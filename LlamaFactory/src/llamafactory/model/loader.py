# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from importlib import import_module
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from ..extras.packages import is_torch_version_greater_than
from .adapter import init_adapter
from .model_utils.ktransformers import load_kt_pretrained_model
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model
from .runtime_patch import apply_data_runtime_patches


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _normalize_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "_").replace(".", "_").replace("/", "_")


def _get_data_model_class(model_type: str, model_name: str):
    data_module = import_module(".data", package=__package__)
    normalized_model_type = _normalize_model_name(model_type)
    normalized_model_name = _normalize_model_name(model_name)
    candidates = [normalized_model_type, normalized_model_name]
    if any("qwen3_vl_moe" in candidate for candidate in candidates) and getattr(data_module, "Qwen3VLMoeDATA", None) is not None:
        return data_module.Qwen3VLMoeDATA
    if any("qwen3_vl" in candidate for candidate in candidates) and getattr(data_module, "Qwen3VLDATA", None) is not None:
        return data_module.Qwen3VLDATA
    if any("qwen2_5_vl" in candidate for candidate in candidates) and getattr(data_module, "Qwen2_5_VLDATA", None) is not None:
        return data_module.Qwen2_5_VLDATA
    if any("t5" in candidate for candidate in candidates) and getattr(data_module, "T5DATA", None) is not None:
        return data_module.T5DATA
    if any("llama" in candidate for candidate in candidates) and getattr(data_module, "LlamaDATA", None) is not None:
        return data_module.LlamaDATA
    if any("qwen2" in candidate for candidate in candidates) and getattr(data_module, "Qwen2DATA", None) is not None:
        return data_module.Qwen2DATA
    return None


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try another one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    is_data_mode = finetuning_args.finetuning_type == "data" or finetuning_args.is_data
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    if is_data_mode and finetuning_args.runtime_local_patch:
        apply_data_runtime_patches(getattr(config, "model_type", ""), model_args.model_name_or_path)

    model = None
    lazy_load = False
    if model_args.use_kt:
        from ktransformers.sft.monkey_patch_torch_module import install_patch

        install_patch()
        model = load_kt_pretrained_model(config, model_args)
    elif model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        if finetuning_args.is_data:
            config.is_data = finetuning_args.is_data
            config.adaprompt = finetuning_args.adaprompt
            config.n_tasks = getattr(finetuning_args, "n_tasks", getattr(config, "n_tasks", None))
            config.task_id = finetuning_args.task_id
            config.data_rank1 = model_args.data_rank1
            config.data_rank2 = model_args.data_rank2
            config.restore = finetuning_args.restore
            config.scale = finetuning_args.scale
            config.unc_thr = finetuning_args.unc_thr
            config.ema_data = finetuning_args.ema_data
            config.ema_teacher = finetuning_args.ema_teacher
            config.gap_layers = finetuning_args.gap_layers
            config.reinit = finetuning_args.reinit
            config.ortho_mu = finetuning_args.ortho_mu
            config.scale_bakebone = finetuning_args.scale_bakebone
            config.nomlp = finetuning_args.nomlp
            config.project = finetuning_args.cl_project

        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        init_kwargs["torch_dtype"] = "auto"

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            auto_map = getattr(config, "auto_map", None) or {}
            if is_data_mode:
                load_class = _get_data_model_class(getattr(config, "model_type", ""), model_args.model_name_or_path)
                if load_class is None:
                    auto_map = getattr(config, "auto_map", None) or {}
                    if isinstance(auto_map, dict) and "AutoModelForImageTextToText" in auto_map:
                        load_class = AutoModelForImageTextToText
                    elif type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                        load_class = AutoModelForImageTextToText
                    elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                        load_class = AutoModelForSeq2SeqLM
                    elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio-text for qwen omni
                        load_class = AutoModelForTextToWaveform
                    else:
                        load_class = AutoModelForCausalLM
            elif isinstance(auto_map, dict) and "AutoModelForImageTextToText" in auto_map:
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio-text for qwen omni
                load_class = AutoModelForTextToWaveform
            elif "data" in model_args.model_name_or_path.lower():
                load_class = _get_data_model_class(getattr(config, "model_type", ""), model_args.model_name_or_path)
                if load_class is None:
                    load_class = AutoModelForCausalLM
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
                    model = getattr(model, "thinker")

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    # Wrap with DATA adapters if requested via finetuning_type or explicit flag.
    if is_data_mode:
        base_model = model
        data_model_class = _get_data_model_class(getattr(config, "model_type", ""), model_args.model_name_or_path)
        if data_model_class is not None and not isinstance(base_model, data_model_class):
            model = data_model_class(base_model.config)
        else:
            # Fallback: keep base and let adapter decide
            model = base_model
        if model is not base_model:
            model.load_state_dict(base_model.state_dict(), strict=False)
            if getattr(finetuning_args, "adaprompt", False) and hasattr(model, "model"):
                post_prompt_init = getattr(model.model, "post_prompt_init", None)
                if callable(post_prompt_init):
                    post_prompt_init()

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    # Conv3D is not recommended when using torch 2.9.x
    if is_torch_version_greater_than("2.9.0") and not is_torch_version_greater_than("2.10.0"):
        if any(isinstance(m, torch.nn.Conv3d) for m in model.modules()):
            raise ValueError(
                "Unsupported torch version detected: torch 2.9.x with Conv3D. "
                "This combination is known to cause severe performance regression. "
                "Please downgrade torch to <2.9 or remove Conv3D. "
                "See https://github.com/pytorch/pytorch/issues/166122"
            )

    if is_data_mode:
        logger.info_rank0("Fine-tuning method: DATA (freeze backbone, train injected data parameters only)")
        for name, param in model.named_parameters():
            param.requires_grad_("data_" in name)

        if is_trainable:
            prepare_model_for_training(model, model_args)

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    # Borrowing the kernel plugins ability of v1 to temporarily apply the NPU fusion operator to v0,
    # it is turned off by default, and can be discarded after the transition period ends.
    if model_args.use_v1_kernels and is_trainable:
        logger.warning_rank0(
            "You are try to using future feature about kernels, please note that this feature "
            "is not supported for all models. If get any error, please disable this feature, or report the issue."
        )
        from ..v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = apply_default_kernels(model, include_kernels=model_args.use_v1_kernels)

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
