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

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments

if is_transformers_version_greater_than("4.57.0"):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe


logger = logging.get_logger(__name__)

_GENERATION_REQUIRED_METHODS: tuple[str, ...] = (
    "generate",
    "prepare_inputs_for_generation",
)


def _iter_generation_mixin_methods() -> tuple[str, ...]:
    method_names = []
    for name in dir(GenerationMixin):
        if name.startswith("__"):
            continue
        attr = getattr(GenerationMixin, name, None)
        if callable(attr):
            method_names.append(name)

    return tuple(sorted(set(method_names) | set(_GENERATION_REQUIRED_METHODS)))


def _reorder_flm_audio_cache(self, past_key_values, beam_idx: "torch.Tensor"):
    """Generic cache reorder fallback for remote-code models lacking `_reorder_cache`."""

    if past_key_values is None:
        return past_key_values

    if torch.is_tensor(past_key_values):
        try:
            return past_key_values.index_select(0, beam_idx.to(past_key_values.device))
        except Exception:
            return past_key_values

    if isinstance(past_key_values, tuple):
        return tuple(_reorder_flm_audio_cache(self, value, beam_idx) for value in past_key_values)

    if isinstance(past_key_values, list):
        return [_reorder_flm_audio_cache(self, value, beam_idx) for value in past_key_values]

    if isinstance(past_key_values, dict):
        return {key: _reorder_flm_audio_cache(self, value, beam_idx) for key, value in past_key_values.items()}

    return past_key_values


def _ensure_generation_mixin_methods(model: "PreTrainedModel") -> None:
    """Attach missing GenerationMixin methods required by the current transformers version.

    Some remote-code models don't inherit GenerationMixin, so even if we bind
    `GenerationMixin.generate`, the call may fail later on missing internal helpers
    (e.g. `_extract_generation_mode_kwargs`).
    """
    model_cls = type(model)
    for name in _iter_generation_mixin_methods():
        # Preserve descriptor types (staticmethod/classmethod) when copying from GenerationMixin.
        # `getattr(GenerationMixin, name)` resolves descriptors, so we must use `__dict__` to keep
        # staticmethods like `_expand_inputs_for_generation` from being turned into instance methods.
        raw_attr = GenerationMixin.__dict__.get(name, None)
        resolved_attr = getattr(GenerationMixin, name, None)

        if raw_attr is None and not callable(resolved_attr):
            continue

        # Some generation paths resolve helpers on `type(self)` rather than on the instance,
        # so class-level patching is required for remote-code models.
        existing_raw = model_cls.__dict__.get(name, None)
        if isinstance(raw_attr, staticmethod):
            # Ensure the target class keeps it as a staticmethod (override incorrect prior injection).
            if not isinstance(existing_raw, staticmethod):
                setattr(model_cls, name, raw_attr)
        elif isinstance(raw_attr, classmethod):
            if not isinstance(existing_raw, classmethod):
                setattr(model_cls, name, raw_attr)
        else:
            if not hasattr(model_cls, name) and callable(resolved_attr):
                setattr(model_cls, name, resolved_attr)

        # Instance-level patching should not be applied to staticmethod/classmethod helpers.
        # They must stay unbound to avoid kwarg/positional collisions inside transformers generate().
        if isinstance(raw_attr, (staticmethod, classmethod)):
            continue

        if callable(resolved_attr) and not hasattr(model, name):
            setattr(model, name, MethodType(resolved_attr, model))

    if not hasattr(model_cls, "_reorder_cache"):
        setattr(model_cls, "_reorder_cache", _reorder_flm_audio_cache)

    if not hasattr(model, "_reorder_cache"):
        model._reorder_cache = MethodType(_reorder_flm_audio_cache, model)


def patch_qwen3_omni_moe_thinker_text_sparse_moe_block():
    if is_transformers_version_greater_than("4.57.0") and not is_transformers_version_greater_than("4.58.0"):
        from .model_utils.moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

        logger.warning_rank0(
            "You are using transformers with 4.x version, the Qwen3OmniMoeThinkerTextSparseMoeBlock will have some issues about deepspeed zero2 and fsdp2 training, so that we patched this model to avoid it. Transformers v5.0.0rc0 has fixed the issue, you can also try to update the transformers to using qwen3_omni. See more information on https://github.com/hiyouga/LLaMA-Factory/issues/9628."
        )

        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock = Qwen3OmniMoeThinkerTextSparseMoeBlock


def patch_youtu_vl_model(model: "PreTrainedModel") -> None:
    original_forward = model.forward

    def forward(self, *args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if "loss" not in outputs and "labels" in kwargs:
            logits = outputs.get("logits")
            labels = kwargs.get("labels")
            if logits is not None and labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                outputs["loss"] = loss

        return outputs

    model.forward = MethodType(forward, model)


def patch_flm_audio_model(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    original_forward = model.forward
    original_forward_text = getattr(model, "_forward_text", None)

    def _get_audio_pad_token_id() -> int:
        mm_token_info = getattr(model.config, "mm_token_info", None)
        aud_pad_token_id = getattr(mm_token_info, "aud_pad_token_id", None)
        if aud_pad_token_id is None:
            aud_pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if aud_pad_token_id is None:
            aud_pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if aud_pad_token_id is None:
            aud_pad_token_id = 0
        return int(aud_pad_token_id)

    def _get_logits(result: Any):
        if result is None:
            return None
        if isinstance(result, dict):
            return result.get("logits")
        return getattr(result, "logits", None)

    def _attach_loss(result: Any, loss: "torch.Tensor", return_dict: bool):
        if return_dict:
            if isinstance(result, dict):
                result["loss"] = loss
            else:
                setattr(result, "loss", loss)
                try:
                    result["loss"] = loss
                except Exception:
                    pass
            return result

        if isinstance(result, tuple):
            return (loss,) + result

        return (loss, result)

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        listen_ids = kwargs.get("listen_ids", None)
        speak_ids = kwargs.get("speak_ids", None)

        if input_ids is not None and listen_ids is None and speak_ids is None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if attention_mask is not None:
                valid_token_count = int(attention_mask.sum().item())
            elif pad_token_id is not None:
                valid_token_count = int(input_ids.ne(pad_token_id).sum().item())
            else:
                valid_token_count = int(input_ids.numel())

            aud_channel = getattr(self.config, "aud_channel", 8)
            aud_pad_token_id = _get_audio_pad_token_id()
            default_audio_ids = torch.full(
                (valid_token_count, aud_channel),
                aud_pad_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            kwargs["listen_ids"] = default_audio_ids
            kwargs["speak_ids"] = default_audio_ids.clone()

        return original_forward(*args, **kwargs)

    def forward_text(self, outputs, labels, return_dict):
        result = None
        if callable(original_forward_text):
            try:
                result = original_forward_text(outputs, None, return_dict)
            except NotImplementedError:
                result = outputs

        if result is None:
            result = outputs

        if labels is None:
            return result

        logits = _get_logits(result)
        if logits is None and result is not outputs:
            logits = _get_logits(outputs)
        if logits is None:
            return result

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return _attach_loss(result, loss, return_dict)

    model.forward = MethodType(forward, model)
    model._forward_text = MethodType(forward_text, model)


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, is_trainable, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and "InternVLChatModel" in architectures:
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if isinstance(architectures, list) and "LlavaLlamaForCausalLM" in architectures:
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    if getattr(config, "model_type", None) == "lfm2_vl" and not is_transformers_version_greater_than("4.58.0"):
        raise RuntimeError(
            "LFM2.5-VL model requires transformers>=4.58.0 or install from commit: "
            "pip install git+https://github.com/huggingface/transformers.git@3c2517727ce28a30f5044e01663ee204deb1cdbe"
        )

    if getattr(config, "model_type", None) == "qwen3_omni_moe":
        patch_qwen3_omni_moe_thinker_text_sparse_moe_block()

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # fsdp/deepspeed zero3 does not need device map
    if not (is_deepspeed_zero3_enabled() or is_fsdp_enabled()) and init_kwargs["low_cpu_mem_usage"]:
        if "device_map" not in init_kwargs and model_args.device_map:
            init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

        if init_kwargs.get("device_map", None) == "auto":
            init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = getattr(model, "generation_config", None)  # check and fix generation config
    if gen_config is None:
        try:
            gen_config = GenerationConfig.from_model_config(model.config)
        except Exception:
            gen_config = GenerationConfig()

        model.generation_config = gen_config

    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"]:
        generate_func = getattr(model, "generate", None)
        generate_impl = getattr(generate_func, "__func__", generate_func)
        prepare_inputs_func = getattr(model, "prepare_inputs_for_generation", None)
        prepare_inputs_impl = getattr(prepare_inputs_func, "__func__", prepare_inputs_func)
        needs_mixin_generate = (not callable(generate_func)) or ("GenerationMixin" not in str(generate_impl))
        needs_mixin_prepare = (not callable(prepare_inputs_func)) or ("GenerationMixin" not in str(prepare_inputs_impl))
        needs_missing_helpers = not hasattr(model, "_extract_generation_mode_kwargs")

        if needs_mixin_generate or needs_mixin_prepare or needs_missing_helpers:
            _ensure_generation_mixin_methods(model)

    if getattr(model.config, "model_type", None) == "FLMAudio":
        patch_flm_audio_model(model, tokenizer)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(
            model,
            tokenizer,
            new_special_tokens_config=getattr(model_args, "_special_token_descriptions", None),
            init_special_tokens=model_args.init_special_tokens,
        )

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        if getattr(model.config, "model_type", None) == "youtu_vl":
            patch_youtu_vl_model(model)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    def get_rope_index_func(self: "AutoModelForCausalLMWithValueHead"):
        if isinstance(self.pretrained_model, PeftModel):
            base_model = self.pretrained_model.base_model.model
        else:
            base_model = self.pretrained_model

        if base_model and hasattr(base_model, "get_rope_index"):
            return base_model.get_rope_index
        elif base_model and hasattr(base_model, "model") and hasattr(base_model.model, "get_rope_index"):
            return base_model.model.get_rope_index
        else:
            return None

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "get_rope_index", get_rope_index_func(model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
