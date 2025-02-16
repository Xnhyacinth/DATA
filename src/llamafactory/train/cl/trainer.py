# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

from copy import deepcopy
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .aug import get_text_augmentation_transforms

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback

logger = get_logger(__name__)
SUPPORTED_DECODER_MODELS = ['codegen', 'bloomz', 'gpt-neox', 'llama', 'qwen2']
ANSWER_PREFIX = "Answer:"


def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False

def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_truncate(t, limit) for k, t in tensors.items()})

    return tensors[:limit]

def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:
            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append(pred.strip())
    else:
        final_predictions = predictions

    return final_predictions

def update_ema_variables(ema_model, model, alpha_teacher, alpha_vida):#, iteration):
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #     ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    # return ema_model
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        if "vida_" in name:
            ema_param.data[:] = alpha_vida * ema_param[:].data[:] + (1 - alpha_vida) * param[:].data[:]
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.args.restore = self.finetuning_args.restore
        
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if self.finetuning_args.restore:
            # self.restore_step = int(self.finetuning_args.restore * self.state.max_steps)
            logger.info(f"Stochastic restore is enabled. restore ratio: {self.finetuning_args.restore}")
            self.model_state = deepcopy(self.model.state_dict())

        if self.finetuning_args.scale:
            logger.info("scaling is enabled.")
            self.model_ema = deepcopy(self.model)
            self.thr = self.finetuning_args.unc_thr
            self.alpha_teacher = self.finetuning_args.ema_teacher
            self.alpha_vida = self.finetuning_args.ema_vida
            for param in self.model_ema.parameters():
                param.detach_()
                
            self.transform = get_text_augmentation_transforms(
                tokenizer=self.tokenizer,
                synonym_prob=0.3,
                delete_prob=0.1,
                insert_prob=0.1,
                swap_prob=0.1,
                pad_token_id=self.tokenizer.pad_token_id
                )
            # back_model_name1 = f"Helsinki-NLP/opus-mt-en-de"
            # self.back_tokenizer1 = MarianTokenizer.from_pretrained(back_model_name1)
            # self.back_model1 = MarianMTModel.from_pretrained(back_model_name1)
            # back_model_name2 = f"Helsinki-NLP/opus-mt-de-en"
            # self.back_tokenizer2 = MarianTokenizer.from_pretrained(back_model_name2)
            # self.back_model2 = MarianMTModel.from_pretrained(back_model_name2)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def set_scale(self, update_model, high, low):
        for name, module in update_model.named_modules():
            if hasattr(module, 'scale1'):
                module.scale1 = low.item()
            elif hasattr(module, 'scale2'):
                module.scale2 = high.item()
    
    # def set_train(self, update_model, train_mode):
    #     for name, module in update_model.named_modules():
    #         if hasattr(module, 'train_mode'):
    #             module.train_mode = train_mode
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        # if self.finetuning_args.adaprompt:
        task_id = inputs.pop("task_id")
        # inputs = {key: inputs for key in ("input_ids", "attention_mask", "labels", "token_type_ids", "decoder_input_ids")}
        inputs = self._prepare_inputs(inputs)

        if self.finetuning_args.scale and (self.state.global_step + 1) % self.finetuning_args.scale == 0:
            self.model_ema.eval()
            # Teacher Prediction
            N = 5 
            outputs_uncs = []
            for i in range(N):
                inputs_ema = deepcopy(inputs)
                if "labels" in inputs:
                    labels = inputs_ema.pop("labels")
                else:
                    labels = None
                # xxx = self.transform(inputs_ema)
                # self.tokenizer.batch_decode(self.transform(inputs_ema)['input_ids'])
                # self.tokenizer.batch_decode(inputs_ema['input_ids'])
                # breakpoint()
                # outputs_  = self.model_ema(**inputs_ema, return_dict=True).logits.detach()
                outputs_  = self.model_ema(**self.transform(inputs_ema), return_dict=True).logits.detach()
                outputs_uncs.append(outputs_)
            outputs_unc = torch.stack(outputs_uncs)
            variance = torch.var(outputs_unc, dim=0)
            uncertainty = torch.mean(variance) * 0.05
            # print(uncertainty)
            if uncertainty >= self.thr:
                lambda_high = 1 + uncertainty
                lambda_low = 1 - uncertainty
            else:
                lambda_low = 1 + uncertainty
                lambda_high = 1 - uncertainty
            self.set_scale(update_model=model, high=lambda_high, low=lambda_low)
            self.set_scale(update_model=self.model_ema, high=lambda_high, low=lambda_low)
            # breakpoint()
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # # l2-normalization for loranew_A/B
        # l2_loss = 0.
        # for name, param in self.model.named_parameters():
        #     if "lora_" in name:
        #         l2_loss += torch.norm(param, p=2)

        # lamda = self.args.lamda

        # logger.info(f"l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ2: {lamda}")
        # loss = loss + l2_loss * lamda

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        if self.finetuning_args.restore and (self.state.global_step + 1) % self.restore_step == 0:
            # Stochastic restore
            if self.state.max_steps - self.state.global_step >= self.restore_step:
                logger.info(f"Stochastic restore at step {self.state.global_step}")
                for nm, m  in self.model.named_modules():
                    for npp, p in m.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            mask = (torch.rand(p.shape)<0.00001).float().to(self.model.device)
                            with torch.no_grad():
                                p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
        
        if self.finetuning_args.scale and (self.state.global_step + 1) % self.finetuning_args.scale == 0:
            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.alpha_teacher, alpha_vida=self.alpha_vida)

        return loss.detach() / self.args.gradient_accumulation_steps

    
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate and not 't5' in model.config.architectures[0].lower():
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        if self.finetuning_args.scale:
            self.model_ema.eval()
            # Teacher Prediction
            N = 5 
            outputs_uncs = []
            for i in range(N):
                inputs_ema = deepcopy(inputs)
                if "labels" in inputs:
                    labels = inputs_ema.pop("labels")
                else:
                    labels = None
                # xxx = self.transform(inputs_ema)
                # self.tokenizer.batch_decode(self.transform(inputs_ema)['input_ids'])
                # self.tokenizer.batch_decode(inputs_ema['input_ids'])
                # breakpoint()
                # outputs_  = self.model_ema(**inputs_ema, return_dict=True).logits.detach()
                outputs_  = self.model_ema(**self.transform(inputs_ema), return_dict=True).logits.detach()
                outputs_uncs.append(outputs_)
            outputs_unc = torch.stack(outputs_uncs)
            variance = torch.var(outputs_unc, dim=0)
            uncertainty = torch.mean(variance) * 0.05
            # print(uncertainty)
            if uncertainty >= self.thr:
                lambda_high = 1 + uncertainty
                lambda_low = 1 - uncertainty
            else:
                lambda_low = 1 + uncertainty
                lambda_high = 1 - uncertainty
            self.set_scale(update_model=model, high=lambda_high, low=lambda_low)
            self.set_scale(update_model=self.model_ema, high=lambda_high, low=lambda_low)
            # breakpoint()
        task_ids = inputs.pop('task_id')
        if self.args.adaprompt:
            if task_ids[0] != task_ids[-1]:
                unique_task_ids = set(task_ids.tolist())
                all_losses = []
                all_generated_tokens = []
                # logger.info(f'task id diff: {unique_task_ids}')
                for task_id in unique_task_ids:
                    mask = task_ids == task_id
                    task_specific_inputs = {k: v[mask] for k, v in inputs.items()}

                    if 't5' in self.model.config.architectures[0].lower():
                        self.model.encoder.task_id = task_id
                        self.model.decoder.task_id = task_id
                    # elif 'llama' in self.model.config.architectures[0].lower():
                    else:
                        self.model.model.task_id = task_id

                    loss, generated_tokens, _ = super().prediction_step(
                        model, task_specific_inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
                    )
                    # if generated_tokens.shape[-1] < gen_config.max_length:
                    #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
                    # elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
                    # generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self._gen_kwargs['max_new_tokens'] + 1)
                    all_losses.append(loss)
                    all_generated_tokens.append(generated_tokens)
                # logger.info(f'error gen config: {self._gen_kwargs.copy()}')
                loss = sum(all_losses)
                max_len = max([tokens.size(-1) for tokens in all_generated_tokens])
                all_generated_tokens = [self._pad_tensors_to_max_len(generated_tokens, max_len) for generated_tokens in all_generated_tokens]
                generated_tokens = torch.cat(all_generated_tokens, dim=0)
            else:
                if 't5' in self.model.config.architectures[0].lower():
                    self.model.encoder.task_id = task_ids[0]
                    self.model.decoder.task_id = task_ids[0]
                # elif 'llama' in self.model.config.architectures[0].lower():
                else:
                    self.model.model.task_id = task_ids[0]

                loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
                )
        else:
            loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        # if 't5' in model.config.architectures[0].lower():
        #     import pdb;pdb.set_trace()
        # self.tokenizer.decode(inputs["labels"][0][:-2], skip_special_tokens=True) self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        # self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        if generated_tokens is not None and self.args.predict_with_generate and not 't5' in model.config.architectures[0].lower():
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))
            
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)
        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # if self.args.adaprompt:
        #     self.set_train(self.model, False)
    
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # print("woshilogit", model, inputs, prediction_loss_only, ignore_keys)
            # exit()
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                all_inputs.add(inputs_decode)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                all_preds.add(logits)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self.gather_function((labels))
                all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
