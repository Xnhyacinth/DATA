import logging
import torch
from transformers.data.data_collator import *
from ...data import SFTDataCollatorWith4DAttentionMask
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence

SUPPORTED_DECODER_MODELS = ['codegen', 'bloomz', 'gpt-neox', 'llama']
SUPPORTED_SEQ2SEQ_MODELS = ['t5', 'flan-t5']


def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False


@dataclass
class DataCollatorForUIE(SFTDataCollatorWith4DAttentionMask):
    model: Optional[Any] = None

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        model_name = self.model.config._name_or_path
        # print(model_name)
        features = super().__call__(features)
        if check_model(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
            
        return features
        
    def seq2seq_call(self, batch, return_tensors):
        sources = []
        labels = []

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            source = instruction
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        # TODO, support online demo
        if self.text_only:
            model_inputs = {"inputs": sources, "labels": labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    labels,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
            label_mask = labels["attention_mask"].bool()
            model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

            # prepare decoder_input_ids
            if self.model is not None:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
                model_inputs["decoder_input_ids"] = decoder_input_ids

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

