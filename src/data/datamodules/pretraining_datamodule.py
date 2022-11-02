from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset

from transformers import PreTrainedTokenizerBase, DataCollatorForLanguageModeling
from datasets import Dataset

from src.data.datamodules.hf_datamodule import HfDataModule
from src.pl.callbacks.progress import tqdm
from src.utils.fallbacks import handle_fallback_per_split

from src.utils.spans import get_token_span

import logging

logger = logging.getLogger(__name__)


class PretrainingDataModule(HfDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        feature_extractor,
        tasks: List[str],
        image_transform=None,
        partial_masking=True,
        label_corruption_rate: Optional[Union[float, int]] = None,
        whole_label_masking=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self._image_transform = handle_fallback_per_split(image_transform)

        self._mlm_collator = DataCollatorForLanguageModeling(deepcopy(tokenizer)) if "mlm" in tasks else None
        self._partial_masking = partial_masking
        self._label_corruption_rate = label_corruption_rate
        self._whole_label_masking = whole_label_masking

        self._tasks = tasks

    @property
    def _model_kwargs(self):
        model_kwargs = super()._model_kwargs
        model_kwargs["tokenizer"] = self.tokenizer
        return model_kwargs

    @property
    def general_preprocessing_kwargs(self):
        return {
            "tokenizer": self.tokenizer,
        }

    @property
    def general_key_mapping(self):
        return {
            "language_inputs": self.tokenizer.model_input_names,
        }

    @staticmethod
    def general_preprocessing(examples, idxs, split, tokenizer):
        result = {}
        result["idx"] = idxs

        # Input
        is_caption_from_labels = "is_caption_from_labels" in examples and examples["is_caption_from_labels"][0] is True
        if is_caption_from_labels:
            assert all(v is True for v in examples["is_caption_from_labels"])

        language_inputs = tokenizer(
            examples["caption"], return_special_tokens_mask=True, return_offsets_mapping=is_caption_from_labels
        )
        result.update(**language_inputs)
        vision_inputs = {"image_file_name": examples["image_file_name"]}
        result.update(vision_inputs)

        if is_caption_from_labels:
            # For each label, find the token indices associated with it.
            # In the collator we will choose % of labels and mask them in these indices.
            label_to_token_indices_list = []
            for labels, caption, offset_mapping, special_tokens_mask in zip(
                examples["labels"],
                examples["caption"],
                language_inputs["offset_mapping"],
                language_inputs["special_tokens_mask"],
            ):
                label_to_token_indices_list.append([])
                for label in labels:
                    start_index = caption.index(label)
                    end_index = start_index + len(label)
                    span = get_token_span(offset_mapping, special_tokens_mask, start_index, end_index, 0)
                    label_to_token_indices_list[-1].append(list(range(span[0], span[1] + 1)))
            result["label_to_token_indices"] = label_to_token_indices_list
        else:
            result["label_to_token_indices"] = [None for _ in range(len(idxs))]

        if "source_container_name" in examples:
            result["source_container_name"] = examples["source_container_name"]

        return result

    def get_collate_fn(self, split=None):
        return PretrainingDataCollator(
            tokenizer=self.tokenizer,
            image_transform=self._image_transform[split],
            feature_extractor=self.feature_extractor,
            key_mapping=self._get_key_mapping(split),
            mlm_collator=self._mlm_collator,
            tasks=self._tasks,
            dataset_container=self.dataset_container,
            split=split,
            partial_masking=self._partial_masking,
            label_corruption_rate=self._label_corruption_rate,
            whole_label_masking=self._whole_label_masking,
        )


@dataclass
class PretrainingDataCollator:
    tokenizer: PreTrainedTokenizerBase
    image_transform: Any
    feature_extractor: Any
    key_mapping: Dict[str, List[str]]
    mlm_collator: Optional[DataCollatorForLanguageModeling]
    tasks: List[str]
    dataset_container: Any
    split: str
    partial_masking: bool
    label_corruption_rate: bool
    whole_label_masking: bool

    def __call__(self, raw_batch) -> Dict[str, Any]:
        batch_size = len(raw_batch)
        elem = raw_batch[0]
        key_mapping = deepcopy(self.key_mapping)
        tensor_keys = []
        batch = {
            "key_mapping": key_mapping,
            "tensor_keys": tensor_keys,
            "pidx": [example["pidx"] for example in raw_batch],
        }
        if "source_container_name" in elem:
            batch["source_container_name"] = [example["source_container_name"] for example in raw_batch]

        hf_raw_batch = [
            {k: v for k, v in example.items() if k in key_mapping["language_inputs"] + ["special_tokens_mask"]}
            for example in raw_batch
        ]
        tensor_keys.extend(key_mapping["language_inputs"])
        batch.update(
            self.tokenizer.pad(
                hf_raw_batch,
                padding=True,
                return_tensors="pt",
            )
        )

        batch["num_of_images"] = 1

        images = []
        for example in raw_batch:
            image = self.get_images_dataset(example)[example["image_file_name"]]
            images.append(self.image_transform(image).numpy())

        extracted_features = self.feature_extractor(images, return_tensors="pt")
        key_mapping["vision_inputs"] = self.feature_extractor.model_input_names
        tensor_keys.extend(key_mapping["vision_inputs"])
        batch.update(**extracted_features)

        # MLM
        if "mlm" in self.tasks:
            if self.partial_masking:
                if elem["label_to_token_indices"] is not None and self.whole_label_masking:
                    # Doesn't support mixing of instances without labels
                    batch["mlm_input_ids"], batch["mlm_labels"] = label_masking(
                        self.tokenizer,
                        batch["input_ids"],
                        label_to_token_indices_list=[example["label_to_token_indices"] for example in raw_batch],
                        label_corruption_rate=self.label_corruption_rate,
                    )
                else:
                    batch["mlm_input_ids"], batch["mlm_labels"] = self.mlm_collator.torch_mask_tokens(
                        batch["input_ids"], batch["special_tokens_mask"]
                    )
            else:
                batch["mlm_input_ids"], batch["mlm_labels"] = full_masking(
                    self.tokenizer, batch["input_ids"], batch["special_tokens_mask"]
                )

            tensor_keys.extend(["mlm_input_ids", "mlm_labels"])
            key_mapping["mlm_language_inputs"] = {
                **{k: k for k in self.tokenizer.model_input_names},
                **{"mlm_input_ids": "input_ids"},
            }

        if "itm" in self.tasks:
            # ITM
            pos_len = batch_size // 2
            neg_len = batch_size - pos_len
            itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)])
            itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]
            batch["itm_labels"] = itm_labels.long()

            false_images = []
            for i, example in enumerate(raw_batch):
                if itm_labels[i].item() == 0:
                    raw_dataset = self.get_dataset(example)
                    false_example = raw_dataset[np.random.choice(len(raw_dataset))]
                    image = self.get_images_dataset(example)[false_example["image_file_name"]]
                    false_images.append(self.image_transform(image).numpy())

            batch["itm_false_pixel_values"] = self.feature_extractor(false_images, return_tensors="pt")["pixel_values"]
            tensor_keys.extend(["itm_false_pixel_values", "itm_labels"])

        return batch

    def get_images_dataset(self, example):
        if "source_container_name" in example:
            dataset_container = self.dataset_container[example["source_container_name"]]
        else:
            dataset_container = self.dataset_container
        images = dataset_container["vision"].ready_datasets[self.split]
        return images

    def get_dataset(self, example):
        if "source_container_name" in example:
            dataset_container = self.dataset_container[example["source_container_name"]]
        else:
            dataset_container = self.dataset_container
        dataset = dataset_container["language"].ready_datasets[self.split]
        return dataset


def full_masking(tokenizer, inputs: Any, special_tokens_mask: Any, mask_prob=0.8):
    labels = inputs.clone()
    probability_matrix = torch.ones(labels.shape)
    special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # mask_prob of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_prob)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    if mask_prob < 1.0:
        # rest of the time, we replace masked input tokens with random word
        indices_random = masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

    return inputs, labels


def label_masking(
    tokenizer,
    inputs: Any,
    label_to_token_indices_list: List[List[List[int]]],
    label_corruption_rate: Union[float, int],
    mask_prob=0.8,
):
    mlm_labels = inputs.clone()

    masked_indices = torch.zeros(mlm_labels.shape)
    for i, label_to_token_indices in enumerate(label_to_token_indices_list):
        if label_corruption_rate < 1:
            # Corrupt by % of labels
            num_of_labels_to_corrupt = round(label_corruption_rate * len(label_to_token_indices))
        else:
            # Corrupt by number of labels
            num_of_labels_to_corrupt = label_corruption_rate

        # Mask at least one label
        if num_of_labels_to_corrupt == 0:
            label_corruption_rate = 1

        if num_of_labels_to_corrupt > 0:
            num_of_labels = len(label_to_token_indices)
            indices_to_mask = torch.randperm(num_of_labels)[:num_of_labels_to_corrupt]
            for label_index in indices_to_mask:
                masked_indices[i][label_to_token_indices[label_index]] = 1
    masked_indices = masked_indices.bool()

    mlm_labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # mask_prob of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(mlm_labels.shape, mask_prob)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(mlm_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), mlm_labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, mlm_labels
