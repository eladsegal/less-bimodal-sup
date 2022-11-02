from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from copy import deepcopy

import torch
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedTokenizerBase

from src.data.datamodules.discriminative_vision_language_datamodule import DiscriminativeVisionLanguageDataModule
from src.data.dataset_containers.images_dataset_container import ImagesDatasetContainer

import logging

logger = logging.getLogger(__name__)


class FinetuningDataModule(DiscriminativeVisionLanguageDataModule):
    def __init__(
        self,
        feature_extractor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.feature_extractor = feature_extractor

    def get_collate_fn(self, split=None, images=None):
        return FinetuningDataCollator(
            tokenizer=self.tokenizer,
            image_transform=self._image_transform[split],
            feature_extractor=self.feature_extractor,
            images=images,
            key_mapping=self._get_key_mapping(split),
            ans2label=self._ans2label,
            objective_format=self._objective_format,
            dataset_container=self.dataset_container,
            split=split,
        )

    def prepare_examples(
        self,
        *args,
        **kwargs: Optional[Dict[str, Any]],
    ):
        split = args[1] if len(args) > 1 else kwargs["split"]
        if self.dataset_container is None or "vision" not in self.dataset_container:
            kwargs["collator_kwargs"] = kwargs.get("collator_kwargs", {})
            if "images" not in kwargs["collator_kwargs"]:
                dataset_container = ImagesDatasetContainer()
                dataset_container.setup()
                kwargs["collator_kwargs"]["images"] = dataset_container.ready_datasets[split]
        return super().prepare_examples(*args, **kwargs)


@dataclass
class FinetuningDataCollator:
    tokenizer: PreTrainedTokenizerBase
    image_transform: Any
    feature_extractor: Any
    images: Dataset
    ans2label: Dict[str, int]
    objective_format: str
    key_mapping: Dict[str, List[str]]
    dataset_container: Any
    split: str

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
            # TODO: Add a wrapper function in a collator parent class that does this
            batch["source_container_name"] = [example["source_container_name"] for example in raw_batch]

        hf_raw_batch = [
            {k: v for k, v in example.items() if k in key_mapping["language_inputs"]} for example in raw_batch
        ]
        tensor_keys.extend(key_mapping["language_inputs"])
        batch.update(
            self.tokenizer.pad(
                hf_raw_batch,
                padding=True,
                return_tensors="pt",
            )
        )

        if self.objective_format == "cross_entropy":
            if "labels" in elem:
                labels = []
                for example in raw_batch:
                    labels.append(example["labels"])
                tensor_keys.append("labels")
                batch["labels"] = torch.as_tensor(labels)
        elif self.objective_format == "binary_cross_entropy_with_logits":
            if "labels_list" in elem and "scores_list" in elem:
                targets = torch.zeros(batch_size, len(self.ans2label))
                for i, example in enumerate(raw_batch):
                    labels = example["labels_list"]
                    scores = example["scores_list"]
                    for (label, score) in zip(labels, scores):
                        targets[i, label] = score
                tensor_keys.append("targets")
                batch["targets"] = targets

        num_of_images = len([key for key in elem.keys() if "image_file_name" in key])
        batch["num_of_images"] = num_of_images

        for image_index in range(num_of_images):
            if num_of_images == 1:
                image_file_name_str = f"image_file_name"
                vision_inputs_str = f"vision_inputs"
                rename_extracted_features = False
                key_fstr = "{k}"
            else:
                image_file_name_str = f"image_file_name_{image_index}"
                vision_inputs_str = f"vision_inputs_{image_index}"
                rename_extracted_features = True
                key_fstr = "{k}_{image_index}"

            images = []
            for example in raw_batch:
                image_file_name = example[image_file_name_str.format(image_index=image_index)]
                images_dataset = self.get_images_dataset(example) if self.images is None else self.images
                images.append(self.image_transform(images_dataset[image_file_name]).numpy())

            extracted_features = self.feature_extractor(images, return_tensors="pt")
            if rename_extracted_features:
                extracted_features = {
                    key_fstr.format(k=k, image_index=image_index): v for k, v in extracted_features.items()
                }
            key_mapping[vision_inputs_str] = {
                key_fstr.format(k=k, image_index=image_index): k for k in self.feature_extractor.model_input_names
            }
            tensor_keys.extend(key_mapping[vision_inputs_str])
            batch.update(**extracted_features)

        return batch

    def get_images_dataset(self, example):
        if "source_container_name" in example:
            dataset_container = self.dataset_container[example["source_container_name"]]
        else:
            dataset_container = self.dataset_container
        images = dataset_container["vision"].ready_datasets[self.split]
        return images
