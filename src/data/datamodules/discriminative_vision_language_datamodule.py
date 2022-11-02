from typing import Optional, List, Iterator
from collections import Counter, defaultdict

from copy import deepcopy
from pyrsistent import b

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedTokenizerBase

from src.data.datamodules.mixins.answers_labels_mixin import AnswersLabelsMixin
from src.data.datamodules.hf_datamodule import HfDataModule
from src.metrics.concrete.vqa_score import get_score
from src.pl.callbacks.progress import tqdm
from src.utils.fallbacks import handle_fallback_per_split

import logging

logger = logging.getLogger(__name__)


class DiscriminativeVisionLanguageDataModule(AnswersLabelsMixin, HfDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_of_images: int,
        image_transform=None,
        batch_by_sets: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self._image_transform = handle_fallback_per_split(image_transform)

        self.num_of_images = num_of_images

        self._batch_by_sets = batch_by_sets
        self.num_batches_per_key = {}

    @property
    def _model_kwargs(self):
        model_kwargs = super()._model_kwargs
        model_kwargs["num_of_images"] = self.num_of_images
        return model_kwargs

    @property
    def general_preprocessing_kwargs(self):
        return {
            "tokenizer": self.tokenizer,
            "ans2label": self._ans2label_current,
            "objective_format": self._objective_format,
            "batch_by_sets": self._batch_by_sets,
        }

    @property
    def general_key_mapping(self):
        return {
            "language_inputs": self.tokenizer.model_input_names,
        }

    @staticmethod
    def general_preprocessing(examples, idxs, split, tokenizer, ans2label, objective_format, batch_by_sets):
        result = {}
        result["idx"] = idxs

        # Input
        text_input = examples["text_input"]
        language_inputs = tokenizer(text_input)
        result.update(**language_inputs)

        if "image_file_name" in examples:
            vision_inputs = {"image_file_name": examples["image_file_name"]}
        else:
            vision_inputs = {}
            image_index = 0
            while True:
                key = f"image_file_name_{image_index}"
                if key not in examples:
                    break
                vision_inputs[key] = examples[key]
                image_index += 1
        result.update(vision_inputs)

        # Supervision
        if "answers" in examples:
            labels_list = []
            scores_list = []
            for i, answers in enumerate(examples["answers"]):
                answer_count = Counter(answers)

                labels_list.append([])
                scores_list.append([])
                for answer in answer_count:
                    if answer not in ans2label:
                        continue

                    labels_list[-1].append(ans2label[answer])
                    score = get_score(answer_count[answer])
                    scores_list[-1].append(score)
            result.update({"labels_list": labels_list, "scores_list": scores_list})
        elif "answer" in examples:

            def answer_to_label(answer):
                return (
                    ans2label[answer]
                    if split == "train"
                    else ans2label.get(answer, ans2label["UNKNOWN"] if "UNKNOWN" in ans2label else 0)
                )

            result.update({"labels": [answer_to_label(answer) for answer in examples["answer"]]})
        elif "label" in examples:
            result.update({"labels": [ans2label[answer] for answer in examples["label"]]})

        """
        In order to support this again, we need to transform the input according to the objective_format
        in a mapper. This means that ans2label should be available from the start
        if dataset_name == "gqa.default":
            if "answer" in examples:

                def answer_to_label(answer):
                    return (
                        ans2label[answer]
                        if split == "train"
                        else ans2label.get(answer, ans2label["UNKNOWN"] if "UNKNOWN" in ans2label else 0)
                    )

                if objective_format == "binary_cross_entropy_with_logits" or objective_format == "volta":
                    labels_list = [[answer_to_label(answer)] for answer in examples["answer"]]
                    scores_list = [[get_score(10)]] * len(labels_list)
                    result.update({"labels_list": labels_list, "scores_list": scores_list})
                else:
                    result.update({"labels": [answer_to_label(answer) for answer in examples["answer"]]})
        elif dataset_name == "nlvr2.default":
            if "label" in examples:
                if objective_format == "binary_cross_entropy_with_logits" or objective_format == "volta":
                    labels_list = [[ans2label[answer]] for answer in examples["label"]]
                    scores_list = [[get_score(10)]] * len(labels_list)
                    result.update({"labels_list": labels_list, "scores_list": scores_list})
                else:
                    result.update({"labels": [ans2label[answer] for answer in examples["label"]]})"""

        if objective_format == "volta" and "official_split" in examples:
            result.update({"official_split": examples["official_split"]})

        if batch_by_sets:
            result.update({"set_id": examples["set_id"]})

        if "source_container_name" in examples:
            result["source_container_name"] = examples["source_container_name"]

        return result

    def train_dataloader(self):
        if not (self._batch_by_sets):
            return super().train_dataloader()

        if "train" not in self.preprocessed_datasets:
            return None

        return DataLoader(
            self.preprocessed_datasets["train"],
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn("train"),
            batch_sampler=self._get_batch_sampler(key="train", shuffle_per_epoch=True),
        )

    def _get_batch_sampler(self, key, shuffle_per_epoch, dataset=None):
        kwargs = {
            "num_batches": self.num_batches_per_key.get(key),
            "num_epochs": self.trainer.max_epochs,
        }
        batch_sampler = self.get_batch_sampler(Nlvr2BatchSampler, key, shuffle_per_epoch, dataset, **kwargs)
        self.num_batches_per_key[key] = batch_sampler.num_batches
        return batch_sampler


class Nlvr2BatchSampler(Sampler[List[int]]):
    """
    WARNING: FOR A GIVEN SEED AND EPOCH, IT WILL CREATE THE SAME BATCHES EVERY TIME.
    IT IS MEANT TO BE USED WITH `reload_dataloaders_every_n_epochs=1`, AS IT TAKES EPOCH AS AN ARGUMENT.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_epochs: int,
        num_batches: Optional[int] = None,
        seed: int = 0,
        shuffle_per_epoch: bool = True,
        epoch: int = 0,
        num_replicas: int = 1,
        rank: Optional[int] = None,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size)
            )
        self.batch_size = batch_size
        self.num_epochs = (
            num_epochs  # Needed in order to choose the minimal num_batches so every epoch would be of the same size
        )
        self.seed = seed
        self.shuffle_per_epoch = shuffle_per_epoch
        self.epoch = epoch

        self.num_replicas = num_replicas
        self.rank = rank

        self._example_indices_per_set = defaultdict(list)
        for i, set_id in enumerate(dataset["set_id"]):
            self._example_indices_per_set[set_id].append(i)
        max_set_size = max(len(v) for v in self._example_indices_per_set.values())
        if max_set_size > self.batch_size:
            raise ValueError(
                f"The maximum set size is {max_set_size}, which is greater than the batch size {self.batch_size}. Current behavior of _generate_batches will result in an infinite loop."
            )

        if num_batches is None:
            logger.info("Batch sampling for every epoch to get the number of batches per epoch")
            minimal_num_batches = len(dataset)
            for i in tqdm(range(num_epochs)):
                batches = list(
                    self._generate_batches(
                        minimal_num_batches - (minimal_num_batches % num_replicas),
                        _get_generator(seed, i, shuffle_per_epoch),
                    )
                )
                minimal_num_batches = min(minimal_num_batches, len(batches))
            logger.info(f"Setting the number of batches per epoch to {minimal_num_batches}")
            self.num_batches = minimal_num_batches - (minimal_num_batches % num_replicas)
        else:
            self.num_batches = num_batches

    def __iter__(self) -> Iterator[List[int]]:
        generator = _get_generator(self.seed, self.epoch, self.shuffle_per_epoch)
        batches = list(
            self._generate_batches(
                self.num_batches,
                generator,
            )
        )
        batches = batches[self.rank :: self.num_replicas]
        yield from batches

    def __len__(self) -> int:
        return int(self.num_batches / self.num_replicas)

    def _generate_batches(self, num_batches, generator):
        example_indices_per_set = deepcopy(self._example_indices_per_set)
        batch_counter = 0
        while len(example_indices_per_set) > 0:
            batch = []
            selected_set_indices = torch.randperm(len(example_indices_per_set), generator=generator)[: self.batch_size]
            set_ids = list(example_indices_per_set.keys())
            selected_set_ids = [set_ids[i] for i in selected_set_indices]
            for set_id in selected_set_ids:
                set_size = len(example_indices_per_set[set_id])
                if len(batch) + set_size <= self.batch_size:
                    batch.extend(example_indices_per_set[set_id])
                    del example_indices_per_set[set_id]
                else:
                    break
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                break


def _get_generator(seed, epoch, shuffle_per_epoch):
    generator = torch.Generator()
    generator.manual_seed(
        (seed + epoch) if shuffle_per_epoch else 0
    )  # If not shuffling per epoch, we should use the same order independently of the seed
    return generator
