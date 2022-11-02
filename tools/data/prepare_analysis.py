from typing import Optional, Union
from collections.abc import Mapping

import os
import sys
import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
import fire
from tokenizers.pre_tokenizers import Whitespace
from hydra.utils import instantiate
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset

from utils.jsonnet import evaluate
from utils.general import resolve_relative_paths

import logging

logger = logging.getLogger(__name__)


class Analysis:
    @staticmethod
    def get_stats(dataset: Union[str, Dataset], text_field, cache_path=None):
        if isinstance(dataset, str):
            dataset = hf_load_dataset("json", data_files=dataset, split="train")
        tokens_per_example = _get_tokens_per_example(dataset, text_field, cache_path)
        tokenized_lengths = _get_tokenized_lengths(tokens_per_example)
        token_set = _get_token_set(tokens_per_example)

        average_length = sum(tokenized_lengths) / len(tokenized_lengths)
        num_of_unique_tokens = len(token_set)
        return average_length, num_of_unique_tokens

    @staticmethod
    def create_analysis_subset(
        config_path: str,
        subset_name: str,
        output_dir: str,
        text_field: str,
        subset_size: Optional[Union[float, int]] = None,
        seed: int = 42,
    ):
        train_dataset = _load_dataset(config_path, "train")
        rng = np.random.default_rng(seed)

        if subset_name == "tdiuc":
            with open("data/vqa/tdiuc_subset_ids.json") as f:
                subset_ids = set(json.load(f))

        cache_path = os.path.join(output_dir, Path(config_path).stem + "_tokens.json")
        tokens_per_example = _get_tokens_per_example(train_dataset, text_field, cache_path)

        tokenized_lengths = _get_tokenized_lengths(tokens_per_example)

        if subset_name == "max_length" or subset_name == "min_length":
            reverse = False if subset_name == "min_length" else True
            subset_indices = sorted(
                range(len(tokenized_lengths)), key=lambda k: tokenized_lengths[k], reverse=reverse
            )[:subset_size]
            subset_ids = set([train_dataset[i]["id"] for i in subset_indices])
        elif subset_name == "max_vocab" or subset_name == "min_vocab":
            token_set_per_example = []
            for tokens in tokens_per_example:
                token_set_per_example.append(set(tokens))

            op = max if subset_name == "max_vocab" else min
            RANDOM_SELECTION_SIZE = 10000
            subset_indices = set()
            subset_tokens = set()
            pbar = tqdm(total=subset_size)
            while len(subset_indices) < subset_size:
                # This is ok when getting only a very small subset. The chance of getting selected indices is low.
                selection_subset = rng.choice(len(train_dataset), size=RANDOM_SELECTION_SIZE, replace=False)
                selection_subset = [i.item() for i in selection_subset if i not in subset_indices]

                best_intersection_size = float("-inf") if subset_name == "max_vocab" else float("inf")
                best_i = None
                for i in selection_subset:
                    addition_size = len(token_set_per_example[i] - subset_tokens)
                    if op(addition_size, best_intersection_size) == addition_size:
                        best_intersection_size = addition_size
                        best_i = i
                subset_indices.add(best_i)
                subset_tokens.update(token_set_per_example[best_i])
                pbar.update(1)
            pbar.close()
            subset_ids = set([train_dataset[i]["id"] for i in subset_indices])

        full_name = f"{subset_name}_subset" if subset_size is None else f"{subset_name}_subset_{subset_size}"
        subset = train_dataset.filter(
            lambda examples, idxs: [id_ in subset_ids for id_ in examples["id"]], batched=True, with_indices=True
        )

        random_subset_indices = rng.choice(len(train_dataset), size=subset_size, replace=False)
        random_subset_ids = set([train_dataset[i.item()]["id"] for i in random_subset_indices])
        random_subset = train_dataset.filter(
            lambda examples, idxs: [id_ in random_subset_ids for id_ in examples["id"]],
            batched=True,
            with_indices=True,
        )

        total_average_length, total_num_of_unique_tokens = Analysis.get_stats(train_dataset, text_field, cache_path)
        random_average_length, random_num_of_unique_tokens = Analysis.get_stats(random_subset, text_field)
        average_length, num_of_unique_tokens = Analysis.get_stats(subset, text_field)

        logger.info(f"Average Number of length in tokens in the train dataset: {total_average_length}")
        logger.info(f"Average Number of length in tokens in the random subset: {random_average_length}")
        logger.info(f"Average Number of length in tokens in the subset: {average_length}")

        logger.info(f"Number of unique tokens in the train dataset: {total_num_of_unique_tokens}")
        logger.info(f"Number of unique tokens in the random subset: {random_num_of_unique_tokens}")
        logger.info(f"Number of unique tokens in the subset: {num_of_unique_tokens}")

        if subset_name == "tdiuc":
            output_dir_with_seed = output_dir
        else:
            output_dir_with_seed = os.path.join(output_dir, str(seed))
        os.makedirs(output_dir_with_seed, exist_ok=True)
        subset.to_json(os.path.join(output_dir_with_seed, f"{full_name}.jsonl"))


def _load_dataset(config_path, split):
    cfg = evaluate(config_path)
    load_dataset_kwargs = cfg["dataset_container"]["language"]["load_dataset_kwargs"]

    if load_dataset_kwargs["path"].endswith(".py"):
        load_dataset_kwargs["path"] = resolve_relative_paths(load_dataset_kwargs["path"])

    for kwarg in ["data_files", "data_dir"]:
        load_dataset_kwargs[kwarg] = (
            resolve_relative_paths(load_dataset_kwargs[kwarg]) if load_dataset_kwargs.get(kwarg) is not None else None
        )

    _split = load_dataset_kwargs.pop("custom_split") or load_dataset_kwargs.pop("split")
    if isinstance(_split, Mapping):
        _split = _split["train"]

    return hf_load_dataset(**load_dataset_kwargs, split=_split)


def _get_tokenized_lengths(tokens_per_example):
    tokenized_lengths = []
    for tokens in tokens_per_example:
        tokenized_lengths.append(len(tokens))
    return tokenized_lengths


def _get_token_set(tokens_per_example):
    token_set = set()
    for tokens in tokens_per_example:
        token_set.update(tokens)
    return token_set


def _get_tokens_per_example(dataset, text_field, cache_path=None):
    if cache_path is not None and os.path.isfile(cache_path):
        with open(cache_path) as f:
            tokens_per_example = json.load(f)
    else:
        whitespace_tokenizer = Whitespace()
        tokens_per_example = []
        for example in tqdm(dataset, desc=f"Calculating tokens per {text_field}"):
            tokens = [t[0] for t in whitespace_tokenizer.pre_tokenize_str(example[text_field])]
            tokens_per_example.append(tokens)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(tokens_per_example, f)
    return tokens_per_example


if __name__ == "__main__":
    fire.Fire(Analysis)
