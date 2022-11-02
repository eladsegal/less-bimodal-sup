from typing import Union, Optional
import sys, os
from src.utils.logging import set_logger
import logging

set_logger()
import json
from collections import Counter, defaultdict
import re
from functools import cache
from zipfile import ZipFile

from tqdm import tqdm

import fire

from datasets import load_dataset, DownloadManager

import numpy as np

from utils.general import resolve_relative_paths, smart_open

logger = logging.getLogger(__name__)

NUM_OF_VALIDATION_IMAGES = 1000  # when validation_image_file_names_path is not used


class VqaPreparation:
    @staticmethod
    def create_train_and_validation(
        train,
        train_annotations,
        validation,
        validation_annotations,
        output_dir,
        validation_image_file_names_path=None,
        seed=42,
    ):
        train = resolve_relative_paths(train)
        train_annotations = resolve_relative_paths(train_annotations)
        validation = resolve_relative_paths(validation)
        validation_annotations = resolve_relative_paths(validation_annotations)
        output_dir = output_dir
        validation_image_file_names_path = validation_image_file_names_path

        logger.info("Loading train_validation_dataset")
        train_validation_dataset = load_vqa(
            {
                "train": train,
                "train_annotations": train_annotations,
                "validation": validation,
                "validation_annotations": validation_annotations,
            }
        )
        logger.info("Loading validation_dataset")
        validation_dataset = load_vqa(
            {
                "validation": validation,
                "validation_annotations": validation_annotations,
            }
        )

        logger.info("Preprocessing train_validation_dataset answers")
        train_validation_dataset = preprocess_answers(train_validation_dataset)
        logger.info("Preprocessing validation_dataset answers")
        validation_dataset = preprocess_answers(validation_dataset)

        logger.info("Creating answer labels")
        ans2label, label2ans = get_answers_labels(train_validation_dataset)
        with smart_open(os.path.join(output_dir, "ans2label.json"), "w") as f:
            json.dump(ans2label, f)
        with smart_open(os.path.join(output_dir, "label2ans.json"), "w") as f:
            json.dump(label2ans, f)

        logger.info("Filtering train_validation_dataset by available answers")
        filtered_train_validation_dataset = filter_dataset_by_available_answers(train_validation_dataset, ans2label)

        logger.info("Making the new train and validation splits mutually exclusive")
        new_train_dataset, new_validation_dataset, validation_image_file_names = mutually_exclude(
            filtered_train_validation_dataset, validation_dataset, validation_image_file_names_path, seed
        )

        logger.info("Saving validation_image_file_names")
        if validation_image_file_names_path is None:
            with smart_open(os.path.join(output_dir, "validation_image_file_names.json"), mode="w") as f:
                json.dump(list(validation_image_file_names), f)

        datasets = {
            "train": new_train_dataset,
            "validation": new_validation_dataset,
        }
        for split in ["train", "validation"]:
            logger.info(f"Saving new {split} dataset split")
            datasets[split].to_json(os.path.join(output_dir, f"{split}.jsonl"))

    @staticmethod
    def create_karpathy_split(
        train,
        train_annotations,
        validation,
        validation_annotations,
        output_dir,
        ans2label_path,
    ):
        train = resolve_relative_paths(train)
        train_annotations = resolve_relative_paths(train_annotations)
        validation = resolve_relative_paths(validation)
        validation_annotations = resolve_relative_paths(validation_annotations)
        output_dir = output_dir

        with open(ans2label_path) as f:
            ans2label = json.load(f)

        dl_manager = DownloadManager()

        karpathy_split = json.loads(
            ZipFile(
                dl_manager.download("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip")
            ).read("dataset_coco.json")
        )["images"]
        images_per_split = defaultdict(set)
        source_to_target_split = {
            "train": "train",
            "val": "validation",
            "restval": "train",
            "test": "test",
        }
        for image in karpathy_split:
            images_per_split[source_to_target_split[image["split"]]].add(image["filename"])

        logger.info("Loading train_validation_dataset")
        train_validation_dataset = load_vqa(
            {
                "train": train,
                "train_annotations": train_annotations,
                "validation": validation,
                "validation_annotations": validation_annotations,
            }
        )

        logger.info("Preprocessing train_validation_dataset answers")
        train_validation_dataset = preprocess_answers(train_validation_dataset)

        datasets = {}
        for split, images in images_per_split.keys():
            datasets[split] = train_validation_dataset.filter(
                lambda example: os.path.basename(example["image_file_name"]) in images,
            )
            if split != "test":
                datasets[split] = filter_dataset_by_available_answers(datasets[split], ans2label)

        os.makedirs(output_dir, exist_ok=True)
        for split, dataset in datasets.items():
            logger.info(f"Saving new {split} dataset split")
            dataset.to_json(os.path.join(output_dir, f"{split}.jsonl"))

    @staticmethod
    def convert(split, output_dir, questions_path, annotations_path=None):
        questions_path = resolve_relative_paths(questions_path)
        annotations_path = resolve_relative_paths(annotations_path)
        output_dir = resolve_relative_paths(output_dir)

        data_files = {
            split: questions_path,
        }
        if annotations_path is not None:
            data_files[f"{split}_annotations"] = annotations_path

        if split == "test":
            dataset = load_vqa(data_files, features_granularity="test")
        else:
            dataset = load_vqa(data_files)

        if split != "test":
            dataset = preprocess_answers(dataset)

        os.makedirs(output_dir, exist_ok=True)
        dataset.to_json(os.path.join(output_dir, f"{split}.jsonl"))

    @staticmethod
    def create_analysis_subset(
        subset_name: str, output_dir: str, subset_size: Optional[Union[float, int]] = None, seed: int = 42
    ):
        train_dataset = load_vqa(
            {
                "train": resolve_relative_paths("data/vqa/train_lxmert.jsonl.zip"),
            },
            original_format=False,
        )

        from tokenizers.pre_tokenizers import Whitespace

        whitespace_tokenizer = Whitespace()
        import random

        if subset_name == "tdiuc":
            with open("data/vqa/tdiuc_subset_ids.json") as f:
                subset_ids = set(json.load(f))
        elif subset_name == "max_length" or subset_name == "min_length":
            tokenized_lengths = []
            for example in tqdm(train_dataset, desc="Calculating tokens per question"):
                tokenized_lengths.append(len(whitespace_tokenizer.pre_tokenize_str(example["question"])))

            reverse = False if subset_name == "min_length" else True
            subset_indices = sorted(
                range(len(tokenized_lengths)), key=lambda k: tokenized_lengths[k], reverse=reverse
            )[:subset_size]
            subset_ids = set([train_dataset[i]["id"] for i in subset_indices])

            logger.info(
                f"Average Number of length in tokens in the train dataset: {sum(tokenized_lengths) / len(tokenized_lengths)}"
            )
            logger.info(
                f"Average Number of length in tokens in the subset: {sum(tokenized_lengths[i] for i in subset_indices) / len(subset_indices)}"
            )
        elif subset_name == "max_vocab" or subset_name == "min_vocab":
            rng = np.random.default_rng(seed)

            tokens_per_question = []
            for example in tqdm(train_dataset, desc="Calculating tokens per question"):
                tokens = set([t[0] for t in whitespace_tokenizer.pre_tokenize_str(example["question"])])
                tokens_per_question.append(tokens)

            all_tokens = set(token for tokens in tokens_per_question for token in tokens)

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
                    addition_size = len(tokens_per_question[i] - subset_tokens)
                    if op(addition_size, best_intersection_size) == addition_size:
                        best_intersection_size = addition_size
                        best_i = i
                subset_indices.add(best_i)
                subset_tokens.update(tokens_per_question[best_i])
                pbar.update(1)
            pbar.close()
            subset_ids = set([train_dataset[i]["id"] for i in subset_indices])

            random_subset_indices = rng.choice(len(train_dataset), size=subset_size, replace=False)
            random_train_tokens = set()
            for i in random_subset_indices:
                random_train_tokens.update(tokens_per_question[i])

            logger.info(f"Number of unique tokens in the train dataset: {len(all_tokens)}")
            logger.info(f"Number of unique tokens in the random train dataset subet: {len(random_train_tokens)}")
            logger.info(f"Number of unique tokens in the subset: {len(subset_tokens)}")

        full_name = f"{subset_name}_subset" if subset_size is None else f"{subset_name}_subset_{subset_size}"
        subset = train_dataset.filter(
            lambda examples, idxs: [id_ in subset_ids for id_ in examples["id"]], batched=True, with_indices=True
        )
        os.makedirs(output_dir, exist_ok=True)
        subset.to_json(os.path.join(output_dir, f"{full_name}.jsonl"))


def load_vqa(data_files, *, original_format=True, features_granularity=None):
    kwargs = {
        "path": "src/data/hf_datasets/vqa.py",
        "data_files": data_files,
        "split": "+".join(split for split in data_files if not split.endswith("annotations")),
        "original_format": original_format,
        "group_non_train_by_image": False,
    }
    if features_granularity is not None:
        kwargs["features_granularity"] = features_granularity
    return load_dataset(**kwargs)


def preprocess_answers(dataset):
    dataset = dataset.map(
        lambda examples: {
            "multiple_choice_answer": [
                preprocess_answer(multiple_choice_answer)
                for multiple_choice_answer in examples["multiple_choice_answer"]
            ],
            "answers": [[preprocess_answer(answer) for answer in answers] for answers in examples["answers"]],
        },
        # num_proc=max(1, 1),
        batched=True,
    )
    return dataset


def get_answers_labels(raw_train_validation_dataset):
    ans2label = {}
    label2ans = []
    all_majority_answers = []
    for example in tqdm(raw_train_validation_dataset, desc=f"Creating answer labels"):
        all_majority_answers.append(example["multiple_choice_answer"])
    counter = {k: v for k, v in Counter(all_majority_answers).items() if v >= 9}
    ans2label.update({k: i for i, k in enumerate(counter.keys())})
    label2ans.extend(counter.keys())

    logger.info(f"Got {len(ans2label)} answers")

    return ans2label, label2ans


def filter_dataset_by_available_answers(dataset, ans2label):
    idxs_to_keep = set()
    for i, answers in enumerate(dataset["answers"]):
        answer_count = Counter(answers)
        has_available_answer = False
        for answer in answer_count:
            if answer not in ans2label:
                continue
            has_available_answer = True
        if has_available_answer:
            idxs_to_keep.add(i)
    return dataset.filter(lambda examples, idxs: [i in idxs_to_keep for i in idxs], batched=True, with_indices=True)


def mutually_exclude(train_dataset, validation_dataset, validation_image_file_names_path=None, seed=None):
    if validation_image_file_names_path is not None:
        with open(validation_image_file_names_path, mode="r") as f:
            validation_image_file_names = dict.fromkeys(json.load(f))
    else:
        validation_image_file_names = list(dict.fromkeys(validation_dataset["image_file_name"]))
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(validation_image_file_names), NUM_OF_VALIDATION_IMAGES, replace=False)
        validation_image_file_names = dict.fromkeys([validation_image_file_names[index] for index in indices])

    logger.info("VQA Image-based filtering - Split sizes before:")
    logger.info(f"train: {len(train_dataset)}")
    logger.info(f"validation: {len(validation_dataset)}")

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
    }
    for split, include_images in [("train", False), ("validation", True)]:
        datasets[split] = datasets[split].filter(
            lambda example: (example["image_file_name"] in validation_image_file_names) == include_images,
        )
    new_train_dataset = datasets["train"]
    new_validation_dataset = datasets["validation"]

    logger.info(f"train: {len(new_train_dataset)}")
    logger.info(f"validation: {len(new_validation_dataset)}")

    return new_train_dataset, new_validation_dataset, validation_image_file_names


# START - Copied from https://github.com/allenai/allennlp-models/blob/main/allennlp_models/vision/dataset_readers/utils.py,
contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile(r"(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def process_punctuation(inText: str) -> str:
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (comma_strip.search(inText) is not None):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(input: str) -> str:
    output = []
    for word in input.lower().split():
        word = manual_map.get(word, word)
        if word not in articles:
            output.append(word)
        else:
            pass
    for index, word in enumerate(output):
        if word in contractions:
            output[index] = contractions[word]
    return " ".join(output)


@cache
def preprocess_answer(answer: str) -> str:
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(",", "")
    return answer


# END


if __name__ == "__main__":
    fire.Fire(VqaPreparation)
