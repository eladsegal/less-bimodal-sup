import os
import json

import fire
from datasets import load_dataset
from tqdm import tqdm
from src.data.dataset_containers.hf_dataset_container import HfDatasetContainer
from utils.general import smart_open

from utils.jsonnet import evaluate


def prepare_ans2label(dataset_config_path: str, answer_column: str, add_unknown: bool, output_dir: str):
    if not isinstance(add_unknown, bool):
        raise ValueError("add_unknown must be a boolean (True or False)")

    cfg = evaluate(dataset_config_path)
    load_dataset_kwargs = cfg["dataset_container"]["language"]["load_dataset_kwargs"]
    dataset_container = HfDatasetContainer(load_dataset_kwargs=load_dataset_kwargs)
    dataset_container.setup()

    ans2label, label2ans = _get_answers_labels(
        dataset_split=dataset_container.raw_datasets["train"], answer_column=answer_column, add_unknown=add_unknown
    )
    with smart_open(os.path.join(output_dir, "ans2label.json"), "w") as f:
        json.dump(ans2label, f)
    with smart_open(os.path.join(output_dir, "label2ans.json"), "w") as f:
        json.dump(label2ans, f)


def _get_answers_labels(dataset_split, answer_column: str, add_unknown: bool):
    ans2label_current = {}
    label2ans_current = []

    if add_unknown:
        ans2label_current["UNKNOWN"] = len(ans2label_current)
        label2ans_current.append("UNKNOWN")
    for example in tqdm(dataset_split, desc="Creating answer labels"):
        answer = example[answer_column]
        if answer not in ans2label_current:
            ans2label_current[answer] = len(ans2label_current)
            label2ans_current.append(answer)

    return ans2label_current, label2ans_current


if __name__ == "__main__":
    fire.Fire(prepare_ans2label)
