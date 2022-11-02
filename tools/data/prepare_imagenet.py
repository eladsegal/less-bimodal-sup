from src.utils.logging import set_logger
import logging

set_logger()
from collections import defaultdict
import fire
import os

import numpy as np

from datasets import load_dataset

import logging

logger = logging.getLogger(__name__)


class ImageNetPreparation:
    @staticmethod
    def balanced_collapse(data_dir, seed=42, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join("data", "imagenet", "manual_balanced_class_collapse", str(seed))
            os.makedirs(output_dir, exist_ok=True)

        rng = np.random.default_rng(seed)

        original_datasets = load_dataset(
            "src/data/hf_datasets/imagenet.py", data_dir=data_dir, manual_class_collapse=True
        )
        for split in original_datasets:
            original_dataset = original_datasets[split]

            example_indices_per_class = defaultdict(list)
            for i, class_id in enumerate(original_dataset["class_id"]):
                example_indices_per_class[class_id].append(i)
            min_num_examples = min([len(v) for v in example_indices_per_class.values()])

            new_dataset_indices = []
            for class_id, indices in example_indices_per_class.items():
                selected_indices = rng.choice(len(indices), min_num_examples, replace=False)
                new_dataset_indices.extend(indices[i] for i in selected_indices)

            new_dataset = original_dataset.select(new_dataset_indices)
            output_file = os.path.join(output_dir, f"{split}.jsonl")
            logger.info(f"Saving to {output_file}")
            new_dataset.to_json(output_file)


if __name__ == "__main__":
    fire.Fire(ImageNetPreparation)
