from collections import defaultdict
import numpy as np

from src.data.hf_datasets.mappers import Mapper


class ImageNetToVqaMapper(Mapper):
    @property
    def general_mapping_kwargs(self):
        return {
            "dataset": None,
        }

    @staticmethod
    def general_mapping(examples, idxs, split, dataset):
        num_of_examples = len(idxs)

        class_to_indices = defaultdict(list)
        for i, class_id in enumerate(dataset["class_id"]):
            class_to_indices[class_id].append(i)

        pos_len = num_of_examples // 2
        neg_len = num_of_examples - pos_len
        neg_indices = set(np.random.choice(num_of_examples, neg_len, replace=False).tolist())

        result = {
            "id": [None for _ in range(num_of_examples)],
            "text_input": [None for _ in range(num_of_examples)],
            "answers": [None for _ in range(num_of_examples)],
            "image_file_name": [None for _ in range(num_of_examples)],
        }
        for i in range(num_of_examples):
            result["id"][i] = examples["id"][i]
            result["text_input"][i] = "Is this a photo of " + examples["class_text"][i] + "?"

            if i in neg_indices:
                result["answers"][i] = ["no"] * 10

                # Select a random image from a different class
                class_id = examples["class_id"][i]
                while class_id == examples["class_id"][i]:
                    class_id = np.random.choice(list(class_to_indices.keys()))
                result["image_file_name"][i] = dataset[np.random.choice(class_to_indices[class_id]).item()][
                    "image_file_name"
                ]
            else:
                result["answers"][i] = ["yes"] * 10
                result["image_file_name"][i] = examples["image_file_name"][i]
        return result
