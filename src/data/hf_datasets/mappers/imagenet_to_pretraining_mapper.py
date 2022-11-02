from src.data.hf_datasets.mappers import Mapper


class ImageNetToPretrainingMapper(Mapper):
    @property
    def general_mapping_kwargs(self):
        return {}

    @staticmethod
    def general_mapping(examples, idxs, split, max_num_of_labels=None):
        labels_list = [
            [label.strip() for label in class_text.split(",")[:max_num_of_labels]]
            for class_text in examples["class_text"]
        ]

        result = {k: v for k, v in examples.items() if k in ["id", "image_file_name"]}
        result.update(
            {
                "caption": [", ".join(labels) for labels in labels_list],
                "is_caption_from_labels": [True] * len(examples["id"]),
                "labels": labels_list,
            }
        )
        return result
