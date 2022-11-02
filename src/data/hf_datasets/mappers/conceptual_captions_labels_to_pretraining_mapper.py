from src.data.hf_datasets.mappers import Mapper


class ConceptualCaptionsLabelsToPretrainingMapper(Mapper):
    @property
    def general_mapping_kwargs(self):
        return {}

    @staticmethod
    def general_mapping(examples, idxs, split, confidence_threshold=None, max_num_of_labels=None):
        indices_to_keep_list = [
            [
                i
                for i, confidence_score in enumerate(confidence_scores)
                if confidence_threshold is None or confidence_score > confidence_threshold
            ][:max_num_of_labels]
            for confidence_scores in examples["confidence_scores"]
        ]
        labels_list = [
            [examples["labels"][i][index] for index in indices_to_keep]
            for i, indices_to_keep in enumerate(indices_to_keep_list)
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
