{
    global+: {
        debug: false,
        tags+: ["gqa"],
        dataset_name: "gqa",
    },

    callbacks_dict+:: {
        checkpoint+: {
            monitor: "val/accuracy",
            mode: "max",
        }
    },

    json_predictions_writer+:: {
        settings: {
            _target_: "src.callbacks.json_predictions_writer.JsonPredictionsSettings",
            raw_example_keys: ["id->questionId"],
            preprocessed_example_keys: [],
            interim_batch_keys: [],
            interim_output_keys: [],
            batch_keys: [],
            output_keys: ["predicted_answer->prediction"],
            output_list: true,
            enabled: true,
        },
    },

    dataset_container+: {
        _target_: "src.data.dataset_containers.ComplexDatasetContainer",
        main_key: "language",
        language+: {
            _target_: "src.data.dataset_containers.HfDatasetContainer",
            load_dataset_kwargs: {
                path: "src/data/hf_datasets/gqa.py", 
                data_files: {
                    train: ["../data/gqa/train_balanced_questions.json"],
                    validation: ["../data/gqa/testdev_balanced_questions.json"],
                },
                // download_mode: "reuse_cache_if_exists",
                custom_split: {
                    train: "train" + (if $.global.debug then "[:100]" else ""),
                    validation: "validation" + (if $.global.debug then "[:1000]" else ""),
                },
            },
        },
        vision+: {
            _target_: "src.data.dataset_containers.ImagesDatasetContainer",
            filtering_dataset_container: "language",
            images_base_dir: {
                train: "ROOT_PATH_PARENT/data/gqa/images",
                validation: "ROOT_PATH_PARENT/data/gqa/images",
            },
        },
    },

    metrics+: {
        "accuracy": {
            _target_: "src.metrics.Accuracy",
            name: "accuracy",
        },
    },

    datamodule+: {
        artifacts_dir: "gqa",
        ans2label_current_path: "data/gqa/ans2label.json",
        num_of_images: 1,
    },
}
