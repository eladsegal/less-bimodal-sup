{
    global+: {
        debug: false,
        tags+: ["nlvr2"],
        dataset_name: "nlvr2",
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
            raw_example_keys: ["sentence", "image_file_name_0", "image_file_name_1", "label"],
            preprocessed_example_keys: [],
            interim_batch_keys: [],
            interim_output_keys: [],
            batch_keys: [],
            output_keys: ["predicted_answer"],
            output_list: false,
            enabled: true,
        },
    },

    dataset_container+: {
        _target_:  "src.data.dataset_containers.ComplexDatasetContainer",
        main_key: "language",
        language+: {
            _target_: "src.data.dataset_containers.HfDatasetContainer",
            load_dataset_kwargs: {
                path: "src/data/hf_datasets/nlvr2.py", 
                features_granularity: "full",
                // download_mode: "reuse_cache_if_exists",
                custom_split: {
                    train: "train" + (if $.global.debug then "[:1000]" else ""), 
                    validation: "dev" + (if $.global.debug then "[:1000]" else "")
                },
            },
        },
        vision+: {
            _target_: "src.data.dataset_containers.ImagesDatasetContainer",
            filtering_dataset_container: "language",
            images_base_dir: {
                train: "ROOT_PATH_PARENT/data/nlvr2",
                validation: "ROOT_PATH_PARENT/data/nlvr2",
            },
        },
    },

    metrics+: {
        "accuracy": {
            _target_: "src.metrics.Accuracy",
            name: "accuracy",
        },
        "consistency": {
            _target_: "src.metrics.Nlvr2Consistency",
            name: "consistency",
        },
    },

    datamodule+: {
        artifacts_dir: "nlvr2",
        ans2label_current_path: "data/nlvr2/ans2label.json",
        num_of_images: 2,
    },
}
