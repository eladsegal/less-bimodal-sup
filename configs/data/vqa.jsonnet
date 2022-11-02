{
    global+: {
        debug: false,
        tags+: ["vqa"],
        dataset_name: "vqa",
    },

    callbacks_dict+:: {
        checkpoint+: {
            monitor: "val/vqa_score",
            mode: "max",
        }
    },

    // TODO: This should be defined as a preset for a (dataset, model, [datamodule]) tuple,
    // but overridable from the config
    json_predictions_writer+:: {
        settings: {
            _target_: "src.callbacks.json_predictions_writer.JsonPredictionsSettings",
            raw_example_keys: ["int(id)->question_id"],
            preprocessed_example_keys: [],
            interim_batch_keys: [],
            interim_output_keys: [],
            batch_keys: [],
            output_keys: ["predicted_answer->answer"],
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
                path: "src/data/hf_datasets/vqa.py", 
                data_files: {
                    train: "data/vqa/train_lxmert.jsonl.zip",
                    validation: "data/vqa/validation_lxmert.jsonl",
                },
                // download_mode: "reuse_cache_if_exists",
                custom_split: {
                    train: "train" + (if $.global.debug then "[:100]" else ""),
                    validation: "validation" + (if $.global.debug then "[:1000]" else ""),
                },
                original_format: false,
                group_non_train_by_image: true,
            },
        },
        vision+: {
            _target_: "src.data.dataset_containers.ImagesDatasetContainer",
            filtering_dataset_container: "language",
            images_base_dir: {
                train: "ROOT_PATH_PARENT/data/vqa", 
                validation: "ROOT_PATH_PARENT/data/vqa", 
            },
        },
    },

    metrics+: {
        "vqa_score": {
            _target_: "src.metrics.VqaScore",
            name: "vqa_score",
        },
    },

    datamodule+: {
        artifacts_dir: "vqa",
        ans2label_current_path: "data/vqa/ans2label.json",
        num_of_images: 1,
    },
}
