local path = "ROOT_PATH_PARENT/data/imagenet";
{
    global+: {
        debug: false,
        tags+: ["imagenet"],
        dataset_name: "imagenet",
    },

    # for contrastive learning
    json_predictions_writer+:: {
        settings: {
            raw_example_keys: [],
            preprocessed_example_keys: [],
            interim_batch_keys: [],
            interim_output_keys: [],
            batch_keys: [],
            output_keys: ["top5_image_ids_per_text", "top5_text_ids_per_image"],
            output_list: false,
        },
    },

    dataset_container+: {
        _target_: "src.data.dataset_containers.ComplexDatasetContainer",
        main_key: "language",
        language+: {
            _target_: "src.data.dataset_containers.HfDatasetContainer",
            load_dataset_kwargs: {
                path: "src/data/hf_datasets/imagenet.py", 
                data_dir: path,
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
                train: path, 
                validation: path, 
            },
            image_folder_kwargs: {
                extension: ".JPEG",
            },
        },
    },
}
