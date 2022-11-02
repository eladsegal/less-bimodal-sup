
local path = "ROOT_PATH_PARENT/data/conceptual_captions";
{
    global+: {
        debug: false,
        tags+: ["conceptual_captions"],
        dataset_name: "conceptual_captions",
    },

    dataset_container+: {
        _target_: "src.data.dataset_containers.ComplexDatasetContainer",
        main_key: "language",
        language+: {
            _target_: "src.data.dataset_containers.HfDatasetContainer",
            load_dataset_kwargs: {
                path: "src/data/hf_datasets/conceptual_captions.py", 
                data_dir: path,
                data_files: {
                    train: path + "/Train_GCC-training.tsv",
                    validation: path + "/Validation_GCC-1.1.0-Validation.tsv",
                },
                custom_split: {
                    train: "train" + (if $.global.debug then "[:2000]" else "[:2841600]"),
                    validation: "validation" + (if $.global.debug then "[:100]" else ""),
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
                extension: ".jpg",
            },
        },
    },
}
