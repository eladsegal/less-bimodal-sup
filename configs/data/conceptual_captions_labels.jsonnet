
local path = "ROOT_PATH_PARENT/data/conceptual_captions";
{
    global+: {
        debug: false,
        tags+: ["conceptual_captions_labels"],
        dataset_name: "conceptual_captions_labels",
    },

    dataset_container+: {
        _target_: "src.data.dataset_containers.ComplexDatasetContainer",
        main_key: "language",
        language+: {
            _target_: "src.data.dataset_containers.HfDatasetContainer",
            load_dataset_kwargs: {
                path: "src/data/hf_datasets/conceptual_captions_labels.py", 
                data_dir: path,
                data_files: {
                    train: path + "/Image_Labels_Subset_Train_GCC-Labels-training.tsv"
                },
                custom_split: {
                    train: "train" + (if $.global.debug then "[:2000]" else "[10502:]"),
                    validation: "train" + (if $.global.debug then "[:100]" else "[:10502]"),
                },
                index_to_url_tsv_path: path + "/Train_GCC-training.tsv",
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
