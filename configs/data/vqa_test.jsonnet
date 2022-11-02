local config = import "vqa.jsonnet";

config {
    dataset_container+: {
        language+: {
            load_dataset_kwargs+: {
                features_granularity: "test",
                data_files: {
                    test: ["data/vqa/test.jsonl"],
                },
                custom_split: {
                    test: "test", 
                },
            },
        },
        vision+: {
            images_base_dir: {
                test: "ROOT_PATH_PARENT/data/coco", 
            },
        },
    },
}
