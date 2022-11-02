local config = import "imagenet.jsonnet";

config {
    dataset_container+: {
        language+: {
            load_dataset_kwargs: {
                path: "src/data/hf_datasets/imagenet_from_files.py",
                data_files: {
                    train: "data/imagenet/manual_balanced_class_collapse/43/train.jsonl",
                    validation: "data/imagenet/manual_balanced_class_collapse/43/validation.jsonl",
                },
                custom_split: {
                    train: "train" + (if $.global.debug then "[:2000]" else ""),
                    validation: "validation" + (if $.global.debug then "[:2000]" else ""),
                },
            },
        },
    },
}
