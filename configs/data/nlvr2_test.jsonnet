local config = import "nlvr2.jsonnet";

config {
    dataset_container+: {
        language+: {
            load_dataset_kwargs+: {
                features_granularity: "test",
                custom_split: {
                    test: "test1", 
                },
            },
        },
        vision+: {
            images_base_dir: {
                test: "ROOT_PATH_PARENT/data/nlvr2",
            },
        },
    },

}
