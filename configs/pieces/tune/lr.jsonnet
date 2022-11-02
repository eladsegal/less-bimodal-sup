{
    trainer+: {
        auto_lr_find: true,
    },
    lr_find_kwargs+: {
        update_attr: false,
        mode: "exponential",
        early_stop_threshold: null,
        num_training: 2400,
        min_lr: 1e-7
    },
}
