{
    global+: {
        debug: false,
        tags+: [],
    },

    wandb:: {
        project: "<??>",
        tags: $.global.tags,
        entity: "<??>"
    },

    wandb_logger:: if !($.global.debug) then [{
        _target_: "src.pl.loggers.BetterWandbLogger",
    } + $.wandb] else [],
    tensorboard_logger:: [{
        _target_: "pytorch_lightning.loggers.TensorBoardLogger",
        save_dir: ".",
        name: "tensorboard",
        version: "",
    }],
    trainer+: {
        logger: $.wandb_logger,
    }
}
