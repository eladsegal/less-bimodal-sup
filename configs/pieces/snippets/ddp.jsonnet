{
    trainer+: {
        strategy: {
            _target_: "pytorch_lightning.strategies.ddp.DDPStrategy",
            find_unused_parameters: false
        },
    },
}
