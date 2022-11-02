local loggers = import 'pieces/loggers/default.jsonnet';
local json_predictions_writer = import 'pieces/callbacks/json_predictions_writer.jsonnet';

{
    callbacks_dict:: {
        checkpoint: {
            _target_: "src.pl.callbacks.BetterModelCheckpoint",
            filename: "training_" + (if $.global.val_check_interval == null then "epoch_{epoch}" else "epoch_{epoch}_step_{step}"),
            monitor: "val/loss",
            mode: "min",
            specific_weights_to_save: $.global.specific_weights_to_save,
        },
        progress_bar: {
            _target_: "src.pl.callbacks.BetterTQDMProgressBar",
            refresh_rate: 1,
        },
        upload_output: {
            _target_: "src.callbacks.UploadOutput",
            interval: 300,
        },
        /*find_unused_parameters: {
            _target_: "src.callbacks.FindUnusedParameters",
        },*/
        [if (std.isBoolean($.trainer.logger) && $.trainer.logger) || (std.isArray($.trainer.logger) && std.length($.trainer.logger) > 0) then 
        "learning_rate_monitor" else null]: {
            _target_: "pytorch_lightning.callbacks.LearningRateMonitor",
            logging_interval: "step",
        },
    },

    global: {
        debug: false,
        dataset_name: "",
        objective_format: "",
        val_check_interval: null,
        specific_weights_to_save: null,
        seed: 42,
    },

    dataset_container+: {
        additional_kwargs+: {
            seed: $.global.seed,
            language+: {
                map_num_proc: 6,
            },
            map_num_proc: 6,
        },
    },
    metrics+: {},
    datamodule+: {
        _target_: "<??>",
        objective_format: $.global.objective_format,
        batch_size: "<??>",
        dataloader_num_workers: 6,
        map_num_proc: 6,
        seed: $.global.seed,
    },
    trainer+: {
        _target_: "src.pl.MyTrainer",
        gpus: 1,
        precision: 16,
        callbacks: [$.callbacks_dict[callback_name], for callback_name in std.objectFields($.callbacks_dict)],
        [if $.global.val_check_interval != null then "val_check_interval" else null]: $.global.val_check_interval,
        max_epochs: "<??>",
        accumulate_grad_batches: 1,
        [if true then "num_sanity_val_steps" else null]: 0, // Changes the randomness of the train and val data loaders when on
        reload_dataloaders_every_n_epochs: 1, # TODO: remove this when fault tolerant training is fixed. Currently it's needed to have the right epoch for the dataloader init
    },
    model+: {
        _target_: "<??>",
        objective_format: $.global.objective_format,
        optimizer: {
            _target_: "src.utils.instantiation.Instantiation",
            klass: {
                _target_: "src.utils.instantiation.get_class_type",
                full_class_name: "transformers.AdamW",
            },
            lr: "<??>",
        },
    },
} + loggers + json_predictions_writer
