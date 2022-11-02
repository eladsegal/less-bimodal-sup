local config = import "../../base.jsonnet";

local freezing_strategies = {
    none: [],
    vision: ["_encoder.vision_model"],
    language: ["_encoder.language_model"],
    encoders: ["_encoder.language_model", "_encoder.vision_model"],
};

config {
    wandb+:: {
        group: "pretraining",
    },

    global+: {
        tags+: ["pretraining"],
        image_size: 224,
        pretrained_language_model: "roberta-base",
        pretrained_vision_model: "google/vit-base-patch16-224-in21k",
        freezing_strategy: "none",
        output_predictions: false,
        tasks: ["itm", "mlm"],
    },

    callbacks_dict+:: {
        freezer+: {
            _target_: "src.callbacks.Freezer",
            fqns: freezing_strategies[$.global.freezing_strategy],
        },
        checkpoint+: {
            filename: "training_" + "step_{step}",
            save_also_top_k_weight_only: -1,
            save_top_k: 1,  // due to modifications to the PL model_checkpoint callback only full epochs will be saved
        },
    },

    dataset_container+: {
        additional_kwargs+: {
            language+: {
                target_datamodule: $.datamodule._target_,
            },
            vision+: {
                image_transform: {
                    train: {
                        _target_: "src.data.datasets.utils.image.transforms.center_crop_transform_randaug",
                        size: $.global.image_size,
                        pretrained_vision_model: $.global.pretrained_vision_model,
                    },
                    validation: {
                        _target_: "src.data.datasets.utils.image.transforms.center_crop_transform",
                        size: $.global.image_size,
                        pretrained_vision_model: $.global.pretrained_vision_model,
                    },
                },
            },
        },
    },

    datamodule+: {
        _target_: "src.data.datamodules.pretraining_datamodule.PretrainingDataModule",
        tokenizer: {
            _target_: "transformers.AutoTokenizer.from_pretrained",
            use_fast: true,
            pretrained_model_name_or_path: $.global.pretrained_language_model,
            model_max_length: 50,
        },
        feature_extractor: {
            _target_: "transformers.AutoFeatureExtractor.from_pretrained",
            pretrained_model_name_or_path: $.global.pretrained_vision_model,
            do_resize: false,
            do_normalize: false,
            do_center_crop: false,
        },
        batch_size: "<??>",
        tasks: $.global.tasks,
    },
    trainer+: {
        max_epochs: 10,
        accumulate_grad_batches: "<??>",
        gpus: "<??>",
        log_every_n_steps: 1,
    },
    model+: {
        _target_: "src.models.vl.pretraining_model.PretrainingModel",
        encoder_kwargs: {
            pretrained_language_model: $.global.pretrained_language_model,
            pretrained_vision_model: $.global.pretrained_vision_model,
            modality_embedding_vocab_size: 0,
            transformer_config+: {
                _target_: "src.models.vl.encoder_model.TransformerConfig",
            },
            fusion_type: "coattention",
        },
        optimizer+: {
            lr: 1e-4,
            eps: 1e-8,
            betas: [0.9, 0.98],
            weight_decay: 0.01,
        },
        scheduler: {
            _target_: "src.utils.instantiation.Instantiation",
            klass: {
                _target_: "src.utils.instantiation.get_class_type",
                full_class_name: "src.modules.lr_schedulers.get_hf_scheduler"
            },
            name: "linear",
        },
        num_warmup_steps: 0.1,
        use_every_cls: true,
        lr_mult_head: 5,
        lr_mult_cross_modal: 5,
        tasks: $.global.tasks,
    },
}
