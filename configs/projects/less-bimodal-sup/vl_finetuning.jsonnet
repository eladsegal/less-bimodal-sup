local config = import "../../base.jsonnet";

local freezing_strategies = {
    none: [],
    vision: ["_encoder.vision_model"],
    language: ["_encoder.language_model"],
    encoders: ["_encoder.language_model", "_encoder.vision_model"],
    projection: ["_encoder.language_projection", "_encoder.vision_projection"],
};

local batch_sizes = {
    "": -1,
    gqa: 96,
    vqa: 96,
    vqa_imagenet: 96,
    nlvr2: 48,
};

local lrs = {
    "": -1,
    gqa: 1e-5,
    vqa: 2e-5,
    nlvr2: 1e-5,
};

local objective_format_per_dataset = {
    "": "",
    gqa: "cross_entropy",
    vqa: "binary_cross_entropy_with_logits",
    vqa_imagenet: "binary_cross_entropy_with_logits",
    nlvr2: "cross_entropy",
};

config {
    wandb+:: {
        group: "vl_finetuning",
    },

    global+: {
        tags+: ["vl_finetuning"],
        objective_format: objective_format_per_dataset[$.global.dataset_name],
        pretrained_language_model: "roberta-base",
        pretrained_vision_model: "google/vit-base-patch16-224-in21k",
        freezing_strategy: "none",
        image_size: 224,
    },

    callbacks_dict+:: {
        freezer+: {
            _target_: "src.callbacks.Freezer",
            fqns: freezing_strategies[$.global.freezing_strategy],
            train_bias_only: false,
        }
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
        _target_: "src.data.datamodules.finetuning_datamodule.FinetuningDataModule",
        tokenizer: {
            _target_: "transformers.AutoTokenizer.from_pretrained",
            use_fast: true,
            pretrained_model_name_or_path: $.global.pretrained_language_model,
        },
        feature_extractor: {
            _target_: "transformers.AutoFeatureExtractor.from_pretrained",
            pretrained_model_name_or_path: $.global.pretrained_vision_model,
            do_resize: false,
            do_normalize: false,
            do_center_crop: false,
        },
        batch_size: batch_sizes[$.global.dataset_name],
    },
    trainer+: {
        max_epochs: 10,
    },
    model+: {
        _target_: "src.models.vl.classifier_model.ClassifierModel",
        encoder_kwargs: {
            pretrained_language_model: $.global.pretrained_language_model,
            pretrained_vision_model: $.global.pretrained_vision_model,
            transformer_config+: {
                _target_: "src.models.vl.encoder_model.TransformerConfig",
            },
            fusion_type: "coattention",
        },
        optimizer+: {
            lr: lrs[$.global.dataset_name],
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
        classifier_klass: {
            _target_: "src.utils.instantiation.get_class_type",
            full_class_name: "src.modules.classifiers.ViltClassifier",
        },
        use_every_cls: true,
        lr_mult_head: 10,
        lr_mult_cross_modal: 10,
    },
}
