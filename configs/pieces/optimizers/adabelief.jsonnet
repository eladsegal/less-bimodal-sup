{
    model+: {
        optimizer: {
            _target_: "src.utils.instantiation.Instantiation",
            klass: {
                _target_: "src.utils.instantiation.get_class_type",
                full_class_name: "adabelief_pytorch.AdaBelief",
            },
            lr: "<??>",
            betas: [0.9,0.999],
            eps: 1e-16,
            weight_decay: 5e-5,
            weight_decouple: true,
            rectify: true,
            fixed_decay: false,
        },
    },
}
// pip install adabelief-pytorch==0.2.1