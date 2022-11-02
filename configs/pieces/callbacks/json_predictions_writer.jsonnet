{
    global+: {
        output_predictions: true,
    },

    json_predictions_writer:: {
        _target_: "src.callbacks.JsonPredictionsWriter",
    },

    trainer+: {
        callbacks+: if $.global.output_predictions then [$.json_predictions_writer] else [],
    },
}
