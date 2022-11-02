from typing import Any, Callable, Dict, Optional, Union
from src.data.dataset_containers.base_dataset_container import BaseDatasetContainer
from src.data.dataset_containers.complex_dataset_container import ComplexDatasetContainer

from src.data.dataset_helper import DatasetHelper
from src.evaluators.evaluator import Evaluator
from src.metrics import Metric, HfMetric, TorchMetricsWrapper
from src.utils.inspection import get_fqn


FORMATTERS_PRESETS = {
    (
        "src.models.discriminative_vqa_model.DiscriminativeVQAModel",
        "vqa.default",
        "src.metrics.concrete.vqa_score.VqaScore",
    ): {
        "prediction_formatter": lambda id_, prediction: prediction["predicted_answer"]
        if isinstance(prediction, dict)
        else prediction,
        "example_formatter": lambda id_, example: example["answers"],
    },
    (
        "src.models.discriminative_vqa_model.DiscriminativeVQAModel",
        "image_net_from_files.default",
        "src.metrics.concrete.accuracy.Accuracy",
    ): {
        "prediction_formatter": lambda id_, prediction: str(
            prediction["predicted_label"] in prediction["labels_list"]
        ),
        "example_formatter": lambda id_, example: str(True),
    },
    (
        "src.models.discriminative_vqa_model.DiscriminativeVQAModel",
        "gqa.default",
        "src.metrics.concrete.accuracy.Accuracy",
    ): {
        "prediction_formatter": lambda id_, prediction: prediction["predicted_answer"]
        if isinstance(prediction, dict)
        else prediction,
        "example_formatter": lambda id_, example: example["answer"],
    },
    (
        "src.models.discriminative_vqa_model.DiscriminativeVQAModel",
        "nlvr2.default",
        "src.metrics.concrete.accuracy.Accuracy",
    ): {
        "prediction_formatter": lambda id_, prediction: prediction["predicted_answer"]
        if isinstance(prediction, dict)
        else prediction,
        "example_formatter": lambda id_, example: example["label"],
    },
    (
        "src.models.discriminative_vqa_model.DiscriminativeVQAModel",
        "nlvr2.default",
        "src.metrics.concrete.nlvr2_consistency.Nlvr2Consistency",
    ): {
        "prediction_formatter": lambda id_, prediction: prediction["predicted_answer"]
        if isinstance(prediction, dict)
        else prediction,
        "example_formatter": lambda id_, example: {"identifier": id_, "label": example["label"]},
    },
}


def identity_func(id_, x):
    return x


class HfEvaluator(Evaluator):
    def __init__(
        self,
        metrics: Union[Dict[str, Metric], Dict[str, Dict[str, Metric]]],
        # prediction_formatters: Optional[Dict[str, Callable]] = None,
        # example_formatters: Optional[Dict[str, Callable]] = None,
    ):
        if all(isinstance(metric, Metric) for metric in metrics.values()):
            self._metrics = {None: metrics}
        elif all(isinstance(metrics, dict) for metrics in metrics.values()):
            for key, current_metrics in metrics.items():
                if not all(isinstance(metric, Metric) for metric in current_metrics.values()):
                    raise TypeError(f"At least one value for key {key} is not an instance of Metric")
            self._metrics = metrics
        else:
            raise TypeError(f"metrics is not Dict[str, Metric] or Dict[str, Dict[str, Metric]]]")

        # TODO: Make it per source too
        # self._prediction_formatters = prediction_formatters
        # self._example_formatters = example_formatters

    def __call__(
        self, predictions: Dict[str, Dict[str, Any]], dataset_helper: DatasetHelper, source_model_hierarchy=None
    ) -> Any:
        metrics_to_log = {}
        for source_container_name, source_predictions in predictions.items():
            if source_container_name not in self._metrics:
                continue
            current_metrics = self._metrics[source_container_name]
            for metric_name, metric in current_metrics.items():
                if isinstance(metric, HfMetric):
                    fqn_parts = get_fqn(type(metric._hf_metric)).split(".")
                    fqn_parts = fqn_parts[:3] + [fqn_parts[-1]]
                    metric_fqn = ".".join(fqn_parts)
                elif isinstance(metric, TorchMetricsWrapper):
                    metric_fqn = get_fqn(type(metric._torchmetric))
                else:
                    metric_fqn = get_fqn(type(metric))

                prediction_formatter = None
                example_formatter = None

                # Use explicitly defined formatters
                """if self._prediction_formatters is not None and metric_name in self._prediction_formatters:
                    prediction_formatter = self._prediction_formatters[metric_name]
                if self._example_formatters is not None and metric_name in self._example_formatters:
                    example_formatter = self._example_formatters[metric_name]"""

                # If any of the formatters was defined explicitly, don't check the presets
                if prediction_formatter is None and example_formatter is None:
                    for source_model in source_model_hierarchy:
                        preset = FORMATTERS_PRESETS.get(
                            (source_model, dataset_helper.get_dataset_name(source_container_name), metric_fqn)
                        )
                        if preset is not None:
                            prediction_formatter = preset["prediction_formatter"]
                            example_formatter = preset["example_formatter"]
                            break

                # If a formatter was not set, use the identity function
                if example_formatter is None:
                    example_formatter = identity_func
                if prediction_formatter is None:
                    prediction_formatter = identity_func

                # Apply the formatters
                formatted_predictions, formatted_examples = [], []
                for id_, prediction in source_predictions.items():
                    formatted_predictions.append(prediction_formatter(id_, prediction))
                    formatted_examples.append(
                        example_formatter(id_, dataset_helper.get_example_by_id(id_, source_container_name))
                    )

                # calculate the metrics
                metric.reset()
                computed_metrics = metric(formatted_predictions, formatted_examples, sync=False)

                log_name_suffix = "" if source_container_name is None else f"_{source_container_name}"
                metrics_to_log.update(
                    {result.log_name + log_name_suffix: result.value for result in computed_metrics.values()}
                )

            n_examples_str = "n_examples"
            if source_container_name is not None:
                n_examples_str += "_" + source_container_name
            metrics_to_log.update({n_examples_str: float(len(source_predictions))})  # Use float to prevent warning
        return metrics_to_log
