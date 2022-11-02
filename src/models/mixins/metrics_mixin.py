from typing import Any, Dict, List, Union, Optional, Callable
from collections.abc import Sequence
import functools

from torch import nn

from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.trainer.states import TrainerFn

from hydra._internal.utils import _locate

from src.pl import MyLightningModule
from src.metrics import Metric, Sum, Average

STAGE_NAME_TO_PREFIX_FN = lambda stage: "val" if stage in ["validation", "validate", "sanity_check"] else stage


class MetricsMixin(MyLightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metrics_dict = nn.ModuleDict()
        self._define_metric(["train", "val", "test"], "n_examples", Sum)
        self._define_metric(["train"], "n_examples_overall", Sum)

        self.training_step = self._wrap_step(self.training_step)
        self.validation_step = self._wrap_step(self.validation_step)
        self.test_step = self._wrap_step(self.test_step)

        self._train_losses = {}
        self.define_loss("loss")

        self._init_totals()

        self._model_specific_metrics_definitions = []

        self._best_monitor_metric_value = None

    def _init_totals(self):
        self._totals = {
            f"{name}": {
                "offset": 0,
                "value": None,
            }
            for name in ["epoch", "global_step", "n_examples_overall"]
        }

        self._totals["epoch"]["value_fn"] = init_epoch_fn
        self._totals["global_step"]["value_fn"] = init_global_step_fn
        self._totals["n_examples_overall"]["value_fn"] = init_n_examples_overall_fn

    def define_loss(self, loss_name):
        self._train_losses[loss_name] = None
        self._define_metric(["val", "test"], loss_name, Average)

    def define_metric(
        self,
        prefixes: Union[Optional[str], List[Optional[str]]],
        name: str,
        klass: Union[str, Metric],
        kwargs_list=None,
        *,
        metric_id: str = None,
    ):
        self._model_specific_metrics_definitions.append(
            {
                "prefixes": prefixes,
                "name": name,
                "klass": klass,
                "kwargs_list": kwargs_list,
                "metric_id": metric_id if metric_id is not None else name,
            }
        )
        self._define_metric(**self._model_specific_metrics_definitions[-1])

    def _define_metric(
        self,
        prefixes: Union[Optional[str], List[Optional[str]]],
        name: str,
        klass: Union[str, Metric],
        kwargs_list=None,
        *,
        metric_id: Optional[str] = None,
    ):
        if metric_id is None:
            metric_id = name
        if isinstance(klass, str):
            klass = _locate(klass)
        if not isinstance(prefixes, Sequence) or isinstance(prefixes, str):
            prefixes = [prefixes]
        if kwargs_list is None:
            kwargs_list = {}
        if not isinstance(kwargs_list, Sequence):
            kwargs_list = [kwargs_list for i in range(len(prefixes))]
        for prefix, kwargs in zip(prefixes, kwargs_list):
            member_name = self._get_prefixed_metric_id(prefix, metric_id)
            if member_name in self._metrics_dict:
                raise Exception(f"Member {member_name} already exists in {type(self)}")
            self._metrics_dict[member_name] = klass(prefix=prefix, name=name, **kwargs)

    def get_current_metric(self, metric_id: str) -> Metric:
        return self.get_metric(self.get_current_prefix(), metric_id)

    def get_metric(self, prefix: Optional[str], metric_id: str) -> Metric:
        return self._metrics_dict[self._get_prefixed_metric_id(prefix, metric_id)]

    @staticmethod
    def _get_prefixed_metric_id(prefix: Optional[str], metric_id: str) -> str:
        return "_".join([x for x in [prefix, metric_id] if x is not None and len(x) > 0])

    def get_current_prefix(self) -> str:
        return STAGE_NAME_TO_PREFIX_FN(self.trainer.state.stage.value)

    def on_before_manually_load_weights(self, checkpoint: Dict[str, Any]):
        super().on_before_manually_load_weights(checkpoint)

        checkpoint["state_dict"].pop("_metrics_dict", None)

        if self._cfg is not None and self._cfg.get("global", {}).get("is_continuation", False):
            for name in self._totals.keys():
                name_total = f"{name}_total"
                if name_total in checkpoint:
                    self._totals[name]["offset"] = checkpoint[name_total]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)

        for name in self._totals.keys():
            name_total_for_resuming = f"{name}_total_for_resuming"
            if name_total_for_resuming in checkpoint:
                self._totals[name]["offset"] = checkpoint[name_total_for_resuming]

        if self.trainer is not None:
            checkpoint_callback = self.trainer.checkpoint_callback
            if checkpoint_callback is not None and checkpoint["monitor_metric_name"] == checkpoint_callback.monitor:
                self._best_monitor_metric_value = checkpoint["best_monitor_metric_value"]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)

        for name in self._totals.keys():
            checkpoint[f"{name}_total"] = self._totals[name]["value_fn"](self) + (
                1 if name != "n_examples_overall" else 0
            )
            checkpoint[f"{name}_total_for_resuming"] = self._totals[name]["offset"]

        if self.trainer.checkpoint_callback is not None:
            # None check is needed due to LR-Finder removing the checkpoint callback from the trainer
            checkpoint["monitor_metric_name"] = self.trainer.checkpoint_callback.monitor
            checkpoint["best_monitor_metric_value"] = self._best_monitor_metric_value

    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        self.get_current_metric("n_examples").reset()

        for metric_definition in self._model_specific_metrics_definitions:
            current_prefix = self.get_current_prefix()
            if current_prefix in metric_definition["prefixes"]:
                self.get_metric(current_prefix, metric_definition["metric_id"]).reset()

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        for loss_name in self._train_losses.keys():
            if self._train_losses[loss_name] is None:
                self._train_losses[loss_name] = TensorRunningAccum(window_length=self.trainer.log_every_n_steps)
            else:
                self._train_losses[loss_name].reset()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        for loss_name in self._train_losses.keys():
            self.get_current_metric(loss_name).reset()

    def _wrap_step(self, step_method: Callable) -> Callable:
        @functools.wraps(step_method)
        def wrapped_func(*args, **kwargs) -> Dict[str, Any]:
            outputs = step_method(*args, **kwargs)
            batch = args[0] if len(args) > 0 else kwargs["batch"]
            self.step_metrics_call(batch, outputs)
            return outputs

        return wrapped_func

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
        self.epoch_metrics_call()

    def step_metrics_call(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        batch_size = self.find_batch_size(batch)
        prefix = self.get_current_prefix()

        for loss_name in self._train_losses.keys():
            if loss_name in outputs:
                if self.training:
                    single_step_loss = outputs[loss_name].detach().clone()
                    self.log(
                        f"{prefix}/single_step_{loss_name}",
                        single_step_loss,
                        prog_bar=False,
                        on_step=True,
                        on_epoch=False,
                        logger=True,
                        sync_dist=True,
                    )

                    log_name = f"{prefix}/{loss_name}"
                    self._train_losses[loss_name].append(outputs[loss_name].detach().clone())
                    aggregated_loss = self._train_losses[loss_name].mean()
                else:
                    metric = self.get_current_metric(loss_name)
                    result = metric(outputs[loss_name].detach().clone())[loss_name]
                    log_name, aggregated_loss = result.log_name, result.value

                self.log(
                    log_name,
                    aggregated_loss,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=False,
                    logger=self.training,
                    sync_dist=self.training,
                )

        metric = self.get_current_metric("n_examples")
        self.log_dict(
            {result.log_name: float(result.value) for result in metric(batch_size).values()},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=self.training,
        )

        if self.training:
            metric = self.get_metric(prefix, "n_examples_overall")
            self.log_dict(
                {result.log_name: float(result.value) for result in metric(batch_size).values()},
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

            self.log_dict(
                {f"{name}_total": float(total_obj["value_fn"](self)) for name, total_obj in self._totals.items()},
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                logger=True,
            )

    def epoch_metrics_call(self) -> None:
        if not self.training:
            for loss_name in self._train_losses.keys():
                metric = self.get_current_metric(loss_name)
                if metric._update_called:
                    self.log_dict(
                        {result.log_name: result.value for result in metric.compute().values()},
                        prog_bar=False,
                        logger=True,
                    )

        if self.training:
            metric = self.get_current_metric("n_examples")
            self.log_dict(
                {result.log_name: float(result.value) for result in metric.compute().values()},
                prog_bar=False,
                logger=True,
            )

        if self.trainer.state.fn == TrainerFn.FITTING:
            # Needed in order to plot validation against
            metric = self.get_metric("train", "n_examples_overall")
            self.log_dict(
                {result.log_name: float(result.value) for result in metric.compute().values()},
                prog_bar=False,
                logger=True,
            )
            self.log_dict(
                {f"{name}_total": float(total_obj["value_fn"](self)) for name, total_obj in self._totals.items()},
                prog_bar=False,
                logger=True,
            )

            if not self.training:
                # TODO: Do it for all of the user-defined metrics for "val", unrelated to the checkpoint callback.
                # Requires keeping best for each of the metrics
                checkpoint_callback = self.trainer.checkpoint_callback
                if checkpoint_callback is not None:
                    monitor = checkpoint_callback.monitor
                    if "/" in monitor:
                        slash_index = monitor.index("/")
                        best_metric_name = "".join([monitor[: slash_index + 1], "best_", monitor[slash_index + 1 :]])
                    else:
                        best_metric_name = "best_" + monitor

                    if monitor in self.trainer.callback_metrics:
                        monitor_op = {"min": min, "max": max}[checkpoint_callback.mode]
                        monitor_metric_value = self.trainer.callback_metrics[monitor]
                        if self._best_monitor_metric_value is not None:
                            self._best_monitor_metric_value = monitor_op(
                                self._best_monitor_metric_value, monitor_metric_value
                            )
                        else:
                            self._best_monitor_metric_value = monitor_metric_value
                        self.log(
                            best_metric_name,
                            self._best_monitor_metric_value,
                            prog_bar=False,
                            logger=True,
                        )


def init_epoch_fn(self):
    return self._totals["epoch"]["offset"] + self.trainer.current_epoch


def init_global_step_fn(self):
    return self._totals["global_step"]["offset"] + self.trainer.global_step


def init_n_examples_overall_fn(self):
    return (
        self._totals["n_examples_overall"]["offset"]
        + self._metrics_dict[self._get_prefixed_metric_id("train", "n_examples_overall")]
        .compute()["n_examples_overall"]
        .value
    )
