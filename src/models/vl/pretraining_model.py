from typing import Dict, Any, List, Union

import torch
from torch import nn
import numpy as np

from transformers import PreTrainedTokenizerBase, BertConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from src.models.vl.encoder_model import EncoderModel
from src.models.base_model import BaseModel
from src.modules.initializations import init_weights_for_module
from src.metrics import AccuracyByLabels, TopKAccuracy

from src.data.datamodules.utils.key_mapping import (
    KeyMapping,
)  # TODO: Should be used in the data collator if https://github.com/pytorch/pytorch/issues/67831 gets fixed

import logging

logger = logging.getLogger(__name__)


class PretrainingModel(BaseModel):
    def __init__(
        self,
        tasks: Union[str, List[str]],
        encoder_kwargs: Dict[str, Any],
        lr_mult_head: float = 1.0,
        lr_mult_cross_modal: float = 1.0,
        use_every_cls: bool = False,
        use_pooler: bool = False,
        tokenizer: PreTrainedTokenizerBase = None,
        contrastive_init_temperature: float = 0.07,
        use_global_loss: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if tasks == ["contrastive"]:
            encoder_kwargs["fusion_type"] = None
            encoder_kwargs["no_projection"] = True

        self._encoder = EncoderModel(**encoder_kwargs, cfg=self._cfg)
        hidden_size = self._encoder.hidden_size

        if isinstance(tasks, str):
            tasks = [tasks]
        for task in tasks:
            self.define_loss(f"{task}_loss")

        if "mlm" in tasks:
            self.define_metric(["train", "val", "test"], "mlm_accuracy", AccuracyByLabels)

        if "itm" in tasks:
            self.define_metric(["train", "val", "test"], "itm_accuracy", AccuracyByLabels)

        if "contrastive" in tasks:
            self.define_metric("train", "contrastive_text2image", TopKAccuracy, kwargs_list={"k": [1, 2, 5]})
            self.define_metric("train", "contrastive_image2text", TopKAccuracy, kwargs_list={"k": [1, 2, 5]})
            self.define_metric("val", "contrastive_text2image", TopKAccuracy, kwargs_list={"k": [1, 2, 5]})
            self.define_metric("val", "contrastive_image2text", TopKAccuracy, kwargs_list={"k": [1, 2, 5]})

        self._use_pooler = use_pooler
        if use_pooler:
            self._text_pooler = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
            self._image_pooler = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())

        self._use_every_cls = use_every_cls

        if "mlm" in tasks:
            self._mlm_score = BertLMPredictionHead(
                BertConfig(
                    hidden_size=self._encoder.language_model.config.hidden_size,
                    vocab_size=self._encoder.language_model.config.vocab_size,
                )
            )

        if "itm" in tasks:
            self._itm_score = nn.Linear(hidden_size * (2 if use_every_cls else 1), 2)

        if "contrastive" in tasks:
            contrastive_hidden_size = 256
            self._contrastive_language_projection = nn.Linear(
                self._encoder.language_model.config.hidden_size, contrastive_hidden_size
            )
            self._contrastive_vision_projection = nn.Linear(
                self._encoder.vision_model.config.hidden_size, contrastive_hidden_size
            )
            self._logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / contrastive_init_temperature)
            )  # CLIP's default value, gets to temperature of 1/100 at the end of the training
            self._use_global_loss = use_global_loss

        self._lr_mult_head = lr_mult_head
        self._lr_mult_cross_modal = lr_mult_cross_modal

        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(encoder_kwargs["pretrained_language_model"])
        self._tasks = tasks

        self.init_weights(init_weights_for_module, excluding=["_encoder"])

    def on_train_start(self) -> None:
        super().on_train_start()
        if "mlm" in self._tasks:
            self.get_metric("train", "mlm_accuracy").window_length = self.trainer.log_every_n_steps
        if "itm" in self._tasks:
            self.get_metric("train", "itm_accuracy").window_length = self.trainer.log_every_n_steps
        if "contrastive" in self._tasks:
            self.get_metric("train", "contrastive_text2image").window_length = self.trainer.log_every_n_steps
            self.get_metric("train", "contrastive_image2text").window_length = self.trainer.log_every_n_steps

    def forward(self, batch: Dict[str, Any], enhance_outputs: bool = False, **kwargs) -> Dict[str, Any]:
        outputs = {}
        if "mlm" in self._tasks:
            outputs.update(self.compute_mlm(batch, enhance_outputs=enhance_outputs))
        if "itm" in self._tasks:
            outputs.update(self.compute_itm(batch, enhance_outputs=enhance_outputs))
        if "contrastive" in self._tasks:
            outputs.update(self.compute_contrastive(batch, enhance_outputs=enhance_outputs))
        outputs["loss"] = sum(task_loss for key, task_loss in outputs.items() if key.endswith("_loss"))
        return outputs

    def compute_mlm(self, batch, enhance_outputs: bool = False):
        batch["key_mapping"] = KeyMapping(batch["key_mapping"])
        key_mapping = batch["key_mapping"]
        model_inputs = key_mapping.apply("mlm_language_inputs", batch)
        mlm_batch = {**batch, **model_inputs}

        hidden_states_dict = self._encoder(mlm_batch)[0]
        logits = self._mlm_score(hidden_states_dict["language_hidden_states"])

        outputs = {}

        # If the text is from actual labels, reorder mlm_labels to compute the minimal loss per instance in the batch,
        # as the original ordering doesn't matter
        """is_caption_from_labels = self.get_current_dataset_helper().get_ready_examples(pidxs=[batch["pidx"][0]])[0].get("is_caption_from_labels", False)
        if is_caption_from_labels:
            # For each mlm label, order the masked logits
            # Then, move each label to the highest logit position
            pass """

        outputs["mlm_loss"] = nn.functional.cross_entropy(
            logits.view(-1, self._encoder.language_model.config.vocab_size), batch["mlm_labels"].view(-1)
        )
        outputs["mlm_logits"] = logits.detach()

        if enhance_outputs:
            pass
            # Need to make the output meaningful - include the masked input and only the predictions for the masked tokens
            # outputs["mlm_predictions"] = self._tokenizer.batch_decode(outputs["mlm_logits"].argmax(-1), skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return outputs

    def compute_itm(self, batch, enhance_outputs: bool = False):
        itm_pixel_values = batch["pixel_values"].clone()
        for i, fake_index in enumerate((~(batch["itm_labels"].bool())).nonzero().flatten().tolist()):
            itm_pixel_values[fake_index] = batch["itm_false_pixel_values"][i]

        itm_batch = {**batch, **{"pixel_values": itm_pixel_values}}

        hidden_states_dict = self._encoder(itm_batch)[0]

        # get CLSs
        language_cls = hidden_states_dict["language_hidden_states"][:, 0, :]
        if self._use_every_cls:
            vision_cls = hidden_states_dict["vision_hidden_states"][:, 0, :]

        # pooler
        if self._use_pooler:
            language_cls = self._text_pooler(language_cls)
            if self._use_every_cls:
                vision_cls = self._image_pooler(vision_cls)

        # classifier input
        if self._use_every_cls:
            itm_input = torch.cat([language_cls, vision_cls], dim=-1)
        else:
            itm_input = language_cls

        logits = self._itm_score(itm_input)

        outputs = {}
        outputs["itm_loss"] = nn.functional.cross_entropy(logits, batch["itm_labels"])
        outputs["itm_logits"] = logits.detach()
        return outputs

    def compute_contrastive(self, batch, enhance_outputs: bool = False):
        with torch.no_grad():
            self._logit_scale.clamp_(np.log(1 / 0.5), np.log(1 / 0.001))

        image_embeds = self._encoder.vision_forward(batch)["vision_hidden_states"][:, 0, :]
        text_embeds = self._encoder.language_forward(batch)["language_hidden_states"][:, 0, :]

        image_embeds = self._contrastive_vision_projection(image_embeds)
        text_embeds = self._contrastive_language_projection(text_embeds)

        with torch.no_grad():
            if self._use_global_loss:
                gathered_image_embeds = self.all_gather(image_embeds).view(-1, image_embeds.shape[-1])
                gathered_text_embeds = self.all_gather(text_embeds).view(-1, text_embeds.shape[-1])
            else:
                gathered_image_embeds = image_embeds.detach().clone()
                gathered_text_embeds = text_embeds.detach().clone()
            # normalized features
            gathered_image_embeds = gathered_image_embeds / gathered_image_embeds.norm(dim=-1, keepdim=True)
            gathered_text_embeds = gathered_text_embeds / gathered_text_embeds.norm(dim=-1, keepdim=True)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self._logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, gathered_image_embeds.t()) * logit_scale
        logits_per_image = torch.matmul(image_embeds, gathered_text_embeds.t()) * logit_scale

        offset = self._get_contrastive_offset(batch)
        labels = torch.arange(offset, offset + len(logits_per_text), device=logits_per_text.device)
        loss = clip_loss(logits_per_text, logits_per_image, labels)

        outputs = {}
        outputs["contrastive_loss"] = loss
        outputs["contrastive_logits_per_text"] = logits_per_text.detach()
        outputs["contrastive_logits_per_image"] = logits_per_image.detach()
        batch["contrastive_labels"] = labels
        return outputs

    def step_with_metrics(self, batch: Dict[str, Any], batch_idx: int, enhance_outputs=False) -> Dict[str, Any]:
        batch_size = self.find_batch_size(batch)
        outputs = self(batch, enhance_outputs=enhance_outputs)

        metrics_tuples = []
        if "mlm" in self._tasks:
            metrics_tuples.append(("mlm_accuracy", "mlm_logits", "mlm_labels"))
        if "itm" in self._tasks:
            metrics_tuples.append(("itm_accuracy", "itm_logits", "itm_labels"))
        if "contrastive" in self._tasks:
            metrics_tuples.append(("contrastive_text2image", "contrastive_logits_per_text", "contrastive_labels"))
            metrics_tuples.append(("contrastive_image2text", "contrastive_logits_per_image", "contrastive_labels"))

        for metric_id, logits_key, labels_key in metrics_tuples:
            if logits_key not in outputs or labels_key not in batch:
                continue

            accuracy_metric = self.get_current_metric(metric_id)

            metric_result = accuracy_metric(outputs[logits_key], batch[labels_key])

            self.log_dict(
                {result.log_name: result.value for result in metric_result.values()},
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                logger=self.training,
                add_dataloader_idx=False,
            )

        return outputs

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
        if self.training:
            return

        metric_ids = []
        if "mlm" in self._tasks:
            metric_ids.append("mlm_accuracy")
        if "itm" in self._tasks:
            metric_ids.append("itm_accuracy")
        if "contrastive" in self._tasks:
            metric_ids.append("contrastive_text2image")
            metric_ids.append("contrastive_image2text")

        for metric_id in metric_ids:
            keys_to_log = [result.log_name for result in self.get_current_metric(metric_id).compute().values()]
            self.log_dict(
                {k: v for k, v in self.trainer.progress_bar_metrics.items() if k in keys_to_log},
                prog_bar=True,
                logger=True,
            )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        outputs = self.step_with_metrics(batch, batch_idx, enhance_outputs=False)
        return outputs

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx=0) -> Dict[str, Any]:
        outputs = self.step_with_metrics(batch, batch_idx, enhance_outputs=True)
        return outputs

    def _get_optimizer_grouped_parameters(self, **optimizer_kwargs):
        lr = optimizer_kwargs["lr"]
        wd = optimizer_kwargs.get("weight_decay", 0.0)

        head_names = [
            "_mlm_score.",
            "_itm_score.",
            "_contrastive_language_projection.",
            "_contrastive_vision_projection.",
            "_logit_scale.",
        ]
        cross_modal_names = [
            "_encoder.cross_modal_text_layers.",
            "_encoder.cross_modal_image_layers.",
            "_encoder.cross_modal_layers.",
            "_encoder.language_projection.",
            "_encoder.vision_projection.",
            "_text_pooler.",
            "_image_pooler.",
        ]
        lr_mult_head = self._lr_mult_head
        lr_mult_cross_modal = self._lr_mult_cross_modal

        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.extend(
            self.create_optimizer_group(
                (
                    (n, p)
                    for n, p in filter(lambda n_p: n_p[1].requires_grad, self.named_parameters())
                    if not any(n.startswith(bb) for bb in head_names)
                    and not any(n.startswith(ht) for ht in cross_modal_names)
                ),
                weight_decay=wd,
                lr=lr,
                name="regular",
            )
        )

        optimizer_grouped_parameters.extend(
            self.create_optimizer_group(
                (
                    (n, p)
                    for n, p in filter(lambda n_p: n_p[1].requires_grad, self.named_parameters())
                    if any(n.startswith(bb) for bb in head_names)
                    and not any(n.startswith(ht) for ht in cross_modal_names)
                ),
                weight_decay=wd,
                lr=lr * lr_mult_head,
                name="head",
            )
        )

        optimizer_grouped_parameters.extend(
            self.create_optimizer_group(
                (
                    (n, p)
                    for n, p in filter(lambda n_p: n_p[1].requires_grad, self.named_parameters())
                    if not any(n.startswith(bb) for bb in head_names)
                    and any(n.startswith(ht) for ht in cross_modal_names)
                ),
                weight_decay=wd,
                lr=lr * lr_mult_cross_modal,
                name="cross_modal",
            )
        )

        return optimizer_grouped_parameters

    def _get_contrastive_offset(self, batch) -> int:
        if self._use_global_loss:
            return self.global_rank * self.find_batch_size(batch)
        else:
            return 0


def get_contrastive_labels(logits: torch.Tensor, offset: int) -> torch.Tensor:
    return torch.arange(offset, offset + len(logits), device=logits.device)


# Based on https://github.com/huggingface/transformers/blob/2e11a043374a6229ec129a4765ee4ba7517832b9/src/transformers/models/clip/modeling_clip.py#L63-L72
def contrastive_loss(logits: torch.Tensor, labels) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels)


def clip_loss(similarity_text: torch.Tensor, similarity_image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity_text, labels)
    image_loss = contrastive_loss(similarity_image, labels)
    return (caption_loss + image_loss) / 2.0
