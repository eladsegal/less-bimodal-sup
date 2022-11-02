from typing import Type, Dict, Any

import torch
from torch import nn

from src.models.vl.encoder_model import EncoderModel
from src.models.discriminative_vqa_model import DiscriminativeVQAModel
from src.modules.classifiers import Classifier
from src.modules.initializations import init_weights_for_module
from src.modules.initializations import init_linear_weights, init_linear_bias


import logging

logger = logging.getLogger(__name__)


class ClassifierModel(DiscriminativeVQAModel):
    def __init__(
        self,
        classifier_klass: Type[Classifier],
        num_of_images: int,
        encoder_kwargs: Dict[str, Any],
        use_every_cls: bool = False,
        use_pooler: bool = False,
        lr_mult_head: float = 1.0,
        lr_mult_cross_modal: float = 1.0,
        manually_load_weights_config: Dict[str, Any] = None,
        classifier_classes_reorder=None,
        use_lr_mult_on_language_encoder: bool = False,
        use_lr_mult_on_vision_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._encoder = EncoderModel(**encoder_kwargs, cfg=self._cfg)
        hidden_size = self._encoder.hidden_size

        self._use_pooler = use_pooler
        if use_pooler:
            self._text_pooler = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
            self._image_pooler = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())

        self._use_every_cls = use_every_cls

        classifier_input_dim = num_of_images * hidden_size * (2 if use_every_cls else 1)
        self._classifier = classifier_klass(classifier_input_dim, hidden_size, len(self._label2ans))

        self._lr_mult_head = lr_mult_head
        self._lr_mult_cross_modal = lr_mult_cross_modal

        self._manually_load_weights_config = {
            "remove_classifier": True,
        }
        if manually_load_weights_config is not None:
            self._manually_load_weights_config.update(manually_load_weights_config)
        self._classifier_classes_reorder = classifier_classes_reorder

        self._use_lr_mult_on_language_encoder = use_lr_mult_on_language_encoder
        self._use_lr_mult_on_vision_encoder = use_lr_mult_on_vision_encoder

        self.init_weights(init_weights_for_module, excluding=["_encoder"])

    def _get_names_for_detailed_grad_norms(self):
        return ["_encoder"]

    def forward(self, batch: Dict[str, Any], enhance_outputs: bool = False, **kwargs) -> Dict[str, Any]:
        hidden_states_dict_per_image = self._encoder(batch)

        classifier_input_to_concat = []
        for hidden_states_dict in hidden_states_dict_per_image:
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
                classifier_input = torch.cat([language_cls, vision_cls], dim=-1)
            else:
                classifier_input = language_cls
            classifier_input_to_concat.append(classifier_input)

        final_classifier_input = torch.cat(classifier_input_to_concat, dim=-1)

        logits = self._classifier(final_classifier_input)

        outputs = {}
        if self._objective_format == "binary_cross_entropy_with_logits":
            if "targets" in batch:
                outputs["loss"] = (
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["targets"])
                    * batch["targets"].shape[1]
                )
        elif self._objective_format in ["cross_entropy"]:
            if "labels" in batch:
                outputs["loss"] = torch.nn.functional.cross_entropy(logits, batch["labels"])
        outputs["logits"] = logits.detach()

        if enhance_outputs:
            self.enhance_outputs(batch, outputs)

        return outputs

    def _get_optimizer_grouped_parameters(self, **optimizer_kwargs):
        lr = optimizer_kwargs["lr"]
        wd = optimizer_kwargs.get("weight_decay", 0.0)

        head_names = ["_classifier."]
        cross_modal_names = [
            "_encoder.cross_modal_text_layers.",
            "_encoder.cross_modal_image_layers.",
            "_encoder.cross_modal_layers.",
            "_encoder.language_projection.",
            "_encoder.vision_projection.",
            "_text_pooler.",
            "_image_pooler.",
        ]
        if self._use_lr_mult_on_language_encoder:
            cross_modal_names += ["_encoder.language_model."]
        if self._use_lr_mult_on_vision_encoder:
            cross_modal_names += ["_encoder.vision_model."]

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

    def on_before_manually_load_weights(self, checkpoint: Dict[str, Any]):
        super().on_before_manually_load_weights(checkpoint)

        if self._manually_load_weights_config["remove_classifier"]:
            for key in list(checkpoint["state_dict"].keys()):
                if key.startswith("_classifier"):
                    del checkpoint["state_dict"][key]
        else:
            # Check if there's a mismatch in the output of the classifier, and fix it if so.
            keys_to_fix = []
            new_classes_sizes = []
            if hasattr(self, "_classifier") and hasattr(self._classifier, "_classifier"):
                if isinstance(self._classifier._classifier, nn.Sequential):
                    length = len(self._classifier._classifier)
                    keys_to_fix.append(f"_classifier._classifier.{length - 1}.weight")
                    keys_to_fix.append(f"_classifier._classifier.{length - 1}.bias")
                    new_classes_sizes.append(self._classifier._classifier[-1].out_features)
                    new_classes_sizes.append(self._classifier._classifier[-1].out_features)
                else:
                    keys_to_fix.append(f"_classifier._classifier.weight")
                    keys_to_fix.append(f"_classifier._classifier.bias")
                    new_classes_sizes.append(self._classifier._classifier.out_features)
                    new_classes_sizes.append(self._classifier._classifier.out_features)
            if hasattr(self, "clf") and hasattr(self._classifier, "logit_fc"):
                length = len(self.clf.logit_fc)
                keys_to_fix.append(f"clf.logit_fc.{length - 1}.weight")
                keys_to_fix.append(f"clf.logit_fc.{length - 1}.bias")
                new_classes_sizes.append(self._classifier.logit_fc[-1].out_features)
                new_classes_sizes.append(self._classifier.logit_fc[-1].out_features)

            for key, new_classes_size in zip(keys_to_fix, new_classes_sizes):
                if self._classifier_classes_reorder is not None:
                    old_tensor = checkpoint["state_dict"][key]

                    new_size = list(old_tensor.size())
                    new_size[0] = new_classes_size
                    new_tensor = torch.zeros(new_size, dtype=old_tensor.dtype)

                    if key.endswith(".weight"):
                        init_linear_weights(new_tensor)
                    elif key.endswith(".bias"):
                        init_linear_bias(new_tensor)

                    for old_class, new_class in self._classifier_classes_reorder.items():
                        if key.endswith(".weight"):
                            new_tensor.data[new_class, :] = old_tensor.data[old_class, :]
                        elif key.endswith(".bias"):
                            new_tensor.data[new_class] = old_tensor.data[old_class]
                    checkpoint["state_dict"][key] = new_tensor
