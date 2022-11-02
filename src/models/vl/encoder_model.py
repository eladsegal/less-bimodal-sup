from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

import torch
from torch import nn

from transformers import AutoConfig, AutoModel, CLIPVisionModel

from transformers.models.bert.modeling_bert import BertLayer, BertConfig

from src.models.base_model import BaseModel
from src.modules.classifiers import Classifier
from src.modules.initializations import init_weights_for_module

from src.models.meter.components.clip_model import build_model as meter_build_clip
from src.models.meter.components.bert_model import BertCrossLayer

from src.data.datamodules.utils.key_mapping import (
    KeyMapping,
)  # TODO: Should be used in the data collator if https://github.com/pytorch/pytorch/issues/67831 gets fixed

import logging

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    input_dim: int = 768
    num_hidden_layers: int = 2


class FusionType(Enum):
    COATTENTION = "coattention"
    MERGED = "merged"


class EncoderModel(BaseModel):
    def __init__(
        self,
        pretrained_language_model: str,
        pretrained_vision_model: str,
        language_from_scratch: bool = False,
        vision_from_scratch: bool = False,
        transformer_config: TransformerConfig = TransformerConfig(),
        modality_embedding_vocab_size: int = 0,
        fusion_type: Optional[str] = FusionType.MERGED.value,
        ignore_last_encoder_layer=False,
        pre_projection_ln: bool = False,
        post_projection_ln: bool = False,
        pre_fusion_ln: bool = False,
        post_fusion_ln: bool = False,
        init_projection_to_identity: bool = False,
        no_projection: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        language_config = AutoConfig.from_pretrained(pretrained_language_model)
        language_kwargs = (
            dict(add_pooling_layer=False)
            if all(model_family not in pretrained_language_model for model_family in ["microsoft/deberta"])
            else dict()
        )
        self._language_from_scratch = language_from_scratch
        if language_from_scratch:
            self.language_model = AutoModel.from_config(language_config, **language_kwargs)
        else:
            self.language_model = AutoModel.from_pretrained(
                pretrained_language_model, config=language_config, **language_kwargs
            )
        if "clip" in pretrained_language_model:
            self.language_model = self.language_model.text_model

        vision_config = AutoConfig.from_pretrained(pretrained_vision_model)
        # TODO: find a better way to modify the configs and models specifically for a model
        vision_kwargs = (
            dict(add_pooling_layer=False)
            if all(model_family not in pretrained_vision_model for model_family in ["openai/clip", "facebook/vit-mae"])
            else dict()
        )
        self._vision_from_scratch = vision_from_scratch
        """if pretrained_vision_model == "microsoft/beit-base-patch16-224-pt22k":
            vision_config.use_mask_token = False
            vision_config.use_relative_position_bias = True
            vision_config.use_shared_relative_position_bias = False
            logger.info("Using the config meant for fine-tuning BEiT")"""
        """if pretrained_vision_model == "microsoft/beit-base-patch16-224-pt22k-ft22k":
            vision_config.use_mask_token = False
            vision_config.use_relative_position_bias = False
            vision_config.use_shared_relative_position_bias = True
            logger.info("Using the config meant for pre-training BEiT")"""

        if vision_from_scratch:
            self.vision_model = AutoModel.from_config(vision_config, **vision_kwargs)
        else:
            self.vision_model = AutoModel.from_pretrained(
                pretrained_vision_model, config=vision_config, **vision_kwargs
            )
        if "openai/clip" in pretrained_vision_model:
            self.vision_model = self.vision_model.vision_model
            # Need to copy and edit to drop the pooling layer. Until then trainer.strategy.find_unused_parameters=True is needed
            self.vision_model = meter_build_clip(
                pretrained_vision_model, resolution_after=self._cfg["global"]["image_size"]
            )

        self._ignore_last_encoder_layer = ignore_last_encoder_layer

        self.pre_language_projection_ln = (
            nn.LayerNorm(self.language_model.config.hidden_size) if pre_projection_ln else None
        )
        self.language_projection = (
            nn.Linear(self.language_model.config.hidden_size, transformer_config.input_dim)
            if not no_projection
            else None
        )
        self.post_language_projection_ln = nn.LayerNorm(transformer_config.input_dim) if post_projection_ln else None

        self.pre_vision_projection_ln = (
            nn.LayerNorm(self.vision_model.config.hidden_size) if pre_projection_ln else None
        )
        self.vision_projection = (
            nn.Linear(self.vision_model.config.hidden_size, transformer_config.input_dim)
            if not no_projection
            else None
        )
        self.post_vision_projection_ln = nn.LayerNorm(transformer_config.input_dim) if post_projection_ln else None

        # Modality embeddings
        self.modality_embedding_vocab_size = modality_embedding_vocab_size
        if self.modality_embedding_vocab_size > 0:
            self.token_modality_embeddings = nn.Embedding(modality_embedding_vocab_size, transformer_config.input_dim)

        # Transformer layers
        if fusion_type is not None:
            self.fusion_type = FusionType(fusion_type)
            config = BertConfig.from_pretrained("bert-base-uncased")
            self.hidden_size = config.hidden_size
            if self.fusion_type == FusionType.COATTENTION:
                self.cross_modal_text_layers = nn.ModuleList(
                    [BertCrossLayer(config) for _ in range(transformer_config.num_hidden_layers)]
                )
                self.cross_modal_image_layers = nn.ModuleList(
                    [BertCrossLayer(config) for _ in range(transformer_config.num_hidden_layers)]
                )
                self.post_language_fusion_ln = nn.LayerNorm(self.hidden_size) if post_fusion_ln else None
                self.post_vision_fusion_ln = nn.LayerNorm(self.hidden_size) if post_fusion_ln else None
            elif self.fusion_type == FusionType.MERGED:
                self.pre_fusion_ln = nn.LayerNorm(self.hidden_size) if pre_fusion_ln else None
                self.cross_modal_layers = nn.ModuleList(
                    [BertLayer(config) for _ in range(transformer_config.num_hidden_layers)]
                )
                self.post_fusion_ln = nn.LayerNorm(self.hidden_size) if post_fusion_ln else None
        else:
            # TODO: this is a bad design, but allows keeping changes to the minimum for this non-default case
            self.fusion_type = None
            self.hidden_size = transformer_config.input_dim
            self.post_language_fusion_ln = nn.LayerNorm(self.hidden_size) if post_fusion_ln else None
            self.post_vision_fusion_ln = nn.LayerNorm(self.hidden_size) if post_fusion_ln else None

        exclude_from_initialization = ["language_model", "vision_model"]
        if init_projection_to_identity:
            exclude_from_initialization.extend(["language_projection", "vision_projection"])
            self.language_projection.weight.data.copy_(torch.eye(self.hidden_size))
            logger.info(f"Initialized identity weights for language_projection")
            self.vision_projection.weight.data.copy_(torch.eye(self.hidden_size))
            logger.info(f"Initialized identity weights for vision_projection")
        self.init_weights(init_weights_for_module, excluding=exclude_from_initialization)

    def language_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["key_mapping"] = KeyMapping(batch["key_mapping"])
        key_mapping = batch["key_mapping"]

        model_inputs = key_mapping.apply("language_inputs", batch)
        language_outputs = dict(self.language_model(**model_inputs))
        return {"language_hidden_states": language_outputs["last_hidden_state"]}

    def vision_forward(self, batch: Dict[str, Any], image_index=0) -> Dict[str, Any]:
        batch["key_mapping"] = KeyMapping(batch["key_mapping"])
        key_mapping = batch["key_mapping"]

        vision_inputs_key_str = f"vision_inputs_{image_index}" if batch["num_of_images"] > 1 else "vision_inputs"
        model_inputs = key_mapping.apply(vision_inputs_key_str, batch)

        vision_outputs = dict(self.vision_model(**model_inputs))
        return {"vision_hidden_states": vision_outputs["last_hidden_state"]}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["key_mapping"] = KeyMapping(batch["key_mapping"])
        key_mapping = batch["key_mapping"]
        language_attention_mask = batch["attention_mask"]
        batch_size = self.find_batch_size(batch)

        model_inputs = key_mapping.apply("language_inputs", batch)
        if self._ignore_last_encoder_layer:
            language_outputs = dict(self.language_model(**model_inputs, output_hidden_states=True))
            language_hidden_states = language_outputs["hidden_states"][-2]
        else:
            language_outputs = dict(self.language_model(**model_inputs))
            language_hidden_states = language_outputs["last_hidden_state"]

        if self.pre_language_projection_ln:
            language_hidden_states = self.pre_language_projection_ln(language_hidden_states)
        language_hidden_states = (
            self.language_projection(language_hidden_states)
            if self.language_projection is not None
            else language_hidden_states
        )
        if self.post_language_projection_ln:
            language_hidden_states = self.post_language_projection_ln(language_hidden_states)

        if self.modality_embedding_vocab_size > 0:
            language_hidden_states = language_hidden_states + self.token_modality_embeddings(
                torch.zeros_like(language_attention_mask)
            )

        hidden_states_dict_per_image = []
        for image_index in range(batch["num_of_images"]):
            # TODO: Can use viewXnum_of_images to run once instead of loop, like volta
            vision_inputs_key_str = f"vision_inputs_{image_index}" if batch["num_of_images"] > 1 else "vision_inputs"
            model_inputs = key_mapping.apply(vision_inputs_key_str, batch)

            if self._ignore_last_encoder_layer:
                vision_outputs = dict(self.vision_model(**model_inputs, output_hidden_states=True))
                vision_hidden_states = vision_outputs["hidden_states"][-2]
            else:
                vision_outputs = dict(self.vision_model(**model_inputs))
                vision_hidden_states = vision_outputs["last_hidden_state"]

            if self.pre_vision_projection_ln:
                vision_hidden_states = self.pre_vision_projection_ln(vision_hidden_states)
            vision_hidden_states = (
                self.vision_projection(vision_hidden_states)
                if self.vision_projection is not None
                else vision_hidden_states
            )
            if self.post_vision_projection_ln:
                vision_hidden_states = self.post_vision_projection_ln(vision_hidden_states)

            vision_attention_mask = torch.cat(
                (
                    torch.ones((batch_size, 1)).type_as(language_attention_mask),  # vision CLS
                    torch.ones((batch_size, vision_hidden_states.shape[1] - 1)).type_as(
                        language_attention_mask
                    ),  # non-CLS vision tokens
                ),
                dim=1,
            )

            if self.modality_embedding_vocab_size > 0:
                vision_hidden_states = vision_hidden_states + self.token_modality_embeddings(
                    (image_index + 1)
                    * torch.ones(vision_hidden_states.shape[:2], dtype=torch.long, device=vision_hidden_states.device)
                )

            # transformer output
            if self.fusion_type == FusionType.COATTENTION:
                extended_language_attention_mask = self._get_extended_attention_mask(language_attention_mask)
                extended_vision_attention_mask = self._get_extended_attention_mask(vision_attention_mask)
                for i, (text_layer, image_layer) in enumerate(
                    zip(self.cross_modal_text_layers, self.cross_modal_image_layers)
                ):
                    x = text_layer(
                        language_hidden_states,
                        vision_hidden_states,
                        extended_language_attention_mask,
                        extended_vision_attention_mask,
                    )[0]
                    y = image_layer(
                        vision_hidden_states,
                        language_hidden_states,
                        extended_vision_attention_mask,
                        extended_language_attention_mask,
                    )[0]
                    language_hidden_states, vision_hidden_states = x, y

                if self.post_language_fusion_ln and self.post_vision_fusion_ln:
                    language_hidden_states = (
                        self.post_language_fusion_ln(language_hidden_states)
                        if self.post_language_fusion_ln
                        else language_hidden_states
                    )
                    vision_hidden_states = (
                        self.post_vision_fusion_ln(vision_hidden_states)
                        if self.post_vision_fusion_ln
                        else vision_hidden_states
                    )

                hidden_states_dict_per_image.append(
                    {
                        "language_hidden_states": language_hidden_states,
                        "vision_hidden_states": vision_hidden_states,
                    }
                )
            elif self.fusion_type == FusionType.MERGED:
                hidden_states = torch.cat([language_hidden_states, vision_hidden_states], dim=1)
                attention_mask = torch.cat([language_attention_mask, vision_attention_mask], dim=1)

                if self.pre_fusion_ln:
                    hidden_states = self.pre_fusion_ln(hidden_states) if self.pre_fusion_ln else hidden_states

                extended_attention_mask = self._get_extended_attention_mask(attention_mask)

                for i, layer_module in enumerate(self.cross_modal_layers):
                    layer_outputs = layer_module(
                        hidden_states,
                        extended_attention_mask,
                    )
                    hidden_states = layer_outputs[0]

                if self.post_fusion_ln:
                    hidden_states = self.post_fusion_ln(hidden_states) if self.post_fusion_ln else hidden_states

                vision_offset = language_attention_mask.shape[1]
                hidden_states_dict_per_image.append(
                    {
                        "language_hidden_states": hidden_states[:, :vision_offset, :],
                        "vision_hidden_states": hidden_states[:, vision_offset:, :],
                    }
                )
            elif self.fusion_type is None:
                if self.post_language_fusion_ln and self.post_vision_fusion_ln:
                    language_hidden_states = (
                        self.post_language_fusion_ln(language_hidden_states)
                        if self.post_language_fusion_ln
                        else language_hidden_states
                    )
                    vision_hidden_states = (
                        self.post_vision_fusion_ln(vision_hidden_states)
                        if self.post_vision_fusion_ln
                        else vision_hidden_states
                    )
                hidden_states_dict_per_image.append(
                    {
                        "language_hidden_states": language_hidden_states,
                        "vision_hidden_states": vision_hidden_states,
                    }
                )
            hidden_states_dict_per_image[-1].update(
                {
                    "language_attention_mask": language_attention_mask,
                    "vision_attention_mask": vision_attention_mask,
                }
            )

        return hidden_states_dict_per_image

    def _get_extended_attention_mask(self, attention_mask: torch.Tensor):
        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
