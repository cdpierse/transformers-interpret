from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from captum.attr import (
    IntegratedGradients,
    LayerConductance,
    LayerIntegratedGradients,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)
from captum.attr import visualization as viz

from transformers_interpret.errors import AttributionsNotCalculatedError


class Attributions:
    def __init__(self, custom_forward: Callable, embeddings: nn.Module, tokens: list):
        self.custom_forward = custom_forward
        self.embeddings = embeddings
        self.tokens = tokens


class LIGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        tokens: list,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        sep_id: int,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        ref_token_type_ids: torch.Tensor = None,
        ref_position_ids: torch.Tensor = None,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids

        self.lig = LayerIntegratedGradients(self.custom_forward, self.embeddings)

        if self.token_type_ids is not None and self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids, self.position_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_token_type_ids,
                    self.ref_position_ids,
                ),
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
            )
        elif self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.position_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_position_ids,
                ),
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
            )
        elif self.token_type_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_token_type_ids,
                ),
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
            )

        else:
            self._attributions, self.delta = self.lig.attribute(
                inputs=self.input_ids,
                baselines=self.ref_input_ids,
                return_convergence_delta=True,
            )

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(
                zip(self.tokens, self.attributions_sum)
            ):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self):
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0)
        self.attributions_sum = self.attributions_sum / torch.norm(
            self.attributions_sum
        )

    def visualize_attributions(
        self, pred_prob, pred_class, true_class, attr_class, all_tokens
    ):

        return viz.VisualizationDataRecord(
            self.attributions_sum,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            self.attributions_sum.sum(),
            all_tokens,
            self.delta,
        )
