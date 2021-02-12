from typing import Callable, Dict, List, Tuple

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
    def __init__(self, custom_forward: Callable, embeddings: nn.Module, text: str):
        self.custom_forward = custom_forward
        self.embeddings = embeddings
        self.text = text


class LIGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        text: str,
        input_ids: torch.Tensor,
        ref_input_ids: torch.Tensor,
        sep_id: int,
    ):
        super().__init__(custom_forward, embeddings, text)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.lig = LayerIntegratedGradients(self.custom_forward, self.embeddings)
        self._attributions, self.delta = self.lig.attribute(
            inputs=self.input_ids,
            baselines=self.ref_input_ids,
            return_convergence_delta=True,
        )

    @property
    def word_attributions(self):
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(
                zip(self.text.split(), self.attributions_sum)
            ):
                wa.append((word, float(attribution.data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self):
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0)
        self.attributions_sum = self.attributions_sum / torch.norm(
            self.attributions_sum
        )

    def visualize_attributions(
        self, pred_prob, pred_class, true_class, attr_class, text, all_tokens
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
