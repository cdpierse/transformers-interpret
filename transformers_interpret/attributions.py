from typing import Callable
import torch.nn as nn
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import (
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)
import torch


class Attributions:
    def __init__(self, custom_forward: Callable, embeddings: nn.Module):
        self.custom_forward = custom_forward
        self.embeddings = embeddings


class LIGAttributions(Attributions):
    def __init__(
        self,
        custom_forward: Callable,
        embeddings: nn.Module,
        input_ids,
        ref_input_ids,
        sep_id,
    ):
        super().__init__(custom_forward, embeddings)
        self.custom_forward = custom_forward
        self.embeddings
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.lig = LayerIntegratedGradients(self.custom_forward, self.embeddings)
        self.attributions, self.delta = self.lig.attribute(
            inputs=self.input_ids,
            baselines=self.ref_input_ids,
            return_convergence_delta=True,
        )
        self.summarize_attributions()

    def summarize_attributions(self):
        self.attributions = self.attributions.sum(dim=-1).squeeze(0)
        self.attributions = self.attributions / torch.norm(self.attributions)

    def get_predicted_class(self):
        "will return index of the predicted class etc"
        pass
