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
        # self.custom_forward = custom_forward
        # self.embeddings
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
        if len(self.attributions) >= 1:
            for i, (word, attribution) in enumerate(
                zip(self.text.split(), self.attributions)
            ):
                wa.append((word, float(attribution.data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self):
        print("I run")
        self.attributions = self._attributions.sum(dim=-1).squeeze(0)
        self.attributions = self.attributions / torch.norm(self.attributions)

    def get_predicted_class(self):
        "will return index of the predicted class etc"
        pass
