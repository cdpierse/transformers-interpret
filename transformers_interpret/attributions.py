from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
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
        target: Optional[Union[int, Tuple, torch.Tensor, List]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ref_token_type_ids: Optional[torch.Tensor] = None,
        ref_position_ids: Optional[torch.Tensor] = None,
        internal_batch_size: Optional[int] = None,
        n_steps: int = 50,
    ):
        super().__init__(custom_forward, embeddings, tokens)
        self.input_ids = input_ids
        self.ref_input_ids = ref_input_ids
        self.attention_mask = attention_mask
        self.target = target
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.ref_token_type_ids = ref_token_type_ids
        self.ref_position_ids = ref_position_ids
        self.internal_batch_size = internal_batch_size
        self.n_steps = n_steps

        self.lig = LayerIntegratedGradients(self.custom_forward, self.embeddings)

        if self.token_type_ids is not None and self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids, self.position_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_token_type_ids,
                    self.ref_position_ids,
                ),
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        elif self.position_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.position_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_position_ids,
                ),
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
        elif self.token_type_ids is not None:
            self._attributions, self.delta = self.lig.attribute(
                inputs=(self.input_ids, self.token_type_ids),
                baselines=(
                    self.ref_input_ids,
                    self.ref_token_type_ids,
                ),
                target=self.target,
                return_convergence_delta=True,
                additional_forward_args=(self.attention_mask),
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )

        else:
            self._attributions, self.delta = self.lig.attribute(
                inputs=self.input_ids,
                baselines=self.ref_input_ids,
                target=self.target,
                return_convergence_delta=True,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )

    @property
    def word_attributions(self) -> list:
        wa = []
        if len(self.attributions_sum) >= 1:
            for i, (word, attribution) in enumerate(zip(self.tokens, self.attributions_sum)):
                wa.append((word, float(attribution.cpu().data.numpy())))
            return wa

        else:
            raise AttributionsNotCalculatedError("Attributions are not yet calculated")

    def summarize(self, end_idx=None, flip_sign: bool = False):
        if flip_sign:
            multiplier = -1
        else:
            multiplier = 1
        self.attributions_sum = self._attributions.sum(dim=-1).squeeze(0) * multiplier
        self.attributions_sum = self.attributions_sum[:end_idx] / torch.norm(self.attributions_sum[:end_idx])

    def visualize_attributions(self, pred_prob, pred_class, true_class, attr_class, all_tokens):

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
