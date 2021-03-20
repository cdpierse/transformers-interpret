import sys
import warnings
from typing import Union

import torch
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer, LIGAttributions
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class QuestionAnsweringExplainer(BaseExplainer):
    """
    Explainer for explaining attributions for models of type `{MODEL_NAME}ForQuestionAnswering`
    from the Transformers package.
    """

    def __init__(
        self,
        question: str,
        text: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
    ):

        super().__init__(question, model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type = attribution_type

        self.label2id = model.config.label2id
        self.id2label = model.config.id2label

        self.attributions: Union[None, LIGAttributions] = None
        self.input_ids: torch.Tensor = torch.Tensor()

        self._single_node_output = False
        self.text = text
        self.question = question

    def encode(self, text: str) -> list:  # type: ignore
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    def _make_input_reference_pair(self, question: str, text: str):  # type: ignore
        question_ids = self.encode(question)
        text_ids = self.encode(text)

        input_ids = (
            [self.cls_token_id]
            + question_ids
            + [self.sep_token_id]
            + text_ids
            + [self.sep_token_id]
        )

        ref_input_ids = (
            [self.cls_token_id]
            + [self.ref_token_id] * len(question_ids)
            + [self.sep_token_id]
            + [self.ref_token_id] * len(text_ids)
            + [self.sep_token_id]
        )

        return (
            torch.tensor([input_ids], device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            len(question_ids),
        )

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position: int = 0,
    ):


        if self.accepts_position_ids:
            preds = self.model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            preds = preds[position]
            return preds.max(1).values

    def _run(
        self, question: str = None, text: str = None, embedding_type: int = None
    ) -> LIGAttributions:
        if embedding_type is None:
            embeddings = self.word_embeddings
        elif embedding_type == 0:
            embeddings = self.word_embeddings
        elif embedding_type == 1:
            if self.accepts_position_ids and self.position_embeddings is not None:
                embeddings = self.position_embeddings
            else:
                warnings.warn(
                    f"This model doesn't support position embeddings for attributions. Defaulting to word embeddings"
                )
                embeddings = self.word_embeddings
        else:
            embeddings = self.word_embeddings

        if question is not None:
            self.question = question
        if text is not None:
            self.text = text

        self._calculate_attributions(embeddings)
        return self.attributions  # type: ignore

    def _calculate_attributions(self, embeddings: Embedding):  # type: ignore

        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.question, self.text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        self.attention_mask = self._make_attention_mask(self.input_ids)
        if self.attribution_type == "lig":
            reference_tokens = [
                token.replace("Ġ", "") for token in self.decode(self.input_ids)
            ]
            lig = LIGAttributions(
                self._forward,
                embeddings,
                reference_tokens,
                self.input_ids,
                self.ref_input_ids,
                self.sep_idx,
                self.attention_mask,
                position_ids=self.position_ids,
                ref_position_ids=self.ref_position_ids,
            )
            lig.summarize()
            self.attributions = lig
        else:
            pass
