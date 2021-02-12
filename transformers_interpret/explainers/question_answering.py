import sys
import warnings

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer, LIGAttributions
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)


class QuestionAnsweringExplainer(BaseExplainer):
    """
    Explainer for explaining attributions for models of type `{MODEL_NAME}ForQuestionAnswering`
    from the Transformers package.
    """

    pass

    # def __init__(
    #     self,
    #     text: str,
    #     model: PreTrainedModel,
    #     tokenizer: PreTrainedTokenizer,
    #     attribution_type: str = "lig",
    # ):
    #     super().__init__(text, model, tokenizer)
    #     if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
    #         raise AttributionTypeNotSupportedError(
    #             f"Attribution type '{attribution_type}' is not supported. Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"
    #         )
    #     self.attribution_type = attribution_type

    #     self.label2id = model.config.label2id
    #     self.id2label = model.config.id2label

    #     self.attributions = None
