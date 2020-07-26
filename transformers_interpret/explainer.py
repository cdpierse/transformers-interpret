import abc
from abc import abstractmethod
from typing import Tuple, List, Union

import pandas
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .attributions import LIGAttributions, Attributions


""" transfromers-interpret
adapts the amazing work being done by the team over at Captum (https://github.com/pytorch/captum).
Considering this packages' sole focus on explaining Language Models from the Transformers team
the tutorial found here https://github.com/pytorch/captum/blob/master/tutorials/Bert_SQUAD_Interpret.ipynb
has been a huge help and much of what I've written here adapts @NarineK's work.

Thanks to everyone involved for building a stellar tool for explainable A.I.
"""


class BaseExplainer:
    ALLOWED_MODELS = [
        "model1"
    ]

    SUPPORTED_ATTRIBUTION_TYPES = [
        "lig"
    ]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 attribution_type: str = "lig"):
        """
        [summary]

        Args:
            model (PreTrainedModel): [description]
            tokenizer (PreTrainedTokenizer): [description]
            attribution_type (str, optional): [description]. Defaults to "lig".

        Raises:
            NotImplementedError: [description]
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.attribution_type = attribution_type.lower()

        if attribution_type not in self.SUPPORTED_ATTRIBUTION_TYPES:
            raise NotImplementedError(
                "Model Attribution Explanation is currently not supported for '{}'. \
                Please select a an attribution method from {}.".format(
                    self.attribution_type, self.SUPPORTED_ATTRIBUTION_TYPES)
            )

        self._get_special_token_ids()

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text,
                                     add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        indices = input_ids[0].detach().tolist()
        return self.tokenizer.convert_ids_to_tokens(input_ids)

    def _get_special_token_ids(self):
        self.ref_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id

    def _make_input_reference_pair(self, text: Union[List, str]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Tokenizes `text` to numerical token id  representation `input_ids`,
        as well as creating another reference tensor of the same length
        that will be used as baseline for attributions `ref_input_ids`. Additionally
        the length of text without special tokens appended is prepended is also
        returned.

        Args:
            text (str): Text for which we are creating both input ids
            and their corresponding reference ids

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]
        """

        if isinstance(text, list):
            raise NotImplementedError(
                "Lists of text are not currently supported.")

        text_ids = self.encode(text)
        input_ids = [self.cls_token_id] + text_ids + [self.sep_token_id]
        ref_input_ids = [self.cls_token_id] + \
            [self.ref_token_id] * len(text_ids) + [self.sep_token_id]
        return (torch.tensor([input_ids], device=self.device),
                torch.tensor([ref_input_ids], device=self.device),
                len(text_ids))

    def _make_input_reference_token_type_pair(self, input_ids: torch.Tensor, sep_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two tensors indicating the corresponding token types for the `input_ids`
        and a corresponding all zero reference token type tensor.
        Args:
            input_ids (torch.Tensor): Tensor of text converted to `input_ids`
            sep_idx (int, optional):  Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor(
            [0 if i <= sep_idx else 1 for i in range(seq_len)],
            device=self.device
        )
        ref_token_type_ids = torch.zeros_like(
            token_type_ids, device=self.device)

        return (token_type_ids, ref_token_type_ids)

    def _make_input_reference_position_id_pair(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tensors for positional encoding of tokens for input_ids and zeroed tensor for reference ids.

        Args:
            input_ids (torch.Tensor): inputs to create positional encoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        seq_len = input_ids.size(1)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=self.device)
        ref_position_ids = torch.zeros(
            seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return (position_ids, ref_position_ids)

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(input_ids)

    @staticmethod
    def get_model_namespace(model):
        # TODO: #2 @cdpierse write method for extracting the appropriate namespace from a model
        return "distilbert"


class SequenceClassificationExplainer(BaseExplainer):
    """
    Instantiates an explainer for Sequence Classification based models
    from the Transformer package by Hugging Face.
    """

    def __init__(self,
                 text: Union[List, str],
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 attribution_type: str = "lig"):
        """
        Instantiates an explainer for Sequence Classification based models
        from the Transformer package by Hugging Face.

        Args:
            text (Union[List, str]): The texts or list of texts to derive explanations for.
            model (PreTrainedModel): A pretrained model with a sequence classification head. 
            tokenizer (PreTrainedTokenizer): The official tokenizer for the model
            attribution_type (str, optional): The type of attributions we want to calculate. Defaults to "lig".
        """

        super().__init__(model, tokenizer, attribution_type)

        self.text = text
        self.input_ids, self.ref_input_ids, self.sep_idx = self._make_input_reference_pair(
            self.text)
        self.calculate_attributions()

    def forward(self, inputs):
        preds = self.model(inputs)[0]
        return torch.softmax(preds, dim=1)[0][1].unsqueeze(-1)

    def calculate_attributions(self):
        if self.attribution_type == "lig":
            namespace = self.get_model_namespace(self.model)
            embeddings = getattr(self.model, namespace).embeddings
            lig = LIGAttributions(self.forward,
                                  embeddings,
                                  self.input_ids,
                                  self.ref_input_ids,
                                  self.sep_idx)
            self.attributions = lig
            all_tokens = self.tokenizer.convert_ids_to_tokens(
                self.input_ids[0].detach().tolist())

        else:
            raise NotImplementedError(
                "Other attribution types are not added yet.")

    def get_attributions(self) -> LIGAttributions:
        return self.attributions


class QuestionAnsweringExplainer(BaseExplainer):

    def forward(self):
        "Custom forward function for Question Answering"
        # click link for help implementing
        #  https://github.com/pytorch/captum/blob/master/tutorials/Bert_SQUAD_Interpret.ipynb
        pass

    pass


class NERExplainer(BaseExplainer):
    pass


TokenClassificationExplainer = NERExplainer


class LMExplainer(BaseExplainer):
    pass


class SummarizationExplainer(BaseExplainer):
    pass
