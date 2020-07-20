import pandas
from transformers import PreTrainedModel, PreTrainedTokenizer
from abc import abstractmethod
import abc
from typing import Tuple
import torch


""" transfromers-interpret 
adapts the great work being done by the team over at Captum (https://github.com/pytorch/captum).
Considering this packages' sole focus on explaining Language Models from the transformers team 
the tutorial found here https://github.com/pytorch/captum/blob/master/tutorials/Bert_SQUAD_Interpret.ipynb
has been a huge help and much of what I've written here adapts @NarineK work. 

Thanks to everyone involved for building a stellar tool for explainable A.I.
"""


class BaseExplainer:
    ALLOWED_MODELS = [
        "model1"
    ]

    SUPPORTED_ATTRIBUTION_TYPES = [
        "lig"
    ]
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 attribution_type: str = "lig"):
        """

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

    def get_attributions(self, text: str):

        pass

    @abstractmethod
    def get_layer_attributions(self):
        pass

    @abstractmethod
    def custom_forward(self):
        pass

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

    def _make_input_reference_pair(self, text: str):
        text_ids = self.encode(text)
        input_ids = [self.cls_token_id] + text_ids + [self.sep_token_id]
        ref_input_ids = [self.cls_token_id] + \
            [self.ref_token_id] * len(text_ids) + [self.sep_token_id]
        return (torch.tensor([input_ids], device=self.device),
                torch.tensor([ref_input_ids], device=self.device),
                len(text_ids))

    def _make_input_reference_token_type_pair(self, input_ids: torch.Tensor):
        pass

    def _make_input_reference_position_id_pair(self, input_ids: torch.Tensor):
        pass

    def _make_attention_mask(self, input_ids: torch.Tensor):
        pass


class SequenceClassificationExplainer(BaseExplainer):

    def get_layer_attributions(self):
        print("SCE attributions")

    def get_model_attributions(self):
        print("SCE model attributions")


class QuestionAnsweringExplainer(BaseExplainer):
    pass


class NERExplainer(BaseExplainer):
    pass


TokenClassificationExplainer = NERExplainer


class LMExplainer(BaseExplainer):
    pass


class SummarizationExplainer(BaseExplainer):
    pass
