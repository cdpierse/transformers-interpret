import abc
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseExplainer(ABC):
    def __init__(
        self, text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        text = self._clean_text(text)
        self.text = text

        self.ref_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id

        self.model_type = model.config.model_type

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def encode(self, text: str = None):
        """
        Encode given text with a model's tokenizer.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, input_ids: torch.Tensor) -> List[str]:
        """
        Decode received input_ids into a list of word tokens.


        Args:
            input_ids (torch.Tensor): Input ids representing
            word tokens for a sentence/document.

        """
        raise NotImplementedError

    @abstractmethod
    def run(self):  # Add abstract type return for attribution
        raise NotImplementedError

    @abstractmethod
    def _forward(self):
        """
        Forward defines a function for passing inputs
        through a models's forward method.

        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_attributions(self):
        """
        Internal method for calculating the attribution
        values for the input text.

        """
        raise NotImplementedError

    def _make_input_reference_pair(
        self, text: Union[List, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Tokenizes `text` to numerical token id  representation `input_ids`,
        as well as creating another reference tensor `ref_input_ids` of the same length
        that will be used as baseline for attributions. Additionally
        the length of text without special tokens appended is prepended is also
        returned.

        Args:
            text (str): Text for which we are creating both input ids
            and their corresponding reference ids

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]
        """

        if isinstance(text, list):
            raise NotImplementedError("Lists of text are not currently supported.")

        text_ids = self.encode(text)
        input_ids = [self.cls_token_id] + text_ids + [self.sep_token_id]
        ref_input_ids = (
            [self.cls_token_id]
            + [self.ref_token_id] * len(text_ids)
            + [self.sep_token_id]
        )
        return (
            torch.tensor([input_ids], device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            len(text_ids),
        )

    def _make_input_reference_token_type_pair(
        self, input_ids: torch.Tensor, sep_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            [0 if i <= sep_idx else 1 for i in range(seq_len)], device=self.device
        )
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=self.device)

        return (token_type_ids, ref_token_type_ids)

    def _make_input_reference_position_id_pair(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tensors for positional encoding of tokens for input_ids and zeroed tensor for reference ids.

        Args:
            input_ids (torch.Tensor): inputs to create positional encoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        ref_position_ids = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return (position_ids, ref_position_ids)

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(input_ids)

    def _clean_text(self, text: str) -> str:
        text = re.sub("([.,!?()])", r" \1 ", text)
        text = re.sub("\s{2,}", " ", text)
        return text

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f'\n\ttext="{str(self.text[:10])}...",'
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__}"
        s += ")"

        return s
