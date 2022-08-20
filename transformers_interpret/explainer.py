import inspect
import re
from abc import ABC, abstractmethod, abstractproperty
from typing import List, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseExplainer(ABC):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if self.model.config.model_type == "gpt2":
            self.ref_token_id = self.tokenizer.eos_token_id
        else:
            self.ref_token_id = self.tokenizer.pad_token_id

        self.sep_token_id = (
            self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id
        )
        self.cls_token_id = (
            self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        )

        self.model_prefix = model.base_model_prefix

        nonstandard_model_types = ["roberta"]
        if (
            self._model_forward_signature_accepts_parameter("position_ids")
            and self.model.config.model_type not in nonstandard_model_types
        ):
            self.accepts_position_ids = True
        else:
            self.accepts_position_ids = False

        if (
            self._model_forward_signature_accepts_parameter("token_type_ids")
            and self.model.config.model_type not in nonstandard_model_types
        ):
            self.accepts_token_type_ids = True
        else:
            self.accepts_token_type_ids = False

        self.device = self.model.device

        self.word_embeddings = self.model.get_input_embeddings()
        self.position_embeddings = None
        self.token_type_embeddings = None

        self._set_available_embedding_types()

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

    @abstractproperty
    def word_attributions(self):
        raise NotImplementedError

    @abstractmethod
    def _run(self) -> list:
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

    def _make_input_reference_pair(self, text: Union[List, str]) -> Tuple[torch.Tensor, torch.Tensor, int]:
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
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # if no special tokens were added
        if len(text_ids) == len(input_ids):
            ref_input_ids = [self.ref_token_id] * len(text_ids)
        else:
            ref_input_ids = [self.cls_token_id] + [self.ref_token_id] * len(text_ids) + [self.sep_token_id]

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
        token_type_ids = torch.tensor([0 if i <= sep_idx else 1 for i in range(seq_len)], device=self.device).expand_as(
            input_ids
        )
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=self.device).expand_as(input_ids)

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
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        ref_position_ids = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return (position_ids, ref_position_ids)

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(input_ids)

    def _get_preds(
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):

        if self.accepts_position_ids and self.accepts_token_type_ids:
            preds = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            return preds

        elif self.accepts_position_ids:
            preds = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            return preds
        elif self.accepts_token_type_ids:
            preds = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

            return preds
        else:
            preds = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            return preds

    def _clean_text(self, text: str) -> str:
        text = re.sub("([.,!?()])", r" \1 ", text)
        text = re.sub("\s{2,}", " ", text)
        return text

    def _model_forward_signature_accepts_parameter(self, parameter: str) -> bool:
        signature = inspect.signature(self.model.forward)
        parameters = signature.parameters
        return parameter in parameters

    def _set_available_embedding_types(self):
        model_base = getattr(self.model, self.model_prefix)
        if self.model.config.model_type == "gpt2" and hasattr(model_base, "wpe"):
            self.position_embeddings = model_base.wpe.weight
        else:
            if hasattr(model_base, "embeddings"):
                self.model_embeddings = getattr(model_base, "embeddings")
                if hasattr(self.model_embeddings, "position_embeddings"):
                    self.position_embeddings = self.model_embeddings.position_embeddings
                if hasattr(self.model_embeddings, "token_type_embeddings"):
                    self.token_type_embeddings = self.model_embeddings.token_type_embeddings

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__}"
        s += ")"

        return s
