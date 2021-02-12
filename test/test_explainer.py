from unittest.mock import patch

import pytest
import torch
from torch import Tensor
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers_interpret import BaseExplainer

MODEL = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class DummyExplainer(BaseExplainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, text: str = None):
        if text is None:
            text = self.text
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids):
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    def run(self):
        pass

    def _calculate_attributions(self):
        pass

    def _forward(self):
        pass


def test_explainer_init():
    explainer = DummyExplainer("testing", MODEL, TOKENIZER)
    assert explainer.text == "testing"
    assert isinstance(explainer.model, PreTrainedModel)
    assert isinstance(explainer.tokenizer, PreTrainedTokenizerFast) | isinstance(
        explainer.tokenizer, PreTrainedTokenizer
    )
    assert explainer.model_type == MODEL.config.model_type
    if torch.cuda.is_available():
        assert explainer.device.type == "cuda:0"
    else:
        assert explainer.device.type == "cpu"


def test_explainer_make_input_reference_pair():
    explainer = DummyExplainer("this is a test string", MODEL, TOKENIZER)
    input_ids, ref_input_ids, len_inputs = explainer._make_input_reference_pair(
        "this is a test string"
    )
    assert isinstance(input_ids, Tensor)
    assert isinstance(ref_input_ids, Tensor)
    assert isinstance(len_inputs, int)

    assert len(input_ids[0]) == len(ref_input_ids[0]) == (len_inputs + 2)
    assert ref_input_ids[0][0] == input_ids[0][0]
    assert ref_input_ids[0][-1] == input_ids[0][-1]
    assert ref_input_ids[0][0] == explainer.cls_token_id
    assert ref_input_ids[0][-1] == explainer.sep_token_id


def test_explainer_make_input_token_type_pair_no_sep_idx():
    explainer = DummyExplainer("this is a test string", MODEL, TOKENIZER)
    input_ids, ref_input_ids, len_inputs = explainer._make_input_reference_pair(
        "this is a test string"
    )
    (
        token_type_ids,
        ref_token_type_ids,
    ) = explainer._make_input_reference_token_type_pair(input_ids)

    assert ref_token_type_ids[0] == torch.zeros(len(input_ids[0]))[0]
    for i, val in enumerate(token_type_ids):
        if i == 0:
            assert val == 0
        else:
            assert val == 1


def test_explainer_make_input_token_type_pair_sep_idx():
    explainer = DummyExplainer("this is a test string", MODEL, TOKENIZER)
    input_ids, ref_input_ids, len_inputs = explainer._make_input_reference_pair(
        "this is a test string"
    )
    (
        token_type_ids,
        ref_token_type_ids,
    ) = explainer._make_input_reference_token_type_pair(input_ids, 3)

    assert ref_token_type_ids[0] == torch.zeros(len(input_ids[0]))[0]
    for i, val in enumerate(token_type_ids):
        if i <= 3:
            assert val == 0
        else:
            assert val == 1


def test_explainer_make_input_reference_position_id_pair():
    explainer = DummyExplainer("this is a test string", MODEL, TOKENIZER)
    input_ids, ref_input_ids, len_inputs = explainer._make_input_reference_pair(
        "this is a test string"
    )
    position_ids, ref_position_ids = explainer._make_input_reference_position_id_pair(
        input_ids
    )

    assert ref_position_ids[0] == torch.zeros(len(input_ids[0]))[0]
    for i, val in enumerate(position_ids):
        assert val == i


def test_explainer_make_attention_mask():
    explainer = DummyExplainer("this is a test string", MODEL, TOKENIZER)
    input_ids, ref_input_ids, len_inputs = explainer._make_input_reference_pair(
        "this is a test string"
    )
    attention_mask = explainer._make_attention_mask(input_ids)
    assert len(attention_mask[0]) == len(input_ids[0])
    for i, val in enumerate(attention_mask[0]):
        assert val == 1


def test_explainer_str():
    test_string = "this is a test string"
    explainer = DummyExplainer(test_string, MODEL, TOKENIZER)
    s = "DummyExplainer("
    s += f'\n\ttext="{test_string[:10]}...",'
    s += f"\n\tmodel={MODEL.__class__.__name__},"
    s += f"\n\ttokenizer={TOKENIZER.__class__.__name__}"
    s += ")"
    assert s == explainer.__str__()
