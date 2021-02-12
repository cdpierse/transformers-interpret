from unittest.mock import patch

import pytest
import torch
from IPython.core.display import HTML
from torch import Tensor
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers_interpret import SequenceClassificationExplainer
from transformers_interpret.attributions import LIGAttributions
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)

MODEL = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
TOKENIZER = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)


def test_sequence_classification_explainer_init():
    seq_explainer = SequenceClassificationExplainer(
        "I love you, I hate you", MODEL, TOKENIZER
    )
    assert seq_explainer.attribution_type == "lig"
    assert seq_explainer.label2id == MODEL.config.label2id
    assert seq_explainer.id2label == MODEL.config.id2label
    assert seq_explainer.attributions == None


def test_sequence_classification_explainer_init_attribution_type_error():
    with pytest.raises(AttributionTypeNotSupportedError):
        SequenceClassificationExplainer(
            "I love you, I hate you", MODEL, TOKENIZER, attribution_type="UNSUPPORTED"
        )


def test_sequence_classification_encode():
    seq_explainer = SequenceClassificationExplainer(
        "I love you, I hate you", MODEL, TOKENIZER
    )

    _input = "this is a sample of text to be encode"
    tokens = seq_explainer.encode(_input)
    assert isinstance(tokens, list)
    assert tokens[0] != seq_explainer.cls_token_id
    assert tokens[-1] != seq_explainer.sep_token_id
    assert len(tokens) >= len(_input.split(" "))


def test_sequence_classification_encode_no_text_passed():
    explainer_string = "I love you, I hate you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)
    tokens = seq_explainer.encode()
    assert isinstance(tokens, list)
    assert tokens[0] != seq_explainer.cls_token_id
    assert tokens[-1] != seq_explainer.sep_token_id
    assert len(tokens) >= len(explainer_string.split(" "))


def test_sequence_classification_decode():
    explainer_string = "I love you , I hate you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)
    input_ids, _, _ = seq_explainer._make_input_reference_pair(explainer_string)
    decoded = seq_explainer.decode(input_ids)
    assert decoded[0] == seq_explainer.tokenizer.cls_token
    assert decoded[-1] == seq_explainer.tokenizer.sep_token
    assert " ".join(decoded[1:-1]) == explainer_string.lower()


def test_sequence_classification_run_text_given():
    explainer_string = "I love you , I hate you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)
    attributions = seq_explainer.run("I love you, I just love you")
    assert isinstance(attributions, LIGAttributions)

    actual_tokens = [token for token, _ in attributions.word_attributions]
    expected_tokens = [
        "BOS_TOKEN",
        "I",
        "love",
        "you,",
        "I",
        "just",
        "love",
        "you",
        "EOS_TOKEN",
    ]
    assert actual_tokens == expected_tokens


def test_sequence_classification_no_text_given():
    explainer_string = "I love you , I hate you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)
    attributions = seq_explainer.run()
    assert isinstance(attributions, LIGAttributions)

    actual_tokens = [token for token, _ in attributions.word_attributions]
    expected_tokens = [
        "BOS_TOKEN",
        "I",
        "love",
        "you",
        ",",
        "I",
        "hate",
        "you",
        "EOS_TOKEN",
    ]
    assert actual_tokens == expected_tokens


def test_sequence_classification_explain_on_cls_index():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)
    attributions = seq_explainer.run(index=0)
    assert seq_explainer.predicted_class_index == 1
    assert seq_explainer.predicted_class_index != seq_explainer.selected_index
    assert (
        seq_explainer.predicted_class_name
        != seq_explainer.id2label[seq_explainer.selected_index]
    )
    assert seq_explainer.predicted_class_name != "NEGATIVE"
    assert seq_explainer.predicted_class_name == "POSITIVE"


def test_sequence_classification_explain_on_cls_name():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)
    attributions = seq_explainer.run(class_name="NEGATIVE")
    assert seq_explainer.predicted_class_index == 1
    assert seq_explainer.predicted_class_index != seq_explainer.selected_index
    assert (
        seq_explainer.predicted_class_name
        != seq_explainer.id2label[seq_explainer.selected_index]
    )
    assert seq_explainer.predicted_class_name != "NEGATIVE"
    assert seq_explainer.predicted_class_name == "POSITIVE"


def test_sequence_classification_explain_on_cls_name_not_in_dict():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)
    attributions = seq_explainer.run(class_name="UNKNOWN")
    assert seq_explainer.selected_index == 1
    assert seq_explainer.predicted_class_index == 1


def test_sequence_classification_explain_callable():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)

    seq_explainer.run()
    run_method_predicted_index = seq_explainer.predicted_class_index

    seq_explainer()
    call_method_predicted_index = seq_explainer.predicted_class_index

    assert call_method_predicted_index == run_method_predicted_index


def test_sequence_classification_explain_raises_on_input_ids_not_calculated():
    with pytest.raises(InputIdsNotCalculatedError):
        explainer_string = "I love you , I like you"
        seq_explainer = SequenceClassificationExplainer(
            explainer_string, MODEL, TOKENIZER
        )
        seq_explainer.predicted_class_index


def test_sequence_classification_explainer_str():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)
    s = "SequenceClassificationExplainer("
    s += f'\n\ttext="{explainer_string[:10]}...",'
    s += f"\n\tmodel={MODEL.__class__.__name__},"
    s += f"\n\ttokenizer={TOKENIZER.__class__.__name__},"
    s += "\n\tattribution_type='lig',"
    s += ")"
    assert s == seq_explainer.__str__()


def test_sequence_classification_viz():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(explainer_string, MODEL, TOKENIZER)
    seq_explainer()
    seq_explainer.visualize()

