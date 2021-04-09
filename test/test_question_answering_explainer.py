import pytest
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers_interpret import QuestionAnsweringExplainer
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)

DISTILBERT_QA_MODEL = AutoModelForQuestionAnswering.from_pretrained(
    "sshleifer/tiny-distilbert-base-cased-distilled-squad"
)
DISTILBERT_QA_TOKENIZER = AutoTokenizer.from_pretrained(
    "sshleifer/tiny-distilbert-base-cased-distilled-squad"
)


def test_question_answering_explainer_init_distilbert():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    assert qa_explainer.attribution_type == "lig"
    assert qa_explainer.attributions is None
    assert qa_explainer.position == 0


def test_question_answering_explainer_init_attribution_type_error():
    with pytest.raises(AttributionTypeNotSupportedError):
        QuestionAnsweringExplainer(
            DISTILBERT_QA_MODEL,
            DISTILBERT_QA_TOKENIZER,
            attribution_type="UNSUPPORTED",
        )


def test_question_answering_encode():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )

    _input = "this is a sample of text to be encoded"
    tokens = qa_explainer.encode(_input)
    assert isinstance(tokens, list)
    assert tokens[0] != qa_explainer.cls_token_id
    assert tokens[-1] != qa_explainer.sep_token_id
    assert len(tokens) >= len(_input.split(" "))


def test_question_answering_decode():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is the question ?"
    explainer_text = "this is the text the text that contains the answer"
    input_ids, _, _ = qa_explainer._make_input_reference_pair(
        explainer_question, explainer_text
    )
    decoded = qa_explainer.decode(input_ids)
    assert decoded[0] == qa_explainer.tokenizer.cls_token
    assert decoded[-1] == qa_explainer.tokenizer.sep_token
    assert (
        " ".join(decoded[1:-1])
        == explainer_question.lower() + " [SEP] " + explainer_text.lower()
    )


def test_question_answering_word_attributions():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is the question ?"
    explainer_text = "this is the text the text that contains the answer"
    word_attributions = qa_explainer(
        explainer_question, explainer_text, embedding_type=0
    )
    assert isinstance(word_attributions, dict)
    assert "start" in word_attributions.keys()
    assert "end" in word_attributions.keys()
    assert len(word_attributions["start"]) == len(word_attributions["end"])
