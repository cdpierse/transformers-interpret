import os

import pytest
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers_interpret import QuestionAnsweringExplainer
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)

DISTILBERT_QA_MODEL = AutoModelForQuestionAnswering.from_pretrained(
    "mrm8488/bert-tiny-5-finetuned-squadv2"
)
DISTILBERT_QA_TOKENIZER = AutoTokenizer.from_pretrained(
    "mrm8488/bert-tiny-5-finetuned-squadv2"
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
    explainer_question = "what is his name ?"
    explainer_text = "his name is bob"
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
    explainer_question = "what is his name ?"
    explainer_text = "his name is bob"
    word_attributions = qa_explainer(explainer_question, explainer_text)
    assert isinstance(word_attributions, dict)
    assert "start" in word_attributions.keys()
    assert "end" in word_attributions.keys()
    assert len(word_attributions["start"]) == len(word_attributions["end"])


def test_question_answering_word_attributions_input_ids_not_calculated():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )

    with pytest.raises(ValueError):
        qa_explainer.word_attributions


def test_question_answering_start_pos():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is his name ?"
    explainer_text = "his name is Bob"
    qa_explainer(explainer_question, explainer_text)
    start_pos = qa_explainer.start_pos
    assert start_pos == 10


def test_question_answering_end_pos():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is his name ?"
    explainer_text = "his name is Bob"
    qa_explainer(explainer_question, explainer_text)
    end_pos = qa_explainer.end_pos
    assert end_pos == 10


def test_question_answering_start_pos_input_ids_not_calculated():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    with pytest.raises(InputIdsNotCalculatedError):
        qa_explainer.start_pos


def test_question_answering_end_pos_input_ids_not_calculated():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    with pytest.raises(InputIdsNotCalculatedError):
        qa_explainer.end_pos


def test_question_answering_predicted_answer():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is his name ?"
    explainer_text = "his name is Bob"
    qa_explainer(explainer_question, explainer_text)
    predicted_answer = qa_explainer.predicted_answer
    assert predicted_answer == "bob"


def test_question_answering_predicted_answer_input_ids_not_calculated():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    with pytest.raises(InputIdsNotCalculatedError):
        qa_explainer.predicted_answer


def test_question_answering_visualize():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is his name ?"
    explainer_text = "his name is Bob"
    qa_explainer(explainer_question, explainer_text)
    qa_explainer.visualize()


def test_question_answering_visualize_save():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is his name ?"
    explainer_text = "his name is Bob"
    qa_explainer(explainer_question, explainer_text)

    html_filename = "./test/qa_test.html"
    qa_explainer.visualize(html_filename)
    assert os.path.exists(html_filename)
    os.remove(html_filename)


def test_question_answering_visualize_save_append_html_file_ending():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is his name ?"
    explainer_text = "his name is Bob"
    qa_explainer(explainer_question, explainer_text)

    html_filename = "./test/qa_test"
    qa_explainer.visualize(html_filename)
    assert os.path.exists(html_filename + ".html")
    os.remove(html_filename + ".html")


def xtest_question_answering_custom_steps():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is his name ?"
    explainer_text = "his name is Bob"
    qa_explainer(explainer_question, explainer_text, n_steps=1)


def xtest_question_answering_custom_internal_batch_size():
    qa_explainer = QuestionAnsweringExplainer(
        DISTILBERT_QA_MODEL, DISTILBERT_QA_TOKENIZER
    )
    explainer_question = "what is his name ?"
    explainer_text = "his name is Bob"
    qa_explainer(explainer_question, explainer_text, internal_batch_size=1)
