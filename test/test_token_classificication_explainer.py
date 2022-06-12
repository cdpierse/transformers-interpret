import pytest
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers_interpret import TokenClassificationExplainer
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)

DISTILBERT_MODEL = AutoModelForTokenClassification.from_pretrained(
    "elastic/distilbert-base-cased-finetuned-conll03-english"
)
DISTILBERT_TOKENIZER = AutoTokenizer.from_pretrained(
    "elastic/distilbert-base-cased-finetuned-conll03-english"
)


BERT_MODEL = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
BERT_TOKENIZER = AutoTokenizer.from_pretrained("dslim/bert-base-NER")


def test_token_classification_explainer_init_distilbert():
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    assert ner_explainer.attribution_type == "lig"
    assert ner_explainer.label2id == DISTILBERT_MODEL.config.label2id
    assert ner_explainer.id2label == DISTILBERT_MODEL.config.id2label
    assert ner_explainer.attributions is None


def test_token_classification_explainer_init_bert():
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    assert ner_explainer.attribution_type == "lig"
    assert ner_explainer.label2id == BERT_MODEL.config.label2id
    assert ner_explainer.id2label == BERT_MODEL.config.id2label
    assert ner_explainer.attributions is None


def test_token_classification_explainer_init_attribution_type_error():
    with pytest.raises(AttributionTypeNotSupportedError):
        TokenClassificationExplainer(
            DISTILBERT_MODEL,
            DISTILBERT_TOKENIZER,
            attribution_type="UNSUPPORTED",
        )


def test_token_classification_selected_indexes_only_ignored_indexes():
    explainer_string = (
        "We visited Paris during the weekend, where Emmanuel Macron lives."
    )
    expected_all_indexes = list(range(15))
    indexes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13]
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)

    word_attributions = ner_explainer(explainer_string, ignored_indexes=indexes)

    assert len(ner_explainer._selected_indexes) == (
        len(expected_all_indexes) - len(indexes)
    )

    for index in ner_explainer._selected_indexes:
        assert index in expected_all_indexes
        assert index not in indexes


def test_token_classification_selected_indexes_only_ignored_labels():
    ignored_labels = ["O", "I-LOC", "B-LOC"]
    indexes = [8, 9, 10]
    explainer_string = "We visited Paris last weekend, where Emmanuel Macron lives."

    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)

    word_attributions = ner_explainer(explainer_string, ignored_labels=ignored_labels)

    assert len(indexes) == len(ner_explainer._selected_indexes)

    for index in ner_explainer._selected_indexes:
        assert index in indexes


def test_token_classification_selected_indexes_all():
    explainer_string = (
        "We visited Paris during the weekend, where Emmanuel Macron lives."
    )
    expected_all_indexes = list(range(15))
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)

    word_attributions = ner_explainer(explainer_string)

    assert len(ner_explainer._selected_indexes) == ner_explainer.input_ids.shape[1]

    for i, index in enumerate(ner_explainer._selected_indexes):
        assert i == index


def test_token_classification_selected_indexes_ignored_indexes_and_labels():
    ignored_labels = ["O", "I-PER", "B-PER"]
    ignored_indexes = [4, 5, 6]
    explainer_string = "We visited Paris last weekend"
    selected_indexes = [3]  # this models classifies erroniously '[SEP]' as a location

    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    word_attributions = ner_explainer(
        explainer_string, ignored_indexes=ignored_indexes, ignored_labels=ignored_labels
    )

    assert len(selected_indexes) == len(ner_explainer._selected_indexes)

    for i, index in enumerate(ner_explainer._selected_indexes):
        assert selected_indexes[i] == index


def test_token_classification_encode():
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)

    _input = "this is a sample of text to be encoded"
    tokens = ner_explainer.encode(_input)
    assert isinstance(tokens, list)
    assert tokens[0] != ner_explainer.cls_token_id
    assert tokens[-1] != ner_explainer.sep_token_id
    assert len(tokens) >= len(_input.split(" "))


def test_token_classification_decode():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    input_ids, _, _ = ner_explainer._make_input_reference_pair(explainer_string)
    decoded = ner_explainer.decode(input_ids)
    assert decoded[0] == ner_explainer.tokenizer.cls_token
    assert decoded[-1] == ner_explainer.tokenizer.sep_token
    assert " ".join(decoded[1:-1]) == explainer_string


def test_token_classification_run_text_given():
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    word_attributions = ner_explainer._run("We visited Paris during the weekend")
    assert isinstance(word_attributions, dict)

    actual_tokens = list(word_attributions.keys())
    expected_tokens = [
        "[CLS]",
        "We",
        "visited",
        "Paris",
        "during",
        "the",
        "weekend",
        "[SEP]",
    ]
    assert actual_tokens == expected_tokens


def test_token_classification_explain_position_embeddings():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    pos_attributions = ner_explainer(explainer_string, embedding_type=1)
    word_attributions = ner_explainer(explainer_string, embedding_type=0)

    for token in ner_explainer.word_attributions.keys():
        assert pos_attributions != word_attributions


def test_token_classification_explain_position_embeddings_incorrect_value():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)

    word_attributions = ner_explainer(explainer_string, embedding_type=0)
    incorrect_word_attributions = ner_explainer(explainer_string, embedding_type=-42)

    assert incorrect_word_attributions == word_attributions


def test_token_classification_predicted_class_names():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    ner_explainer._run(explainer_string)
    ground_truths = ["O", "O", "O", "B-LOC", "O", "O", "O", "O"]

    assert len(ground_truths) == len(ner_explainer.predicted_class_names)

    for i, class_id in enumerate(ner_explainer.predicted_class_names):
        assert ground_truths[i] == class_id


def test_token_classification_predicted_class_names_no_id2label_defaults_idx():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    ner_explainer.id2label = {"test": "value"}
    ner_explainer._run(explainer_string)
    class_labels = list(range(9))

    assert len(ner_explainer.predicted_class_names) == 8

    for class_name in ner_explainer.predicted_class_names:
        assert class_name in class_labels


def test_token_classification_explain_raises_on_input_ids_not_calculated():
    with pytest.raises(InputIdsNotCalculatedError):
        ner_explainer = TokenClassificationExplainer(
            DISTILBERT_MODEL, DISTILBERT_TOKENIZER
        )
        ner_explainer.predicted_class_indexes


def test_token_classification_word_attributions():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    ner_explainer(explainer_string)

    assert isinstance(ner_explainer.word_attributions, dict)

    for token, elements in ner_explainer.word_attributions.items():
        assert isinstance(elements, dict)
        assert list(elements.keys()) == ["label", "attribution_scores"]
        assert isinstance(elements["label"], str)
        assert isinstance(elements["attribution_scores"], list)
        for score in elements["attribution_scores"]:
            assert isinstance(score, tuple)
            assert isinstance(score[0], str)
            assert isinstance(score[1], float)


def test_token_classification_word_attributions_not_calculated_raises():
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    with pytest.raises(ValueError):
        ner_explainer.word_attributions


def test_token_classification_explainer_str():
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    s = "TokenClassificationExplainer("
    s += f"\n\tmodel={DISTILBERT_MODEL.__class__.__name__},"
    s += f"\n\ttokenizer={DISTILBERT_TOKENIZER.__class__.__name__},"
    s += "\n\tattribution_type='lig',"
    s += ")"
    assert s == ner_explainer.__str__()


def test_token_classification_viz():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    ner_explainer(explainer_string)
    ner_explainer.visualize()


def test_token_classification_viz_on_true_classes_value_error():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    ner_explainer(explainer_string)
    true_classes = ["None", "Location", "None"]
    with pytest.raises(ValueError):
        ner_explainer.visualize(true_classes=true_classes)


def token_classification_custom_steps():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    ner_explainer(explainer_string, n_steps=1)


def token_classification_internal_batch_size():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    ner_explainer(explainer_string, internal_batch_size=1)
