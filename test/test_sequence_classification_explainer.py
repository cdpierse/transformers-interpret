import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)

DISTILBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
DISTILBERT_TOKENIZER = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

BERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
    "mrm8488/bert-mini-finetuned-age_news-classification"
)
BERT_TOKENIZER = AutoTokenizer.from_pretrained(
    "mrm8488/bert-mini-finetuned-age_news-classification"
)


def test_sequence_classification_explainer_init_distilbert():
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    assert seq_explainer.attribution_type == "lig"
    assert seq_explainer.label2id == DISTILBERT_MODEL.config.label2id
    assert seq_explainer.id2label == DISTILBERT_MODEL.config.id2label
    assert seq_explainer.attributions is None


def test_sequence_classification_explainer_init_bert():
    seq_explainer = SequenceClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    assert seq_explainer.attribution_type == "lig"
    assert seq_explainer.label2id == BERT_MODEL.config.label2id
    assert seq_explainer.id2label == BERT_MODEL.config.id2label
    assert seq_explainer.attributions is None


def test_sequence_classification_explainer_init_attribution_type_error():
    with pytest.raises(AttributionTypeNotSupportedError):
        SequenceClassificationExplainer(
            DISTILBERT_MODEL,
            DISTILBERT_TOKENIZER,
            attribution_type="UNSUPPORTED",
        )


def test_sequence_classification_explainer_init_with_custom_labels():
    labels = ["label_1", "label_2"]
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER, custom_labels=labels
    )
    assert len(labels) == len(seq_explainer.id2label)
    assert len(labels) == len(seq_explainer.label2id)
    for (k1, v1), (k2, v2) in zip(
        seq_explainer.id2label.items(), seq_explainer.label2id.items()
    ):
        assert v1 in labels and k2 in labels


def test_sequence_classification_explainer_init_custom_labels_size_error():
    with pytest.raises(ValueError):
        SequenceClassificationExplainer(
            DISTILBERT_MODEL, DISTILBERT_TOKENIZER, custom_labels=["few_labels"]
        )


def test_sequence_classification_encode():
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )

    _input = "this is a sample of text to be encoded"
    tokens = seq_explainer.encode(_input)
    assert isinstance(tokens, list)
    assert tokens[0] != seq_explainer.cls_token_id
    assert tokens[-1] != seq_explainer.sep_token_id
    assert len(tokens) >= len(_input.split(" "))


def test_sequence_classification_decode():
    explainer_string = "I love you , I hate you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    input_ids, _, _ = seq_explainer._make_input_reference_pair(explainer_string)
    decoded = seq_explainer.decode(input_ids)
    assert decoded[0] == seq_explainer.tokenizer.cls_token
    assert decoded[-1] == seq_explainer.tokenizer.sep_token
    assert " ".join(decoded[1:-1]) == explainer_string.lower()


def test_sequence_classification_run_text_given():
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    word_attributions = seq_explainer._run("I love you, I just love you")
    assert isinstance(word_attributions, list)

    actual_tokens = [token for token, _ in word_attributions]
    expected_tokens = [
        "[CLS]",
        "i",
        "love",
        "you",
        ",",
        "i",
        "just",
        "love",
        "you",
        "[SEP]",
    ]
    assert actual_tokens == expected_tokens


def test_sequence_classification_explain_on_cls_index():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    seq_explainer._run(explainer_string, index=0)
    assert seq_explainer.predicted_class_index == 1
    assert seq_explainer.predicted_class_index != seq_explainer.selected_index
    assert (
        seq_explainer.predicted_class_name
        != seq_explainer.id2label[seq_explainer.selected_index]
    )
    assert seq_explainer.predicted_class_name != "NEGATIVE"
    assert seq_explainer.predicted_class_name == "POSITIVE"


def test_sequence_classification_explain_position_embeddings():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    pos_attributions = seq_explainer(explainer_string, embedding_type=1)
    word_attributions = seq_explainer(explainer_string, embedding_type=0)

    assert pos_attributions != word_attributions


def test_sequence_classification_explain_position_embeddings_not_available():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    pos_attributions = seq_explainer(explainer_string, embedding_type=1)
    word_attributions = seq_explainer(explainer_string, embedding_type=0)

    assert pos_attributions == word_attributions


def test_sequence_classification_explain_embedding_incorrect_value():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )

    word_attributions = seq_explainer(explainer_string, embedding_type=0)
    incorrect_word_attributions = seq_explainer(explainer_string, embedding_type=-42)

    assert incorrect_word_attributions == word_attributions


def test_sequence_classification_predicted_class_name_no_id2label_defaults_idx():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    seq_explainer.id2label = {"test": "value"}
    seq_explainer._run(explainer_string)
    assert seq_explainer.predicted_class_name == 1


def test_sequence_classification_explain_on_cls_name():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    seq_explainer._run(explainer_string, class_name="NEGATIVE")
    assert seq_explainer.predicted_class_index == 1
    assert seq_explainer.predicted_class_index != seq_explainer.selected_index
    assert (
        seq_explainer.predicted_class_name
        != seq_explainer.id2label[seq_explainer.selected_index]
    )
    assert seq_explainer.predicted_class_name != "NEGATIVE"
    assert seq_explainer.predicted_class_name == "POSITIVE"


def test_sequence_classification_explain_on_cls_name_with_custom_labels():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER, custom_labels=["sad", "happy"]
    )
    seq_explainer._run(explainer_string, class_name="sad")
    assert seq_explainer.predicted_class_index == 1
    assert seq_explainer.predicted_class_index != seq_explainer.selected_index
    assert (
        seq_explainer.predicted_class_name
        != seq_explainer.id2label[seq_explainer.selected_index]
    )
    assert seq_explainer.predicted_class_name != "sad"
    assert seq_explainer.predicted_class_name == "happy"


def test_sequence_classification_explain_on_cls_name_not_in_dict():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    seq_explainer._run(explainer_string, class_name="UNKNOWN")
    assert seq_explainer.selected_index == 1
    assert seq_explainer.predicted_class_index == 1


def test_sequence_classification_explain_raises_on_input_ids_not_calculated():
    with pytest.raises(InputIdsNotCalculatedError):
        seq_explainer = SequenceClassificationExplainer(
            DISTILBERT_MODEL, DISTILBERT_TOKENIZER
        )
        seq_explainer.predicted_class_index


def test_sequence_classification_word_attributions():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    seq_explainer(explainer_string)
    assert isinstance(seq_explainer.word_attributions, list)
    for element in seq_explainer.word_attributions:
        assert isinstance(element, tuple)


def test_sequence_classification_word_attributions_not_calculated_raises():
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    with pytest.raises(ValueError):
        seq_explainer.word_attributions


def test_sequence_classification_explainer_str():
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    s = "SequenceClassificationExplainer("
    s += f"\n\tmodel={DISTILBERT_MODEL.__class__.__name__},"
    s += f"\n\ttokenizer={DISTILBERT_TOKENIZER.__class__.__name__},"
    s += "\n\tattribution_type='lig',"
    s += ")"
    assert s == seq_explainer.__str__()


def test_sequence_classification_viz():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    seq_explainer(explainer_string)
    seq_explainer.visualize()


def sequence_classification_custom_steps():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    seq_explainer(explainer_string, n_steps=1)


def sequence_classification_internal_batch_size():
    explainer_string = "I love you , I like you"
    seq_explainer = SequenceClassificationExplainer(
        DISTILBERT_MODEL, DISTILBERT_TOKENIZER
    )
    seq_explainer(explainer_string, internal_batch_size=1)
