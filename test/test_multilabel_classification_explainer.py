import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import MultiLabelClassificationExplainer
from transformers_interpret.errors import AttributionTypeNotSupportedError, InputIdsNotCalculatedError

DISTILBERT_MODEL = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
DISTILBERT_TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

BERT_MODEL = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-mini-finetuned-age_news-classification")
BERT_TOKENIZER = AutoTokenizer.from_pretrained("mrm8488/bert-mini-finetuned-age_news-classification")


def test_multilabel_classification_explainer_init_distilbert():
    seq_explainer = MultiLabelClassificationExplainer(DISTILBERT_MODEL, DISTILBERT_TOKENIZER)
    assert seq_explainer.attribution_type == "lig"
    assert seq_explainer.label2id == DISTILBERT_MODEL.config.label2id
    assert seq_explainer.id2label == DISTILBERT_MODEL.config.id2label
    assert seq_explainer.attributions is None
    assert seq_explainer.labels == []


def test_multilabel_classification_explainer_init_bert():
    seq_explainer = MultiLabelClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    assert seq_explainer.attribution_type == "lig"
    assert seq_explainer.label2id == BERT_MODEL.config.label2id
    assert seq_explainer.id2label == BERT_MODEL.config.id2label
    assert seq_explainer.attributions is None
    assert seq_explainer.labels == []


def test_multilabel_classification_explainer_word_attributes_is_dict():
    seq_explainer = MultiLabelClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    wa = seq_explainer("this is a sample text")
    assert isinstance(wa, dict)


def test_multilabel_classification_explainer_word_attributes_is_equals_label_length():
    seq_explainer = MultiLabelClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    wa = seq_explainer("this is a sample text")
    assert len(wa) == len(BERT_MODEL.config.id2label)


def test_multilabel_classification_word_attributions_not_calculated_raises():
    seq_explainer = MultiLabelClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    with pytest.raises(ValueError):
        seq_explainer.word_attributions


def test_multilabel_classification_viz():
    seq_explainer = MultiLabelClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    wa = seq_explainer("this is a sample text")
    seq_explainer.visualize()


@pytest.mark.skip(reason="Slow test")
def test_multilabel_classification_classification_custom_steps():
    explainer_string = "I love you , I like you"
    seq_explainer = MultiLabelClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    seq_explainer(explainer_string, n_steps=1)


@pytest.mark.skip(reason="Slow test")
def test_multilabel_classification_classification_internal_batch_size():
    explainer_string = "I love you , I like you"
    seq_explainer = MultiLabelClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    seq_explainer(explainer_string, internal_batch_size=1)
