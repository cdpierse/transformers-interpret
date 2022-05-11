import pytest
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers_interpret import TokenClassificationExplainer
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)

BERT_MODEL = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
BERT_TOKENIZER = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

ROBERTA_MODEL = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ROBERTA_TOKENIZER = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")


def test_token_classification_explainer_init_bert():
    ner_explainer = TokenClassificationExplainer(
        BERT_MODEL, BERT_TOKENIZER
    )
    assert ner_explainer.attribution_type == "lig"
    assert ner_explainer.label2id == BERT_MODEL.config.label2id
    assert ner_explainer.id2label == BERT_MODEL.config.id2label
    assert ner_explainer.attributions is None


def test_token_classification_explainer_init_bert():
    ner_explainer = TokenClassificationExplainer(
        ROBERTA_MODEL, ROBERTA_TOKENIZER
    )
    assert ner_explainer.attribution_type == "lig"
    assert ner_explainer.label2id == ROBERTA_MODEL.config.label2id
    assert ner_explainer.id2label == ROBERTA_MODEL.config.id2label
    assert ner_explainer.attributions is None


def test_token_classification_explainer_init_attribution_type_error():
    with pytest.raises(AttributionTypeNotSupportedError):
        TokenClassificationExplainer(
            BERT_MODEL,
            BERT_TOKENIZER,
            attribution_type="UNSUPPORTED",
        )

def test_token_classification_explainer_init_ignored_indexes():
    indexes = [0, 1, 6]
    ner_explainer = TokenClassificationExplainer(
        BERT_MODEL, BERT_TOKENIZER, ignored_indexes=indexes
    )
    assert len(indexes) == len(ner_explainer.ignored_indexes)
    
    for index in ner_explainer.ignored_indexes:
        assert index in indexes


def test_token_classification_explainer_init_ignored_labels():
    labels = ['O', 'I-LOC', 'UNKNOWN_LABEL']
    ner_explainer = TokenClassificationExplainer(
        BERT_MODEL, BERT_TOKENIZER, ignored_labels=labels
    )
    assert len(labels) == len(ner_explainer.ignored_labels)
    
    for label in ner_explainer.ignored_labels:
        assert label in labels


def test_token_classification_selected_indexes():
    explainer_string = "We visited Paris during the weekend, where Emmanuel Macron lives."
    expected_all_indexes = list(range(15))
    indexes = [0,1,2,3,4,5,6,7,8,9,11,12,13]
    ner_explainer = TokenClassificationExplainer(
        BERT_MODEL, BERT_TOKENIZER, ignored_indexes=indexes
    )

    word_attributions = ner_explainer(explainer_string) 

    assert len(ner_explainer.selected_indexes) == (len(expected_all_indexes) - len(indexes))

    for index in ner_explainer.selected_indexes: 
        assert index in expected_all_indexes 
        assert index not in indexes


def test_token_classification_selected_indexes_all():
    explainer_string = "We visited Paris during the weekend, where Emmanuel Macron lives."
    expected_all_indexes = list(range(15))
    ner_explainer = TokenClassificationExplainer(
        BERT_MODEL, BERT_TOKENIZER
    )

    word_attributions = ner_explainer(explainer_string)

    assert len(ner_explainer.selected_indexes) == ner_explainer.input_ids.shape[1]

    for i, index in enumerate(ner_explainer.selected_indexes):
        assert i == index 


def test_token_classification_selected_indexes_raises_on_input_ids_not_calculated():
    indexes = [0,2,3]
    with pytest.raises(InputIdsNotCalculatedError):
        ner_explainer = TokenClassificationExplainer(
            BERT_MODEL, BERT_TOKENIZER, ignored_indexes=indexes
         )
        ner_explainer.selected_indexes


def test_token_classification_selected_labels():
    ignored_labels = ['O', 'I-LOC']
    ner_explainer = TokenClassificationExplainer(
        BERT_MODEL, BERT_TOKENIZER, ignored_labels=ignored_labels
    )
    all_labels = list(BERT_MODEL.config.label2id.keys())

    for label in ner_explainer.selected_labels:
        assert label in all_labels 
        assert label not in ignored_labels 


def test_token_classification_selected_labels_all():
    ner_explainer = TokenClassificationExplainer(
        BERT_MODEL, BERT_TOKENIZER
    )
    all_labels = list(BERT_MODEL.config.label2id.keys())

    assert len(ner_explainer.selected_labels) == len(all_labels)

    for label in ner_explainer.selected_labels:
        assert label in all_labels


def test_token_classification_selected_indexes_and_selected_labels():
    ignored_labels = ['O', 'I-PER', 'B-PER']
    ignored_indexes = [1,2]
    explainer_string = "We visited Paris last weekend"
    selected_indexes = [3]

    ner_explainer = TokenClassificationExplainer(
        BERT_MODEL, BERT_TOKENIZER, ignored_indexes=ignored_indexes, ignored_labels=ignored_labels
    )
    ner_explainer._run(explainer_string)

    assert len(selected_indexes) == len(ner_explainer._selected_indexes) 

    for i, index in enumerate(ner_explainer._selected_indexes):
        assert selected_indexes[i] == index


def test_token_classification_encode():
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)

    _input = "this is a sample of text to be encoded"
    tokens = ner_explainer.encode(_input)
    assert isinstance(tokens, list)
    assert tokens[0] != ner_explainer.cls_token_id
    assert tokens[-1] != ner_explainer.sep_token_id
    assert len(tokens) >= len(_input.split(" "))


def test_token_classification_decode():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    input_ids, _, _ = ner_explainer._make_input_reference_pair(explainer_string)
    decoded = ner_explainer.decode(input_ids)
    assert decoded[0] == ner_explainer.tokenizer.cls_token
    assert decoded[-1] == ner_explainer.tokenizer.sep_token
    assert " ".join(decoded[1:-1]) == explainer_string


def test_token_classification_run_text_given():
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    word_attributions = ner_explainer._run("We visited Paris during the weekend")
    assert isinstance(word_attributions, list)

    actual_tokens = [token for token, _ in word_attributions]
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

    assert pos_attributions != word_attributions


def test_token_classification_explain_position_embeddings_incorrect_value():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    
    word_attributions = ner_explainer(explainer_string, embedding_type=0)
    incorrect_word_attributions = ner_explainer(explainer_string, embedding_type=-42)

    assert incorrect_word_attributions == word_attributions 


def test_token_classification_predicted_class_names():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    ner_explainer._run(explainer_string)
    ground_truths = ['O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O']

    assert len(ground_truths) == len(ner_explainer.predicted_class_names)

    for i, class_id in enumerate(ner_explainer.predicted_class_names):
        assert ground_truths[i] == class_id


def test_token_classification_predicted_class_names_no_id2label_defaults_idx():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    ner_explainer.id2label = {"test": "value"}
    ner_explainer._run(explainer_string)
    ground_truths = [0, 0, 0, 7, 0, 0, 0, 0]

    assert len(ground_truths) == len(ner_explainer.predicted_class_names)

    for i, class_id in enumerate(ner_explainer.predicted_class_names):
        assert ground_truths[i] == class_id


def test_token_classification_explain_raises_on_input_ids_not_calculated():
    with pytest.raises(InputIdsNotCalculatedError):
        ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
        ner_explainer.predicted_class_indexes


def test_token_classification_word_attributions():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    ner_explainer(explainer_string)

    assert isinstance(ner_explainer.word_attributions, list)

    for element in ner_explainer.word_attributions:
        assert isinstance(element, tuple)


def test_token_classification_word_attributions_not_calculated_raises():
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    with pytest.raises(ValueError):
        ner_explainer.word_attributions


def test_token_classification_explainer_str():
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    s = "TokenClassificationExplainer("
    s += f"\n\tmodel={BERT_MODEL.__class__.__name__},"
    s += f"\n\ttokenizer={BERT_TOKENIZER.__class__.__name__},"
    s += "\n\tattribution_type='lig',"
    s += ")"
    assert s == ner_explainer.__str__()


def test_token_classification_viz():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    ner_explainer(explainer_string)
    ner_explainer.visualize()


def test_token_classification_viz_on_true_classes_value_error():
    explainer_string = "We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    ner_explainer(explainer_string)
    true_classes = ['None', 'Location', 'None']
    with pytest.raises(ValueError):
        ner_explainer.visualize(true_classes=true_classes)


def token_classification_custom_steps():
    explainer_string ="We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    ner_explainer(explainer_string, n_steps=1)


def token_classification_internal_batch_size():
    explainer_string ="We visited Paris during the weekend"
    ner_explainer = TokenClassificationExplainer(BERT_MODEL, BERT_TOKENIZER)
    ner_explainer(explainer_string, internal_batch_size=1)
