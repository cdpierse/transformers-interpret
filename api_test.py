from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers_interpret import SequenceClassificationExplainer

model = AutoModelForSequenceClassification.from_pretrained(
    "sampathkethineedi/industry-classification"
)
tokenizer = AutoTokenizer.from_pretrained(
    "sampathkethineedi/industry-classification")

se = SequenceClassificationExplainer(
    "hello there how are you doing", model, tokenizer)


input_ids, ref_input_ids, sep_idx = se._make_input_reference_pair(
    se.text)

print(se.get_attributions().attributions)
