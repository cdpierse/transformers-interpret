from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers_interpret import SequenceClassificationExplainer

model = AutoModelForSequenceClassification.from_pretrained(
    "sampathkethineedi/industry-classification"
)
tokenizer = AutoTokenizer.from_pretrained("sampathkethineedi/industry-classification")
text = "economic impact of competition markets in economic equilibrium and boobs too"
se = SequenceClassificationExplainer(text, model, tokenizer)


input_ids, ref_input_ids, sep_idx = se._make_input_reference_pair(se.text)


attr = se.get_attributions()
wa = attr.word_attributions
for w in wa:
    print(w)
