from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers_interpret import SequenceClassificationExplainer

model = AutoModelForSequenceClassification.from_pretrained(
    "sampathkethineedi/industry-classification"
)
tokenizer = AutoTokenizer.from_pretrained("sampathkethineedi/industry-classification")
text = "Migrating on premises servers to the cloud using kubernetes and distributed computing"
se = SequenceClassificationExplainer(text, model, tokenizer)


input_ids, ref_input_ids, sep_idx = se._make_input_reference_pair(se.text)


attr = se.get_attributions()
wa = attr.word_attributions
for w in wa:
    print(w)

print(se.predicted_class_index)
print(se.predicted_class_name)
