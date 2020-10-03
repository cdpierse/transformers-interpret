from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers_interpret import SequenceClassificationExplainer

model = AutoModelForSequenceClassification.from_pretrained(
    "sampathkethineedi/industry-classification"
)
tokenizer = AutoTokenizer.from_pretrained("sampathkethineedi/industry-classification")
text = "The financial services industry is rife with indexes and economic theory"
se = SequenceClassificationExplainer(text, model, tokenizer)


input_ids, ref_input_ids, sep_idx = se._make_input_reference_pair(se.text)

attr = se.get_attributions()
wa  = attr.word_attributions
for w in wa:
    print(w)
# print(attr)
# # s = ""
# # for i, (word, a) in enumerate(zip(text.split(), attr)):
# #     if i != 0 and i != len(text):
# #         s += f"{word} ({attr[i]}) "
# # print(s)
