import datetime

import captum
from captum.attr import visualization as viz
from IPython.display import Image
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer)

from transformers_interpret import SequenceClassificationExplainer

if captum.__version__ <= "0.3.0":
    from transformers_interpret.custom_visualization import visualize_text
else:
    from viz import visualize_text

# tokenizer = AutoTokenizer.from_pretrained("roberta-large-openai-detector")

# model = AutoModelForSequenceClassification.from_pretrained("roberta-large-openai-detector")

tokenizer = AutoTokenizer.from_pretrained("sampathkethineedi/industry-classification")
model = AutoModelForSequenceClassification.from_pretrained(
    "sampathkethineedi/industry-classification"
)

# tokenizer = AutoTokenizer.from_pretrained("kuzgunlar/electra-turkish-ner")

# model = AutoModelForTokenClassification.from_pretrained("kuzgunlar/electra-turkish-ner")

text ="""Venus Williams was the first of the Williams sisters to make a splash in professional tennis. But Richard Williams, their father, was always convinced that Serena — 15 months younger than Venus, and the youngest of Oracene Price’s five daughters — would go on to have the better career.

Richard Williams was right. While Venus has enjoyed a magnificent career, winning seven Grand Slam singles titles, Serena has won 23 Grand Slams and is widely acclaimed as the best female tennis player (maybe the best tennis player of any gender) of all time. What was true of the Williams sisters — that the younger one went on to be the better athlete — is also true across sports generally. This is the “little sibling effect,” one of the most intriguing findings in sports science: Younger siblings have a significantly higher chance of becoming elite athletes, as University of Utah professor Mark Williams and I explore in our new book, “The Best: How Elite Athletes Are Made.”"""
# NER classification produces output for each named entity it predicts. So if it predicts 2 entities for a sentence we
# have an output of 2*len(input) to generate explanations for, it very similar in many ways to the text classification task

start = datetime.datetime.now()
se = SequenceClassificationExplainer(text, model, tokenizer, attribution_type="lig")

attr = se.get_attributions()
end = datetime.datetime.now()
print(end - start)


wa = attr.word_attributions
for w in wa:
    print(w)

score_viz = se.visualize()
html_object = visualize_text([score_viz],return_html=True)
print(html_object)

print(html_object.data)
print("====")
with open("test.html","w") as file:
    file.write(html_object.data)


print(se.predicted_class_index)
print(se.predicted_class_name)
print(se.selected_index)
print(se)
