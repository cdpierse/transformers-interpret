import datetime

import captum
from captum.attr import visualization as viz
from IPython.display import Image, display
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer)

from transformers_interpret import SequenceClassificationExplainer

if captum.__version__ <= "0.3.0":
    from transformers_interpret.custom_visualization import visualize_text
else:
    from viz import visualize_text


tokenizer = AutoTokenizer.from_pretrained("sampathkethineedi/industry-classification")
model = AutoModelForSequenceClassification.from_pretrained(
    "sampathkethineedi/industry-classification"
)


# tokenizer = AutoTokenizer.from_pretrained("kuzgunlar/electra-turkish-ner")

# model = AutoModelForTokenClassification.from_pretrained("kuzgunlar/electra-turkish-ner")

text = """
The first push will vaccinate about three million people. 

Federal health authorities have recommended that health care workers and nursing home residents be at the front of the line, but the decisions will be left to individual states. 

The US became the sixth country to green light the Pfizer vaccine last night after Britain, Bahrain, Canada, Saudi Arabia and Mexico.

It has been shown to be 95% effective in preventing Covid-19 infection compared to a placebo.

But the Food & Drug Administration (FDA) has advised people who have severe allergies to ingredients in the drug to avoid getting immunised for the time being.

"""
# NER classification produces output for each named entity it predicts. So if it predicts 2 entities for a sentence we
# have an output of 2*len(input) to generate explanations for, it very similar in many ways to the text classification task


se = SequenceClassificationExplainer(text, model, tokenizer, attribution_type="lig")
attr = se.run()

wa = attr.word_attributions
for w in wa:
    print(w)

html_object = se.visualize()
img = display(html_object, display_id=1)
print("====")
print(type(img))
with open("test.html", "w") as file:
    file.write(html_object.data)


print(se.predicted_class_index)
print(se.predicted_class_name)
print(se.selected_index)
print(se)
