from pprint import pprint

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)

from transformers_interpret import BaseExplainer, SequenceClassificationExplainer

tokenizer = AutoTokenizer.from_pretrained("sampathkethineedi/industry-classification")
model = AutoModelForSequenceClassification.from_pretrained(
    "sampathkethineedi/industry-classification"
)

device = "cpu"
# model = BertForSequenceClassification.from_pretrained(
#     'textattack/bert-base-uncased-rotten-tomatoes')
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
# tokenizer = BertTokenizer.from_pretrained(
#     'textattack/bert-base-uncased-rotten-tomatoes')

text = "For those who like their romance movies filled with unnecessary mysteries, murdered dogs, poached lobsters and the ghosts of deceased little girls, \
        “Dirt Music” will fit the bill. All others need not apply, not even if you’re into the kind of Nicholas Sparks-style drama this movie shamelessly marinates in for an interminable 105 minutes. Director Gregor Jordan’s Australia-set \
        potboiler plays like “Wake in Fright” meets “The Notebook”; the toxic masculinity of several characters wreaks havoc before one guy reveals a softer side that bends toward true love as a means of assuaging his guilt."

sce = SequenceClassificationExplainer(text, model, tokenizer)
atr = sce.get_attributions()
print(atr.attributions)
pprint(dir(model))


# print("="*50)
# print("Inputs are: ", inputs)
# print("="*50)
# print("Types are: ", types)
# print("="*50)
# print("Positions are: ", positions)
