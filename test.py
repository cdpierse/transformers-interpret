from transformers_interpret import BaseExplainer, SequenceClassificationExplainer
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

device = "cpu"
model = BertForSequenceClassification.from_pretrained(
    'textattack/bert-base-uncased-rotten-tomatoes')
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    'textattack/bert-base-uncased-rotten-tomatoes')


be = BaseExplainer(model, tokenizer)
inputs, refs, length = be._make_input_reference_pair(
    "Hey there how are you doing today sir?")
types = be._make_input_reference_token_type_pair(inputs)

print("="*30)
print("Inputs are: ", inputs)
print("="*30)
print("Types are: ", types)

