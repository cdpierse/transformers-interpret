from transformers_interpret.explainers import SequenceClassificationExplainer
model, tokenizer = "model1", "tokenizer"
sce = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)
sce