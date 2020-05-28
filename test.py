from transformers_interpret.interpret import SequenceClassificationExplainer
model, tokenizer = "model", "tokenizer"
sce = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)
