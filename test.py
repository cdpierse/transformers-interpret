from transformers_interpret import BaseExplainer
model, tokenizer = "model1", "tokenizer"
sce = BaseExplainer(model=model, tokenizer=tokenizer)