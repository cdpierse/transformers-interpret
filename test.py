from transformers_interpret import BaseExplainer, SequenceClassificationExplainer
model, tokenizer = "model1", "tokenizer"
sce = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)
sce.they_all_implement()
