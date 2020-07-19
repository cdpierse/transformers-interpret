from transformers_interpret import BaseExplainer, SequenceClassificationExplainer
model, tokenizer = "model1", "tokenizer"
sce = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)
sce.get_layer_attributions()
sce.get_model_attributions()
