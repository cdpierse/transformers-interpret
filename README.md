# Transformers-Interpret

Transformers-interpret is a model explainability tool designed to work exclusively with 🤗 transformers.


In line with the philosophy of the transformers package tranformers-interpret allows any transformers model to be explained in just two lines. It even supports visualizations in both notebooks and as savable html files. 

This package stands on the shoulder of the the incredible work being done by the teams at [Pytorch Captum](https://captum.ai/) and  [Hugging Face](https://huggingface.co/) and would not exist if not for the amazing job they are both doing in the fields of nlp and model interpretability respectively.  


## Install

```bash
pip install transformers_interpret
```

## Quick Start

Let's start by import some our auto model and tokenizer class from transformers and initializing a model and tokenizer. 
For this example we are using `distilbert-base-uncased-finetuned-sst-2-english` a distilbert model finetuned on a sentiment analysis task. 

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")    
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")    
```

With both the model and tokenizer initialized we are now able to get get some explanations on some input text. 

```python
from transformers_interpret import SequenceClassificationExplainer   
cls_explainer = SequenceClassificationExplainer("I love you, I like you",model, tokenizer)
attributions = cls_explainer()    
```

Returns the list of tuples below. 

```python
>>> attributions.word_attributions 
[('BOS_TOKEN', 0.0),
 ('I', 0.46820533977856904),
 ('love', 0.4606184697303162),
 ('you,', 0.5664126708457133),
 ('I', -0.017154242497229605),
 ('like', -0.05376360639469018),
 ('you', 0.10987772217503108),
 ('EOS_TOKEN', 0.4822169265102293)]
```

Positive numbers indicate a word contributes positively towards the predicted class, negative numbers indicate the opposite. Here we can see that __I love you__ gets the most attention. 

In case you want to know what the predicted class actually is:
```python
>>> cls_explainer.predicted_class_index    
array(1)
```
And if the model creator has provided label names for each class
```python
>>> cls_explainer.predicted_class_name
'POSITIVE'
```

### Visualizing attributions

Sometimes the numeric attributions can be difficult to read particularly in instances where there is a lot of text. To help with that there
is also an inbuilt visualize method that utilizes Captum's in built viz library to create a HTML file highlighting attributions. If you are in a notebook call the `visualize()` method will display the visualization in line, otherwise you can pass a filepath in as an argument a html file will be created so you can view the explanation in your browser. 

```python 
cls_explainer.visualize("distilbert_viz.html") 
```

<img src="images/distilbert_example.png" width="60%" height="60%" align="center" />