
<p align="center">
    <a id="transformers-intepret" href="#transformers-intepret">
        <img src="https://github.com/cdpierse/transformers-interpret/blob/master/images/tight%401920x_transparent.png" alt="Transformers Intepret Title" title="Transformers Intepret Title" width="600"/>
    </a>
</p>


<p align="center">
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/> 
    </a>
    <img src="./images/coverage.svg">
    <a href="https://github.com/cdpierse/transformers-interpret/releases">
        <img src="https://img.shields.io/pypi/v/transformers_interpret?label=version"/> 
    </a>
    <a href="https://app.circleci.com/pipelines/github/cdpierse/transformers-interpret">
        <img src="https://circleci.com/gh/cdpierse/transformers-interpret.svg?style=shield&circle-token=de18bfcb7476a5a47b8ad39b8cb1d61f5ae9ed52">
    </a>
</p>


Transformers Interpret is a model explainability tool designed to work exclusively with the 🤗  [transformers][transformers] package.

In line with the philosophy of the transformers package Tranformers Interpret allows any transformers model to be explained in just two lines. It even supports visualizations in both notebooks and as savable html files.


## Install

```posh
pip install transformers-interpret
```

Supported:

* Python >= 3.6 
* Pytorch >= 1.5.0 
* [transformers][transformers] >= v3.0.0 
* captum >= 0.3.1 


The package does not work with Python 2.7 or below.



# Documentation

## Quick Start

Let's start by initializing a transformers' model and tokenizer, and running it through the `SequenceClassificationExplainer`.

For this example we are using `distilbert-base-uncased-finetuned-sst-2-english`, a distilbert model finetuned on a sentiment analysis task.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# With both the model and tokenizer initialized we are now able to get explanations on an example text.

from transformers_interpret import SequenceClassificationExplainer
cls_explainer = SequenceClassificationExplainer(
    model, 
    tokenizer)
word_attributions = cls_explainer("I love you, I like you")
```

Which will return the following list of tuples:

```python
>>> word_attributions
[('[CLS]', 0.0),
 ('i', 0.2778544699186709),
 ('love', 0.7792370723380415),
 ('you', 0.38560088858031094),
 (',', -0.01769750505546915),
 ('i', 0.12071898121557832),
 ('like', 0.19091105304734457),
 ('you', 0.33994871536713467),
 ('[SEP]', 0.0)]
```

Positive attribution numbers indicate a word contributes positively towards the predicted class, while negative numbers indicate a word contributes negatively towards the predicted class. Here we can see that **I love you** gets the most attention.

You can use `predicted_class_index` in case you'd want to know what the predicted class actually is:

```python
>>> cls_explainer.predicted_class_index
array(1) 
```

And if the model has label names for each class, we can see these too using `predicted_class_name`:

```python
>>> cls_explainer.predicted_class_name
'POSITIVE'
```

### Visualizing attributions

Sometimes the numeric attributions can be difficult to read particularly in instances where there is a lot of text. To help with that we also provide the `visualize()` method that utilizes Captum's in built viz library to create a HTML file highlighting the attributions.

If you are in a notebook, calls to the `visualize()` method will display the visualization in-line. Alternatively you can pass a filepath in as an argument and an HTML file will be created, allowing you to view the explanation HTML in your browser.

```python
cls_explainer.visualize("distilbert_viz.html")
```

<img src="https://github.com/cdpierse/transformers-interpret/blob/master/images/distilbert_example.png" width="80%" height="80%" align="center"/>


### Explaining Attributions for Non Predicted Class

Attribution explanations are not limited to the predicted class. Let's test a more complex sentence that contains mixed sentiments.

In the example below we pass `class_name="NEGATIVE"` as an argument indicating we would like the attributions to be explained for the **NEGATIVE** class regardless of what the actual prediction is. Effectively because this is a binary classifier we are getting the inverse attributions.

```python
cls_explainer = SequenceClassificationExplainer(model, tokenizer)
attributions = cls_explainer("I love you, I like you, I also kinda dislike you", class_name="NEGATIVE")
```

In this case, `predicted_class_name` still returns a prediction of the **POSITIVE** class, because the model has generated the same prediction but nonetheless we are interested in looking at the attributions for the negative class regardless of the predicted result. 

```python
>>> cls_explainer.predicted_class_name
'POSITIVE'
```

But when we visualize the attributions we can see that the words "**...kinda dislike**" are contributing to a prediction of the "NEGATIVE"
class.

```python
cls_explainer.visualize("distilbert_negative_attr.html")
```

<img src="https://github.com/cdpierse/transformers-interpret/blob/master/images/distilbert_example_negative.png" width="80%" height="80%" align="center" />

Getting attributions for different classes is particularly insightful for multiclass problems as it allows you to inspect model predictions for a number of different classes and sanity-check that the model is "looking" at the right things.

For a detailed explanation of this example please checkout this [multiclass classification notebook.](notebooks/multiclass_classification_example.ipynb)

## Future Development

This package is still in its early days and there is much more planned. For a 1.0.0 release we're aiming to have:

- Clean and thorough documentation
- Support for Question Answering models
- Support for NER models
- Support for Multiple Choice models
- Ability to show attributions for multiple embedding type, rather than just the word embeddings. 
- Additional attribution methods
- In depth examples
- ~~A nice logo~~ (thanks @Voyz)
- and more... feel free to submit your suggestions!

## Questions / Get In Touch

The main contributor to this repository is [@cdpierse](https://github.com/cdpierse).

If you have any questions, suggestions, or would like to make a contribution (please do 😁), feel free to get in touch at charlespierse@gmail.com

I'd also highly suggest checking out [Captum](https://captum.ai/) if you find model explainability and interpretability interesting. They are doing amazing and important work. In fact, this package stands on the shoulders of the the incredible work being done by the teams at [Pytorch Captum](https://captum.ai/) and [Hugging Face](https://huggingface.co/) and would not exist if not for the amazing job they are both doing in the fields of NLP and model interpretability respectively.

## Miscellaneous

**Captum Links**

Below are some links I used to help me get this package together using Captum. Thank you to @davidefiocco for your very insightful GIST.

- [Link to useful GIST on captum](https://gist.github.com/davidefiocco/3e1a0ed030792230a33c726c61f6b3a5)
- [Link to runnable colab of captum with BERT](https://colab.research.google.com/drive/1snFbxdVDtL3JEFW7GNfRs1PZKgNHfoNz)

[transformers]: https://huggingface.co/transformers/
