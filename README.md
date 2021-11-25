<p align="center">
    <a id="transformers-intepret" href="#transformers-intepret">
        <img src="https://github.com/cdpierse/transformers-interpret/blob/master/images/tight%401920x_transparent.png" alt="Transformers Intepret Title" title="Transformers Intepret Title" width="600"/>
    </a>
</p>

<p align="center"> Explainability for 🤗 Transformers models in 2 lines.</p>

<h1 align="center"></h1>

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
        <a href="https://pepy.tech/project/transformers-interpret">
        <img src="https://static.pepy.tech/personalized-badge/transformers-interpret?period=total&units=abbreviation&left_color=black&right_color=brightgreen&left_text=Downloads">
    </a>
</p>

Transformers Interpret is a model explainability tool designed to work exclusively with the 🤗 [transformers][transformers] package.

In line with the philosophy of the transformers package Tranformers Interpret allows any transformers model to be explained in just two lines. It even supports visualizations in both notebooks and as savable html files.

Check out the streamlit [demo app here](https://share.streamlit.io/cdpierse/transformers-interpret-streamlit/main/app.py)

#### Table of Contents

- [Install](#install)

- [Documentation](#documentation)
  - [Quick Start](#quick-start)
    - [Sequence Classification Explainer](#sequence-classification-explainer)
      - [Visualize Classification attributions](#visualize-classification-attributions)
      - [Explaining Attributions for Non Predicted Class](#explaining-attributions-for-non-predicted-class)
    - [Zero Shot Classification Explainer](#zero-shot-classification-explainer)
      - [Visualize Zero Shot Classification attributions](#visualize-zero-shot-classification-attributions)
    - [Question Answering Explainer (Experimental)](#question-answering-explainer-experimental)
      - [Visualize Question Answering attributions](#visualize-question-answering-attributions)
  - [Future Development](#future-development)
  - [Contributing](#contributing)
  - [Questions / Get In Touch](#questions--get-in-touch)
  - [Reading and Resources](#reading-and-resources)
  - [Miscellaneous](#miscellaneous)

<a name="install"/>

## Install

```posh
pip install transformers-interpret
```

Supported:

- Python >= 3.6
- Pytorch >= 1.5.0
- [transformers][transformers] >= v3.0.0
- captum >= 0.3.1

The package does not work with Python 2.7 or below.

# Documentation

## Quick Start


### Sequence Classification Explainer

<details><summary>Click to expand</summary>

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

#### Visualize Classification attributions

Sometimes the numeric attributions can be difficult to read particularly in instances where there is a lot of text. To help with that we also provide the `visualize()` method that utilizes Captum's in built viz library to create a HTML file highlighting the attributions.

If you are in a notebook, calls to the `visualize()` method will display the visualization in-line. Alternatively you can pass a filepath in as an argument and an HTML file will be created, allowing you to view the explanation HTML in your browser.

```python
cls_explainer.visualize("distilbert_viz.html")
```

<a href="https://github.com/cdpierse/transformers-interpret/blob/master/images/distilbert_example.png">
<img src="https://github.com/cdpierse/transformers-interpret/blob/master/images/distilbert_example.png" width="80%" height="80%" align="center"/>
</a>

#### Explaining Attributions for Non Predicted Class

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

<a href="https://github.com/cdpierse/transformers-interpret/blob/master/images/distilbert_example_negative.png">
<img src="https://github.com/cdpierse/transformers-interpret/blob/master/images/distilbert_example_negative.png" width="80%" height="80%" align="center" />
</a>

Getting attributions for different classes is particularly insightful for multiclass problems as it allows you to inspect model predictions for a number of different classes and sanity-check that the model is "looking" at the right things.

For a detailed explanation of this example please checkout this [multiclass classification notebook.](notebooks/multiclass_classification_example.ipynb)


</details>

### Zero Shot Classification Explainer

<details><summary>Click to expand</summary>



_Models using this explainer must be previously trained on NLI classification downstream tasks and have a label in the model's config called either "entailment" or "ENTAILMENT"._

This explainer allows for attributions to be calculated for zero shot classification like models. In order to achieve this we use the same methodology employed by Hugging face. For those not familiar method employed by Hugging Face to achieve zero shot classification the way this works is by exploiting the "entailment" label of NLI models. Here is a [link](https://arxiv.org/abs/1909.00161) to a paper explaining more about it. A list of NLI models guaranteed to be compatible with this explainer can be found on the [model hub](https://huggingface.co/models?filter=pytorch&pipeline_tag=zero-shot-classification).

Let's start by initializing a transformers' sequence classification model and tokenizer trained specifically on a NLI task, and passing it to the ZeroShotClassificationExplainer.

For this example we are using `facebook/bart-large-mnli` which is a checkpoint for a bart-large model trained on the
[MNLI dataset](https://huggingface.co/datasets/multi_nli). This model typically predicts whether a sentence pair are an entailment, neutral, or a contradiction, however for zero-shot we only look the entailment label.

Notice that we pass our own custom labels `["finance", "technology", "sports"]` to the class instance. Any number of labels can be passed including as little as one. Whichever label scores highest for entailment can be accessed via `predicted_label`, however the attributions themselves are calculated for every label. If you want to see the attributions for a particular label it is recommended just to pass in that one label and then the attributions will be guaranteed to be calculated w.r.t. that label.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import ZeroShotClassificationExplainer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")


zero_shot_explainer = ZeroShotClassificationExplainer(model, tokenizer)


word_attributions = zero_shot_explainer(
    "Today apple released the new Macbook showing off a range of new features found in the proprietary silicon chip computer. ",
    labels = ["finance", "technology", "sports"],
)

```

Which will return the following dict of attribution tuple lists for each label:

```python
>>> word_attributions
{'finance': [('<s>', 0.0),
  ('Today', 0.0),
  ('apple', -0.016100065046282107),
  ('released', 0.3348383988281792),
  ('the', -0.8932952916127369),
  ('new', 0.14207183688642497),
  ('Mac', 0.016309545780430777),
  ('book', -0.06956802041125129),
  ('showing', -0.12661404114316252),
  ('off', -0.11470154900720078),
  ('a', -0.03299250484912159),
  ('range', -0.002532332125100561),
  ('of', -0.022451943898971004),
  ('new', -0.01859870581213379),
  ('features', -0.020774327263810944),
  ('found', -0.007734346326330102),
  ('in', 0.005100588658589585),
  ('the', 0.04711084622588314),
  ('proprietary', 0.046352064964644286),
  ('silicon', -0.0033502000158946127),
  ('chip', -0.010419324929115785),
  ('computer', -0.11507972995022273),
  ('.', 0.12237840300907425)],
 'technology': [('<s>', 0.0),
  ('Today', 0.0),
  ('apple', 0.22505152647747717),
  ('released', -0.16164146624851905),
  ('the', 0.5026975657258089),
  ('new', 0.052589263167955536),
  ('Mac', 0.2528325960993759),
  ('book', -0.06445090203729663),
  ('showing', -0.21204922293777534),
  ('off', 0.06319714817612732),
  ('a', 0.032048012090796815),
  ('range', 0.08553079346908955),
  ('of', 0.1409201107994034),
  ('new', 0.0515261917112576),
  ('features', -0.09656406466213506),
  ('found', 0.02336613296843605),
  ('in', -0.0011649894272190678),
  ('the', 0.14229640664777807),
  ('proprietary', -0.23169065661847646),
  ('silicon', 0.5963924257008087),
  ('chip', -0.19908474233975806),
  ('computer', 0.030620295844734646),
  ('.', 0.1995076958535378)],
 'sports': [('<s>', 0.0),
  ('Today', 0.0),
  ('apple', 0.1776618164760026),
  ('released', 0.10067773539491479),
  ('the', 0.4813466937627506),
  ('new', -0.018555244191949295),
  ('Mac', 0.016338241133536224),
  ('book', 0.39311969562943677),
  ('showing', 0.03579210145504227),
  ('off', 0.0016710813632476176),
  ('a', 0.04367940034297261),
  ('range', 0.06076859006993011),
  ('of', 0.11039711284328052),
  ('new', 0.003932416031994724),
  ('features', -0.009660883377622588),
  ('found', -0.06507586539836184),
  ('in', 0.2957812911667922),
  ('the', 0.1584106228974514),
  ('proprietary', 0.0005789280604917397),
  ('silicon', -0.04693795680472678),
  ('chip', -0.1699508539245465),
  ('computer', -0.4290823663975582),
  ('.', 0.469314992542427)]}
```

We can find out which label was predicted with:

```python
>>> zero_shot_explainer.predicted_label
'technology'
```

#### Visualize Zero Shot Classification attributions

For the `ZeroShotClassificationExplainer` the visualize() method returns a table similar to the `SequenceClassificationExplainer` but with attributions for every label.

```python
zero_shot_explainer.visualize("zero_shot.html")
```

<a href="https://github.com/cdpierse/transformers-interpret/blob/master/images/zero_shot_example.png">
<img src="https://github.com/cdpierse/transformers-interpret/blob/master/images/zero_shot_example.png" width="100%" height="100%" align="center" />
</a>

</details>

### Question Answering Explainer (Experimental)
    
<details><summary>Click to expand</summary>

_This is currently an experimental explainer under active development and is not yet fully tested. The explainers' API is subject to change as are the attribution methods, if you find any bugs please let me know._

Let's start by initializing a transformers' Question Answering model and tokenizer, and running it through the `QuestionAnsweringExplainer`.

For this example we are using `bert-large-uncased-whole-word-masking-finetuned-squad`, a bert model finetuned on a SQuAD.

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers_interpret import QuestionAnsweringExplainer

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

qa_explainer = QuestionAnsweringExplainer(
    model,
    tokenizer,
)

context = """
In Artificial Intelligence and machine learning, Natural Language Processing relates to the usage of machines to process and understand human language.
Many researchers currently work in this space.
"""

word_attributions = qa_explainer(
    "What is natural language processing ?",
    context,
)
```

Which will return the following dict containing word attributions for both the predicted start and end positions for the answer.

```python
>>> word_attributions
{'start': [('[CLS]', 0.0),
  ('what', 0.9177170660377296),
  ('is', 0.13382234898765258),
  ('natural', 0.08061747350142005),
  ('language', 0.013138062762511409),
  ('processing', 0.11135923869816286),
  ('?', 0.00858057388924361),
  ('[SEP]', -0.09646373141894966),
  ('in', 0.01545633993975799),
  ('artificial', 0.0472082598707737),
  ('intelligence', 0.026687249355110867),
  ('and', 0.01675371260058537),
  ('machine', -0.08429502436554961),
  ('learning', 0.0044827685126163355),
  (',', -0.02401013152520878),
  ('natural', -0.0016756080249823537),
  ('language', 0.0026815068421401885),
  ('processing', 0.06773157580722854),
  ('relates', 0.03884601576992908),
  ('to', 0.009783797821526368),
  ('the', -0.026650922910540952),
  ('usage', -0.010675019721821147),
  ('of', 0.015346787885898537),
  ('machines', -0.08278008270160107),
  ('to', 0.12861387892768839),
  ('process', 0.19540146386642743),
  ('and', 0.009942879959615826),
  ('understand', 0.006836894853320319),
  ('human', 0.05020451122579102),
  ('language', -0.012980795199301),
  ('.', 0.00804358248127772),
  ('many', 0.02259009321498161),
  ('researchers', -0.02351650942555469),
  ('currently', 0.04484573078852946),
  ('work', 0.00990399948294476),
  ('in', 0.01806961211334615),
  ('this', 0.13075899776164499),
  ('space', 0.004298315347838973),
  ('.', -0.003767904539347979),
  ('[SEP]', -0.08891544093454595)],
 'end': [('[CLS]', 0.0),
  ('what', 0.8227231947501547),
  ('is', 0.0586864942952253),
  ('natural', 0.0938903563379123),
  ('language', 0.058596976016400674),
  ('processing', 0.1632374290269829),
  ('?', 0.09695686057123237),
  ('[SEP]', -0.11644447033554006),
  ('in', -0.03769172371919206),
  ('artificial', 0.06736158404049886),
  ('intelligence', 0.02496399001288386),
  ('and', -0.03526028847762427),
  ('machine', -0.20846431491771975),
  ('learning', 0.00904892847529654),
  (',', -0.02949905488474854),
  ('natural', 0.011024507784743872),
  ('language', 0.0870741751282507),
  ('processing', 0.11482449622317169),
  ('relates', 0.05008962090922852),
  ('to', 0.04079118393166258),
  ('the', -0.005069048880616451),
  ('usage', -0.011992752445836278),
  ('of', 0.01715183316135495),
  ('machines', -0.29823535624026265),
  ('to', -0.0043760160855057925),
  ('process', 0.10503217484645223),
  ('and', 0.06840313586976698),
  ('understand', 0.057184000619403944),
  ('human', 0.0976805947708315),
  ('language', 0.07031163646606695),
  ('.', 0.10494566513897102),
  ('many', 0.019227154676079487),
  ('researchers', -0.038173913797800885),
  ('currently', 0.03916641120002003),
  ('work', 0.03705371672439422),
  ('in', -0.0003155975107591203),
  ('this', 0.17254932354022232),
  ('space', 0.0014311439625599323),
  ('.', 0.060637932829867736),
  ('[SEP]', -0.09186286505530596)]}
```

We can get the text span for the predicted answer with:

```python
>>> qa_explainer.predicted_answer
'usage of machines to process and understand human language'
```

#### Visualize Question Answering attributions

For the `QuestionAnsweringExplainer` the visualize() method returns a table with two rows. The first row represents the attributions for the answers' start position and the second row represents the attributions for the answers' end position.

```python
qa_explainer.visualize("bert_qa_viz.html")
```

<a href="https://github.com/cdpierse/transformers-interpret/blob/master/images/bert_qa_explainer.png">
<img src="https://github.com/cdpierse/transformers-interpret/blob/master/images/bert_qa_explainer.png" width="120%" height="120%" align="center" />
</a>

</details>

<a name="future"/>

## Future Development

This package is still in its early days and there is much more planned. For a 1.0.0 release we're aiming to have:

- Clean and thorough documentation website (**In Progress**)
- ~~Support for Question Answering models~~
- Support for NER models
- Support for Multiple Choice models
- ~~Support for Zero Shot Classification models.~~
- ~~Ability to show attributions for multiple embedding type, rather than just the word embeddings.~~
- Additional attribution methods
- In depth examples
- ~~A nice logo~~ (thanks @Voyz)
- and more... feel free to submit your suggestions!

<a name="contributing"/>

## Contributing

If you would like to make a contribution please checkout our [contribution guidelines](https://github.com/cdpierse/transformers-interpret/blob/master/CONTRIBUTING.md)

<a name="contact"/>

## Questions / Get In Touch

The main contributor to this repository is [@cdpierse](https://github.com/cdpierse).

If you have any questions, suggestions, or would like to make a contribution (please do 😁), feel free to get in touch at charlespierse@gmail.com

I'd also highly suggest checking out [Captum](https://captum.ai/) if you find model explainability and interpretability interesting. They are doing amazing and important work. In fact, this package stands on the shoulders of the the incredible work being done by the teams at [Pytorch Captum](https://captum.ai/) and [Hugging Face](https://huggingface.co/) and would not exist if not for the amazing job they are both doing in the fields of NLP and model interpretability respectively.

## Reading and Resources

<a name="reading-resources"/>

**Captum**

All of the attributions within this package are calculated using PyTorch's explainability package [Captum](https://captum.ai/). See below for some useful links related to Captum.

- [Captum Algorithm Overview](https://captum.ai/docs/algorithms)
- [Bert QA Example](https://captum.ai/tutorials/Bert_SQUAD_Interpret) this an implementation acheived purely using Captum.
- [API Reference](https://captum.ai/api/)
- [Model Interpretability with Captum - Narine Kokhilkyan (Video)](https://www.youtube.com/watch?v=iVSIFm0UN9I)

**Attributions**

Integrated Gradients (IG) and a variation of it Layer Integrated Gradients (LIG) are the core attribution methods on which Transformers Interpret is currently built. Below are some useful resources including the original paper and some video links explaining the inner mechanics. If you are curious about what is going on inside of Transformers Interpret I highly recommend checking out at least one of these resources.

- [Axiomatic Attributions for Deep Networks](https://arxiv.org/abs/1703.01365) the original paper [2017] where Integrated Gradients was specified.
- [Fiddler AI YouTube video on IG](https://www.youtube.com/watch?v=9AaDc35JYiI)
- [Henry AI Labs YouTube Primer on IG](https://www.youtube.com/watch?v=MB8KYX5UzKw)
- [Explaining Explanations: Axiomatic Feature Interactions for Deep Networks](http://export.arxiv.org/abs/2002.04138) more recent paper [2020] extending the work of the original paper.

## Miscellaneous

**Captum Links**

Below are some links I used to help me get this package together using Captum. Thank you to @davidefiocco for your very insightful GIST.

- [Link to useful GIST on captum](https://gist.github.com/davidefiocco/3e1a0ed030792230a33c726c61f6b3a5)
- [Link to runnable colab of captum with BERT](https://colab.research.google.com/drive/1snFbxdVDtL3JEFW7GNfRs1PZKgNHfoNz)

[transformers]: https://huggingface.co/transformers/
