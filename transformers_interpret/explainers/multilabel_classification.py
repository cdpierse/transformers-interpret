from typing import Dict, List, Optional, Tuple, Union

from captum.attr import visualization as viz
from transformers import PreTrainedModel, PreTrainedTokenizer

from .sequence_classification import SequenceClassificationExplainer

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class MultiLabelClassificationExplainer(SequenceClassificationExplainer):
    """
    Explainer for Multi-Label Classification models.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type="lig",
        custom_labels: Optional[List[str]] = None,
    ):
        super().__init__(model, tokenizer, attribution_type, custom_labels)

    @property
    def word_attributions(self) -> dict:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."
        if self.attributions != []:

            return dict(
                zip(
                    self.labels,
                    [attr.word_attributions for attr in self.attributions],
                )
            )

        else:
            raise ValueError("Attributions have not yet been calculated. Please call the explainer on text first.")

    def visualize(self, html_filepath: str = None, true_class: str = None):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.

        Otherwise pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.

        If the true class is known for the text that can be passed to `true_class`

        """
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]

        score_viz = [
            self.attributions[i].visualize_attributions(  # type: ignore
                self.pred_probs[i],
                "",  # including a predicted class name does not make sense for this explainer
                "n/a" if not true_class else true_class,  # no true class name for this explainer by default
                self.labels[i],
                tokens,
            )
            for i in range(len(self.attributions))
        ]

        html = viz.visualize_text(score_viz)

        new_html_data = html._repr_html_().replace("Predicted Label", "Prediction Score")
        new_html_data = new_html_data.replace("True Label", "n/a")
        html.data = new_html_data

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)
        return html

    def __call__(
        self,
        text: str,
        embedding_type: int = 0,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> list:
        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size

        self.attributions = []
        self.pred_probs = []
        self.labels = list(self.label2id.keys())
        self.label_probs_dict = {}
        for i in range(self.model.config.num_labels):
            explainer = SequenceClassificationExplainer(
                self.model,
                self.tokenizer,
            )
            explainer(text, i, embedding_type)

            self.attributions.append(explainer.attributions)
            self.input_ids = explainer.input_ids
            self.pred_probs.append(explainer.pred_probs)
            self.label_probs_dict[self.id2label[i]] = explainer.pred_probs

        return self.word_attributions

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += f"\n\tcustom_labels={self.custom_labels},"
        s += ")"

        return s
