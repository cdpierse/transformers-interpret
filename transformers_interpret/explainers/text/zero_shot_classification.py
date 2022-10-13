from typing import List, Tuple

import torch
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import LIGAttributions
from transformers_interpret.errors import AttributionTypeNotSupportedError

from .question_answering import QuestionAnsweringExplainer
from .sequence_classification import SequenceClassificationExplainer

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class ZeroShotClassificationExplainer(SequenceClassificationExplainer, QuestionAnsweringExplainer):
    """
    Explainer for explaining attributions for models that can perform
    zero-shot classification, specifically models trained on nli downstream tasks.

    This explainer uses the same "trick" as Huggingface to achieve attributions on
    arbitrary labels provided at inference time.

    Model's provided to this explainer must be nli sequence classification models
    and must have the label "entailment" or "ENTAILMENT" in
    `model.config.label2id.keys()` in order for it to work correctly.

    This explainer works by forcing the model to explain it's output with respect to
    the entailment class. For each label passed at inference the explainer forms a hypothesis with each
    and calculates attributions for each hypothesis label. The label with the highest predicted probability
    can be accessed via the attribute `predicted_label`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
    ):
        """

        Args:
            model (PreTrainedModel):Pretrained huggingface Sequence Classification model. Must be a NLI model.
            tokenizer (PreTrainedTokenizer): Pretrained huggingface tokenizer
            attribution_type (str, optional): The attribution method to calculate on. Defaults to "lig".

        Raises:
            AttributionTypeNotSupportedError: [description]
            ValueError: [description]
        """
        super().__init__(model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.label_exists, self.entailment_key = self._entailment_label_exists()
        if not self.label_exists:
            raise ValueError('Expected label "entailment" in `model.label2id` ')

        self.entailment_idx = self.label2id[self.entailment_key]
        self.include_hypothesis = False
        self.attributions = []

        self.internal_batch_size = None
        self.n_steps = 50

    @property
    def word_attributions(self) -> dict:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."
        if self.attributions != []:
            if self.include_hypothesis:
                return dict(
                    zip(
                        self.labels,
                        [attr.word_attributions for attr in self.attributions],
                    )
                )
            else:
                spliced_wa = [attr.word_attributions[: self.sep_idx] for attr in self.attributions]
                return dict(zip(self.labels, spliced_wa))
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

        if not self.include_hypothesis:
            tokens = tokens[: self.sep_idx]

        score_viz = [
            self.attributions[i].visualize_attributions(  # type: ignore
                self.pred_probs[i],
                self.labels[i],
                self.labels[i],
                self.labels[i],
                tokens,
            )
            for i in range(len(self.attributions))
        ]
        html = viz.visualize_text(score_viz)

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)
        return html

    def _entailment_label_exists(self) -> bool:
        if "entailment" in self.label2id.keys():
            return True, "entailment"
        elif "ENTAILMENT" in self.label2id.keys():
            return True, "ENTAILMENT"

        return False, None

    def _get_top_predicted_label_idx(self, text, hypothesis_labels: List[str]) -> int:

        entailment_outputs = []
        for label in hypothesis_labels:
            input_ids, _, sep_idx = self._make_input_reference_pair(text, label)
            position_ids, _ = self._make_input_reference_position_id_pair(input_ids)
            token_type_ids, _ = self._make_input_reference_token_type_pair(input_ids, sep_idx)
            attention_mask = self._make_attention_mask(input_ids)
            preds = self._get_preds(input_ids, token_type_ids, position_ids, attention_mask)
            entailment_outputs.append(float(torch.sigmoid(preds[0])[0][self.entailment_idx]))

        normed_entailment_outputs = [float(i) / sum(entailment_outputs) for i in entailment_outputs]

        self.pred_probs = normed_entailment_outputs

        return entailment_outputs.index(max(entailment_outputs))

    def _make_input_reference_pair(
        self,
        text: str,
        hypothesis_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        hyp_ids = self.encode(hypothesis_text)
        text_ids = self.encode(text)

        input_ids = [self.cls_token_id] + text_ids + [self.sep_token_id] + hyp_ids + [self.sep_token_id]

        ref_input_ids = (
            [self.cls_token_id]
            + [self.ref_token_id] * len(text_ids)
            + [self.sep_token_id]
            + [self.ref_token_id] * len(hyp_ids)
            + [self.sep_token_id]
        )

        return (
            torch.tensor([input_ids], device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            len(text_ids),
        )

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):

        preds = self._get_preds(input_ids, token_type_ids, position_ids, attention_mask)
        preds = preds[0]

        return torch.softmax(preds, dim=1)[:, self.selected_index]

    def _calculate_attributions(self, embeddings: Embedding, class_name: str, index: int = None):  # type: ignore
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.text, self.hypothesis_text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        (
            self.token_type_ids,
            self.ref_token_type_ids,
        ) = self._make_input_reference_token_type_pair(self.input_ids, self.sep_idx)

        self.attention_mask = self._make_attention_mask(self.input_ids)

        self.selected_index = int(self.label2id[class_name])

        reference_tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        lig = LIGAttributions(
            self._forward,
            embeddings,
            reference_tokens,
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
            self.attention_mask,
            position_ids=self.position_ids,
            ref_position_ids=self.ref_position_ids,
            token_type_ids=self.token_type_ids,
            ref_token_type_ids=self.ref_token_type_ids,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )
        if self.include_hypothesis:
            lig.summarize()
        else:
            lig.summarize(self.sep_idx)
        self.attributions.append(lig)

    def __call__(
        self,
        text: str,
        labels: List[str],
        embedding_type: int = 0,
        hypothesis_template="this text is about {} .",
        include_hypothesis: bool = False,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> dict:
        """
        Calculates attribution for `text` using the model and
        tokenizer given in the constructor. Since `self.model` is
        a NLI type model each label in `labels` is formatted to the
        `hypothesis_template`. By default attributions are provided for all
        labels. The top predicted label can be found in the `predicted_label`
        attribute.

        Attribution is forced to be on the axis of whatever index
        the entailment class resolves to. e.g. {"entailment": 0, "neutral": 1, "contradiction": 2 }
        in the above case attributions would be for the label at index 0.

        This explainer also allows for attributions with respect to a particlar embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.

        The default `hypothesis_template` can also be overridden by providing a formattable
        string which accepts exactly one formattable value for the label.

        If `include_hypothesis` is set to `True` then the word attributions and visualization
        of the attributions will also included the hypothesis text which gives a complete indication
        of what the model sees at inference.

        Args:
            text (str): Text to provide attributions for.
            labels (List[str]): The labels to classify the text to. If only one label is provided in the list then
                attributions are guaranteed to be for that label.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for.
                Defaults to 0.
            hypothesis_template (str, optional): Hypothesis presetned to NLI model given text.
                Defaults to "this text is about {} .".
            include_hypothesis (bool, optional): Alternative option to include hypothesis text in attributions
                and visualization. Defaults to False.
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. If internal_batch_size is None, then all evaluations are
                processed in one batch.
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.
        Returns:
            list: List of tuples containing words and their associated attribution scores.
        """

        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size
        self.attributions = []
        self.pred_probs = []
        self.include_hypothesis = include_hypothesis
        self.labels = labels
        self.hypothesis_labels = [hypothesis_template.format(label) for label in labels]

        predicted_text_idx = self._get_top_predicted_label_idx(text, self.hypothesis_labels)

        for i, _ in enumerate(self.labels):
            self.hypothesis_text = self.hypothesis_labels[i]
            self.predicted_label = labels[i] + " (" + self.entailment_key.lower() + ")"
            super().__call__(
                text,
                class_name=self.entailment_key,
                embedding_type=embedding_type,
            )
        self.predicted_label = self.labels[predicted_text_idx]
        return self.word_attributions
