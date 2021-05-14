import warnings
from typing import List, Tuple, Union

import torch
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import LIGAttributions
from transformers_interpret.errors import AttributionTypeNotSupportedError
from transformers_interpret.explainers.question_answering import (
    QuestionAnsweringExplainer,
)

from .sequence_classification import SequenceClassificationExplainer

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class ZeroShotClassificationExplainer(
    SequenceClassificationExplainer, QuestionAnsweringExplainer
):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
    ):
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

    @property
    def word_attributions(self) -> list:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."
        if self.attributions is not None:
            return self.attributions.word_attributions[: self.sep_idx]
        else:
            raise ValueError(
                "Attributions have not yet been calculated. Please call the explainer on text first."
            )

    def visualize(self, html_filepath: str = None, true_class: str = None):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.

        Otherwise pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.

        If the true class is known for the text that can be passed to `true_class`

        """
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        attr_class = self.id2label[self.selected_index]

        score_viz = self.attributions.visualize_attributions(  # type: ignore
            self.pred_probs,
            self.predicted_label,
            self.entailment_key,
            attr_class,
            tokens[: self.sep_idx],
        )
        html = viz.visualize_text([score_viz])

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
            token_type_ids, _ = self._make_input_reference_token_type_pair(
                input_ids, sep_idx
            )
            attention_mask = self._make_attention_mask(input_ids)
            preds = self._get_preds(
                input_ids, token_type_ids, position_ids, attention_mask
            )
            entailment_outputs.append(
                float(torch.softmax(preds[0], dim=1)[0][self.entailment_idx])
            )

        return entailment_outputs.index(max(entailment_outputs))

    def _make_input_reference_pair(
        self,
        text: str,
        hypothesis_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        hyp_ids = self.encode(hypothesis_text)
        text_ids = self.encode(text)

        input_ids = (
            [self.cls_token_id]
            + text_ids
            + [self.sep_token_id]
            + hyp_ids
            + [self.sep_token_id]
        )

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

    def _calculate_attributions(  # type: ignore
        self, embeddings: Embedding, index: int = None, class_name: str = None
    ):
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.text, self.hypothesis_text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        self.attention_mask = self._make_attention_mask(self.input_ids)

        if index is not None:
            self.selected_index = index
        elif class_name is not None:
            if class_name in self.label2id.keys():
                self.selected_index = int(self.label2id[class_name])
            else:
                s = f"'{class_name}' is not found in self.label2id keys."
                s += "Defaulting to predicted index instead."
                warnings.warn(s)
                self.selected_index = int(self.predicted_class_index)
        else:
            self.selected_index = int(self.predicted_class_index)
        if self.attribution_type == "lig":
            reference_tokens = [
                token.replace("Ġ", "") for token in self.decode(self.input_ids)
            ]
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
            )
            lig.summarize()
            self.attributions = lig

    def __call__(
        self,
        text: str,
        labels: List[str],
        embedding_type: int = 0,
        hypothesis_template="this text is about {}",
    ) -> list:
        hypothesis_labels = [hypothesis_template.format(label) for label in labels]

        text_idx = self._get_top_predicted_label_idx(text, hypothesis_labels)
        self.hypothesis_text = hypothesis_labels[text_idx]
        self.predicted_label = labels[text_idx]

        return super().__call__(
            text,
            class_name=self.entailment_key,
            embedding_type=embedding_type,
        )
