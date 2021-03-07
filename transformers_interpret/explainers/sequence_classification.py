import warnings
from enum import Enum
from typing import Union

import captum
import torch
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer, LIGAttributions
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class SequenceClassificationExplainer(BaseExplainer):
    """
    Explainer for explaining attributions for models of type
    `{MODEL_NAME}ForSequenceClassification` from the Transformers package.
    """

    def __init__(
        self,
        text: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
    ):
        super().__init__(text, model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type = attribution_type

        self.label2id = model.config.label2id
        self.id2label = model.config.id2label

        self.attributions: Union[None, LIGAttributions] = None
        self.input_ids: torch.Tensor = torch.Tensor()

        self._single_node_output = False

    def encode(self, text: str = None) -> list:
        if text is None:
            text = self.text
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):

        if self.accepts_position_ids:
            preds = self.model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            preds = preds[0]

        else:
            preds = self.model(input_ids, attention_mask)[0]

        # if it is a single output node
        if len(preds[0]) == 1:
            self._single_node_output = True
            self.pred_probs = torch.sigmoid(preds)[0][0]
            return torch.sigmoid(preds)[:, :]

        self.pred_probs = torch.softmax(preds, dim=1)[0][self.selected_index]
        return torch.softmax(preds, dim=1)[:, self.selected_index]

    @property
    def predicted_class_index(self):
        if len(self.input_ids) > 0:
            # we call this before _forward() so it has to be calculated twice
            preds = self.model(self.input_ids)[0]
            self.pred_class = torch.argmax(torch.softmax(preds, dim=0)[0])
            return torch.argmax(torch.softmax(preds, dim=1)[0]).cpu().detach().numpy()

        else:
            raise InputIdsNotCalculatedError(
                "input_ids have not been created yet. Please call `get_attributions()`"
            )

    @property
    def predicted_class_name(self):
        try:
            index = self.predicted_class_index
            return self.id2label[int(index)]
        except Exception:
            return self.predicted_class_index

    def visualize(self, html_filepath: str = None, true_class: str = None):
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        attr_class = self.id2label[int(self.selected_index)]

        if self._single_node_output:
            if true_class is None:
                true_class = round(float(self.pred_probs))
            predicted_class = round(float(self.pred_probs))
            attr_class = round(float(self.pred_probs))
        else:
            if true_class is None:
                true_class = str(self.selected_index)
            predicted_class = self.predicted_class_name

        score_viz = self.attributions.visualize_attributions(  # type: ignore
            self.pred_probs,
            predicted_class,
            true_class,
            attr_class,
            self.text,
            tokens,
        )
        html = viz.visualize_text([score_viz])

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)
        return html

    def _calculate_attributions(  # type: ignore
        self, embeddings: Embedding, index: int = None, class_name: str = None
    ):
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        self.attention_mask = self._make_attention_mask(self.input_ids)

        if index is not None:
            self.selected_index = index
        elif class_name is not None:
            if class_name in self.label2id.keys():
                self.selected_index = self.label2id[class_name]
            else:
                s = f"'{class_name}' is not found in self.label2id keys."
                s += "Defaulting to predicted index instead."
                warnings.warn(s)
                self.selected_index = self.predicted_class_index
        else:
            self.selected_index = self.predicted_class_index
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
        else:
            pass

    def _run(
        self,
        text: str = None,
        index: int = None,
        class_name: str = None,
        embedding_type: int = None,
    ) -> LIGAttributions:
        if embedding_type is None:
            embeddings = self.word_embeddings
        else:
            if embedding_type == 0:
                embeddings = self.word_embeddings
            elif embedding_type == 1:
                if self.accepts_position_ids and self.position_embeddings is not None:
                    embeddings = self.position_embeddings
                else:
                    warnings.warn(
                        f"This model doesn't support position embeddings for attributions. Defaulting to word embeddings"
                    )
                    embeddings = self.word_embeddings
            else:
                embeddings = self.word_embeddings

        if text is not None:
            self.text = text

        self._calculate_attributions(
            embeddings=embeddings, index=index, class_name=class_name
        )
        return self.attributions  # type: ignore

    def __call__(
        self,
        text: str = None,
        index: int = None,
        class_name: str = None,
        embedding_type: int = 0,
    ) -> LIGAttributions:
        """
        Calculates attribution for `text` or `self.text` using the given model
        and tokenizer. If `text` is not passed it is expected that text was passed
        in the constructor.

        Attributions can be forced along the axis of a particular output index or class name.
        To do this provide either a valid `index` for the class label's output or if the outputs
        have provided labels you can pass a `class_name`.

        This explainer also allows for attributions with respect to a particlar embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.

        Args:
            text (str, optional): Text to provide attributions for. Defaults to None.
            index (int, optional): Optional output index to provide attributions for. Defaults to None.
            class_name (str, optional): Optional output class name to provide attributions for. Defaults to None.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.

        Returns:
            LIGAttributions:
        """
        return self._run(text, index, class_name, embedding_type=embedding_type)

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f'\n\ttext="{str(self.text[:10])}...",'
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += ")"

        return s
