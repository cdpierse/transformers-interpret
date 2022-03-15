import warnings
from typing import List, Dict, Optional, Union

import torch
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer
from transformers_interpret.attributions import LIGAttributions
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)


SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class TokenClassificationExplainer(BaseExplainer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type="lig",
    ):

        """
        Args:
            model (PreTrainedModel): Pretrained huggingface Sequence Classification model.
            tokenizer (PreTrainedTokenizer): Pretrained huggingface tokenizer
            attribution_type (str, optional): The attribution method to calculate on. Defaults to "lig".

        Raises:
            AttributionTypeNotSupportedError:
        """
        super().__init__(model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type: str = attribution_type

        self.label2id = model.config.label2id
        self.id2label = model.config.id2label

        self.ignored_indexes: Optional[List[int]] = None
        self.ignored_labels: Optional[List[str]] = None

        self.attributions: Union[None, Dict[int, LIGAttributions]] = None
        self.input_ids: torch.Tensor = torch.Tensor()

        self.internal_batch_size = None
        self.n_steps = 50

    def encode(self, text: str = None) -> list:
        "Encode the text using tokenizer"
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        "Decode 'input_ids' to string using tokenizer"
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    @property
    def predicted_class_indexes(self) -> List[int]:
        "Returns the predicted class indexes (int) for model with last calculated `input_ids`"
        if len(self.input_ids) > 0:

            preds = self.model(self.input_ids)
            preds = preds[0]
            self.pred_class = torch.softmax(preds, dim=2)[0]

            return torch.argmax(torch.softmax(preds, dim=2), dim=2)[0].cpu().detach().numpy()

        else:
            raise InputIdsNotCalculatedError("input_ids have not been created yet.`")

    @property
    def predicted_class_names(self):
        "Returns predicted class names (str) for model with last calculated `input_ids`"
        try:
            indexes = self.predicted_class_indexes
            return [self.id2label[int(index)] for index in indexes]
        except Exception:
            return self.predicted_class_indexes

    @property
    def word_attributions(self) -> Dict:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."

        if self.attributions is not None:
            word_attr = dict()
            tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
            labels = self.predicted_class_names

            for index, attr in self.attributions.items():
                try:
                    predicted_class = self.id2label[torch.argmax(self.pred_probs[index]).item()]
                except KeyError:
                    predicted_class = torch.argmax(self.pred_probs[index]).item()

                word_attr[tokens[index]] = {
                    "label": predicted_class,
                    "attribution_scores": attr.word_attributions,
                }

            return word_attr
        else:
            raise ValueError("Attributions have not yet been calculated. Please call the explainer on text first.")

    @property
    def _selected_indexes(self) -> List[int]:
        """Returns the indexes for which the attributions must be calculated considering the
        ignored indexes and the ignored labels, in that order of priority"""

        selected_indexes = set(range(self.input_ids.shape[1]))  # all indexes

        if self.ignored_indexes is not None:
            selected_indexes = selected_indexes.difference(set(self.ignored_indexes))

        if self.ignored_labels is not None:
            ignored_indexes_extra = []
            pred_labels = [self.id2label[id] for id in self.predicted_class_indexes]

            for index, label in enumerate(pred_labels):
                if label in self.ignored_labels:
                    ignored_indexes_extra.append(index)
            selected_indexes = selected_indexes.difference(ignored_indexes_extra)

        return sorted(list(selected_indexes))

    def visualize(self, html_filepath: str = None, true_classes: List[str] = None):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.

        Otherwise pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.

        If the true class is known for the text that can be passed to `true_class`

        """
        if true_classes is not None and len(true_classes) != self.input_ids.shape[1]:
            raise ValueError(f"""The length of `true_classes` must be equal to the number of tokens""")

        score_vizs = []
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]

        for index in self._selected_indexes:
            pred_prob = torch.max(self.pred_probs[index])
            predicted_class = self.id2label[torch.argmax(self.pred_probs[index]).item()]

            attr_class = tokens[index]
            if true_classes is None:
                true_class = predicted_class
            else:
                true_class = true_classes[index]

            score_vizs.append(
                self.attributions[index].visualize_attributions(
                    pred_prob,
                    predicted_class,
                    true_class,
                    attr_class,
                    tokens,
                )
            )

        html = viz.visualize_text(score_vizs)

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)
        return html

    def _forward(
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
        else:
            preds = self.model(input_ids, attention_mask)

        preds = preds.logits  # preds.shape = [N_BATCH, N_TOKENS, N_CLASSES]

        self.pred_probs = torch.softmax(preds, dim=2)[0]
        return torch.softmax(preds, dim=2)[:, self.index, :]

    def _calculate_attributions(
        self,
        embeddings: Embedding,
    ) -> None:
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

        pred_classes = self.predicted_class_indexes
        reference_tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]

        ligs = {}

        for index in self._selected_indexes:
            self.index = index
            lig = LIGAttributions(
                self._forward,
                embeddings,
                reference_tokens,
                self.input_ids,
                self.ref_input_ids,
                self.sep_idx,
                self.attention_mask,
                target=int(pred_classes[index]),
                position_ids=self.position_ids,
                ref_position_ids=self.ref_position_ids,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
            )
            lig.summarize()
            ligs[index] = lig

        self.attributions = ligs

    def _run(
        self,
        text: str,
        embedding_type: int = None,
    ) -> dict:
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
                        "This model doesn't support position embeddings for attributions. Defaulting to word embeddings"
                    )
                    embeddings = self.word_embeddings
            else:
                embeddings = self.word_embeddings

        self.text = self._clean_text(text)

        self._calculate_attributions(embeddings=embeddings)
        return self.word_attributions

    def __call__(
        self,
        text: str,
        embedding_type: int = 0,
        internal_batch_size: Optional[int] = None,
        n_steps: Optional[int] = None,
        ignored_indexes: Optional[List[int]] = None,
        ignored_labels: Optional[List[str]] = None,
    ) -> dict:
        """
        Args:
            text (str): Sentence whose NER predictions are to be explained.
            embedding_type (int, default = 0): Custom type of embedding.
            internal_batch_size (int, optional): Custom internal batch size for the attributions calculation.
            n_steps (int): Custom number of steps in the approximation used in the attributions calculation.
            ignored_indexes (List[int], optional): Indexes that are to be ignored by the explainer.
            ignored_labels (List[str], optional)): NER labels that are to be ignored by the explainer. The
                                                explainer will ignore those indexes whose predicted label is
                                                in `ignored_labels`.
        """

        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size

        self.ignored_indexes = ignored_indexes
        self.ignored_labels = ignored_labels

        return self._run(text, embedding_type=embedding_type)

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += ")"

        return s
