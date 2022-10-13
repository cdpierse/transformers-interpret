import warnings
from typing import Union

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


class QuestionAnsweringExplainer(BaseExplainer):
    """
    Explainer for explaining attributions for models of type `{MODEL_NAME}ForQuestionAnswering`
    from the Transformers package.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
    ):
        """
        Args:
            model (PreTrainedModel): Pretrained huggingface Question Answering model.
            tokenizer (PreTrainedTokenizer): Pretrained huggingface tokenizer
            attribution_type (str, optional): The attribution method to calculate on. Defaults to "lig".

        Raises:
            AttributionTypeNotSupportedError: [description]
        """
        super().__init__(model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type = attribution_type

        self.attributions: Union[None, LIGAttributions] = None
        self.start_attributions = None
        self.end_attributions = None
        self.input_ids: torch.Tensor = torch.Tensor()

        self.position = 0

        self.internal_batch_size = None
        self.n_steps = 50

    def encode(self, text: str) -> list:  # type: ignore
        "Encode 'text' using tokenizer, special tokens are not added"
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        "Decode 'input_ids' to string using tokenizer"
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    @property
    def word_attributions(self) -> dict:
        """
        Returns the word attributions (as `dict`) for both start and end positions of QA model.

        Raises error if attributions not calculated.

        """
        if self.start_attributions is not None and self.end_attributions is not None:
            return {
                "start": self.start_attributions.word_attributions,
                "end": self.end_attributions.word_attributions,
            }

        else:
            raise ValueError("Attributions have not yet been calculated. Please call the explainer on text first.")

    @property
    def start_pos(self):
        "Returns predicted start position for answer"
        if len(self.input_ids) > 0:
            preds = self._get_preds(
                self.input_ids,
                self.token_type_ids,
                self.position_ids,
                self.attention_mask,
            )

            preds = preds[0]
            return int(preds.argmax())
        else:
            raise InputIdsNotCalculatedError("input_ids have not been created yet.`")

    @property
    def end_pos(self):
        "Returns predicted end position for answer"
        if len(self.input_ids) > 0:
            preds = self._get_preds(
                self.input_ids,
                self.token_type_ids,
                self.position_ids,
                self.attention_mask,
            )

            preds = preds[1]
            return int(preds.argmax())
        else:
            raise InputIdsNotCalculatedError("input_ids have not been created yet.`")

    @property
    def predicted_answer(self):
        "Returns predicted answer span from provided `text`"
        if len(self.input_ids) > 0:
            preds = self._get_preds(
                self.input_ids,
                self.token_type_ids,
                self.position_ids,
                self.attention_mask,
            )

            start = preds[0].argmax()
            end = preds[1].argmax()
            return " ".join(self.decode(self.input_ids)[start : end + 1])
        else:
            raise InputIdsNotCalculatedError("input_ids have not been created yet.`")

    def visualize(self, html_filepath: str = None):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.

        Otherwise pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.
        """
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        predicted_answer = self.predicted_answer

        self.position = 0
        start_pred_probs = self._forward(self.input_ids, self.token_type_ids, self.position_ids)
        start_pos = self.start_pos
        start_pos_str = tokens[start_pos] + " (" + str(start_pos) + ")"
        start_score_viz = self.start_attributions.visualize_attributions(
            float(start_pred_probs),
            str(predicted_answer),
            start_pos_str,
            start_pos_str,
            tokens,
        )

        self.position = 1

        end_pred_probs = self._forward(self.input_ids, self.token_type_ids, self.position_ids)
        end_pos = self.end_pos
        end_pos_str = tokens[end_pos] + " (" + str(end_pos) + ")"
        end_score_viz = self.end_attributions.visualize_attributions(
            float(end_pred_probs),
            str(predicted_answer),
            end_pos_str,
            end_pos_str,
            tokens,
        )

        html = viz.visualize_text([start_score_viz, end_score_viz])

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)
        return html

    def _make_input_reference_pair(self, question: str, text: str):  # type: ignore
        question_ids = self.encode(question)
        text_ids = self.encode(text)

        input_ids = [self.cls_token_id] + question_ids + [self.sep_token_id] + text_ids + [self.sep_token_id]

        ref_input_ids = (
            [self.cls_token_id]
            + [self.ref_token_id] * len(question_ids)
            + [self.sep_token_id]
            + [self.ref_token_id] * len(text_ids)
            + [self.sep_token_id]
        )

        return (
            torch.tensor([input_ids], device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            len(question_ids),
        )

    def _get_preds(
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        if self.accepts_position_ids and self.accepts_token_type_ids:
            preds = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            return preds

        elif self.accepts_position_ids:
            preds = self.model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            return preds
        elif self.accepts_token_type_ids:
            preds = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

            return preds
        else:
            preds = self.model(
                input_ids,
                attention_mask=attention_mask,
            )

            return preds

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):

        preds = self._get_preds(input_ids, token_type_ids, position_ids, attention_mask)

        preds = preds[self.position]

        return preds.max(1).values

    def _run(self, question: str, text: str, embedding_type: int) -> dict:
        if embedding_type == 0:
            embeddings = self.word_embeddings
        try:
            if embedding_type == 1:
                if self.accepts_position_ids and self.position_embeddings is not None:
                    embeddings = self.position_embeddings
                else:
                    warnings.warn(
                        "This model doesn't support position embeddings for attributions. Defaulting to word embeddings"
                    )
                    embeddings = self.word_embeddings
            elif embedding_type == 2:
                embeddings = self.model_embeddings

            else:
                embeddings = self.word_embeddings
        except Exception:
            warnings.warn(
                "This model doesn't support the embedding type you selected for attributions. Defaulting to word embeddings"
            )
            embeddings = self.word_embeddings

        self.question = question
        self.text = text

        self._calculate_attributions(embeddings)
        return self.word_attributions

    def _calculate_attributions(self, embeddings: Embedding):  # type: ignore

        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.question, self.text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        (
            self.token_type_ids,
            self.ref_token_type_ids,
        ) = self._make_input_reference_token_type_pair(self.input_ids, self.sep_idx)

        self.attention_mask = self._make_attention_mask(self.input_ids)

        reference_tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        self.position = 0
        start_lig = LIGAttributions(
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
        start_lig.summarize()
        self.start_attributions = start_lig

        self.position = 1
        end_lig = LIGAttributions(
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
        end_lig.summarize()
        self.end_attributions = end_lig
        self.attributions = [self.start_attributions, self.end_attributions]

    def __call__(
        self,
        question: str,
        text: str,
        embedding_type: int = 2,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> dict:
        """
        Calculates start and end position word attributions for `question` and `text` using the model
        and tokenizer given in the constructor.

        This explainer also allows for attributions with respect to a particlar embedding type.
        This can be selected by passing a `embedding_type`. The default value is `2` which
        attempts to calculate for all embeddings. If `0` is passed then attributions are w.r.t word_embeddings,
        if `1` is passed then attributions are w.r.t position_embeddings.


        Args:
            question (str): The question text
            text (str): The text or context from which the model finds an answers
            embedding_type (int, optional): The embedding type word(0), position(1), all(2) to calculate attributions for.
                Defaults to 2.
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. If internal_batch_size is None, then all evaluations are
                processed in one batch.
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.

        Returns:
            dict: Dict for start and end position word attributions.
        """

        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size
        return self._run(question, text, embedding_type)
