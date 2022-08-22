import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer, LIGAttributions
from transformers_interpret.errors import AttributionTypeNotSupportedError, InputIdsNotCalculatedError

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class SequenceClassificationExplainer(BaseExplainer):
    """
    Explainer for explaining attributions for models of type
    `{MODEL_NAME}ForSequenceClassification` from the Transformers package.

    Calculates attribution for `text` using the given model
    and tokenizer.

    Attributions can be forced along the axis of a particular output index or class name.
    To do this provide either a valid `index` for the class label's output or if the outputs
    have provided labels you can pass a `class_name`.

    This explainer also allows for attributions with respect to a particlar embedding type.
    This can be selected by passing a `embedding_type`. The default value is `0` which
    is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
    If a model does not take position ids in its forward method (distilbert) a warning will
    occur and the default word_embeddings will be chosen instead.

    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
        custom_labels: Optional[List[str]] = None,
    ):
        """
        Args:
            model (PreTrainedModel): Pretrained huggingface Sequence Classification model.
            tokenizer (PreTrainedTokenizer): Pretrained huggingface tokenizer
            attribution_type (str, optional): The attribution method to calculate on. Defaults to "lig".
            custom_labels (List[str], optional): Applies custom labels to label2id and id2label configs.
                                                 Labels must be same length as the base model configs' labels.
                                                 Labels and ids are applied index-wise. Defaults to None.

        Raises:
            AttributionTypeNotSupportedError:
        """
        super().__init__(model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type = attribution_type

        if custom_labels is not None:
            if len(custom_labels) != len(model.config.label2id):
                raise ValueError(
                    f"""`custom_labels` size '{len(custom_labels)}' should match pretrained model's label2id size
                    '{len(model.config.label2id)}'"""
                )

            self.id2label, self.label2id = self._get_id2label_and_label2id_dict(custom_labels)
        else:
            self.label2id = model.config.label2id
            self.id2label = model.config.id2label

        self.attributions: Union[None, LIGAttributions] = None
        self.input_ids: torch.Tensor = torch.Tensor()

        self._single_node_output = False

        self.internal_batch_size = None
        self.n_steps = 50

    @staticmethod
    def _get_id2label_and_label2id_dict(
        labels: List[str],
    ) -> Tuple[Dict[int, str], Dict[str, int]]:
        id2label: Dict[int, str] = dict()
        label2id: Dict[str, int] = dict()
        for idx, label in enumerate(labels):
            id2label[idx] = label
            label2id[label] = idx

        return id2label, label2id

    def encode(self, text: str = None) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        "Decode 'input_ids' to string using tokenizer"
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    @property
    def predicted_class_index(self) -> int:
        "Returns predicted class index (int) for model with last calculated `input_ids`"
        if len(self.input_ids) > 0:
            # we call this before _forward() so it has to be calculated twice
            preds = self.model(self.input_ids)[0]
            self.pred_class = torch.argmax(torch.softmax(preds, dim=0)[0])
            return torch.argmax(torch.softmax(preds, dim=1)[0]).cpu().detach().numpy()

        else:
            raise InputIdsNotCalculatedError("input_ids have not been created yet.`")

    @property
    def predicted_class_name(self):
        "Returns predicted class name (str) for model with last calculated `input_ids`"
        try:
            index = self.predicted_class_index
            return self.id2label[int(index)]
        except Exception:
            return self.predicted_class_index

    @property
    def word_attributions(self) -> list:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."
        if self.attributions is not None:
            return self.attributions.word_attributions
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
        attr_class = self.id2label[self.selected_index]

        if self._single_node_output:
            if true_class is None:
                true_class = round(float(self.pred_probs))
            predicted_class = round(float(self.pred_probs))
            attr_class = round(float(self.pred_probs))

        else:
            if true_class is None:
                true_class = self.selected_index
            predicted_class = self.predicted_class_name

        score_viz = self.attributions.visualize_attributions(  # type: ignore
            self.pred_probs,
            predicted_class,
            true_class,
            attr_class,
            tokens,
        )
        html = viz.visualize_text([score_viz])

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)
        return html

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):

        preds = self._get_preds(input_ids, token_type_ids, position_ids, attention_mask)
        preds = preds[0]

        # if it is a single output node
        if len(preds[0]) == 1:
            self._single_node_output = True
            self.pred_probs = torch.sigmoid(preds)[0][0]
            return torch.sigmoid(preds)[:, :]

        self.pred_probs = torch.softmax(preds, dim=1)[0][self.selected_index]
        return torch.softmax(preds, dim=1)[:, self.selected_index]

    def _calculate_attributions(self, embeddings: Embedding, index: int = None, class_name: str = None):  # type: ignore
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        (
            self.token_type_ids,
            self.ref_token_type_ids,
        ) = self._make_input_reference_token_type_pair(self.input_ids, self.sep_idx)

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

        reference_tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        lig = LIGAttributions(
            custom_forward=self._forward,
            embeddings=embeddings,
            tokens=reference_tokens,
            input_ids=self.input_ids,
            ref_input_ids=self.ref_input_ids,
            sep_id=self.sep_idx,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            ref_position_ids=self.ref_position_ids,
            token_type_ids=self.token_type_ids,
            ref_token_type_ids=self.ref_token_type_ids,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )

        lig.summarize()
        self.attributions = lig

    def _run(
        self,
        text: str,
        index: int = None,
        class_name: str = None,
        embedding_type: int = None,
    ) -> list:  # type: ignore
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

        self._calculate_attributions(embeddings=embeddings, index=index, class_name=class_name)
        return self.word_attributions  # type: ignore

    def __call__(
        self,
        text: str,
        index: int = None,
        class_name: str = None,
        embedding_type: int = 0,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> list:
        """
        Calculates attribution for `text` using the model
        and tokenizer given in the constructor.

        Attributions can be forced along the axis of a particular output index or class name.
        To do this provide either a valid `index` for the class label's output or if the outputs
        have provided labels you can pass a `class_name`.

        This explainer also allows for attributions with respect to a particular embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.

        Args:
            text (str): Text to provide attributions for.
            index (int, optional): Optional output index to provide attributions for. Defaults to None.
            class_name (str, optional): Optional output class name to provide attributions for. Defaults to None.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.
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
        return self._run(text, index, class_name, embedding_type=embedding_type)

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += ")"

        return s


class PairwiseSequenceClassificationExplainer(SequenceClassificationExplainer):
    def _make_input_reference_pair(
        self, text1: Union[List, str], text2: Union[List, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:

        t1_ids = self.tokenizer.encode(text1, add_special_tokens=False)
        t2_ids = self.tokenizer.encode(text2, add_special_tokens=False)
        input_ids = self.tokenizer.encode([text1, text2], add_special_tokens=True)
        if self.model.config.model_type == "roberta":
            ref_input_ids = (
                [self.cls_token_id]
                + [self.ref_token_id] * len(t1_ids)
                + [self.sep_token_id]
                + [self.sep_token_id]
                + [self.ref_token_id] * len(t2_ids)
                + [self.sep_token_id]
            )

        else:

            ref_input_ids = (
                [self.cls_token_id]
                + [self.ref_token_id] * len(t1_ids)
                + [self.sep_token_id]
                + [self.ref_token_id] * len(t2_ids)
                + [self.sep_token_id]
            )

        return (
            torch.tensor([input_ids], device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            len(t1_ids) + 1,  # +1 for CLS token
        )

    def _calculate_attributions(
        self,
        embeddings: Embedding,
        index: int = None,
        class_name: str = None,
        flip_sign: bool = False,
    ):  # type: ignore
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.text1, self.text2)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        (
            self.token_type_ids,
            self.ref_token_type_ids,
        ) = self._make_input_reference_token_type_pair(self.input_ids, self.sep_idx)

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

        reference_tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]
        lig = LIGAttributions(
            custom_forward=self._forward,
            embeddings=embeddings,
            tokens=reference_tokens,
            input_ids=self.input_ids,
            ref_input_ids=self.ref_input_ids,
            sep_id=self.sep_idx,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            ref_position_ids=self.ref_position_ids,
            token_type_ids=self.token_type_ids,
            ref_token_type_ids=self.ref_token_type_ids,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )
        if self._single_node_output:
            lig.summarize(flip_sign=flip_sign)
        else:
            lig.summarize()
        self.attributions = lig

    def _run(
        self,
        text1: str,
        text2: str,
        index: int = None,
        class_name: str = None,
        embedding_type: int = None,
        flip_sign: bool = False,
    ) -> list:  # type: ignore
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

        self.text1 = text1
        self.text2 = text2

        self._calculate_attributions(
            embeddings=embeddings,
            index=index,
            class_name=class_name,
            flip_sign=flip_sign,
        )
        return self.word_attributions  # type: ignore

    def __call__(
        self,
        text1: str,
        text2: str,
        index: int = None,
        class_name: str = None,
        embedding_type: int = 0,
        internal_batch_size: int = None,
        n_steps: int = None,
        flip_sign: bool = False,
    ):
        """
        Calculates pairwise attributions for two inputs `text1` and `text2` using the model
        and tokenizer given in the constructor. Pairwise attributions are useful for models where
        two distinct inputs separated by the model separator token are fed to the model, such as cross-encoder
        models for similarity classification.

        Attributions can be forced along the axis of a particular output index or class name if there is more than one.
        To do this provide either a valid `index` for the class label's output or if the outputs
        have provided labels you can pass a `class_name`.

        This explainer also allows for attributions with respect to a particular embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.

        Additionally, this explainer allows for attributions signs to be flipped in cases where the model
        only outputs a single node. By default for models that output a single node the attributions are
        with respect to the inputs pushing the scores closer to 1.0, however if you want to see the
        attributions with respect to scores closer to 0.0 you can pass `flip_sign=True`. For similarity
        based models this is useful, as the model might predict a score closer to 0.0 for the two inputs
        and in that case we would flip the attributions sign to explain why the two inputs are dissimilar.

        Args:
            text1 (str): First text input to provide pairwise attributions for.
            text2 (str): Second text to provide pairwise attributions for.
            index (int, optional): Optional output index to provide attributions for. Defaults to None.
            class_name (str, optional):  Optional output class name to provide attributions for. Defaults to None.
            embedding_type (int, optional):The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. If internal_batch_size is None, then all evaluations are
                processed in one batch.
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.
            flip_sign (bool, optional): Boolean flag determining whether to flip the sign of attributions. Defaults to False.

        Returns:
            _type_: _description_
        """
        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size
        return self._run(
            text1=text1,
            text2=text2,
            embedding_type=embedding_type,
            index=index,
            class_name=class_name,
            flip_sign=flip_sign,
        )
