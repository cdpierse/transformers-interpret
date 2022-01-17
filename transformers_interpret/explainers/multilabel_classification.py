import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer, LIGAttributions, SequenceClassificationExplainer
from transformers_interpret.errors import AttributionTypeNotSupportedError, InputIdsNotCalculatedError

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class MultiLabelClassificationExplainer(BaseExplainer):
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
        # TODO: Custom implementation for this explainer see ZeroShot
        pass

    def visualize(self, html_filepath: str = None, true_class: str = None):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.

        Otherwise pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.

        If the true class is known for the text that can be passed to `true_class`

        """
        # TODO: Custom implementation for this explainer see ZeroShot
        pass

    def _forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        # TODO: possibly needs a custom implementation.
        pass

    def _calculate_attributions(self, embeddings: Embedding, index: int = None, class_name: str = None):  # type: ignore
        # TODO: possibly needs a custom implementation.
        pass

    def _run(
        self,
        text: str,
        index: int = None,
        class_name: str = None,
        embedding_type: int = None,
    ) -> list:  # type: ignore

        # TODO: possibly needs a custom implementation.
        pass

    def __call__(
        self,
        text: str,
        embedding_type: int = 0,
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> list:
        # TODO: definitely needs a custom implementation.
        pass
