import warnings

import captum
import torch
from captum.attr import visualization as viz
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer, LIGAttributions
from transformers_interpret.errors import (AttributionTypeNotSupportedError,
                                           InputIdsNotCalculatedError)

SUPPORTED_ATTRIBUTION_TYPES: list = ["lig"]


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

        self.attributions = None
        self.input_ids = None

    def encode(self, text: str = None) -> list:
        if text is None:
            text = self.text
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids):
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    def run(self, text: str = None, index: int = None, class_name: str = None):
        if text is not None:
            self.text = text

        self._calculate_attributions(index=index, class_name=class_name)
        return self.attributions

    def _forward(self, input_ids):
        preds = self.model(input_ids)[0]
        self.pred_probs = torch.softmax(preds, dim=1)[0][1]
        return torch.softmax(preds, dim=1)[0][self.selected_index].unsqueeze(-1)

    @property
    def predicted_class_index(self):
        if self.input_ids is not None:
            preds = self.model(self.input_ids)[0]
            self.pred_class = torch.argmax(torch.softmax(preds, dim=0)[0])
            return torch.argmax(torch.softmax(preds, dim=1)[0]).detach().numpy()

        else:
            raise InputIdsNotCalculatedError(
                "input_ids have not been created yet. Please call `get_attributions()`"
            )

    @property
    def predicted_class_name(self):
        try:
            index = self.predicted_class_index
            return self.id2label[int(index)]
        except ValueError:
            return self.predicted_class_index

    def visualize(self, html_filepath: str = None, true_class: str = None):
        tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids[0])
        attr_class = self.id2label[int(self.selected_index)]
        if true_class is None:
            true_class = self.predicted_class_name
        score_viz = self.attributions.visualize_attributions(
            self.pred_probs, self.predicted_class_name, true_class, attr_class, self.text, tokens
        )
        html = viz.visualize_text([score_viz])

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)


    def _calculate_attributions(self, index: int = None, class_name: str = None):
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.text)

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
            embeddings = getattr(self.model, self.model_type).embeddings
            reference_text = "BOS_TOKEN " + self.text + " EOS_TOKEN"
            lig = LIGAttributions(
                self._forward,
                embeddings,
                reference_text,
                self.input_ids,
                self.ref_input_ids,
                self.sep_idx,
            )
            lig.summarize()
            self.attributions = lig
        else:
            pass

    def __call__(self, text: str = None, index: int = None, class_name: str = None):
        return self.run(text, index, class_name)

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f'\n\ttext="{str(self.text[:10])}...",'
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += ")"

        return s
