import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from transformers_interpret import BaseExplainer, LIGAttributions

SUPPORTED_ATTRIBUTION_TYPES: list = ["lig"]


class SequenceClassificationExplainer(BaseExplainer):
    """
    Explainer for explaining attributions for models of type `{MODEL_NAME}ForSequenceClassification`
    from the Transformers package.
    """

    def __init__(
        self,
        text: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
    ):
        super().__init__(text, model, tokenizer)
        self.attribution_type = attribution_type

        self.label2id = model.config.label2id
        self.id2label = model.config.id2label

    def encode(self, text: str = None):
        if text is None:
            text = self.text
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self):
        indices = input_ids[0].detach().tolist()
        return self.tokenizer.convert_ids_to_tokens(input_ids)

    def get_attributions(self, text: str = None):
        if text is not None:
            self.text = text
        self._calculate_attributions()
        return self.attributions

    def _forward(self, input_ids):
        preds = self.model(input_ids)[0]
        return torch.softmax(preds, dim=1)[0][1].unsqueeze(-1)

    def _calculate_attributions(self):

        self.input_ids, self.ref_input_ids, self.sep_idx = self._make_input_reference_pair(
            self.text)

        if self.attribution_type == "lig":
            embeddings = getattr(self.model, self.model_type).embeddings
            self.attributions = LIGAttributions(
                self._forward,
                embeddings,
                self.input_ids,
                self.ref_input_ids,
                self.sep_idx
            )

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f'\n\ttext="{str(self.text[:10])}...",'
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += ")"

        return s
