import pandas
from transformers import PreTrainedModel, PreTrainedTokenizer
from abc import abstractmethod


class BaseExplainer:
    ALLOWED_MODELS = [
        "model1"
    ]

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 attribution_type: str = "shap"):
        self.model = model
        self.tokenizer = tokenizer

        # if self.model.__class__.__name__ not in self.ALLOWED_MODELS:
        #     raise NotImplementedError(
        #         "Model Interpretation is currently not supported for {}. Please select a model from {} for interpretation.".format(
        #             self.model.__class__.__name__, self.ALLOWED_MODELS
        #         )
        #     )

    @abstractmethod
    def get_model_attributions(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_layer_attributions(self):
        raise NotImplementedError

    @ staticmethod
    def encode_inputs(tokenizer: PreTrainedTokenizer, inputs: str):
        tokenizer.encode_plus(inputs,
                              pad_to_max_length=True)

    @ staticmethod
    def decode_inputs(tokenizer: PreTrainedTokenizer, token_ids: str):
        tokenizer.decode(token_ids)


class SequenceClassificationExplainer(BaseExplainer):
    pass


class QuestionAnsweringExplainer(BaseExplainer):
    pass


class NERExplainer(BaseExplainer):
    pass


TokenClassificationExplainer = NERExplainer


class LMExplainer(BaseExplainer):
    pass


class SummarizationExplainer(BaseExplainer):
    pass
