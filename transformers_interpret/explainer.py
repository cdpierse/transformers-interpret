import pandas
from transformers import PreTrainedModel, PreTrainedTokenizer
from abc import abstractmethod
import abc


class BaseExplainer:
    ALLOWED_MODELS = [
        "model1"
    ]
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 attribution_type: str = "shap"):
        """

        Args:
            model (PreTrainedModel): [description]
            tokenizer (PreTrainedTokenizer): [description]
            attribution_type (str, optional): [description]. Defaults to "shap".

        Raises:
            NotImplementedError: [description]
        """
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
        pass

    @abstractmethod
    def get_layer_attributions(self):
        pass

    @staticmethod
    def encode_inputs(tokenizer: PreTrainedTokenizer, inputs: str):
        tokenizer.encode_plus(inputs,
                              pad_to_max_length=True)

    @staticmethod
    def decode_inputs(tokenizer: PreTrainedTokenizer, token_ids: str):
        tokenizer.decode(token_ids)


class SequenceClassificationExplainer(BaseExplainer):

    def get_layer_attributions(self):
        print("SCE attributions")

    def get_model_attributions(self):
        print("SCE model attributions")


class QuestionAnsweringExplainer(BaseExplainer):
    pass


class NERExplainer(BaseExplainer):
    pass


TokenClassificationExplainer = NERExplainer


class LMExplainer(BaseExplainer):
    pass


class SummarizationExplainer(BaseExplainer):
    pass
