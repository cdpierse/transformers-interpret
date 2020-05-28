import pandas
from transformers import PreTrainedModel


class BaseExplainer:

    def __init__(self, model: PreTrainedModel, model_type: str):
        pass


class SequenceClassificationExplainer(BaseExplainer):
    pass


class QuestionAnsweringExplainer(BaseExplainer):
    pass


class TokenClassificationExplainer(BaseExplainer):
    pass


class LMExplainer(BaseExplainer):
    pass
