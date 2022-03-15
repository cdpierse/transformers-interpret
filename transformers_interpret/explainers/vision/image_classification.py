from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers.image_utils import ImageFeatureExtractionMixin
from transformers.modeling_utils import PreTrainedModel


class ImageClassificationExplainer:
    """
    This class is used to explain the output of a model on an image.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        feature_extractor: ImageFeatureExtractionMixin,
        attribution_type: str = "ig",
        add_noise_tunnel: bool = False,
        custom_labels: Optional[List[str]] = None,
    ):
        self.model = model
        self.feature_extractor = feature_extractor

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

        self.device = self.model.device

    def visualize(self):
        pass

    def _calculate_attributions(self, inputs: torch.Tensor, target: int):
        pass

    def __call__(self, image: Image):
        self.inputs = self.feature_extractor(image, return_tensors="tf")
