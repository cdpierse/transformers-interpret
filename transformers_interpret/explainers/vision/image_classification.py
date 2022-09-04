from enum import Enum, unique
from typing import List, Optional, Union

import numpy as np
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers.image_utils import ImageFeatureExtractionMixin
from transformers.modeling_utils import PreTrainedModel

from .attribution_types import AttributionType, NoiseTunnelType


class ImageClassificationExplainer:
    """
    This class is used to explain the output of a model on an image.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        feature_extractor: ImageFeatureExtractionMixin,
        attribution_type: str = AttributionType.INTEGRATED_GRADIENTS_NOISE_TUNNEL,
        custom_labels: Optional[List[str]] = None,
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        if not isinstance(attribution_type, AttributionType):
            raise TypeError("attribution_type must be an instance of AttributionType Enum")

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

        self.device = self.model.device

        self.internal_batch_size = None
        self.n_steps = 50
        self.n_steps_noise_tunell = 5
        self.noise_tunell_n_samples = 10
        self.noise_tunell_type = NoiseTunnelType.SMOOTHGRAD.value

        self.attributions = None

    def visualize(self, save_path: str = None, show: bool = True, method: str = "overlay", sign: str = "all"):

        np_attributions = np.transpose(self.attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))
        np_image = np.transpose(self.inputs["pixel_values"].squeeze().cpu().detach().numpy(), (1, 2, 0))
        visualizer = ImageAttributionVisualizer(
            attributions=np_attributions,
            pixel_values=np_image,
            outlier_threshold=self.outlier_threshold,
            pred_class=self.id2label[self.predicted_index],
        )

    def get_image(self):
        return self.inputs["pixel_values"]

    def _forward_func(self, inputs):
        outputs = self.model(inputs)
        return outputs["logits"]

    def _calculate_attributions(self, inputs: torch.Tensor, class_name: Union[int, None], index: Union[int, None]):

        if class_name:
            self.selected_index = self.label2id[class_name]

        if index:
            self.selected_index = index
        else:
            self.selected_index = self.predicted_index

        if self.attribution_type == AttributionType.INTEGRATED_GRADIENTS:
            ig = IntegratedGradients(self._forward_func)
            self.attributions, self.delta = ig.attribute(
                inputs["pixel_values"],
                target=self.selected_index,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
                return_convergence_delta=True,
            )
        if self.attribution_type == AttributionType.INTEGRATED_GRADIENTS_NOISE_TUNNEL:
            ig_nt = IntegratedGradients(self._forward_func)
            nt = NoiseTunnel(ig_nt)
            self.attributions = nt.attribute(
                inputs["pixel_values"],
                nt_samples=self.noise_tunell_n_samples,
                nt_type=self.noise_tunell_type,
                target=self.selected_index,
                n_steps=self.n_steps_noise_tunell,
            )

    def __call__(
        self,
        image: Image,
        index: int = None,
        class_name: str = None,
        internal_batch_size: Union[int, None] = None,
        n_steps: Union[int, None] = None,
        n_steps_noise_tunell: Union[int, None] = None,
        noise_tunell_n_samples: Union[int, None] = None,
        noise_tunell_type: NoiseTunnelType = NoiseTunnelType.SMOOTHGRAD,
        outlier_threshold: Union[float, None] = 0.1,
    ):
        self.noise_tunell_type = noise_tunell_type.value
        self.inputs = self.feature_extractor(image, return_tensors="pt")
        self.predicted_index = self.model(self.inputs["pixel_values"]).logits.argmax().item()
        self.outlier_threshold = outlier_threshold * 100

        if n_steps:
            self.n_steps = n_steps
        if n_steps_noise_tunell:
            self.n_steps_noise_tunell = n_steps_noise_tunell
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size
        if noise_tunell_n_samples:
            self.noise_tunell_n_samples = noise_tunell_n_samples

        self._calculate_attributions(self.inputs, class_name, index)


class VisualizationMethods(Enum):
    HEATMAP = "heatmap"
    OVERLAY = "overlay"


class SignType(Enum):
    ALL = "all"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ABSOULTE = "absolute"


# TODO: create a class specific for image classification that can visualize the attributions
# in a variety of ways e.g. overlay, heatmap, side-by-side, etc.
class ImageAttributionVisualizer:
    def __init__(
        self,
        attributions: np.ndarray,
        pixel_values: np.ndarray,
        outlier_threshold: float,
        pred_class: str,
    ):
        self.attributions = attributions
        self.pixel_values = pixel_values
        self.outlier_threshold = outlier_threshold
        self.pred_class = pred_class

    def overlay(self):
        _ = viz.visualize_image_attr(
            attr=self.attributions,
            sign="all",
            method="blended_heat_map",
            show_colorbar=True,
            outlier_perc=self.outlier_threshold,
            title=f"Heatmap Overlay IG. Prediction - {self.pred_class}",
        )

    def heatmap(self):
        _ = viz.visualize_image_attr(
            attr=self.attributions,
            sign="all",
            method="heat_map",
            show_colorbar=True,
            outlier_perc=self.outlier_threshold,
            title=f"Heatmap IG. Prediction - {self.pred_class}",
        )

    def side_by_side(self):
        pass

    def original_image(self):
        img = viz.visualize_image_attr(
            None, self.pixel_values, method="original_image", title=f"Original Image. Prediction - {self.pred_class}"
        )

    def save(self):
        pass
