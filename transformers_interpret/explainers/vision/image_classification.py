import warnings
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Tuple, Union

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
        attribution_type: str = AttributionType.INTEGRATED_GRADIENTS_NOISE_TUNNEL.value,
        custom_labels: Optional[List[str]] = None,
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        if attribution_type not in [attribution.value for attribution in AttributionType]:
            raise ValueError(f"Attribution type {attribution_type} not supported.")

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
        self.n_steps_noise_tunnel = 5
        self.noise_tunnel_n_samples = 10
        self.noise_tunnel_type = NoiseTunnelType.SMOOTHGRAD.value

        self.attributions = None

    def visualize(
        self,
        save_path: Union[str, None] = None,
        method: str = "overlay",
        sign: str = "all",
        outlier_threshold: float = 0.1,
        use_original_image_pixels: bool = True,
        side_by_side: bool = False,
    ):
        outlier_threshold = min(outlier_threshold * 100, 100)
        np_attributions = np.transpose(self.attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))
        if use_original_image_pixels:
            np_image = np.asarray(
                self.feature_extractor.resize(self._image, size=(np_attributions.shape[0], np_attributions.shape[1]))
            )
        else:
            # uses the normalized image pixels which is what the model sees, but can be hard to interpret visually
            np_image = np.transpose(self.inputs["pixel_values"].squeeze().cpu().detach().numpy(), (1, 2, 0))

        if sign == "all" and method in ["alpha_scaling", "masked_image"]:
            warnings.warn(
                "sign='all' is not supported for method='alpha_scaling' or method='masked_image'. "
                "Please use sign='positive', sign='negative', or sign='absolute'. "
                "Changing sign to default 'positive'."
            )
            sign = "positive"

        visualizer = ImageAttributionVisualizer(
            attributions=np_attributions,
            pixel_values=np_image,
            outlier_threshold=outlier_threshold,
            pred_class=self.id2label[self.predicted_index],
            visualization_method=method,
            sign=sign,
            side_by_side=side_by_side,
        )

        viz_result = visualizer()
        if save_path:
            viz_result[0].savefig(save_path)

        return viz_result

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

        if self.attribution_type == AttributionType.INTEGRATED_GRADIENTS.value:
            ig = IntegratedGradients(self._forward_func)
            self.attributions, self.delta = ig.attribute(
                inputs["pixel_values"],
                target=self.selected_index,
                internal_batch_size=self.internal_batch_size,
                n_steps=self.n_steps,
                return_convergence_delta=True,
            )
        if self.attribution_type == AttributionType.INTEGRATED_GRADIENTS_NOISE_TUNNEL.value:
            ig_nt = IntegratedGradients(self._forward_func)
            nt = NoiseTunnel(ig_nt)
            self.attributions = nt.attribute(
                inputs["pixel_values"],
                nt_samples=self.noise_tunnel_n_samples,
                nt_type=self.noise_tunnel_type,
                target=self.selected_index,
                n_steps=self.n_steps_noise_tunnel,
            )

        return self.attributions

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

    def __call__(
        self,
        image: Image,
        index: int = None,
        class_name: str = None,
        internal_batch_size: Union[int, None] = None,
        n_steps: Union[int, None] = None,
        n_steps_noise_tunnel: Union[int, None] = None,
        noise_tunnel_n_samples: Union[int, None] = None,
        noise_tunnel_type: NoiseTunnelType = NoiseTunnelType.SMOOTHGRAD.value,
    ):
        self._image: Image = image
        try:
            self.noise_tunnel_type = NoiseTunnelType(noise_tunnel_type).value
        except ValueError:
            raise ValueError(f"noise_tunnel_type must be one of {NoiseTunnelType.__members__}")

        self.inputs = self.feature_extractor(image, return_tensors="pt")
        self.predicted_index = self.model(self.inputs["pixel_values"]).logits.argmax().item()

        if n_steps:
            self.n_steps = n_steps
        if n_steps_noise_tunnel:
            self.n_steps_noise_tunnel = n_steps_noise_tunnel
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size
        if noise_tunnel_n_samples:
            self.noise_tunnel_n_samples = noise_tunnel_n_samples

        return self._calculate_attributions(self.inputs, class_name, index)


class VisualizationMethods(Enum):
    HEATMAP = "heatmap"
    OVERLAY = "overlay"
    ALPHA_SCALING = "alpha_scaling"
    MASKED_IMAGE = "masked_image"


class SignType(Enum):
    ALL = "all"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ABSOLUTE = "absolute"


class ImageAttributionVisualizer:
    def __init__(
        self,
        attributions: np.ndarray,
        pixel_values: np.ndarray,
        outlier_threshold: float,
        pred_class: str,
        sign: str,
        visualization_method: str,
        side_by_side: bool = False,
    ):
        self.attributions = attributions
        self.pixel_values = pixel_values
        self.outlier_threshold = outlier_threshold
        self.pred_class = pred_class
        self.render_pyplot = self._using_ipython()
        self.side_by_side = side_by_side
        try:
            self.visualization_method = VisualizationMethods(visualization_method)
        except ValueError:
            raise ValueError(
                f"""`visualization_method` must be one of the following: {list(VisualizationMethods.__members__.keys())}"""
            )

        try:
            self.sign = SignType(sign)
        except ValueError:
            raise ValueError(f"""`sign` must be one of the following: {list(SignType.__members__.keys())}""")
        if self.visualization_method == VisualizationMethods.HEATMAP:
            self.plot_function = self.heatmap
        elif self.visualization_method == VisualizationMethods.OVERLAY:
            self.plot_function = self.overlay
        elif self.visualization_method == VisualizationMethods.ALPHA_SCALING:
            self.plot_function = self.alpha_scaling
        elif self.visualization_method == VisualizationMethods.MASKED_IMAGE:
            self.plot_function = self.masked_image

    def overlay(self):
        if self.side_by_side:
            return viz.visualize_image_attr_multiple(
                attr=self.attributions,
                original_image=self.pixel_values,
                methods=["original_image", "blended_heat_map"],
                signs=["all", "absolute_value" if self.sign.value == "absolute" else self.sign.value],
                show_colorbar=True,
                use_pyplot=self.render_pyplot,
                outlier_perc=self.outlier_threshold,
                titles=["Original Image", f"Heatmap overlay IG. Prediction: {self.pred_class}"],
            )
        return viz.visualize_image_attr(
            original_image=self.pixel_values,
            attr=self.attributions,
            sign="absolute_value" if self.sign.value == "absolute" else self.sign.value,
            method="blended_heat_map",
            show_colorbar=True,
            outlier_perc=self.outlier_threshold,
            title=f"Heatmap Overlay IG. Prediction - {self.pred_class}",
            use_pyplot=self.render_pyplot,
        )

    def heatmap(self):
        if self.side_by_side:
            return viz.visualize_image_attr_multiple(
                attr=self.attributions,
                original_image=self.pixel_values,
                methods=["original_image", "heat_map"],
                signs=["all", "absolute_value" if self.sign.value == "absolute" else self.sign.value],
                show_colorbar=True,
                use_pyplot=self.render_pyplot,
                outlier_perc=self.outlier_threshold,
                titles=["Original Image", f"Heatmap IG. Prediction: {self.pred_class}"],
            )

        return viz.visualize_image_attr(
            original_image=self.pixel_values,
            attr=self.attributions,
            sign="absolute_value" if self.sign.value == "absolute" else self.sign.value,
            method="heat_map",
            show_colorbar=True,
            outlier_perc=self.outlier_threshold,
            title=f"Heatmap IG. Prediction - {self.pred_class}",
            use_pyplot=self.render_pyplot,
        )

    def alpha_scaling(self):
        if self.side_by_side:
            return viz.visualize_image_attr_multiple(
                attr=self.attributions,
                original_image=self.pixel_values,
                methods=["original_image", "alpha_scaling"],
                signs=["all", "absolute_value" if self.sign.value == "absolute" else self.sign.value],
                show_colorbar=True,
                use_pyplot=self.render_pyplot,
                outlier_perc=self.outlier_threshold,
                titles=["Original Image", f"Alpha Scaled IG. Prediction: {self.pred_class}"],
            )

        return viz.visualize_image_attr(
            original_image=self.pixel_values,
            attr=self.attributions,
            sign="absolute_value" if self.sign.value == "absolute" else self.sign.value,
            method="alpha_scaling",
            show_colorbar=True,
            outlier_perc=self.outlier_threshold,
            title=f"Alpha Scaling IG. Prediction - {self.pred_class}",
            use_pyplot=self.render_pyplot,
        )

    def masked_image(self):
        if self.side_by_side:
            return viz.visualize_image_attr_multiple(
                attr=self.attributions,
                original_image=self.pixel_values,
                methods=["original_image", "masked_image"],
                signs=["all", "absolute_value" if self.sign.value == "absolute" else self.sign.value],
                show_colorbar=True,
                use_pyplot=self.render_pyplot,
                outlier_perc=self.outlier_threshold,
                titles=["Original Image", f"Masked Image IG. Prediction: {self.pred_class}"],
            )
        return viz.visualize_image_attr(
            original_image=self.pixel_values,
            attr=self.attributions,
            sign="absolute_value" if self.sign.value == "absolute" else self.sign.value,
            method="masked_image",
            show_colorbar=True,
            outlier_perc=self.outlier_threshold,
            title=f"Masked Image IG. Prediction - {self.pred_class}",
            use_pyplot=self.render_pyplot,
        )

    def _using_ipython(self) -> bool:
        try:
            eval("__IPYTHON__")
        except NameError:
            return False
        else:  # pragma: no cover
            return True

    def __call__(self):
        return self.plot_function()
