import imp
import pytest
import requests
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers_interpret import ImageClassificationExplainer
from transformers_interpret.explainers.vision.attribution_types import AttributionType
import os

model_name = "apple/mobilevit-small"
MODEL = AutoModelForImageClassification.from_pretrained(model_name)
FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(model_name)

IMAGE_LINK = "https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2342&q=80"
TEST_IMAGE = Image.open(requests.get(IMAGE_LINK, stream=True).raw)


def test_image_classification_init():
    img_cls_explainer = ImageClassificationExplainer(model=MODEL, feature_extractor=FEATURE_EXTRACTOR)

    assert img_cls_explainer.model == MODEL
    assert img_cls_explainer.feature_extractor == FEATURE_EXTRACTOR
    assert img_cls_explainer.id2label == MODEL.config.id2label
    assert img_cls_explainer.label2id == MODEL.config.label2id

    assert img_cls_explainer.attributions is None


def test_image_classification_init_attribution_type_not_supported():
    with pytest.raises(ValueError):
        ImageClassificationExplainer(model=MODEL, feature_extractor=FEATURE_EXTRACTOR, attribution_type="not_supported")


def test_image_classification_init_custom_labels():
    labels = [f"label_{i}" for i in range(len(MODEL.config.id2label) - 1)]
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL, feature_extractor=FEATURE_EXTRACTOR, custom_labels=labels
    )

    assert list(img_cls_explainer.label2id.keys()) == labels


def test_image_classification_init_custom_labels_not_valid():
    with pytest.raises(ValueError):
        ImageClassificationExplainer(model=MODEL, feature_extractor=FEATURE_EXTRACTOR, custom_labels=["label_0"])


def test_image_classification_call():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1)

    assert img_cls_explainer.attributions is not None
    assert img_cls_explainer.predicted_index is not None
    assert img_cls_explainer.n_steps == 1
    assert img_cls_explainer.n_steps_noise_tunnel == 1
    assert img_cls_explainer.noise_tunnel_n_samples == 1
    assert img_cls_explainer.internal_batch_size == 1


def test_image_classification_call_attribution_type_not_supported():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    with pytest.raises(ValueError):
        img_cls_explainer(
            TEST_IMAGE,
            n_steps=1,
            n_steps_noise_tunnel=1,
            noise_tunnel_n_samples=1,
            internal_batch_size=1,
            noise_tunnel_type="not_supported",
        )


def test_image_classification_visualize():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1)

    img_cls_explainer.visualize(method="alpha_scaling", save_path=None, sign="all", outlier_threshold=0.15)


def test_image_classification_visualize_use_normed_pixel_values():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1)

    img_cls_explainer.visualize(
        method="overlay", save_path=None, sign="all", outlier_threshold=0.15, use_original_image_pixels=False
    )


def test_image_classification_visualize_wrt_class_name():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(
        TEST_IMAGE,
        n_steps=1,
        n_steps_noise_tunnel=1,
        noise_tunnel_n_samples=1,
        internal_batch_size=1,
        class_name="banana",
    )

    img_cls_explainer.visualize(method="heatmap", save_path=None, sign="all", outlier_threshold=0.15)


def test_image_classification_visualize_wrt_class_index():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(
        TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1, index=3
    )

    img_cls_explainer.visualize(method="overlay", save_path=None, sign="all", outlier_threshold=0.15)


def test_image_classification_visualize_integrated_gradients_no_noise_tunnel():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL, feature_extractor=FEATURE_EXTRACTOR, attribution_type=AttributionType.INTEGRATED_GRADIENTS.value
    )

    img_cls_explainer(
        TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1, index=3
    )

    img_cls_explainer.visualize(method="masked_image", save_path=None, sign="all", outlier_threshold=0.15)


def test_image_classification_visualize_save_image():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1)

    img_cls_explainer.visualize(
        method="overlay",
        save_path="./test.png",
        sign="all",
        outlier_threshold=0.15,
        side_by_side=True,
    )

    os.remove("./test.png")


def test_image_classification_visualize_positive_sign_for_unsupported_methods():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1)

    img_cls_explainer.visualize(
        method="masked_image",
        save_path="./test.png",
        sign="all",
        outlier_threshold=0.15,
        side_by_side=True,
    )

    os.remove("./test.png")


def test_image_classification_visualize_unsupported_viz_method():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1)

    with pytest.raises(ValueError):
        img_cls_explainer.visualize(method="not_supported", save_path=None, sign="all", outlier_threshold=0.15)


def test_image_classification_visualize_unsupported_sign():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1)

    with pytest.raises(ValueError):
        img_cls_explainer.visualize(method="overlay", save_path=None, sign="not_supported", outlier_threshold=0.15)


def test_image_classification_visualize_side_by_side():
    img_cls_explainer = ImageClassificationExplainer(
        model=MODEL,
        feature_extractor=FEATURE_EXTRACTOR,
    )

    img_cls_explainer(TEST_IMAGE, n_steps=1, n_steps_noise_tunnel=1, noise_tunnel_n_samples=1, internal_batch_size=1)

    methods = ["overlay", "heatmap", "masked_image", "alpha_scaling"]
    for method in methods:
        img_cls_explainer.visualize(
            method=method, save_path=None, sign="all", outlier_threshold=0.15, side_by_side=True
        )
