from enum import Enum, unique


@unique
class AttributionType(Enum):
    INTEGRATED_GRADIENTS = "IntegratedGradients"
    INTEGRATED_GRADIENTS_NOISE_TUNNEL = "IntegratedGradientsNoiseTunnell"


class NoiseTunellType(Enum):
    SMOOTHGRAD = "smoothgrad"
    SMOOTHGRAD_SQUARED = "smoothgrad_sq"
    VARGRAD = "vargrad"
