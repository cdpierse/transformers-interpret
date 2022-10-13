from enum import Enum, unique


@unique
class AttributionType(Enum):
    INTEGRATED_GRADIENTS = "IG"
    INTEGRATED_GRADIENTS_NOISE_TUNNEL = "IGNT"


class NoiseTunnelType(Enum):
    SMOOTHGRAD = "smoothgrad"
    SMOOTHGRAD_SQUARED = "smoothgrad_sq"
    VARGRAD = "vargrad"
