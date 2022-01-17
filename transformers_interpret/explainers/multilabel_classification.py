import warnings
from typing import Union

import torch
from captum.attr import visualization as viz
from torch.nn.modules.sparse import Embedding
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer, LIGAttributions
from transformers_interpret.errors import AttributionTypeNotSupportedError, InputIdsNotCalculatedError

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]
