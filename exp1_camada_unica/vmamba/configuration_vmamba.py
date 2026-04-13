# coding=utf-8
# @Author  : Saurabhchand Bhati
# @Affiliation  : Massachusetts Institute of Technology
"""VMamba: Visual State Space Model configuration"""

from typing import Any, Dict

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class VMambaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VMambaModel`]. It is used to instantiate a VMamba
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [VMamba-T](https://github.com/MzeroMiko/VMamba/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch.
        embed_dim (`int`, *optional*, defaults to 96):
            Dimensionality of patch embedding.
        depths (`list(int)`, *optional*, defaults to `[2, 2, 8, 2]`):
            Depth of each layer in the VMamba encoder.
        dims (`list(int)`, *optional*, defaults to `[96, 192, 384, 768]`):
            Dimensionality of each layer in the VMamba encoder.
        drop_path_rate (`float`, *optional*, defaults to 0.2):
            Stochastic depth rate.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classes for classification.
        max_length (`int`, *optional*, defaults to 1024):
            Temporal dimension of the spectrograms.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Frequency dimension of the spectrograms (number of Mel-frequency bins).
        use_checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to use checkpointing to save memory.

    Example:

    ```python
    >>> from transformers import VMambaConfig, VMambaModel

    >>> # Initializing a VMamba tiny style configuration
    >>> configuration = VMambaConfig()

    >>> # Initializing a model (with random weights) from the VMamba tiny style configuration
    >>> model = VMambaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vmamba"

    def __init__(
        self,
        num_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: list = [2, 2, 8, 2],
        dims: list =[96, 192, 384, 768],
        drop_path_rate: float = 0.2,
        num_classes: int = 1000,
        max_length: int = 1024,
        num_mel_bins: int = 128,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins
        self.use_checkpoint = use_checkpoint

    # Overwritten from the parent class: VMamba is not compatible with `generate`, but has a config parameter sharing the
    # same name (`max_length`). Sharing the same name triggers checks regarding the config -> generation_config
    # generative parameters deprecation cycle, overwriting this function prevents this from happening.
    def _get_non_default_generation_parameters(self) -> Dict[str, Any]:
        return {}


__all__ = ["VMambaConfig"]
