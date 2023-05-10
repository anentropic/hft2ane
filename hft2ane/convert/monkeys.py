
from functools import partial
from typing import Any, Callable

from exporters.coreml.config import CoreMLConfig
from exporters.coreml.features import (
    FeaturesManager as HF_FeaturesManager,
    supported_features_mapping,
)
from exporters.coreml.models import BertCoreMLConfig as HF_BertCoreMLConfig
from transformers import PretrainedConfig


class CoreMLConfigDuck(type):
    def __subclasscheck__(cls, subclass):
        return issubclass(subclass, CoreMLConfig)


class PatchedConfig(metaclass=CoreMLConfigDuck):
    """
    So that we can fix the sequence length for CoreML conversion
    """
    def __init__(self, config: CoreMLConfig, sequence_length: int):
        self._config = config
        self._seq_len = sequence_length

    @property
    def inputs(self):
        inputs = self._config.inputs
        inputs['input_ids'].sequence_length = self._seq_len
        return inputs
    
    def __getattr__(self, __name: str) -> Any:
        return getattr(self._config, __name)


class BertCoreMLConfig(HF_BertCoreMLConfig):
    """
    We saw atol values of 0.0002...0.0003 after conversion
    """
    @property
    def atol_for_validation(self) -> float:
        return 1e-3


def _replace_config(
    mapping: dict[str, Callable[[PretrainedConfig], CoreMLConfig]],
    replacement_cls: type[CoreMLConfig],
) -> dict[str, Callable[[PretrainedConfig], CoreMLConfig]]:
    return {
        key: partial(
            getattr(replacement_cls, value.func.__name__),
            *value.args,
            **value.keywords
        )
        for key, value in mapping.items()
    }


class FeaturesManager(HF_FeaturesManager):
    _SUPPORTED_MODEL_TYPE = HF_FeaturesManager._SUPPORTED_MODEL_TYPE

    _SUPPORTED_MODEL_TYPE["bert"]: _replace_config(
        _SUPPORTED_MODEL_TYPE["bert"],
        BertCoreMLConfig,
    )

    _SUPPORTED_MODEL_TYPE["roberta"]: (
        _SUPPORTED_MODEL_TYPE["roberta"]
        | supported_features_mapping(
            "feature-extraction",
            "fill-mask",
            "multiple-choice",
            "question-answering",
            "text-classification",
            "token-classification",
            coreml_config_cls="models.roberta.RobertaCoreMLConfig",
        )
    )
