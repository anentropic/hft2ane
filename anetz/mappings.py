import importlib
import pkgutil
from collections import defaultdict
from types import ModuleType
from typing import Type, TypeVar

from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto import modeling_auto

from anetz.exceptions import ModelNotFoundError


T = TypeVar("T", bound=Type)


def _get_public_classes(module: ModuleType, base_class: Type[T]) -> list[Type[T]]:
    return [
        cls
        for attr in dir(module)
        if not attr.startswith("_")
        if (cls := getattr(module, attr))
        if isinstance(cls, type)
        if issubclass(cls, base_class)
    ]


def _auto_models() -> list[Type[modeling_auto._BaseAutoModelClass]]:
    return _get_public_classes(modeling_auto, modeling_auto._BaseAutoModelClass)


_AUTO_MODELS = _auto_models()


def _names_to_auto_models() -> dict[str, list[Type[modeling_auto._BaseAutoModelClass]]]:
    mapping = defaultdict(list)
    for cls in _AUTO_MODELS:
        for name in cls._model_mapping._model_mapping.keys():
            mapping[name].append(cls)
    return mapping


_BASE_NAMES_TO_AUTO_MODELS = _names_to_auto_models()


def get_hf_auto_models(name: str) -> list[Type[modeling_auto._BaseAutoModelClass]]:
    """
    For most pre-trained model names on HuggingFace Hub, this function returns a
    list of the corresponding HF AutoModel classes valid for that model type.
    """
    config = AutoConfig.from_pretrained(name)
    return _BASE_NAMES_TO_AUTO_MODELS[config.model_type]


def get_hf_concrete_models(name: str) -> list[Type[PreTrainedModel]]:
    """
    For most pre-trained model names on HuggingFace Hub, this function returns a
    list of the corresponding HF concrete model classes valid for that model type.
    """
    config = AutoConfig.from_pretrained(name)
    return [
        cls._model_mapping[config.__class__]
        for cls in _BASE_NAMES_TO_AUTO_MODELS[config.model_type]
    ]


def _names_to_anetz_models() -> dict[str, list[Type[PreTrainedModel]]]:
    from . import models

    mapping = {}
    for info in pkgutil.iter_modules(models.__path__):
        module = importlib.import_module(f"{models.__name__}.{info.name}")
        try:
            model_type = module.MODEL_TYPE
        except AttributeError:
            continue
        mapping[model_type] = _get_public_classes(module, PreTrainedModel)
    return mapping


_BASE_NAMES_TO_ANETZ_MODELS = _names_to_anetz_models()


def get_anetz_model_names(name: str) -> list[str]:
    """
    For most pre-trained model names on HuggingFace Hub, this function returns a
    list of the corresponding anetz model classes valid for that model type.

    pre-trained models define specific model class/es to use, e.g.:

        >>> config = AutoConfig.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        >>> model.config.architectures
        ['DistilBertForSequenceClassification']

    TODO: do all HF models define this attribute (correctly)?
    """
    config = AutoConfig.from_pretrained(name)
    return [
        model.__name__
        for model in _BASE_NAMES_TO_ANETZ_MODELS[config.model_type]
        if model.__name__ in config.architectures
    ]


def get_anetz_model(model: Type[PreTrainedModel]) -> Type[PreTrainedModel]:
    """
    For a given HuggingFace model class, this function returns the corresponding
    anetz model class, if there is one.
    """
    for cls in _BASE_NAMES_TO_ANETZ_MODELS[model.config_class.model_type]:
        if cls.__name__ == model.__name__:
            return cls
    raise ModelNotFoundError(f"Could not find anetz model matching: {model.__name__}")
