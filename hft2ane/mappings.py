import importlib
import pkgutil
from collections import defaultdict
from types import ModuleType
from typing import Type, TypeVar

from transformers import modeling_outputs
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto import modeling_auto

from hft2ane.exceptions import ModelNotFoundError


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
    """
    Maps model type names to the AutoModel classes registered for that model type.
    """
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


def _names_to_hft2ane_models() -> dict[str, list[Type[PreTrainedModel]]]:
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


_BASE_NAMES_TO_ANETZ_MODELS = _names_to_hft2ane_models()


def get_hft2ane_model_names(name: str) -> list[str]:
    """
    For most pre-trained model names on HuggingFace Hub, this function returns a
    list of the corresponding hft2ane model classes valid for that model type.

    pre-trained models define specific model class/es to use, e.g.:

        >>> config = AutoConfig.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        >>> model.config.architectures
        ['DistilBertForSequenceClassification']

    TODO: do all HF models define this attribute correctly?
    (i.e. would we ever need to allow manual override of this mapping?)
    """
    config = AutoConfig.from_pretrained(name)
    names = [
        model.__name__
        for model in _BASE_NAMES_TO_ANETZ_MODELS[config.model_type]
        if model.__name__ in config.architectures
    ]
    if not names:
        raise ModelNotFoundError(f"Could not find hft2ane model matching: {name}")
    return names


def get_hft2ane_model(model: Type[PreTrainedModel]) -> Type[PreTrainedModel]:
    """
    For a given HuggingFace model class, this function returns the corresponding
    hft2ane model class, if there is one.
    """
    for cls in _BASE_NAMES_TO_ANETZ_MODELS[model.config_class.model_type]:
        if cls.__name__ == model.__name__:
            return cls
    raise ModelNotFoundError(f"Could not find hft2ane model matching: {model.__name__}")


_CONCRETE_TO_AUTO = {
    concrete_model: auto_model
    for auto_model in _AUTO_MODELS
    for concrete_model in auto_model._model_mapping.values()
}


def get_hf_auto_model(
    model: Type[PreTrainedModel],
) -> Type[modeling_auto._BaseAutoModelClass]:
    """
    For a given HuggingFace model class, this function returns the corresponding
    HF AutoModel class, if there is one.
    """
    try:
        return _CONCRETE_TO_AUTO[model]
    except KeyError:
        raise ModelNotFoundError(
            f"Could not find HF AutoModel matching: {model.__name__}"
        )


_AUTO_MODEL_TO_OUTPUT = {
    modeling_auto.AutoBackbone: modeling_outputs.BackboneOutput,
    modeling_auto.AutoModel: modeling_outputs.BaseModelOutput,
    # modeling_auto.AutoModelForAudioClassification: ,
    # modeling_auto.AutoModelForAudioFrameClassification: ,
    # modeling_auto.AutoModelForAudioXVector: ,
    # modeling_auto.AutoModelForCTC: ,
    modeling_auto.AutoModelForCausalLM: modeling_outputs.CausalLMOutput,
    modeling_auto.AutoModelForDepthEstimation: modeling_outputs.DepthEstimatorOutput,
    # modeling_auto.AutoModelForDocumentQuestionAnswering: ,
    modeling_auto.AutoModelForImageClassification: modeling_outputs.ImageClassifierOutput,
    # modeling_auto.AutoModelForImageSegmentation: ,
    # modeling_auto.AutoModelForInstanceSegmentation: ,
    # modeling_auto.AutoModelForMaskedImageModeling: ,
    modeling_auto.AutoModelForMaskedLM: modeling_outputs.MaskedLMOutput,
    modeling_auto.AutoModelForMultipleChoice: modeling_outputs.MultipleChoiceModelOutput,
    modeling_auto.AutoModelForNextSentencePrediction: modeling_outputs.NextSentencePredictorOutput,
    # modeling_auto.AutoModelForObjectDetection: ,
    # modeling_auto.AutoModelForPreTraining: ,
    modeling_auto.AutoModelForQuestionAnswering: modeling_outputs.QuestionAnsweringModelOutput,
    modeling_auto.AutoModelForSemanticSegmentation: modeling_outputs.SemanticSegmenterOutput,
    modeling_auto.AutoModelForSeq2SeqLM: modeling_outputs.Seq2SeqLMOutput,
    modeling_auto.AutoModelForSequenceClassification: modeling_outputs.SequenceClassifierOutput,
    # modeling_auto.AutoModelForSpeechSeq2Seq: ,
    # modeling_auto.AutoModelForTableQuestionAnswering: ,
    modeling_auto.AutoModelForTokenClassification: modeling_outputs.TokenClassifierOutput,
    # modeling_auto.AutoModelForUniversalSegmentation: ,
    # modeling_auto.AutoModelForVideoClassification: ,
    # modeling_auto.AutoModelForVision2Seq: ,
    # modeling_auto.AutoModelForVisualQuestionAnswering: ,
    # modeling_auto.AutoModelForZeroShotImageClassification: ,
    # modeling_auto.AutoModelForZeroShotObjectDetection: ,
    # modeling_auto.AutoModelWithLMHead: ,
}


def get_output_for_auto_model(
    model: Type[modeling_auto._BaseAutoModelClass],
) -> Type[modeling_outputs.ModelOutput]:
    """
    For a given HF AutoModel class, this function returns the corresponding
    transformers ModelOutput class, if there is one.
    """
    try:
        return _AUTO_MODEL_TO_OUTPUT[model]
    except KeyError:
        raise ModelNotFoundError(
            f"Could not find ModelOutput matching: {model.__name__}"
        )
