import os
from typing import Any, Type, Union
from warnings import warn

import torch
import coremltools as ct
import numpy as np
from exporters.coreml.config import CoreMLConfig
from exporters.coreml.convert import export
from exporters.coreml.features import FeaturesManager
from huggingface_hub import model_info
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import _BaseAutoModelClass
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.onnx.utils import get_preprocessor

from hft2ane.mappings import get_hft2ane_model
from hft2ane.utils.introspection import model_config


"""
This is adapted from the method here:
https://github.com/apple/ml-ane-transformers#tutorial-optimized-deployment-of-hugging-face-distilbert

We implement a specific HuggingFace-specific workflow. If you have other needs
you are best off using the original tutorial as a guide.

TODO:
How do we support a model like this?
https://huggingface.co/mosaicml/mpt-1b-redpajama-200b/blob/main/mosaic_gpt.py
...it has custom code but inherits from PreTrainedModel and can be loaded
via AutoModelForCausalLM.from_pretrained
...implies we should support choosing AutoModelFor* classes in the cli
from_pretrained(trust_remote_code=True) allows the AutoModel to use custom code
from the model's HF repo (as opposed to only the `transformers` package)
"""


ModelT = Union[Type[PreTrainedModel], Type[_BaseAutoModelClass]]

PreprocessorT = Union[PreTrainedTokenizerBase, FeatureExtractionMixin, ProcessorMixin]


METADATA_MODEL_NAME_KEY = "com.github.anentropic.hft2ane.model"
METADATA_MODEL_CLS_KEY = "com.github.anentropic.hft2ane.type"
METADATA_MODEL_SEQ_LEN_KEY = "com.github.anentropic.hft2ane.sequenceLength"


def get_baseline_model(model_name: str, model_cls: ModelT) -> PreTrainedModel:
    return model_cls.from_pretrained(
        model_name,
        return_dict=False,
        torchscript=True,
    ).eval()


def _init_hft2ane_model(baseline_model: PreTrainedModel) -> PreTrainedModel:
    hft2ane_model_cls = get_hft2ane_model(baseline_model.__class__)
    hft2ane_model = hft2ane_model_cls(baseline_model.config).eval()
    hft2ane_model.load_state_dict(baseline_model.state_dict())
    return hft2ane_model


def get_models_for_conversion(
    model_name: str, model_cls: ModelT
) -> tuple[PreTrainedModel, PreTrainedModel]:
    baseline_model = get_baseline_model(model_name, model_cls)
    hft2ane_model = _init_hft2ane_model(baseline_model)
    return baseline_model, hft2ane_model


def _pt_to_np_dtype(pt_dtype: torch.dtype) -> np.generic:
    """
    ct.TensorType.dtype expects a numpy dtype (or a CoreML MIL type).
    We exploit the fact that the torch and numpy dtypes are named the same.

    https://github.com/apple/coremltools/issues/1498
    it seems that coremltools does not support int64, the default int type,
    for prediction (inference) and we have to use int32 instead.
    Only a problem if your inputs have values > 2**31. (~2B)
    """
    name = str(pt_dtype).replace("torch.", "")
    if name == "int64":
        name = "int32"
        warn(f"Converting {pt_dtype} input tensor to {name} for CoreML compatibility.")
    return getattr(np, np.dtype(name).name)


def _get_ct_inputs(model: PreTrainedModel) -> list[ct.TensorType]:
    return [
        ct.TensorType(
            name,
            shape=tensor.shape,
            dtype=_pt_to_np_dtype(tensor.dtype),
        )
        for name, tensor in model.dummy_inputs.items()
    ]


def _get_ct_outputs(model: PreTrainedModel) -> list[ct.TensorType]:
    with model_config(model, return_dict=True, torchscript=False):
        dummy_outputs = model(**model.dummy_inputs)
    return [
        ct.TensorType(
            name,
            dtype=_pt_to_np_dtype(tensor.dtype),
        )
        for name, tensor in dummy_outputs.items()
    ]


def _set_metadata(
    mlmodel: ct.models.MLModel,
    model_name: str,
    model_cls_name: str | None,
    sequence_length: int | None = None,
) -> None:
    try:
        hfinfo = model_info(model_name)
    except HfHubHTTPError as e:
        warn(
            f"Failed to fetch model info for {model_name} due to: {e!r} "
            "Unable to set author metadata."
        )
    else:
        try:
            mlmodel.license = hfinfo.cardData["license"]
        except (AttributeError, KeyError):
            pass
        mlmodel.author = hfinfo.author or "<via Hugging Face>"
        mlmodel.version = hfinfo.sha
        mlmodel.short_description = (
            f"{hfinfo.modelId} re-implemented using hft2ane "
            "for compatibility with execution on Neural Engine."
        )
    mlmodel.user_defined_metadata[METADATA_MODEL_NAME_KEY] = model_name
    if sequence_length is not None:
        mlmodel.user_defined_metadata[METADATA_MODEL_SEQ_LEN_KEY] = str(sequence_length)
    if model_cls_name:
        mlmodel.user_defined_metadata[METADATA_MODEL_CLS_KEY] = model_cls_name


def hf_to_coreml(
    hft2ane_model: PreTrainedModel,
    task: str,
    preprocessor_name: str,
    sequence_length: int | None,
    out_path: str,
) -> tuple[ct.models.MLModel, PreprocessorT, CoreMLConfig]:
    # TODO: FeaturesManager has weird gaps in coverage for e.g. RoBERTa
    _, model_coreml_config = FeaturesManager.check_supported_model_or_raise(
        hft2ane_model, feature=task
    )

    overrides: dict[str, Any] = (
        {"inferSequenceLengthFromConfig": True}
        if sequence_length is None
        else {"sequenceLength": sequence_length}
    )

    # TODO: use_past (pre-cached Decoder) or seq2seq (EncoderDecoder)
    coreml_config = model_coreml_config(hft2ane_model.config, overrides=overrides)

    model_name = hft2ane_model.name_or_path

    # Instantiate the appropriate preprocessor
    preprocessor = None
    if preprocessor_name == "auto":
        preprocessor = get_preprocessor(model_name)
    elif preprocessor_name == "tokenizer":
        preprocessor = AutoTokenizer.from_pretrained(model_name)
    elif preprocessor_name == "feature_extractor":
        preprocessor = AutoFeatureExtractor.from_pretrained(model_name)
    elif preprocessor_name == "processor":
        preprocessor = AutoProcessor.from_pretrained(model_name)

    if not preprocessor:
        raise ValueError(
            f"Unknown preprocessor type '{preprocessor_name}' for '{model_name}'"
        )

    converted = export(
        preprocessor,
        hft2ane_model,
        coreml_config,
        quantize="float16",
    )
    # so far HF `export` doesn't set author etc metadata
    # (but it does set input/output descriptions and classifier config)
    _set_metadata(
        converted,
        model_name,
        hft2ane_model.__class__.__name__,
        coreml_config.sequenceLength,
    )

    # workaround for https://github.com/apple/coremltools/issues/1680
    ct.models.MLModel(
        converted._spec,
        weights_dir=converted._weights_dir,
        skip_model_load=True,
    ).save(out_path)

    converted.package_path = os.path.abspath(out_path)
    return converted, preprocessor, coreml_config


def to_coreml_internal(
    baseline_model: PreTrainedModel,
    hft2ane_model: PreTrainedModel,
    out_path: str,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
    model_cls_name: str | None = None,  # for metadata only
    is_classifier: bool = False,
) -> ct.models.MLModel:
    """
    NOTE: some models have a config.id2label attribute but are not classifiers
    ...ct.convert then complains that output shape does not match num labels
    (e.g. deepset/roberta-base-squad2)

    hence we have to manually specify `is_classifier`

    NOTE: it's unclear if setting `compute_units` at conversion time does anything
    ...when you later load the model the flag reverts to ALL, as if it wasn't stored.
    Which implies that it is only relevant at inference time, and its presence in
    `ct.convert` signature is just a convenience for setting the flag on the returned
    model instance if you intend to use it immediately (which we do, for verification)
    Confirmed: https://github.com/apple/coremltools/issues/1849
    """
    if baseline_model.num_parameters() > 1e9:
        warn(
            "Model has > 1B parameters. This is unlikely to fit within the "
            "Neural Engine's ~3GB memory constraints and will execute on CPU. "
            "Run the --confirm-neural-engine evaluation to confirm."
        )

    traced_optimized_model = torch.jit.trace(
        hft2ane_model,
        list(baseline_model.dummy_inputs.values()),
    )

    kwargs: dict[str, Any] = {}
    # if is_classifier and (class_labels := getattr(baseline_model.config, "id2label", None)):
    #     kwargs["classifier_config"] = _get_classifier_config(class_labels)
    mlmodel = ct.convert(
        traced_optimized_model,
        convert_to="mlprogram",
        inputs=_get_ct_inputs(baseline_model),
        # outputs=_get_ct_outputs(baseline_model),
        compute_units=compute_units,
        **kwargs,
    )
    assert isinstance(mlmodel, ct.models.MLModel)

    # (it's a shame we can't do this in the convert call above)
    _set_metadata(mlmodel, baseline_model.name_or_path, model_cls_name)

    # workaround for https://github.com/apple/coremltools/issues/1680
    ct.models.MLModel(
        mlmodel._spec,
        weights_dir=mlmodel._weights_dir,
        skip_model_load=True,
    ).save(out_path)

    mlmodel.package_path = os.path.abspath(out_path)
    return mlmodel


def to_coreml(
    model_name: str,
    model_cls: Type[PreTrainedModel],
    out_path: str,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
) -> ct.models.MLModel:
    """
    Args:
        model_name: the name of the pre-trained model to load from HuggingFace
        model_cls: the class of the `transformers` model to use as a base
        out_path: must include the .mlpackage extension
        compute_units: the target hardware for the CoreML model

    Suggested `out_path`:
        build/{model_name}/{model_cls.__name}.mlpackage

    TODO: we just use default config, might want to support customising it
    TODO: model_cls_name should include module path for custom models
    """
    baseline_model, hft2ane_model = get_models_for_conversion(model_name, model_cls)
    return to_coreml_internal(
        baseline_model=baseline_model,
        hft2ane_model=hft2ane_model,
        out_path=out_path,
        compute_units=compute_units,
        model_cls_name=model_cls.__name__,
    )
