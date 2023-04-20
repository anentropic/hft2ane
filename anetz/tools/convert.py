from typing import Type
from warnings import warn

import torch
import coremltools as ct
import numpy as np
from huggingface_hub import model_info
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import _BaseAutoModelClass

from anetz.mappings import get_anetz_model


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
"""


ModelT = Type[PreTrainedModel] | Type[_BaseAutoModelClass]


def _get_baseline_model(model_name: str, model_cls: ModelT) -> PreTrainedModel:
    return model_cls.from_pretrained(
        model_name,
        return_dict=False,
        torchscript=True,
    ).eval()


def _init_anetz_model(baseline_model: PreTrainedModel) -> PreTrainedModel:
    anetz_model_cls = get_anetz_model(baseline_model.__class__)
    anetz_model = anetz_model_cls(baseline_model.config).eval()
    anetz_model.load_state_dict(baseline_model.state_dict())
    return anetz_model


def get_models_for_conversion(
    model_name: str, model_cls: ModelT
) -> tuple[PreTrainedModel, PreTrainedModel]:
    baseline_model = _get_baseline_model(model_name, model_cls)
    anetz_model = _init_anetz_model(baseline_model)
    return baseline_model, anetz_model


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
    """
    TODO: do all HuggingFace models reliably define `dummy_inputs`?
    """
    return [
        ct.TensorType(
            name,
            shape=tensor.shape,
            dtype=_pt_to_np_dtype(tensor.dtype),
        )
        for name, tensor in model.dummy_inputs.items()
    ]


def _set_metadata(mlmodel: ct.models.MLModel, model_name: str) -> None:
    hfinfo = model_info(model_name)
    mlmodel.license = hfinfo.cardData["license"]
    mlmodel.author = hfinfo.author or "<via Hugging Face>"
    mlmodel.version = hfinfo.sha
    mlmodel.short_description = (
        f"{hfinfo.modelId} re-implemented using anetz.{hfinfo.config['model_type']} "
        "for compatibility with execution on Neural Engine."
    )


def to_coreml_internal(
    baseline_model: PreTrainedModel,
    anetz_model: PreTrainedModel,
    out_path: str,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
) -> ct.models.MLModel:
    traced_optimized_model = torch.jit.trace(
        anetz_model,
        list(baseline_model.dummy_inputs.values()),
    )
    # TODO: for some models we might want to be able to set the `classifier_config` here
    # https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#classifierconfig
    mlmodel = ct.convert(
        traced_optimized_model,
        convert_to="mlprogram",
        inputs=_get_ct_inputs(baseline_model),
        compute_units=compute_units,
    )
    _set_metadata(mlmodel, baseline_model.name_or_path)
    mlmodel.save(out_path)
    assert isinstance(mlmodel, ct.models.MLModel)
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
    """
    baseline_model, anetz_model = get_models_for_conversion(model_name, model_cls)
    return to_coreml_internal(
        baseline_model=baseline_model,
        anetz_model=anetz_model,
        out_path=out_path,
        compute_units=compute_units,
    )
