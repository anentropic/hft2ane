from typing import Type
from warnings import warn

import torch
import coremltools as ct
import numpy as np
from huggingface_hub import model_info
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import _BaseAutoModelClass

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


ModelT = Type[PreTrainedModel] | Type[_BaseAutoModelClass]

METADATA_MODEL_NAME_KEY = "com.github.anentropic.hft2ane.model"
METADATA_MODEL_CLS_KEY = "com.github.anentropic.hft2ane.type"


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


def _get_classifier_config(id2label: dict[str, int | str]) -> ct.ClassifierConfig:
    """
    TODO: we can't use this because coremltools complains about the shape of outputs
    from unaltered HF models. HF `exporters` gets around this by adding a Wrapper
    module to the model which adds pre and post processing, akin to a `pipeline`.
    https://github.com/huggingface/exporters/blob/7f82edfcda2fe39790f93ba5a9500866709fc71b/src/exporters/coreml/convert.py#L292

    I think solution for us will be to piggy-back on `exporters` completely.
    """
    # we could probably just assume id2label keys are contiguous and start at 0
    # but the data structure does not guarantee this, so play it safe
    label_type = type(next(id2label.values().__iter__()))
    default = "" if label_type is str else 0
    labels = [default] * (max(id2label.keys()) + 1)
    for i, label in id2label.items():
        labels[i] = label
    # they only specify `class_labels``
    # https://github.com/huggingface/exporters/blob/7f82edfcda2fe39790f93ba5a9500866709fc71b/src/exporters/coreml/convert.py#L539
    return ct.ClassifierConfig(
        class_labels=labels,
    )


def _set_metadata(
    mlmodel: ct.models.MLModel, model_name: str, model_cls_name: str | None
) -> None:
    hfinfo = model_info(model_name)
    mlmodel.license = hfinfo.cardData["license"]
    mlmodel.author = hfinfo.author or "<via Hugging Face>"
    mlmodel.version = hfinfo.sha
    mlmodel.short_description = (
        f"{hfinfo.modelId} re-implemented using hft2ane "
        "for compatibility with execution on Neural Engine."
    )
    mlmodel.user_defined_metadata[METADATA_MODEL_NAME_KEY] = model_name
    if model_cls_name:
        mlmodel.user_defined_metadata[METADATA_MODEL_CLS_KEY] = model_cls_name


def to_coreml_internal(
    baseline_model: PreTrainedModel,
    hft2ane_model: PreTrainedModel,
    out_path: str,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
    model_cls_name: str | None = None,
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

    kwargs = {}
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
        is_temp_package=True,
        skip_model_load=True,
    ).save(out_path)

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
