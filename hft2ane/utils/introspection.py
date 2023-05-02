import warnings
from contextlib import contextmanager
from types import NoneType
from typing import Any, Iterator, get_type_hints, get_args, Type, TypeGuard

import torch.nn
from transformers import PreTrainedModel
from transformers.utils.fx import symbolic_trace
from transformers.modeling_outputs import ModelOutput

from hft2ane.mappings import get_hf_auto_model, get_output_for_auto_model


TensorT = (
    torch.Tensor
    | torch.DoubleTensor
    | torch.FloatTensor
    | torch.LongTensor
    | torch.IntTensor
    | torch.ShortTensor
    | torch.HalfTensor
    | torch.CharTensor
    | torch.ByteTensor
    | torch.BoolTensor
)


@contextmanager
def model_config(model: PreTrainedModel, **kwargs):
    """
    Temporarily set the config of a model.
    """
    old_config = model.config
    model.config = model.config_class(**{**old_config.to_dict(), **kwargs})
    yield
    model.config = old_config


def _get_by_trace(model: PreTrainedModel) -> tuple[list[str], list[str]]:
    """
    Get the output of a model by tracing it with a dummy input.
    This is useful for getting the output shape of a model, which is
    not available from the model's config.
    """
    if not model.config.return_dict:
        raise ValueError(
            "Model must be initialized with `return_dict=True` to return output names via symbolic_trace."
        )
    # NOTE: not all models can be symbolically traced, e.g. MosaicGPT fails
    traced = symbolic_trace(model)
    inputs = []
    outputs = []
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node.name)
        elif node.op == "output":
            outputs.extend(node.args[0].keys())
    return inputs, outputs


class ModelOutputNotFound(Exception):
    pass


class MultipleModelOutputsFound(Exception):
    pass


def _flatten_type_args(type_args: tuple[object, ...]) -> Iterator[object]:
    for t in type_args:
        if t_args := get_args(t):
            yield from _flatten_type_args(t_args)
        else:
            yield t


def _fields_from_model_output(
    model_output: Type[ModelOutput],
) -> tuple[dict[str, TensorT], dict[str, TensorT]]:
    required_fields = {}
    optional_fields = {}
    for f in model_output.__dataclass_fields__.values():
        if NoneType in get_args(f.type):
            optional_fields[f.name] = f.type
        else:
            required_fields[f.name] = f.type
    return required_fields, optional_fields


def _is_model_output(t: Any) -> TypeGuard[ModelOutput]:
    try:
        return issubclass(t, ModelOutput)
    except TypeError:
        return False


def _get_by_model_outputs(model: PreTrainedModel) -> tuple[list[str], list[str]]:
    """
    Get the output of a model by inspecting the annotations of the forward method.
    This is useful for getting the output shape of a model, which is not available
    from the model's config.
    (For inputs we can use model.dummy_inputs or model.main_input_name)
    """
    model_output = None
    try:
        RetT = get_type_hints(model.forward)["return"]
    except KeyError:
        # no return type annotation
        pass
    else:
        type_args = _flatten_type_args(get_args(RetT))
        model_outputs = list(filter(_is_model_output, type_args))
        if not model_outputs:
            warnings.warn(
                "No ModelOutput found in model.forward return type annotations. "
                "Falling back to inference from AutoModel."
            )
        if len(model_outputs) > 1:
            warnings.warn(
                f"Multiple ModelOutputs found in model.forward return type annotations: {model_outputs}. "
                "Falling back to inference from AutoModel."
            )
        model_output = model_outputs[0]
    if model_output is None:
        # infer from AutoModel type
        auto_model = get_hf_auto_model(model)
        model_output = get_output_for_auto_model(auto_model)
    required_fields, optional_fields = _fields_from_model_output(model_output)
    # TODO: we'll only consider the required fields for now
    return list(model.dummy_inputs.keys()), list(required_fields.keys())


def get_input_and_output_key_names(
    model: PreTrainedModel,
) -> tuple[list[str], list[str]]:
    """
    See coremltools.converters._converters_entry.convert and _validate_outputs_argument
    it seems that we can pass a list of strings to `outputs` to specify the output names
    (or else a list of coremltools.converters.mil.input_types.TensorType(name=name))

    NOTE:
    All the ane_transformer models override the HF base model output to NOT return a
    dict (or even able to). Presumably returning a dict is not useful for ct.convert.
    So the `model` passed here should be the original HF model, not the ane_transformer.

    TODO:
    Can we make use of this https://github.com/huggingface/exporters ?
    It does some nice things e.g. already applied Softmax to the classifier outputs.
    """
    with model_config(model, return_dict=True):
        try:
            return _get_by_trace(model)
        except Exception as e:
            warnings.warn(
                f"Failed to get output names via tracing: {e}. "
                "Falling back to using model config."
            )
    try:
        return _get_by_model_outputs(model)
    except Exception as e:
        # TODO: probably we should just use this in all cases as it is guaranteed to be
        # accurate. Slow for large models, but we already had to instantiate the model.
        warnings.warn(
            f"Failed to get output names via model outputs: {e}. "
            "Falling back to running model."
        )
        with model_config(model, return_dict=True):
            result = model(**model.dummy_inputs)
            return list(model.dummy_inputs.keys()), list(result.keys())
