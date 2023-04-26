import warnings

import torch
import torch.nn as nn


WARN_MSG_FOR_TRAINING_ATTEMPT = (
    "This model is optimized for on-device execution only. "
    "Please use the original implementation from Hugging Face for training"
)

WARN_MSG_FOR_DICT_RETURN = (
    "coremltools does not support dict outputs. Please set return_dict=False"
)

# Note: Original implementation of distilbert uses an epsilon value of 1e-12
# which is not friendly with the float16 precision that ANE uses by default
EPS = 1e-7


# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    state_dict[prefix + "bias"] = (
        state_dict[prefix + "bias"] / state_dict[prefix + "weight"]
    )
    return state_dict


def last_conv2d_reshape(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.squeeze(2).permute(0, 2, 1)


class ANEMixin:
    # suffixes of the state_dict keys that should be unsqueezed
    # due to converting nn.Linear to nn.Conv2d for ANE compatibility
    # e.g ["classifier.weight"]
    _linear_to_conv2d_layers: list[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Register hook for unsqueezing nn.Linear parameters to match nn.Conv2d parameter spec
        self._register_load_state_dict_pre_hook(self.linear_to_conv2d_map)

    def linear_to_conv2d_map(self, state_dict, *args, **kwargs):
        """
        Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights

        (This is a fixed version of the same function in `ane_transformers`
        which didn't match all the Conv2d layers from all the model classes)
        """
        for k in state_dict:
            if any(k.endswith(layer) for layer in self._linear_to_conv2d_layers):
                if len(state_dict[k].shape) == 2:
                    state_dict[k] = state_dict[k][:, :, None, None]

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """
        Tie or clone module weights depending of whether we are using TorchScript or not
        """
        # hft2ane: added this logic to unsqueeze input where the output is a Conv2d
        if input_embeddings.weight.shape != output_embeddings.weight.shape:
            if (
                len(input_embeddings.weight.shape) == 2
                and len(output_embeddings.weight.shape) == 4
            ):
                normalised_input_weight = nn.Parameter(
                    input_embeddings.weight.unsqueeze(-1).unsqueeze(-1)
                )
            else:
                warnings.warn(
                    f"Input and output embeddings should have the same shape or be "
                    f"expandable to the same shape (e.g. ANE unsqueeze to 4D). "
                    f"But got the shapes of input: {input_embeddings.weight.shape} and "
                    f"output: {output_embeddings.weight.shape}"
                )
        else:
            normalised_input_weight = input_embeddings.weight

        # rest of logic as in huggingface...
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(normalised_input_weight.clone())
        else:
            output_embeddings.weight = normalised_input_weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(
            input_embeddings, "num_embeddings"
        ):
            output_embeddings.out_features = input_embeddings.num_embeddings
