import re

from ane_transformers.huggingface import distilbert
from ane_transformers.huggingface.distilbert import (
    DistilBertForMaskedLM,  # noqa: F401
    DistilBertForSequenceClassification,  # noqa: F401
    DistilBertForQuestionAnswering,  # noqa: F401
    DistilBertForTokenClassification,  # noqa: F401
    DistilBertForMultipleChoice,  # noqa: F401
    Embeddings,
    Transformer,
)
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertModel as _DistilBertModel,
)

MODEL_TYPE = "distilbert"


_INTERNAL_PROJ_RE = re.compile(r".*lin.*\.weight")
_OUTPUT_PROJ_RE = re.compile(
    r".*({}).*\.weight".format(
        "|".join(
            [
                "classifier",
                "pre_classifier",
                "vocab_transform",
                "vocab_projector",
                "qa_outputs",
            ]
        )
    )
)


def linear_to_conv2d_map(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """
    Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights

    (This is a fixed version of the same function in `ane_transformers`
    which didn't match all the Conv2d layers from all the model classes)
    """
    for k in state_dict:
        is_internal_proj = _INTERNAL_PROJ_RE.match(k)
        is_output_proj = _OUTPUT_PROJ_RE.match(k)
        if is_internal_proj or is_output_proj:
            if len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]


"""
We have to monkeypatch the `ane_transformers` models to use the fixed version
of the `linear_to_conv2d_map` function above.
"""


class DistilBertModel(_DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        setattr(self, "embeddings", Embeddings(config))
        setattr(self, "transformer", Transformer(config))

        # Register hook for unsqueezing nn.Linear parameters to match nn.Conv2d parameter spec
        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError


distilbert.DistilBertModel = DistilBertModel
