import torch
import torch.nn as nn
from ane_transformers.reference.layer_norm import LayerNormANE
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert import modeling_bert

from hft2ane.models._common import (
    EPS,
    WARN_MSG_FOR_DICT_RETURN,
    WARN_MSG_FOR_TRAINING_ATTEMPT,
    ANEMixin,
    correct_for_bias_scale_order_inversion,
    last_conv2d_reshape,
)

MODEL_TYPE = "bert"


class ANEBertMixin(ANEMixin):
    _linear_to_conv2d_layers = [
        "query.weight",
        "key.weight",
        "value.weight",
        "dense.weight",
        "decoder.weight",
        "seq_relationship.weight",
        "classifier.weight",
        "qa_outputs.weight",
    ]


class LayerNormANE(LayerNormANE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)


class BertEmbeddings(modeling_bert.BertEmbeddings):
    """
    Embeddings module optimized for Apple Neural Engine
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.LayerNorm = LayerNormANE(config.hidden_size, eps=EPS)


class BertSelfAttention(modeling_bert.BertSelfAttention):
    def __init__(self, config: PretrainedConfig, is_causal=False, layer_idx=None):
        super().__init__(config, is_causal=is_causal, layer_idx=layer_idx)
        self.query = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=self.all_head_size, kernel_size=1
        )
        self.key = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=self.all_head_size, kernel_size=1
        )
        self.value = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=self.all_head_size, kernel_size=1
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters:
            hidden_states: torch.tensor(bs, dim, 1, seq_length)
            attention_mask: torch.tensor(bs, seq_length) or torch.tensor(bs, seq_length, 1, 1)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            dim, 1, seq_length) Contextualized layer. Optional: only if `output_attentions=True`
        """
        output_attentions = kwargs.get("output_attentions", False)

        # Parse tensor shapes for source and target sequences
        assert len(hidden_states.size()) == 4

        bs, dim, dummy, seqlen = hidden_states.size()

        # Project q, k and v
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Validate mask
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.logical_not().float() * -1e4
            elif attention_mask.dtype == torch.int64:
                attention_mask = (1 - attention_mask).float() * -1e4
            elif attention_mask.dtype != torch.float32:
                raise TypeError(f"Unexpected dtype for mask: {attention_mask.dtype}")

            if len(attention_mask.size()) == 2:
                # Raw 2D mask (bs, seqlen) → (bs, seqlen, 1, 1) for ANE attention
                attention_mask = attention_mask.unsqueeze(2).unsqueeze(2)

            expected_mask_shape = [bs, seqlen, 1, 1]
            if list(attention_mask.size()) != expected_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `mask` (Expected {expected_mask_shape}, got {list(attention_mask.size())}"
                )

        # Compute scaled dot-product attention
        dim_per_head = self.attention_head_size
        # Principle 2: Chunking Large Intermediate Tensors  (machinelearning.apple.com/research/apple-neural-engine)
        # - Split q, k and v to compute a list of single-head attention functions
        # Principle 3: Minimizing Memory Copies
        # - Avoid as many transposes and reshapes as possible
        mh_q = q.split(dim_per_head, dim=1)  # (bs, dim_per_head, 1, max_seq_length) * n_heads
        mh_k = k.transpose(1, 3).split(
            dim_per_head, dim=3
        )  # (bs, max_seq_length, 1, dim_per_head) * n_heads
        mh_v = v.split(dim_per_head, dim=1)  # (bs, dim_per_head, 1, max_seq_length) * n_heads

        normalize_factor = float(dim_per_head) ** -0.5
        attention_scores = [
            torch.einsum("bchq,bkhc->bkhq", [qi, ki]) * normalize_factor
            for qi, ki in zip(mh_q, mh_k)
        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads

        if attention_mask is not None:
            for head_idx in range(self.num_attention_heads):
                attention_scores[head_idx] = attention_scores[head_idx] + attention_mask

        attention_probs = [
            aw.softmax(dim=1) for aw in attention_scores
        ]  # (bs, max_seq_length, 1, max_seq_length) * n_heads
        context_layer = [
            torch.einsum("bkhq,bchk->bchq", wi, vi) for wi, vi in zip(attention_probs, mh_v)
        ]  # (bs, dim_per_head, 1, max_seq_length) * n_heads

        context_layer = torch.cat(context_layer, dim=1)  # (bs, dim, 1, max_seq_length)

        attn_weights = attention_probs if output_attentions else None

        return context_layer, attn_weights


class BertSelfOutput(modeling_bert.BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=1
        )
        self.LayerNorm = LayerNormANE(config.hidden_size, eps=EPS)


class BertAttention(modeling_bert.BertAttention):
    def __init__(
        self, config: PretrainedConfig, is_causal=False, layer_idx=None, is_cross_attention=False
    ):
        super().__init__(
            config, is_causal=is_causal, layer_idx=layer_idx, is_cross_attention=is_cross_attention
        )
        self.self = BertSelfAttention(config, is_causal=is_causal, layer_idx=layer_idx)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        attention_output, attn_weights = self.self(
            hidden_states, attention_mask=attention_mask, **kwargs
        )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attn_weights

    def prune_heads(self, heads):
        raise NotImplementedError


class BertIntermediate(modeling_bert.BertIntermediate):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dense = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=config.intermediate_size, kernel_size=1
        )


class BertOutput(modeling_bert.BertOutput):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dense = nn.Conv2d(
            in_channels=config.intermediate_size, out_channels=config.hidden_size, kernel_size=1
        )
        self.LayerNorm = LayerNormANE(config.hidden_size, eps=EPS)


class BertLayer(modeling_bert.BertLayer):
    def __init__(self, config: PretrainedConfig, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attention = BertAttention(config, layer_idx=layer_idx)
        if hasattr(self, "crossattention"):
            self.crossattention = BertAttention(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


class BertEncoder(modeling_bert.BertEncoder):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [BertLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )


class BertPooler(modeling_bert.BertPooler):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dense = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=1
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hft2ane:
        first_token_tensor = hidden_states[:, :, :, 0].unsqueeze(-1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(modeling_bert.BertPredictionHeadTransform):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dense = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=1
        )
        self.LayerNorm = LayerNormANE(config.hidden_size, eps=EPS)


class BertLMPredictionHead(modeling_bert.BertLMPredictionHead):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.vocab_size,
            kernel_size=1,
            bias=False,
        )
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = super().forward(hidden_states)
        hidden_states = last_conv2d_reshape(hidden_states)
        return hidden_states


class BertOnlyMLMHead(modeling_bert.BertOnlyMLMHead):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.predictions = BertLMPredictionHead(config)


class BertOnlyNSPHead(modeling_bert.BertOnlyNSPHead):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.seq_relationship = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=2, kernel_size=1
        )

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        # hft2ane:
        seq_relationship_score = seq_relationship_score.squeeze(-1).squeeze(-1)
        return seq_relationship_score


class BertModel(ANEBertMixin, modeling_bert.BertModel):
    def __init__(self, config: PretrainedConfig, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        if self.pooler:
            self.pooler = BertPooler(config)

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Embeddings output is 4D (bs, dim, 1, seq_len) from LayerNormANE
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        # Skip parent's _create_attention_masks / create_bidirectional_mask which
        # expects 3D embeddings (accesses shape[1] for seq_len) but ours are 4D.
        # Our SelfAttention handles 2D attention_mask conversion internally.
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            **kwargs,
        )

        if not isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs.to_tuple()

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class BertLMHeadModel(ANEBertMixin, modeling_bert.BertLMHeadModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)


class BertForMaskedLM(ANEBertMixin, modeling_bert.BertForMaskedLM):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)


class BertForNextSentencePrediction(ANEBertMixin, modeling_bert.BertForNextSentencePrediction):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)


class BertForSequenceClassification(ANEBertMixin, modeling_bert.BertForSequenceClassification):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=config.num_labels, kernel_size=1
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, ...]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None or self.training:
            raise NotImplementedError(WARN_MSG_FOR_TRAINING_ATTEMPT)

        if return_dict:
            raise ValueError(WARN_MSG_FOR_DICT_RETURN)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.squeeze(-1).squeeze(-1)  # (bs, num_labels)

        output = (logits,) + outputs[2:]
        return output


class BertForMultipleChoice(ANEBertMixin, modeling_bert.BertForMultipleChoice):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Conv2d(in_channels=config.hidden_size, out_channels=1, kernel_size=1)


class BertForTokenClassification(ANEBertMixin, modeling_bert.BertForTokenClassification):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=config.num_labels, kernel_size=1
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, ...]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None or self.training:
            raise NotImplementedError(WARN_MSG_FOR_TRAINING_ATTEMPT)

        if return_dict:
            raise ValueError(WARN_MSG_FOR_DICT_RETURN)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = last_conv2d_reshape(logits)

        output = (logits,) + outputs[2:]
        return output


class BertForQuestionAnswering(ANEBertMixin, modeling_bert.BertForQuestionAnswering):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Conv2d(
            in_channels=config.hidden_size, out_channels=config.num_labels, kernel_size=1
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, ...]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if start_positions is not None or end_positions is not None or self.training:
            raise NotImplementedError(WARN_MSG_FOR_TRAINING_ATTEMPT)

        if return_dict:
            raise ValueError(WARN_MSG_FOR_DICT_RETURN)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        # hft2ane:
        logits = last_conv2d_reshape(logits)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        output = (start_logits, end_logits) + outputs[2:]
        return output
