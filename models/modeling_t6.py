import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from models.configuration_t6 import T6Config

logger = logging.getLogger(__name__)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        # 预先计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).to(device) / dim))
        # 注册为 buffer，确保在 model.to(device) 时跟随转移，并避免 meta tensor 问题
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().to(dtype=x.dtype)
            self.sin_cached = freqs.sin().to(dtype=x.dtype)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CPLinear(nn.Module):
    # Bilinear form of x using CP decomposition
    def __init__(self, config: T6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_head = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rank = config.rank
        self.q_rank = config.q_rank
        self.attention_bias = config.attention_bias

        # Define linear transformations for A projections
        self.W_A_q = nn.Linear(
            self.hidden_size,
            self.n_head * self.q_rank,
            bias=self.attention_bias,
        )
        self.W_A_k = nn.Linear(
            self.hidden_size,
            self.n_head * self.rank,
            bias=self.attention_bias,
        )
        self.W_A_v = nn.Linear(
            self.hidden_size,
            self.n_head * self.rank,
            bias=self.attention_bias,
        )

        # Define B projection parameters for Q, K, V
        self.W_B_q = nn.Linear(
            self.hidden_size,
            self.q_rank * self.head_dim,
            bias=self.attention_bias,
        )
        self.W_B_k = nn.Linear(
            self.hidden_size,
            self.rank * self.head_dim,
            bias=self.attention_bias,
        )
        self.W_B_v = nn.Linear(
            self.hidden_size,
            self.rank * self.head_dim,
            bias=self.attention_bias,
        )
        self.rotary = Rotary(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self):
        W_A_q_tensor = self.W_A_q.weight.view(
            self.hidden_size, self.n_head, self.q_rank
        )
        W_A_k_tensor = self.W_A_k.weight.view(self.hidden_size, self.n_head, self.rank)
        W_A_v_tensor = self.W_A_v.weight.view(self.hidden_size, self.n_head, self.rank)
        nn.init.xavier_uniform_(W_A_q_tensor)
        nn.init.xavier_uniform_(W_A_k_tensor)
        nn.init.xavier_uniform_(W_A_v_tensor)
        self.W_A_q.weight.data = W_A_q_tensor.view_as(self.W_A_q.weight)
        self.W_A_k.weight.data = W_A_k_tensor.view_as(self.W_A_k.weight)
        self.W_A_v.weight.data = W_A_v_tensor.view_as(self.W_A_v.weight)

        W_B_q_tensor = self.W_B_q.weight.view(
            self.hidden_size, self.q_rank, self.head_dim
        )
        W_B_k_tensor = self.W_B_k.weight.view(
            self.hidden_size, self.rank, self.head_dim
        )
        W_B_v_tensor = self.W_B_v.weight.view(
            self.hidden_size, self.rank, self.head_dim
        )
        nn.init.xavier_uniform_(W_B_q_tensor)
        nn.init.xavier_uniform_(W_B_k_tensor)
        nn.init.xavier_uniform_(W_B_v_tensor)
        self.W_B_q.weight.data = W_B_q_tensor.view_as(self.W_B_q.weight)
        self.W_B_k.weight.data = W_B_k_tensor.view_as(self.W_B_k.weight)
        self.W_B_v.weight.data = W_B_v_tensor.view_as(self.W_B_v.weight)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Compute intermediate variables A for Q, K, and V
        A_q = self.W_A_q(x).view(batch_size, seq_len, self.n_head, self.q_rank)
        A_k = self.W_A_k(x).view(batch_size, seq_len, self.n_head, self.rank)
        A_v = self.W_A_v(x).view(batch_size, seq_len, self.n_head, self.rank)

        # Compute intermediate variables B for Q, K, and V
        B_q = self.W_B_q(x).view(batch_size, seq_len, self.q_rank, self.head_dim)
        B_k = self.W_B_k(x).view(batch_size, seq_len, self.rank, self.head_dim)
        B_v = self.W_B_v(x).view(batch_size, seq_len, self.rank, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary(B_q)
        B_q, B_k = apply_rotary_emb(B_q, cos, sin), apply_rotary_emb(B_k, cos, sin)

        # Reshape A_q, A_k, A_v
        A_q = A_q.view(batch_size * seq_len, self.n_head, self.q_rank)
        A_k = A_k.view(batch_size * seq_len, self.n_head, self.rank)
        A_v = A_v.view(batch_size * seq_len, self.n_head, self.rank)

        # Reshape B_k, B_v
        B_q = B_q.view(batch_size * seq_len, self.q_rank, self.head_dim)
        B_k = B_k.view(batch_size * seq_len, self.rank, self.head_dim)
        B_v = B_v.view(batch_size * seq_len, self.rank, self.head_dim)

        q = (
            torch.bmm(A_q, B_q)
            .div_(self.q_rank)
            .view(batch_size, seq_len, self.n_head, self.head_dim)
        )
        k = (
            torch.bmm(A_k, B_k)
            .div_(self.rank)
            .view(batch_size, seq_len, self.n_head, self.head_dim)
        )
        v = (
            torch.bmm(A_v, B_v)
            .div_(self.rank)
            .view(batch_size, seq_len, self.n_head, self.head_dim)
        )

        return q, k, v


class CausalSelfAttention(nn.Module):
    def __init__(self, config: T6Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.num_attention_heads
        self.head_dim = config.head_dim

        # CPLinear projections directly output multi-head dimensions
        self.c_qkv = CPLinear(config)

        # Output projection from (n_head * head_dim) back to n_embd
        self.c_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.c_proj.weight.data.zero_()

        # Add group norm
        self.using_groupnorm = config.using_groupnorm

    def forward(self, x):
        B, T, C = x.size()  # (batch_size, seq_length, n_embd)

        # Project inputs to queries, keys, and values directly with multi-head shape
        q, k, v = self.c_qkv(x)  # Each has shape (B, T, n_head, head_dim)

        # Scaled dot-product attention
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # (B, n_head, T, head_dim)
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )

        if self.using_groupnorm:
            # Apply RMSNorm directly to each head's output
            # y = self.subln(y)
            y = F.rms_norm(y, (self.head_dim,), eps=self.config.rms_norm_eps)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: T6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Split the linear projection into two parts for SwiGLU
        self.c_fc1 = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.c_fc2 = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )

        # Output projection
        self.c_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        # Apply the first linear layer to produce two projections
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)

        # Apply the SwiGLU gating: SILU on one projection, and gate with the other
        x = self.act_fn(x1) * x2

        # Apply the final output projection
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: T6Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.rms_norm_eps = config.rms_norm_eps

        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x):
        norm1 = F.rms_norm(x, (x.size(-1),), eps=self.rms_norm_eps)
        attn_out = self.attn(norm1)
        x = x + attn_out

        norm2 = F.rms_norm(x, (x.size(-1),), eps=self.rms_norm_eps)
        mlp_out = self.mlp(norm2)
        x = x + mlp_out
        return x


class T6PreTrainedModel(PreTrainedModel):
    config_class = T6Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class T6Model(T6PreTrainedModel):
    def __init__(self, config: T6Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [Block(config, i) for i in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_logits: bool = True,
        output_all_seq: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # 打印每一层的输出信息
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)

        hidden_states = F.rms_norm(
            hidden_states, (self.config.hidden_size,), eps=self.config.rms_norm_eps
        )

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            # past_key_values=past_key_values if use_cache else None,
            # hidden_states=all_hidden_states,
            # attentions=all_self_attns,
        )
        return output

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        # return n_params
        return n_params

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        super().save_pretrained(save_directory, safe_serialization=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        model = super().from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )
        return model


class T6ForCausalLM(T6PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = T6Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_logits: bool = True,
        output_all_seq: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            # position_ids=position_ids,
            # past_key_values=past_key_values,
            # inputs_embeds=inputs_embeds,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            # cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]

        # logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
