import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from models.configuration_t6 import T6Config


@dataclass
class T6Output(ModelOutput):
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine=True,
        memory_efficient=False,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        # 预先计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
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
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
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
    def __init__(self, in_features, n_head, head_dim, rank: int = 1, q_rank: int = 12):
        super(CPLinear, self).__init__()
        self.in_features = in_features
        self.n_head = n_head
        self.head_dim = head_dim
        self.rank = rank
        self.q_rank = q_rank

        # Define linear transformations for A projections
        self.W_A_q = nn.Linear(in_features, n_head * q_rank, bias=False)
        self.W_A_k = nn.Linear(in_features, n_head * rank, bias=False)
        self.W_A_v = nn.Linear(in_features, n_head * rank, bias=False)

        # Define B projection parameters for Q, K, V
        self.W_B_q = nn.Linear(in_features, q_rank * head_dim, bias=False)
        self.W_B_k = nn.Linear(in_features, rank * head_dim, bias=False)
        self.W_B_v = nn.Linear(in_features, rank * head_dim, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self):
        W_A_q_tensor = self.W_A_q.weight.view(
            self.in_features, self.n_head, self.q_rank
        )
        W_A_k_tensor = self.W_A_k.weight.view(self.in_features, self.n_head, self.rank)
        W_A_v_tensor = self.W_A_v.weight.view(self.in_features, self.n_head, self.rank)
        nn.init.xavier_uniform_(W_A_q_tensor)
        nn.init.xavier_uniform_(W_A_k_tensor)
        nn.init.xavier_uniform_(W_A_v_tensor)
        self.W_A_q.weight.data = W_A_q_tensor.view_as(self.W_A_q.weight)
        self.W_A_k.weight.data = W_A_k_tensor.view_as(self.W_A_k.weight)
        self.W_A_v.weight.data = W_A_v_tensor.view_as(self.W_A_v.weight)

        W_B_q_tensor = self.W_B_q.weight.view(
            self.in_features, self.q_rank, self.head_dim
        )
        W_B_k_tensor = self.W_B_k.weight.view(
            self.in_features, self.rank, self.head_dim
        )
        W_B_v_tensor = self.W_B_v.weight.view(
            self.in_features, self.rank, self.head_dim
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
    def __init__(self, config: T6Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.hidden_size  # Fixed embedding dimension
        self.rank = config.rank
        self.q_rank = config.q_rank

        # CPLinear projections directly output multi-head dimensions
        self.c_qkv = CPLinear(
            self.n_embd, self.n_head, self.head_dim, self.rank, self.q_rank
        )

        # Output projection from (n_head * head_dim) back to n_embd
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

        # Add group norm
        self.using_groupnorm = getattr(config, "using_groupnorm", False)
        if self.using_groupnorm:
            # Apply RMSNorm to each head's output dimension
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

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
            y = self.subln(y)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: T6Config):
        super().__init__()
        # Calculate the floored hidden dimension size
        hidden_dim = math.floor(8 / 3 * config.hidden_size)

        # Split the linear projection into two parts for SwiGLU
        self.c_fc1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.hidden_size, hidden_dim, bias=False)

        # Output projection
        self.c_proj = nn.Linear(hidden_dim, config.hidden_size, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        # Apply the first linear layer to produce two projections
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)

        # Apply the SwiGLU gating: SILU on one projection, and gate with the other
        x = F.silu(x1) * x2

        # Apply the final output projection
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: T6Config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model


class T6(PreTrainedModel):
    config_class = T6Config
    base_model_prefix = "t6"
    supports_gradient_checkpointing = True

    def __init__(self, config: T6Config):
        super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.num_hidden_layers)]
                ),
            )
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

    def forward(
        self, input_ids, labels=None, return_logits=True, output_all_seq=False, **kwargs
    ) -> T6Output:
        # forward the GPT model itself
        x = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if labels is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1
            )
        elif output_all_seq:
            logits = self.lm_head(
                x[:, :, :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return T6Output(logits=logits, loss=loss)

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        # assert block_size <= self.config.block_size
        # self.config.block_size = block_size
        # self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # for block in self.transformer.h:
        #     block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
        pass

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.num_hidden_layers,
            cfg.n_head,
            cfg.hidden_size // cfg.n_head,
            cfg.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

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
