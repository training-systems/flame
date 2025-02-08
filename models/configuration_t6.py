from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig


@dataclass
class T6Config(PretrainedConfig):
    model_type = "t6"

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 768,  # Fixed embedding dimension
        num_hidden_layers: int = 12,
        n_head: int = 22,  # Number of attention heads
        head_dim: int = 64,  # Dimension per head
        rank: int = 2,  # CP rank for key and value
        q_rank: int = 6,  # CP rank for query
        block_size: int = 1024,  # Maximum sequence length
        bias: bool = False,  # Use bias in all linear layers
        dropout: float = 0.0,  # Dropout rate
        scale_attn_by_inverse_layer_idx: bool = (
            False  # Scale attention by 1/sqrt(layer_idx)
        ),
        using_groupnorm: bool = False,  # Whether to use Group Layernorm
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.n_head = n_head
        self.head_dim = head_dim
        self.rank = rank
        self.q_rank = q_rank
        self.block_size = block_size
        self.bias = bias
        self.dropout = dropout
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.using_groupnorm = using_groupnorm
        super().__init__(**kwargs)
