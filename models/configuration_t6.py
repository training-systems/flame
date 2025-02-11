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
        num_attention_heads: int = 22,  # Number of attention heads
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rank: int = 2,  # CP rank for key and value
        q_rank: int = 6,  # CP rank for query
        block_size: int = 1024,  # Maximum sequence length
        attention_bias: bool = False,
        bias: bool = False,  # Use bias in all linear layers
        dropout: float = 0.0,  # Dropout rate
        scale_attn_by_inverse_layer_idx: bool = (
            False  # Scale attention by 1/sqrt(layer_idx)
        ),
        using_groupnorm: bool = False,  # Whether to use Group Layernorm
        mlp_bias: bool = False,
        head_dim: int = None,  # Dimension per head
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pretraining_tp = pretraining_tp
        self.tie_word_embeddings = tie_word_embeddings
        self.rank = rank
        self.q_rank = q_rank
        self.block_size = block_size
        self.attention_bias = attention_bias
        self.bias = bias
        self.dropout = dropout
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.using_groupnorm = using_groupnorm
        self.mlp_bias = mlp_bias
        self.head_dim = (
            head_dim
            if head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
