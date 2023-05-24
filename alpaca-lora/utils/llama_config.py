from transformers import LlamaConfig

low_footprint_config = LlamaConfig(
    hidden_size=4096,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=1,
)
