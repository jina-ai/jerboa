from transformers import LlamaConfig

# Configuration for a Llama model with minimal footprint
low_footprint_config = LlamaConfig(
    hidden_size=4096,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=1,
)

low_footprint_general = {
    'intermediate_size': 64,
    'num_hidden_layers': 1,
    'num_attention_heads': 1,
}
