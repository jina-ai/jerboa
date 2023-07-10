from jerboa.utils.load_model import load_model
import pytest

@pytest.mark.slow
def test_load_huggingface_model():
    lora_dir = 'jinaai/falcon-7b'
    load_model(lora_dir=lora_dir)


@pytest.mark.slow
def test_load_wandb_model():
    lora_dir = 'jina-ai/jerboa/lora_weight:v19'
    load_model(lora_dir=lora_dir)

