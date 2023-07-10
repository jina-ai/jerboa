import tempfile
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)
import wandb
import torch


def load_model(
        base_model: str = 'tiiuae/falcon-7b',
        lora_dir: str = '',
        load_in_4bit: bool = False,
        load_in_8bit: bool = True,
        device_map: str = 'auto',
) -> AutoModelForCausalLM:
    """Load a model from a base model, optionally  include lora weights

    Args:
        base_model (str): The base model to load
        lora_dir (str): The lora weights to load
        load_in_4bit (bool): Whether to load in 4bit, ONLY ON GPU
        load_in_8bit (bool): Whether to load in 8bit, ONLY ON GPU
        device_map (str): The device map to use

    Returns:
        AutoModelForCausalLM: The loaded model
    """
    assert not (load_in_4bit and load_in_8bit), "Cannot load in both 4bit and 8bit"

    # Define constants
    QUANT_CONFIG = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    QUANT_CONFIG = QUANT_CONFIG if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=QUANT_CONFIG,
        trust_remote_code=True,
    )

    def load_peft_model(base_model: str, lora_dir: str) -> PeftModel:
        """Incorporate lora weights into a base model

        Args:
            base_model (str): The instantiated base model to be adapted
            lora_dir (str): The directory containing the lora weights and configuration

        Returns:
            PeftModel: The loaded model
        """

        return PeftModel.from_pretrained(
            model=base_model,
            model_id=lora_dir,
        )

    if lora_dir:
        # Check if the lora weights are from wandb or huggingface
        # Wandb sources have two forward slashes, huggingface sources have one
        from_wandb = len(lora_dir.split('/')) == 3

        if from_wandb:
            with tempfile.TemporaryDirectory() as tmpdir:
                api = wandb.Api()
                artifact = api.artifact(lora_dir)
                lora_dir = artifact.download(tmpdir)
                model = load_peft_model(base_model=model, lora_dir=lora_dir)
        else:
            model = load_peft_model(base_model=model, lora_dir=lora_dir)

    return model


if __name__ == '__main__':
    model = load_model(lora_dir='jinaai/falcon-7b')
    print(model)
    print("Done!")
