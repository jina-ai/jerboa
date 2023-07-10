import tempfile
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import wandb


def load_model(
    base_model: str = 'tiiuae/falcon-7b',
    lora_dir: str = '',
) -> AutoModelForCausalLM:
    """Load a model from a base model, optionally lora weights

    Args:
        base_model (str): The base model to load
        lora_weights (str): The lora weights to load

    Returns:
        AutoModelForCausalLM: The loaded model
    """

    # Define constants
    QUANT_CONFIG = BitsAndBytesConfig(
        load_in_8bit=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=QUANT_CONFIG,
    )

    def load_peft_model(model: str, lora_dir: str) -> PeftModel:
        """Load a model with lora weights

        Args:
            model (str): The instantiated base model to be adapted
            lora_dir (str): The directory containing the lora weights and configuration

        Returns:
            PeftModel: The loaded model
        """

        return PeftModel.from_pretrained(
            model=model,
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
                model = load_peft_model(model=model, lora_dir=lora_dir)
        else:
            model = load_peft_model(model=model, lora_dir=lora_dir)

    return model

if __name__ == '__main__':
    model = load_model(lora_dir='jina-ai/jerboa/lora_weight:v18')
    print(model)
    print("Done!")
