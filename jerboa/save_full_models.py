import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from typer import Typer

app = Typer(pretty_exceptions_enable=False)


@app.command()
def save_full_model(
    base_model: str = "tiiuae/falcon-7b",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = True,
    lora_weights: str = "artifacts/lora_weight:v12/",
):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        config=model_config,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    model.base_model.save_pretrained("full_model")


if __name__ == "__main__":
    app()
