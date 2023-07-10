import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from typer import Typer
from jerboa.utils.load_model import load_model

app = Typer(pretty_exceptions_enable=False)

@app.command()
def save_full_model(
    base_model: str = "tiiuae/falcon-7b",
    load_in_4bit: bool = False,
    load_in_8bit: bool = True,
    lora_weights: str = "jina-ai/jerboa/lora_weight:v12",
):

    model = load_model(
        base_model,
        lora_weights,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    model.base_model.save_pretrained("full_model")


if __name__ == "__main__":
    app()
