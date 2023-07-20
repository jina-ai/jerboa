from typer import Typer
from jerboa.utils.load_model import load_model

app = Typer(pretty_exceptions_enable=False)


@app.command()
def save_full_model(
    base_model: str = 'tiiuae/falcon-40b',
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    lora_weights: str = 'wandb:jina-ai/jerboa/lora_weight:v18',
):
    model = load_model(
        base_model,
        lora_weights,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    model = model.merge_and_unload()
    model.base_model.save_pretrained("full_model", push_to_hub=True, repo_id='jinaai/falcon-40b-code-alpaca')


if __name__ == "__main__":
    app()
