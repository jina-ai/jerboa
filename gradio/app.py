import gradio as gr
from jerboa.utils.prompter import Prompter
from jerboa.utils.load_model import load_model
from transformers import (
    AutoTokenizer,
    GenerationConfig,
)
from typer import Typer
import torch

app = Typer(pretty_exceptions_enable=False)


@app.command()
def launch_app(
    base_model: str = 'tiiuae/falcon-7b',
    lora_repo: str = 'wandb:jina-ai/jerboa/lora_weight:v19',
):
    model = load_model(base_model=base_model, lora_dir=lora_repo)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side='left',
    )
    tokenizer.pad_token = tokenizer.eos_token
    prompter = Prompter('')

    def evaluate(
        instruction,
        max_new_tokens,
        input=None,
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
    ):
        device = 'cuda'
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        return prompter.get_response(output)

    demo = gr.Interface(
        fn=evaluate,
        inputs=[gr.Textbox(lines=2, placeholder="Prompt here"), gr.Slider(5, 1024)],
        outputs="text",
    )

    demo.launch()


if __name__ == "__main__":
    app()
