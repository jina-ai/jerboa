import gradio as gr
from jerboa.utils.prompter import Prompter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    AutoConfig,
)
from peft import PeftConfig, PeftModel
import torch


# Load model
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
)

BASE_MODEL = 'tiiuae/falcon-40b'
LORA_WEIGHTS = '../artifacts/falcon40b_code_alpaca'
MODEL_CONFIG = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=BASE_MODEL,
    torch_dtype=torch.float16,
    device_map='auto',
    config=MODEL_CONFIG,
    # load_in_8bit=True,
    quantization_config=QUANT_CONFIG,
    trust_remote_code=True,
)

PEFT_CONFIG = PeftConfig.from_pretrained(
    LORA_WEIGHTS,
    trust_remote=True,
)

model = PeftModel.from_pretrained(
    model,
    LORA_WEIGHTS,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    PEFT_CONFIG.base_model_name_or_path,
    trust_remote_code=True,
    padding_side='left',
)
tokenizer.pad_token = tokenizer.eos_token
prompter = Prompter('')


def evaluate(
    instruction,
    max_new_tokens,
    input=None,
    temperature=0.1,
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

if __name__ == "__main__":
    demo.launch(share=True)
