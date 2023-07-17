from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel, GenerationConfig,
)
from core import run_eval, filter_code, fix_indents
import os
import torch
from jerboa.run_code_eval import create_configuration
from jerboa.utils.load_model import load_model

from core import run_eval, instruct_prompt



@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(
        input_batch, return_tensors="pt", return_token_type_ids=False
    ).to(model.device)
    prompt_input = instruct_prompt(prompt)
    input_batch = [prompt_input for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    GENERATION_CONFIG = create_configuration(tokenizer)
    generated_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        generation_config=GenerationConfig(**GENERATION_CONFIG),
    )


    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # breakpoint()
    # fix_indents is required to fix the tab character that is generated from starcoder model
    return batch_completions


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 1
    out_path = "results/falcon7b_pure/eval.jsonl"
    os.makedirs("results/falcon7b_pure", exist_ok=True)

    model = torch.compile(
        load_model(
            "tiiuae/falcon-7b",
            lora_dir="jinaai/falcon-7b-lora",
            load_in_8bit=True,
            device_map="auto",
        ).eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "tiiuae/falcon-7b",
        trust_remote_code=True,
        padding_side='left',
    )

    run_eval(
        model, tokenizer, num_samples_per_task, out_path, generate_batch_completion
    )
