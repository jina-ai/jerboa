from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from core import run_eval, filter_code, fix_indents
import os
import torch
from jerboa.utils.load_model import load_model

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, batch_size: int
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(
        input_batch, return_tensors="pt", return_token_type_ids=False
    ).to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    # fix_indents is required to fix the tab character that is generated from starcoder model
    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    out_path = "results/falcon7b_code_alpaca/eval.jsonl"
    os.makedirs("results/falcon7b_code_alpaca", exist_ok=True)

    model = torch.compile(
        load_model(
            "tiiuae/falcon-7b",
            lora_dir="jinaai/falcon-7b-lora",
            load_in_8bit=True,
            device_map="auto",
        ).eval()
    )

    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", use_fast=False)

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )
