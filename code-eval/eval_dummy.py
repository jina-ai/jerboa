from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from core import run_eval, filter_code, fix_indents
import os
import torch
from core import run_eval, instruct_prompt



@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    prompt_input = instruct_prompt(prompt)
    input_batch = [prompt_input for _ in range(batch_size)]
    batch_completions = [prompt + "def add(x,y):\n x+y" for prompt in input_batch]
    breakpoint()
    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 1
    out_path = "results/dummy/eval.jsonl"
    os.makedirs("results/dummy", exist_ok=True)

    tokenizer = None
    model = None

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
    )
