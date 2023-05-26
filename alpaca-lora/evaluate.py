import json
import sys

import fire
from typing import Optional
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer
from utils.llama_config import low_footprint_config
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def evaluate(
    load_8bit: bool = False,
    base_model: str = "huggyllama/llama-7b",
    lora_weights: str = "tloen/alpaca-lora-7b",
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    prompt_template: str = "",
    eval_file: str = "",
    eval_limit: int = 0,
    debug: bool = False,
):
    assert (
        base_model or model
    ), "Please specify a --base_model or model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    if not tokenizer:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # Default configurations
    llama_args = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "device_map": {"": device},
    }
    peft_args = {
        "model_id": lora_weights,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "device_map": {"": device},
    }

    # Conditional configurations
    if device == "cuda":
        llama_args.update({"load_in_8bit": load_8bit, "device_map": "auto"})
    elif device == "mps":
        pass  # No changes needed
    else:
        llama_args["low_cpu_mem_usage"] = True


    # Instantiate models
    if not model:
        if debug or base_model == 'debug_llama':
            # Debugging configuration for the Llama model, reduces parameters
            # If a gpu is available the model will run on the gpu, otherwise cpu
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            llama_config = low_footprint_config
            model = LlamaForCausalLM(llama_config).to(device)
        else:
            model = LlamaForCausalLM.from_pretrained(base_model, **llama_args)

        model = PeftModel.from_pretrained(model, **peft_args)

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit and device == "cuda":
            model.half()  # seems to fix bugs for some users.

    model.eval()

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
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
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    eval_data = []
    with open(eval_file, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))

    results = []
    for i, eval_instance in enumerate(eval_data):
        output = list(
            evaluate(
                eval_instance["instruction"], eval_instance["instances"][0]["input"]
            )
        )[0]
        if eval_limit != 0 and i == eval_limit:
            break
        results.append(
            {
                "id": i,
                "instruction": eval_instance["instruction"],
                "input": eval_instance["instances"][0]["input"],
                "output": output,
                "to_compare": eval_instance["instances"][0]["output"],
            }
        )
    return results


if __name__ == "__main__":
    fire.Fire(evaluate)
