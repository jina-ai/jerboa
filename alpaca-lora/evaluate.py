import json

import fire
import torch
from transformers import GenerationConfig, PreTrainedTokenizer
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def evaluate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt_template: str = "",
    eval_file: str = "",
    eval_limit: int = 0,
):
    prompter = Prompter(prompt_template)

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
    for i, eval_instance in enumerate(eval_data[0]):
        output = list(
            evaluate(
                eval_instance["instruction"], eval_instance["input"]
            )
        )[0]
        if eval_limit != 0 and i == eval_limit:
            break
        results.append(
            {
                "id": i,
                "instruction": eval_instance["instruction"],
                "input": eval_instance["input"],
                "output": output,
                "to_compare": eval_instance["output"],
            }
        )
    return results


if __name__ == "__main__":
    fire.Fire(evaluate)
