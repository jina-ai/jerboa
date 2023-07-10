import json
from typing import List, Tuple, Dict, Union, Any
from pathlib import Path
import os
from jerboa.utils.prompter import Prompter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedTokenizer,
)
from peft import PeftConfig
from typer import Typer

# Define constants
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
)


def create_configuration(tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    return {
        'max_new_tokens': 512,
        'do_sample': True,
        'top_k': 50,
        'temperature': 0.3,
        'num_return_sequences': 1,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.eos_token_id,
        'repetition_penalty': 3.0,
        'length_penalty': -100,
    }


def load_model_and_tokenizer(
    model_repo: str = 'jinaai/falcon-7b',
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from Huggingface model repository

    Args:
        model_repo: Huggingface ID of repository with LORA weights, FUNCTION DOES NOT LOAD FULL MODEL

    Returns:
        model: Huggingface model in eval mode
        tokenizer: Huggingface tokenizer corresponding to model
    """

    PEFT_CONFIG = PeftConfig.from_pretrained(
        model_repo,
        trust_remote=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        PEFT_CONFIG.base_model_name_or_path,
        trust_remote_code=True,
        quantization_config=QUANT_CONFIG,
    )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        PEFT_CONFIG.base_model_name_or_path,
        trust_remote_code=True,
        padding_side='left',
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_data(eval_file: Union[str, os.PathLike] = '') -> List[Dict[str, str]]:
    assert eval_file, "Please provide a path to the evaluation data"
    eval_data = []
    with open(eval_file, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))

    return eval_data


app = Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    model: str = 'jinaai/falcon-7b',
    eval_file: str = "eval.jsonl",
    output_file: str = "output.jsonl",
):
    EVAL_PATH = Path(eval_file)
    results = []

    eval_data = load_data(eval_file=str(EVAL_PATH))
    model, tokenizer = load_model_and_tokenizer(model_repo=model)
    prompter = Prompter('alpaca')

    # Variable to select instances from eval_data based on index
    targets = list(range(len(eval_data)))
    # targets = np.random.randint(0, 250, 5)

    # Loop over evaluation data
    for i, eval_instance in enumerate(map(eval_data.__getitem__, targets)):
        print(f"Instance {i} of {len(eval_data)}")
        prompt = prompter.generate_prompt(
            eval_instance['instruction'],
            eval_instance['instances'][0]['input'],
        )
        tokenized_prompt = tokenizer(
            prompt,
            padding=True,
            return_tensors='pt',
        ).to(0)

        GENERATION_CONFIG = create_configuration(tokenizer)
        output = model.generate(
            input_ids=tokenized_prompt['input_ids'],
            attention_mask=tokenized_prompt['attention_mask'],
            generation_config=GenerationConfig(**GENERATION_CONFIG),
        )

        result = tokenizer.decode(
            output[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        print("Result:", result.split("### Response:")[1])
        output = result.split("### Response:")[1]

        results.append(
            {
                "id": i,
                "instruction": eval_instance["instruction"],
                "input": eval_instance["instances"][0]["input"],
                "alpaca_lora_output": eval_instance["instances"][0][
                    "stanford_alpaca_output"
                ],
                "falcon_no_config": eval_instance["instances"][0][
                    "falcon_output"
                ].split("### Response:")[1],
                "falcon_with_config": output,
                "human_evaluation": eval_instance["instances"][0]["output"],
            }
        )

    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    app()
