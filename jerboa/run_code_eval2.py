import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftConfig
from jerboa.utils.prompter import Prompter
import numpy as np


# Define constants
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
)

PEFT_MODEL_ID = "jinaai/falcon-7b"
EVAL_FILE = "eval.jsonl"

PEFT_CONFIG = PeftConfig.from_pretrained(
    PEFT_MODEL_ID,
    trust_remote=True,
)

model = AutoModelForCausalLM.from_pretrained(
    PEFT_CONFIG.base_model_name_or_path,
    trust_remote_code=True,
    quantization_config=QUANT_CONFIG,
)

tokenizer = AutoTokenizer.from_pretrained(
    PEFT_CONFIG.base_model_name_or_path,
    trust_remote_code=True,
    padding_side='left',
)
tokenizer.pad_token = tokenizer.eos_token

model.eval()
prompter = Prompter('alpaca')

GENERATION_DICTIONARY = {
    'max_length': 1024,
    'do_sample': True,
    'top_k': 50,
    'temperature': 0.3,
    'num_return_sequences': 1,
    'eos_token_id': tokenizer.eos_token_id,
    'pad_token_id': tokenizer.eos_token_id,
    'repetition_penalty': 3.0,
    'length_penalty': -100,
}


def load_data():
    eval_data = []
    with open(EVAL_FILE, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))

    return eval_data


def main():
    results = []
    targets = list(range(45))

    eval_data = load_data()
    for eval_instance in map(eval_data.__getitem__, targets):
        prompt = (
            prompter.generate_prompt(
                eval_instance['instruction'],
                eval_instance['instances'][0]['input'],
            )
        )
        tokenized_prompt = tokenizer(
            prompt,
            padding=True,
            return_tensors='pt',
        ).to(0)

        output = model.generate(
            input_ids=tokenized_prompt['input_ids'],
            attention_mask=tokenized_prompt['attention_mask'],
            generation_config=GenerationConfig(**GENERATION_DICTIONARY)
        )

        result = tokenizer.decode(
            output[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print("Result:", result.split("### Response:")[1])



main()
