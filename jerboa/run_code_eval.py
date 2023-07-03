import json

import torch
from peft import PeftModel, PeftConfig
from typer import Typer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from jerboa.utils.prompter import Prompter


app = Typer(pretty_exceptions_enable=False)
device = "cuda"
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

peft_model_id = "jinaai/falcon-7b"
config = PeftConfig.from_pretrained(
    peft_model_id,
    trust_remote=True,
)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True,
    quantization_config=quant_config,
)
model = PeftModel.from_pretrained(model, peft_model_id).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, trust_remote_code=True
)


@app.command()
def run_eval(eval_file: str = "eval.jsonl"):
    eval_data = []
    with open(eval_file, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))

    results = []
    targets = list(range(3))
    # Create tokenized data
    # Create prompts
    prompts = []
    for eval_instance in map(eval_data.__getitem__, targets):
        prompts.append(
            "### Instruction: \n"
            + eval_instance['instruction']
            + "### Input: \n"
            + eval_instance["instances"][0]["input"]
            + "### Response: \n"
        )

    tokenized_prompts = tokenizer(
        prompts,
        return_tensors='pt',
    )

    GEN_CONFIG_PATH = 'tiiuae/falcon-7b'
    GenerationConfig.from_pretrained(GEN_CONFIG_PATH)
    with torch.no_grad():
        y = model.generate(
            input_ids=tokenized_prompts['input_ids'],
            attention_mask=tokenized_prompts['attention_mask'],
            #generation_config=generation_config,
            max_length=1048,
            do_sample=True,
            top_p=0.95,
            top_k=4,
            temperature=0.2,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=2.0,
            length_penalty=-100,
        )

        result = tokenizer.decode(
            y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        pr
    with open(eval_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    app()
