import json

import torch
import transformers
from transformers import AutoTokenizer
from utils.prompter import Prompter

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
prompter = Prompter("")
eval_data = []
with open('eval.jsonl', 'r') as f:
    for line in f:
        eval_data.append(json.loads(line))

for i, eval_instance in enumerate(eval_data):
    print(i)
    prompt = prompter.generate_prompt(
        eval_instance["instruction"], eval_instance["instances"][0]["input"]
    )
    sequences = pipeline(
        prompt,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(sequences[0]["generated_text"])
    eval_instance["instances"][0]["falcon_output"] = sequences[0]["generated_text"]
    with open('eval.jsonl', 'w') as f:
        for instance in eval_data:
            f.write(json.dumps(instance) + '\n')
