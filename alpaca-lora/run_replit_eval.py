import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "teknium/Replit-v2-CodeInstruct-3B", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "teknium/Replit-v2-CodeInstruct-3B", trust_remote_code=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
eval_data = []
with open('code_eval.jsonl', 'r') as f:
    for line in f:
        eval_data.append(json.loads(line))

results = []
for i, eval_instance in enumerate(eval_data):
    x = tokenizer.encode(
        "### Instruction: \n"
        + eval_instance['instruction']
        + "### Input: \n"
        + eval_instance["instances"][0]["input"]
        + "### Response: \n",
        return_tensors='pt',
    )
    x = x.to(device)
    print(x.is_cuda)
    print(next(model.parameters()).is_cuda)
    y = model.generate(
        x,
        max_length=256,
        do_sample=True,
        top_p=0.95,
        top_k=4,
        temperature=0.2,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(
        tokenizer.decode(
            y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    )
    eval_instance["instances"][0]["replit_code_output"] = tokenizer.decode(
        y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    results.append(eval_instance)

with open('code_eval.jsonl', 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')
