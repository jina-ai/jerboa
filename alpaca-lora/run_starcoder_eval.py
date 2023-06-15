import json

from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceH4/starchat-alpha"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

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
    ).to(device)
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
    eval_instance["instances"][0]["starcoder_code_output"] = tokenizer.decode(
        y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    results.append(eval_instance)

with open('code_eval.jsonl', 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')
