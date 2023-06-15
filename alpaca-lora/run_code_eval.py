import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from typer import Typer

app = Typer(pretty_exceptions_enable=False)


@app.command()
def run_eval(checkpoint: str, device: str = "cuda", eval_file: str = "code_eval.jsonl"):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    eval_data = []
    with open(eval_file, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))

    results = []
    for eval_instance in eval_data:
        x = tokenizer.encode(
            "### Instruction: \n"
            + eval_instance['instruction']
            + "### Input: \n"
            + eval_instance["instances"][0]["input"]
            + "### Response: \n",
            return_tensors='pt',
        ).to(device)
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

    with open(eval_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    app()
