import json

from peft import PeftModel, PeftConfig
from typer import Typer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


app = Typer(pretty_exceptions_enable=False)
device = "cuda"
quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

peft_model_id = "jinaai/falcon-7b"
config = PeftConfig.from_pretrained(peft_model_id, trust_remote=True, )
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True,
    quantization_config=quant_config,
)
model = PeftModel.from_pretrained(model, peft_model_id).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

# model = model.to(device)
# model.eval()

# with torch.no_grad():
#   outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=10)
#   print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
# # 'complaint'

@app.command()
def run_eval(eval_file: str = "code_eval.jsonl"):
    eval_data = []
    with open(eval_file, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line))

    results = []
    for eval_instance in eval_data[:2]:
        x = tokenizer(
            "### Instruction: \n"
            + eval_instance['instruction']
            + "### Input: \n"
            + eval_instance["instances"][0]["input"]
            + "### Response: \n",
            return_tensors='pt',
        ).to(device)

        y = model.generate(
            x["input_ids"],
            # max_length=256,
            # do_sample=True,
            # top_p=0.95,
            # top_k=4,
            # temperature=0.2,
            # num_return_sequences=1,
            # eos_token_id=tokenizer.eos_token_id,
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

    # with open(eval_file, 'w') as f:
    #     for result in results:
    #         f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    app()
