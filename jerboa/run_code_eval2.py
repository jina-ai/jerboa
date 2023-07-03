import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftConfig
from jerboa.utils.prompter import Prompter

device = "cuda"
device_map = "auto"
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
    device_map=device_map,
)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token


prompter = Prompter('alpaca')
eval_file: str = "eval.jsonl"
eval_data = []
with open(eval_file, 'r') as f:
    for line in f:
        eval_data.append(json.loads(line))

results = []
targets = list(range(2))
# Create tokenized data
# Create prompts
prompts = []
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
        # generation_config=generation_config,
        max_length=512,
        do_sample=True,
        top_p=0.75,
        top_k=50,
        temperature=0.1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=3.0,
        length_penalty=-100,
    )
    result = prompter.get_response(tokenizer.decode(
        output[0].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ))

    print("Input:", eval_instance)
    print("Prompt:", prompt)
    print("Result:", result)



print("Done")
