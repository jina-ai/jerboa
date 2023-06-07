import os
import sys
from typing import List, Optional, Tuple

import fire
import torch
import transformers
import wandb
from datasets import load_dataset
from evaluate import evaluate
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from utils.model_config import low_footprint_general
from utils.prompter import Prompter


def load_model_tokenizer(
    base_model: str,
    device_map: dict,
    lora_config: dict,
    device: str,
    debug: bool = False,
    load_in_4bit: object = False,
) -> Tuple[torch.nn.Module, transformers.PreTrainedTokenizer]:
    load_in_8bit = True if not load_in_4bit else False
    # No quantization available on cpu
    if device == 'cpu' and load_in_4bit:
        raise ValueError("Quantization (4bit and 8bit) does not work on cpu")

    # Load small memory config for llama in debugging model
    if debug:
        low_footprint_model_config = low_footprint_general
        model_config = AutoConfig.from_pretrained(
            base_model, trust_remote_code=True, **low_footprint_model_config
        )
        debug_model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=True,
        )
        debug_model.save_pretrained('./trash/empty_model')
    else:
        model_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    quant_config = quant_config if device == "cuda" else None
    model = base_model if not debug else './trash/empty_model'

    # Instantiate Llama model either from base model or from empty model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model,
        torch_dtype=torch.float16,
        device_map=device_map,
        config=model_config,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    # Move model to cpu in debugging mode
    if debug and device == "cpu":
        model = model.to(device)

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**lora_config)
    model = get_peft_model(model, lora_config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    return model, tokenizer


def train(
    # model/data params
    base_model: str = "yahma/llama-7b-hf",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    load_in_4bit=False,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    use_wandb: bool = True,  # flag to completely remove wandb
    wandb_project: str = "jerboa-debug",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: bool = True,  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    debug: bool = False,  # Debug mode to load a small model for quick debugging
    n_samples: Optional[int] = None,
    eval_file: str = "",  # path to file you want to evaluate on
    eval_limit: int = 0,  # limit the number of instructions to evaluate on
):
    if debug:
        batch_size = 2
        micro_batch_size = 1
        num_epochs = 1
        eval_file = 'resources/eval_sample.jsonl'
        eval_limit = 1

    is_master_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    if is_master_process:
        params_dict = {
            "base_model": base_model,
            "data_path": data_path,
            "debug:": debug,
            "n_samples": n_samples,
            "output_dir": output_dir,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "cutoff_len": cutoff_len,
            "val_set_size": val_set_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
            "train_on_inputs": train_on_inputs,
            "add_eos_token": add_eos_token,
            "group_by_length": group_by_length,
            "wandb_project": wandb_project,
            "wandb_run_name": wandb_run_name,
            "wandb_watch": wandb_watch,
            "wandb_log_model": wandb_log_model,
            "resume_from_checkpoint": resume_from_checkpoint or False,
            "prompt template": prompt_template_name,
        }

        print("Training Alpaca-LoRA model with params:", params_dict)

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = use_wandb and (
        len(wandb_project) > 0
        or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    )
    # Only overwrite environ if wandb param passed
    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
        if is_master_process:
            run = wandb.init(wandb_project)
    else:
        os.environ["WANDB_MODE"] = "disabled"
    if use_wandb and len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # No quantization available on cpu
    if device == 'cpu' and (load_in_4bit):
        raise Exception("Quantization (4bit and 8bit) does not work on cpu")

    _lora_config = {
        'r': lora_r,
        'lora_alpha': lora_alpha,
        'target_modules': lora_target_modules,
        'lora_dropout': lora_dropout,
        'bias': "none",
        'task_type': "CAUSAL_LM",
    }

    model, tokenizer = load_model_tokenizer(
        base_model=base_model,
        device=device,
        device_map=device_map,
        lora_config=_lora_config,
        debug=debug,
    )

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if debug:
        train_data = data["train"].select(range(10)).map(generate_and_tokenize_prompt)
        val_data = None
    elif val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if n_samples:
        train_data = train_data.select(range(n_samples))

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not debug,
            logging_steps=10,
            optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else "none",
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if is_master_process:
        lora_dir = f"{output_dir}/lora_adapter"
        model.save_pretrained(lora_dir)

        if wandb_log_model and use_wandb:
            artifact = wandb.Artifact(name='lora_weight', type='model')
            artifact.add_dir(lora_dir)
            run.log_artifact(artifact)

        if eval_file:
            results = evaluate(
                model=model,
                tokenizer=tokenizer,
                eval_file=eval_file,
                eval_limit=eval_limit,
            )

            if use_wandb:
                columns = list(results[0].keys())
                results_data = [[d[key] for key in columns] for d in results]
                eval_table = wandb.Table(columns=columns, data=results_data)
                run.log({"Evaluation": eval_table})
            else:
                print(results)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
