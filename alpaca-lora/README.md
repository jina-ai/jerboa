# Jina AI alpca lora fork


This folder is the fork of the great [alpaca lora](https://github.com/tloen/alpaca-lora) repo. 


the rest of this readme is the original README from the repo.


## Jina specific stuff

!!! To follow the rest be sure to have enabled your virtual env with poetry by looking at the top root README.md 

## Runpod
To run this repository on runpod, use the latest PyTorch container on runpod. 
Then run the following command to install the necessary dependencies and login to github. 
Afterward you can continue with the training and inference explained below. 

```bash
bash <(curl -Ls https://raw.githubusercontent.com/sebastian-weisshaar/config_jerboa/main/config.sh)
```

### debug mode

We can run the code in debug mode, this allows to test the code with low resource (small model and small dataset)

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py --debug
```

this still use wandb. If you want to disable wandb you can do

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py --debug --use-wandb=False
```

## Distributed training
It is possible to train the model on multiple GPUs. This allows to train the model faster.
Training on 2x3090 GPUs: 

```bash
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=1234 finetune.py --base_model 'yahma/llama-7b-hf' --output_dir './lora-alpaca' --batch_size 128 --micro_batch_size 4 --eval_limit 30 --eval_file eval.jsonl --wandb_log_model true --wandb_project jerboa --wandb_run_name jerboa-intial-train --wandb_watch gradients  --num_epochs 3
```

Training on 3x3090 GPUs: 

```bash
WORLD_SIZE=3 CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=1234 finetune.py --base_model 'yahma/llama-7b-hf' --output_dir './lora-alpaca' --batch_size 128 --micro_batch_size 4 --eval_limit 30 --eval_file eval.jsonl --wandb_log_model true --wandb_project jerboa --wandb_run_name jerboa-intial-train --wandb_watch gradients  --num_epochs 3
```

## Training Datasets
Currently, the training pipeline supports 2 training datasets:
- `yahma/alpaca-cleaned`: cleaned version of the alpaca dataset, available on the HF datasets hub. This is the used dataset by default
  - `code_alpaca_20k.json`: a dataset of 20k code snippets, available locally. To use this dataset, specify the following parameter in the training command: `--data_path ./code_alpaca_20k.json`


## Tests

You can run our tests by doing


```bash
CUDA_VISIBLE_DEVICES=0 pytest test_finetune.py
```


this should take a couple of second to run on a singe 3090. Just doing one epoch over 100 data points



### Target modules
You need to specify the target `lora_target_modules` as for each different model that is used. For Falcon 7b `lora_target_modules=["query_key_value"]`
For Llama 7b `lora_target_modules=["q_proj", "v_proj"]`. 

## Evaluation
To run evaluation you first need an evaluation file or dataset.
This evaluation looks like the following:

```bash
{"id": "user_oriented_task_0", "motivation_app": "Grammarly", "instruction": "The sentence you are given might be too wordy, complicated, or unclear. Rewrite the sentence and make your writing clearer by keeping it concise. Whenever possible, break complex sentences into multiple sentences and eliminate unnecessary words.", "instances": [{"input": "If you have any questions about my rate or if you find it necessary to increase or decrease the scope for this project, please let me know.", "output": "If you have any questions about my rate or find it necessary to increase or decrease this project's scope, please let me know."}]}
```

You can download self-instruct evaluation data using this command:

```bash
wget https://raw.githubusercontent.com/yizhongw/self-instruct/main/human_eval/user_oriented_instructions.jsonl
```

To run evaluation after finetuning you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=2 \
python finetune.py \
  --base_model 'yahma/llama-7b-hf' \
  --lora_target_modules "[q_proj, v_proj]" \
  --data_path <Your-data-path> \
  --output_dir './lora-alpaca' \
  --wandb_project 'jerboa' \
  --wandb_run_name 'test-run' \
  --wandb_watch 'gradients' \
  --wandb_log_model 'true' \
  --num_epochs '2' \
  --eval_file 'user_oriented_instructions.jsonl' \
  --eval_limit '5'
```

--eval_file: path to the evaluation file<br>
--eval_limit: number of examples to evaluate on

Evaluation results will be automatically logged to wandb.
