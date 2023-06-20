# Jerboa

Jerboa is an experimental repo to finetune several open source LLM (llama, falcon, ...) on several datasets ( alpaca, code-alpaca, ...).

The repo is shared publicly to allow the community to reproduce our results. Though it is still very experimental and a lot
of breaking change will happen. This is not production ready software. Check ouf [finetuner](https://github.com/jina-ai/finetuner) for a production ready software.

credits: this project is originally a fork of the great [alpaca lora](https://github.com/tloen/alpaca-lora) repo.

## Set up the codebase for dev

### 1 set up poetry env

first install poetry

```bash
pip install -U poetry
```

If you encounter a keyring error on the Berlin GPU run: 
```bash
poetry run python -m pip install keyring
poetry run python -m keyring --disable
```

You don't need to setup a virtual env, poetry will take care of it.
```bash
poetry install
```

this is needed to fixe the OOM problem

For Berlin GPU install torch:
```bash
pip install torch
```

then activate the virtual env

```bash
poetry shell
```


### 2 pre commit

install pre commit hook

```bash
pre-commit install
```


# Jina AI alpca lora fork

to follow the rest of the readme, you need to be in the `jerboa` folder.

```bash
cd jerboa
```

This folder is the fork of the great [alpaca lora](https://github.com/tloen/alpaca-lora) repo. 


the rest of this readme is the original README from the repo.


## Finetuning


!!! To follow the rest be sure to have enabled your virtual env with poetry (see above)

## Runpod
To run this repository on runpod, use the latest PyTorch container on runpod.
Connect to the VM via SSH, then run the following command to install the necessary dependencies and login to github. 
You can now continue with the training and inference explained below. 

```bash
bash <(curl -Ls https://raw.githubusercontent.com/sebastian-weisshaar/config_jerboa/main/config.sh)
```

To run a training run and automatically shutdown the runpod afterwards run the following command in a screen on runpod:
ATTENTION: The runpod shuts down immediately if you run the command before logging in to WandB
```bash
./training_run.sh "python <your training run setup>"
```

### debug mode

We can run the code in debug mode, this allows to test the code with little resources (small model and small dataset).

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
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=1234 finetune.py --base-model 'yahma/llama-7b-hf' --output-dir './lora-alpaca' --batch-size 128 --micro-batch-size 4 --eval-limit 30 --eval-file eval.jsonl --wandb-log-model --wandb-project jerboa --wandb-run-name jerboa-intial-train --wandb-watch gradients  --num-epochs 3
```

Training on 3x3090 GPUs: 

```bash
WORLD_SIZE=3 CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=1234 finetune.py --base-model 'yahma/llama-7b-hf' --output-dir './lora-alpaca' --batch-size 128 --micro-batch-size 4 --eval-limit 30 --eval-file eval.jsonl --wandb-log-model --wandb-project jerboa --wandb-run-name jerboa-intial-train --wandb-watch gradients  --num-epochs 3
```

## Training Datasets
Currently, the training pipeline supports 2 training datasets:
- `yahma/alpaca-cleaned`: cleaned version of the alpaca dataset, available on the HF datasets hub. This is the used dataset by default
- `sahil2801/CodeAlpaca-20k`: a dataset of 20k code snippets, available on the HF datasets hub. To use this dataset, specify the following parameter in the training command: `--data_path "sahil2801/CodeAlpaca-20k"`
- `togethercomputer/RedPajama-Data-Instruct`: this dataset is provided by `togethercomputer` and contains 2 subsets:
  - NI (Natural Instructions): An instruction-tuning dataset comprising a diverse set of tasks in natural languages.
  To use this dataset, simply add the flags `--data-path togethercomputer/RedPajama-Data-Instruct --data-files data/NI_decontaminated.jsonl.zst`
  - P3 (Public Pool of Prompts): A large dataset featuring various creative tasks obtained from crowdsourcing efforts.
  To use this dataset, simply add the flags `--data-path togethercomputer/RedPajama-Data-Instruct --data-files data/P3_decontaminated.jsonl.zst`

You can also come up with a different dataset if it follows the alpaca dataset format. If it follows a different format similar to one of the previously supported formats, you can specify one of the existing dataset preprocessors to transform it to alpaca format during training.
Just add the following flags:
`--data-path curated_dataset_name --data-files curated_dataset_data_files --dataset-preprocessor redpajamas_ni_to_alpaca_format `
## Tests

You can run our tests by doing at the root folder level


```bash
CUDA_VISIBLE_DEVICES=0 pytest tests
```


this should take a couple of second to run on a singe 3090. Just doing one epoch over 100 data points



### Target modules
You need to specify the target `lora_target_modules` as for each different model that is used. For Falcon 7b `lora_target_modules=["query_key_value"]`
For Llama 7b `lora_target_modules=["q_proj", "v_proj"]`. However, in the command line the target modules need to be passed as individual arguments. 
See the example below for an illustration. 

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
  --base-model 'yahma/llama-7b-hf' \
  --lora-target-modules q_proj \
  --lora-target-modules v_proj \
  --data-path <Your-data-path> \
  --output-dir './lora-alpaca' \
  --wandb-project 'jerboa' \
  --wandb-run-name 'test-run' \
  --wandb-watch 'gradients' \
  --wandb-log-model \
  --num-epochs '2' \
  --eval-file 'user_oriented_instructions.jsonl' \
  --eval-limit '5'
```

--eval-file: path to the evaluation file<br>
--eval-limit: number of examples to evaluate on

Evaluation results will be automatically logged to wandb.
