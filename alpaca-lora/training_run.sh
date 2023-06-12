
# use this script by passing the training argument as string
# Example:
 :' ./executor "python finetune.py  \
  --base_model yahma/llama-7b-hf \
  --output_dir ./lora-alpaca \
  --batch_size 128 \
  --micro_batch_size 4 \
  --eval_limit 30 -\
  -eval_file eval.json \
  --wandb_log_model true  \
  --wandb_watch gradients \
  --num_epochs 3"
'
echo starting run
screen -dm eval "$1"
echo training done

#Stop the runpod instance
runpodctl stop pod $RUNPOD_POD_ID
