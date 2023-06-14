
# use this script by passing the training argument as string
# Example:
 : '
  ./executor "python finetune.py  \
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

if [ -z "$STY" ]
then
	echo "Please work in a screen"
	exit 1
else
	echo "Working from screen,"
fi
echo Starting run
{ eval "$1"; echo done; runpodctl stop pod $RUNPOD_POD_ID; } &
screen -X detach
echo Detached screen

#Stop the runpod instance
# runpodctl stop pod $RUNPOD_POD_ID
