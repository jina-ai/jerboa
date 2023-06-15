
# use this script by passing the training argument as string
# Example:
 : '
  ./training_run.sh "python finetune.py  \
  --base-model yahma/llama-7b-hf \
  --output-dir ./lora-alpaca \
  --batch-size 128 \
  --micro-batch_size 4 \
  --eval-limit 30 \
  --eval-file eval.json \
  --wandb-log_model true  \
  --wandb-watch gradients \
  --num-epochs 3"
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
