# Setup of runpod instance
apt update -y
apt install gh screen neovim nvtop -y
pip install -U poetry
gh auth login
gh repo clone jina-ai/jerboa
cd jerboa
pwd
poetry config virtualenvs.create false \
     && poetry install --no-interaction --no-ansi

poetry run pip install torch
wandb login
screen -S training_screen
