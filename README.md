# jerboa
alpaca reproducing 


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

fix bitsandbytes

```bash
pip install https://github.com/samsja/bitsandbytes/blob/feat-save-col-wheel/bitsandbytes-0.39.1-py3-none-any.whl\?raw\=true
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


## What to do with this repo

for now you can take a look at the [alpaca lora folder](/alpaca-lora/README.md) 
