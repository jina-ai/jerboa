import os

import torch

from jerboa.evaluate import evaluate
from jerboa.finetune import load_model_tokenizer, train


def test_debug_mode(tmp_path):
    train(
        base_model='yahma/llama-7b-hf',
        output_dir=str(tmp_path),
        debug=True,
        use_wandb=False,
    )
    assert os.path.getsize(tmp_path / 'lora_adapter/adapter_model.bin') > 443


lora_r: int = 8
lora_alpha: int = 16
lora_dropout: float = 0.05
lora_target_modules = [
    "q_proj",
    "v_proj",
]
_lora_config = {
    'r': lora_r,
    'lora_alpha': lora_alpha,
    'target_modules': lora_target_modules,
    'lora_dropout': lora_dropout,
    'bias': "none",
    'task_type': "CAUSAL_LM",
}


def test_eval():
    model, tokenizer = load_model_tokenizer(
        base_model='yahma/llama-7b-hf',
        device_map="auto",
        debug=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lora_config=_lora_config,
    )
    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        eval_file='jerboa/resources/eval_sample.jsonl',
        eval_limit=2,
    )
    assert len(results) == 2
    for res in results:
        assert 'id' in res
        assert 'input' in res
        assert 'output' in res
    assert (
        results[0]['instruction']
        == "The sentence you are given might be too wordy, complicated, or unclear. "
        "Rewrite the sentence and make your writing clearer by keeping it concise. "
        "Whenever possible, break complex sentences into multiple sentences and "
        "eliminate unnecessary words."
    )
