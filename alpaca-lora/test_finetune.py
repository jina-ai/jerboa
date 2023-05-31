import torch
from evaluate import evaluate
from finetune import load_model_tokenizer, train


def test_debug_mode():
    train(
        base_model='yahma/llama-7b-hf',
        output_dir='trash',
        debug=True,
        use_wandb=False,
    )


def test_eval():
    model, tokenizer = load_model_tokenizer(
        base_model='decapoda-research/llama-7b-hf',
        debug=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        eval_file='resources/eval_sample.jsonl',
        eval_limit=2,
    )
    assert len(results) == 2
    for res in results:
        assert 'id' in res
        assert 'input' in res
        assert 'output' in res
        assert 'to_compare' in res
    assert (
        results[0]['instruction']
        == "The sentence you are given might be too wordy, complicated, or unclear. "
        "Rewrite the sentence and make your writing clearer by keeping it concise. "
        "Whenever possible, break complex sentences into multiple sentences and "
        "eliminate unnecessary words."
    )
