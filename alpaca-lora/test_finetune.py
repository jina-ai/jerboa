from finetune import train
from evaluate import evaluate


def test_debug_mode():
    train(base_model='decapoda-research/llama-7b-hf', output_dir='trash', debug=True)


def test_eval():
    results = evaluate(base_model='decapoda-research/llama-7b-hf', eval_file='alpaca-lora/resources/eval_sample.jsonl',
                       eval_limit=2)
    assert len(results) == 2
