from finetune import train


def test_debug_mode():
    train(base_model='decapoda-research/llama-7b-hf', output_dir='trash', debug=True)
