"""Check tokenizer details for Qwen2.5-7B."""
import json
base = 'C:/Users/ww109/.cache/modelscope/Qwen/Qwen2___5-7B-Instruct'
cfg = json.load(open(f'{base}/tokenizer_config.json'))
print('tokenizer_class:', cfg.get('tokenizer_class'))
print('model_max_length:', cfg.get('model_max_length'))
print('chat_template present:', 'chat_template' in cfg)
# Check special tokens
print('bos_token_id:', cfg.get('bos_token_id'))
print('eos_token_id:', cfg.get('eos_token_id'))
print('pad_token_id:', cfg.get('pad_token_id'))
