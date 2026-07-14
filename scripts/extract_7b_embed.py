"""Extract embedding table from Qwen2.5-7B-Instruct safetensors."""
from safetensors import safe_open
import numpy as np
import os

base = 'C:/Users/ww109/.cache/modelscope/Qwen/Qwen2___5-7B-Instruct'
out_dir = 'finding-order/data'
os.makedirs(out_dir, exist_ok=True)

# Extract embed_tokens from shard 1
with safe_open(f'{base}/model-00001-of-00004.safetensors', framework='pt') as f:
    embed = f.get_tensor('model.embed_tokens.weight')
    print(f'embed_tokens.shape: {embed.shape}, dtype: {embed.dtype}')

# Save as float32 numpy (for compatibility)
np.save(f'{out_dir}/qwen7b_embed_tokens.npy', embed.float().numpy())
size_mb = os.path.getsize(f'{out_dir}/qwen7b_embed_tokens.npy') / 1024**2
print(f'Saved embed_tokens: {size_mb:.0f} MB')
