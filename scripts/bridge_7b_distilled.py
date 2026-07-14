"""
桥接：7B蒸馏向量 → 反向查表 → JSON（bagua_nn兼容格式）
"""
import numpy as np
import json
from transformers import AutoTokenizer

MODEL_DIR = 'C:/Users/ww109/.cache/modelscope/Qwen/Qwen2___5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
embed = np.load('finding-order/data/qwen7b_embed_tokens.npy')
distilled = np.load('finding-order/data/distilled_7b.npz')

BAGUA = ['乾', '兑', '离', '震', '巽', '坎', '艮', '坤']

def find_nearest(vec, top_k=30):
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    embed_norm = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(embed_norm, vec_norm)
    
    # 过滤：只要中文词
    mask = np.ones(len(sims), dtype=bool)
    mask[:5] = False
    for i in range(len(sims)):
        token = tokenizer.decode([i]).strip()
        has_cjk = any('\u4e00' <= c <= '\u9fff' for c in token)
        if not has_cjk:
            mask[i] = False
    
    sims[~mask] = -2
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [(tokenizer.decode([idx]), float(sims[idx])) for idx in top_indices]

gua_to_words = {}
for gua in BAGUA:
    vec_key = f'{gua}_vec'
    if vec_key not in distilled:
        print(f"  WARNING: {vec_key} not found in npz")
        continue
    
    gua_vec = distilled[vec_key]
    neighbors = find_nearest(gua_vec, top_k=30)
    
    # 清理
    clean = []
    for word, sim in neighbors:
        word = word.strip()
        if not word or len(word) < 1:
            continue
        # 过滤噪声
        if any(ord(c) > 0x2ffff for c in word):
            continue
        clean.append([word, sim])
    
    gua_to_words[gua] = clean[:30]
    print(f"  {gua}: {len(clean)} words, top: {[w for w,s in clean[:8]]}")

result = {
    "model_name": "Qwen2.5-7B-Instruct (modelscope)",
    "gua_to_words": gua_to_words,
}

out_path = 'finding-order/engine/gua8_bait_distilled_7b.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n[OK] Saved to {out_path}")
print(f"     {len(gua_to_words)} gua, {sum(len(v) for v in gua_to_words.values())} words total")
