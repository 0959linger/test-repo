"""
桥接 v2：7B蒸馏向量 → 反向查表 → 语义去重 → JSON
解决"渗透霸屏"问题：对邻居词做语义聚类，每个簇只保留最强代表
"""
import numpy as np
import json
from transformers import AutoTokenizer

MODEL_DIR = 'C:/Users/ww109/.cache/modelscope/Qwen/Qwen2___5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
embed = np.load('finding-order/data/qwen7b_embed_tokens.npy')
distilled = np.load('finding-order/data/distilled_7b.npz')

BAGUA = ['乾', '兑', '离', '震', '巽', '坎', '艮', '坤']

def get_token_embed(tid):
    """获取token的embedding向量"""
    if tid < len(embed):
        return embed[tid]
    return None

def cosine(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))

def find_nearest(vec, top_k=50):
    """余弦相似度找最近邻（中文词）"""
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    embed_norm = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(embed_norm, vec_norm)
    
    mask = np.ones(len(sims), dtype=bool)
    mask[:5] = False
    for i in range(len(sims)):
        token = tokenizer.decode([i]).strip()
        has_cjk = any('\u4e00' <= c <= '\u9fff' for c in token)
        if not has_cjk:
            mask[i] = False
    
    sims[~mask] = -2
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [(tokenizer.decode([idx]), float(sims[idx]), int(idx)) for idx in top_indices]

def semantic_deduplicate(neighbors, min_diversity=0.25):
    """
    语义去重：贪婪筛选，每个保留词之间的余弦距离 > min_diversity
    保留的是每个语义簇的最强代表
    """
    if not neighbors:
        return []
    
    # 按相似度排序（已经排好了）
    kept = [neighbors[0]]  # 保留最强词
    
    for word, sim, tid in neighbors[1:]:
        w_emb = get_token_embed(tid)
        if w_emb is None:
            continue
        
        # 检查与已保留词的语义距离
        is_diverse = True
        for kw, ks, ktid in kept:
            k_emb = get_token_embed(ktid)
            if k_emb is None:
                continue
            cos_sim = cosine(w_emb, k_emb)
            if cos_sim > (1.0 - min_diversity):  # 距离太近，视为同一语义簇
                is_diverse = False
                break
        
        if is_diverse:
            kept.append((word, sim, tid))
        
        if len(kept) >= 15:  # 最多保留15个
            break
    
    return kept

print("=" * 60)
print("7B 语义去重蒸馏")
print("=" * 60)

gua_to_words = {}
for gua in BAGUA:
    vec_key = f'{gua}_vec'
    if vec_key not in distilled:
        continue
    
    gua_vec = distilled[vec_key]
    neighbors = find_nearest(gua_vec, top_k=50)
    
    # 过滤噪声
    clean = []
    for word, sim, tid in neighbors:
        word = word.strip()
        if not word or len(word) < 1:
            continue
        if any(ord(c) > 0x2ffff for c in word):
            continue
        clean.append((word, sim, tid))
    
    # 语义去重
    diverse = semantic_deduplicate(clean, min_diversity=0.40)
    
    gua_to_words[gua] = [[w, s] for w, s, tid in diverse]
    print(f"  {gua}: {len(clean)} → {len(diverse)} 去重后")
    print(f"     {[w for w,s,tid in diverse]}")
    print()

result = {
    "model_name": "Qwen2.5-7B-Instruct (modelscope, diversity-filtered)",
    "gua_to_words": gua_to_words,
}

out_path = 'finding-order/engine/gua8_bait_distilled_7b.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n[OK] Saved to {out_path}")
print(f"     {len(gua_to_words)} gua, {sum(len(v) for v in gua_to_words.values())} words total")
