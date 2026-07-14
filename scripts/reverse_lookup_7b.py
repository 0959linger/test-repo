"""
反向查表测试：蒸馏词 → 7B embedding → 最近邻
验证7B的embedding空间是否有真正的语义扩展（非镜子效应）
"""
import numpy as np
from transformers import AutoTokenizer

MODEL_DIR = 'C:/Users/ww109/.cache/modelscope/Qwen/Qwen2___5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
embed = np.load('finding-order/data/qwen7b_embed_tokens.npy')
distilled = np.load('finding-order/data/distilled_7b.npz')

# 八卦对应的蒸馏关键词（从蒸馏结果中得出每个卦的核心词）
GUA_CORE_WORDS = {
    "乾": ["刚", "强", "阳", "天", "主动", "力量"],
    "兑": ["传播", "扩散", "交流", "愉悦", "流动"],
    "离": ["温暖", "明亮", "光明", "文明", "美丽"],
    "震": ["惊", "轰", "震动", "撼", "激动"],
    "巽": ["风", "潜", "飘", "顺", "渗透", "入"],
    "坎": ["沉", "深", "隐", "暗", "陷", "流"],
    "艮": ["静", "止", "停止", "稳定", "凝"],
    "坤": ["柔", "承载", "包容", "母", "顺", "从"],
}

def find_nearest(vec, top_k=15, exclude_ids=None):
    """余弦相似度找最近邻"""
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    embed_norm = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(embed_norm, vec_norm)
    
    # 排除特殊token + 纯标点/空白/数字
    mask = np.ones(len(sims), dtype=bool)
    mask[:5] = False
    if exclude_ids:
        for eid in exclude_ids:
            if eid < len(mask):
                mask[eid] = False
    
    # 排除非中文字符/非有意义词
    for i in range(len(sims)):
        token = tokenizer.decode([i]).strip()
        # 保留中文单字
        if '\u4e00' <= token <= '\u9fff':
            continue
        # 保留中文多字词
        if len(token) >= 2 and all('\u4e00' <= c <= '\u9fff' for c in token):
            continue
        # 保留有意义的英文词（>=3字母）
        if token.isalpha() and len(token) >= 3:
            continue
        mask[i] = False
    
    sims[~mask] = -2
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [(tokenizer.decode([idx]), float(sims[idx]), int(idx)) for idx in top_indices]

print("=" * 70)
print("反向查表测试 — 7B embedding空间")
print("=" * 70)

for gua, core_words in GUA_CORE_WORDS.items():
    gua_vec = distilled[f'{gua}_vec']
    
    # 收集所有输入词的token id用于排除（看是否镜子）
    input_ids = set()
    for kw in core_words:
        for tid in tokenizer.encode(kw, add_special_tokens=False):
            input_ids.add(tid)
    
    # 找最近邻（排除输入词本身）
    neighbors = find_nearest(gua_vec, top_k=20, exclude_ids=input_ids)
    
    # 过滤掉输入词
    filtered = [(t, s, i) for t, s, i in neighbors if t not in core_words]
    
    print(f"\n{gua}卦 蒸馏向量 → 7B空间最近邻 (排除核心词后):")
    for token, sim, tid in filtered[:12]:
        marker = " ← 新词!" if token not in core_words else ""
        print(f"  {token:12s} cos={sim:.3f}{marker}")
