"""
词级打窝蒸馏 v5 — 7B模型版
用Qwen2.5-7B的embedding空间重新蒸馏八卦知识词
"""
import numpy as np
from transformers import AutoTokenizer
import os

# ── 加载7B的tokenizer和embedding表 ──
model_dir = 'C:/Users/ww109/.cache/modelscope/Qwen/Qwen2___5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
embed = np.load('finding-order/data/qwen7b_embed_tokens.npy')  # [152064, 3584]

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Embed table shape: {embed.shape}")
print(f"Tokenizer type: {type(tokenizer).__name__}")
print()

# ── 八卦描述关键词 ──
BAGUA_KEYWORDS = {
    "乾": ["强", "天", "刚健", "力量", "创造", "领导", "主动", "阳刚"],
    "兑": ["扩散", "流动", "交流", "喜悦", "泽", "言语", "愉悦", "传播"],
    "离": ["光明", "火", "照亮", "温暖", "依附", "文明", "美丽", "明亮"],
    "震": ["雷", "震动", "撼", "轰鸣", "唤醒", "惊吓", "行动", "激动"],
    "巽": ["风", "顺从", "渗透", "柔软", "飘散", "潜入", "细致", "温和"],
    "坎": ["漩", "陷", "沉", "暗流", "涡", "危险", "深", "隐"],
    "艮": ["止", "停止", "凝", "静", "坚固", "山", "阻挡", "稳定"],
    "坤": ["大地", "柔", "包容", "厚重", "承载", "母性", "顺从", "接纳"],
}

# ── 蒸馏函数 ──
def distill_gua(gua_name, keywords):
    """对一组关键词做embedding均值蒸馏"""
    vecs = []
    tokens_used = []
    
    for kw in keywords:
        token_ids = tokenizer.encode(kw, add_special_tokens=False)
        for tid in token_ids:
            if tid < len(embed):
                vecs.append(embed[tid])
                tokens_used.append((tid, tokenizer.decode([tid]), kw))
    
    if not vecs:
        return None, []
    
    mean_vec = np.mean(vecs, axis=0)
    return mean_vec, tokens_used

# ── 最近邻查找 ──
def find_nearest(vec, top_k=20):
    """余弦相似度找最近邻词"""
    # 归一化
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    embed_norm = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(embed_norm, vec_norm)
    
    # 排除特殊token (id < 5 和一些控制符)
    mask = np.ones(len(sims), dtype=bool)
    mask[:5] = False  # 排除最前面的特殊token
    
    # 排除纯标点/空白
    for i in range(len(sims)):
        token = tokenizer.decode([i])
        if len(token.strip()) <= 1 and not token.strip().isalpha():
            # 但保留中文字符
            if not ('\u4e00' <= token <= '\u9fff' or '\u3400' <= token <= '\u4dbf'):
                mask[i] = False
    
    sims[~mask] = -2  # 排除的设为极低值
    
    top_indices = np.argsort(sims)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        token = tokenizer.decode([idx])
        results.append((token, float(sims[idx])))
    return results

# ── 主流程 ──
print("=" * 60)
print("7B 词级打窝蒸馏")
print("=" * 60)

distilled = {}
for gua, keywords in BAGUA_KEYWORDS.items():
    mean_vec, tokens_used = distill_gua(gua, keywords)
    if mean_vec is None:
        print(f"\n{gua}: ❌ 无有效token")
        continue
    
    distilled[gua] = mean_vec
    
    print(f"\n{gua} 卦 — 输入词: {keywords}")
    print(f"  使用了 {len(tokens_used)} 个token")
    for tid, token, src in tokens_used[:8]:
        print(f"    [{tid}] '{token}' ← {src}")
    
    # 最近邻
    neighbors = find_nearest(mean_vec, top_k=10)
    print(f"  最近邻词:")
    for token, sim in neighbors:
        print(f"    {token:12s}  (cos={sim:.3f})")

# ── 保存蒸馏结果 ──
os.makedirs('finding-order/data', exist_ok=True)
np.savez('finding-order/data/distilled_7b.npz',
         **{f'{gua}_vec': vec for gua, vec in distilled.items()})
print(f"\n[OK] Distilled {len(distilled)} gua -> data/distilled_7b.npz")
