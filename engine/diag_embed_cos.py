"""诊断：7B嵌入表词向量之间的余弦距离"""
import numpy as np, json, sys, os

WORKSPACE = r"C:\Users\ww109\.qwenpaw\workspaces\default"
sys.path.insert(0, os.path.join(WORKSPACE, "finding-order", "engine"))
from engine_v8 import Engine

# 加载嵌入表
embed = np.load(os.path.join(WORKSPACE, 'finding-order', 'data', 'qwen7b_embed_tokens.npy'))
print(f'嵌入表维度: {embed.shape}')

# 随机采样1000个token
import random
random.seed(42)
idx = random.sample(range(len(embed)), 1000)
vecs = embed[idx]
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_n = vecs / (norms + 1e-10)

# 余弦矩阵
cos_mat = vecs_n @ vecs_n.T
mask = ~np.eye(1000, dtype=bool)
off_diag = cos_mat[mask]
print(f'词间余弦均值: {off_diag.mean():.4f}')
print(f'词间余弦std:   {off_diag.std():.4f}')
print(f'词间余弦min:   {off_diag.min():.4f}')
print(f'词间余弦max:   {off_diag.max():.4f}')
for p in [10,25,50,75,90,95,99]:
    print(f'  P{p}: {np.percentile(off_diag, p):.4f}')

# 加载蒸馏数据和分词器，算8卦锚点间的余弦
engine = Engine(hour=12)
engine._ensure_backend()

with open(os.path.join(WORKSPACE, 'finding-order', 'engine', 'gua8_bait_distilled_7b.json'),
          'r', encoding='utf-8') as f:
    dd = json.load(f)

BAGUA = ['乾','兑','离','震','坤','艮','坎','巽']
gua_vecs = {}
for gua in BAGUA:
    ws = [w for w,s in dd['gua_to_words'].get(gua,[])][:15]
    vecs_g = []
    for w in ws:
        v = engine._word_to_vec(w)
        if v is not None and np.linalg.norm(v) > 0:
            vecs_g.append(v)
    if vecs_g:
        gua_vecs[gua] = np.mean(vecs_g, axis=0)
        gua_vecs[gua] = gua_vecs[gua] / np.linalg.norm(gua_vecs[gua])

print('\n8卦锚点间余弦:')
for i, g1 in enumerate(BAGUA):
    if g1 not in gua_vecs: continue
    row = []
    for g2 in BAGUA:
        if g2 not in gua_vecs: continue
        c = np.dot(gua_vecs[g1], gua_vecs[g2])
        row.append(f'{c:.4f}')
    print(f'  {g1}: {" ".join(row)}')
