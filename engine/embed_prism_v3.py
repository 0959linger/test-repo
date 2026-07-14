"""
嵌入棱镜 v3：纯偶极子 — 去均值 → 7D → 八卦
不投票。不看绝对位置。只看形状。
"""
import numpy as np
from typing import Dict

BAGUA = ['乾','兑','离','震','坤','艮','坎','巽']
ANGLES = {n: i * 45 for i, n in enumerate(BAGUA)}

YANG = {'乾', '兑', '离', '震'}
YIN  = {'坤', '艮', '坎', '巽'}

COORDS = {
    '乾': ( 1.0,  0.0), '兑': ( 0.707,  0.707), '离': ( 0.0,  1.0),
    '震': (-0.707,  0.707), '坤': (-1.0,  0.0), '艮': (-0.707, -0.707),
    '坎': ( 0.0, -1.0), '巽': ( 0.707, -0.707),
}


class EmbedPrismV3:
    """纯偶极子棱镜：向量形状 → 八卦"""
    
    def project(self, vec: np.ndarray) -> Dict:
        v = vec.astype(np.float64)
        
        # ═══ 去均值：只看形状，不看绝对位置 ═══
        v_centered = v - np.mean(v)
        
        mean = 0.0  # 去均值后必然为0，不用
        std = float(np.std(v_centered))
        if std < 1e-10:
            std = 1e-10
        
        # 偏度：分布的对称性 → 阳（正偏） vs 阴（负偏）
        skew = float(np.mean((v_centered / std) ** 3))
        
        # 峰度：分布的尖锐度 → 集中（阳） vs 扁平（阴）
        kurt = float(np.mean((v_centered / std) ** 4) - 3)
        
        # 趋势：后半段 vs 前半段 → 升（阳） vs 降（阴）
        mid = len(v_centered) // 2
        trend = float(np.mean(v_centered[mid:]) - np.mean(v_centered[:mid]))
        
        # 曲率：后半段波动 vs 前半段波动 → 发散（阳） vs 收敛（阴）
        curv = float(np.std(v_centered[mid:]) - np.std(v_centered[:mid]))
        
        # 熵：分布的混乱度 → 有序（阳） vs 混沌（阴）
        p = np.exp(v_centered - np.max(v_centered))
        p = p / np.sum(p)
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)))
        
        # ═══ 偶极子投影 ═══
        # 全部用去均值后的形状特征
        skew_n = np.clip(skew * 2.0, -5, 5)
        kurt_n = np.clip(kurt * 0.5, -5, 5)
        trend_n = np.clip(trend * 20.0, -5, 5)
        curv_n = np.clip(curv * 5.0, -5, 5)
        entropy_n = np.clip(entropy * 0.5, 0, 3)
        
        scores = np.zeros(8)
        for i, t in enumerate(BAGUA):
            x, y = COORDS[t]
            
            # x 轴（乾-坤）：偏度 + 趋势 → 外放 vs 内收
            # y 轴（离-坎）：峰度 + 曲率 → 尖锐 vs 平坦
            scores[i] = (
                # 主项：偏度沿x，峰度沿y
                x * skew_n * 1.5 +
                y * kurt_n * 1.0 +
                # 趋势增强：升/降主要影响乾-坤轴
                x * trend_n * 1.0 +
                # 曲率增强：发散/收敛主要影响离-坎轴  
                y * curv_n * 0.5 +
                # 熵作为全局"场温"
                entropy_n * 0.3
            )
        
        # 数值稳定的 softmax
        scores -= scores.max()
        exp_s = np.exp(np.clip(scores, -50, 50))
        distribution = exp_s / exp_s.sum()
        
        sorted_idx = np.argsort(distribution)[::-1]
        
        yang_total = sum(distribution[BAGUA.index(t)] for t in YANG)
        yin_total = sum(distribution[BAGUA.index(t)] for t in YIN)
        
        return {
            'distribution': {BAGUA[i]: float(distribution[i]) for i in range(8)},
            'winner': BAGUA[sorted_idx[0]],
            'top3': [BAGUA[i] for i in sorted_idx[:3]],
            'cv': float(np.std(distribution) / (np.mean(distribution) + 1e-10)),
            'yang': float(yang_total),
            'yin': float(yin_total),
            '_shape': {
                'skew': float(skew), 'kurt': float(kurt),
                'trend': float(trend), 'curv': float(curv),
                'entropy': float(entropy),
            }
        }


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    from gguf import GGUFReader
    
    model_path = r"C:\Users\ww109\.qwenpaw\llama.cpp\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    reader = GGUFReader(model_path)
    embed = None
    for t in reader.tensors:
        if t.name == 'token_embd.weight':
            embed = t.data.astype(np.float32)
            break
    
    byte_to_tid = {i: i for i in range(256)}
    f = reader.fields['tokenizer.ggml.tokens']
    parts = f.parts
    decode_map, tid = {}, 0
    idx = 5
    while idx + 1 < len(parts):
        L = int(parts[idx][0])
        decode_map[tid] = bytes(parts[idx+1][:L])
        tid += 1; idx += 2
    bpe_index = {tok: tid for tid, tok in decode_map.items() if len(tok) >= 2}
    
    def word_to_vec(text):
        utf8 = text.encode('utf-8')
        tids = [byte_to_tid[b] for b in utf8]
        for start in range(len(utf8)):
            for end in range(len(utf8), start, -1):
                sub = utf8[start:end]
                if sub in bpe_index:
                    tids.append(bpe_index[sub])
                    break
        return embed[tids].mean(axis=0).astype(np.float64) if tids else np.zeros(952)
    
    prism = EmbedPrismV3()
    
    # 英文词
    print("=" * 80)
    print(f"{'EN':<12} {'卦':<4} {'top3 分布':<50} {'CV':<6} {'阳':<7} {'阴':<7}")
    print("=" * 80)
    
    en_words = ["love", "hate", "peace", "war", "death", "spring", "sky", "ocean",
                "king", "queen", "fire", "water", "light", "dark", "sun", "moon"]
    
    for w in en_words:
        vec = word_to_vec(w)
        r = prism.project(vec)
        d = r['distribution']
        top = ' > '.join(f"{t}={d[t]:.3f}" for t in r['top3'][:3])
        shape = r['_shape']
        print(f"  {w:<10} → {r['winner']:<4} | {top:<50} | {r['cv']:.2f} | "
              f"{r['yang']:.3f} | {r['yin']:.3f}  "
              f"⚙s={shape['skew']:+.2f} k={shape['kurt']:+.2f} t={shape['trend']:+.3f}")
    
    print()
    print("=" * 80)
    print(f"{'CN':<12} {'卦':<4} {'top3 分布':<50} {'CV':<6} {'阳':<7} {'阴':<7}")
    print("=" * 80)
    
    cn_words = ["爱", "恨", "和平", "战争", "死亡", "春天", "天空", "海洋",
                "王", "后", "火", "水", "光", "暗", "太阳", "月亮"]
    
    for w in cn_words:
        vec = word_to_vec(w)
        r = prism.project(vec)
        d = r['distribution']
        top = ' > '.join(f"{t}={d[t]:.3f}" for t in r['top3'][:3])
        shape = r['_shape']
        print(f"  {w:<4} → {r['winner']:<4} | {top:<50} | {r['cv']:.2f} | "
              f"{r['yang']:.3f} | {r['yin']:.3f}  "
              f"⚙s={shape['skew']:+.2f} k={shape['kurt']:+.2f} t={shape['trend']:+.3f}")
    
    print()
    print("=" * 80)
    print("关键配对：")
    pairs = [("love","hate"), ("peace","war"), ("light","dark"), ("sun","moon"),
             ("fire","water"), ("king","queen")]
    for a, b in pairs:
        ra = prism.project(word_to_vec(a))
        rb = prism.project(word_to_vec(b))
        print(f"  {a}→{ra['winner']}(阳={ra['yang']:.2f})  vs  "
              f"{b}→{rb['winner']}(阳={rb['yang']:.2f})  "
              f"Δ阳={ra['yang']-rb['yang']:+.3f}")
