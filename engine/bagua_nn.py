"""
八卦神经网络 —— qi状态 → 自然语言

不是预训练的黑盒。三层：
  8 → 64 → 蒸馏词 → 句子

权重来源（不靠梯度下降）：
  - 层1 (8→64): 先天方位 + 五行生克 → 扩散场
  - 层2 (64→词): 蒸馏词典（从模型 embedding 空间提取）

输入: qi状态 {卦: 分布, 偏角, 场温, 运动方向}
输出: 候选词 + 自然语言句子
"""

import numpy as np
import json
import os
import random
from typing import Dict, List, Tuple, Optional

# ============================================================
# 八卦基础
# ============================================================

BAGUA = ['乾', '兑', '离', '震', '巽', '坎', '艮', '坤']

BAGUA_ANGLE = {
    '乾': 0.0, '兑': 45.0, '离': 90.0, '震': 135.0,
    '巽': 180.0, '坎': 225.0, '艮': 270.0, '坤': 315.0,
}

WUXING = {
    '乾': '金', '兑': '金', '离': '火', '震': '木',
    '巽': '木', '坎': '水', '艮': '土', '坤': '土',
}

WUXING_SHENG = {'金': '水', '水': '木', '木': '火', '火': '土', '土': '金'}
WUXING_KE    = {'金': '木', '木': '土', '土': '水', '水': '火', '火': '金'}

# 64卦名（标准顺序）
GUA64_NAMES = [
    '乾', '坤', '屯', '蒙', '需', '讼', '师', '比',
    '小畜', '履', '泰', '否', '同人', '大有', '谦', '豫',
    '随', '蛊', '临', '观', '噬嗑', '贲', '剥', '复',
    '无妄', '大畜', '颐', '大过', '坎', '离', '咸', '恒',
    '遁', '大壮', '晋', '明夷', '家人', '睽', '蹇', '解',
    '损', '益', '夬', '姤', '萃', '升', '困', '井',
    '革', '鼎', '震', '艮', '渐', '归妹', '丰', '旅',
    '巽', '兑', '涣', '节', '中孚', '小过', '既济', '未济',
]

# 64卦→(上卦,下卦)
GUA64_COMPONENTS = {}
for idx in range(64):
    GUA64_COMPONENTS[idx] = (BAGUA[idx // 8], BAGUA[idx % 8])


# ============================================================
# BaguaNeuralNet
# ============================================================

class BaguaNeuralNet:
    """
    八卦神经网络：qi状态 → 自然语言
    
    层1: 8→64（扩散场）
    层2: 64→蒸馏词（模型知识）
    层3: 词→句子（句法模板）
    """
    
    def __init__(self, distilled_path: str = None):
        self.gua8_words = {}  # {卦: [(词, 相似度), ...]}
        if distilled_path and os.path.exists(distilled_path):
            self._load_distilled(distilled_path)
    
    # ============================================================
    # 蒸馏词典加载
    # ============================================================
    
    def _load_distilled(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        raw = data['gua_to_words']
        for gua, words in raw.items():
            clean = []
            seen = set()
            for w, s in words:
                w = w.strip()
                if not w:
                    continue
                # 7B模型：单字中文是合法token，保留
                is_single_cjk = len(w) == 1 and '\u4e00' <= w <= '\u9fff'
                if len(w) <= 1 and not is_single_cjk:
                    continue
                if w in seen:
                    continue
                # 只保留纯中文、纯英文/ASCII词，排除多语言噪声
                has_cjk = any('\u4e00' <= c <= '\u9fff' for c in w)
                has_latin = any(c.isascii() and c.isalpha() for c in w)
                has_noise = any(
                    (not c.isascii() and not ('\u4e00' <= c <= '\u9fff') and not c.isspace())
                    for c in w
                )
                if has_noise:
                    continue
                if not has_cjk and not has_latin:
                    continue
                # 排除单个大写字母开头的缩写
                if has_latin and not has_cjk:
                    if len(w) <= 2 and w.isupper():
                        continue
                seen.add(w)
                clean.append((w, float(s)))
            self.gua8_words[gua] = clean[:20]
        print(f"  蒸馏词典: {len(self.gua8_words)}卦, "
              f"{sum(len(v) for v in self.gua8_words.values())}词")
    
    def _get_gua64_words(self, gua64_idx: int,
                         from_gua: str = None, to_gua: str = None,
                         boost_from: float = 1.0, boost_to: float = 1.0) -> List[Tuple[str, float]]:
        """64卦从上下卦继承蒸馏词，支持 from/to 双方向加权"""
        shang, xia = GUA64_COMPONENTS[gua64_idx]
        words = {}
        for w, s in self.gua8_words.get(shang, []):
            boost = boost_from if shang == from_gua else (boost_to if shang == to_gua else 1.0)
            words[w] = words.get(w, 0.0) + s * 0.55 * boost
        for w, s in self.gua8_words.get(xia, []):
            boost = boost_from if xia == from_gua else (boost_to if xia == to_gua else 1.0)
            words[w] = words.get(w, 0.0) + s * 0.45 * boost
        return sorted(words.items(), key=lambda x: -x[1])
    
    # ============================================================
    # 扩散场 (层1)
    # ============================================================
    
    def _raw_affinity(self, input_gua: str, shang: str, xia: str) -> float:
        def angle_dist(a, b):
            d = abs(a - b)
            return min(d, 360 - d) / 180.0
        d_s = angle_dist(BAGUA_ANGLE[input_gua], BAGUA_ANGLE[shang])
        d_x = angle_dist(BAGUA_ANGLE[input_gua], BAGUA_ANGLE[xia])
        pos_aff = (1.0 - d_s) * 0.5 + (1.0 - d_x) * 0.5
        wx_i = WUXING[input_gua]
        def wuxing_bonus(target):
            wx_t = WUXING[target]
            if WUXING_SHENG.get(wx_t) == wx_i: return 1.5
            elif WUXING_KE.get(wx_i) == wx_t: return 0.3
            return 1.0
        return pos_aff * (wuxing_bonus(shang) + wuxing_bonus(xia)) / 2
    
    def _diffuse(self, qi_state: np.ndarray, temperature: float,
                 trace: list = None) -> np.ndarray:
        """扩散场：qi_state扩散 + 可选轨迹能量注入"""
        radius = 2.0 + temperature * 1.5
        gua64_act = np.zeros(64)
        for idx in range(64):
            shang, xia = GUA64_COMPONENTS[idx]
            energy = 0.0
            for i, g in enumerate(BAGUA):
                if qi_state[i] < 0.01:
                    continue
                raw_aff = self._raw_affinity(g, shang, xia)
                d_s = self._angle_dist(BAGUA_ANGLE[g], BAGUA_ANGLE[shang])
                d_x = self._angle_dist(BAGUA_ANGLE[g], BAGUA_ANGLE[xia])
                avg_dist = (d_s + d_x) / 2
                decay = np.exp(-avg_dist * radius)
                energy += qi_state[i] * raw_aff * decay
            gua64_act[idx] = energy
        
        # ── 轨迹注入：v94内部中间态作为额外能量源 ──
        if trace:
            n_trace = len(trace)
            for ti, tstep in enumerate(trace):
                # 时间衰减：越早的步骤权重越低
                time_weight = (ti + 1) / n_trace * 0.3  # max 0.3
                qc = tstep.get('qichang', {})
                for i, g in enumerate(BAGUA):
                    qi_val = qc.get(g, 0)
                    if qi_val < 0.01:
                        continue
                    # 每个中间态的卦位放一个衰减能量源
                    for idx in range(64):
                        shang, xia = GUA64_COMPONENTS[idx]
                        raw_aff = self._raw_affinity(g, shang, xia)
                        d_s = self._angle_dist(BAGUA_ANGLE[g], BAGUA_ANGLE[shang])
                        d_x = self._angle_dist(BAGUA_ANGLE[g], BAGUA_ANGLE[xia])
                        avg_dist = (d_s + d_x) / 2
                        decay = np.exp(-avg_dist * (radius * 1.3))  # 轨迹源散得更开
                        gua64_act[idx] += qi_val * raw_aff * decay * time_weight
        
        total = gua64_act.sum()
        if total > 1e-10:
            gua64_act = gua64_act / total
        return gua64_act
    
    @staticmethod
    def _angle_dist(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 360 - d) / 180.0
    
    # ============================================================
    # 前向传播
    # ============================================================
    
    def forward(self, qi_state: np.ndarray, deviation: float = 0.0,
                direction: Optional[Tuple[str, str]] = None,
                temperature: float = 1.0,
                physics_winner: str = None,
                trace: list = None) -> Dict:
        # 层1: 8→64（扩散场 + 轨迹注入）
        gua64_activation = self._diffuse(qi_state, temperature, trace=trace)
        
        if direction:
            from_g, to_g = direction
            # 运动即场：偏差驱动加权，一致态(dev≈0)不加权=纯物理
            boost_from = 1.0 + deviation * 1.0
            boost_to   = 1.0 + deviation * 1.15  # 略强，克服上/下卦权重差(0.55 vs 0.45)
            # 在to卦方位注入扩散场能量
            for idx in range(64):
                shang, xia = GUA64_COMPONENTS[idx]
                if shang == from_g and xia == to_g:
                    gua64_activation[idx] += deviation * 0.5
                if xia == from_g and shang == to_g:
                    gua64_activation[idx] += deviation * 0.3
        elif physics_winner:
            from_g = physics_winner
            to_g = None
            boost_from = 2.5  # 一致态 from卦强力加权
            boost_to = 1.0
        else:
            from_g = None
            to_g = None
            boost_from = 1.0
            boost_to = 1.0
        
        if temperature > 1.5:
            gua64_activation = np.power(gua64_activation + 1e-10, 0.7)
        elif temperature < 1.0:
            gua64_activation = np.power(gua64_activation + 1e-10, 1.5)
        gua64_activation = gua64_activation / (gua64_activation.sum() + 1e-10)
        
        # 层2: 64→蒸馏词（双方向加权）
        word_scores = {}
        for idx, act in enumerate(gua64_activation):
            if act < 0.01:
                continue
            for word, w_score in self._get_gua64_words(idx,
                    from_gua=from_g, to_gua=to_g,
                    boost_from=boost_from, boost_to=boost_to):
                word_scores[word] = word_scores.get(word, 0.0) + act * w_score
        
        sorted_gua64 = sorted(
            [(GUA64_NAMES[i], float(gua64_activation[i]))
             for i in range(64) if gua64_activation[i] > 0.01],
            key=lambda x: -x[1])[:5]
        
        sorted_words = sorted(word_scores.items(), key=lambda x: -x[1])[:15]
        
        # 句法线索
        if deviation < 0.2:
            temporal, particles = "静态", ["了", "着"]
        elif deviation < 0.5:
            temporal, particles = "渐变", ["开始", "正在", "渐渐"]
        else:
            temporal, particles = "动态", ["在", "着", "了"]
        
        if direction and deviation > 0.2:
            from_g, to_g = direction
            templates = [f"{from_g}里{to_g}的", f"{from_g}往{to_g}", f"{from_g}中{to_g}"]
        else:
            templates = ["{词}", "{词}了", "{词}着"]
        
        return {
            'top_gua64': sorted_gua64,
            'candidate_words': sorted_words,
            'sentence_hints': {
                'temporal': temporal, 'particles': particles, 'templates': templates,
            },
        }


# ============================================================
# 自然语言生成器
# ============================================================

class QiToLanguage:
    """qi状态 → 自然语言（分歧态: from词 vs to词 刻意对比）"""
    
    def __init__(self, distilled_path: str = None):
        if distilled_path is None:
            distilled_path = os.path.join(os.path.dirname(__file__),
                                          'gua8_bait_distilled_7b.json')
        self.nn = BaguaNeuralNet(distilled_path=distilled_path)
    
    def _split_words_by_gua(self, words, from_gua, to_gua):
        """把候选词按来源卦分组"""
        from_words, to_words, other_words = [], [], []
        word_gua_map = {}  # 缓存
        for gua, gw_list in self.nn.gua8_words.items():
            for gw, gs in gw_list:
                word_gua_map[gw] = gua
        
        for word, score in words:
            g = word_gua_map.get(word)
            if g == from_gua:
                from_words.append((word, score))
            elif to_gua and g == to_gua:
                to_words.append((word, score))
            else:
                other_words.append((word, score))
        return from_words, to_words, other_words
    
    def generate(self, qi_state: np.ndarray, physics_winner: str,
                 deviation: float, prism_annotation: str,
                 temperature: float,
                 trace: list = None,
                 momentum: dict = None) -> str:
        direction = None
        if prism_annotation != "一致" and "→" in prism_annotation:
            parts = prism_annotation.split("→")
            if len(parts) == 2:
                direction = (parts[0], parts[1])
        
        result = self.nn.forward(qi_state, deviation=deviation,
                                 direction=direction, temperature=temperature,
                                 physics_winner=physics_winner,
                                 trace=trace)
        words = result['candidate_words']
        if not words:
            return "…"
        
        if prism_annotation == "一致":
            top = words[0][0]
            if temperature > 2.5:
                return f"很{top}，感觉{top}得不行"
            elif temperature > 1.8:
                return f"挺{top}的"
            elif temperature > 1.0:
                return f"{top}的"
            elif temperature > 0.5:
                return f"有点{top}"
            else:
                return f"淡淡的{top}"
        else:
            # 分歧态：刻意选 from卦词 vs to卦词
            from_g, to_g = direction
            from_words, to_words, other = self._split_words_by_gua(words, from_g, to_g)
            
            w_from = from_words[0][0] if from_words else words[0][0]
            w_to   = to_words[0][0] if to_words else (other[0][0] if other else (words[1][0] if len(words) > 1 else w_from))
            
            if deviation < 0.3:
                patterns = [
                    f"{w_from}中带点{w_to}",
                    f"看起来{w_from}，其实{w_to}",
                    f"表面{w_from}，里面{w_to}",
                ]
            elif deviation < 0.5:
                patterns = [
                    f"{w_from}，但底下{w_to}",
                    f"{w_from}里面藏着{w_to}",
                    f"说是{w_from}，更像{w_to}",
                ]
            elif deviation < 0.7:
                patterns = [
                    f"{w_from}和{w_to}在拉扯",
                    f"{w_from}往{w_to}那边偏",
                    f"{w_from}被{w_to}拽着",
                ]
            else:
                patterns = [
                    f"{w_from}和{w_to}完全相反",
                    f"外{w_from}内{w_to}，很分裂",
                    f"{w_from}但深处是{w_to}",
                ]
            sentence = random.choice(patterns)
            
            # ── 动量调制 ──
            if momentum:
                v = momentum.get('vel_mean', 0)
                a = momentum.get('accel_abs_mean', 0)
                if v > 0.4 and a > 0.15:
                    if random.random() < 0.5:
                        sentence += '，很快'
                elif momentum.get('stable') and v < 0.2:
                    # 稳定慢速：保持原样
                    pass
            
            return sentence


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    distilled_path = os.path.join(os.path.dirname(__file__), 'gua8_bait_distilled_7b.json')
    
    print("=== 蒸馏八卦NN ===\n")
    nn = BaguaNeuralNet(distilled_path=distilled_path)
    
    # 模拟一个巽主导的 qi 状态
    qi = np.array([0.05, 0.05, 0.05, 0.10, 0.40, 0.15, 0.10, 0.10])
    
    print("\n--- 一致（低偏角） ---")
    r = nn.forward(qi, deviation=0.15, direction=None, temperature=1.2)
    print(f"  64卦: {r['top_gua64']}")
    print(f"  候选词: {[w for w,s in r['candidate_words'][:8]]}")
    
    print("\n--- 巽→坤（高分歧） ---")
    r = nn.forward(qi, deviation=0.60, direction=('巽', '坤'), temperature=1.5)
    print(f"  64卦: {r['top_gua64']}")
    print(f"  候选词: {[w for w,s in r['candidate_words'][:8]]}")
    
    print("\n=== 自然语言生成 ===")
    gen = QiToLanguage(distilled_path=distilled_path)
    
    tests = [
        (qi, "巽", 0.15, "一致", 1.2),
        (qi, "巽", 0.60, "巽→坤", 1.5),
        (qi, "兑", 0.45, "兑→巽", 1.0),
    ]
    
    for qs, pw, dev, ann, temp in tests:
        s = gen.generate(qs, pw, dev, ann, temp)
        print(f"  {ann} (偏{dev:.2f}/温{temp:.1f}): {s}")
