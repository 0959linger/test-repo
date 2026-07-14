"""
语言偶极子场 v2 (Language Qichang)

核心机制：
  - 3584维 embedding → 8个蒸馏锚点 → 任意词的 qi 分布
  - 句子生成 = 场状态匹配 + 语法约束
  - 0 参数，纯物理路由

输入：qi_state + 蒸馏词候选 + 对话历史
输出：自然中文句子
"""

import random
import numpy as np
import sys
from typing import Dict, List, Tuple, Optional


# ============================================================
# 八卦语法映射
# ============================================================

BAGUA = ['乾', '兑', '离', '震', '巽', '坎', '艮', '坤']

# 卦 → 句法模式（何卦主导时倾向什么句子结构）
GUA_SYNTAX = {
    '乾': {'pattern': '断言',    'terminal': '。',  'tone': ''},
    '兑': {'pattern': '感叹',    'terminal': '啊',  'tone': '吧'},
    '离': {'pattern': '描述',    'terminal': '的',  'tone': '呢'},
    '震': {'pattern': '行动',    'terminal': '！',  'tone': '吗'},
    '巽': {'pattern': '柔和陈述','terminal': '…',  'tone': '吧'},
    '坎': {'pattern': '内敛',    'terminal': '…',  'tone': ''},
    '艮': {'pattern': '肯定',    'terminal': '。',  'tone': '吧'},
    '坤': {'pattern': '接纳',    'terminal': '。',  'tone': '了'},
}

# 对话策略模板
STRATEGY_TEMPLATES = {
    '共鸣': [
        "嗯，{state}的", "我懂，{state}", "{state}……我也是",
        "对，{state}", "是啊，{state}",
    ],
    '追问': [
        "然后呢？", "{state}？", "什么意思", "怎么说",
        "接着说",
    ],
    '推进': [
        "继续", "说说看", "我听着呢",
        "{state}，然后呢",
    ],
    '回避': [
        "……", "是么", "嗯",
        "不说这个了", "换一个话题？",
    ],
    '确认': [
        "你说{state}？", "是这样么？", "真的{state}？",
    ],
    '总结': [
        "就是说{state}了", "所以{state}", "那{state}吧",
    ],
    '回应': [
        "嗯", "好", "明白了", "知道了",
    ],
}

# 句法模板（按模式分组）
SYNTAX_TEMPLATES = {
    '断言':      ['是{state}的', '{state}', '就是{state}'],
    '感叹':      ['真{state}啊', '好{state}', '{state}呢'],
    '描述':      ['挺{state}的', '很{state}', '有点{state}'],
    '行动':      ['{state}！', '太{state}了', '真{state}'],
    '柔和陈述':  ['{state}的', '感觉{state}', '好像{state}'],
    '内敛':      ['{state}…', '有点{state}…', '{state}，但…'],
    '肯定':      ['确实{state}', '很{state}了', '{state}'],
    '接纳':      ['{state}吧', '那就{state}', '{state}了'],
}


# ============================================================
# 词→qi映射引擎
# ============================================================

class WordToQi:
    """
    将任意词的embedding映射到八卦qi分布。
    机制：词向量 → 8个蒸馏锚点的余弦距离 → qi分布。
    0 参数，纯几何。
    """
    
    def __init__(self):
        self.anchors = {}  # {卦: 锚点向量}
        self._word_cache = {}  # {词: qi分布}
    
    def set_anchors(self, anchors: Dict[str, np.ndarray]):
        """设置8个蒸馏锚点"""
        self.anchors = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in anchors.items()}
        self._word_cache = {}
    
    def word_to_qi(self, word_vec: np.ndarray) -> np.ndarray:
        """单个词向量 → qi分布[8]"""
        if not self.anchors:
            return np.ones(8) / 8
        
        vec_norm = word_vec / (np.linalg.norm(word_vec) + 1e-8)
        qi = np.zeros(8)
        for i, gua in enumerate(BAGUA):
            anchor = self.anchors.get(gua)
            if anchor is not None:
                cos_sim = np.dot(vec_norm, anchor)
                qi[i] = max(0, cos_sim)  # 只取正向相似度
        
        total = qi.sum()
        if total > 1e-10:
            qi = qi / total
        else:
            qi = np.ones(8) / 8
        return qi
    
    def get_word_qi(self, word: str, word_embed: Dict[str, np.ndarray]) -> np.ndarray:
        """获取词的qi分布（带缓存）"""
        if word in self._word_cache:
            return self._word_cache[word]
        
        if word in word_embed:
            qi = self.word_to_qi(word_embed[word])
        else:
            qi = np.ones(8) / 8
        
        self._word_cache[word] = qi
        return qi
    
    def match_score(self, qi_state: np.ndarray, word_qi: np.ndarray) -> float:
        """
        词与当前场状态的匹配度。
        使用余弦相似度（不是点积）以确保方向匹配而非大小匹配。
        """
        cos = np.dot(qi_state, word_qi) / (
            np.linalg.norm(qi_state) * np.linalg.norm(word_qi) + 1e-8)
        return float(cos)


# ============================================================
# 语言生成器
# ============================================================

class LanguageQichang:
    """qi状态 → 自然语言"""
    
    def __init__(self, distill_anchors: Dict[str, np.ndarray] = None):
        self.w2q = WordToQi()
        if distill_anchors:
            self.w2q.set_anchors(distill_anchors)
        
        self._word_embed = {}
        self._debug = False  # 调试开关
    
    def set_word_embed(self, word_embed: Dict[str, np.ndarray]):
        self._word_embed = word_embed
    
    def _best_word(self, qi_state: np.ndarray, candidates: List[str],
                   n: int = 1) -> List[str]:
        """从候选词中选和场状态最匹配的词"""
        if not candidates:
            return []
        
        scored = []
        for w in candidates:
            wqi = self.w2q.get_word_qi(w, self._word_embed)
            score = self.w2q.match_score(qi_state, wqi)
            scored.append((w, score))
        
        scored.sort(key=lambda x: -x[1])
        return [w for w, s in scored[:n]]
    
    def select_strategy(self, deviation: float,
                        context: dict = None) -> str:
        """
        从MemoryLayer上下文决定对话策略。
        引擎不维护策略状态——所有状态从上层MemoryLayer读取。
        
        context 必须包含: strategy_fatigue, gua_cooling, cooling_high, cooling_max
        """
        fatigue = context.get('strategy_fatigue', False) if context else False
        cooling_high = context.get('cooling_high', False) if context else False
        cooling_max = context.get('cooling_max', False) if context else False
        last_strategy = context.get('prev', {}).get('strategy', '描述') if context else '描述'
        strategy_streak = context.get('strategy_streak', 1) if context else 1
        
        strategy = '描述'
        if deviation > 0.55:
            if cooling_max:
                strategy = '回避'
            elif fatigue and last_strategy == '确认':
                strategy = '回避'
            elif strategy_streak >= 2 and last_strategy == '确认':
                strategy = '总结'
            else:
                strategy = '确认'
        elif deviation > 0.35:
            if cooling_high and last_strategy != '追问':
                strategy = '追问'
            elif fatigue and last_strategy == '描述':
                strategy = '追问'
            else:
                strategy = '描述'
        else:
            if cooling_max:
                strategy = '推进'
            elif fatigue:
                if last_strategy == '描述':
                    strategy = '推进'
                elif last_strategy == '共鸣':
                    strategy = '追问'
                elif last_strategy == '推进':
                    strategy = '总结'
            elif strategy_streak >= 2 and last_strategy == '描述':
                strategy = '共鸣'
            else:
                strategy = '描述'
        
        return strategy
    
    def _gen_strategy(self, qi_state, dominant_gua, deviation, temperature,
                      prism_annotation, from_state_words, to_state_words,
                      strategy: str) -> str:
        """用策略模板生成回答"""
        templates = STRATEGY_TEMPLATES.get(strategy, ['{state}'])
        
        # 获取状态词
        state_word = ""
        if prism_annotation == "一致":
            candidates = from_state_words or []
            best = self._best_word(qi_state, candidates, n=2)
            state_word = best[0] if best else ""
        else:
            from_best = self._best_word(qi_state, from_state_words or [], n=1)
            to_best = self._best_word(qi_state, to_state_words or [], n=1)
            
            if deviation > 0.55 and from_best and to_best:
                # 强分歧：策略模板里用拉扯表达
                fw, tw = from_best[0], to_best[0]
                state_word = f"{fw}又{tw}"
            elif from_best:
                state_word = from_best[0]
            else:
                state_word = to_best[0] if to_best else ""
        
        # 策略追问/确认——不需要状态词的模板
        no_state_templates = [t for t in templates if '{state}' not in t]
        with_state_templates = [t for t in templates if '{state}' in t]
        
        if strategy in ('追问', '推进') and no_state_templates and random.random() < 0.6:
            # 60%概率用纯策略模板（不需要状态词）
            template = random.choice(no_state_templates)
            return template
        elif state_word and with_state_templates:
            template = random.choice(with_state_templates)
            return template.format(state=state_word)
        elif state_word:
            return state_word
        elif no_state_templates:
            return random.choice(no_state_templates)
        else:
            return ""
    
    def generate(self, qi_state: np.ndarray,
                 dominant_gua: str,
                 deviation: float = 0.0,
                 temperature: float = 1.0,
                 prism_annotation: str = "一致",
                 from_state_words: List[str] = None,
                 to_state_words: List[str] = None,
                 extremeness: float = 0.0,
                 context: dict = None) -> str:
        """
        生成自然语言句子。策略状态从MemoryLayer读取。
        
        qi_state: [8] qi分布
        dominant_gua: 主导卦名
        deviation: 偏角
        temperature: 场温
        prism_annotation: 棱镜旁注
        from_state_words: from卦蒸馏词
        to_state_words: to卦蒸馏词（分歧态）
        extremeness: 极端度
        context: MemoryLayer返回的上下文
        """
        
        # ── 策略选择（从MemoryLayer读状态）──
        strategy = self.select_strategy(deviation, context)
        
        # 策略→句法映射：追问/确认/回避/共鸣 用策略模板，其余用原句法模板
        if strategy in ('追问', '确认', '回避', '总结', '共鸣', '推进', '回应'):
            # 用策略模板生成
            return self._gen_strategy(
                qi_state, dominant_gua, deviation, temperature,
                prism_annotation, from_state_words, to_state_words, strategy,
            )
        
        # 默认：描述模式（原逻辑）
        syntax = GUA_SYNTAX.get(dominant_gua, GUA_SYNTAX['坤'])
        pattern = syntax['pattern']
        
        # ── 选状态词 ──
        if prism_annotation == "一致":
            # 一致态：用主导卦的词，选匹配度最高的
            candidates = from_state_words if from_state_words else []
            best = self._best_word(qi_state, candidates, n=3)
            
            # 加入场温后再次筛选
            if temperature < 0.8:
                state_word = best[0] if best else "的"
            elif temperature > 2.0:
                state_word = random.choice(best) if best else "的"
            else:
                state_word = best[0] if best else "的"
        else:
            # 分歧态：from词 + to词 混合
            from_best = self._best_word(qi_state, from_state_words or [], n=2)
            to_best = self._best_word(qi_state, to_state_words or [], n=2)
            
            # 按偏差决定用哪种结构
            if deviation < 0.4:
                # 轻微分歧：from词为主
                state_word = from_best[0] if from_best else (to_best[0] if to_best else "…")
            elif deviation < 0.6:
                # 中等分歧：from词 + 但 + to倾向
                fw = from_best[0] if from_best else ""
                tw = to_best[0] if to_best else ""
                t = syntax['terminal']
                if fw and tw:
                    template = random.choice([
                        f"{fw}，但{tw}{t}",
                        f"{fw}里有点{tw}",
                        f"表面{fw}，其实{tw}",
                    ])
                    return template
                state_word = fw or tw or "…"
            else:
                # 强分歧：from词和to词并存
                fw = from_best[0] if from_best else ""
                tw = to_best[0] if to_best else ""
                if fw and tw:
                    template = random.choice([
                        f"{fw}和{tw}在拉扯",
                        f"{fw}往{tw}偏过去了",
                        f"{fw}被{tw}拽着",
                    ])
                    return template
                state_word = fw or tw or "…"
        
        # ── 套模板 ──
        templates = SYNTAX_TEMPLATES.get(pattern, ['{state}'])
        template = random.choice(templates)
        sentence = template.format(state=state_word)
        
        return sentence


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    import sys, io, os, json
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    from transformers import AutoTokenizer
    
    # ── 加载数据 ──
    print("Loading...")
    embed = np.load('data/qwen7b_embed_tokens.npy')
    MODEL_DIR = 'C:/Users/ww109/.cache/modelscope/Qwen/Qwen2___5-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    distilled = np.load('data/distilled_7b.npz')
    
    # 设置蒸馏锚点
    anchors = {}
    for gua in BAGUA:
        key = f'{gua}_vec'
        if key in distilled:
            anchors[gua] = distilled[key]
    
    lq = LanguageQichang(distill_anchors=anchors)
    
    # 加载蒸馏词
    with open('engine/gua8_bait_distilled_7b.json', 'r', encoding='utf-8') as f:
        dd = json.load(f)
    
    # 构建词嵌入表
    from language_qichang import SYNTAX_TEMPLATES
    all_words = set()
    for gua, words in dd['gua_to_words'].items():
        all_words.update(w for w, s in words[:15])
    
    word_embed = {}
    for word in all_words:
        tids = tokenizer.encode(word, add_special_tokens=False)
        vecs = [embed[tid] for tid in tids if tid < len(embed)]
        if vecs:
            word_embed[word] = np.mean(vecs, axis=0)
    lq.set_word_embed(word_embed)
    
    print(f"Ready: {len(all_words)} words, {len(anchors)} anchors\n")
    
    # ── 测试 ──
    tests = [
        {
            'label': '巽一致',
            'qi': np.array([0.05, 0.05, 0.10, 0.05, 0.40, 0.10, 0.10, 0.15]),
            'gua': '巽',
            'from': ['顺', '温和', '柔软', '飘', '风', '渗透', '细致'],
            'to': [],
            'dev': 0.15, 'temp': 1.2, 'ann': '一致',
        },
        {
            'label': '巽→坤',
            'qi': np.array([0.05, 0.05, 0.05, 0.10, 0.35, 0.10, 0.10, 0.20]),
            'gua': '巽',
            'from': ['顺', '风', '渗透', '飘', '温和', '柔软'],
            'to': ['柔', '承载', '包容', '大地', '从', '接纳'],
            'dev': 0.65, 'temp': 1.8, 'ann': '巽→坤',
        },
        {
            'label': '乾→震',
            'qi': np.array([0.30, 0.10, 0.10, 0.25, 0.05, 0.05, 0.10, 0.05]),
            'gua': '乾',
            'from': ['刚', '强', '力量', '阳', '主动', '创造'],
            'to': ['惊', '震动', '激动', '轰', '撼', '冲击'],
            'dev': 0.55, 'temp': 2.0, 'ann': '乾→震',
        },
        {
            'label': '艮一致',
            'qi': np.array([0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.40, 0.20]),
            'gua': '艮',
            'from': ['静', '稳定', '停止', '坚固', '凝', '阻挡'],
            'to': [],
            'dev': 0.10, 'temp': 0.8, 'ann': '一致',
        },
    ]
    
    for t in tests:
        print(f"  [{t['label']}] dev={t['dev']} temp={t['temp']}")
        for run in range(3):
            s = lq.generate(
                qi_state=t['qi'],
                dominant_gua=t['gua'],
                deviation=t['dev'],
                temperature=t['temp'],
                prism_annotation=t['ann'],
                from_state_words=t['from'],
                to_state_words=t['to'],
            )
            print(f"    run {run+1}: {s}")
        print()
