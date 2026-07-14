"""
╔══════════════════════════════════════════════════════════════╗
║  嘴巴 v8 — 共用组件（版本 A / B 都调用）                  ║
║                                                            ║
║  不依赖河图也不依赖汉字物理。                              ║
║  输入 qi_state（8维）→ 先天八卦环推演 → 双输出。         ║
║                                                            ║
║  对照 → ../ARCHITECTURE_VERSIONS.md                        ║
╚══════════════════════════════════════════════════════════════╝

嘴巴 v8 — 双输出：洞察回应 + 日常共情

架构哲学：
  不是一个算卦模拟器。是一个会思考的朋友。
  推演路径不输出为卦辞——输出为有实质洞见的自然回应。
  算卦智慧内化在系统里，不在前台展示。

双输出：
  1. insight（洞察）：推演路径 → 有内容的建议/判断
  2. chatter（唠嗑）：推演路径 → 日常共情/陪伴
"""
import numpy as np, json, os, hashlib
from collections import defaultdict
from typing import List, Dict

BAGUA = ['乾', '兑', '离', '震', '坤', '艮', '坎', '巽']
RI = [0, 1, 2, 3, 4, 5, 6, 7]

# ═══════════════════════════════════
# 洞察回应模板
# ═══════════════════════════════════
INSIGHT = {
    ('乾', +1): ["方向没问题，放手去做。", "时机到了，不用犹豫。", "该你上场了，自信点。"],
    ('乾', 0):  ["现在这样就挺好，稳住。", "守得住就是赢。", "不急，已经在正轨上了。"],
    ('乾', -1): ["稍微收一收，太刚容易折。", "别冲太猛，慢一点更稳。", "有力量是好事，但别全使出去。"],
    ('兑', +1): ["说出来就好了，有人在听。", "聊聊吧，话说开就通了。", "你的感受很重要，别憋着。"],
    ('兑', 0):  ["开心就好，不用想太多。", "就这样，挺好的。", "说够了就停，刚刚好。"],
    ('兑', -1): ["少说两句，听一听别人。", "别太急着表达，先感受。", "话说多了反而乱了。"],
    ('离', +1): ["看清楚了就往前走。", "心里有数了，可以行动了。", "你的直觉是对的，信它。"],
    ('离', 0):  ["已经看得很清楚了，不急。", "保持光亮，但别烧着自己。", "看清了就好，不用急着做。"],
    ('离', -1): ["火太旺了，降降温。", "别太激动，冷静一下再看。", "热情是好的，但别灼伤自己。"],
    ('震', +1): ["变动是机会，接住它。", "别怕突发的事，后面有好东西。", "意外来了也别慌，站稳就行。"],
    ('震', 0):  ["先稳住，别被吓到。", "等等再看，现在不是动的时候。", "惊了一下没关系，缓缓。"],
    ('震', -1): ["别被带着跑，定一定。", "太快了，先停下来。", "激动过了，现在该冷静了。"],
    ('坤', +1): ["承受住了就是你的力量。", "不用争，包容本身就是赢。", "你接得住，那就没问题。"],
    ('坤', 0):  ["就这样承受着也行，不急变。", "安安静静的，挺好的。", "不用急着改变什么。"],
    ('坤', -1): ["别什么都自己扛。", "太包容了反而累，放一放。", "厚德没错，但别过度。"],
    ('艮', +1): ["停一下是为了看清楚。", "不急，等一等自然就明白了。", "止步不是退，是蓄力。"],
    ('艮', 0):  ["现在就该停下来。", "别动了，安静待着。", "够了，不用再想了。"],
    ('艮', -1): ["该开口了，别闷着。", "停太久了，动一动吧。", "憋着不好，说出来。"],
    ('坎', +1): ["深处难明，但水总会清的。", "现在是难，但往下走就能浮起来。", "别怕深水区，顺流就能过。"],
    ('坎', 0):  ["现在看不清就先等着。", "深水不急，自然会清。", "不太确定的话，先不动。"],
    ('坎', -1): ["别陷进去，抬头看看。", "越想越深，先出来透口气。", "水太深了，换个方向。"],
    ('巽', +1): ["慢慢来，春风总会吹到的。", "不着急，好事在后面。", "徐徐前进，比猛冲更稳。"],
    ('巽', 0):  ["就这样温柔的，挺好的。", "不急不徐，你做得对。", "轻轻推一推，不用大力。"],
    ('巽', -1): ["风要进来了，别关着。", "太慢了，稍微加快一点。", "放开一点，别太小心翼翼。"],
}

# ═══════════════════════════════════
# 唠嗑短语
# ═══════════════════════════════════
CHATTER = {
    '乾': ['没问题的', '就是这样的', '可以', '行得通', '往前走', '没错', '放心', '撑得住'],
    '兑': ['对吧', '嗯嗯', '说出来就好', '聊聊吧', '告诉我', '太好了', '好的', '听你的'],
    '离': ['看到了', '我明白', '清楚了', '原来是这样的', '往前看', '会好的', '明白了', '是这样啊'],
    '震': ['真的吗', '天啊', '怎么会', '好意外', '不敢相信', '等等', '让我缓缓', '居然'],
    '坤': ['没关系的', '我可以', '就这样吧', '接受了', '包容', '也行', '随它去吧', '好吧'],
    '艮': ['停一下', '等等', '不急', '想一想', '安静', '不说了', '够了', '算了吧'],
    '坎': ['不太好', '有点怕', '不太确定', '不会吧', '担心', '想不通', '难', '有点悬'],
    '巽': ['慢慢来', '不着急', '轻轻的', '温柔点', '等一等', '随他去', '自然就好', '从容'],
}


# ═══════════════════════════════════
# 环推演引擎（保持不变）
# ═══════════════════════════════════

def divine_path(qi_state: np.ndarray, input_text: str = "") -> dict:
    """从 qi_state 推演先天八卦环上的路径。"""
    ri_qi = np.array([qi_state[i] for i in RI])
    main_idx = int(np.argmax(ri_qi))
    main_gua = BAGUA[RI[main_idx]]
    
    if main_idx > 0:   prev_qi = ri_qi[main_idx - 1]
    else:              prev_qi = ri_qi[-1]
    if main_idx < 7:   next_qi = ri_qi[main_idx + 1]
    else:              next_qi = ri_qi[0]
    
    if next_qi > prev_qi + 0.02:     direction = +1
    elif prev_qi > next_qi + 0.02:   direction = -1
    else:                             direction = 0
    
    # SHA256 扰动：同卦不同输入 → 不同方向/步数
    if input_text:
        h = hashlib.sha256(input_text.encode()).digest()
        if h[0] < 85 and direction == 0:           direction = -1
        elif h[0] > 170 and direction == 0:         direction = +1
        steps = 1 if h[1] < 128 else 2
    else:
        steps = 2
    
    path_guas = []
    if direction != 0:
        for step in range(1, steps + 1):
            pos = (main_idx + direction * step) % 8
            path_guas.append(BAGUA[RI[pos]])
    
    return {
        'main_gua': main_gua,
        'path_guas': path_guas,
        'direction': direction,
    }


# ═══════════════════════════════════
# 嘴巴类
# ═══════════════════════════════════

class Mouth:
    """双输出嘴巴：洞察 + 唠嗑"""
    
    BOOST = 0.04
    DECAY = 0.995
    
    def __init__(self, state_path: str = None):
        self.anchor_force: Dict[str, float] = {}
        for phrases in CHATTER.values():
            for p in phrases:
                self.anchor_force[p] = 0.35
        
        self.total_speaks = 0
        self.state_path = state_path
        if state_path and os.path.exists(state_path):
            self._load_state(state_path)
        
        print(f"嘴巴 v8: 洞察{sum(len(v) for v in INSIGHT.values())}句 + 唠嗑{sum(len(v) for v in CHATTER.values())}短语")
    
    def _load_state(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                s = json.load(f)
            self.anchor_force.update(s.get('anchor_force', {}))
            self.total_speaks = s.get('total_speaks', 0)
        except: pass
    
    def _save_state(self):
        if not self.state_path: return
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump({'anchor_force': self.anchor_force,
                       'total_speaks': self.total_speaks}, f, ensure_ascii=False, indent=2)
    
    # ─── 洞察回应 ───
    
    def _speak_insight(self, path_info: dict, input_text: str = "") -> str:
        """推演路径 → 有内容的洞察回应"""
        main = path_info['main_gua']
        direction = path_info['direction']
        key = (main, direction)
        
        templates = INSIGHT.get(key, ["嗯。"])
        
        # SHA256 确定性选一句
        h_input = f"{input_text}_insight_{self.total_speaks}"
        h = hashlib.sha256(h_input.encode()).digest()
        idx = h[0] % len(templates)
        return templates[idx]
    
    # ─── 唠嗑回应 ───
    
    def _speak_chatter(self, path_info: dict, input_text: str = "") -> str:
        """推演路径 → 日常唠嗑"""
        main = path_info['main_gua']
        path = path_info['path_guas']
        
        picked = []
        used = set()
        gua_sequence = [main] + path[:1]
        if not path:
            main_ri = RI.index(BAGUA.index(main))
            neighbor = BAGUA[RI[(main_ri + 1) % 8]]
            gua_sequence.append(neighbor)
        
        for i, gua in enumerate(gua_sequence):
            candidates = [w for w in CHATTER.get(gua, []) if w not in used]
            if not candidates:
                all_cands = []
                for g in [main] + path:
                    all_cands.extend([w for w in CHATTER.get(g, []) if w not in used])
                candidates = all_cands
                if not candidates: continue
            
            h_input = f"{input_text}_{i}_{main}_chatter_{self.total_speaks}"
            h = hashlib.sha256(h_input.encode()).digest()
            
            candidates.sort()
            top_n = max(2, len(candidates) // 2)
            top = candidates[:top_n]
            idx = h[0] % len(top)
            picked.append(top[idx])
            used.add(top[idx])
        
        if not picked:
            return "嗯。"
        
        dialogue = ''.join(picked)
        
        # 语气后缀
        tone = {'乾': '！', '兑': '~', '离': '！', '震': '！',
                '坤': '…', '艮': '…', '坎': '…', '巽': '~'}[main]
        dialogue += tone
        
        return dialogue
    
    # ─── 学习 ───
    
    def learn_from_usage(self, chatter_words: List[str]):
        for w, f in self.anchor_force.items():
            if w in chatter_words:
                self.anchor_force[w] = max(0.05, f * self.DECAY)
            else:
                self.anchor_force[w] = min(1.0, f + self.BOOST)
        self.total_speaks += 1
        if self.total_speaks % 10 == 0:
            self._save_state()
    
    # ─── 统一入口 ───
    
    def speak(self, qi_state: np.ndarray, input_text: str = "", crystal_words: List[str] = None) -> dict:
        """
        嘴巴统一入口。
        
        Args:
            qi_state: 8维物理场状态
            input_text: 原始输入文本
            crystal_words: 太极演化结晶词（可选，如果提供则使用）
        
        Returns:
            dict: 包含 insight, chatter, full, main_gua, direction
        """
        self.total_speaks += 1
        path_info = divine_path(qi_state, input_text)
        
        # 如果有结晶词，使用结晶词组装主要内容
        if crystal_words and len(crystal_words) > 0:
            insight = self._speak_from_crystal(crystal_words, path_info)
        else:
            # 没有结晶词，使用传统模板
            insight = self._speak_insight(path_info, input_text)
        
        chatter = self._speak_chatter(path_info, input_text)
        
        self.learn_from_usage([w for w in self.anchor_force if w in chatter])
        
        return {
            'insight': insight,
            'chatter': chatter,
            'full': f"{insight}\n{chatter}",
            'main_gua': path_info['main_gua'],
            'direction': path_info['direction'],
            'crystal_words': crystal_words or [],
        }
    
    def _speak_from_crystal(self, crystal_words: List[str], path_info: dict) -> str:
        """
        从结晶词组装自然文本。
        
        太极演化的核心：词从热场中浮现，不是从模板匹配。
        """
        if not crystal_words:
            return "嗯。"
        
        # 简单模板：词1，词2，词3...
        # 或者更复杂的句式生成
        main = path_info['main_gua']
        
        if len(crystal_words) >= 3:
            # 3个词：用逗号连接
            content = f"{crystal_words[0]}，{crystal_words[1]}，{crystal_words[2]}"
        elif len(crystal_words) == 2:
            content = f"{crystal_words[0]}，{crystal_words[1]}"
        else:
            content = crystal_words[0]
        
        # 语气后缀
        tone = {'乾': '！', '兑': '~', '离': '！', '震': '！',
                '坤': '…', '艮': '…', '坎': '…', '巽': '~'}[main]
        
        return content + tone
