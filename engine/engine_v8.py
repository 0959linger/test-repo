"""
╔══════════════════════════════════════════════════════════════╗
║  版本 B：engine_v8 — 纯物理引擎（不含河图）               ║
║                                                            ║
║  三条感知路径：                                            ║
║    ① 汉字物理 → qi_physics （进 v94）                    ║
║    ② 棱镜     → qi_prism   （只算偏角，不进 v94）        ║
║    ③ 查表     → qi_lookup  （未使用）                     ║
║                                                            ║
║  v94 只吃汉字物理一条信号。                                ║
║  河图、记忆层、近窗微风、C层推理、呼吸回路 — 均未接入。   ║
║                                                            ║
║  对照 → ARCHITECTURE_VERSIONS.md                           ║
╚══════════════════════════════════════════════════════════════╝

引擎核心 v8：多偶极子（场自决）

  文本
  ├─ 物理 → qi_A
  ├─ 棱镜 → qi_B  
  └─ 查表 → qi_C
  
  三条 qi 独立注入 v94 → 多偶极子互推 → 场自决收敛
  不做加权平均。Δ = 多路自然分歧（v94 内部计算）。
"""
import math, numpy as np, re, sys, os
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, _src_dir)
from PIL import Image, ImageDraw, ImageFont
from v94_qichang.core_enhanced import V94QichangEnhanced
from embed_prism_v3 import EmbedPrismV3

BAGUA = ['乾','兑','离','震','坤','艮','坎','巽']
BAGUA_ANGLE = {
    '乾': 0, '兑': 45, '离': 90, '震': 135,
    '坤': 180, '艮': 225, '坎': 270, '巽': 315,
}
FONT = None

def _get_font():
    global FONT
    if FONT is None:
        FONT = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 128)
    return FONT

# ============================================================
# 汉字物理
# ============================================================
def hanzi_physics(text):
    chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    n = len(chars)
    if n < 2: return None, None, 0.1, 0.0
    codes = np.array([(ord(c) - 0x4E00) / (0x9FFF - 0x4E00) * 2 - 1 for c in chars])
    d1 = np.median([(codes[i+1] - codes[i]) * 180 for i in range(n-1)]) % 360
    if n >= 3:
        m, s = np.mean(codes), np.std(codes) or 1e-10
        d3 = (np.mean(((codes - m) / s) ** 3) * 90 + 180) % 360
    else: d3 = d1
    font = _get_font()
    densities = []
    for c in chars[:30]:
        try:
            img = Image.new('L', (128, 128), 255)
            d = ImageDraw.Draw(img)
            bb = d.textbbox((0, 0), c, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            x, y = (128 - tw) // 2 - bb[0], (128 - th) // 2 - bb[1]
            d.text((x, y), c, fill=0, font=font)
            densities.append(np.mean(np.array(img, dtype=np.float32) < 128))
        except: pass
    arr = np.array(densities) if densities else np.array([0.1])
    return d1, d3, float(np.mean(arr)), float(np.std(arr))

def digit_angle(text):
    digits = re.findall(r'[0-9]', text)
    if not digits: return None, 0
    codes = [(ord(d) - 0x30) / 9.0 for d in digits]
    angle = np.median([(codes[i+1]-codes[i])*180 for i in range(len(codes)-1)])%360 if len(codes)>=2 else codes[0]*360
    return angle, min(len(digits)*0.15, 0.6)

def punct_angle(text):
    MAP = {'？':0,'?':0,'！':90,'!':90,'，':180,',':180,'。':270,'.':270}
    puncts = [c for c in text if c in MAP]
    if not puncts: return None, 0
    angles = [MAP[c] for c in puncts]
    angle = (np.median(np.diff(angles))+360)%360 if len(angles)>=2 else angles[0]
    return angle, min(len(puncts)*0.15, 0.6)

def time_membrane(hour):
    phase = (hour/24.0)*2*math.pi
    return np.array([1.0+math.cos(phase-i*math.pi/4)*0.08 for i in range(8)])

# ═══ 物理空间提取 ═══
# 句式 / 人称 / 时态 / 长度 → 8维偏置
def extract_space(text):
    bias = np.zeros(8)
    
    # 句式
    if text.rstrip().endswith(('？', '?')):
        d = {'兑': 0.8, '离': 0.5, '巽': 0.3}
    elif text.rstrip().endswith(('！', '!')):
        d = {'兑': 0.9, '震': 0.6, '离': 0.3}
    elif re.search(r'请|别|不要|来|去|做|别再说', text):
        d = {'震': 0.8, '乾': 0.5}
    else:
        d = {'艮': 0.5, '坤': 0.5}
    for gua, w in d.items(): bias[BAGUA.index(gua)] += w
    
    # 人称
    if re.search(r'(?<!们)我(?!们)', text): d = {'坤': 0.7, '坎': 0.5, '艮': 0.2}
    elif '我们' in text: d = {'兑': 0.7, '坤': 0.5, '离': 0.3}
    elif '你' in text or '您' in text: d = {'乾': 0.7, '离': 0.5, '兑': 0.2}
    elif re.search(r'他|她|它|他们', text): d = {'艮': 0.6, '坎': 0.6, '坤': 0.2}
    else: d = {}
    for gua, w in d.items(): bias[BAGUA.index(gua)] += w
    
    # 时态
    if re.search(r'曾经|过去|以前|上次|那时|当初', text):
        d = {'坎': 0.7, '艮': 0.5}
    elif re.search(r'正在|在.*[着呢]|目前|当下|现在', text):
        d = {'离': 0.7, '震': 0.3}
    elif re.search(r'将要|以后|未来|明天|下次|会.*吗|会不会', text):
        d = {'乾': 0.6, '巽': 0.6, '震': 0.3}
    else: d = {}
    for gua, w in d.items(): bias[BAGUA.index(gua)] += w
    
    # 长度
    L = len(text)
    if L <= 4: d = {'艮': 0.6, '坤': 0.4}
    elif L <= 12: d = {}
    else: d = {'巽': 0.5, '兑': 0.4}
    for gua, w in d.items(): bias[BAGUA.index(gua)] += w
    
    # 规范化
    s = bias.sum()
    if s > 0:
        bias = bias / s - 1.0 / 8
    return bias

class Persona:
    def __init__(self, hour=None):
        self.lens = np.ones(8)*0.125; self.steps = 0
        self._tm = time_membrane(hour) if hour is not None else np.ones(8)
    def view(self, raw_qi):
        if self.steps==0: weighted = raw_qi.copy()
        else:
            dev = self.lens-0.125; res = 1.0+dev*4.0
            weighted = raw_qi*np.clip(res,0.3,1.7)
        weighted = weighted*self._tm
        return weighted/weighted.sum()*raw_qi.sum()
    def update(self, qi):
        rates = [0.5,0.3,0.2,0.15]
        self.lens = self.lens*(1-rates[min(self.steps,3)])+qi*rates[min(self.steps,3)]
        self.steps += 1
    def cv(self):
        m,s = np.mean(self.lens),np.std(self.lens)
        return s/m if m>0 else 0
    def dominant(self):
        return BAGUA[np.argmax(self.lens)], self.lens[np.argmax(self.lens)]

# ============================================================
# 引擎 v8：多偶极子
# ============================================================
class Engine:
    def __init__(self, hour=None, inertia=0.35):
        self.v94 = V94QichangEnhanced()
        self.persona = Persona(hour=hour)
        self.prev_qi = None
        self.inertia = inertia
        self.round = 0
        self.hour = hour
        
        self._prism = None
        self._embed = None
        self._word_to_vec = None
    
    def _ensure_backend(self):
        if self._prism is not None: return
        from gguf import GGUFReader
        path = r"C:\Users\ww109\.qwenpaw\llama.cpp\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
        reader = GGUFReader(path)
        embed = None
        for t in reader.tensors:
            if t.name == 'token_embd.weight':
                embed = t.data.astype(np.float32); break
        
        byte_to_tid = {i: i for i in range(256)}
        f = reader.fields['tokenizer.ggml.tokens']; parts = f.parts
        decode_map, tid = {}, 0; idx = 5
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
                        tids.append(bpe_index[sub]); break
            return embed[tids].mean(axis=0).astype(np.float64) if tids else np.zeros(952)
        
        self._prism = EmbedPrismV3()
        self._embed = embed
        self._word_to_vec = word_to_vec
        self._emb_norm = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-10)
        self._byte_to_tid = byte_to_tid
        self._bpe_index = bpe_index
        self._decode_map = decode_map
    
    def _lookup_neighbors(self, text, top_k=5):
        utf8 = text.encode('utf-8')
        tids = [self._byte_to_tid[b] for b in utf8]
        for start in range(len(utf8)):
            for end in range(len(utf8), start, -1):
                sub = utf8[start:end]
                if sub in self._bpe_index:
                    tids.append(self._bpe_index[sub]); break
        if not tids: return []
        avg = self._embed[tids].mean(axis=0)
        v_norm = avg / (np.linalg.norm(avg) + 1e-10)
        scores = self._emb_norm @ v_norm
        top = np.argsort(scores)[-(top_k*3):][::-1]
        results, seen = [], set(tids)
        for tid in top:
            if int(tid) in seen: continue
            raw = self._decode_map.get(int(tid), b'')
            txt = raw.decode('utf-8', errors='replace')
            txt = txt.replace('\u0120',' ').replace('\u2581',' ').strip()
            if txt and len(txt) >= 1 and txt not in seen:
                results.append((txt, float(scores[int(tid)]), int(tid)))
                seen.add(txt)
            if len(results) >= top_k: break
        return results
    
    def perceive(self, text):
        self.round += 1
        self._ensure_backend()
        
        # ═══ 路径1：汉字物理 → qi ═══
        d1, d3, den_m, den_s = hanzi_physics(text)
        mag = den_m * 2.0 * (1 + den_s * 3) if den_m else 0.15
        angles = []
        if d1 is not None: angles.extend([(d1, 1.0), (d3, 1.0)])
        else: angles.extend([(0, 0.05), (180, 0.05)])
        for (a, s) in [digit_angle(text), punct_angle(text)]:
            if a is not None and s > 0: angles.append((a, s))
        
        ANGLES_DICT = {n: i * 45 for i, n in enumerate(BAGUA)}
        qi_physics = np.array([0.2] * 8)
        for angle, m in angles:
            for i, name in enumerate(BAGUA):
                diff = min(abs(angle - ANGLES_DICT[name]), 360 - abs(angle - ANGLES_DICT[name]))
                qi_physics[i] += max(0, math.cos(math.radians(diff))) * m * mag
        for h, l in [('乾','坤'),('兑','艮'),('离','坎'),('震','巽')]:
            hi, li = BAGUA.index(h), BAGUA.index(l)
            if qi_physics[hi] > qi_physics[li]:
                flow = (qi_physics[hi] - qi_physics[li]) * 0.1
                qi_physics[hi] -= flow; qi_physics[li] += flow
        
        qi_physics_dist = qi_physics / (qi_physics.sum() + 1e-10)
        
        # ═══ 时+空 物理偏置 ═══
        # 时：系统节律（cos相位）
        if self.hour is not None:
            time_bias = time_membrane(self.hour)
            qi_physics_dist = qi_physics_dist + (time_bias - 1.0) * 0.15
            qi_physics_dist = qi_physics_dist / (qi_physics_dist.sum() + 1e-10)
        # 空：问题自带物理空间（句式+人称+时态+长度→偏置）
        space_bias = extract_space(text)
        if np.abs(space_bias).sum() > 0.001:
            qi_physics_dist = qi_physics_dist + space_bias * 0.05
            qi_physics_dist = qi_physics_dist / (qi_physics_dist.sum() + 1e-10)
        
        # ═══ 路径2：棱镜 → qi ═══
        vec = self._word_to_vec(text)
        prism_r = self._prism.project(vec)
        qi_prism = {t: prism_r['distribution'][t] for t in BAGUA}
        
        # ═══ 路径3：查表 → qi ═══
        neighbors = self._lookup_neighbors(text, top_k=5)
        qi_lookup = {t: 0.125 for t in BAGUA}  # 默认均匀
        
        if neighbors:
            lookup_dist = np.zeros(8)
            for word, cos, _ in neighbors[:5]:
                if cos < 0.5: continue
                nb_vec = self._word_to_vec(word)
                nb_prism = self._prism.project(nb_vec)
                nb_qi = np.array([nb_prism['distribution'][t] for t in BAGUA])
                lookup_dist += nb_qi * cos
            if lookup_dist.sum() > 0:
                lookup_dist = lookup_dist / lookup_dist.sum()
                qi_lookup = {t: float(lookup_dist[i]) for i, t in enumerate(BAGUA)}
        
        # ═══ 异步双卦 ═══
        # 第1步：物理独立走完全程
        qi_physics_dict = {t: float(qi_physics_dist[i]) for i, t in enumerate(BAGUA)}
        result_physics = self.v94.divine_from_qi(qi_physics_dict, trace=True)
        physics_winner = result_physics['winner']
        physics_final = np.array([result_physics['distribution'].get(g, 0) for g in BAGUA])
        
        # 提取场内部信号
        extremeness = result_physics.get('extremeness', 0.0)
        cooling = result_physics.get('cooling', {t: 0.0 for t in BAGUA})
        cooling_arr = np.array([cooling.get(g, 0.0) for g in BAGUA])
        
        # 第2步：棱镜读取物理结论，不独立演化——只算偏角
        # 棱镜 qi 保持原始分布（不过 v94），直接和物理结果对比
        prism_raw = np.array([qi_prism.get(g, 0) for g in BAGUA])
        prism_raw = prism_raw / (prism_raw.sum() + 1e-10)
        
        # 偏角 = 棱镜方向与物理方向的余弦距离
        cos_sim = np.dot(physics_final, prism_raw) / (
            np.linalg.norm(physics_final) * np.linalg.norm(prism_raw) + 1e-10)
        deviation = float(1.0 - cos_sim)
        
        # 旁注：棱镜按自己的分布"投票"，但不改变物理卦
        prism_winner_idx = int(np.argmax(prism_raw))
        prism_winner = BAGUA[prism_winner_idx]
        
        # 比较时态：物理卦固定，看棱镜往哪个方向偏
        if physics_winner == prism_winner:
            annotation = "一致"
        else:
            # 偏角方向：物理→棱镜
            annotation = f"{physics_winner}→{prism_winner}"
        
        # ═══ 场惯性：基于物理最终分布 ═══
        movement = None
        if self.prev_qi is not None and self.round > 1:
            qi_with_inertia = physics_final * (1 - self.inertia) + self.prev_qi * self.inertia
            qi_with_inertia = qi_with_inertia / qi_with_inertia.sum()
            movement = qi_with_inertia - self.prev_qi
            physics_final = qi_with_inertia
        
        # ═══ 人物透镜 ═══
        viewed = self.persona.view(physics_final)
        self.persona.update(physics_final)
        self.prev_qi = physics_final.copy()
        
        sorted_idx = np.argsort(physics_final)[::-1]
        tension = float(physics_final[sorted_idx[0]] - physics_final[sorted_idx[-1]])
        
        move_meta = {}
        if movement is not None:
            move_meta = {
                'direction': BAGUA[int(np.argmax(movement))],
                'away_from': BAGUA[int(np.argmin(movement))],
                'speed': float(np.linalg.norm(movement)),
            }
        
        # ── 动量计算（从v94内部trace提取）──
        trace = result_physics.get('trace', [])
        momentum = {}
        if trace and len(trace) >= 3:
            # 速度：主导卦在各帧间的角位移
            velocities = []
            for i in range(1, len(trace)):
                g_prev = trace[i-1]['top_gua']
                g_curr = trace[i]['top_gua']
                # 先天方位角距离
                a_prev = BAGUA_ANGLE.get(g_prev, 0)
                a_curr = BAGUA_ANGLE.get(g_curr, 0)
                d = abs(a_curr - a_prev)
                d = min(d, 360 - d) / 180.0  # 归一化 [0,1]
                velocities.append(d)
            # 加速度：速度变化
            accels = [velocities[i+1] - velocities[i] for i in range(len(velocities)-1)]
            momentum = {
                'vel_mean': float(np.mean(velocities)),
                'vel_max': float(np.max(velocities)),
                'accel_mean': float(np.mean(accels)) if accels else 0,
                'accel_abs_mean': float(np.mean([abs(a) for a in accels])) if accels else 0,
                'stable': float(np.std(velocities)) < 0.15 if len(velocities) > 2 else True,
            }
        
        return {
            'round': self.round, 'text': text,
            'winner': physics_winner,
            'cv': result_physics.get('depth_cv', 0),
            'top3': [BAGUA[i] for i in sorted_idx[:3]],
            'tension': tension,
            'persona_cv': self.persona.cv(),
            'persona_dominant': self.persona.dominant()[0],
            'movement': move_meta,
            # v9：棱镜旁注
            'physics_winner': physics_winner,
            'prism_annotation': annotation,
            'deviation': deviation,
            'prism_winner': prism_winner,
            'prism_yang': prism_r['yang'],
            'prism_yin': prism_r['yin'],
            # qi_state for NN
            'qi_state': physics_final,
            # 轨迹 + 动量（给NN做过程注入）
            'trace': result_physics.get('trace', []),
            'momentum': momentum,
            # 场信号（冷却度+极端度 → 对话策略）
            'cooling': cooling_arr,
            'extremeness': extremeness,
            'inertia': self.inertia,
        }
    
    def reset(self):
        self.prev_qi = None; self.round = 0


# ============================================================
# 测试（端到端：引擎 + 八卦NN + 语言偶极子场）
# ============================================================
if __name__ == '__main__':
    import sys, io, os, json
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    from bagua_nn import BaguaNeuralNet
    from language_qichang import LanguageQichang
    from transformers import AutoTokenizer
    import numpy as np
    
    # ── 加载7B嵌入表和蒸馏数据 ──
    embed_table = np.load('data/qwen7b_embed_tokens.npy')
    MODEL_DIR = 'C:/Users/ww109/.cache/modelscope/Qwen/Qwen2___5-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    distilled_npz = np.load('data/distilled_7b.npz')
    
    # 蒸馏锚点
    anchors = {}
    for gua in ['乾', '兑', '离', '震', '巽', '坎', '艮', '坤']:
        key = f'{gua}_vec'
        if key in distilled_npz:
            anchors[gua] = distilled_npz[key]
    
    # 加载蒸馏词典JSON
    distilled_json_path = os.path.join(os.path.dirname(__file__), 'gua8_bait_distilled_7b.json')
    with open(distilled_json_path, 'r', encoding='utf-8') as f:
        distilled_data = json.load(f)
    
    # ── 构建词嵌入表 ──
    all_words = set()
    for gua, words in distilled_data['gua_to_words'].items():
        all_words.update(w for w, s in words[:15])
    word_embed = {}
    for word in all_words:
        tids = tokenizer.encode(word, add_special_tokens=False)
        vecs = [embed_table[tid] for tid in tids if tid < len(embed_table)]
        if vecs:
            word_embed[word] = np.mean(vecs, axis=0)
    
    # ── 初始化模块 ──
    nn = BaguaNeuralNet(distilled_path=distilled_json_path)
    lq = LanguageQichang(distill_anchors=anchors)
    lq.set_word_embed(word_embed)
    
    print(f"  蒸馏词典: {len(nn.gua8_words)}卦, {sum(len(v) for v in nn.gua8_words.values())}词")
    print(f"  词嵌入: {len(word_embed)}词")
    print(f"  蒸馏锚点: {len(anchors)}卦")
    
    dialogues = {
        "分手": [
            "我们谈谈吧", "好", "我觉得我们之间变了", "我知道",
            "你不再像以前那样看我了", "对不起", "不是你的错。是我。",
            "不要说这种话", "真的。我累了。",
        ],
        "温暖日常": [
            "今天做了什么", "在阳台看了会书。阳光很好。",
            "真好。我这边下雨。", "带伞了吗", "带了。你提醒过的。",
            "那就好。晚上想吃什么", "你做的都可以", "那煮面吧。加个蛋。",
        ],
    }
    
    def run_dialogue(engine, texts, title, nn, lq, reset_lq=True):
        """跑一组对话"""
        if reset_lq:
            lq._strategy_history.clear()
            lq._strategy_streak.clear()
            lq._gua_streak.clear()
            lq._round_count = 0
            lq._last_strategy = "描述"
            lq._last_gua = None
            lq._cooling = 0.0
        print(f"\n{'='*80}")
        print(f"【{title}】")
        print(f"{'='*80}")
        print(f"{'轮':<3} {'卦':<4} {'旁注':<12} {'偏角':<5} {'Cd':<4} {'策略':<5} {'连':<2} {'输出'}")
        print(f"{'-'*80}")
        
        for t in texts:
            r = engine.perceive(t)
            qi = r['qi_state']
            
            direction = None
            if r['prism_annotation'] != "一致" and "→" in r['prism_annotation']:
                parts = r['prism_annotation'].split("→")
                if len(parts) == 2:
                    direction = (parts[0], parts[1])
            
            nn_result = nn.forward(
                qi, deviation=r['deviation'],
                direction=direction,
                temperature=r['cv'] + 0.5,
                physics_winner=r['physics_winner'],
                trace=r.get('trace', []),
            )
            candidate_words = nn_result['candidate_words']
            
            word_gua_map = {}
            for gua, gw_list in nn.gua8_words.items():
                for gw, gs in gw_list:
                    word_gua_map[gw] = gua
            
            from_words = []
            to_words = []
            if direction:
                from_g, to_g = direction
                for w, s in candidate_words:
                    g = word_gua_map.get(w)
                    if g == from_g:
                        from_words.append(w)
                    elif g == to_g:
                        to_words.append(w)
            
            sentence = lq.generate(
                qi_state=qi,
                dominant_gua=r['physics_winner'],
                deviation=r['deviation'],
                temperature=r['cv'] + 0.5,
                prism_annotation=r['prism_annotation'],
                from_state_words=from_words if from_words else [w for w, s in candidate_words[:6]],
                to_state_words=to_words,
                cooling=r.get('cooling', np.zeros(8)),
                extremeness=r.get('extremeness', 0.0),
                inertia=r.get('inertia', 0.35),
            )
            
            # 从_select_strategy获取实际策略
            actual_strategy = lq._select_strategy(
                r.get('cooling', np.zeros(8)),
                r.get('extremeness', 0.0),
                r.get('inertia', 0.35),
                r['deviation'],
                r['physics_winner'],
            )
            
            print(f"  {r['round']:<3} {r['physics_winner']:<4} "
                  f"{r['prism_annotation']:<12} "
                  f"{r['deviation']:.2f}  "
                  f"{lq._cooling:.2f} "
                  f"{actual_strategy:<5} "
                  f"{lq._strategy_streak.get(actual_strategy, 0):<2} "
                  f"{sentence}")
    
    # ═══ 全局观测：三组对话串联，不重置 ═══
    print(f"\n{'#'*80}")
    print(f"全局观测：三组对话串联（不重置语言场）")
    print(f"{'#'*80}")
    
    all_dialogues = [
        ("分手", ["我们谈谈吧", "好", "我觉得我们之间变了", "我知道",
         "你不再像以前那样看我了", "对不起", "不是你的错。是我。",
         "不要说这种话", "真的。我累了。"]),
        ("温暖日常", ["今天做了什么", "在阳台看了会书。阳光很好。",
         "真好。我这边下雨。", "带伞了吗", "带了。你提醒过的。",
         "那就好。晚上想吃什么", "你做的都可以", "那煮面吧。加个蛋。"]),
        ("困住自语", ["我真的很累", "每天都一样", "上班下班回家睡觉", 
         "没有尽头", "我试过改变", "但总是回到原地",
         "感觉被困住了", "像在笼子里", "门开着但我出不去",
         "不是不想走", "是不知道往哪走", "你知道吗",
         "就是那种感觉", "被什么无形的力量按住了",
         "我说不清楚", "但你能感觉到吗", "那种沉重",
         "每天早上醒来", "什么都没变", "我有时候想",
         "是不是只有我这样", "是不是别人都很清醒"]),
    ]
    
    engine_global = Engine(hour=20, inertia=0.35)
    lq_global = LanguageQichang(distill_anchors=anchors)
    lq_global.set_word_embed(word_embed)
    
    for title, texts in all_dialogues:
        print(f"\n{'─'*80}")
        print(f"【{title}】")
        print(f"{'轮':<3} {'卦':<4} {'旁注':<12} {'偏角':<5} {'空':<12} {'输出'}")
        print(f"{'─'*80}")
        for t in texts:
            r = engine_global.perceive(t)
            qi = r['qi_state']
            
            direction = None
            if r['prism_annotation'] != "一致" and "→" in r['prism_annotation']:
                parts = r['prism_annotation'].split("→")
                if len(parts) == 2:
                    direction = (parts[0], parts[1])
            
            nn_result = nn.forward(
                qi, deviation=r['deviation'],
                direction=direction,
                temperature=r['cv'] + 0.5,
                physics_winner=r['physics_winner'],
                trace=r.get('trace', []),
            )
            candidate_words = nn_result['candidate_words']
            
            word_gua_map = {}
            for gua, gw_list in nn.gua8_words.items():
                for gw, gs in gw_list:
                    word_gua_map[gw] = gua
            
            from_words = []
            to_words = []
            if direction:
                from_g, to_g = direction
                for w, s in candidate_words:
                    g = word_gua_map.get(w)
                    if g == from_g:
                        from_words.append(w)
                    elif g == to_g:
                        to_words.append(w)
            
            sentence = lq_global.generate(
                qi_state=qi,
                dominant_gua=r['physics_winner'],
                deviation=r['deviation'],
                temperature=r['cv'] + 0.5,
                prism_annotation=r['prism_annotation'],
                from_state_words=from_words if from_words else [w for w, s in candidate_words[:6]],
                to_state_words=to_words,
                cooling=r.get('cooling', np.zeros(8)),
                extremeness=r.get('extremeness', 0.0),
                inertia=r.get('inertia', 0.35),
            )
            
            gua_s = lq_global._gua_streak.get(r['physics_winner'], 0)
            strat_s = lq_global._strategy_streak.get(lq_global._last_strategy, 0)
            
            # 空间偏置信息
            sp = extract_space(t)
            sp_top = sorted([(BAGUA[i], sp[i]) for i in range(8) if sp[i] > 0.02], key=lambda x: -x[1])
            sp_str = '+'.join(g for g, v in sp_top[:3]) if sp_top else '—'
            
            print(f"  {r['round']:<3} {r['physics_winner']:<4} "
                  f"{r['prism_annotation']:<12} "
                  f"{r['deviation']:.2f}  "
                  f"{sp_str:<12} "
                  f"{sentence}")
    
    print(f"\n{'='*80}")
    print(f"最终状态: 卦冷却={lq_global._cooling:.2f} | 策略疲劳={lq_global._strategy_streak}")
    print(f"卦位连任: {lq_global._gua_streak}")
    
    # ═══ 诊断：冷却机制验证 ═══
    print(f"\n{'#'*80}")
    print(f"诊断：冷却机制纯注入验证（同卦×15轮）")
    print(f"{'#'*80}")
    lq_test = LanguageQichang(distill_anchors=anchors)
    lq_test.set_word_embed(word_embed)
    lq_test._debug = True
    qi_diag = np.array([0.3, 0.05, 0.05, 0.05, 0.45, 0.05, 0.03, 0.02])  # 巽主导
    for i in range(15):
        s = lq_test.generate(
            qi_state=qi_diag, dominant_gua='巽', deviation=0.15,
            prism_annotation='一致',
            from_state_words=['风', '柔', '顺', '温和'], to_state_words=[],
            cooling=np.zeros(8), extremeness=3.0, inertia=0.5,
        )
        gua_s = lq_test._gua_streak.get('巽', 0)
        print(f"  轮{i+1:>2} 卦连={gua_s}  Cd={lq_test._cooling:.3f}  策略={lq_test._last_strategy}  → {s}")
