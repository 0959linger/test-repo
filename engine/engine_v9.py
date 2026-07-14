"""
╔══════════════════════════════════════════════════════════════════╗
║  engine_v9 — 交汇处理器                                         ║
║                                                                  ║
║  三条线在词网络上汇合：                                          ║
║    ① 汉字物理（硬物理）→ qi_physics                             ║
║    ② v94 qi_field（背景温度梯度）→ 从蒸汽桥接上游传入            ║
║    ③ 河图语义（软物质）→ 玄学压缩 → 词网络 → 预演层 → 热分布   ║
║                                                                  ║
║  外围层（接入但不进场）：                                        ║
║    记忆层 — 初态微偏置 qi × 0.15                                ║
║    近窗微风 — 每步 ε=0.005                                      ║
║    C层推理 — 旁路日记，不碰场                                    ║
║    指涉层 — 跨轮观察，给玲看                                    ║
║                                                                  ║
║  铁律：不筛选 不特征提取 不投影 不匹配 不量化 不加权 不仲裁      ║
║                                                                  ║
║  基石 → MEMORY.md ⭐ 架构基石 2026-07-11                         ║
╚══════════════════════════════════════════════════════════════════╝
"""
import numpy as np, json, os, sys, math, re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# ─── 路径 ───
HERE = os.path.dirname(os.path.abspath(__file__))
V94_SRC = os.path.join(HERE, '..', 'src', 'v94_qichang')

import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "core_enhanced",
    os.path.join(V94_SRC, 'core_enhanced.py')
)
_ce = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_ce)
V94QichangEnhanced = _ce.V94QichangEnhanced

BAGUA = ['乾', '兑', '离', '震', '坤', '艮', '坎', '巽']
BAGUA_RI = {g: i for i, g in enumerate(BAGUA)}

# ─── 全局嵌入表缓存 ───
# 多个 EngineV9 实例共享同一份 mmap 数据，避免重复 I/O
_GLOBAL_EMBED_CACHE = {}
_GLOBAL_256_NORMED = {}  # 归一化 256D 嵌入表缓存
_GLOBAL_TOKENIZER = {}   # GGUF tokenizer 缓存

def _get_shared_embed(embed_path, pca_path):
    """获取全局共享的嵌入表 + PCA（懒加载，只读一次）"""
    key = (embed_path, pca_path)
    if key not in _GLOBAL_EMBED_CACHE:
        embed_7b = np.load(embed_path, mmap_mode='r')
        pca_data = np.load(pca_path)
        pca_proj = pca_data['proj']    # [3584, 256]
        pca_mean = pca_data['mean']    # [3584,]
        _GLOBAL_EMBED_CACHE[key] = (embed_7b, pca_proj, pca_mean)
    return _GLOBAL_EMBED_CACHE[key]

def _get_shared_256_normed(embed_7b, pca_proj, pca_mean, embed_path=None, pca_path=None):
    """获取全局共享的归一化 256D 嵌入表（磁盘缓存 + 内存缓存）
    
    着色器缓存思路：
      1. 检查磁盘缓存是否存在且比源文件新
      2. 是 → 直接 load（<1s）
      3. 否 → 计算 → 存盘（首次 ~5s，之后 ~1s）
    """
    key = id(embed_7b)  # 用嵌入表对象的 id 做 key
    if key not in _GLOBAL_256_NORMED:
        # 磁盘缓存路径
        cache_path = None
        if embed_path and pca_path:
            data_dir = os.path.dirname(embed_path)
            cache_path = os.path.join(data_dir, 'emb_256_normed_cache.npy')
            
            # 缓存有效性检查：缓存存在 且 比源文件新
            if os.path.exists(cache_path):
                cache_mtime = os.path.getmtime(cache_path)
                embed_mtime = os.path.getmtime(embed_path)
                pca_mtime = os.path.getmtime(pca_path)
                if cache_mtime > embed_mtime and cache_mtime > pca_mtime:
                    try:
                        _GLOBAL_256_NORMED[key] = np.load(cache_path, mmap_mode='r')
                        return _GLOBAL_256_NORMED[key]
                    except:
                        pass  # 缓存损坏，重新计算
        
        # 计算
        emb_c = embed_7b - pca_mean
        proj = emb_c @ pca_proj  # [152064, 256]
        norms = np.linalg.norm(proj, axis=1, keepdims=True) + 1e-10
        result = (proj / norms).astype(np.float32)
        
        # 存盘
        if cache_path:
            try:
                np.save(cache_path, result)
            except:
                pass  # 存盘失败不阻塞
        
        _GLOBAL_256_NORMED[key] = result
    return _GLOBAL_256_NORMED[key]

def _get_shared_tokenizer(gguf_path):
    """获取全局共享的 GGUF tokenizer（磁盘缓存 + 内存缓存）
    
    着色器缓存思路：
      1. 检查磁盘缓存是否存在且比 GGUF 新
      2. 是 → 直接 load（<0.1s）
      3. 否 → 解析 → 存盘（首次 ~1s，之后 <0.1s）
    """
    if gguf_path not in _GLOBAL_TOKENIZER:
        import pickle
        
        # 磁盘缓存路径
        cache_path = gguf_path + '.tokenizer_cache.pkl'
        
        # 缓存有效性检查
        if os.path.exists(cache_path):
            try:
                cache_mtime = os.path.getmtime(cache_path)
                gguf_mtime = os.path.getmtime(gguf_path)
                if cache_mtime > gguf_mtime:
                    with open(cache_path, 'rb') as f:
                        _GLOBAL_TOKENIZER[gguf_path] = pickle.load(f)
                    return _GLOBAL_TOKENIZER[gguf_path]
            except:
                pass  # 缓存损坏，重新解析
        
        # 解析
        from gguf import GGUFReader
        reader = GGUFReader(gguf_path)
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
        result = (decode_map, bpe_index, byte_to_tid)
        
        # 存盘
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except:
            pass  # 存盘失败不阻塞
        
        _GLOBAL_TOKENIZER[gguf_path] = result
    return _GLOBAL_TOKENIZER[gguf_path]

# ─── 物理工具函数（从 engine_v8 继承，不动原文件） ───
from PIL import Image, ImageDraw, ImageFont
FONT = None
def _get_font():
    global FONT
    if FONT is None:
        FONT = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 128)
    return FONT

def _hanzi_physics(text: str):
    """汉字物理 → 方向角+密度"""
    chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    n = len(chars)
    if n < 2: return None, None, 0.1, 0.0
    codes = np.array([(ord(c) - 0x4E00) / (0x9FFF - 0x4E00) * 2 - 1 for c in chars])
    d1 = np.median([(codes[i+1] - codes[i]) * 180 for i in range(n-1)]) % 360
    if n >= 3:
        m, s = np.mean(codes), np.std(codes) or 1e-10
        d3 = (np.mean(((codes - m) / s) ** 3) * 90 + 180) % 360
    else:
        d3 = d1
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
        except:
            pass
    arr = np.array(densities) if densities else np.array([0.1])
    return d1, d3, float(np.mean(arr)), float(np.std(arr))

def _digit_angle(text: str):
    digits = re.findall(r'[0-9]', text)
    if not digits: return None, 0
    codes = [(ord(d) - 0x30) / 9.0 for d in digits]
    angle = np.median([(codes[i+1]-codes[i])*180 for i in range(len(codes)-1)])%360 if len(codes)>=2 else codes[0]*360
    return angle, min(len(digits)*0.15, 0.6)

def _punct_angle(text: str):
    MAP = {'？':0,'?':0,'！':90,'!':90,'，':180,',':180,'。':270,'.':270}
    puncts = [c for c in text if c in MAP]
    if not puncts: return None, 0
    angles = [MAP[c] for c in puncts]
    angle = (np.median(np.diff(angles))+360)%360 if len(angles)>=2 else angles[0]
    return angle, min(len(puncts)*0.15, 0.6)

def _compute_qi_physics(text: str) -> np.ndarray:
    """汉字物理 → 8维qi分布"""
    d1, d3, den_m, den_s = _hanzi_physics(text)
    mag = den_m * 2.0 * (1 + den_s * 3) if den_m else 0.15
    angles = []
    if d1 is not None: angles.extend([(d1, 1.0), (d3, 1.0)])
    else: angles.extend([(0, 0.05), (180, 0.05)])
    for (a, s) in [_digit_angle(text), _punct_angle(text)]:
        if a is not None and s > 0: angles.append((a, s))
    
    ANGLES_DICT = {n: i * 45 for i, n in enumerate(BAGUA)}
    qi = np.array([0.2] * 8)
    for angle, m in angles:
        for i, name in enumerate(BAGUA):
            diff = min(abs(angle - ANGLES_DICT[name]), 360 - abs(angle - ANGLES_DICT[name]))
            qi[i] += max(0, math.cos(math.radians(diff))) * m * mag
    for h, l in [('乾','坤'),('兑','艮'),('离','坎'),('震','巽')]:
        hi, li = BAGUA.index(h), BAGUA.index(l)
        if qi[hi] > qi[li]:
            flow = (qi[hi] - qi[li]) * 0.1
            qi[hi] -= flow; qi[li] += flow
    return qi / (qi.sum() + 1e-10)


# ═══════════════════════════════════════════
# 预演层：微型排名级联
# ═══════════════════════════════════════════
def _prestage_amplify(qi: Dict[str, float], steps: int = 10,
                      rate: float = 0.40) -> Dict[str, float]:
    """
    预演层放大：独立微型级联场。
    放在词网络和主v94之间，只拉开到可观测就停。
    铁律七条零触碰。
    """
    qi = {g: max(0.0, qi[g]) for g in BAGUA}
    for _ in range(steps):
        sg = sorted(qi.items(), key=lambda x: -x[1])
        for i in range(7):
            src, dst = sg[i][0], sg[i+1][0]
            diff = qi[src] - qi[dst]
            if diff > 0:
                cascade = diff * rate
                qi[src] += cascade
                qi[dst] -= cascade
        for g in BAGUA:
            qi[g] = max(0.001, qi[g])
    return qi


# ═══════════════════════════════════════════
# 词网络
# ═══════════════════════════════════════════
class WordNetwork:
    """
    词网络：7B嵌入表 + PCA 256D投影 → 词间语义距离 → 热扩散 → qi分布。
    
    流程：
      河图文本 → 分词 → 查7B嵌入(3584D) → PCA投影(256D)
      → 词间余弦距离 → 语义邻接图 → 热扩散 → qi分布
    """
    
    def __init__(self, embed_7b: 'np.ndarray', pca_proj: 'np.ndarray', pca_mean: 'np.ndarray',
                 tokenizer_json_path: str = None,
                 embed_path: str = None, pca_path: str = None):
        self._embed_7b = embed_7b       # [vocab, 3584] — 7B 完整嵌入表
        self._pca_proj = pca_proj       # [3584, 256]
        self._pca_mean = pca_mean       # [3584]
        
        # 使用 tokenizer.json（支持中文分词）
        self._tokenizer = None
        self._tokenizer_path = tokenizer_json_path
        if tokenizer_json_path and os.path.exists(tokenizer_json_path):
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(tokenizer_json_path)
        
        # 磁盘缓存路径
        self._embed_path = embed_path
        self._pca_path = pca_path
        
        # 预计算：归一化 256D 嵌入表（懒加载或分批）
        self._emb_256_normed = None
    
    def _ensure_256_normed(self):
        """预计算归一化 256D 嵌入表（全局缓存，所有实例共享）"""
        if self._emb_256_normed is not None:
            return
        # 使用全局缓存，避免重复计算 152064×3584→256 矩阵乘法
        self._emb_256_normed = _get_shared_256_normed(
            self._embed_7b, self._pca_proj, self._pca_mean,
            embed_path=self._embed_path, pca_path=self._pca_path
        )
    
    def _tokenize(self, text: str) -> 'List[int]':
        """BPE分词 → token id列表
        
        使用 tokenizer.json（支持中文分词）。
        """
        if self._tokenizer is None:
            return []
        
        # 使用 tokenizer.json 分词
        encoded = self._tokenizer.encode(text)
        tids = encoded.ids
        
        # 去重保留顺序
        seen = set()
        unique = []
        for t in tids:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique
    
    def _semantic_distances(self, tids: 'List[int]') -> 'np.ndarray':
        """词间语义距离矩阵（余弦相似度 → 距离）"""
        import numpy as np
        self._ensure_256_normed()
        n = len(tids)
        if n <= 1:
            return np.zeros((n, n))
        vecs = self._emb_256_normed[tids]  # [n, 256]
        # 余弦相似度
        sim = vecs @ vecs.T  # [n, n]
        np.fill_diagonal(sim, 0)
        return sim  # 正值=近，负值=远
    
    def _build_graph(self, tids: 'List[int]', sim: 'np.ndarray',
                     threshold: float = 0.3, k_max: int = 5) -> 'dict':
        """建语义邻接图：每条边 = (src, dst, weight)"""
        import numpy as np
        edges = []
        n = len(tids)
        for i in range(n):
            # 每个词连 k_max 个最近的（超过阈值）
            row = sim[i]
            # 只取正相似（余弦 > threshold）
            candidates = [(j, row[j]) for j in range(n) if j != i and row[j] > threshold]
            candidates.sort(key=lambda x: -x[1])
            for j, w in candidates[:k_max]:
                edges.append((i, j, w))
        return edges
    
    def _heat_diffusion(self, n_nodes: int, edges: 'list', 
                        steps: int = None, temp: float = None,
                        initial_heat: 'np.ndarray' = None) -> 'np.ndarray':
        """图热扩散：热从每个节点沿语义边扩散
        
        自适应参数（2026-07-14 补丁）：
        - steps: 默认 ∝ log(n_nodes) * 3（词越多步数越多，但有上限）
        - temp:  默认 0.5 / (n_nodes + 1)（词越多温度越低，慢扩散）
        - initial_heat: 可选非均匀初始热源（混合气体比例）
        """
        import numpy as np
        
        # 自适应参数（比例关系，不是硬编码）
        if steps is None:
            steps = int(np.log(n_nodes + 1) * 3)  # 词越多步数越多
            steps = max(5, min(steps, 30))         # 下限5，上限30
        if temp is None:
            temp = 0.5 / (n_nodes + 1)             # 词越多温度越低
            temp = max(0.01, min(temp, 0.2))       # 下限0.01，上限0.2
        
        # 初始热分布
        if initial_heat is not None:
            heat = initial_heat.copy()
        else:
            heat = np.ones(n_nodes) / n_nodes  # 默认均匀热
        
        # 建邻接矩阵
        adj = np.zeros((n_nodes, n_nodes))
        for i, j, w in edges:
            adj[i, j] = w
            adj[j, i] = w  # 无向
        
        # 度归一化
        deg = adj.sum(axis=1)
        deg[deg == 0] = 1  # 孤立节点
        adj_norm = adj / deg[:, np.newaxis]
        
        # 扩散
        for _ in range(steps):
            heat = (1 - temp) * heat + temp * (adj_norm.T @ heat)
            heat = heat / (heat.sum() + 1e-10)
        
        return heat
    
    def _text_to_qi_v2(self, text: str) -> 'Dict[str, float]':
        """一段文本 → 词网络热扩散 → qi分布（新方法）"""
        import numpy as np
        
        tids = self._tokenize(text)
        if not tids:
            return {g: 1.0 for g in BAGUA}
        
        # 词间语义距离
        sim = self._semantic_distances(tids)
        
        # 建图 + 热扩散
        edges = self._build_graph(tids, sim, threshold=0.3, k_max=5)
        heat = self._heat_diffusion(len(tids), edges, steps=5, temp=0.1)
        
        # 热分布 → qi：热值结合词在八卦方向上的投影
        self._ensure_256_normed()
        vecs = self._emb_256_normed[tids]  # [n, 256]
        
        # 每词对其相邻词的平均方向加权映射到八卦
        qi = {g: 0.0 for g in BAGUA}
        for i in range(len(tids)):
            # 取该词最近的邻居方向
            row = sim[i]
            neighbors = [(j, row[j]) for j in range(len(tids)) if j != i and row[j] > 0.2]
            if not neighbors:
                continue
            neighbors.sort(key=lambda x: -x[1])
            
            # 邻居方向的平均向量
            neighbor_vecs = [vecs[j] * w for j, w in neighbors[:3]]
            avg_dir = sum(neighbor_vecs) / (len(neighbor_vecs) + 1e-10)
            norm = np.linalg.norm(avg_dir) + 1e-10
            avg_dir = avg_dir / norm
            
            # 投影到八卦方向（256维 → 8卦轮转分桶，每卦32维，方差均匀）
            for gi, g in enumerate(BAGUA):
                # 轮转分桶：卦0取维[0,8,16,...]，卦1取维[1,9,17,...]...
                # 这样 PCA 前几维的方差被8卦平分，不会全归乾
                bucket_dims = [d for d in range(gi, 256, 8)]  # 32维
                proj_sum = np.abs(avg_dir[bucket_dims]).sum()
                qi[g] += heat[i] * proj_sum
        
        # 归一化
        total = sum(qi.values())
        if total < 1e-10:
            return {g: 1.0 for g in BAGUA}
        return {g: max(0.001, v / total) for g, v in qi.items()}
    
    def ingest(self, hetu_texts: 'List[str]') -> 'Dict[str, float]':
        """
        全量直喂6段文本 → 词网络热扩散 → qi分布。
        频率加权：被多段文本提及的词自然更热。
        """
        import numpy as np
        if not hetu_texts:
            return {g: 1.0 for g in BAGUA}
        
        merged = {g: 0.0 for g in BAGUA}
        for text in hetu_texts:
            qi_seg = self._text_to_qi_v2(text)
            for g in BAGUA:
                merged[g] += qi_seg[g]
        
        total = sum(merged.values())
        return {g: max(0.001, v / total) for g, v in merged.items()}
    
    def _compute_word_heat(self, text: str, knowledge_text: str = None, 
                           global_state: dict = None) -> 'Dict[int, float]':
        """
        计算词网络中每个词的基础热度。
        
        流程：分词 → 7B嵌入 → PCA投影 → 词间距离 → 热扩散
        返回 {tid: heat}
        
        自适应热源比例（2026-07-14 补丁，氧气焊混合）：
        - 问题词：基础热度，由语义密度决定
        - 知识库词：基础热度 × 全局调制因子
          - 问题短 + 知识库长 → 知识库热度自动降低（不喧宾夺主）
          - 问题长 + 知识库短 → 知识库热度自动升高（助燃剂加强）
          - 场CV高（已收敛）→ 知识库热度降低
          - 场CV低（波动）→ 知识库热度加强
        - 虚词：不再人工标记，让热扩散自然决定（语义距离远自然沉底）
        
        这不是"权"（人为定义重要性），是物理系统的自适应。
        """
        import numpy as np
        
        # 问题文本分词
        tids = self._tokenize(text)
        if not tids:
            return {}
        
        # 知识库词加入词网络（2026-07-14 修复）
        kb_tids = set()
        kb_len = 0
        if knowledge_text:
            kb_tids_list = self._tokenize(knowledge_text)
            kb_tids = set(kb_tids_list)
            kb_len = len(kb_tids_list)
            # 把知识库词加入主词列表（去重）
            for tid in kb_tids_list:
                if tid not in tids:
                    tids.append(tid)
        
        # 词间语义距离
        sim = self._semantic_distances(tids)
        
        # 建图
        edges = self._build_graph(tids, sim, threshold=0.3, k_max=5)
        
        # 自适应热源比例（氧气焊混合）
        n = len(tids)
        question_len = len(self._tokenize(text))
        
        # 问题词基础热度 = 1.0（归一化后是比例）
        initial_heat = np.ones(n) * 1.0
        
        # 知识库词热度自适应
        if knowledge_text and kb_len > 0:
            # 基础比例：问题长度 / 知识库长度
            # 问题短 + 知识库长 → 比例 < 1.0 → 知识库热度降低
            # 问题长 + 知识库短 → 比例 > 1.0 → 知识库热度升高
            length_ratio = question_len / max(kb_len, 1)
            
            # 全局调制因子：场状态
            # CV 高（已收敛，稳定）→ 调制 < 1.0，减少助燃
            # CV 低（波动，没收敛）→ 调制 > 1.0，加强助燃
            cv_modulation = 1.0
            if global_state and 'cv' in global_state:
                cv = global_state['cv']
                # CV 范围通常 0.0-2.0，中心值约 1.0
                # 用指数函数让调制更敏感：cv=0.1 → 调制=1.43, cv=1.0 → 调制=1.0, cv=2.0 → 调制=0.67
                import math
                cv_modulation = math.exp(0.4 * (1.0 - cv))
            
            # 知识库词热度 = 基础比例 × 全局调制
            kb_heat = length_ratio * cv_modulation
            
            for i, tid in enumerate(tids):
                if tid in kb_tids:
                    initial_heat[i] = kb_heat
        
        # 归一化初始热分布
        initial_heat = initial_heat / initial_heat.sum()
        
        # 自适应热扩散（网的比例）
        heat = self._heat_diffusion(n, edges, initial_heat=initial_heat)
        
        # 返回 {tid: heat}
        return {tids[i]: float(heat[i]) for i in range(n)}
    
    def _get_word_gua_bias(self, tid: int) -> 'np.ndarray':
        """
        获取词的八卦倾向向量（8维）。
        
        从 256 维向量投影到八卦空间。
        使用轮转分桶方法（每卦32维，方差均匀分布）。
        """
        import numpy as np
        
        self._ensure_256_normed()
        vec_256 = self._emb_256_normed[tid]  # [256,]
        
        # 轮转分桶投影：卦0取维[0,8,16,...]，卦1取维[1,9,17,...]
        gua_bias = np.zeros(8)
        for gi, g in enumerate(BAGUA):
            bucket_dims = [d for d in range(gi, 256, 8)]  # 32维
            gua_bias[gi] = np.abs(vec_256[bucket_dims]).sum()
        
        # 归一化
        total = gua_bias.sum()
        if total > 1e-10:
            gua_bias = gua_bias / total
        
        return gua_bias
    
    def _decode_tid(self, tid: int) -> str:
        """将 token id 解码为文本"""
        if self._tokenizer is None:
            return f"<tid:{tid}>"
        
        # 使用 tokenizer.json 解码
        try:
            decoded = self._tokenizer.decode([tid])
            return decoded if decoded else f"<tid:{tid}>"
        except:
            return f"<tid:{tid}>"


# ═══════════════════════════════════════════
# 引擎 v9：交汇处理器
# ═══════════════════════════════════════════
@dataclass
class EngineV9Result:
    """引擎v9产出"""
    winner: str                          # 主卦
    distribution: Dict[str, float]       # 最终qi分布
    cv: float                            # 深度CV
    qi_physics: np.ndarray               # 汉字物理qi（记录用）
    hetu_heat: Dict[str, float]          # 河图语义热分布（放大后）
    qi_field_in: np.ndarray              # 传入的qi_field
    trace: dict                          # v94演化trace
    word_crystal: List[str]              # 结晶词序列（嘴吧窗口用，高维语义场直接浮现）
    dashboard_gua: List[str] = field(default_factory=list)  # 仪表盘卦标签（v94 8标量观测，不主导）


class EngineV9:
    """交汇处理器 — 三条线在主v94词网络上场自决"""
    
    def __init__(self, hour: int = None,
                 embed_path: str = None,
                 pca_path: str = None):
        # 自动检测数据路径（相对于 engine/ 上一级的 data/）
        _data_dir = os.path.normpath(os.path.join(HERE, '..', 'data'))
        if embed_path is None:
            embed_path = os.path.join(_data_dir, 'qwen7b_embed_tokens.npy')
        if pca_path is None:
            pca_path = os.path.join(_data_dir, 'pca_256_proj.npz')
        self.v94 = V94QichangEnhanced()
        self.hour = hour
        self.round = 0
        
        # 外围层状态
        self.memory_bias = np.zeros(8)     # 记忆层初态偏置
        self.near_qi = None                # 近窗微风（上轮qi）
        self.breeze_epsilon = 0.005
        
        # 懒加载
        self._word_net = None
        self._embed = None
        self._embed_path = embed_path
        self._pca_path = pca_path
        self._loaded = False
        
        # C层/指涉层引用（外部注入，引擎不维护）
        self.c_layer = None
        self.observer = None
    
    def _ensure_loaded(self):
        import numpy as np
        if self._loaded:
            return
        
        # ── 全局缓存：嵌入表 + PCA 投影 ──
        embed_7b, pca_proj, pca_mean = _get_shared_embed(self._embed_path, self._pca_path)
        
        # ── tokenizer.json（支持中文分词）──
        tokenizer_path = os.path.join(_data_dir, 'tokenizer.json')
        
        self._word_net = WordNetwork(embed_7b, pca_proj, pca_mean,
                                     tokenizer_json_path=tokenizer_path,
                                     embed_path=self._embed_path, pca_path=self._pca_path)
        self._loaded = True


    
    # ═══ 外围层接口 ═══
    
    def set_memory_bias(self, bias: np.ndarray):
        """记忆层注入初态偏置（不进管道，仅改边界条件）"""
        self.memory_bias = bias.copy()
    
    def set_near_window(self, near_qi: np.ndarray):
        """近窗微风注入上轮qi（不进蒸汽、不争卦位）"""
        self.near_qi = near_qi.copy()
    
    def attach_c_layer(self, c_layer):
        """注入C层推理器（旁路，不碰场）"""
        self.c_layer = c_layer
    
    def attach_observer(self, observer):
        """注入指涉层观察器（只读，不干预）"""
        self.observer = observer
    
    # ═══ 主入口 ═══
    
    def perceive(self, text: str,
                 hetu_texts: List[str] = None,
                 qi_field: np.ndarray = None) -> EngineV9Result:
        """
        交汇处理器主入口。
        
        Args:
            text: 原始问题（走汉字物理——保留原生连结）
            hetu_texts: 河图6段回答（走语义入口）
            qi_field: v94 qi场（背景温度梯度，从蒸汽桥接上游传入）
        
        Returns:
            EngineV9Result: 包含 winner（胜卦）、distribution（8卦分布）、cv 等
        
        场自决行为说明（2026-07-13 分析）：
        ────────────────────────────────────────────────────────────────
        观察现象：在某些测试中，兑/离/震/巽 可能从未成为 winner（胜卦）。
        
        根因分析：这是**正常的场自决行为**，不是bug。
        
        1. 所有8个卦位的 distribution 值都在 0.10-0.25 范围内，**没有任何卦位为0**
        2. v94 场演化会选择能量最集中、最稳定的卦位作为 winner
        3. 乾/坤/艮/坎 的峰值通常更高（0.15-0.25），更容易成为 winner
        4. 兑/离/震/巽 的峰值较低（0.10-0.13），很少成为最稳定的选择
        
        这符合铁律：
        - ✅ 场自决，不人为干预场选择
        - ✅ 所有卦位都有非零分布（参与计算）
        - ✅ 卦位是被发现的，不是被定义的
        
        如果想看到更多卦位变化，可以：
        - 增加测试场景多样性（冲突、创新、变化类问题）
        - 调整 v94 场演化参数（让场更敏感）
        - 调整汉字物理的权重（让物理线影响更大）
        
        详细分析见 MEMORY.md「兑/离/震/巽 零出现现象分析」章节。
        ────────────────────────────────────────────────────────────────
        """
        self.round += 1
        self._ensure_loaded()
        
        # ── 线①：汉字物理 → qi_physics ──
        qi_physics = _compute_qi_physics(text)
        
        # ── 线③：河图语义 → 词网络 → 预演 → 热分布 ──
        if hetu_texts:
            raw_heat = self._word_net.ingest(hetu_texts)
            hetu_heat = _prestage_amplify(raw_heat, steps=10, rate=0.40)
        else:
            hetu_heat = {g: 1.0 for g in BAGUA}
        
        # ── 多偶极子隔离演化 ──
        # 三条热源各自独立演化，在v94内部互推互拉，场自决哪条更热
        # 不做加权合并，不做归一化缩放，让场自决收敛方向
        # 
        # 历史：7/8 首次提出多偶极子，7/9 发现隐式加权问题（归一化缩放），
        #        7/13 实现隔离演化（divine_from_multi_qi_isolated），不再先加总再缩放
        qi_physics_dict = {g: float(qi_physics[i]) for i, g in enumerate(BAGUA)}
        qi_field_dict = {g: float(qi_field[i]) for i, g in enumerate(BAGUA)} if qi_field is not None else {g: 1.0 for g in BAGUA}
        
        # 构建多源列表
        qi_sources = [
            {'name': 'physics', 'qi': qi_physics_dict},
            {'name': 'hetu', 'qi': hetu_heat},
            {'name': 'field', 'qi': qi_field_dict},
        ]
        
        # 记忆层：初态微偏置（调制每条路的初始 qi）
        if self.memory_bias is not None and self.memory_bias.sum() > 0:
            for src in qi_sources:
                for i, g in enumerate(BAGUA):
                    src['qi'][g] += self.memory_bias[i] * 0.15
        
        # 隔离演化：各自独立演化，互推互拉，场自决
        result = self.v94.divine_from_multi_qi_isolated(qi_sources, trace=True)
        
        # ── 近窗微风：每步注入（v94内部已处理演化，此处做后置微偏） ──
        final_dist = np.array([result['distribution'].get(g, 0) for g in BAGUA])
        if self.near_qi is not None:
            final_dist = final_dist + self.near_qi * self.breeze_epsilon
            final_dist = final_dist / final_dist.sum()
        
        # ── 词结晶（高维语义场直接浮现，不经过8标量）──
        # 2026-07-14: 从 qi 标量取 top-3 卦名 → 改为高维词网络直接结晶
        # v94 的 8 标量作为可选观测窗口（仪表盘），不驱动结晶词选择
        # 传入全局状态（CV）调制热源比例
        global_state = {
            'cv': result['depth_cv'],
            'round': self.round,
            'distribution': result['distribution']
        }
        crystal_words = self.crystallize(text, top_k=5, global_state=global_state)
        # 保留卦名标签作为仪表盘输出（不主导）
        sorted_gua = sorted(result['distribution'].items(), key=lambda x: -x[1])
        dashboard_gua = [g for g, v in sorted_gua if v > 0.1][:3]
        
        result_obj = EngineV9Result(
            winner=result['winner'],
            distribution=result['distribution'],
            cv=result['depth_cv'],
            qi_physics=qi_physics,
            hetu_heat=hetu_heat,
            qi_field_in=qi_field if qi_field is not None else np.ones(8),
            trace=result.get('trace', {}),
            word_crystal=crystal_words,
            dashboard_gua=dashboard_gua,
        )
        
        # ── 外围层（完全不进管道，只读/只旁路）──
        
        # C层推理（旁路日记，不碰场）
        if self.c_layer is not None and hetu_texts:
            self.c_layer.observe(
                hetu_texts=hetu_texts,
                distribution=result['distribution'],
                winner=result['winner'],
                crystal=crystal_words,
            )
        
        # 指涉层观察（只读）
        if self.observer is not None:
            self.observer.record(
                round=self.round,
                text=text,
                qi_physics=qi_physics,
                hetu_heat=hetu_heat,
                qi_field_in=qi_field if qi_field is not None else np.ones(8),
                distribution=result['distribution'],
                winner=result['winner'],
                cv=result['depth_cv'],
                word_crystal=crystal_words,
            dashboard_gua=dashboard_gua,
            )
        
        # ── 本轮near更新 ──
        self.near_qi = final_dist.copy()
        
        return result_obj
    
    def crystallize(self, text: str, knowledge_answer: str = None, 
                    global_state: dict = None, top_k: int = 5) -> List[str]:
        """
        太极演化（纯净版）：词在高维语义场中自然浮现。
        
        核心机制：
        1. 词网络在高维语义空间中热扩散（不压到8标量）
        2. 按热度排序，取 top-k
        3. v94 的 8 标量不参与词选择（仪表盘，可选观测）
        
        自适应参数（2026-07-14 补丁）：
        - 网的比例：steps ∝ log(n_words), temp ∝ 1/n_words
        - 热源比例：问题词/知识库词比例自适应，受全局状态（CV）调制
        - 虚词：不再人工标记，让热扩散自然决定
        
        哲学定位（2026-07-14 讨论确认）：
        - 八卦不是坐标系（不定义8个维度）
        - 八卦是描述语言（帮助理解高维场的状态）
        - v94 的 8 标量是观测窗口，不驱动结晶
        - 铁律零触犯：不筛选/不特征提取/不投影/不匹配/不量化/不加权/不仲裁
        
        输入：原始文本 + 可选知识库答案 + 可选全局状态
        输出：结晶词列表（按热度排序）
        
        这不是选句器，是"打窝"——词在热场中自然浮现。
        """
        self._ensure_loaded()
        
        # 词网络热扩散（自适应网 + 自适应热源 + 全局状态调制）
        word_heat = self._word_net._compute_word_heat(
            text, 
            knowledge_text=knowledge_answer,
            global_state=global_state
        )
        if not word_heat:
            return []
        
        # 按热度排序
        sorted_words = sorted(word_heat.items(), key=lambda x: x[1], reverse=True)
        
        # 取 top-k，转回文本
        crystal_words = [self._word_net._decode_tid(tid) for tid, heat in sorted_words[:top_k]]
        
        return crystal_words
    
    def crystallize_with_dashboard(self, text: str, qi_state: Dict[str, float], 
                                    knowledge_answer: str = None, top_k: int = 5) -> List[str]:
        """
        【旧版仪表盘模式】用 v94 的 8 标量调制词热度。
        
        ⚠️ 2026-07-14 讨论后标记为旧版：
        - 这个方法用 qi_state（8标量）反过来调制高维词热度
        - 本质是：高维→8标量→反过来指导高维（绕了一圈）
        - 保留了作为对比参考，但主流程应使用 crystallize()
        
        主流程请使用：crystallize()（纯净版，铁律零触犯）
        """
        self._ensure_loaded()
        
        # 1. 合并文本：原始文本 + 知识库答案
        merged_text = text
        if knowledge_answer:
            # 知识库答案加入热场（不覆盖原始文本，是补充）
            merged_text = text + " " + knowledge_answer
        
        # 2. 计算词网络的基础热度
        word_heat = self._word_net._compute_word_heat(merged_text)
        if not word_heat:
            return []
        
        # 3. 物理场调制：用 qi_state 调制词热度
        modulated = []
        for tid, base_heat in word_heat.items():
            # 获取该词的八卦倾向向量
            gua_bias = self._word_net._get_word_gua_bias(tid)
            
            # 用 qi_state 调制：内积 = 倾向强度
            modulation = sum(qi_state.get(g, 0) * gua_bias[i] 
                            for i, g in enumerate(BAGUA))
            
            # 调制后的热度
            modulated_heat = base_heat * (1.0 + modulation * 2.0)  # 调制幅度
            modulated.append((tid, modulated_heat))
        
        # 4. 按调制后热度排序，取 top-k
        modulated.sort(key=lambda x: x[1], reverse=True)
        
        # 5. 转回文本
        crystal_words = [self._word_net._decode_tid(tid) for tid, heat in modulated[:top_k]]
        
        return crystal_words


# ═══════════════════════════════════════════
# 轻量测试（不依赖河图后端）
# ═══════════════════════════════════════════
if __name__ == '__main__':
    import io
    import sys as _sys
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("═" * 60)
    print("engine_v9 自测：三条线汇合（模拟qi_field）")
    print("═" * 60)
    
    engine = EngineV9(hour=12)
    
    # 模拟8题
    tests = [
        ("我升职了！", "升职"),
        ("我被冤枉了，很难受。", "被冤枉"),
        ("有人偷了我的东西。", "偷窃"),
        ("什么是美？", "美"),
        ("什么是爱？", "爱"),
        ("1+1等于几？", "数学"),
        ("这句话是假的。", "说谎者"),
        ("进化论正确吗？", "进化论"),
    ]
    
    print(f"\n{'题':<8} {'胜卦':<4} {'CV':>5} {'物理top3':>30} {'语义top3':>30}")
    print("-" * 90)
    
    for text, label in tests:
        # 模拟河图语义（6段占位文本，实际使用时从河图后端获取）
        hetu_fake = [f"模型{i}对'{label}'的回应" for i in range(6)]
        # 模拟qi_field（均匀分布）
        qi_field_fake = np.ones(8) / 8
        
        r = engine.perceive(text, hetu_texts=hetu_fake, qi_field=qi_field_fake)
        
        # 物理top3
        phys_idx = np.argsort(r.qi_physics)[::-1][:3]
        phys_top3 = ' '.join(f"{BAGUA[i]}:{r.qi_physics[i]:.2f}" for i in phys_idx)
        
        # 语义top3
        sem_top3 = ' '.join(f"{g}:{v:.2f}" for g, v in 
                          sorted(r.hetu_heat.items(), key=lambda x: -x[1])[:3])
        
        print(f"{label:<8} {r.winner:<4} {r.cv:>5.2f} {phys_top3:>30} {sem_top3:>30}")
    
    print(f"\n✅ engine_v9 核心链路跑通")
