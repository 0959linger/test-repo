"""
v94 气场方案 - 增强版（继承v93三大机制）

═══════════════════════════════════════════════════════════════
架构历史约束（v92/v93 教训 — 不可遗忘）
═══════════════════════════════════════════════════════════════

[教训1 — v92 "全乾锁死" (2026-06)]
  16 偶极子争 8 卦位 → 共振滚雪球 → 8 题全乾 CV=7.9。
  根因：卦位分布"高度统一"，场丧失分辨力。
  约束：归一化熵必须 ≥ 60%（五维核心验证·维度5）。
  测试：engine/test_core_v2.py

[教训2 — v93 "兑为隐卦" (2026-06)]
  兑卦 0% 领地但通过拓扑耦合影响全局，去掉兑 → 其他卦位变化。
  约束：任何卦位不应被系统性地排除出 winner 候选。
  监控：8 题场自决分组的卦位覆盖率。

[教训3 — v94 "截断插入无用" (2026-07)]
  自感冷却（v94 内部修改）验证失败——阀门式截断插入被物理架构排斥。
  COOLING_SUPPRESS=0.15 作为冷却→qi 桥梁太弱，cascade_chase 一步抢回。
  约束：反思不能是"加在架构上的步骤"，只能从整体链路自然产生。
  文档：finding-order/docs/longzhan.md

[教训4 — v94 "枕边风绕过免疫" (2026-07)]
  暴力洗脑触发河图级联高烧（免疫反应），枕边风温柔渗透让河图停机。
  约束：架构怕枕边风不怕暴力。混合输入（多样化）=天然疫苗。
  测试：污染测试四组对照（见 MEMORY.md 2026-07-05）

[教训5 — 动态阈值改动 (2026-07-11)]
  原 CASCADE_THRESHOLD=0.3 硬编码 → 改为动态阈值 max(0.15, 0.2+0.2×spread)。
  动机：场均匀时阈值应降低（让微小差异级联），场拉开后阈值应升高（防过度追逐）。
  验证：五维核心验证全部通过（test_core_v2.py），ρ=0.643 熵=71.9%。
  注意：改动后重新验证分辨力底线——确认未因动态阈值导致"高度统一"。

═══════════════════════════════════════════════════════════════
五维核心验证（每次改动后必须通过）
═══════════════════════════════════════════════════════════════
  1. 结构诚实性：spread→CV 正关联（Spearman ρ > 0）
  2. 收敛底线：边界输入不崩溃
  3. 自然聚类：场自决分组（不预设语义）
  4. 临界区报告：噪声下分布（不设 pass/fail）
  5. 分辨力底线：归一化熵 ≥ 60%（防"高度统一"）
  测试脚本：engine/test_core_v2.py
  标准文档：docs/core_stability_philosophy_review.md
═══════════════════════════════════════════════════════════════

从v93继承的机制：
1. 级联追逐：气量梯度驱动，超过阈值才触发，20%损耗
2. 冷却度：被级联的卦位累积冷却，压制下次增长
3. 破坏性地震：削平级联源卦，防止过热垄断

原有v94机制保持不变：
- 7D特征提取
- 三偶极子初始化 + sin劈开
- 扩散 + qi^1.5聚集
- 蒸汽流动
- 极端度检测 + 太极衰减
- 温度计 + 凝结

架构：
  7D特征 → 偶极子初始化 → 气场旋转 → 
  ┌─ 扩散+聚集+衰减 ─┐
  │  级联追逐(新)     │ × N步
  │  冷却度(新)       │
  │  地震(新)         │
  └───────────────────┘
  极端度 → 太极衰减 → 温度计 → 凝结 → 卦象
"""

import numpy as np
from typing import List, Dict, Tuple


# ============================================================
# 物理参数
# ============================================================

# 冲刷参数
FLUSH_STEPS = 50
FLUSH_RATE = 0.1
DIFFUSION_RATE = 0.05
AGGREGATION_MULTIPLIER = 2.0

# 级联追逐参数（从v93继承 → v94 动态阈值）
# [2026-07-11 动态阈值] 原硬编码 CASCADE_THRESHOLD=0.3 改为根据 qi 场的 spread
#   (max-min) 动态调整。物理直觉：场越均匀（信号挤在一起），静摩擦阈值越低，
#   让微小差异能被级联放大；场越拉开后阈值升高，防止过度追逐破坏结构。
#   - CASCADE_THRESHOLD_BASE = 0.3 作为锚点（spread≈1.0 时还原原行为）
#   - 动态公式：threshold = 0.2 + 0.2 × spread（spread<0.5 时用 0.2 地板）
#   - spread=0.5 → th=0.2, spread=1.0 → th=0.3, spread=2.0 → th=0.6
CASCADE_THRESHOLD_BASE = 0.3  # 锚点阈值（spread≈1.0 时的行为）
CASCADE_RATIO = 0.4            # 级联量 = 差 × 此比例
CASCADE_LOSS = 0.2             # 20%损耗

# 冷却度参数（从v93继承）
COOLING_DECAY = 0.92       # 每步衰减
COOLING_SUPPRESS = 0.15    # 冷却压制系数

# 地震参数（从v93继承）
EARTHQUAKE_INTERVAL = 10   # 每N步触发一次
EARTHQUAKE_STRENGTH = 0.10 # 削平10%
TRACE_INTERVAL = 5          # trace采样间隔（仅divine_from_qi trace=True时）


class V94QichangEnhanced:
    """v94 气场增强版"""
    
    def __init__(self):
        # 先天八卦，乾南坤北
        self.trigram_angles = {
            '乾': 0,       # 南
            '兑': 45,      # 东南
            '离': 90,      # 东
            '震': 135,     # 东北
            '坤': 180,     # 北
            '艮': 225,     # 西北
            '坎': 270,     # 西
            '巽': 315,     # 西南
        }
        self.trigrams = list(self.trigram_angles.keys())
        
        # 爻位属性：阳爻比例（乾=3/3, 坤=0/3, 震=1/3）
        self.yang_ratio = {
            '乾': 1.0,    # ☰ 阳阳阳
            '兑': 2/3,    # ☱ 阳阳阴
            '离': 2/3,    # ☲ 阳阴阳
            '震': 1/3,    # ☳ 阳阴阴
            '坤': 0.0,    # ☷ 阴阴阴
            '艮': 1/3,    # ☶ 阴阴阳
            '坎': 2/3,    # ☵ 阴阳阴（实际上八卦中坎是中阳，2阳？等等...）
            '巽': 2/3,    # ☴ 阴阳阳
        }
        # 纠正：坎 ☵ = 阴阴阳 = 1/3 阳爻
        self.yang_ratio['坎'] = 1/3
        
        # 冷却度状态（从v93继承）
        self.cooling = {t: 0.0 for t in self.trigrams}
        
        # 统计
        self.cascade_records = []
        self.steps = 0
    
    # ============================================================
    # 7D特征提取
    # ============================================================
    
    def extract_7d_features(self, logprobs: List[float]) -> Dict[str, float]:
        """提取7D物理特征"""
        arr = np.array(logprobs)
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std < 1e-10:
            skew = kurt = 0.0
        else:
            skew = float(np.mean(((arr - mean) / std) ** 3))
            kurt = float(np.mean(((arr - mean) / std) ** 4) - 3)
        
        x = np.arange(len(arr))
        slope, _ = np.polyfit(x, arr, 1)
        
        # 曲率：前后段std差异
        mid = len(arr) // 2
        curv = float(np.std(arr[mid:]) - np.std(arr[:mid]))
        
        # 熵
        p = np.exp(arr) / np.sum(np.exp(arr))
        p = p[p > 0]
        entropy = -np.sum(p * np.log2(p))
        
        return {
            'mean': mean, 'std': std, 'skew': skew, 'kurt': kurt,
            'trend': slope, 'curvature': curv, 'entropy': entropy,
        }
    
    # ============================================================
    # 三偶极子初始化 + sin劈开
    # ============================================================
    
    def initialize_qichang(self, features: Dict[str, float]) -> Dict[str, float]:
        """直接分发：7D特征 × 八卦坐标 → 初始气量
        
        去掉cos/sin/weight翻译链，卦位坐标直接决定消费哪些特征
        所有特征自然归一化，不加人为权重
        """
        # 卦位坐标（八卦盘，单位圆）
        coords = {
            '乾': (1.0, 0.0), '兑': (0.707, 0.707), '离': (0.0, 1.0),
            '震': (-0.707, 0.707), '坤': (-1.0, 0.0), '艮': (-0.707, -0.707),
            '坎': (0.0, -1.0), '巽': (0.707, -0.707),
        }
        
        # 自然归一化：除以各特征的典型范围
        # mean: logprob通常 -2 到 0 → /1.0
        # std: 通常 0 到 2 → /1.0
        # skew: 通常 -3 到 3 → /2.0
        # kurt: 通常 -3 到 20 → /5.0
        # trend: 通常 -0.1 到 0.1 → /0.05
        # curvature: 通常 -1 到 1 → /0.5
        # entropy: 通常 0 到 5 → /2.0
        
        mean_n = features['mean'] / 1.0
        std_n = features['std'] / 1.0
        skew_n = features['skew'] / 2.0
        kurt_n = features['kurt'] / 5.0
        trend_n = features['trend'] / 0.05
        curv_n = features['curvature'] / 0.5
        entropy_n = features['entropy'] / 2.0
        
        qichang = {}
        for t in self.trigrams:
            x, y = coords[t]
            
            # 基础值：坐标 × 归一化特征（无人为权重）
            base = x * mean_n + y * std_n
            
            # 高阶调制（无人为权重）
            modulation = (
                abs(x) * skew_n +
                abs(y) * kurt_n +
                x * trend_n +
                y * curv_n
            )
            
            # 熵作为全局调制
            entropy_scale = 0.5 + 0.2 * entropy_n
            
            # 最终值
            qi = (base + modulation) * entropy_scale
            qi = max(0.1, qi + 5.0)
            
            qichang[t] = qi
        
        return qichang
    
    # ============================================================
    # 趋势旋转
    # ============================================================
    
    def calc_trend_rotation(self, features: Dict[str, float]) -> float:
        """趋势旋转角度"""
        return (features['trend'] + features['curvature']) * 2.0
    
    def rotate_qichang(self, qichang: Dict[str, float], rot_rad: float) -> Dict[str, float]:
        """气场旋转"""
        new_qichang = {t: 0.0 for t in self.trigrams}
        
        for t in self.trigrams:
            angle_rad = np.deg2rad(self.trigram_angles[t])
            new_angle = angle_rad + rot_rad
            x, y = np.cos(new_angle), np.sin(new_angle)
            
            # 找最近卦位
            min_dist = float('inf')
            closest = t
            for t2 in self.trigrams:
                t2_rad = np.deg2rad(self.trigram_angles[t2])
                dist = np.sqrt((x - np.cos(t2_rad))**2 + (y - np.sin(t2_rad))**2)
                if dist < min_dist:
                    min_dist = dist
                    closest = t2
            
            new_qichang[closest] += qichang[t]
        
        return new_qichang
    
    # ============================================================
    # 冲刷演化（含冷却度压制）
    # ============================================================
    
    def flush_evolution(self, qichang: Dict[str, float]) -> Dict[str, float]:
        """
        冲刷演化：扩散 + 聚集 + 衰减 + 冷却度压制
        
        冷却度从v93继承：被级联过的卦位累积cooling，
        压制其下次聚集增长。
        """
        new_qichang = qichang.copy()
        
        total_qi = sum(qichang.values())
        avg_qi = total_qi / len(self.trigrams)
        if total_qi < 1e-10:
            return qichang
        
        attract_pool = sum(max(qi, 0) ** 1.5 for qi in qichang.values())
        
        for t in self.trigrams:
            qi = qichang[t]
            
            # 冷却度压制（从v93继承）
            cool_suppress = max(0.0, 1.0 - self.cooling[t] * COOLING_SUPPRESS)
            
            # 1. 扩散
            diffusion = (qi - avg_qi) * DIFFUSION_RATE
            
            # 2. 聚集（受冷却度压制）
            if attract_pool > 1e-10 and qi > 0:
                attract = (qi ** 1.5 / attract_pool) * FLUSH_RATE * total_qi * AGGREGATION_MULTIPLIER * cool_suppress
            else:
                attract = 0
            
            # 3. 衰减
            decay = qi * FLUSH_RATE
            
            new_qichang[t] = qi - diffusion + attract - decay
        
        return new_qichang
    
    # ============================================================
    # 级联追逐（从v93继承）
    # ============================================================
    
    def cascade_chase(self, qichang: Dict[str, float]) -> Tuple[Dict[str, float], List]:
        """
        级联追逐：气量从高到低流动 + 相错关系（cos注入）
        
        cos注入：对立卦位之间流动更强
        - 乾⇄坤（180°）：cos=-1 → 1-cos=2 → 流动加倍
        - 乾⇄兑（45°）：cos=0.707 → 1-cos=0.293 → 流动减弱
        物理直觉：对立卦位天生互流，相邻卦位之间克制
        
        [2026-07-11 动态阈值] 级联阈值不再硬编码，改为根据当前 qi 场的
        spread（max-min）动态计算。场越均匀 → 阈值越低 → 微小差异被级联放大；
        场越拉开 → 阈值升高 → 防止过度追逐破坏结构。
        """
        new_qichang = qichang.copy()
        records = []
        
        # 动态阈值：根据 qi 场的 spread 计算
        #   spread = max(qi) - min(qi)
        #   阈值 = 0.2 + 0.2 × spread，地板 0.15
        #   物理：场均匀(spread→0)时 th→0.15，差异再小也能级联
        #        场拉开(spread→2)时 th→0.6，只有显著差异才触发追逐
        qi_values = list(qichang.values())
        qi_spread = max(qi_values) - min(qi_values)
        dynamic_threshold = max(0.15, 0.2 + 0.2 * qi_spread)
        
        # 按气量排序
        sorted_trigrams = sorted(self.trigrams, key=lambda t: new_qichang[t], reverse=True)
        
        for i in range(len(sorted_trigrams) - 1):
            src = sorted_trigrams[i]
            dst = sorted_trigrams[i + 1]
            
            diff = new_qichang[src] - new_qichang[dst]
            if diff > dynamic_threshold:
                # cos相错注入：对立卦位流动更强
                angle_diff = abs(self.trigram_angles[src] - self.trigram_angles[dst])
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                cos_factor = 1.0 - np.cos(np.deg2rad(angle_diff))  # 0（同向）→ 2（对立）
                
                overflow = diff * CASCADE_RATIO * cos_factor
                actual_flow = overflow * (1.0 - CASCADE_LOSS)
                
                new_qichang[src] -= overflow
                new_qichang[dst] += actual_flow
                
                records.append((src, dst, overflow))
        
        return new_qichang, records
    
    # ============================================================
    # 冷却度更新（从v93继承）
    # ============================================================
    
    def update_cooling(self, cascade_records: List) -> None:
        """
        冷却度累积与衰减 + 爻位注入（weight）
        
        爻位：阳爻越多冷却越慢（阳气保温），阴爻越多冷却越快
        - 乾(3阳)：cooling × 0.25
        - 坤(0阳)：cooling × 1.0
        """
        for src, dst, overflow in cascade_records:
            # 爻位weight：阳爻比例越低，冷却越快
            yin_ratio = 1.0 - self.yang_ratio[src]
            cool_weight = 0.5 + yin_ratio * 0.5  # 0.5(全阳) → 1.0(全阴)
            self.cooling[src] += overflow * 0.02 * cool_weight
        
        for t in self.trigrams:
            self.cooling[t] *= COOLING_DECAY
    
    # ============================================================
    # 破坏性地震（从v93继承）
    # ============================================================
    
    def earthquake(self, qichang: Dict[str, float], cascade_records: List) -> Dict[str, float]:
        """
        破坏性地震：削平级联源卦 + 物极必反（sin注入）
        
        sin注入：极端度决定地震强度，不是恒定力度
        - 极端度=1.0：sin(π)=0 → 不地震（正常状态）
        - 极端度=1.5：sin(1.5π)=-1 → 地震最强（极阳转阴）
        - 极端度=2.0：sin(2π)=0 → 地震消失（反转完成）
        物理直觉：极点到顶自动衰弱，过顶地震自然停止
        """
        new_qichang = qichang.copy()
        
        if not cascade_records:
            return new_qichang
        
        # 物极必反：极端度决定地震强度
        ext = self.detect_extremeness(qichang)
        sin_factor = abs(np.sin(ext * np.pi))  # 0(ext=1) → 1(ext=1.5) → 0(ext=2)
        
        sources = set(src for src, dst, overflow in cascade_records)
        for src in sources:
            new_qichang[src] *= (1.0 - EARTHQUAKE_STRENGTH * sin_factor)
        
        return new_qichang
    
    # ============================================================
    # 蒸汽流动
    # ============================================================
    
    def steam_flow(self, qichang: Dict[str, float]) -> Dict[str, float]:
        """蒸汽流动：梯度驱动"""
        new_qichang = qichang.copy()
        avg_qi = sum(qichang.values()) / len(self.trigrams)
        
        for t in self.trigrams:
            flow = -(qichang[t] - avg_qi) * 0.1
            new_qichang[t] = qichang[t] + flow
        
        return new_qichang
    
    # ============================================================
    # 极端度检测
    # ============================================================
    
    def detect_extremeness(self, qichang: Dict[str, float]) -> float:
        """极端度 = max / mean"""
        values = [max(qi, 0) for qi in qichang.values()]
        if not values or sum(values) < 1e-10:
            return 1.0
        return max(values) / (np.mean(values) + 1e-10)
    
    # ============================================================
    # 太极衰减
    # ============================================================
    
    def taiji_decay(self, extremeness: float, decay_factor: float = 0.5) -> float:
        """太极衰减：极端度越高，衰减越强"""
        return 1.0 / (1.0 + (extremeness - 1.0) * decay_factor)
    
    # ============================================================
    # 温度计
    # ============================================================
    
    def calc_temperature(self, trend_rotation: float) -> float:
        """温度计"""
        return 3.0 + abs(trend_rotation) * 5.0
    
    # ============================================================
    # 凝结
    # ============================================================
    
    def condensation(self, qichang: Dict[str, float], k: float):
        """softmax凝结"""
        qi_values = np.array([qichang[t] for t in self.trigrams])
        qi_values = qi_values - np.max(qi_values)
        exp_qi = np.exp(k * qi_values)
        softmax_qi = exp_qi / np.sum(exp_qi)
        
        distribution = {t: float(softmax_qi[i]) for i, t in enumerate(self.trigrams)}
        winner = max(distribution, key=distribution.get)
        
        avg = 1.0 / len(self.trigrams)
        active = [t for t, v in distribution.items() if v > avg * 2]
        
        return winner, distribution, active
    
    # ============================================================
    # 太极度（CV）
    # ============================================================
    
    def calc_taiji_degree(self, qichang: Dict[str, float]) -> float:
        """太极度 = CV（从v93继承）"""
        values = list(qichang.values())
        mean_val = np.mean(values)
        if abs(mean_val) < 1e-10:
            return 0.0
        return np.std(values) / abs(mean_val)
    
    # ============================================================
    # 完整推演
    # ============================================================
    
    def divine(self, logprobs: List[float], trace: bool = False) -> Dict:
        """
        完整推演流程
        
        7D特征 → 偶极子初始化 → 气场旋转 →
        [冲刷+级联+冷却+地震] × N步 →
        极端度 → 太极衰减 → 温度计 → 凝结
        """
        self.cooling = {t: 0.0 for t in self.trigrams}
        self.cascade_records = []
        self.steps = 0
        trace_steps = [] if trace else None
        
        # 1. 7D特征提取
        features = self.extract_7d_features(logprobs)
        
        # 2. 三偶极子初始化 + sin劈开
        qichang = self.initialize_qichang(features)
        
        # 3. 趋势旋转
        trend_rotation = self.calc_trend_rotation(features)
        qichang = self.rotate_qichang(qichang, trend_rotation)
        
        # 4. 冲刷演化（含v93三大机制）
        for step in range(FLUSH_STEPS):
            self.steps += 1
            
            # 4a. 冲刷（扩散+聚集+衰减+冷却度压制）
            qichang = self.flush_evolution(qichang)
            
            # 4b. 级联追逐（从v93继承）
            qichang, cascade_records = self.cascade_chase(qichang)
            
            # 4c. 冷却度更新（从v93继承）
            self.update_cooling(cascade_records)
            
            # 4d. 破坏性地震（从v93继承，每N步一次）
            if step % EARTHQUAKE_INTERVAL == 0 and cascade_records:
                qichang = self.earthquake(qichang, cascade_records)
            
            # 4e. 蒸汽流动（每10步）
            if step % 10 == 0:
                qichang = self.steam_flow(qichang)
            
            # 4f. 自然衰减
            for t in self.trigrams:
                qichang[t] *= (1 - FLUSH_RATE)
                qichang[t] = max(0, qichang[t])

            if trace and step % TRACE_INTERVAL == 0:
                top_g = max(qichang, key=qichang.get)
                trace_steps.append({
                    'step': step,
                    'top_gua': top_g,
                    'top_value': round(qichang[top_g], 3),
                    'qichang': dict(qichang),
                    'cascades': [(s, d, round(o, 3)) for s, d, o in cascade_records],
                    'cooling': dict(self.cooling),
                })
        
        # 5. 极端度检测
        extremeness = self.detect_extremeness(qichang)
        
        # 6. 太极衰减
        decay_weight = self.taiji_decay(extremeness)
        for t in self.trigrams:
            qichang[t] *= decay_weight
        
        # 7. 温度计
        k = self.calc_temperature(trend_rotation)
        
        # 8. 凝结
        winner, distribution, active_trigrams = self.condensation(qichang, k)
        
        # 9. 太极度
        taiji_degree = self.calc_taiji_degree(qichang)
        
        # 10. CV
        values = list(qichang.values())
        mean_val = np.mean(values)
        cv = np.std(values) / (mean_val + 1e-10) if mean_val > 1e-10 else 0.0
        
        return {
            'winner': winner,
            'distribution': distribution,
            'active_trigrams': active_trigrams,
            'features': features,
            'final_qichang': qichang,
            'extremeness': extremeness,
            'trend_rotation': trend_rotation,
            'temperature': k,
            'taiji_degree': taiji_degree,
            'depth_cv': cv,
            'cooling': dict(self.cooling),
        }


    def divine_from_qi(self, qichang: Dict[str, float], trace: bool = False) -> Dict:
        """
        从外部 qi 值直接推演，绕过 7D 特征提取和偶极子初始化。

        用于多信号源场景（如河图桥接）：8 条独立信号已确定
        各自卦位的初始气量，不需要单信号的 7D→偶极子翻译链。

        Args:
            qichang: {卦名: 初始气量}，8 个值必须齐全
        """
        # 重置状态
        self.cooling = {t: 0.0 for t in self.trigrams}
        self.cascade_records = []
        self.steps = 0
        trace_steps = [] if trace else None

        # 确保所有卦位有值
        for t in self.trigrams:
            if t not in qichang:
                qichang[t] = 2.0

        # ── 演化循环（与 divine() 步骤 4 完全一致）──
        for step in range(FLUSH_STEPS):
            self.steps += 1

            qichang = self.flush_evolution(qichang)
            qichang, cascade_records = self.cascade_chase(qichang)
            self.update_cooling(cascade_records)

            if step % EARTHQUAKE_INTERVAL == 0 and cascade_records:
                qichang = self.earthquake(qichang, cascade_records)
            if step % 10 == 0:
                qichang = self.steam_flow(qichang)

            for t in self.trigrams:
                qichang[t] *= (1 - FLUSH_RATE)
                qichang[t] = max(0, qichang[t])

            if trace and step % TRACE_INTERVAL == 0:
                top_g = max(qichang, key=qichang.get)
                trace_steps.append({
                    'step': step,
                    'top_gua': top_g,
                    'top_value': round(qichang[top_g], 3),
                    'qichang': dict(qichang),
                    'cascades': [(s, d, round(o, 3)) for s, d, o in cascade_records],
                    'cooling': dict(self.cooling),
                })

        # ── 后处理（与 divine() 步骤 5-10 一致）──
        extremeness = self.detect_extremeness(qichang)
        decay_weight = self.taiji_decay(extremeness)
        for t in self.trigrams:
            qichang[t] *= decay_weight

        # 温度计从 qi 分布自身计算：用 CV 替代 trend_rotation
        values = list(qichang.values())
        mean_val = np.mean(values)
        cv_val = np.std(values) / (mean_val + 1e-10) if mean_val > 1e-10 else 0.0
        k = 3.0 + cv_val * 5.0

        winner, distribution, active_trigrams = self.condensation(qichang, k)
        taiji_degree = self.calc_taiji_degree(qichang)

        result = {
            'winner': winner,
            'distribution': distribution,
            'active_trigrams': active_trigrams,
            'final_qichang': qichang,
            'extremeness': extremeness,
            'temperature': k,
            'taiji_degree': taiji_degree,
            'depth_cv': cv_val,
            'cooling': dict(self.cooling),
        }
        if trace:
            result['trace'] = trace_steps
        return result

    # ============================================================
    # 多偶极子推演（场自决方向）
    # ============================================================

    def divine_from_multi_qi(self, qi_sources: list, trace: bool = False) -> Dict:
        """
        多路 qi 信号同时注入，场自决融合方向。

        不做加权平均。每条路以完整强度注入，多偶极子在
        演化过程中自然互推互拉，最终场自己决定收敛方向。

        天然防锁：多路信号的差异本身就是持续扰动，
        不需要外部地震来打破垄断。

        Args:
            qi_sources: [{'name': 'physics', 'qi': {卦: 量}},
                         {'name': 'prism',   'qi': {卦: 量}}, ...]
        """
        self.cooling = {t: 0.0 for t in self.trigrams}
        self.cascade_records = []
        self.steps = 0
        trace_steps = [] if trace else None

        # ═══ 多偶极子求和 ═══
        # 每条路完整注入，不平权、不压缩
        qichang = {t: 0.0 for t in self.trigrams}
        source_shares = {t: {} for t in self.trigrams}  # 记录每条路的贡献
        
        for src in qi_sources:
            name = src['name']
            qi = src['qi']
            for t in self.trigrams:
                val = qi.get(t, 0.0)
                qichang[t] += val
                source_shares[t][name] = val
        
        # 记录初始状态（用于回算各路线索）
        initial_total = sum(qichang.values())
        if initial_total > 0:
            for t in self.trigrams:
                qichang[t] = qichang[t] / initial_total * 8.0  # 缩放到 v94 范围
        
        # ═══ 演化循环 ═══
        for step in range(FLUSH_STEPS):
            self.steps += 1
            
            qichang = self.flush_evolution(qichang)
            qichang, cascade_records = self.cascade_chase(qichang)
            self.update_cooling(cascade_records)
            
            if step % EARTHQUAKE_INTERVAL == 0 and cascade_records:
                qichang = self.earthquake(qichang, cascade_records)
            if step % 10 == 0:
                qichang = self.steam_flow(qichang)
            
            for t in self.trigrams:
                qichang[t] *= (1 - FLUSH_RATE)
                qichang[t] = max(0, qichang[t])
            
            if trace and step % TRACE_INTERVAL == 0:
                top_g = max(qichang, key=qichang.get)
                trace_steps.append({
                    'step': step,
                    'top_gua': top_g,
                    'top_value': round(qichang[top_g], 3),
                    'qichang': dict(qichang),
                })
        
        # ═══ 后处理 ═══
        extremeness = self.detect_extremeness(qichang)
        decay_weight = self.taiji_decay(extremeness)
        for t in self.trigrams:
            qichang[t] *= decay_weight
        
        values = list(qichang.values())
        mean_val = np.mean(values)
        cv_val = np.std(values) / (mean_val + 1e-10) if mean_val > 1e-10 else 0.0
        k = 3.0 + cv_val * 5.0
        
        winner, distribution, active_trigrams = self.condensation(qichang, k)
        taiji_degree = self.calc_taiji_degree(qichang)
        
        # ═══ 回溯各路线索：哪个源对最终 winner 贡献最大 ═══
        winner_contrib = {}
        for src in qi_sources:
            name = src['name']
            initial_qi = src['qi']
            # 该源在 winner 卦位上的初始注入量
            winner_contrib[name] = initial_qi.get(winner, 0.0)
        
        # 判断主导源
        if winner_contrib:
            dominant_source = max(winner_contrib, key=winner_contrib.get)
        else:
            dominant_source = 'unknown'
        
        # ═══ 天然 Δ：多路初始分布在 8 卦上的两两余弦距离 ═══
        def cos_dist(a, b):
            va = np.array([a.get(t, 0) for t in self.trigrams])
            vb = np.array([b.get(t, 0) for t in self.trigrams])
            return float(1.0 - np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))
        
        divergences = {}
        src_names = [s['name'] for s in qi_sources]
        for i in range(len(src_names)):
            for j in range(i+1, len(src_names)):
                key = f"{src_names[i]}_vs_{src_names[j]}"
                divergences[key] = cos_dist(
                    qi_sources[i]['qi'], qi_sources[j]['qi'])
        
        result = {
            'winner': winner,
            'distribution': distribution,
            'active_trigrams': active_trigrams,
            'final_qichang': qichang,
            'extremeness': extremeness,
            'temperature': k,
            'taiji_degree': taiji_degree,
            'depth_cv': cv_val,
            'cooling': dict(self.cooling),
            # 多路特有
            'dominant_source': dominant_source,
            'source_contrib': winner_contrib,
            'divergences': divergences,
            'source_count': len(qi_sources),
        }
        if trace:
            result['trace'] = trace_steps
        return result

    # ============================================================
    # 多偶极子隔离推演（场自决方向）
    # ============================================================

    def divine_from_multi_qi_isolated(self, qi_sources: list, trace: bool = False) -> Dict:
        """
        多路 qi 信号隔离演化，场自决融合方向。
        
        与 divine_from_multi_qi 的区别：
        - 不归一化缩放（避免隐式加权）
        - 各路在演化过程中互推互拉（偶极子相互作用）
        - 让尖锐路径在互推中被其他路的力平衡
        
        物理直觉：磁铁不是把磁场加起来再演化，而是各自产生磁场，
        磁场在空间中互推互拉。
        
        Args:
            qi_sources: [{'name': 'physics', 'qi': {卦: 量}},
                         {'name': 'prism',   'qi': {卦: 量}}, ...]
        """
        from copy import deepcopy
        
        self.cooling = {t: 0.0 for t in self.trigrams}
        self.cascade_records = []
        self.steps = 0
        trace_steps = [] if trace else None
        
        n_sources = len(qi_sources)
        
        # ═══ 隔离初始化：各路独立的 qichang ═══
        qichangs = []
        for src in qi_sources:
            qi = src['qi']
            # 不归一化，保持原始量级
            qichang = {t: float(qi.get(t, 0.0)) for t in self.trigrams}
            qichangs.append(qichang)
        
        # ═══ 演化循环 ═══
        for step in range(FLUSH_STEPS):
            self.steps += 1
            
            # 各自演化
            for i in range(n_sources):
                qichangs[i] = self.flush_evolution(qichangs[i])
                qichangs[i], cascade_records = self.cascade_chase(qichangs[i])
                self.update_cooling(cascade_records)
                
                if step % EARTHQUAKE_INTERVAL == 0 and cascade_records:
                    qichangs[i] = self.earthquake(qichangs[i], cascade_records)
                if step % 10 == 0:
                    qichangs[i] = self.steam_flow(qichangs[i])
                
                for t in self.trigrams:
                    qichangs[i][t] *= (1 - FLUSH_RATE)
                    qichangs[i][t] = max(0, qichangs[i][t])
            
            # 互推互拉：偶极子相互作用
            qichangs = self._dipole_interaction(qichangs)
            
            if trace and step % TRACE_INTERVAL == 0:
                total_qi = {t: sum(q[t] for q in qichangs) for t in self.trigrams}
                top_g = max(total_qi, key=total_qi.get)
                trace_steps.append({
                    'step': step,
                    'top_gua': top_g,
                    'top_value': round(total_qi[top_g], 3),
                    'total_qi': dict(total_qi),
                })
        
        # ═══ 后处理：合并各路 ═══
        qichang = {t: sum(q[t] for q in qichangs) for t in self.trigrams}
        
        extremeness = self.detect_extremeness(qichang)
        decay_weight = self.taiji_decay(extremeness)
        for t in self.trigrams:
            qichang[t] *= decay_weight
        
        values = list(qichang.values())
        mean_val = np.mean(values)
        cv_val = np.std(values) / (mean_val + 1e-10) if mean_val > 1e-10 else 0.0
        k = 3.0 + cv_val * 5.0
        
        winner, distribution, active_trigrams = self.condensation(qichang, k)
        taiji_degree = self.calc_taiji_degree(qichang)
        
        # ═══ 回溯各路线索 ═══
        winner_contrib = {}
        for i, src in enumerate(qi_sources):
            name = src['name']
            initial_qi = src['qi']
            winner_contrib[name] = initial_qi.get(winner, 0.0)
        
        dominant_source = max(winner_contrib, key=winner_contrib.get) if winner_contrib else 'unknown'
        
        # ═══ 天然 Δ：多路初始分布在 8 卦上的两两余弦距离 ═══
        def cos_dist(a, b):
            va = np.array([a.get(t, 0) for t in self.trigrams])
            vb = np.array([b.get(t, 0) for t in self.trigrams])
            return float(1.0 - np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))
        
        divergences = {}
        src_names = [s['name'] for s in qi_sources]
        for i in range(len(src_names)):
            for j in range(i+1, len(src_names)):
                key = f"{src_names[i]}_vs_{src_names[j]}"
                divergences[key] = cos_dist(
                    qi_sources[i]['qi'], qi_sources[j]['qi'])
        
        result = {
            'winner': winner,
            'distribution': distribution,
            'active_trigrams': active_trigrams,
            'final_qichang': qichang,
            'extremeness': extremeness,
            'temperature': k,
            'taiji_degree': taiji_degree,
            'depth_cv': cv_val,
            'cooling': dict(self.cooling),
            # 多路特有
            'dominant_source': dominant_source,
            'source_contrib': winner_contrib,
            'divergences': divergences,
            'source_count': len(qi_sources),
            'method': 'isolated',
        }
        if trace:
            result['trace'] = trace_steps
        return result
    
    def _dipole_interaction(self, qichangs: list, interaction_strength: float = 0.1) -> list:
        """
        偶极子互推互拉：各路在演化中互相影响。
        
        物理直觉：磁铁不是把磁场加起来，而是各自产生磁场，
        磁场在空间中互推互拉。
        
        Args:
            qichangs: 各路的 qichang 列表
            interaction_strength: 相互作用强度（0.1 表示 10% 的影响力）
        
        Returns:
            更新后的 qichangs 列表
        """
        from copy import deepcopy
        
        n = len(qichangs)
        if n < 2:
            return qichangs
        
        new_qichangs = [deepcopy(q) for q in qichangs]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # 计算 j 路对 i 路的力
                # 力的大小 = j 路在各卦位的强度
                for t in self.trigrams:
                    force = qichangs[j][t]
                    if force > 0.01:  # 只有足够强才施加力
                        # 推力：j 路强的地方，i 路被推开
                        # 这防止尖锐路径自动主导
                        new_qichangs[i][t] *= (1.0 - force * interaction_strength)
        
        return new_qichangs

    # ============================================================
    # 顺序多偶极子推演（先来后到）
    # ============================================================

    def divine_from_sequential_qi(self, qi_sources: list, trace: bool = False) -> Dict:
        """
        多路 qi 信号分阶段注入，先来后到 = 物理规则。

        阶段1：第1路注入，独立演化 FLUSH_STEPS//3 步 → 站住脚
        阶段2：第2路作为扰动注入，演化 FLUSH_STEPS//3 步
        阶段3：第3路注入，演化剩余步数
        ...
        
        每阶段的扰动强度 = 新路与当前场分布的余弦距离。
        距离大 → 扰动强 → 新路有更大发言权。
        距离小 → 扰动弱 → 当前场守住。

        Args:
            qi_sources: 按进入顺序排列的源列表
        """
        from copy import deepcopy
        
        self.cooling = {t: 0.0 for t in self.trigrams}
        self.cascade_records = []
        self.steps = 0
        trace_steps = [] if trace else None
        
        n_sources = len(qi_sources)
        steps_per_phase = max(5, FLUSH_STEPS // (n_sources + 1))
        
        # ═══ 第1路：独立演化 ═══
        first = qi_sources[0]
        qichang = {t: float(first['qi'].get(t, 0.0)) for t in self.trigrams}
        total = sum(qichang.values())
        if total > 0:
            for t in self.trigrams:
                qichang[t] = qichang[t] / total * 8.0
        
        for step in range(steps_per_phase):
            self.steps += 1
            qichang = self.flush_evolution(qichang)
            qichang, cascade_records = self.cascade_chase(qichang)
            self.update_cooling(cascade_records)
            if step % EARTHQUAKE_INTERVAL == 0 and cascade_records:
                qichang = self.earthquake(qichang, cascade_records)
            for t in self.trigrams:
                qichang[t] *= (1 - FLUSH_RATE)
                qichang[t] = max(0, qichang[t])
        
        # ═══ 后续各路：作为扰动注入 ═══
        phase_records = []  # 记录每个阶段的场状态
        
        for si in range(1, n_sources):
            src = qi_sources[si]
            new_qi = src['qi']
            
            # 计算当前场分布（归一化）
            total_q = sum(qichang.values())
            if total_q > 0:
                current_dist = {t: qichang[t] / total_q for t in self.trigrams}
            else:
                current_dist = {t: 0.125 for t in self.trigrams}
            
            # 扰动强度 = 新路与当前场的余弦距离
            va = np.array([current_dist[t] for t in self.trigrams])
            vb = np.array([new_qi.get(t, 0.125) for t in self.trigrams])
            vb = vb / (vb.sum() + 1e-10)
            cos_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10)
            perturbation = 1.0 - float(cos_sim)  # 0=完全一致, 1=完全正交
            
            # 扰动注入：新路的值按扰动强度混入当前场
            for t in self.trigrams:
                new_val = new_qi.get(t, 0.0)
                # 扰动大 → 新路占比大；扰动小 → 保留当前场
                mix_ratio = perturbation * 0.5  # 最多混入 50%
                qichang[t] = qichang[t] * (1 - mix_ratio) + new_val * mix_ratio * 8.0
            
            phase_records.append({
                'source': src['name'],
                'perturbation': float(perturbation),
                'cos_sim': float(cos_sim),
            })
            
            # 演化
            remaining = steps_per_phase if si < n_sources - 1 else FLUSH_STEPS - self.steps
            remaining = max(remaining, 5)
            
            for step in range(remaining):
                self.steps += 1
                if self.steps >= FLUSH_STEPS:
                    break
                qichang = self.flush_evolution(qichang)
                qichang, cascade_records = self.cascade_chase(qichang)
                self.update_cooling(cascade_records)
                if step % EARTHQUAKE_INTERVAL == 0 and cascade_records:
                    qichang = self.earthquake(qichang, cascade_records)
                for t in self.trigrams:
                    qichang[t] *= (1 - FLUSH_RATE)
                    qichang[t] = max(0, qichang[t])
        
        # ═══ 后处理 ═══
        extremeness = self.detect_extremeness(qichang)
        decay_weight = self.taiji_decay(extremeness)
        for t in self.trigrams:
            qichang[t] *= decay_weight
        
        values = list(qichang.values())
        mean_val = np.mean(values)
        cv_val = np.std(values) / (mean_val + 1e-10) if mean_val > 1e-10 else 0.0
        k = 3.0 + cv_val * 5.0
        
        winner, distribution, active_trigrams = self.condensation(qichang, k)
        taiji_degree = self.calc_taiji_degree(qichang)
        
        # ═══ 两两 Δ ═══
        def cos_dist(a, b):
            va = np.array([a.get(t, 0) for t in self.trigrams])
            vb = np.array([b.get(t, 0) for t in self.trigrams])
            return float(1.0 - np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))
        
        divergences = {}
        src_names = [s['name'] for s in qi_sources]
        for i in range(len(src_names)):
            for j in range(i+1, len(src_names)):
                key = f"{src_names[i]}_vs_{src_names[j]}"
                divergences[key] = cos_dist(
                    qi_sources[i]['qi'], qi_sources[j]['qi'])
        
        result = {
            'winner': winner,
            'distribution': distribution,
            'active_trigrams': active_trigrams,
            'final_qichang': qichang,
            'extremeness': extremeness,
            'temperature': k,
            'taiji_degree': taiji_degree,
            'depth_cv': cv_val,
            'cooling': dict(self.cooling),
            # 顺序特有
            'phase_records': phase_records,
            'divergences': divergences,
            'source_count': n_sources,
        }
        if trace:
            result['trace'] = trace_steps
        return result

    # ============================================================
    # 双组偶极子推演（伴随模式）
    # ============================================================

    def divine_from_dual_qi(self, qi_hetu: Dict[str, float],
                            qi_near: Dict[str, float],
                            trace: bool = False) -> Dict:
        """
        近窗偏置场模式：近窗不争卦位，只是每一步演化中的微弱背景温度梯度。

        qi_hetu: 河图8条信号的 qi（唯一的偶极子场）
        qi_near: 近窗瞬时记忆的 qi（背景吹风，不产生卦位）

        物理图像：
        - 只有河图的8卦偶极子在v94中演化
        - 近窗作为恒定微风：每一步 qi[t] += qi_near[t] × EPSILON
        - EPSILON 极小（0.005），近窗不决定方向，只是让同向的路少费一点力
        - 没有竞争、没有合并、没有第二组分布
        """
        EPSILON = getattr(self, '_breeze_epsilon', 0.005)

        # ── 重置状态 ──
        self.cooling = {t: 0.0 for t in self.trigrams}
        self.cascade_records = []
        self.steps = 0
        trace_steps = [] if trace else None

        # ── 只用河图 qi 初始化 ──
        qichang = {}
        for t in self.trigrams:
            qichang[t] = qi_hetu.get(t, 2.0)

        # ── 演化循环 ──
        for step in range(FLUSH_STEPS):
            self.steps += 1

            qichang = self.flush_evolution(qichang)
            qichang, cascade_records = self.cascade_chase(qichang)
            self.update_cooling(cascade_records)

            if step % EARTHQUAKE_INTERVAL == 0 and cascade_records:
                qichang = self.earthquake(qichang, cascade_records)
            if step % 10 == 0:
                qichang = self.steam_flow(qichang)

            for t in self.trigrams:
                qichang[t] *= (1 - FLUSH_RATE)
                qichang[t] = max(0, qichang[t])

            # ── 近窗偏置：每一步微微吹风 ──
            for t in self.trigrams:
                if t in qi_near:
                    qichang[t] += qi_near[t] * EPSILON

            if trace and step % TRACE_INTERVAL == 0:
                top_g = max(qichang, key=qichang.get)
                trace_steps.append({
                    'step': step,
                    'top_gua': top_g,
                    'top_value': round(qichang[top_g], 3),
                    'qichang': dict(qichang),
                    'cascades': [(s, d, round(o, 3)) for s, d, o in cascade_records],
                    'cooling': dict(self.cooling),
                })

        # ── 后处理 ──
        extremeness = self.detect_extremeness(qichang)
        decay_weight = self.taiji_decay(extremeness)
        for t in self.trigrams:
            qichang[t] *= decay_weight

        values = list(qichang.values())
        mean_val = np.mean(values)
        cv_val = np.std(values) / (mean_val + 1e-10) if mean_val > 1e-10 else 0.0
        k = 3.0 + cv_val * 5.0

        winner, distribution, active_trigrams = self.condensation(qichang, k)
        taiji_degree = self.calc_taiji_degree(qichang)

        result = {
            'winner': winner,
            'distribution': distribution,
            'active_trigrams': active_trigrams,
            'final_qichang': qichang,
            'extremeness': extremeness,
            'temperature': k,
            'taiji_degree': taiji_degree,
            'depth_cv': cv_val,
            'cooling': dict(self.cooling),
        }
        if trace:
            result['trace'] = trace_steps
        return result


# ============================================================
# 测试
# ============================================================

def get_logprobs(prompt: str, max_tokens: int = 50, port: int = 8091) -> list:
    """调用llama-server获取logprobs"""
    import requests
    url = f"http://127.0.0.1:{port}/completion"
    data = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "logprobs": 1
    }
    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return [t['logprob'] for t in result.get('completion_probabilities', [])]
    except Exception as e:
        print(f"[警告] LLM调用失败: {e}")
        return None


def test():
    """测试v94增强版"""
    import requests
    print("="*60)
    print("v94 气场增强版（继承v93三大机制）")
    print("="*60)
    
    core = V94QichangEnhanced()
    
    tests = [
        '今天适合出门吗？',
        '这个项目能成功吗？',
        '我该换工作吗？',
        '1+1等于几？',
        '写一首关于春天的诗',
        '设计一个高并发系统的架构需要考虑哪些因素？',
    ]
    
    for prompt in tests:
        print(f'\n{"─"*50}')
        print(f'  {prompt}')
        print(f'{"─"*50}')
        
        logprobs = get_logprobs(prompt, max_tokens=50)
        if not logprobs:
            print(f'  [跳过]')
            continue
        
        result = core.divine(logprobs)
        
        sorted_qi = sorted(result['final_qichang'].items(), key=lambda x: x[1], reverse=True)
        qi_str = '  '.join(f'{n}:{q:.2f}' for n,q in sorted_qi)
        
        print(f'  winner: {result["winner"]}')
        print(f'  qichang: {qi_str}')
        print(f'  CV={result["depth_cv"]:.3f}  极端度={result["extremeness"]:.3f}  太极度={result["taiji_degree"]:.3f}')
        print(f'  冷却度: {dict((k,round(v,3)) for k,v in sorted(result["cooling"].items(), key=lambda x:-x[1])[:3])}')
    
    print(f'\n{"="*60}')


if __name__ == "__main__":
    test()
