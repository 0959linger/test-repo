"""
v94 气场方案 - 完整版

核心思想：
- 高维感知（7D特征）
- 气场弥漫（不是几何映射，是物理弥漫）
- 自然演化（卦位相互作用 + 物理规则）
- 卦象涌现（稳态）

与v93的区别：
- v93：信号 → 2D点 → 匹配卦位 → 冲刷 → 凝结
- v94：多维信号 → 弥漫到场 → 场自然演化 → 凝结

v94的"气场"：
- 不是几何容器（没有x/y/z坐标）
- 是物理场（像气场、磁场）
- 高维数据进入场 → 场的物理规则自然处理
- "进来空间了数据就得少" = 从精确数值变成"气质"

完整物理管线（从v93移植）：
1. 7D特征提取
2. 三偶极子初始化 + sin劈开空间
3. 扩散 + qi^1.5聚集（替代v91冲刷）
4. 蒸汽流动（气量梯度驱动）
5. 极端度检测
6. 太极衰减
7. 温度计计算
8. 水蒸气凝结（softmax）
"""

import numpy as np
from typing import List, Dict, Tuple


class V94Qichang:
    """v94 气场方案"""
    
    def __init__(self):
        # 8卦位在八卦圆上的角度（纯几何，不含语义）
        # 乾南坤北，先天八卦序
        # 先天八卦，乾南坤北，阴阳对位
        self.trigram_angles = {
            '乾': 0,       # 南
            '兑': 45,      # 东南
            '离': 90,      # 东
            '震': 135,     # 东北
            '坤': 180,     # 北  ← 纯阴对纯阳
            '艮': 225,     # 西北
            '坎': 270,     # 西
            '巽': 315      # 西南
        }
        
        self.trigrams = list(self.trigram_angles.keys())
        
        # 物理参数
        self.flush_steps = 30
        self.flush_rate = 0.1
        self.diffusion_rate = 0.05
        self.aggregation_multiplier = 2.0  # 聚集强度（qi^1.5的系数）
        
    def extract_7d_features(self, logprobs: List[float]) -> Dict[str, float]:
        """提取7D物理特征"""
        arr = np.array(logprobs)
        
        # 手写skew和kurtosis
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std < 1e-10:
            skew = 0.0
            kurt = 0.0
        else:
            # 偏度：三阶矩
            skew = np.mean(((arr - mean) / std) ** 3)
            # 峰度：四阶矩 - 3（excess kurtosis）
            kurt = np.mean(((arr - mean) / std) ** 4) - 3
        
        return {
            'mean': mean,           # 中心
            'std': std,             # 波动
            'skew': skew,           # 偏斜
            'kurt': kurt,           # 尖锐
            'trend': self._calc_trend(arr), # 趋势
            'curvature': self._calc_curvature(arr),  # 曲率
            'entropy': self._calc_entropy(arr)       # 熵
        }
    
    def _calc_trend(self, arr: np.ndarray) -> float:
        """计算线性趋势"""
        x = np.arange(len(arr))
        slope, _ = np.polyfit(x, arr, 1)
        return slope
    
    def _calc_curvature(self, arr: np.ndarray) -> float:
        """计算曲率（前后段std差）"""
        mid = len(arr) // 2
        std1 = np.std(arr[:mid])
        std2 = np.std(arr[mid:])
        return std2 - std1
    
    def _calc_entropy(self, arr: np.ndarray) -> float:
        """计算熵（混乱程度）"""
        # 离散化
        hist, _ = np.histogram(arr, bins=20, density=True)
        hist = hist + 1e-10  # 避免0
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def initialize_qichang(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        初始化气场（三偶极子叠加 + sin物极必反）
        
        3个偶极子分别由不同特征对驱动，叠加后形成复杂空间模式
        不同输入 → 3个偶极子朝不同方向 → 叠加出多个热点
        全部7D特征都参与
        
        加入sin(π·x)机制：让气量可以取负值（阴），实现物极必反
        """
        total_qi = sum(abs(v) for v in features.values())
        
        # 3个偶极子方向（用6个特征对）
        # 偶极子1: mean/std
        d1 = np.rad2deg(np.arctan2(features['std']/5, features['mean']/10)) % 360
        # 偶极子2: skew/kurt
        d2 = np.rad2deg(np.arctan2(features['kurt']/5, features['skew']/2)) % 360
        # 偶极子3: trend/curvature
        d3 = np.rad2deg(np.arctan2(features['curvature']/2, features['trend']/0.1)) % 360
        
        # entropy 作为整体调制（高entropy放大差异，低entropy压缩）
        entropy_mod = 0.5 + 0.5 * np.tanh(features['entropy'] / 3)
        
        # 3个偶极子的权重（由各自特征对的幅度决定）
        w1 = np.sqrt((features['mean']/10)**2 + (features['std']/5)**2)
        w2 = np.sqrt((features['skew']/2)**2 + (features['kurt']/5)**2)
        w3 = np.sqrt((features['trend']/0.1)**2 + (features['curvature']/2)**2)
        w_total = w1 + w2 + w3 + 1e-10
        
        qichang = {}
        for trigram in self.trigrams:
            angle = self.trigram_angles[trigram]
            
            # 3个偶极子贡献
            qi = 0
            for d, w in [(d1, w1/w_total), (d2, w2/w_total), (d3, w3/w_total)]:
                diff = abs(angle - d)
                if diff > 180:
                    diff = 360 - diff
                qi += w * np.cos(np.deg2rad(diff))
            
            # entropy调制
            qi *= entropy_mod
            
            # 物极必反：sin(π·qi) 让气量可以取负值（阴）
            # 这是v93的核心机制，让某些卦位得到"负气"
            qi = np.sin(np.pi * qi)
            
            # 映射到 [0.3, 0.7]（sin后范围是[-1,1]）
            weight = 0.5 + 0.2 * qi
            
            qichang[trigram] = total_qi / 8 * weight * 2
        
        return qichang
    
    def flush_evolution(self, qichang: Dict[str, float]) -> Dict[str, float]:
        """
        冲刷演化（扩散 + 聚集 + 衰减）
        
        三种物理力共存：
        1. 扩散：高浓度 → 低浓度（线性，抹平差异）
        2. 聚集：气量^1.5吸引更多（超线性正反馈，类似引力）
        3. 衰减：自然耗散
        
        关键：聚集是超线性的（qi^1.5），所以高qi节点获得的优势
        比线性增长更快 → 对抗扩散的抹平 → 保留差异
        """
        new_qichang = qichang.copy()
        
        total_qi = sum(qichang.values())
        avg_qi = total_qi / len(self.trigrams)
        
        if total_qi < 1e-10:
            return qichang
        
        # 计算聚集池（超线性：qi^1.5 的总和）
        # 比qi²温和，但仍比线性qi强 → 保留差异但不垄断
        attract_pool = sum(qi ** 1.5 for qi in qichang.values())
        
        for trigram in self.trigrams:
            qi = qichang[trigram]
            
            # 1. 扩散：高于平均的流失，低于平均的获得（线性）
            diffusion = (qi - avg_qi) * self.diffusion_rate
            
            # 2. 聚集：超线性正反馈（qi^1.5 / sum(qi^1.5)）
            if attract_pool > 1e-10:
                attract = (qi ** 1.5 / attract_pool) * self.flush_rate * total_qi * self.aggregation_multiplier
            else:
                attract = 0
            
            # 3. 衰减：自然耗散（线性）
            decay = qi * self.flush_rate
            
            new_qichang[trigram] = qi - diffusion + attract - decay
        
        return new_qichang
    
    def steam_flow(self, qichang: Dict[str, float]) -> Dict[str, float]:
        """
        蒸汽流动（气量梯度驱动）
        
        从v93移植：高浓度区域向低浓度区域扩散
        但这里不是2D空间，是气场内的抽象流动
        """
        new_qichang = qichang.copy()
        
        # 计算平均气量
        avg_qi = sum(qichang.values()) / len(self.trigrams)
        
        # 每个卦位的气量变化 = 梯度驱动流动
        for trigram in self.trigrams:
            qi = qichang[trigram]
            gradient = qi - avg_qi
            
            # 正梯度（高于平均）：流失
            # 负梯度（低于平均）：获得
            flow = -gradient * 0.1  # 流动系数
            
            new_qichang[trigram] = qi + flow
        
        return new_qichang
    
    def detect_extremeness(self, qichang: Dict[str, float]) -> float:
        """
        极端度检测
        
        从v93移植：检测系统是否过热
        extremeness = max / mean
        """
        if not qichang:
            return 0.0
        
        max_qi = max(qichang.values())
        mean_qi = sum(qichang.values()) / len(qichang)
        
        if mean_qi < 1e-10:
            return 0.0
        
        return max_qi / mean_qi
    
    def taiji_decay(self, extremeness: float, decay_factor: float = 0.3) -> float:
        """
        太极衰减
        
        从v93移植：极端度越高，衰减越强
        物理意义：物极必反，过热则冷却
        
        v93公式：rotation_weight = 1.0 / (1.0 + (extremeness - 1.0) * decay_factor)
        """
        return 1.0 / (1.0 + (extremeness - 1.0) * decay_factor)
    
    def calc_trend_rotation(self, features: Dict[str, float]) -> float:
        """
        趋势旋转（从v93移植，适配v94）
        
        基于trend的信号方向偏转，物极必反的动态实现
        
        v93: trend_rotation = signal_trend * curvature * 0.5（基于2D方向）
        v94: rotation = (trend + curvature) * 2.0（基于标量，需要更大scale）
        """
        trend = features['trend']
        curvature = features['curvature']
        
        # 趋势旋转角度（弧度）
        # 用加法而非乘法，避免两个小数相乘导致过小
        rotation = (trend + curvature) * 2.0
        
        return rotation
    
    def rotate_qichang(self, qichang: Dict[str, float], rotation_rad: float) -> Dict[str, float]:
        """
        气场旋转（趋势旋转的气场版本）
        
        v93旋转2D信号方向，v94旋转气场分布
        物理意义：趋势驱动气场流动，物极必反的动态实现
        """
        # 将卦位角度加上旋转
        new_qichang = {}
        for trigram in self.trigrams:
            # 原始角度
            angle_deg = self.trigram_angles[trigram]
            angle_rad = np.deg2rad(angle_deg)
            
            # 旋转后的角度
            new_angle_rad = angle_rad + rotation_rad
            
            # 计算旋转后的位置（用于重新计算气量）
            x_new = np.cos(new_angle_rad)
            y_new = np.sin(new_angle_rad)
            
            # 找到最近的卦位
            min_dist = float('inf')
            closest_trigram = trigram
            for t in self.trigrams:
                t_angle = np.deg2rad(self.trigram_angles[t])
                t_x = np.cos(t_angle)
                t_y = np.sin(t_angle)
                dist = np.sqrt((x_new - t_x)**2 + (y_new - t_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_trigram = t
            
            # 将气量转移到最近的卦位
            if closest_trigram not in new_qichang:
                new_qichang[closest_trigram] = 0
            new_qichang[closest_trigram] += qichang[trigram]
        
        # 保证所有卦位都存在
        for trigram in self.trigrams:
            if trigram not in new_qichang:
                new_qichang[trigram] = 0
        
        return new_qichang
    
    def calc_temperature(self, trend_rotation_rad: float) -> float:
        """
        温度计计算（从v93移植）
        
        v93公式：k = base_k + |trend_rotation| × factor
        base_k = 3.0, factor = 5.0
        """
        k = 3.0 + abs(trend_rotation_rad) * 5.0
        return k
    
    def condensation(self, qichang: Dict[str, float], k: float = 3.0) -> Tuple[str, Dict[str, float], List[str]]:
        """
        水蒸气凝结（softmax + 活跃卦位）
        
        从v93移植：用softmax做平滑归一化
        k越大，分布越尖锐（winner-take-all）
        
        活跃卦位 = condensation > avg × 2（从v93移植）
        
        Returns:
            winner: 最大卦象
            distribution: softmax分布
            active_trigrams: 活跃卦位列表
        """
        if not qichang:
            return '', {}, []
        
        # softmax
        qi_values = np.array([qichang[t] for t in self.trigrams])
        
        # 数值稳定性
        qi_values = qi_values - np.max(qi_values)
        
        # exp(k * qi) / sum(exp(k * qi))
        exp_qi = np.exp(k * qi_values)
        softmax_qi = exp_qi / np.sum(exp_qi)
        
        # 构建分布
        distribution = {trigram: float(softmax_qi[i]) for i, trigram in enumerate(self.trigrams)}
        
        # 找最大值
        winner = max(distribution, key=distribution.get)
        
        # 活跃卦位 = condensation > avg × 2
        avg = 1.0 / len(self.trigrams)
        active_trigrams = [t for t, v in distribution.items() if v > avg * 2]
        
        return winner, distribution, active_trigrams
    
    def divine(self, logprobs: List[float]) -> Dict:
        """
        完整推演流程（从v93移植完整物理管线）
        
        多维信号 → 弥漫到场 → 场自然演化 → 卦象涌现
        
        1. 7D特征提取
        2. 三偶极子初始化 + sin劈开空间
        3. 趋势旋转（气场旋转）
        4. 冲刷演化（扩散 + 聚集 + 衰减）
        5. 蒸汽流动
        6. 极端度检测
        7. 太极衰减
        8. 温度计计算（基于trend_rotation）
        9. 水蒸气凝结（softmax + 活跃卦位）
        """
        # 1. 7D特征提取
        features = self.extract_7d_features(logprobs)
        
        # 2. 三偶极子初始化 + sin劈开空间
        qichang = self.initialize_qichang(features)
        
        # 3. 趋势旋转（气场旋转）
        trend_rotation = self.calc_trend_rotation(features)
        qichang = self.rotate_qichang(qichang, trend_rotation)
        
        # 4. 冲刷演化（50步）
        for step in range(self.flush_steps):
            qichang = self.flush_evolution(qichang)
            
            # 蒸汽流动（每10步一次）
            if step % 10 == 0:
                qichang = self.steam_flow(qichang)
            
            # 自然衰减（避免无限增长）
            for trigram in self.trigrams:
                qichang[trigram] *= (1 - self.flush_rate)
            
            # 保证非负
            for trigram in self.trigrams:
                qichang[trigram] = max(0, qichang[trigram])
        
        # 5. 极端度检测
        extremeness = self.detect_extremeness(qichang)
        
        # 6. 太极衰减
        decay_weight = self.taiji_decay(extremeness)
        for trigram in self.trigrams:
            qichang[trigram] *= decay_weight
        
        # 7. 温度计计算（基于trend_rotation，从v93移植）
        k = self.calc_temperature(trend_rotation)
        
        # 8. 水蒸气凝结（softmax + 活跃卦位）
        winner, distribution, active_trigrams = self.condensation(qichang, k)
        
        return {
            'winner': winner,
            'distribution': distribution,
            'active_trigrams': active_trigrams,
            'features': features,
            'final_qichang': qichang,
            'extremeness': extremeness,
            'trend_rotation': trend_rotation,
            'temperature': k
        }
