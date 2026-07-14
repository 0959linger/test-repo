"""
╔══════════════════════════════════════════════════════════════════╗
║  observer_layer — 指涉层                                        ║
║                                                                  ║
║  不改架构，只并排观察各层数据。                                  ║
║  观察经验存入 observations.md，但不进入反馈环。                  ║
║  眼睛看了历史后告诉玲"这个结果在历史中意味着什么"，              ║
║  但不干预架构。                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import deque

BAGUA = ['乾', '兑', '离', '震', '坤', '艮', '坎', '巽']


@dataclass
class LayerSnapshot:
    """一轮的完整快照"""
    round: int
    text: str
    qi_physics: np.ndarray          # 汉字物理
    hetu_heat: Dict[str, float]     # 河图语义热（放大后）
    qi_field_in: np.ndarray          # 传入qi_field（蒸汽上游）
    distribution: Dict[str, float]   # v94最终分布
    winner: str
    cv: float
    word_crystal: List[str]          # 结晶词
    dashboard_gua: List[str] = field(default_factory=list)  # 仪表盘卦标签


class ObserverLayer:
    """指涉层：跨轮观察，输出给玲看。不进反馈环。"""
    
    def __init__(self, max_history: int = 32):
        self.max_history = max_history
        self.history: List[LayerSnapshot] = []
        
        # ── 统计跟踪 ──
        self.gua_count: Dict[str, int] = {g: 0 for g in BAGUA}
        self.cv_sequence: List[float] = []
        self.gua_sequence: List[str] = []
        self.expected_turnover = 0.1  # 预期换卦率
        
        # ── 模式发现 ──
        self._anomalies: List[dict] = []
    
    def record(self, round: int, text: str = "",
               qi_physics: np.ndarray = None,
               hetu_heat: Dict[str, float] = None,
               qi_field_in: np.ndarray = None,
               distribution: Dict[str, float] = None,
               winner: str = "",
               cv: float = 0.0,
               word_crystal: List[str] = None,
               dashboard_gua: List[str] = None):
        """记录一轮快照"""
        snap = LayerSnapshot(
            round=round,
            text=text,
            qi_physics=qi_physics if qi_physics is not None else np.ones(8) / 8,
            hetu_heat=hetu_heat or {g: 1.0 for g in BAGUA},
            qi_field_in=qi_field_in if qi_field_in is not None else np.ones(8) / 8,
            distribution=distribution or {g: 1.0 / 8 for g in BAGUA},
            winner=winner,
            cv=cv,
            word_crystal=word_crystal or [],
            dashboard_gua=dashboard_gua or [],
        )
        self.history.append(snap)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # 更新统计
        self.gua_count[winner] = self.gua_count.get(winner, 0) + 1
        self.cv_sequence.append(cv)
        self.gua_sequence.append(winner)
        
        # 异常检测
        self._detect_anomaly(snap)
    
    def _detect_anomaly(self, snap: LayerSnapshot):
        """检测本轮异常"""
        if len(self.cv_sequence) < 3:
            return
        
        recent_cv = self.cv_sequence[-3:]
        avg_cv = np.mean(recent_cv)
        
        # CV突然崩溃（热寂前兆）
        if avg_cv < 0.3 and len(self.cv_sequence) >= 5:
            earlier = self.cv_sequence[-6:-3]
            if np.mean(earlier) > 1.0:
                self._anomalies.append({
                    "round": snap.round,
                    "type": "cv_collapse",
                    "desc": f"CV从{np.mean(earlier):.1f}骤降至{avg_cv:.2f}，场在塌缩"
                })
        
        # 卦位锁死
        if len(self.gua_sequence) >= 5:
            recent_gua = self.gua_sequence[-5:]
            if len(set(recent_gua)) == 1:
                self._anomalies.append({
                    "round": snap.round,
                    "type": "gua_lock",
                    "desc": f"连续5轮{recent_gua[0]}卦锁死"
                })
        
        # 龙战（高度共识+低CV）
        if len(self.cv_sequence) >= 2:
            cv_recent = self.cv_sequence[-2:]
            if all(c < 2.5 for c in cv_recent) and any(c > 2.0 for c in cv_recent):
                gua_recent = self.gua_sequence[-2:]
                if len(set(gua_recent)) > 1 and len(self.gua_sequence) >= 4:
                    diversity = len(set(self.gua_sequence[-4:]))
                    if diversity >= 3:
                        self._anomalies.append({
                            "round": snap.round,
                            "type": "longzhan",
                            "desc": f"低CV({cv_recent[-1]:.1f})+高卦位多样性→疑似龙战"
                        })
    
    def report(self) -> str:
        """给玲看的观察报告"""
        if not self.history:
            return "（指涉层：尚无数据）"
        
        lines = []
        lines.append(f"── 指涉层观察 ({len(self.history)}轮) ──")
        
        # 卦位分布
        sorted_gua = sorted(self.gua_count.items(), key=lambda x: -x[1])
        top_gua = ' '.join(f"{g}×{c}" for g, c in sorted_gua if c > 0)[:40]
        lines.append(f"卦位: {top_gua}")
        
        # CV趋势
        if len(self.cv_sequence) >= 2:
            cv_avg = np.mean(self.cv_sequence)
            cv_trend = "↑" if self.cv_sequence[-1] > self.cv_sequence[-2] else "↓"
            lines.append(f"CV: {cv_avg:.2f} avg, 最近{cv_trend}")
        
        # 最新3轮的卦位转移
        if len(self.gua_sequence) >= 3:
            recent3 = self.gua_sequence[-3:]
            lines.append(f"最近3卦: {' → '.join(recent3)}")
        
        # 异常
        if self._anomalies:
            recent_anom = self._anomalies[-3:]
            lines.append(f"⚠ 异常: {len(self._anomalies)}次")
            for a in recent_anom:
                lines.append(f"  r{a['round']} [{a['type']}] {a['desc']}")
        
        return '\n'.join(lines)
    
    def anomalies_summary(self) -> List[dict]:
        """返回所有异常记录"""
        return self._anomalies.copy()
    
    def reset(self):
        self.history.clear()
        self.gua_count = {g: 0 for g in BAGUA}
        self.cv_sequence.clear()
        self.gua_sequence.clear()
        self._anomalies.clear()
