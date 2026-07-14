"""
MemoryLayer — 上层窗口的记忆模块

三个存储层次：
  1. 双轮窗口 — 安全网，只做差分参考
  2. N轮滑窗 — 策略+语言用
  3. 轨迹统计 — 物理过程记忆

记忆不进引擎，只影响策略和语言。
"""

from collections import deque
import numpy as np
from typing import Dict, List, Optional, Tuple

BAGUA = ['乾', '兑', '离', '震', '坤', '艮', '坎', '巽']


class MemoryLayer:
    def __init__(self, window_size: int = 8):
        self.window_size = window_size

        # ═══ 双轮窗口 ═══
        self.prev_text: Optional[str] = None
        self.prev_qi: Optional[np.ndarray] = None
        self.prev_winner: Optional[str] = None
        self.prev_deviation: Optional[float] = None
        self.prev_strategy: Optional[str] = None

        # ═══ N轮滑窗 ═══
        self.texts: deque = deque(maxlen=window_size)
        self.qi_states: deque = deque(maxlen=window_size)
        self.winners: deque = deque(maxlen=window_size)
        self.deviations: deque = deque(maxlen=window_size)
        self.strategies: deque = deque(maxlen=window_size)

        # ═══ 轨迹统计 ═══
        self.gua_dwell: Dict[str, int] = {g: 0 for g in BAGUA}
        self._gua_streak: Dict[str, int] = {}
        self._strategy_streak: Dict[str, int] = {}
        self._last_gua: Optional[str] = None
        self._last_strategy: Optional[str] = None
        self._cooling: float = 0.0

        # 全程轨迹
        self.deviation_sequence: List[float] = []
        self.gua_sequence: List[str] = []

    # ── 更新 ──
    def update(self, text: str, qi_state: np.ndarray,
               winner: str, deviation: float,
               strategy: str,
               extremeness: float = 0.0) -> dict:
        """每轮调用一次，返回上层需要的完整上下文"""

        # 双轮差分
        delta_qi = None
        delta_deviation = None
        if self.prev_qi is not None:
            delta_qi = qi_state - self.prev_qi
            delta_deviation = deviation - (self.prev_deviation or 0)

        # ── 更新双轮 ──
        self.prev_text = text
        self.prev_qi = qi_state.copy()
        self.prev_winner = winner
        self.prev_deviation = deviation
        self.prev_strategy = strategy

        # ── 更新N轮滑窗 ──
        self.texts.append(text)
        self.qi_states.append(qi_state.copy())
        self.winners.append(winner)
        self.deviations.append(deviation)
        self.strategies.append(strategy)

        # ── 更新轨迹统计 ──
        self.gua_dwell[winner] += 1
        self.gua_sequence.append(winner)
        self.deviation_sequence.append(deviation)

        # ── 卦位冷却（同一卦连任检测）──
        if winner == self._last_gua:
            self._gua_streak[winner] = self._gua_streak.get(winner, 0) + 1
        else:
            self._gua_streak[winner] = 1
            self._cooling *= 0.5  # 换卦衰减

        gua_s = self._gua_streak.get(winner, 0)
        if gua_s <= 3:
            self._cooling = max(self._cooling * 0.5, 0.0)
        else:
            self._cooling = min(1.0, (gua_s - 3) / 7.0)
        self._last_gua = winner

        # ── 策略疲劳（同一策略连用检测）──
        if strategy == self._last_strategy:
            self._strategy_streak[strategy] = self._strategy_streak.get(strategy, 0) + 1
        else:
            self._strategy_streak[strategy] = 1
        self._last_strategy = strategy

        # ── 话题回环检测 ──
        topic_loop = self._detect_loop()

        # ── 偏角趋势 ──
        deviation_trend = self._deviation_trend()

        return {
            # 双轮差分
            'prev': {
                'text': self.prev_text,
                'winner': self.prev_winner,
                'deviation': self.prev_deviation,
                'strategy': self.prev_strategy,
            },
            'delta_qi': delta_qi,
            'delta_deviation': delta_deviation,

            # 滑窗
            'winners': list(self.winners),
            'strategies': list(self.strategies),
            'gua_sequence': self.gua_sequence[-self.window_size:],

            # 策略信号
            'strategy_fatigue': self._strategy_streak.get(strategy, 1) >= 4,
            'strategy_streak': self._strategy_streak.get(strategy, 1),
            'gua_cooling': self._cooling,
            'gua_streak': gua_s,
            'cooling_high': self._cooling > 0.5,
            'cooling_max': self._cooling > 0.8,

            # 话题信号
            'topic_loop': topic_loop,
            'dominant_recent_gua': self._dominant_recent(),
            'deviation_trend': deviation_trend,
            'deviation_expanding': deviation_trend > 0.02,
            'deviation_narrowing': deviation_trend < -0.02,

            # 长期统计
            'gua_dwell': dict(self.gua_dwell),
            'total_rounds': len(self.gua_sequence),
        }

    # ── 内部诊断 ──
    def _detect_loop(self) -> bool:
        """话题回环：最近4轮卦序列出现重复模式"""
        if len(self.gua_sequence) < 4:
            return False
        recent = self.gua_sequence[-4:]
        earlier = self.gua_sequence[:-2]
        # 最近4轮是否在历史中出现过
        for i in range(len(earlier) - 3):
            if earlier[i:i + 4] == recent:
                return True
        return False

    def _deviation_trend(self) -> float:
        """偏角趋势：最近4轮偏角的线性斜率"""
        if len(self.deviation_sequence) < 4:
            return 0.0
        recent = self.deviation_sequence[-4:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)

    def _dominant_recent(self) -> str:
        """最近N轮中最频繁的卦"""
        if not self.winners:
            return '—'
        recent = list(self.winners)[-min(4, len(self.winners)):]
        counts = {g: recent.count(g) for g in set(recent)}
        return max(counts, key=counts.get)

    def reset(self):
        """重置记忆"""
        self.prev_text = None
        self.prev_qi = None
        self.prev_winner = None
        self.prev_deviation = None
        self.prev_strategy = None
        self.texts.clear()
        self.qi_states.clear()
        self.winners.clear()
        self.deviations.clear()
        self.strategies.clear()
        self.gua_dwell = {g: 0 for g in BAGUA}
        self._gua_streak.clear()
        self._strategy_streak.clear()
        self._last_gua = None
        self._last_strategy = None
        self._cooling = 0.0
        self.deviation_sequence.clear()
        self.gua_sequence.clear()
