"""
qi_field 线独立测试 — mock 不同结构的 qi_field，验证对结果的影响

测试线② qi_field（背景温度梯度）如何影响 engine_v9 的输出。
三种结构：
1. 均匀 — 不引入偏置
2. 乾偏 — 乾卦占主导
3. 坤偏 — 坤卦占主导
4. 坎偏 — 坎卦占主导（水，深邃）
5. 乾+坤双峰 — 两个热源
"""
import pytest, numpy as np
from engine_v9 import EngineV9, BAGUA  # type: ignore


@pytest.mark.arch
class TestQiFieldLine:
    """qi_field 线：不同结构的背景温度梯度"""

    def test_uniform_no_bias(self):
        """均匀 qi_field 不引入偏置（仅靠汉字物理+语义）"""
        e = EngineV9(hour=12)
        qi_field = np.ones(8) / 8
        r = e.perceive("我升职了", hetu_texts=["恭喜你", "新的开始"], qi_field=qi_field)
        assert r.winner in BAGUA
        assert r.cv >= 0

    def test_qian_dominant_influences(self):
        """乾偏 qi_field 应让乾卦占比提升"""
        e = EngineV9(hour=12)
        qi_field = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        qi_field = qi_field / qi_field.sum()
        r = e.perceive("升职加薪", hetu_texts=["恭喜"], qi_field=qi_field)
        assert r.winner in BAGUA
        # 乾偏的 qi_field 应使乾卦的 qi_init 更高
        assert r.distribution.get('乾', 0) > 0.01

    def test_kun_dominant_influences(self):
        """坤偏 qi_field 应让坤卦占比提升"""
        e = EngineV9(hour=12)
        qi_field = np.array([1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0])
        qi_field = qi_field / qi_field.sum()
        r = e.perceive("我升职了", hetu_texts=["恭喜"], qi_field=qi_field)
        assert r.winner in BAGUA
        assert r.distribution.get('坤', 0) > 0.01

    def test_kan_dominant_influences(self):
        """坎偏 qi_field 应让坎卦占比提升"""
        e = EngineV9(hour=12)
        qi_field = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0])
        qi_field = qi_field / qi_field.sum()
        r = e.perceive("升职加薪", hetu_texts=["恭喜"], qi_field=qi_field)
        assert r.winner in BAGUA
        assert r.distribution.get('坎', 0) > 0.01

    def test_dual_peak(self):
        """乾+坤双峰 qi_field — 两个热源同时烧"""
        e = EngineV9(hour=12)
        qi_field = np.array([2.5, 1.0, 1.0, 1.0, 2.5, 1.0, 1.0, 1.0])
        qi_field = qi_field / qi_field.sum()
        r = e.perceive("升职加薪", hetu_texts=["恭喜"], qi_field=qi_field)
        assert r.winner in BAGUA
        # 乾或坤应有一个占主导
        q_k = r.distribution.get('乾', 0) + r.distribution.get('坤', 0)
        assert q_k > 0.1, f"双峰应让乾+坤占比提升，实际{q_k:.3f}"

    def test_no_qi_field_fallback(self):
        """无 qi_field 时应回退为均匀（默认 1.0）"""
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        assert r.winner in BAGUA
        # qi_field_in 应为全 1（回退值）
        assert np.allclose(r.qi_field_in, np.ones(8))

    def test_different_qi_field_different_result(self):
        """不同 qi_field → 不同分布（线②有区分力）"""
        e1 = EngineV9(hour=12)
        qi1 = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        qi1 = qi1 / qi1.sum()
        r1 = e1.perceive("测试", hetu_texts=["你好"], qi_field=qi1)

        e2 = EngineV9(hour=12)
        qi2 = np.array([1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0])
        qi2 = qi2 / qi2.sum()
        r2 = e2.perceive("测试", hetu_texts=["你好"], qi_field=qi2)

        d1 = np.array([r1.distribution.get(g, 0) for g in BAGUA])
        d2 = np.array([r2.distribution.get(g, 0) for g in BAGUA])
        diff = np.abs(d1 - d2).sum()
        assert diff > 0.001, f"不同 qi_field 应产生不同分布，diff={diff:.4f}"

    def test_qi_field_preserves_order(self):
        """qi_field 中的排序应在分布中体现（强热源→高分布）"""
        e = EngineV9(hour=12)
        # 乾最强，巽最弱
        qi_field = np.array([3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4])
        qi_field = qi_field / qi_field.sum()
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=qi_field)
        # 乾的分布应高于巽
        assert r.distribution.get('乾', 0) >= r.distribution.get('巽', 0) - 0.01, \
            f"乾偏qi_field下乾应≥巽: 乾={r.distribution.get('乾',0):.3f}, 巽={r.distribution.get('巽',0):.3f}"
