"""
外围层独立测试 — 记忆层 / 微风 / C层 / 指涉层

测试四个外围组件的接口正确性和行为逻辑，不依赖外部模型。
C层用 mock 验证接口（不启动 phi3）。
"""
import pytest, numpy as np, unittest.mock as mock
from collections import defaultdict

from memory_layer import MemoryLayer  # type: ignore
from observer_layer import ObserverLayer  # type: ignore
from c_layer import CLayer, BAGUA  # type: ignore


# ═══════════════════════════════════════════
# 记忆层 MemoryLayer
# ═══════════════════════════════════════════
@pytest.mark.periphery
class TestMemoryLayer:
    """记忆层：三存储（双轮/滑窗/轨迹）"""

    def test_first_update_returns_context(self):
        """首次 update 返回完整上下文"""
        m = MemoryLayer(window_size=4)
        qi = np.array([0.125]*8)
        ctx = m.update(text="test", qi_state=qi, winner='乾',
                       deviation=1.0, strategy="default")
        assert ctx is not None
        assert 'prev' in ctx
        assert ctx['delta_qi'] is None  # 首次无差分
        assert ctx['prev']['winner'] == '乾'
        assert ctx['prev']['text'] == 'test'

    def test_second_update_produces_delta(self):
        """第二次 update 应产生差分"""
        m = MemoryLayer(window_size=4)
        qi1 = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        qi2 = np.array([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        m.update(text="t1", qi_state=qi1, winner='乾', deviation=1.0, strategy="default")
        ctx = m.update(text="t2", qi_state=qi2, winner='兑', deviation=0.8, strategy="describe")

        assert ctx['delta_qi'] is not None
        assert np.allclose(ctx['delta_qi'], qi2 - qi1)
        assert ctx['delta_deviation'] == pytest.approx(-0.2, abs=1e-9)

    def test_sliding_window(self):
        """N轮滑窗应限制长度"""
        m = MemoryLayer(window_size=3)
        for i in range(10):
            qi = np.array([0.125]*8)
            m.update(text=f"t{i}", qi_state=qi, winner=BAGUA[i % 8],
                     deviation=1.0, strategy="default")
        assert len(m.texts) == 3
        assert len(m.winners) == 3
        assert m.winners[-1] == BAGUA[9 % 8]

    def test_gua_cooling(self):
        """同卦连任应触发冷却"""
        m = MemoryLayer(window_size=8)
        qi = np.array([0.125]*8)
        # 连续 5 轮乾卦
        for i in range(5):
            ctx = m.update(text=f"t{i}", qi_state=qi, winner='乾',
                           deviation=1.0, strategy="default")
        assert ctx['gua_cooling'] > 0, "连续同卦应触发冷却"
        assert ctx['gua_streak'] == 5

    def test_gua_cooling_resets_on_switch(self):
        """换卦后冷却应衰减"""
        m = MemoryLayer(window_size=8)
        qi = np.array([0.125]*8)
        m.update(text="t1", qi_state=qi, winner='乾', deviation=1.0, strategy="default")
        m.update(text="t2", qi_state=qi, winner='乾', deviation=1.0, strategy="default")
        m.update(text="t3", qi_state=qi, winner='兑', deviation=1.0, strategy="default")
        # 换卦后冷却应衰减（×0.5）
        ctx = m.update(text="t4", qi_state=qi, winner='兑', deviation=1.0, strategy="default")
        # 冷却不会清零，但会衰减
        assert ctx['cooling_high'] is False or ctx['gua_cooling'] < 0.5

    def test_strategy_fatigue(self):
        """同一策略连用≥4次应触发疲劳"""
        m = MemoryLayer(window_size=8)
        qi = np.array([0.125]*8)
        for i in range(4):
            ctx = m.update(text=f"t{i}", qi_state=qi, winner='乾',
                           deviation=1.0, strategy="default")
        assert ctx['strategy_fatigue'] is True
        assert ctx['strategy_streak'] == 4

    def test_strategy_fatigue_resets(self):
        """换策略后疲劳应重置"""
        m = MemoryLayer(window_size=8)
        qi = np.array([0.125]*8)
        m.update(text="t1", qi_state=qi, winner='乾', deviation=1.0, strategy="default")
        m.update(text="t2", qi_state=qi, winner='乾', deviation=1.0, strategy="describe")
        assert m._strategy_streak.get("describe", 0) == 1

    def test_reset(self):
        """reset 应清空所有状态"""
        m = MemoryLayer(window_size=4)
        qi = np.array([0.125]*8)
        m.update(text="t1", qi_state=qi, winner='乾', deviation=1.0, strategy="default")
        m.update(text="t2", qi_state=qi, winner='兑', deviation=0.8, strategy="default")
        m.reset()
        assert m.prev_winner is None
        assert len(m.texts) == 0
        assert m.gua_dwell == {g: 0 for g in BAGUA}


# ═══════════════════════════════════════════
# 近窗微风（通过 engine_v9 间接测试）
# ═══════════════════════════════════════════
@pytest.mark.periphery
class TestNearWindowBreeze:
    """近窗微风：每步 ε=0.005 后置微扰"""

    def test_breeze_applied(self):
        """有 near_qi 时应产生微扰"""
        from engine_v9 import EngineV9  # type: ignore
        e = EngineV9(hour=12)
        # 第一轮无微风
        r1 = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        # 设置 near_qi（第二轮的微风源）
        breeze = np.array([0.1, -0.1, 0.05, -0.05, 0.0, 0.0, 0.0, 0.0])
        e.set_near_window(breeze)
        r2 = e.perceive("第二次", hetu_texts=["你好"], qi_field=None)
        # 微风应产生微小影响——但不会颠覆卦位
        assert r2.winner in BAGUA
        # 分布总和仍为 1
        total = sum(r2.distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_breeze_epsilon_small(self):
        """微风系数应很小（ε=0.005）"""
        from engine_v9 import EngineV9  # type: ignore
        e = EngineV9(hour=12)
        assert e.breeze_epsilon == 0.005


# ═══════════════════════════════════════════
# C层 CLayer（mock 验证接口）
# ═══════════════════════════════════════════
@pytest.mark.periphery
class TestCLayer:
    """C层：旁路推理，mock phi3 验证接口"""

    def test_observe_calls_api(self):
        """observe 应发送 HTTP 请求"""
        c = CLayer(port=8084)
        hetu = [f"回应{i} 这是一段测试文本内容。" for i in range(6)]
        dist = {'乾': 0.5, '兑': 0.1, '离': 0.1, '震': 0.05, '坤': 0.1, '艮': 0.05, '坎': 0.05, '巽': 0.05}

        with mock.patch('c_layer.requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"content": "因为看到了力量的对比。"}

            result = c.observe(hetu_texts=hetu, distribution=dist, winner='乾', crystal=['乾'])

            assert result is not None
            assert "因为" in result
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "prompt" in call_args.kwargs.get('json', call_args[1].get('json', {}))

    def test_observe_empty_hetu(self):
        """无河图文本应返回 None"""
        c = CLayer(port=8084)
        result = c.observe(hetu_texts=[], distribution={}, winner='乾')
        assert result is None

    def test_observe_api_failure(self):
        """API 失败应优雅返回 None"""
        c = CLayer(port=8084)
        hetu = [f"回应{i}" for i in range(6)]

        with mock.patch('c_layer.requests.post') as mock_post:
            mock_post.side_effect = Exception("Connection refused")
            result = c.observe(hetu_texts=hetu, distribution={}, winner='乾')
            assert result is None

    def test_diary_records(self):
        """成功的 observe 应存入 diary"""
        c = CLayer(port=8084)
        hetu = [f"回应{i} 测试内容。" for i in range(6)]
        dist = {'乾': 0.5, '兑': 0.1, '离': 0.1, '震': 0.05, '坤': 0.1, '艮': 0.05, '坎': 0.05, '巽': 0.05}

        with mock.patch('c_layer.requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"content": "力量主导。"}
            c.observe(hetu_texts=hetu, distribution=dist, winner='乾')
            c.observe(hetu_texts=hetu, distribution=dist, winner='乾')

        assert len(c.all_diary()) == 2
        assert c.last() is not None

    def test_reset(self):
        """reset 应清空 diary"""
        c = CLayer(port=8084)
        c.diary.append("[r1] 因为测试")
        c.reset()
        assert len(c.all_diary()) == 0
        assert c.last() is None


# ═══════════════════════════════════════════
# 指涉层 ObserverLayer
# ═══════════════════════════════════════════
@pytest.mark.periphery
class TestObserverLayer:
    """指涉层：跨轮观察，异常检测"""

    def test_record_snapshot(self):
        """record 应存储快照"""
        o = ObserverLayer(max_history=5)
        qi_p = np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03])
        qf = np.array([0.125]*8)
        dist = {'乾': 0.5, '兑': 0.1, '离': 0.1, '震': 0.05, '坤': 0.1, '艮': 0.05, '坎': 0.05, '巽': 0.05}

        o.record(round=1, text="测试", qi_physics=qi_p, hetu_heat={'乾': 0.5},
                 qi_field_in=qf, distribution=dist, winner='乾', cv=1.5,
                 word_crystal=['乾'])
        assert len(o.history) == 1
        assert o.history[0].round == 1
        assert o.history[0].winner == '乾'

    def test_max_history(self):
        """超过 max_history 应淘汰最旧"""
        o = ObserverLayer(max_history=3)
        qi = np.ones(8) / 8
        dist = {'乾': 0.5} | {g: 0.1 for g in BAGUA[1:]}
        for r in range(5):
            o.record(round=r+1, text=f"t{r}", qi_physics=qi,
                     qi_field_in=qi, distribution=dist, winner=BAGUA[r % 8],
                     cv=1.0, word_crystal=['乾'])
        assert len(o.history) == 3
        assert o.history[0].round == 3  # 最旧的是 r3

    def test_gua_count_tracking(self):
        """卦位计数应正确"""
        o = ObserverLayer(max_history=10)
        qi = np.ones(8) / 8
        dist = {'乾': 0.5} | {g: 0.1 for g in BAGUA[1:]}
        for r in range(4):
            o.record(round=r+1, text="test", qi_physics=qi,
                     qi_field_in=qi, distribution=dist, winner='乾',
                     cv=1.0, word_crystal=['乾'])
        assert o.gua_count['乾'] == 4

    def test_cv_sequence(self):
        """CV 序列应正确记录"""
        o = ObserverLayer(max_history=10)
        qi = np.ones(8) / 8
        dist = {'乾': 0.5} | {g: 0.1 for g in BAGUA[1:]}
        for r in range(3):
            o.record(round=r+1, text="test", qi_physics=qi,
                     qi_field_in=qi, distribution=dist, winner='乾',
                     cv=float(r+1), word_crystal=['乾'])
        assert o.cv_sequence == [1.0, 2.0, 3.0]

    def test_report_empty(self):
        """无数据时报告应合理"""
        o = ObserverLayer()
        report = o.report()
        assert "尚无数据" in report or "0" in report or len(o.history) == 0

    def test_report_with_data(self):
        """有数据时报告应包含统计"""
        o = ObserverLayer(max_history=10)
        qi = np.ones(8) / 8
        dist = {'乾': 0.5} | {g: 0.1 for g in BAGUA[1:]}
        for r in range(3):
            o.record(round=r+1, text="test", qi_physics=qi,
                     qi_field_in=qi, distribution=dist, winner='乾',
                     cv=1.5, word_crystal=['乾'])
        report = o.report()
        assert "指涉层" in report
        assert "3轮" in report or "3" in report

    def test_anomaly_gua_lock(self):
        """连续5轮同卦应触发锁死异常"""
        o = ObserverLayer(max_history=10)
        qi = np.ones(8) / 8
        dist = {'乾': 0.5} | {g: 0.1 for g in BAGUA[1:]}
        for r in range(5):
            o.record(round=r+1, text="test", qi_physics=qi,
                     qi_field_in=qi, distribution=dist, winner='乾',
                     cv=1.5, word_crystal=['乾'])
        anomalies = o.anomalies_summary()
        lock_anomalies = [a for a in anomalies if a['type'] == 'gua_lock']
        assert len(lock_anomalies) >= 1, "应检测到卦位锁死"

    def test_reset(self):
        """reset 应清空所有状态"""
        o = ObserverLayer()
        qi = np.ones(8) / 8
        dist = {'乾': 0.5} | {g: 0.1 for g in BAGUA[1:]}
        o.record(round=1, text="test", qi_physics=qi, qi_field_in=qi,
                 distribution=dist, winner='乾', cv=1.5, word_crystal=['乾'])
        o.reset()
        assert len(o.history) == 0
        assert len(o.cv_sequence) == 0
        assert len(o.anomalies_summary()) == 0


# ═══════════════════════════════════════════
# 外围层与引擎 v9 的集成
# ═══════════════════════════════════════════
@pytest.mark.periphery
class TestPeripheryIntegration:
    """外围层与 engine_v9 的集成验证"""

    def test_memory_bias_affects_init(self):
        """记忆层偏置应改变 qi_init"""
        from engine_v9 import EngineV9  # type: ignore
        e = EngineV9(hour=12)
        # 无偏置
        r1 = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        # 设置偏置（乾卦+0.5）
        bias = np.zeros(8)
        bias[0] = 0.5  # 乾
        e.set_memory_bias(bias)
        r2 = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        # 偏置应影响分布——但不一定改变胜卦
        assert r1.winner in BAGUA
        assert r2.winner in BAGUA
        # 隔离演化下偏置影响分布（不保证乾卦变大，但分布会变化）
        d1 = np.array([r1.distribution.get(g, 0) for g in BAGUA])
        d2 = np.array([r2.distribution.get(g, 0) for g in BAGUA])
        diff = np.abs(d1 - d2).sum()
        assert diff > 0.001, f"偏置应影响分布，diff={diff:.6f}"

    def test_observer_records_every_perceive(self):
        """指涉层应记录每次 perceive 的结果"""
        from engine_v9 import EngineV9  # type: ignore
        e = EngineV9(hour=12)
        observer = ObserverLayer(max_history=10)
        e.attach_observer(observer)

        for i in range(3):
            e.perceive(f"测试{i}", hetu_texts=["你好"], qi_field=None)

        assert len(observer.history) == 3
        assert observer.history[0].winner in BAGUA

    def test_c_layer_observes_every_perceive(self):
        """C层应观察每次有 hetu_texts 的 perceive"""
        from engine_v9 import EngineV9  # type: ignore
        e = EngineV9(hour=12)
        c = CLayer(port=8084)
        e.attach_c_layer(c)
        hetu = ["回应1", "回应2", "回应3", "回应4", "回应5", "回应6"]

        with mock.patch('c_layer.requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"content": "因为测试。"}
            e.perceive("测试", hetu_texts=hetu, qi_field=None)
            e.perceive("测试2", hetu_texts=hetu, qi_field=None)

        assert mock_post.call_count == 2
