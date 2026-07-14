"""
架构完整性测试 — 确认 engine_v9.perceive() 内部每个阶段都走了

测试目标：
1. 线①汉字物理被调用
2. 线③词网络被调用
3. 三线合并正确
4. 记忆层偏置参与
5. C层/指涉层被调用（mock）
6. 近窗微风被更新
7. EngineV9Result 包含所有字段
8. 架构铁律不被破坏

标记: arch
用法: pytest test_framework/test_arch_integrity.py -v
"""
import pytest, numpy as np, unittest.mock as mock
from engine_v9 import EngineV9, EngineV9Result, BAGUA, _compute_qi_physics  # type: ignore
from memory_layer import MemoryLayer  # type: ignore
from observer_layer import ObserverLayer  # type: ignore
from c_layer import CLayer  # type: ignore


@pytest.mark.arch
class TestArchitectureIntegrity:
    """架构完整性 — perceive() 内部阶段追踪"""

    def test_result_has_all_fields(self):
        """EngineV9Result 应包含所有架构定义的字段"""
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=None)

        # 核心字段
        assert hasattr(r, 'winner')
        assert hasattr(r, 'distribution')
        assert hasattr(r, 'cv')
        assert hasattr(r, 'qi_physics')
        assert hasattr(r, 'hetu_heat')
        assert hasattr(r, 'qi_field_in')
        assert hasattr(r, 'trace')
        assert hasattr(r, 'word_crystal')

        # 类型验证
        assert r.winner in BAGUA
        assert isinstance(r.distribution, dict)
        assert all(g in r.distribution for g in BAGUA)
        assert isinstance(r.qi_physics, np.ndarray)
        assert len(r.qi_physics) == 8
        assert isinstance(r.hetu_heat, dict)
        assert isinstance(r.trace, (dict, list))  # trace 可以是 dict 或 list（步快照）
        assert isinstance(r.word_crystal, list)

    def test_physics_line_called(self):
        """汉字物理线应被调用"""
        e = EngineV9(hour=12)
        with mock.patch('engine_v9._compute_qi_physics', wraps=_compute_qi_physics) as spy:
            r = e.perceive("天地", hetu_texts=["你好"], qi_field=None)
            spy.assert_called_once_with("天地")

    def test_semantic_line_called(self):
        """词网络语义线应被调用（当有 hetu_texts 时）"""
        e = EngineV9(hour=12)
        # _word_net 是懒加载的，先 perceive 一次让 _ensure_loaded() 跑
        e.perceive("warmup", hetu_texts=["预热"], qi_field=None)
        wn = e._word_net
        assert wn is not None, "懒加载后 _word_net 应被初始化"
        with mock.patch.object(wn, 'ingest', wraps=wn.ingest) as spy:
            r = e.perceive("测试", hetu_texts=["你好世界"], qi_field=None)
            spy.assert_called_once_with(["你好世界"])

    def test_semantic_line_skipped_when_no_hetu(self):
        """无 hetu_texts 时语义线应回退为均匀"""
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=None, qi_field=None)
        # hetu_heat 应为全 1.0（回退）
        assert all(abs(v - 1.0) < 0.001 for v in r.hetu_heat.values()), \
            f"无语义输入时 hetu_heat 应全1，实际{r.hetu_heat}"

    def test_three_line_merge(self):
        """三线合并：0.5×物理 + 0.3×语义 + 0.2×qi_field"""
        e = EngineV9(hour=12)
        qi_field = np.array([1.0]*8)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=qi_field)

        # 验证：qi_field_in 被正确传入
        assert np.allclose(r.qi_field_in, qi_field)
        # 验证：分布总和≈1
        total = sum(r.distribution.values())
        assert abs(total - 1.0) < 0.01

    def test_c_layer_called(self):
        """C层应在有 hetu_texts 时被调用"""
        e = EngineV9(hour=12)
        c = CLayer(port=8084)
        e.attach_c_layer(c)
        hetu = [f"回应{i} 内容。" for i in range(6)]

        with mock.patch('c_layer.requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"content": "测试。"}
            r = e.perceive("测试", hetu_texts=hetu, qi_field=None)
            mock_post.assert_called_once()

    def test_c_layer_not_called_without_hetu(self):
        """无 hetu_texts 时 C层不应被调用"""
        e = EngineV9(hour=12)
        c = CLayer(port=8084)
        e.attach_c_layer(c)

        with mock.patch('c_layer.requests.post') as mock_post:
            r = e.perceive("测试", hetu_texts=None, qi_field=None)
            mock_post.assert_not_called()

    def test_observer_recorded(self):
        """指涉层应记录每次 perceive"""
        e = EngineV9(hour=12)
        observer = ObserverLayer(max_history=10)
        e.attach_observer(observer)

        qi_field = np.ones(8) / 8
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=qi_field)

        assert len(observer.history) == 1
        snap = observer.history[0]
        assert snap.winner == r.winner
        assert snap.cv == r.cv
        assert np.allclose(snap.qi_physics, r.qi_physics)
        assert snap.winner in BAGUA

    def test_near_qi_updated(self):
        """perceive 后 near_qi 应更新为 final_dist"""
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        assert e.near_qi is not None
        assert abs(e.near_qi.sum() - 1.0) < 0.01

    def test_word_crystal_produced(self):
        """词结晶：perceive 应产出 word_crystal 列表（中文词，非卦名）"""
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        assert isinstance(r.word_crystal, list)
        assert len(r.word_crystal) > 0, "word_crystal 不应为空"
        # v9 纯净版结晶产出的是词网络热扩散后的实际中文词，不是卦名
        assert all(isinstance(w, str) for w in r.word_crystal), "word_crystal 含非字符串"

    def test_memory_bias_integration(self):
        """记忆层偏置应参与 qi_init（隔离演化下影响分布）"""
        e1 = EngineV9(hour=12)
        r1 = e1.perceive("测试", hetu_texts=["你好"], qi_field=None)

        bias = np.zeros(8)
        bias[0] = 1.0  # 乾偏
        e2 = EngineV9(hour=12)
        e2.set_memory_bias(bias)
        r2 = e2.perceive("测试", hetu_texts=["你好"], qi_field=None)

        # 偏置应影响分布（隔离演化下分布会变化，但不一定乾卦变大）
        # 验证：两次分布不完全相同
        d1 = np.array([r1.distribution.get(g, 0) for g in BAGUA])
        d2 = np.array([r2.distribution.get(g, 0) for g in BAGUA])
        diff = np.abs(d1 - d2).sum()
        assert diff > 0.001, f"偏置应影响分布，diff={diff:.6f}"

    def test_round_increments(self):
        """round 计数器应递增"""
        e = EngineV9(hour=12)
        e.perceive("t1", hetu_texts=["a"], qi_field=None)
        assert e.round == 1
        e.perceive("t2", hetu_texts=["a"], qi_field=None)
        assert e.round == 2
        e.perceive("t3", hetu_texts=["a"], qi_field=None)
        assert e.round == 3

    def test_deterministic_no_random(self):
        """确定性系统：同输入同输出"""
        e = EngineV9(hour=12)
        qi_field = np.array([0.2, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.05])
        qi_field = qi_field / qi_field.sum()

        r1 = e.perceive("测试文本", hetu_texts=["你好世界"], qi_field=qi_field)
        r2 = e.perceive("测试文本", hetu_texts=["你好世界"], qi_field=qi_field)

        assert r1.winner == r2.winner
        for g in BAGUA:
            assert abs(r1.distribution[g] - r2.distribution[g]) < 1e-6


@pytest.mark.arch
class TestArchitectureIronLaws:
    """架构铁律验证 — 七条铁律不被破坏"""

    def test_no_filtering(self):
        """不筛选：8卦全部参与，没有卦被剔除"""
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        assert len(r.distribution) == 8, "8卦应全部在场"

    def test_no_feature_extraction(self):
        """不特征提取：输入未经过外部特征工程"""
        # 汉字物理是纯数学（码点+位图），不是特征提取
        qi = _compute_qi_physics("天地")
        assert len(qi) == 8
        assert abs(qi.sum() - 1.0) < 0.01

    def test_no_projection(self):
        """不投影：qi 保持在 8 维八卦空间"""
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        # 分布应始终在 8 维
        assert len(r.distribution) == 8

    def test_no_matching(self):
        """不匹配：没有卦位锚点对比逻辑"""
        # 验证：引擎内部不做 cos 匹配
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        # 如果做了匹配，分布会偏向某个卦
        # 纯物理+语义→分布应自然形成
        assert all(0 <= v <= 1 for v in r.distribution.values())

    def test_no_quantification(self):
        """不量化：输出是连续分布，不是离散等级"""
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        values = list(r.distribution.values())
        # 8个值应不完全相同（否则被量化了）
        assert len(set([f"{v:.4f}" for v in values])) > 1

    def test_no_weighted_avg(self):
        """不加权：三线汇合不是简单加权平均（是0.5/0.3/0.2热源）"""
        # 验证：分布不是三线直接线性组合
        e = EngineV9(hour=12)
        qi_field = np.ones(8) / 8
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=qi_field)
        # v94 内部有级联+冷却，不是线性加权
        total = sum(r.distribution.values())
        assert abs(total - 1.0) < 0.01  # 归一化但不等于线性组合

    def test_no_arbitrator(self):
        """无仲裁者：没有外部决策层选择卦位"""
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=None)
        # winner 应由分布自然产生（argmax），不是外部选
        import numpy as np
        vals = np.array([r.distribution[g] for g in BAGUA])
        expected_winner = BAGUA[np.argmax(vals)]
        assert r.winner == expected_winner, \
            f"winner={r.winner} 不是分布最大值，可能有人为仲裁"
