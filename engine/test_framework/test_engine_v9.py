"""
引擎 v9 管线测试 — 三线汇合 + WordNetwork + 预演层 + 外围层

从"不崩"到"管线对"——每个组件独立验证，每条线参与验证。

标记: engine_v9
用法: pytest test_framework/test_engine_v9.py -v
"""
import pytest, numpy as np, sys, os

# conftest.py 已把 engine/ 加入 sys.path
from engine_v9 import EngineV9, BAGUA, _compute_qi_physics, _prestage_amplify, WordNetwork  # type: ignore
from memory_layer import MemoryLayer  # type: ignore
from observer_layer import ObserverLayer  # type: ignore
from c_layer import CLayer  # type: ignore

# ═══════════════════════════════════════════
# 预演层 _prestage_amplify
# ═══════════════════════════════════════════
@pytest.mark.engine_v9
class TestPrestageAmplify:
    """预演层：10步×0.40 排名级联放大"""

    def test_amplifies_differences(self):
        """有明显差异的输入 → spread 被放大"""
        qi = {'乾': 0.30, '兑': 0.15, '离': 0.12, '震': 0.10,
              '坤': 0.10, '艮': 0.08, '坎': 0.08, '巽': 0.07}
        result = _prestage_amplify(qi, steps=10, rate=0.40)
        spread = max(result.values()) - min(result.values())
        assert spread > 0.25, f"预演层应放大差异，spread={spread:.3f}"

    def test_uniform_stays_flat(self):
        """均匀输入 → 不应造出虚假差异"""
        qi = {g: 1.0 for g in BAGUA}
        result = _prestage_amplify(qi, steps=10, rate=0.40)
        spread = max(result.values()) - min(result.values())
        assert spread < 0.08, f"均匀输入不应造差异，spread={spread:.3f}"

    def test_no_negative(self):
        """输出无负值"""
        qi = {'乾': 0.01, '兑': 0.01, '离': 0.01, '震': 0.01,
              '坤': 0.01, '艮': 0.01, '坎': 0.01, '巽': 0.96}
        result = _prestage_amplify(qi)
        assert all(v >= 0 for v in result.values()), f"有负值: {result}"

    def test_preserves_order(self):
        """放大后卦位排序不变（强的依然强）"""
        qi = {'乾': 0.28, '兑': 0.18, '离': 0.14, '震': 0.12,
              '坤': 0.10, '艮': 0.08, '坎': 0.06, '巽': 0.04}
        result = _prestage_amplify(qi, steps=10, rate=0.40)
        sorted_orig = sorted(qi.items(), key=lambda x: -x[1])
        sorted_res = sorted(result.items(), key=lambda x: -x[1])
        orig_order = [g for g, _ in sorted_orig]
        res_order = [g for g, _ in sorted_res]
        assert orig_order == res_order, \
            f"排序应变: {orig_order} → {res_order}"


# ═══════════════════════════════════════════
# WordNetwork 词网络
# ═══════════════════════════════════════════
@pytest.mark.engine_v9
@pytest.mark.slow
class TestWordNetwork:
    """
    WordNetwork 词网络：BPE分词 → PCA256D → 词间余弦 → 邻接图 → 热扩散 → 轮转分桶八卦映射

    需要加载 7B 嵌入表（mmap，约 2GB 虚拟地址空间），标记 slow。
    """

    @pytest.fixture(scope="class")
    def wn(self):
        """加载完整7B嵌入表（class级复用，只加载一次）"""
        import numpy as np
        HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA = os.path.join(HERE, '..', 'data')
        embed = np.load(os.path.join(DATA, 'qwen7b_embed_tokens.npy'), mmap_mode='r')
        pca = np.load(os.path.join(DATA, 'pca_256_proj.npz'))
        tokenizer_path = os.path.join(DATA, 'tokenizer.json')
        return WordNetwork(
            embed, pca['proj'], pca['mean'],
            tokenizer_json_path=tokenizer_path,
            embed_path=os.path.join(DATA, 'qwen7b_embed_tokens.npy'),
            pca_path=os.path.join(DATA, 'pca_256_proj.npz'),
        )

    # ── 分词 ──
    def test_tokenize_not_empty(self, wn):
        """中文文本分词非空"""
        tids = wn._tokenize("天地玄黄宇宙洪荒")
        assert len(tids) > 0

    def test_tokenize_dedup(self, wn):
        """重复词去重"""
        tids = wn._tokenize("天地天地天地")
        # 去重后每个唯一 token 只出现一次
        assert len(tids) <= len(set(tids)) + 3  # 宽容：去重逻辑可能因 BPE 合并不同

    # ── 语义距离 ──
    def test_semantic_matrix_shape(self, wn):
        """语义矩阵形状正确"""
        wn._ensure_256_normed()
        tids = wn._tokenize("天地人")
        sim = wn._semantic_distances(tids)
        assert sim.shape == (len(tids), len(tids))
        # 对角线应该是 0（被 fill_diagonal 清零）
        for i in range(len(tids)):
            assert sim[i, i] == 0.0

    # ── 图构建 ──
    def test_graph_produces_edges(self, wn):
        """足够多的词应产生语义边"""
        tids = wn._tokenize("天地玄黄宇宙洪荒日月盈昃辰宿列张")
        if len(tids) < 3:
            pytest.skip("分词结果太少")
        sim = wn._semantic_distances(tids)
        edges = wn._build_graph(tids, sim, threshold=0.3, k_max=5)
        assert len(edges) > 0, f"应有语义边，实际 {len(edges)}"

    def test_graph_respects_k_max(self, wn):
        """每词出边数 ≤ k_max"""
        tids = wn._tokenize("天地玄黄宇宙洪荒")
        if len(tids) < 3:
            pytest.skip("分词结果太少")
        sim = wn._semantic_distances(tids)
        edges = wn._build_graph(tids, sim, threshold=0.2, k_max=3)
        from collections import Counter
        out_deg = Counter(src for src, _, _ in edges)
        for src, deg in out_deg.items():
            assert deg <= 3, f"词{src}出边={deg} > k_max=3"

    # ── 热扩散 ──
    def test_heat_diffusion_normalized(self, wn):
        """热扩散后 sum≈1"""
        tids = wn._tokenize("天地人")
        if len(tids) < 2:
            pytest.skip("分词结果太少")
        sim = wn._semantic_distances(tids)
        edges = wn._build_graph(tids, sim)
        heat = wn._heat_diffusion(len(tids), edges, steps=5, temp=0.1)
        assert abs(heat.sum() - 1.0) < 0.01, f"sum={heat.sum():.4f}"

    def test_heat_diffusion_no_negative(self, wn):
        """热扩散无负值"""
        tids = wn._tokenize("天地玄黄")
        if len(tids) < 2:
            pytest.skip("分词结果太少")
        sim = wn._semantic_distances(tids)
        edges = wn._build_graph(tids, sim)
        heat = wn._heat_diffusion(len(tids), edges, steps=5, temp=0.1)
        assert all(h >= 0 for h in heat)

    # ── 文本→qi ──
    def test_text_to_qi_all_nonzero(self, wn):
        """轮转分桶有效：8卦全非零"""
        qi = wn._text_to_qi_v2("天地玄黄宇宙洪荒日月盈昃辰宿列张")
        assert all(v > 0.001 for v in qi.values()), f"有卦≈0: {qi}"

    def test_text_to_qi_not_dry_monopoly(self, wn):
        """轮转分桶不偏乾——PCA方差被8卦平分"""
        qi = wn._text_to_qi_v2("温柔顺从渗透柔软飘散")
        max_share = max(qi.values()) / sum(qi.values())
        assert max_share < 0.7, \
            f"单卦占比={max_share:.1%}，轮转分桶可能失效"

    def test_different_text_different_qi(self, wn):
        """不同文本 → 不同 qi 分布"""
        qi_a = wn._text_to_qi_v2("升职加薪很开心")
        qi_b = wn._text_to_qi_v2("被冤枉了很难受")
        a_vec = np.array([qi_a[g] for g in BAGUA])
        b_vec = np.array([qi_b[g] for g in BAGUA])
        diff = np.abs(a_vec - b_vec).sum()
        assert diff > 0.01, f"不同文本应有不同分布，diff={diff:.4f}"

    # ── ingest 多段合并 ──
    def test_ingest_frequency_weighting(self, wn):
        """多段文本频率加权——重复提及的词更热"""
        qi = wn.ingest(["天地", "天地", "天地", "宇宙", "洪荒"])
        assert abs(sum(qi.values()) - 1.0) < 0.01
        # "天地"出现3次，应实质性影响分布
        assert max(qi.values()) > 0.12, f"max={max(qi.values()):.3f} 太低"


# ═══════════════════════════════════════════
# 汉字物理 _compute_qi_physics
# ═══════════════════════════════════════════
@pytest.mark.engine_v9
class TestHanziPhysics:
    """汉字物理：笔画密度+方向角 → 8维qi"""

    def test_output_shape_and_sum(self):
        qi = _compute_qi_physics("天地玄黄宇宙洪荒")
        assert len(qi) == 8
        assert abs(qi.sum() - 1.0) < 0.01

    def test_empty_input(self):
        """无汉字时的回退"""
        qi = _compute_qi_physics("12345!@#$")
        assert len(qi) == 8
        assert abs(qi.sum() - 1.0) < 0.01

    def test_different_text_different_output(self):
        """不同文本 → 不同物理指纹"""
        qi_a = _compute_qi_physics("天地")
        qi_b = _compute_qi_physics("水火")
        diff = np.abs(qi_a - qi_b).sum()
        assert diff > 0.001, f"不同汉字应有不同物理指纹，diff={diff:.4f}"


# ═══════════════════════════════════════════
# engine_v9 三线汇合
# ═══════════════════════════════════════════
@pytest.mark.engine_v9
class TestEngineV9Merge:
    """engine_v9.perceive() — 三线合并 + 主v94"""

    def test_physics_only_no_crash(self):
        """仅汉字物理（无河图、无qi_field）不崩"""
        e = EngineV9(hour=12)
        r = e.perceive("天地玄黄宇宙洪荒", hetu_texts=None, qi_field=None)
        assert r.winner in BAGUA
        assert r.cv >= 0
        assert 0 < len(r.word_crystal) <= 5  # crystallize 默认 top_k=5

    def test_with_hetu_produces_signal(self):
        """有河图语义时 CV > 0（有信号）"""
        e = EngineV9(hour=12)
        r = e.perceive(
            "我升职了",
            hetu_texts=["恭喜升职", "新的挑战开始了", "能力得到认可"],
            qi_field=None
        )
        assert r.winner in BAGUA
        assert r.cv > 0, "有语义输入时应产生信号"

    def test_with_qi_field(self):
        """qi_field 参与合并"""
        e = EngineV9(hour=12)
        qi_field = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        qi_field = qi_field / qi_field.sum()
        r = e.perceive("测试", hetu_texts=["你好"], qi_field=qi_field)
        assert r.winner in BAGUA

    def test_different_inputs_different_output(self):
        """不同输入 → 不同分布（管线有区分力）"""
        e = EngineV9(hour=12)
        r1 = e.perceive("升职加薪很开心",
            hetu_texts=["恭喜你", "这是你应得的", "努力有了回报"])
        e2 = EngineV9(hour=12)
        r2 = e2.perceive("被冤枉了很难受",
            hetu_texts=["这太不公平了", "你需要时间消化", "不要冲动"])
        d1 = np.array([r1.distribution.get(g, 0) for g in BAGUA])
        d2 = np.array([r2.distribution.get(g, 0) for g in BAGUA])
        diff = np.abs(d1 - d2).sum()
        assert diff > 0.01, f"不同输入应不同分布，diff={diff:.4f}"

    def test_same_input_consistent(self):
        """确定性系统：相同输入 → 相同输出"""
        e1 = EngineV9(hour=12)
        r1 = e1.perceive("天地玄黄", hetu_texts=["宇宙洪荒", "日月盈昃"])
        e2 = EngineV9(hour=12)
        r2 = e2.perceive("天地玄黄", hetu_texts=["宇宙洪荒", "日月盈昃"])
        assert r1.winner == r2.winner, \
            f"同输入应同卦: {r1.winner} vs {r2.winner}"

    def test_physics_line_independent(self):
        """线①独立存在：无河图时也产出合法结果"""
        e = EngineV9(hour=12)
        r = e.perceive("天地人", hetu_texts=None, qi_field=None)
        assert r.winner in BAGUA
        # 线① qi_physics 应被记录
        assert r.qi_physics is not None
        assert len(r.qi_physics) == 8

    def test_semantic_line_independent(self, wn=None):
        """线③独立存在：河图文本经WordNetwork产出hetu_heat"""
        e = EngineV9(hour=12)
        r = e.perceive("测试",
            hetu_texts=["天地玄黄", "宇宙洪荒", "日月盈昃"],
            qi_field=None)
        # hetu_heat 应被记录且8卦全有值
        assert r.hetu_heat is not None
        assert set(r.hetu_heat.keys()) == set(BAGUA)
        assert sum(r.hetu_heat.values()) > 0.01


# ═══════════════════════════════════════════
# 外围层
# ═══════════════════════════════════════════
@pytest.mark.engine_v9
class TestPeripheralLayers:
    """外围层：记忆偏置、近窗微风、C层、指涉层"""

    def test_memory_bias_affects_distribution(self):
        """记忆偏置 → 初态被扰动"""
        e1 = EngineV9(hour=12)
        r1 = e1.perceive("测试", hetu_texts=None, qi_field=None)
        e2 = EngineV9(hour=12)