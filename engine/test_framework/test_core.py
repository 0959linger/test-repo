"""
五维核心验证（pytest 版）
从 test_core_v2.py 迁移，适配 test_framework 目录结构。

标记: core
用法: pytest test_framework/test_core.py -v
"""
import pytest, math
import numpy as np
from collections import defaultdict

# conftest.py 已把 engine/ 加入 sys.path
from core_enhanced import V94QichangEnhanced  # type: ignore

BAGUA = ['乾', '兑', '离', '震', '坤', '艮', '坎', '巽']

QUESTIONS = {
    "升职":   {'乾':5.0,'兑':3.5,'离':4.2,'震':3.0,'坤':2.8,'艮':2.5,'坎':2.2,'巽':3.0},
    "被冤枉": {'乾':3.0,'兑':2.8,'离':4.5,'震':3.5,'坤':5.0,'艮':3.2,'坎':4.0,'巽':3.0},
    "偷窃":   {'乾':3.5,'兑':4.8,'离':3.0,'震':3.2,'坤':3.0,'艮':2.5,'坎':4.0,'巽':2.8},
    "美":     {'乾':4.5,'兑':3.2,'离':4.0,'震':3.5,'坤':4.8,'艮':3.0,'坎':3.2,'巽':4.2},
    "爱":     {'乾':4.2,'兑':3.0,'离':4.5,'震':3.8,'坤':4.0,'艮':2.8,'坎':3.2,'巽':3.5},
    "数学":   {'乾':4.8,'兑':3.0,'离':3.5,'震':2.8,'坤':3.2,'艮':2.5,'坎':3.0,'巽':2.2},
    "说谎者": {'乾':3.5,'兑':3.8,'离':3.0,'震':2.5,'坤':3.3,'艮':2.0,'坎':4.2,'巽':2.7},
    "进化论": {'乾':4.5,'兑':2.8,'离':3.5,'震':3.0,'坤':3.2,'艮':2.5,'坎':3.8,'巽':2.2},
}

# ─── 维度1：结构诚实性 ───
@pytest.mark.core
def test_dim1_structural_honesty():
    """spread 与 CV 应正关联 (Spearman ρ > 0)"""
    spreads, cvs = [], []
    for name, qi in QUESTIONS.items():
        s = max(qi.values()) - min(qi.values())
        v94 = V94QichangEnhanced()
        r = v94.divine_from_qi(qi.copy())
        spreads.append(s)
        cvs.append(r['depth_cv'])
    
    n = len(spreads)
    rank_s = np.argsort(np.argsort(spreads))
    rank_c = np.argsort(np.argsort(cvs))
    rho = 1 - 6 * np.sum((rank_s - rank_c) ** 2) / (n * (n * n - 1))
    assert rho > 0, f"Spread-CV 应为正关联，实际 ρ={rho:.3f}"


# ─── 维度2：收敛底线 ───
@pytest.mark.core
@pytest.mark.parametrize("name,qi", [
    ("全零", {t: 0.001 for t in BAGUA}),
    ("全等", {t: 5.0 for t in BAGUA}),
    ("含负", {'乾':-1,'兑':0.5,'离':2,'震':1,'坤':3,'艮':1.5,'坎':0,'巽':2.5}),
    ("单热", {'乾':20,'兑':2,'离':2,'震':2,'坤':2,'艮':2,'坎':2,'巽':2}),
    ("极大", {'乾':1e10,'兑':1e9,'离':1e8,'震':1e7,'坤':1e6,'艮':1e5,'坎':1e4,'巽':1e3}),
])
def test_dim2_convergence(name, qi):
    """边界输入不崩溃，输出合法卦位"""
    v94 = V94QichangEnhanced()
    r = v94.divine_from_qi(qi)
    ds = sum(r['distribution'].values())
    assert r['winner'] in BAGUA, f"winner={r['winner']} 不合法"
    assert abs(ds - 1.0) < 0.01, f"Σdist={ds:.3f} 偏离1.0"


@pytest.mark.core
def test_dim2_random_no_crash():
    """随机输入 50 次零崩溃"""
    for seed in range(50):
        np.random.seed(seed)
        qi = {t: np.random.random() * 5 for t in BAGUA}
        v94 = V94QichangEnhanced()
        r = v94.divine_from_qi(qi)
        assert r['winner'] in BAGUA, f"seed={seed} winner={r['winner']}"


# ─── 维度3：自然聚类 ───
@pytest.mark.core
def test_dim3_natural_clustering():
    """场自决分组（不预设语义）"""
    clusters = defaultdict(list)
    for name, qi in QUESTIONS.items():
        v94 = V94QichangEnhanced()
        r = v94.divine_from_qi(qi.copy())
        clusters[r['winner']].append(name)
    
    n_groups = len(clusters)
    assert n_groups >= 2, f"聚类数={n_groups}，至少应有2组"


# ─── 维度4：临界区报告（不设 pass/fail，仅记录） ───
@pytest.mark.core
def test_dim4_critical_zone():
    """噪声下分布报告"""
    results = {}
    for name, qi_base in QUESTIONS.items():
        np.random.seed(42)
        counts = defaultdict(int)
        for _ in range(30):
            qi = {t: v * (1 + np.random.uniform(-0.02, 0.02)) for t, v in qi_base.items()}
            v94 = V94QichangEnhanced()
            r = v94.divine_from_qi(qi)
            counts[r['winner']] += 1
        top2 = sorted(counts.items(), key=lambda x: -x[1])[:2]
        results[name] = {"top": top2[0], "top_pct": top2[0][1] / 30}
    
    # 记录但不断言——临界区只报告
    critical = [(k, v) for k, v in results.items() if v["top_pct"] < 0.67]
    if critical:
        pytest.skip(f"临界区: {critical}")


# ─── 维度5：分辨力底线 ───
@pytest.mark.core
def test_dim5_resolution_floor():
    """归一化熵 ≥ 60%，单卦占比 < 75%"""
    winners = []
    for name, qi in QUESTIONS.items():
        v94 = V94QichangEnhanced()
        r = v94.divine_from_qi(qi.copy())
        winners.append(r['winner'])
    
    total = len(winners)
    counts = defaultdict(int)
    for w in winners:
        counts[w] += 1
    
    max_entropy = math.log2(8)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    norm_entropy = entropy / max_entropy
    
    assert norm_entropy >= 0.60, f"归一化熵={norm_entropy:.1%} < 60%，分布={dict(counts)}"
    
    max_share = max(counts.values()) / total
    assert max_share < 0.75, f"单卦占比={max_share:.0%} ≥ 75%，接近锁死"
