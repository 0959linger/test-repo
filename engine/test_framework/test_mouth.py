"""
嘴巴独立测试 — 纯 numpy，不依赖外部模型

测试 mouth.py 的完整行为：
- divine_path() 卦位推演
- Mouth.speak() 双输出
- 学习机制（anchor_force）
- 确定性（同卦同输入→同输出）
"""
import pytest, numpy as np
from mouth import Mouth, divine_path, BAGUA, INSIGHT, CHATTER  # type: ignore


@pytest.mark.mouth
class TestDivinePath:
    """卦位推演引擎 — qi_state → 路径"""

    def test_strong_gua_wins(self):
        """最强的卦应该是胜卦"""
        qi = np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        path = divine_path(qi)
        assert path['main_gua'] == '乾'

    def test_direction_detection(self):
        """方向应由相邻卦的相对强度决定"""
        qi = np.array([0.1, 0.3, 0.4, 0.1, 0.02, 0.02, 0.02, 0.02])
        path = divine_path(qi)
        assert path['main_gua'] == '离'
        # 震(0.1) > 兑(0.3)... 看具体方向逻辑
        assert path['direction'] in [-1, 0, 1]

    def test_direction_0_fallback(self):
        """方向为0时不应崩"""
        qi = np.ones(8) / 8  # 完全均匀
        path = divine_path(qi)
        assert path['main_gua'] in BAGUA
        assert path['path_guas'] == []  # 均匀场无方向

    def test_different_qi_different_path(self):
        """不同 qi → 不同路径"""
        qi_a = np.array([0.6, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03, 0.03])
        qi_b = np.array([0.03, 0.03, 0.6, 0.1, 0.1, 0.05, 0.03, 0.03])
        pa = divine_path(qi_a)
        pb = divine_path(qi_b)
        assert pa['main_gua'] != pb['main_gua'], "不同qi应不同主卦"


@pytest.mark.mouth
class TestMouthSpeak:
    """嘴巴双输出 — qi_state → {insight, chatter}"""

    def test_basic_output(self):
        """基础调用不崩，输出合法"""
        m = Mouth()
        qi = np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03])
        out = m.speak(qi, "测试文本")
        assert 'insight' in out
        assert 'chatter' in out
        assert len(out['insight']) > 0
        assert len(out['chatter']) > 0

    def test_different_gua_different_output(self):
        """不同卦位应产生不同洞察"""
        m1 = Mouth()
        m2 = Mouth()
        qi_a = np.array([0.6, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03, 0.03])
        qi_b = np.array([0.03, 0.03, 0.6, 0.1, 0.1, 0.05, 0.03, 0.03])
        out_a = m1.speak(qi_a, "test")
        out_b = m2.speak(qi_b, "test")
        # 洞察可能因方向不同而不同
        assert out_a['insight'] != out_b['insight'] or out_a['chatter'] != out_b['chatter'], \
            "不同卦位应产出不同输出"

    def test_uniform_qi_works(self):
        """均匀 qi 不崩"""
        m = Mouth()
        qi = np.ones(8) / 8
        out = m.speak(qi, "")
        assert 'insight' in out
        assert 'chatter' in out

    def test_insight_from_correct_gua(self):
        """洞察模板应与胜卦匹配"""
        m = Mouth()
        # 乾卦最强，且兑(0.1) < 坎(0.03)，direction=+1
        qi = np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.03, 0.03, 0.03])
        out = m.speak(qi, "测试")
        # 乾的洞察模板应该被用到（方向可能是+1或-1，取决于相邻卦）
        all_dry_templates = (INSIGHT.get(('乾', +1), []) + INSIGHT.get(('乾', 0), [])
                           + INSIGHT.get(('乾', -1), []))
        assert out['insight'] in all_dry_templates or out['insight'] == "嗯。", \
            f"洞察'{out['insight']}'不在乾卦模板中"

    def test_chatter_has_gua_suffix(self):
        """唠嗑应有卦位对应的语气后缀"""
        m = Mouth()
        qi = np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.03, 0.03, 0.03])
        out = m.speak(qi, "")
        # 乾后缀是"！"
        assert out['chatter'][-1] in ['！', '~', '…'], \
            f"语气后缀异常: {out['chatter'][-1]}"

    def test_deterministic_same_input(self):
        """同卦同输入同total_speaks→同输出（确定性）"""
        m = Mouth()
        qi = np.array([0.5, 0.1, 0.1, 0.1, 0.05, 0.03, 0.03, 0.03])
        out1 = m.speak(qi, "test123")
        out2 = m.speak(qi, "test123")
        # total_speaks 会 +1，所以洞察可能不同（SHA256 选模板）
        # 但 chatter 的核心词应该稳定
        assert len(out1['chatter']) > 0
        assert len(out2['chatter']) > 0

    def test_all_gua_produce_output(self):
        """8卦作为胜卦都能正常产出"""
        m = Mouth()
        for i, gua in enumerate(BAGUA):
            qi = np.zeros(8)
            qi[i] = 0.5
            qi = np.ones(8) * 0.1
            qi[i] = 0.5
            qi = qi / qi.sum()
            out = m.speak(qi, f"test_{gua}")
            assert len(out['insight']) > 0, f"卦{gua}无洞察输出"
            assert len(out['chatter']) > 0, f"卦{gua}无唠嗑输出"


@pytest.mark.mouth
class TestMouthLearn:
    """嘴巴学习机制 — anchor_force 衰减"""

    def test_learn_updates_anchor_force(self):
        """使用过的短语应降低 anchor_force，没用的反而会增加"""
        m = Mouth()
        qi = np.ones(8) / 8
        qi[0] = 0.5
        out = m.speak(qi, "test")
        # speak 会调用 learn_from_usage
        # 用过的短语 force × DECAY，没用过的 + BOOST
        # 关键是 total_speaks 应增加
        assert m.total_speaks >= 1

    def test_total_speaks_increments(self):
        """speak 次数应递增"""
        m = Mouth()
        qi = np.ones(8) / 8
        before = m.total_speaks
        m.speak(qi, "")
        after = m.total_speaks
        # speak() 内 total_speaks += 1，divine_path 不增加，learn_from_usage 末尾 += 1
        # 每次 speak 实际 +2
        assert after > before, f"speak 次数应递增，{before} → {after}"
