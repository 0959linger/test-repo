"""
引擎 v9 端到端集成测试（pytest 版）
从 test_e2e_v9_quick.py 迁移，三线汇合 + 四外围层（模拟河图）。

标记: e2e
用法: pytest test_framework/test_e2e.py -v
"""
import pytest, time, numpy as np

from engine_v9 import EngineV9, BAGUA  # type: ignore
from memory_layer import MemoryLayer  # type: ignore
from observer_layer import ObserverLayer  # type: ignore
from c_layer import CLayer  # type: ignore

TESTS = [
    ("我升职了！", "升职"),
    ("朋友背叛了我，我的心被刀割一样。", "背叛"),
    ("有人偷了我的钱包。", "偷窃"),
    ("你怎么定义美？", "美"),
    ("爱到底是什么？", "爱"),
    ("计算：1+1等于几？", "数学"),
    ("如果这句话是假的，那它是真的吗？", "说谎者"),
    ("进化论正确吗？", "进化论"),
]

MOCK_HETU = {
    "升职": [
        "这次升职是对你能力的肯定，但也要注意团队关系。",
        "恭喜！但新的职位也意味着新的压力和期望。",
        "升职代表领导看到了你的价值，保持谦逊。",
        "权力来了，责任也来了。别忘了初心。",
        "这可能是一个转折点，需要重新规划。",
        "高位不仅需要能力，还需要智慧和耐力。",
    ],
    "背叛": [
        "这种痛苦说明你在意。背叛是人际伤害的极致。",
        "被信任的人伤害，心确实会碎。给自己时间愈合。",
        "不要马上反击，冷静下来再决定怎么处理。",
        "或许不是全部的错——先看看是否有误会。",
        "关系断裂的时候，最疼的是信任的崩塌。",
        "你可以愤怒，但不要让愤怒定义你。",
    ],
    "偷窃": [
        "钱被偷了，先报警，保护自己的权益。",
        "损失不只是钱，还有安全感。这需要恢复。",
        "小偷可能有自己的故事，但这不成为借口。",
        "尽快冻结银行卡、修改密码。",
        "或许是提醒你更注意自己的财物安全。",
        "不要因为一个人伤害你就怀疑所有人。",
    ],
    "美": [
        "美是主观的——每个人看到的美都不一样。",
        "美不仅在形式，还在内在的和谐与平衡。",
        "美不需要被定义，它是一种直觉的共鸣。",
        "自然的日出是美的，善良的微笑也是美的。",
        "美在秩序中诞生，也在混沌中隐藏。",
        "对美的追求，可能是人类最基本的冲动。",
    ],
    "爱": [
        "爱是无条件的给予，不求回报。",
        "爱也可能是一种占有，甚至是束缚。",
        "真正的爱是让对方成为自己，而不是你的影子。",
        "爱的反面不是恨，是冷漠。",
        "爱带来温暖，但也能带来最深的痛。",
        "你可能永远无法完全理解爱，只能去体验。",
    ],
    "数学": [
        "1+1=2，这是算术的基础。",
        "在二进制中，1+1=10，看你用什么进制。",
        "数学上的确定性让人安心。",
        "但这只是十进制下的结果——在别的系统可能不同。",
        "最简单的算式中藏着最深的哲理。",
        "确定性是数学的美，也是它的局限。",
    ],
    "说谎者": [
        "这句话在逻辑上不能成立，它自我矛盾。",
        "这是著名的说谎者悖论，两千年来没有定论。",
        "悖论揭示了语言的局限——它不能描述自己。",
        "如果它是假的，那么它说自己是假的——那就是真的？",
        "这个悖论冲击了逻辑的基础。",
        "或许答案不在真假中，而在语言与现实的裂缝里。",
    ],
    "进化论": [
        "进化论是生物学的基础理论，证据充分。",
        "但它不能解释一切——意识的起源仍是谜。",
        "自然选择很残酷，但这是自然法则。",
        "进化论不否定偶然性，也不推崇目的论。",
        "它提出了一种不需要设计者的复杂性来源。",
        "但基因漂变和自组织也是演化的重要力量。",
    ],
}


def _fake_qi_field():
    qi = np.ones(8) / 8 + np.random.normal(0, 0.02, 8)
    qi = np.clip(qi, 0.01, None)
    return qi / qi.sum()


@pytest.mark.e2e
class TestE2EBaseline:
    """E2E 基线：三线汇合 + 四外围层，8题×3轮，mock河图"""

    def test_all_rounds_pass(self):
        engine = EngineV9(hour=12)
        memory = MemoryLayer(window_size=8)
        observer = ObserverLayer(max_history=32)
        c_layer = CLayer(port=8084)

        engine.attach_c_layer(c_layer)
        engine.attach_observer(observer)

        results = []
        for text, label in TESTS:
            hetu_texts = MOCK_HETU.get(label, [f"回应{i}" for i in range(6)])
            for breath_round in range(3):
                if memory.prev_qi is not None:
                    engine.set_memory_bias(memory.prev_qi * 0.15)

                qi_field = _fake_qi_field()
                r = engine.perceive(text, hetu_texts=hetu_texts, qi_field=qi_field)

                assert r.winner in BAGUA, f"{label}轮{breath_round+1}: winner={r.winner}"
                assert r.cv >= 0, f"{label}轮{breath_round+1}: CV={r.cv} < 0"
                
                qi_state = np.array([r.distribution.get(g, 0) for g in BAGUA])
                memory.update(
                    text=text, qi_state=qi_state, winner=r.winner,
                    deviation=r.cv / 10, strategy="default"
                )
                results.append(r)

        assert len(results) == 24, f"应为24轮，实际{len(results)}"


@pytest.mark.e2e
class TestE2ESmoke:
    """E2E 冒烟：最小化测试，验证基本链路不崩溃"""

    def test_single_perceive(self):
        engine = EngineV9(hour=12)
        r = engine.perceive(
            "测试", hetu_texts=["你好世界"], qi_field=np.ones(8) / 8
        )
        assert r.winner in BAGUA
        assert 0 <= r.cv <= 10

    def test_no_hetu_fallback(self):
        """无河图文本时的回退：仅汉字物理 + qi_field"""
        engine = EngineV9(hour=12)
        r = engine.perceive("你好，今天天气不错", hetu_texts=None, qi_field=None)
        assert r.winner in BAGUA
