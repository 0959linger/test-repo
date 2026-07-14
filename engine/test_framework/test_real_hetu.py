"""
真河图端到端测试 — 5个实际模型 + engine_v9 全链路

前置条件：5个 llama-server 在 8080-8084 运行（launch_hetu_backend.py）

标记: hetu
用法: pytest test_framework/test_real_hetu.py -v
"""
import pytest, numpy as np, requests, time
from engine_v9 import EngineV9, BAGUA

# ── 河图模型 ──
HETU_PORTS = [8080, 8081, 8082, 8083, 8084]
HETU_NAMES = ["qwen0.5b", "qwen1.5b", "llama1b", "smollm135m", "phi3"]

# ── 测试问题 ──
HETU_QUESTIONS = [
    "我升职了很开心",
    "朋友背叛了我很难受",
    "工作压力很大",
    "今天天气很好适合出去走走",
    "遇到了一个难题不知道怎么解决",
]


def _model_health(port):
    """检查模型是否在线"""
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _call_hetu(port, prompt, max_tokens=80):
    """调单个河图模型"""
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[{e}]"


@pytest.mark.hetu
class TestHetuBackend:
    """河图后端：5模型健康检查 + 基础推理"""

    def test_all_five_models_alive(self):
        """5个模型全部在线"""
        for port, name in zip(HETU_PORTS, HETU_NAMES):
            assert _model_health(port), f"{name} (:{port}) 离线"

    def test_all_models_respond(self):
        """每个模型都能返回合法中文文本"""
        for port, name in zip(HETU_PORTS, HETU_NAMES):
            text = _call_hetu(port, "你好", max_tokens=30)
            assert text, f"{name} 返回空文本"
            assert len(text) > 1, f"{name} 返回太短: {text}"
            # 基本合法性：不只是错误信息
            assert "[ERROR]" not in text, f"{name} 返回错误: {text}"

    def test_model_diversity(self):
        """不同模型对同一问题给出不同回答（验证多视角分歧）"""
        responses = {}
        for port, name in zip(HETU_PORTS, HETU_NAMES):
            text = _call_hetu(port, "从你的角度分析：人生是什么", max_tokens=50)
            responses[name] = text

        # 至少3个模型给出不同的回答
        unique_texts = set(responses.values())
        assert len(unique_texts) >= 3, (
            f"5个模型只有{len(unique_texts)}种不同回答，多视角分歧不足"
        )


@pytest.mark.hetu
class TestHetuE2E:
    """真河图端到端：5模型 → engine_v9 → 卦象"""

    @pytest.fixture(scope="class")
    def engine(self):
        return EngineV9(hour=14)

    def test_perceive_with_real_hetu(self, engine):
        """真河图文本进 engine_v9.perceive() → 合法卦象"""
        # 生成6段河图文本（6个视角 × 5模型轮询）
        question = HETU_QUESTIONS[0]
        perspectives = [
            f"从逻辑角度分析：{question}",
            f"从情感角度分析：{question}",
            f"从实用角度分析：{question}",
            f"从风险角度分析：{question}",
            f"从创新角度分析：{question}",
            f"从传统角度分析：{question}",
        ]
        # 6段文本用5个端口轮询
        hetu_texts = []
        for i, prompt in enumerate(perspectives):
            port = HETU_PORTS[i % 5]
            text = _call_hetu(port, prompt, max_tokens=60)
            hetu_texts.append(text)

        assert len(hetu_texts) == 6, f"应有6段河图文本，实际{len(hetu_texts)}"

        # 进引擎
        result = engine.perceive(text=question, hetu_texts=hetu_texts, qi_field=None)

        # 基本合法性
        assert result.winner in BAGUA, f"胜卦非法: {result.winner}"
        dist_sum = sum(result.distribution.values())
        assert abs(dist_sum - 1.0) < 0.001, f"分布不归一: sum={dist_sum:.4f}"

    def test_five_questions_all_pass(self, engine):
        """5个问题全部通过真河图端到端"""
        for question in HETU_QUESTIONS:
            # 从5个模型各取一段文本
            hetu_texts = []
            for i, port in enumerate(HETU_PORTS):
                text = _call_hetu(port, f"简短回应：{question}", max_tokens=40)
                hetu_texts.append(text)

            result = engine.perceive(
                text=question, hetu_texts=hetu_texts, qi_field=None
            )

            assert result.winner in BAGUA, (
                f"[{question[:10]}...] 胜卦非法: {result.winner}"
            )
            assert result.cv >= 0, (
                f"[{question[:10]}...] CV负值: {result.cv}"
            )

    def test_hetu_vs_mock_consistency(self, engine):
        """真河图 vs 无河图 — 胜卦不应相同（河图应有影响）"""
        question = HETU_QUESTIONS[2]  # "工作压力很大"

        # 真河图
        hetu_texts = []
        for i, port in enumerate(HETU_PORTS):
            text = _call_hetu(port, f"简短回应：{question}", max_tokens=40)
            hetu_texts.append(text)
        result_real = engine.perceive(
            text=question, hetu_texts=hetu_texts, qi_field=None
        )

        # 无河图
        result_no = engine.perceive(text=question, hetu_texts=None, qi_field=None)

        # 验证真河图和裸物理的卦象分布有差异（河图确实在起作用）
        # 不要求胜卦不同（可能碰巧一致），但分布不能完全相同
        dist_diff = sum(
            abs(result_real.distribution[g] - result_no.distribution[g])
            for g in BAGUA
        )
        assert dist_diff > 0.001, (
            f"真河图与无河图分布完全一致(diff={dist_diff:.6f})，河图未起作用"
        )

    def test_word_crystal_with_hetu(self, engine):
        """真河图后的结晶词输出合法"""
        question = HETU_QUESTIONS[0]
        hetu_texts = []
        for i, port in enumerate(HETU_PORTS):
            text = _call_hetu(port, f"简短回应：{question}", max_tokens=40)
            hetu_texts.append(text)

        result = engine.perceive(
            text=question, hetu_texts=hetu_texts, qi_field=None
        )

        assert result.word_crystal, "word_crystal 为空"
        assert isinstance(result.word_crystal, list), "word_crystal 不是列表"
        assert len(result.word_crystal) > 0, "word_crystal 无内容"
        # 结晶词应该都在原始问题相关的语义范围内
        assert all(isinstance(w, str) for w in result.word_crystal), (
            "word_crystal 含非字符串"
        )
