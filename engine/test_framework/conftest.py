"""
八卦构架2 测试框架 — pytest 配置
路径初始化 + 共享 fixture，所有测试文件从此继承。

用法（在 finding-order/engine/ 目录下）：
    pytest test_framework/ -v           # 全量
    pytest test_framework/ -v -m core   # 仅核心验证
    pytest test_framework/ -v -m e2e    # 仅 E2E
    pytest test_framework/ -v -m mouth  # 嘴巴
    pytest test_framework/ -v -m periphery  # 外围层
    pytest test_framework/ -v -m arch   # 架构完整性
    pytest test_framework/ -v -m patch  # 仅补丁
    pytest test_framework/ -v -k "v94"  # 按名称筛
"""
import sys, os, pytest

# 把 engine/ 加入路径，所有测试可以直接 import engine_v9 等
ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

# 把 v94_qichang 加入路径（core_enhanced 在这里）
V94_SRC = os.path.join(ENGINE_DIR, '..', 'src', 'v94_qichang')
sys.path.insert(0, os.path.normpath(V94_SRC))


# ═══════════════════════════════════════════
# 共享 fixture — 嵌入表只加载一次
# ═══════════════════════════════════════════

@pytest.fixture(scope="session")
def shared_engine():
    """全局共享的 EngineV9 实例，嵌入表只加载一次。
    
    用法:
        def test_xxx(shared_engine):
            r = shared_engine.perceive(...)
    """
    from engine_v9 import EngineV9
    e = EngineV9(hour=12)
    # 预热：触发 _ensure_loaded()，加载嵌入表
    e.perceive("预热", hetu_texts=["你好"], qi_field=None)
    return e


@pytest.fixture(scope="session")
def fresh_engine_factory():
    """工厂 fixture：每次调用返回新的 EngineV9 实例。
    用于需要隔离状态的测试（如 round 计数、near_qi）。
    注意：每个实例都会重新加载嵌入表，尽量用 shared_engine。
    """
    from engine_v9 import EngineV9
    def _factory(hour=12):
        return EngineV9(hour=hour)
    return _factory
