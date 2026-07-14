"""
补丁 2026-07-13 — 测试框架初始化

每当我们讨论改代码，在这里追加一条对应测试。
格式：一个函数 = 一个补丁，函数名带日期和简短描述。
标记: patch
"""
import pytest, numpy as np

from engine_v9 import EngineV9, BAGUA  # type: ignore


@pytest.mark.patch
class TestPatch20260713:
    """2026-07-13: 框架初始化验证"""

    def test_engine_v9_imports(self):
        """engine_v9 所有依赖模块可正常导入"""
        from memory_layer import MemoryLayer  # type: ignore
        from observer_layer import ObserverLayer  # type: ignore
        from c_layer import CLayer  # type: ignore
        from core_enhanced import V94QichangEnhanced  # type: ignore
        assert V94QichangEnhanced is not None
        assert MemoryLayer is not None
        assert ObserverLayer is not None
        assert CLayer is not None

    def test_hanzi_physics_not_crash(self):
        """汉字物理管线不崩溃"""
        from engine_v9 import _compute_qi_physics  # type: ignore
        qi = _compute_qi_physics("天地玄黄宇宙洪荒日月盈昃辰宿列张")
        assert len(qi) == 8
        assert abs(qi.sum() - 1.0) < 0.01

    def test_bagua_constants(self):
        """八卦常量完整且有序"""
        assert len(BAGUA) == 8
        assert BAGUA[0] == '乾'
        assert BAGUA[7] == '巽'


    # ── 真河图全链路 ──
    @pytest.mark.skip(reason="需要5个河图模型(8080-8084)+phi3在线")
    def test_real_hetu_full_pipeline(self):
        """真河图焊接全链路: 河图5模型 → engine_v9 → 嘴巴"""
        import os, sys, numpy as np
        ROOT = r"C:\Users\ww109\.qwenpaw\workspaces\default"
        sys.path.insert(0, os.path.join(ROOT, "hetu"))

        from breathing_v38 import once
        from memory_layer import MemoryLayer as HetuMemoryLayer
        from observer_layer import ObserverLayer
        from c_layer import CLayer
        from mouth import Mouth

        memory = HetuMemoryLayer()
        observer = ObserverLayer(max_history=32)
        c_layer = CLayer(port=8084)
        engine = EngineV9(hour=12,
            embed_path=os.path.join(ROOT, "finding-order", "data", "qwen7b_embed_tokens.npy"),
            pca_path=os.path.join(ROOT, "finding-order", "data", "pca_256_proj.npz"))
        engine.attach_c_layer(c_layer)
        engine.attach_observer(observer)
        mouth = Mouth()

        # 河图
        r_hetu = once(question="我升职了！", memory=memory,
            steam_url="http://127.0.0.1:8080", trace=False, reason_c=True, near_window=None)
        ht = [e.text for e in r_hetu["hetu"].final_8[:6]]
        qf = np.array([r_hetu["qi"].get(g, 1.0/8) for g in BAGUA])

        # engine_v9
        r_v9 = engine.perceive(text="我升职了！", hetu_texts=ht, qi_field=qf)

        # 嘴巴
        qi_state = np.array([r_v9.distribution.get(g, 0) for g in BAGUA])
        mouth_out = mouth.speak(qi_state, "我升职了！")

        # 验证
        assert r_v9.winner in BAGUA, f"非法卦位: {r_v9.winner}"
        assert r_v9.cv > 0, f"CV异常: {r_v9.cv}"
        assert len(mouth_out.get("insight", "")) > 0, "嘴巴无输出"


# ═══════════════════════════════════════════════
# 追加模板（复制下面的结构写新补丁）：
#
# @pytest.mark.patch
# class TestPatchYYYYMMDD:
#     """YYYY-MM-DD: 简短描述改了什么"""
#
#     def test_某功能(self):
#         """具体测试描述"""
#         ...你的测试...
#         assert 条件
#
# ═══════════════════════════════════════════════
