"""
八卦构架2 测试框架 — 便捷入口

用法:
    python run_tests.py              # 全量测试（排除 slow）
    python run_tests.py core         # 仅核心验证
    python run_tests.py e2e          # 仅 E2E
    python run_tests.py mouth        # 嘴巴独立测试
    python run_tests.py periphery    # 外围层（记忆/微风/C层/指涉层）
    python run_tests.py arch         # 架构完整性测试
    python run_tests.py qi_field     # qi_field 线测试
    python run_tests.py engine_v9    # 引擎管线
    python run_tests.py patch        # 仅补丁
    python run_tests.py smoke        # 快速冒烟
    python run_tests.py full         # 全量（含 slow）
    python run_tests.py report       # 生成白话报告（给玲看的）

依赖:
    需要 pytest (pip install pytest)。若未安装会提示安装命令，
    同时提供纯 Python 回退模式 (python run_tests.py core --fallback)。
    report 模式不需要 pytest。
"""
import sys, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

arg = sys.argv[1] if len(sys.argv) > 1 else ""
use_fallback = "--fallback" in sys.argv

# 清理 --fallback 标记
sys.argv = [a for a in sys.argv if a != "--fallback"]

# --- report 模式: 生成白话报告 ---
if arg == "report":
    subprocess.call([sys.executable, "test_framework/reporter.py"])
    print("\n玲，报告已经生成好了:")
    report_path = os.path.join(HERE, "test_report.md")
    with open(report_path, "r", encoding="utf-8") as f:
        print(f.read())
    sys.exit(0)

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    if not use_fallback:
        print("⚠ pytest 未安装。运行: pip install pytest")
        print("  或用纯 Python 回退: python run_tests.py core --fallback\n")

if HAS_PYTEST and not use_fallback:
    # ─── pytest 模式 ───
    cmd = ["pytest", "test_framework/", "-v", "--tb=short"]
    
    # 默认排除 slow（7B嵌入表加载）
    if arg not in ("full", "engine_v9"):
        cmd += ["-m", "not slow"]
    
    if arg == "core":
        cmd += ["-m", "core"]
    elif arg == "e2e":
        cmd += ["-m", "e2e"]
    elif arg == "mouth":
        cmd += ["-m", "mouth"]
    elif arg == "periphery":
        cmd += ["-m", "periphery"]
    elif arg == "arch":
        cmd += ["-m", "arch"]
    elif arg == "qi_field":
        cmd += ["-m", "qi_field"]
    elif arg == "engine_v9":
        cmd += ["-m", "engine_v9"]
    elif arg == "patch":
        cmd += ["-m", "patch"]
    elif arg == "smoke":
        cmd = ["pytest", "test_framework/", "-v", "--tb=short",
               "-m", "not slow",
               "-k", "smoke or core or test_single_perceive or patch"]
    elif arg == "full":
        cmd = ["pytest", "test_framework/", "-v", "--tb=short"]  # 包含 slow

    print(f"▶ {' '.join(cmd)}\n")
    sys.exit(subprocess.call(cmd))

else:
    # ─── 纯 Python 回退 ───
    print(" " * 13 + "八卦构架2 测试框架 (纯 Python 回退)\n")

    import importlib.util as _iu

    def _load(name, path):
        spec = _iu.spec_from_file_location(name, os.path.join(HERE, path))
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    all_ok = True
    import traceback

    if arg in ("", "core"):
        print("═══ 五维核心验证 ═══\n")
        try:
            mod = _load("test_core", "test_framework/test_core.py")
            mod.test_dim1_structural_honesty()
            mod.test_dim2_random_no_crash()
            for name, qi in mod.test_dim2_convergence.pytestmark[0].args[1]:
                v94 = mod.V94QichangEnhanced()
                r = v94.divine_from_qi(qi)
                assert r['winner'] in mod.BAGUA
            mod.test_dim3_natural_clustering()
            mod.test_dim5_resolution_floor()
            print("\n  ✅ 五维核心验证通过")
        except:
            print(traceback.format_exc())
            all_ok = False

    if arg in ("", "e2e"):
        print("\n═══ E2E 冒烟 ═══\n")
        try:
            from engine_v9 import EngineV9  # type: ignore
            import numpy as np
            e = EngineV9(hour=12)
            r = e.perceive("测试", hetu_texts=["你好世界"], qi_field=np.ones(8) / 8)
            assert r.winner in ['乾','兑','离','震','坤','艮','坎','巽']
            print(f"  winner={r.winner} cv={r.cv:.2f}")
            print("  ✅ E2E 冒烟通过")
        except:
            print(traceback.format_exc())
            all_ok = False

    if arg in ("", "patch"):
        print("\n═══ 补丁测试 ═══\n")
        try:
            mod = _load("patches", "test_framework/patches/2026-07-13.py")
            tc = mod.TestPatch20260713()
            tc.test_engine_v9_imports()
            tc.test_hanzi_physics_not_crash()
            tc.test_bagua_constants()
            print("  ✅ 补丁测试通过")
        except:
            print(traceback.format_exc())
            all_ok = False

    print("\n" + "=" * 50)
    if all_ok:
        print("✅ 全部通过")
        sys.exit(0)
    else:
        print("❌ 有测试失败")
        sys.exit(1)
