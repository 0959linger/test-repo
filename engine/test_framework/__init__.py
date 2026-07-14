"""
测试框架 v2 — 对齐八卦构架2 完整落地结构

v1 → v2 升级清单：
1. 嘴巴独立测试 — 纯 numpy，不依赖外部
2. qi_field 线测试 — mock 不同结构的 qi_field
3. 记忆层/微风独立测试 — 验证初态偏置和每步微扰
4. C层/指涉层 mock — 验证接口正确性（不启动 phi3）
5. 架构完整性测试 — perceive() 内部阶段追踪
6. 散落测试合并 — 有用的逻辑整合

用法:
    python run_tests.py core     — 五维核心
    python run_tests.py e2e      — 端到端
    python run_tests.py engine_v9 — 引擎管线
    python run_tests.py mouth    — 嘴巴
    python run_tests.py periphery — 外围层
    python run_tests.py arch     — 架构完整性
    python run_tests.py report   — 白话报告
"""
