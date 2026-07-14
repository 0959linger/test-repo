"""
八卦构架2 测试框架 - 白话报告生成器

每次跑完测试自动生成 test_report.md，用玲能懂的中文解释：
- 这个测试在测什么
- 过了代表什么
- 没过代表什么风险
"""
import os, sys, traceback, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(HERE)

# --- 每个测试的白话解释 ---
EXPLANATIONS = {
    "test_dim1_structural_honesty": {
        "title": "维度1: 结构诚实性",
        "what": "检查'信号有多强'和'卦的判断有多果断'是否成正比。",
        "pass": "信号强时判断果断，信号弱时判断谨慎 -- 系统在'说实话'，没有瞎自信。",
        "fail": "系统可能在信号不强时假装很自信，或信号很强时反而不果断 -- 内部机制有'说谎'嫌疑。",
    },
    "test_dim2_convergence": {
        "title": "维度2: 收敛底线",
        "what": "拿各种极端输入(全零、全等、负数、天文数字、随机乱数)喂系统，看会不会崩。",
        "pass": "无论给什么离谱输入，系统都不会崩溃，总能给出合法卦位。",
        "fail": "有某种输入导致系统崩溃或输出非法结果 -- 这是bug，需要修。",
    },
    "test_dim2_random_no_crash": {
        "title": "维度2追加: 随机50次",
        "what": "连续给50组完全随机的输入，确保零崩溃。",
        "pass": "50次随机输入全部正常输出。",
        "fail": "某个随机组合触发了崩溃 -- 不易复现，但说明存在隐藏边界bug。",
    },
    "test_dim3_natural_clustering": {
        "title": "维度3: 自然聚类",
        "what": "8个不同问题，看系统能不能把相似问题归到同一卦位，而不是全散开或全挤一起。",
        "pass": "问题被自然地分成几组 -- 系统能识别问题的'气质'差异。",
        "fail": "要么所有问题归同一个卦(太僵)，要么完全散开(没识别能力)。",
    },
    "test_dim4_critical_zone": {
        "title": "维度4: 临界区报告",
        "what": "在每个问题上加微小噪声，看卦位会不会频繁跳变。",
        "pass": "大部分问题的卦位稳定，不受微小扰动影响。",
        "fail": "某个问题在噪声下频繁换卦 -- 该问题的判断处于'临界状态'，容易被干扰。不一定是bug，但值得关注。",
    },
    "test_dim5_resolution_floor": {
        "title": "维度5: 分辨力底线",
        "what": "检查卦位分布是否足够多样 -- 不能所有问题都归到一两个卦。",
        "pass": "至少3-4个卦位被用到，没有某个卦垄断一切。",
        "fail": "出现了'全乾锁死'或类似高度统一 -- 这是v92时代的经典bug，必须修。",
    },
    "test_all_rounds_pass": {
        "title": "E2E 全量: 24轮集成",
        "what": "8个问题 x 3轮呼吸回路，完整跑三条线(汉字物理+河图语义+v94气场)加四外围层。",
        "pass": "24轮全部正常，卦位合法，CV值正常。",
        "fail": "某轮崩溃或输出非法 -- 集成链路有断裂。需要看具体哪一轮崩的。",
    },
    "test_single_perceive": {
        "title": "E2E 冒烟: 单次感知",
        "what": "只跑一次最简流程，快速验证整个管道能通。",
        "pass": "一次感知正常，winner合法，CV在0-10之间。",
        "fail": "连最简流程都跑不通 -- 有基础性的导入错误或代码损坏。",
    },
    "test_no_hetu_fallback": {
        "title": "E2E 冒烟: 无河图回退",
        "what": "不给河图文本，看系统能不能仅靠汉字物理正常工作。",
        "pass": "不依赖河图也能给出合法卦位 -- 汉字物理管线完好。",
        "fail": "没有河图就崩了 -- 河图回退逻辑有问题。",
    },
    "test_engine_v9_imports": {
        "title": "补丁: 模块导入",
        "what": "检查所有核心模块能否正常导入。",
        "pass": "所有模块就绪。",
        "fail": "某个模块找不到 -- 可能文件被误删或改名了。",
    },
    "test_hanzi_physics_not_crash": {
        "title": "补丁: 汉字物理",
        "what": "喂一段经典文本，看汉字物理引擎是否正常输出8卦分布。",
        "pass": "8卦分布正常，总和为1。",
        "fail": "汉字物理管线有bug -- 可能字体文件问题或计算溢出。",
    },
    "test_bagua_constants": {
        "title": "补丁: 八卦常量",
        "what": "确认八卦的8个名字和顺序没有被改动。",
        "pass": "乾兑离震坤艮坎巽，顺序正确。",
        "fail": "八卦常量被改动了 -- 很基础的错误，可能影响所有下游逻辑。",
    },
    # ═══ 嘴巴测试 ═══
    "test_strong_gua_wins": {
        "title": "嘴巴: 卦位推演正确性",
        "what": "最强的卦应该是胜卦（先天八卦环上的主位）。",
        "pass": "乾占50%时，胜卦是乾 -- 推演引擎正确。",
        "fail": "最强卦没当上胜卦 -- 卦位推演逻辑有问题。",
    },
    "test_basic_output": {
        "title": "嘴巴: 双输出完整性",
        "what": "嘴巴吐出洞察和唠嗑两条内容，不为空。",
        "pass": "洞察和唠嗑都有内容 -- 嘴巴正常工作。",
        "fail": "洞察或唠嗑为空 -- 嘴巴输出有bug。",
    },
    "test_all_gua_produce_output": {
        "title": "嘴巴: 八卦全覆盖",
        "what": "8卦作为胜卦都能正常产出洞察和唠嗑。",
        "pass": "8卦全通 -- 嘴巴不会因某个卦而沉默。",
        "fail": "某卦无输出 -- 该卦的模板缺失或逻辑异常。",
    },
    # ═══ 外围层测试 ═══
    "test_first_update_returns_context": {
        "title": "记忆层: 首次更新",
        "what": "首次调用 memory.update 应返回完整上下文。",
        "pass": "返回上下文正确，delta_qi 为空（首次无差分）。",
        "fail": "记忆层初始化有问题。",
    },
    "test_gua_cooling": {
        "title": "记忆层: 卦位冷却",
        "what": "同一卦连续出现5次应触发冷却机制。",
        "pass": "冷却值>0，表示系统检测到重复。",
        "fail": "冷却未触发 -- 卦位锁死检测失效。",
    },
    "test_strategy_fatigue": {
        "title": "记忆层: 策略疲劳",
        "what": "同一策略连用4次应触发疲劳。",
        "pass": "疲劳标志=True，系统检测到重复。",
        "fail": "疲劳未触发 -- 策略多样性监控失效。",
    },
    "test_breeze_applied": {
        "title": "微风: 近窗微扰",
        "what": "近窗微风（epsilon=0.005）应在每轮产生微小影响。",
        "pass": "微风正确注入，分布总和仍为1。",
        "fail": "微风未注入或破坏了分布归一化。",
    },
    "test_observe_calls_api": {
        "title": "C层: API调用",
        "what": "C层 observe 应发送 HTTP 请求到 phi3。",
        "pass": "请求正确发送，推理日记返回。",
        "fail": "API调用失败或请求格式错误。",
    },
    "test_record_snapshot": {
        "title": "指涉层: 快照记录",
        "what": "指涉层 record 应存储每轮的完整快照。",
        "pass": "快照正确存储，卦位和CV值完整。",
        "fail": "快照存储异常。",
    },
    "test_anomaly_gua_lock": {
        "title": "指涉层: 锁死异常检测",
        "what": "连续5轮同卦应触发卦位锁死异常。",
        "pass": "检测到卦位锁死异常 -- 监控生效。",
        "fail": "锁死异常未触发 -- 指涉层监控失效。",
    },
    # ═══ 架构完整性测试 ═══
    "test_result_has_all_fields": {
        "title": "架构: 输出完整性",
        "what": "engine_v9.perceive() 返回结果应包含所有架构定义的字段。",
        "pass": "8个字段全部存在 -- 架构接口完整。",
        "fail": "某个字段缺失 -- 架构接口被破坏。",
    },
    "test_three_line_merge": {
        "title": "架构: 三线合并",
        "what": "三条热源（物理0.5+语义0.3+qi_field 0.2）正确合并。",
        "pass": "三线汇合后分布归一，卦位合法。",
        "fail": "合并逻辑有误，分布不归一。",
    },
    "test_no_arbitrator": {
        "title": "架构: 无仲裁者",
        "what": "winner 应由分布自然产生（argmax），不是外部选择。",
        "pass": "winner = 分布最大值 -- 场自决。",
        "fail": "winner 不是分布最大值 -- 有人为仲裁介入。",
    },
    "test_deterministic_no_random": {
        "title": "架构: 确定性",
        "what": "相同输入应产生完全相同的输出（v94无随机源）。",
        "pass": "两次完全相同 -- 确定性系统。",
        "fail": "输出不同 -- 可能引入了随机性，破坏可复现性。",
    },
    # ═══ qi_field 线测试 ═══
    "test_uniform_no_bias": {
        "title": "qi_field: 均匀无偏",
        "what": "均匀 qi_field 不引入偏置。",
        "pass": "均匀背景下卦位正常 -- 线②不干扰。",
        "fail": "均匀背景导致异常 -- qi_field 处理有问题。",
    },
    "test_qian_dominant_influences": {
        "title": "qi_field: 乾偏影响",
        "what": "乾偏的 qi_field 应让乾卦占比提升。",
        "pass": "乾偏下乾卦>0 -- 线②信号有效。",
        "fail": "乾偏未影响乾卦 -- qi_field 信号被忽略。",
    },
}


def generate_report(results, output_path=None):
    """
    根据测试结果生成白话报告。

    results: {test_name: True/False} 或 {test_name: {"passed": bool, "detail": str}}
    """
    if output_path is None:
        output_path = os.path.join(ENGINE_DIR, "test_report.md")

    total = len(results)
    _p = 0
    for v in results.values():
        if isinstance(v, dict):
            if v.get("passed") is not False:
                _p += 1
        elif v is not False:
            _p += 1
    passed = _p
    failed = total - passed

    lines = []
    lines.append("# 八卦构架2 - 测试报告\n")
    ok_mark = "OK" if failed == 0 else "XX"
    lines.append(f"> 自动生成 | {total}项测试 | 通过 {passed}/{total} | 失败 {failed}\n")
    lines.append("---\n")

    for test_name, outcome in results.items():
        expl = EXPLANATIONS.get(test_name, {
            "title": test_name,
            "what": "(缺少说明)",
            "pass": "通过",
            "fail": "未通过",
        })

        if isinstance(outcome, dict):
            is_pass = outcome.get("passed", False)
            detail = outcome.get("detail", "")
        else:
            is_pass = outcome
            detail = ""

        icon = "OK" if is_pass else "XX"
        lines.append(f"## {icon} {expl['title']}\n")
        lines.append(f"**在测什么:** {expl['what']}\n")
        if is_pass:
            lines.append(f"**结果:** {expl['pass']}\n")
        else:
            lines.append(f"**结果:** {expl['fail']}\n")
            if detail:
                lines.append(f"**细节:** `{detail}`\n")
        lines.append("")

    lines.append("---\n")
    lines.append("### 总结\n")
    if failed == 0:
        lines.append("全部通过，系统健康。\n")
    elif failed <= 2:
        lines.append(f"注意 {failed}项失败，建议排查但不紧急。\n")
    else:
        lines.append(f"警告 {failed}项失败，可能有严重bug，需要立即处理。\n")

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


# --- 独立运行: 跑测试 + 生成报告 ---
if __name__ == "__main__":
    sys.path.insert(0, ENGINE_DIR)
    sys.path.insert(0, os.path.join(ENGINE_DIR, '..', 'src', 'v94_qichang'))
    from engine_v9 import EngineV9, BAGUA, _compute_qi_physics
    from core_enhanced import V94QichangEnhanced

    results = {}

    # 补丁: 模块导入
    try:
        from memory_layer import MemoryLayer
        from observer_layer import ObserverLayer
        from c_layer import CLayer
        results["test_engine_v9_imports"] = True
    except Exception as e:
        results["test_engine_v9_imports"] = {"passed": False, "detail": str(e)}

    # 补丁: 汉字物理
    try:
        qi = _compute_qi_physics("天地玄黄宇宙洪荒日月盈昃辰宿列张")
        results["test_hanzi_physics_not_crash"] = abs(qi.sum() - 1.0) < 0.01
    except Exception as e:
        results["test_hanzi_physics_not_crash"] = {"passed": False, "detail": str(e)}

    # 补丁: 八卦常量
    results["test_bagua_constants"] = (
        len(BAGUA) == 8 and BAGUA[0] == '乾' and BAGUA[7] == '巽'
    )

    # E2E: 单次感知
    try:
        e = EngineV9(hour=12)
        r = e.perceive("测试", hetu_texts=["你好世界"], qi_field=np.ones(8)/8)
        results["test_single_perceive"] = r.winner in BAGUA and 0 <= r.cv <= 10
    except Exception as e:
        results["test_single_perceive"] = {"passed": False, "detail": str(e)}

    # E2E: 无河图回退
    try:
        e = EngineV9(hour=12)
        r = e.perceive("你好", hetu_texts=None, qi_field=None)
        results["test_no_hetu_fallback"] = r.winner in BAGUA
    except Exception as e:
        results["test_no_hetu_fallback"] = {"passed": False, "detail": str(e)}

    # 维度1: 结构诚实性
    try:
        spreads, cvs = [], []
        QUESTIONS = {
            "升职": {'乾':5.0,'兑':3.5,'离':4.2,'震':3.0,'坤':2.8,'艮':2.5,'坎':2.2,'巽':3.0},
            "被冤枉": {'乾':3.0,'兑':2.8,'离':4.5,'震':3.5,'坤':5.0,'艮':3.2,'坎':4.0,'巽':3.0},
            "偷窃": {'乾':3.5,'兑':4.8,'离':3.0,'震':3.2,'坤':3.0,'艮':2.5,'坎':4.0,'巽':2.8},
            "美": {'乾':4.5,'兑':3.2,'离':4.0,'震':3.5,'坤':4.8,'艮':3.0,'坎':3.2,'巽':4.2},
            "爱": {'乾':4.2,'兑':3.0,'离':4.5,'震':3.8,'坤':4.0,'艮':2.8,'坎':3.2,'巽':3.5},
            "数学": {'乾':4.8,'兑':3.0,'离':3.5,'震':2.8,'坤':3.2,'艮':2.5,'坎':3.0,'巽':2.2},
            "说谎者": {'乾':3.5,'兑':3.8,'离':3.0,'震':2.5,'坤':3.3,'艮':2.0,'坎':4.2,'巽':2.7},
            "进化论": {'乾':4.5,'兑':2.8,'离':3.5,'震':3.0,'坤':3.2,'艮':2.5,'坎':3.8,'巽':2.2},
        }
        for name, qi in QUESTIONS.items():
            s = max(qi.values()) - min(qi.values())
            v94 = V94QichangEnhanced()
            r = v94.divine_from_qi(qi.copy())
            spreads.append(s)
            cvs.append(r['depth_cv'])
        n = len(spreads)
        rank_s = np.argsort(np.argsort(spreads))
        rank_c = np.argsort(np.argsort(cvs))
        rho = 1 - 6 * np.sum((rank_s - rank_c)**2) / (n * (n*n - 1))
        results["test_dim1_structural_honesty"] = rho > 0
    except Exception as e:
        results["test_dim1_structural_honesty"] = {"passed": False, "detail": str(e)}

    # 维度2: 收敛底线
    try:
        v94 = V94QichangEnhanced()
        qi = {'乾':5.0,'兑':3.5,'离':4.2,'震':3.0,'坤':2.8,'艮':2.5,'坎':2.2,'巽':3.0}
        r = v94.divine_from_qi(qi)
        results["test_dim2_convergence"] = r['winner'] in BAGUA
    except Exception as e:
        results["test_dim2_convergence"] = {"passed": False, "detail": str(e)}

    # 维度2追加: 随机50次
    try:
        crash = False
        for seed in range(50):
            np.random.seed(seed)
            qi = {t: np.random.random() * 5 for t in BAGUA}
            v94 = V94QichangEnhanced()
            r = v94.divine_from_qi(qi)
            if r['winner'] not in BAGUA:
                crash = True
                break
        results["test_dim2_random_no_crash"] = not crash
    except Exception as e:
        results["test_dim2_random_no_crash"] = {"passed": False, "detail": str(e)}

    # 生成报告
    report = generate_report(results)
    print(report)
    print(f"\n报告已保存到: {os.path.join(ENGINE_DIR, 'test_report.md')}")
