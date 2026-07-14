# v91+v92 完整架构图

## 节点：v92-v91restored-golden
日期：2026-06-12
文件：`v92_v91restored_golden.py`

---

## 一、整体分层

```
┌─────────────────────────────────────────────────────────────┐
│                      v92 观测层（纯观察）                     │
│  ┌─────────┐  ┌───────────┐  ┌──────────────────────┐       │
│  │ Model   │  │ BaguaLoad │  │ depth_history        │       │
│  │ Infer-  │  │ Scheduler │  │ (最近50步)           │       │
│  │ ence    │  │ 动态加载  │  │                      │       │
│  │ logprob │  │           │  │                      │       │
│  │ →factor │  │           │  │                      │       │
│  └────┬────┘  └─────┬─────┘  └──────────────────────┘       │
│       │             │              │                         │
│       ▼             ▼              ▼                         │
│  result["model_factors"]  result["loaded"]  外部分析         │
├─────────────────────────────────────────────────────────────┤
│              ← 隔离边界：v92不可越过 →                       │
├─────────────────────────────────────────────────────────────┤
│                      v91 物理层（完全自主）                   │
│                                                             │
│  信号 ──→ SignalSmoother ──→ flow_rate                      │
│              │                                              │
│              ▼                                              │
│        RhythmCircle ──→ disturbance[0..7]                   │
│              │                                              │
│              ▼                                              │
│   ┌──────────────────────────────────────────┐             │
│   │   CircleChannel × 8 (erode)              │             │
│   │                                          │             │
│   │   depth_change =                         │             │
│   │     flow * depth * growth * disturbance  │  ← 自反馈  │
│   │   + flow * growth * disturbance          │  ← 线性项  │
│   │   - decay * time_delta                   │  ← 衰减    │
│   │                                          │             │
│   │   熔断：std > 0.30% → 归零               │             │
│   └──────────────────────────────────────────┘             │
│                                                             │
│   桥接：scale-free diff > 4.9 → bridge_rate                 │
└─────────────────────────────────────────────────────────────┘
```

## 二、v91 核心不变部分（永不触碰）

### 第一层：IronLaws 铁律常数
```
熔断线          0.30%    不可变
桥梁阈值        4.90     不可变
侵蚀基础        0.001    不可变
衰减率          0.005    不可变
最大流量        20.00    不可变
节奏范围        [0.7, 1.0]
共振衰减        0.08
节奏耦合        0.12
比例衰减阈值    8000
```

### 第二层：BAGUA_TOPOLOGY（八卦骨架）
```
   ┌─────┐          ┌─────┐
   │ 0乾  │──────────│ 1坤  │
   └──┬───┘          └──┬───┘
      │  ↖              │
   ┌──┴───┐         ┌───┴──┐
   │ 7兑  │         │ 6巽  │
   └──┬───┘         └──┬───┘
      │                 │
   ┌──┴───┐         ┌───┴──┐
   │ 2离  │         │ 5艮  │
   └──┬───┘         └──┬───┘
      │                 │
   ┌──┴───┐         ┌───┴──┐
   │ 3坎  │─────────│ 4震  │
   └──────┘         └──────┘

   每卦恰好2连接，无自连接
   总连接数 = 16
```

### 第六层：CircleChannel.erode()（冲刷公式）
```python
# 完整v91公式，v92不修改
depth_change = (
    flow_rate * self.depth * growth_factor * disturbance  # 自反馈（沟壑加深）
  + flow_rate * growth_factor * disturbance              # 线性项（新沟壑）
  - decay_rate * time_delta                              # 衰减
)
self.depth = max(0.0, self.depth + depth_change)

# 比例衰减
if self.depth > 8000:
    self.depth *= 0.995

# 通道微差：每个通道的 growth_factor 差 0.0001
growth_factor = 0.001 * (1.0 + channel_idx * 0.0001)
```

### 熔断机制
```python
honesty_check():
    std = get_std()       # 归一化波动
    threshold = 0.30      # 铁律不可变
    if flow_rate > 15:
        threshold = min(0.5, 0.30 + (flow_rate - 15) * 0.01)
    if std > threshold:
        fuse_all()        # 全部通道归零
        raise HonestDeath
```

## 三、v92 扩展部分（纯观测层）

### 第三层：模型绑定（BaguaModelBinding）
```
常驻(4个，~4GB):
  0乾: Qwen2.5-3B (1.8GB)     ─ 逻辑
  1坤: Qwen2.5-Coder-1.5B (0.9GB) ─ 数据
  4震: Qwen2.5-1.5B (0.9GB)   ─ 怀疑
  7兑: Qwen2.5-3B (1.8GB)     ─ 表达

按需(4个，最大9GB):
  2离: Huihui-Qwen3-8B (4.7GB)   ─ 直觉
  3坎: DeepSeek-V2-Lite (9.1GB)  ─ 情感
  5艮: Qwen2.5-0.5B (0.4GB)     ─ 记忆
  6巽: QwenPaw-Flash-9B (5.4GB)  ─ 联想
```

### ModelInference（logprob → factor）
```python
infer(signal):
    1. llama-server /completion API
    2. 提取 completion_probabilities[].logprob
    3. 平均 logprob → factor 映射
    
    baseline = -0.8    # 模型平均困惑度
    scale = 0.2        # 映射系数
    factor = 1.0 + (avg_logprob - baseline) * scale
    factor = clamp(factor, 0.9, 1.1)
    
    物理意义：factor ≠ 质量分数
    factor = 模型处理此信号时的认知消耗（纯物理量）
```

### BaguaLoadScheduler（纯调度，不影响冲刷）
```
predict_needed(signal):
    1. 信号分解器分析文本特征
    2. 匹配卦位性质（逻辑/数据/情感…）
    3. 返回需要加载的卦位列表
    
ensure_loaded(needed):
    1. 检查已有server进程
    2. 按需启动/停止llama-server
    3. 等待就绪
```

### flush_step() 执行顺序
```
1. verify_bagua_integrity()     ← v91 检查
2. SignalSmoother.smooth()      ← v91
3. RhythmCircle.disturbance()   ← v91
4. CircleChannel.erode() ×8    ← v91 核心（统一flow_rate）
5. honesty_check()              ← v91 熔断
6. ModelInference.infer() ×8   ← v92 观测（在v91之后）
7. depth_history记录            ← v92 观测
```

## 四、不变性保证

| 层级 | 元素 | 状态 | 证明 |
|------|------|------|------|
| 铁律 | 所有常数 | 绝对不变 | verify_v91_restore 10/10 ✅ |
| 骨架 | BAGUA_TOPOLOGY | 16连接固定 | integrity check ✅ |
| 公式 | erode() | 不含任何v92引用 | diff vs v91 ✓ |
| 熔断 | honesty_check() | 仅v91逻辑 | 无趋势检测 ✅ |
| 冲刷 | flow_rate | 统一流速 | 各通道一致 ✅ |
| 观测 | model_factors | 不进入depth_change | 冲刷后获取 ✅ |

## 五、v92 与 v91 的关系

```
v91 = 物理层
        │
        ├── 冲刷公式（自然沟壑形成）
        ├── 熔断机制（防虚假共识）
        ├── 节奏环（全局时钟）
        └── 极限平滑器（信号稳定）
        
v92 = 观测层（建立在v91之上）
        │
        ├── 模型logprob采集（不影响冲刷）
        ├── 深度历史记录（不影响冲刷）
        ├── 动态调度（不影响冲刷）
        └── 输出扩展字段（model_factors, loaded）
        
关键：v92 位于 v91 的"下游"
      v91 冲刷完成 → v92 观测 → 返回合并结果
      两者之间只有单向数据流
```

## 六、关键决策记录

1. **factor 不进 erode()** — 否则指数放大导致熔断
2. **不修改熔断阈值** — v91铁律不可触碰
3. **不引入趋势检测** — "判断差异真假"本质是仲裁
4. **v92 模型调用在冲刷之后** — 确保v91独立运行
5. **model_factors 仅作为观测输出** — 外部分析使用

## 七、已知限制

- v92的model_factor差异反映真实物理差异（logprob）
- 但从未被用于任何决策或冲刷过程
- 如何在"不仲裁"前提下让这些物理差异有用 → 下一个研究点
