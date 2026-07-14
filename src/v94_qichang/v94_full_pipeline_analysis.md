# v94 完整管线分析 — 翻译层与权重地图

## 管线全流程

```
7D特征提取
  ↓
initialize_qichang (初始化)
  ├─ 翻译1: features → arctan2 → 3个角度 (d1,d2,d3)
  ├─ 翻译2: 角度 + trigram角度 → cos(diff) → qi相似度
  ├─ 翻译3: qi *= entropy_mod (熵调制)
  ├─ 翻译4: sin(π·qi) (物极必反劈开)
  ├─ 翻译5: weight = 0.5 + 0.2*qi (sin→权重)
  └─ 翻译6: total_qi/8 * weight * 2 (权重→绝对值)
  ↓
flush_evolution × N步 (冲刷演化)
  ├─ 翻译7: qi → diffusion = (qi-avg)*rate (值→流)
  ├─ 翻译8: qi^1.5 → attract (值→引力)
  └─ 翻译9: qi*rate → decay (值→衰减)
  ↓
cascade_chase (级联追逐)
  └─ 翻译10: 按值排序 → 相邻配对 → diff → overflow (值→排序对→流)
  ↓
cooling + earthquake (冷却+地震)
  ├─ 翻译11: cascade_records → src累积 (记录→个体累积)
  └─ 翻译12: cascade_records → sources *= (1-strength) (记录→个体削减)
  ↓
taiji_decay (太极衰减)
  └─ 翻译13: extremeness → 1/(1+(ext-1)*factor) (极端度→乘数)
  ↓
condensation (凝结)
  └─ 翻译14: softmax(qi) → distribution (qi→概率分布)
```

## 14个翻译层分类

### 🔴 有问题的翻译层（格式不匹配，靠数学胶水）

| # | 位置 | 翻译内容 | 问题 |
|---|------|---------|------|
| 1 | initialize | 7D特征 → 3个角度 | 静态特征→角度，d1/d2被锁死 |
| 2 | initialize | 角度 → cos相似度 | 角度→标量，丢失方向性 |
| 5 | initialize | sin → weight [0.3,0.7] | 负值被压缩，物极必反失效 |
| 6 | initialize | weight → 绝对值 | 权重→绝对值，丢失相对结构 |
| 10 | cascade | 值 → 排序配对 | 完全无视trigram关系 |

### 🟡 中性的翻译层（格式合理但有改进空间）

| # | 位置 | 翻译内容 | 备注 |
|---|------|---------|------|
| 3 | initialize | entropy调制 | 全局缩放，合理 |
| 4 | initialize | sin劈开 | 物理直觉正确，但被翻译5破坏 |
| 7 | flush | diffusion | 线性扩散，标准做法 |
| 8 | flush | qi^1.5吸引 | 超线性正反馈，合理 |
| 9 | flush | decay | 线性衰减，标准做法 |
| 13 | taiji | extremeness→衰减 | 合理，但factor=0.5是硬编码 |
| 14 | condense | softmax | 标准概率化，合理 |

### 🟢 无问题的翻译层

| # | 位置 | 翻译内容 | 备注 |
|---|------|---------|------|
| 11 | cooling | records→累积 | 直接操作，无格式转换 |
| 12 | earthquake | records→削减 | 直接操作，无格式转换 |

## 9个常数权重/比率

| 名称 | 值 | 位置 | 是否可八卦化 |
|------|-----|------|-------------|
| flush_rate | 0.1 | line 57 | ❌ 全局时间尺度 |
| diffusion_rate | 0.05 | line 58 | ✅ 可按卦位关系调整 |
| aggregation_multiplier | 2.0 | line 59 | ✅ 可按卦位关系调整 |
| CASCADE_RATIO | 0.4 | line 254 | ✅ 可按相生关系调整 |
| CASCADE_LOSS | 0.2 | line 255 | ✅ 可按相克关系调整 |
| COOLING_SUPPRESS | 0.15 | line 211 | ✅ 可按卦位状态调整 |
| COOLING_DECAY | 0.92 | line 279 | ❌ 全局时间衰减 |
| EARTHQUAKE_STRENGTH | 0.10 | line 301 | ✅ 可按卦位状态调整 |
| decay_factor | 0.5 | line 335 | ❌ 全局太极参数 |

## 核心发现

1. **翻译层集中在 initialize_qichang** — 6步翻译链，每步都在丢失信息
2. **级联追逐完全无视trigram结构** — 只看值大小，不看谁生谁克
3. **9个常数权重中6个可以八卦化** — 但需要物理直觉，不能硬编码
4. **物极必反在翻译5被破坏** — sin产生负值是对的，但weight=0.5+0.2*qi把它压回正值

## 重整方向

不是"升级到权重"，而是**让组件的输入输出自然匹配八卦结构**：
- 去掉翻译1-6的6步链，让7D特征直接映射到八卦空间
- 让级联追逐感知相生关系，而不是盲排
- 让常数权重感知卦位状态，而不是固定数字
