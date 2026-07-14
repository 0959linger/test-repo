"""
异步管线测试：码点方向（即刻） + 位图气量（后台线程）

验证：
1. 时间配合度（位图会不会拖慢整体）
2. 效果一致性（异步 vs 同步结果是否相同）
3. 两条管线的扩展自由度
"""

import sys, os, time, threading, math, numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from core_enhanced import V94QichangEnhanced

font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 64)

# ============================================================
# 码点方向（即刻，μs级）
# ============================================================
def codepoint_direction_v1(text):
    """中位角法"""
    chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    if len(chars) < 2:
        return 0.0
    codes = [(ord(c) - 0x4E00) / (0x9FFF - 0x4E00) * 2 - 1 for c in chars]
    diffs = [codes[i+1] - codes[i] for i in range(len(codes)-1)]
    angles = [d * 180 for d in diffs]
    return np.median(angles) % 360 if angles else 0.0


# ============================================================
# 位图气量（后台线程，可独立扩展精度/维度）
# ============================================================

FINE_SIZE = 128  # 异步版可以用大尺寸

def bitmap_render_fine(text):
    """精细位图渲染（可在独立线程中运行）"""
    chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    if not chars:
        return {'density': 0.1, 'density_std': 0.0, 'symmetry': 0.5, 
                'horiz_ratio': 0.5, 'vert_ratio': 0.5, 'quad_balance': 0.5}
    
    densities = []
    symmetries = []
    horiz_ratios = []
    vert_ratios = []
    quad_balances = []
    
    for char in chars:
        try:
            img = Image.new('L', (FINE_SIZE, FINE_SIZE), 255)
            draw = ImageDraw.Draw(img)
            font_large = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", FINE_SIZE)
            bbox = draw.textbbox((0, 0), char, font=font_large)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (FINE_SIZE - tw) // 2 - bbox[0]
            y = (FINE_SIZE - th) // 2 - bbox[1]
            draw.text((x, y), char, fill=0, font=font_large)
            
            arr = np.array(img, dtype=np.float32)
            black = arr < 128
            density = np.mean(black)
            
            # 左右对称
            h = FINE_SIZE // 2
            left = np.mean(black[:, :h])
            right = np.mean(black[:, h:])
            sym = 1.0 - abs(left - right) / max(density, 0.001)
            
            # 横竖笔画比例
            horiz_runs = 0
            for row in black:
                in_run = False
                for p in row:
                    if p and not in_run:
                        horiz_runs += 1; in_run = True
                    elif not p: in_run = False
            
            vert_runs = 0
            for col in black.T:
                in_run = False
                for p in col:
                    if p and not in_run:
                        vert_runs += 1; in_run = True
                    elif not p: in_run = False
            
            total_runs = horiz_runs + vert_runs + 1
            horiz_ratio = horiz_runs / total_runs
            
            # 四象限平衡度
            qh = FINE_SIZE // 2
            tl = np.mean(black[:qh, :qh])
            tr = np.mean(black[:qh, qh:])
            bl = np.mean(black[qh:, :qh])
            br = np.mean(black[qh:, qh:])
            quad_diff = abs(tl - br) + abs(tr - bl)
            quad_balance = 1.0 - quad_diff / max(density * 2, 0.001)
            
            densities.append(density)
            symmetries.append(sym)
            horiz_ratios.append(horiz_ratio)
            vert_ratios.append(horiz_runs)
            quad_balances.append(quad_balance)
        except:
            densities.append(0.1); symmetries.append(0.5)
            horiz_ratios.append(0.5); vert_ratios.append(0.5)
            quad_balances.append(0.5)
    
    return {
        'density': np.mean(densities),
        'density_std': np.std(densities),
        'symmetry': np.mean(symmetries),
        'horiz_ratio': np.mean(horiz_ratios),
        'quad_balance': np.mean(quad_balances),
    }


# ============================================================
# 合成气量
# ============================================================
def compose(direction_deg, density_info, base_qi=0.3):
    trigram_angles = {
        '乾': 0, '兑': 45, '离': 90, '震': 135,
        '坤': 180, '艮': 225, '坎': 270, '巽': 315,
    }
    density = density_info.get('density', 0.15)
    density_std = density_info.get('density_std', 0.01)
    symmetry = density_info.get('symmetry', 0.5)
    quad_balance = density_info.get('quad_balance', 0.5)
    
    magnitude = density * 2.0
    richness_bonus = 1.0 + density_std * 3.0
    asymmetry_bonus = 1.0 + (1.0 - symmetry) * 0.5
    balance_bonus = 0.8 + quad_balance * 0.4
    
    total = magnitude * richness_bonus * asymmetry_bonus * balance_bonus
    
    qichang = {}
    for name, angle in trigram_angles.items():
        diff = abs(direction_deg - angle)
        diff = min(diff, 360 - diff)
        weight = max(0, math.cos(math.radians(diff)))
        qichang[name] = base_qi + weight * total
    return qichang


# ============================================================
# 异步管线
# ============================================================

class AsyncPipeline:
    def __init__(self, v94):
        self.v94 = v94
        self._future = None
        self._direction = None
    
    def start(self, text):
        """启动异步处理：方向即刻，位图后台"""
        self._direction = codepoint_direction_v1(text)
        self._future = threading.Thread(
            target=self._render_job, args=(text,)
        )
        self._future.start()
    
    def _render_job(self, text):
        self._density_result = bitmap_render_fine(text)
    
    def ready(self):
        """检查位图是否就绪"""
        return self._future is not None and not self._future.is_alive()
    
    def get_result(self):
        """阻塞直到位图就绪，合成并推演"""
        if self._future and self._future.is_alive():
            self._future.join()
        
        qichang = compose(self._direction, self._density_result)
        result = self.v94.divine_from_qi(qichang, trace=False)
        return result, self._direction, self._density_result


# ============================================================
# 测试
# ============================================================

questions = [
    ("偷窃是错误的吗", "道德"),
    ("什么是美", "哲学"),
    ("量子力学的基本原理是什么", "技术"),
    ("今天天气真好", "日常"),
    ("一加一等于几", "事实"),
]

print("=" * 65)
print("异步管线测试：码点方向（即刻） + 位图渲染（后台）")
print("=" * 65)
print()

print(f"位图精度: {FINE_SIZE}×{FINE_SIZE}")
print()

v94 = V94QichangEnhanced()
pipeline = AsyncPipeline(v94)

# ----- 测试1：时间配合度 -----
print("【测试1】时间配合度")
print("-" * 65)
print(f"{'问题':<30} {'方向(before)':>12} {'渲染(ms)':>10} {'总耗时(ms)':>10}")
print("-" * 65)

for q, cat in questions:
    # 即刻取方向
    t0 = time.perf_counter()
    direction = codepoint_direction_v1(q)
    t_dir = (time.perf_counter() - t0) * 1000
    
    # 后台启动渲染
    t1 = time.perf_counter()
    density = bitmap_render_fine(q)
    t_render = (time.perf_counter() - t1) * 1000
    
    # 合成+推演
    qichang = compose(direction, density)
    t2 = time.perf_counter()
    result = v94.divine_from_qi(qichang, trace=False)
    t_divine = (time.perf_counter() - t2) * 1000
    
    total = t_dir + t_render + t_divine
    print(f"{q:<30} {direction:>8.1f}°   {t_render:>6.1f}ms   {total:>6.1f}ms  "
          f"→ {result['winner']}")

print()
print(f"  平均方向: <1us级别 -> 即刻")
print(f"  平均渲染: —— 见上表")
print()

# ----- 测试2：异步 vs 同步一致性 -----
print("【测试2】异步 vs 同步 一致性")
print("-" * 65)

for q, cat in questions:
    # 异步流程
    pipeline.start(q)
    t_wait0 = time.perf_counter()
    if not pipeline.ready():
        pipeline._future.join()
    t_wait = (time.perf_counter() - t_wait0) * 1000
    async_result, _, _ = pipeline.get_result()
    
    # 同步流程
    d = codepoint_direction_v1(q)
    di = bitmap_render_fine(q)
    sync_qichang = compose(d, di)
    sync_result = v94.divine_from_qi(sync_qichang, trace=False)
    
    match = "OK" if async_result['winner'] == sync_result['winner'] else "DIFF"
    print(f"  {q:<30} 异步={async_result['winner']} 同步={sync_result['winner']} {match}"
          f"  位图等待={t_wait:.1f}ms")

print()

# ----- 测试3：多条管线并行 -----
print("【测试3】多条管线并行启动")
print("-" * 65)

t_start = time.perf_counter()

pipelines = []
for q, cat in questions:
    p = AsyncPipeline(V94QichangEnhanced())
    p.start(q)
    pipelines.append((q, cat, p))

# 收集结果
results = []
for q, cat, p in pipelines:
    result, direction, density = p.get_result()
    results.append((q, cat, result['winner']))
    print(f"  {q:<30} → {result['winner']}  "
          f"方向={direction:>6.1f}°  密度={density['density']:.4f}")

t_total = (time.perf_counter() - t_start) * 1000
print(f"\n  5条管线并行总耗时: {t_total:.1f}ms")

# 多样性
winners = [r[2] for r in results]
counts = Counter(winners)
print(f"  卦象分布: {dict(counts)}  ({len(counts)}/8)")

print()
print("=" * 65)
print("异步管线测试完成")
print("=" * 65)
