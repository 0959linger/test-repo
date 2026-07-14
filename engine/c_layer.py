"""
╔══════════════════════════════════════════════════════════════════╗
║  c_layer — C层旁路推理                                          ║
║                                                                  ║
║  旁路日记，不碰场。                                              ║
║  看到的是河图6模型对问题的回应（分歧信号），                    ║
║  而非原始问题文本 — 三层稀释链的关键免疫环节。                   ║
║                                                                  ║
║  推理结果通过 memory.remember() 存入记忆层，                     ║
║  与信号文本平权，受同样衰减淘汰。                                ║
║                                                                  ║
║  铁律核验（逐条通过）：                                          ║
║    不筛选 ✓  不特征提取 ✓  不投影 ✓                             ║
║    不匹配 ✓  不量化 ✓  不进蒸汽管道 ✓                           ║
║    不截断插入 ✓  不预设八卦定义 ✓                               ║
╚══════════════════════════════════════════════════════════════════╝
"""
import requests
import json
from typing import List, Dict, Optional

BAGUA = ['乾', '兑', '离', '震', '坤', '艮', '坎', '巽']

# phi3推理模型端口
REASON_PORT = 8084
REASON_URL = f"http://127.0.0.1:{REASON_PORT}/completion"

# 温度配置
REASON_TEMP = 0.7
REASON_MAX_TOKENS = 256


def call_phi3_reason(hetu_texts: List[str],
                     distribution: Dict[str, float],
                     winner: str,
                     timeout: int = 60) -> Optional[str]:
    """
    调用phi3做旁路推理。
    
    输入：河图6段回应 + v94分布 + 主导卦
    输出：推理日记文本（"因为..."）
    """
    if not hetu_texts:
        return None
    
    # 构造prompt：phi3看到的是分歧信号，不是原始问题
    signals_text = '\n'.join(f"[模型{i+1}] {t[:200]}" for i, t in enumerate(hetu_texts[:6]))
    
    # 气场分布
    dist_sorted = sorted(distribution.items(), key=lambda x: -x[1])[:3]
    dist_str = ' '.join(f"{g}:{v:.2f}" for g, v in dist_sorted)
    
    prompt = f"""观察以下六个视角对同一问题的回应：

{signals_text}

这些回应经过物理场处理后，气场分布为：{dist_str}
主导卦位：{winner}

请以日记形式写下这段观察的解读，格式为"因为...。"
只写解读，不重复数据。控制在三句话以内。

因为"""
    
    try:
        resp = requests.post(REASON_URL, json={
            "prompt": prompt,
            "temperature": REASON_TEMP,
            "n_predict": REASON_MAX_TOKENS,
            "stream": False,
        }, timeout=timeout)
        
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("content", "")
            # 清理输出
            content = content.strip().split('\n\n')[0]  # 只取第一段
            content = content.strip()
            if content:
                return f"因为{content}"
        return None
    
    except Exception:
        return None


class CLayer:
    """C层推理器 — 旁路日记"""
    
    def __init__(self, port: int = 8084):
        self.port = port
        self._url = f"http://127.0.0.1:{port}/completion"
        self.diary: List[str] = []
    
    def observe(self, hetu_texts: List[str],
                distribution: Dict[str, float],
                winner: str,
                crystal: List[str] = None,
                timeout: int = 60) -> Optional[str]:
        """
        观察本轮数据，生成推理日记。
        返回推理文本，同时存入self.diary。
        不修改任何架构内部状态——只产出文本。
        """
        # 取前6条（河图标准输出）
        texts = hetu_texts[:6] if hetu_texts else []
        
        if not texts:
            return None
        
        # 构造prompt
        signals_text = '\n'.join(
            f"[模型{i+1}] {t[:200]}" for i, t in enumerate(texts)
        )
        
        dist_sorted = sorted(distribution.items(), key=lambda x: -x[1])[:3]
        dist_str = ' '.join(f"{g}:{v:.2f}" for g, v in dist_sorted)
        
        prompt = f"""观察以下六个视角对同一问题的回应：

{signals_text}

这些回应经过物理场处理后，气场分布为：{dist_str}
主导卦位：{winner}

请以日记形式写下这段观察的解读，格式为"因为...。"
只写解读，不重复数据。控制在三句话以内。

因为"""
        
        try:
            resp = requests.post(self._url, json={
                "prompt": prompt,
                "temperature": REASON_TEMP,
                "n_predict": REASON_MAX_TOKENS,
                "stream": False,
            }, timeout=timeout)
            
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("content", "")
                content = content.strip()
                # 取第一段
                if '\n' in content:
                    content = content.split('\n')[0]
                if content:
                    reasoning = f"因为{content}"
                    self.diary.append(f"[r{len(self.diary)+1}] {reasoning}")
                    return reasoning
            return None
        
        except Exception:
            return None
    
    def last(self) -> Optional[str]:
        """返回最新一条推理日记"""
        return self.diary[-1] if self.diary else None
    
    def all_diary(self) -> List[str]:
        return self.diary.copy()
    
    def reset(self):
        self.diary.clear()
