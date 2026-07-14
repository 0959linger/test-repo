"""
卦→词 蒸馏器

从模型的 embedding 空间提取64卦各自对应的词。
不是手写词典——是模型"认为"这个卦该说什么。

原理：
  1. 对每个卦构造"理想 qi 分布"（纯卦模板）
  2. 在模型 embedding 空间中，找到这个分布的语义位置
  3. 查询最近邻词 = 模型的卦→词映射

用法：
  distiller = GuaWordDistiller(model_name="Qwen/Qwen2.5-0.5B")
  bagua_dict = distiller.distill(top_k=30)
  # bagua_dict['巽'] = ['gentle', 'wind', 'soft', ...]
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


BAGUA = ['乾', '兑', '离', '震', '巽', '坎', '艮', '坤']
BAGUA_ORDER = BAGUA  # 保持与 engine 一致


class GuaTemplateGenerator:
    """生成64卦的"理想 qi 分布"——纯卦模板"""
    
    @staticmethod
    def pure_gua(gua_idx: int) -> np.ndarray:
        """单卦模板：该卦=1，其余接近0"""
        qi = np.zeros(8)
        qi[gua_idx] = 1.0
        # 给它一点"扩散"——相邻卦也有微弱激活
        # 先天八卦环：乾兑离震巽坎艮坤
        neighbors = [(gua_idx - 1) % 8, (gua_idx + 1) % 8]
        for n in neighbors:
            qi[n] = 0.15
        qi = qi / qi.sum()
        return qi
    
    @staticmethod
    def composite_gua64(idx: int) -> np.ndarray:
        """64卦模板：上卦+下卦的加权混合"""
        shang_idx = idx // 8   # 上卦
        xia_idx = idx % 8      # 下卦
        
        qi = np.zeros(8)
        qi[shang_idx] = 0.55  # 上卦主导
        qi[xia_idx] = 0.35    # 下卦从属
        # 微弱扩散
        for offset in [-1, 1]:
            qi[(shang_idx + offset) % 8] += 0.03
            qi[(xia_idx + offset) % 8] += 0.02
        qi = qi / qi.sum()
        return qi


class GuaWordDistiller:
    """
    从模型 embedding 空间蒸馏卦→词映射。
    
    每个卦的理想 qi 分布 → embedding 向量 → 最近邻词。
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B",
                 device: str = None):
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self.embed_matrix = None      # [vocab_size, hidden_dim]
        self.vocab = []
        self.gua_to_qi = {}           # {卦名: qi分布}
        self.gua_to_words = {}        # {卦名: [(词, 相似度), ...]}
        
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    def load_model(self):
        """加载模型，提取 embedding 表"""
        print(f"  加载 {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)
        
        # 提取 embedding 权重矩阵
        embed = self.model.get_input_embeddings()
        self.embed_matrix = embed.weight.detach().float().cpu().numpy()  # [vocab, dim]
        
        # 构建词表
        self.vocab = [self.tokenizer.decode([i]) for i in range(len(self.embed_matrix))]
        
        print(f"  词表大小: {len(self.vocab)}, embedding维数: {self.embed_matrix.shape[1]}")
    
    # ============================================================
    # 打窝蒸馏：用上下文激活的 hidden state
    # ============================================================
    
    BAIT_TEMPLATES = {
        '乾': '乾为天，代表刚健、创造、力量、天行健君子以自强不息。',
        '兑': '兑为泽，代表喜悦、交流、流动、外显，像湖面的涟漪扩散。',
        '离': '离为火，代表光明、温暖、依附、跳动，火焰照亮四周。',
        '震': '震为雷，代表震动、惊雷、撼动、轰鸣，春雷唤醒万物。',
        '巽': '巽为风，代表顺从、渗透、柔软、飘散，如春风潜入万物。',
        '坎': '坎为水，代表险陷、深渊、暗流、漩涡，水往低处沉入。',
        '艮': '艮为山，代表停止、静止、沉重、凝固，如山般不可动摇。',
        '坤': '坤为地，代表柔顺、包容、厚重、承载，大地承载万物。',
    }
    
    # 八卦"打窝"关键词——从知识描述中提取的描述词
    BAIT_KEYWORDS = {
        '乾': ['天', '刚健', '创造', '力量', '自强不息'],
        '兑': ['泽', '喜悦', '交流', '流动', '外显', '扩散'],
        '离': ['火', '光明', '温暖', '跳动', '照亮'],
        '震': ['雷', '震动', '惊雷', '撼动', '轰鸣', '唤醒'],
        '巽': ['风', '顺从', '渗透', '柔软', '飘散', '潜入'],
        '坎': ['水', '险陷', '深渊', '暗流', '漩涡', '沉入'],
        '艮': ['山', '停止', '静止', '沉重', '凝固', '坚固'],
        '坤': ['地', '柔顺', '包容', '厚重', '承载', '大地'],
    }
    
    def bait_hidden_state(self, gua: str) -> np.ndarray:
        """
        打窝蒸馏 v4：词级蒸馏。
        
        不打整句hidden state——直接从打窝关键词中取 embedding 均值。
        "巽为风，顺从柔软" → 取"风""顺从""柔软""渗透""飘散"的 embedding
        → 均值 = 模型对"巽"的理解位置。
        
        不需要跑模型（纯查表），embedding 几何天然对齐。
        """
        keywords = self.BAIT_KEYWORDS.get(gua, [gua])
        
        vec = np.zeros(self.embed_matrix.shape[1])
        count = 0
        for kw in keywords:
            tok_ids = self.tokenizer.encode(kw, add_special_tokens=False)
            for tid in tok_ids:
                if tid < len(self.embed_matrix):
                    vec += self.embed_matrix[tid]
                    count += 1
        
        if count > 0:
            vec = vec / count
        
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        return vec
    
    def qi_to_embedding(self, qi_dist: np.ndarray) -> np.ndarray:
        """
        qi分布 → 模型 embedding 向量。
        
        方法：用 BAGUA 中文字作为 token，按 qi 分布加权平均。
        乾→'乾'的embedding, 兑→'兑'的embedding, ...
        
        这样 qi 向量在模型空间中有一个明确的位置。
        """
        tokens = [self.tokenizer.encode(g, add_special_tokens=False) for g in BAGUA_ORDER]
        # 每个卦可能被tokenizer切成多个token，取第一个的embedding
        embs = []
        for t in tokens:
            if len(t) > 0:
                embs.append(self.embed_matrix[t[0]])
            else:
                embs.append(np.zeros(self.embed_matrix.shape[1]))
        
        embs = np.array(embs)
        qi_embedding = np.dot(qi_dist, embs)  # 加权平均
        
        # 归一化
        norm = np.linalg.norm(qi_embedding)
        if norm > 1e-10:
            qi_embedding = qi_embedding / norm
        
        return qi_embedding
    
    def nearest_words(self, qi_embedding: np.ndarray, top_k: int = 30,
                      skip_special: bool = True) -> List[Tuple[str, float]]:
        """找 embedding 空间中最近的词"""
        sims = cosine_similarity([qi_embedding], self.embed_matrix)[0]
        
        # 排序
        sorted_idx = np.argsort(sims)[::-1]
        
        results = []
        for idx in sorted_idx:
            if skip_special:
                word = self.vocab[idx]
                # 跳过特殊token和空字符
                if word.strip() == '' or word.startswith('<'):
                    continue
                if len(word) <= 1 and not word.isalpha() and not self._is_chinese(word):
                    continue
            results.append((self.vocab[idx], float(sims[idx])))
            if len(results) >= top_k:
                break
        
        return results
    
    @staticmethod
    def _is_chinese(s: str) -> bool:
        return any('\u4e00' <= c <= '\u9fff' for c in s)
    
    def distill(self, mode: str = "gua8", top_k: int = 30) -> Dict[str, List[Tuple[str, float]]]:
        """
        蒸馏：对每个卦跑一次，记录模型空间最近邻词。
        
        Args:
            mode: "gua8"=八卦, "gua64"=六十四卦
            top_k: 每卦返回多少词
        
        Returns:
            {卦名: [(词, 相似度), ...]}
        """
        if self.model is None:
            self.load_model()
        
        if mode == "gua8":
            gua_list = BAGUA
            template_fn = GuaTemplateGenerator.pure_gua
        else:
            gua_list = [f"{BAGUA[i//8]}{BAGUA[i%8]}" for i in range(64)]
            template_fn = GuaTemplateGenerator.composite_gua64
        
        result = {}
        
        for i, gua in enumerate(gua_list):
            qi = template_fn(i)
            qi_emb = self.qi_to_embedding(qi)
            words = self.nearest_words(qi_emb, top_k=top_k)
            result[gua] = words
            
            if (i + 1) % 8 == 0:
                print(f"  已蒸馏 {i+1}/{len(gua_list)} 卦")
        
        self.gua_to_words = result
        return result
    
    def distill_bait(self, top_k: int = 30) -> Dict[str, List[Tuple[str, float]]]:
        """
        打窝蒸馏：用上下文激活的 hidden state 替代静态 embedding。
        
        对每个八卦字，先注入知识上下文，再取 hidden state 查最近邻词。
        """
        if self.model is None:
            self.load_model()
        
        result = {}
        for i, gua in enumerate(BAGUA):
            hidden_vec = self.bait_hidden_state(gua)
            words = self.nearest_words(hidden_vec, top_k=top_k)
            result[gua] = words
            print(f"  打窝蒸馏 {gua}: {words[0][0]}({words[0][1]:.3f}) {words[1][0]}({words[1][1]:.3f}) ...")
        
        self.gua_to_words = result
        return result
    
    def print_summary(self, top_n: int = 5):
        """打印蒸馏结果摘要"""
        if not self.gua_to_words:
            print("  尚未蒸馏。先调用 distill()。")
            return
        
        print(f"\n{'='*60}")
        print(f"卦→词 蒸馏结果 ({self.model_name})")
        print(f"{'='*60}")
        
        for gua, words in self.gua_to_words.items():
            top = words[:top_n]
            line = "  ".join([f"{w}({s:.3f})" for w, s in top])
            print(f"  {gua:<4} → {line}")
    
    def save(self, filepath: str):
        """保存蒸馏结果到文件"""
        import json
        # 转为可序列化格式
        data = {
            'model_name': self.model_name,
            'gua_to_words': {gua: [(w, float(s)) for w, s in words]
                             for gua, words in self.gua_to_words.items()},
            'vocab_size': len(self.vocab),
            'embed_dim': int(self.embed_matrix.shape[1]) if self.embed_matrix is not None else 0,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GuaWordDistiller':
        """从文件加载蒸馏结果（不需要重新跑模型）"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        d = cls(model_name=data['model_name'])
        d.gua_to_words = data['gua_to_words']
        # 不需要加载模型（节省显存）
        return d


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("=== 卦→词 蒸馏器 ===")
    
    # 使用本地缓存的 Qwen2.5-0.5B-Instruct（sentencepiece，中文不裂字）
    local_path = r"C:\Users\ww109\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
    
    distiller = GuaWordDistiller(model_name=local_path)
    
    # 静态蒸馏（对比基线）
    print("\n[静态蒸馏（embedding表）]")
    result_static = distiller.distill(mode="gua8", top_k=30)
    distiller.print_summary(top_n=3)
    
    # 打窝蒸馏
    print("\n[打窝蒸馏（hidden state）]")
    result_bait = distiller.distill_bait(top_k=30)
    distiller.print_summary(top_n=3)
    
    distiller.save("gua8_bait_distilled.json")
    
    # 蒸馏64卦（耗时较长）
    # distiller.distill(mode="gua64", top_k=30)
    # distiller.save("gua64_distilled.json")
