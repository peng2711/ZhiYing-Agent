"""测试用的确定性 embedding，替代需要下载的真实模型。"""
import hashlib

from chromadb import Documents, EmbeddingFunction, Embeddings


class BigramHashEmbedding(EmbeddingFunction[Documents]):
    """字符二元组哈希向量（64 维），相似文本得到相近向量。"""

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = [0.0] * 64
            for i in range(len(text) - 1):
                vec[int(hashlib.md5(text[i:i + 2].encode()).hexdigest(), 16) % 64] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class ConstantEmbedding(EmbeddingFunction[Documents]):
    """模拟旧模型：8 维常量向量，维度与新模型不同。"""

    def __call__(self, input: Documents) -> Embeddings:
        return [[1.0] + [0.0] * 7 for _ in input]
