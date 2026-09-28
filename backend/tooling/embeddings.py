"""
知识库 embedding 模型选择。

默认沿用 ChromaDB 内置的 all-MiniLM-L6-v2（以英文语料训练）。可选的中文模型通过 fastembed
以 ONNX 方式运行，不依赖 PyTorch：
  - bge-small-zh：BAAI/bge-small-zh-v1.5，512 维

不同模型的向量维度和语义空间都不同，不能写进同一个 collection，所以每个模型使用自己的
collection 名；切换模型后需要重新导入知识库。
"""
from typing import Any, List, Optional

from chromadb import Documents, EmbeddingFunction, Embeddings

DEFAULT_EMBEDDING = "default"

_FASTEMBED_MODELS = {
    "bge-small-zh": "BAAI/bge-small-zh-v1.5",
}

SUPPORTED_EMBEDDINGS = (DEFAULT_EMBEDDING, *_FASTEMBED_MODELS)


class FastEmbedFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        from fastembed import TextEmbedding  # 可选依赖，只在选用中文模型时导入

        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def __call__(self, input: Documents) -> Embeddings:
        return [vector.tolist() for vector in self._model.embed(list(input))]


def get_embedding_function(name: str, cache_dir: Optional[str] = None) -> Any:
    """返回 ChromaDB 可用的 embedding function；default 返回 None，表示用 ChromaDB 默认模型。"""
    if name == DEFAULT_EMBEDDING:
        return None
    if name not in _FASTEMBED_MODELS:
        raise ValueError(f"不支持的 embedding 模型 {name!r}，可选: {', '.join(SUPPORTED_EMBEDDINGS)}")
    return FastEmbedFunction(_FASTEMBED_MODELS[name], cache_dir=cache_dir)


def collection_name_for(name: str, base: str = "knowledge_base") -> str:
    return base if name == DEFAULT_EMBEDDING else f"{base}__{name}"


def list_supported() -> List[str]:
    return list(SUPPORTED_EMBEDDINGS)
