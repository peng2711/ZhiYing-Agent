"""
知识库和记忆使用的 embedding 模型。

默认使用 bge-small-zh（BAAI/bge-small-zh-v1.5，512 维），通过 fastembed 以 ONNX 方式运行，
不依赖 PyTorch。检索评测（evaluation/retrieval_evaluator.py）上它的 Hit@3 为 0.977，
而 ChromaDB 内置的 all-MiniLM-L6-v2（以英文语料训练，384 维）只有 0.333。

  - ZHIYING_EMBEDDING_MODEL=bge-small-zh（默认）或 default（ChromaDB 内置模型）
  - ZHIYING_EMBEDDING_CACHE_DIR：模型缓存目录，默认 ~/.cache/fastembed

不同模型的向量维度和语义空间都不同，不能写进同一个 collection：每个模型使用自己的
collection 名（ChromaDB 内置模型沿用原名，其他模型加 __<模型名> 后缀）。
从旧 collection 迁移数据见 tooling/migrate_embeddings.py。
"""
import os
from functools import lru_cache
from typing import Any, List, Optional

from chromadb import Documents, EmbeddingFunction, Embeddings

DEFAULT_EMBEDDING = "default"          # ChromaDB 内置 all-MiniLM-L6-v2
RECOMMENDED_EMBEDDING = "bge-small-zh"

_FASTEMBED_MODELS = {
    "bge-small-zh": "BAAI/bge-small-zh-v1.5",
}

SUPPORTED_EMBEDDINGS = (DEFAULT_EMBEDDING, *_FASTEMBED_MODELS)


class FastEmbedFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def __call__(self, input: Documents) -> Embeddings:
        return [vector.tolist() for vector in self._model.embed(list(input))]


def configured_embedding() -> str:
    name = os.getenv("ZHIYING_EMBEDDING_MODEL", RECOMMENDED_EMBEDDING).strip() or RECOMMENDED_EMBEDDING
    if name not in SUPPORTED_EMBEDDINGS:
        raise ValueError(f"不支持的 embedding 模型 {name!r}，可选: {', '.join(SUPPORTED_EMBEDDINGS)}")
    return name


def embedding_cache_dir() -> str:
    return os.getenv("ZHIYING_EMBEDDING_CACHE_DIR", "").strip() or os.path.expanduser("~/.cache/fastembed")


@lru_cache(maxsize=None)
def get_embedding_function(name: str, cache_dir: Optional[str] = None) -> Any:
    """返回 ChromaDB 可用的 embedding function；default 返回 None，表示用 ChromaDB 内置模型。

    同一模型在进程内只加载一次，知识库和记忆模块共用。
    """
    if name == DEFAULT_EMBEDDING:
        return None
    if name not in _FASTEMBED_MODELS:
        raise ValueError(f"不支持的 embedding 模型 {name!r}，可选: {', '.join(SUPPORTED_EMBEDDINGS)}")
    return FastEmbedFunction(_FASTEMBED_MODELS[name], cache_dir=cache_dir)


def configured_embedding_function() -> Any:
    return get_embedding_function(configured_embedding(), embedding_cache_dir())


def collection_name_for(name: str, base: str = "knowledge_base") -> str:
    return base if name == DEFAULT_EMBEDDING else f"{base}__{name}"


def legacy_collection_for(name: str, base: str = "knowledge_base") -> Optional[str]:
    """切换到非默认模型时，旧数据所在的 collection（ChromaDB 内置模型时代的原名）。"""
    return None if name == DEFAULT_EMBEDDING else base


def list_supported() -> List[str]:
    return list(SUPPORTED_EMBEDDINGS)
