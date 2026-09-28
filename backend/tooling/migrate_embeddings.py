"""
把 ChromaDB collection 的数据迁移到另一个 embedding 模型的 collection。

只复制文本和元数据，向量由目标 collection 的 embedding function 重新计算；ID 保持不变，
所以重复执行是幂等的。源 collection 不做任何修改，回滚时把 ZHIYING_EMBEDDING_MODEL 改回即可。

服务启动时会自动迁移（目标为空且旧 collection 有数据时），也可以提前手动执行：
  python -m tooling.migrate_embeddings --from default --to bge-small-zh
"""
import argparse
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 知识库、情景记忆、用户画像
COLLECTION_BASES = ("knowledge_base", "episodic", "user_profile")


def migrate_collection(client: Any, source_name: str, target: Any, batch_size: int = 100) -> int:
    """把 source_name 的全部记录写入 target collection，返回复制的条数。

    target 已有数据时跳过：说明已经迁移过，或者新模型已经在使用，不能覆盖。
    """
    if source_name == target.name:
        return 0
    try:
        source = client.get_collection(source_name)
    except Exception:
        return 0
    total = source.count()
    if total == 0 or target.count() > 0:
        return 0

    copied = 0
    while copied < total:
        rows = source.get(include=["documents", "metadatas"], limit=batch_size, offset=copied)
        ids = rows.get("ids") or []
        if not ids:
            break
        target.upsert(ids=ids, documents=rows["documents"], metadatas=rows["metadatas"])
        copied += len(ids)
    logger.info("已将 %s 的 %d 条记录迁移到 %s（重新计算向量）", source_name, copied, target.name)
    return copied


def migrate_all(client: Any, from_model: str, to_model: str) -> Dict[str, int]:
    from tooling.embeddings import collection_name_for, embedding_cache_dir, get_embedding_function

    embedding_function = get_embedding_function(to_model, embedding_cache_dir())
    kwargs = {"embedding_function": embedding_function} if embedding_function is not None else {}
    report: Dict[str, int] = {}
    for base in COLLECTION_BASES:
        source_name = collection_name_for(from_model, base)
        try:
            source_meta = client.get_collection(source_name).metadata
        except Exception:
            report[base] = 0
            continue
        target = client.get_or_create_collection(
            collection_name_for(to_model, base), metadata=source_meta or None, **kwargs,
        )
        report[base] = migrate_collection(client, source_name, target)
    return report


def _client_from_env() -> Any:
    import chromadb

    settings = chromadb.Settings(anonymized_telemetry=False)
    host = os.getenv("CHROMA_HOST", "localhost").strip() or "localhost"
    port = int(os.getenv("CHROMA_PORT", "8001"))
    try:
        client = chromadb.HttpClient(host=host, port=port, settings=settings)
        client.heartbeat()
        return client
    except Exception:
        root = pathlib.Path(__file__).resolve().parent.parent
        path = os.getenv("CHROMA_PERSIST_DIRECTORY", str(root / "data" / "chroma"))
        return chromadb.PersistentClient(path=path, settings=settings)


def main(argv: Optional[List[str]] = None) -> None:
    from tooling.embeddings import SUPPORTED_EMBEDDINGS

    parser = argparse.ArgumentParser(description="迁移 ChromaDB 数据到新的 embedding 模型")
    parser.add_argument("--from", dest="from_model", default="default", choices=SUPPORTED_EMBEDDINGS)
    parser.add_argument("--to", dest="to_model", default="bge-small-zh", choices=SUPPORTED_EMBEDDINGS)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

    report = migrate_all(_client_from_env(), args.from_model, args.to_model)
    for base, count in report.items():
        print(f"{base}: {count} 条")


if __name__ == "__main__":
    main()
