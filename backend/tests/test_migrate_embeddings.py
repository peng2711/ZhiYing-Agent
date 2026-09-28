import chromadb
import pytest

from tests_support_embedding import BigramHashEmbedding, ConstantEmbedding
from tooling import embeddings
from tooling.knowledge_base import KnowledgeBase
from tooling.migrate_embeddings import migrate_all, migrate_collection


def _client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))


def test_migration_copies_text_and_metadata_and_reembeds(tmp_path):
    client = _client(tmp_path)
    legacy = client.get_or_create_collection("knowledge_base", embedding_function=ConstantEmbedding())
    legacy.add(ids=["a", "b"], documents=["退款政策正文", "发票规则正文"],
               metadatas=[{"source_id": "refund"}, {"source_id": "invoice"}])
    target = client.get_or_create_collection("knowledge_base__new", embedding_function=BigramHashEmbedding())

    assert migrate_collection(client, "knowledge_base", target, batch_size=1) == 2
    rows = target.get(include=["documents", "metadatas", "embeddings"])
    assert sorted(rows["ids"]) == ["a", "b"]
    assert {m["source_id"] for m in rows["metadatas"]} == {"refund", "invoice"}
    # 向量由目标模型重新计算，而不是沿用旧模型的
    assert len(rows["embeddings"][0]) == 64
    # 旧 collection 保持原样，便于回滚
    assert legacy.count() == 2


def test_migration_is_skipped_when_target_already_has_data(tmp_path):
    client = _client(tmp_path)
    client.get_or_create_collection("src", embedding_function=ConstantEmbedding()).add(ids=["a"], documents=["旧"])
    target = client.get_or_create_collection("dst", embedding_function=BigramHashEmbedding())
    target.add(ids=["x"], documents=["新模型已经在用的数据"])

    assert migrate_collection(client, "src", target) == 0
    assert migrate_collection(client, "missing", target) == 0
    assert target.get()["ids"] == ["x"]


def test_knowledge_base_migrates_instead_of_loading_demo_docs(tmp_path):
    client = _client(tmp_path)
    client.get_or_create_collection("knowledge_base", embedding_function=ConstantEmbedding()).add(
        ids=["u1"], documents=["用户自己导入的会员积分规则"],
        metadatas=[{"source_id": "points", "title": "积分规则", "status": "active",
                    "effective_from": "2025-01-01", "version": "1.0"}],
    )
    kb = KnowledgeBase(
        client=client, embedding_function=BigramHashEmbedding(),
        collection_name="knowledge_base__bge-small-zh", migrate_from_collection="knowledge_base",
    )
    assert kb.doc_count == 1
    assert kb.search("积分规则", top_k=1)[0]["source_id"] == "points"


def test_memory_collections_are_migrated_with_the_configured_model(tmp_path, monkeypatch):
    from memory.conversation_memory import MemoryManager

    monkeypatch.setattr(embeddings, "get_embedding_function", lambda name, cache_dir=None: BigramHashEmbedding())
    client = _client(tmp_path)
    client.get_or_create_collection("episodic", embedding_function=ConstantEmbedding()).add(
        ids=["e1"], documents=["用户上次问过退款进度"], metadatas=[{"user_id": "u1"}],
    )

    collection = MemoryManager._open_collection(client, "episodic", "bge-small-zh")

    assert collection.name == "episodic__bge-small-zh"
    assert collection.get()["ids"] == ["e1"]


def test_migrate_all_covers_knowledge_and_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedding_function", lambda name, cache_dir=None: BigramHashEmbedding())
    client = _client(tmp_path)
    for base in ("knowledge_base", "episodic", "user_profile"):
        client.get_or_create_collection(base, embedding_function=ConstantEmbedding()).add(ids=[base], documents=[base])

    assert migrate_all(client, "default", "bge-small-zh") == {"knowledge_base": 1, "episodic": 1, "user_profile": 1}
    assert migrate_all(client, "default", "bge-small-zh") == {"knowledge_base": 0, "episodic": 0, "user_profile": 0}


def test_default_embedding_is_the_chinese_model(monkeypatch):
    monkeypatch.delenv("ZHIYING_EMBEDDING_MODEL", raising=False)
    assert embeddings.configured_embedding() == "bge-small-zh"
    assert embeddings.legacy_collection_for("bge-small-zh") == "knowledge_base"
    assert embeddings.legacy_collection_for("default") is None
    monkeypatch.setenv("ZHIYING_EMBEDDING_MODEL", "text-embedding-3")
    with pytest.raises(ValueError):
        embeddings.configured_embedding()
