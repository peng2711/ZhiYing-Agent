import asyncio
import hashlib

from chromadb import Documents, EmbeddingFunction, Embeddings

from evaluation import retrieval_evaluator as rev
from tooling.embeddings import collection_name_for


class BigramHashEmbedding(EmbeddingFunction[Documents]):
    """确定性的字符二元组哈希向量，测试里替代需要下载的模型。"""

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = [0.0] * 64
            for i in range(len(text) - 1):
                vec[int(hashlib.md5(text[i:i + 2].encode()).hexdigest(), 16) % 64] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


def test_score_query_counts_documents_not_chunks():
    ranked = rev.ranked_sources([
        {"source_id": "a"}, {"source_id": "a"}, {"source_id": "b"}, {"source_id": "c"},
    ])
    assert ranked == ["a", "b", "c"]
    scores = rev.score_query(ranked, {"b", "z"})
    assert scores["hit@1"] == 0.0
    assert scores["hit@3"] == 1.0
    assert scores["recall@3"] == 0.5
    assert scores["mrr"] == 0.5


def test_dataset_labels_point_to_existing_documents():
    documents, cases = rev.load_dataset()
    sources = {doc["source_id"] for doc in documents}
    queries = [case["query"] for case in cases]
    assert len(queries) == len(set(queries))
    for case in cases:
        assert case["relevant"] and set(case["relevant"]) <= sources, case
        assert case.get("category"), case


def test_pipeline_on_isolated_kb_respects_version_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(rev, "get_embedding_function", lambda name, cache_dir=None: BigramHashEmbedding())
    documents = [
        {"source_id": "refund", "title": "退款", "version": "1.0", "status": "expired",
         "effective_from": "2025-01-01", "effective_to": "2025-12-31", "content": "旧规则：七天内可以无理由退款"},
        {"source_id": "refund", "title": "退款", "version": "2.0", "effective_from": "2026-01-01",
         "content": "新规则：十五天内可以无理由退款"},
        {"source_id": "invoice", "title": "发票", "content": "确认收货后可以申请开具电子发票"},
        {"source_id": "login", "title": "登录", "content": "返回 401 表示登录状态已经失效，请重新登录"},
    ]
    kb = rev.build_knowledge_base("default", documents, str(tmp_path))

    cases = [
        {"query": "无理由退款几天", "relevant": ["refund"], "category": "version"},
        {"query": "怎么开电子发票", "relevant": ["invoice"], "category": "paraphrase"},
        {"query": "登录状态失效 401", "relevant": ["login"], "category": "exact_term"},
    ]
    searchers = rev.build_searchers(kb, ["vector"], top_k=5)
    report = asyncio.run(rev.evaluate(searchers["vector"], cases))

    assert report["overall"]["hit@1"] == 1.0
    assert report["misses"] == []
    refund_hits = asyncio.run(kb.search_async("无理由退款几天", top_k=5))
    assert [hit["version"] for hit in refund_hits if hit["source_id"] == "refund"] == ["2.0"]


def test_llm_modes_require_an_api_key(monkeypatch):
    import pytest

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="LLM_API_KEY"):
        rev.build_searchers(object(), ["vector", "rerank"], top_k=5)


def test_each_embedding_model_gets_its_own_collection():
    assert collection_name_for("default") == "knowledge_base"
    assert collection_name_for("bge-small-zh") == "knowledge_base__bge-small-zh"
