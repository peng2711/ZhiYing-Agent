from tooling.knowledge_base import KnowledgeBase


class FakeCollection:
    def __init__(self):
        self.rows = {}

    @staticmethod
    def _matches(meta, where):
        if not where:
            return True
        if "$and" in where:
            return all(FakeCollection._matches(meta, item) for item in where["$and"])
        return all(meta.get(key) == value for key, value in where.items())

    def count(self):
        return len(self.rows)

    def get(self, where=None, include=None):
        rows = [(item_id, row) for item_id, row in self.rows.items()
                if self._matches(row["meta"], where)]
        return {
            "ids": [item_id for item_id, _ in rows],
            "metadatas": [row["meta"] for _, row in rows],
        }

    def delete(self, where=None, ids=None):
        if ids is not None:
            for item_id in ids:
                self.rows.pop(item_id, None)
            return
        for item_id in list(self.rows):
            if self._matches(self.rows[item_id]["meta"], where):
                del self.rows[item_id]

    def update(self, ids, metadatas):
        for item_id, meta in zip(ids, metadatas):
            self.rows[item_id]["meta"] = dict(meta)

    def upsert(self, ids, documents, metadatas):
        for item_id, document, meta in zip(ids, documents, metadatas):
            self.rows[item_id] = {"doc": document, "meta": dict(meta)}

    def query(self, query_texts, n_results, where=None):
        rows = [row for row in self.rows.values() if self._matches(row["meta"], where)][:n_results]
        return {
            "documents": [[row["doc"] for row in rows]],
            "metadatas": [[row["meta"] for row in rows]],
            "distances": [[0.1 for _ in rows]],
        }


def make_kb():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = FakeCollection()
    kb._distance_space = "cosine"
    return kb


def test_activating_new_version_expires_old_and_searches_only_current():
    kb = make_kb()
    kb.add_documents([{
        "source_id": "refund-policy", "title": "退款政策", "version": "1.0",
        "effective_from": "2020-01-01", "content": "支持七天退款。",
    }])
    kb.add_documents([{
        "source_id": "refund-policy", "title": "退款政策", "version": "2.0",
        "effective_from": "2021-01-01", "content": "支持十五天退款。",
    }])

    versions = {item["version"]: item for item in kb.list_versions("refund-policy")}
    assert versions["1.0"]["status"] == "expired"
    assert versions["1.0"]["effective_to"] == "2020-12-31"
    assert versions["2.0"]["status"] == "active"
    results = kb.search("退款", top_k=5)
    assert {item["version"] for item in results} == {"2.0"}
    assert results[0]["content"] == "支持十五天退款。"


def test_old_version_can_be_reactivated_for_rollback():
    kb = make_kb()
    for version, content in (("1.0", "七天"), ("2.0", "十五天")):
        kb.add_documents([{
            "source_id": "refund-policy", "title": "退款政策", "version": version,
            "effective_from": "2020-01-01", "content": content,
        }])

    activated = kb.set_version_status("refund-policy", "1.0", "active", "2020-01-01")
    assert activated["status"] == "active"
    assert {item["version"] for item in kb.search("退款", top_k=5)} == {"1.0"}


def test_legacy_chunks_receive_lifecycle_defaults():
    kb = make_kb()
    kb._collection.rows["old"] = {
        "doc": "旧内容",
        "meta": {"source_id": "legacy", "version": "1.0", "title": "旧政策"},
    }
    kb._migrate_legacy_metadata()
    meta = kb._collection.rows["old"]["meta"]
    assert meta["status"] == "active"
    assert meta["effective_from"] == "1970-01-01"
    assert meta["effective_to"] == ""


def test_delete_version_only_removes_exact_version():
    kb = make_kb()
    for version in ("1.0", "2.0"):
        kb.add_documents([{
            "source_id": "refund-policy", "title": "退款政策", "version": version,
            "effective_from": "2020-01-01", "content": version,
        }])
    assert kb.delete_version("refund-policy", "1.0") == 1
    assert {item["version"] for item in kb.list_versions("refund-policy")} == {"2.0"}


def _open_kb(path, monkeypatch):
    # 端口 1 无服务，走本地持久化模式；跳过默认文档导入以免下载 embedding 模型。
    monkeypatch.setattr(KnowledgeBase, "_load_default_docs", lambda self: None)
    return KnowledgeBase(chroma_host="127.0.0.1", chroma_port=1, chroma_path=str(path))


def test_new_collection_uses_cosine_distance(tmp_path, monkeypatch):
    kb = _open_kb(tmp_path, monkeypatch)

    assert kb._distance_space == "cosine"
    assert kb._similarity(0.25) == 0.75


def test_legacy_l2_collection_scores_are_converted_to_cosine(tmp_path, monkeypatch):
    import chromadb

    legacy = chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.Settings(anonymized_telemetry=False))
    legacy.get_or_create_collection(KnowledgeBase.COLLECTION_NAME, metadata={"description": "旧版本"})

    kb = _open_kb(tmp_path, monkeypatch)

    assert kb._distance_space == "l2"
    # 归一化向量的平方欧氏距离 d = 2 - 2cos：cos=0.9 时 d=0.2。
    assert abs(kb._similarity(0.2) - 0.9) < 1e-9
