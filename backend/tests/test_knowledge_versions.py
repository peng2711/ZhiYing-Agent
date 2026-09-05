from mcp.knowledge_base import KnowledgeBase


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
