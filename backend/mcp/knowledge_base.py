"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "ZhiYing Agent RAG 知识库"},
        )
        self._migrate_legacy_metadata()

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            # source_id 用于文档更新时清理旧切片；未提供时按标题生成稳定键。
            source_id = str(doc.get("source_id") or hashlib.sha256(title.encode("utf-8")).hexdigest()[:16])
            document_name = str(doc.get("document_name") or title)
            version = str(doc.get("version") or "1.0")
            updated_at = str(doc.get("updated_at") or datetime.now(timezone.utc).date().isoformat())
            section = str(doc.get("section") or title)
            status = str(doc.get("status") or "active").lower()
            if status not in {"active", "expired"}:
                raise ValueError("知识库版本状态只能是 active 或 expired")
            effective_from = self._normalize_date(
                doc.get("effective_from"), default=date.today().isoformat()
            )
            effective_to = self._normalize_date(doc.get("effective_to"), default="")
            if effective_to and effective_to < effective_from:
                raise ValueError("effective_to 不能早于 effective_from")
            if status == "active" and effective_from > date.today().isoformat():
                raise ValueError("active 版本的 effective_from 不能晚于今天")
            chunks  = self._chunk_text(content, chunk_size=500)

            # 只覆盖同一来源的同一版本；其他版本保留，便于审计和回滚。
            try:
                self._collection.delete(where={"$and": [
                    {"source_id": source_id}, {"version": version},
                ]})
            except Exception:
                # 兼容历史 collection（旧切片没有 source_id 元数据）。
                logger.debug("知识库旧版本清理跳过: source_id=%s", source_id, exc_info=True)

            for i, chunk in enumerate(chunks):
                # 使用完整 chunk 生成稳定 ID，重复导入相同文档时幂等。
                doc_id = hashlib.sha256(
                    f"{source_id}\0{version}\0{title}\0{i}\0{chunk}".encode("utf-8")
                ).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({
                    "title": title,
                    "source_id": source_id,
                    "document_name": document_name,
                    "version": version,
                    "updated_at": updated_at,
                    "section": section,
                    "status": status,
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })

            if status == "active":
                self._expire_other_versions(source_id, version, effective_from)

        if ids:
            # ChromaDB 会自动生成 Embedding
            # upsert 兼容文档重复导入和内容更新；老版本客户端没有 upsert 时回退到 add。
            upsert = getattr(self._collection, "upsert", None)
            if upsert is not None:
                upsert(ids=ids, documents=docs, metadatas=metas)
            else:
                self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        # 多取一些候选，日期过滤和同源版本选择后仍尽量满足 top_k。
        candidate_k = min(max(top_k * 4, 20), max(self._collection.count(), 1))
        results = self._collection.query(
            query_texts=[query],
            n_results=candidate_k,
            where={"status": "active"},
        )

        candidates = []
        today = date.today().isoformat()
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                effective_from = str(meta.get("effective_from", "1970-01-01"))
                effective_to = str(meta.get("effective_to", ""))
                if effective_from > today or (effective_to and effective_to < today):
                    continue
                candidates.append({
                    "title":    meta.get("title", ""),
                    "document_name": meta.get("document_name", meta.get("title", "")),
                    "source_id": meta.get("source_id", ""),
                    "version": meta.get("version", "1.0"),
                    "updated_at": meta.get("updated_at", ""),
                    "section": meta.get("section", meta.get("title", "")),
                    "status": meta.get("status", "active"),
                    "effective_from": effective_from,
                    "effective_to": effective_to,
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        # 若历史数据存在重叠 active 版本，每个 source 只保留生效日期最新的一版。
        selected_versions: Dict[str, tuple[str, str]] = {}
        for item in candidates:
            source_id = item["source_id"]
            candidate = (item["effective_from"], item["version"])
            if source_id not in selected_versions or candidate > selected_versions[source_id]:
                selected_versions[source_id] = candidate
        return [
            item for item in candidates
            if (item["effective_from"], item["version"]) == selected_versions[item["source_id"]]
        ][:top_k]

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    def list_versions(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出文档版本；每个版本只返回一条摘要，不返回重复 chunk。"""
        kwargs: Dict[str, Any] = {"include": ["metadatas"]}
        if source_id:
            kwargs["where"] = {"source_id": source_id}
        result = self._collection.get(**kwargs)
        versions: Dict[tuple[str, str], Dict[str, Any]] = {}
        for meta in result.get("metadatas") or []:
            key = (str(meta.get("source_id", "")), str(meta.get("version", "1.0")))
            item = versions.setdefault(key, {
                "source_id": key[0], "version": key[1],
                "document_name": meta.get("document_name", meta.get("title", "")),
                "status": meta.get("status", "active"),
                "effective_from": meta.get("effective_from", "1970-01-01"),
                "effective_to": meta.get("effective_to", ""),
                "updated_at": meta.get("updated_at", ""), "chunks": 0,
            })
            item["chunks"] += 1
        return sorted(versions.values(), key=lambda item: (
            item["source_id"], item["effective_from"], item["version"]
        ), reverse=True)

    async def list_versions_async(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.list_versions, source_id)

    def set_version_status(
        self, source_id: str, version: str, status: str, effective_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """激活或停用一个版本；激活时自动停用同源的其他版本。"""
        status = status.lower()
        if status not in {"active", "expired"}:
            raise ValueError("知识库版本状态只能是 active 或 expired")
        result = self._collection.get(
            where={"$and": [{"source_id": source_id}, {"version": version}]},
            include=["metadatas"],
        )
        ids, metas = list(result.get("ids") or []), list(result.get("metadatas") or [])
        if not ids:
            raise KeyError(f"未找到知识库版本: {source_id}@{version}")
        active_from = self._normalize_date(effective_from, default=date.today().isoformat())
        if status == "active" and active_from > date.today().isoformat():
            raise ValueError("active 版本的 effective_from 不能晚于今天")
        if status == "active":
            self._expire_other_versions(source_id, version, active_from)
        updated = []
        for meta in metas:
            next_meta = dict(meta)
            next_meta["status"] = status
            if status == "active":
                next_meta["effective_from"] = active_from
                next_meta["effective_to"] = ""
            else:
                next_meta["effective_to"] = date.today().isoformat()
            updated.append(next_meta)
        self._collection.update(ids=ids, metadatas=updated)
        versions = self.list_versions(source_id=source_id)
        return next(item for item in versions if item["version"] == version)

    async def set_version_status_async(
        self, source_id: str, version: str, status: str, effective_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.set_version_status, source_id, version, status, effective_from
        )

    def delete_version(self, source_id: str, version: str) -> int:
        """精确删除一个来源的一个版本，返回删除的 chunk 数。"""
        result = self._collection.get(
            where={"$and": [{"source_id": source_id}, {"version": version}]},
            include=["metadatas"],
        )
        ids = list(result.get("ids") or [])
        if not ids:
            raise KeyError(f"未找到知识库版本: {source_id}@{version}")
        self._collection.delete(ids=ids)
        return len(ids)

    async def delete_version_async(self, source_id: str, version: str) -> int:
        return await asyncio.to_thread(self.delete_version, source_id, version)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_date(value: Any, default: str) -> str:
        text = str(value or default)
        if not text:
            return ""
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise ValueError(f"无效日期 {text}，应使用 YYYY-MM-DD") from exc

    def _expire_other_versions(self, source_id: str, version: str, active_from: str) -> None:
        result = self._collection.get(where={"source_id": source_id}, include=["metadatas"])
        ids_to_update, metas_to_update = [], []
        expiry = (date.fromisoformat(active_from) - timedelta(days=1)).isoformat()
        for item_id, meta in zip(result.get("ids") or [], result.get("metadatas") or []):
            if str(meta.get("version", "1.0")) == version or meta.get("status", "active") != "active":
                continue
            next_meta = dict(meta)
            next_meta["status"] = "expired"
            next_meta["effective_to"] = expiry
            ids_to_update.append(item_id)
            metas_to_update.append(next_meta)
        if ids_to_update:
            self._collection.update(ids=ids_to_update, metadatas=metas_to_update)

    def _migrate_legacy_metadata(self) -> None:
        """为升级前没有版本生命周期字段的 chunk 补默认值。"""
        result = self._collection.get(include=["metadatas"])
        ids_to_update, metas_to_update = [], []
        for item_id, meta in zip(result.get("ids") or [], result.get("metadatas") or []):
            if all(key in meta for key in ("status", "effective_from", "effective_to")):
                continue
            next_meta = dict(meta)
            next_meta.setdefault("status", "active")
            next_meta.setdefault("effective_from", "1970-01-01")
            next_meta.setdefault("effective_to", "")
            ids_to_update.append(item_id)
            metas_to_update.append(next_meta)
        if ids_to_update:
            self._collection.update(ids=ids_to_update, metadatas=metas_to_update)
            logger.info("已迁移 %s 个历史知识片段的版本元数据", len(ids_to_update))

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        text = str(text or "")
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                # 单句超长时继续硬切，保证 chunk 上限真实生效。
                while len(sent) > chunk_size:
                    chunks.append(sent[:chunk_size])
                    sent = sent[chunk_size:]
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "电子发票开具规则",
                "content": (
                    "订单完成支付后可以申请电子发票。"
                    "申请发票时需要提供订单号、发票抬头和纳税人识别号。"
                    "发票申请提交后，系统通常会在 1-3 个工作日内开具并发送到用户邮箱。"
                    "已经退款的订单不能继续申请该订单的发票。"
                    "需要修改发票信息时，请在开票前联系支持人员处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
