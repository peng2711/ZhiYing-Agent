"""
ZhiYing Agent 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import hashlib
import hmac
import logging
import os
import pathlib
import re
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request as FastAPIRequest, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║  ZhiYing Agent v1.0  ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_business_backend = None

def _llm_cfg() -> Dict[str, Any]:
    key = os.getenv("LLM_API_KEY", "")
    if key.strip().lower() in {"", "xxx", "your_api_key", "your-key", "replace-with-your-key"}:
        raise RuntimeError("未设置有效的 LLM_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("LLM_MODEL", "deepseek-v4-pro").strip(),
    }
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


def _chroma_cfg() -> Dict[str, Any]:
    """本地默认连接 8001 并回退仓库数据目录；Compose 会显式覆盖。"""
    return {
        "host": os.getenv("CHROMA_HOST", "localhost").strip() or "localhost",
        "port": int(os.getenv("CHROMA_PORT", "8001")),
        "path": os.getenv("CHROMA_PERSIST_DIRECTORY", str(pathlib.Path(_ROOT) / "data" / "chroma")),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _business_backend

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request, build_shared_rag_tools
    from agents.tools import build_business_tools
    from business import BusinessWorkflow, MockBusinessBackend
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager

    cfg = _llm_cfg()
    chroma_cfg = _chroma_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("ZHIYING_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("ZHIYING_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=chroma_cfg["host"],
        chroma_port=chroma_cfg["port"],
        chroma_path=chroma_cfg["path"],
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    _business_backend = MockBusinessBackend(
        os.getenv("BUSINESS_DB_PATH", str(pathlib.Path(_ROOT) / "data" / "business" / "business.db"))
    )
    _orchestrator.set_domain_tools(build_business_tools(_business_backend, _memory))
    _orchestrator.set_business_workflow(BusinessWorkflow(_business_backend, _memory))

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=chroma_cfg["host"],
        chroma_port=chroma_cfg["port"],
        chroma_path=chroma_cfg["path"],
    )
    logger.info(f"知识库已加载: {await kb.doc_count_async()} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或转人工客服确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))
    if _orchestrator is not None:
        _orchestrator.set_shared_tools(build_shared_rag_tools(_tool_manager))

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    logger.info("ZhiYing Agent 已就绪")
    yield

    await _monitor.stop()
    if _memory is not None:
        await _memory.close()
    logger.info("ZhiYing Agent 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ZhiYing Agent 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)


_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/metrics"}
_GUEST_COOKIE = "zhiying_guest_id"
_GUEST_COOKIE_MAX_AGE = 30 * 24 * 60 * 60
_AUTH_USER_HEADER = "X-Authenticated-User"
_AUTH_SIGNATURE_HEADER = "X-Authenticated-User-Signature"
_PLACEHOLDER_API_KEYS = {
    "xxx", "your_api_key", "your-key", "replace-with-a-long-random-key",
}


@app.middleware("http")
async def api_key_guard(request: FastAPIRequest, call_next):
    """可选 API Key 鉴权；生产环境未配置密钥时拒绝业务请求。"""
    # CORS 预检请求不会携带业务 API Key，交给 CORSMiddleware 处理。
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path.rstrip("/") or "/"
    configured = os.getenv("ZHIYING_API_KEY", "").strip()
    if configured.lower() in _PLACEHOLDER_API_KEYS:
        configured = ""
    is_public = path in _PUBLIC_PATHS or path.startswith("/docs/") or path.startswith("/redoc/")

    if not is_public:
        if not configured:
            if os.getenv("APP_ENV", "development").lower() == "production":
                return Response("ZHIYING_API_KEY 未配置", status_code=503)
        else:
            provided = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(provided, configured):
                return Response("Unauthorized", status_code=401)

        identity_error = _authenticate_user_header(request)
        if identity_error is not None:
            return identity_error

    return await call_next(request)


def _valid_guest_id(value: Optional[str]) -> bool:
    """只接受服务端签发格式，避免把任意 Cookie 内容直接用于 Redis/Chroma 键。"""
    return bool(value and re.fullmatch(r"guest_[0-9a-f]{32}", value))


def _valid_authenticated_user_id(value: Optional[str]) -> bool:
    """限制认证网关传入的用户 ID，避免控制字符进入存储键和日志。"""
    return bool(value and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}", value))


def _authenticate_user_header(request: FastAPIRequest) -> Optional[Response]:
    """验签上游认证网关传入的用户身份；返回错误响应表示请求应被拒绝。"""
    user_id = request.headers.get(_AUTH_USER_HEADER, "").strip()
    signature = request.headers.get(_AUTH_SIGNATURE_HEADER, "").strip().lower()
    if not user_id and not signature:
        return None
    if not user_id or not signature or not _valid_authenticated_user_id(user_id):
        return Response("Invalid authenticated user headers", status_code=401)

    secret = os.getenv("ZHIYING_USER_ID_SECRET", "").strip()
    if secret.lower() in {"", "xxx", "your_secret", "replace-with-a-random-secret"}:
        if os.getenv("APP_ENV", "development").lower() == "production":
            return Response("ZHIYING_USER_ID_SECRET 未配置", status_code=503)
        return None

    expected = hmac.new(secret.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return Response("Invalid authenticated user signature", status_code=401)
    request.state.authenticated_user_id = user_id
    return None


def _resolve_chat_identity(request: FastAPIRequest, requested_user_id: str) -> tuple[str, bool, bool]:
    """返回 (memory_user_id, is_guest, newly_issued_guest_id)。

    body.user_id 仅为兼容旧客户端，默认不可信；生产环境应由真正的登录 Token
    中间件设置 request.state.authenticated_user_id 后再接入真实账号。
    """
    authenticated_user_id = getattr(request.state, "authenticated_user_id", None)
    if authenticated_user_id:
        return str(authenticated_user_id), False, False

    allow_client_id = os.getenv("ZHIYING_ALLOW_CLIENT_USER_ID", "false").lower() in {"1", "true", "yes"}
    if allow_client_id and requested_user_id and requested_user_id != "anonymous":
        return requested_user_id, False, False

    guest_id = request.cookies.get(_GUEST_COOKIE)
    if _valid_guest_id(guest_id):
        return guest_id, True, False
    return f"guest_{uuid.uuid4().hex}", True, True

_cors_origins = [origin.strip() for origin in os.getenv(
    "ZHIYING_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",") if origin.strip()]
if os.getenv("APP_ENV", "development").lower() != "production" and not _cors_origins:
    _cors_origins = ["*"]
_cors_allow_credentials = "*" not in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str = Field(min_length=1, max_length=8000)
    user_id:     str = Field(
        default="anonymous",
        min_length=1,
        max_length=128,
        description="兼容旧客户端的字段；默认不作为身份依据，服务端使用 guest_id Cookie",
    )
    conv_id:     Optional[str] = Field(default=None, min_length=1, max_length=128)


class DemoResetRequest(BaseModel):
    conv_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    conv_id:     str
    request_id:  str = ""
    response:    str
    intent:      str
    intent_group: str = "other"
    agent_type:  str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_attempted: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    pending_action: Optional[Dict[str, Any]] = None
    ticket: Optional[Dict[str, Any]] = None


class ToolTraceResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response, request: FastAPIRequest):
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from memory.conversation_memory import MsgRole

    memory_user_id, is_guest, newly_issued_guest_id = _resolve_chat_identity(request, req.user_id)
    if newly_issued_guest_id:
        secure_cookie = os.getenv("APP_ENV", "development").lower() == "production"
        response.set_cookie(
            key=_GUEST_COOKIE,
            value=memory_user_id,
            max_age=_GUEST_COOKIE_MAX_AGE,
            httponly=True,
            secure=secure_cookie,
            samesite="lax",
        )
    # 匿名访客只保留短期记忆；只有认证用户（或明确开启旧兼容模式）才写长期记忆。
    persist_long_term = not is_guest

    conv_id = req.conv_id or str(uuid.uuid4())

    # 1. 读取记忆上下文
    mem_ctx = await _memory.get_context(
        memory_user_id,
        conv_id,
        query=req.message,
        include_long_term=persist_long_term,
    )

    # 2. 构建编排请求（含对话历史，用于意图识别上下文）
    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None

    intent_result = await _orchestrator.recognize_intent(req.message, history=history)
    # 当前版本采用 Agent Tool Use：RAG 由 Agent 按策略调用，不在 API 层重复预取。
    full_context = mem_ctx.to_prompt_text()

    orch_req = OrcReq(
        message=req.message,
        user_id=memory_user_id,
        conv_id=conv_id,
        context=full_context,
        history=history,
        entities=intent_result.entities,
        intent=intent_result.intent,
        intent_group=intent_result.intent_group,
        urgency=intent_result.urgency,
        intent_confidence=intent_result.confidence,
    )

    # 3. 执行
    result = await _orchestrator.run(orch_req)

    # 4. 写入记忆
    await _memory.add_message(
        memory_user_id, conv_id, MsgRole.USER, req.message,
        persist_long_term=persist_long_term,
    )
    await _memory.add_message(
        memory_user_id, conv_id, MsgRole.ASSISTANT, result.response,
        persist_long_term=persist_long_term,
    )

    # 5. 异步更新用户画像（不阻塞响应）
    if persist_long_term:
        asyncio.create_task(_memory.update_profile(memory_user_id, conv_id))

    return ChatResponse(
        conv_id=conv_id,
        request_id=result.request_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        intent_group=intent_result.intent_group,
        agent_type=result.agent_type.value,
        agent_types=[agent_type.value for agent_type in result.agent_types],
        primary_agent=result.primary_agent.value if result.primary_agent else result.agent_type.value,
        supporting_agents=[agent_type.value for agent_type in result.supporting_agents],
        tools_attempted=result.tools_attempted,
        tools_used=result.tools_used,
        routing_reason=result.routing_reason,
        routing_confidence=result.routing_confidence,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used="search_knowledge_base" in result.tools_used,
        entities=intent_result.entities,
        intent_confidence=round(intent_result.confidence, 4),
        intent_source_scores=intent_result.source_scores,
        citations=result.citations,
        pending_action=result.pending_action,
        ticket=result.ticket,
    )


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    """查看某次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = Query(default=20, ge=1, le=100)):
    """查看最近 N 次请求的工具调用明细。"""
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return RecentToolTracesResponse(items=_orchestrator.get_recent_tool_traces(limit=limit))


@app.get("/tickets/{ticket_id}", tags=["业务后台"])
async def get_ticket(ticket_id: str, request: FastAPIRequest):
    if _business_backend is None:
        raise HTTPException(503, "业务后台未初始化")
    user_id, _, _ = _resolve_chat_identity(request, "anonymous")
    try:
        return _business_backend.get_ticket(ticket_id, user_id)
    except Exception as exc:
        raise HTTPException(404, "未找到该工单") from exc


@app.post("/demo/reset", tags=["业务后台"])
async def reset_demo(request: FastAPIRequest, body: Optional[DemoResetRequest] = None):
    """恢复演示订单。生产环境强制禁用，避免成为危险管理接口。"""
    if os.getenv("APP_ENV", "development").lower() not in {"development", "test"}:
        raise HTTPException(404, "接口不存在")
    if _business_backend is None or _memory is None:
        raise HTTPException(503, "业务后台未初始化")
    user_id, _, _ = _resolve_chat_identity(request, "anonymous")
    if body and body.conv_id:
        await _memory.clear_task_state(user_id, body.conv_id)
    return {"message": "演示业务数据已恢复", **_business_backend.reset_demo_data()}


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str = Query(min_length=1, max_length=8000), top_k: int = Query(default=5, ge=1, le=10)):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    source_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    title:   str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1, max_length=200_000)
    document_name: Optional[str] = Field(default=None, max_length=256)
    version: str = Field(default="1.0", min_length=1, max_length=64)
    updated_at: Optional[str] = Field(default=None, max_length=64)
    section: Optional[str] = Field(default=None, max_length=256)


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput] = Field(min_length=1, max_length=100)


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str = Field(min_length=1, max_length=8000)
    expected_intent: str = Field(min_length=1, max_length=64)
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = Field(default=None, max_length=8000)
    turns: Optional[List[str]] = Field(default=None, max_length=50)
    user_id: Optional[str] = Field(default=None, max_length=128)
    conv_id: Optional[str] = Field(default=None, max_length=128)


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = Field(default=None, max_length=500)
    dialog_cases: Optional[List[EvalDialogInput]] = Field(default=None, max_length=100)


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = await kb.add_documents_async([
        d.model_dump(exclude_none=True) for d in body.documents
    ])
    total = await kb.doc_count_async()
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": total}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    if not isinstance(docs, list) or not 1 <= len(docs) <= 100:
        raise HTTPException(400, "文档数量必须在 1-100 之间")
    try:
        docs = [DocInput(**doc).model_dump(exclude_none=True) for doc in docs]
    except Exception as ex:
        raise HTTPException(400, f"文档格式不合法: {ex}")

    count = await kb.add_documents_async(docs)
    total = await kb.doc_count_async()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": await kb.doc_count_async()}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("ZhiYing Agent CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _llm_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("ZHIYING_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("ZHIYING_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    chroma_cfg = _chroma_cfg()
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=chroma_cfg["host"],
        chroma_port=chroma_cfg["port"],
        chroma_path=chroma_cfg["path"],
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nZhiYing [{result.agent_type.value}]: {result.response}\n")

    await mem.close()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
