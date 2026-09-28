import asyncio

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentOrchestrator,
    AgentType,
    BillingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    TechnicalAgent,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "登录时报 401，同时这笔订单被重复扣款",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.TECHNICAL_LOGIN,
        "intent_group": "technical",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"error_code": ["401"], "amount": ["99 元"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != TechnicalAgent.profile.role
    assert TechnicalAgent.profile.workflow != BillingAgent.profile.workflow
    assert TechnicalAgent.profile.temperature < GeneralAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "lookup_error_code" in TechnicalAgent.profile.tool_scope
    assert "check_billing_fields" in BillingAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    technical_packet = TechnicalAgent(FakeClient(), "test-model")._build_role_packet(req)
    billing_packet = BillingAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "diagnostic_fields" in technical_packet
    assert "verification_fields" in billing_packet
    assert general_packet != technical_packet != billing_packet


def test_agent_system_prompt_contains_executable_response_protocol():
    agent = BillingAgent(FakeClient(), "test-model")
    prompt = agent._build_system_prompt(make_request(intent=IntentCategory.REFUND))

    assert "最终回答协议" in prompt
    assert "结论 → 依据/限制 → 下一步" in prompt
    assert "不做成功承诺" in prompt


def test_escalation_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")

    result = asyncio.run(agent.handle(make_request(
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.CRITICAL,
    )))

    assert result.success is True
    assert result.escalate is True
    assert "人工升级" in result.content
    assert client.calls == []


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.TECHNICAL, "先排查 Token 是否过期。", True),
        AgentResponse(AgentType.BILLING, "请提供两笔扣款的时间和金额。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("先排查 Token 是否过期。")
    assert "补充说明" in content
    assert "两笔扣款" in content


def test_routing_decision_can_target_escalation_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    technical_tools = set(TechnicalAgent(FakeClient(), "test-model").get_tools())
    billing_tools = set(BillingAgent(FakeClient(), "test-model").get_tools())
    escalation_tools = set(EscalationAgent(FakeClient(), "test-model").get_tools())

    handoff = {"request_human_handoff"}
    assert general_tools == {"inspect_request_context", "suggest_required_fields"} | handoff
    assert technical_tools == {"lookup_error_code", "build_diagnostic_plan"} | handoff
    assert billing_tools == {"check_billing_fields", "compare_amounts"} | handoff
    assert escalation_tools == {"create_handoff_summary"}
    assert general_tools & technical_tools == handoff
    assert technical_tools & billing_tools == handoff


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "7 天内可退款"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    billing = BillingAgent(FakeClient(), "test-model")
    escalation = EscalationAgent(FakeClient(), "test-model")

    for agent in (general, technical, billing, escalation):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_tool_input_validation_rejects_unknown_fields():
    agent = TechnicalAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["lookup_error_code"]

    try:
        agent._validate_tool_input(spec, {"error_code": "401", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "lookup_error_code"
        input = {"error_code": "401"}

    class TextBlock:
        type = "text"
        text = "已根据 401 错误码给出排查建议。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = TechnicalAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_attempted == ["lookup_error_code"]
    assert response.tools_used == ["lookup_error_code"]
    assert len(response.tool_traces) == 1
    assert response.tool_traces[0]["tool_name"] == "lookup_error_code"
    assert response.tool_traces[0]["input"] == {"error_code": "401"}
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "lookup_error_code",
        "build_diagnostic_plan",
        "request_human_handoff",
    }
    assert "tool_result" in str(client.calls[1]["messages"])


def test_rag_results_are_exposed_as_structured_citations():
    class ToolUseBlock:
        type, id, name, input = "tool_use", "toolu_rag", "search_knowledge_base", {"query": "退款政策"}
    class TextBlock:
        type, text = "text", "购买后 7 天内可以申请退款。"
    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [type("Response", (), {"content": [ToolUseBlock()]})(),
                              type("Response", (), {"content": [TextBlock()]})()]
        class Messages:
            def __init__(self, owner): self.owner = owner
            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)
        @property
        def messages(self): return self.Messages(self)
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            data = [{"source_id": "refund-policy", "document_name": "退款政策", "version": "3.0",
                     "updated_at": "2026-08-20", "section": "退款时限", "chunk": 2,
                     "content": "购买后 7 天内可以申请无理由退款。", "score": 0.93}]
            return type("Result", (), {"success": True, "data": data, "reranked": True})()
    client = ToolClient()
    agent = BillingAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(RagManager()))
    response = asyncio.run(agent.handle(make_request(
        message="退款政策是什么？请给出依据",
        intent=IntentCategory.QUERY,
    )))
    assert client.calls[0]["tool_choice"]["name"] == "search_knowledge_base"
    assert response.citations[0]["source_id"] == "refund-policy"
    assert response.citations[0]["version"] == "3.0"
    assert response.citations[0]["chunk"] == 2


def test_same_domain_multi_goal_executes_each_goal_and_keeps_sections():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._business_workflow = None
    calls = []

    async def execute(req, agent_type):
        calls.append((
            req.focus_intent,
            req.intent,
            agent_type,
            req.task_state["explicit_intents"],
        ))
        return AgentResponse(
            agent_type=AgentType.BILLING,
            content=f"已处理 {req.focus_intent.value}",
            success=True,
        )

    orchestrator._execute = execute
    orchestrator._route = lambda intent, urgency: AgentType.BILLING
    orchestrator._record_tool_trace = lambda result: None
    req = make_request(
        message="我想退款并申请发票",
        intent=IntentCategory.REFUND,
        intent_group="billing",
        task_state={"explicit_intents": ["refund", "invoice"]},
    )

    result = asyncio.run(orchestrator.run_multi_goal(
        req,
        [IntentCategory.REFUND, IntentCategory.INVOICE],
    ))

    assert "### 退款" in result.response
    assert "### 发票" in result.response
    assert [call[0] for call in calls] == [IntentCategory.REFUND, IntentCategory.INVOICE]
    assert all(call[1] == call[0] for call in calls)
    assert [call[3] for call in calls] == [["refund"], ["invoice"]]
    assert result.task_state["explicit_intents"] == ["refund", "invoice"]
    assert result.routing_reason == "当前轮多目标拆分：refund, invoice"


def test_focused_prompt_forbids_answering_other_goals():
    agent = BillingAgent(client=FakeClient(), model="test")
    req = make_request(
        message="我想退款并申请发票",
        intent=IntentCategory.INVOICE,
        focus_intent=IntentCategory.INVOICE,
    )

    prompt = agent._build_system_prompt(req)

    assert "当前唯一允许回答的主题是“发票”" in prompt
    assert "不要解释、总结或重复其他主题" in prompt


def test_concurrent_requests_do_not_share_pending_action():
    """同一 Agent 实例被并发请求共享时，退款确认单只能出现在发起者自己的响应里。"""
    from agents.tools import make_tool

    class ToolUseBlock:
        type, id, name = "tool_use", "toolu_refund", "prepare_refund"
        input = {"order_id": "10086", "reason": "不想要了"}

    class TextBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    a_prepared = asyncio.Event()
    others_done = asyncio.Event()

    class InterleavingClient:
        class Messages:
            async def create(self, **kwargs):
                messages = kwargs["messages"]
                if "A 申请退款" in str(messages):
                    if isinstance(messages[-1]["content"], str):
                        return type("Response", (), {"content": [ToolUseBlock()]})()
                    # A 已执行 prepare_refund，等其他请求结束后才返回最终回复。
                    await others_done.wait()
                    return type("Response", (), {"content": [TextBlock("A 的退款确认单已生成")]})()
                if "B 咨询" in str(messages):
                    # B 在 A 调用工具之前进入，在 A 结束之前返回。
                    await a_prepared.wait()
                return type("Response", (), {"content": [TextBlock("已回复")]})()

        messages = Messages()

    def prepare_refund(req, args):
        a_prepared.set()
        return {"success": True, "order_id": args["order_id"], "owner": req.user_id,
                "confirmation_token": "secret"}

    agent = BillingAgent(InterleavingClient(), "test-model")
    agent.set_domain_tools({"prepare_refund": make_tool(
        "prepare_refund", "生成退款确认单",
        {"order_id": {"type": "string"}, "reason": {"type": "string"}},
        prepare_refund, ["order_id", "reason"],
    )})

    def billing_request(message, user_id):
        return make_request(message=message, user_id=user_id, conv_id=user_id,
                            intent=IntentCategory.BILLING, entities={})

    async def scenario():
        task_b = asyncio.create_task(agent.handle(billing_request("B 咨询发票", "user_b")))
        await asyncio.sleep(0)
        task_a = asyncio.create_task(agent.handle(billing_request("A 申请退款 10086", "user_a")))
        response_b = await task_b
        # C 在 A 工具调用之后、A 返回之前开始并结束。
        response_c = await agent.handle(billing_request("C 咨询发票", "user_c"))
        others_done.set()
        return await task_a, response_b, response_c

    response_a, response_b, response_c = asyncio.run(scenario())

    assert response_a.pending_action is not None
    assert response_a.pending_action["owner"] == "user_a"
    assert "confirmation_token" not in response_a.pending_action
    assert response_a.tools_used == ["prepare_refund"]
    assert response_b.pending_action is None
    assert response_b.tools_attempted == []
    assert response_c.pending_action is None


class _ScriptedClient:
    """按顺序返回预设响应的 LLM 客户端。"""

    def __init__(self, *responses):
        self.responses = list(responses)

        class Messages:
            async def create(inner, **kwargs):
                return self.responses.pop(0)

        self.messages = Messages()


def _blocks(*blocks):
    return type("Response", (), {"content": list(blocks)})()


def test_mentioning_human_support_in_text_does_not_escalate():
    client = _ScriptedClient(_blocks({"type": "text", "text": "可以按以下步骤操作，如仍有疑问可联系人工客服。"}))
    agent = GeneralAgent(client, "test-model")

    response = asyncio.run(agent.handle(make_request(intent=IntentCategory.QUERY, entities={})))

    assert response.success is True
    assert response.escalate is False


def test_explicit_handoff_tool_escalates_and_orchestrator_creates_ticket():
    client = _ScriptedClient(
        _blocks({"type": "tool_use", "id": "toolu_h", "name": "request_human_handoff",
                 "input": {"reason": "需要后台核对账户权限"}}),
        _blocks({"type": "text", "text": "该问题需要人工核对，已为您转交。"}),
    )
    orchestrator = AgentOrchestrator(api_key="test", model="test-model")
    for agent in orchestrator._pool[AgentType.GENERAL]:
        agent._client = client
    tickets = []
    orchestrator.set_domain_tools({"escalation": {"create_ticket": build_ticket_tool(tickets)}})

    result = asyncio.run(orchestrator.run(make_request(
        message="我的账号权限异常", intent=IntentCategory.QUERY, intent_group="general",
        urgency=UrgencyLevel.MEDIUM, entities={},
    )))

    assert result.escalated is True
    assert "request_human_handoff" in result.tools_used
    assert result.ticket == {"ticket_id": "CS-TEST", "status": "open"}
    assert len(tickets) == 1


def build_ticket_tool(tickets):
    from agents.tools import make_tool

    def create_ticket(req, args):
        tickets.append(args)
        return {"success": True, "ticket": {"ticket_id": "CS-TEST", "status": "open"}}

    return make_tool("create_ticket", "创建工单",
                     {"issue_type": {"type": "string"}, "priority": {"type": "string"}, "summary": {"type": "string"}},
                     create_ticket, ["issue_type", "priority", "summary"], "write")


def _counting_tool(counter):
    from agents.tools import make_tool

    def handler(req, args):
        counter.append(dict(args))
        return {"success": True, "status": "已发货"}

    return make_tool("lookup_status", "查询状态", {"order_id": {"type": "string"}}, handler, required=["order_id"])


def _tool_call(call_id, order_id="A1"):
    return _blocks({"type": "tool_use", "id": call_id, "name": "lookup_status", "input": {"order_id": order_id}})


def test_repeated_identical_tool_call_is_not_executed_again():
    executed = []
    client = _ScriptedClient(_tool_call("t1"), _tool_call("t2"), _blocks({"type": "text", "text": "订单已发货。"}))
    agent = GeneralAgent(client, "test-model")
    agent.set_domain_tools({"lookup_status": _counting_tool(executed)})

    response = asyncio.run(agent.handle(make_request(intent=IntentCategory.QUERY, entities={})))

    assert response.success is True
    assert executed == [{"order_id": "A1"}]
    assert "已经用相同参数调用过" in response.tool_traces[1]["error"]


def test_second_repeated_call_stops_the_loop_as_no_progress():
    executed = []
    client = _ScriptedClient(_tool_call("t1"), _tool_call("t2"), _tool_call("t3"))
    agent = GeneralAgent(client, "test-model")
    agent.set_domain_tools({"lookup_status": _counting_tool(executed)})

    response = asyncio.run(agent.handle(make_request(intent=IntentCategory.QUERY, entities={})))

    assert response.success is False
    assert "无进展" in response.error
    assert len(executed) == 1


def test_llm_timeout_is_capped_by_remaining_agent_deadline(monkeypatch):
    import time

    monkeypatch.setenv("ZHIYING_AGENT_DEADLINE_S", "1")

    class SlowClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                await asyncio.sleep(5)

    agent = GeneralAgent(SlowClient(), "test-model")
    t0 = time.monotonic()
    response = asyncio.run(agent.handle(make_request(intent=IntentCategory.QUERY, entities={})))

    assert response.success is False
    assert time.monotonic() - t0 < 2.5


def test_request_token_budget_stops_further_llm_rounds(monkeypatch):
    import types

    from core.llm_client import MeteredClient
    from core.usage import track_request_usage

    monkeypatch.setenv("ZHIYING_REQUEST_TOKEN_BUDGET", "100")
    calls = []

    class Messages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            response = _tool_call(f"t{len(calls)}", order_id=f"A{len(calls)}")
            response.usage = {"input_tokens": 100, "output_tokens": 20}
            return response

    agent = GeneralAgent(MeteredClient(types.SimpleNamespace(messages=Messages())), "test-model")
    agent.set_domain_tools({"lookup_status": _counting_tool([])})

    async def main():
        with track_request_usage():
            return await agent.handle(make_request(intent=IntentCategory.QUERY, entities={}))

    response = asyncio.run(main())

    assert response.success is False
    assert "token 预算" in response.error
    assert len(calls) == 1
