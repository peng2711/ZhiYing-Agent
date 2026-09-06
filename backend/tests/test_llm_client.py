import asyncio
from types import SimpleNamespace

from core.llm_client import (
    ToolUseBlock,
    anthropic_messages_to_openai,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_response_to_anthropic,
)


def test_anthropic_history_converts_tool_round_trip():
    messages = anthropic_messages_to_openai(
        [
            {"role": "user", "content": "查一下发票政策"},
            {
                "role": "assistant",
                "content": [ToolUseBlock(id="toolu_1", name="search_knowledge_base", input={"query": "发票"})],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"success": true}'}],
            },
        ],
        system="你是客服 Agent",
    )

    assert messages[0] == {"role": "system", "content": "你是客服 Agent"}
    assert messages[2]["tool_calls"][0]["function"]["name"] == "search_knowledge_base"
    assert messages[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": '{"success": true}'}


def test_tools_and_forced_choice_convert_to_openai_shape():
    tools = anthropic_tools_to_openai([{
        "name": "search_knowledge_base",
        "description": "检索知识库",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }])

    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"
    assert anthropic_tool_choice_to_openai({"type": "tool", "name": "search_knowledge_base"}) == {
        "type": "function",
        "function": {"name": "search_knowledge_base"},
    }


def test_openai_tool_response_converts_to_internal_tool_use_block():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="",
            tool_calls=[SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="search_knowledge_base", arguments='{"query":"发票"}'),
            )],
        ))]
    )

    converted = openai_response_to_anthropic(response)
    assert len(converted.content) == 1
    block = converted.content[0]
    assert block.type == "tool_use"
    assert block.id == "call_1"
    assert block.input == {"query": "发票"}


def test_qwen_uses_its_own_non_thinking_parameter(monkeypatch):
    from core.llm_client import OpenAICompatibleMessages

    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setenv("QWEN_THINKING", "disabled")
    messages = OpenAICompatibleMessages(client, provider="qwen")
    asyncio.run(messages.create(
        model="qwen3.7-plus",
        messages=[{"role": "user", "content": "hello"}],
    ))

    assert captured["extra_body"] == {"enable_thinking": False}
    assert "thinking" not in captured["extra_body"]
