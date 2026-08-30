"""LLM client adapters used by ZhiYing Agent.

The application historically used Anthropic's Messages API shape internally.
DeepSeek's native API is OpenAI-compatible, so this module translates between
the two shapes and lets the existing Agent/RAG code keep its Anthropic-style
tool loop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.input is None:
            self.input = {}


@dataclass
class LLMMessageResponse:
    content: List[Any]
    raw: Any = None


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _block_type(block: Any) -> Optional[str]:
    return _value(block, "type")


def _block_value(block: Any, key: str, default: Any = None) -> Any:
    return _value(block, key, default)


def _json_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def anthropic_messages_to_openai(
    messages: Iterable[Dict[str, Any]],
    system: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Translate the internal Anthropic-style message history to OpenAI form."""
    output: List[Dict[str, Any]] = []
    if system:
        output.append({"role": "system", "content": system})

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")

        if role == "assistant" and isinstance(content, list):
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for block in content:
                block_type = _block_type(block)
                if block_type == "text":
                    text = _block_value(block, "text", "")
                    if text:
                        text_parts.append(str(text))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": str(_block_value(block, "id", "")),
                        "type": "function",
                        "function": {
                            "name": str(_block_value(block, "name", "")),
                            "arguments": json.dumps(
                                _block_value(block, "input", {}) or {},
                                ensure_ascii=False,
                            ),
                        },
                    })
            assistant: Dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) or None,
            }
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            output.append(assistant)
            continue

        if role == "user" and isinstance(content, list):
            # Anthropic sends tool results as one user message containing a list;
            # OpenAI expects one role=tool message for each result.
            tool_results = [b for b in content if _block_type(b) == "tool_result"]
            if tool_results and len(tool_results) == len(content):
                for result in tool_results:
                    output.append({
                        "role": "tool",
                        "tool_call_id": str(_block_value(result, "tool_use_id", "")),
                        "content": str(_block_value(result, "content", "")),
                    })
                continue

            text_parts = []
            for block in content:
                text = _block_value(block, "text", None)
                if text is not None:
                    text_parts.append(str(text))
            content = "\n".join(text_parts)

        output.append({"role": role, "content": content})

    return output


def anthropic_tools_to_openai(tools: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools or []:
        converted.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return converted


def anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return choice
    if choice.get("type") == "tool":
        return {
            "type": "function",
            "function": {"name": choice.get("name", "")},
        }
    return choice


def openai_response_to_anthropic(response: Any) -> LLMMessageResponse:
    message = response.choices[0].message
    content: List[Any] = []
    text = getattr(message, "content", None)
    if text:
        content.append(TextBlock(text=str(text)))

    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        content.append(ToolUseBlock(
            id=str(getattr(tool_call, "id", "")),
            name=str(getattr(function, "name", "")),
            input=_json_arguments(getattr(function, "arguments", "{}")),
        ))
    return LLMMessageResponse(content=content, raw=response)


class DeepSeekMessages:
    def __init__(self, client: AsyncOpenAI):
        self._client = client

    async def create(self, **kwargs: Any) -> LLMMessageResponse:
        system = kwargs.pop("system", None)
        messages = anthropic_messages_to_openai(kwargs.pop("messages", []), system=system)
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        # DeepSeek V4 enables Thinking Mode by default. In that mode the API
        # rejects a named tool_choice, while ZhiYing deliberately forces RAG
        # for fact intents. Disable thinking by default for deterministic
        # function-calling; callers can opt in once reasoning_content round
        # tripping is implemented.
        extra_body.setdefault(
            "thinking",
            {"type": os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower() or "disabled"},
        )
        if tools:
            kwargs["tools"] = anthropic_tools_to_openai(tools)
        if tool_choice is not None:
            kwargs["tool_choice"] = anthropic_tool_choice_to_openai(tool_choice)
        kwargs["extra_body"] = extra_body
        response = await self._client.chat.completions.create(
            messages=messages,
            **kwargs,
        )
        return openai_response_to_anthropic(response)


class DeepSeekClient:
    """Small Anthropic-shaped facade over DeepSeek's native OpenAI API."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=(base_url or "https://api.deepseek.com").rstrip("/"),
        )
        self.messages = DeepSeekMessages(self._client)


LLMClient = Any


def create_llm_client(api_key: str, base_url: Optional[str] = None) -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if provider in {"deepseek", "deepseek_openai", "openai"}:
        return DeepSeekClient(api_key=api_key, base_url=base_url)

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncAnthropic(**kwargs)
