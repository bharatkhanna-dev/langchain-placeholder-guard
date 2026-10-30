"""Internal helpers for normalising model-call handler results."""

from __future__ import annotations

import dataclasses
from typing import Any

from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, BaseMessage


def tool_name(tool: Any) -> str | None:
    """Return the name of a ``BaseTool``, provider dict tool, or plain string."""
    if isinstance(tool, str):
        return tool
    name = getattr(tool, "name", None)
    if name:
        return str(name)
    if isinstance(tool, dict):
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        return str(name) if name else None
    return None


def result_messages(response: Any) -> list[BaseMessage]:
    """Return the message list from any shape a handler may return."""
    if isinstance(response, AIMessage):
        return [response]
    inner = getattr(response, "model_response", None)  # ExtendedModelResponse
    if inner is not None:
        response = inner
    return list(getattr(response, "result", None) or [])


def last_ai(response: Any) -> AIMessage | None:
    """Return the last ``AIMessage`` in the response, if any."""
    for message in reversed(result_messages(response)):
        if isinstance(message, AIMessage):
            return message
    return None


def replace_last_ai(response: Any, new: AIMessage) -> Any:
    """Return ``response`` with its last ``AIMessage`` replaced, preserving its shape."""
    if isinstance(response, AIMessage):
        return new
    inner = getattr(response, "model_response", None)
    if inner is not None:
        return dataclasses.replace(response, model_response=replace_last_ai(inner, new))
    messages = list(response.result)
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            messages[i] = new
            break
    return ModelResponse(
        result=messages, structured_response=getattr(response, "structured_response", None)
    )
