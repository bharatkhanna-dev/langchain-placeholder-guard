"""Detect unfilled placeholders and forbidden terms in model output."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from ._common import last_ai, replace_last_ai, tool_name
from .detection import (
    DEFAULT_PLACEHOLDER_PATTERNS,
    Detector,
    Finding,
    Term,
    dedupe,
    find_placeholders,
    format_findings,
    iter_strings,
    validate_patterns,
)

logger = logging.getLogger(__name__)

DEFAULT_PLACEHOLDER_NUDGE = (
    "Your previous response contained unfilled placeholders or internal terms that must "
    "not appear in the output:\n{findings}\nRewrite the response replacing each with the "
    "real value from the conversation or tool results. If a value is unknown, rephrase "
    "the sentence without it. Do not mention tools or internal steps."
)


class PlaceholderLeakError(RuntimeError):
    """Raised for ``strategy="block"`` or when retries are exhausted with ``"error"``."""

    def __init__(self, findings: Sequence[Finding]) -> None:
        super().__init__(f"{len(findings)} placeholder finding(s) in model output")
        self.findings = list(findings)


class PlaceholderGuardMiddleware(AgentMiddleware):
    """Retry, block, or flag model output containing unfilled placeholders."""

    def __init__(
        self,
        *,
        strategy: Literal["retry", "block", "flag"] = "retry",
        max_retries: int = 2,
        check_final_text: bool = True,
        check_tool_args: bool | Sequence[str] = False,
        patterns: Mapping[str, str] | None = None,
        extra_patterns: Mapping[str, str] | None = None,
        forbidden_terms: Sequence[Term] = (),
        flag_bound_tool_names: bool = False,
        ignore: Sequence[str] = (),
        ignore_code_blocks: bool = True,
        detectors: Sequence[Detector] = (),
        nudge: str = DEFAULT_PLACEHOLDER_NUDGE,
        on_exhausted: str = "return",
    ) -> None:
        """Initialize the middleware.

        Args:
            strategy: ``"retry"`` nudges and re-calls, ``"block"`` raises
                ``PlaceholderLeakError``, ``"flag"`` attaches findings to
                ``response_metadata["placeholder_guard"]``.
            max_retries: Retries after the first attempt (``retry`` only). Must be >= 0.
            check_final_text: Scan the text of final responses (no tool calls).
            check_tool_args: ``True`` scans every tool call's string arguments; a
                sequence of tool names scans only those tools.
            patterns: Named regexes replacing the defaults.
            extra_patterns: Named regexes merged on top of ``patterns``.
            forbidden_terms: Case-insensitive word-bounded strings or compiled regexes.
            flag_bound_tool_names: Treat bound tool names as forbidden in final text.
            ignore: Exact matches that are never reported.
            ignore_code_blocks: Skip fenced and inline code.
            detectors: Custom callables returning ``Finding`` lists.
            nudge: Text containing ``{findings}``, replaced by a bullet list of findings.
            on_exhausted: ``"return"``, ``"error"``, or a fallback reply string.
                Any other string is returned as the reply.
        """
        if strategy not in ("retry", "block", "flag"):
            raise ValueError("strategy must be 'retry', 'block' or 'flag'")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        merged = dict(DEFAULT_PLACEHOLDER_PATTERNS if patterns is None else patterns)
        merged.update(extra_patterns or {})
        validate_patterns(merged)
        super().__init__()
        self.strategy = strategy
        self.max_retries = max_retries
        self.check_final_text = check_final_text
        self.check_tool_args = check_tool_args
        self.patterns = merged
        self.forbidden_terms = tuple(forbidden_terms)
        self.flag_bound_tool_names = flag_bound_tool_names
        self.ignore = tuple(ignore)
        self.ignore_code_blocks = ignore_code_blocks
        self.detectors = tuple(detectors)
        self.nudge = nudge
        self.on_exhausted = on_exhausted

    def _find(
        self, text: str, location: str, extra_terms: Sequence[Term] = ()
    ) -> list[Finding]:
        return find_placeholders(
            text,
            patterns=self.patterns,
            forbidden_terms=(*self.forbidden_terms, *extra_terms),
            ignore=self.ignore,
            ignore_code_blocks=self.ignore_code_blocks,
            detectors=self.detectors,
            location=location,
        )

    def _tool_in_scope(self, name: str) -> bool:
        if self.check_tool_args is True:
            return True
        if not self.check_tool_args:
            return False
        return name in self.check_tool_args

    def _scan(self, response: Any, request: ModelRequest) -> list[Finding]:
        last = last_ai(response)
        if last is None:
            return []
        findings: list[Finding] = []
        if last.tool_calls:
            if self.check_tool_args:
                for call in last.tool_calls:
                    name = call["name"]
                    if not self._tool_in_scope(name):
                        continue
                    for path, value in iter_strings(call.get("args") or {}):
                        findings += self._find(value, f"tool:{name}:{path}")
        elif self.check_final_text:
            extra: list[str] = []
            if self.flag_bound_tool_names:
                extra = [n for n in map(tool_name, request.tools or []) if n]
            findings += self._find(last.text, "text", extra)
        return dedupe(findings)

    def _attach_findings(self, response: Any, findings: list[Finding]) -> Any:
        last = last_ai(response)
        if last is None:
            return response
        meta = {**last.response_metadata, "placeholder_guard": [asdict(f) for f in findings]}
        return replace_last_ai(response, last.model_copy(update={"response_metadata": meta}))

    def _retry_request(self, request: ModelRequest, findings: list[Finding]) -> ModelRequest:
        text = self.nudge.replace("{findings}", format_findings(findings))
        return request.override(messages=[*request.messages, HumanMessage(content=text)])

    def _exhausted(self, response: Any, findings: list[Finding]) -> Any:
        if self.on_exhausted == "return":
            return response
        if self.on_exhausted == "error":
            raise PlaceholderLeakError(findings)
        return ModelResponse(result=[AIMessage(content=self.on_exhausted)])

    def _handle(self, response: Any, findings: list[Finding]) -> Any | None:
        """Return a terminal response for block/flag strategies, else ``None``."""
        if self.strategy == "block":
            raise PlaceholderLeakError(findings)
        if self.strategy == "flag":
            return self._attach_findings(response, findings)
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        current = request
        response: Any = None
        findings: list[Finding] = []
        for attempt in range(self.max_retries + 1):
            response = handler(current)
            findings = self._scan(response, request)
            if not findings:
                return response
            terminal = self._handle(response, findings)
            if terminal is not None:
                return terminal
            if attempt < self.max_retries:
                logger.warning(
                    "placeholder findings (%d); retrying (attempt %d)", len(findings), attempt + 1
                )
                current = self._retry_request(request, findings)
        return self._exhausted(response, findings)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        current = request
        response: Any = None
        findings: list[Finding] = []
        for attempt in range(self.max_retries + 1):
            response = await handler(current)
            findings = self._scan(response, request)
            if not findings:
                return response
            terminal = self._handle(response, findings)
            if terminal is not None:
                return terminal
            if attempt < self.max_retries:
                logger.warning(
                    "placeholder findings (%d); retrying (attempt %d)", len(findings), attempt + 1
                )
                current = self._retry_request(request, findings)
        return self._exhausted(response, findings)
