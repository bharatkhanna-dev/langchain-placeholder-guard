"""Pure, deterministic detection of unfilled placeholders and forbidden terms."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

DEFAULT_PLACEHOLDER_PATTERNS: dict[str, str] = {
    # {{ name }}, {{ctx.first_name}}
    "double_brace": r"\{\{\s*[^{}\n]{1,60}?\s*\}\}",
    # ${name}
    "dollar_brace": r"\$\{[A-Za-z_][\w.]{0,60}\}",
    # [Client Name], [NAME], [Insert Date] -- not "[text](url)", "[1]" or "[ ]"
    "bracket_label": r"\[(?:[A-Z][a-z]+|[A-Z]{2,})(?:[ _-](?:[A-Z][a-z]+|[A-Z]{2,})){0,5}\](?!\()",
    # <NAME>, <CLIENT_NAME>
    "angle_label": r"<[A-Z][A-Z0-9_]{1,40}>",
    # "insert ... here"
    "insert_here": r"(?i)\binsert\b[^.\n]{0,40}\bhere\b",
}

OPTIONAL_PATTERNS: dict[str, str] = {
    "todo_marker": r"\b(?:TODO|TBD|FIXME|XXX)\b",
    "single_brace": r"(?<!\{)\{[A-Za-z_][\w.]{0,40}\}(?!\})",
}

_HTML_TAGS = frozenset(
    {
        "A", "B", "I", "U", "P", "BR", "HR", "LI", "UL", "OL", "DL", "DT", "DD", "TD", "TR",
        "TH", "H1", "H2", "H3", "H4", "H5", "H6", "EM", "IMG", "DIV", "PRE", "SUB", "SUP",
        "SPAN", "BODY", "HEAD", "HTML", "FONT", "CODE", "BLOCKQUOTE", "TABLE", "THEAD",
        "TBODY", "TFOOT", "STRONG",
    }
)

_FENCED_CODE = re.compile(r"```[\s\S]*?(?:```|$)")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_SNIPPET_RADIUS = 40


@dataclass(frozen=True)
class Finding:
    """A single placeholder or forbidden-term match."""

    kind: str
    match: str
    location: str
    snippet: str


Detector = Callable[[str], list[Finding]]
Term = str | re.Pattern[str]


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


@lru_cache(maxsize=256)
def _compile_term(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def validate_patterns(patterns: Mapping[str, str]) -> None:
    """Raise ``ValueError`` if any pattern is not a valid regex."""
    for name, pattern in patterns.items():
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid pattern {name!r}: {exc}") from exc


def _blank_code(text: str) -> str:
    def blank(m: re.Match[str]) -> str:
        return " " * len(m.group(0))

    return _INLINE_CODE.sub(blank, _FENCED_CODE.sub(blank, text))


def _snippet(text: str, start: int, end: int) -> str:
    lo = max(0, start - _SNIPPET_RADIUS)
    hi = min(len(text), end + _SNIPPET_RADIUS)
    return " ".join(text[lo:hi].split())


def find_placeholders(
    text: str,
    *,
    patterns: Mapping[str, str] = DEFAULT_PLACEHOLDER_PATTERNS,
    forbidden_terms: Sequence[Term] = (),
    ignore: Sequence[str] = (),
    ignore_code_blocks: bool = True,
    detectors: Sequence[Detector] = (),
    location: str = "text",
) -> list[Finding]:
    """Return placeholders, forbidden terms, and custom detector hits in ``text``.

    ``location`` is recorded on each finding (``"text"`` or ``"tool:<name>:<arg path>"``).
    Matches equal to an entry in ``ignore`` are skipped.
    """
    if not text:
        return []
    scan = _blank_code(text) if ignore_code_blocks else text
    ignored = set(ignore)
    findings: list[Finding] = []

    for kind, pattern in patterns.items():
        for m in _compile(pattern).finditer(scan):
            value = m.group(0)
            if value in ignored:
                continue
            if kind == "angle_label" and value[1:-1] in _HTML_TAGS:
                continue
            findings.append(Finding(kind, value, location, _snippet(text, m.start(), m.end())))

    for term in forbidden_terms:
        regex = term if isinstance(term, re.Pattern) else _compile_term(term)
        for m in regex.finditer(scan):
            value = m.group(0)
            if value in ignored:
                continue
            findings.append(
                Finding("forbidden_term", value, location, _snippet(text, m.start(), m.end()))
            )

    for detector in detectors:
        for f in detector(text):
            if f.match not in ignored:
                findings.append(f)
    return findings


def iter_strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(path, string)`` for every string in nested dicts and lists."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from iter_strings(item, f"{path}[{i}]")


def dedupe(findings: Sequence[Finding]) -> list[Finding]:
    """Remove duplicate findings, preserving order."""
    seen: set[tuple[str, str]] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.match, f.location)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def format_findings(findings: Sequence[Finding], limit: int = 10) -> str:
    """Format findings as a de-duplicated bullet list capped at ``limit`` lines."""
    lines = [
        f'- "{f.match}" in {f.location}: "...{f.snippet}..."' for f in dedupe(findings)[:limit]
    ]
    return "\n".join(lines)
