# langchain-placeholder-guard

A model-agnostic [LangChain](https://github.com/langchain-ai/langchain) agent middleware that deterministically detects **unfilled template placeholders** and **forbidden internal terms** in model output and outgoing tool arguments, then retries, blocks, or flags. No extra model call.

## Why

Agents drafting user-facing content regularly ship `[Client Name]`, `{{first_name}}`, `<AMOUNT>`, `${date}`, "INSERT LINK HERE", or internal terms such as tool names and routing labels. Prompt instructions reduce this but do not eliminate it; LLM-judge checks cost a model call and are non-deterministic.

## Install

```bash
pip install langchain-placeholder-guard
```

## Usage

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_placeholder_guard import PlaceholderGuardMiddleware


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return "sent"


agent = create_agent(
    model="openai:gpt-4o",  # any chat model
    tools=[send_email],
    middleware=[
        PlaceholderGuardMiddleware(
            check_tool_args=["send_email"],           # also scan outgoing email args
            forbidden_terms=["send_email", "[INTERNAL]"],
        ),
    ],
)
```

### What a retry looks like

```text
model call 1: [Human: "Write a greeting"]
              -> AIMessage("Dear [Client Name], about 5 days.")
model call 2: [Human: "Write a greeting",
               Human: "Your previous response contained unfilled placeholders ...
                       - '[Client Name]' in text: '...Dear [Client Name], about 5 days....'"]
              -> AIMessage("Dear Ann, about 5 days.")
```

The bad message is not replayed, because a message with tool calls cannot be sent back without its tool results.

### Block or flag instead of retrying

```python
# Raise PlaceholderLeakError(findings) on the first finding.
PlaceholderGuardMiddleware(strategy="block")

# Return the response unchanged, with findings in response_metadata["placeholder_guard"].
PlaceholderGuardMiddleware(strategy="flag")
```

### Tuning the patterns

```python
from langchain_placeholder_guard import OPTIONAL_PATTERNS

PlaceholderGuardMiddleware(
    extra_patterns=OPTIONAL_PATTERNS,        # also flag TODO/TBD/FIXME and {single_brace}
    ignore=["[Note]", "[Important]"],        # known-safe labels
    detectors=[my_domain_check],             # (str) -> list[Finding]
)
```

Combine with `HumanInTheLoopMiddleware` on `send_email` so reviewers see drafts whose placeholders are already corrected.

## How it works

- Scans the final text of responses without tool calls and, optionally, the string arguments of tool calls (recursively; locations look like `tool:send_email:recipients[0].name`) before the tool runs.
- Strategies:
  - `retry` (default): re-calls with `original messages + [nudge]`. The nudge quotes each finding with surrounding context.
  - `block`: raises `PlaceholderLeakError(findings)`.
  - `flag`: returns a copy of the response with findings in `response_metadata["placeholder_guard"]`.
- Fenced and inline code is ignored by default.

Default patterns: `double_brace` (`{{ x }}`), `dollar_brace` (`${x}`), `bracket_label` (`[Client Name]`), `angle_label` (`<AMOUNT>`, common HTML tags excluded), `insert_here`. Opt-in via `extra_patterns=OPTIONAL_PATTERNS`: `todo_marker`, `single_brace`.

The detector is exported:

```python
from langchain_placeholder_guard import find_placeholders

find_placeholders("Dear [Client Name], total {{ amount }}")
# -> two Finding objects: kind='double_brace' match='{{ amount }}'
#                          kind='bracket_label' match='[Client Name]'
# Each has .kind, .match, .location ('text' or 'tool:<name>:<arg path>') and .snippet.
```

## Configuration

| Constructor arg | Default | Description |
|---|---|---|
| `strategy` | `"retry"` | `"retry"`, `"block"` or `"flag"`. |
| `max_retries` | `2` | Only for `retry`. |
| `check_final_text` | `True` | Scan final response text. |
| `check_tool_args` | `False` | `True` for all tools, or a list of tool names. |
| `patterns` | defaults | Named regexes replacing the defaults. |
| `extra_patterns` | `None` | Named regexes merged in. |
| `forbidden_terms` | `()` | Strings (case-insensitive, word-bounded) or compiled regexes. |
| `flag_bound_tool_names` | `False` | Treat bound tool names as forbidden in final text only. |
| `ignore` | `()` | Exact matches never reported (for example `"[Note]"`). |
| `ignore_code_blocks` | `True` | Skip fenced and inline code. |
| `detectors` | `()` | Custom `(str) -> list[Finding]` callables. |
| `nudge` | see `DEFAULT_PLACEHOLDER_NUDGE` | Text containing `{findings}`. |
| `on_exhausted` | `"return"` | `"return"` returns the last response unchanged, `"error"` raises `PlaceholderLeakError`, any other string is returned to the user as the reply. |

## Ordering

Middleware order is outermost first.

- Place **inside** middleware that trims message history, so the nudge is not trimmed away.
- Put a `ModelCallLimitMiddleware` outermost; retries across several guards multiply model calls.

## Limitations

- Regex heuristics: tune `ignore` and `patterns` for your domain. `[Note]`-style labels in legitimate prose are a known false-positive class.
- **Streaming:** the rejected attempt may already have streamed. For strict guarantees on final text, do not stream raw tokens to end users, or rely on tool-argument checks for outbound messages.
- Does not validate that a filled value is correct, only that it is not a placeholder.
- Not a PII tool. Combine with `PIIMiddleware` if needed.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
