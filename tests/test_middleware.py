import re
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_placeholder_guard import (
    OPTIONAL_PATTERNS,
    Finding,
    PlaceholderGuardMiddleware,
    PlaceholderLeakError,
    find_placeholders,
)

from .conftest import make_request, scripted_async_handler, scripted_handler

H = [HumanMessage("hi")]


@tool
def send_email(body: str) -> str:
    """Send."""
    return body


def kinds(text, **kw):
    return {f.kind for f in find_placeholders(text, **kw)}


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Hi {{ first_name }}", "double_brace"),
        ("Hi {{ctx.first_name}}", "double_brace"),
        ("Total ${amount}", "dollar_brace"),
        ("Dear [Client Name],", "bracket_label"),
        ("Dear [NAME],", "bracket_label"),
        ("Pay <AMOUNT> now", "angle_label"),
        ("<CLIENT_NAME>", "angle_label"),
        ("Please INSERT LINK HERE", "insert_here"),
    ],
)
def test_default_patterns_positive(text, kind):
    assert kind in kinds(text)


@pytest.mark.parametrize(
    "text",
    ["see [link](http://x.y)", "ref [1]", "task [ ]", '{"a": 1}', "line<br>break", "a<BR>b", "<HR>"],
)
def test_documented_negatives(text):
    assert not find_placeholders(text)


def test_ignore_exact_match():
    assert not find_placeholders("[Note] this", ignore=["[Note]"])
    assert find_placeholders("[Note] this")


def test_code_blocks_ignored_by_default():
    text = "Use ```\n{{ name }}\n``` and `[Client Name]`"
    assert not find_placeholders(text)
    assert find_placeholders(text, ignore_code_blocks=False)


def test_optional_patterns():
    assert not find_placeholders("TODO {x}")
    found = kinds("TODO {x}", patterns={**OPTIONAL_PATTERNS})
    assert found == {"todo_marker", "single_brace"}


def test_forbidden_terms_string_and_regex():
    assert find_placeholders("I used Lookup_Customer", forbidden_terms=["lookup_customer"])
    assert not find_placeholders("lookup_customers", forbidden_terms=["lookup_customer"])
    assert find_placeholders("route-7", forbidden_terms=[re.compile(r"route-\d")])
    assert find_placeholders("see [INTERNAL] here", forbidden_terms=["[INTERNAL]"])


def test_regex_performance_and_robustness():
    text = ("[A " * 30000) + ("{{ " * 30000) + "\u202e\x00" * 1000
    start = time.perf_counter()
    find_placeholders(text)
    assert time.perf_counter() - start < 5


def test_clean_text_same_object():
    mw = PlaceholderGuardMiddleware()
    resp = _resp("All good")
    handler, calls = scripted_handler(resp)
    assert mw.wrap_model_call(make_request(H), handler) is resp
    assert len(calls) == 1


def _resp(text):
    from langchain.agents.middleware import ModelResponse

    return ModelResponse(result=[AIMessage(content=text)])


def test_retry_request_shape():
    mw = PlaceholderGuardMiddleware()
    bad = AIMessage(content="Dear [Client Name], hello")
    handler, calls = scripted_handler(bad, AIMessage(content="Dear Ann, hello"))
    request = make_request(H)
    out = mw.wrap_model_call(request, handler)
    assert out.result[0].content == "Dear Ann, hello"
    retry = calls[1].messages
    assert len(retry) == 2 and bad not in retry
    assert isinstance(retry[1], HumanMessage)
    assert "[Client Name]" in retry[1].content and "Dear" in retry[1].content
    assert request.messages == H


def test_nudge_with_braces_in_findings_does_not_break():
    mw = PlaceholderGuardMiddleware()
    handler, calls = scripted_handler(AIMessage(content="Hi {{ x }}"), AIMessage(content="Hi"))
    mw.wrap_model_call(make_request(H), handler)
    assert "{{ x }}" in calls[1].messages[-1].content


def test_tool_arg_scan_with_nested_paths_and_scope():
    mw = PlaceholderGuardMiddleware(check_tool_args=["send_email"])
    bad = AIMessage(
        content="",
        tool_calls=[
            {"name": "send_email", "args": {"to": [{"name": "[Client Name]"}], "body": "Hi {{n}}"}, "id": "1"},
            {"name": "other", "args": {"body": "{{n}}"}, "id": "2"},
        ],
    )
    findings = mw._scan(_resp_msg(bad), make_request(H))
    locs = {f.location for f in findings}
    assert locs == {"tool:send_email:to[0].name", "tool:send_email:body"}


def _resp_msg(msg):
    from langchain.agents.middleware import ModelResponse

    return ModelResponse(result=[msg])


def test_tool_arg_retry_does_not_replay_tool_call_message():
    mw = PlaceholderGuardMiddleware(check_tool_args=True)
    bad = AIMessage(
        content="", tool_calls=[{"name": "send_email", "args": {"body": "Hi [NAME]"}, "id": "1"}]
    )
    good = AIMessage(
        content="", tool_calls=[{"name": "send_email", "args": {"body": "Hi Ann"}, "id": "2"}]
    )
    handler, calls = scripted_handler(bad, good)
    mw.wrap_model_call(make_request(H), handler)
    assert len(calls) == 2
    assert all(not getattr(m, "tool_calls", None) for m in calls[1].messages)


def test_tool_calls_not_scanned_when_check_tool_args_false():
    mw = PlaceholderGuardMiddleware(check_tool_args=False)
    msg = AIMessage(
        content="[Client Name]",
        tool_calls=[{"name": "send_email", "args": {"body": "{{x}}"}, "id": "1"}],
    )
    handler, calls = scripted_handler(msg)
    mw.wrap_model_call(make_request(H), handler)
    assert len(calls) == 1


def test_check_final_text_false():
    mw = PlaceholderGuardMiddleware(check_final_text=False)
    handler, calls = scripted_handler(AIMessage(content="[Client Name]"))
    mw.wrap_model_call(make_request(H), handler)
    assert len(calls) == 1


def test_block_raises_with_findings():
    mw = PlaceholderGuardMiddleware(strategy="block")
    handler, calls = scripted_handler(AIMessage(content="Hi [NAME]"))
    with pytest.raises(PlaceholderLeakError) as exc:
        mw.wrap_model_call(make_request(H), handler)
    assert exc.value.findings[0].match == "[NAME]"
    assert len(calls) == 1


def test_flag_attaches_metadata_on_copy():
    mw = PlaceholderGuardMiddleware(strategy="flag")
    original = AIMessage(content="Hi [NAME]", response_metadata={"k": 1})
    handler, calls = scripted_handler(original)
    out = mw.wrap_model_call(make_request(H), handler)
    assert len(calls) == 1
    meta = out.result[0].response_metadata
    assert meta["k"] == 1 and meta["placeholder_guard"][0]["match"] == "[NAME]"
    assert "placeholder_guard" not in original.response_metadata
    assert out.result[0] is not original


def test_flag_bound_tool_names_final_text_only():
    mw = PlaceholderGuardMiddleware(flag_bound_tool_names=True, max_retries=0)
    handler, _ = scripted_handler(AIMessage(content="I called send_email."))
    with pytest.raises(PlaceholderLeakError):
        PlaceholderGuardMiddleware(
            flag_bound_tool_names=True, max_retries=0, on_exhausted="error"
        ).wrap_model_call(make_request(H, tools=[send_email]), handler)
    handler, _ = scripted_handler(AIMessage(content="clean"))
    assert mw.wrap_model_call(make_request(H, tools=[send_email]), handler)

    tool_msg = AIMessage(
        content="", tool_calls=[{"name": "send_email", "args": {"body": "send_email"}, "id": "1"}]
    )
    found = mw._scan(_resp_msg(tool_msg), make_request(H, tools=[send_email]))
    assert found == []


def test_dedup_and_cap():
    mw = PlaceholderGuardMiddleware()
    words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    words += ["Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu"]
    text = "[NAME] [NAME] " + " ".join(f"[Item {w}]" for w in words)
    handler, calls = scripted_handler(AIMessage(content=text), AIMessage(content="ok"))
    mw.wrap_model_call(make_request(H), handler)
    nudge = calls[1].messages[-1].content
    assert sum(line.startswith("- ") for line in nudge.splitlines()) == 10
    assert nudge.count('"[NAME]"') == 1


def test_custom_detector():
    def det(text):
        return [Finding("custom", "bad", "text", text)] if "bad" in text else []

    mw = PlaceholderGuardMiddleware(detectors=[det], strategy="block")
    handler, _ = scripted_handler(AIMessage(content="so bad"))
    with pytest.raises(PlaceholderLeakError):
        mw.wrap_model_call(make_request(H), handler)


def test_exhaustion_modes():
    bad = AIMessage(content="[NAME]")
    handler, calls = scripted_handler(bad, bad, bad)
    PlaceholderGuardMiddleware().wrap_model_call(make_request(H), handler)
    assert len(calls) == 3

    handler, _ = scripted_handler(bad)
    with pytest.raises(PlaceholderLeakError):
        PlaceholderGuardMiddleware(max_retries=0, on_exhausted="error").wrap_model_call(
            make_request(H), handler
        )

    handler, _ = scripted_handler(bad)
    out = PlaceholderGuardMiddleware(max_retries=0, on_exhausted="Sorry").wrap_model_call(
        make_request(H), handler
    )
    assert out.result[0].content == "Sorry"


def test_invalid_config():
    with pytest.raises(ValueError):
        PlaceholderGuardMiddleware(strategy="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PlaceholderGuardMiddleware(max_retries=-1)
    with pytest.raises(ValueError):
        PlaceholderGuardMiddleware(extra_patterns={"bad": "("})


async def test_async_parity():
    mw = PlaceholderGuardMiddleware(check_tool_args=True)
    bad = AIMessage(
        content="", tool_calls=[{"name": "send_email", "args": {"body": "{{x}}"}, "id": "1"}]
    )
    handler, calls = scripted_async_handler(bad, AIMessage(content="ok"))
    out = await mw.awrap_model_call(make_request(H), handler)
    assert len(calls) == 2 and out.result[0].content == "ok"

    flag = PlaceholderGuardMiddleware(strategy="flag")
    handler, _ = scripted_async_handler(AIMessage(content="[NAME]"))
    out = await flag.awrap_model_call(make_request(H), handler)
    assert out.result[0].response_metadata["placeholder_guard"]

    block = PlaceholderGuardMiddleware(strategy="block")
    handler, _ = scripted_async_handler(AIMessage(content="[NAME]"))
    with pytest.raises(PlaceholderLeakError):
        await block.awrap_model_call(make_request(H), handler)
