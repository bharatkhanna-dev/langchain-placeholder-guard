from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage


def make_request(messages, tools=(), system=None):
    return ModelRequest(
        model=None,
        messages=list(messages),
        system_message=SystemMessage(content=system) if system else None,
        tool_choice=None,
        tools=list(tools),
        response_format=None,
        state={"messages": list(messages)},
        runtime=None,
    )


def _wrap(reply):
    return ModelResponse(result=[reply]) if isinstance(reply, AIMessage) else reply


def scripted_handler(*replies):
    it = iter(replies)
    calls = []

    def handler(req):
        calls.append(req)
        return _wrap(next(it))

    return handler, calls


def scripted_async_handler(*replies):
    it = iter(replies)
    calls = []

    async def handler(req):
        calls.append(req)
        return _wrap(next(it))

    return handler, calls
