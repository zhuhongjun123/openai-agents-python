from __future__ import annotations

import pytest
from pydantic import BaseModel

from agents import (
    Agent,
    ModelBehaviorError,
    RunConfig,
    RunContextWrapper,
    Runner,
    RunState,
    UserError,
    handoff,
)
from agents.tool import Tool, function_tool

from .fake_model import FakeModel
from .mcp.helpers import FakeMCPServer
from .test_responses import get_function_tool_call, get_handoff_tool_call, get_text_message


class StructuredResult(BaseModel):
    status: str


@pytest.mark.asyncio
async def test_run_rejects_repeated_function_tool_instance_in_error_mode() -> None:
    tool = function_tool(lambda: "result", name_override="lookup")
    model = FakeModel(initial_output=[get_text_message("done")])
    agent = Agent(name="agent", model=model, tools=[tool, tool])

    with pytest.raises(UserError, match="the tool name `lookup` is used by multiple tools"):
        await Runner.run(
            agent,
            "Look this up",
            run_config=RunConfig(tool_name_collision_policy="error"),
        )

    assert model.first_turn_args is None


@pytest.mark.asyncio
async def test_run_allows_disabled_duplicate_function_tool_in_error_mode() -> None:
    enabled_tool = function_tool(lambda: "enabled", name_override="lookup")
    disabled_tool = function_tool(
        lambda: "disabled",
        name_override="lookup",
        is_enabled=False,
    )
    model = FakeModel(initial_output=[get_text_message("done")])
    agent = Agent(name="agent", model=model, tools=[enabled_tool, disabled_tool])

    await Runner.run(
        agent,
        "Look this up",
        run_config=RunConfig(tool_name_collision_policy="error"),
    )

    assert model.first_turn_args is not None
    assert model.first_turn_args["tools"] == [enabled_tool]


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.asyncio
async def test_run_allows_visible_and_deferred_same_name(
    reverse_order: bool,
) -> None:
    visible_tool = function_tool(lambda: "visible", name_override="lookup")
    deferred_tool = function_tool(
        lambda: "deferred",
        name_override="lookup",
        defer_loading=True,
    )
    tools: list[Tool] = (
        [deferred_tool, visible_tool] if reverse_order else [visible_tool, deferred_tool]
    )
    model = FakeModel(initial_output=[get_text_message("done")])
    agent = Agent(name="agent", model=model, tools=tools)

    await Runner.run(
        agent,
        "Look this up",
        run_config=RunConfig(tool_name_collision_policy="error"),
    )

    assert model.first_turn_args is not None
    assert model.first_turn_args["tools"] == tools


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.asyncio
async def test_resume_rejects_duplicate_function_tools_before_approved_call(
    streamed: bool,
) -> None:
    calls: list[str] = []

    def first_lookup() -> str:
        calls.append("first")
        return "first"

    def second_lookup() -> str:
        calls.append("second")
        return "second"

    first_tool = function_tool(
        first_lookup,
        name_override="lookup",
        needs_approval=True,
    )
    second_tool = function_tool(
        second_lookup,
        name_override="lookup",
        needs_approval=True,
    )
    model = FakeModel(initial_output=[get_function_tool_call("lookup", "{}")])
    agent = Agent(name="agent", model=model, tools=[first_tool, second_tool])

    initial_result = await Runner.run(agent, "Look this up")
    state = await RunState.from_json(agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])

    run_config = RunConfig(tool_name_collision_policy="error")
    if streamed:
        resumed_result = Runner.run_streamed(agent, state, run_config=run_config)
        with pytest.raises(UserError, match="the tool name `lookup` is used by multiple tools"):
            async for _ in resumed_result.stream_events():
                pass
    else:
        with pytest.raises(UserError, match="the tool name `lookup` is used by multiple tools"):
            await Runner.run(agent, state, run_config=run_config)

    assert calls == []


@pytest.mark.asyncio
async def test_resume_warn_mode_executes_last_duplicate_function_tool() -> None:
    calls: list[str] = []

    def first_lookup() -> str:
        calls.append("first")
        return "first"

    def second_lookup() -> str:
        calls.append("second")
        return "second"

    first_tool = function_tool(
        first_lookup,
        name_override="lookup",
        needs_approval=True,
    )
    second_tool = function_tool(
        second_lookup,
        name_override="lookup",
        needs_approval=True,
    )
    model = FakeModel(initial_output=[get_function_tool_call("lookup", "{}")])
    agent = Agent(name="agent", model=model, tools=[first_tool, second_tool])

    initial_result = await Runner.run(agent, "Look this up")
    state = await RunState.from_json(agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])
    model.set_next_output([get_text_message("done")])

    resumed_result = await Runner.run(agent, state)

    assert resumed_result.final_output == "done"
    assert calls == ["second"]


@pytest.mark.asyncio
async def test_resume_rejects_function_and_handoff_collision_before_approved_call() -> None:
    calls: list[str] = []

    def approved_tool() -> str:
        calls.append("approved")
        return "approved"

    approval_tool = function_tool(
        approved_tool,
        name_override="approval_tool",
        needs_approval=True,
    )
    route_tool = function_tool(lambda: "tool", name_override="route")
    route_handoff = handoff(Agent(name="target"), tool_name_override="route")
    model = FakeModel(initial_output=[get_function_tool_call("approval_tool", "{}")])
    agent = Agent(
        name="agent",
        model=model,
        tools=[approval_tool, route_tool],
        handoffs=[route_handoff],
    )

    initial_result = await Runner.run(agent, "Route this request")
    state = await RunState.from_json(agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])

    with pytest.raises(
        UserError,
        match="the tool name `route` is used by both a function tool and a handoff",
    ):
        await Runner.run(
            agent,
            state,
            run_config=RunConfig(tool_name_collision_policy="error"),
        )

    assert calls == []


@pytest.mark.asyncio
async def test_resume_rejects_duplicate_mcp_tools_before_approved_call() -> None:
    server = FakeMCPServer(require_approval="always")
    server.add_tool("lookup", {"type": "object", "properties": {}})
    local_tool = function_tool(lambda: "local", name_override="lookup")
    model = FakeModel(initial_output=[get_function_tool_call("lookup", "{}")])
    agent = Agent(
        name="agent",
        model=model,
        mcp_servers=[server],
    )

    initial_result = await Runner.run(agent, "Look this up")
    state = await RunState.from_json(agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])
    agent.tools = [local_tool]

    with pytest.raises(UserError, match="the tool name `lookup` is used by multiple tools"):
        await Runner.run(
            agent,
            state,
            run_config=RunConfig(tool_name_collision_policy="error"),
        )

    assert server.tool_calls == []


@pytest.mark.asyncio
async def test_resume_warn_mode_rebinds_queued_mcp_call_to_local_winner() -> None:
    calls: list[str] = []
    server = FakeMCPServer(require_approval="always")
    server.add_tool("lookup", {"type": "object", "properties": {}})

    def local_lookup() -> str:
        calls.append("local")
        return "local"

    local_tool = function_tool(local_lookup, name_override="lookup")
    model = FakeModel(initial_output=[get_function_tool_call("lookup", "{}")])
    agent = Agent(
        name="agent",
        model=model,
        mcp_servers=[server],
    )

    initial_result = await Runner.run(agent, "Look this up")
    state = await RunState.from_json(agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])
    agent.tools = [local_tool]
    model.set_next_output([get_text_message("done")])

    resumed_result = await Runner.run(agent, state)

    assert resumed_result.final_output == "done"
    assert calls == ["local"]
    assert server.tool_calls == []


@pytest.mark.asyncio
async def test_resume_rebind_drops_displaced_agent_tool_state() -> None:
    calls: list[str] = []

    sensitive_tool = function_tool(
        lambda: "sensitive",
        name_override="sensitive",
        needs_approval=True,
    )
    inner_model = FakeModel(initial_output=[get_function_tool_call("sensitive", "{}")])
    inner_agent = Agent(name="inner", model=inner_model, tools=[sensitive_tool])
    nested_tool = inner_agent.as_tool(
        tool_name="lookup",
        tool_description="Look up a value with the inner agent.",
    )
    outer_model = FakeModel(initial_output=[get_function_tool_call("lookup", '{"input":"hello"}')])
    outer_agent = Agent(name="outer", model=outer_model, tools=[nested_tool])

    initial_result = await Runner.run(outer_agent, "Look this up")
    state = await RunState.from_json(outer_agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])

    def local_lookup(input: str) -> str:
        calls.append(input)
        return "local"

    outer_agent.tools = [function_tool(local_lookup, name_override="lookup")]
    outer_model.set_next_output([get_text_message("done")])

    resumed_result = await Runner.run(outer_agent, state)

    assert resumed_result.final_output == "done"
    assert resumed_result.interruptions == []
    assert calls == ["hello"]


@pytest.mark.asyncio
async def test_resume_rebind_validates_current_winner_caller_policy() -> None:
    calls: list[str] = []
    original_tool = function_tool(
        lambda: "original",
        name_override="lookup",
        needs_approval=True,
    )
    model = FakeModel(initial_output=[get_function_tool_call("lookup", "{}")])
    agent = Agent(name="agent", model=model, tools=[original_tool])

    initial_result = await Runner.run(agent, "Look this up")
    state = await RunState.from_json(agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])

    def programmatic_only_lookup() -> str:
        calls.append("replacement")
        return "replacement"

    agent.tools = [
        function_tool(
            programmatic_only_lookup,
            name_override="lookup",
            allowed_callers=["programmatic"],
        )
    ]

    with pytest.raises(ModelBehaviorError, match="caller direct"):
        await Runner.run(agent, state)

    assert calls == []


@pytest.mark.asyncio
async def test_resume_revalidates_unchanged_winner_caller_policy() -> None:
    calls: list[str] = []

    def lookup() -> str:
        calls.append("lookup")
        return "lookup"

    approval_tool = function_tool(
        lookup,
        name_override="lookup",
        needs_approval=True,
    )
    model = FakeModel(initial_output=[get_function_tool_call("lookup", "{}")])
    agent = Agent(name="agent", model=model, tools=[approval_tool])

    initial_result = await Runner.run(agent, "Look this up")
    state = initial_result.to_state()
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])
    approval_tool.allowed_callers = ["programmatic"]

    with pytest.raises(ModelBehaviorError, match="caller direct"):
        await Runner.run(agent, state)

    assert calls == []


@pytest.mark.asyncio
async def test_resume_rejects_missing_current_function_before_stale_call() -> None:
    calls: list[str] = []

    def route_tool() -> str:
        calls.append("function")
        return "function"

    function_tool_with_approval = function_tool(
        route_tool,
        name_override="route",
        needs_approval=True,
    )
    model = FakeModel(initial_output=[get_function_tool_call("route", "{}")])
    agent = Agent(name="agent", model=model, tools=[function_tool_with_approval])

    initial_result = await Runner.run(agent, "Route this request")
    state = await RunState.from_json(agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])
    agent.tools = []

    with pytest.raises(ModelBehaviorError, match="Tool route not found in agent agent"):
        await Runner.run(agent, state)

    assert calls == []


@pytest.mark.asyncio
async def test_resume_returns_missing_current_function_error_to_model() -> None:
    calls: list[str] = []

    def route_tool() -> str:
        calls.append("function")
        return "function"

    function_tool_with_approval = function_tool(
        route_tool,
        name_override="route",
        needs_approval=True,
    )
    model = FakeModel(initial_output=[get_function_tool_call("route", "{}")])
    agent = Agent(name="agent", model=model, tools=[function_tool_with_approval])

    initial_result = await Runner.run(agent, "Route this request")
    state = await RunState.from_json(agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])
    agent.tools = []
    model.set_next_output([get_text_message("done")])

    resumed_result = await Runner.run(
        agent,
        state,
        run_config=RunConfig(tool_not_found_behavior="return_error_to_model"),
    )

    assert resumed_result.final_output == "done"
    assert calls == []


@pytest.mark.asyncio
async def test_resume_ignores_completed_function_after_tool_removal() -> None:
    calls: list[str] = []

    def completed_lookup() -> str:
        calls.append("completed")
        return "completed"

    def pending_lookup() -> str:
        calls.append("pending")
        return "pending"

    completed_tool = function_tool(completed_lookup, name_override="completed_lookup")
    pending_tool = function_tool(
        pending_lookup,
        name_override="pending_lookup",
        needs_approval=True,
    )
    model = FakeModel(
        initial_output=[
            get_function_tool_call("completed_lookup", "{}", call_id="completed_call"),
            get_function_tool_call("pending_lookup", "{}", call_id="pending_call"),
        ]
    )
    agent = Agent(name="agent", model=model, tools=[completed_tool, pending_tool])

    initial_result = await Runner.run(agent, "Look these up")
    state = initial_result.to_state()
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    assert calls == ["completed"]
    state.approve(interruptions[0])
    agent.tools = [pending_tool]
    model.set_next_output([get_text_message("done")])

    resumed_result = await Runner.run(agent, state)

    assert resumed_result.final_output == "done"
    assert calls == ["completed", "pending"]


@pytest.mark.asyncio
async def test_resume_rejects_missing_function_dropped_during_state_restore() -> None:
    calls: list[str] = []

    def lookup() -> str:
        calls.append("lookup")
        return "lookup"

    approval_tool = function_tool(
        lookup,
        name_override="lookup",
        needs_approval=True,
    )
    model = FakeModel(initial_output=[get_function_tool_call("lookup", "{}")])
    agent = Agent(name="agent", model=model, tools=[approval_tool])

    initial_result = await Runner.run(agent, "Look this up")
    state_json = initial_result.to_state().to_json()
    agent.tools = []
    state = await RunState.from_json(agent, state_json)
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])

    with pytest.raises(ModelBehaviorError, match="Tool lookup not found in agent agent"):
        await Runner.run(agent, state)

    assert calls == []


@pytest.mark.asyncio
async def test_resume_keeps_malformed_approval_only_call_pending() -> None:
    calls: list[str] = []

    def lookup() -> str:
        calls.append("lookup")
        return "lookup"

    approval_tool = function_tool(
        lookup,
        name_override="lookup",
        needs_approval=True,
    )
    model = FakeModel(initial_output=[get_function_tool_call("lookup", "{}")])
    agent = Agent(name="agent", model=model, tools=[approval_tool])

    initial_result = await Runner.run(agent, "Look this up")
    state_json = initial_result.to_state().to_json()
    state_json["last_processed_response"]["functions"] = []
    raw_approval = state_json["current_step"]["data"]["interruptions"][0]["raw_item"]
    raw_approval.pop("arguments")
    state = await RunState.from_json(agent, state_json)
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])

    resumed_result = await Runner.run(agent, state)

    assert len(resumed_result.interruptions) == 1
    assert calls == []


@pytest.mark.asyncio
async def test_resume_preserves_completed_sdk_json_tool_call() -> None:
    calls: list[str] = []

    def approved_lookup() -> str:
        calls.append("lookup")
        return "lookup"

    approval_tool = function_tool(
        approved_lookup,
        name_override="lookup",
        needs_approval=True,
    )
    model = FakeModel(
        initial_output=[
            get_function_tool_call(
                "json_tool_call",
                '{"status":"ok"}',
                call_id="json_call",
            ),
            get_function_tool_call("lookup", "{}", call_id="lookup_call"),
        ]
    )
    agent = Agent(
        name="agent",
        model=model,
        tools=[approval_tool],
        output_type=StructuredResult,
    )

    initial_result = await Runner.run(agent, "Look this up")
    state = initial_result.to_state()
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])
    model.set_next_output([get_text_message('{"status":"done"}')])

    resumed_result = await Runner.run(agent, state)

    assert resumed_result.final_output == StructuredResult(status="done")
    assert calls == ["lookup"]


@pytest.mark.asyncio
async def test_resume_rejects_collision_before_pending_handoff() -> None:
    calls: list[str] = []

    def approved_tool() -> str:
        calls.append("approved")
        return "approved"

    def record_handoff(_: RunContextWrapper[None]) -> None:
        calls.append("handoff")

    approval_tool = function_tool(
        approved_tool,
        name_override="approval_tool",
        needs_approval=True,
    )
    route_tool = function_tool(lambda: "tool", name_override="route")
    target = Agent(name="target")
    route_handoff = handoff(
        target,
        tool_name_override="route",
        on_handoff=record_handoff,
    )
    model = FakeModel(
        initial_output=[
            get_function_tool_call("approval_tool", "{}"),
            get_handoff_tool_call(target, override_name="route", args="{}"),
        ]
    )
    agent = Agent(
        name="agent",
        model=model,
        tools=[approval_tool, route_tool],
        handoffs=[route_handoff],
    )

    initial_result = await Runner.run(agent, "Route this request")
    state = await RunState.from_json(agent, initial_result.to_state().to_json())
    interruptions = state.get_interruptions()
    assert len(interruptions) == 1
    state.approve(interruptions[0])

    with pytest.raises(
        UserError,
        match="the tool name `route` is used by both a function tool and a handoff",
    ):
        await Runner.run(
            agent,
            state,
            run_config=RunConfig(tool_name_collision_policy="error"),
        )

    assert calls == []
