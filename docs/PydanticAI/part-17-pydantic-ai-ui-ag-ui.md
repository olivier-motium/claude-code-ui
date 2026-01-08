<!-- Source: _complete-doc.md -->

# `pydantic_ai.ui.ag_ui`

AG-UI protocol integration for Pydantic AI agents.

### AGUIAdapter

Bases: `UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]`

UI adapter for the Agent-User Interaction (AG-UI) protocol.

Source code in `pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py`

```python
class AGUIAdapter(UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]):
    """UI adapter for the Agent-User Interaction (AG-UI) protocol."""

    @classmethod
    def build_run_input(cls, body: bytes) -> RunAgentInput:
        """Build an AG-UI run input object from the request body."""
        return RunAgentInput.model_validate_json(body)

    def build_event_stream(self) -> UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]:
        """Build an AG-UI event stream transformer."""
        return AGUIEventStream(self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the AG-UI run input."""
        return self.load_messages(self.run_input.messages)

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Toolset representing frontend tools from the AG-UI run input."""
        if self.run_input.tools:
            return _AGUIFrontendToolset[AgentDepsT](self.run_input.tools)
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Frontend state from the AG-UI run input."""
        state = self.run_input.state
        if state is None:
            return None

        if isinstance(state, Mapping) and not state:
            return None

        return cast('dict[str, Any]', state)

    @classmethod
    def load_messages(cls, messages: Sequence[Message]) -> list[ModelMessage]:
        """Transform AG-UI messages into Pydantic AI messages."""
        builder = MessagesBuilder()
        tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.

        for msg in messages:
            if isinstance(msg, UserMessage | SystemMessage | DeveloperMessage) or (
                isinstance(msg, ToolMessage) and not msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
            ):
                if isinstance(msg, UserMessage):
                    builder.add(UserPromptPart(content=msg.content))
                elif isinstance(msg, SystemMessage | DeveloperMessage):
                    builder.add(SystemPromptPart(content=msg.content))
                else:
                    tool_call_id = msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                    builder.add(
                        ToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=tool_call_id,
                        )
                    )

            elif isinstance(msg, AssistantMessage) or (  # pragma: no branch
                isinstance(msg, ToolMessage) and msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
            ):
                if isinstance(msg, AssistantMessage):
                    if msg.content:
                        builder.add(TextPart(content=msg.content))

                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_call_id = tool_call.id
                            tool_name = tool_call.function.name
                            tool_calls[tool_call_id] = tool_name

                            if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
                                _, provider_name, tool_call_id = tool_call_id.split('|', 2)
                                builder.add(
                                    BuiltinToolCallPart(
                                        tool_name=tool_name,
                                        args=tool_call.function.arguments,
                                        tool_call_id=tool_call_id,
                                        provider_name=provider_name,
                                    )
                                )
                            else:
                                builder.add(
                                    ToolCallPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        args=tool_call.function.arguments,
                                    )
                                )
                else:
                    tool_call_id = msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')
                    _, provider_name, tool_call_id = tool_call_id.split('|', 2)

                    builder.add(
                        BuiltinToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=tool_call_id,
                            provider_name=provider_name,
                        )
                    )

        return builder.messages

```

#### build_run_input

```python
build_run_input(body: bytes) -> RunAgentInput

```

Build an AG-UI run input object from the request body.

Source code in `pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py`

```python
@classmethod
def build_run_input(cls, body: bytes) -> RunAgentInput:
    """Build an AG-UI run input object from the request body."""
    return RunAgentInput.model_validate_json(body)

```

#### build_event_stream

```python
build_event_stream() -> (
    UIEventStream[
        RunAgentInput, BaseEvent, AgentDepsT, OutputDataT
    ]
)

```

Build an AG-UI event stream transformer.

Source code in `pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py`

```python
def build_event_stream(self) -> UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]:
    """Build an AG-UI event stream transformer."""
    return AGUIEventStream(self.run_input, accept=self.accept)

```

#### messages

```python
messages: list[ModelMessage]

```

Pydantic AI messages from the AG-UI run input.

#### toolset

```python
toolset: AbstractToolset[AgentDepsT] | None

```

Toolset representing frontend tools from the AG-UI run input.

#### state

```python
state: dict[str, Any] | None

```

Frontend state from the AG-UI run input.

#### load_messages

```python
load_messages(
    messages: Sequence[Message],
) -> list[ModelMessage]

```

Transform AG-UI messages into Pydantic AI messages.

Source code in `pydantic_ai_slim/pydantic_ai/ui/ag_ui/_adapter.py`

```python
@classmethod
def load_messages(cls, messages: Sequence[Message]) -> list[ModelMessage]:
    """Transform AG-UI messages into Pydantic AI messages."""
    builder = MessagesBuilder()
    tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.

    for msg in messages:
        if isinstance(msg, UserMessage | SystemMessage | DeveloperMessage) or (
            isinstance(msg, ToolMessage) and not msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
        ):
            if isinstance(msg, UserMessage):
                builder.add(UserPromptPart(content=msg.content))
            elif isinstance(msg, SystemMessage | DeveloperMessage):
                builder.add(SystemPromptPart(content=msg.content))
            else:
                tool_call_id = msg.tool_call_id
                tool_name = tool_calls.get(tool_call_id)
                if tool_name is None:  # pragma: no cover
                    raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                builder.add(
                    ToolReturnPart(
                        tool_name=tool_name,
                        content=msg.content,
                        tool_call_id=tool_call_id,
                    )
                )

        elif isinstance(msg, AssistantMessage) or (  # pragma: no branch
            isinstance(msg, ToolMessage) and msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
        ):
            if isinstance(msg, AssistantMessage):
                if msg.content:
                    builder.add(TextPart(content=msg.content))

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_call_id = tool_call.id
                        tool_name = tool_call.function.name
                        tool_calls[tool_call_id] = tool_name

                        if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
                            _, provider_name, tool_call_id = tool_call_id.split('|', 2)
                            builder.add(
                                BuiltinToolCallPart(
                                    tool_name=tool_name,
                                    args=tool_call.function.arguments,
                                    tool_call_id=tool_call_id,
                                    provider_name=provider_name,
                                )
                            )
                        else:
                            builder.add(
                                ToolCallPart(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    args=tool_call.function.arguments,
                                )
                            )
            else:
                tool_call_id = msg.tool_call_id
                tool_name = tool_calls.get(tool_call_id)
                if tool_name is None:  # pragma: no cover
                    raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')
                _, provider_name, tool_call_id = tool_call_id.split('|', 2)

                builder.add(
                    BuiltinToolReturnPart(
                        tool_name=tool_name,
                        content=msg.content,
                        tool_call_id=tool_call_id,
                        provider_name=provider_name,
                    )
                )

    return builder.messages

```

### AGUIEventStream

Bases: `UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]`

UI event stream transformer for the Agent-User Interaction (AG-UI) protocol.

Source code in `pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py`

```python
@dataclass
class AGUIEventStream(UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the Agent-User Interaction (AG-UI) protocol."""

    _thinking_text: bool = False
    _builtin_tool_call_ids: dict[str, str] = field(default_factory=dict)
    _error: bool = False

    @property
    def _event_encoder(self) -> EventEncoder:
        return EventEncoder(accept=self.accept or SSE_CONTENT_TYPE)

    @property
    def content_type(self) -> str:
        return self._event_encoder.get_content_type()

    def encode_event(self, event: BaseEvent) -> str:
        return self._event_encoder.encode(event)

    async def before_stream(self) -> AsyncIterator[BaseEvent]:
        yield RunStartedEvent(
            thread_id=self.run_input.thread_id,
            run_id=self.run_input.run_id,
        )

    async def before_response(self) -> AsyncIterator[BaseEvent]:
        # Prevent parts from a subsequent response being tied to parts from an earlier response.
        # See https://github.com/pydantic/pydantic-ai/issues/3316
        self.new_message_id()
        return
        yield  # Make this an async generator

    async def after_stream(self) -> AsyncIterator[BaseEvent]:
        if not self._error:
            yield RunFinishedEvent(
                thread_id=self.run_input.thread_id,
                run_id=self.run_input.run_id,
            )

    async def on_error(self, error: Exception) -> AsyncIterator[BaseEvent]:
        self._error = True
        yield RunErrorEvent(message=str(error))

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseEvent]:
        if follows_text:
            message_id = self.message_id
        else:
            message_id = self.new_message_id()
            yield TextMessageStartEvent(message_id=message_id)

        if part.content:  # pragma: no branch
            yield TextMessageContentEvent(message_id=message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseEvent]:
        if delta.content_delta:  # pragma: no branch
            yield TextMessageContentEvent(message_id=self.message_id, delta=delta.content_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[BaseEvent]:
        if not followed_by_text:
            yield TextMessageEndEvent(message_id=self.message_id)

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[BaseEvent]:
        if not follows_thinking:
            yield ThinkingStartEvent(type=EventType.THINKING_START)

        if part.content:
            yield ThinkingTextMessageStartEvent(type=EventType.THINKING_TEXT_MESSAGE_START)
            yield ThinkingTextMessageContentEvent(type=EventType.THINKING_TEXT_MESSAGE_CONTENT, delta=part.content)
            self._thinking_text = True

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseEvent]:
        if not delta.content_delta:
            return  # pragma: no cover

        if not self._thinking_text:
            yield ThinkingTextMessageStartEvent(type=EventType.THINKING_TEXT_MESSAGE_START)
            self._thinking_text = True

        yield ThinkingTextMessageContentEvent(type=EventType.THINKING_TEXT_MESSAGE_CONTENT, delta=delta.content_delta)

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[BaseEvent]:
        if self._thinking_text:
            yield ThinkingTextMessageEndEvent(type=EventType.THINKING_TEXT_MESSAGE_END)
            self._thinking_text = False

        if not followed_by_thinking:
            yield ThinkingEndEvent(type=EventType.THINKING_END)

    def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        return self._handle_tool_call_start(part)

    def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        tool_call_id = part.tool_call_id
        builtin_tool_call_id = '|'.join([BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', tool_call_id])
        self._builtin_tool_call_ids[tool_call_id] = builtin_tool_call_id
        tool_call_id = builtin_tool_call_id

        return self._handle_tool_call_start(part, tool_call_id)

    async def _handle_tool_call_start(
        self, part: ToolCallPart | BuiltinToolCallPart, tool_call_id: str | None = None
    ) -> AsyncIterator[BaseEvent]:
        tool_call_id = tool_call_id or part.tool_call_id
        parent_message_id = self.message_id

        yield ToolCallStartEvent(
            tool_call_id=tool_call_id, tool_call_name=part.tool_name, parent_message_id=parent_message_id
        )
        if part.args:
            yield ToolCallArgsEvent(tool_call_id=tool_call_id, delta=part.args_as_json_str())

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseEvent]:
        tool_call_id = delta.tool_call_id
        assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
        if tool_call_id in self._builtin_tool_call_ids:
            tool_call_id = self._builtin_tool_call_ids[tool_call_id]
        yield ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            delta=delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta),
        )

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[BaseEvent]:
        yield ToolCallEndEvent(tool_call_id=part.tool_call_id)

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        yield ToolCallEndEvent(tool_call_id=self._builtin_tool_call_ids[part.tool_call_id])

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[BaseEvent]:
        tool_call_id = self._builtin_tool_call_ids[part.tool_call_id]
        yield ToolCallResultEvent(
            message_id=self.new_message_id(),
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=tool_call_id,
            content=part.model_response_str(),
        )

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseEvent]:
        result = event.result
        output = result.model_response() if isinstance(result, RetryPromptPart) else result.model_response_str()

        yield ToolCallResultEvent(
            message_id=self.new_message_id(),
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=result.tool_call_id,
            content=output,
        )

        # ToolCallResultEvent.content may hold user parts (e.g. text, images) that AG-UI does not currently have events for

        if isinstance(result, ToolReturnPart):
            # Check for AG-UI events returned by tool calls.
            possible_event = result.metadata or result.content
            if isinstance(possible_event, BaseEvent):
                yield possible_event
            elif isinstance(possible_event, str | bytes):  # pragma: no branch
                # Avoid iterable check for strings and bytes.
                pass
            elif isinstance(possible_event, Iterable):  # pragma: no branch
                for item in possible_event:  # type: ignore[reportUnknownMemberType]
                    if isinstance(item, BaseEvent):  # pragma: no branch
                        yield item

```

AG-UI protocol integration for Pydantic AI agents.

### AGUIApp

Bases: `Generic[AgentDepsT, OutputDataT]`, `Starlette`

ASGI application for running Pydantic AI agents with AG-UI protocol support.

Source code in `pydantic_ai_slim/pydantic_ai/ui/ag_ui/app.py`

```python
class AGUIApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """ASGI application for running Pydantic AI agents with AG-UI protocol support."""

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        # AGUIAdapter.dispatch_request parameters
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[Any] | None = None,
        # Starlette parameters
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[Self] | None = None,
    ) -> None:
        """An ASGI application that handles every request by running the agent and streaming the response.

        Note that the `deps` will be the same for each request, with the exception of the frontend state that's
        injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ui.StateHandler] protocol.
        To provide different `deps` for each request (e.g. based on the authenticated user),
        use [`AGUIAdapter.run_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.run_stream] or
        [`AGUIAdapter.dispatch_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.dispatch_request] instead.

        Args:
            agent: The agent to run.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request. A starlette application will always
                automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
                outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
                `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
                exception cases occurring in the routing or endpoints.
            exception_handlers: A mapping of either integer status codes, or exception class types onto
                callables which handle the exceptions. Exception handler callables should be of the form
                `handler(request, exc) -> response` and may be either standard functions, or async functions.
            on_startup: A list of callables to run on application startup. Startup handler callables do not
                take any arguments, and may be either standard functions, or async functions.
            on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
                not take any arguments, and may be either standard functions, or async functions.
            lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
                This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
                the other, not both.
        """
        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )

        async def run_agent(request: Request) -> Response:
            """Endpoint to run the agent with the provided input data."""
            # `dispatch_request` will store the frontend state from the request on `deps.state` (if it implements the `StateHandler` protocol),
            # so we need to copy the deps to avoid different requests mutating the same deps object.
            nonlocal deps
            if isinstance(deps, StateHandler):  # pragma: no branch
                deps = replace(deps)

            return await AGUIAdapter[AgentDepsT, OutputDataT].dispatch_request(
                request,
                agent=agent,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                on_complete=on_complete,
            )

        self.router.add_route('/', run_agent, methods=['POST'])

```

#### __init__

```python
__init__(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    on_complete: OnCompleteFunc[Any] | None = None,
    debug: bool = False,
    routes: Sequence[BaseRoute] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: (
        Mapping[Any, ExceptionHandler] | None
    ) = None,
    on_startup: Sequence[Callable[[], Any]] | None = None,
    on_shutdown: Sequence[Callable[[], Any]] | None = None,
    lifespan: Lifespan[Self] | None = None
) -> None

```

An ASGI application that handles every request by running the agent and streaming the response.

Note that the `deps` will be the same for each request, with the exception of the frontend state that's injected into the `state` field of a `deps` object that implements the StateHandler protocol. To provide different `deps` for each request (e.g. based on the authenticated user), use AGUIAdapter.run_stream() or AGUIAdapter.dispatch_request() instead.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `agent` | `AbstractAgent[AgentDepsT, OutputDataT]` | The agent to run. | *required* | | `output_type` | `OutputSpec[Any] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` | | `on_complete` | `OnCompleteFunc[Any] | None` | Optional callback function called when the agent run completes successfully. The callback receives the completed AgentRunResult and can access all_messages() and other result data. | `None` | | `debug` | `bool` | Boolean indicating if debug tracebacks should be returned on errors. | `False` | | `routes` | `Sequence[BaseRoute] | None` | A list of routes to serve incoming HTTP and WebSocket requests. | `None` | | `middleware` | `Sequence[Middleware] | None` | A list of middleware to run for every request. A starlette application will always automatically include two middleware classes. ServerErrorMiddleware is added as the very outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack. ExceptionMiddleware is added as the very innermost middleware, to deal with handled exception cases occurring in the routing or endpoints. | `None` | | `exception_handlers` | `Mapping[Any, ExceptionHandler] | None` | A mapping of either integer status codes, or exception class types onto callables which handle the exceptions. Exception handler callables should be of the form handler(request, exc) -> response and may be either standard functions, or async functions. | `None` | | `on_startup` | `Sequence[Callable[[], Any]] | None` | A list of callables to run on application startup. Startup handler callables do not take any arguments, and may be either standard functions, or async functions. | `None` | | `on_shutdown` | `Sequence[Callable[[], Any]] | None` | A list of callables to run on application shutdown. Shutdown handler callables do not take any arguments, and may be either standard functions, or async functions. | `None` | | `lifespan` | `Lifespan[Self] | None` | A lifespan context function, which can be used to perform startup and shutdown tasks. This is a newer style that replaces the on_startup and on_shutdown handlers. Use one or the other, not both. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/ag_ui/app.py`

```python
def __init__(
    self,
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    # AGUIAdapter.dispatch_request parameters
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    on_complete: OnCompleteFunc[Any] | None = None,
    # Starlette parameters
    debug: bool = False,
    routes: Sequence[BaseRoute] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
    on_startup: Sequence[Callable[[], Any]] | None = None,
    on_shutdown: Sequence[Callable[[], Any]] | None = None,
    lifespan: Lifespan[Self] | None = None,
) -> None:
    """An ASGI application that handles every request by running the agent and streaming the response.

    Note that the `deps` will be the same for each request, with the exception of the frontend state that's
    injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ui.StateHandler] protocol.
    To provide different `deps` for each request (e.g. based on the authenticated user),
    use [`AGUIAdapter.run_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.run_stream] or
    [`AGUIAdapter.dispatch_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.dispatch_request] instead.

    Args:
        agent: The agent to run.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
            no output validators since output validators would expect an argument that matches the agent's
            output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

        debug: Boolean indicating if debug tracebacks should be returned on errors.
        routes: A list of routes to serve incoming HTTP and WebSocket requests.
        middleware: A list of middleware to run for every request. A starlette application will always
            automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
            outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
            `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
            exception cases occurring in the routing or endpoints.
        exception_handlers: A mapping of either integer status codes, or exception class types onto
            callables which handle the exceptions. Exception handler callables should be of the form
            `handler(request, exc) -> response` and may be either standard functions, or async functions.
        on_startup: A list of callables to run on application startup. Startup handler callables do not
            take any arguments, and may be either standard functions, or async functions.
        on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
            not take any arguments, and may be either standard functions, or async functions.
        lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
            This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
            the other, not both.
    """
    super().__init__(
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        lifespan=lifespan,
    )

    async def run_agent(request: Request) -> Response:
        """Endpoint to run the agent with the provided input data."""
        # `dispatch_request` will store the frontend state from the request on `deps.state` (if it implements the `StateHandler` protocol),
        # so we need to copy the deps to avoid different requests mutating the same deps object.
        nonlocal deps
        if isinstance(deps, StateHandler):  # pragma: no branch
            deps = replace(deps)

        return await AGUIAdapter[AgentDepsT, OutputDataT].dispatch_request(
            request,
            agent=agent,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        )

    self.router.add_route('/', run_agent, methods=['POST'])

```

# `pydantic_ai.ui`

### StateDeps

Bases: `Generic[StateT]`

Dependency type that holds state.

This class is used to manage the state of an agent run. It allows setting the state of the agent run with a specific type of state model, which must be a subclass of `BaseModel`.

The state is set using the `state` setter by the `Adapter` when the run starts.

Implements the `StateHandler` protocol.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@dataclass
class StateDeps(Generic[StateT]):
    """Dependency type that holds state.

    This class is used to manage the state of an agent run. It allows setting
    the state of the agent run with a specific type of state model, which must
    be a subclass of `BaseModel`.

    The state is set using the `state` setter by the `Adapter` when the run starts.

    Implements the `StateHandler` protocol.
    """

    state: StateT

```

### StateHandler

Bases: `Protocol`

Protocol for state handlers in agent runs. Requires the class to be a dataclass with a `state` field.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs. Requires the class to be a dataclass with a `state` field."""

    # Has to be a dataclass so we can use `replace` to update the state.
    # From https://github.com/python/typeshed/blob/9ab7fde0a0cd24ed7a72837fcb21093b811b80d8/stdlib/_typeshed/__init__.pyi#L352
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def state(self) -> Any:
        """Get the current state of the agent run."""
        ...

    @state.setter
    def state(self, state: Any) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Args:
            state: The run state.
        """
        ...

```

#### state

```python
state: Any

```

Get the current state of the agent run.

### UIAdapter

Bases: `ABC`, `Generic[RunInputT, MessageT, EventT, AgentDepsT, OutputDataT]`

Base class for UI adapters.

This class is responsible for transforming agent run input received from the frontend into arguments for Agent.run_stream_events(), running the agent, and then transforming Pydantic AI events into protocol-specific events.

The event stream transformation is handled by a protocol-specific UIEventStream subclass.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@dataclass
class UIAdapter(ABC, Generic[RunInputT, MessageT, EventT, AgentDepsT, OutputDataT]):
    """Base class for UI adapters.

    This class is responsible for transforming agent run input received from the frontend into arguments for [`Agent.run_stream_events()`][pydantic_ai.Agent.run_stream_events], running the agent, and then transforming Pydantic AI events into protocol-specific events.

    The event stream transformation is handled by a protocol-specific [`UIEventStream`][pydantic_ai.ui.UIEventStream] subclass.
    """

    agent: AbstractAgent[AgentDepsT, OutputDataT]
    """The Pydantic AI agent to run."""

    run_input: RunInputT
    """The protocol-specific run input object."""

    _: KW_ONLY

    accept: str | None = None
    """The `Accept` header value of the request, used to determine how to encode the protocol-specific events for the streaming response."""

    @classmethod
    async def from_request(
        cls, request: Request, *, agent: AbstractAgent[AgentDepsT, OutputDataT]
    ) -> UIAdapter[RunInputT, MessageT, EventT, AgentDepsT, OutputDataT]:
        """Create an adapter from a request."""
        return cls(
            agent=agent,
            run_input=cls.build_run_input(await request.body()),
            accept=request.headers.get('accept'),
        )

    @classmethod
    @abstractmethod
    def build_run_input(cls, body: bytes) -> RunInputT:
        """Build a protocol-specific run input object from the request body."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_messages(cls, messages: Sequence[MessageT]) -> list[ModelMessage]:
        """Transform protocol-specific messages into Pydantic AI messages."""
        raise NotImplementedError

    @abstractmethod
    def build_event_stream(self) -> UIEventStream[RunInputT, EventT, AgentDepsT, OutputDataT]:
        """Build a protocol-specific event stream transformer."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the protocol-specific run input."""
        raise NotImplementedError

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Toolset representing frontend tools from the protocol-specific run input."""
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Frontend state from the protocol-specific run input."""
        return None

    def transform_stream(
        self,
        stream: AsyncIterator[NativeEvent],
        on_complete: OnCompleteFunc[EventT] | None = None,
    ) -> AsyncIterator[EventT]:
        """Transform a stream of Pydantic AI events into protocol-specific events.

        Args:
            stream: The stream of Pydantic AI events to transform.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
        """
        return self.build_event_stream().transform_stream(stream, on_complete=on_complete)

    def encode_stream(self, stream: AsyncIterator[EventT]) -> AsyncIterator[str]:
        """Encode a stream of protocol-specific events as strings according to the `Accept` header value.

        Args:
            stream: The stream of protocol-specific events to encode.
        """
        return self.build_event_stream().encode_stream(stream)

    def streaming_response(self, stream: AsyncIterator[EventT]) -> StreamingResponse:
        """Generate a streaming response from a stream of protocol-specific events.

        Args:
            stream: The stream of protocol-specific events to encode.
        """
        return self.build_event_stream().streaming_response(stream)

    def run_stream_native(
        self,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[NativeEvent]:
        """Run the agent with the protocol-specific run input and stream Pydantic AI events.

        Args:
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools to use for this run.
        """
        message_history = [*(message_history or []), *self.messages]

        toolset = self.toolset
        if toolset:
            output_type = [output_type or self.agent.output_type, DeferredToolRequests]
            toolsets = [*(toolsets or []), toolset]

        if isinstance(deps, StateHandler):
            raw_state = self.state or {}
            if isinstance(deps.state, BaseModel):
                state = type(deps.state).model_validate(raw_state)
            else:
                state = raw_state

            deps.state = state
        elif self.state:
            warnings.warn(
                f'State was provided but `deps` of type `{type(deps).__name__}` does not implement the `StateHandler` protocol, so the state was ignored. Use `StateDeps[...]` or implement `StateHandler` to receive AG-UI state.',
                UserWarning,
                stacklevel=2,
            )

        return self.agent.run_stream_events(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            instructions=instructions,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        )

    def run_stream(
        self,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[EventT] | None = None,
    ) -> AsyncIterator[EventT]:
        """Run the agent with the protocol-specific run input and stream protocol-specific events.

        Args:
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools to use for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
        """
        return self.transform_stream(
            self.run_stream_native(
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
            ),
            on_complete=on_complete,
        )

    @classmethod
    async def dispatch_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        output_type: OutputSpec[Any] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[EventT] | None = None,
    ) -> Response:
        """Handle a protocol-specific HTTP request by running the agent and returning a streaming response of protocol-specific events.

        Args:
            request: The incoming Starlette/FastAPI request.
            agent: The agent to run.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools to use for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.

        Returns:
            A streaming Starlette response with protocol-specific events encoded per the request's `Accept` header value.
        """
        try:
            from starlette.responses import Response
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'Please install the `starlette` package to use `dispatch_request()` method, '
                'you can use the `ui` optional group â€” `pip install "pydantic-ai-slim[ui]"`'
            ) from e

        try:
            adapter = await cls.from_request(request, agent=agent)
        except ValidationError as e:  # pragma: no cover
            return Response(
                content=e.json(),
                media_type='application/json',
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        return adapter.streaming_response(
            adapter.run_stream(
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                deps=deps,
                output_type=output_type,
                model=model,
                instructions=instructions,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                on_complete=on_complete,
            ),
        )

```

#### agent

```python
agent: AbstractAgent[AgentDepsT, OutputDataT]

```

The Pydantic AI agent to run.

#### run_input

```python
run_input: RunInputT

```

The protocol-specific run input object.

#### accept

```python
accept: str | None = None

```

The `Accept` header value of the request, used to determine how to encode the protocol-specific events for the streaming response.

#### from_request

```python
from_request(
    request: Request,
    *,
    agent: AbstractAgent[AgentDepsT, OutputDataT]
) -> UIAdapter[
    RunInputT, MessageT, EventT, AgentDepsT, OutputDataT
]

```

Create an adapter from a request.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@classmethod
async def from_request(
    cls, request: Request, *, agent: AbstractAgent[AgentDepsT, OutputDataT]
) -> UIAdapter[RunInputT, MessageT, EventT, AgentDepsT, OutputDataT]:
    """Create an adapter from a request."""
    return cls(
        agent=agent,
        run_input=cls.build_run_input(await request.body()),
        accept=request.headers.get('accept'),
    )

```

#### build_run_input

```python
build_run_input(body: bytes) -> RunInputT

```

Build a protocol-specific run input object from the request body.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@classmethod
@abstractmethod
def build_run_input(cls, body: bytes) -> RunInputT:
    """Build a protocol-specific run input object from the request body."""
    raise NotImplementedError

```

#### load_messages

```python
load_messages(
    messages: Sequence[MessageT],
) -> list[ModelMessage]

```

Transform protocol-specific messages into Pydantic AI messages.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@classmethod
@abstractmethod
def load_messages(cls, messages: Sequence[MessageT]) -> list[ModelMessage]:
    """Transform protocol-specific messages into Pydantic AI messages."""
    raise NotImplementedError

```

#### build_event_stream

```python
build_event_stream() -> (
    UIEventStream[
        RunInputT, EventT, AgentDepsT, OutputDataT
    ]
)

```

Build a protocol-specific event stream transformer.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@abstractmethod
def build_event_stream(self) -> UIEventStream[RunInputT, EventT, AgentDepsT, OutputDataT]:
    """Build a protocol-specific event stream transformer."""
    raise NotImplementedError

```

#### messages

```python
messages: list[ModelMessage]

```

Pydantic AI messages from the protocol-specific run input.

#### toolset

```python
toolset: AbstractToolset[AgentDepsT] | None

```

Toolset representing frontend tools from the protocol-specific run input.

#### state

```python
state: dict[str, Any] | None

```

Frontend state from the protocol-specific run input.

#### transform_stream

```python
transform_stream(
    stream: AsyncIterator[NativeEvent],
    on_complete: OnCompleteFunc[EventT] | None = None,
) -> AsyncIterator[EventT]

```

Transform a stream of Pydantic AI events into protocol-specific events.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `stream` | `AsyncIterator[NativeEvent]` | The stream of Pydantic AI events to transform. | *required* | | `on_complete` | `OnCompleteFunc[EventT] | None` | Optional callback function called when the agent run completes successfully. The callback receives the completed AgentRunResult and can optionally yield additional protocol-specific events. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
def transform_stream(
    self,
    stream: AsyncIterator[NativeEvent],
    on_complete: OnCompleteFunc[EventT] | None = None,
) -> AsyncIterator[EventT]:
    """Transform a stream of Pydantic AI events into protocol-specific events.

    Args:
        stream: The stream of Pydantic AI events to transform.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
    """
    return self.build_event_stream().transform_stream(stream, on_complete=on_complete)

```

#### encode_stream

```python
encode_stream(
    stream: AsyncIterator[EventT],
) -> AsyncIterator[str]

```

Encode a stream of protocol-specific events as strings according to the `Accept` header value.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `stream` | `AsyncIterator[EventT]` | The stream of protocol-specific events to encode. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
def encode_stream(self, stream: AsyncIterator[EventT]) -> AsyncIterator[str]:
    """Encode a stream of protocol-specific events as strings according to the `Accept` header value.

    Args:
        stream: The stream of protocol-specific events to encode.
    """
    return self.build_event_stream().encode_stream(stream)

```

#### streaming_response

```python
streaming_response(
    stream: AsyncIterator[EventT],
) -> StreamingResponse

```

Generate a streaming response from a stream of protocol-specific events.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `stream` | `AsyncIterator[EventT]` | The stream of protocol-specific events to encode. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
def streaming_response(self, stream: AsyncIterator[EventT]) -> StreamingResponse:
    """Generate a streaming response from a stream of protocol-specific events.

    Args:
        stream: The stream of protocol-specific events to encode.
    """
    return self.build_event_stream().streaming_response(stream)

```

#### run_stream_native

```python
run_stream_native(
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AsyncIterator[NativeEvent]

```

Run the agent with the protocol-specific run input and stream Pydantic AI events.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_type` | `OutputSpec[Any] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools to use for this run. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
def run_stream_native(
    self,
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
) -> AsyncIterator[NativeEvent]:
    """Run the agent with the protocol-specific run input and stream Pydantic AI events.

    Args:
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools to use for this run.
    """
    message_history = [*(message_history or []), *self.messages]

    toolset = self.toolset
    if toolset:
        output_type = [output_type or self.agent.output_type, DeferredToolRequests]
        toolsets = [*(toolsets or []), toolset]

    if isinstance(deps, StateHandler):
        raw_state = self.state or {}
        if isinstance(deps.state, BaseModel):
            state = type(deps.state).model_validate(raw_state)
        else:
            state = raw_state

        deps.state = state
    elif self.state:
        warnings.warn(
            f'State was provided but `deps` of type `{type(deps).__name__}` does not implement the `StateHandler` protocol, so the state was ignored. Use `StateDeps[...]` or implement `StateHandler` to receive AG-UI state.',
            UserWarning,
            stacklevel=2,
        )

    return self.agent.run_stream_events(
        output_type=output_type,
        message_history=message_history,
        deferred_tool_results=deferred_tool_results,
        model=model,
        deps=deps,
        model_settings=model_settings,
        instructions=instructions,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
        builtin_tools=builtin_tools,
    )

```

#### run_stream

```python
run_stream(
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    on_complete: OnCompleteFunc[EventT] | None = None
) -> AsyncIterator[EventT]

```

Run the agent with the protocol-specific run input and stream protocol-specific events.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_type` | `OutputSpec[Any] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools to use for this run. | `None` | | `on_complete` | `OnCompleteFunc[EventT] | None` | Optional callback function called when the agent run completes successfully. The callback receives the completed AgentRunResult and can optionally yield additional protocol-specific events. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
def run_stream(
    self,
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    on_complete: OnCompleteFunc[EventT] | None = None,
) -> AsyncIterator[EventT]:
    """Run the agent with the protocol-specific run input and stream protocol-specific events.

    Args:
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools to use for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
    """
    return self.transform_stream(
        self.run_stream_native(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        ),
        on_complete=on_complete,
    )

```

#### dispatch_request

```python
dispatch_request(
    request: Request,
    *,
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    output_type: OutputSpec[Any] | None = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    on_complete: OnCompleteFunc[EventT] | None = None
) -> Response

```

Handle a protocol-specific HTTP request by running the agent and returning a streaming response of protocol-specific events.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `request` | `Request` | The incoming Starlette/FastAPI request. | *required* | | `agent` | `AbstractAgent[AgentDepsT, OutputDataT]` | The agent to run. | *required* | | `output_type` | `OutputSpec[Any] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools to use for this run. | `None` | | `on_complete` | `OnCompleteFunc[EventT] | None` | Optional callback function called when the agent run completes successfully. The callback receives the completed AgentRunResult and can optionally yield additional protocol-specific events. | `None` |

Returns:

| Type | Description | | --- | --- | | `Response` | A streaming Starlette response with protocol-specific events encoded per the request's Accept header value. |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@classmethod
async def dispatch_request(
    cls,
    request: Request,
    *,
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    output_type: OutputSpec[Any] | None = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    on_complete: OnCompleteFunc[EventT] | None = None,
) -> Response:
    """Handle a protocol-specific HTTP request by running the agent and returning a streaming response of protocol-specific events.

    Args:
        request: The incoming Starlette/FastAPI request.
        agent: The agent to run.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools to use for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.

    Returns:
        A streaming Starlette response with protocol-specific events encoded per the request's `Accept` header value.
    """
    try:
        from starlette.responses import Response
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'Please install the `starlette` package to use `dispatch_request()` method, '
            'you can use the `ui` optional group â€” `pip install "pydantic-ai-slim[ui]"`'
        ) from e

    try:
        adapter = await cls.from_request(request, agent=agent)
    except ValidationError as e:  # pragma: no cover
        return Response(
            content=e.json(),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    return adapter.streaming_response(
        adapter.run_stream(
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            deps=deps,
            output_type=output_type,
            model=model,
            instructions=instructions,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        ),
    )

```

### SSE_CONTENT_TYPE

```python
SSE_CONTENT_TYPE = 'text/event-stream'

```

Content type header value for Server-Sent Events (SSE).

### NativeEvent

```python
NativeEvent: TypeAlias = (
    AgentStreamEvent | AgentRunResultEvent[Any]
)

```

Type alias for the native event type, which is either an `AgentStreamEvent` or an `AgentRunResultEvent`.

### OnCompleteFunc

```python
OnCompleteFunc: TypeAlias = (
    Callable[[AgentRunResult[Any]], None]
    | Callable[[AgentRunResult[Any]], Awaitable[None]]
    | Callable[[AgentRunResult[Any]], AsyncIterator[EventT]]
)

```

Callback function type that receives the `AgentRunResult` of the completed run. Can be sync, async, or an async generator of protocol-specific events.

### UIEventStream

Bases: `ABC`, `Generic[RunInputT, EventT, AgentDepsT, OutputDataT]`

Base class for UI event stream transformers.

This class is responsible for transforming Pydantic AI events into protocol-specific events.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
@dataclass
class UIEventStream(ABC, Generic[RunInputT, EventT, AgentDepsT, OutputDataT]):
    """Base class for UI event stream transformers.

    This class is responsible for transforming Pydantic AI events into protocol-specific events.
    """

    run_input: RunInputT

    accept: str | None = None
    """The `Accept` header value of the request, used to determine how to encode the protocol-specific events for the streaming response."""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    """The message ID to use for the next event."""

    _turn: Literal['request', 'response'] | None = None

    _result: AgentRunResult[OutputDataT] | None = None
    _final_result_event: FinalResultEvent | None = None

    def new_message_id(self) -> str:
        """Generate and store a new message ID."""
        self.message_id = str(uuid4())
        return self.message_id

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        """Response headers to return to the frontend."""
        return None

    @property
    def content_type(self) -> str:
        """Get the content type for the event stream, compatible with the `Accept` header value.

        By default, this returns the Server-Sent Events content type (`text/event-stream`).
        If a subclass supports other types as well, it should consider `self.accept` in [`encode_event()`][pydantic_ai.ui.UIEventStream.encode_event] and return the resulting content type.
        """
        return SSE_CONTENT_TYPE

    @abstractmethod
    def encode_event(self, event: EventT) -> str:
        """Encode a protocol-specific event as a string."""
        raise NotImplementedError

    async def encode_stream(self, stream: AsyncIterator[EventT]) -> AsyncIterator[str]:
        """Encode a stream of protocol-specific events as strings according to the `Accept` header value."""
        async for event in stream:
            yield self.encode_event(event)

    def streaming_response(self, stream: AsyncIterator[EventT]) -> StreamingResponse:
        """Generate a streaming response from a stream of protocol-specific events."""
        try:
            from starlette.responses import StreamingResponse
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'Please install the `starlette` package to use the `streaming_response()` method, '
                'you can use the `ui` optional group â€” `pip install "pydantic-ai-slim[ui]"`'
            ) from e

        return StreamingResponse(
            self.encode_stream(stream),
            headers=self.response_headers,
            media_type=self.content_type,
        )

    async def transform_stream(  # noqa: C901
        self, stream: AsyncIterator[NativeEvent], on_complete: OnCompleteFunc[EventT] | None = None
    ) -> AsyncIterator[EventT]:
        """Transform a stream of Pydantic AI events into protocol-specific events.

        This method dispatches to specific hooks and `handle_*` methods that subclasses can override:
        - [`before_stream()`][pydantic_ai.ui.UIEventStream.before_stream]
        - [`after_stream()`][pydantic_ai.ui.UIEventStream.after_stream]
        - [`on_error()`][pydantic_ai.ui.UIEventStream.on_error]
        - [`before_request()`][pydantic_ai.ui.UIEventStream.before_request]
        - [`after_request()`][pydantic_ai.ui.UIEventStream.after_request]
        - [`before_response()`][pydantic_ai.ui.UIEventStream.before_response]
        - [`after_response()`][pydantic_ai.ui.UIEventStream.after_response]
        - [`handle_event()`][pydantic_ai.ui.UIEventStream.handle_event]

        Args:
            stream: The stream of Pydantic AI events to transform.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
        """
        async for e in self.before_stream():
            yield e

        try:
            async for event in stream:
                if isinstance(event, PartStartEvent):
                    async for e in self._turn_to('response'):
                        yield e
                elif isinstance(event, FunctionToolCallEvent):
                    async for e in self._turn_to('request'):
                        yield e
                elif isinstance(event, AgentRunResultEvent):
                    if (
                        self._final_result_event
                        and (tool_call_id := self._final_result_event.tool_call_id)
                        and (tool_name := self._final_result_event.tool_name)
                    ):
                        async for e in self._turn_to('request'):
                            yield e

                        self._final_result_event = None
                        # Ensure the stream does not end on a dangling tool call without a result.
                        output_tool_result_event = FunctionToolResultEvent(
                            result=ToolReturnPart(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                content='Final result processed.',
                            )
                        )
                        async for e in self.handle_function_tool_result(output_tool_result_event):
                            yield e

                    result = cast(AgentRunResult[OutputDataT], event.result)
                    self._result = result

                    async for e in self._turn_to(None):
                        yield e

                    if on_complete is not None:
                        if inspect.isasyncgenfunction(on_complete):
                            async for e in on_complete(result):
                                yield e
                        elif _utils.is_async_callable(on_complete):
                            await on_complete(result)
                        else:
                            await _utils.run_in_executor(on_complete, result)
                elif isinstance(event, FinalResultEvent):
                    self._final_result_event = event

                if isinstance(event, BuiltinToolCallEvent | BuiltinToolResultEvent):  # pyright: ignore[reportDeprecated]
                    # These events were deprecated before this feature was introduced
                    continue

                async for e in self.handle_event(event):
                    yield e
        except Exception as e:
            async for e in self.on_error(e):
                yield e
        finally:
            async for e in self._turn_to(None):
                yield e

            async for e in self.after_stream():
                yield e

    async def _turn_to(self, to_turn: Literal['request', 'response'] | None) -> AsyncIterator[EventT]:
        """Fire hooks when turning from request to response or vice versa."""
        if to_turn == self._turn:
            return

        if self._turn == 'request':
            async for e in self.after_request():
                yield e
        elif self._turn == 'response':
            async for e in self.after_response():
                yield e

        self._turn = to_turn

        if to_turn == 'request':
            async for e in self.before_request():
                yield e
        elif to_turn == 'response':
            async for e in self.before_response():
                yield e

    async def handle_event(self, event: NativeEvent) -> AsyncIterator[EventT]:
        """Transform a Pydantic AI event into one or more protocol-specific events.

        This method dispatches to specific `handle_*` methods based on event type:

        - [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] -> [`handle_part_start()`][pydantic_ai.ui.UIEventStream.handle_part_start]
        - [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] -> `handle_part_delta`
        - [`PartEndEvent`][pydantic_ai.messages.PartEndEvent] -> `handle_part_end`
        - [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] -> `handle_final_result`
        - [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] -> `handle_function_tool_call`
        - [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] -> `handle_function_tool_result`
        - [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] -> `handle_run_result`

        Subclasses are encouraged to override the individual `handle_*` methods rather than this one.
        If you need specific behavior for all events, make sure you call the super method.
        """
        match event:
            case PartStartEvent():
                async for e in self.handle_part_start(event):
                    yield e
            case PartDeltaEvent():
                async for e in self.handle_part_delta(event):
                    yield e
            case PartEndEvent():
                async for e in self.handle_part_end(event):
                    yield e
            case FinalResultEvent():
                async for e in self.handle_final_result(event):
                    yield e
            case FunctionToolCallEvent():
                async for e in self.handle_function_tool_call(event):
                    yield e
            case FunctionToolResultEvent():
                async for e in self.handle_function_tool_result(event):
                    yield e
            case AgentRunResultEvent():
                async for e in self.handle_run_result(event):
                    yield e
            case _:
                pass

    async def handle_part_start(self, event: PartStartEvent) -> AsyncIterator[EventT]:
        """Handle a `PartStartEvent`.

        This method dispatches to specific `handle_*` methods based on part type:

        - [`TextPart`][pydantic_ai.messages.TextPart] -> [`handle_text_start()`][pydantic_ai.ui.UIEventStream.handle_text_start]
        - [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] -> [`handle_thinking_start()`][pydantic_ai.ui.UIEventStream.handle_thinking_start]
        - [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] -> [`handle_tool_call_start()`][pydantic_ai.ui.UIEventStream.handle_tool_call_start]
        - [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] -> [`handle_builtin_tool_call_start()`][pydantic_ai.ui.UIEventStream.handle_builtin_tool_call_start]
        - [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] -> [`handle_builtin_tool_return()`][pydantic_ai.ui.UIEventStream.handle_builtin_tool_return]
        - [`FilePart`][pydantic_ai.messages.FilePart] -> [`handle_file()`][pydantic_ai.ui.UIEventStream.handle_file]

        Subclasses are encouraged to override the individual `handle_*` methods rather than this one.
        If you need specific behavior for all part start events, make sure you call the super method.

        Args:
            event: The part start event.
        """
        part = event.part
        previous_part_kind = event.previous_part_kind
        match part:
            case TextPart():
                async for e in self.handle_text_start(part, follows_text=previous_part_kind == 'text'):
                    yield e
            case ThinkingPart():
                async for e in self.handle_thinking_start(part, follows_thinking=previous_part_kind == 'thinking'):
                    yield e
            case ToolCallPart():
                async for e in self.handle_tool_call_start(part):
                    yield e
            case BuiltinToolCallPart():
                async for e in self.handle_builtin_tool_call_start(part):
                    yield e
            case BuiltinToolReturnPart():
                async for e in self.handle_builtin_tool_return(part):
                    yield e
            case FilePart():  # pragma: no branch
                async for e in self.handle_file(part):
                    yield e

    async def handle_part_delta(self, event: PartDeltaEvent) -> AsyncIterator[EventT]:
        """Handle a PartDeltaEvent.

        This method dispatches to specific `handle_*_delta` methods based on part delta type:

        - [`TextPartDelta`][pydantic_ai.messages.TextPartDelta] -> [`handle_text_delta()`][pydantic_ai.ui.UIEventStream.handle_text_delta]
        - [`ThinkingPartDelta`][pydantic_ai.messages.ThinkingPartDelta] -> [`handle_thinking_delta()`][pydantic_ai.ui.UIEventStream.handle_thinking_delta]
        - [`ToolCallPartDelta`][pydantic_ai.messages.ToolCallPartDelta] -> [`handle_tool_call_delta()`][pydantic_ai.ui.UIEventStream.handle_tool_call_delta]

        Subclasses are encouraged to override the individual `handle_*_delta` methods rather than this one.
        If you need specific behavior for all part delta events, make sure you call the super method.

        Args:
            event: The PartDeltaEvent.
        """
        delta = event.delta
        match delta:
            case TextPartDelta():
                async for e in self.handle_text_delta(delta):
                    yield e
            case ThinkingPartDelta():
                async for e in self.handle_thinking_delta(delta):
                    yield e
            case ToolCallPartDelta():  # pragma: no branch
                async for e in self.handle_tool_call_delta(delta):
                    yield e

    async def handle_part_end(self, event: PartEndEvent) -> AsyncIterator[EventT]:
        """Handle a `PartEndEvent`.

        This method dispatches to specific `handle_*_end` methods based on part type:

        - [`TextPart`][pydantic_ai.messages.TextPart] -> [`handle_text_end()`][pydantic_ai.ui.UIEventStream.handle_text_end]
        - [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] -> [`handle_thinking_end()`][pydantic_ai.ui.UIEventStream.handle_thinking_end]
        - [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] -> [`handle_tool_call_end()`][pydantic_ai.ui.UIEventStream.handle_tool_call_end]
        - [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] -> [`handle_builtin_tool_call_end()`][pydantic_ai.ui.UIEventStream.handle_builtin_tool_call_end]

        Subclasses are encouraged to override the individual `handle_*_end` methods rather than this one.
        If you need specific behavior for all part end events, make sure you call the super method.

        Args:
            event: The part end event.
        """
        part = event.part
        next_part_kind = event.next_part_kind
        match part:
            case TextPart():
                async for e in self.handle_text_end(part, followed_by_text=next_part_kind == 'text'):
                    yield e
            case ThinkingPart():
                async for e in self.handle_thinking_end(part, followed_by_thinking=next_part_kind == 'thinking'):
                    yield e
            case ToolCallPart():
                async for e in self.handle_tool_call_end(part):
                    yield e
            case BuiltinToolCallPart():
                async for e in self.handle_builtin_tool_call_end(part):
                    yield e
            case BuiltinToolReturnPart() | FilePart():  # pragma: no cover
                # These don't have deltas, so they don't need to be ended.
                pass

    async def before_stream(self) -> AsyncIterator[EventT]:
        """Yield events before agent streaming starts.

        This hook is called before any agent events are processed.
        Override this to inject custom events at the start of the stream.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def after_stream(self) -> AsyncIterator[EventT]:
        """Yield events after agent streaming completes.

        This hook is called after all agent events have been processed.
        Override this to inject custom events at the end of the stream.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def on_error(self, error: Exception) -> AsyncIterator[EventT]:
        """Handle errors that occur during streaming.

        Args:
            error: The error that occurred during streaming.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def before_request(self) -> AsyncIterator[EventT]:
        """Yield events before a model request is processed.

        Override this to inject custom events at the start of the request.
        """
        return  # pragma: lax no cover
        yield  # Make this an async generator

    async def after_request(self) -> AsyncIterator[EventT]:
        """Yield events after a model request is processed.

        Override this to inject custom events at the end of the request.
        """
        return  # pragma: lax no cover
        yield  # Make this an async generator

    async def before_response(self) -> AsyncIterator[EventT]:
        """Yield events before a model response is processed.

        Override this to inject custom events at the start of the response.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def after_response(self) -> AsyncIterator[EventT]:
        """Yield events after a model response is processed.

        Override this to inject custom events at the end of the response.
        """
        return  # pragma: lax no cover
        yield  # Make this an async generator

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[EventT]:
        """Handle the start of a `TextPart`.

        Args:
            part: The text part.
            follows_text: Whether the part is directly preceded by another text part. In this case, you may want to yield a "text-delta" event instead of a "text-start" event.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[EventT]:
        """Handle a `TextPartDelta`.

        Args:
            delta: The text part delta.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[EventT]:
        """Handle the end of a `TextPart`.

        Args:
            part: The text part.
            followed_by_text: Whether the part is directly followed by another text part. In this case, you may not want to yield a "text-end" event yet.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_thinking_start(self, part: ThinkingPart, follows_thinking: bool = False) -> AsyncIterator[EventT]:
        """Handle the start of a `ThinkingPart`.

        Args:
            part: The thinking part.
            follows_thinking: Whether the part is directly preceded by another thinking part. In this case, you may want to yield a "thinking-delta" event instead of a "thinking-start" event.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[EventT]:
        """Handle a `ThinkingPartDelta`.

        Args:
            delta: The thinking part delta.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[EventT]:
        """Handle the end of a `ThinkingPart`.

        Args:
            part: The thinking part.
            followed_by_thinking: Whether the part is directly followed by another thinking part. In this case, you may not want to yield a "thinking-end" event yet.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[EventT]:
        """Handle the start of a `ToolCallPart`.

        Args:
            part: The tool call part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[EventT]:
        """Handle a `ToolCallPartDelta`.

        Args:
            delta: The tool call part delta.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[EventT]:
        """Handle the end of a `ToolCallPart`.

        Args:
            part: The tool call part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[EventT]:
        """Handle a `BuiltinToolCallPart` at start.

        Args:
            part: The builtin tool call part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[EventT]:
        """Handle the end of a `BuiltinToolCallPart`.

        Args:
            part: The builtin tool call part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[EventT]:
        """Handle a `BuiltinToolReturnPart`.

        Args:
            part: The builtin tool return part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_file(self, part: FilePart) -> AsyncIterator[EventT]:
        """Handle a `FilePart`.

        Args:
            part: The file part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[EventT]:
        """Handle a `FinalResultEvent`.

        Args:
            event: The final result event.
        """
        return
        yield  # Make this an async generator

    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[EventT]:
        """Handle a `FunctionToolCallEvent`.

        Args:
            event: The function tool call event.
        """
        return
        yield  # Make this an async generator

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[EventT]:
        """Handle a `FunctionToolResultEvent`.

        Args:
            event: The function tool result event.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[EventT]:
        """Handle an `AgentRunResultEvent`.

        Args:
            event: The agent run result event.
        """
        return
        yield  # Make this an async generator

```

#### accept

```python
accept: str | None = None

```

The `Accept` header value of the request, used to determine how to encode the protocol-specific events for the streaming response.

#### message_id

```python
message_id: str = field(
    default_factory=lambda: str(uuid4())
)

```

The message ID to use for the next event.

#### new_message_id

```python
new_message_id() -> str

```

Generate and store a new message ID.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
def new_message_id(self) -> str:
    """Generate and store a new message ID."""
    self.message_id = str(uuid4())
    return self.message_id

```

#### response_headers

```python
response_headers: Mapping[str, str] | None

```

Response headers to return to the frontend.

#### content_type

```python
content_type: str

```

Get the content type for the event stream, compatible with the `Accept` header value.

By default, this returns the Server-Sent Events content type (`text/event-stream`). If a subclass supports other types as well, it should consider `self.accept` in encode_event() and return the resulting content type.

#### encode_event

```python
encode_event(event: EventT) -> str

```

Encode a protocol-specific event as a string.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
@abstractmethod
def encode_event(self, event: EventT) -> str:
    """Encode a protocol-specific event as a string."""
    raise NotImplementedError

```

#### encode_stream

```python
encode_stream(
    stream: AsyncIterator[EventT],
) -> AsyncIterator[str]

```

Encode a stream of protocol-specific events as strings according to the `Accept` header value.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def encode_stream(self, stream: AsyncIterator[EventT]) -> AsyncIterator[str]:
    """Encode a stream of protocol-specific events as strings according to the `Accept` header value."""
    async for event in stream:
        yield self.encode_event(event)

```

#### streaming_response

```python
streaming_response(
    stream: AsyncIterator[EventT],
) -> StreamingResponse

```

Generate a streaming response from a stream of protocol-specific events.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
def streaming_response(self, stream: AsyncIterator[EventT]) -> StreamingResponse:
    """Generate a streaming response from a stream of protocol-specific events."""
    try:
        from starlette.responses import StreamingResponse
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'Please install the `starlette` package to use the `streaming_response()` method, '
            'you can use the `ui` optional group â€” `pip install "pydantic-ai-slim[ui]"`'
        ) from e

    return StreamingResponse(
        self.encode_stream(stream),
        headers=self.response_headers,
        media_type=self.content_type,
    )

```

#### transform_stream

```python
transform_stream(
    stream: AsyncIterator[NativeEvent],
    on_complete: OnCompleteFunc[EventT] | None = None,
) -> AsyncIterator[EventT]

```

Transform a stream of Pydantic AI events into protocol-specific events.

This method dispatches to specific hooks and `handle_*` methods that subclasses can override:

- before_stream()
- after_stream()
- on_error()
- before_request()
- after_request()
- before_response()
- after_response()
- handle_event()

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `stream` | `AsyncIterator[NativeEvent]` | The stream of Pydantic AI events to transform. | *required* | | `on_complete` | `OnCompleteFunc[EventT] | None` | Optional callback function called when the agent run completes successfully. The callback receives the completed AgentRunResult and can optionally yield additional protocol-specific events. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def transform_stream(  # noqa: C901
    self, stream: AsyncIterator[NativeEvent], on_complete: OnCompleteFunc[EventT] | None = None
) -> AsyncIterator[EventT]:
    """Transform a stream of Pydantic AI events into protocol-specific events.

    This method dispatches to specific hooks and `handle_*` methods that subclasses can override:
    - [`before_stream()`][pydantic_ai.ui.UIEventStream.before_stream]
    - [`after_stream()`][pydantic_ai.ui.UIEventStream.after_stream]
    - [`on_error()`][pydantic_ai.ui.UIEventStream.on_error]
    - [`before_request()`][pydantic_ai.ui.UIEventStream.before_request]
    - [`after_request()`][pydantic_ai.ui.UIEventStream.after_request]
    - [`before_response()`][pydantic_ai.ui.UIEventStream.before_response]
    - [`after_response()`][pydantic_ai.ui.UIEventStream.after_response]
    - [`handle_event()`][pydantic_ai.ui.UIEventStream.handle_event]

    Args:
        stream: The stream of Pydantic AI events to transform.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
    """
    async for e in self.before_stream():
        yield e

    try:
        async for event in stream:
            if isinstance(event, PartStartEvent):
                async for e in self._turn_to('response'):
                    yield e
            elif isinstance(event, FunctionToolCallEvent):
                async for e in self._turn_to('request'):
                    yield e
            elif isinstance(event, AgentRunResultEvent):
                if (
                    self._final_result_event
                    and (tool_call_id := self._final_result_event.tool_call_id)
                    and (tool_name := self._final_result_event.tool_name)
                ):
                    async for e in self._turn_to('request'):
                        yield e

                    self._final_result_event = None
                    # Ensure the stream does not end on a dangling tool call without a result.
                    output_tool_result_event = FunctionToolResultEvent(
                        result=ToolReturnPart(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            content='Final result processed.',
                        )
                    )
                    async for e in self.handle_function_tool_result(output_tool_result_event):
                        yield e

                result = cast(AgentRunResult[OutputDataT], event.result)
                self._result = result

                async for e in self._turn_to(None):
                    yield e

                if on_complete is not None:
                    if inspect.isasyncgenfunction(on_complete):
                        async for e in on_complete(result):
                            yield e
                    elif _utils.is_async_callable(on_complete):
                        await on_complete(result)
                    else:
                        await _utils.run_in_executor(on_complete, result)
            elif isinstance(event, FinalResultEvent):
                self._final_result_event = event

            if isinstance(event, BuiltinToolCallEvent | BuiltinToolResultEvent):  # pyright: ignore[reportDeprecated]
                # These events were deprecated before this feature was introduced
                continue

            async for e in self.handle_event(event):
                yield e
    except Exception as e:
        async for e in self.on_error(e):
            yield e
    finally:
        async for e in self._turn_to(None):
            yield e

        async for e in self.after_stream():
            yield e

```

#### handle_event

```python
handle_event(event: NativeEvent) -> AsyncIterator[EventT]

```

Transform a Pydantic AI event into one or more protocol-specific events.

This method dispatches to specific `handle_*` methods based on event type:

- PartStartEvent -> handle_part_start()
- PartDeltaEvent -> `handle_part_delta`
- PartEndEvent -> `handle_part_end`
- FinalResultEvent -> `handle_final_result`
- FunctionToolCallEvent -> `handle_function_tool_call`
- FunctionToolResultEvent -> `handle_function_tool_result`
- AgentRunResultEvent -> `handle_run_result`

Subclasses are encouraged to override the individual `handle_*` methods rather than this one. If you need specific behavior for all events, make sure you call the super method.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_event(self, event: NativeEvent) -> AsyncIterator[EventT]:
    """Transform a Pydantic AI event into one or more protocol-specific events.

    This method dispatches to specific `handle_*` methods based on event type:

    - [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] -> [`handle_part_start()`][pydantic_ai.ui.UIEventStream.handle_part_start]
    - [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] -> `handle_part_delta`
    - [`PartEndEvent`][pydantic_ai.messages.PartEndEvent] -> `handle_part_end`
    - [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] -> `handle_final_result`
    - [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] -> `handle_function_tool_call`
    - [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] -> `handle_function_tool_result`
    - [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] -> `handle_run_result`

    Subclasses are encouraged to override the individual `handle_*` methods rather than this one.
    If you need specific behavior for all events, make sure you call the super method.
    """
    match event:
        case PartStartEvent():
            async for e in self.handle_part_start(event):
                yield e
        case PartDeltaEvent():
            async for e in self.handle_part_delta(event):
                yield e
        case PartEndEvent():
            async for e in self.handle_part_end(event):
                yield e
        case FinalResultEvent():
            async for e in self.handle_final_result(event):
                yield e
        case FunctionToolCallEvent():
            async for e in self.handle_function_tool_call(event):
                yield e
        case FunctionToolResultEvent():
            async for e in self.handle_function_tool_result(event):
                yield e
        case AgentRunResultEvent():
            async for e in self.handle_run_result(event):
                yield e
        case _:
            pass

```

#### handle_part_start

```python
handle_part_start(
    event: PartStartEvent,
) -> AsyncIterator[EventT]

```

Handle a `PartStartEvent`.

This method dispatches to specific `handle_*` methods based on part type:

- TextPart -> handle_text_start()
- ThinkingPart -> handle_thinking_start()
- ToolCallPart -> handle_tool_call_start()
- BuiltinToolCallPart -> handle_builtin_tool_call_start()
- BuiltinToolReturnPart -> handle_builtin_tool_return()
- FilePart -> handle_file()

Subclasses are encouraged to override the individual `handle_*` methods rather than this one. If you need specific behavior for all part start events, make sure you call the super method.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `PartStartEvent` | The part start event. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_part_start(self, event: PartStartEvent) -> AsyncIterator[EventT]:
    """Handle a `PartStartEvent`.

    This method dispatches to specific `handle_*` methods based on part type:

    - [`TextPart`][pydantic_ai.messages.TextPart] -> [`handle_text_start()`][pydantic_ai.ui.UIEventStream.handle_text_start]
    - [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] -> [`handle_thinking_start()`][pydantic_ai.ui.UIEventStream.handle_thinking_start]
    - [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] -> [`handle_tool_call_start()`][pydantic_ai.ui.UIEventStream.handle_tool_call_start]
    - [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] -> [`handle_builtin_tool_call_start()`][pydantic_ai.ui.UIEventStream.handle_builtin_tool_call_start]
    - [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] -> [`handle_builtin_tool_return()`][pydantic_ai.ui.UIEventStream.handle_builtin_tool_return]
    - [`FilePart`][pydantic_ai.messages.FilePart] -> [`handle_file()`][pydantic_ai.ui.UIEventStream.handle_file]

    Subclasses are encouraged to override the individual `handle_*` methods rather than this one.
    If you need specific behavior for all part start events, make sure you call the super method.

    Args:
        event: The part start event.
    """
    part = event.part
    previous_part_kind = event.previous_part_kind
    match part:
        case TextPart():
            async for e in self.handle_text_start(part, follows_text=previous_part_kind == 'text'):
                yield e
        case ThinkingPart():
            async for e in self.handle_thinking_start(part, follows_thinking=previous_part_kind == 'thinking'):
                yield e
        case ToolCallPart():
            async for e in self.handle_tool_call_start(part):
                yield e
        case BuiltinToolCallPart():
            async for e in self.handle_builtin_tool_call_start(part):
                yield e
        case BuiltinToolReturnPart():
            async for e in self.handle_builtin_tool_return(part):
                yield e
        case FilePart():  # pragma: no branch
            async for e in self.handle_file(part):
                yield e

```

#### handle_part_delta

```python
handle_part_delta(
    event: PartDeltaEvent,
) -> AsyncIterator[EventT]

```

Handle a PartDeltaEvent.

This method dispatches to specific `handle_*_delta` methods based on part delta type:

- TextPartDelta -> handle_text_delta()
- ThinkingPartDelta -> handle_thinking_delta()
- ToolCallPartDelta -> handle_tool_call_delta()

Subclasses are encouraged to override the individual `handle_*_delta` methods rather than this one. If you need specific behavior for all part delta events, make sure you call the super method.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `PartDeltaEvent` | The PartDeltaEvent. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_part_delta(self, event: PartDeltaEvent) -> AsyncIterator[EventT]:
    """Handle a PartDeltaEvent.

    This method dispatches to specific `handle_*_delta` methods based on part delta type:

    - [`TextPartDelta`][pydantic_ai.messages.TextPartDelta] -> [`handle_text_delta()`][pydantic_ai.ui.UIEventStream.handle_text_delta]
    - [`ThinkingPartDelta`][pydantic_ai.messages.ThinkingPartDelta] -> [`handle_thinking_delta()`][pydantic_ai.ui.UIEventStream.handle_thinking_delta]
    - [`ToolCallPartDelta`][pydantic_ai.messages.ToolCallPartDelta] -> [`handle_tool_call_delta()`][pydantic_ai.ui.UIEventStream.handle_tool_call_delta]

    Subclasses are encouraged to override the individual `handle_*_delta` methods rather than this one.
    If you need specific behavior for all part delta events, make sure you call the super method.

    Args:
        event: The PartDeltaEvent.
    """
    delta = event.delta
    match delta:
        case TextPartDelta():
            async for e in self.handle_text_delta(delta):
                yield e
        case ThinkingPartDelta():
            async for e in self.handle_thinking_delta(delta):
                yield e
        case ToolCallPartDelta():  # pragma: no branch
            async for e in self.handle_tool_call_delta(delta):
                yield e

```

#### handle_part_end

```python
handle_part_end(
    event: PartEndEvent,
) -> AsyncIterator[EventT]

```

Handle a `PartEndEvent`.

This method dispatches to specific `handle_*_end` methods based on part type:

- TextPart -> handle_text_end()
- ThinkingPart -> handle_thinking_end()
- ToolCallPart -> handle_tool_call_end()
- BuiltinToolCallPart -> handle_builtin_tool_call_end()

Subclasses are encouraged to override the individual `handle_*_end` methods rather than this one. If you need specific behavior for all part end events, make sure you call the super method.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `PartEndEvent` | The part end event. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_part_end(self, event: PartEndEvent) -> AsyncIterator[EventT]:
    """Handle a `PartEndEvent`.

    This method dispatches to specific `handle_*_end` methods based on part type:

    - [`TextPart`][pydantic_ai.messages.TextPart] -> [`handle_text_end()`][pydantic_ai.ui.UIEventStream.handle_text_end]
    - [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] -> [`handle_thinking_end()`][pydantic_ai.ui.UIEventStream.handle_thinking_end]
    - [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] -> [`handle_tool_call_end()`][pydantic_ai.ui.UIEventStream.handle_tool_call_end]
    - [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] -> [`handle_builtin_tool_call_end()`][pydantic_ai.ui.UIEventStream.handle_builtin_tool_call_end]

    Subclasses are encouraged to override the individual `handle_*_end` methods rather than this one.
    If you need specific behavior for all part end events, make sure you call the super method.

    Args:
        event: The part end event.
    """
    part = event.part
    next_part_kind = event.next_part_kind
    match part:
        case TextPart():
            async for e in self.handle_text_end(part, followed_by_text=next_part_kind == 'text'):
                yield e
        case ThinkingPart():
            async for e in self.handle_thinking_end(part, followed_by_thinking=next_part_kind == 'thinking'):
                yield e
        case ToolCallPart():
            async for e in self.handle_tool_call_end(part):
                yield e
        case BuiltinToolCallPart():
            async for e in self.handle_builtin_tool_call_end(part):
                yield e
        case BuiltinToolReturnPart() | FilePart():  # pragma: no cover
            # These don't have deltas, so they don't need to be ended.
            pass

```

#### before_stream

```python
before_stream() -> AsyncIterator[EventT]

```

Yield events before agent streaming starts.

This hook is called before any agent events are processed. Override this to inject custom events at the start of the stream.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def before_stream(self) -> AsyncIterator[EventT]:
    """Yield events before agent streaming starts.

    This hook is called before any agent events are processed.
    Override this to inject custom events at the start of the stream.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### after_stream

```python
after_stream() -> AsyncIterator[EventT]

```

Yield events after agent streaming completes.

This hook is called after all agent events have been processed. Override this to inject custom events at the end of the stream.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def after_stream(self) -> AsyncIterator[EventT]:
    """Yield events after agent streaming completes.

    This hook is called after all agent events have been processed.
    Override this to inject custom events at the end of the stream.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### on_error

```python
on_error(error: Exception) -> AsyncIterator[EventT]

```

Handle errors that occur during streaming.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `error` | `Exception` | The error that occurred during streaming. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def on_error(self, error: Exception) -> AsyncIterator[EventT]:
    """Handle errors that occur during streaming.

    Args:
        error: The error that occurred during streaming.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### before_request

```python
before_request() -> AsyncIterator[EventT]

```

Yield events before a model request is processed.

Override this to inject custom events at the start of the request.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def before_request(self) -> AsyncIterator[EventT]:
    """Yield events before a model request is processed.

    Override this to inject custom events at the start of the request.
    """
    return  # pragma: lax no cover
    yield  # Make this an async generator

```

#### after_request

```python
after_request() -> AsyncIterator[EventT]

```

Yield events after a model request is processed.

Override this to inject custom events at the end of the request.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def after_request(self) -> AsyncIterator[EventT]:
    """Yield events after a model request is processed.

    Override this to inject custom events at the end of the request.
    """
    return  # pragma: lax no cover
    yield  # Make this an async generator

```

#### before_response

```python
before_response() -> AsyncIterator[EventT]

```

Yield events before a model response is processed.

Override this to inject custom events at the start of the response.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def before_response(self) -> AsyncIterator[EventT]:
    """Yield events before a model response is processed.

    Override this to inject custom events at the start of the response.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### after_response

```python
after_response() -> AsyncIterator[EventT]

```

Yield events after a model response is processed.

Override this to inject custom events at the end of the response.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def after_response(self) -> AsyncIterator[EventT]:
    """Yield events after a model response is processed.

    Override this to inject custom events at the end of the response.
    """
    return  # pragma: lax no cover
    yield  # Make this an async generator

```

#### handle_text_start

```python
handle_text_start(
    part: TextPart, follows_text: bool = False
) -> AsyncIterator[EventT]

```

Handle the start of a `TextPart`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `TextPart` | The text part. | *required* | | `follows_text` | `bool` | Whether the part is directly preceded by another text part. In this case, you may want to yield a "text-delta" event instead of a "text-start" event. | `False` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[EventT]:
    """Handle the start of a `TextPart`.

    Args:
        part: The text part.
        follows_text: Whether the part is directly preceded by another text part. In this case, you may want to yield a "text-delta" event instead of a "text-start" event.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_text_delta

```python
handle_text_delta(
    delta: TextPartDelta,
) -> AsyncIterator[EventT]

```

Handle a `TextPartDelta`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `delta` | `TextPartDelta` | The text part delta. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[EventT]:
    """Handle a `TextPartDelta`.

    Args:
        delta: The text part delta.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_text_end

```python
handle_text_end(
    part: TextPart, followed_by_text: bool = False
) -> AsyncIterator[EventT]

```

Handle the end of a `TextPart`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `TextPart` | The text part. | *required* | | `followed_by_text` | `bool` | Whether the part is directly followed by another text part. In this case, you may not want to yield a "text-end" event yet. | `False` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[EventT]:
    """Handle the end of a `TextPart`.

    Args:
        part: The text part.
        followed_by_text: Whether the part is directly followed by another text part. In this case, you may not want to yield a "text-end" event yet.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_thinking_start

```python
handle_thinking_start(
    part: ThinkingPart, follows_thinking: bool = False
) -> AsyncIterator[EventT]

```

Handle the start of a `ThinkingPart`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `ThinkingPart` | The thinking part. | *required* | | `follows_thinking` | `bool` | Whether the part is directly preceded by another thinking part. In this case, you may want to yield a "thinking-delta" event instead of a "thinking-start" event. | `False` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_thinking_start(self, part: ThinkingPart, follows_thinking: bool = False) -> AsyncIterator[EventT]:
    """Handle the start of a `ThinkingPart`.

    Args:
        part: The thinking part.
        follows_thinking: Whether the part is directly preceded by another thinking part. In this case, you may want to yield a "thinking-delta" event instead of a "thinking-start" event.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_thinking_delta

```python
handle_thinking_delta(
    delta: ThinkingPartDelta,
) -> AsyncIterator[EventT]

```

Handle a `ThinkingPartDelta`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `delta` | `ThinkingPartDelta` | The thinking part delta. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[EventT]:
    """Handle a `ThinkingPartDelta`.

    Args:
        delta: The thinking part delta.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_thinking_end

```python
handle_thinking_end(
    part: ThinkingPart, followed_by_thinking: bool = False
) -> AsyncIterator[EventT]

```

Handle the end of a `ThinkingPart`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `ThinkingPart` | The thinking part. | *required* | | `followed_by_thinking` | `bool` | Whether the part is directly followed by another thinking part. In this case, you may not want to yield a "thinking-end" event yet. | `False` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_thinking_end(
    self, part: ThinkingPart, followed_by_thinking: bool = False
) -> AsyncIterator[EventT]:
    """Handle the end of a `ThinkingPart`.

    Args:
        part: The thinking part.
        followed_by_thinking: Whether the part is directly followed by another thinking part. In this case, you may not want to yield a "thinking-end" event yet.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_tool_call_start

```python
handle_tool_call_start(
    part: ToolCallPart,
) -> AsyncIterator[EventT]

```

Handle the start of a `ToolCallPart`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `ToolCallPart` | The tool call part. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[EventT]:
    """Handle the start of a `ToolCallPart`.

    Args:
        part: The tool call part.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_tool_call_delta

```python
handle_tool_call_delta(
    delta: ToolCallPartDelta,
) -> AsyncIterator[EventT]

```

Handle a `ToolCallPartDelta`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `delta` | `ToolCallPartDelta` | The tool call part delta. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[EventT]:
    """Handle a `ToolCallPartDelta`.

    Args:
        delta: The tool call part delta.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_tool_call_end

```python
handle_tool_call_end(
    part: ToolCallPart,
) -> AsyncIterator[EventT]

```

Handle the end of a `ToolCallPart`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `ToolCallPart` | The tool call part. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[EventT]:
    """Handle the end of a `ToolCallPart`.

    Args:
        part: The tool call part.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_builtin_tool_call_start

```python
handle_builtin_tool_call_start(
    part: BuiltinToolCallPart,
) -> AsyncIterator[EventT]

```

Handle a `BuiltinToolCallPart` at start.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `BuiltinToolCallPart` | The builtin tool call part. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[EventT]:
    """Handle a `BuiltinToolCallPart` at start.

    Args:
        part: The builtin tool call part.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_builtin_tool_call_end

```python
handle_builtin_tool_call_end(
    part: BuiltinToolCallPart,
) -> AsyncIterator[EventT]

```

Handle the end of a `BuiltinToolCallPart`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `BuiltinToolCallPart` | The builtin tool call part. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[EventT]:
    """Handle the end of a `BuiltinToolCallPart`.

    Args:
        part: The builtin tool call part.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_builtin_tool_return

```python
handle_builtin_tool_return(
    part: BuiltinToolReturnPart,
) -> AsyncIterator[EventT]

```

Handle a `BuiltinToolReturnPart`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `BuiltinToolReturnPart` | The builtin tool return part. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[EventT]:
    """Handle a `BuiltinToolReturnPart`.

    Args:
        part: The builtin tool return part.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_file

```python
handle_file(part: FilePart) -> AsyncIterator[EventT]

```

Handle a `FilePart`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `part` | `FilePart` | The file part. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_file(self, part: FilePart) -> AsyncIterator[EventT]:
    """Handle a `FilePart`.

    Args:
        part: The file part.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_final_result

```python
handle_final_result(
    event: FinalResultEvent,
) -> AsyncIterator[EventT]

```

Handle a `FinalResultEvent`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `FinalResultEvent` | The final result event. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[EventT]:
    """Handle a `FinalResultEvent`.

    Args:
        event: The final result event.
    """
    return
    yield  # Make this an async generator

```

#### handle_function_tool_call

```python
handle_function_tool_call(
    event: FunctionToolCallEvent,
) -> AsyncIterator[EventT]

```

Handle a `FunctionToolCallEvent`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `FunctionToolCallEvent` | The function tool call event. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[EventT]:
    """Handle a `FunctionToolCallEvent`.

    Args:
        event: The function tool call event.
    """
    return
    yield  # Make this an async generator

```

#### handle_function_tool_result

```python
handle_function_tool_result(
    event: FunctionToolResultEvent,
) -> AsyncIterator[EventT]

```

Handle a `FunctionToolResultEvent`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `FunctionToolResultEvent` | The function tool result event. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[EventT]:
    """Handle a `FunctionToolResultEvent`.

    Args:
        event: The function tool result event.
    """
    return  # pragma: no cover
    yield  # Make this an async generator

```

#### handle_run_result

```python
handle_run_result(
    event: AgentRunResultEvent,
) -> AsyncIterator[EventT]

```

Handle an `AgentRunResultEvent`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `event` | `AgentRunResultEvent` | The agent run result event. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py`

```python
async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[EventT]:
    """Handle an `AgentRunResultEvent`.

    Args:
        event: The agent run result event.
    """
    return
    yield  # Make this an async generator

```

### MessagesBuilder

Helper class to build Pydantic AI messages from request/response parts.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_messages_builder.py`

```python
@dataclass
class MessagesBuilder:
    """Helper class to build Pydantic AI messages from request/response parts."""

    messages: list[ModelMessage] = field(default_factory=list)

    def add(self, part: ModelRequestPart | ModelResponsePart) -> None:
        """Add a new part, creating a new request or response message if necessary."""
        last_message = self.messages[-1] if self.messages else None
        if isinstance(part, get_union_args(ModelRequestPart)):
            part = cast(ModelRequestPart, part)
            if isinstance(last_message, ModelRequest):
                last_message.parts = [*last_message.parts, part]
            else:
                self.messages.append(ModelRequest(parts=[part]))
        else:
            part = cast(ModelResponsePart, part)
            if isinstance(last_message, ModelResponse):
                last_message.parts = [*last_message.parts, part]
            else:
                self.messages.append(ModelResponse(parts=[part]))

```

#### add

```python
add(part: ModelRequestPart | ModelResponsePart) -> None

```

Add a new part, creating a new request or response message if necessary.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_messages_builder.py`

```python
def add(self, part: ModelRequestPart | ModelResponsePart) -> None:
    """Add a new part, creating a new request or response message if necessary."""
    last_message = self.messages[-1] if self.messages else None
    if isinstance(part, get_union_args(ModelRequestPart)):
        part = cast(ModelRequestPart, part)
        if isinstance(last_message, ModelRequest):
            last_message.parts = [*last_message.parts, part]
        else:
            self.messages.append(ModelRequest(parts=[part]))
    else:
        part = cast(ModelResponsePart, part)
        if isinstance(last_message, ModelResponse):
            last_message.parts = [*last_message.parts, part]
        else:
            self.messages.append(ModelResponse(parts=[part]))

```

# `pydantic_ai.ui.vercel_ai`

Vercel AI protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the Vercel AI protocol, enabling streaming event-based communication for interactive AI applications.

Converted to Python from: https://github.com/vercel/ai/blob/ai%405.0.34/packages/ai/src/ui/ui-messages.ts

### VercelAIAdapter

Bases: `UIAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT, OutputDataT]`

UI adapter for the Vercel AI protocol.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_adapter.py`

```python
@dataclass
class VercelAIAdapter(UIAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT, OutputDataT]):
    """UI adapter for the Vercel AI protocol."""

    @classmethod
    def build_run_input(cls, body: bytes) -> RequestData:
        """Build a Vercel AI run input object from the request body."""
        return request_data_ta.validate_json(body)

    def build_event_stream(self) -> UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]:
        """Build a Vercel AI event stream transformer."""
        return VercelAIEventStream(self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the Vercel AI run input."""
        return self.load_messages(self.run_input.messages)

    @classmethod
    def load_messages(cls, messages: Sequence[UIMessage]) -> list[ModelMessage]:  # noqa: C901
        """Transform Vercel AI messages into Pydantic AI messages."""
        builder = MessagesBuilder()

        for msg in messages:
            if msg.role == 'system':
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        builder.add(SystemPromptPart(content=part.text))
                    else:  # pragma: no cover
                        raise ValueError(f'Unsupported system message part type: {type(part)}')
            elif msg.role == 'user':
                user_prompt_content: str | list[UserContent] = []
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        user_prompt_content.append(part.text)
                    elif isinstance(part, FileUIPart):
                        try:
                            file = BinaryContent.from_data_uri(part.url)
                        except ValueError:
                            media_type_prefix = part.media_type.split('/', 1)[0]
                            match media_type_prefix:
                                case 'image':
                                    file = ImageUrl(url=part.url, media_type=part.media_type)
                                case 'video':
                                    file = VideoUrl(url=part.url, media_type=part.media_type)
                                case 'audio':
                                    file = AudioUrl(url=part.url, media_type=part.media_type)
                                case _:
                                    file = DocumentUrl(url=part.url, media_type=part.media_type)
                        user_prompt_content.append(file)
                    else:  # pragma: no cover
                        raise ValueError(f'Unsupported user message part type: {type(part)}')

                if user_prompt_content:  # pragma: no branch
                    if len(user_prompt_content) == 1 and isinstance(user_prompt_content[0], str):
                        user_prompt_content = user_prompt_content[0]
                    builder.add(UserPromptPart(content=user_prompt_content))

            elif msg.role == 'assistant':
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        builder.add(TextPart(content=part.text))
                    elif isinstance(part, ReasoningUIPart):
                        builder.add(ThinkingPart(content=part.text))
                    elif isinstance(part, FileUIPart):
                        try:
                            file = BinaryContent.from_data_uri(part.url)
                        except ValueError as e:  # pragma: no cover
                            # We don't yet handle non-data-URI file URLs returned by assistants, as no Pydantic AI models do this.
                            raise ValueError(
                                'Vercel AI integration can currently only handle assistant file parts with data URIs.'
                            ) from e
                        builder.add(FilePart(content=file))
                    elif isinstance(part, ToolUIPart | DynamicToolUIPart):
                        if isinstance(part, DynamicToolUIPart):
                            tool_name = part.tool_name
                            builtin_tool = False
                        else:
                            tool_name = part.type.removeprefix('tool-')
                            builtin_tool = part.provider_executed

                        tool_call_id = part.tool_call_id
                        args = part.input

                        if builtin_tool:
                            call_part = BuiltinToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)
                            builder.add(call_part)

                            if isinstance(part, ToolOutputAvailablePart | ToolOutputErrorPart):
                                if part.state == 'output-available':
                                    output = part.output
                                else:
                                    output = {'error_text': part.error_text, 'is_error': True}

                                provider_name = (
                                    (part.call_provider_metadata or {}).get('pydantic_ai', {}).get('provider_name')
                                )
                                call_part.provider_name = provider_name

                                builder.add(
                                    BuiltinToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        content=output,
                                        provider_name=provider_name,
                                    )
                                )
                        else:
                            builder.add(ToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args))

                            if part.state == 'output-available':
                                builder.add(
                                    ToolReturnPart(tool_name=tool_name, tool_call_id=tool_call_id, content=part.output)
                                )
                            elif part.state == 'output-error':
                                builder.add(
                                    RetryPromptPart(
                                        tool_name=tool_name, tool_call_id=tool_call_id, content=part.error_text
                                    )
                                )
                    elif isinstance(part, DataUIPart):  # pragma: no cover
                        # Contains custom data that shouldn't be sent to the model
                        pass
                    elif isinstance(part, SourceUrlUIPart):  # pragma: no cover
                        # TODO: Once we support citations: https://github.com/pydantic/pydantic-ai/issues/3126
                        pass
                    elif isinstance(part, SourceDocumentUIPart):  # pragma: no cover
                        # TODO: Once we support citations: https://github.com/pydantic/pydantic-ai/issues/3126
                        pass
                    elif isinstance(part, StepStartUIPart):  # pragma: no cover
                        # Nothing to do here
                        pass
                    else:
                        assert_never(part)
            else:
                assert_never(msg.role)

        return builder.messages

```

#### build_run_input

```python
build_run_input(body: bytes) -> RequestData

```

Build a Vercel AI run input object from the request body.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_adapter.py`

```python
@classmethod
def build_run_input(cls, body: bytes) -> RequestData:
    """Build a Vercel AI run input object from the request body."""
    return request_data_ta.validate_json(body)

```

#### build_event_stream

```python
build_event_stream() -> (
    UIEventStream[
        RequestData, BaseChunk, AgentDepsT, OutputDataT
    ]
)

```

Build a Vercel AI event stream transformer.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_adapter.py`

```python
def build_event_stream(self) -> UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]:
    """Build a Vercel AI event stream transformer."""
    return VercelAIEventStream(self.run_input, accept=self.accept)

```

#### messages

```python
messages: list[ModelMessage]

```

Pydantic AI messages from the Vercel AI run input.

#### load_messages

```python
load_messages(
    messages: Sequence[UIMessage],
) -> list[ModelMessage]

```

Transform Vercel AI messages into Pydantic AI messages.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_adapter.py`

```python
@classmethod
def load_messages(cls, messages: Sequence[UIMessage]) -> list[ModelMessage]:  # noqa: C901
    """Transform Vercel AI messages into Pydantic AI messages."""
    builder = MessagesBuilder()

    for msg in messages:
        if msg.role == 'system':
            for part in msg.parts:
                if isinstance(part, TextUIPart):
                    builder.add(SystemPromptPart(content=part.text))
                else:  # pragma: no cover
                    raise ValueError(f'Unsupported system message part type: {type(part)}')
        elif msg.role == 'user':
            user_prompt_content: str | list[UserContent] = []
            for part in msg.parts:
                if isinstance(part, TextUIPart):
                    user_prompt_content.append(part.text)
                elif isinstance(part, FileUIPart):
                    try:
                        file = BinaryContent.from_data_uri(part.url)
                    except ValueError:
                        media_type_prefix = part.media_type.split('/', 1)[0]
                        match media_type_prefix:
                            case 'image':
                                file = ImageUrl(url=part.url, media_type=part.media_type)
                            case 'video':
                                file = VideoUrl(url=part.url, media_type=part.media_type)
                            case 'audio':
                                file = AudioUrl(url=part.url, media_type=part.media_type)
                            case _:
                                file = DocumentUrl(url=part.url, media_type=part.media_type)
                    user_prompt_content.append(file)
                else:  # pragma: no cover
                    raise ValueError(f'Unsupported user message part type: {type(part)}')

            if user_prompt_content:  # pragma: no branch
                if len(user_prompt_content) == 1 and isinstance(user_prompt_content[0], str):
                    user_prompt_content = user_prompt_content[0]
                builder.add(UserPromptPart(content=user_prompt_content))

        elif msg.role == 'assistant':
            for part in msg.parts:
                if isinstance(part, TextUIPart):
                    builder.add(TextPart(content=part.text))
                elif isinstance(part, ReasoningUIPart):
                    builder.add(ThinkingPart(content=part.text))
                elif isinstance(part, FileUIPart):
                    try:
                        file = BinaryContent.from_data_uri(part.url)
                    except ValueError as e:  # pragma: no cover
                        # We don't yet handle non-data-URI file URLs returned by assistants, as no Pydantic AI models do this.
                        raise ValueError(
                            'Vercel AI integration can currently only handle assistant file parts with data URIs.'
                        ) from e
                    builder.add(FilePart(content=file))
                elif isinstance(part, ToolUIPart | DynamicToolUIPart):
                    if isinstance(part, DynamicToolUIPart):
                        tool_name = part.tool_name
                        builtin_tool = False
                    else:
                        tool_name = part.type.removeprefix('tool-')
                        builtin_tool = part.provider_executed

                    tool_call_id = part.tool_call_id
                    args = part.input

                    if builtin_tool:
                        call_part = BuiltinToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)
                        builder.add(call_part)

                        if isinstance(part, ToolOutputAvailablePart | ToolOutputErrorPart):
                            if part.state == 'output-available':
                                output = part.output
                            else:
                                output = {'error_text': part.error_text, 'is_error': True}

                            provider_name = (
                                (part.call_provider_metadata or {}).get('pydantic_ai', {}).get('provider_name')
                            )
                            call_part.provider_name = provider_name

                            builder.add(
                                BuiltinToolReturnPart(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    content=output,
                                    provider_name=provider_name,
                                )
                            )
                    else:
                        builder.add(ToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args))

                        if part.state == 'output-available':
                            builder.add(
                                ToolReturnPart(tool_name=tool_name, tool_call_id=tool_call_id, content=part.output)
                            )
                        elif part.state == 'output-error':
                            builder.add(
                                RetryPromptPart(
                                    tool_name=tool_name, tool_call_id=tool_call_id, content=part.error_text
                                )
                            )
                elif isinstance(part, DataUIPart):  # pragma: no cover
                    # Contains custom data that shouldn't be sent to the model
                    pass
                elif isinstance(part, SourceUrlUIPart):  # pragma: no cover
                    # TODO: Once we support citations: https://github.com/pydantic/pydantic-ai/issues/3126
                    pass
                elif isinstance(part, SourceDocumentUIPart):  # pragma: no cover
                    # TODO: Once we support citations: https://github.com/pydantic/pydantic-ai/issues/3126
                    pass
                elif isinstance(part, StepStartUIPart):  # pragma: no cover
                    # Nothing to do here
                    pass
                else:
                    assert_never(part)
        else:
            assert_never(msg.role)

    return builder.messages

```

### VercelAIEventStream

Bases: `UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]`

UI event stream transformer for the Vercel AI protocol.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_event_stream.py`

```python
@dataclass
class VercelAIEventStream(UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the Vercel AI protocol."""

    _step_started: bool = False

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        return VERCEL_AI_DSP_HEADERS

    def encode_event(self, event: BaseChunk) -> str:
        return f'data: {event.encode()}\n\n'

    async def before_stream(self) -> AsyncIterator[BaseChunk]:
        yield StartChunk()

    async def before_response(self) -> AsyncIterator[BaseChunk]:
        if self._step_started:
            yield FinishStepChunk()

        self._step_started = True
        yield StartStepChunk()

    async def after_stream(self) -> AsyncIterator[BaseChunk]:
        yield FinishStepChunk()

        yield FinishChunk()
        yield DoneChunk()

    async def on_error(self, error: Exception) -> AsyncIterator[BaseChunk]:
        yield ErrorChunk(error_text=str(error))

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseChunk]:
        if follows_text:
            message_id = self.message_id
        else:
            message_id = self.new_message_id()
            yield TextStartChunk(id=message_id)

        if part.content:
            yield TextDeltaChunk(id=message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseChunk]:
        if delta.content_delta:  # pragma: no branch
            yield TextDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[BaseChunk]:
        if not followed_by_text:
            yield TextEndChunk(id=self.message_id)

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[BaseChunk]:
        message_id = self.new_message_id()
        yield ReasoningStartChunk(id=message_id)
        if part.content:
            yield ReasoningDeltaChunk(id=message_id, delta=part.content)

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseChunk]:
        if delta.content_delta:  # pragma: no branch
            yield ReasoningDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[BaseChunk]:
        yield ReasoningEndChunk(id=self.message_id)

    def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        return self._handle_tool_call_start(part)

    def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        return self._handle_tool_call_start(part, provider_executed=True)

    async def _handle_tool_call_start(
        self,
        part: ToolCallPart | BuiltinToolCallPart,
        tool_call_id: str | None = None,
        provider_executed: bool | None = None,
    ) -> AsyncIterator[BaseChunk]:
        tool_call_id = tool_call_id or part.tool_call_id
        yield ToolInputStartChunk(
            tool_call_id=tool_call_id,
            tool_name=part.tool_name,
            provider_executed=provider_executed,
        )
        if part.args:
            yield ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=part.args_as_json_str())

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseChunk]:
        tool_call_id = delta.tool_call_id or ''
        assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
        yield ToolInputDeltaChunk(
            tool_call_id=tool_call_id,
            input_text_delta=delta.args_delta if isinstance(delta.args_delta, str) else _json_dumps(delta.args_delta),
        )

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[BaseChunk]:
        yield ToolInputAvailableChunk(
            tool_call_id=part.tool_call_id, tool_name=part.tool_name, input=part.args_as_dict()
        )

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        yield ToolInputAvailableChunk(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            input=part.args_as_dict(),
            provider_executed=True,
            provider_metadata={'pydantic_ai': {'provider_name': part.provider_name}},
        )

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[BaseChunk]:
        yield ToolOutputAvailableChunk(
            tool_call_id=part.tool_call_id,
            output=self._tool_return_output(part),
            provider_executed=True,
        )

    async def handle_file(self, part: FilePart) -> AsyncIterator[BaseChunk]:
        file = part.content
        yield FileChunk(url=file.data_uri, media_type=file.media_type)

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseChunk]:
        part = event.result
        if isinstance(part, RetryPromptPart):
            yield ToolOutputErrorChunk(tool_call_id=part.tool_call_id, error_text=part.model_response())
        else:
            yield ToolOutputAvailableChunk(tool_call_id=part.tool_call_id, output=self._tool_return_output(part))

        # ToolCallResultEvent.content may hold user parts (e.g. text, images) that Vercel AI does not currently have events for

    def _tool_return_output(self, part: BaseToolReturnPart) -> Any:
        output = part.model_response_object()
        # Unwrap the return value from the output dictionary if it exists
        return output.get('return_value', output)

```

Vercel AI request types (UI messages).

Converted to Python from: https://github.com/vercel/ai/blob/ai%405.0.59/packages/ai/src/ui/ui-messages.ts

### ProviderMetadata

```python
ProviderMetadata = dict[str, dict[str, JSONValue]]

```

Provider metadata.

### BaseUIPart

Bases: `CamelBaseModel`, `ABC`

Abstract base class for all UI parts.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class BaseUIPart(CamelBaseModel, ABC):
    """Abstract base class for all UI parts."""

```

### TextUIPart

Bases: `BaseUIPart`

A text part of a message.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class TextUIPart(BaseUIPart):
    """A text part of a message."""

    type: Literal['text'] = 'text'

    text: str
    """The text content."""

    state: Literal['streaming', 'done'] | None = None
    """The state of the text part."""

    provider_metadata: ProviderMetadata | None = None
    """The provider metadata."""

```

#### text

```python
text: str

```

The text content.

#### state

```python
state: Literal['streaming', 'done'] | None = None

```

The state of the text part.

#### provider_metadata

```python
provider_metadata: ProviderMetadata | None = None

```

The provider metadata.

### ReasoningUIPart

Bases: `BaseUIPart`

A reasoning part of a message.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class ReasoningUIPart(BaseUIPart):
    """A reasoning part of a message."""

    type: Literal['reasoning'] = 'reasoning'

    text: str
    """The reasoning text."""

    state: Literal['streaming', 'done'] | None = None
    """The state of the reasoning part."""

    provider_metadata: ProviderMetadata | None = None
    """The provider metadata."""

```

#### text

```python
text: str

```

The reasoning text.

#### state

```python
state: Literal['streaming', 'done'] | None = None

```

The state of the reasoning part.

#### provider_metadata

```python
provider_metadata: ProviderMetadata | None = None

```

The provider metadata.

### SourceUrlUIPart

Bases: `BaseUIPart`

A source part of a message.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class SourceUrlUIPart(BaseUIPart):
    """A source part of a message."""

    type: Literal['source-url'] = 'source-url'
    source_id: str
    url: str
    title: str | None = None
    provider_metadata: ProviderMetadata | None = None

```

### SourceDocumentUIPart

Bases: `BaseUIPart`

A document source part of a message.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class SourceDocumentUIPart(BaseUIPart):
    """A document source part of a message."""

    type: Literal['source-document'] = 'source-document'
    source_id: str
    media_type: str
    title: str
    filename: str | None = None
    provider_metadata: ProviderMetadata | None = None

```

### FileUIPart

Bases: `BaseUIPart`

A file part of a message.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class FileUIPart(BaseUIPart):
    """A file part of a message."""

    type: Literal['file'] = 'file'

    media_type: str
    """
    IANA media type of the file.
    @see https://www.iana.org/assignments/media-types/media-types.xhtml
    """

    filename: str | None = None
    """Optional filename of the file."""

    url: str
    """
    The URL of the file.
    It can either be a URL to a hosted file or a [Data URL](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs).
    """

    provider_metadata: ProviderMetadata | None = None
    """The provider metadata."""

```

#### media_type

```python
media_type: str

```

IANA media type of the file. @see https://www.iana.org/assignments/media-types/media-types.xhtml

#### filename

```python
filename: str | None = None

```

Optional filename of the file.

#### url

```python
url: str

```

The URL of the file. It can either be a URL to a hosted file or a [Data URL](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs).

#### provider_metadata

```python
provider_metadata: ProviderMetadata | None = None

```

The provider metadata.

### StepStartUIPart

Bases: `BaseUIPart`

A step boundary part of a message.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class StepStartUIPart(BaseUIPart):
    """A step boundary part of a message."""

    type: Literal['step-start'] = 'step-start'

```

### DataUIPart

Bases: `BaseUIPart`

Data part with dynamic type based on data name.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class DataUIPart(BaseUIPart):
    """Data part with dynamic type based on data name."""

    type: Annotated[str, Field(pattern=r'^data-')]
    id: str | None = None
    data: Any

```

### ToolInputStreamingPart

Bases: `BaseUIPart`

Tool part in input-streaming state.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class ToolInputStreamingPart(BaseUIPart):
    """Tool part in input-streaming state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['input-streaming'] = 'input-streaming'
    input: Any | None = None
    provider_executed: bool | None = None

```

### ToolInputAvailablePart

Bases: `BaseUIPart`

Tool part in input-available state.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class ToolInputAvailablePart(BaseUIPart):
    """Tool part in input-available state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['input-available'] = 'input-available'
    input: Any | None = None
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None

```

### ToolOutputAvailablePart

Bases: `BaseUIPart`

Tool part in output-available state.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class ToolOutputAvailablePart(BaseUIPart):
    """Tool part in output-available state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['output-available'] = 'output-available'
    input: Any | None = None
    output: Any | None = None
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None
    preliminary: bool | None = None

```

### ToolOutputErrorPart

Bases: `BaseUIPart`

Tool part in output-error state.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class ToolOutputErrorPart(BaseUIPart):
    """Tool part in output-error state."""

    type: Annotated[str, Field(pattern=r'^tool-')]
    tool_call_id: str
    state: Literal['output-error'] = 'output-error'
    input: Any | None = None
    raw_input: Any | None = None
    error_text: str
    provider_executed: bool | None = None
    call_provider_metadata: ProviderMetadata | None = None

```

### ToolUIPart

```python
ToolUIPart = (
    ToolInputStreamingPart
    | ToolInputAvailablePart
    | ToolOutputAvailablePart
    | ToolOutputErrorPart
)

```

Union of all tool part types.

### DynamicToolInputStreamingPart

Bases: `BaseUIPart`

Dynamic tool part in input-streaming state.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class DynamicToolInputStreamingPart(BaseUIPart):
    """Dynamic tool part in input-streaming state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['input-streaming'] = 'input-streaming'
    input: Any | None = None

```

### DynamicToolInputAvailablePart

Bases: `BaseUIPart`

Dynamic tool part in input-available state.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class DynamicToolInputAvailablePart(BaseUIPart):
    """Dynamic tool part in input-available state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['input-available'] = 'input-available'
    input: Any
    call_provider_metadata: ProviderMetadata | None = None

```

### DynamicToolOutputAvailablePart

Bases: `BaseUIPart`

Dynamic tool part in output-available state.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class DynamicToolOutputAvailablePart(BaseUIPart):
    """Dynamic tool part in output-available state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['output-available'] = 'output-available'
    input: Any
    output: Any
    call_provider_metadata: ProviderMetadata | None = None
    preliminary: bool | None = None

```

### DynamicToolOutputErrorPart

Bases: `BaseUIPart`

Dynamic tool part in output-error state.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class DynamicToolOutputErrorPart(BaseUIPart):
    """Dynamic tool part in output-error state."""

    type: Literal['dynamic-tool'] = 'dynamic-tool'
    tool_name: str
    tool_call_id: str
    state: Literal['output-error'] = 'output-error'
    input: Any
    error_text: str
    call_provider_metadata: ProviderMetadata | None = None

```

### DynamicToolUIPart

```python
DynamicToolUIPart = (
    DynamicToolInputStreamingPart
    | DynamicToolInputAvailablePart
    | DynamicToolOutputAvailablePart
    | DynamicToolOutputErrorPart
)

```

Union of all dynamic tool part types.

### UIMessagePart

```python
UIMessagePart = (
    TextUIPart
    | ReasoningUIPart
    | ToolUIPart
    | DynamicToolUIPart
    | SourceUrlUIPart
    | SourceDocumentUIPart
    | FileUIPart
    | DataUIPart
    | StepStartUIPart
)

```

Union of all message part types.

### UIMessage

Bases: `CamelBaseModel`

A message as displayed in the UI by Vercel AI Elements.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class UIMessage(CamelBaseModel):
    """A message as displayed in the UI by Vercel AI Elements."""

    id: str
    """A unique identifier for the message."""

    role: Literal['system', 'user', 'assistant']
    """The role of the message."""

    metadata: Any | None = None
    """The metadata of the message."""

    parts: list[UIMessagePart]
    """
    The parts of the message. Use this for rendering the message in the UI.
    System messages should be avoided (set the system prompt on the server instead).
    They can have text parts.
    User messages can have text parts and file parts.
    Assistant messages can have text, reasoning, tool invocation, and file parts.
    """

```

#### id

```python
id: str

```

A unique identifier for the message.

#### role

```python
role: Literal['system', 'user', 'assistant']

```

The role of the message.

#### metadata

```python
metadata: Any | None = None

```

The metadata of the message.

#### parts

```python
parts: list[UIMessagePart]

```

The parts of the message. Use this for rendering the message in the UI. System messages should be avoided (set the system prompt on the server instead). They can have text parts. User messages can have text parts and file parts. Assistant messages can have text, reasoning, tool invocation, and file parts.

### SubmitMessage

Bases: `CamelBaseModel`

Submit message request.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class SubmitMessage(CamelBaseModel, extra='allow'):
    """Submit message request."""

    trigger: Literal['submit-message'] = 'submit-message'
    id: str
    messages: list[UIMessage]

```

### RegenerateMessage

Bases: `CamelBaseModel`

Ask the agent to regenerate a message.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/request_types.py`

```python
class RegenerateMessage(CamelBaseModel, extra='allow'):
    """Ask the agent to regenerate a message."""

    trigger: Literal['regenerate-message']
    id: str
    messages: list[UIMessage]
    message_id: str

```

### RequestData

```python
RequestData = Annotated[
    SubmitMessage | RegenerateMessage,
    Discriminator("trigger"),
]

```

Union of all request data types.

Vercel AI response types (SSE chunks).

Converted to Python from: https://github.com/vercel/ai/blob/ai%405.0.59/packages/ai/src/ui-message-stream/ui-message-chunks.ts

### ProviderMetadata

```python
ProviderMetadata = dict[str, dict[str, JSONValue]]

```

Provider metadata.

### BaseChunk

Bases: `CamelBaseModel`, `ABC`

Abstract base class for response SSE events.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class BaseChunk(CamelBaseModel, ABC):
    """Abstract base class for response SSE events."""

    def encode(self) -> str:
        return self.model_dump_json(by_alias=True, exclude_none=True)

```

### TextStartChunk

Bases: `BaseChunk`

Text start chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class TextStartChunk(BaseChunk):
    """Text start chunk."""

    type: Literal['text-start'] = 'text-start'
    id: str
    provider_metadata: ProviderMetadata | None = None

```

### TextDeltaChunk

Bases: `BaseChunk`

Text delta chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class TextDeltaChunk(BaseChunk):
    """Text delta chunk."""

    type: Literal['text-delta'] = 'text-delta'
    delta: str
    id: str
    provider_metadata: ProviderMetadata | None = None

```

### TextEndChunk

Bases: `BaseChunk`

Text end chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class TextEndChunk(BaseChunk):
    """Text end chunk."""

    type: Literal['text-end'] = 'text-end'
    id: str
    provider_metadata: ProviderMetadata | None = None

```

### ReasoningStartChunk

Bases: `BaseChunk`

Reasoning start chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ReasoningStartChunk(BaseChunk):
    """Reasoning start chunk."""

    type: Literal['reasoning-start'] = 'reasoning-start'
    id: str
    provider_metadata: ProviderMetadata | None = None

```

### ReasoningDeltaChunk

Bases: `BaseChunk`

Reasoning delta chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ReasoningDeltaChunk(BaseChunk):
    """Reasoning delta chunk."""

    type: Literal['reasoning-delta'] = 'reasoning-delta'
    id: str
    delta: str
    provider_metadata: ProviderMetadata | None = None

```

### ReasoningEndChunk

Bases: `BaseChunk`

Reasoning end chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ReasoningEndChunk(BaseChunk):
    """Reasoning end chunk."""

    type: Literal['reasoning-end'] = 'reasoning-end'
    id: str
    provider_metadata: ProviderMetadata | None = None

```

### ErrorChunk

Bases: `BaseChunk`

Error chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ErrorChunk(BaseChunk):
    """Error chunk."""

    type: Literal['error'] = 'error'
    error_text: str

```

### ToolInputStartChunk

Bases: `BaseChunk`

Tool input start chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ToolInputStartChunk(BaseChunk):
    """Tool input start chunk."""

    type: Literal['tool-input-start'] = 'tool-input-start'
    tool_call_id: str
    tool_name: str
    provider_executed: bool | None = None
    dynamic: bool | None = None

```

### ToolInputDeltaChunk

Bases: `BaseChunk`

Tool input delta chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ToolInputDeltaChunk(BaseChunk):
    """Tool input delta chunk."""

    type: Literal['tool-input-delta'] = 'tool-input-delta'
    tool_call_id: str
    input_text_delta: str

```

### ToolOutputAvailableChunk

Bases: `BaseChunk`

Tool output available chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ToolOutputAvailableChunk(BaseChunk):
    """Tool output available chunk."""

    type: Literal['tool-output-available'] = 'tool-output-available'
    tool_call_id: str
    output: Any
    provider_executed: bool | None = None
    dynamic: bool | None = None
    preliminary: bool | None = None

```

### ToolInputAvailableChunk

Bases: `BaseChunk`

Tool input available chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ToolInputAvailableChunk(BaseChunk):
    """Tool input available chunk."""

    type: Literal['tool-input-available'] = 'tool-input-available'
    tool_call_id: str
    tool_name: str
    input: Any
    provider_executed: bool | None = None
    provider_metadata: ProviderMetadata | None = None
    dynamic: bool | None = None

```

### ToolInputErrorChunk

Bases: `BaseChunk`

Tool input error chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ToolInputErrorChunk(BaseChunk):
    """Tool input error chunk."""

    type: Literal['tool-input-error'] = 'tool-input-error'
    tool_call_id: str
    tool_name: str
    input: Any
    provider_executed: bool | None = None
    provider_metadata: ProviderMetadata | None = None
    dynamic: bool | None = None
    error_text: str

```

### ToolOutputErrorChunk

Bases: `BaseChunk`

Tool output error chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class ToolOutputErrorChunk(BaseChunk):
    """Tool output error chunk."""

    type: Literal['tool-output-error'] = 'tool-output-error'
    tool_call_id: str
    error_text: str
    provider_executed: bool | None = None
    dynamic: bool | None = None

```

### SourceUrlChunk

Bases: `BaseChunk`

Source URL chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class SourceUrlChunk(BaseChunk):
    """Source URL chunk."""

    type: Literal['source-url'] = 'source-url'
    source_id: str
    url: str
    title: str | None = None
    provider_metadata: ProviderMetadata | None = None

```

### SourceDocumentChunk

Bases: `BaseChunk`

Source document chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class SourceDocumentChunk(BaseChunk):
    """Source document chunk."""

    type: Literal['source-document'] = 'source-document'
    source_id: str
    media_type: str
    title: str
    filename: str | None = None
    provider_metadata: ProviderMetadata | None = None

```

### FileChunk

Bases: `BaseChunk`

File chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class FileChunk(BaseChunk):
    """File chunk."""

    type: Literal['file'] = 'file'
    url: str
    media_type: str

```

### DataChunk

Bases: `BaseChunk`

Data chunk with dynamic type.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class DataChunk(BaseChunk):
    """Data chunk with dynamic type."""

    type: Annotated[str, Field(pattern=r'^data-')]
    data: Any

```

### StartStepChunk

Bases: `BaseChunk`

Start step chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class StartStepChunk(BaseChunk):
    """Start step chunk."""

    type: Literal['start-step'] = 'start-step'

```

### FinishStepChunk

Bases: `BaseChunk`

Finish step chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class FinishStepChunk(BaseChunk):
    """Finish step chunk."""

    type: Literal['finish-step'] = 'finish-step'

```

### StartChunk

Bases: `BaseChunk`

Start chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class StartChunk(BaseChunk):
    """Start chunk."""

    type: Literal['start'] = 'start'
    message_id: str | None = None
    message_metadata: Any | None = None

```

### FinishChunk

Bases: `BaseChunk`

Finish chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class FinishChunk(BaseChunk):
    """Finish chunk."""

    type: Literal['finish'] = 'finish'
    message_metadata: Any | None = None

```

### AbortChunk

Bases: `BaseChunk`

Abort chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class AbortChunk(BaseChunk):
    """Abort chunk."""

    type: Literal['abort'] = 'abort'

```

### MessageMetadataChunk

Bases: `BaseChunk`

Message metadata chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class MessageMetadataChunk(BaseChunk):
    """Message metadata chunk."""

    type: Literal['message-metadata'] = 'message-metadata'
    message_metadata: Any

```

### DoneChunk

Bases: `BaseChunk`

Done chunk.

Source code in `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/response_types.py`

```python
class DoneChunk(BaseChunk):
    """Done chunk."""

    type: Literal['done'] = 'done'

    def encode(self) -> str:
        return '[DONE]'

```
