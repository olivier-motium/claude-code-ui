<!-- Source: _complete-doc.md -->

# `pydantic_ai.exceptions`

### ModelRetry

Bases: `Exception`

Exception to raise when a tool function should be retried.

The agent will return the message to the model and ask it to try calling the function/tool again.

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class ModelRetry(Exception):
    """Exception to raise when a tool function should be retried.

    The agent will return the message to the model and ask it to try calling the function/tool again.
    """

    message: str
    """The message to return to the model."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and other.message == self.message

    def __hash__(self) -> int:
        return hash((self.__class__, self.message))

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> core_schema.CoreSchema:
        """Pydantic core schema to allow `ModelRetry` to be (de)serialized."""
        schema = core_schema.typed_dict_schema(
            {
                'message': core_schema.typed_dict_field(core_schema.str_schema()),
                'kind': core_schema.typed_dict_field(core_schema.literal_schema(['model-retry'])),
            }
        )
        return core_schema.no_info_after_validator_function(
            lambda dct: ModelRetry(dct['message']),
            schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: {'message': x.message, 'kind': 'model-retry'},
                return_schema=schema,
            ),
        )

```

#### message

```python
message: str = message

```

The message to return to the model.

#### __get_pydantic_core_schema__

```python
__get_pydantic_core_schema__(_: Any, __: Any) -> CoreSchema

```

Pydantic core schema to allow `ModelRetry` to be (de)serialized.

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
@classmethod
def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> core_schema.CoreSchema:
    """Pydantic core schema to allow `ModelRetry` to be (de)serialized."""
    schema = core_schema.typed_dict_schema(
        {
            'message': core_schema.typed_dict_field(core_schema.str_schema()),
            'kind': core_schema.typed_dict_field(core_schema.literal_schema(['model-retry'])),
        }
    )
    return core_schema.no_info_after_validator_function(
        lambda dct: ModelRetry(dct['message']),
        schema,
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: {'message': x.message, 'kind': 'model-retry'},
            return_schema=schema,
        ),
    )

```

### CallDeferred

Bases: `Exception`

Exception to raise when a tool call should be deferred.

See [tools docs](../../deferred-tools/#deferred-tools) for more information.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `metadata` | `dict[str, Any] | None` | Optional dictionary of metadata to attach to the deferred tool call. This metadata will be available in DeferredToolRequests.metadata keyed by tool_call_id. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class CallDeferred(Exception):
    """Exception to raise when a tool call should be deferred.

    See [tools docs](../deferred-tools.md#deferred-tools) for more information.

    Args:
        metadata: Optional dictionary of metadata to attach to the deferred tool call.
            This metadata will be available in `DeferredToolRequests.metadata` keyed by `tool_call_id`.
    """

    def __init__(self, metadata: dict[str, Any] | None = None):
        self.metadata = metadata
        super().__init__()

```

### ApprovalRequired

Bases: `Exception`

Exception to raise when a tool call requires human-in-the-loop approval.

See [tools docs](../../deferred-tools/#human-in-the-loop-tool-approval) for more information.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `metadata` | `dict[str, Any] | None` | Optional dictionary of metadata to attach to the deferred tool call. This metadata will be available in DeferredToolRequests.metadata keyed by tool_call_id. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class ApprovalRequired(Exception):
    """Exception to raise when a tool call requires human-in-the-loop approval.

    See [tools docs](../deferred-tools.md#human-in-the-loop-tool-approval) for more information.

    Args:
        metadata: Optional dictionary of metadata to attach to the deferred tool call.
            This metadata will be available in `DeferredToolRequests.metadata` keyed by `tool_call_id`.
    """

    def __init__(self, metadata: dict[str, Any] | None = None):
        self.metadata = metadata
        super().__init__()

```

### UserError

Bases: `RuntimeError`

Error caused by a usage mistake by the application developer â€” You!

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class UserError(RuntimeError):
    """Error caused by a usage mistake by the application developer â€” You!"""

    message: str
    """Description of the mistake."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

```

#### message

```python
message: str = message

```

Description of the mistake.

### AgentRunError

Bases: `RuntimeError`

Base class for errors occurring during an agent run.

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class AgentRunError(RuntimeError):
    """Base class for errors occurring during an agent run."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message

```

#### message

```python
message: str = message

```

The error message.

### UsageLimitExceeded

Bases: `AgentRunError`

Error raised when a Model's usage exceeds the specified limits.

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class UsageLimitExceeded(AgentRunError):
    """Error raised when a Model's usage exceeds the specified limits."""

```

### UnexpectedModelBehavior

Bases: `AgentRunError`

Error caused by unexpected Model behavior, e.g. an unexpected response code.

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class UnexpectedModelBehavior(AgentRunError):
    """Error caused by unexpected Model behavior, e.g. an unexpected response code."""

    message: str
    """Description of the unexpected behavior."""
    body: str | None
    """The body of the response, if available."""

    def __init__(self, message: str, body: str | None = None):
        self.message = message
        if body is None:
            self.body: str | None = None
        else:
            try:
                self.body = json.dumps(json.loads(body), indent=2)
            except ValueError:
                self.body = body
        super().__init__(message)

    def __str__(self) -> str:
        if self.body:
            return f'{self.message}, body:\n{self.body}'
        else:
            return self.message

```

#### message

```python
message: str = message

```

Description of the unexpected behavior.

#### body

```python
body: str | None = dumps(loads(body), indent=2)

```

The body of the response, if available.

### ModelAPIError

Bases: `AgentRunError`

Raised when a model provider API request fails.

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class ModelAPIError(AgentRunError):
    """Raised when a model provider API request fails."""

    model_name: str
    """The name of the model associated with the error."""

    def __init__(self, model_name: str, message: str):
        self.model_name = model_name
        super().__init__(message)

```

#### model_name

```python
model_name: str = model_name

```

The name of the model associated with the error.

### ModelHTTPError

Bases: `ModelAPIError`

Raised when an model provider response has a status code of 4xx or 5xx.

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class ModelHTTPError(ModelAPIError):
    """Raised when an model provider response has a status code of 4xx or 5xx."""

    status_code: int
    """The HTTP status code returned by the API."""

    body: object | None
    """The body of the response, if available."""

    def __init__(self, status_code: int, model_name: str, body: object | None = None):
        self.status_code = status_code
        self.body = body
        message = f'status_code: {status_code}, model_name: {model_name}, body: {body}'
        super().__init__(model_name=model_name, message=message)

```

#### status_code

```python
status_code: int = status_code

```

The HTTP status code returned by the API.

#### body

```python
body: object | None = body

```

The body of the response, if available.

### FallbackExceptionGroup

Bases: `ExceptionGroup[Any]`

A group of exceptions that can be raised when all fallback models fail.

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class FallbackExceptionGroup(ExceptionGroup[Any]):
    """A group of exceptions that can be raised when all fallback models fail."""

```

### IncompleteToolCall

Bases: `UnexpectedModelBehavior`

Error raised when a model stops due to token limit while emitting a tool call.

Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`

```python
class IncompleteToolCall(UnexpectedModelBehavior):
    """Error raised when a model stops due to token limit while emitting a tool call."""

```

# `pydantic_ai.ext`

### tool_from_langchain

```python
tool_from_langchain(langchain_tool: LangChainTool) -> Tool

```

Creates a Pydantic AI tool proxy from a LangChain tool.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `langchain_tool` | `LangChainTool` | The LangChain tool to wrap. | *required* |

Returns:

| Type | Description | | --- | --- | | `Tool` | A Pydantic AI tool that corresponds to the LangChain tool. |

Source code in `pydantic_ai_slim/pydantic_ai/ext/langchain.py`

```python
def tool_from_langchain(langchain_tool: LangChainTool) -> Tool:
    """Creates a Pydantic AI tool proxy from a LangChain tool.

    Args:
        langchain_tool: The LangChain tool to wrap.

    Returns:
        A Pydantic AI tool that corresponds to the LangChain tool.
    """
    function_name = langchain_tool.name
    function_description = langchain_tool.description
    inputs = langchain_tool.args.copy()
    required = sorted({name for name, detail in inputs.items() if 'default' not in detail})
    schema: JsonSchemaValue = langchain_tool.get_input_jsonschema()
    if 'additionalProperties' not in schema:
        schema['additionalProperties'] = False
    if required:
        schema['required'] = required

    defaults = {name: detail['default'] for name, detail in inputs.items() if 'default' in detail}

    # restructures the arguments to match langchain tool run
    def proxy(*args: Any, **kwargs: Any) -> str:
        assert not args, 'This should always be called with kwargs'
        kwargs = defaults | kwargs
        return langchain_tool.run(kwargs)

    return Tool.from_schema(
        function=proxy,
        name=function_name,
        description=function_description,
        json_schema=schema,
    )

```

### LangChainToolset

Bases: `FunctionToolset`

A toolset that wraps LangChain tools.

Source code in `pydantic_ai_slim/pydantic_ai/ext/langchain.py`

```python
class LangChainToolset(FunctionToolset):
    """A toolset that wraps LangChain tools."""

    def __init__(self, tools: list[LangChainTool], *, id: str | None = None):
        super().__init__([tool_from_langchain(tool) for tool in tools], id=id)

```

### tool_from_aci

```python
tool_from_aci(
    aci_function: str, linked_account_owner_id: str
) -> Tool

```

Creates a Pydantic AI tool proxy from an ACI.dev function.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `aci_function` | `str` | The ACI.dev function to wrap. | *required* | | `linked_account_owner_id` | `str` | The ACI user ID to execute the function on behalf of. | *required* |

Returns:

| Type | Description | | --- | --- | | `Tool` | A Pydantic AI tool that corresponds to the ACI.dev tool. |

Source code in `pydantic_ai_slim/pydantic_ai/ext/aci.py`

```python
def tool_from_aci(aci_function: str, linked_account_owner_id: str) -> Tool:
    """Creates a Pydantic AI tool proxy from an ACI.dev function.

    Args:
        aci_function: The ACI.dev function to wrap.
        linked_account_owner_id: The ACI user ID to execute the function on behalf of.

    Returns:
        A Pydantic AI tool that corresponds to the ACI.dev tool.
    """
    aci = ACI()
    function_definition = aci.functions.get_definition(aci_function)
    function_name = function_definition['function']['name']
    function_description = function_definition['function']['description']
    inputs = function_definition['function']['parameters']

    json_schema = {
        'additionalProperties': inputs.get('additionalProperties', False),
        'properties': inputs.get('properties', {}),
        'required': inputs.get('required', []),
        # Default to 'object' if not specified
        'type': inputs.get('type', 'object'),
    }

    # Clean the schema
    json_schema = _clean_schema(json_schema)

    def implementation(*args: Any, **kwargs: Any) -> str:
        if args:
            raise TypeError('Positional arguments are not allowed')
        return aci.handle_function_call(
            function_name,
            kwargs,
            linked_account_owner_id=linked_account_owner_id,
            allowed_apps_only=True,
        )

    return Tool.from_schema(
        function=implementation,
        name=function_name,
        description=function_description,
        json_schema=json_schema,
    )

```

### ACIToolset

Bases: `FunctionToolset`

A toolset that wraps ACI.dev tools.

Source code in `pydantic_ai_slim/pydantic_ai/ext/aci.py`

```python
class ACIToolset(FunctionToolset):
    """A toolset that wraps ACI.dev tools."""

    def __init__(self, aci_functions: Sequence[str], linked_account_owner_id: str, *, id: str | None = None):
        super().__init__(
            [tool_from_aci(aci_function, linked_account_owner_id) for aci_function in aci_functions], id=id
        )

```

# `fasta2a`

### FastA2A

Bases: `Starlette`

The main class for the FastA2A library.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/applications.py`

```python
class FastA2A(Starlette):
    """The main class for the FastA2A library."""

    def __init__(
        self,
        *,
        storage: Storage,
        broker: Broker,
        # Agent card
        name: str | None = None,
        url: str = 'http://localhost:8000',
        version: str = '1.0.0',
        description: str | None = None,
        provider: AgentProvider | None = None,
        skills: list[Skill] | None = None,
        # Starlette
        debug: bool = False,
        routes: Sequence[Route] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: dict[Any, ExceptionHandler] | None = None,
        lifespan: Lifespan[FastA2A] | None = None,
    ):
        if lifespan is None:
            lifespan = _default_lifespan

        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            lifespan=lifespan,
        )

        self.name = name or 'My Agent'
        self.url = url
        self.version = version
        self.description = description
        self.provider = provider
        self.skills = skills or []
        # NOTE: For now, I don't think there's any reason to support any other input/output modes.
        self.default_input_modes = ['application/json']
        self.default_output_modes = ['application/json']

        self.task_manager = TaskManager(broker=broker, storage=storage)

        # Setup
        self._agent_card_json_schema: bytes | None = None
        self.router.add_route('/.well-known/agent.json', self._agent_card_endpoint, methods=['HEAD', 'GET', 'OPTIONS'])
        self.router.add_route('/', self._agent_run_endpoint, methods=['POST'])
        self.router.add_route('/docs', self._docs_endpoint, methods=['GET'])

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] == 'http' and not self.task_manager.is_running:
            raise RuntimeError('TaskManager was not properly initialized.')
        await super().__call__(scope, receive, send)

    async def _agent_card_endpoint(self, request: Request) -> Response:
        if self._agent_card_json_schema is None:
            agent_card = AgentCard(
                name=self.name,
                description=self.description or 'An AI agent exposed as an A2A agent.',
                url=self.url,
                version=self.version,
                protocol_version='0.2.5',
                skills=self.skills,
                default_input_modes=self.default_input_modes,
                default_output_modes=self.default_output_modes,
                capabilities=AgentCapabilities(
                    streaming=False, push_notifications=False, state_transition_history=False
                ),
            )
            if self.provider is not None:
                agent_card['provider'] = self.provider
            self._agent_card_json_schema = agent_card_ta.dump_json(agent_card, by_alias=True)
        return Response(content=self._agent_card_json_schema, media_type='application/json')

    async def _docs_endpoint(self, request: Request) -> Response:
        """Serve the documentation interface."""
        docs_path = Path(__file__).parent / 'static' / 'docs.html'
        return FileResponse(docs_path, media_type='text/html')

    async def _agent_run_endpoint(self, request: Request) -> Response:
        """This is the main endpoint for the A2A server.

        Although the specification allows freedom of choice and implementation, I'm pretty sure about some decisions.

        1. The server will always either send a "submitted" or a "failed" on `tasks/send`.
            Never a "completed" on the first message.
        2. There are three possible ends for the task:
            2.1. The task was "completed" successfully.
            2.2. The task was "canceled".
            2.3. The task "failed".
        3. The server will send a "working" on the first chunk on `tasks/pushNotification/get`.
        """
        data = await request.body()
        a2a_request = a2a_request_ta.validate_json(data)

        if a2a_request['method'] == 'message/send':
            jsonrpc_response = await self.task_manager.send_message(a2a_request)
        elif a2a_request['method'] == 'tasks/get':
            jsonrpc_response = await self.task_manager.get_task(a2a_request)
        elif a2a_request['method'] == 'tasks/cancel':
            jsonrpc_response = await self.task_manager.cancel_task(a2a_request)
        else:
            raise NotImplementedError(f'Method {a2a_request["method"]} not implemented.')
        return Response(
            content=a2a_response_ta.dump_json(jsonrpc_response, by_alias=True), media_type='application/json'
        )

```

### Broker

Bases: `ABC`

The broker class is in charge of scheduling the tasks.

The HTTP server uses the broker to schedule tasks.

The simple implementation is the `InMemoryBroker`, which is the broker that runs the tasks in the same process as the HTTP server. That said, this class can be extended to support remote workers.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/broker.py`

```python
@dataclass
class Broker(ABC):
    """The broker class is in charge of scheduling the tasks.

    The HTTP server uses the broker to schedule tasks.

    The simple implementation is the `InMemoryBroker`, which is the broker that
    runs the tasks in the same process as the HTTP server. That said, this class can be
    extended to support remote workers.
    """

    @abstractmethod
    async def run_task(self, params: TaskSendParams) -> None:
        """Send a task to be executed by the worker."""
        raise NotImplementedError('send_run_task is not implemented yet.')

    @abstractmethod
    async def cancel_task(self, params: TaskIdParams) -> None:
        """Cancel a task."""
        raise NotImplementedError('send_cancel_task is not implemented yet.')

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any): ...

    @abstractmethod
    def receive_task_operations(self) -> AsyncIterator[TaskOperation]:
        """Receive task operations from the broker.

        On a multi-worker setup, the broker will need to round-robin the task operations
        between the workers.
        """

```

#### run_task

```python
run_task(params: TaskSendParams) -> None

```

Send a task to be executed by the worker.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/broker.py`

```python
@abstractmethod
async def run_task(self, params: TaskSendParams) -> None:
    """Send a task to be executed by the worker."""
    raise NotImplementedError('send_run_task is not implemented yet.')

```

#### cancel_task

```python
cancel_task(params: TaskIdParams) -> None

```

Cancel a task.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/broker.py`

```python
@abstractmethod
async def cancel_task(self, params: TaskIdParams) -> None:
    """Cancel a task."""
    raise NotImplementedError('send_cancel_task is not implemented yet.')

```

#### receive_task_operations

```python
receive_task_operations() -> AsyncIterator[TaskOperation]

```

Receive task operations from the broker.

On a multi-worker setup, the broker will need to round-robin the task operations between the workers.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/broker.py`

```python
@abstractmethod
def receive_task_operations(self) -> AsyncIterator[TaskOperation]:
    """Receive task operations from the broker.

    On a multi-worker setup, the broker will need to round-robin the task operations
    between the workers.
    """

```

### Skill

Bases: `TypedDict`

Skills are a unit of capability that an agent can perform.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class Skill(TypedDict):
    """Skills are a unit of capability that an agent can perform."""

    id: str
    """A unique identifier for the skill."""

    name: str
    """Human readable name of the skill."""

    description: str
    """A human-readable description of the skill.

    It will be used by the client or a human as a hint to understand the skill.
    """

    tags: list[str]
    """Set of tag-words describing classes of capabilities for this specific skill.

    Examples: "cooking", "customer support", "billing".
    """

    examples: NotRequired[list[str]]
    """The set of example scenarios that the skill can perform.

    Will be used by the client as a hint to understand how the skill can be used. (e.g. "I need a recipe for bread")
    """

    input_modes: list[str]
    """Supported mime types for input data."""

    output_modes: list[str]
    """Supported mime types for output data."""

```

#### id

```python
id: str

```

A unique identifier for the skill.

#### name

```python
name: str

```

Human readable name of the skill.

#### description

```python
description: str

```

A human-readable description of the skill.

It will be used by the client or a human as a hint to understand the skill.

#### tags

```python
tags: list[str]

```

Set of tag-words describing classes of capabilities for this specific skill.

Examples: "cooking", "customer support", "billing".

#### examples

```python
examples: NotRequired[list[str]]

```

The set of example scenarios that the skill can perform.

Will be used by the client as a hint to understand how the skill can be used. (e.g. "I need a recipe for bread")

#### input_modes

```python
input_modes: list[str]

```

Supported mime types for input data.

#### output_modes

```python
output_modes: list[str]

```

Supported mime types for output data.

### Storage

Bases: `ABC`, `Generic[ContextT]`

A storage to retrieve and save tasks, as well as retrieve and save context.

The storage serves two purposes:

1. Task storage: Stores tasks in A2A protocol format with their status, artifacts, and message history
1. Context storage: Stores conversation context in a format optimized for the specific agent implementation

Source code in `.venv/lib/python3.12/site-packages/fasta2a/storage.py`

```python
class Storage(ABC, Generic[ContextT]):
    """A storage to retrieve and save tasks, as well as retrieve and save context.

    The storage serves two purposes:
    1. Task storage: Stores tasks in A2A protocol format with their status, artifacts, and message history
    2. Context storage: Stores conversation context in a format optimized for the specific agent implementation
    """

    @abstractmethod
    async def load_task(self, task_id: str, history_length: int | None = None) -> Task | None:
        """Load a task from storage.

        If the task is not found, return None.
        """

    @abstractmethod
    async def submit_task(self, context_id: str, message: Message) -> Task:
        """Submit a task to storage."""

    @abstractmethod
    async def update_task(
        self,
        task_id: str,
        state: TaskState,
        new_artifacts: list[Artifact] | None = None,
        new_messages: list[Message] | None = None,
    ) -> Task:
        """Update the state of a task. Appends artifacts and messages, if specified."""

    @abstractmethod
    async def load_context(self, context_id: str) -> ContextT | None:
        """Retrieve the stored context given the `context_id`."""

    @abstractmethod
    async def update_context(self, context_id: str, context: ContextT) -> None:
        """Updates the context for a `context_id`.

        Implementing agent can decide what to store in context.
        """

```

#### load_task

```python
load_task(
    task_id: str, history_length: int | None = None
) -> Task | None

```

Load a task from storage.

If the task is not found, return None.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/storage.py`

```python
@abstractmethod
async def load_task(self, task_id: str, history_length: int | None = None) -> Task | None:
    """Load a task from storage.

    If the task is not found, return None.
    """

```

#### submit_task

```python
submit_task(context_id: str, message: Message) -> Task

```

Submit a task to storage.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/storage.py`

```python
@abstractmethod
async def submit_task(self, context_id: str, message: Message) -> Task:
    """Submit a task to storage."""

```

#### update_task

```python
update_task(
    task_id: str,
    state: TaskState,
    new_artifacts: list[Artifact] | None = None,
    new_messages: list[Message] | None = None,
) -> Task

```

Update the state of a task. Appends artifacts and messages, if specified.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/storage.py`

```python
@abstractmethod
async def update_task(
    self,
    task_id: str,
    state: TaskState,
    new_artifacts: list[Artifact] | None = None,
    new_messages: list[Message] | None = None,
) -> Task:
    """Update the state of a task. Appends artifacts and messages, if specified."""

```

#### load_context

```python
load_context(context_id: str) -> ContextT | None

```

Retrieve the stored context given the `context_id`.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/storage.py`

```python
@abstractmethod
async def load_context(self, context_id: str) -> ContextT | None:
    """Retrieve the stored context given the `context_id`."""

```

#### update_context

```python
update_context(context_id: str, context: ContextT) -> None

```

Updates the context for a `context_id`.

Implementing agent can decide what to store in context.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/storage.py`

```python
@abstractmethod
async def update_context(self, context_id: str, context: ContextT) -> None:
    """Updates the context for a `context_id`.

    Implementing agent can decide what to store in context.
    """

```

### Worker

Bases: `ABC`, `Generic[ContextT]`

A worker is responsible for executing tasks.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/worker.py`

```python
@dataclass
class Worker(ABC, Generic[ContextT]):
    """A worker is responsible for executing tasks."""

    broker: Broker
    storage: Storage[ContextT]

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        """Run the worker.

        It connects to the broker, and it makes itself available to receive commands.
        """
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._loop)
            yield
            tg.cancel_scope.cancel()

    async def _loop(self) -> None:
        async for task_operation in self.broker.receive_task_operations():
            await self._handle_task_operation(task_operation)

    async def _handle_task_operation(self, task_operation: TaskOperation) -> None:
        try:
            with use_span(task_operation['_current_span']):
                with tracer.start_as_current_span(
                    f'{task_operation["operation"]} task', attributes={'logfire.tags': ['fasta2a']}
                ):
                    if task_operation['operation'] == 'run':
                        await self.run_task(task_operation['params'])
                    elif task_operation['operation'] == 'cancel':
                        await self.cancel_task(task_operation['params'])
                    else:
                        assert_never(task_operation)
        except Exception:
            await self.storage.update_task(task_operation['params']['id'], state='failed')

    @abstractmethod
    async def run_task(self, params: TaskSendParams) -> None: ...

    @abstractmethod
    async def cancel_task(self, params: TaskIdParams) -> None: ...

    @abstractmethod
    def build_message_history(self, history: list[Message]) -> list[Any]: ...

    @abstractmethod
    def build_artifacts(self, result: Any) -> list[Artifact]: ...

```

#### run

```python
run() -> AsyncIterator[None]

```

Run the worker.

It connects to the broker, and it makes itself available to receive commands.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/worker.py`

```python
@asynccontextmanager
async def run(self) -> AsyncIterator[None]:
    """Run the worker.

    It connects to the broker, and it makes itself available to receive commands.
    """
    async with anyio.create_task_group() as tg:
        tg.start_soon(self._loop)
        yield
        tg.cancel_scope.cancel()

```

This module contains the schema for the agent card.

### AgentCard

Bases: `TypedDict`

The card that describes an agent.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class AgentCard(TypedDict):
    """The card that describes an agent."""

    name: str
    """Human readable name of the agent e.g. "Recipe Agent"."""

    description: str
    """A human-readable description of the agent.

    Used to assist users and other agents in understanding what the agent can do.
    (e.g. "Agent that helps users with recipes and cooking.")
    """

    url: str
    """A URL to the address the agent is hosted at."""

    version: str
    """The version of the agent - format is up to the provider. (e.g. "1.0.0")"""

    protocol_version: str
    """The version of the A2A protocol this agent supports."""

    provider: NotRequired[AgentProvider]
    """The service provider of the agent."""

    documentation_url: NotRequired[str]
    """A URL to documentation for the agent."""

    icon_url: NotRequired[str]
    """A URL to an icon for the agent."""

    preferred_transport: NotRequired[str]
    """The transport of the preferred endpoint. If empty, defaults to JSONRPC."""

    additional_interfaces: NotRequired[list[AgentInterface]]
    """Announcement of additional supported transports."""

    capabilities: AgentCapabilities
    """The capabilities of the agent."""

    security: NotRequired[list[dict[str, list[str]]]]
    """Security requirements for contacting the agent."""

    security_schemes: NotRequired[dict[str, SecurityScheme]]
    """Security scheme definitions."""

    default_input_modes: list[str]
    """Supported mime types for input data."""

    default_output_modes: list[str]
    """Supported mime types for output data."""

    skills: list[Skill]

```

#### name

```python
name: str

```

Human readable name of the agent e.g. "Recipe Agent".

#### description

```python
description: str

```

A human-readable description of the agent.

Used to assist users and other agents in understanding what the agent can do. (e.g. "Agent that helps users with recipes and cooking.")

#### url

```python
url: str

```

A URL to the address the agent is hosted at.

#### version

```python
version: str

```

The version of the agent - format is up to the provider. (e.g. "1.0.0")

#### protocol_version

```python
protocol_version: str

```

The version of the A2A protocol this agent supports.

#### provider

```python
provider: NotRequired[AgentProvider]

```

The service provider of the agent.

#### documentation_url

```python
documentation_url: NotRequired[str]

```

A URL to documentation for the agent.

#### icon_url

```python
icon_url: NotRequired[str]

```

A URL to an icon for the agent.

#### preferred_transport

```python
preferred_transport: NotRequired[str]

```

The transport of the preferred endpoint. If empty, defaults to JSONRPC.

#### additional_interfaces

```python
additional_interfaces: NotRequired[list[AgentInterface]]

```

Announcement of additional supported transports.

#### capabilities

```python
capabilities: AgentCapabilities

```

The capabilities of the agent.

#### security

```python
security: NotRequired[list[dict[str, list[str]]]]

```

Security requirements for contacting the agent.

#### security_schemes

```python
security_schemes: NotRequired[dict[str, SecurityScheme]]

```

Security scheme definitions.

#### default_input_modes

```python
default_input_modes: list[str]

```

Supported mime types for input data.

#### default_output_modes

```python
default_output_modes: list[str]

```

Supported mime types for output data.

### AgentProvider

Bases: `TypedDict`

The service provider of the agent.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
class AgentProvider(TypedDict):
    """The service provider of the agent."""

    organization: str
    url: str

```

### AgentCapabilities

Bases: `TypedDict`

The capabilities of the agent.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class AgentCapabilities(TypedDict):
    """The capabilities of the agent."""

    streaming: NotRequired[bool]
    """Whether the agent supports streaming."""

    push_notifications: NotRequired[bool]
    """Whether the agent can notify updates to client."""

    state_transition_history: NotRequired[bool]
    """Whether the agent exposes status change history for tasks."""

```

#### streaming

```python
streaming: NotRequired[bool]

```

Whether the agent supports streaming.

#### push_notifications

```python
push_notifications: NotRequired[bool]

```

Whether the agent can notify updates to client.

#### state_transition_history

```python
state_transition_history: NotRequired[bool]

```

Whether the agent exposes status change history for tasks.

### HttpSecurityScheme

Bases: `TypedDict`

HTTP security scheme.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class HttpSecurityScheme(TypedDict):
    """HTTP security scheme."""

    type: Literal['http']
    scheme: str
    """The name of the HTTP Authorization scheme."""
    bearer_format: NotRequired[str]
    """A hint to the client to identify how the bearer token is formatted."""
    description: NotRequired[str]
    """Description of this security scheme."""

```

#### scheme

```python
scheme: str

```

The name of the HTTP Authorization scheme.

#### bearer_format

```python
bearer_format: NotRequired[str]

```

A hint to the client to identify how the bearer token is formatted.

#### description

```python
description: NotRequired[str]

```

Description of this security scheme.

### ApiKeySecurityScheme

Bases: `TypedDict`

API Key security scheme.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class ApiKeySecurityScheme(TypedDict):
    """API Key security scheme."""

    type: Literal['apiKey']
    name: str
    """The name of the header, query or cookie parameter to be used."""
    in_: Literal['query', 'header', 'cookie']
    """The location of the API key."""
    description: NotRequired[str]
    """Description of this security scheme."""

```

#### name

```python
name: str

```

The name of the header, query or cookie parameter to be used.

#### in\_

```python
in_: Literal['query', 'header', 'cookie']

```

The location of the API key.

#### description

```python
description: NotRequired[str]

```

Description of this security scheme.

### OAuth2SecurityScheme

Bases: `TypedDict`

OAuth2 security scheme.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class OAuth2SecurityScheme(TypedDict):
    """OAuth2 security scheme."""

    type: Literal['oauth2']
    flows: dict[str, Any]
    """An object containing configuration information for the flow types supported."""
    description: NotRequired[str]
    """Description of this security scheme."""

```

#### flows

```python
flows: dict[str, Any]

```

An object containing configuration information for the flow types supported.

#### description

```python
description: NotRequired[str]

```

Description of this security scheme.

### OpenIdConnectSecurityScheme

Bases: `TypedDict`

OpenID Connect security scheme.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class OpenIdConnectSecurityScheme(TypedDict):
    """OpenID Connect security scheme."""

    type: Literal['openIdConnect']
    open_id_connect_url: str
    """OpenId Connect URL to discover OAuth2 configuration values."""
    description: NotRequired[str]
    """Description of this security scheme."""

```

#### open_id_connect_url

```python
open_id_connect_url: str

```

OpenId Connect URL to discover OAuth2 configuration values.

#### description

```python
description: NotRequired[str]

```

Description of this security scheme.

### SecurityScheme

```python
SecurityScheme = Annotated[
    Union[
        HttpSecurityScheme,
        ApiKeySecurityScheme,
        OAuth2SecurityScheme,
        OpenIdConnectSecurityScheme,
    ],
    Field(discriminator="type"),
]

```

A security scheme for authentication.

### AgentInterface

Bases: `TypedDict`

An interface that the agent supports.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class AgentInterface(TypedDict):
    """An interface that the agent supports."""

    transport: str
    """The transport protocol (e.g., 'jsonrpc', 'websocket')."""

    url: str
    """The URL endpoint for this transport."""

    description: NotRequired[str]
    """Description of this interface."""

```

#### transport

```python
transport: str

```

The transport protocol (e.g., 'jsonrpc', 'websocket').

#### url

```python
url: str

```

The URL endpoint for this transport.

#### description

```python
description: NotRequired[str]

```

Description of this interface.

### AgentExtension

Bases: `TypedDict`

A declaration of an extension supported by an Agent.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class AgentExtension(TypedDict):
    """A declaration of an extension supported by an Agent."""

    uri: str
    """The URI of the extension."""

    description: NotRequired[str]
    """A description of how this agent uses this extension."""

    required: NotRequired[bool]
    """Whether the client must follow specific requirements of the extension."""

    params: NotRequired[dict[str, Any]]
    """Optional configuration for the extension."""

```

#### uri

```python
uri: str

```

The URI of the extension.

#### description

```python
description: NotRequired[str]

```

A description of how this agent uses this extension.

#### required

```python
required: NotRequired[bool]

```

Whether the client must follow specific requirements of the extension.

#### params

```python
params: NotRequired[dict[str, Any]]

```

Optional configuration for the extension.

### Skill

Bases: `TypedDict`

Skills are a unit of capability that an agent can perform.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class Skill(TypedDict):
    """Skills are a unit of capability that an agent can perform."""

    id: str
    """A unique identifier for the skill."""

    name: str
    """Human readable name of the skill."""

    description: str
    """A human-readable description of the skill.

    It will be used by the client or a human as a hint to understand the skill.
    """

    tags: list[str]
    """Set of tag-words describing classes of capabilities for this specific skill.

    Examples: "cooking", "customer support", "billing".
    """

    examples: NotRequired[list[str]]
    """The set of example scenarios that the skill can perform.

    Will be used by the client as a hint to understand how the skill can be used. (e.g. "I need a recipe for bread")
    """

    input_modes: list[str]
    """Supported mime types for input data."""

    output_modes: list[str]
    """Supported mime types for output data."""

```

#### id

```python
id: str

```

A unique identifier for the skill.

#### name

```python
name: str

```

Human readable name of the skill.

#### description

```python
description: str

```

A human-readable description of the skill.

It will be used by the client or a human as a hint to understand the skill.

#### tags

```python
tags: list[str]

```

Set of tag-words describing classes of capabilities for this specific skill.

Examples: "cooking", "customer support", "billing".

#### examples

```python
examples: NotRequired[list[str]]

```

The set of example scenarios that the skill can perform.

Will be used by the client as a hint to understand how the skill can be used. (e.g. "I need a recipe for bread")

#### input_modes

```python
input_modes: list[str]

```

Supported mime types for input data.

#### output_modes

```python
output_modes: list[str]

```

Supported mime types for output data.

### Artifact

Bases: `TypedDict`

Agents generate Artifacts as an end result of a Task.

Artifacts are immutable, can be named, and can have multiple parts. A streaming response can append parts to existing Artifacts.

A single Task can generate many Artifacts. For example, "create a webpage" could create separate HTML and image Artifacts.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class Artifact(TypedDict):
    """Agents generate Artifacts as an end result of a Task.

    Artifacts are immutable, can be named, and can have multiple parts. A streaming response can append parts to
    existing Artifacts.

    A single Task can generate many Artifacts. For example, "create a webpage" could create separate HTML and image
    Artifacts.
    """

    artifact_id: str
    """Unique identifier for the artifact."""

    name: NotRequired[str]
    """The name of the artifact."""

    description: NotRequired[str]
    """A description of the artifact."""

    parts: list[Part]
    """The parts that make up the artifact."""

    metadata: NotRequired[dict[str, Any]]
    """Metadata about the artifact."""

    extensions: NotRequired[list[str]]
    """Array of extensions."""

    append: NotRequired[bool]
    """Whether to append this artifact to an existing one."""

    last_chunk: NotRequired[bool]
    """Whether this is the last chunk of the artifact."""

```

#### artifact_id

```python
artifact_id: str

```

Unique identifier for the artifact.

#### name

```python
name: NotRequired[str]

```

The name of the artifact.

#### description

```python
description: NotRequired[str]

```

A description of the artifact.

#### parts

```python
parts: list[Part]

```

The parts that make up the artifact.

#### metadata

```python
metadata: NotRequired[dict[str, Any]]

```

Metadata about the artifact.

#### extensions

```python
extensions: NotRequired[list[str]]

```

Array of extensions.

#### append

```python
append: NotRequired[bool]

```

Whether to append this artifact to an existing one.

#### last_chunk

```python
last_chunk: NotRequired[bool]

```

Whether this is the last chunk of the artifact.

### PushNotificationConfig

Bases: `TypedDict`

Configuration for push notifications.

A2A supports a secure notification mechanism whereby an agent can notify a client of an update outside of a connected session via a PushNotificationService. Within and across enterprises, it is critical that the agent verifies the identity of the notification service, authenticates itself with the service, and presents an identifier that ties the notification to the executing Task.

The target server of the PushNotificationService should be considered a separate service, and is not guaranteed (or even expected) to be the client directly. This PushNotificationService is responsible for authenticating and authorizing the agent and for proxying the verified notification to the appropriate endpoint (which could be anything from a pub/sub queue, to an email inbox or other service, etc).

For contrived scenarios with isolated client-agent pairs (e.g. local service mesh in a contained VPC, etc.) or isolated environments without enterprise security concerns, the client may choose to simply open a port and act as its own PushNotificationService. Any enterprise implementation will likely have a centralized service that authenticates the remote agents with trusted notification credentials and can handle online/offline scenarios. (This should be thought of similarly to a mobile Push Notification Service).

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class PushNotificationConfig(TypedDict):
    """Configuration for push notifications.

    A2A supports a secure notification mechanism whereby an agent can notify a client of an update
    outside of a connected session via a PushNotificationService. Within and across enterprises,
    it is critical that the agent verifies the identity of the notification service, authenticates
    itself with the service, and presents an identifier that ties the notification to the executing
    Task.

    The target server of the PushNotificationService should be considered a separate service, and
    is not guaranteed (or even expected) to be the client directly. This PushNotificationService is
    responsible for authenticating and authorizing the agent and for proxying the verified notification
    to the appropriate endpoint (which could be anything from a pub/sub queue, to an email inbox or
    other service, etc).

    For contrived scenarios with isolated client-agent pairs (e.g. local service mesh in a contained
    VPC, etc.) or isolated environments without enterprise security concerns, the client may choose to
    simply open a port and act as its own PushNotificationService. Any enterprise implementation will
    likely have a centralized service that authenticates the remote agents with trusted notification
    credentials and can handle online/offline scenarios. (This should be thought of similarly to a
    mobile Push Notification Service).
    """

    id: NotRequired[str]
    """Server-assigned identifier."""

    url: str
    """The URL to send push notifications to."""

    token: NotRequired[str]
    """Token unique to this task/session."""

    authentication: NotRequired[SecurityScheme]
    """Authentication details for push notifications."""

```

#### id

```python
id: NotRequired[str]

```

Server-assigned identifier.

#### url

```python
url: str

```

The URL to send push notifications to.

#### token

```python
token: NotRequired[str]

```

Token unique to this task/session.

#### authentication

```python
authentication: NotRequired[SecurityScheme]

```

Authentication details for push notifications.

### TaskPushNotificationConfig

Bases: `TypedDict`

Configuration for task push notifications.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class TaskPushNotificationConfig(TypedDict):
    """Configuration for task push notifications."""

    id: str
    """The task id."""

    push_notification_config: PushNotificationConfig
    """The push notification configuration."""

```

#### id

```python
id: str

```

The task id.

#### push_notification_config

```python
push_notification_config: PushNotificationConfig

```

The push notification configuration.

### Message

Bases: `TypedDict`

A Message contains any content that is not an Artifact.

This can include things like agent thoughts, user context, instructions, errors, status, or metadata.

All content from a client comes in the form of a Message. Agents send Messages to communicate status or to provide instructions (whereas generated results are sent as Artifacts).

A Message can have multiple parts to denote different pieces of content. For example, a user request could include a textual description from a user and then multiple files used as context from the client.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class Message(TypedDict):
    """A Message contains any content that is not an Artifact.

    This can include things like agent thoughts, user context, instructions, errors, status, or metadata.

    All content from a client comes in the form of a Message. Agents send Messages to communicate status or to provide
    instructions (whereas generated results are sent as Artifacts).

    A Message can have multiple parts to denote different pieces of content. For example, a user request could include
    a textual description from a user and then multiple files used as context from the client.
    """

    role: Literal['user', 'agent']
    """The role of the message."""

    parts: list[Part]
    """The parts of the message."""

    kind: Literal['message']
    """Event type."""

    metadata: NotRequired[dict[str, Any]]
    """Metadata about the message."""

    # Additional fields
    message_id: str
    """Identifier created by the message creator."""

    context_id: NotRequired[str]
    """The context the message is associated with."""

    task_id: NotRequired[str]
    """Identifier of task the message is related to."""

    reference_task_ids: NotRequired[list[str]]
    """Array of task IDs this message references."""

    extensions: NotRequired[list[str]]
    """Array of extensions."""

```

#### role

```python
role: Literal['user', 'agent']

```

The role of the message.

#### parts

```python
parts: list[Part]

```

The parts of the message.

#### kind

```python
kind: Literal['message']

```

Event type.

#### metadata

```python
metadata: NotRequired[dict[str, Any]]

```

Metadata about the message.

#### message_id

```python
message_id: str

```

Identifier created by the message creator.

#### context_id

```python
context_id: NotRequired[str]

```

The context the message is associated with.

#### task_id

```python
task_id: NotRequired[str]

```

Identifier of task the message is related to.

#### reference_task_ids

```python
reference_task_ids: NotRequired[list[str]]

```

Array of task IDs this message references.

#### extensions

```python
extensions: NotRequired[list[str]]

```

Array of extensions.

### TextPart

Bases: `_BasePart`

A part that contains text.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class TextPart(_BasePart):
    """A part that contains text."""

    kind: Literal['text']
    """The kind of the part."""

    text: str
    """The text of the part."""

```

#### kind

```python
kind: Literal['text']

```

The kind of the part.

#### text

```python
text: str

```

The text of the part.

### FileWithBytes

Bases: `TypedDict`

File with base64 encoded data.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class FileWithBytes(TypedDict):
    """File with base64 encoded data."""

    bytes: str
    """The base64 encoded content of the file."""

    mime_type: NotRequired[str]
    """Optional mime type for the file."""

```

#### bytes

```python
bytes: str

```

The base64 encoded content of the file.

#### mime_type

```python
mime_type: NotRequired[str]

```

Optional mime type for the file.

### FileWithUri

Bases: `TypedDict`

File with URI reference.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class FileWithUri(TypedDict):
    """File with URI reference."""

    uri: str
    """The URI of the file."""

    mime_type: NotRequired[str]
    """The mime type of the file."""

```

#### uri

```python
uri: str

```

The URI of the file.

#### mime_type

```python
mime_type: NotRequired[str]

```

The mime type of the file.

### FilePart

Bases: `_BasePart`

A part that contains a file.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class FilePart(_BasePart):
    """A part that contains a file."""

    kind: Literal['file']
    """The kind of the part."""

    file: FileWithBytes | FileWithUri
    """The file content - either bytes or URI."""

```

#### kind

```python
kind: Literal['file']

```

The kind of the part.

#### file

```python
file: FileWithBytes | FileWithUri

```

The file content - either bytes or URI.

### DataPart

Bases: `_BasePart`

A part that contains structured data.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class DataPart(_BasePart):
    """A part that contains structured data."""

    kind: Literal['data']
    """The kind of the part."""

    data: dict[str, Any]
    """The data of the part."""

```

#### kind

```python
kind: Literal['data']

```

The kind of the part.

#### data

```python
data: dict[str, Any]

```

The data of the part.

### Part

```python
Part = Annotated[
    Union[TextPart, FilePart, DataPart],
    Field(discriminator="kind"),
]

```

A fully formed piece of content exchanged between a client and a remote agent as part of a Message or an Artifact.

Each Part has its own content type and metadata.

### TaskState

```python
TaskState: TypeAlias = Literal[
    "submitted",
    "working",
    "input-required",
    "completed",
    "canceled",
    "failed",
    "rejected",
    "auth-required",
    "unknown",
]

```

The possible states of a task.

### TaskStatus

Bases: `TypedDict`

Status and accompanying message for a task.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class TaskStatus(TypedDict):
    """Status and accompanying message for a task."""

    state: TaskState
    """The current state of the task."""

    message: NotRequired[Message]
    """Additional status updates for client."""

    timestamp: NotRequired[str]
    """ISO datetime value of when the status was updated."""

```

#### state

```python
state: TaskState

```

The current state of the task.

#### message

```python
message: NotRequired[Message]

```

Additional status updates for client.

#### timestamp

```python
timestamp: NotRequired[str]

```

ISO datetime value of when the status was updated.

### Task

Bases: `TypedDict`

A Task is a stateful entity that allows Clients and Remote Agents to achieve a specific outcome.

Clients and Remote Agents exchange Messages within a Task. Remote Agents generate results as Artifacts. A Task is always created by a Client and the status is always determined by the Remote Agent.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class Task(TypedDict):
    """A Task is a stateful entity that allows Clients and Remote Agents to achieve a specific outcome.

    Clients and Remote Agents exchange Messages within a Task. Remote Agents generate results as Artifacts.
    A Task is always created by a Client and the status is always determined by the Remote Agent.
    """

    id: str
    """Unique identifier for the task."""

    context_id: str
    """The context the task is associated with."""

    kind: Literal['task']
    """Event type."""

    status: TaskStatus
    """Current status of the task."""

    history: NotRequired[list[Message]]
    """Optional history of messages."""

    artifacts: NotRequired[list[Artifact]]
    """Collection of artifacts created by the agent."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""

```

#### id

```python
id: str

```

Unique identifier for the task.

#### context_id

```python
context_id: str

```

The context the task is associated with.

#### kind

```python
kind: Literal['task']

```

Event type.

#### status

```python
status: TaskStatus

```

Current status of the task.

#### history

```python
history: NotRequired[list[Message]]

```

Optional history of messages.

#### artifacts

```python
artifacts: NotRequired[list[Artifact]]

```

Collection of artifacts created by the agent.

#### metadata

```python
metadata: NotRequired[dict[str, Any]]

```

Extension metadata.

### TaskStatusUpdateEvent

Bases: `TypedDict`

Sent by server during message/stream requests.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class TaskStatusUpdateEvent(TypedDict):
    """Sent by server during message/stream requests."""

    task_id: str
    """The id of the task."""

    context_id: str
    """The context the task is associated with."""

    kind: Literal['status-update']
    """Event type."""

    status: TaskStatus
    """The status of the task."""

    final: bool
    """Indicates the end of the event stream."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""

```

#### task_id

```python
task_id: str

```

The id of the task.

#### context_id

```python
context_id: str

```

The context the task is associated with.

#### kind

```python
kind: Literal['status-update']

```

Event type.

#### status

```python
status: TaskStatus

```

The status of the task.

#### final

```python
final: bool

```

Indicates the end of the event stream.

#### metadata

```python
metadata: NotRequired[dict[str, Any]]

```

Extension metadata.

### TaskArtifactUpdateEvent

Bases: `TypedDict`

Sent by server during message/stream requests.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class TaskArtifactUpdateEvent(TypedDict):
    """Sent by server during message/stream requests."""

    task_id: str
    """The id of the task."""

    context_id: str
    """The context the task is associated with."""

    kind: Literal['artifact-update']
    """Event type identification."""

    artifact: Artifact
    """The artifact that was updated."""

    append: NotRequired[bool]
    """Whether to append to existing artifact (true) or replace (false)."""

    last_chunk: NotRequired[bool]
    """Indicates this is the final chunk of the artifact."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""

```

#### task_id

```python
task_id: str

```

The id of the task.

#### context_id

```python
context_id: str

```

The context the task is associated with.

#### kind

```python
kind: Literal['artifact-update']

```

Event type identification.

#### artifact

```python
artifact: Artifact

```

The artifact that was updated.

#### append

```python
append: NotRequired[bool]

```

Whether to append to existing artifact (true) or replace (false).

#### last_chunk

```python
last_chunk: NotRequired[bool]

```

Indicates this is the final chunk of the artifact.

#### metadata

```python
metadata: NotRequired[dict[str, Any]]

```

Extension metadata.

### TaskIdParams

Bases: `TypedDict`

Parameters for a task id.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class TaskIdParams(TypedDict):
    """Parameters for a task id."""

    id: str
    metadata: NotRequired[dict[str, Any]]

```

### TaskQueryParams

Bases: `TaskIdParams`

Query parameters for a task.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class TaskQueryParams(TaskIdParams):
    """Query parameters for a task."""

    history_length: NotRequired[int]
    """Number of recent messages to be retrieved."""

```

#### history_length

```python
history_length: NotRequired[int]

```

Number of recent messages to be retrieved.

### MessageSendConfiguration

Bases: `TypedDict`

Configuration for the send message request.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class MessageSendConfiguration(TypedDict):
    """Configuration for the send message request."""

    accepted_output_modes: list[str]
    """Accepted output modalities by the client."""

    blocking: NotRequired[bool]
    """If the server should treat the client as a blocking request."""

    history_length: NotRequired[int]
    """Number of recent messages to be retrieved."""

    push_notification_config: NotRequired[PushNotificationConfig]
    """Where the server should send notifications when disconnected."""

```

#### accepted_output_modes

```python
accepted_output_modes: list[str]

```

Accepted output modalities by the client.

#### blocking

```python
blocking: NotRequired[bool]

```

If the server should treat the client as a blocking request.

#### history_length

```python
history_length: NotRequired[int]

```

Number of recent messages to be retrieved.

#### push_notification_config

```python
push_notification_config: NotRequired[
    PushNotificationConfig
]

```

Where the server should send notifications when disconnected.

### MessageSendParams

Bases: `TypedDict`

Parameters for message/send method.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class MessageSendParams(TypedDict):
    """Parameters for message/send method."""

    configuration: NotRequired[MessageSendConfiguration]
    """Send message configuration."""

    message: Message
    """The message being sent to the server."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""

```

#### configuration

```python
configuration: NotRequired[MessageSendConfiguration]

```

Send message configuration.

#### message

```python
message: Message

```

The message being sent to the server.

#### metadata

```python
metadata: NotRequired[dict[str, Any]]

```

Extension metadata.

### TaskSendParams

Bases: `TypedDict`

Internal parameters for task execution within the framework.

Note: This is not part of the A2A protocol - it's used internally for broker/worker communication.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class TaskSendParams(TypedDict):
    """Internal parameters for task execution within the framework.

    Note: This is not part of the A2A protocol - it's used internally
    for broker/worker communication.
    """

    id: str
    """The id of the task."""

    context_id: str
    """The context id for the task."""

    message: Message
    """The message to process."""

    history_length: NotRequired[int]
    """Number of recent messages to be retrieved."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""

```

#### id

```python
id: str

```

The id of the task.

#### context_id

```python
context_id: str

```

The context id for the task.

#### message

```python
message: Message

```

The message to process.

#### history_length

```python
history_length: NotRequired[int]

```

Number of recent messages to be retrieved.

#### metadata

```python
metadata: NotRequired[dict[str, Any]]

```

Extension metadata.

### ListTaskPushNotificationConfigParams

Bases: `TypedDict`

Parameters for getting list of pushNotificationConfigurations associated with a Task.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class ListTaskPushNotificationConfigParams(TypedDict):
    """Parameters for getting list of pushNotificationConfigurations associated with a Task."""

    id: str
    """Task id."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""

```

#### id

```python
id: str

```

Task id.

#### metadata

```python
metadata: NotRequired[dict[str, Any]]

```

Extension metadata.

### DeleteTaskPushNotificationConfigParams

Bases: `TypedDict`

Parameters for removing pushNotificationConfiguration associated with a Task.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
@pydantic.with_config({'alias_generator': to_camel})
class DeleteTaskPushNotificationConfigParams(TypedDict):
    """Parameters for removing pushNotificationConfiguration associated with a Task."""

    id: str
    """Task id."""

    push_notification_config_id: str
    """The push notification config id to delete."""

    metadata: NotRequired[dict[str, Any]]
    """Extension metadata."""

```

#### id

```python
id: str

```

Task id.

#### push_notification_config_id

```python
push_notification_config_id: str

```

The push notification config id to delete.

#### metadata

```python
metadata: NotRequired[dict[str, Any]]

```

Extension metadata.

### JSONRPCMessage

Bases: `TypedDict`

A JSON RPC message.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
class JSONRPCMessage(TypedDict):
    """A JSON RPC message."""

    jsonrpc: Literal['2.0']
    """The JSON RPC version."""

    id: int | str | None
    """The request id."""

```

#### jsonrpc

```python
jsonrpc: Literal['2.0']

```

The JSON RPC version.

#### id

```python
id: int | str | None

```

The request id.

### JSONRPCRequest

Bases: `JSONRPCMessage`, `Generic[Method, Params]`

A JSON RPC request.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
class JSONRPCRequest(JSONRPCMessage, Generic[Method, Params]):
    """A JSON RPC request."""

    method: Method
    """The method to call."""

    params: Params
    """The parameters to pass to the method."""

```

#### method

```python
method: Method

```

The method to call.

#### params

```python
params: Params

```

The parameters to pass to the method.

### JSONRPCError

Bases: `TypedDict`, `Generic[CodeT, MessageT]`

A JSON RPC error.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
class JSONRPCError(TypedDict, Generic[CodeT, MessageT]):
    """A JSON RPC error."""

    code: CodeT
    message: MessageT
    data: NotRequired[Any]

```

### JSONRPCResponse

Bases: `JSONRPCMessage`, `Generic[ResultT, ErrorT]`

A JSON RPC response.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/schema.py`

```python
class JSONRPCResponse(JSONRPCMessage, Generic[ResultT, ErrorT]):
    """A JSON RPC response."""

    result: NotRequired[ResultT]
    error: NotRequired[ErrorT]

```

### JSONParseError

```python
JSONParseError = JSONRPCError[
    Literal[-32700], Literal["Invalid JSON payload"]
]

```

A JSON RPC error for a parse error.

### InvalidRequestError

```python
InvalidRequestError = JSONRPCError[
    Literal[-32600],
    Literal["Request payload validation error"],
]

```

A JSON RPC error for an invalid request.

### MethodNotFoundError

```python
MethodNotFoundError = JSONRPCError[
    Literal[-32601], Literal["Method not found"]
]

```

A JSON RPC error for a method not found.

### InvalidParamsError

```python
InvalidParamsError = JSONRPCError[
    Literal[-32602], Literal["Invalid parameters"]
]

```

A JSON RPC error for invalid parameters.

### InternalError

```python
InternalError = JSONRPCError[
    Literal[-32603], Literal["Internal error"]
]

```

A JSON RPC error for an internal error.

### TaskNotFoundError

```python
TaskNotFoundError = JSONRPCError[
    Literal[-32001], Literal["Task not found"]
]

```

A JSON RPC error for a task not found.

### TaskNotCancelableError

```python
TaskNotCancelableError = JSONRPCError[
    Literal[-32002], Literal["Task not cancelable"]
]

```

A JSON RPC error for a task not cancelable.

### PushNotificationNotSupportedError

```python
PushNotificationNotSupportedError = JSONRPCError[
    Literal[-32003],
    Literal["Push notification not supported"],
]

```

A JSON RPC error for a push notification not supported.

### UnsupportedOperationError

```python
UnsupportedOperationError = JSONRPCError[
    Literal[-32004],
    Literal["This operation is not supported"],
]

```

A JSON RPC error for an unsupported operation.

### ContentTypeNotSupportedError

```python
ContentTypeNotSupportedError = JSONRPCError[
    Literal[-32005], Literal["Incompatible content types"]
]

```

A JSON RPC error for incompatible content types.

### InvalidAgentResponseError

```python
InvalidAgentResponseError = JSONRPCError[
    Literal[-32006], Literal["Invalid agent response"]
]

```

A JSON RPC error for invalid agent response.

### SendMessageRequest

```python
SendMessageRequest = JSONRPCRequest[
    Literal["message/send"], MessageSendParams
]

```

A JSON RPC request to send a message.

### SendMessageResponse

```python
SendMessageResponse = JSONRPCResponse[
    Union[Task, Message], JSONRPCError[Any, Any]
]

```

A JSON RPC response to send a message.

### StreamMessageRequest

```python
StreamMessageRequest = JSONRPCRequest[
    Literal["message/stream"], MessageSendParams
]

```

A JSON RPC request to stream a message.

### GetTaskRequest

```python
GetTaskRequest = JSONRPCRequest[
    Literal["tasks/get"], TaskQueryParams
]

```

A JSON RPC request to get a task.

### GetTaskResponse

```python
GetTaskResponse = JSONRPCResponse[Task, TaskNotFoundError]

```

A JSON RPC response to get a task.

### CancelTaskRequest

```python
CancelTaskRequest = JSONRPCRequest[
    Literal["tasks/cancel"], TaskIdParams
]

```

A JSON RPC request to cancel a task.

### CancelTaskResponse

```python
CancelTaskResponse = JSONRPCResponse[
    Task, Union[TaskNotCancelableError, TaskNotFoundError]
]

```

A JSON RPC response to cancel a task.

### SetTaskPushNotificationRequest

```python
SetTaskPushNotificationRequest = JSONRPCRequest[
    Literal["tasks/pushNotification/set"],
    TaskPushNotificationConfig,
]

```

A JSON RPC request to set a task push notification.

### SetTaskPushNotificationResponse

```python
SetTaskPushNotificationResponse = JSONRPCResponse[
    TaskPushNotificationConfig,
    PushNotificationNotSupportedError,
]

```

A JSON RPC response to set a task push notification.

### GetTaskPushNotificationRequest

```python
GetTaskPushNotificationRequest = JSONRPCRequest[
    Literal["tasks/pushNotification/get"], TaskIdParams
]

```

A JSON RPC request to get a task push notification.

### GetTaskPushNotificationResponse

```python
GetTaskPushNotificationResponse = JSONRPCResponse[
    TaskPushNotificationConfig,
    PushNotificationNotSupportedError,
]

```

A JSON RPC response to get a task push notification.

### ResubscribeTaskRequest

```python
ResubscribeTaskRequest = JSONRPCRequest[
    Literal["tasks/resubscribe"], TaskIdParams
]

```

A JSON RPC request to resubscribe to a task.

### ListTaskPushNotificationConfigRequest

```python
ListTaskPushNotificationConfigRequest = JSONRPCRequest[
    Literal["tasks/pushNotificationConfig/list"],
    ListTaskPushNotificationConfigParams,
]

```

A JSON RPC request to list task push notification configs.

### DeleteTaskPushNotificationConfigRequest

```python
DeleteTaskPushNotificationConfigRequest = JSONRPCRequest[
    Literal["tasks/pushNotificationConfig/delete"],
    DeleteTaskPushNotificationConfigParams,
]

```

A JSON RPC request to delete a task push notification config.

### A2ARequest

```python
A2ARequest = Annotated[
    Union[
        SendMessageRequest,
        StreamMessageRequest,
        GetTaskRequest,
        CancelTaskRequest,
        SetTaskPushNotificationRequest,
        GetTaskPushNotificationRequest,
        ResubscribeTaskRequest,
        ListTaskPushNotificationConfigRequest,
        DeleteTaskPushNotificationConfigRequest,
    ],
    Discriminator("method"),
]

```

A JSON RPC request to the A2A server.

### A2AResponse

```python
A2AResponse: TypeAlias = Union[
    SendMessageResponse,
    GetTaskResponse,
    CancelTaskResponse,
    SetTaskPushNotificationResponse,
    GetTaskPushNotificationResponse,
]

```

A JSON RPC response from the A2A server.

### A2AClient

A client for the A2A protocol.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/client.py`

```python
class A2AClient:
    """A client for the A2A protocol."""

    def __init__(self, base_url: str = 'http://localhost:8000', http_client: httpx.AsyncClient | None = None) -> None:
        if http_client is None:
            self.http_client = httpx.AsyncClient(base_url=base_url)
        else:
            self.http_client = http_client
            self.http_client.base_url = base_url

    async def send_message(
        self,
        message: Message,
        *,
        metadata: dict[str, Any] | None = None,
        configuration: MessageSendConfiguration | None = None,
    ) -> SendMessageResponse:
        """Send a message using the A2A protocol.

        Returns a JSON-RPC response containing either a result (Task) or an error.
        """
        params = MessageSendParams(message=message)
        if metadata is not None:
            params['metadata'] = metadata
        if configuration is not None:
            params['configuration'] = configuration

        request_id = str(uuid.uuid4())
        payload = SendMessageRequest(jsonrpc='2.0', id=request_id, method='message/send', params=params)
        content = send_message_request_ta.dump_json(payload, by_alias=True)
        response = await self.http_client.post('/', content=content, headers={'Content-Type': 'application/json'})
        self._raise_for_status(response)

        return send_message_response_ta.validate_json(response.content)

    async def get_task(self, task_id: str) -> GetTaskResponse:
        payload = GetTaskRequest(jsonrpc='2.0', id=None, method='tasks/get', params={'id': task_id})
        content = a2a_request_ta.dump_json(payload, by_alias=True)
        response = await self.http_client.post('/', content=content, headers={'Content-Type': 'application/json'})
        self._raise_for_status(response)
        return get_task_response_ta.validate_json(response.content)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise UnexpectedResponseError(response.status_code, response.text)

```

#### send_message

```python
send_message(
    message: Message,
    *,
    metadata: dict[str, Any] | None = None,
    configuration: MessageSendConfiguration | None = None
) -> SendMessageResponse

```

Send a message using the A2A protocol.

Returns a JSON-RPC response containing either a result (Task) or an error.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/client.py`

```python
async def send_message(
    self,
    message: Message,
    *,
    metadata: dict[str, Any] | None = None,
    configuration: MessageSendConfiguration | None = None,
) -> SendMessageResponse:
    """Send a message using the A2A protocol.

    Returns a JSON-RPC response containing either a result (Task) or an error.
    """
    params = MessageSendParams(message=message)
    if metadata is not None:
        params['metadata'] = metadata
    if configuration is not None:
        params['configuration'] = configuration

    request_id = str(uuid.uuid4())
    payload = SendMessageRequest(jsonrpc='2.0', id=request_id, method='message/send', params=params)
    content = send_message_request_ta.dump_json(payload, by_alias=True)
    response = await self.http_client.post('/', content=content, headers={'Content-Type': 'application/json'})
    self._raise_for_status(response)

    return send_message_response_ta.validate_json(response.content)

```

### UnexpectedResponseError

Bases: `Exception`

An error raised when an unexpected response is received from the server.

Source code in `.venv/lib/python3.12/site-packages/fasta2a/client.py`

```python
class UnexpectedResponseError(Exception):
    """An error raised when an unexpected response is received from the server."""

    def __init__(self, status_code: int, content: str) -> None:
        self.status_code = status_code
        self.content = content

```

# `pydantic_ai.format_prompt`

### format_as_xml

```python
format_as_xml(
    obj: Any,
    root_tag: str | None = None,
    item_tag: str = "item",
    none_str: str = "null",
    indent: str | None = "  ",
    include_field_info: Literal["once"] | bool = False,
) -> str

```

Format a Python object as XML.

This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML, rather than JSON etc.

Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `time`, `timedelta`, `Enum`, `Mapping`, `Iterable`, `dataclass`, and `BaseModel`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `obj` | `Any` | Python Object to serialize to XML. | *required* | | `root_tag` | `str | None` | Outer tag to wrap the XML in, use None to omit the outer tag. | `None` | | `item_tag` | `str` | Tag to use for each item in an iterable (e.g. list), this is overridden by the class name for dataclasses and Pydantic models. | `'item'` | | `none_str` | `str` | String to use for None values. | `'null'` | | `indent` | `str | None` | Indentation string to use for pretty printing. | `' '` | | `include_field_info` | `Literal['once'] | bool` | Whether to include attributes like Pydantic Field attributes and dataclasses field() metadata as XML attributes. In both cases the allowed Field attributes and field() metadata keys are title and description. If a field is repeated in the data (e.g. in a list) by setting once the attributes are included only in the first occurrence of an XML element relative to the same field. | `False` |

Returns:

| Type | Description | | --- | --- | | `str` | XML representation of the object. |

Example: format_as_xml_example.py

```python
from pydantic_ai import format_as_xml

print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
'''
<user>
  <name>John</name>
  <height>6</height>
  <weight>200</weight>
</user>
'''

```

Source code in `pydantic_ai_slim/pydantic_ai/format_prompt.py`

````python
def format_as_xml(
    obj: Any,
    root_tag: str | None = None,
    item_tag: str = 'item',
    none_str: str = 'null',
    indent: str | None = '  ',
    include_field_info: Literal['once'] | bool = False,
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `time`, `timedelta`, `Enum`,
    `Mapping`, `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.
        include_field_info: Whether to include attributes like Pydantic `Field` attributes and dataclasses `field()`
            `metadata` as XML attributes. In both cases the allowed `Field` attributes and `field()` metadata keys are
            `title` and `description`. If a field is repeated in the data (e.g. in a list) by setting `once`
            the attributes are included only in the first occurrence of an XML element relative to the same field.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai import format_as_xml

    print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
    '''
    <user>
      <name>John</name>
      <height>6</height>
      <weight>200</weight>
    </user>
    '''
    ```
    """
    el = _ToXml(
        data=obj,
        item_tag=item_tag,
        none_str=none_str,
        include_field_info=include_field_info,
    ).to_xml(root_tag)
    if root_tag is None and el.text is None:
        join = '' if indent is None else '\n'
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')

````

# `pydantic_ai.mcp`

### MCPError

Bases: `RuntimeError`

Raised when an MCP server returns an error response.

This exception wraps error responses from MCP servers, following the ErrorData schema from the MCP specification.

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
class MCPError(RuntimeError):
    """Raised when an MCP server returns an error response.

    This exception wraps error responses from MCP servers, following the ErrorData schema
    from the MCP specification.
    """

    message: str
    """The error message."""

    code: int
    """The error code returned by the server."""

    data: dict[str, Any] | None
    """Additional information about the error, if provided by the server."""

    def __init__(self, message: str, code: int, data: dict[str, Any] | None = None):
        self.message = message
        self.code = code
        self.data = data
        super().__init__(message)

    @classmethod
    def from_mcp_sdk(cls, error: mcp_exceptions.McpError) -> MCPError:
        """Create an MCPError from an MCP SDK McpError.

        Args:
            error: An McpError from the MCP SDK.
        """
        # Extract error data from the McpError.error attribute
        error_data = error.error
        return cls(message=error_data.message, code=error_data.code, data=error_data.data)

    def __str__(self) -> str:
        if self.data:
            return f'{self.message} (code: {self.code}, data: {self.data})'
        return f'{self.message} (code: {self.code})'

```

#### message

```python
message: str = message

```

The error message.

#### code

```python
code: int = code

```

The error code returned by the server.

#### data

```python
data: dict[str, Any] | None = data

```

Additional information about the error, if provided by the server.

#### from_mcp_sdk

```python
from_mcp_sdk(error: McpError) -> MCPError

```

Create an MCPError from an MCP SDK McpError.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `error` | `McpError` | An McpError from the MCP SDK. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@classmethod
def from_mcp_sdk(cls, error: mcp_exceptions.McpError) -> MCPError:
    """Create an MCPError from an MCP SDK McpError.

    Args:
        error: An McpError from the MCP SDK.
    """
    # Extract error data from the McpError.error attribute
    error_data = error.error
    return cls(message=error_data.message, code=error_data.code, data=error_data.data)

```

### ResourceAnnotations

Additional properties describing MCP entities.

See the [resource annotations in the MCP specification](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#annotations).

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@dataclass(repr=False, kw_only=True)
class ResourceAnnotations:
    """Additional properties describing MCP entities.

    See the [resource annotations in the MCP specification](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#annotations).
    """

    audience: list[mcp_types.Role] | None = None
    """Intended audience for this entity."""

    priority: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Priority level for this entity, ranging from 0.0 to 1.0."""

    __repr__ = _utils.dataclasses_no_defaults_repr

    @classmethod
    def from_mcp_sdk(cls, mcp_annotations: mcp_types.Annotations) -> ResourceAnnotations:
        """Convert from MCP SDK Annotations to ResourceAnnotations.

        Args:
            mcp_annotations: The MCP SDK annotations object.
        """
        return cls(audience=mcp_annotations.audience, priority=mcp_annotations.priority)

```

#### audience

```python
audience: list[Role] | None = None

```

Intended audience for this entity.

#### priority

```python
priority: Annotated[float, Field(ge=0.0, le=1.0)] | None = (
    None
)

```

Priority level for this entity, ranging from 0.0 to 1.0.

#### from_mcp_sdk

```python
from_mcp_sdk(
    mcp_annotations: Annotations,
) -> ResourceAnnotations

```

Convert from MCP SDK Annotations to ResourceAnnotations.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `mcp_annotations` | `Annotations` | The MCP SDK annotations object. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@classmethod
def from_mcp_sdk(cls, mcp_annotations: mcp_types.Annotations) -> ResourceAnnotations:
    """Convert from MCP SDK Annotations to ResourceAnnotations.

    Args:
        mcp_annotations: The MCP SDK annotations object.
    """
    return cls(audience=mcp_annotations.audience, priority=mcp_annotations.priority)

```

### Resource

Bases: `BaseResource`

A resource that can be read from an MCP server.

See the [resources in the MCP specification](https://modelcontextprotocol.io/specification/2025-06-18/server/resources).

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@dataclass(repr=False, kw_only=True)
class Resource(BaseResource):
    """A resource that can be read from an MCP server.

    See the [resources in the MCP specification](https://modelcontextprotocol.io/specification/2025-06-18/server/resources).
    """

    uri: str
    """The URI of the resource."""

    size: int | None = None
    """The size of the raw resource content in bytes (before base64 encoding), if known."""

    @classmethod
    def from_mcp_sdk(cls, mcp_resource: mcp_types.Resource) -> Resource:
        """Convert from MCP SDK Resource to PydanticAI Resource.

        Args:
            mcp_resource: The MCP SDK Resource object.
        """
        return cls(
            uri=str(mcp_resource.uri),
            name=mcp_resource.name,
            title=mcp_resource.title,
            description=mcp_resource.description,
            mime_type=mcp_resource.mimeType,
            size=mcp_resource.size,
            annotations=ResourceAnnotations.from_mcp_sdk(mcp_resource.annotations)
            if mcp_resource.annotations
            else None,
            metadata=mcp_resource.meta,
        )

```

#### uri

```python
uri: str

```

The URI of the resource.

#### size

```python
size: int | None = None

```

The size of the raw resource content in bytes (before base64 encoding), if known.

#### from_mcp_sdk

```python
from_mcp_sdk(mcp_resource: Resource) -> Resource

```

Convert from MCP SDK Resource to PydanticAI Resource.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `mcp_resource` | `Resource` | The MCP SDK Resource object. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@classmethod
def from_mcp_sdk(cls, mcp_resource: mcp_types.Resource) -> Resource:
    """Convert from MCP SDK Resource to PydanticAI Resource.

    Args:
        mcp_resource: The MCP SDK Resource object.
    """
    return cls(
        uri=str(mcp_resource.uri),
        name=mcp_resource.name,
        title=mcp_resource.title,
        description=mcp_resource.description,
        mime_type=mcp_resource.mimeType,
        size=mcp_resource.size,
        annotations=ResourceAnnotations.from_mcp_sdk(mcp_resource.annotations)
        if mcp_resource.annotations
        else None,
        metadata=mcp_resource.meta,
    )

```

### ResourceTemplate

Bases: `BaseResource`

A template for parameterized resources on an MCP server.

See the [resource templates in the MCP specification](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#resource-templates).

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@dataclass(repr=False, kw_only=True)
class ResourceTemplate(BaseResource):
    """A template for parameterized resources on an MCP server.

    See the [resource templates in the MCP specification](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#resource-templates).
    """

    uri_template: str
    """URI template (RFC 6570) for constructing resource URIs."""

    @classmethod
    def from_mcp_sdk(cls, mcp_template: mcp_types.ResourceTemplate) -> ResourceTemplate:
        """Convert from MCP SDK ResourceTemplate to PydanticAI ResourceTemplate.

        Args:
            mcp_template: The MCP SDK ResourceTemplate object.
        """
        return cls(
            uri_template=mcp_template.uriTemplate,
            name=mcp_template.name,
            title=mcp_template.title,
            description=mcp_template.description,
            mime_type=mcp_template.mimeType,
            annotations=ResourceAnnotations.from_mcp_sdk(mcp_template.annotations)
            if mcp_template.annotations
            else None,
            metadata=mcp_template.meta,
        )

```

#### uri_template

```python
uri_template: str

```

URI template (RFC 6570) for constructing resource URIs.

#### from_mcp_sdk

```python
from_mcp_sdk(
    mcp_template: ResourceTemplate,
) -> ResourceTemplate

```

Convert from MCP SDK ResourceTemplate to PydanticAI ResourceTemplate.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `mcp_template` | `ResourceTemplate` | The MCP SDK ResourceTemplate object. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@classmethod
def from_mcp_sdk(cls, mcp_template: mcp_types.ResourceTemplate) -> ResourceTemplate:
    """Convert from MCP SDK ResourceTemplate to PydanticAI ResourceTemplate.

    Args:
        mcp_template: The MCP SDK ResourceTemplate object.
    """
    return cls(
        uri_template=mcp_template.uriTemplate,
        name=mcp_template.name,
        title=mcp_template.title,
        description=mcp_template.description,
        mime_type=mcp_template.mimeType,
        annotations=ResourceAnnotations.from_mcp_sdk(mcp_template.annotations)
        if mcp_template.annotations
        else None,
        metadata=mcp_template.meta,
    )

```

### ServerCapabilities

Capabilities that an MCP server supports.

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@dataclass(repr=False, kw_only=True)
class ServerCapabilities:
    """Capabilities that an MCP server supports."""

    experimental: list[str] | None = None
    """Experimental, non-standard capabilities that the server supports."""

    logging: bool = False
    """Whether the server supports sending log messages to the client."""

    prompts: bool = False
    """Whether the server offers any prompt templates."""

    resources: bool = False
    """Whether the server offers any resources to read."""

    tools: bool = False
    """Whether the server offers any tools to call."""

    completions: bool = False
    """Whether the server offers autocompletion suggestions for prompts and resources."""

    __repr__ = _utils.dataclasses_no_defaults_repr

    @classmethod
    def from_mcp_sdk(cls, mcp_capabilities: mcp_types.ServerCapabilities) -> ServerCapabilities:
        """Convert from MCP SDK ServerCapabilities to PydanticAI ServerCapabilities.

        Args:
            mcp_capabilities: The MCP SDK ServerCapabilities object.
        """
        return cls(
            experimental=list(mcp_capabilities.experimental.keys()) if mcp_capabilities.experimental else None,
            logging=mcp_capabilities.logging is not None,
            prompts=mcp_capabilities.prompts is not None,
            resources=mcp_capabilities.resources is not None,
            tools=mcp_capabilities.tools is not None,
            completions=mcp_capabilities.completions is not None,
        )

```

#### experimental

```python
experimental: list[str] | None = None

```

Experimental, non-standard capabilities that the server supports.

#### logging

```python
logging: bool = False

```

Whether the server supports sending log messages to the client.

#### prompts

```python
prompts: bool = False

```

Whether the server offers any prompt templates.

#### resources

```python
resources: bool = False

```

Whether the server offers any resources to read.

#### tools

```python
tools: bool = False

```

Whether the server offers any tools to call.

#### completions

```python
completions: bool = False

```

Whether the server offers autocompletion suggestions for prompts and resources.

#### from_mcp_sdk

```python
from_mcp_sdk(
    mcp_capabilities: ServerCapabilities,
) -> ServerCapabilities

```

Convert from MCP SDK ServerCapabilities to PydanticAI ServerCapabilities.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `mcp_capabilities` | `ServerCapabilities` | The MCP SDK ServerCapabilities object. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@classmethod
def from_mcp_sdk(cls, mcp_capabilities: mcp_types.ServerCapabilities) -> ServerCapabilities:
    """Convert from MCP SDK ServerCapabilities to PydanticAI ServerCapabilities.

    Args:
        mcp_capabilities: The MCP SDK ServerCapabilities object.
    """
    return cls(
        experimental=list(mcp_capabilities.experimental.keys()) if mcp_capabilities.experimental else None,
        logging=mcp_capabilities.logging is not None,
        prompts=mcp_capabilities.prompts is not None,
        resources=mcp_capabilities.resources is not None,
        tools=mcp_capabilities.tools is not None,
        completions=mcp_capabilities.completions is not None,
    )

```

### MCPServer

Bases: `AbstractToolset[Any]`, `ABC`

Base class for attaching agents to MCP servers.

See <https://modelcontextprotocol.io> for more information.

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
class MCPServer(AbstractToolset[Any], ABC):
    """Base class for attaching agents to MCP servers.

    See <https://modelcontextprotocol.io> for more information.
    """

    tool_prefix: str | None
    """A prefix to add to all tools that are registered with the server.

    If not empty, will include a trailing underscore(`_`).

    e.g. if `tool_prefix='foo'`, then a tool named `bar` will be registered as `foo_bar`
    """

    log_level: mcp_types.LoggingLevel | None
    """The log level to set when connecting to the server, if any.

    See <https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/logging#logging> for more details.

    If `None`, no log level will be set.
    """

    log_handler: LoggingFnT | None
    """A handler for logging messages from the server."""

    timeout: float
    """The timeout in seconds to wait for the client to initialize."""

    read_timeout: float
    """Maximum time in seconds to wait for new messages before timing out.

    This timeout applies to the long-lived connection after it's established.
    If no new messages are received within this time, the connection will be considered stale
    and may be closed. Defaults to 5 minutes (300 seconds).
    """

    process_tool_call: ProcessToolCallback | None
    """Hook to customize tool calling and optionally pass extra metadata."""

    allow_sampling: bool
    """Whether to allow MCP sampling through this client."""

    sampling_model: models.Model | None
    """The model to use for sampling."""

    max_retries: int
    """The maximum number of times to retry a tool call."""

    elicitation_callback: ElicitationFnT | None = None
    """Callback function to handle elicitation requests from the server."""

    _id: str | None

    _enter_lock: Lock = field(compare=False)
    _running_count: int
    _exit_stack: AsyncExitStack | None

    _client: ClientSession
    _read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    _write_stream: MemoryObjectSendStream[SessionMessage]
    _server_info: mcp_types.Implementation
    _server_capabilities: ServerCapabilities
    _instructions: str | None

    def __init__(
        self,
        tool_prefix: str | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        log_handler: LoggingFnT | None = None,
        timeout: float = 5,
        read_timeout: float = 5 * 60,
        process_tool_call: ProcessToolCallback | None = None,
        allow_sampling: bool = True,
        sampling_model: models.Model | None = None,
        max_retries: int = 1,
        elicitation_callback: ElicitationFnT | None = None,
        *,
        id: str | None = None,
    ):
        self.tool_prefix = tool_prefix
        self.log_level = log_level
        self.log_handler = log_handler
        self.timeout = timeout
        self.read_timeout = read_timeout
        self.process_tool_call = process_tool_call
        self.allow_sampling = allow_sampling
        self.sampling_model = sampling_model
        self.max_retries = max_retries
        self.elicitation_callback = elicitation_callback

        self._id = id or tool_prefix

        self.__post_init__()

    def __post_init__(self):
        self._enter_lock = Lock()
        self._running_count = 0
        self._exit_stack = None

    @abstractmethod
    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        """Create the streams for the MCP server."""
        raise NotImplementedError('MCP Server subclasses must implement this method.')
        yield

    @property
    def id(self) -> str | None:
        return self._id

    @id.setter
    def id(self, value: str | None):
        self._id = value

    @property
    def label(self) -> str:
        if self.id:
            return super().label  # pragma: no cover
        else:
            return repr(self)

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Set the `tool_prefix` attribute to avoid name conflicts.'

    @property
    def server_info(self) -> mcp_types.Implementation:
        """Access the information send by the MCP server during initialization."""
        if getattr(self, '_server_info', None) is None:
            raise AttributeError(
                f'The `{self.__class__.__name__}.server_info` is only instantiated after initialization.'
            )
        return self._server_info

    @property
    def capabilities(self) -> ServerCapabilities:
        """Access the capabilities advertised by the MCP server during initialization."""
        if getattr(self, '_server_capabilities', None) is None:
            raise AttributeError(
                f'The `{self.__class__.__name__}.capabilities` is only instantiated after initialization.'
            )
        return self._server_capabilities

    @property
    def instructions(self) -> str | None:
        """Access the instructions sent by the MCP server during initialization."""
        if not hasattr(self, '_instructions'):
            raise AttributeError(
                f'The `{self.__class__.__name__}.instructions` is only available after initialization.'
            )
        return self._instructions

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Retrieve tools that are currently active on the server.

        Note:
        - We don't cache tools as they might change.
        - We also don't subscribe to the server to avoid complexity.
        """
        async with self:  # Ensure server is running
            result = await self._client.list_tools()
        return result.tools

    async def direct_call_tool(
        self,
        name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Call a tool on the server.

        Args:
            name: The name of the tool to call.
            args: The arguments to pass to the tool.
            metadata: Request-level metadata (optional)

        Returns:
            The result of the tool call.

        Raises:
            ModelRetry: If the tool call fails.
        """
        async with self:  # Ensure server is running
            try:
                result = await self._client.send_request(
                    mcp_types.ClientRequest(
                        mcp_types.CallToolRequest(
                            method='tools/call',
                            params=mcp_types.CallToolRequestParams(
                                name=name,
                                arguments=args,
                                _meta=mcp_types.RequestParams.Meta(**metadata) if metadata else None,
                            ),
                        )
                    ),
                    mcp_types.CallToolResult,
                )
            except mcp_exceptions.McpError as e:
                raise exceptions.ModelRetry(e.error.message)

        if result.isError:
            message: str | None = None
            if result.content:  # pragma: no branch
                text_parts = [part.text for part in result.content if isinstance(part, mcp_types.TextContent)]
                message = '\n'.join(text_parts)

            raise exceptions.ModelRetry(message or 'MCP tool call failed')

        # Prefer structured content if there are only text parts, which per the docs would contain the JSON-encoded structured content for backward compatibility.
        # See https://github.com/modelcontextprotocol/python-sdk#structured-output
        if (structured := result.structuredContent) and not any(
            not isinstance(part, mcp_types.TextContent) for part in result.content
        ):
            # The MCP SDK wraps primitives and generic types like list in a `result` key, but we want to use the raw value returned by the tool function.
            # See https://github.com/modelcontextprotocol/python-sdk#structured-output
            if isinstance(structured, dict) and len(structured) == 1 and 'result' in structured:
                return structured['result']
            return structured

        mapped = [await self._map_tool_result_part(part) for part in result.content]
        return mapped[0] if len(mapped) == 1 else mapped

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> ToolResult:
        if self.tool_prefix:
            name = name.removeprefix(f'{self.tool_prefix}_')
            ctx = replace(ctx, tool_name=name)

        if self.process_tool_call is not None:
            return await self.process_tool_call(ctx, self.direct_call_tool, name, tool_args)
        else:
            return await self.direct_call_tool(name, tool_args)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        return {
            name: self.tool_for_tool_def(
                ToolDefinition(
                    name=name,
                    description=mcp_tool.description,
                    parameters_json_schema=mcp_tool.inputSchema,
                    metadata={
                        'meta': mcp_tool.meta,
                        'annotations': mcp_tool.annotations.model_dump() if mcp_tool.annotations else None,
                        'output_schema': mcp_tool.outputSchema or None,
                    },
                ),
            )
            for mcp_tool in await self.list_tools()
            if (name := f'{self.tool_prefix}_{mcp_tool.name}' if self.tool_prefix else mcp_tool.name)
        }

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[Any]:
        return ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=self.max_retries,
            args_validator=TOOL_SCHEMA_VALIDATOR,
        )

    async def list_resources(self) -> list[Resource]:
        """Retrieve resources that are currently present on the server.

        Note:
        - We don't cache resources as they might change.
        - We also don't subscribe to resource changes to avoid complexity.

        Raises:
            MCPError: If the server returns an error.
        """
        async with self:  # Ensure server is running
            if not self.capabilities.resources:
                return []
            try:
                result = await self._client.list_resources()
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e
        return [Resource.from_mcp_sdk(r) for r in result.resources]

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """Retrieve resource templates that are currently present on the server.

        Raises:
            MCPError: If the server returns an error.
        """
        async with self:  # Ensure server is running
            if not self.capabilities.resources:
                return []
            try:
                result = await self._client.list_resource_templates()
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e
        return [ResourceTemplate.from_mcp_sdk(t) for t in result.resourceTemplates]

    @overload
    async def read_resource(self, uri: str) -> str | messages.BinaryContent | list[str | messages.BinaryContent]: ...

    @overload
    async def read_resource(
        self, uri: Resource
    ) -> str | messages.BinaryContent | list[str | messages.BinaryContent]: ...

    async def read_resource(
        self, uri: str | Resource
    ) -> str | messages.BinaryContent | list[str | messages.BinaryContent]:
        """Read the contents of a specific resource by URI.

        Args:
            uri: The URI of the resource to read, or a Resource object.

        Returns:
            The resource contents. If the resource has a single content item, returns that item directly.
            If the resource has multiple content items, returns a list of items.

        Raises:
            MCPError: If the server returns an error.
        """
        resource_uri = uri if isinstance(uri, str) else uri.uri
        async with self:  # Ensure server is running
            try:
                result = await self._client.read_resource(AnyUrl(resource_uri))
            except mcp_exceptions.McpError as e:
                raise MCPError.from_mcp_sdk(e) from e

        return (
            self._get_content(result.contents[0])
            if len(result.contents) == 1
            else [self._get_content(resource) for resource in result.contents]
        )

    async def __aenter__(self) -> Self:
        """Enter the MCP server context.

        This will initialize the connection to the server.
        If this server is an [`MCPServerStdio`][pydantic_ai.mcp.MCPServerStdio], the server will first be started as a subprocess.

        This is a no-op if the MCP server has already been entered.
        """
        async with self._enter_lock:
            if self._running_count == 0:
                async with AsyncExitStack() as exit_stack:
                    self._read_stream, self._write_stream = await exit_stack.enter_async_context(self.client_streams())
                    client = ClientSession(
                        read_stream=self._read_stream,
                        write_stream=self._write_stream,
                        sampling_callback=self._sampling_callback if self.allow_sampling else None,
                        elicitation_callback=self.elicitation_callback,
                        logging_callback=self.log_handler,
                        read_timeout_seconds=timedelta(seconds=self.read_timeout),
                    )
                    self._client = await exit_stack.enter_async_context(client)

                    with anyio.fail_after(self.timeout):
                        result = await self._client.initialize()
                        self._server_info = result.serverInfo
                        self._server_capabilities = ServerCapabilities.from_mcp_sdk(result.capabilities)
                        self._instructions = result.instructions
                        if log_level := self.log_level:
                            await self._client.set_logging_level(log_level)

                    self._exit_stack = exit_stack.pop_all()
            self._running_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if self._running_count == 0:
            raise ValueError('MCPServer.__aexit__ called more times than __aenter__')
        async with self._enter_lock:
            self._running_count -= 1
            if self._running_count == 0 and self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None

    @property
    def is_running(self) -> bool:
        """Check if the MCP server is running."""
        return bool(self._running_count)

    async def _sampling_callback(
        self, context: RequestContext[ClientSession, Any], params: mcp_types.CreateMessageRequestParams
    ) -> mcp_types.CreateMessageResult | mcp_types.ErrorData:
        """MCP sampling callback."""
        if self.sampling_model is None:
            raise ValueError('Sampling model is not set')  # pragma: no cover

        pai_messages = _mcp.map_from_mcp_params(params)
        model_settings = models.ModelSettings()
        if max_tokens := params.maxTokens:  # pragma: no branch
            model_settings['max_tokens'] = max_tokens
        if temperature := params.temperature:  # pragma: no branch
            model_settings['temperature'] = temperature
        if stop_sequences := params.stopSequences:  # pragma: no branch
            model_settings['stop_sequences'] = stop_sequences

        model_response = await model_request(self.sampling_model, pai_messages, model_settings=model_settings)
        return mcp_types.CreateMessageResult(
            role='assistant',
            content=_mcp.map_from_model_response(model_response),
            model=self.sampling_model.model_name,
        )

    async def _map_tool_result_part(
        self, part: mcp_types.ContentBlock
    ) -> str | messages.BinaryContent | dict[str, Any] | list[Any]:
        # See https://github.com/jlowin/fastmcp/blob/main/docs/servers/tools.mdx#return-values

        if isinstance(part, mcp_types.TextContent):
            text = part.text
            if text.startswith(('[', '{')):
                try:
                    return pydantic_core.from_json(text)
                except ValueError:
                    pass
            return text
        elif isinstance(part, mcp_types.ImageContent):
            return messages.BinaryContent(data=base64.b64decode(part.data), media_type=part.mimeType)
        elif isinstance(part, mcp_types.AudioContent):
            # NOTE: The FastMCP server doesn't support audio content.
            # See <https://github.com/modelcontextprotocol/python-sdk/issues/952> for more details.
            return messages.BinaryContent(
                data=base64.b64decode(part.data), media_type=part.mimeType
            )  # pragma: no cover
        elif isinstance(part, mcp_types.EmbeddedResource):
            resource = part.resource
            return self._get_content(resource)
        elif isinstance(part, mcp_types.ResourceLink):
            return await self.read_resource(str(part.uri))
        else:
            assert_never(part)

    def _get_content(
        self, resource: mcp_types.TextResourceContents | mcp_types.BlobResourceContents
    ) -> str | messages.BinaryContent:
        if isinstance(resource, mcp_types.TextResourceContents):
            return resource.text
        elif isinstance(resource, mcp_types.BlobResourceContents):
            return messages.BinaryContent(
                data=base64.b64decode(resource.blob), media_type=resource.mimeType or 'application/octet-stream'
            )
        else:
            assert_never(resource)

    def __eq__(self, value: object, /) -> bool:
        return isinstance(value, MCPServer) and self.id == value.id and self.tool_prefix == value.tool_prefix

```

#### tool_prefix

```python
tool_prefix: str | None = tool_prefix

```

A prefix to add to all tools that are registered with the server.

If not empty, will include a trailing underscore(`_`).

e.g. if `tool_prefix='foo'`, then a tool named `bar` will be registered as `foo_bar`

#### log_level

```python
log_level: LoggingLevel | None = log_level

```

The log level to set when connecting to the server, if any.

See <https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/logging#logging> for more details.

If `None`, no log level will be set.

#### log_handler

```python
log_handler: LoggingFnT | None = log_handler

```

A handler for logging messages from the server.

#### timeout

```python
timeout: float = timeout

```

The timeout in seconds to wait for the client to initialize.

#### read_timeout

```python
read_timeout: float = read_timeout

```

Maximum time in seconds to wait for new messages before timing out.

This timeout applies to the long-lived connection after it's established. If no new messages are received within this time, the connection will be considered stale and may be closed. Defaults to 5 minutes (300 seconds).

#### process_tool_call

```python
process_tool_call: ProcessToolCallback | None = (
    process_tool_call
)

```

Hook to customize tool calling and optionally pass extra metadata.

#### allow_sampling

```python
allow_sampling: bool = allow_sampling

```

Whether to allow MCP sampling through this client.

#### sampling_model

```python
sampling_model: Model | None = sampling_model

```

The model to use for sampling.

#### max_retries

```python
max_retries: int = max_retries

```

The maximum number of times to retry a tool call.

#### elicitation_callback

```python
elicitation_callback: ElicitationFnT | None = (
    elicitation_callback
)

```

Callback function to handle elicitation requests from the server.

#### client_streams

```python
client_streams() -> AsyncIterator[
    tuple[
        MemoryObjectReceiveStream[
            SessionMessage | Exception
        ],
        MemoryObjectSendStream[SessionMessage],
    ]
]

```

Create the streams for the MCP server.

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
@abstractmethod
@asynccontextmanager
async def client_streams(
    self,
) -> AsyncIterator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]
]:
    """Create the streams for the MCP server."""
    raise NotImplementedError('MCP Server subclasses must implement this method.')
    yield

```

#### server_info

```python
server_info: Implementation

```

Access the information send by the MCP server during initialization.

#### capabilities

```python
capabilities: ServerCapabilities

```

Access the capabilities advertised by the MCP server during initialization.

#### instructions

```python
instructions: str | None

```

Access the instructions sent by the MCP server during initialization.

#### list_tools

```python
list_tools() -> list[Tool]

```

Retrieve tools that are currently active on the server.

Note:

- We don't cache tools as they might change.
- We also don't subscribe to the server to avoid complexity.

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
async def list_tools(self) -> list[mcp_types.Tool]:
    """Retrieve tools that are currently active on the server.

    Note:
    - We don't cache tools as they might change.
    - We also don't subscribe to the server to avoid complexity.
    """
    async with self:  # Ensure server is running
        result = await self._client.list_tools()
    return result.tools

```

#### direct_call_tool

```python
direct_call_tool(
    name: str,
    args: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> ToolResult

```

Call a tool on the server.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of the tool to call. | *required* | | `args` | `dict[str, Any]` | The arguments to pass to the tool. | *required* | | `metadata` | `dict[str, Any] | None` | Request-level metadata (optional) | `None` |

Returns:

| Type | Description | | --- | --- | | `ToolResult` | The result of the tool call. |

Raises:

| Type | Description | | --- | --- | | `ModelRetry` | If the tool call fails. |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
async def direct_call_tool(
    self,
    name: str,
    args: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    """Call a tool on the server.

    Args:
        name: The name of the tool to call.
        args: The arguments to pass to the tool.
        metadata: Request-level metadata (optional)

    Returns:
        The result of the tool call.

    Raises:
        ModelRetry: If the tool call fails.
    """
    async with self:  # Ensure server is running
        try:
            result = await self._client.send_request(
                mcp_types.ClientRequest(
                    mcp_types.CallToolRequest(
                        method='tools/call',
                        params=mcp_types.CallToolRequestParams(
                            name=name,
                            arguments=args,
                            _meta=mcp_types.RequestParams.Meta(**metadata) if metadata else None,
                        ),
                    )
                ),
                mcp_types.CallToolResult,
            )
        except mcp_exceptions.McpError as e:
            raise exceptions.ModelRetry(e.error.message)

    if result.isError:
        message: str | None = None
        if result.content:  # pragma: no branch
            text_parts = [part.text for part in result.content if isinstance(part, mcp_types.TextContent)]
            message = '\n'.join(text_parts)

        raise exceptions.ModelRetry(message or 'MCP tool call failed')

    # Prefer structured content if there are only text parts, which per the docs would contain the JSON-encoded structured content for backward compatibility.
    # See https://github.com/modelcontextprotocol/python-sdk#structured-output
    if (structured := result.structuredContent) and not any(
        not isinstance(part, mcp_types.TextContent) for part in result.content
    ):
        # The MCP SDK wraps primitives and generic types like list in a `result` key, but we want to use the raw value returned by the tool function.
        # See https://github.com/modelcontextprotocol/python-sdk#structured-output
        if isinstance(structured, dict) and len(structured) == 1 and 'result' in structured:
            return structured['result']
        return structured

    mapped = [await self._map_tool_result_part(part) for part in result.content]
    return mapped[0] if len(mapped) == 1 else mapped

```

#### list_resources

```python
list_resources() -> list[Resource]

```

Retrieve resources that are currently present on the server.

Note:

- We don't cache resources as they might change.
- We also don't subscribe to resource changes to avoid complexity.

Raises:

| Type | Description | | --- | --- | | `MCPError` | If the server returns an error. |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
async def list_resources(self) -> list[Resource]:
    """Retrieve resources that are currently present on the server.

    Note:
    - We don't cache resources as they might change.
    - We also don't subscribe to resource changes to avoid complexity.

    Raises:
        MCPError: If the server returns an error.
    """
    async with self:  # Ensure server is running
        if not self.capabilities.resources:
            return []
        try:
            result = await self._client.list_resources()
        except mcp_exceptions.McpError as e:
            raise MCPError.from_mcp_sdk(e) from e
    return [Resource.from_mcp_sdk(r) for r in result.resources]

```

#### list_resource_templates

```python
list_resource_templates() -> list[ResourceTemplate]

```

Retrieve resource templates that are currently present on the server.

Raises:

| Type | Description | | --- | --- | | `MCPError` | If the server returns an error. |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
async def list_resource_templates(self) -> list[ResourceTemplate]:
    """Retrieve resource templates that are currently present on the server.

    Raises:
        MCPError: If the server returns an error.
    """
    async with self:  # Ensure server is running
        if not self.capabilities.resources:
            return []
        try:
            result = await self._client.list_resource_templates()
        except mcp_exceptions.McpError as e:
            raise MCPError.from_mcp_sdk(e) from e
    return [ResourceTemplate.from_mcp_sdk(t) for t in result.resourceTemplates]

```

#### read_resource

```python
read_resource(
    uri: str,
) -> str | BinaryContent | list[str | BinaryContent]

```

```python
read_resource(
    uri: Resource,
) -> str | BinaryContent | list[str | BinaryContent]

```

```python
read_resource(
    uri: str | Resource,
) -> str | BinaryContent | list[str | BinaryContent]

```

Read the contents of a specific resource by URI.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `uri` | `str | Resource` | The URI of the resource to read, or a Resource object. | *required* |

Returns:

| Type | Description | | --- | --- | | `str | BinaryContent | list[str | BinaryContent]` | The resource contents. If the resource has a single content item, returns that item directly. | | `str | BinaryContent | list[str | BinaryContent]` | If the resource has multiple content items, returns a list of items. |

Raises:

| Type | Description | | --- | --- | | `MCPError` | If the server returns an error. |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
async def read_resource(
    self, uri: str | Resource
) -> str | messages.BinaryContent | list[str | messages.BinaryContent]:
    """Read the contents of a specific resource by URI.

    Args:
        uri: The URI of the resource to read, or a Resource object.

    Returns:
        The resource contents. If the resource has a single content item, returns that item directly.
        If the resource has multiple content items, returns a list of items.

    Raises:
        MCPError: If the server returns an error.
    """
    resource_uri = uri if isinstance(uri, str) else uri.uri
    async with self:  # Ensure server is running
        try:
            result = await self._client.read_resource(AnyUrl(resource_uri))
        except mcp_exceptions.McpError as e:
            raise MCPError.from_mcp_sdk(e) from e

    return (
        self._get_content(result.contents[0])
        if len(result.contents) == 1
        else [self._get_content(resource) for resource in result.contents]
    )

```

#### __aenter__

```python
__aenter__() -> Self

```

Enter the MCP server context.

This will initialize the connection to the server. If this server is an MCPServerStdio, the server will first be started as a subprocess.

This is a no-op if the MCP server has already been entered.

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
async def __aenter__(self) -> Self:
    """Enter the MCP server context.

    This will initialize the connection to the server.
    If this server is an [`MCPServerStdio`][pydantic_ai.mcp.MCPServerStdio], the server will first be started as a subprocess.

    This is a no-op if the MCP server has already been entered.
    """
    async with self._enter_lock:
        if self._running_count == 0:
            async with AsyncExitStack() as exit_stack:
                self._read_stream, self._write_stream = await exit_stack.enter_async_context(self.client_streams())
                client = ClientSession(
                    read_stream=self._read_stream,
                    write_stream=self._write_stream,
                    sampling_callback=self._sampling_callback if self.allow_sampling else None,
                    elicitation_callback=self.elicitation_callback,
                    logging_callback=self.log_handler,
                    read_timeout_seconds=timedelta(seconds=self.read_timeout),
                )
                self._client = await exit_stack.enter_async_context(client)

                with anyio.fail_after(self.timeout):
                    result = await self._client.initialize()
                    self._server_info = result.serverInfo
                    self._server_capabilities = ServerCapabilities.from_mcp_sdk(result.capabilities)
                    self._instructions = result.instructions
                    if log_level := self.log_level:
                        await self._client.set_logging_level(log_level)

                self._exit_stack = exit_stack.pop_all()
        self._running_count += 1
    return self

```

#### is_running

```python
is_running: bool

```

Check if the MCP server is running.

### MCPServerStdio

Bases: `MCPServer`

Runs an MCP server in a subprocess and communicates with it over stdin/stdout.

This class implements the stdio transport from the MCP specification. See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio> for more information.

Note

Using this class as an async context manager will start the server as a subprocess when entering the context, and stop it when exiting the context.

Example:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(  # (1)!
    'uv', args=['run', 'mcp-run-python', 'stdio'], timeout=10
)
agent = Agent('openai:gpt-4o', toolsets=[server])

```

1. See [MCP Run Python](https://github.com/pydantic/mcp-run-python) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

````python
class MCPServerStdio(MCPServer):
    """Runs an MCP server in a subprocess and communicates with it over stdin/stdout.

    This class implements the stdio transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio> for more information.

    !!! note
        Using this class as an async context manager will start the server as a subprocess when entering the context,
        and stop it when exiting the context.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio

    server = MCPServerStdio(  # (1)!
        'uv', args=['run', 'mcp-run-python', 'stdio'], timeout=10
    )
    agent = Agent('openai:gpt-4o', toolsets=[server])
    ```

    1. See [MCP Run Python](https://github.com/pydantic/mcp-run-python) for more information.
    """

    command: str
    """The command to run."""

    args: Sequence[str]
    """The arguments to pass to the command."""

    env: dict[str, str] | None
    """The environment variables the CLI server will have access to.

    By default the subprocess will not inherit any environment variables from the parent process.
    If you want to inherit the environment variables from the parent process, use `env=os.environ`.
    """

    cwd: str | Path | None
    """The working directory to use when spawning the process."""

    # last fields are re-defined from the parent class so they appear as fields
    tool_prefix: str | None
    log_level: mcp_types.LoggingLevel | None
    log_handler: LoggingFnT | None
    timeout: float
    read_timeout: float
    process_tool_call: ProcessToolCallback | None
    allow_sampling: bool
    sampling_model: models.Model | None
    max_retries: int
    elicitation_callback: ElicitationFnT | None = None

    def __init__(
        self,
        command: str,
        args: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        tool_prefix: str | None = None,
        log_level: mcp_types.LoggingLevel | None = None,
        log_handler: LoggingFnT | None = None,
        timeout: float = 5,
        read_timeout: float = 5 * 60,
        process_tool_call: ProcessToolCallback | None = None,
        allow_sampling: bool = True,
        sampling_model: models.Model | None = None,
        max_retries: int = 1,
        elicitation_callback: ElicitationFnT | None = None,
        id: str | None = None,
    ):
        """Build a new MCP server.

        Args:
            command: The command to run.
            args: The arguments to pass to the command.
            env: The environment variables to set in the subprocess.
            cwd: The working directory to use when spawning the process.
            tool_prefix: A prefix to add to all tools that are registered with the server.
            log_level: The log level to set when connecting to the server, if any.
            log_handler: A handler for logging messages from the server.
            timeout: The timeout in seconds to wait for the client to initialize.
            read_timeout: Maximum time in seconds to wait for new messages before timing out.
            process_tool_call: Hook to customize tool calling and optionally pass extra metadata.
            allow_sampling: Whether to allow MCP sampling through this client.
            sampling_model: The model to use for sampling.
            max_retries: The maximum number of times to retry a tool call.
            elicitation_callback: Callback function to handle elicitation requests from the server.
            id: An optional unique ID for the MCP server. An MCP server needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the server's activities within the workflow.
        """
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd

        super().__init__(
            tool_prefix,
            log_level,
            log_handler,
            timeout,
            read_timeout,
            process_tool_call,
            allow_sampling,
            sampling_model,
            max_retries,
            elicitation_callback,
            id=id,
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerStdio(**dct),
            core_schema.typed_dict_schema(
                {
                    'command': core_schema.typed_dict_field(core_schema.str_schema()),
                    'args': core_schema.typed_dict_field(core_schema.list_schema(core_schema.str_schema())),
                    'env': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()),
                        required=False,
                    ),
                }
            ),
        )

    @asynccontextmanager
    async def client_streams(
        self,
    ) -> AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]:
        server = StdioServerParameters(command=self.command, args=list(self.args), env=self.env, cwd=self.cwd)
        async with stdio_client(server=server) as (read_stream, write_stream):
            yield read_stream, write_stream

    def __repr__(self) -> str:
        repr_args = [
            f'command={self.command!r}',
            f'args={self.args!r}',
        ]
        if self.id:
            repr_args.append(f'id={self.id!r}')  # pragma: lax no cover
        return f'{self.__class__.__name__}({", ".join(repr_args)})'

    def __eq__(self, value: object, /) -> bool:
        return (
            super().__eq__(value)
            and isinstance(value, MCPServerStdio)
            and self.command == value.command
            and self.args == value.args
            and self.env == value.env
            and self.cwd == value.cwd
        )

````

#### __init__

```python
__init__(
    command: str,
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    tool_prefix: str | None = None,
    log_level: LoggingLevel | None = None,
    log_handler: LoggingFnT | None = None,
    timeout: float = 5,
    read_timeout: float = 5 * 60,
    process_tool_call: ProcessToolCallback | None = None,
    allow_sampling: bool = True,
    sampling_model: Model | None = None,
    max_retries: int = 1,
    elicitation_callback: ElicitationFnT | None = None,
    id: str | None = None
)

```

Build a new MCP server.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `command` | `str` | The command to run. | *required* | | `args` | `Sequence[str]` | The arguments to pass to the command. | *required* | | `env` | `dict[str, str] | None` | The environment variables to set in the subprocess. | `None` | | `cwd` | `str | Path | None` | The working directory to use when spawning the process. | `None` | | `tool_prefix` | `str | None` | A prefix to add to all tools that are registered with the server. | `None` | | `log_level` | `LoggingLevel | None` | The log level to set when connecting to the server, if any. | `None` | | `log_handler` | `LoggingFnT | None` | A handler for logging messages from the server. | `None` | | `timeout` | `float` | The timeout in seconds to wait for the client to initialize. | `5` | | `read_timeout` | `float` | Maximum time in seconds to wait for new messages before timing out. | `5 * 60` | | `process_tool_call` | `ProcessToolCallback | None` | Hook to customize tool calling and optionally pass extra metadata. | `None` | | `allow_sampling` | `bool` | Whether to allow MCP sampling through this client. | `True` | | `sampling_model` | `Model | None` | The model to use for sampling. | `None` | | `max_retries` | `int` | The maximum number of times to retry a tool call. | `1` | | `elicitation_callback` | `ElicitationFnT | None` | Callback function to handle elicitation requests from the server. | `None` | | `id` | `str | None` | An optional unique ID for the MCP server. An MCP server needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the server's activities within the workflow. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
def __init__(
    self,
    command: str,
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    tool_prefix: str | None = None,
    log_level: mcp_types.LoggingLevel | None = None,
    log_handler: LoggingFnT | None = None,
    timeout: float = 5,
    read_timeout: float = 5 * 60,
    process_tool_call: ProcessToolCallback | None = None,
    allow_sampling: bool = True,
    sampling_model: models.Model | None = None,
    max_retries: int = 1,
    elicitation_callback: ElicitationFnT | None = None,
    id: str | None = None,
):
    """Build a new MCP server.

    Args:
        command: The command to run.
        args: The arguments to pass to the command.
        env: The environment variables to set in the subprocess.
        cwd: The working directory to use when spawning the process.
        tool_prefix: A prefix to add to all tools that are registered with the server.
        log_level: The log level to set when connecting to the server, if any.
        log_handler: A handler for logging messages from the server.
        timeout: The timeout in seconds to wait for the client to initialize.
        read_timeout: Maximum time in seconds to wait for new messages before timing out.
        process_tool_call: Hook to customize tool calling and optionally pass extra metadata.
        allow_sampling: Whether to allow MCP sampling through this client.
        sampling_model: The model to use for sampling.
        max_retries: The maximum number of times to retry a tool call.
        elicitation_callback: Callback function to handle elicitation requests from the server.
        id: An optional unique ID for the MCP server. An MCP server needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the server's activities within the workflow.
    """
    self.command = command
    self.args = args
    self.env = env
    self.cwd = cwd

    super().__init__(
        tool_prefix,
        log_level,
        log_handler,
        timeout,
        read_timeout,
        process_tool_call,
        allow_sampling,
        sampling_model,
        max_retries,
        elicitation_callback,
        id=id,
    )

```

#### command

```python
command: str = command

```

The command to run.

#### args

```python
args: Sequence[str] = args

```

The arguments to pass to the command.

#### env

```python
env: dict[str, str] | None = env

```

The environment variables the CLI server will have access to.

By default the subprocess will not inherit any environment variables from the parent process. If you want to inherit the environment variables from the parent process, use `env=os.environ`.

#### cwd

```python
cwd: str | Path | None = cwd

```

The working directory to use when spawning the process.

### MCPServerSSE

Bases: `_MCPServerHTTP`

An MCP server that connects over streamable HTTP connections.

This class implements the SSE transport from the MCP specification. See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

Note

Using this class as an async context manager will create a new pool of HTTP connections to connect to a server which should already be running.

Example:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE('http://localhost:3001/sse')
agent = Agent('openai:gpt-4o', toolsets=[server])

```

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

````python
class MCPServerSSE(_MCPServerHTTP):
    """An MCP server that connects over streamable HTTP connections.

    This class implements the SSE transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerSSE

    server = MCPServerSSE('http://localhost:3001/sse')
    agent = Agent('openai:gpt-4o', toolsets=[server])
    ```
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerSSE(**dct),
            core_schema.typed_dict_schema(
                {
                    'url': core_schema.typed_dict_field(core_schema.str_schema()),
                    'headers': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()), required=False
                    ),
                }
            ),
        )

    @property
    def _transport_client(self):
        return sse_client  # pragma: no cover

    def __eq__(self, value: object, /) -> bool:
        return super().__eq__(value) and isinstance(value, MCPServerSSE) and self.url == value.url

````

### MCPServerHTTP

Bases: `MCPServerSSE`

Deprecated

The `MCPServerHTTP` class is deprecated, use `MCPServerSSE` instead.

An MCP server that connects over HTTP using the old SSE transport.

This class implements the SSE transport from the MCP specification. See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

Note

Using this class as an async context manager will create a new pool of HTTP connections to connect to a server which should already be running.

Example:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

server = MCPServerHTTP('http://localhost:3001/sse')
agent = Agent('openai:gpt-4o', toolsets=[server])

```

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

````python
@deprecated('The `MCPServerHTTP` class is deprecated, use `MCPServerSSE` instead.')
class MCPServerHTTP(MCPServerSSE):
    """An MCP server that connects over HTTP using the old SSE transport.

    This class implements the SSE transport from the MCP specification.
    See <https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10" test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerHTTP

    server = MCPServerHTTP('http://localhost:3001/sse')
    agent = Agent('openai:gpt-4o', toolsets=[server])
    ```
    """

````

### MCPServerStreamableHTTP

Bases: `_MCPServerHTTP`

An MCP server that connects over HTTP using the Streamable HTTP transport.

This class implements the Streamable HTTP transport from the MCP specification. See <https://modelcontextprotocol.io/introduction#streamable-http> for more information.

Note

Using this class as an async context manager will create a new pool of HTTP connections to connect to a server which should already be running.

Example:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')
agent = Agent('openai:gpt-4o', toolsets=[server])

```

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

````python
class MCPServerStreamableHTTP(_MCPServerHTTP):
    """An MCP server that connects over HTTP using the Streamable HTTP transport.

    This class implements the Streamable HTTP transport from the MCP specification.
    See <https://modelcontextprotocol.io/introduction#streamable-http> for more information.

    !!! note
        Using this class as an async context manager will create a new pool of HTTP connections to connect
        to a server which should already be running.

    Example:
    ```python {py="3.10"}
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStreamableHTTP

    server = MCPServerStreamableHTTP('http://localhost:8000/mcp')
    agent = Agent('openai:gpt-4o', toolsets=[server])
    ```
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda dct: MCPServerStreamableHTTP(**dct),
            core_schema.typed_dict_schema(
                {
                    'url': core_schema.typed_dict_field(core_schema.str_schema()),
                    'headers': core_schema.typed_dict_field(
                        core_schema.dict_schema(core_schema.str_schema(), core_schema.str_schema()), required=False
                    ),
                }
            ),
        )

    @property
    def _transport_client(self):
        return streamablehttp_client

    def __eq__(self, value: object, /) -> bool:
        return super().__eq__(value) and isinstance(value, MCPServerStreamableHTTP) and self.url == value.url

````

### load_mcp_servers

```python
load_mcp_servers(
    config_path: str | Path,
) -> list[
    MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE
]

```

Load MCP servers from a configuration file.

Environment variables can be referenced in the configuration file using:

- `${VAR_NAME}` syntax - expands to the value of VAR_NAME, raises error if not defined
- `${VAR_NAME:-default}` syntax - expands to VAR_NAME if set, otherwise uses the default value

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `config_path` | `str | Path` | The path to the configuration file. | *required* |

Returns:

| Type | Description | | --- | --- | | `list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE]` | A list of MCP servers. |

Raises:

| Type | Description | | --- | --- | | `FileNotFoundError` | If the configuration file does not exist. | | `ValidationError` | If the configuration file does not match the schema. | | `ValueError` | If an environment variable referenced in the configuration is not defined and no default value is provided. |

Source code in `pydantic_ai_slim/pydantic_ai/mcp.py`

```python
def load_mcp_servers(config_path: str | Path) -> list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE]:
    """Load MCP servers from a configuration file.

    Environment variables can be referenced in the configuration file using:
    - `${VAR_NAME}` syntax - expands to the value of VAR_NAME, raises error if not defined
    - `${VAR_NAME:-default}` syntax - expands to VAR_NAME if set, otherwise uses the default value

    Args:
        config_path: The path to the configuration file.

    Returns:
        A list of MCP servers.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValidationError: If the configuration file does not match the schema.
        ValueError: If an environment variable referenced in the configuration is not defined and no default value is provided.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f'Config file {config_path} not found')

    config_data = pydantic_core.from_json(config_path.read_bytes())
    expanded_config_data = _expand_env_vars(config_data)
    config = MCPServerConfig.model_validate(expanded_config_data)

    servers: list[MCPServerStdio | MCPServerStreamableHTTP | MCPServerSSE] = []
    for name, server in config.mcp_servers.items():
        server.id = name
        server.tool_prefix = name
        servers.append(server)

    return servers

```

