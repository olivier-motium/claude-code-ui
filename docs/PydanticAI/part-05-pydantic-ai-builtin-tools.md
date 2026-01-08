<!-- Source: _complete-doc.md -->

# `pydantic_ai.builtin_tools`

### AbstractBuiltinTool

Bases: `ABC`

A builtin tool that can be used by an agent.

This class is abstract and cannot be instantiated directly.

The builtin tools are passed to the model as part of the `ModelRequestParameters`.

Source code in `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

```python
@dataclass(kw_only=True)
class AbstractBuiltinTool(ABC):
    """A builtin tool that can be used by an agent.

    This class is abstract and cannot be instantiated directly.

    The builtin tools are passed to the model as part of the `ModelRequestParameters`.
    """

    kind: str = 'unknown_builtin_tool'
    """Built-in tool identifier, this should be available on all built-in tools as a discriminator."""

    @property
    def unique_id(self) -> str:
        """A unique identifier for the builtin tool.

        If multiple instances of the same builtin tool can be passed to the model, subclasses should override this property to allow them to be distinguished.
        """
        return self.kind

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _BUILTIN_TOOL_TYPES[cls.kind] = cls

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        if cls is not AbstractBuiltinTool:
            return handler(cls)

        tools = _BUILTIN_TOOL_TYPES.values()
        if len(tools) == 1:  # pragma: no cover
            tools_type = next(iter(tools))
        else:
            tools_annotated = [Annotated[tool, pydantic.Tag(tool.kind)] for tool in tools]
            tools_type = Annotated[Union[tuple(tools_annotated)], pydantic.Discriminator(_tool_discriminator)]  # noqa: UP007

        return handler(tools_type)

```

#### kind

```python
kind: str = 'unknown_builtin_tool'

```

Built-in tool identifier, this should be available on all built-in tools as a discriminator.

#### unique_id

```python
unique_id: str

```

A unique identifier for the builtin tool.

If multiple instances of the same builtin tool can be passed to the model, subclasses should override this property to allow them to be distinguished.

### WebSearchTool

Bases: `AbstractBuiltinTool`

A builtin tool that allows your agent to search the web for information.

The parameters that PydanticAI passes depend on the model, as some parameters may not be supported by certain models.

Supported by:

- Anthropic
- OpenAI Responses
- Groq
- Google

Source code in `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

```python
@dataclass(kw_only=True)
class WebSearchTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to search the web for information.

    The parameters that PydanticAI passes depend on the model, as some parameters may not be supported by certain models.

    Supported by:

    * Anthropic
    * OpenAI Responses
    * Groq
    * Google
    """

    search_context_size: Literal['low', 'medium', 'high'] = 'medium'
    """The `search_context_size` parameter controls how much context is retrieved from the web to help the tool formulate a response.

    Supported by:

    * OpenAI Responses
    """

    user_location: WebSearchUserLocation | None = None
    """The `user_location` parameter allows you to localize search results based on a user's location.

    Supported by:

    * Anthropic
    * OpenAI Responses
    """

    blocked_domains: list[str] | None = None
    """If provided, these domains will never appear in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
    * Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>
    """

    allowed_domains: list[str] | None = None
    """If provided, only these domains will be included in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
    * Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>
    """

    max_uses: int | None = None
    """If provided, the tool will stop searching the web after the given number of uses.

    Supported by:

    * Anthropic
    """

    kind: str = 'web_search'
    """The kind of tool."""

```

#### search_context_size

```python
search_context_size: Literal["low", "medium", "high"] = (
    "medium"
)

```

The `search_context_size` parameter controls how much context is retrieved from the web to help the tool formulate a response.

Supported by:

- OpenAI Responses

#### user_location

```python
user_location: WebSearchUserLocation | None = None

```

The `user_location` parameter allows you to localize search results based on a user's location.

Supported by:

- Anthropic
- OpenAI Responses

#### blocked_domains

```python
blocked_domains: list[str] | None = None

```

If provided, these domains will never appear in results.

With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

Supported by:

- Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
- Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>

#### allowed_domains

```python
allowed_domains: list[str] | None = None

```

If provided, only these domains will be included in results.

With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

Supported by:

- Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
- Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>

#### max_uses

```python
max_uses: int | None = None

```

If provided, the tool will stop searching the web after the given number of uses.

Supported by:

- Anthropic

#### kind

```python
kind: str = 'web_search'

```

The kind of tool.

### WebSearchUserLocation

Bases: `TypedDict`

Allows you to localize search results based on a user's location.

Supported by:

- Anthropic
- OpenAI Responses

Source code in `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

```python
class WebSearchUserLocation(TypedDict, total=False):
    """Allows you to localize search results based on a user's location.

    Supported by:

    * Anthropic
    * OpenAI Responses
    """

    city: str
    """The city where the user is located."""

    country: str
    """The country where the user is located. For OpenAI, this must be a 2-letter country code (e.g., 'US', 'GB')."""

    region: str
    """The region or state where the user is located."""

    timezone: str
    """The timezone of the user's location."""

```

#### city

```python
city: str

```

The city where the user is located.

#### country

```python
country: str

```

The country where the user is located. For OpenAI, this must be a 2-letter country code (e.g., 'US', 'GB').

#### region

```python
region: str

```

The region or state where the user is located.

#### timezone

```python
timezone: str

```

The timezone of the user's location.

### CodeExecutionTool

Bases: `AbstractBuiltinTool`

A builtin tool that allows your agent to execute code.

Supported by:

- Anthropic
- OpenAI Responses
- Google

Source code in `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

```python
@dataclass(kw_only=True)
class CodeExecutionTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to execute code.

    Supported by:

    * Anthropic
    * OpenAI Responses
    * Google
    """

    kind: str = 'code_execution'
    """The kind of tool."""

```

#### kind

```python
kind: str = 'code_execution'

```

The kind of tool.

### UrlContextTool

Bases: `AbstractBuiltinTool`

Allows your agent to access contents from URLs.

Supported by:

- Google

Source code in `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

```python
@dataclass(kw_only=True)
class UrlContextTool(AbstractBuiltinTool):
    """Allows your agent to access contents from URLs.

    Supported by:

    * Google
    """

    kind: str = 'url_context'
    """The kind of tool."""

```

#### kind

```python
kind: str = 'url_context'

```

The kind of tool.

### ImageGenerationTool

Bases: `AbstractBuiltinTool`

A builtin tool that allows your agent to generate images.

Supported by:

- OpenAI Responses
- Google

Source code in `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

```python
@dataclass(kw_only=True)
class ImageGenerationTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to generate images.

    Supported by:

    * OpenAI Responses
    * Google
    """

    background: Literal['transparent', 'opaque', 'auto'] = 'auto'
    """Background type for the generated image.

    Supported by:

    * OpenAI Responses. 'transparent' is only supported for 'png' and 'webp' output formats.
    """

    input_fidelity: Literal['high', 'low'] | None = None
    """
    Control how much effort the model will exert to match the style and features,
    especially facial features, of input images.

    Supported by:

    * OpenAI Responses. Default: 'low'.
    """

    moderation: Literal['auto', 'low'] = 'auto'
    """Moderation level for the generated image.

    Supported by:

    * OpenAI Responses
    """

    output_compression: int = 100
    """Compression level for the output image.

    Supported by:

    * OpenAI Responses. Only supported for 'png' and 'webp' output formats.
    """

    output_format: Literal['png', 'webp', 'jpeg'] | None = None
    """The output format of the generated image.

    Supported by:

    * OpenAI Responses. Default: 'png'.
    """

    partial_images: int = 0
    """
    Number of partial images to generate in streaming mode.

    Supported by:

    * OpenAI Responses. Supports 0 to 3.
    """

    quality: Literal['low', 'medium', 'high', 'auto'] = 'auto'
    """The quality of the generated image.

    Supported by:

    * OpenAI Responses
    """

    size: Literal['1024x1024', '1024x1536', '1536x1024', 'auto'] = 'auto'
    """The size of the generated image.

    Supported by:

    * OpenAI Responses
    """

    kind: str = 'image_generation'
    """The kind of tool."""

```

#### background

```python
background: Literal["transparent", "opaque", "auto"] = (
    "auto"
)

```

Background type for the generated image.

Supported by:

- OpenAI Responses. 'transparent' is only supported for 'png' and 'webp' output formats.

#### input_fidelity

```python
input_fidelity: Literal['high', 'low'] | None = None

```

Control how much effort the model will exert to match the style and features, especially facial features, of input images.

Supported by:

- OpenAI Responses. Default: 'low'.

#### moderation

```python
moderation: Literal['auto', 'low'] = 'auto'

```

Moderation level for the generated image.

Supported by:

- OpenAI Responses

#### output_compression

```python
output_compression: int = 100

```

Compression level for the output image.

Supported by:

- OpenAI Responses. Only supported for 'png' and 'webp' output formats.

#### output_format

```python
output_format: Literal['png', 'webp', 'jpeg'] | None = None

```

The output format of the generated image.

Supported by:

- OpenAI Responses. Default: 'png'.

#### partial_images

```python
partial_images: int = 0

```

Number of partial images to generate in streaming mode.

Supported by:

- OpenAI Responses. Supports 0 to 3.

#### quality

```python
quality: Literal['low', 'medium', 'high', 'auto'] = 'auto'

```

The quality of the generated image.

Supported by:

- OpenAI Responses

#### size

```python
size: Literal[
    "1024x1024", "1024x1536", "1536x1024", "auto"
] = "auto"

```

The size of the generated image.

Supported by:

- OpenAI Responses

#### kind

```python
kind: str = 'image_generation'

```

The kind of tool.

### MemoryTool

Bases: `AbstractBuiltinTool`

A builtin tool that allows your agent to use memory.

Supported by:

- Anthropic

Source code in `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

```python
@dataclass(kw_only=True)
class MemoryTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to use memory.

    Supported by:

    * Anthropic
    """

    kind: str = 'memory'
    """The kind of tool."""

```

#### kind

```python
kind: str = 'memory'

```

The kind of tool.

### MCPServerTool

Bases: `AbstractBuiltinTool`

A builtin tool that allows your agent to use MCP servers.

Supported by:

- OpenAI Responses
- Anthropic

Source code in `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

```python
@dataclass(kw_only=True)
class MCPServerTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to use MCP servers.

    Supported by:

    * OpenAI Responses
    * Anthropic
    """

    id: str
    """A unique identifier for the MCP server."""

    url: str
    """The URL of the MCP server to use.

    For OpenAI Responses, it is possible to use `connector_id` by providing it as `x-openai-connector:<connector_id>`.
    """

    authorization_token: str | None = None
    """Authorization header to use when making requests to the MCP server.

    Supported by:

    * OpenAI Responses
    * Anthropic
    """

    description: str | None = None
    """A description of the MCP server.

    Supported by:

    * OpenAI Responses
    """

    allowed_tools: list[str] | None = None
    """A list of tools that the MCP server can use.

    Supported by:

    * OpenAI Responses
    * Anthropic
    """

    headers: dict[str, str] | None = None
    """Optional HTTP headers to send to the MCP server.

    Use for authentication or other purposes.

    Supported by:

    * OpenAI Responses
    """

    kind: str = 'mcp_server'

    @property
    def unique_id(self) -> str:
        return ':'.join([self.kind, self.id])

```

#### id

```python
id: str

```

A unique identifier for the MCP server.

#### url

```python
url: str

```

The URL of the MCP server to use.

For OpenAI Responses, it is possible to use `connector_id` by providing it as `x-openai-connector:<connector_id>`.

#### authorization_token

```python
authorization_token: str | None = None

```

Authorization header to use when making requests to the MCP server.

Supported by:

- OpenAI Responses
- Anthropic

#### description

```python
description: str | None = None

```

A description of the MCP server.

Supported by:

- OpenAI Responses

#### allowed_tools

```python
allowed_tools: list[str] | None = None

```

A list of tools that the MCP server can use.

Supported by:

- OpenAI Responses
- Anthropic

#### headers

```python
headers: dict[str, str] | None = None

```

Optional HTTP headers to send to the MCP server.

Use for authentication or other purposes.

Supported by:

- OpenAI Responses

# `pydantic_ai.common_tools`

### duckduckgo_search_tool

```python
duckduckgo_search_tool(
    duckduckgo_client: DDGS | None = None,
    max_results: int | None = None,
)

```

Creates a DuckDuckGo search tool.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `duckduckgo_client` | `DDGS | None` | The DuckDuckGo search client. | `None` | | `max_results` | `int | None` | The maximum number of results. If None, returns results only from the first response. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/common_tools/duckduckgo.py`

```python
def duckduckgo_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None):
    """Creates a DuckDuckGo search tool.

    Args:
        duckduckgo_client: The DuckDuckGo search client.
        max_results: The maximum number of results. If None, returns results only from the first response.
    """
    return Tool[Any](
        DuckDuckGoSearchTool(client=duckduckgo_client or DDGS(), max_results=max_results).__call__,
        name='duckduckgo_search',
        description='Searches DuckDuckGo for the given query and returns the results.',
    )

```

### tavily_search_tool

```python
tavily_search_tool(api_key: str)

```

Creates a Tavily search tool.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `api_key` | `str` | The Tavily API key. You can get one by signing up at https://app.tavily.com/home. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/common_tools/tavily.py`

```python
def tavily_search_tool(api_key: str):
    """Creates a Tavily search tool.

    Args:
        api_key: The Tavily API key.

            You can get one by signing up at [https://app.tavily.com/home](https://app.tavily.com/home).
    """
    return Tool[Any](
        TavilySearchTool(client=AsyncTavilyClient(api_key)).__call__,
        name='tavily_search',
        description='Searches Tavily for the given query and returns the results.',
    )

```

# `pydantic_ai.direct`

Methods for making imperative requests to language models with minimal abstraction.

These methods allow you to make requests to LLMs where the only abstraction is input and output schema translation so you can use all models with the same API.

These methods are thin wrappers around Model implementations.

### model_request

```python
model_request(
    model: Model | KnownModelName | str,
    messages: Sequence[ModelMessage],
    *,
    model_settings: ModelSettings | None = None,
    model_request_parameters: (
        ModelRequestParameters | None
    ) = None,
    instrument: InstrumentationSettings | bool | None = None
) -> ModelResponse

```

Make a non-streamed request to a model.

model_request_example.py

```py
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request


async def main():
    model_response = await model_request(
        'anthropic:claude-haiku-4-5',
        [ModelRequest.user_text_prompt('What is the capital of France?')]  # (1)!
    )
    print(model_response)
    '''
    ModelResponse(
        parts=[TextPart(content='The capital of France is Paris.')],
        usage=RequestUsage(input_tokens=56, output_tokens=7),
        model_name='claude-haiku-4-5',
        timestamp=datetime.datetime(...),
    )
    '''

```

1. See ModelRequest.user_text_prompt for details.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model` | `Model | KnownModelName | str` | The model to make a request to. We allow str here since the actual list of allowed models changes frequently. | *required* | | `messages` | `Sequence[ModelMessage]` | Messages to send to the model | *required* | | `model_settings` | `ModelSettings | None` | optional model settings | `None` | | `model_request_parameters` | `ModelRequestParameters | None` | optional model request parameters | `None` | | `instrument` | `InstrumentationSettings | bool | None` | Whether to instrument the request with OpenTelemetry/Logfire, if None the value from logfire.instrument_pydantic_ai is used. | `None` |

Returns:

| Type | Description | | --- | --- | | `ModelResponse` | The model response and token usage associated with the request. |

Source code in `pydantic_ai_slim/pydantic_ai/direct.py`

````python
async def model_request(
    model: models.Model | models.KnownModelName | str,
    messages: Sequence[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> messages.ModelResponse:
    """Make a non-streamed request to a model.

    ```py title="model_request_example.py"
    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request


    async def main():
        model_response = await model_request(
            'anthropic:claude-haiku-4-5',
            [ModelRequest.user_text_prompt('What is the capital of France?')]  # (1)!
        )
        print(model_response)
        '''
        ModelResponse(
            parts=[TextPart(content='The capital of France is Paris.')],
            usage=RequestUsage(input_tokens=56, output_tokens=7),
            model_name='claude-haiku-4-5',
            timestamp=datetime.datetime(...),
        )
        '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        The model response and token usage associated with the request.
    """
    model_instance = _prepare_model(model, instrument)
    return await model_instance.request(
        list(messages),
        model_settings,
        model_request_parameters or models.ModelRequestParameters(),
    )

````

### model_request_sync

```python
model_request_sync(
    model: Model | KnownModelName | str,
    messages: Sequence[ModelMessage],
    *,
    model_settings: ModelSettings | None = None,
    model_request_parameters: (
        ModelRequestParameters | None
    ) = None,
    instrument: InstrumentationSettings | bool | None = None
) -> ModelResponse

```

Make a Synchronous, non-streamed request to a model.

This is a convenience method that wraps model_request with `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

model_request_sync_example.py

```py
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

model_response = model_request_sync(
    'anthropic:claude-haiku-4-5',
    [ModelRequest.user_text_prompt('What is the capital of France?')]  # (1)!
)
print(model_response)
'''
ModelResponse(
    parts=[TextPart(content='The capital of France is Paris.')],
    usage=RequestUsage(input_tokens=56, output_tokens=7),
    model_name='claude-haiku-4-5',
    timestamp=datetime.datetime(...),
)
'''

```

1. See ModelRequest.user_text_prompt for details.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model` | `Model | KnownModelName | str` | The model to make a request to. We allow str here since the actual list of allowed models changes frequently. | *required* | | `messages` | `Sequence[ModelMessage]` | Messages to send to the model | *required* | | `model_settings` | `ModelSettings | None` | optional model settings | `None` | | `model_request_parameters` | `ModelRequestParameters | None` | optional model request parameters | `None` | | `instrument` | `InstrumentationSettings | bool | None` | Whether to instrument the request with OpenTelemetry/Logfire, if None the value from logfire.instrument_pydantic_ai is used. | `None` |

Returns:

| Type | Description | | --- | --- | | `ModelResponse` | The model response and token usage associated with the request. |

Source code in `pydantic_ai_slim/pydantic_ai/direct.py`

````python
def model_request_sync(
    model: models.Model | models.KnownModelName | str,
    messages: Sequence[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> messages.ModelResponse:
    """Make a Synchronous, non-streamed request to a model.

    This is a convenience method that wraps [`model_request`][pydantic_ai.direct.model_request] with
    `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

    ```py title="model_request_sync_example.py"
    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request_sync

    model_response = model_request_sync(
        'anthropic:claude-haiku-4-5',
        [ModelRequest.user_text_prompt('What is the capital of France?')]  # (1)!
    )
    print(model_response)
    '''
    ModelResponse(
        parts=[TextPart(content='The capital of France is Paris.')],
        usage=RequestUsage(input_tokens=56, output_tokens=7),
        model_name='claude-haiku-4-5',
        timestamp=datetime.datetime(...),
    )
    '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        The model response and token usage associated with the request.
    """
    return _get_event_loop().run_until_complete(
        model_request(
            model,
            list(messages),
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
            instrument=instrument,
        )
    )

````

### model_request_stream

```python
model_request_stream(
    model: Model | KnownModelName | str,
    messages: Sequence[ModelMessage],
    *,
    model_settings: ModelSettings | None = None,
    model_request_parameters: (
        ModelRequestParameters | None
    ) = None,
    instrument: InstrumentationSettings | bool | None = None
) -> AbstractAsyncContextManager[StreamedResponse]

```

Make a streamed async request to a model.

model_request_stream_example.py

```py
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream


async def main():
    messages = [ModelRequest.user_text_prompt('Who was Albert Einstein?')]  # (1)!
    async with model_request_stream('openai:gpt-4.1-mini', messages) as stream:
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        print(chunks)
        '''
        [
            PartStartEvent(index=0, part=TextPart(content='Albert Einstein was ')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=0, delta=TextPartDelta(content_delta='a German-born theoretical ')
            ),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='physicist.')),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content='Albert Einstein was a German-born theoretical physicist.'
                ),
            ),
        ]
        '''

```

1. See ModelRequest.user_text_prompt for details.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model` | `Model | KnownModelName | str` | The model to make a request to. We allow str here since the actual list of allowed models changes frequently. | *required* | | `messages` | `Sequence[ModelMessage]` | Messages to send to the model | *required* | | `model_settings` | `ModelSettings | None` | optional model settings | `None` | | `model_request_parameters` | `ModelRequestParameters | None` | optional model request parameters | `None` | | `instrument` | `InstrumentationSettings | bool | None` | Whether to instrument the request with OpenTelemetry/Logfire, if None the value from logfire.instrument_pydantic_ai is used. | `None` |

Returns:

| Type | Description | | --- | --- | | `AbstractAsyncContextManager[StreamedResponse]` | A stream response async context manager. |

Source code in `pydantic_ai_slim/pydantic_ai/direct.py`

````python
def model_request_stream(
    model: models.Model | models.KnownModelName | str,
    messages: Sequence[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> AbstractAsyncContextManager[models.StreamedResponse]:
    """Make a streamed async request to a model.

    ```py {title="model_request_stream_example.py"}

    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request_stream


    async def main():
        messages = [ModelRequest.user_text_prompt('Who was Albert Einstein?')]  # (1)!
        async with model_request_stream('openai:gpt-4.1-mini', messages) as stream:
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            print(chunks)
            '''
            [
                PartStartEvent(index=0, part=TextPart(content='Albert Einstein was ')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(
                    index=0, delta=TextPartDelta(content_delta='a German-born theoretical ')
                ),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='physicist.')),
                PartEndEvent(
                    index=0,
                    part=TextPart(
                        content='Albert Einstein was a German-born theoretical physicist.'
                    ),
                ),
            ]
            '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        A [stream response][pydantic_ai.models.StreamedResponse] async context manager.
    """
    model_instance = _prepare_model(model, instrument)
    return model_instance.request_stream(
        list(messages),
        model_settings,
        model_request_parameters or models.ModelRequestParameters(),
    )

````

### model_request_stream_sync

```python
model_request_stream_sync(
    model: Model | KnownModelName | str,
    messages: Sequence[ModelMessage],
    *,
    model_settings: ModelSettings | None = None,
    model_request_parameters: (
        ModelRequestParameters | None
    ) = None,
    instrument: InstrumentationSettings | bool | None = None
) -> StreamedResponseSync

```

Make a streamed synchronous request to a model.

This is the synchronous version of model_request_stream. It uses threading to run the asynchronous stream in the background while providing a synchronous iterator interface.

model_request_stream_sync_example.py

```py
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream_sync

messages = [ModelRequest.user_text_prompt('Who was Albert Einstein?')]
with model_request_stream_sync('openai:gpt-4.1-mini', messages) as stream:
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    print(chunks)
    '''
    [
        PartStartEvent(index=0, part=TextPart(content='Albert Einstein was ')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta='a German-born theoretical ')
        ),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='physicist.')),
        PartEndEvent(
            index=0,
            part=TextPart(
                content='Albert Einstein was a German-born theoretical physicist.'
            ),
        ),
    ]
    '''

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model` | `Model | KnownModelName | str` | The model to make a request to. We allow str here since the actual list of allowed models changes frequently. | *required* | | `messages` | `Sequence[ModelMessage]` | Messages to send to the model | *required* | | `model_settings` | `ModelSettings | None` | optional model settings | `None` | | `model_request_parameters` | `ModelRequestParameters | None` | optional model request parameters | `None` | | `instrument` | `InstrumentationSettings | bool | None` | Whether to instrument the request with OpenTelemetry/Logfire, if None the value from logfire.instrument_pydantic_ai is used. | `None` |

Returns:

| Type | Description | | --- | --- | | `StreamedResponseSync` | A sync stream response context manager. |

Source code in `pydantic_ai_slim/pydantic_ai/direct.py`

````python
def model_request_stream_sync(
    model: models.Model | models.KnownModelName | str,
    messages: Sequence[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
    instrument: instrumented_models.InstrumentationSettings | bool | None = None,
) -> StreamedResponseSync:
    """Make a streamed synchronous request to a model.

    This is the synchronous version of [`model_request_stream`][pydantic_ai.direct.model_request_stream].
    It uses threading to run the asynchronous stream in the background while providing a synchronous iterator interface.

    ```py {title="model_request_stream_sync_example.py"}

    from pydantic_ai import ModelRequest
    from pydantic_ai.direct import model_request_stream_sync

    messages = [ModelRequest.user_text_prompt('Who was Albert Einstein?')]
    with model_request_stream_sync('openai:gpt-4.1-mini', messages) as stream:
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        print(chunks)
        '''
        [
            PartStartEvent(index=0, part=TextPart(content='Albert Einstein was ')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=0, delta=TextPartDelta(content_delta='a German-born theoretical ')
            ),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='physicist.')),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content='Albert Einstein was a German-born theoretical physicist.'
                ),
            ),
        ]
        '''
    ```

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters
        instrument: Whether to instrument the request with OpenTelemetry/Logfire, if `None` the value from
            [`logfire.instrument_pydantic_ai`][logfire.Logfire.instrument_pydantic_ai] is used.

    Returns:
        A [sync stream response][pydantic_ai.direct.StreamedResponseSync] context manager.
    """
    async_stream_cm = model_request_stream(
        model=model,
        messages=list(messages),
        model_settings=model_settings,
        model_request_parameters=model_request_parameters,
        instrument=instrument,
    )

    return StreamedResponseSync(async_stream_cm)

````

### StreamedResponseSync

Synchronous wrapper to async streaming responses by running the async producer in a background thread and providing a synchronous iterator.

This class must be used as a context manager with the `with` statement.

Source code in `pydantic_ai_slim/pydantic_ai/direct.py`

```python
@dataclass
class StreamedResponseSync:
    """Synchronous wrapper to async streaming responses by running the async producer in a background thread and providing a synchronous iterator.

    This class must be used as a context manager with the `with` statement.
    """

    _async_stream_cm: AbstractAsyncContextManager[StreamedResponse]
    _queue: queue.Queue[messages.ModelResponseStreamEvent | Exception | None] = field(
        default_factory=queue.Queue, init=False
    )
    _thread: threading.Thread | None = field(default=None, init=False)
    _stream_response: StreamedResponse | None = field(default=None, init=False)
    _exception: Exception | None = field(default=None, init=False)
    _context_entered: bool = field(default=False, init=False)
    _stream_ready: threading.Event = field(default_factory=threading.Event, init=False)

    def __enter__(self) -> StreamedResponseSync:
        self._context_entered = True
        self._start_producer()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        self._cleanup()

    def __iter__(self) -> Iterator[messages.ModelResponseStreamEvent]:
        """Stream the response as an iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
        self._check_context_manager_usage()

        while True:
            item = self._queue.get()
            if item is None:  # End of stream
                break
            elif isinstance(item, Exception):
                raise item
            else:
                yield item

    def __repr__(self) -> str:
        if self._stream_response:
            return repr(self._stream_response)
        else:
            return f'{self.__class__.__name__}(context_entered={self._context_entered})'

    __str__ = __repr__

    def _check_context_manager_usage(self) -> None:
        if not self._context_entered:
            raise RuntimeError(
                'StreamedResponseSync must be used as a context manager. '
                'Use: `with model_request_stream_sync(...) as stream:`'
            )

    def _ensure_stream_ready(self) -> StreamedResponse:
        self._check_context_manager_usage()

        if self._stream_response is None:
            # Wait for the background thread to signal that the stream is ready
            if not self._stream_ready.wait(timeout=STREAM_INITIALIZATION_TIMEOUT):
                raise RuntimeError('Stream failed to initialize within timeout')

            if self._stream_response is None:  # pragma: no cover
                raise RuntimeError('Stream failed to initialize')

        return self._stream_response

    def _start_producer(self):
        self._thread = threading.Thread(target=self._async_producer, daemon=True)
        self._thread.start()

    def _async_producer(self):
        async def _consume_async_stream():
            try:
                async with self._async_stream_cm as stream:
                    self._stream_response = stream
                    # Signal that the stream is ready
                    self._stream_ready.set()
                    async for event in stream:
                        self._queue.put(event)
            except Exception as e:
                # Signal ready even on error so waiting threads don't hang
                self._stream_ready.set()
                self._queue.put(e)
            finally:
                self._queue.put(None)  # Signal end

        _get_event_loop().run_until_complete(_consume_async_stream())

    def _cleanup(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()

    # TODO (v2): Drop in favor of `response` property
    def get(self) -> messages.ModelResponse:
        """Build a ModelResponse from the data received from the stream so far."""
        return self._ensure_stream_ready().get()

    @property
    def response(self) -> messages.ModelResponse:
        """Get the current state of the response."""
        return self.get()

    # TODO (v2): Make this a property
    def usage(self) -> RequestUsage:
        """Get the usage of the response so far."""
        return self._ensure_stream_ready().usage()

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._ensure_stream_ready().model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._ensure_stream_ready().timestamp

```

#### __iter__

```python
__iter__() -> Iterator[ModelResponseStreamEvent]

```

Stream the response as an iterable of ModelResponseStreamEvents.

Source code in `pydantic_ai_slim/pydantic_ai/direct.py`

```python
def __iter__(self) -> Iterator[messages.ModelResponseStreamEvent]:
    """Stream the response as an iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
    self._check_context_manager_usage()

    while True:
        item = self._queue.get()
        if item is None:  # End of stream
            break
        elif isinstance(item, Exception):
            raise item
        else:
            yield item

```

#### get

```python
get() -> ModelResponse

```

Build a ModelResponse from the data received from the stream so far.

Source code in `pydantic_ai_slim/pydantic_ai/direct.py`

```python
def get(self) -> messages.ModelResponse:
    """Build a ModelResponse from the data received from the stream so far."""
    return self._ensure_stream_ready().get()

```

#### response

```python
response: ModelResponse

```

Get the current state of the response.

#### usage

```python
usage() -> RequestUsage

```

Get the usage of the response so far.

Source code in `pydantic_ai_slim/pydantic_ai/direct.py`

```python
def usage(self) -> RequestUsage:
    """Get the usage of the response so far."""
    return self._ensure_stream_ready().usage()

```

#### model_name

```python
model_name: str

```

Get the model name of the response.

#### timestamp

```python
timestamp: datetime

```

Get the timestamp of the response.

# `pydantic_ai.durable_exec`

### TemporalAgent

Bases: `WrapperAgent[AgentDepsT, OutputDataT]`

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`

````python
class TemporalAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        activity_config: ActivityConfig | None = None,
        model_activity_config: ActivityConfig | None = None,
        toolset_activity_config: dict[str, ActivityConfig] | None = None,
        tool_activity_config: dict[str, dict[str, ActivityConfig | Literal[False]]] | None = None,
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
        temporalize_toolset_func: Callable[
            [
                AbstractToolset[AgentDepsT],
                str,
                ActivityConfig,
                dict[str, ActivityConfig | Literal[False]],
                type[AgentDepsT],
                type[TemporalRunContext[AgentDepsT]],
            ],
            AbstractToolset[AgentDepsT],
        ] = temporalize_toolset,
    ):
        """Wrap an agent to enable it to be used inside a Temporal workflow, by automatically offloading model requests, tool calls, and MCP server communication to Temporal activities.

        After wrapping, the original agent can still be used as normal outside of the Temporal workflow, but any changes to its model or toolsets after wrapping will not be reflected in the durable agent.

        Args:
            wrapped: The agent to wrap.
            name: Optional unique agent name to use in the Temporal activities' names. If not provided, the agent's `name` will be used.
            event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
            activity_config: The base Temporal activity config to use for all activities. If no config is provided, a `start_to_close_timeout` of 60 seconds is used.
            model_activity_config: The Temporal activity config to use for model request activities. This is merged with the base activity config.
            toolset_activity_config: The Temporal activity config to use for get-tools and call-tool activities for specific toolsets identified by ID. This is merged with the base activity config.
            tool_activity_config: The Temporal activity config to use for specific tool call activities identified by toolset ID and tool name.
                This is merged with the base and toolset-specific activity configs.
                If a tool does not use IO, you can specify `False` to disable using an activity.
                Note that the tool is required to be defined as an `async` function as non-async tools are run in threads which are non-deterministic and thus not supported outside of activities.
            run_context_type: The `TemporalRunContext` subclass to use to serialize and deserialize the run context for use inside a Temporal activity.
                By default, only the `deps`, `retries`, `tool_call_id`, `tool_name`, `retry` and `run_step` attributes will be available.
                To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute.
            temporalize_toolset_func: Optional function to use to prepare "leaf" toolsets (i.e. those that implement their own tool listing and calling) for Temporal by wrapping them in a `TemporalWrapperToolset` that moves methods that require IO to Temporal activities.
                If not provided, only `FunctionToolset` and `MCPServer` will be prepared for Temporal.
                The function takes the toolset, the activity name prefix, the toolset-specific activity config, the tool-specific activity configs and the run context type.
        """
        super().__init__(wrapped)

        self._name = name
        self._event_stream_handler = event_stream_handler
        self.run_context_type = run_context_type

        # start_to_close_timeout is required
        activity_config = activity_config or ActivityConfig(start_to_close_timeout=timedelta(seconds=60))

        # `pydantic_ai.exceptions.UserError` and `pydantic.errors.PydanticUserError` are not retryable
        retry_policy = activity_config.get('retry_policy') or RetryPolicy()
        retry_policy.non_retryable_error_types = [
            *(retry_policy.non_retryable_error_types or []),
            UserError.__name__,
            PydanticUserError.__name__,
        ]
        activity_config['retry_policy'] = retry_policy
        self.activity_config = activity_config

        model_activity_config = model_activity_config or {}
        toolset_activity_config = toolset_activity_config or {}
        tool_activity_config = tool_activity_config or {}

        if self.name is None:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with Temporal. The name will be used to identify the agent's activities within the workflow."
            )

        activity_name_prefix = f'agent__{self.name}'

        activities: list[Callable[..., Any]] = []
        if not isinstance(wrapped.model, Model):
            raise UserError(
                'An agent needs to have a `model` in order to be used with Temporal, it cannot be set at agent run time.'
            )

        async def event_stream_handler_activity(params: _EventStreamHandlerParams, deps: AgentDepsT) -> None:
            # We can never get here without an `event_stream_handler`, as `TemporalAgent.run_stream` and `TemporalAgent.iter` raise an error saying to use `TemporalAgent.run` instead,
            # and that only ends up calling `event_stream_handler` if it is set.
            assert self.event_stream_handler is not None

            run_context = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)

            async def streamed_response():
                yield params.event

            await self.event_stream_handler(run_context, streamed_response())

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        event_stream_handler_activity.__annotations__['deps'] = self.deps_type

        self.event_stream_handler_activity = activity.defn(name=f'{activity_name_prefix}__event_stream_handler')(
            event_stream_handler_activity
        )
        activities.append(self.event_stream_handler_activity)

        temporal_model = TemporalModel(
            wrapped.model,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config | model_activity_config,
            deps_type=self.deps_type,
            run_context_type=self.run_context_type,
            event_stream_handler=self.event_stream_handler,
        )
        activities.extend(temporal_model.temporal_activities)

        def temporalize_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            id = toolset.id
            if id is None:
                raise UserError(
                    "Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) need to have a unique `id` in order to be used with Temporal. The ID will be used to identify the toolset's activities within the workflow."
                )

            toolset = temporalize_toolset_func(
                toolset,
                activity_name_prefix,
                activity_config | toolset_activity_config.get(id, {}),
                tool_activity_config.get(id, {}),
                self.deps_type,
                self.run_context_type,
            )
            if isinstance(toolset, TemporalWrapperToolset):
                activities.extend(toolset.temporal_activities)
            return toolset

        temporal_toolsets = [toolset.visit_and_replace(temporalize_toolset) for toolset in wrapped.toolsets]

        self._model = temporal_model
        self._toolsets = temporal_toolsets
        self._temporal_activities = activities

        self._temporal_overrides_active: ContextVar[bool] = ContextVar('_temporal_overrides_active', default=False)

    @property
    def name(self) -> str | None:
        return self._name or super().name

    @name.setter
    def name(self, value: str | None) -> None:  # pragma: no cover
        raise UserError(
            'The agent name cannot be changed after creation. If you need to change the name, create a new agent.'
        )

    @property
    def model(self) -> Model:
        return self._model

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        handler = self._event_stream_handler or super().event_stream_handler
        if handler is None:
            return None
        elif workflow.in_workflow():
            return self._call_event_stream_handler_activity
        else:
            return handler

    async def _call_event_stream_handler_activity(
        self, ctx: RunContext[AgentDepsT], stream: AsyncIterable[_messages.AgentStreamEvent]
    ) -> None:
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        async for event in stream:
            await workflow.execute_activity(
                activity=self.event_stream_handler_activity,
                args=[
                    _EventStreamHandlerParams(
                        event=event,
                        serialized_run_context=serialized_run_context,
                    ),
                    ctx.deps,
                ],
                **self.activity_config,
            )

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._temporal_overrides():
            return super().toolsets

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return self._temporal_activities

    @contextmanager
    def _temporal_overrides(self) -> Iterator[None]:
        # We reset tools here as the temporalized function toolset is already in self._toolsets.
        with super().override(model=self._model, toolsets=self._toolsets, tools=[]):
            token = self._temporal_overrides_active.set(True)
            try:
                yield
            except PydanticSerializationError as e:
                raise UserError(
                    "The `deps` object failed to be serialized. Temporal requires all objects that are passed to activities to be serializable using Pydantic's `TypeAdapter`."
                ) from e
            finally:
                self._temporal_overrides_active.reset(token)

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode.

        This method builds an internal agent graph (using system prompts, tools and result schemas) and then
        runs the graph to completion. The result of the run is returned.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            agent_run = await agent.run('What is the capital of France?')
            print(agent_run.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            event_stream_handler: Optional event stream handler to use for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow() and event_stream_handler is not None:
            raise UserError(
                'Event stream handler cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
            )

        with self._temporal_overrides():
            return await super().run(
                user_prompt,
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
                event_stream_handler=event_stream_handler or self.event_stream_handler,
                **_deprecated_kwargs,
            )

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Synchronously run the agent with a user prompt.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        result_sync = agent.run_sync('What is the capital of Italy?')
        print(result_sync.output)
        #> The capital of Italy is Rome.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            event_stream_handler: Optional event stream handler to use for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow():
            raise UserError(
                '`agent.run_sync()` cannot be used inside a Temporal workflow. Use `await agent.run()` instead.'
            )

        return super().run_sync(
            user_prompt,
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
            event_stream_handler=event_stream_handler,
            **_deprecated_kwargs,
        )

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async mode, returning a streamed response.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.run_stream('What is the capital of the UK?') as response:
                print(await response.get_output())
                #> The capital of the UK is London.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow():
            raise UserError(
                '`agent.run_stream()` cannot be used inside a Temporal workflow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        async with super().run_stream(
            user_prompt,
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
            event_stream_handler=event_stream_handler,
            builtin_tools=builtin_tools,
            **_deprecated_kwargs,
        ) as result:
            yield result

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[OutputDataT]]: ...

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]]: ...

    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
        """Run the agent with a user prompt in async mode and stream events from the run.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
        uses the `event_stream_handler` kwarg to get a stream of events from the run.

        Example:
        ```python
        from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

        agent = Agent('openai:gpt-4o')

        async def main():
            events: list[AgentStreamEvent | AgentRunResultEvent] = []
            async for event in agent.run_stream_events('What is the capital of France?'):
                events.append(event)
            print(events)
            '''
            [
                PartStartEvent(index=0, part=TextPart(content='The capital of ')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
                PartEndEvent(
                    index=0, part=TextPart(content='The capital of France is Paris. ')
                ),
                AgentRunResultEvent(
                    result=AgentRunResult(output='The capital of France is Paris. ')
                ),
            ]
            '''
        ```

        Arguments are the same as for [`self.run`][pydantic_ai.agent.AbstractAgent.run],
        except that `event_stream_handler` is now allowed.

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
            run result.
        """
        if workflow.in_workflow():
            raise UserError(
                '`agent.run_stream_events()` cannot be used inside a Temporal workflow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        return super().run_stream_events(
            user_prompt,
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
        )

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
            async with agent.iter('What is the capital of France?') as agent_run:
                async for node in agent_run:
                    nodes.append(node)
            print(nodes)
            '''
            [
                UserPromptNode(
                    user_prompt='What is the capital of France?',
                    instructions_functions=[],
                    system_prompts=(),
                    system_prompt_functions=[],
                    system_prompt_dynamic_functions={},
                ),
                ModelRequestNode(
                    request=ModelRequest(
                        parts=[
                            UserPromptPart(
                                content='What is the capital of France?',
                                timestamp=datetime.datetime(...),
                            )
                        ],
                        run_id='...',
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
                        run_id='...',
                    )
                ),
                End(data=FinalResult(output='The capital of France is Paris.')),
            ]
            '''
            print(agent_run.result.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow():
            if not self._temporal_overrides_active.get():
                raise UserError(
                    '`agent.iter()` cannot be used inside a Temporal workflow. '
                    'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
                )

            if model is not None:
                raise UserError(
                    'Model cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
                )
            if toolsets is not None:
                raise UserError(
                    'Toolsets cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
                )

        async with super().iter(
            user_prompt=user_prompt,
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
            **_deprecated_kwargs,
        ) as run:
            yield run

    @contextmanager
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            instructions: The instructions to use instead of the instructions registered with the agent.
        """
        if workflow.in_workflow():
            if _utils.is_set(model):
                raise UserError(
                    'Model cannot be contextually overridden inside a Temporal workflow, it must be set at agent creation time.'
                )
            if _utils.is_set(toolsets):
                raise UserError(
                    'Toolsets cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
                )
            if _utils.is_set(tools):
                raise UserError(
                    'Tools cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
                )

        with super().override(
            name=name,
            deps=deps,
            model=model,
            toolsets=toolsets,
            tools=tools,
            instructions=instructions,
        ):
            yield

````

#### __init__

```python
__init__(
    wrapped: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    name: str | None = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    activity_config: ActivityConfig | None = None,
    model_activity_config: ActivityConfig | None = None,
    toolset_activity_config: (
        dict[str, ActivityConfig] | None
    ) = None,
    tool_activity_config: (
        dict[
            str, dict[str, ActivityConfig | Literal[False]]
        ]
        | None
    ) = None,
    run_context_type: type[
        TemporalRunContext[AgentDepsT]
    ] = TemporalRunContext[AgentDepsT],
    temporalize_toolset_func: Callable[
        [
            AbstractToolset[AgentDepsT],
            str,
            ActivityConfig,
            dict[str, ActivityConfig | Literal[False]],
            type[AgentDepsT],
            type[TemporalRunContext[AgentDepsT]],
        ],
        AbstractToolset[AgentDepsT],
    ] = temporalize_toolset
)

```

Wrap an agent to enable it to be used inside a Temporal workflow, by automatically offloading model requests, tool calls, and MCP server communication to Temporal activities.

After wrapping, the original agent can still be used as normal outside of the Temporal workflow, but any changes to its model or toolsets after wrapping will not be reflected in the durable agent.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `wrapped` | `AbstractAgent[AgentDepsT, OutputDataT]` | The agent to wrap. | *required* | | `name` | `str | None` | Optional unique agent name to use in the Temporal activities' names. If not provided, the agent's name will be used. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use instead of the one set on the wrapped agent. | `None` | | `activity_config` | `ActivityConfig | None` | The base Temporal activity config to use for all activities. If no config is provided, a start_to_close_timeout of 60 seconds is used. | `None` | | `model_activity_config` | `ActivityConfig | None` | The Temporal activity config to use for model request activities. This is merged with the base activity config. | `None` | | `toolset_activity_config` | `dict[str, ActivityConfig] | None` | The Temporal activity config to use for get-tools and call-tool activities for specific toolsets identified by ID. This is merged with the base activity config. | `None` | | `tool_activity_config` | `dict[str, dict[str, ActivityConfig | Literal[False]]] | None` | The Temporal activity config to use for specific tool call activities identified by toolset ID and tool name. This is merged with the base and toolset-specific activity configs. If a tool does not use IO, you can specify False to disable using an activity. Note that the tool is required to be defined as an async function as non-async tools are run in threads which are non-deterministic and thus not supported outside of activities. | `None` | | `run_context_type` | `type[TemporalRunContext[AgentDepsT]]` | The TemporalRunContext subclass to use to serialize and deserialize the run context for use inside a Temporal activity. By default, only the deps, retries, tool_call_id, tool_name, retry and run_step attributes will be available. To make another attribute available, create a TemporalRunContext subclass with a custom serialize_run_context class method that returns a dictionary that includes the attribute. | `TemporalRunContext[AgentDepsT]` | | `temporalize_toolset_func` | `Callable[[AbstractToolset[AgentDepsT], str, ActivityConfig, dict[str, ActivityConfig | Literal[False]], type[AgentDepsT], type[TemporalRunContext[AgentDepsT]]], AbstractToolset[AgentDepsT]]` | Optional function to use to prepare "leaf" toolsets (i.e. those that implement their own tool listing and calling) for Temporal by wrapping them in a TemporalWrapperToolset that moves methods that require IO to Temporal activities. If not provided, only FunctionToolset and MCPServer will be prepared for Temporal. The function takes the toolset, the activity name prefix, the toolset-specific activity config, the tool-specific activity configs and the run context type. | `temporalize_toolset` |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`

```python
def __init__(
    self,
    wrapped: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    name: str | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    activity_config: ActivityConfig | None = None,
    model_activity_config: ActivityConfig | None = None,
    toolset_activity_config: dict[str, ActivityConfig] | None = None,
    tool_activity_config: dict[str, dict[str, ActivityConfig | Literal[False]]] | None = None,
    run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    temporalize_toolset_func: Callable[
        [
            AbstractToolset[AgentDepsT],
            str,
            ActivityConfig,
            dict[str, ActivityConfig | Literal[False]],
            type[AgentDepsT],
            type[TemporalRunContext[AgentDepsT]],
        ],
        AbstractToolset[AgentDepsT],
    ] = temporalize_toolset,
):
    """Wrap an agent to enable it to be used inside a Temporal workflow, by automatically offloading model requests, tool calls, and MCP server communication to Temporal activities.

    After wrapping, the original agent can still be used as normal outside of the Temporal workflow, but any changes to its model or toolsets after wrapping will not be reflected in the durable agent.

    Args:
        wrapped: The agent to wrap.
        name: Optional unique agent name to use in the Temporal activities' names. If not provided, the agent's `name` will be used.
        event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
        activity_config: The base Temporal activity config to use for all activities. If no config is provided, a `start_to_close_timeout` of 60 seconds is used.
        model_activity_config: The Temporal activity config to use for model request activities. This is merged with the base activity config.
        toolset_activity_config: The Temporal activity config to use for get-tools and call-tool activities for specific toolsets identified by ID. This is merged with the base activity config.
        tool_activity_config: The Temporal activity config to use for specific tool call activities identified by toolset ID and tool name.
            This is merged with the base and toolset-specific activity configs.
            If a tool does not use IO, you can specify `False` to disable using an activity.
            Note that the tool is required to be defined as an `async` function as non-async tools are run in threads which are non-deterministic and thus not supported outside of activities.
        run_context_type: The `TemporalRunContext` subclass to use to serialize and deserialize the run context for use inside a Temporal activity.
            By default, only the `deps`, `retries`, `tool_call_id`, `tool_name`, `retry` and `run_step` attributes will be available.
            To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute.
        temporalize_toolset_func: Optional function to use to prepare "leaf" toolsets (i.e. those that implement their own tool listing and calling) for Temporal by wrapping them in a `TemporalWrapperToolset` that moves methods that require IO to Temporal activities.
            If not provided, only `FunctionToolset` and `MCPServer` will be prepared for Temporal.
            The function takes the toolset, the activity name prefix, the toolset-specific activity config, the tool-specific activity configs and the run context type.
    """
    super().__init__(wrapped)

    self._name = name
    self._event_stream_handler = event_stream_handler
    self.run_context_type = run_context_type

    # start_to_close_timeout is required
    activity_config = activity_config or ActivityConfig(start_to_close_timeout=timedelta(seconds=60))

    # `pydantic_ai.exceptions.UserError` and `pydantic.errors.PydanticUserError` are not retryable
    retry_policy = activity_config.get('retry_policy') or RetryPolicy()
    retry_policy.non_retryable_error_types = [
        *(retry_policy.non_retryable_error_types or []),
        UserError.__name__,
        PydanticUserError.__name__,
    ]
    activity_config['retry_policy'] = retry_policy
    self.activity_config = activity_config

    model_activity_config = model_activity_config or {}
    toolset_activity_config = toolset_activity_config or {}
    tool_activity_config = tool_activity_config or {}

    if self.name is None:
        raise UserError(
            "An agent needs to have a unique `name` in order to be used with Temporal. The name will be used to identify the agent's activities within the workflow."
        )

    activity_name_prefix = f'agent__{self.name}'

    activities: list[Callable[..., Any]] = []
    if not isinstance(wrapped.model, Model):
        raise UserError(
            'An agent needs to have a `model` in order to be used with Temporal, it cannot be set at agent run time.'
        )

    async def event_stream_handler_activity(params: _EventStreamHandlerParams, deps: AgentDepsT) -> None:
        # We can never get here without an `event_stream_handler`, as `TemporalAgent.run_stream` and `TemporalAgent.iter` raise an error saying to use `TemporalAgent.run` instead,
        # and that only ends up calling `event_stream_handler` if it is set.
        assert self.event_stream_handler is not None

        run_context = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)

        async def streamed_response():
            yield params.event

        await self.event_stream_handler(run_context, streamed_response())

    # Set type hint explicitly so that Temporal can take care of serialization and deserialization
    event_stream_handler_activity.__annotations__['deps'] = self.deps_type

    self.event_stream_handler_activity = activity.defn(name=f'{activity_name_prefix}__event_stream_handler')(
        event_stream_handler_activity
    )
    activities.append(self.event_stream_handler_activity)

    temporal_model = TemporalModel(
        wrapped.model,
        activity_name_prefix=activity_name_prefix,
        activity_config=activity_config | model_activity_config,
        deps_type=self.deps_type,
        run_context_type=self.run_context_type,
        event_stream_handler=self.event_stream_handler,
    )
    activities.extend(temporal_model.temporal_activities)

    def temporalize_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        id = toolset.id
        if id is None:
            raise UserError(
                "Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) need to have a unique `id` in order to be used with Temporal. The ID will be used to identify the toolset's activities within the workflow."
            )

        toolset = temporalize_toolset_func(
            toolset,
            activity_name_prefix,
            activity_config | toolset_activity_config.get(id, {}),
            tool_activity_config.get(id, {}),
            self.deps_type,
            self.run_context_type,
        )
        if isinstance(toolset, TemporalWrapperToolset):
            activities.extend(toolset.temporal_activities)
        return toolset

    temporal_toolsets = [toolset.visit_and_replace(temporalize_toolset) for toolset in wrapped.toolsets]

    self._model = temporal_model
    self._toolsets = temporal_toolsets
    self._temporal_activities = activities

    self._temporal_overrides_active: ContextVar[bool] = ContextVar('_temporal_overrides_active', default=False)

```

#### run

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[OutputDataT]

```

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[RunOutputDataT]

```

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AgentRunResult[Any]

```

Run the agent with a user prompt in async mode.

This method builds an internal agent graph (using system prompts, tools and result schemas) and then runs the graph to completion. The result of the run is returned.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    agent_run = await agent.run('What is the capital of France?')
    print(agent_run.output)
    #> The capital of France is Paris.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AgentRunResult[Any]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`

````python
async def run(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Never,
) -> AgentRunResult[Any]:
    """Run the agent with a user prompt in async mode.

    This method builds an internal agent graph (using system prompts, tools and result schemas) and then
    runs the graph to completion. The result of the run is returned.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        agent_run = await agent.run('What is the capital of France?')
        print(agent_run.output)
        #> The capital of France is Paris.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        event_stream_handler: Optional event stream handler to use for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    if workflow.in_workflow() and event_stream_handler is not None:
        raise UserError(
            'Event stream handler cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
        )

    with self._temporal_overrides():
        return await super().run(
            user_prompt,
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
            event_stream_handler=event_stream_handler or self.event_stream_handler,
            **_deprecated_kwargs,
        )

````

#### run_sync

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[OutputDataT]

```

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[RunOutputDataT]

```

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AgentRunResult[Any]

```

Synchronously run the agent with a user prompt.

This is a convenience method that wraps self.run with `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)
#> The capital of Italy is Rome.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AgentRunResult[Any]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`

````python
def run_sync(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Never,
) -> AgentRunResult[Any]:
    """Synchronously run the agent with a user prompt.

    This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
    You therefore can't use this method inside async code or if there's an active event loop.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    result_sync = agent.run_sync('What is the capital of Italy?')
    print(result_sync.output)
    #> The capital of Italy is Rome.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        event_stream_handler: Optional event stream handler to use for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    if workflow.in_workflow():
        raise UserError(
            '`agent.run_sync()` cannot be used inside a Temporal workflow. Use `await agent.run()` instead.'
        )

    return super().run_sync(
        user_prompt,
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
        event_stream_handler=event_stream_handler,
        **_deprecated_kwargs,
    )

````

#### run_stream

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AbstractAsyncContextManager[
    StreamedRunResult[AgentDepsT, OutputDataT]
]

```

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AbstractAsyncContextManager[
    StreamedRunResult[AgentDepsT, RunOutputDataT]
]

```

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]

```

Run the agent with a user prompt in async mode, returning a streamed response.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.run_stream('What is the capital of the UK?') as response:
        print(await response.get_output())
        #> The capital of the UK is London.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[StreamedRunResult[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`

````python
@asynccontextmanager
async def run_stream(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Never,
) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
    """Run the agent with a user prompt in async mode, returning a streamed response.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        async with agent.run_stream('What is the capital of the UK?') as response:
            print(await response.get_output())
            #> The capital of the UK is London.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.
        event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

    Returns:
        The result of the run.
    """
    if workflow.in_workflow():
        raise UserError(
            '`agent.run_stream()` cannot be used inside a Temporal workflow. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        )

    async with super().run_stream(
        user_prompt,
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
        event_stream_handler=event_stream_handler,
        builtin_tools=builtin_tools,
        **_deprecated_kwargs,
    ) as result:
        yield result

````

#### run_stream_events

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[OutputDataT]
]

```

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]
]

```

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[Any]
]

```

Run the agent with a user prompt in async mode and stream events from the run.

This is a convenience method that wraps self.run and uses the `event_stream_handler` kwarg to get a stream of events from the run.

Example:

```python
from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

agent = Agent('openai:gpt-4o')

async def main():
    events: list[AgentStreamEvent | AgentRunResultEvent] = []
    async for event in agent.run_stream_events('What is the capital of France?'):
        events.append(event)
    print(events)
    '''
    [
        PartStartEvent(index=0, part=TextPart(content='The capital of ')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
        PartEndEvent(
            index=0, part=TextPart(content='The capital of France is Paris. ')
        ),
        AgentRunResultEvent(
            result=AgentRunResult(output='The capital of France is Paris. ')
        ),
    ]
    '''

```

Arguments are the same as for self.run, except that `event_stream_handler` is now allowed.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]` | An async iterable of stream events AgentStreamEvent and finally a AgentRunResultEvent with the final | | `AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]` | run result. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`

````python
def run_stream_events(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
    """Run the agent with a user prompt in async mode and stream events from the run.

    This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
    uses the `event_stream_handler` kwarg to get a stream of events from the run.

    Example:
    ```python
    from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

    agent = Agent('openai:gpt-4o')

    async def main():
        events: list[AgentStreamEvent | AgentRunResultEvent] = []
        async for event in agent.run_stream_events('What is the capital of France?'):
            events.append(event)
        print(events)
        '''
        [
            PartStartEvent(index=0, part=TextPart(content='The capital of ')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
            PartEndEvent(
                index=0, part=TextPart(content='The capital of France is Paris. ')
            ),
            AgentRunResultEvent(
                result=AgentRunResult(output='The capital of France is Paris. ')
            ),
        ]
        '''
    ```

    Arguments are the same as for [`self.run`][pydantic_ai.agent.AbstractAgent.run],
    except that `event_stream_handler` is now allowed.

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
        run result.
    """
    if workflow.in_workflow():
        raise UserError(
            '`agent.run_stream_events()` cannot be used inside a Temporal workflow. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        )

    return super().run_stream_events(
        user_prompt,
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
    )

````

#### iter

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, OutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    **_deprecated_kwargs: Never
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, RunOutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    **_deprecated_kwargs: Never
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]

```

A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the stream of events coming from the execution of tools.

The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics, and the final result of the run once it has completed.

For more details, see the documentation of `AgentRun`.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    nodes = []
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            nodes.append(node)
    print(nodes)
    '''
    [
        UserPromptNode(
            user_prompt='What is the capital of France?',
            instructions_functions=[],
            system_prompts=(),
            system_prompt_functions=[],
            system_prompt_dynamic_functions={},
        ),
        ModelRequestNode(
            request=ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=datetime.datetime(...),
                    )
                ],
                run_id='...',
            )
        ),
        CallToolsNode(
            model_response=ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(input_tokens=56, output_tokens=7),
                model_name='gpt-4o',
                timestamp=datetime.datetime(...),
                run_id='...',
            )
        ),
        End(data=FinalResult(output='The capital of France is Paris.')),
    ]
    '''
    print(agent_run.result.output)
    #> The capital of France is Paris.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentRun[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`

````python
@asynccontextmanager
async def iter(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    **_deprecated_kwargs: Never,
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
    """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

    This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
    `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
    executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
    stream of events coming from the execution of tools.

    The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
    and the final result of the run once it has completed.

    For more details, see the documentation of `AgentRun`.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
        async with agent.iter('What is the capital of France?') as agent_run:
            async for node in agent_run:
                nodes.append(node)
        print(nodes)
        '''
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                instructions_functions=[],
                system_prompts=(),
                system_prompt_functions=[],
                system_prompt_dynamic_functions={},
            ),
            ModelRequestNode(
                request=ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=datetime.datetime(...),
                        )
                    ],
                    run_id='...',
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='The capital of France is Paris.')],
                    usage=RequestUsage(input_tokens=56, output_tokens=7),
                    model_name='gpt-4o',
                    timestamp=datetime.datetime(...),
                    run_id='...',
                )
            ),
            End(data=FinalResult(output='The capital of France is Paris.')),
        ]
        '''
        print(agent_run.result.output)
        #> The capital of France is Paris.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    if workflow.in_workflow():
        if not self._temporal_overrides_active.get():
            raise UserError(
                '`agent.iter()` cannot be used inside a Temporal workflow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        if model is not None:
            raise UserError(
                'Model cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
            )
        if toolsets is not None:
            raise UserError(
                'Toolsets cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
            )

    async with super().iter(
        user_prompt=user_prompt,
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
        **_deprecated_kwargs,
    ) as run:
        yield run

````

#### override

```python
override(
    *,
    name: str | Unset = UNSET,
    deps: AgentDepsT | Unset = UNSET,
    model: Model | KnownModelName | str | Unset = UNSET,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | Unset
    ) = UNSET,
    tools: (
        Sequence[
            Tool[AgentDepsT]
            | ToolFuncEither[AgentDepsT, ...]
        ]
        | Unset
    ) = UNSET,
    instructions: Instructions[AgentDepsT] | Unset = UNSET
) -> Iterator[None]

```

Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

This is particularly useful when testing. You can find an example of this [here](../../testing/#overriding-model-via-pytest-fixtures).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | Unset` | The name to use instead of the name passed to the agent constructor and agent run. | `UNSET` | | `deps` | `AgentDepsT | Unset` | The dependencies to use instead of the dependencies passed to the agent run. | `UNSET` | | `model` | `Model | KnownModelName | str | Unset` | The model to use instead of the model passed to the agent run. | `UNSET` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | Unset` | The toolsets to use instead of the toolsets passed to the agent constructor and agent run. | `UNSET` | | `tools` | `Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | Unset` | The tools to use instead of the tools registered with the agent. | `UNSET` | | `instructions` | `Instructions[AgentDepsT] | Unset` | The instructions to use instead of the instructions registered with the agent. | `UNSET` |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`

```python
@contextmanager
def override(
    self,
    *,
    name: str | _utils.Unset = _utils.UNSET,
    deps: AgentDepsT | _utils.Unset = _utils.UNSET,
    model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
    instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
) -> Iterator[None]:
    """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

    This is particularly useful when testing.
    You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

    Args:
        name: The name to use instead of the name passed to the agent constructor and agent run.
        deps: The dependencies to use instead of the dependencies passed to the agent run.
        model: The model to use instead of the model passed to the agent run.
        toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
        tools: The tools to use instead of the tools registered with the agent.
        instructions: The instructions to use instead of the instructions registered with the agent.
    """
    if workflow.in_workflow():
        if _utils.is_set(model):
            raise UserError(
                'Model cannot be contextually overridden inside a Temporal workflow, it must be set at agent creation time.'
            )
        if _utils.is_set(toolsets):
            raise UserError(
                'Toolsets cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
            )
        if _utils.is_set(tools):
            raise UserError(
                'Tools cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
            )

    with super().override(
        name=name,
        deps=deps,
        model=model,
        toolsets=toolsets,
        tools=tools,
        instructions=instructions,
    ):
        yield

```

### LogfirePlugin

Bases: `SimplePlugin`

Temporal client plugin for Logfire.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_logfire.py`

```python
class LogfirePlugin(SimplePlugin):
    """Temporal client plugin for Logfire."""

    def __init__(self, setup_logfire: Callable[[], Logfire] = _default_setup_logfire, *, metrics: bool = True):
        try:
            import logfire  # noqa: F401 # pyright: ignore[reportUnusedImport]
            from opentelemetry.trace import get_tracer
            from temporalio.contrib.opentelemetry import TracingInterceptor
        except ImportError as _import_error:
            raise ImportError(
                'Please install the `logfire` package to use the Logfire plugin, '
                'you can use the `logfire` optional group â€” `pip install "pydantic-ai-slim[logfire]"`'
            ) from _import_error

        self.setup_logfire = setup_logfire
        self.metrics = metrics

        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='LogfirePlugin',
            client_interceptors=[TracingInterceptor(get_tracer('temporalio'))],
        )

    async def connect_service_client(
        self, config: ConnectConfig, next: Callable[[ConnectConfig], Awaitable[ServiceClient]]
    ) -> ServiceClient:
        logfire = self.setup_logfire()

        if self.metrics:
            logfire_config = logfire.config
            token = logfire_config.token
            if logfire_config.send_to_logfire and token is not None and logfire_config.metrics is not False:
                base_url = logfire_config.advanced.generate_base_url(token)
                metrics_url = base_url + '/v1/metrics'
                headers = {'Authorization': f'Bearer {token}'}

                config.runtime = Runtime(
                    telemetry=TelemetryConfig(metrics=OpenTelemetryConfig(url=metrics_url, headers=headers))
                )

        return await next(config)

```

### TemporalRunContext

Bases: `RunContext[AgentDepsT]`

The RunContext subclass to use to serialize and deserialize the run context for use inside a Temporal activity.

By default, only the `deps`, `run_id`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `retry`, `max_retries`, `run_step`, `usage`, and `partial_output` attributes will be available. To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to TemporalAgent.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_run_context.py`

```python
class TemporalRunContext(RunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] subclass to use to serialize and deserialize the run context for use inside a Temporal activity.

    By default, only the `deps`, `run_id`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `retry`, `max_retries`, `run_step`, `usage`, and `partial_output` attributes will be available.
    To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent].
    """

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        self.__dict__ = {**kwargs, 'deps': deps}
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in self.__dict__},
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError as e:  # pragma: no cover
            if name in RunContext.__dataclass_fields__:
                raise UserError(
                    f'{self.__class__.__name__!r} object has no attribute {name!r}. '
                    'To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it to `TemporalAgent`.'
                )
            else:
                raise e

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context to a `dict[str, Any]`."""
        return {
            'run_id': ctx.run_id,
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'tool_call_approved': ctx.tool_call_approved,
            'retry': ctx.retry,
            'max_retries': ctx.max_retries,
            'run_step': ctx.run_step,
            'partial_output': ctx.partial_output,
            'usage': ctx.usage,
        }

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> TemporalRunContext[Any]:
        """Deserialize the run context from a `dict[str, Any]`."""
        return cls(**ctx, deps=deps)

```

#### serialize_run_context

```python
serialize_run_context(
    ctx: RunContext[Any],
) -> dict[str, Any]

```

Serialize the run context to a `dict[str, Any]`.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_run_context.py`

```python
@classmethod
def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
    """Serialize the run context to a `dict[str, Any]`."""
    return {
        'run_id': ctx.run_id,
        'retries': ctx.retries,
        'tool_call_id': ctx.tool_call_id,
        'tool_name': ctx.tool_name,
        'tool_call_approved': ctx.tool_call_approved,
        'retry': ctx.retry,
        'max_retries': ctx.max_retries,
        'run_step': ctx.run_step,
        'partial_output': ctx.partial_output,
        'usage': ctx.usage,
    }

```

#### deserialize_run_context

```python
deserialize_run_context(
    ctx: dict[str, Any], deps: Any
) -> TemporalRunContext[Any]

```

Deserialize the run context from a `dict[str, Any]`.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_run_context.py`

```python
@classmethod
def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> TemporalRunContext[Any]:
    """Deserialize the run context from a `dict[str, Any]`."""
    return cls(**ctx, deps=deps)

```

### PydanticAIPlugin

Bases: `SimplePlugin`

Temporal client and worker plugin for Pydantic AI.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/__init__.py`

```python
class PydanticAIPlugin(SimplePlugin):
    """Temporal client and worker plugin for Pydantic AI."""

    def __init__(self):
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='PydanticAIPlugin',
            data_converter=_data_converter,
            workflow_runner=_workflow_runner,
            workflow_failure_exception_types=[UserError, PydanticUserError],
        )

```

### AgentPlugin

Bases: `SimplePlugin`

Temporal worker plugin for a specific Pydantic AI agent.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/__init__.py`

```python
class AgentPlugin(SimplePlugin):
    """Temporal worker plugin for a specific Pydantic AI agent."""

    def __init__(self, agent: TemporalAgent[Any, Any]):
        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='AgentPlugin',
            activities=agent.temporal_activities,
        )

```

### DBOSAgent

Bases: `WrapperAgent[AgentDepsT, OutputDataT]`, `DBOSConfiguredInstance`

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`

````python
@DBOS.dbos_class()
class DBOSAgent(WrapperAgent[AgentDepsT, OutputDataT], DBOSConfiguredInstance):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        mcp_step_config: StepConfig | None = None,
        model_step_config: StepConfig | None = None,
    ):
        """Wrap an agent to enable it with DBOS durable workflows, by automatically offloading model requests, tool calls, and MCP server communication to DBOS steps.

        After wrapping, the original agent can still be used as normal outside of the DBOS workflow.

        Args:
            wrapped: The agent to wrap.
            name: Optional unique agent name to use as the DBOS configured instance name. If not provided, the agent's `name` will be used.
            event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
            mcp_step_config: The base DBOS step config to use for MCP server steps. If no config is provided, use the default settings of DBOS.
            model_step_config: The DBOS step config to use for model request steps. If no config is provided, use the default settings of DBOS.
        """
        super().__init__(wrapped)

        self._name = name or wrapped.name
        self._event_stream_handler = event_stream_handler
        if self._name is None:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with DBOS. The name will be used to identify the agent's workflows and steps."
            )

        # Merge the config with the default DBOS config
        self._mcp_step_config = mcp_step_config or {}
        self._model_step_config = model_step_config or {}

        if not isinstance(wrapped.model, Model):
            raise UserError(
                'An agent needs to have a `model` in order to be used with DBOS, it cannot be set at agent run time.'
            )

        dbos_model = DBOSModel(
            wrapped.model,
            step_name_prefix=self._name,
            step_config=self._model_step_config,
            event_stream_handler=self.event_stream_handler,
        )
        self._model = dbos_model

        dbosagent_name = self._name

        def dbosify_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            # Replace MCPServer with DBOSMCPServer
            try:
                from pydantic_ai.mcp import MCPServer

                from ._mcp_server import DBOSMCPServer
            except ImportError:
                pass
            else:
                if isinstance(toolset, MCPServer):
                    return DBOSMCPServer(
                        wrapped=toolset,
                        step_name_prefix=dbosagent_name,
                        step_config=self._mcp_step_config,
                    )

            return toolset

        dbos_toolsets = [toolset.visit_and_replace(dbosify_toolset) for toolset in wrapped.toolsets]
        self._toolsets = dbos_toolsets
        DBOSConfiguredInstance.__init__(self, self._name)

        # Wrap the `run` method in a DBOS workflow
        @DBOS.workflow(name=f'{self._name}.run')
        async def wrapped_run_workflow(
            user_prompt: str | Sequence[_messages.UserContent] | None = None,
            *,
            output_type: OutputSpec[RunOutputDataT] | None = None,
            message_history: Sequence[_messages.ModelMessage] | None = None,
            deferred_tool_results: DeferredToolResults | None = None,
            model: models.Model | models.KnownModelName | str | None = None,
            instructions: Instructions[AgentDepsT] = None,
            deps: AgentDepsT,
            model_settings: ModelSettings | None = None,
            usage_limits: _usage.UsageLimits | None = None,
            usage: _usage.RunUsage | None = None,
            infer_name: bool = True,
            toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
            builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
            event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
            **_deprecated_kwargs: Never,
        ) -> AgentRunResult[Any]:
            with self._dbos_overrides():
                return await super(WrapperAgent, self).run(
                    user_prompt,
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
                    event_stream_handler=event_stream_handler,
                    **_deprecated_kwargs,
                )

        self.dbos_wrapped_run_workflow = wrapped_run_workflow

        # Wrap the `run_sync` method in a DBOS workflow
        @DBOS.workflow(name=f'{self._name}.run_sync')
        def wrapped_run_sync_workflow(
            user_prompt: str | Sequence[_messages.UserContent] | None = None,
            *,
            output_type: OutputSpec[RunOutputDataT] | None = None,
            message_history: Sequence[_messages.ModelMessage] | None = None,
            deferred_tool_results: DeferredToolResults | None = None,
            model: models.Model | models.KnownModelName | str | None = None,
            deps: AgentDepsT,
            model_settings: ModelSettings | None = None,
            instructions: Instructions[AgentDepsT] = None,
            usage_limits: _usage.UsageLimits | None = None,
            usage: _usage.RunUsage | None = None,
            infer_name: bool = True,
            toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
            builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
            event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
            **_deprecated_kwargs: Never,
        ) -> AgentRunResult[Any]:
            with self._dbos_overrides():
                return super(DBOSAgent, self).run_sync(
                    user_prompt,
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
                    event_stream_handler=event_stream_handler,
                    **_deprecated_kwargs,
                )

        self.dbos_wrapped_run_sync_workflow = wrapped_run_sync_workflow

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:  # pragma: no cover
        raise UserError(
            'The agent name cannot be changed after creation. If you need to change the name, create a new agent.'
        )

    @property
    def model(self) -> Model:
        return self._model

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        handler = self._event_stream_handler or super().event_stream_handler
        if handler is None:
            return None
        elif DBOS.workflow_id is not None and DBOS.step_id is None:
            # Special case if it's in a DBOS workflow but not a step, we need to iterate through all events and call the handler.
            return self._call_event_stream_handler_in_workflow
        else:
            return handler

    async def _call_event_stream_handler_in_workflow(
        self, ctx: RunContext[AgentDepsT], stream: AsyncIterable[_messages.AgentStreamEvent]
    ) -> None:
        handler = self._event_stream_handler or super().event_stream_handler
        assert handler is not None

        async def streamed_response(event: _messages.AgentStreamEvent):
            yield event

        async for event in stream:
            await handler(ctx, streamed_response(event))

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._dbos_overrides():
            return super().toolsets

    @contextmanager
    def _dbos_overrides(self) -> Iterator[None]:
        # Override with DBOSModel and DBOSMCPServer in the toolsets.
        with (
            super().override(model=self._model, toolsets=self._toolsets, tools=[]),
            self.sequential_tool_calls(),
        ):
            yield

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode.

        This method builds an internal agent graph (using system prompts, tools and result schemas) and then
        runs the graph to completion. The result of the run is returned.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            agent_run = await agent.run('What is the capital of France?')
            print(agent_run.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run.

        Returns:
            The result of the run.
        """
        return await self.dbos_wrapped_run_workflow(
            user_prompt,
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
            event_stream_handler=event_stream_handler,
            **_deprecated_kwargs,
        )

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Synchronously run the agent with a user prompt.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        result_sync = agent.run_sync('What is the capital of Italy?')
        print(result_sync.output)
        #> The capital of Italy is Rome.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run.

        Returns:
            The result of the run.
        """
        return self.dbos_wrapped_run_sync_workflow(
            user_prompt,
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
            event_stream_handler=event_stream_handler,
            **_deprecated_kwargs,
        )

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        instructions: Instructions[AgentDepsT] = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async mode, returning a streamed response.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.run_stream('What is the capital of the UK?') as response:
                print(await response.get_output())
                #> The capital of the UK is London.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

        Returns:
            The result of the run.
        """
        if DBOS.workflow_id is not None and DBOS.step_id is None:
            raise UserError(
                '`agent.run_stream()` cannot be used inside a DBOS workflow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        async with super().run_stream(
            user_prompt,
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
            event_stream_handler=event_stream_handler,
            **_deprecated_kwargs,
        ) as result:
            yield result

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[OutputDataT]]: ...

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]]: ...

    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
        """Run the agent with a user prompt in async mode and stream events from the run.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
        uses the `event_stream_handler` kwarg to get a stream of events from the run.

        Example:
        ```python
        from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

        agent = Agent('openai:gpt-4o')

        async def main():
            events: list[AgentStreamEvent | AgentRunResultEvent] = []
            async for event in agent.run_stream_events('What is the capital of France?'):
                events.append(event)
            print(events)
            '''
            [
                PartStartEvent(index=0, part=TextPart(content='The capital of ')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
                PartEndEvent(
                    index=0, part=TextPart(content='The capital of France is Paris. ')
                ),
                AgentRunResultEvent(
                    result=AgentRunResult(output='The capital of France is Paris. ')
                ),
            ]
            '''
        ```

        Arguments are the same as for [`self.run`][pydantic_ai.agent.AbstractAgent.run],
        except that `event_stream_handler` is now allowed.

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
            run result.
        """
        raise UserError(
            '`agent.run_stream_events()` cannot be used with DBOS. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        )

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
            async with agent.iter('What is the capital of France?') as agent_run:
                async for node in agent_run:
                    nodes.append(node)
            print(nodes)
            '''
            [
                UserPromptNode(
                    user_prompt='What is the capital of France?',
                    instructions_functions=[],
                    system_prompts=(),
                    system_prompt_functions=[],
                    system_prompt_dynamic_functions={},
                ),
                ModelRequestNode(
                    request=ModelRequest(
                        parts=[
                            UserPromptPart(
                                content='What is the capital of France?',
                                timestamp=datetime.datetime(...),
                            )
                        ],
                        run_id='...',
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
                        run_id='...',
                    )
                ),
                End(data=FinalResult(output='The capital of France is Paris.')),
            ]
            '''
            print(agent_run.result.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if model is not None and not isinstance(model, DBOSModel):
            raise UserError(
                'Non-DBOS model cannot be set at agent run time inside a DBOS workflow, it must be set at agent creation time.'
            )

        with self._dbos_overrides():
            async with super().iter(
                user_prompt=user_prompt,
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
                **_deprecated_kwargs,
            ) as run:
                yield run

    @contextmanager
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            instructions: The instructions to use instead of the instructions registered with the agent.
        """
        if _utils.is_set(model) and not isinstance(model, (DBOSModel)):
            raise UserError(
                'Non-DBOS model cannot be contextually overridden inside a DBOS workflow, it must be set at agent creation time.'
            )

        with super().override(
            name=name,
            deps=deps,
            model=model,
            toolsets=toolsets,
            tools=tools,
            instructions=instructions,
        ):
            yield

````

#### __init__

```python
__init__(
    wrapped: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    name: str | None = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    mcp_step_config: StepConfig | None = None,
    model_step_config: StepConfig | None = None
)

```

Wrap an agent to enable it with DBOS durable workflows, by automatically offloading model requests, tool calls, and MCP server communication to DBOS steps.

After wrapping, the original agent can still be used as normal outside of the DBOS workflow.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `wrapped` | `AbstractAgent[AgentDepsT, OutputDataT]` | The agent to wrap. | *required* | | `name` | `str | None` | Optional unique agent name to use as the DBOS configured instance name. If not provided, the agent's name will be used. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use instead of the one set on the wrapped agent. | `None` | | `mcp_step_config` | `StepConfig | None` | The base DBOS step config to use for MCP server steps. If no config is provided, use the default settings of DBOS. | `None` | | `model_step_config` | `StepConfig | None` | The DBOS step config to use for model request steps. If no config is provided, use the default settings of DBOS. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`

```python
def __init__(
    self,
    wrapped: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    name: str | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    mcp_step_config: StepConfig | None = None,
    model_step_config: StepConfig | None = None,
):
    """Wrap an agent to enable it with DBOS durable workflows, by automatically offloading model requests, tool calls, and MCP server communication to DBOS steps.

    After wrapping, the original agent can still be used as normal outside of the DBOS workflow.

    Args:
        wrapped: The agent to wrap.
        name: Optional unique agent name to use as the DBOS configured instance name. If not provided, the agent's `name` will be used.
        event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
        mcp_step_config: The base DBOS step config to use for MCP server steps. If no config is provided, use the default settings of DBOS.
        model_step_config: The DBOS step config to use for model request steps. If no config is provided, use the default settings of DBOS.
    """
    super().__init__(wrapped)

    self._name = name or wrapped.name
    self._event_stream_handler = event_stream_handler
    if self._name is None:
        raise UserError(
            "An agent needs to have a unique `name` in order to be used with DBOS. The name will be used to identify the agent's workflows and steps."
        )

    # Merge the config with the default DBOS config
    self._mcp_step_config = mcp_step_config or {}
    self._model_step_config = model_step_config or {}

    if not isinstance(wrapped.model, Model):
        raise UserError(
            'An agent needs to have a `model` in order to be used with DBOS, it cannot be set at agent run time.'
        )

    dbos_model = DBOSModel(
        wrapped.model,
        step_name_prefix=self._name,
        step_config=self._model_step_config,
        event_stream_handler=self.event_stream_handler,
    )
    self._model = dbos_model

    dbosagent_name = self._name

    def dbosify_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # Replace MCPServer with DBOSMCPServer
        try:
            from pydantic_ai.mcp import MCPServer

            from ._mcp_server import DBOSMCPServer
        except ImportError:
            pass
        else:
            if isinstance(toolset, MCPServer):
                return DBOSMCPServer(
                    wrapped=toolset,
                    step_name_prefix=dbosagent_name,
                    step_config=self._mcp_step_config,
                )

        return toolset

    dbos_toolsets = [toolset.visit_and_replace(dbosify_toolset) for toolset in wrapped.toolsets]
    self._toolsets = dbos_toolsets
    DBOSConfiguredInstance.__init__(self, self._name)

    # Wrap the `run` method in a DBOS workflow
    @DBOS.workflow(name=f'{self._name}.run')
    async def wrapped_run_workflow(
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        with self._dbos_overrides():
            return await super(WrapperAgent, self).run(
                user_prompt,
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
                event_stream_handler=event_stream_handler,
                **_deprecated_kwargs,
            )

    self.dbos_wrapped_run_workflow = wrapped_run_workflow

    # Wrap the `run_sync` method in a DBOS workflow
    @DBOS.workflow(name=f'{self._name}.run_sync')
    def wrapped_run_sync_workflow(
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT,
        model_settings: ModelSettings | None = None,
        instructions: Instructions[AgentDepsT] = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        with self._dbos_overrides():
            return super(DBOSAgent, self).run_sync(
                user_prompt,
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
                event_stream_handler=event_stream_handler,
                **_deprecated_kwargs,
            )

    self.dbos_wrapped_run_sync_workflow = wrapped_run_sync_workflow

```

#### run

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[OutputDataT]

```

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[RunOutputDataT]

```

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AgentRunResult[Any]

```

Run the agent with a user prompt in async mode.

This method builds an internal agent graph (using system prompts, tools and result schemas) and then runs the graph to completion. The result of the run is returned.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    agent_run = await agent.run('What is the capital of France?')
    print(agent_run.output)
    #> The capital of France is Paris.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AgentRunResult[Any]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`

````python
async def run(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Never,
) -> AgentRunResult[Any]:
    """Run the agent with a user prompt in async mode.

    This method builds an internal agent graph (using system prompts, tools and result schemas) and then
    runs the graph to completion. The result of the run is returned.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        agent_run = await agent.run('What is the capital of France?')
        print(agent_run.output)
        #> The capital of France is Paris.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.
        event_stream_handler: Optional event stream handler to use for this run.

    Returns:
        The result of the run.
    """
    return await self.dbos_wrapped_run_workflow(
        user_prompt,
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
        event_stream_handler=event_stream_handler,
        **_deprecated_kwargs,
    )

````

#### run_sync

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[OutputDataT]

```

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[RunOutputDataT]

```

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AgentRunResult[Any]

```

Synchronously run the agent with a user prompt.

This is a convenience method that wraps self.run with `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)
#> The capital of Italy is Rome.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AgentRunResult[Any]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`

````python
def run_sync(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Never,
) -> AgentRunResult[Any]:
    """Synchronously run the agent with a user prompt.

    This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
    You therefore can't use this method inside async code or if there's an active event loop.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    result_sync = agent.run_sync('What is the capital of Italy?')
    print(result_sync.output)
    #> The capital of Italy is Rome.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.
        event_stream_handler: Optional event stream handler to use for this run.

    Returns:
        The result of the run.
    """
    return self.dbos_wrapped_run_sync_workflow(
        user_prompt,
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
        event_stream_handler=event_stream_handler,
        **_deprecated_kwargs,
    )

````

#### run_stream

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AbstractAsyncContextManager[
    StreamedRunResult[AgentDepsT, OutputDataT]
]

```

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    instructions: Instructions[AgentDepsT] = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AbstractAsyncContextManager[
    StreamedRunResult[AgentDepsT, RunOutputDataT]
]

```

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]

```

Run the agent with a user prompt in async mode, returning a streamed response.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.run_stream('What is the capital of the UK?') as response:
        print(await response.get_output())
        #> The capital of the UK is London.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[StreamedRunResult[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`

````python
@asynccontextmanager
async def run_stream(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Never,
) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
    """Run the agent with a user prompt in async mode, returning a streamed response.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        async with agent.run_stream('What is the capital of the UK?') as response:
            print(await response.get_output())
            #> The capital of the UK is London.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.
        event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

    Returns:
        The result of the run.
    """
    if DBOS.workflow_id is not None and DBOS.step_id is None:
        raise UserError(
            '`agent.run_stream()` cannot be used inside a DBOS workflow. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        )

    async with super().run_stream(
        user_prompt,
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
        event_stream_handler=event_stream_handler,
        **_deprecated_kwargs,
    ) as result:
        yield result

````

#### run_stream_events

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[OutputDataT]
]

```

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]
]

```

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[Any]
]

```

Run the agent with a user prompt in async mode and stream events from the run.

This is a convenience method that wraps self.run and uses the `event_stream_handler` kwarg to get a stream of events from the run.

Example:

```python
from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

agent = Agent('openai:gpt-4o')

async def main():
    events: list[AgentStreamEvent | AgentRunResultEvent] = []
    async for event in agent.run_stream_events('What is the capital of France?'):
        events.append(event)
    print(events)
    '''
    [
        PartStartEvent(index=0, part=TextPart(content='The capital of ')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
        PartEndEvent(
            index=0, part=TextPart(content='The capital of France is Paris. ')
        ),
        AgentRunResultEvent(
            result=AgentRunResult(output='The capital of France is Paris. ')
        ),
    ]
    '''

```

Arguments are the same as for self.run, except that `event_stream_handler` is now allowed.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]` | An async iterable of stream events AgentStreamEvent and finally a AgentRunResultEvent with the final | | `AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]` | run result. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`

````python
def run_stream_events(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
    """Run the agent with a user prompt in async mode and stream events from the run.

    This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
    uses the `event_stream_handler` kwarg to get a stream of events from the run.

    Example:
    ```python
    from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

    agent = Agent('openai:gpt-4o')

    async def main():
        events: list[AgentStreamEvent | AgentRunResultEvent] = []
        async for event in agent.run_stream_events('What is the capital of France?'):
            events.append(event)
        print(events)
        '''
        [
            PartStartEvent(index=0, part=TextPart(content='The capital of ')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
            PartEndEvent(
                index=0, part=TextPart(content='The capital of France is Paris. ')
            ),
            AgentRunResultEvent(
                result=AgentRunResult(output='The capital of France is Paris. ')
            ),
        ]
        '''
    ```

    Arguments are the same as for [`self.run`][pydantic_ai.agent.AbstractAgent.run],
    except that `event_stream_handler` is now allowed.

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
        run result.
    """
    raise UserError(
        '`agent.run_stream_events()` cannot be used with DBOS. '
        'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
    )

````

#### iter

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    **_deprecated_kwargs: Never
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, OutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    **_deprecated_kwargs: Never
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, RunOutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    **_deprecated_kwargs: Never
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]

```

A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the stream of events coming from the execution of tools.

The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics, and the final result of the run once it has completed.

For more details, see the documentation of `AgentRun`.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    nodes = []
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            nodes.append(node)
    print(nodes)
    '''
    [
        UserPromptNode(
            user_prompt='What is the capital of France?',
            instructions_functions=[],
            system_prompts=(),
            system_prompt_functions=[],
            system_prompt_dynamic_functions={},
        ),
        ModelRequestNode(
            request=ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=datetime.datetime(...),
                    )
                ],
                run_id='...',
            )
        ),
        CallToolsNode(
            model_response=ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(input_tokens=56, output_tokens=7),
                model_name='gpt-4o',
                timestamp=datetime.datetime(...),
                run_id='...',
            )
        ),
        End(data=FinalResult(output='The capital of France is Paris.')),
    ]
    '''
    print(agent_run.result.output)
    #> The capital of France is Paris.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentRun[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`

````python
@asynccontextmanager
async def iter(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    **_deprecated_kwargs: Never,
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
    """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

    This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
    `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
    executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
    stream of events coming from the execution of tools.

    The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
    and the final result of the run once it has completed.

    For more details, see the documentation of `AgentRun`.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
        async with agent.iter('What is the capital of France?') as agent_run:
            async for node in agent_run:
                nodes.append(node)
        print(nodes)
        '''
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                instructions_functions=[],
                system_prompts=(),
                system_prompt_functions=[],
                system_prompt_dynamic_functions={},
            ),
            ModelRequestNode(
                request=ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=datetime.datetime(...),
                        )
                    ],
                    run_id='...',
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='The capital of France is Paris.')],
                    usage=RequestUsage(input_tokens=56, output_tokens=7),
                    model_name='gpt-4o',
                    timestamp=datetime.datetime(...),
                    run_id='...',
                )
            ),
            End(data=FinalResult(output='The capital of France is Paris.')),
        ]
        '''
        print(agent_run.result.output)
        #> The capital of France is Paris.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    if model is not None and not isinstance(model, DBOSModel):
        raise UserError(
            'Non-DBOS model cannot be set at agent run time inside a DBOS workflow, it must be set at agent creation time.'
        )

    with self._dbos_overrides():
        async with super().iter(
            user_prompt=user_prompt,
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
            **_deprecated_kwargs,
        ) as run:
            yield run

````

#### override

```python
override(
    *,
    name: str | Unset = UNSET,
    deps: AgentDepsT | Unset = UNSET,
    model: Model | KnownModelName | str | Unset = UNSET,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | Unset
    ) = UNSET,
    tools: (
        Sequence[
            Tool[AgentDepsT]
            | ToolFuncEither[AgentDepsT, ...]
        ]
        | Unset
    ) = UNSET,
    instructions: Instructions[AgentDepsT] | Unset = UNSET
) -> Iterator[None]

```

Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

This is particularly useful when testing. You can find an example of this [here](../../testing/#overriding-model-via-pytest-fixtures).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | Unset` | The name to use instead of the name passed to the agent constructor and agent run. | `UNSET` | | `deps` | `AgentDepsT | Unset` | The dependencies to use instead of the dependencies passed to the agent run. | `UNSET` | | `model` | `Model | KnownModelName | str | Unset` | The model to use instead of the model passed to the agent run. | `UNSET` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | Unset` | The toolsets to use instead of the toolsets passed to the agent constructor and agent run. | `UNSET` | | `tools` | `Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | Unset` | The tools to use instead of the tools registered with the agent. | `UNSET` | | `instructions` | `Instructions[AgentDepsT] | Unset` | The instructions to use instead of the instructions registered with the agent. | `UNSET` |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`

```python
@contextmanager
def override(
    self,
    *,
    name: str | _utils.Unset = _utils.UNSET,
    deps: AgentDepsT | _utils.Unset = _utils.UNSET,
    model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
    instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
) -> Iterator[None]:
    """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

    This is particularly useful when testing.
    You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

    Args:
        name: The name to use instead of the name passed to the agent constructor and agent run.
        deps: The dependencies to use instead of the dependencies passed to the agent run.
        model: The model to use instead of the model passed to the agent run.
        toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
        tools: The tools to use instead of the tools registered with the agent.
        instructions: The instructions to use instead of the instructions registered with the agent.
    """
    if _utils.is_set(model) and not isinstance(model, (DBOSModel)):
        raise UserError(
            'Non-DBOS model cannot be contextually overridden inside a DBOS workflow, it must be set at agent creation time.'
        )

    with super().override(
        name=name,
        deps=deps,
        model=model,
        toolsets=toolsets,
        tools=tools,
        instructions=instructions,
    ):
        yield

```

### DBOSMCPServer

Bases: `WrapperToolset[AgentDepsT]`, `ABC`

A wrapper for MCPServer that integrates with DBOS, turning call_tool and get_tools to DBOS steps.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_mcp_server.py`

```python
class DBOSMCPServer(WrapperToolset[AgentDepsT], ABC):
    """A wrapper for MCPServer that integrates with DBOS, turning call_tool and get_tools to DBOS steps."""

    def __init__(
        self,
        wrapped: MCPServer,
        *,
        step_name_prefix: str,
        step_config: StepConfig,
    ):
        super().__init__(wrapped)
        self._step_config = step_config or {}
        self._step_name_prefix = step_name_prefix
        id_suffix = f'__{wrapped.id}' if wrapped.id else ''
        self._name = f'{step_name_prefix}__mcp_server{id_suffix}'

        # Wrap get_tools in a DBOS step.
        @DBOS.step(
            name=f'{self._name}.get_tools',
            **self._step_config,
        )
        async def wrapped_get_tools_step(
            ctx: RunContext[AgentDepsT],
        ) -> dict[str, ToolsetTool[AgentDepsT]]:
            return await super(DBOSMCPServer, self).get_tools(ctx)

        self._dbos_wrapped_get_tools_step = wrapped_get_tools_step

        # Wrap call_tool in a DBOS step.
        @DBOS.step(
            name=f'{self._name}.call_tool',
            **self._step_config,
        )
        async def wrapped_call_tool_step(
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> ToolResult:
            return await super(DBOSMCPServer, self).call_tool(name, tool_args, ctx, tool)

        self._dbos_wrapped_call_tool_step = wrapped_call_tool_step

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        # The wrapped MCPServer enters itself around listing and calling tools
        # so we don't need to enter it here (nor could we because we're not inside a DBOS step).
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # DBOS-ified toolsets cannot be swapped out after the fact.
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return await self._dbos_wrapped_get_tools_step(ctx)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        return await self._dbos_wrapped_call_tool_step(name, tool_args, ctx, tool)

```

### DBOSModel

Bases: `WrapperModel`

A wrapper for Model that integrates with DBOS, turning request and request_stream to DBOS steps.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_model.py`

```python
class DBOSModel(WrapperModel):
    """A wrapper for Model that integrates with DBOS, turning request and request_stream to DBOS steps."""

    def __init__(
        self,
        model: Model,
        *,
        step_name_prefix: str,
        step_config: StepConfig,
        event_stream_handler: EventStreamHandler[Any] | None = None,
    ):
        super().__init__(model)
        self.step_config = step_config
        self.event_stream_handler = event_stream_handler
        self._step_name_prefix = step_name_prefix

        # Wrap the request in a DBOS step.
        @DBOS.step(
            name=f'{self._step_name_prefix}__model.request',
            **self.step_config,
        )
        async def wrapped_request_step(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return await super(DBOSModel, self).request(messages, model_settings, model_request_parameters)

        self._dbos_wrapped_request_step = wrapped_request_step

        # Wrap the request_stream in a DBOS step.
        @DBOS.step(
            name=f'{self._step_name_prefix}__model.request_stream',
            **self.step_config,
        )
        async def wrapped_request_stream_step(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any] | None = None,
        ) -> ModelResponse:
            async with super(DBOSModel, self).request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                if self.event_stream_handler is not None:
                    assert run_context is not None, (
                        'A DBOS model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
                    )
                    await self.event_stream_handler(run_context, streamed_response)

                async for _ in streamed_response:
                    pass
            return streamed_response.get()

        self._dbos_wrapped_request_stream_step = wrapped_request_stream_step

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await self._dbos_wrapped_request_step(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        # If not in a workflow (could be in a step), just call the wrapped request_stream method.
        if DBOS.workflow_id is None or DBOS.step_id is not None:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                yield streamed_response
                return

        response = await self._dbos_wrapped_request_stream_step(
            messages, model_settings, model_request_parameters, run_context
        )
        yield DBOSStreamedResponse(model_request_parameters, response)

```

### StepConfig

Bases: `TypedDict`

Configuration for a step in the DBOS workflow.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_utils.py`

```python
class StepConfig(TypedDict, total=False):
    """Configuration for a step in the DBOS workflow."""

    retries_allowed: bool
    interval_seconds: float
    max_attempts: int
    backoff_rate: float

```

### PrefectAgent

Bases: `WrapperAgent[AgentDepsT, OutputDataT]`

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`

````python
class PrefectAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        mcp_task_config: TaskConfig | None = None,
        model_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
        tool_task_config_by_name: dict[str, TaskConfig | None] | None = None,
        event_stream_handler_task_config: TaskConfig | None = None,
        prefectify_toolset_func: Callable[
            [AbstractToolset[AgentDepsT], TaskConfig, TaskConfig, dict[str, TaskConfig | None]],
            AbstractToolset[AgentDepsT],
        ] = prefectify_toolset,
    ):
        """Wrap an agent to enable it with Prefect durable flows, by automatically offloading model requests, tool calls, and MCP server communication to Prefect tasks.

        After wrapping, the original agent can still be used as normal outside of the Prefect flow.

        Args:
            wrapped: The agent to wrap.
            name: Optional unique agent name to use as the Prefect flow name prefix. If not provided, the agent's `name` will be used.
            event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
            mcp_task_config: The base Prefect task config to use for MCP server tasks. If no config is provided, use the default settings of Prefect.
            model_task_config: The Prefect task config to use for model request tasks. If no config is provided, use the default settings of Prefect.
            tool_task_config: The default Prefect task config to use for tool calls. If no config is provided, use the default settings of Prefect.
            tool_task_config_by_name: Per-tool task configuration. Keys are tool names, values are TaskConfig or None (None disables task wrapping for that tool).
            event_stream_handler_task_config: The Prefect task config to use for the event stream handler task. If no config is provided, use the default settings of Prefect.
            prefectify_toolset_func: Optional function to use to prepare toolsets for Prefect by wrapping them in a `PrefectWrapperToolset` that moves methods that require IO to Prefect tasks.
                If not provided, only `FunctionToolset` and `MCPServer` will be prepared for Prefect.
                The function takes the toolset, the task config, the tool-specific task config, and the tool-specific task config by name.
        """
        super().__init__(wrapped)

        self._name = name or wrapped.name
        self._event_stream_handler = event_stream_handler
        if self._name is None:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with Prefect. The name will be used to identify the agent's flows and tasks."
            )

        # Merge the config with the default Prefect config
        self._mcp_task_config = default_task_config | (mcp_task_config or {})
        self._model_task_config = default_task_config | (model_task_config or {})
        self._tool_task_config = default_task_config | (tool_task_config or {})
        self._tool_task_config_by_name = tool_task_config_by_name or {}
        self._event_stream_handler_task_config = default_task_config | (event_stream_handler_task_config or {})

        if not isinstance(wrapped.model, Model):
            raise UserError(
                'An agent needs to have a `model` in order to be used with Prefect, it cannot be set at agent run time.'
            )

        prefect_model = PrefectModel(
            wrapped.model,
            task_config=self._model_task_config,
            event_stream_handler=self.event_stream_handler,
        )
        self._model = prefect_model

        def _prefectify_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            """Convert a toolset to its Prefect equivalent."""
            return prefectify_toolset_func(
                toolset,
                self._mcp_task_config,
                self._tool_task_config,
                self._tool_task_config_by_name,
            )

        prefect_toolsets = [toolset.visit_and_replace(_prefectify_toolset) for toolset in wrapped.toolsets]
        self._toolsets = prefect_toolsets

        # Context variable to track when we're inside this agent's Prefect flow
        self._in_prefect_agent_flow: ContextVar[bool] = ContextVar(
            f'_in_prefect_agent_flow_{self._name}', default=False
        )

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:  # pragma: no cover
        raise UserError(
            'The agent name cannot be changed after creation. If you need to change the name, create a new agent.'
        )

    @property
    def model(self) -> Model:
        return self._model

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        handler = self._event_stream_handler or super().event_stream_handler
        if handler is None:
            return None
        elif FlowRunContext.get() is not None:
            # Special case if it's in a Prefect flow, we need to iterate through all events and call the handler.
            return self._call_event_stream_handler_in_flow
        else:
            return handler

    async def _call_event_stream_handler_in_flow(
        self, ctx: RunContext[AgentDepsT], stream: AsyncIterable[_messages.AgentStreamEvent]
    ) -> None:
        handler = self._event_stream_handler or super().event_stream_handler
        assert handler is not None

        # Create a task to handle each event
        @task(name='Handle Stream Event', **self._event_stream_handler_task_config)
        async def event_stream_handler_task(event: _messages.AgentStreamEvent) -> None:
            async def streamed_response():
                yield event

            await handler(ctx, streamed_response())

        async for event in stream:
            await event_stream_handler_task(event)

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._prefect_overrides():
            return super().toolsets

    @contextmanager
    def _prefect_overrides(self) -> Iterator[None]:
        # Override with PrefectModel and PrefectMCPServer in the toolsets.
        with super().override(model=self._model, toolsets=self._toolsets, tools=[]):
            yield

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode.

        This method builds an internal agent graph (using system prompts, tools and result schemas) and then
        runs the graph to completion. The result of the run is returned.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            agent_run = await agent.run('What is the capital of France?')
            print(agent_run.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            event_stream_handler: Optional event stream handler to use for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """

        @flow(name=f'{self._name} Run')
        async def wrapped_run_flow() -> AgentRunResult[Any]:
            # Mark that we're inside a PrefectAgent flow
            token = self._in_prefect_agent_flow.set(True)
            try:
                with self._prefect_overrides():
                    result = await super(WrapperAgent, self).run(
                        user_prompt,
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
                        event_stream_handler=event_stream_handler,
                    )
                    return result
            finally:
                self._in_prefect_agent_flow.reset(token)

        return await wrapped_run_flow()

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Synchronously run the agent with a user prompt.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        result_sync = agent.run_sync('What is the capital of Italy?')
        print(result_sync.output)
        #> The capital of Italy is Rome.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            event_stream_handler: Optional event stream handler to use for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """

        @flow(name=f'{self._name} Sync Run')
        def wrapped_run_sync_flow() -> AgentRunResult[Any]:
            # Mark that we're inside a PrefectAgent flow
            token = self._in_prefect_agent_flow.set(True)
            try:
                with self._prefect_overrides():
                    # Using `run_coro_as_sync` from Prefect with async `run` to avoid event loop conflicts.
                    result = run_coro_as_sync(
                        super(PrefectAgent, self).run(
                            user_prompt,
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
                            event_stream_handler=event_stream_handler,
                        )
                    )
                    return result
            finally:
                self._in_prefect_agent_flow.reset(token)

        return wrapped_run_sync_flow()

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async mode, returning a streamed response.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.run_stream('What is the capital of the UK?') as response:
                print(await response.get_output())
                #> The capital of the UK is London.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

        Returns:
            The result of the run.
        """
        if FlowRunContext.get() is not None:
            raise UserError(
                '`agent.run_stream()` cannot be used inside a Prefect flow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        async with super().run_stream(
            user_prompt,
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
            event_stream_handler=event_stream_handler,
            builtin_tools=builtin_tools,
            **_deprecated_kwargs,
        ) as result:
            yield result

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[OutputDataT]]: ...

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]]: ...

    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
        """Run the agent with a user prompt in async mode and stream events from the run.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
        uses the `event_stream_handler` kwarg to get a stream of events from the run.

        Example:
        ```python
        from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

        agent = Agent('openai:gpt-4o')

        async def main():
            events: list[AgentStreamEvent | AgentRunResultEvent] = []
            async for event in agent.run_stream_events('What is the capital of France?'):
                events.append(event)
            print(events)
            '''
            [
                PartStartEvent(index=0, part=TextPart(content='The capital of ')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
                PartEndEvent(
                    index=0, part=TextPart(content='The capital of France is Paris. ')
                ),
                AgentRunResultEvent(
                    result=AgentRunResult(output='The capital of France is Paris. ')
                ),
            ]
            '''
        ```

        Arguments are the same as for [`self.run`][pydantic_ai.agent.AbstractAgent.run],
        except that `event_stream_handler` is now allowed.

        Args:
            user_prompt: User input to start/continue the conversation.
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
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
            run result.
        """
        if FlowRunContext.get() is not None:
            raise UserError(
                '`agent.run_stream_events()` cannot be used inside a Prefect flow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        return super().run_stream_events(
            user_prompt,
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
        )

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
            async with agent.iter('What is the capital of France?') as agent_run:
                async for node in agent_run:
                    nodes.append(node)
            print(nodes)
            '''
            [
                UserPromptNode(
                    user_prompt='What is the capital of France?',
                    instructions_functions=[],
                    system_prompts=(),
                    system_prompt_functions=[],
                    system_prompt_dynamic_functions={},
                ),
                ModelRequestNode(
                    request=ModelRequest(
                        parts=[
                            UserPromptPart(
                                content='What is the capital of France?',
                                timestamp=datetime.datetime(...),
                            )
                        ],
                        run_id='...',
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
                        run_id='...',
                    )
                ),
                End(data=FinalResult(output='The capital of France is Paris.')),
            ]
            '''
            print(agent_run.result.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            instructions: Optional additional instructions to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if model is not None and not isinstance(model, PrefectModel):
            raise UserError(
                'Non-Prefect model cannot be set at agent run time inside a Prefect flow, it must be set at agent creation time.'
            )

        with self._prefect_overrides():
            async with super().iter(
                user_prompt=user_prompt,
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
            ) as run:
                yield run

    @contextmanager
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies, model, toolsets, tools, or instructions.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            instructions: The instructions to use instead of the instructions registered with the agent.
        """
        if _utils.is_set(model) and not isinstance(model, PrefectModel):
            raise UserError(
                'Non-Prefect model cannot be contextually overridden inside a Prefect flow, it must be set at agent creation time.'
            )

        with super().override(
            name=name, deps=deps, model=model, toolsets=toolsets, tools=tools, instructions=instructions
        ):
            yield

````

#### __init__

```python
__init__(
    wrapped: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    name: str | None = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    mcp_task_config: TaskConfig | None = None,
    model_task_config: TaskConfig | None = None,
    tool_task_config: TaskConfig | None = None,
    tool_task_config_by_name: (
        dict[str, TaskConfig | None] | None
    ) = None,
    event_stream_handler_task_config: (
        TaskConfig | None
    ) = None,
    prefectify_toolset_func: Callable[
        [
            AbstractToolset[AgentDepsT],
            TaskConfig,
            TaskConfig,
            dict[str, TaskConfig | None],
        ],
        AbstractToolset[AgentDepsT],
    ] = prefectify_toolset
)

```

Wrap an agent to enable it with Prefect durable flows, by automatically offloading model requests, tool calls, and MCP server communication to Prefect tasks.

After wrapping, the original agent can still be used as normal outside of the Prefect flow.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `wrapped` | `AbstractAgent[AgentDepsT, OutputDataT]` | The agent to wrap. | *required* | | `name` | `str | None` | Optional unique agent name to use as the Prefect flow name prefix. If not provided, the agent's name will be used. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use instead of the one set on the wrapped agent. | `None` | | `mcp_task_config` | `TaskConfig | None` | The base Prefect task config to use for MCP server tasks. If no config is provided, use the default settings of Prefect. | `None` | | `model_task_config` | `TaskConfig | None` | The Prefect task config to use for model request tasks. If no config is provided, use the default settings of Prefect. | `None` | | `tool_task_config` | `TaskConfig | None` | The default Prefect task config to use for tool calls. If no config is provided, use the default settings of Prefect. | `None` | | `tool_task_config_by_name` | `dict[str, TaskConfig | None] | None` | Per-tool task configuration. Keys are tool names, values are TaskConfig or None (None disables task wrapping for that tool). | `None` | | `event_stream_handler_task_config` | `TaskConfig | None` | The Prefect task config to use for the event stream handler task. If no config is provided, use the default settings of Prefect. | `None` | | `prefectify_toolset_func` | `Callable[[AbstractToolset[AgentDepsT], TaskConfig, TaskConfig, dict[str, TaskConfig | None]], AbstractToolset[AgentDepsT]]` | Optional function to use to prepare toolsets for Prefect by wrapping them in a PrefectWrapperToolset that moves methods that require IO to Prefect tasks. If not provided, only FunctionToolset and MCPServer will be prepared for Prefect. The function takes the toolset, the task config, the tool-specific task config, and the tool-specific task config by name. | `prefectify_toolset` |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`

```python
def __init__(
    self,
    wrapped: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    name: str | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    mcp_task_config: TaskConfig | None = None,
    model_task_config: TaskConfig | None = None,
    tool_task_config: TaskConfig | None = None,
    tool_task_config_by_name: dict[str, TaskConfig | None] | None = None,
    event_stream_handler_task_config: TaskConfig | None = None,
    prefectify_toolset_func: Callable[
        [AbstractToolset[AgentDepsT], TaskConfig, TaskConfig, dict[str, TaskConfig | None]],
        AbstractToolset[AgentDepsT],
    ] = prefectify_toolset,
):
    """Wrap an agent to enable it with Prefect durable flows, by automatically offloading model requests, tool calls, and MCP server communication to Prefect tasks.

    After wrapping, the original agent can still be used as normal outside of the Prefect flow.

    Args:
        wrapped: The agent to wrap.
        name: Optional unique agent name to use as the Prefect flow name prefix. If not provided, the agent's `name` will be used.
        event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
        mcp_task_config: The base Prefect task config to use for MCP server tasks. If no config is provided, use the default settings of Prefect.
        model_task_config: The Prefect task config to use for model request tasks. If no config is provided, use the default settings of Prefect.
        tool_task_config: The default Prefect task config to use for tool calls. If no config is provided, use the default settings of Prefect.
        tool_task_config_by_name: Per-tool task configuration. Keys are tool names, values are TaskConfig or None (None disables task wrapping for that tool).
        event_stream_handler_task_config: The Prefect task config to use for the event stream handler task. If no config is provided, use the default settings of Prefect.
        prefectify_toolset_func: Optional function to use to prepare toolsets for Prefect by wrapping them in a `PrefectWrapperToolset` that moves methods that require IO to Prefect tasks.
            If not provided, only `FunctionToolset` and `MCPServer` will be prepared for Prefect.
            The function takes the toolset, the task config, the tool-specific task config, and the tool-specific task config by name.
    """
    super().__init__(wrapped)

    self._name = name or wrapped.name
    self._event_stream_handler = event_stream_handler
    if self._name is None:
        raise UserError(
            "An agent needs to have a unique `name` in order to be used with Prefect. The name will be used to identify the agent's flows and tasks."
        )

    # Merge the config with the default Prefect config
    self._mcp_task_config = default_task_config | (mcp_task_config or {})
    self._model_task_config = default_task_config | (model_task_config or {})
    self._tool_task_config = default_task_config | (tool_task_config or {})
    self._tool_task_config_by_name = tool_task_config_by_name or {}
    self._event_stream_handler_task_config = default_task_config | (event_stream_handler_task_config or {})

    if not isinstance(wrapped.model, Model):
        raise UserError(
            'An agent needs to have a `model` in order to be used with Prefect, it cannot be set at agent run time.'
        )

    prefect_model = PrefectModel(
        wrapped.model,
        task_config=self._model_task_config,
        event_stream_handler=self.event_stream_handler,
    )
    self._model = prefect_model

    def _prefectify_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Convert a toolset to its Prefect equivalent."""
        return prefectify_toolset_func(
            toolset,
            self._mcp_task_config,
            self._tool_task_config,
            self._tool_task_config_by_name,
        )

    prefect_toolsets = [toolset.visit_and_replace(_prefectify_toolset) for toolset in wrapped.toolsets]
    self._toolsets = prefect_toolsets

    # Context variable to track when we're inside this agent's Prefect flow
    self._in_prefect_agent_flow: ContextVar[bool] = ContextVar(
        f'_in_prefect_agent_flow_{self._name}', default=False
    )

```

#### run

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[OutputDataT]

```

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[RunOutputDataT]

```

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AgentRunResult[Any]

```

Run the agent with a user prompt in async mode.

This method builds an internal agent graph (using system prompts, tools and result schemas) and then runs the graph to completion. The result of the run is returned.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    agent_run = await agent.run('What is the capital of France?')
    print(agent_run.output)
    #> The capital of France is Paris.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AgentRunResult[Any]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`

````python
async def run(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Never,
) -> AgentRunResult[Any]:
    """Run the agent with a user prompt in async mode.

    This method builds an internal agent graph (using system prompts, tools and result schemas) and then
    runs the graph to completion. The result of the run is returned.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        agent_run = await agent.run('What is the capital of France?')
        print(agent_run.output)
        #> The capital of France is Paris.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        event_stream_handler: Optional event stream handler to use for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """

    @flow(name=f'{self._name} Run')
    async def wrapped_run_flow() -> AgentRunResult[Any]:
        # Mark that we're inside a PrefectAgent flow
        token = self._in_prefect_agent_flow.set(True)
        try:
            with self._prefect_overrides():
                result = await super(WrapperAgent, self).run(
                    user_prompt,
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
                    event_stream_handler=event_stream_handler,
                )
                return result
        finally:
            self._in_prefect_agent_flow.reset(token)

    return await wrapped_run_flow()

````

#### run_sync

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[OutputDataT]

```

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[RunOutputDataT]

```

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AgentRunResult[Any]

```

Synchronously run the agent with a user prompt.

This is a convenience method that wraps self.run with `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)
#> The capital of Italy is Rome.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AgentRunResult[Any]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`

````python
def run_sync(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Never,
) -> AgentRunResult[Any]:
    """Synchronously run the agent with a user prompt.

    This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
    You therefore can't use this method inside async code or if there's an active event loop.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    result_sync = agent.run_sync('What is the capital of Italy?')
    print(result_sync.output)
    #> The capital of Italy is Rome.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        event_stream_handler: Optional event stream handler to use for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """

    @flow(name=f'{self._name} Sync Run')
    def wrapped_run_sync_flow() -> AgentRunResult[Any]:
        # Mark that we're inside a PrefectAgent flow
        token = self._in_prefect_agent_flow.set(True)
        try:
            with self._prefect_overrides():
                # Using `run_coro_as_sync` from Prefect with async `run` to avoid event loop conflicts.
                result = run_coro_as_sync(
                    super(PrefectAgent, self).run(
                        user_prompt,
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
                        event_stream_handler=event_stream_handler,
                    )
                )
                return result
        finally:
            self._in_prefect_agent_flow.reset(token)

    return wrapped_run_sync_flow()

````

#### run_stream

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AbstractAsyncContextManager[
    StreamedRunResult[AgentDepsT, OutputDataT]
]

```

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AbstractAsyncContextManager[
    StreamedRunResult[AgentDepsT, RunOutputDataT]
]

```

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Never
) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]

```

Run the agent with a user prompt in async mode, returning a streamed response.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.run_stream('What is the capital of the UK?') as response:
        print(await response.get_output())
        #> The capital of the UK is London.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[StreamedRunResult[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`

````python
@asynccontextmanager
async def run_stream(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Never,
) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
    """Run the agent with a user prompt in async mode, returning a streamed response.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        async with agent.run_stream('What is the capital of the UK?') as response:
            print(await response.get_output())
            #> The capital of the UK is London.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.
        event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

    Returns:
        The result of the run.
    """
    if FlowRunContext.get() is not None:
        raise UserError(
            '`agent.run_stream()` cannot be used inside a Prefect flow. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        )

    async with super().run_stream(
        user_prompt,
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
        event_stream_handler=event_stream_handler,
        builtin_tools=builtin_tools,
        **_deprecated_kwargs,
    ) as result:
        yield result

````

#### run_stream_events

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[OutputDataT]
]

```

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]
]

```

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[Any]
]

```

Run the agent with a user prompt in async mode and stream events from the run.

This is a convenience method that wraps self.run and uses the `event_stream_handler` kwarg to get a stream of events from the run.

Example:

```python
from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

agent = Agent('openai:gpt-4o')

async def main():
    events: list[AgentStreamEvent | AgentRunResultEvent] = []
    async for event in agent.run_stream_events('What is the capital of France?'):
        events.append(event)
    print(events)
    '''
    [
        PartStartEvent(index=0, part=TextPart(content='The capital of ')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
        PartEndEvent(
            index=0, part=TextPart(content='The capital of France is Paris. ')
        ),
        AgentRunResultEvent(
            result=AgentRunResult(output='The capital of France is Paris. ')
        ),
    ]
    '''

```

Arguments are the same as for self.run, except that `event_stream_handler` is now allowed.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]` | An async iterable of stream events AgentStreamEvent and finally a AgentRunResultEvent with the final | | `AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]` | run result. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`

````python
def run_stream_events(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
    """Run the agent with a user prompt in async mode and stream events from the run.

    This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
    uses the `event_stream_handler` kwarg to get a stream of events from the run.

    Example:
    ```python
    from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

    agent = Agent('openai:gpt-4o')

    async def main():
        events: list[AgentStreamEvent | AgentRunResultEvent] = []
        async for event in agent.run_stream_events('What is the capital of France?'):
            events.append(event)
        print(events)
        '''
        [
            PartStartEvent(index=0, part=TextPart(content='The capital of ')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
            PartEndEvent(
                index=0, part=TextPart(content='The capital of France is Paris. ')
            ),
            AgentRunResultEvent(
                result=AgentRunResult(output='The capital of France is Paris. ')
            ),
        ]
        '''
    ```

    Arguments are the same as for [`self.run`][pydantic_ai.agent.AbstractAgent.run],
    except that `event_stream_handler` is now allowed.

    Args:
        user_prompt: User input to start/continue the conversation.
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
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
        run result.
    """
    if FlowRunContext.get() is not None:
        raise UserError(
            '`agent.run_stream_events()` cannot be used inside a Prefect flow. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        )

    return super().run_stream_events(
        user_prompt,
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
    )

````

#### iter

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
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
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, OutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
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
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, RunOutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
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
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]

```

A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the stream of events coming from the execution of tools.

The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics, and the final result of the run once it has completed.

For more details, see the documentation of `AgentRun`.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    nodes = []
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            nodes.append(node)
    print(nodes)
    '''
    [
        UserPromptNode(
            user_prompt='What is the capital of France?',
            instructions_functions=[],
            system_prompts=(),
            system_prompt_functions=[],
            system_prompt_dynamic_functions={},
        ),
        ModelRequestNode(
            request=ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=datetime.datetime(...),
                    )
                ],
                run_id='...',
            )
        ),
        CallToolsNode(
            model_response=ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(input_tokens=56, output_tokens=7),
                model_name='gpt-4o',
                timestamp=datetime.datetime(...),
                run_id='...',
            )
        ),
        End(data=FinalResult(output='The capital of France is Paris.')),
    ]
    '''
    print(agent_run.result.output)
    #> The capital of France is Paris.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentRun[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`

````python
@asynccontextmanager
async def iter(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
    """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

    This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
    `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
    executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
    stream of events coming from the execution of tools.

    The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
    and the final result of the run once it has completed.

    For more details, see the documentation of `AgentRun`.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
        async with agent.iter('What is the capital of France?') as agent_run:
            async for node in agent_run:
                nodes.append(node)
        print(nodes)
        '''
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                instructions_functions=[],
                system_prompts=(),
                system_prompt_functions=[],
                system_prompt_dynamic_functions={},
            ),
            ModelRequestNode(
                request=ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=datetime.datetime(...),
                        )
                    ],
                    run_id='...',
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='The capital of France is Paris.')],
                    usage=RequestUsage(input_tokens=56, output_tokens=7),
                    model_name='gpt-4o',
                    timestamp=datetime.datetime(...),
                    run_id='...',
                )
            ),
            End(data=FinalResult(output='The capital of France is Paris.')),
        ]
        '''
        print(agent_run.result.output)
        #> The capital of France is Paris.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        instructions: Optional additional instructions to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    if model is not None and not isinstance(model, PrefectModel):
        raise UserError(
            'Non-Prefect model cannot be set at agent run time inside a Prefect flow, it must be set at agent creation time.'
        )

    with self._prefect_overrides():
        async with super().iter(
            user_prompt=user_prompt,
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
        ) as run:
            yield run

````

#### override

```python
override(
    *,
    name: str | Unset = UNSET,
    deps: AgentDepsT | Unset = UNSET,
    model: Model | KnownModelName | str | Unset = UNSET,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | Unset
    ) = UNSET,
    tools: (
        Sequence[
            Tool[AgentDepsT]
            | ToolFuncEither[AgentDepsT, ...]
        ]
        | Unset
    ) = UNSET,
    instructions: Instructions[AgentDepsT] | Unset = UNSET
) -> Iterator[None]

```

Context manager to temporarily override agent dependencies, model, toolsets, tools, or instructions.

This is particularly useful when testing. You can find an example of this [here](../../testing/#overriding-model-via-pytest-fixtures).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | Unset` | The name to use instead of the name passed to the agent constructor and agent run. | `UNSET` | | `deps` | `AgentDepsT | Unset` | The dependencies to use instead of the dependencies passed to the agent run. | `UNSET` | | `model` | `Model | KnownModelName | str | Unset` | The model to use instead of the model passed to the agent run. | `UNSET` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | Unset` | The toolsets to use instead of the toolsets passed to the agent constructor and agent run. | `UNSET` | | `tools` | `Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | Unset` | The tools to use instead of the tools registered with the agent. | `UNSET` | | `instructions` | `Instructions[AgentDepsT] | Unset` | The instructions to use instead of the instructions registered with the agent. | `UNSET` |

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`

```python
@contextmanager
def override(
    self,
    *,
    name: str | _utils.Unset = _utils.UNSET,
    deps: AgentDepsT | _utils.Unset = _utils.UNSET,
    model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
    instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
) -> Iterator[None]:
    """Context manager to temporarily override agent dependencies, model, toolsets, tools, or instructions.

    This is particularly useful when testing.
    You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

    Args:
        name: The name to use instead of the name passed to the agent constructor and agent run.
        deps: The dependencies to use instead of the dependencies passed to the agent run.
        model: The model to use instead of the model passed to the agent run.
        toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
        tools: The tools to use instead of the tools registered with the agent.
        instructions: The instructions to use instead of the instructions registered with the agent.
    """
    if _utils.is_set(model) and not isinstance(model, PrefectModel):
        raise UserError(
            'Non-Prefect model cannot be contextually overridden inside a Prefect flow, it must be set at agent creation time.'
        )

    with super().override(
        name=name, deps=deps, model=model, toolsets=toolsets, tools=tools, instructions=instructions
    ):
        yield

```

### PrefectFunctionToolset

Bases: `PrefectWrapperToolset[AgentDepsT]`

A wrapper for FunctionToolset that integrates with Prefect, turning tool calls into Prefect tasks.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_function_toolset.py`

```python
class PrefectFunctionToolset(PrefectWrapperToolset[AgentDepsT]):
    """A wrapper for FunctionToolset that integrates with Prefect, turning tool calls into Prefect tasks."""

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        task_config: TaskConfig,
        tool_task_config: dict[str, TaskConfig | None],
    ):
        super().__init__(wrapped)
        self._task_config = default_task_config | (task_config or {})
        self._tool_task_config = tool_task_config or {}

        @task
        async def _call_tool_task(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> Any:
            return await super(PrefectFunctionToolset, self).call_tool(tool_name, tool_args, ctx, tool)

        self._call_tool_task = _call_tool_task

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Call a tool, wrapped as a Prefect task with a descriptive name."""
        # Check if this specific tool has custom config or is disabled
        tool_specific_config = self._tool_task_config.get(name, default_task_config)
        if tool_specific_config is None:
            # None means this tool should not be wrapped as a task
            return await super().call_tool(name, tool_args, ctx, tool)

        # Merge tool-specific config with default config
        merged_config = self._task_config | tool_specific_config

        return await self._call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(
            name, tool_args, ctx, tool
        )

```

#### call_tool

```python
call_tool(
    name: str,
    tool_args: dict[str, Any],
    ctx: RunContext[AgentDepsT],
    tool: ToolsetTool[AgentDepsT],
) -> Any

```

Call a tool, wrapped as a Prefect task with a descriptive name.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_function_toolset.py`

```python
async def call_tool(
    self,
    name: str,
    tool_args: dict[str, Any],
    ctx: RunContext[AgentDepsT],
    tool: ToolsetTool[AgentDepsT],
) -> Any:
    """Call a tool, wrapped as a Prefect task with a descriptive name."""
    # Check if this specific tool has custom config or is disabled
    tool_specific_config = self._tool_task_config.get(name, default_task_config)
    if tool_specific_config is None:
        # None means this tool should not be wrapped as a task
        return await super().call_tool(name, tool_args, ctx, tool)

    # Merge tool-specific config with default config
    merged_config = self._task_config | tool_specific_config

    return await self._call_tool_task.with_options(name=f'Call Tool: {name}', **merged_config)(
        name, tool_args, ctx, tool
    )

```

### PrefectMCPServer

Bases: `PrefectWrapperToolset[AgentDepsT]`, `ABC`

A wrapper for MCPServer that integrates with Prefect, turning call_tool and get_tools into Prefect tasks.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_mcp_server.py`

```python
class PrefectMCPServer(PrefectWrapperToolset[AgentDepsT], ABC):
    """A wrapper for MCPServer that integrates with Prefect, turning call_tool and get_tools into Prefect tasks."""

    def __init__(
        self,
        wrapped: MCPServer,
        *,
        task_config: TaskConfig,
    ):
        super().__init__(wrapped)
        self._task_config = default_task_config | (task_config or {})
        self._mcp_id = wrapped.id

        @task
        async def _call_tool_task(
            tool_name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
        ) -> ToolResult:
            return await super(PrefectMCPServer, self).call_tool(tool_name, tool_args, ctx, tool)

        self._call_tool_task = _call_tool_task

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        """Call an MCP tool, wrapped as a Prefect task with a descriptive name."""
        return await self._call_tool_task.with_options(name=f'Call MCP Tool: {name}', **self._task_config)(
            name, tool_args, ctx, tool
        )

```

#### call_tool

```python
call_tool(
    name: str,
    tool_args: dict[str, Any],
    ctx: RunContext[AgentDepsT],
    tool: ToolsetTool[AgentDepsT],
) -> ToolResult

```

Call an MCP tool, wrapped as a Prefect task with a descriptive name.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_mcp_server.py`

```python
async def call_tool(
    self,
    name: str,
    tool_args: dict[str, Any],
    ctx: RunContext[AgentDepsT],
    tool: ToolsetTool[AgentDepsT],
) -> ToolResult:
    """Call an MCP tool, wrapped as a Prefect task with a descriptive name."""
    return await self._call_tool_task.with_options(name=f'Call MCP Tool: {name}', **self._task_config)(
        name, tool_args, ctx, tool
    )

```

### PrefectModel

Bases: `WrapperModel`

A wrapper for Model that integrates with Prefect, turning request and request_stream into Prefect tasks.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_model.py`

```python
class PrefectModel(WrapperModel):
    """A wrapper for Model that integrates with Prefect, turning request and request_stream into Prefect tasks."""

    def __init__(
        self,
        model: Any,
        *,
        task_config: TaskConfig,
        event_stream_handler: EventStreamHandler[Any] | None = None,
    ):
        super().__init__(model)
        self.task_config = default_task_config | (task_config or {})
        self.event_stream_handler = event_stream_handler

        @task
        async def wrapped_request(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            response = await super(PrefectModel, self).request(messages, model_settings, model_request_parameters)
            return response

        self._wrapped_request = wrapped_request

        @task
        async def request_stream_task(
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            ctx: RunContext[Any] | None,
        ) -> ModelResponse:
            async with super(PrefectModel, self).request_stream(
                messages, model_settings, model_request_parameters, ctx
            ) as streamed_response:
                if self.event_stream_handler is not None:
                    assert ctx is not None, (
                        'A Prefect model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. '
                        'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
                    )
                    await self.event_stream_handler(ctx, streamed_response)

                # Consume the entire stream
                async for _ in streamed_response:
                    pass
            response = streamed_response.get()
            return response

        self._wrapped_request_stream = request_stream_task

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a model request, wrapped as a Prefect task when in a flow."""
        return await self._wrapped_request.with_options(
            name=f'Model Request: {self.wrapped.model_name}', **self.task_config
        )(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming model request.

        When inside a Prefect flow, the stream is consumed within a task and
        a non-streaming response is returned. When not in a flow, behaves normally.
        """
        # Check if we're in a flow context
        flow_run_context = FlowRunContext.get()

        # If not in a flow, just call the wrapped request_stream method
        if flow_run_context is None:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                yield streamed_response
                return

        # If in a flow, consume the stream in a task and return the final response
        response = await self._wrapped_request_stream.with_options(
            name=f'Model Request (Streaming): {self.wrapped.model_name}', **self.task_config
        )(messages, model_settings, model_request_parameters, run_context)
        yield PrefectStreamedResponse(model_request_parameters, response)

```

#### request

```python
request(
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse

```

Make a model request, wrapped as a Prefect task when in a flow.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_model.py`

```python
async def request(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse:
    """Make a model request, wrapped as a Prefect task when in a flow."""
    return await self._wrapped_request.with_options(
        name=f'Model Request: {self.wrapped.model_name}', **self.task_config
    )(messages, model_settings, model_request_parameters)

```

#### request_stream

```python
request_stream(
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    run_context: RunContext[Any] | None = None,
) -> AsyncIterator[StreamedResponse]

```

Make a streaming model request.

When inside a Prefect flow, the stream is consumed within a task and a non-streaming response is returned. When not in a flow, behaves normally.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_model.py`

```python
@asynccontextmanager
async def request_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    run_context: RunContext[Any] | None = None,
) -> AsyncIterator[StreamedResponse]:
    """Make a streaming model request.

    When inside a Prefect flow, the stream is consumed within a task and
    a non-streaming response is returned. When not in a flow, behaves normally.
    """
    # Check if we're in a flow context
    flow_run_context = FlowRunContext.get()

    # If not in a flow, just call the wrapped request_stream method
    if flow_run_context is None:
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as streamed_response:
            yield streamed_response
            return

    # If in a flow, consume the stream in a task and return the final response
    response = await self._wrapped_request_stream.with_options(
        name=f'Model Request (Streaming): {self.wrapped.model_name}', **self.task_config
    )(messages, model_settings, model_request_parameters, run_context)
    yield PrefectStreamedResponse(model_request_parameters, response)

```

### TaskConfig

Bases: `TypedDict`

Configuration for a task in Prefect.

These options are passed to the `@task` decorator.

Source code in `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_types.py`

```python
class TaskConfig(TypedDict, total=False):
    """Configuration for a task in Prefect.

    These options are passed to the `@task` decorator.
    """

    retries: int
    """Maximum number of retries for the task."""

    retry_delay_seconds: float | list[float]
    """Delay between retries in seconds. Can be a single value or a list for custom backoff."""

    timeout_seconds: float
    """Maximum time in seconds for the task to complete."""

    cache_policy: CachePolicy
    """Prefect cache policy for the task."""

    persist_result: bool
    """Whether to persist the task result."""

    result_storage: ResultStorage
    """Prefect result storage for the task. Should be a storage block or a block slug like `s3-bucket/my-storage`."""

    log_prints: bool
    """Whether to log print statements from the task."""

```

#### retries

```python
retries: int

```

Maximum number of retries for the task.

#### retry_delay_seconds

```python
retry_delay_seconds: float | list[float]

```

Delay between retries in seconds. Can be a single value or a list for custom backoff.

#### timeout_seconds

```python
timeout_seconds: float

```

Maximum time in seconds for the task to complete.

#### cache_policy

```python
cache_policy: CachePolicy

```

Prefect cache policy for the task.

#### persist_result

```python
persist_result: bool

```

Whether to persist the task result.

#### result_storage

```python
result_storage: ResultStorage

```

Prefect result storage for the task. Should be a storage block or a block slug like `s3-bucket/my-storage`.

#### log_prints

```python
log_prints: bool

```

Whether to log print statements from the task.

