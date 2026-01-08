<!-- Source: _complete-doc.md -->

# pydantic_ai.models.mcp_sampling

### MCPSamplingModelSettings

Bases: `ModelSettings`

Settings used for an MCP Sampling model request.

Source code in `pydantic_ai_slim/pydantic_ai/models/mcp_sampling.py`

```python
class MCPSamplingModelSettings(ModelSettings, total=False):
    """Settings used for an MCP Sampling model request."""

    # ALL FIELDS MUST BE `mcp_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    mcp_model_preferences: ModelPreferences
    """Model preferences to use for MCP Sampling."""

```

#### mcp_model_preferences

```python
mcp_model_preferences: ModelPreferences

```

Model preferences to use for MCP Sampling.

### MCPSamplingModel

Bases: `Model`

A model that uses MCP Sampling.

[MCP Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) allows an MCP server to make requests to a model by calling back to the MCP client that connected to it.

Source code in `pydantic_ai_slim/pydantic_ai/models/mcp_sampling.py`

```python
@dataclass
class MCPSamplingModel(Model):
    """A model that uses MCP Sampling.

    [MCP Sampling](https://modelcontextprotocol.io/docs/concepts/sampling)
    allows an MCP server to make requests to a model by calling back to the MCP client that connected to it.
    """

    session: ServerSession
    """The MCP server session to use for sampling."""

    _: KW_ONLY

    default_max_tokens: int = 16_384
    """Default max tokens to use if not set in [`ModelSettings`][pydantic_ai.settings.ModelSettings.max_tokens].

    Max tokens is a required parameter for MCP Sampling, but optional on
    [`ModelSettings`][pydantic_ai.settings.ModelSettings], so this value is used as fallback.
    """

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        system_prompt, sampling_messages = _mcp.map_from_pai_messages(messages)

        model_settings, _ = self.prepare_request(model_settings, model_request_parameters)
        model_settings = cast(MCPSamplingModelSettings, model_settings or {})

        result = await self.session.create_message(
            sampling_messages,
            max_tokens=model_settings.get('max_tokens', self.default_max_tokens),
            system_prompt=system_prompt,
            temperature=model_settings.get('temperature'),
            model_preferences=model_settings.get('mcp_model_preferences'),
            stop_sequences=model_settings.get('stop_sequences'),
        )
        if result.role == 'assistant':
            return ModelResponse(
                parts=[_mcp.map_from_sampling_content(result.content)],
                model_name=result.model,
            )
        else:
            raise exceptions.UnexpectedModelBehavior(
                f'Unexpected result from MCP sampling, expected "assistant" role, got {result.role}.'
            )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        raise NotImplementedError('MCP Sampling does not support streaming')
        yield

    @property
    def model_name(self) -> str:
        """The model name.

        Since the model name isn't known until the request is made, this property always returns `'mcp-sampling'`.
        """
        return 'mcp-sampling'

    @property
    def system(self) -> str:
        """The system / model provider, returns `'MCP'`."""
        return 'MCP'

```

#### session

```python
session: ServerSession

```

The MCP server session to use for sampling.

#### default_max_tokens

```python
default_max_tokens: int = 16384

```

Default max tokens to use if not set in ModelSettings.

Max tokens is a required parameter for MCP Sampling, but optional on ModelSettings, so this value is used as fallback.

#### model_name

```python
model_name: str

```

The model name.

Since the model name isn't known until the request is made, this property always returns `'mcp-sampling'`.

#### system

```python
system: str

```

The system / model provider, returns `'MCP'`.

# `pydantic_ai.models.mistral`

## Setup

For details on how to set up authentication with this model, see [model configuration for Mistral](../../../models/mistral/).

### LatestMistralModelNames

```python
LatestMistralModelNames = Literal[
    "mistral-large-latest",
    "mistral-small-latest",
    "codestral-latest",
    "mistral-moderation-latest",
]

```

Latest Mistral models.

### MistralModelName

```python
MistralModelName = str | LatestMistralModelNames

```

Possible Mistral model names.

Since Mistral supports a variety of date-stamped models, we explicitly list the most popular models but allow any name in the type hints. Since [the Mistral docs](https://docs.mistral.ai/getting-started/models/models_overview/) for a full list.

### MistralModelSettings

Bases: `ModelSettings`

Settings used for a Mistral model request.

Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`

```python
class MistralModelSettings(ModelSettings, total=False):
    """Settings used for a Mistral model request."""

```

### MistralModel

Bases: `Model`

A model that uses Mistral.

Internally, this uses the [Mistral Python client](https://github.com/mistralai/client-python) to interact with the API.

[API Documentation](https://docs.mistral.ai/)

Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`

````python
@dataclass(init=False)
class MistralModel(Model):
    """A model that uses Mistral.

    Internally, this uses the [Mistral Python client](https://github.com/mistralai/client-python) to interact with the API.

    [API Documentation](https://docs.mistral.ai/)
    """

    client: Mistral = field(repr=False)
    json_mode_schema_prompt: str

    _model_name: MistralModelName = field(repr=False)
    _provider: Provider[Mistral] = field(repr=False)

    def __init__(
        self,
        model_name: MistralModelName,
        *,
        provider: Literal['mistral'] | Provider[Mistral] = 'mistral',
        profile: ModelProfileSpec | None = None,
        json_mode_schema_prompt: str = """Answer in JSON Object, respect the format:\n```\n{schema}\n```\n""",
        settings: ModelSettings | None = None,
    ):
        """Initialize a Mistral model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use for authentication and API access. Can be either the string
                'mistral' or an instance of `Provider[Mistral]`. If not provided, a new provider will be
                created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            json_mode_schema_prompt: The prompt to show when the model expects a JSON object as input.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name
        self.json_mode_schema_prompt = json_mode_schema_prompt

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> MistralModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a non-streaming request to the model from Pydantic AI call."""
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._completions_create(
            messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the model from Pydantic AI call."""
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._stream_completions_create(
            messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        model_settings: MistralModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> MistralChatCompletionResponse:
        """Make a non-streaming request to the model."""
        # TODO(Marcelo): We need to replace the current MistralAI client to use the beta client.
        # See https://docs.mistral.ai/agents/connectors/websearch/ to support web search.
        if model_request_parameters.builtin_tools:
            raise UserError('Mistral does not support built-in tools')

        try:
            response = await self.client.chat.complete_async(
                model=str(self._model_name),
                messages=self._map_messages(messages, model_request_parameters),
                n=1,
                tools=self._map_function_and_output_tools_definition(model_request_parameters) or UNSET,
                tool_choice=self._get_tool_choice(model_request_parameters),
                stream=False,
                max_tokens=model_settings.get('max_tokens', UNSET),
                temperature=model_settings.get('temperature', UNSET),
                top_p=model_settings.get('top_p', 1),
                timeout_ms=self._get_timeout_ms(model_settings.get('timeout')),
                random_seed=model_settings.get('seed', UNSET),
                stop=model_settings.get('stop_sequences', None),
                http_headers={'User-Agent': get_user_agent()},
            )
        except SDKError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

        assert response, 'A unexpected empty response from Mistral.'
        return response

    async def _stream_completions_create(
        self,
        messages: list[ModelMessage],
        model_settings: MistralModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> MistralEventStreamAsync[MistralCompletionEvent]:
        """Create a streaming completion request to the Mistral model."""
        response: MistralEventStreamAsync[MistralCompletionEvent] | None
        mistral_messages = self._map_messages(messages, model_request_parameters)

        # TODO(Marcelo): We need to replace the current MistralAI client to use the beta client.
        # See https://docs.mistral.ai/agents/connectors/websearch/ to support web search.
        if model_request_parameters.builtin_tools:
            raise UserError('Mistral does not support built-in tools')

        if model_request_parameters.function_tools:
            # Function Calling
            response = await self.client.chat.stream_async(
                model=str(self._model_name),
                messages=mistral_messages,
                n=1,
                tools=self._map_function_and_output_tools_definition(model_request_parameters) or UNSET,
                tool_choice=self._get_tool_choice(model_request_parameters),
                temperature=model_settings.get('temperature', UNSET),
                top_p=model_settings.get('top_p', 1),
                max_tokens=model_settings.get('max_tokens', UNSET),
                timeout_ms=self._get_timeout_ms(model_settings.get('timeout')),
                presence_penalty=model_settings.get('presence_penalty'),
                frequency_penalty=model_settings.get('frequency_penalty'),
                stop=model_settings.get('stop_sequences', None),
                http_headers={'User-Agent': get_user_agent()},
            )

        elif model_request_parameters.output_tools:
            # TODO: Port to native "manual JSON" mode
            # Json Mode
            parameters_json_schemas = [tool.parameters_json_schema for tool in model_request_parameters.output_tools]
            user_output_format_message = self._generate_user_output_format(parameters_json_schemas)
            mistral_messages.append(user_output_format_message)

            response = await self.client.chat.stream_async(
                model=str(self._model_name),
                messages=mistral_messages,
                response_format={
                    'type': 'json_object'
                },  # TODO: Should be able to use json_schema now: https://docs.mistral.ai/capabilities/structured-output/custom_structured_output/, https://github.com/mistralai/client-python/blob/bc4adf335968c8a272e1ab7da8461c9943d8e701/src/mistralai/extra/utils/response_format.py#L9
                stream=True,
                http_headers={'User-Agent': get_user_agent()},
            )

        else:
            # Stream Mode
            response = await self.client.chat.stream_async(
                model=str(self._model_name),
                messages=mistral_messages,
                stream=True,
                http_headers={'User-Agent': get_user_agent()},
            )
        assert response, 'A unexpected empty response from Mistral.'
        return response

    def _get_tool_choice(self, model_request_parameters: ModelRequestParameters) -> MistralToolChoiceEnum | None:
        """Get tool choice for the model.

        - "auto": Default mode. Model decides if it uses the tool or not.
        - "any": Select any tool.
        - "none": Prevents tool use.
        - "required": Forces tool use.
        """
        if not model_request_parameters.function_tools and not model_request_parameters.output_tools:
            return None
        elif not model_request_parameters.allow_text_output:
            return 'required'
        else:
            return 'auto'

    def _map_function_and_output_tools_definition(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[MistralTool] | None:
        """Map function and output tools to MistralTool format.

        Returns None if both function_tools and output_tools are empty.
        """
        tools = [
            MistralTool(
                function=MistralFunction(
                    name=r.name, parameters=r.parameters_json_schema, description=r.description or ''
                )
            )
            for r in model_request_parameters.tool_defs.values()
        ]
        return tools if tools else None

    def _process_response(self, response: MistralChatCompletionResponse) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        assert response.choices, 'Unexpected empty response choice.'

        if response.created:
            timestamp = number_to_datetime(response.created)
        else:
            timestamp = _now_utc()

        choice = response.choices[0]
        content = choice.message.content
        tool_calls = choice.message.tool_calls

        parts: list[ModelResponsePart] = []
        text, thinking = _map_content(content)
        for thought in thinking:
            parts.append(ThinkingPart(content=thought))
        if text:
            parts.append(TextPart(content=text))

        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                tool = self._map_mistral_to_pydantic_tool_call(tool_call=tool_call)
                parts.append(tool)

        raw_finish_reason = choice.finish_reason
        provider_details = {'finish_reason': raw_finish_reason}
        finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=parts,
            usage=_map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self,
        response: MistralEventStreamAsync[MistralCompletionEvent],
        model_request_parameters: ModelRequestParameters,
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        if first_chunk.data.created:
            timestamp = number_to_datetime(first_chunk.data.created)
        else:
            timestamp = _now_utc()

        return MistralStreamedResponse(
            model_request_parameters=model_request_parameters,
            _response=peekable_response,
            _model_name=first_chunk.data.model,
            _timestamp=timestamp,
            _provider_name=self._provider.name,
        )

    @staticmethod
    def _map_mistral_to_pydantic_tool_call(tool_call: MistralToolCall) -> ToolCallPart:
        """Maps a MistralToolCall to a ToolCall."""
        tool_call_id = tool_call.id or _generate_tool_call_id()
        func_call = tool_call.function

        return ToolCallPart(func_call.name, func_call.arguments, tool_call_id)

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> MistralToolCall:
        """Maps a pydantic-ai ToolCall to a MistralToolCall."""
        return MistralToolCall(
            id=_utils.guard_tool_call_id(t=t),
            type='function',
            function=MistralFunctionCall(name=t.tool_name, arguments=t.args or {}),
        )

    def _generate_user_output_format(self, schemas: list[dict[str, Any]]) -> MistralUserMessage:
        """Get a message with an example of the expected output format."""
        examples: list[dict[str, Any]] = []
        for schema in schemas:
            typed_dict_definition: dict[str, Any] = {}
            for key, value in schema.get('properties', {}).items():
                typed_dict_definition[key] = self._get_python_type(value)
            examples.append(typed_dict_definition)

        example_schema = examples[0] if len(examples) == 1 else examples
        return MistralUserMessage(content=self.json_mode_schema_prompt.format(schema=example_schema))

    @classmethod
    def _get_python_type(cls, value: dict[str, Any]) -> str:
        """Return a string representation of the Python type for a single JSON schema property.

        This function handles recursion for nested arrays/objects and `anyOf`.
        """
        # 1) Handle anyOf first, because it's a different schema structure
        if any_of := value.get('anyOf'):
            # Simplistic approach: pick the first option in anyOf
            # (In reality, you'd possibly want to merge or union types)
            return f'Optional[{cls._get_python_type(any_of[0])}]'

        # 2) If we have a top-level "type" field
        value_type = value.get('type')
        if not value_type:
            # No explicit type; fallback
            return 'Any'

        # 3) Direct simple type mapping (string, integer, float, bool, None)
        if value_type in SIMPLE_JSON_TYPE_MAPPING and value_type != 'array' and value_type != 'object':
            return SIMPLE_JSON_TYPE_MAPPING[value_type]

        # 4) Array: Recursively get the item type
        if value_type == 'array':
            items = value.get('items', {})
            return f'list[{cls._get_python_type(items)}]'

        # 5) Object: Check for additionalProperties
        if value_type == 'object':
            additional_properties = value.get('additionalProperties', {})
            if isinstance(additional_properties, bool):
                return 'bool'  # pragma: lax no cover
            additional_properties_type = additional_properties.get('type')
            if (
                additional_properties_type in SIMPLE_JSON_TYPE_MAPPING
                and additional_properties_type != 'array'
                and additional_properties_type != 'object'
            ):
                # dict[str, bool/int/float/etc...]
                return f'dict[str, {SIMPLE_JSON_TYPE_MAPPING[additional_properties_type]}]'
            elif additional_properties_type == 'array':
                array_items = additional_properties.get('items', {})
                return f'dict[str, list[{cls._get_python_type(array_items)}]]'
            elif additional_properties_type == 'object':
                # nested dictionary of unknown shape
                return 'dict[str, dict[str, Any]]'
            else:
                # If no additionalProperties type or something else, default to a generic dict
                return 'dict[str, Any]'

        # 6) Fallback
        return 'Any'

    @staticmethod
    def _get_timeout_ms(timeout: Timeout | float | None) -> int | None:
        """Convert a timeout to milliseconds."""
        if timeout is None:
            return None
        if isinstance(timeout, float):  # pragma: no cover
            return int(1000 * timeout)
        raise NotImplementedError('Timeout object is not yet supported for MistralModel.')

    def _map_user_message(self, message: ModelRequest) -> Iterable[MistralMessages]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield MistralSystemMessage(content=part.content)
            elif isinstance(part, UserPromptPart):
                yield self._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield MistralToolMessage(
                    tool_call_id=part.tool_call_id,
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield MistralUserMessage(content=part.model_response())  # pragma: no cover
                else:
                    yield MistralToolMessage(
                        tool_call_id=part.tool_call_id,
                        content=part.model_response(),
                    )
            else:
                assert_never(part)

    def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[MistralMessages]:
        """Just maps a `pydantic_ai.Message` to a `MistralMessage`."""
        mistral_messages: list[MistralMessages] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                mistral_messages.extend(self._map_user_message(message))
            elif isinstance(message, ModelResponse):
                content_chunks: list[MistralContentChunk] = []
                thinking_chunks: list[MistralTextChunk | MistralReferenceChunk] = []
                tool_calls: list[MistralToolCall] = []

                for part in message.parts:
                    if isinstance(part, TextPart):
                        content_chunks.append(MistralTextChunk(text=part.content))
                    elif isinstance(part, ThinkingPart):
                        thinking_chunks.append(MistralTextChunk(text=part.content))
                    elif isinstance(part, ToolCallPart):
                        tool_calls.append(self._map_tool_call(part))
                    elif isinstance(part, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                        # This is currently never returned from mistral
                        pass
                    elif isinstance(part, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
                        pass
                    else:
                        assert_never(part)
                if thinking_chunks:
                    content_chunks.insert(0, MistralThinkChunk(thinking=thinking_chunks))
                mistral_messages.append(MistralAssistantMessage(content=content_chunks, tool_calls=tool_calls))
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages, model_request_parameters):
            mistral_messages.insert(0, MistralSystemMessage(content=instructions))

        # Post-process messages to insert fake assistant message after tool message if followed by user message
        # to work around `Unexpected role 'user' after role 'tool'` error.
        processed_messages: list[MistralMessages] = []
        for i, current_message in enumerate(mistral_messages):
            processed_messages.append(current_message)

            if isinstance(current_message, MistralToolMessage) and i + 1 < len(mistral_messages):
                next_message = mistral_messages[i + 1]
                if isinstance(next_message, MistralUserMessage):
                    # Insert a dummy assistant message
                    processed_messages.append(MistralAssistantMessage(content=[MistralTextChunk(text='OK')]))

        return processed_messages

    def _map_user_prompt(self, part: UserPromptPart) -> MistralUserMessage:
        content: str | list[MistralContentChunk]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(MistralTextChunk(text=item))
                elif isinstance(item, ImageUrl):
                    content.append(MistralImageURLChunk(image_url=MistralImageURL(url=item.url)))
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        image_url = MistralImageURL(url=item.data_uri)
                        content.append(MistralImageURLChunk(image_url=image_url, type='image_url'))
                    elif item.media_type == 'application/pdf':
                        content.append(MistralDocumentURLChunk(document_url=item.data_uri, type='document_url'))
                    else:
                        raise RuntimeError('BinaryContent other than image or PDF is not supported in Mistral.')
                elif isinstance(item, DocumentUrl):
                    if item.media_type == 'application/pdf':
                        content.append(MistralDocumentURLChunk(document_url=item.url, type='document_url'))
                    else:
                        raise RuntimeError('DocumentUrl other than PDF is not supported in Mistral.')
                elif isinstance(item, VideoUrl):
                    raise RuntimeError('VideoUrl is not supported in Mistral.')
                else:  # pragma: no cover
                    raise RuntimeError(f'Unsupported content type: {type(item)}')
        return MistralUserMessage(content=content)

````

#### __init__

````python
__init__(
    model_name: MistralModelName,
    *,
    provider: (
        Literal["mistral"] | Provider[Mistral]
    ) = "mistral",
    profile: ModelProfileSpec | None = None,
    json_mode_schema_prompt: str = "Answer in JSON Object, respect the format:\n```\n{schema}\n```\n",
    settings: ModelSettings | None = None
)

````

Initialize a Mistral model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model_name` | `MistralModelName` | The name of the model to use. | *required* | | `provider` | `Literal['mistral'] | Provider[Mistral]` | The provider to use for authentication and API access. Can be either the string 'mistral' or an instance of Provider[Mistral]. If not provided, a new provider will be created using the other parameters. | `'mistral'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider based on the model name. | `None` | | `json_mode_schema_prompt` | `str` | The prompt to show when the model expects a JSON object as input. | ```` 'Answer in JSON Object, respect the format:\n```\n{schema}\n```\n' ```` | | `settings` | `ModelSettings | None` | Model-specific settings that will be used as defaults for this model. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`

````python
def __init__(
    self,
    model_name: MistralModelName,
    *,
    provider: Literal['mistral'] | Provider[Mistral] = 'mistral',
    profile: ModelProfileSpec | None = None,
    json_mode_schema_prompt: str = """Answer in JSON Object, respect the format:\n```\n{schema}\n```\n""",
    settings: ModelSettings | None = None,
):
    """Initialize a Mistral model.

    Args:
        model_name: The name of the model to use.
        provider: The provider to use for authentication and API access. Can be either the string
            'mistral' or an instance of `Provider[Mistral]`. If not provided, a new provider will be
            created using the other parameters.
        profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
        json_mode_schema_prompt: The prompt to show when the model expects a JSON object as input.
        settings: Model-specific settings that will be used as defaults for this model.
    """
    self._model_name = model_name
    self.json_mode_schema_prompt = json_mode_schema_prompt

    if isinstance(provider, str):
        provider = infer_provider(provider)
    self._provider = provider
    self.client = provider.client

    super().__init__(settings=settings, profile=profile or provider.model_profile)

````

#### model_name

```python
model_name: MistralModelName

```

The model name.

#### system

```python
system: str

```

The model provider.

#### request

```python
request(
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse

```

Make a non-streaming request to the model from Pydantic AI call.

Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`

```python
async def request(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse:
    """Make a non-streaming request to the model from Pydantic AI call."""
    check_allow_model_requests()
    model_settings, model_request_parameters = self.prepare_request(
        model_settings,
        model_request_parameters,
    )
    response = await self._completions_create(
        messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
    )
    model_response = self._process_response(response)
    return model_response

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

Make a streaming request to the model from Pydantic AI call.

Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`

```python
@asynccontextmanager
async def request_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    run_context: RunContext[Any] | None = None,
) -> AsyncIterator[StreamedResponse]:
    """Make a streaming request to the model from Pydantic AI call."""
    check_allow_model_requests()
    model_settings, model_request_parameters = self.prepare_request(
        model_settings,
        model_request_parameters,
    )
    response = await self._stream_completions_create(
        messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
    )
    async with response:
        yield await self._process_streamed_response(response, model_request_parameters)

```

### MistralStreamedResponse

Bases: `StreamedResponse`

Implementation of `StreamedResponse` for Mistral models.

Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`

```python
@dataclass
class MistralStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Mistral models."""

    _model_name: MistralModelName
    _response: AsyncIterable[MistralCompletionEvent]
    _timestamp: datetime
    _provider_name: str

    _delta_content: str = field(default='', init=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        chunk: MistralCompletionEvent
        async for chunk in self._response:
            self._usage += _map_usage(chunk.data)

            if chunk.data.id:  # pragma: no branch
                self.provider_response_id = chunk.data.id

            try:
                choice = chunk.data.choices[0]
            except IndexError:
                continue

            if raw_finish_reason := choice.finish_reason:
                self.provider_details = {'finish_reason': raw_finish_reason}
                self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

            # Handle the text part of the response
            content = choice.delta.content
            text, thinking = _map_content(content)
            for thought in thinking:
                self._parts_manager.handle_thinking_delta(vendor_part_id='thinking', content=thought)
            if text:
                # Attempt to produce an output tool call from the received text
                output_tools = {c.name: c for c in self.model_request_parameters.output_tools}
                if output_tools:
                    self._delta_content += text
                    # TODO: Port to native "manual JSON" mode
                    maybe_tool_call_part = self._try_get_output_tool_from_text(self._delta_content, output_tools)
                    if maybe_tool_call_part:
                        yield self._parts_manager.handle_tool_call_part(
                            vendor_part_id='output',
                            tool_name=maybe_tool_call_part.tool_name,
                            args=maybe_tool_call_part.args_as_dict(),
                            tool_call_id=maybe_tool_call_part.tool_call_id,
                        )
                else:
                    maybe_event = self._parts_manager.handle_text_delta(vendor_part_id='content', content=text)
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

            # Handle the explicit tool calls
            for index, dtc in enumerate(choice.delta.tool_calls or []):
                # It seems that mistral just sends full tool calls, so we just use them directly, rather than building
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=index, tool_name=dtc.function.name, args=dtc.function.arguments, tool_call_id=dtc.id
                )

    @property
    def model_name(self) -> MistralModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp

    @staticmethod
    def _try_get_output_tool_from_text(text: str, output_tools: dict[str, ToolDefinition]) -> ToolCallPart | None:
        output_json: dict[str, Any] | None = pydantic_core.from_json(text, allow_partial='trailing-strings')
        if output_json:
            for output_tool in output_tools.values():
                # NOTE: Additional verification to prevent JSON validation to crash
                # Ensures required parameters in the JSON schema are respected, especially for stream-based return types.
                # Example with BaseModel and required fields.
                if not MistralStreamedResponse._validate_required_json_schema(
                    output_json, output_tool.parameters_json_schema
                ):
                    continue

                # The following part_id will be thrown away
                return ToolCallPart(tool_name=output_tool.name, args=output_json)

    @staticmethod
    def _validate_required_json_schema(json_dict: dict[str, Any], json_schema: dict[str, Any]) -> bool:
        """Validate that all required parameters in the JSON schema are present in the JSON dictionary."""
        required_params = json_schema.get('required', [])
        properties = json_schema.get('properties', {})

        for param in required_params:
            if param not in json_dict:
                return False

            param_schema = properties.get(param, {})
            param_type = param_schema.get('type')
            param_items_type = param_schema.get('items', {}).get('type')

            if param_type == 'array' and param_items_type:
                if not isinstance(json_dict[param], list):
                    return False
                for item in json_dict[param]:
                    if not isinstance(item, VALID_JSON_TYPE_MAPPING[param_items_type]):
                        return False
            elif param_type and not isinstance(json_dict[param], VALID_JSON_TYPE_MAPPING[param_type]):
                return False

            if isinstance(json_dict[param], dict) and 'properties' in param_schema:
                nested_schema = param_schema
                if not MistralStreamedResponse._validate_required_json_schema(json_dict[param], nested_schema):
                    return False

        return True

```

#### model_name

```python
model_name: MistralModelName

```

Get the model name of the response.

#### provider_name

```python
provider_name: str

```

Get the provider name.

#### timestamp

```python
timestamp: datetime

```

Get the timestamp of the response.

# `pydantic_ai.models.openai`

## Setup

For details on how to set up authentication with this model, see [model configuration for OpenAI](../../../models/openai/).

### OpenAIModelName

```python
OpenAIModelName = str | AllModels

```

Possible OpenAI model names.

Since OpenAI supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [the OpenAI docs](https://platform.openai.com/docs/models) for a full list.

Using this more broad type for the model name instead of the ChatModel definition allows this model to be used more easily with other model types (ie, Ollama, Deepseek).

### OpenAIChatModelSettings

Bases: `ModelSettings`

Settings used for an OpenAI model request.

Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`

```python
class OpenAIChatModelSettings(ModelSettings, total=False):
    """Settings used for an OpenAI model request."""

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openai_reasoning_effort: ReasoningEffort
    """Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning).

    Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
    result in faster responses and fewer tokens used on reasoning in a response.
    """

    openai_logprobs: bool
    """Include log probabilities in the response."""

    openai_top_logprobs: int
    """Include log probabilities of the top n tokens in the response."""

    openai_user: str
    """A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.

    See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.
    """

    openai_service_tier: Literal['auto', 'default', 'flex', 'priority']
    """The service tier to use for the model request.

    Currently supported values are `auto`, `default`, `flex`, and `priority`.
    For more information, see [OpenAI's service tiers documentation](https://platform.openai.com/docs/api-reference/chat/object#chat/object-service_tier).
    """

    openai_prediction: ChatCompletionPredictionContentParam
    """Enables [predictive outputs](https://platform.openai.com/docs/guides/predicted-outputs).

    This feature is currently only supported for some OpenAI models.
    """

```

#### openai_reasoning_effort

```python
openai_reasoning_effort: ReasoningEffort

```

Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning).

Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response.

#### openai_logprobs

```python
openai_logprobs: bool

```

Include log probabilities in the response.

#### openai_top_logprobs

```python
openai_top_logprobs: int

```

Include log probabilities of the top n tokens in the response.

#### openai_user

```python
openai_user: str

```

A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.

See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.

#### openai_service_tier

```python
openai_service_tier: Literal[
    "auto", "default", "flex", "priority"
]

```

The service tier to use for the model request.

Currently supported values are `auto`, `default`, `flex`, and `priority`. For more information, see [OpenAI's service tiers documentation](https://platform.openai.com/docs/api-reference/chat/object#chat/object-service_tier).

#### openai_prediction

```python
openai_prediction: ChatCompletionPredictionContentParam

```

Enables [predictive outputs](https://platform.openai.com/docs/guides/predicted-outputs).

This feature is currently only supported for some OpenAI models.

### OpenAIModelSettings

Bases: `OpenAIChatModelSettings`

Deprecated

Use `OpenAIChatModelSettings` instead.

Deprecated alias for `OpenAIChatModelSettings`.

Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`

```python
@deprecated('Use `OpenAIChatModelSettings` instead.')
class OpenAIModelSettings(OpenAIChatModelSettings, total=False):
    """Deprecated alias for `OpenAIChatModelSettings`."""

```

### OpenAIResponsesModelSettings

Bases: `OpenAIChatModelSettings`

Settings used for an OpenAI Responses model request.

ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`

```python
class OpenAIResponsesModelSettings(OpenAIChatModelSettings, total=False):
    """Settings used for an OpenAI Responses model request.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_builtin_tools: Sequence[FileSearchToolParam | WebSearchToolParam | ComputerToolParam]
    """The provided OpenAI built-in tools to use.

    See [OpenAI's built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses) for more details.
    """

    openai_reasoning_generate_summary: Literal['detailed', 'concise']
    """Deprecated alias for `openai_reasoning_summary`."""

    openai_reasoning_summary: Literal['detailed', 'concise']
    """A summary of the reasoning performed by the model.

    This can be useful for debugging and understanding the model's reasoning process.
    One of `concise` or `detailed`.

    Check the [OpenAI Reasoning documentation](https://platform.openai.com/docs/guides/reasoning?api-mode=responses#reasoning-summaries)
    for more details.
    """

    openai_send_reasoning_ids: bool
    """Whether to send the unique IDs of reasoning, text, and function call parts from the message history to the model. Enabled by default for reasoning models.

    This can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."`
    if the message history you're sending does not match exactly what was received from the Responses API in a previous response,
    for example if you're using a [history processor](../../message-history.md#processing-message-history).
    In that case, you'll want to disable this.
    """

    openai_truncation: Literal['disabled', 'auto']
    """The truncation strategy to use for the model response.

    It can be either:
    - `disabled` (default): If a model response will exceed the context window size for a model, the
        request will fail with a 400 error.
    - `auto`: If the context of this response and previous ones exceeds the model's context window size,
        the model will truncate the response to fit the context window by dropping input items in the
        middle of the conversation.
    """

    openai_text_verbosity: Literal['low', 'medium', 'high']
    """Constrains the verbosity of the model's text response.

    Lower values will result in more concise responses, while higher values will
    result in more verbose responses. Currently supported values are `low`,
    `medium`, and `high`.
    """

    openai_previous_response_id: Literal['auto'] | str
    """The ID of a previous response from the model to use as the starting point for a continued conversation.

    When set to `'auto'`, the request automatically uses the most recent
    `provider_response_id` from the message history and omits earlier messages.

    This enables the model to use server-side conversation state and faithfully reference previous reasoning.
    See the [OpenAI Responses API documentation](https://platform.openai.com/docs/guides/reasoning#keeping-reasoning-items-in-context)
    for more information.
    """

    openai_include_code_execution_outputs: bool
    """Whether to include the code execution results in the response.

    Corresponds to the `code_interpreter_call.outputs` value of the `include` parameter in the Responses API.
    """

    openai_include_web_search_sources: bool
    """Whether to include the web search results in the response.

    Corresponds to the `web_search_call.action.sources` value of the `include` parameter in the Responses API.
    """

```

#### openai_builtin_tools

```python
openai_builtin_tools: Sequence[
    FileSearchToolParam
    | WebSearchToolParam
    | ComputerToolParam
]

```

The provided OpenAI built-in tools to use.

See [OpenAI's built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses) for more details.

#### openai_reasoning_generate_summary

```python
openai_reasoning_generate_summary: Literal[
    "detailed", "concise"
]

```

Deprecated alias for `openai_reasoning_summary`.

#### openai_reasoning_summary

```python
openai_reasoning_summary: Literal['detailed', 'concise']

```

A summary of the reasoning performed by the model.

This can be useful for debugging and understanding the model's reasoning process. One of `concise` or `detailed`.

Check the [OpenAI Reasoning documentation](https://platform.openai.com/docs/guides/reasoning?api-mode=responses#reasoning-summaries) for more details.

#### openai_send_reasoning_ids

```python
openai_send_reasoning_ids: bool

```

Whether to send the unique IDs of reasoning, text, and function call parts from the message history to the model. Enabled by default for reasoning models.

This can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."` if the message history you're sending does not match exactly what was received from the Responses API in a previous response, for example if you're using a [history processor](../../../message-history/#processing-message-history). In that case, you'll want to disable this.

#### openai_truncation

```python
openai_truncation: Literal['disabled', 'auto']

```

The truncation strategy to use for the model response.

It can be either:

- `disabled` (default): If a model response will exceed the context window size for a model, the request will fail with a 400 error.
- `auto`: If the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.

#### openai_text_verbosity

```python
openai_text_verbosity: Literal['low', 'medium', 'high']

```

Constrains the verbosity of the model's text response.

Lower values will result in more concise responses, while higher values will result in more verbose responses. Currently supported values are `low`, `medium`, and `high`.

#### openai_previous_response_id

```python
openai_previous_response_id: Literal['auto'] | str

```

The ID of a previous response from the model to use as the starting point for a continued conversation.

When set to `'auto'`, the request automatically uses the most recent `provider_response_id` from the message history and omits earlier messages.

This enables the model to use server-side conversation state and faithfully reference previous reasoning. See the [OpenAI Responses API documentation](https://platform.openai.com/docs/guides/reasoning#keeping-reasoning-items-in-context) for more information.

#### openai_include_code_execution_outputs

```python
openai_include_code_execution_outputs: bool

```

Whether to include the code execution results in the response.

Corresponds to the `code_interpreter_call.outputs` value of the `include` parameter in the Responses API.

#### openai_include_web_search_sources

```python
openai_include_web_search_sources: bool

```

Whether to include the web search results in the response.

Corresponds to the `web_search_call.action.sources` value of the `include` parameter in the Responses API.

### OpenAIChatModel

Bases: `Model`

A model that uses the OpenAI API.

Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.

Apart from `__init__`, all methods are private or match those of the base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`

```python
@dataclass(init=False)
class OpenAIChatModel(Model):
    """A model that uses the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncOpenAI = field(repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    @overload
    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
            'nebius',
            'ovhcloud',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    @deprecated('Set the `system_prompt_role` in the `OpenAIModelProfile` instead.')
    @overload
    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
            'nebius',
            'ovhcloud',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'azure',
            'deepseek',
            'cerebras',
            'fireworks',
            'github',
            'grok',
            'heroku',
            'moonshotai',
            'ollama',
            'openai',
            'openai-chat',
            'openrouter',
            'together',
            'vercel',
            'litellm',
            'nebius',
            'ovhcloud',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI model.

        Args:
            model_name: The name of the OpenAI model to use. List of model names available
                [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
                (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            system_prompt_role: The role to use for the system prompt message. If not provided, defaults to `'system'`.
                In the future, this may be inferred from the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

        if system_prompt_role is not None:
            self.profile = OpenAIModelProfile(openai_system_prompt_role=system_prompt_role).update(self.profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    @property
    @deprecated('Set the `system_prompt_role` in the `OpenAIModelProfile` instead.')
    def system_prompt_role(self) -> OpenAISystemPromptRole | None:
        return OpenAIModelProfile.from_profile(self.profile).openai_system_prompt_role

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._completions_create(
            messages, False, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._completions_create(
            messages, True, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)
        web_search_options = self._get_web_search_options(model_request_parameters)

        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif (
            not model_request_parameters.allow_text_output
            and OpenAIModelProfile.from_profile(self.profile).openai_supports_tool_choice_required
        ):
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages = await self._map_messages(messages, model_request_parameters)

        response_format: chat.completion_create_params.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output
        ):  # pragma: no branch
            response_format = {'type': 'json_object'}

        unsupported_model_settings = OpenAIModelProfile.from_profile(self.profile).openai_unsupported_model_settings
        for setting in unsupported_model_settings:
            model_settings.pop(setting, None)

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', OMIT),
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                stream=stream,
                stream_options={'include_usage': True} if stream else OMIT,
                stop=model_settings.get('stop_sequences', OMIT),
                max_completion_tokens=model_settings.get('max_tokens', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                response_format=response_format or OMIT,
                seed=model_settings.get('seed', OMIT),
                reasoning_effort=model_settings.get('openai_reasoning_effort', OMIT),
                user=model_settings.get('openai_user', OMIT),
                web_search_options=web_search_options or OMIT,
                service_tier=model_settings.get('openai_service_tier', OMIT),
                prediction=model_settings.get('openai_prediction', OMIT),
                temperature=model_settings.get('temperature', OMIT),
                top_p=model_settings.get('top_p', OMIT),
                presence_penalty=model_settings.get('presence_penalty', OMIT),
                frequency_penalty=model_settings.get('frequency_penalty', OMIT),
                logit_bias=model_settings.get('logit_bias', OMIT),
                logprobs=model_settings.get('openai_logprobs', OMIT),
                top_logprobs=model_settings.get('openai_top_logprobs', OMIT),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

    def _validate_completion(self, response: chat.ChatCompletion) -> chat.ChatCompletion:
        """Hook that validates chat completions before processing.

        This method may be overridden by subclasses of `OpenAIChatModel` to apply custom completion validations.
        """
        return chat.ChatCompletion.model_validate(response.model_dump())

    def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any]:
        """Hook that response content to provider details.

        This method may be overridden by subclasses of `OpenAIChatModel` to apply custom mappings.
        """
        return _map_provider_details(response.choices[0])

    def _process_response(self, response: chat.ChatCompletion | str) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        # Although the OpenAI SDK claims to return a Pydantic model (`ChatCompletion`) from the chat completions function:
        # * it hasn't actually performed validation (presumably they're creating the model with `model_construct` or something?!)
        # * if the endpoint returns plain text, the return type is a string
        # Thus we validate it fully here.
        if not isinstance(response, chat.ChatCompletion):
            raise UnexpectedModelBehavior(
                f'Invalid response from {self.system} chat completions endpoint, expected JSON data'
            )

        if response.created:
            timestamp = number_to_datetime(response.created)
        else:
            timestamp = _now_utc()
            response.created = int(timestamp.timestamp())

        # Workaround for local Ollama which sometimes returns a `None` finish reason.
        if response.choices and (choice := response.choices[0]) and choice.finish_reason is None:  # pyright: ignore[reportUnnecessaryComparison]
            choice.finish_reason = 'stop'

        try:
            response = self._validate_completion(response)
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from {self.system} chat completions endpoint: {e}') from e

        choice = response.choices[0]
        items: list[ModelResponsePart] = []

        if thinking_parts := self._process_thinking(choice.message):
            items.extend(thinking_parts)

        if choice.message.content:
            items.extend(
                (replace(part, id='content', provider_name=self.system) if isinstance(part, ThinkingPart) else part)
                for part in split_content_into_text_and_thinking(choice.message.content, self.profile.thinking_tags)
            )
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                if isinstance(c, ChatCompletionMessageFunctionToolCall):
                    part = ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id)
                elif isinstance(c, ChatCompletionMessageCustomToolCall):  # pragma: no cover
                    # NOTE: Custom tool calls are not supported.
                    # See <https://github.com/pydantic/pydantic-ai/issues/2513> for more details.
                    raise RuntimeError('Custom tool calls are not supported')
                else:
                    assert_never(c)
                part.tool_call_id = _guard_tool_call_id(part)
                items.append(part)

        return ModelResponse(
            parts=items,
            usage=self._map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            provider_details=self._process_provider_details(response),
            provider_response_id=response.id,
            provider_name=self._provider.name,
            finish_reason=self._map_finish_reason(choice.finish_reason),
        )

    def _process_thinking(self, message: chat.ChatCompletionMessage) -> list[ThinkingPart] | None:
        """Hook that maps reasoning tokens to thinking parts.

        This method may be overridden by subclasses of `OpenAIChatModel` to apply custom mappings.
        """
        items: list[ThinkingPart] = []

        # The `reasoning_content` field is only present in DeepSeek models.
        # https://api-docs.deepseek.com/guides/reasoning_model
        if reasoning_content := getattr(message, 'reasoning_content', None):
            items.append(ThinkingPart(id='reasoning_content', content=reasoning_content, provider_name=self.system))

        # The `reasoning` field is only present in gpt-oss via Ollama and OpenRouter.
        # - https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot#chat-completions-api
        # - https://openrouter.ai/docs/use-cases/reasoning-tokens#basic-usage-with-reasoning-tokens
        if reasoning := getattr(message, 'reasoning', None):
            items.append(ThinkingPart(id='reasoning', content=reasoning, provider_name=self.system))

        return items

    async def _process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk], model_request_parameters: ModelRequestParameters
    ) -> OpenAIStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        # When using Azure OpenAI and a content filter is enabled, the first chunk will contain a `''` model name,
        # so we set it from a later chunk in `OpenAIChatStreamedResponse`.
        model_name = first_chunk.model or self._model_name

        return self._streamed_response_cls(
            model_request_parameters=model_request_parameters,
            _model_name=model_name,
            _model_profile=self.profile,
            _response=peekable_response,
            _timestamp=number_to_datetime(first_chunk.created),
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
        )

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        """Returns the `StreamedResponse` type that will be used for streamed responses.

        This method may be overridden by subclasses of `OpenAIChatModel` to provide their own `StreamedResponse` type.
        """
        return OpenAIStreamedResponse

    def _map_usage(self, response: chat.ChatCompletion) -> usage.RequestUsage:
        return _map_usage(response, self._provider.name, self._provider.base_url, self._model_name)

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_web_search_options(self, model_request_parameters: ModelRequestParameters) -> WebSearchOptions | None:
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):  # pragma: no branch
                if not OpenAIModelProfile.from_profile(self.profile).openai_chat_supports_web_search:
                    raise UserError(
                        f'WebSearchTool is not supported with `OpenAIChatModel` and model {self.model_name!r}. '
                        f'Please use `OpenAIResponsesModel` instead.'
                    )

                if tool.user_location:
                    return WebSearchOptions(
                        search_context_size=tool.search_context_size,
                        user_location=WebSearchOptionsUserLocation(
                            type='approximate',
                            approximate=WebSearchOptionsUserLocationApproximate(**tool.user_location),
                        ),
                    )
                return WebSearchOptions(search_context_size=tool.search_context_size)
            else:
                raise UserError(
                    f'`{tool.__class__.__name__}` is not supported by `OpenAIChatModel`. If it should be, please file an issue.'
                )

    @dataclass
    class _MapModelResponseContext:
        """Context object for mapping a `ModelResponse` to OpenAI chat completion parameters.

        This class is designed to be subclassed to add new fields for custom logic,
        collecting various parts of the model response (like text and tool calls)
        to form a single assistant message.
        """

        _model: OpenAIChatModel

        texts: list[str] = field(default_factory=list)
        tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = field(default_factory=list)

        def map_assistant_message(self, message: ModelResponse) -> chat.ChatCompletionAssistantMessageParam:
            for item in message.parts:
                if isinstance(item, TextPart):
                    self._map_response_text_part(item)
                elif isinstance(item, ThinkingPart):
                    self._map_response_thinking_part(item)
                elif isinstance(item, ToolCallPart):
                    self._map_response_tool_call_part(item)
                elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                    self._map_response_builtin_part(item)
                elif isinstance(item, FilePart):  # pragma: no cover
                    self._map_response_file_part(item)
                else:
                    assert_never(item)
            return self._into_message_param()

        def _into_message_param(self) -> chat.ChatCompletionAssistantMessageParam:
            """Converts the collected texts and tool calls into a single OpenAI `ChatCompletionAssistantMessageParam`.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for how collected parts are transformed into the final message parameter.

            Returns:
                An OpenAI `ChatCompletionAssistantMessageParam` object representing the assistant's response.
            """
            message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
            if self.texts:
                # Note: model responses from this model should only have one text item, so the following
                # shouldn't merge multiple texts into one unless you switch models between runs:
                message_param['content'] = '\n\n'.join(self.texts)
            else:
                message_param['content'] = None
            if self.tool_calls:
                message_param['tool_calls'] = self.tool_calls
            return message_param

        def _map_response_text_part(self, item: TextPart) -> None:
            """Maps a `TextPart` to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling text parts.
            """
            self.texts.append(item.content)

        def _map_response_thinking_part(self, item: ThinkingPart) -> None:
            """Maps a `ThinkingPart` to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling thinking parts.
            """
            # NOTE: DeepSeek `reasoning_content` field should NOT be sent back per https://api-docs.deepseek.com/guides/reasoning_model,
            # but we currently just send it in `<think>` tags anyway as we don't want DeepSeek-specific checks here.
            # If you need this changed, please file an issue.
            start_tag, end_tag = self._model.profile.thinking_tags
            self.texts.append('\n'.join([start_tag, item.content, end_tag]))

        def _map_response_tool_call_part(self, item: ToolCallPart) -> None:
            """Maps a `ToolCallPart` to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling tool call parts.
            """
            self.tool_calls.append(self._model._map_tool_call(item))

        def _map_response_builtin_part(self, item: BuiltinToolCallPart | BuiltinToolReturnPart) -> None:
            """Maps a built-in tool call or return part to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling built-in tool parts.
            """
            # OpenAI doesn't return built-in tool calls
            pass

        def _map_response_file_part(self, item: FilePart) -> None:
            """Maps a `FilePart` to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling file parts.
            """
            # Files generated by models are not sent back to models that don't themselves generate files.
            pass

    def _map_model_response(self, message: ModelResponse) -> chat.ChatCompletionMessageParam:
        """Hook that determines how `ModelResponse` is mapped into `ChatCompletionMessageParam` objects before sending.

        Subclasses of `OpenAIChatModel` may override this method to provide their own mapping logic.
        """
        return self._MapModelResponseContext(self).map_assistant_message(message)

    def _map_finish_reason(
        self, key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']
    ) -> FinishReason | None:
        """Hooks that maps a finish reason key to a [FinishReason][pydantic_ai.messages.FinishReason].

        This method may be overridden by subclasses of `OpenAIChatModel` to accommodate custom keys.
        """
        return _CHAT_FINISH_REASON_MAP.get(key)

    async def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        openai_messages: list[chat.ChatCompletionMessageParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                async for item in self._map_user_message(message):
                    openai_messages.append(item)
            elif isinstance(message, ModelResponse):
                openai_messages.append(self._map_model_response(message))
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages, model_request_parameters):
            openai_messages.insert(0, chat.ChatCompletionSystemMessageParam(content=instructions, role='system'))
        return openai_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ChatCompletionMessageFunctionToolCallParam:
        return ChatCompletionMessageFunctionToolCallParam(
            id=_guard_tool_call_id(t=t),
            type='function',
            function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
        )

    def _map_json_schema(self, o: OutputObjectDefinition) -> chat.completion_create_params.ResponseFormat:
        response_format_param: chat.completion_create_params.ResponseFormatJSONSchema = {  # pyright: ignore[reportPrivateImportUsage]
            'type': 'json_schema',
            'json_schema': {'name': o.name or DEFAULT_OUTPUT_TOOL_NAME, 'schema': o.json_schema},
        }
        if o.description:
            response_format_param['json_schema']['description'] = o.description
        if OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:  # pragma: no branch
            response_format_param['json_schema']['strict'] = o.strict
        return response_format_param

    def _map_tool_definition(self, f: ToolDefinition) -> chat.ChatCompletionToolParam:
        tool_param: chat.ChatCompletionToolParam = {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description or '',
                'parameters': f.parameters_json_schema,
            },
        }
        if f.strict and OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:
            tool_param['function']['strict'] = f.strict
        return tool_param

    async def _map_user_message(self, message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                system_prompt_role = OpenAIModelProfile.from_profile(self.profile).openai_system_prompt_role
                if system_prompt_role == 'developer':
                    yield chat.ChatCompletionDeveloperMessageParam(role='developer', content=part.content)
                elif system_prompt_role == 'user':
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.content)
                else:
                    yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield await self._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)

    async def _map_user_prompt(self, part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:  # noqa: C901
        content: str | list[ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url: ImageURL = {'url': item.url}
                    if metadata := item.vendor_metadata:
                        image_url['detail'] = metadata.get('detail', 'auto')
                    if item.force_download:
                        image_content = await download_item(item, data_format='base64_uri', type_format='extension')
                        image_url['url'] = image_content['data']
                    content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, BinaryContent):
                    if self._is_text_like_media_type(item.media_type):
                        # Inline text-like binary content as a text block
                        content.append(
                            self._inline_text_file_part(
                                item.data.decode('utf-8'),
                                media_type=item.media_type,
                                identifier=item.identifier,
                            )
                        )
                    elif item.is_image:
                        image_url = ImageURL(url=item.data_uri)
                        if metadata := item.vendor_metadata:
                            image_url['detail'] = metadata.get('detail', 'auto')
                        content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                    elif item.is_audio:
                        assert item.format in ('wav', 'mp3')
                        audio = InputAudio(data=base64.b64encode(item.data).decode('utf-8'), format=item.format)
                        content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                    elif item.is_document:
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=item.data_uri,
                                    filename=f'filename.{item.format}',
                                ),
                                type='file',
                            )
                        )
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):
                    downloaded_item = await download_item(item, data_format='base64', type_format='extension')
                    assert downloaded_item['data_type'] in (
                        'wav',
                        'mp3',
                    ), f'Unsupported audio format: {downloaded_item["data_type"]}'
                    audio = InputAudio(data=downloaded_item['data'], format=downloaded_item['data_type'])
                    content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                elif isinstance(item, DocumentUrl):
                    if self._is_text_like_media_type(item.media_type):
                        downloaded_text = await download_item(item, data_format='text')
                        content.append(
                            self._inline_text_file_part(
                                downloaded_text['data'],
                                media_type=item.media_type,
                                identifier=item.identifier,
                            )
                        )
                    else:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=downloaded_item['data'],
                                    filename=f'filename.{downloaded_item["data_type"]}',
                                ),
                                type='file',
                            )
                        )
                elif isinstance(item, VideoUrl):  # pragma: no cover
                    raise NotImplementedError('VideoUrl is not supported for OpenAI')
                elif isinstance(item, CachePoint):
                    # OpenAI doesn't support prompt caching via CachePoint, so we filter it out
                    pass
                else:
                    assert_never(item)
        return chat.ChatCompletionUserMessageParam(role='user', content=content)

    @staticmethod
    def _is_text_like_media_type(media_type: str) -> bool:
        return (
            media_type.startswith('text/')
            or media_type == 'application/json'
            or media_type.endswith('+json')
            or media_type == 'application/xml'
            or media_type.endswith('+xml')
            or media_type in ('application/x-yaml', 'application/yaml')
        )

    @staticmethod
    def _inline_text_file_part(text: str, *, media_type: str, identifier: str) -> ChatCompletionContentPartTextParam:
        text = '\n'.join(
            [
                f'-----BEGIN FILE id="{identifier}" type="{media_type}"-----',
                text,
                f'-----END FILE id="{identifier}"-----',
            ]
        )
        return ChatCompletionContentPartTextParam(text=text, type='text')

```

#### __init__

```python
__init__(
    model_name: OpenAIModelName,
    *,
    provider: (
        Literal[
            "azure",
            "deepseek",
            "cerebras",
            "fireworks",
            "github",
            "grok",
            "heroku",
            "moonshotai",
            "ollama",
            "openai",
            "openai-chat",
            "openrouter",
            "together",
            "vercel",
            "litellm",
            "nebius",
            "ovhcloud",
            "gateway",
        ]
        | Provider[AsyncOpenAI]
    ) = "openai",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
) -> None

```

```python
__init__(
    model_name: OpenAIModelName,
    *,
    provider: (
        Literal[
            "azure",
            "deepseek",
            "cerebras",
            "fireworks",
            "github",
            "grok",
            "heroku",
            "moonshotai",
            "ollama",
            "openai",
            "openai-chat",
            "openrouter",
            "together",
            "vercel",
            "litellm",
            "nebius",
            "ovhcloud",
            "gateway",
        ]
        | Provider[AsyncOpenAI]
    ) = "openai",
    profile: ModelProfileSpec | None = None,
    system_prompt_role: (
        OpenAISystemPromptRole | None
    ) = None,
    settings: ModelSettings | None = None
) -> None

```

```python
__init__(
    model_name: OpenAIModelName,
    *,
    provider: (
        Literal[
            "azure",
            "deepseek",
            "cerebras",
            "fireworks",
            "github",
            "grok",
            "heroku",
            "moonshotai",
            "ollama",
            "openai",
            "openai-chat",
            "openrouter",
            "together",
            "vercel",
            "litellm",
            "nebius",
            "ovhcloud",
            "gateway",
        ]
        | Provider[AsyncOpenAI]
    ) = "openai",
    profile: ModelProfileSpec | None = None,
    system_prompt_role: (
        OpenAISystemPromptRole | None
    ) = None,
    settings: ModelSettings | None = None
)

```

Initialize an OpenAI model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model_name` | `OpenAIModelName` | The name of the OpenAI model to use. List of model names available here (Unfortunately, despite being ask to do so, OpenAI do not provide .inv files for their API). | *required* | | `provider` | `Literal['azure', 'deepseek', 'cerebras', 'fireworks', 'github', 'grok', 'heroku', 'moonshotai', 'ollama', 'openai', 'openai-chat', 'openrouter', 'together', 'vercel', 'litellm', 'nebius', 'ovhcloud', 'gateway'] | Provider[AsyncOpenAI]` | The provider to use. Defaults to 'openai'. | `'openai'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider based on the model name. | `None` | | `system_prompt_role` | `OpenAISystemPromptRole | None` | The role to use for the system prompt message. If not provided, defaults to 'system'. In the future, this may be inferred from the model name. | `None` | | `settings` | `ModelSettings | None` | Default model settings for this model instance. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`

```python
def __init__(
    self,
    model_name: OpenAIModelName,
    *,
    provider: Literal[
        'azure',
        'deepseek',
        'cerebras',
        'fireworks',
        'github',
        'grok',
        'heroku',
        'moonshotai',
        'ollama',
        'openai',
        'openai-chat',
        'openrouter',
        'together',
        'vercel',
        'litellm',
        'nebius',
        'ovhcloud',
        'gateway',
    ]
    | Provider[AsyncOpenAI] = 'openai',
    profile: ModelProfileSpec | None = None,
    system_prompt_role: OpenAISystemPromptRole | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize an OpenAI model.

    Args:
        model_name: The name of the OpenAI model to use. List of model names available
            [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
            (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
        provider: The provider to use. Defaults to `'openai'`.
        profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
        system_prompt_role: The role to use for the system prompt message. If not provided, defaults to `'system'`.
            In the future, this may be inferred from the model name.
        settings: Default model settings for this model instance.
    """
    self._model_name = model_name

    if isinstance(provider, str):
        provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
    self._provider = provider
    self.client = provider.client

    super().__init__(settings=settings, profile=profile or provider.model_profile)

    if system_prompt_role is not None:
        self.profile = OpenAIModelProfile(openai_system_prompt_role=system_prompt_role).update(self.profile)

```

#### model_name

```python
model_name: OpenAIModelName

```

The model name.

#### system

```python
system: str

```

The model provider.

### OpenAIModel

Bases: `OpenAIChatModel`

Deprecated

`OpenAIModel` was renamed to `OpenAIChatModel` to clearly distinguish it from `OpenAIResponsesModel` which uses OpenAI's newer Responses API. Use that unless you're using an OpenAI Chat Completions-compatible API, or require a feature that the Responses API doesn't support yet like audio.

Deprecated alias for `OpenAIChatModel`.

Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`

```python
@deprecated(
    '`OpenAIModel` was renamed to `OpenAIChatModel` to clearly distinguish it from `OpenAIResponsesModel` which '
    "uses OpenAI's newer Responses API. Use that unless you're using an OpenAI Chat Completions-compatible API, or "
    "require a feature that the Responses API doesn't support yet like audio."
)
@dataclass(init=False)
class OpenAIModel(OpenAIChatModel):
    """Deprecated alias for `OpenAIChatModel`."""

```

### OpenAIResponsesModel

Bases: `Model`

A model that uses the OpenAI Responses API.

The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) is the new API for OpenAI models.

If you are interested in the differences between the Responses API and the Chat Completions API, see the [OpenAI API docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions).

Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`

```python
@dataclass(init=False)
class OpenAIResponsesModel(Model):
    """A model that uses the OpenAI Responses API.

    The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) is the
    new API for OpenAI models.

    If you are interested in the differences between the Responses API and the Chat Completions API,
    see the [OpenAI API docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions).
    """

    client: AsyncOpenAI = field(repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal[
            'openai',
            'deepseek',
            'azure',
            'openrouter',
            'grok',
            'fireworks',
            'together',
            'nebius',
            'ovhcloud',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Responses model.

        Args:
            model_name: The name of the OpenAI model to use.
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._responses_create(
            messages, False, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        return self._process_response(response, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._responses_create(
            messages, True, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    def _process_response(  # noqa: C901
        self, response: responses.Response, model_request_parameters: ModelRequestParameters
    ) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = number_to_datetime(response.created_at)
        items: list[ModelResponsePart] = []
        for item in response.output:
            if isinstance(item, responses.ResponseReasoningItem):
                signature = item.encrypted_content
                if item.summary:
                    for summary in item.summary:
                        # We use the same id for all summaries so that we can merge them on the round trip.
                        items.append(
                            ThinkingPart(
                                content=summary.text,
                                id=item.id,
                                signature=signature,
                                provider_name=self.system if signature else None,
                            )
                        )
                        # We only need to store the signature once.
                        signature = None
                elif signature:
                    items.append(
                        ThinkingPart(
                            content='',
                            id=item.id,
                            signature=signature,
                            provider_name=self.system,
                        )
                    )
                # NOTE: We don't currently handle the raw CoT from gpt-oss `reasoning_text`: https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot
                # If you need this, please file an issue.
            elif isinstance(item, responses.ResponseOutputMessage):
                for content in item.content:
                    if isinstance(content, responses.ResponseOutputText):  # pragma: no branch
                        items.append(TextPart(content.text, id=item.id))
            elif isinstance(item, responses.ResponseFunctionToolCall):
                items.append(
                    ToolCallPart(
                        item.name,
                        item.arguments,
                        tool_call_id=item.call_id,
                        id=item.id,
                    )
                )
            elif isinstance(item, responses.ResponseCodeInterpreterToolCall):
                call_part, return_part, file_parts = _map_code_interpreter_tool_call(item, self.system)
                items.append(call_part)
                if file_parts:
                    items.extend(file_parts)
                items.append(return_part)
            elif isinstance(item, responses.ResponseFunctionWebSearch):
                call_part, return_part = _map_web_search_tool_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.ImageGenerationCall):
                call_part, return_part, file_part = _map_image_generation_tool_call(item, self.system)
                items.append(call_part)
                if file_part:  # pragma: no branch
                    items.append(file_part)
                items.append(return_part)
            elif isinstance(item, responses.ResponseComputerToolCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the ComputerUse built-in tool
                pass
            elif isinstance(item, responses.ResponseCustomToolCall):  # pragma: no cover
                # Support is being implemented in https://github.com/pydantic/pydantic-ai/pull/2572
                pass
            elif isinstance(item, responses.response_output_item.LocalShellCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the `codex-mini-latest` LocalShell built-in tool
                pass
            elif isinstance(item, responses.ResponseFileSearchToolCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the FileSearch built-in tool
                pass
            elif isinstance(item, responses.response_output_item.McpCall):
                call_part, return_part = _map_mcp_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpListTools):
                call_part, return_part = _map_mcp_list_tools(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpApprovalRequest):  # pragma: no cover
                # Pydantic AI doesn't yet support McpApprovalRequest (explicit tool usage approval)
                pass

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] | None = None
        raw_finish_reason = details.reason if (details := response.incomplete_details) else response.status
        if raw_finish_reason:
            provider_details = {'finish_reason': raw_finish_reason}
            finish_reason = _RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response, self._provider.name, self._provider.base_url, self._model_name),
            model_name=response.model,
            provider_response_id=response.id,
            timestamp=timestamp,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self,
        response: AsyncStream[responses.ResponseStreamEvent],
        model_request_parameters: ModelRequestParameters,
    ) -> OpenAIResponsesStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        assert isinstance(first_chunk, responses.ResponseCreatedEvent)
        return OpenAIResponsesStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.response.model,
            _response=peekable_response,
            _timestamp=number_to_datetime(first_chunk.response.created_at),
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
        )

    @overload
    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[False],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response: ...

    @overload
    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[True],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[responses.ResponseStreamEvent]: ...

    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent]:
        tools = (
            self._get_builtin_tools(model_request_parameters)
            + list(model_settings.get('openai_builtin_tools', []))
            + self._get_tools(model_request_parameters)
        )
        profile = OpenAIModelProfile.from_profile(self.profile)
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output and profile.openai_supports_tool_choice_required:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        previous_response_id = model_settings.get('openai_previous_response_id')
        if previous_response_id == 'auto':
            previous_response_id, messages = self._get_previous_response_id_and_new_messages(messages)

        instructions, openai_messages = await self._map_messages(messages, model_settings, model_request_parameters)
        reasoning = self._get_reasoning(model_settings)

        text: responses.ResponseTextConfigParam | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            text = {'format': self._map_json_schema(output_object)}
        elif (
            model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output
        ):  # pragma: no branch
            text = {'format': {'type': 'json_object'}}

            # Without this trick, we'd hit this error:
            # > Response input messages must contain the word 'json' in some form to use 'text.format' of type 'json_object'.
            # Apparently they're only checking input messages for "JSON", not instructions.
            assert isinstance(instructions, str)
            openai_messages.insert(0, responses.EasyInputMessageParam(role='system', content=instructions))
            instructions = OMIT

        if verbosity := model_settings.get('openai_text_verbosity'):
            text = text or {}
            text['verbosity'] = verbosity

        unsupported_model_settings = profile.openai_unsupported_model_settings
        for setting in unsupported_model_settings:
            model_settings.pop(setting, None)

        include: list[responses.ResponseIncludable] = []
        if profile.openai_supports_encrypted_reasoning_content:
            include.append('reasoning.encrypted_content')
        if model_settings.get('openai_include_code_execution_outputs'):
            include.append('code_interpreter_call.outputs')
        if model_settings.get('openai_include_web_search_sources'):
            include.append('web_search_call.action.sources')

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.responses.create(
                input=openai_messages,
                model=self._model_name,
                instructions=instructions,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', OMIT),
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                max_output_tokens=model_settings.get('max_tokens', OMIT),
                stream=stream,
                temperature=model_settings.get('temperature', OMIT),
                top_p=model_settings.get('top_p', OMIT),
                truncation=model_settings.get('openai_truncation', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                service_tier=model_settings.get('openai_service_tier', OMIT),
                previous_response_id=previous_response_id or OMIT,
                reasoning=reasoning,
                user=model_settings.get('openai_user', OMIT),
                text=text or OMIT,
                include=include or OMIT,
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

    def _get_reasoning(self, model_settings: OpenAIResponsesModelSettings) -> Reasoning | Omit:
        reasoning_effort = model_settings.get('openai_reasoning_effort', None)
        reasoning_summary = model_settings.get('openai_reasoning_summary', None)
        reasoning_generate_summary = model_settings.get('openai_reasoning_generate_summary', None)

        if reasoning_summary and reasoning_generate_summary:  # pragma: no cover
            raise ValueError('`openai_reasoning_summary` and `openai_reasoning_generate_summary` cannot both be set.')

        if reasoning_generate_summary is not None:  # pragma: no cover
            warnings.warn(
                '`openai_reasoning_generate_summary` is deprecated, use `openai_reasoning_summary` instead',
                DeprecationWarning,
            )
            reasoning_summary = reasoning_generate_summary

        if reasoning_effort is None and reasoning_summary is None:
            return OMIT
        return Reasoning(effort=reasoning_effort, summary=reasoning_summary)

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.FunctionToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_builtin_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.ToolParam]:
        tools: list[responses.ToolParam] = []
        has_image_generating_tool = False
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                web_search_tool = responses.WebSearchToolParam(
                    type='web_search', search_context_size=tool.search_context_size
                )
                if tool.user_location:
                    web_search_tool['user_location'] = responses.web_search_tool_param.UserLocation(
                        type='approximate', **tool.user_location
                    )
                tools.append(web_search_tool)
            elif isinstance(tool, CodeExecutionTool):
                has_image_generating_tool = True
                tools.append({'type': 'code_interpreter', 'container': {'type': 'auto'}})
            elif isinstance(tool, MCPServerTool):
                mcp_tool = responses.tool_param.Mcp(
                    type='mcp',
                    server_label=tool.id,
                    require_approval='never',
                )

                if tool.authorization_token:  # pragma: no branch
                    mcp_tool['authorization'] = tool.authorization_token

                if tool.allowed_tools is not None:  # pragma: no branch
                    mcp_tool['allowed_tools'] = tool.allowed_tools

                if tool.description:  # pragma: no branch
                    mcp_tool['server_description'] = tool.description

                if tool.headers:  # pragma: no branch
                    mcp_tool['headers'] = tool.headers

                if tool.url.startswith(MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME + ':'):
                    _, connector_id = tool.url.split(':', maxsplit=1)
                    mcp_tool['connector_id'] = connector_id  # pyright: ignore[reportGeneralTypeIssues]
                else:
                    mcp_tool['server_url'] = tool.url

                tools.append(mcp_tool)
            elif isinstance(tool, ImageGenerationTool):  # pragma: no branch
                has_image_generating_tool = True
                tools.append(
                    responses.tool_param.ImageGeneration(
                        type='image_generation',
                        background=tool.background,
                        input_fidelity=tool.input_fidelity,
                        moderation=tool.moderation,
                        output_compression=tool.output_compression,
                        output_format=tool.output_format or 'png',
                        partial_images=tool.partial_images,
                        quality=tool.quality,
                        size=tool.size,
                    )
                )
            else:
                raise UserError(  # pragma: no cover
                    f'`{tool.__class__.__name__}` is not supported by `OpenAIResponsesModel`. If it should be, please file an issue.'
                )

        if model_request_parameters.allow_image_output and not has_image_generating_tool:
            tools.append({'type': 'image_generation'})
        return tools

    def _map_tool_definition(self, f: ToolDefinition) -> responses.FunctionToolParam:
        return {
            'name': f.name,
            'parameters': f.parameters_json_schema,
            'type': 'function',
            'description': f.description,
            'strict': bool(
                f.strict and OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition
            ),
        }

    def _get_previous_response_id_and_new_messages(
        self, messages: list[ModelMessage]
    ) -> tuple[str | None, list[ModelMessage]]:
        # When `openai_previous_response_id` is set to 'auto', the most recent
        # `provider_response_id` from the message history is selected and all
        # earlier messages are omitted. This allows the OpenAI SDK to reuse
        # server-side history for efficiency. The returned tuple contains the
        # `previous_response_id` (if found) and the trimmed list of messages.
        previous_response_id = None
        trimmed_messages: list[ModelMessage] = []
        for m in reversed(messages):
            if isinstance(m, ModelResponse) and m.provider_name == self.system:
                previous_response_id = m.provider_response_id
                break
            else:
                trimmed_messages.append(m)

        if previous_response_id and trimmed_messages:
            return previous_response_id, list(reversed(trimmed_messages))
        else:
            return None, messages

    async def _map_messages(  # noqa: C901
        self,
        messages: list[ModelMessage],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[str | Omit, list[responses.ResponseInputItemParam]]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.responses.ResponseInputParam`."""
        profile = OpenAIModelProfile.from_profile(self.profile)
        send_item_ids = model_settings.get(
            'openai_send_reasoning_ids', profile.openai_supports_encrypted_reasoning_content
        )

        openai_messages: list[responses.ResponseInputItemParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        openai_messages.append(responses.EasyInputMessageParam(role='system', content=part.content))
                    elif isinstance(part, UserPromptPart):
                        openai_messages.append(await self._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        call_id = _guard_tool_call_id(t=part)
                        call_id, _ = _split_combined_tool_call_id(call_id)
                        item = FunctionCallOutput(
                            type='function_call_output',
                            call_id=call_id,
                            output=part.model_response_str(),
                        )
                        openai_messages.append(item)
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            openai_messages.append(
                                Message(role='user', content=[{'type': 'input_text', 'text': part.model_response()}])
                            )
                        else:
                            call_id = _guard_tool_call_id(t=part)
                            call_id, _ = _split_combined_tool_call_id(call_id)
                            item = FunctionCallOutput(
                                type='function_call_output',
                                call_id=call_id,
                                output=part.model_response(),
                            )
                            openai_messages.append(item)
                    else:
                        assert_never(part)
            elif isinstance(message, ModelResponse):
                send_item_ids = send_item_ids and message.provider_name == self.system

                message_item: responses.ResponseOutputMessageParam | None = None
                reasoning_item: responses.ResponseReasoningItemParam | None = None
                web_search_item: responses.ResponseFunctionWebSearchParam | None = None
                code_interpreter_item: responses.ResponseCodeInterpreterToolCallParam | None = None
                for item in message.parts:
                    if isinstance(item, TextPart):
                        if item.id and send_item_ids:
                            if message_item is None or message_item['id'] != item.id:  # pragma: no branch
                                message_item = responses.ResponseOutputMessageParam(
                                    role='assistant',
                                    id=item.id,
                                    content=[],
                                    type='message',
                                    status='completed',
                                )
                                openai_messages.append(message_item)

                            message_item['content'] = [
                                *message_item['content'],
                                responses.ResponseOutputTextParam(
                                    text=item.content, type='output_text', annotations=[]
                                ),
                            ]
                        else:
                            openai_messages.append(
                                responses.EasyInputMessageParam(role='assistant', content=item.content)
                            )
                    elif isinstance(item, ToolCallPart):
                        call_id = _guard_tool_call_id(t=item)
                        call_id, id = _split_combined_tool_call_id(call_id)
                        id = id or item.id

                        param = responses.ResponseFunctionToolCallParam(
                            name=item.tool_name,
                            arguments=item.args_as_json_str(),
                            call_id=call_id,
                            type='function_call',
                        )
                        if profile.openai_responses_requires_function_call_status_none:
                            param['status'] = None  # type: ignore[reportGeneralTypeIssues]
                        if id and send_item_ids:  # pragma: no branch
                            param['id'] = id
                        openai_messages.append(param)
                    elif isinstance(item, BuiltinToolCallPart):
                        if item.provider_name == self.system and send_item_ids:
                            if (
                                item.tool_name == CodeExecutionTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                                and (container_id := args.get('container_id'))
                            ):
                                code_interpreter_item = responses.ResponseCodeInterpreterToolCallParam(
                                    id=item.tool_call_id,
                                    code=args.get('code'),
                                    container_id=container_id,
                                    outputs=None,  # These can be read server-side
                                    status='completed',
                                    type='code_interpreter_call',
                                )
                                openai_messages.append(code_interpreter_item)
                            elif (
                                item.tool_name == WebSearchTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                            ):
                                web_search_item = responses.ResponseFunctionWebSearchParam(
                                    id=item.tool_call_id,
                                    action=cast(responses.response_function_web_search_param.Action, args),
                                    status='completed',
                                    type='web_search_call',
                                )
                                openai_messages.append(web_search_item)
                            elif item.tool_name == ImageGenerationTool.kind and item.tool_call_id:
                                # The cast is necessary because of https://github.com/openai/openai-python/issues/2648
                                image_generation_item = cast(
                                    responses.response_input_item_param.ImageGenerationCall,
                                    {
                                        'id': item.tool_call_id,
                                        'type': 'image_generation_call',
                                    },
                                )
                                openai_messages.append(image_generation_item)
                            elif (  # pragma: no branch
                                item.tool_name.startswith(MCPServerTool.kind)
                                and item.tool_call_id
                                and (server_id := item.tool_name.split(':', 1)[1])
                                and (args := item.args_as_dict())
                                and (action := args.get('action'))
                            ):
                                if action == 'list_tools':
                                    mcp_list_tools_item = responses.response_input_item_param.McpListTools(
                                        id=item.tool_call_id,
                                        type='mcp_list_tools',
                                        server_label=server_id,
                                        tools=[],  # These can be read server-side
                                    )
                                    openai_messages.append(mcp_list_tools_item)
                                elif (  # pragma: no branch
                                    action == 'call_tool'
                                    and (tool_name := args.get('tool_name'))
                                    and (tool_args := args.get('tool_args'))
                                ):
                                    mcp_call_item = responses.response_input_item_param.McpCall(
                                        id=item.tool_call_id,
                                        server_label=server_id,
                                        name=tool_name,
                                        arguments=to_json(tool_args).decode(),
                                        error=None,  # These can be read server-side
                                        output=None,  # These can be read server-side
                                        type='mcp_call',
                                    )
                                    openai_messages.append(mcp_call_item)

                    elif isinstance(item, BuiltinToolReturnPart):
                        if item.provider_name == self.system and send_item_ids:
                            if (
                                item.tool_name == CodeExecutionTool.kind
                                and code_interpreter_item is not None
                                and isinstance(item.content, dict)
                                and (content := cast(dict[str, Any], item.content))  # pyright: ignore[reportUnknownMemberType]
                                and (status := content.get('status'))
                            ):
                                code_interpreter_item['status'] = status
                            elif (
                                item.tool_name == WebSearchTool.kind
                                and web_search_item is not None
                                and isinstance(item.content, dict)  # pyright: ignore[reportUnknownMemberType]
                                and (content := cast(dict[str, Any], item.content))  # pyright: ignore[reportUnknownMemberType]
                                and (status := content.get('status'))
                            ):
                                web_search_item['status'] = status
                            elif item.tool_name == ImageGenerationTool.kind:
                                # Image generation result does not need to be sent back, just the `id` off of `BuiltinToolCallPart`.
                                pass
                            elif item.tool_name.startswith(MCPServerTool.kind):  # pragma: no branch
                                # MCP call result does not need to be sent back, just the fields off of `BuiltinToolCallPart`.
                                pass
                    elif isinstance(item, FilePart):
                        # This was generated by the `ImageGenerationTool` or `CodeExecutionTool`,
                        # and does not need to be sent back separately from the corresponding `BuiltinToolReturnPart`.
                        # If `send_item_ids` is false, we won't send the `BuiltinToolReturnPart`, but OpenAI does not have a type for files from the assistant.
                        pass
                    elif isinstance(item, ThinkingPart):
                        if item.id and send_item_ids:
                            signature: str | None = None
                            if (
                                item.signature
                                and item.provider_name == self.system
                                and profile.openai_supports_encrypted_reasoning_content
                            ):
                                signature = item.signature

                            if (reasoning_item is None or reasoning_item['id'] != item.id) and (
                                signature or item.content
                            ):  # pragma: no branch
                                reasoning_item = responses.ResponseReasoningItemParam(
                                    id=item.id,
                                    summary=[],
                                    encrypted_content=signature,
                                    type='reasoning',
                                )
                                openai_messages.append(reasoning_item)

                            if item.content:
                                # The check above guarantees that `reasoning_item` is not None
                                assert reasoning_item is not None
                                reasoning_item['summary'] = [
                                    *reasoning_item['summary'],
                                    Summary(text=item.content, type='summary_text'),
                                ]
                        else:
                            start_tag, end_tag = profile.thinking_tags
                            openai_messages.append(
                                responses.EasyInputMessageParam(
                                    role='assistant', content='\n'.join([start_tag, item.content, end_tag])
                                )
                            )
                    else:
                        assert_never(item)
            else:
                assert_never(message)
        instructions = self._get_instructions(messages, model_request_parameters) or OMIT
        return instructions, openai_messages

    def _map_json_schema(self, o: OutputObjectDefinition) -> responses.ResponseFormatTextJSONSchemaConfigParam:
        response_format_param: responses.ResponseFormatTextJSONSchemaConfigParam = {
            'type': 'json_schema',
            'name': o.name or DEFAULT_OUTPUT_TOOL_NAME,
            'schema': o.json_schema,
        }
        if o.description:
            response_format_param['description'] = o.description
        if OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:  # pragma: no branch
            response_format_param['strict'] = o.strict
        return response_format_param

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> responses.EasyInputMessageParam:  # noqa: C901
        content: str | list[responses.ResponseInputContentParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(responses.ResponseInputTextParam(text=item, type='input_text'))
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        detail: Literal['auto', 'low', 'high'] = 'auto'
                        if metadata := item.vendor_metadata:
                            detail = cast(
                                Literal['auto', 'low', 'high'],
                                metadata.get('detail', 'auto'),
                            )
                        content.append(
                            responses.ResponseInputImageParam(
                                image_url=item.data_uri,
                                type='input_image',
                                detail=detail,
                            )
                        )
                    elif item.is_document:
                        content.append(
                            responses.ResponseInputFileParam(
                                type='input_file',
                                file_data=item.data_uri,
                                # NOTE: Type wise it's not necessary to include the filename, but it's required by the
                                # API itself. If we add empty string, the server sends a 500 error - which OpenAI needs
                                # to fix. In any case, we add a placeholder name.
                                filename=f'filename.{item.format}',
                            )
                        )
                    elif item.is_audio:
                        raise NotImplementedError('Audio as binary content is not supported for OpenAI Responses API.')
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, ImageUrl):
                    detail: Literal['auto', 'low', 'high'] = 'auto'
                    image_url = item.url
                    if metadata := item.vendor_metadata:
                        detail = cast(Literal['auto', 'low', 'high'], metadata.get('detail', 'auto'))
                    if item.force_download:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        image_url = downloaded_item['data']

                    content.append(
                        responses.ResponseInputImageParam(
                            image_url=image_url,
                            type='input_image',
                            detail=detail,
                        )
                    )
                elif isinstance(item, AudioUrl):  # pragma: no cover
                    downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                    content.append(
                        responses.ResponseInputFileParam(
                            type='input_file',
                            file_data=downloaded_item['data'],
                            filename=f'filename.{downloaded_item["data_type"]}',
                        )
                    )
                elif isinstance(item, DocumentUrl):
                    downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                    content.append(
                        responses.ResponseInputFileParam(
                            type='input_file',
                            file_data=downloaded_item['data'],
                            filename=f'filename.{downloaded_item["data_type"]}',
                        )
                    )
                elif isinstance(item, VideoUrl):  # pragma: no cover
                    raise NotImplementedError('VideoUrl is not supported for OpenAI.')
                elif isinstance(item, CachePoint):
                    # OpenAI doesn't support prompt caching via CachePoint, so we filter it out
                    pass
                else:
                    assert_never(item)
        return responses.EasyInputMessageParam(role='user', content=content)

```

#### __init__

```python
__init__(
    model_name: OpenAIModelName,
    *,
    provider: (
        Literal[
            "openai",
            "deepseek",
            "azure",
            "openrouter",
            "grok",
            "fireworks",
            "together",
            "nebius",
            "ovhcloud",
            "gateway",
        ]
        | Provider[AsyncOpenAI]
    ) = "openai",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize an OpenAI Responses model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model_name` | `OpenAIModelName` | The name of the OpenAI model to use. | *required* | | `provider` | `Literal['openai', 'deepseek', 'azure', 'openrouter', 'grok', 'fireworks', 'together', 'nebius', 'ovhcloud', 'gateway'] | Provider[AsyncOpenAI]` | The provider to use. Defaults to 'openai'. | `'openai'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider based on the model name. | `None` | | `settings` | `ModelSettings | None` | Default model settings for this model instance. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`

```python
def __init__(
    self,
    model_name: OpenAIModelName,
    *,
    provider: Literal[
        'openai',
        'deepseek',
        'azure',
        'openrouter',
        'grok',
        'fireworks',
        'together',
        'nebius',
        'ovhcloud',
        'gateway',
    ]
    | Provider[AsyncOpenAI] = 'openai',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize an OpenAI Responses model.

    Args:
        model_name: The name of the OpenAI model to use.
        provider: The provider to use. Defaults to `'openai'`.
        profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
        settings: Default model settings for this model instance.
    """
    self._model_name = model_name

    if isinstance(provider, str):
        provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
    self._provider = provider
    self.client = provider.client

    super().__init__(settings=settings, profile=profile or provider.model_profile)

```

#### model_name

```python
model_name: OpenAIModelName

```

The model name.

#### system

```python
system: str

```

The model provider.

# `pydantic_ai.models.outlines`

## Setup

For details on how to set up this model, see [model configuration for Outlines](../../../models/outlines/).

### OutlinesModel

Bases: `Model`

A model that relies on the Outlines library to run non API-based models.

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
@dataclass(init=False)
class OutlinesModel(Model):
    """A model that relies on the Outlines library to run non API-based models."""

    def __init__(
        self,
        model: OutlinesBaseModel | OutlinesAsyncBaseModel,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Outlines model.

        Args:
            model: The Outlines model used for the model.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        self.model: OutlinesBaseModel | OutlinesAsyncBaseModel = model
        self._model_name: str = 'outlines-model'

        if isinstance(provider, str):
            provider = infer_provider(provider)

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @classmethod
    def from_transformers(
        cls,
        hf_model: transformers.modeling_utils.PreTrainedModel,
        hf_tokenizer_or_processor: transformers.tokenization_utils.PreTrainedTokenizer
        | transformers.processing_utils.ProcessorMixin,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a Hugging Face model and tokenizer.

        Args:
            hf_model: The Hugging Face PreTrainedModel or any model that is compatible with the
                `transformers` API.
            hf_tokenizer_or_processor: Either a HuggingFace `PreTrainedTokenizer` or any tokenizer that is compatible
                with the `transformers` API, or a HuggingFace processor inheriting from `ProcessorMixin`. If a
                tokenizer is provided, a regular model will be used, while if you provide a processor, it will be a
                multimodal model.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel = from_transformers(hf_model, hf_tokenizer_or_processor)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @classmethod
    def from_llamacpp(
        cls,
        llama_model: llama_cpp.Llama,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a LlamaCpp model.

        Args:
            llama_model: The llama_cpp.Llama model to use.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel = from_llamacpp(llama_model)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @classmethod
    def from_mlxlm(  # pragma: no cover
        cls,
        mlx_model: nn.Module,
        mlx_tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a MLXLM model.

        Args:
            mlx_model: The nn.Module model to use.
            mlx_tokenizer: The PreTrainedTokenizer to use.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel = from_mlxlm(mlx_model, mlx_tokenizer)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @classmethod
    def from_sglang(
        cls,
        base_url: str,
        api_key: str | None = None,
        model_name: str | None = None,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model to send requests to an SGLang server.

        Args:
            base_url: The url of the SGLang server.
            api_key: The API key to use for authenticating requests to the SGLang server.
            model_name: The name of the model to use.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as _import_error:
            raise ImportError(
                'Please install `openai` to use the Outlines SGLang model, '
                'you can use the `openai` optional group â€” `pip install "pydantic-ai-slim[openai]"`'
            ) from _import_error

        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_sglang(openai_client, model_name)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @classmethod
    def from_vllm_offline(  # pragma: no cover
        cls,
        vllm_model: Any,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a vLLM offline inference model.

        Args:
            vllm_model: The vllm.LLM local model to use.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_vllm_offline(vllm_model)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return 'outlines'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        """Make a request to the model."""
        prompt, output_type, inference_kwargs = await self._build_generation_arguments(
            messages, model_settings, model_request_parameters
        )
        # Async is available for SgLang
        response: str
        if isinstance(self.model, OutlinesAsyncBaseModel):
            response = await self.model(prompt, output_type, None, **inference_kwargs)
        else:
            response = self.model(prompt, output_type, None, **inference_kwargs)
        return self._process_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )

        prompt, output_type, inference_kwargs = await self._build_generation_arguments(
            messages, model_settings, model_request_parameters
        )
        # Async is available for SgLang
        if isinstance(self.model, OutlinesAsyncBaseModel):
            response = self.model.stream(prompt, output_type, None, **inference_kwargs)
            yield await self._process_streamed_response(response, model_request_parameters)
        else:
            response = self.model.stream(prompt, output_type, None, **inference_kwargs)

            async def async_response():
                for chunk in response:
                    yield chunk

            yield await self._process_streamed_response(async_response(), model_request_parameters)

    async def _build_generation_arguments(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[Chat, JsonSchema | None, dict[str, Any]]:
        """Build the generation arguments for the model."""
        if (
            model_request_parameters.function_tools
            or model_request_parameters.builtin_tools
            or model_request_parameters.output_tools
        ):
            raise UserError('Outlines does not support function tools and builtin tools yet.')

        if model_request_parameters.output_object:
            output_type = JsonSchema(model_request_parameters.output_object.json_schema)
        else:
            output_type = None

        prompt = await self._format_prompt(messages, model_request_parameters)
        inference_kwargs = self.format_inference_kwargs(model_settings)

        return prompt, output_type, inference_kwargs

    def format_inference_kwargs(self, model_settings: ModelSettings | None) -> dict[str, Any]:
        """Format the model settings for the inference kwargs."""
        settings_dict: dict[str, Any] = dict(model_settings) if model_settings else {}

        if isinstance(self.model, Transformers):
            settings_dict = self._format_transformers_inference_kwargs(settings_dict)
        elif isinstance(self.model, LlamaCpp):
            settings_dict = self._format_llama_cpp_inference_kwargs(settings_dict)
        elif isinstance(self.model, MLXLM):  # pragma: no cover
            settings_dict = self._format_mlxlm_inference_kwargs(settings_dict)
        elif isinstance(self.model, SGLang | AsyncSGLang):
            settings_dict = self._format_sglang_inference_kwargs(settings_dict)
        elif isinstance(self.model, VLLMOffline):  # pragma: no cover
            settings_dict = self._format_vllm_offline_inference_kwargs(settings_dict)

        extra_body = settings_dict.pop('extra_body', {})
        settings_dict.update(extra_body)

        return settings_dict

    def _format_transformers_inference_kwargs(self, model_settings: dict[str, Any]) -> dict[str, Any]:
        """Select the model settings supported by the Transformers model."""
        supported_args = [
            'max_tokens',
            'temperature',
            'top_p',
            'logit_bias',
            'extra_body',
        ]
        filtered_settings = {k: model_settings[k] for k in supported_args if k in model_settings}

        return filtered_settings

    def _format_llama_cpp_inference_kwargs(self, model_settings: dict[str, Any]) -> dict[str, Any]:
        """Select the model settings supported by the LlamaCpp model."""
        supported_args = [
            'max_tokens',
            'temperature',
            'top_p',
            'seed',
            'presence_penalty',
            'frequency_penalty',
            'logit_bias',
            'extra_body',
        ]
        filtered_settings = {k: model_settings[k] for k in supported_args if k in model_settings}

        return filtered_settings

    def _format_mlxlm_inference_kwargs(  # pragma: no cover
        self, model_settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Select the model settings supported by the MLXLM model."""
        supported_args = [
            'extra_body',
        ]
        filtered_settings = {k: model_settings[k] for k in supported_args if k in model_settings}

        return filtered_settings

    def _format_sglang_inference_kwargs(self, model_settings: dict[str, Any]) -> dict[str, Any]:
        """Select the model settings supported by the SGLang model."""
        supported_args = [
            'max_tokens',
            'temperature',
            'top_p',
            'presence_penalty',
            'frequency_penalty',
            'extra_body',
        ]
        filtered_settings = {k: model_settings[k] for k in supported_args if k in model_settings}

        return filtered_settings

    def _format_vllm_offline_inference_kwargs(  # pragma: no cover
        self, model_settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Select the model settings supported by the vLLMOffline model."""
        from vllm.sampling_params import SamplingParams

        supported_args = [
            'max_tokens',
            'temperature',
            'top_p',
            'seed',
            'presence_penalty',
            'frequency_penalty',
            'logit_bias',
            'extra_body',
        ]
        # The arguments that are part of the fields of `ModelSettings` must be put in a `SamplingParams` object and
        # provided through the `sampling_params` argument to vLLM
        sampling_params = model_settings.get('extra_body', {}).pop('sampling_params', SamplingParams())

        for key in supported_args:
            setattr(sampling_params, key, model_settings.get(key, None))

        filtered_settings = {
            'sampling_params': sampling_params,
            **model_settings.get('extra_body', {}),
        }

        return filtered_settings

    async def _format_prompt(  # noqa: C901
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> Chat:
        """Turn the model messages into an Outlines Chat instance."""
        chat = Chat()

        if instructions := self._get_instructions(messages, model_request_parameters):
            chat.add_system_message(instructions)

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        chat.add_system_message(part.content)
                    elif isinstance(part, UserPromptPart):
                        if isinstance(part.content, str):
                            chat.add_user_message(part.content)
                        elif isinstance(part.content, Sequence):
                            outlines_input: Sequence[str | Image] = []
                            for item in part.content:
                                if isinstance(item, str):
                                    outlines_input.append(item)
                                elif isinstance(item, ImageUrl):
                                    image_content: DownloadedItem[bytes] = await download_item(
                                        item, data_format='bytes', type_format='mime'
                                    )
                                    image = self._create_PIL_image(image_content['data'], image_content['data_type'])
                                    outlines_input.append(Image(image))
                                elif isinstance(item, BinaryContent) and item.is_image:
                                    image = self._create_PIL_image(item.data, item.media_type)
                                    outlines_input.append(Image(image))
                                else:
                                    raise UserError(
                                        'Each element of the content sequence must be a string, an `ImageUrl`'
                                        + ' or a `BinaryImage`.'
                                    )
                            chat.add_user_message(outlines_input)
                        else:
                            assert_never(part.content)
                    elif isinstance(part, RetryPromptPart):
                        chat.add_user_message(part.model_response())
                    elif isinstance(part, ToolReturnPart):
                        raise UserError('Tool calls are not supported for Outlines models yet.')
                    else:
                        assert_never(part)
            elif isinstance(message, ModelResponse):
                text_parts: list[str] = []
                image_parts: list[Image] = []
                for part in message.parts:
                    if isinstance(part, TextPart):
                        text_parts.append(part.content)
                    elif isinstance(part, ThinkingPart):
                        # NOTE: We don't send ThinkingPart to the providers yet.
                        pass
                    elif isinstance(part, ToolCallPart | BuiltinToolCallPart | BuiltinToolReturnPart):
                        raise UserError('Tool calls are not supported for Outlines models yet.')
                    elif isinstance(part, FilePart):
                        if isinstance(part.content, BinaryContent) and part.content.is_image:
                            image = self._create_PIL_image(part.content.data, part.content.media_type)
                            image_parts.append(Image(image))
                        else:
                            raise UserError(
                                'File parts other than `BinaryImage` are not supported for Outlines models yet.'
                            )
                    else:
                        assert_never(part)
                if len(text_parts) == 1 and len(image_parts) == 0:
                    chat.add_assistant_message(text_parts[0])
                else:
                    chat.add_assistant_message([*text_parts, *image_parts])
            else:
                assert_never(message)
        return chat

    def _create_PIL_image(self, data: bytes, data_type: str) -> PILImage.Image:
        """Create a PIL Image from the data and data type."""
        image = PILImage.open(io.BytesIO(data))
        image.format = data_type.split('/')[-1]
        return image

    def _process_response(self, response: str) -> ModelResponse:
        """Turn the Outlines text response into a Pydantic AI model response instance."""
        return ModelResponse(
            parts=cast(
                list[ModelResponsePart], split_content_into_text_and_thinking(response, self.profile.thinking_tags)
            ),
        )

    async def _process_streamed_response(
        self, response: AsyncIterable[str], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Turn the Outlines text response into a Pydantic AI streamed response instance."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        timestamp = datetime.now(tz=timezone.utc)
        return OutlinesStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _model_profile=self.profile,
            _response=peekable_response,
            _timestamp=timestamp,
            _provider_name='outlines',
        )

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        """Customize the model request parameters for the model."""
        if model_request_parameters.output_mode in ('auto', 'native'):
            # This way the JSON schema will be included in the instructions.
            return replace(model_request_parameters, output_mode='prompted')
        else:
            return model_request_parameters

```

#### __init__

```python
__init__(
    model: Model | AsyncModel,
    *,
    provider: (
        Literal["outlines"] | Provider[Model]
    ) = "outlines",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize an Outlines model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model` | `Model | AsyncModel` | The Outlines model used for the model. | *required* | | `provider` | `Literal['outlines'] | Provider[Model]` | The provider to use for OutlinesModel. Can be either the string 'outlines' or an instance of Provider[OutlinesBaseModel]. If not provided, the other parameters will be used. | `'outlines'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider. | `None` | | `settings` | `ModelSettings | None` | Default model settings for this model instance. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
def __init__(
    self,
    model: OutlinesBaseModel | OutlinesAsyncBaseModel,
    *,
    provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize an Outlines model.

    Args:
        model: The Outlines model used for the model.
        provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
            instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
        profile: The model profile to use. Defaults to a profile picked by the provider.
        settings: Default model settings for this model instance.
    """
    self.model: OutlinesBaseModel | OutlinesAsyncBaseModel = model
    self._model_name: str = 'outlines-model'

    if isinstance(provider, str):
        provider = infer_provider(provider)

    super().__init__(settings=settings, profile=profile or provider.model_profile)

```

#### from_transformers

```python
from_transformers(
    hf_model: PreTrainedModel,
    hf_tokenizer_or_processor: (
        PreTrainedTokenizer | ProcessorMixin
    ),
    *,
    provider: (
        Literal["outlines"] | Provider[Model]
    ) = "outlines",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Create an Outlines model from a Hugging Face model and tokenizer.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `hf_model` | `PreTrainedModel` | The Hugging Face PreTrainedModel or any model that is compatible with the transformers API. | *required* | | `hf_tokenizer_or_processor` | `PreTrainedTokenizer | ProcessorMixin` | Either a HuggingFace PreTrainedTokenizer or any tokenizer that is compatible with the transformers API, or a HuggingFace processor inheriting from ProcessorMixin. If a tokenizer is provided, a regular model will be used, while if you provide a processor, it will be a multimodal model. | *required* | | `provider` | `Literal['outlines'] | Provider[Model]` | The provider to use for OutlinesModel. Can be either the string 'outlines' or an instance of Provider[OutlinesBaseModel]. If not provided, the other parameters will be used. | `'outlines'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider. | `None` | | `settings` | `ModelSettings | None` | Default model settings for this model instance. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
@classmethod
def from_transformers(
    cls,
    hf_model: transformers.modeling_utils.PreTrainedModel,
    hf_tokenizer_or_processor: transformers.tokenization_utils.PreTrainedTokenizer
    | transformers.processing_utils.ProcessorMixin,
    *,
    provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Create an Outlines model from a Hugging Face model and tokenizer.

    Args:
        hf_model: The Hugging Face PreTrainedModel or any model that is compatible with the
            `transformers` API.
        hf_tokenizer_or_processor: Either a HuggingFace `PreTrainedTokenizer` or any tokenizer that is compatible
            with the `transformers` API, or a HuggingFace processor inheriting from `ProcessorMixin`. If a
            tokenizer is provided, a regular model will be used, while if you provide a processor, it will be a
            multimodal model.
        provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
            instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
        profile: The model profile to use. Defaults to a profile picked by the provider.
        settings: Default model settings for this model instance.
    """
    outlines_model: OutlinesBaseModel = from_transformers(hf_model, hf_tokenizer_or_processor)
    return cls(outlines_model, provider=provider, profile=profile, settings=settings)

```

#### from_llamacpp

```python
from_llamacpp(
    llama_model: Llama,
    *,
    provider: (
        Literal["outlines"] | Provider[Model]
    ) = "outlines",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Create an Outlines model from a LlamaCpp model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `llama_model` | `Llama` | The llama_cpp.Llama model to use. | *required* | | `provider` | `Literal['outlines'] | Provider[Model]` | The provider to use for OutlinesModel. Can be either the string 'outlines' or an instance of Provider[OutlinesBaseModel]. If not provided, the other parameters will be used. | `'outlines'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider. | `None` | | `settings` | `ModelSettings | None` | Default model settings for this model instance. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
@classmethod
def from_llamacpp(
    cls,
    llama_model: llama_cpp.Llama,
    *,
    provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Create an Outlines model from a LlamaCpp model.

    Args:
        llama_model: The llama_cpp.Llama model to use.
        provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
            instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
        profile: The model profile to use. Defaults to a profile picked by the provider.
        settings: Default model settings for this model instance.
    """
    outlines_model: OutlinesBaseModel = from_llamacpp(llama_model)
    return cls(outlines_model, provider=provider, profile=profile, settings=settings)

```

#### from_mlxlm

```python
from_mlxlm(
    mlx_model: Module,
    mlx_tokenizer: PreTrainedTokenizer,
    *,
    provider: (
        Literal["outlines"] | Provider[Model]
    ) = "outlines",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Create an Outlines model from a MLXLM model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `mlx_model` | `Module` | The nn.Module model to use. | *required* | | `mlx_tokenizer` | `PreTrainedTokenizer` | The PreTrainedTokenizer to use. | *required* | | `provider` | `Literal['outlines'] | Provider[Model]` | The provider to use for OutlinesModel. Can be either the string 'outlines' or an instance of Provider[OutlinesBaseModel]. If not provided, the other parameters will be used. | `'outlines'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider. | `None` | | `settings` | `ModelSettings | None` | Default model settings for this model instance. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
@classmethod
def from_mlxlm(  # pragma: no cover
    cls,
    mlx_model: nn.Module,
    mlx_tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
    *,
    provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Create an Outlines model from a MLXLM model.

    Args:
        mlx_model: The nn.Module model to use.
        mlx_tokenizer: The PreTrainedTokenizer to use.
        provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
            instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
        profile: The model profile to use. Defaults to a profile picked by the provider.
        settings: Default model settings for this model instance.
    """
    outlines_model: OutlinesBaseModel = from_mlxlm(mlx_model, mlx_tokenizer)
    return cls(outlines_model, provider=provider, profile=profile, settings=settings)

```

#### from_sglang

```python
from_sglang(
    base_url: str,
    api_key: str | None = None,
    model_name: str | None = None,
    *,
    provider: (
        Literal["outlines"] | Provider[Model]
    ) = "outlines",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Create an Outlines model to send requests to an SGLang server.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `base_url` | `str` | The url of the SGLang server. | *required* | | `api_key` | `str | None` | The API key to use for authenticating requests to the SGLang server. | `None` | | `model_name` | `str | None` | The name of the model to use. | `None` | | `provider` | `Literal['outlines'] | Provider[Model]` | The provider to use for OutlinesModel. Can be either the string 'outlines' or an instance of Provider[OutlinesBaseModel]. If not provided, the other parameters will be used. | `'outlines'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider. | `None` | | `settings` | `ModelSettings | None` | Default model settings for this model instance. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
@classmethod
def from_sglang(
    cls,
    base_url: str,
    api_key: str | None = None,
    model_name: str | None = None,
    *,
    provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Create an Outlines model to send requests to an SGLang server.

    Args:
        base_url: The url of the SGLang server.
        api_key: The API key to use for authenticating requests to the SGLang server.
        model_name: The name of the model to use.
        provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
            instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
        profile: The model profile to use. Defaults to a profile picked by the provider.
        settings: Default model settings for this model instance.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as _import_error:
        raise ImportError(
            'Please install `openai` to use the Outlines SGLang model, '
            'you can use the `openai` optional group â€” `pip install "pydantic-ai-slim[openai]"`'
        ) from _import_error

    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_sglang(openai_client, model_name)
    return cls(outlines_model, provider=provider, profile=profile, settings=settings)

```

#### from_vllm_offline

```python
from_vllm_offline(
    vllm_model: Any,
    *,
    provider: (
        Literal["outlines"] | Provider[Model]
    ) = "outlines",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Create an Outlines model from a vLLM offline inference model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `vllm_model` | `Any` | The vllm.LLM local model to use. | *required* | | `provider` | `Literal['outlines'] | Provider[Model]` | The provider to use for OutlinesModel. Can be either the string 'outlines' or an instance of Provider[OutlinesBaseModel]. If not provided, the other parameters will be used. | `'outlines'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider. | `None` | | `settings` | `ModelSettings | None` | Default model settings for this model instance. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
@classmethod
def from_vllm_offline(  # pragma: no cover
    cls,
    vllm_model: Any,
    *,
    provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Create an Outlines model from a vLLM offline inference model.

    Args:
        vllm_model: The vllm.LLM local model to use.
        provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
            instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
        profile: The model profile to use. Defaults to a profile picked by the provider.
        settings: Default model settings for this model instance.
    """
    outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_vllm_offline(vllm_model)
    return cls(outlines_model, provider=provider, profile=profile, settings=settings)

```

#### format_inference_kwargs

```python
format_inference_kwargs(
    model_settings: ModelSettings | None,
) -> dict[str, Any]

```

Format the model settings for the inference kwargs.

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
def format_inference_kwargs(self, model_settings: ModelSettings | None) -> dict[str, Any]:
    """Format the model settings for the inference kwargs."""
    settings_dict: dict[str, Any] = dict(model_settings) if model_settings else {}

    if isinstance(self.model, Transformers):
        settings_dict = self._format_transformers_inference_kwargs(settings_dict)
    elif isinstance(self.model, LlamaCpp):
        settings_dict = self._format_llama_cpp_inference_kwargs(settings_dict)
    elif isinstance(self.model, MLXLM):  # pragma: no cover
        settings_dict = self._format_mlxlm_inference_kwargs(settings_dict)
    elif isinstance(self.model, SGLang | AsyncSGLang):
        settings_dict = self._format_sglang_inference_kwargs(settings_dict)
    elif isinstance(self.model, VLLMOffline):  # pragma: no cover
        settings_dict = self._format_vllm_offline_inference_kwargs(settings_dict)

    extra_body = settings_dict.pop('extra_body', {})
    settings_dict.update(extra_body)

    return settings_dict

```

#### customize_request_parameters

```python
customize_request_parameters(
    model_request_parameters: ModelRequestParameters,
) -> ModelRequestParameters

```

Customize the model request parameters for the model.

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
    """Customize the model request parameters for the model."""
    if model_request_parameters.output_mode in ('auto', 'native'):
        # This way the JSON schema will be included in the instructions.
        return replace(model_request_parameters, output_mode='prompted')
    else:
        return model_request_parameters

```

### OutlinesStreamedResponse

Bases: `StreamedResponse`

Implementation of `StreamedResponse` for Outlines models.

Source code in `pydantic_ai_slim/pydantic_ai/models/outlines.py`

```python
@dataclass
class OutlinesStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Outlines models."""

    _model_name: str
    _model_profile: ModelProfile
    _response: AsyncIterable[str]
    _timestamp: datetime
    _provider_name: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in self._response:
            event = self._parts_manager.handle_text_delta(
                vendor_part_id='content',
                content=event,
                thinking_tags=self._model_profile.thinking_tags,
                ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
            )
            if event is not None:  # pragma: no branch
                yield event

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp

```

#### model_name

```python
model_name: str

```

Get the model name of the response.

#### provider_name

```python
provider_name: str

```

Get the provider name.

#### timestamp

```python
timestamp: datetime

```

Get the timestamp of the response.

# `pydantic_ai.models.test`

Utility model for quickly testing apps built with Pydantic AI.

Here's a minimal example:

[Learn about Gateway](../../../gateway) test_model_usage.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

my_agent = Agent('gateway/openai:gpt-5', system_prompt='...')


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    m = TestModel()
    with my_agent.override(model=m):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'success (no tool calls)'
    assert m.last_model_request_parameters.function_tools == []

```

test_model_usage.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

my_agent = Agent('openai:gpt-5', system_prompt='...')


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    m = TestModel()
    with my_agent.override(model=m):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'success (no tool calls)'
    assert m.last_model_request_parameters.function_tools == []

```

See [Unit testing with `TestModel`](../../../testing/#unit-testing-with-testmodel) for detailed documentation.

### TestModel

Bases: `Model`

A model specifically for testing purposes.

This will (by default) call all tools in the agent, then return a tool response if possible, otherwise a plain response.

How useful this model is will vary significantly.

Apart from `__init__` derived by the `dataclass` decorator, all methods are private or match those of the base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/test.py`

```python
@dataclass(init=False)
class TestModel(Model):
    """A model specifically for testing purposes.

    This will (by default) call all tools in the agent, then return a tool response if possible,
    otherwise a plain response.

    How useful this model is will vary significantly.

    Apart from `__init__` derived by the `dataclass` decorator, all methods are private or match those
    of the base class.
    """

    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    call_tools: list[str] | Literal['all'] = 'all'
    """List of tools to call. If `'all'`, all tools will be called."""
    custom_output_text: str | None = None
    """If set, this text is returned as the final output."""
    custom_output_args: Any | None = None
    """If set, these args will be passed to the output tool."""
    seed: int = 0
    """Seed for generating random data."""
    last_model_request_parameters: ModelRequestParameters | None = field(default=None, init=False)
    """The last ModelRequestParameters passed to the model in a request.

    The ModelRequestParameters contains information about the function and output tools available during request handling.

    This is set when a request is made, so will reflect the function tools from the last step of the last run.
    """
    _model_name: str = field(default='test', repr=False)
    _system: str = field(default='test', repr=False)

    def __init__(
        self,
        *,
        call_tools: list[str] | Literal['all'] = 'all',
        custom_output_text: str | None = None,
        custom_output_args: Any | None = None,
        seed: int = 0,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize TestModel with optional settings and profile."""
        self.call_tools = call_tools
        self.custom_output_text = custom_output_text
        self.custom_output_args = custom_output_args
        self.seed = seed
        self.last_model_request_parameters = None
        self._model_name = 'test'
        self._system = 'test'
        super().__init__(settings=settings, profile=profile)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        self.last_model_request_parameters = model_request_parameters
        model_response = self._request(messages, model_settings, model_request_parameters)
        model_response.usage = _estimate_usage([*messages, model_response])
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        self.last_model_request_parameters = model_request_parameters

        model_response = self._request(messages, model_settings, model_request_parameters)
        yield TestStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _structured_response=model_response,
            _messages=messages,
            _provider_name=self._system,
        )

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._system

    def gen_tool_args(self, tool_def: ToolDefinition) -> Any:
        return _JsonSchemaTestData(tool_def.parameters_json_schema, self.seed).generate()

    def _get_tool_calls(self, model_request_parameters: ModelRequestParameters) -> list[tuple[str, ToolDefinition]]:
        if self.call_tools == 'all':
            return [(r.name, r) for r in model_request_parameters.function_tools]
        else:
            function_tools_lookup = {t.name: t for t in model_request_parameters.function_tools}
            tools_to_call = (function_tools_lookup[name] for name in self.call_tools)
            return [(r.name, r) for r in tools_to_call]

    def _get_output(self, model_request_parameters: ModelRequestParameters) -> _WrappedTextOutput | _WrappedToolOutput:
        if self.custom_output_text is not None:
            assert model_request_parameters.output_mode != 'tool', (
                'Plain response not allowed, but `custom_output_text` is set.'
            )
            assert self.custom_output_args is None, 'Cannot set both `custom_output_text` and `custom_output_args`.'
            return _WrappedTextOutput(self.custom_output_text)
        elif self.custom_output_args is not None:
            assert model_request_parameters.output_tools is not None, (
                'No output tools provided, but `custom_output_args` is set.'
            )
            output_tool = model_request_parameters.output_tools[0]

            if k := output_tool.outer_typed_dict_key:
                return _WrappedToolOutput({k: self.custom_output_args})
            else:
                return _WrappedToolOutput(self.custom_output_args)
        elif model_request_parameters.allow_text_output:
            return _WrappedTextOutput(None)
        elif model_request_parameters.output_tools:
            return _WrappedToolOutput(None)
        else:
            return _WrappedTextOutput(None)

    def _request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if model_request_parameters.builtin_tools:
            raise UserError('TestModel does not support built-in tools')

        tool_calls = self._get_tool_calls(model_request_parameters)
        output_wrapper = self._get_output(model_request_parameters)
        output_tools = model_request_parameters.output_tools

        # if there are tools, the first thing we want to do is call all of them
        if tool_calls and not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(name, self.gen_tool_args(args), tool_call_id=f'pyd_ai_tool_call_id__{name}')
                    for name, args in tool_calls
                ],
                model_name=self._model_name,
            )

        if messages:  # pragma: no branch
            last_message = messages[-1]
            assert isinstance(last_message, ModelRequest), 'Expected last message to be a `ModelRequest`.'

            # check if there are any retry prompts, if so retry them
            new_retry_names = {p.tool_name for p in last_message.parts if isinstance(p, RetryPromptPart)}
            if new_retry_names:
                # Handle retries for both function tools and output tools
                # Check function tools first
                retry_parts: list[ModelResponsePart] = [
                    ToolCallPart(name, self.gen_tool_args(args)) for name, args in tool_calls if name in new_retry_names
                ]
                # Check output tools
                if output_tools:
                    retry_parts.extend(
                        [
                            ToolCallPart(
                                tool.name,
                                output_wrapper.value
                                if isinstance(output_wrapper, _WrappedToolOutput) and output_wrapper.value is not None
                                else self.gen_tool_args(tool),
                                tool_call_id=f'pyd_ai_tool_call_id__{tool.name}',
                            )
                            for tool in output_tools
                            if tool.name in new_retry_names
                        ]
                    )
                return ModelResponse(parts=retry_parts, model_name=self._model_name)

        if isinstance(output_wrapper, _WrappedTextOutput):
            if (response_text := output_wrapper.value) is None:
                # build up details of tool responses
                output: dict[str, Any] = {}
                for message in messages:
                    if isinstance(message, ModelRequest):
                        for part in message.parts:
                            if isinstance(part, ToolReturnPart):
                                output[part.tool_name] = part.content
                if output:
                    return ModelResponse(
                        parts=[TextPart(pydantic_core.to_json(output).decode())], model_name=self._model_name
                    )
                else:
                    return ModelResponse(parts=[TextPart('success (no tool calls)')], model_name=self._model_name)
            else:
                return ModelResponse(parts=[TextPart(response_text)], model_name=self._model_name)
        else:
            assert output_tools, 'No output tools provided'
            custom_output_args = output_wrapper.value
            output_tool = output_tools[self.seed % len(output_tools)]
            if custom_output_args is not None:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            output_tool.name,
                            custom_output_args,
                            tool_call_id=f'pyd_ai_tool_call_id__{output_tool.name}',
                        )
                    ],
                    model_name=self._model_name,
                )
            else:
                response_args = self.gen_tool_args(output_tool)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            output_tool.name,
                            response_args,
                            tool_call_id=f'pyd_ai_tool_call_id__{output_tool.name}',
                        )
                    ],
                    model_name=self._model_name,
                )

```

#### __init__

```python
__init__(
    *,
    call_tools: list[str] | Literal["all"] = "all",
    custom_output_text: str | None = None,
    custom_output_args: Any | None = None,
    seed: int = 0,
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize TestModel with optional settings and profile.

Source code in `pydantic_ai_slim/pydantic_ai/models/test.py`

```python
def __init__(
    self,
    *,
    call_tools: list[str] | Literal['all'] = 'all',
    custom_output_text: str | None = None,
    custom_output_args: Any | None = None,
    seed: int = 0,
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize TestModel with optional settings and profile."""
    self.call_tools = call_tools
    self.custom_output_text = custom_output_text
    self.custom_output_args = custom_output_args
    self.seed = seed
    self.last_model_request_parameters = None
    self._model_name = 'test'
    self._system = 'test'
    super().__init__(settings=settings, profile=profile)

```

#### call_tools

```python
call_tools: list[str] | Literal['all'] = call_tools

```

List of tools to call. If `'all'`, all tools will be called.

#### custom_output_text

```python
custom_output_text: str | None = custom_output_text

```

If set, this text is returned as the final output.

#### custom_output_args

```python
custom_output_args: Any | None = custom_output_args

```

If set, these args will be passed to the output tool.

#### seed

```python
seed: int = seed

```

Seed for generating random data.

#### last_model_request_parameters

```python
last_model_request_parameters: (
    ModelRequestParameters | None
) = None

```

The last ModelRequestParameters passed to the model in a request.

The ModelRequestParameters contains information about the function and output tools available during request handling.

This is set when a request is made, so will reflect the function tools from the last step of the last run.

#### model_name

```python
model_name: str

```

The model name.

#### system

```python
system: str

```

The model provider.

### TestStreamedResponse

Bases: `StreamedResponse`

A structured response that streams test data.

Source code in `pydantic_ai_slim/pydantic_ai/models/test.py`

```python
@dataclass
class TestStreamedResponse(StreamedResponse):
    """A structured response that streams test data."""

    _model_name: str
    _structured_response: ModelResponse
    _messages: InitVar[Iterable[ModelMessage]]
    _provider_name: str
    _timestamp: datetime = field(default_factory=_utils.now_utc, init=False)

    def __post_init__(self, _messages: Iterable[ModelMessage]):
        self._usage = _estimate_usage(_messages)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for i, part in enumerate(self._structured_response.parts):
            if isinstance(part, TextPart):
                text = part.content
                *words, last_word = text.split(' ')
                words = [f'{word} ' for word in words]
                words.append(last_word)
                if len(words) == 1 and len(text) > 2:
                    mid = len(text) // 2
                    words = [text[:mid], text[mid:]]
                self._usage += _get_string_usage('')
                maybe_event = self._parts_manager.handle_text_delta(vendor_part_id=i, content='')
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event
                for word in words:
                    self._usage += _get_string_usage(word)
                    maybe_event = self._parts_manager.handle_text_delta(vendor_part_id=i, content=word)
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
            elif isinstance(part, ToolCallPart):
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=i, tool_name=part.tool_name, args=part.args, tool_call_id=part.tool_call_id
                )
            elif isinstance(part, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                # NOTE: These parts are not generated by TestModel, but we need to handle them for type checking
                assert False, f'Unexpected part type in TestModel: {type(part).__name__}'
            elif isinstance(part, ThinkingPart):  # pragma: no cover
                # NOTE: There's no way to reach this part of the code, since we don't generate ThinkingPart on TestModel.
                assert False, "This should be unreachable â€” we don't generate ThinkingPart on TestModel."
            elif isinstance(part, FilePart):  # pragma: no cover
                # NOTE: There's no way to reach this part of the code, since we don't generate FilePart on TestModel.
                assert False, "This should be unreachable â€” we don't generate FilePart on TestModel."
            else:
                assert_never(part)

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp

```

#### model_name

```python
model_name: str

```

Get the model name of the response.

#### provider_name

```python
provider_name: str

```

Get the provider name.

#### timestamp

```python
timestamp: datetime

```

Get the timestamp of the response.

