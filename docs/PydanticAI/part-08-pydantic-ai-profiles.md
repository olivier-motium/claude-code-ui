<!-- Source: _complete-doc.md -->

# `pydantic_ai.profiles`

Describes how requests to and responses from specific models or families of models need to be constructed and processed to get the best results, independent of the model and provider classes used.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/__init__.py`

```python
@dataclass(kw_only=True)
class ModelProfile:
    """Describes how requests to and responses from specific models or families of models need to be constructed and processed to get the best results, independent of the model and provider classes used."""

    supports_tools: bool = True
    """Whether the model supports tools."""
    supports_json_schema_output: bool = False
    """Whether the model supports JSON schema output."""
    supports_json_object_output: bool = False
    """Whether the model supports JSON object output."""
    supports_image_output: bool = False
    """Whether the model supports image output."""
    default_structured_output_mode: StructuredOutputMode = 'tool'
    """The default structured output mode to use for the model."""
    prompted_output_template: str = dedent(
        """
        Always respond with a JSON object that's compatible with this schema:

        {schema}

        Don't include any text or Markdown fencing before or after.
        """
    )
    """The instructions template to use for prompted structured output. The '{schema}' placeholder will be replaced with the JSON schema for the output."""
    json_schema_transformer: type[JsonSchemaTransformer] | None = None
    """The transformer to use to make JSON schemas for tools and structured output compatible with the model."""

    thinking_tags: tuple[str, str] = ('<think>', '</think>')
    """The tags used to indicate thinking parts in the model's output. Defaults to ('<think>', '</think>')."""

    ignore_streamed_leading_whitespace: bool = False
    """Whether to ignore leading whitespace when streaming a response.

    This is a workaround for models that emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
    which we don't want to end up treating as a final result when using `run_stream` with `str` a valid `output_type`.

    This is currently only used by `OpenAIChatModel`, `HuggingFaceModel`, and `GroqModel`.
    """

    @classmethod
    def from_profile(cls, profile: ModelProfile | None) -> Self:
        """Build a ModelProfile subclass instance from a ModelProfile instance."""
        if isinstance(profile, cls):
            return profile
        return cls().update(profile)

    def update(self, profile: ModelProfile | None) -> Self:
        """Update this ModelProfile (subclass) instance with the non-default values from another ModelProfile instance."""
        if not profile:
            return self
        field_names = set(f.name for f in fields(self))
        non_default_attrs = {
            f.name: getattr(profile, f.name)
            for f in fields(profile)
            if f.name in field_names and getattr(profile, f.name) != f.default
        }
        return replace(self, **non_default_attrs)

```

### supports_tools

```python
supports_tools: bool = True

```

Whether the model supports tools.

### supports_json_schema_output

```python
supports_json_schema_output: bool = False

```

Whether the model supports JSON schema output.

### supports_json_object_output

```python
supports_json_object_output: bool = False

```

Whether the model supports JSON object output.

### supports_image_output

```python
supports_image_output: bool = False

```

Whether the model supports image output.

### default_structured_output_mode

```python
default_structured_output_mode: StructuredOutputMode = (
    "tool"
)

```

The default structured output mode to use for the model.

### prompted_output_template

```python
prompted_output_template: str = dedent(
    "\n        Always respond with a JSON object that's compatible with this schema:\n\n        {schema}\n\n        Don't include any text or Markdown fencing before or after.\n        "
)

```

The instructions template to use for prompted structured output. The '{schema}' placeholder will be replaced with the JSON schema for the output.

### json_schema_transformer

```python
json_schema_transformer: (
    type[JsonSchemaTransformer] | None
) = None

```

The transformer to use to make JSON schemas for tools and structured output compatible with the model.

### thinking_tags

```python
thinking_tags: tuple[str, str] = ('<think>', '</think>')

```

The tags used to indicate thinking parts in the model's output. Defaults to ('', '').

### ignore_streamed_leading_whitespace

```python
ignore_streamed_leading_whitespace: bool = False

```

Whether to ignore leading whitespace when streaming a response.

```text
This is a workaround for models that emit `<think>

```

`or an empty text part ahead of tool calls (e.g. Ollama + Qwen3), which we don't want to end up treating as a final result when using`run_stream`with`str`a valid`output_type\`.

```text
This is currently only used by `OpenAIChatModel`, `HuggingFaceModel`, and `GroqModel`.

```

### from_profile

```python
from_profile(profile: ModelProfile | None) -> Self

```

Build a ModelProfile subclass instance from a ModelProfile instance.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/__init__.py`

```python
@classmethod
def from_profile(cls, profile: ModelProfile | None) -> Self:
    """Build a ModelProfile subclass instance from a ModelProfile instance."""
    if isinstance(profile, cls):
        return profile
    return cls().update(profile)

```

### update

```python
update(profile: ModelProfile | None) -> Self

```

Update this ModelProfile (subclass) instance with the non-default values from another ModelProfile instance.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/__init__.py`

```python
def update(self, profile: ModelProfile | None) -> Self:
    """Update this ModelProfile (subclass) instance with the non-default values from another ModelProfile instance."""
    if not profile:
        return self
    field_names = set(f.name for f in fields(self))
    non_default_attrs = {
        f.name: getattr(profile, f.name)
        for f in fields(profile)
        if f.name in field_names and getattr(profile, f.name) != f.default
    }
    return replace(self, **non_default_attrs)

```

### OpenAIModelProfile

Bases: `ModelProfile`

Profile for models used with `OpenAIChatModel`.

ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/openai.py`

```python
@dataclass(kw_only=True)
class OpenAIModelProfile(ModelProfile):
    """Profile for models used with `OpenAIChatModel`.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_supports_strict_tool_definition: bool = True
    """This can be set by a provider or user if the OpenAI-"compatible" API doesn't support strict tool definitions."""

    openai_supports_sampling_settings: bool = True
    """Turn off to don't send sampling settings like `temperature` and `top_p` to models that don't support them, like OpenAI's o-series reasoning models."""

    openai_unsupported_model_settings: Sequence[str] = ()
    """A list of model settings that are not supported by this model."""

    # Some OpenAI-compatible providers (e.g. MoonshotAI) currently do **not** accept
    # `tool_choice="required"`.  This flag lets the calling model know whether it's
    # safe to pass that value along.  Default is `True` to preserve existing
    # behaviour for OpenAI itself and most providers.
    openai_supports_tool_choice_required: bool = True
    """Whether the provider accepts the value ``tool_choice='required'`` in the request payload."""

    openai_system_prompt_role: OpenAISystemPromptRole | None = None
    """The role to use for the system prompt message. If not provided, defaults to `'system'`."""

    openai_chat_supports_web_search: bool = False
    """Whether the model supports web search in Chat Completions API."""

    openai_supports_encrypted_reasoning_content: bool = False
    """Whether the model supports including encrypted reasoning content in the response."""

    openai_responses_requires_function_call_status_none: bool = False
    """Whether the Responses API requires the `status` field on function tool calls to be `None`.

    This is required by vLLM Responses API versions before https://github.com/vllm-project/vllm/pull/26706.
    See https://github.com/pydantic/pydantic-ai/issues/3245 for more details.
    """

    def __post_init__(self):  # pragma: no cover
        if not self.openai_supports_sampling_settings:
            warnings.warn(
                'The `openai_supports_sampling_settings` has no effect, and it will be removed in future versions. '
                'Use `openai_unsupported_model_settings` instead.',
                DeprecationWarning,
            )

```

#### openai_supports_strict_tool_definition

```python
openai_supports_strict_tool_definition: bool = True

```

This can be set by a provider or user if the OpenAI-"compatible" API doesn't support strict tool definitions.

#### openai_supports_sampling_settings

```python
openai_supports_sampling_settings: bool = True

```

Turn off to don't send sampling settings like `temperature` and `top_p` to models that don't support them, like OpenAI's o-series reasoning models.

#### openai_unsupported_model_settings

```python
openai_unsupported_model_settings: Sequence[str] = ()

```

A list of model settings that are not supported by this model.

#### openai_supports_tool_choice_required

```python
openai_supports_tool_choice_required: bool = True

```

Whether the provider accepts the value `tool_choice='required'` in the request payload.

#### openai_system_prompt_role

```python
openai_system_prompt_role: OpenAISystemPromptRole | None = (
    None
)

```

The role to use for the system prompt message. If not provided, defaults to `'system'`.

#### openai_chat_supports_web_search

```python
openai_chat_supports_web_search: bool = False

```

Whether the model supports web search in Chat Completions API.

#### openai_supports_encrypted_reasoning_content

```python
openai_supports_encrypted_reasoning_content: bool = False

```

Whether the model supports including encrypted reasoning content in the response.

#### openai_responses_requires_function_call_status_none

```python
openai_responses_requires_function_call_status_none: (
    bool
) = False

```

Whether the Responses API requires the `status` field on function tool calls to be `None`.

This is required by vLLM Responses API versions before https://github.com/vllm-project/vllm/pull/26706. See https://github.com/pydantic/pydantic-ai/issues/3245 for more details.

### openai_model_profile

```python
openai_model_profile(model_name: str) -> ModelProfile

```

Get the model profile for an OpenAI model.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/openai.py`

```python
def openai_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for an OpenAI model."""
    is_gpt_5 = model_name.startswith('gpt-5')
    is_o_series = model_name.startswith('o')
    is_reasoning_model = is_o_series or (is_gpt_5 and 'gpt-5-chat' not in model_name)

    # Check if the model supports web search (only specific search-preview models)
    supports_web_search = '-search-preview' in model_name

    # Structured Outputs (output mode 'native') is only supported with the gpt-4o-mini, gpt-4o-mini-2024-07-18, and gpt-4o-2024-08-06 model snapshots and later.
    # We leave it in here for all models because the `default_structured_output_mode` is `'tool'`, so `native` is only used
    # when the user specifically uses the `NativeOutput` marker, so an error from the API is acceptable.

    if is_reasoning_model:
        openai_unsupported_model_settings = (
            'temperature',
            'top_p',
            'presence_penalty',
            'frequency_penalty',
            'logit_bias',
            'logprobs',
            'top_logprobs',
        )
    else:
        openai_unsupported_model_settings = ()

    # The o1-mini model doesn't support the `system` role, so we default to `user`.
    # See https://github.com/pydantic/pydantic-ai/issues/974 for more details.
    openai_system_prompt_role = 'user' if model_name.startswith('o1-mini') else None

    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_image_output=is_gpt_5 or 'o3' in model_name or '4.1' in model_name or '4o' in model_name,
        openai_unsupported_model_settings=openai_unsupported_model_settings,
        openai_system_prompt_role=openai_system_prompt_role,
        openai_chat_supports_web_search=supports_web_search,
        openai_supports_encrypted_reasoning_content=is_reasoning_model,
    )

```

### OpenAIJsonSchemaTransformer

Bases: `JsonSchemaTransformer`

Recursively handle the schema to make it compatible with OpenAI strict mode.

See https://platform.openai.com/docs/guides/function-calling?api-mode=responses#strict-mode for more details, but this basically just requires: * `additionalProperties` must be set to false for each object in the parameters * all fields in properties must be marked as required

Source code in `pydantic_ai_slim/pydantic_ai/profiles/openai.py`

```python
@dataclass(init=False)
class OpenAIJsonSchemaTransformer(JsonSchemaTransformer):
    """Recursively handle the schema to make it compatible with OpenAI strict mode.

    See https://platform.openai.com/docs/guides/function-calling?api-mode=responses#strict-mode for more details,
    but this basically just requires:
    * `additionalProperties` must be set to false for each object in the parameters
    * all fields in properties must be marked as required
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict)
        self.root_ref = schema.get('$ref')

    def walk(self) -> JsonSchema:
        # Note: OpenAI does not support anyOf at the root in strict mode
        # However, we don't need to check for it here because we ensure in pydantic_ai._utils.check_object_json_schema
        # that the root schema either has type 'object' or is recursive.
        result = super().walk()

        # For recursive models, we need to tweak the schema to make it compatible with strict mode.
        # Because the following should never change the semantics of the schema we apply it unconditionally.
        if self.root_ref is not None:
            result.pop('$ref', None)  # We replace references to the self.root_ref with just '#' in the transform method
            root_key = re.sub(r'^#/\$defs/', '', self.root_ref)
            result.update(self.defs.get(root_key) or {})

        return result

    def transform(self, schema: JsonSchema) -> JsonSchema:  # noqa C901
        # Remove unnecessary keys
        schema.pop('title', None)
        schema.pop('$schema', None)
        schema.pop('discriminator', None)

        default = schema.get('default', _sentinel)
        if default is not _sentinel:
            # the "default" keyword is not allowed in strict mode, but including it makes some Ollama models behave
            # better, so we keep it around when not strict
            if self.strict is True:
                schema.pop('default', None)
            elif self.strict is None:  # pragma: no branch
                self.is_strict_compatible = False

        if schema_ref := schema.get('$ref'):
            if schema_ref == self.root_ref:
                schema['$ref'] = '#'
            if len(schema) > 1:
                # OpenAI Strict mode doesn't support siblings to "$ref", but _does_ allow siblings to "anyOf".
                # So if there is a "description" field or any other extra info, we move the "$ref" into an "anyOf":
                schema['anyOf'] = [{'$ref': schema.pop('$ref')}]

        # Track strict-incompatible keys
        incompatible_values: dict[str, Any] = {}
        for key in _STRICT_INCOMPATIBLE_KEYS:
            value = schema.get(key, _sentinel)
            if value is not _sentinel:
                incompatible_values[key] = value
        if format := schema.get('format'):
            if format not in _STRICT_COMPATIBLE_STRING_FORMATS:
                incompatible_values['format'] = format
        description = schema.get('description')
        if incompatible_values:
            if self.strict is True:
                notes: list[str] = []
                for key, value in incompatible_values.items():
                    schema.pop(key)
                    notes.append(f'{key}={value}')
                notes_string = ', '.join(notes)
                schema['description'] = notes_string if not description else f'{description} ({notes_string})'
            elif self.strict is None:  # pragma: no branch
                self.is_strict_compatible = False

        schema_type = schema.get('type')
        if 'oneOf' in schema:
            # OpenAI does not support oneOf in strict mode
            if self.strict is True:
                schema['anyOf'] = schema.pop('oneOf')
            else:
                self.is_strict_compatible = False

        if schema_type == 'object':
            if self.strict is True:
                # additional properties are disallowed
                schema['additionalProperties'] = False

                # all properties are required
                if 'properties' not in schema:
                    schema['properties'] = dict[str, Any]()
                schema['required'] = list(schema['properties'].keys())

            elif self.strict is None:
                if schema.get('additionalProperties', None) not in (None, False):
                    self.is_strict_compatible = False
                else:
                    # additional properties are disallowed by default
                    schema['additionalProperties'] = False

                if 'properties' not in schema or 'required' not in schema:
                    self.is_strict_compatible = False
                else:
                    required = schema['required']
                    for k in schema['properties'].keys():
                        if k not in required:
                            self.is_strict_compatible = False
        return schema

```

### anthropic_model_profile

```python
anthropic_model_profile(
    model_name: str,
) -> ModelProfile | None

```

Get the model profile for an Anthropic model.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/anthropic.py`

```python
def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    return ModelProfile(thinking_tags=('<thinking>', '</thinking>'))

```

### GoogleModelProfile

Bases: `ModelProfile`

Profile for models used with `GoogleModel`.

ALL FIELDS MUST BE `google_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/google.py`

```python
@dataclass(kw_only=True)
class GoogleModelProfile(ModelProfile):
    """Profile for models used with `GoogleModel`.

    ALL FIELDS MUST BE `google_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    google_supports_native_output_with_builtin_tools: bool = False
    """Whether the model supports native output with builtin tools.
    See https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#structured_outputs_with_tools"""

```

#### google_supports_native_output_with_builtin_tools

```python
google_supports_native_output_with_builtin_tools: bool = (
    False
)

```

Whether the model supports native output with builtin tools. See https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#structured_outputs_with_tools

### google_model_profile

```python
google_model_profile(
    model_name: str,
) -> ModelProfile | None

```

Get the model profile for a Google model.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/google.py`

```python
def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    is_image_model = 'image' in model_name
    is_3_or_newer = 'gemini-3' in model_name
    return GoogleModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_image_output=is_image_model,
        supports_json_schema_output=not is_image_model,
        supports_json_object_output=not is_image_model,
        supports_tools=not is_image_model,
        google_supports_native_output_with_builtin_tools=is_3_or_newer,
    )

```

### GoogleJsonSchemaTransformer

Bases: `JsonSchemaTransformer`

Transforms the JSON Schema from Pydantic to be suitable for Gemini.

Gemini supports [a subset of OpenAPI v3.0.3](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations).

Source code in `pydantic_ai_slim/pydantic_ai/profiles/google.py`

```python
class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini supports [a subset of OpenAPI v3.0.3](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations).
    """

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove properties not supported by Gemini
        schema.pop('$schema', None)
        if (const := schema.pop('const', None)) is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]
        schema.pop('discriminator', None)
        schema.pop('examples', None)

        # Remove 'title' due to https://github.com/googleapis/python-genai/issues/1732
        schema.pop('title', None)

        type_ = schema.get('type')
        if type_ == 'string' and (fmt := schema.pop('format', None)):
            description = schema.get('description')
            if description:
                schema['description'] = f'{description} (format: {fmt})'
            else:
                schema['description'] = f'Format: {fmt}'

        # Note: exclusiveMinimum/exclusiveMaximum are NOT yet supported
        schema.pop('exclusiveMinimum', None)
        schema.pop('exclusiveMaximum', None)

        return schema

```

### meta_model_profile

```python
meta_model_profile(model_name: str) -> ModelProfile | None

```

Get the model profile for a Meta model.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/meta.py`

```python
def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)

```

### amazon_model_profile

```python
amazon_model_profile(
    model_name: str,
) -> ModelProfile | None

```

Get the model profile for an Amazon model.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/amazon.py`

```python
def amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)

```

### deepseek_model_profile

```python
deepseek_model_profile(
    model_name: str,
) -> ModelProfile | None

```

Get the model profile for a DeepSeek model.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/deepseek.py`

```python
def deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model."""
    return ModelProfile(ignore_streamed_leading_whitespace='r1' in model_name)

```

### grok_model_profile

```python
grok_model_profile(model_name: str) -> ModelProfile | None

```

Get the model profile for a Grok model.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/grok.py`

```python
def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    return None

```

### mistral_model_profile

```python
mistral_model_profile(
    model_name: str,
) -> ModelProfile | None

```

Get the model profile for a Mistral model.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/mistral.py`

```python
def mistral_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Mistral model."""
    return None

```

### qwen_model_profile

```python
qwen_model_profile(model_name: str) -> ModelProfile | None

```

Get the model profile for a Qwen model.

Source code in `pydantic_ai_slim/pydantic_ai/profiles/qwen.py`

```python
def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    if model_name.startswith('qwen-3-coder'):
        return OpenAIModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,
            openai_supports_tool_choice_required=False,
            openai_supports_strict_tool_definition=False,
            ignore_streamed_leading_whitespace=True,
        )
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        ignore_streamed_leading_whitespace=True,
    )

```

# `pydantic_ai.providers`

Bases: `ABC`, `Generic[InterfaceClient]`

Abstract class for a provider.

The provider is in charge of providing an authenticated client to the API.

Each provider only supports a specific interface. A interface can be supported by multiple providers.

For example, the `OpenAIChatModel` interface can be supported by the `OpenAIProvider` and the `DeepSeekProvider`.

Source code in `pydantic_ai_slim/pydantic_ai/providers/__init__.py`

```python
class Provider(ABC, Generic[InterfaceClient]):
    """Abstract class for a provider.

    The provider is in charge of providing an authenticated client to the API.

    Each provider only supports a specific interface. A interface can be supported by multiple providers.

    For example, the `OpenAIChatModel` interface can be supported by the `OpenAIProvider` and the `DeepSeekProvider`.
    """

    _client: InterfaceClient

    @property
    @abstractmethod
    def name(self) -> str:
        """The provider name."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def base_url(self) -> str:
        """The base URL for the provider API."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def client(self) -> InterfaceClient:
        """The client for the provider."""
        raise NotImplementedError()

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """The model profile for the named model, if available."""
        return None  # pragma: no cover

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name}, base_url={self.base_url})'  # pragma: lax no cover

```

### name

```python
name: str

```

The provider name.

### base_url

```python
base_url: str

```

The base URL for the provider API.

### client

```python
client: InterfaceClient

```

The client for the provider.

### model_profile

```python
model_profile(model_name: str) -> ModelProfile | None

```

The model profile for the named model, if available.

Source code in `pydantic_ai_slim/pydantic_ai/providers/__init__.py`

```python
def model_profile(self, model_name: str) -> ModelProfile | None:
    """The model profile for the named model, if available."""
    return None  # pragma: no cover

```

### GoogleProvider

Bases: `Provider[Client]`

Provider for Google.

Source code in `pydantic_ai_slim/pydantic_ai/providers/google.py`

```python
class GoogleProvider(Provider[Client]):
    """Provider for Google."""

    @property
    def name(self) -> str:
        return 'google-vertex' if self._client._api_client.vertexai else 'google-gla'  # type: ignore[reportPrivateUsage]

    @property
    def base_url(self) -> str:
        return str(self._client._api_client._http_options.base_url)  # type: ignore[reportPrivateUsage]

    @property
    def client(self) -> Client:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return google_model_profile(model_name)

    @overload
    def __init__(
        self, *, api_key: str, http_client: httpx.AsyncClient | None = None, base_url: str | None = None
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: VertexAILocation | Literal['global'] | str | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None: ...

    @overload
    def __init__(self, *, client: Client) -> None: ...

    @overload
    def __init__(
        self,
        *,
        vertexai: bool = False,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: VertexAILocation | Literal['global'] | str | None = None,
        vertexai: bool | None = None,
        client: Client | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create a new Google provider.

        Args:
            api_key: The `API key <https://ai.google.dev/gemini-api/docs/api-key>`_ to
                use for authentication. It can also be set via the `GOOGLE_API_KEY` environment variable.
            credentials: The credentials to use for authentication when calling the Vertex AI APIs. Credentials can be
                obtained from environment variables and default credentials. For more information, see Set up
                Application Default Credentials. Applies to the Vertex AI API only.
            project: The Google Cloud project ID to use for quota. Can be obtained from environment variables
                (for example, GOOGLE_CLOUD_PROJECT). Applies to the Vertex AI API only.
            location: The location to send API requests to (for example, us-central1). Can be obtained from environment variables.
                Applies to the Vertex AI API only.
            vertexai: Force the use of the Vertex AI API. If `False`, the Google Generative Language API will be used.
                Defaults to `False` unless `location`, `project`, or `credentials` are provided.
            client: A pre-initialized client to use.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
            base_url: The base URL for the Google API.
        """
        if client is None:
            # NOTE: We are keeping GEMINI_API_KEY for backwards compatibility.
            api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

            vertex_ai_args_used = bool(location or project or credentials)
            if vertexai is None:
                vertexai = vertex_ai_args_used

            http_client = http_client or cached_async_http_client(
                provider='google-vertex' if vertexai else 'google-gla'
            )
            http_options = HttpOptions(
                base_url=base_url,
                headers={'User-Agent': get_user_agent()},
                httpx_async_client=http_client,
            )
            if not vertexai:
                if api_key is None:
                    raise UserError(
                        'Set the `GOOGLE_API_KEY` environment variable or pass it via `GoogleProvider(api_key=...)`'
                        'to use the Google Generative Language API.'
                    )
                self._client = Client(vertexai=False, api_key=api_key, http_options=http_options)
            else:
                if vertex_ai_args_used:
                    api_key = None

                if api_key is None:
                    project = project or os.getenv('GOOGLE_CLOUD_PROJECT')
                    # From https://github.com/pydantic/pydantic-ai/pull/2031/files#r2169682149:
                    # Currently `us-central1` supports the most models by far of any region including `global`, but not
                    # all of them. `us-central1` has all google models but is missing some Anthropic partner models,
                    # which use `us-east5` instead. `global` has fewer models but higher availability.
                    # For more details, check: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
                    location = location or os.getenv('GOOGLE_CLOUD_LOCATION') or 'us-central1'

                self._client = Client(
                    vertexai=True,
                    api_key=api_key,
                    project=project,
                    location=location,
                    credentials=credentials,
                    http_options=http_options,
                )
        else:
            self._client = client  # pragma: no cover

```

#### __init__

```python
__init__(
    *,
    api_key: str,
    http_client: AsyncClient | None = None,
    base_url: str | None = None
) -> None

```

```python
__init__(
    *,
    credentials: Credentials | None = None,
    project: str | None = None,
    location: (
        VertexAILocation | Literal["global"] | str | None
    ) = None,
    http_client: AsyncClient | None = None,
    base_url: str | None = None
) -> None

```

```python
__init__(*, client: Client) -> None

```

```python
__init__(
    *,
    vertexai: bool = False,
    api_key: str | None = None,
    http_client: AsyncClient | None = None,
    base_url: str | None = None
) -> None

```

```python
__init__(
    *,
    api_key: str | None = None,
    credentials: Credentials | None = None,
    project: str | None = None,
    location: (
        VertexAILocation | Literal["global"] | str | None
    ) = None,
    vertexai: bool | None = None,
    client: Client | None = None,
    http_client: AsyncClient | None = None,
    base_url: str | None = None
) -> None

```

Create a new Google provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `api_key` | `str | None` | The API key <https://ai.google.dev/gemini-api/docs/api-key>\_ to use for authentication. It can also be set via the GOOGLE_API_KEY environment variable. | `None` | | `credentials` | `Credentials | None` | The credentials to use for authentication when calling the Vertex AI APIs. Credentials can be obtained from environment variables and default credentials. For more information, see Set up Application Default Credentials. Applies to the Vertex AI API only. | `None` | | `project` | `str | None` | The Google Cloud project ID to use for quota. Can be obtained from environment variables (for example, GOOGLE_CLOUD_PROJECT). Applies to the Vertex AI API only. | `None` | | `location` | `VertexAILocation | Literal['global'] | str | None` | The location to send API requests to (for example, us-central1). Can be obtained from environment variables. Applies to the Vertex AI API only. | `None` | | `vertexai` | `bool | None` | Force the use of the Vertex AI API. If False, the Google Generative Language API will be used. Defaults to False unless location, project, or credentials are provided. | `None` | | `client` | `Client | None` | A pre-initialized client to use. | `None` | | `http_client` | `AsyncClient | None` | An existing httpx.AsyncClient to use for making HTTP requests. | `None` | | `base_url` | `str | None` | The base URL for the Google API. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/google.py`

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    credentials: Credentials | None = None,
    project: str | None = None,
    location: VertexAILocation | Literal['global'] | str | None = None,
    vertexai: bool | None = None,
    client: Client | None = None,
    http_client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
) -> None:
    """Create a new Google provider.

    Args:
        api_key: The `API key <https://ai.google.dev/gemini-api/docs/api-key>`_ to
            use for authentication. It can also be set via the `GOOGLE_API_KEY` environment variable.
        credentials: The credentials to use for authentication when calling the Vertex AI APIs. Credentials can be
            obtained from environment variables and default credentials. For more information, see Set up
            Application Default Credentials. Applies to the Vertex AI API only.
        project: The Google Cloud project ID to use for quota. Can be obtained from environment variables
            (for example, GOOGLE_CLOUD_PROJECT). Applies to the Vertex AI API only.
        location: The location to send API requests to (for example, us-central1). Can be obtained from environment variables.
            Applies to the Vertex AI API only.
        vertexai: Force the use of the Vertex AI API. If `False`, the Google Generative Language API will be used.
            Defaults to `False` unless `location`, `project`, or `credentials` are provided.
        client: A pre-initialized client to use.
        http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        base_url: The base URL for the Google API.
    """
    if client is None:
        # NOTE: We are keeping GEMINI_API_KEY for backwards compatibility.
        api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

        vertex_ai_args_used = bool(location or project or credentials)
        if vertexai is None:
            vertexai = vertex_ai_args_used

        http_client = http_client or cached_async_http_client(
            provider='google-vertex' if vertexai else 'google-gla'
        )
        http_options = HttpOptions(
            base_url=base_url,
            headers={'User-Agent': get_user_agent()},
            httpx_async_client=http_client,
        )
        if not vertexai:
            if api_key is None:
                raise UserError(
                    'Set the `GOOGLE_API_KEY` environment variable or pass it via `GoogleProvider(api_key=...)`'
                    'to use the Google Generative Language API.'
                )
            self._client = Client(vertexai=False, api_key=api_key, http_options=http_options)
        else:
            if vertex_ai_args_used:
                api_key = None

            if api_key is None:
                project = project or os.getenv('GOOGLE_CLOUD_PROJECT')
                # From https://github.com/pydantic/pydantic-ai/pull/2031/files#r2169682149:
                # Currently `us-central1` supports the most models by far of any region including `global`, but not
                # all of them. `us-central1` has all google models but is missing some Anthropic partner models,
                # which use `us-east5` instead. `global` has fewer models but higher availability.
                # For more details, check: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
                location = location or os.getenv('GOOGLE_CLOUD_LOCATION') or 'us-central1'

            self._client = Client(
                vertexai=True,
                api_key=api_key,
                project=project,
                location=location,
                credentials=credentials,
                http_options=http_options,
            )
    else:
        self._client = client  # pragma: no cover

```

### VertexAILocation

```python
VertexAILocation = Literal[
    "asia-east1",
    "asia-east2",
    "asia-northeast1",
    "asia-northeast3",
    "asia-south1",
    "asia-southeast1",
    "australia-southeast1",
    "europe-central2",
    "europe-north1",
    "europe-southwest1",
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west6",
    "europe-west8",
    "europe-west9",
    "me-central1",
    "me-central2",
    "me-west1",
    "northamerica-northeast1",
    "southamerica-east1",
    "us-central1",
    "us-east1",
    "us-east4",
    "us-east5",
    "us-south1",
    "us-west1",
    "us-west4",
]

```

Regions available for Vertex AI. More details [here](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#genai-locations).

### OpenAIProvider

Bases: `Provider[AsyncOpenAI]`

Provider for OpenAI API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/openai.py`

```python
class OpenAIProvider(Provider[AsyncOpenAI]):
    """Provider for OpenAI API."""

    @property
    def name(self) -> str:
        return 'openai'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return openai_model_profile(model_name)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new OpenAI provider.

        Args:
            base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to OpenAI's base url.
            api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
        # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
        if api_key is None and 'OPENAI_API_KEY' not in os.environ and base_url is not None and openai_client is None:
            api_key = 'api-key-not-set'

        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='openai')
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

```

#### __init__

```python
__init__(*, openai_client: AsyncOpenAI) -> None

```

```python
__init__(
    base_url: str | None = None,
    api_key: str | None = None,
    openai_client: None = None,
    http_client: AsyncClient | None = None,
) -> None

```

```python
__init__(
    base_url: str | None = None,
    api_key: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: AsyncClient | None = None,
) -> None

```

Create a new OpenAI provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `base_url` | `str | None` | The base url for the OpenAI requests. If not provided, the OPENAI_BASE_URL environment variable will be used if available. Otherwise, defaults to OpenAI's base url. | `None` | | `api_key` | `str | None` | The API key to use for authentication, if not provided, the OPENAI_API_KEY environment variable will be used if available. | `None` | | `openai_client` | `AsyncOpenAI | None` | An existing AsyncOpenAI client to use. If provided, base_url, api_key, and http_client must be None. | `None` | | `http_client` | `AsyncClient | None` | An existing httpx.AsyncClient to use for making HTTP requests. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/openai.py`

```python
def __init__(
    self,
    base_url: str | None = None,
    api_key: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Create a new OpenAI provider.

    Args:
        base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
            will be used if available. Otherwise, defaults to OpenAI's base url.
        api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
            will be used if available.
        openai_client: An existing
            [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
            client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
        http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
    """
    # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
    # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
    if api_key is None and 'OPENAI_API_KEY' not in os.environ and base_url is not None and openai_client is None:
        api_key = 'api-key-not-set'

    if openai_client is not None:
        assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
        assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
        assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
        self._client = openai_client
    elif http_client is not None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    else:
        http_client = cached_async_http_client(provider='openai')
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

```

### DeepSeekProvider

Bases: `Provider[AsyncOpenAI]`

Provider for DeepSeek API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/deepseek.py`

```python
class DeepSeekProvider(Provider[AsyncOpenAI]):
    """Provider for DeepSeek API."""

    @property
    def name(self) -> str:
        return 'deepseek'

    @property
    def base_url(self) -> str:
        return 'https://api.deepseek.com'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = deepseek_model_profile(model_name)

        # As DeepSeekProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly.
        # This was not the case when using a DeepSeek model with another model class (e.g. BedrockConverseModel or GroqModel),
        # so we won't do this in `deepseek_model_profile` unless we learn it's always needed.
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `DEEPSEEK_API_KEY` environment variable or pass it via `DeepSeekProvider(api_key=...)`'
                'to use the DeepSeek provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='deepseek')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

### BedrockModelProfile

Bases: `ModelProfile`

Profile for models used with BedrockModel.

ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

Source code in `pydantic_ai_slim/pydantic_ai/providers/bedrock.py`

```python
@dataclass(kw_only=True)
class BedrockModelProfile(ModelProfile):
    """Profile for models used with BedrockModel.

    ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    bedrock_supports_tool_choice: bool = False
    bedrock_tool_result_format: Literal['text', 'json'] = 'text'
    bedrock_send_back_thinking_parts: bool = False

```

### bedrock_amazon_model_profile

```python
bedrock_amazon_model_profile(
    model_name: str,
) -> ModelProfile | None

```

Get the model profile for an Amazon model used via Bedrock.

Source code in `pydantic_ai_slim/pydantic_ai/providers/bedrock.py`

```python
def bedrock_amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model used via Bedrock."""
    profile = amazon_model_profile(model_name)
    if 'nova' in model_name:
        return BedrockModelProfile(bedrock_supports_tool_choice=True).update(profile)
    return profile

```

### bedrock_deepseek_model_profile

```python
bedrock_deepseek_model_profile(
    model_name: str,
) -> ModelProfile | None

```

Get the model profile for a DeepSeek model used via Bedrock.

Source code in `pydantic_ai_slim/pydantic_ai/providers/bedrock.py`

```python
def bedrock_deepseek_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a DeepSeek model used via Bedrock."""
    profile = deepseek_model_profile(model_name)
    if 'r1' in model_name:
        return BedrockModelProfile(bedrock_send_back_thinking_parts=True).update(profile)
    return profile  # pragma: no cover

```

### BedrockProvider

Bases: `Provider[BaseClient]`

Provider for AWS Bedrock.

Source code in `pydantic_ai_slim/pydantic_ai/providers/bedrock.py`

```python
class BedrockProvider(Provider[BaseClient]):
    """Provider for AWS Bedrock."""

    @property
    def name(self) -> str:
        return 'bedrock'

    @property
    def base_url(self) -> str:
        return self._client.meta.endpoint_url

    @property
    def client(self) -> BaseClient:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile: dict[str, Callable[[str], ModelProfile | None]] = {
            'anthropic': lambda model_name: BedrockModelProfile(
                bedrock_supports_tool_choice=True, bedrock_send_back_thinking_parts=True
            ).update(anthropic_model_profile(model_name)),
            'mistral': lambda model_name: BedrockModelProfile(bedrock_tool_result_format='json').update(
                mistral_model_profile(model_name)
            ),
            'cohere': cohere_model_profile,
            'amazon': bedrock_amazon_model_profile,
            'meta': meta_model_profile,
            'deepseek': bedrock_deepseek_model_profile,
        }

        # Split the model name into parts
        parts = model_name.split('.', 2)

        # Handle regional prefixes (e.g. "us.")
        if len(parts) > 2 and len(parts[0]) == 2:
            parts = parts[1:]

        if len(parts) < 2:
            return None

        provider = parts[0]
        model_name_with_version = parts[1]

        # Remove version suffix if it matches the format (e.g. "-v1:0" or "-v14")
        version_match = re.match(r'(.+)-v\d+(?::\d+)?$', model_name_with_version)
        if version_match:
            model_name = version_match.group(1)
        else:
            model_name = model_name_with_version

        if provider in provider_to_profile:
            return provider_to_profile[provider](model_name)

        return None

    @overload
    def __init__(self, *, bedrock_client: BaseClient) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        base_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        bedrock_client: BaseClient | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        base_url: str | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
        api_key: str | None = None,
        aws_read_timeout: float | None = None,
        aws_connect_timeout: float | None = None,
    ) -> None:
        """Initialize the Bedrock provider.

        Args:
            bedrock_client: A boto3 client for Bedrock Runtime. If provided, other arguments are ignored.
            aws_access_key_id: The AWS access key ID. If not set, the `AWS_ACCESS_KEY_ID` environment variable will be used if available.
            aws_secret_access_key: The AWS secret access key. If not set, the `AWS_SECRET_ACCESS_KEY` environment variable will be used if available.
            aws_session_token: The AWS session token. If not set, the `AWS_SESSION_TOKEN` environment variable will be used if available.
            api_key: The API key for Bedrock client. Can be used instead of `aws_access_key_id`, `aws_secret_access_key`, and `aws_session_token`. If not set, the `AWS_BEARER_TOKEN_BEDROCK` environment variable will be used if available.
            base_url: The base URL for the Bedrock client.
            region_name: The AWS region name. If not set, the `AWS_DEFAULT_REGION` environment variable will be used if available.
            profile_name: The AWS profile name.
            aws_read_timeout: The read timeout for Bedrock client.
            aws_connect_timeout: The connect timeout for Bedrock client.
        """
        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            read_timeout = aws_read_timeout or float(os.getenv('AWS_READ_TIMEOUT', 300))
            connect_timeout = aws_connect_timeout or float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
            config: dict[str, Any] = {
                'read_timeout': read_timeout,
                'connect_timeout': connect_timeout,
            }
            try:
                if api_key is not None:
                    session = boto3.Session(
                        botocore_session=_BearerTokenSession(api_key),
                        region_name=region_name,
                        profile_name=profile_name,
                    )
                    config['signature_version'] = 'bearer'
                else:
                    session = boto3.Session(
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        aws_session_token=aws_session_token,
                        region_name=region_name,
                        profile_name=profile_name,
                    )
                self._client = session.client(  # type: ignore[reportUnknownMemberType]
                    'bedrock-runtime',
                    config=Config(**config),
                    endpoint_url=base_url,
                )
            except NoRegionError as exc:  # pragma: no cover
                raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') from exc

```

#### __init__

```python
__init__(*, bedrock_client: BaseClient) -> None

```

```python
__init__(
    *,
    api_key: str,
    base_url: str | None = None,
    region_name: str | None = None,
    profile_name: str | None = None,
    aws_read_timeout: float | None = None,
    aws_connect_timeout: float | None = None
) -> None

```

```python
__init__(
    *,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    base_url: str | None = None,
    region_name: str | None = None,
    profile_name: str | None = None,
    aws_read_timeout: float | None = None,
    aws_connect_timeout: float | None = None
) -> None

```

```python
__init__(
    *,
    bedrock_client: BaseClient | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    base_url: str | None = None,
    region_name: str | None = None,
    profile_name: str | None = None,
    api_key: str | None = None,
    aws_read_timeout: float | None = None,
    aws_connect_timeout: float | None = None
) -> None

```

Initialize the Bedrock provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `bedrock_client` | `BaseClient | None` | A boto3 client for Bedrock Runtime. If provided, other arguments are ignored. | `None` | | `aws_access_key_id` | `str | None` | The AWS access key ID. If not set, the AWS_ACCESS_KEY_ID environment variable will be used if available. | `None` | | `aws_secret_access_key` | `str | None` | The AWS secret access key. If not set, the AWS_SECRET_ACCESS_KEY environment variable will be used if available. | `None` | | `aws_session_token` | `str | None` | The AWS session token. If not set, the AWS_SESSION_TOKEN environment variable will be used if available. | `None` | | `api_key` | `str | None` | The API key for Bedrock client. Can be used instead of aws_access_key_id, aws_secret_access_key, and aws_session_token. If not set, the AWS_BEARER_TOKEN_BEDROCK environment variable will be used if available. | `None` | | `base_url` | `str | None` | The base URL for the Bedrock client. | `None` | | `region_name` | `str | None` | The AWS region name. If not set, the AWS_DEFAULT_REGION environment variable will be used if available. | `None` | | `profile_name` | `str | None` | The AWS profile name. | `None` | | `aws_read_timeout` | `float | None` | The read timeout for Bedrock client. | `None` | | `aws_connect_timeout` | `float | None` | The connect timeout for Bedrock client. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/bedrock.py`

```python
def __init__(
    self,
    *,
    bedrock_client: BaseClient | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    base_url: str | None = None,
    region_name: str | None = None,
    profile_name: str | None = None,
    api_key: str | None = None,
    aws_read_timeout: float | None = None,
    aws_connect_timeout: float | None = None,
) -> None:
    """Initialize the Bedrock provider.

    Args:
        bedrock_client: A boto3 client for Bedrock Runtime. If provided, other arguments are ignored.
        aws_access_key_id: The AWS access key ID. If not set, the `AWS_ACCESS_KEY_ID` environment variable will be used if available.
        aws_secret_access_key: The AWS secret access key. If not set, the `AWS_SECRET_ACCESS_KEY` environment variable will be used if available.
        aws_session_token: The AWS session token. If not set, the `AWS_SESSION_TOKEN` environment variable will be used if available.
        api_key: The API key for Bedrock client. Can be used instead of `aws_access_key_id`, `aws_secret_access_key`, and `aws_session_token`. If not set, the `AWS_BEARER_TOKEN_BEDROCK` environment variable will be used if available.
        base_url: The base URL for the Bedrock client.
        region_name: The AWS region name. If not set, the `AWS_DEFAULT_REGION` environment variable will be used if available.
        profile_name: The AWS profile name.
        aws_read_timeout: The read timeout for Bedrock client.
        aws_connect_timeout: The connect timeout for Bedrock client.
    """
    if bedrock_client is not None:
        self._client = bedrock_client
    else:
        read_timeout = aws_read_timeout or float(os.getenv('AWS_READ_TIMEOUT', 300))
        connect_timeout = aws_connect_timeout or float(os.getenv('AWS_CONNECT_TIMEOUT', 60))
        config: dict[str, Any] = {
            'read_timeout': read_timeout,
            'connect_timeout': connect_timeout,
        }
        try:
            if api_key is not None:
                session = boto3.Session(
                    botocore_session=_BearerTokenSession(api_key),
                    region_name=region_name,
                    profile_name=profile_name,
                )
                config['signature_version'] = 'bearer'
            else:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=region_name,
                    profile_name=profile_name,
                )
            self._client = session.client(  # type: ignore[reportUnknownMemberType]
                'bedrock-runtime',
                config=Config(**config),
                endpoint_url=base_url,
            )
        except NoRegionError as exc:  # pragma: no cover
            raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') from exc

```

### groq_moonshotai_model_profile

```python
groq_moonshotai_model_profile(
    model_name: str,
) -> ModelProfile | None

```

Get the model profile for an MoonshotAI model used with the Groq provider.

Source code in `pydantic_ai_slim/pydantic_ai/providers/groq.py`

```python
def groq_moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an MoonshotAI model used with the Groq provider."""
    return ModelProfile(supports_json_object_output=True, supports_json_schema_output=True).update(
        moonshotai_model_profile(model_name)
    )

```

### meta_groq_model_profile

```python
meta_groq_model_profile(
    model_name: str,
) -> ModelProfile | None

```

Get the model profile for a Meta model used with the Groq provider.

Source code in `pydantic_ai_slim/pydantic_ai/providers/groq.py`

```python
def meta_groq_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model used with the Groq provider."""
    if model_name in {'llama-4-maverick-17b-128e-instruct', 'llama-4-scout-17b-16e-instruct'}:
        return ModelProfile(supports_json_object_output=True, supports_json_schema_output=True).update(
            meta_model_profile(model_name)
        )
    else:
        return meta_model_profile(model_name)

```

### GroqProvider

Bases: `Provider[AsyncGroq]`

Provider for Groq API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/groq.py`

```python
class GroqProvider(Provider[AsyncGroq]):
    """Provider for Groq API."""

    @property
    def name(self) -> str:
        return 'groq'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncGroq:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        prefix_to_profile = {
            'llama': meta_model_profile,
            'meta-llama/': meta_groq_model_profile,
            'gemma': google_model_profile,
            'qwen': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'moonshotai/': groq_moonshotai_model_profile,
            'compound-': groq_model_profile,
            'openai/': openai_model_profile,
        }

        for prefix, profile_func in prefix_to_profile.items():
            model_name = model_name.lower()
            if model_name.startswith(prefix):
                if prefix.endswith('/'):
                    model_name = model_name[len(prefix) :]
                return profile_func(model_name)

        return None

    @overload
    def __init__(self, *, groq_client: AsyncGroq | None = None) -> None: ...

    @overload
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, http_client: httpx.AsyncClient | None = None
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        groq_client: AsyncGroq | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Groq provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
                will be used if available.
            base_url: The base url for the Groq requests. If not provided, the `GROQ_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to Groq's base url.
            groq_client: An existing
                [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `AsyncHTTPClient` to use for making HTTP requests.
        """
        if groq_client is not None:
            assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `groq_client` and `base_url`'
            self._client = groq_client
        else:
            api_key = api_key or os.getenv('GROQ_API_KEY')
            base_url = base_url or os.getenv('GROQ_BASE_URL', 'https://api.groq.com')

            if not api_key:
                raise UserError(
                    'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                    'to use the Groq provider.'
                )
            elif http_client is not None:
                self._client = AsyncGroq(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='groq')
                self._client = AsyncGroq(base_url=base_url, api_key=api_key, http_client=http_client)

```

#### __init__

```python
__init__(*, groq_client: AsyncGroq | None = None) -> None

```

```python
__init__(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: AsyncClient | None = None
) -> None

```

```python
__init__(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    groq_client: AsyncGroq | None = None,
    http_client: AsyncClient | None = None
) -> None

```

Create a new Groq provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `api_key` | `str | None` | The API key to use for authentication, if not provided, the GROQ_API_KEY environment variable will be used if available. | `None` | | `base_url` | `str | None` | The base url for the Groq requests. If not provided, the GROQ_BASE_URL environment variable will be used if available. Otherwise, defaults to Groq's base url. | `None` | | `groq_client` | `AsyncGroq | None` | An existing AsyncGroq client to use. If provided, api_key and http_client must be None. | `None` | | `http_client` | `AsyncClient | None` | An existing AsyncHTTPClient to use for making HTTP requests. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/groq.py`

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    groq_client: AsyncGroq | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Create a new Groq provider.

    Args:
        api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
            will be used if available.
        base_url: The base url for the Groq requests. If not provided, the `GROQ_BASE_URL` environment variable
            will be used if available. Otherwise, defaults to Groq's base url.
        groq_client: An existing
            [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
            client to use. If provided, `api_key` and `http_client` must be `None`.
        http_client: An existing `AsyncHTTPClient` to use for making HTTP requests.
    """
    if groq_client is not None:
        assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
        assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
        assert base_url is None, 'Cannot provide both `groq_client` and `base_url`'
        self._client = groq_client
    else:
        api_key = api_key or os.getenv('GROQ_API_KEY')
        base_url = base_url or os.getenv('GROQ_BASE_URL', 'https://api.groq.com')

        if not api_key:
            raise UserError(
                'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                'to use the Groq provider.'
            )
        elif http_client is not None:
            self._client = AsyncGroq(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='groq')
            self._client = AsyncGroq(base_url=base_url, api_key=api_key, http_client=http_client)

```

### AzureProvider

Bases: `Provider[AsyncOpenAI]`

Provider for Azure OpenAI API.

See <https://azure.microsoft.com/en-us/products/ai-foundry> for more information.

Source code in `pydantic_ai_slim/pydantic_ai/providers/azure.py`

```python
class AzureProvider(Provider[AsyncOpenAI]):
    """Provider for Azure OpenAI API.

    See <https://azure.microsoft.com/en-us/products/ai-foundry> for more information.
    """

    @property
    def name(self) -> str:
        return 'azure'

    @property
    def base_url(self) -> str:
        assert self._base_url is not None
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        model_name = model_name.lower()

        prefix_to_profile = {
            'llama': meta_model_profile,
            'meta-': meta_model_profile,
            'deepseek': deepseek_model_profile,
            'mistralai-': mistral_model_profile,
            'mistral': mistral_model_profile,
            'cohere-': cohere_model_profile,
            'grok': grok_model_profile,
        }

        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix):
                if prefix.endswith('-'):
                    model_name = model_name[len(prefix) :]

                profile = profile_func(model_name)

                # As AzureProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
                # we need to maintain that behavior unless json_schema_transformer is set explicitly
                return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

        # OpenAI models are unprefixed
        return openai_model_profile(model_name)

    @overload
    def __init__(self, *, openai_client: AsyncAzureOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncAzureOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Azure provider.

        Args:
            azure_endpoint: The Azure endpoint to use for authentication, if not provided, the `AZURE_OPENAI_ENDPOINT`
                environment variable will be used if available.
            api_version: The API version to use for authentication, if not provided, the `OPENAI_API_VERSION`
                environment variable will be used if available.
            api_key: The API key to use for authentication, if not provided, the `AZURE_OPENAI_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncAzureOpenAI`](https://github.com/openai/openai-python#microsoft-azure-openai)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            assert azure_endpoint is None, 'Cannot provide both `openai_client` and `azure_endpoint`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._base_url = str(openai_client.base_url)
            self._client = openai_client
        else:
            azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
            if not azure_endpoint:
                raise UserError(
                    'Must provide one of the `azure_endpoint` argument or the `AZURE_OPENAI_ENDPOINT` environment variable'
                )

            if not api_key and 'AZURE_OPENAI_API_KEY' not in os.environ:  # pragma: no cover
                raise UserError(
                    'Must provide one of the `api_key` argument or the `AZURE_OPENAI_API_KEY` environment variable'
                )

            if not api_version and 'OPENAI_API_VERSION' not in os.environ:  # pragma: no cover
                raise UserError(
                    'Must provide one of the `api_version` argument or the `OPENAI_API_VERSION` environment variable'
                )

            http_client = http_client or cached_async_http_client(provider='azure')
            self._client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                http_client=http_client,
            )
            self._base_url = str(self._client.base_url)

```

#### __init__

```python
__init__(*, openai_client: AsyncAzureOpenAI) -> None

```

```python
__init__(
    *,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    http_client: AsyncClient | None = None
) -> None

```

```python
__init__(
    *,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    openai_client: AsyncAzureOpenAI | None = None,
    http_client: AsyncClient | None = None
) -> None

```

Create a new Azure provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `azure_endpoint` | `str | None` | The Azure endpoint to use for authentication, if not provided, the AZURE_OPENAI_ENDPOINT environment variable will be used if available. | `None` | | `api_version` | `str | None` | The API version to use for authentication, if not provided, the OPENAI_API_VERSION environment variable will be used if available. | `None` | | `api_key` | `str | None` | The API key to use for authentication, if not provided, the AZURE_OPENAI_API_KEY environment variable will be used if available. | `None` | | `openai_client` | `AsyncAzureOpenAI | None` | An existing AsyncAzureOpenAI client to use. If provided, base_url, api_key, and http_client must be None. | `None` | | `http_client` | `AsyncClient | None` | An existing httpx.AsyncClient to use for making HTTP requests. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/azure.py`

```python
def __init__(
    self,
    *,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    openai_client: AsyncAzureOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Create a new Azure provider.

    Args:
        azure_endpoint: The Azure endpoint to use for authentication, if not provided, the `AZURE_OPENAI_ENDPOINT`
            environment variable will be used if available.
        api_version: The API version to use for authentication, if not provided, the `OPENAI_API_VERSION`
            environment variable will be used if available.
        api_key: The API key to use for authentication, if not provided, the `AZURE_OPENAI_API_KEY` environment variable
            will be used if available.
        openai_client: An existing
            [`AsyncAzureOpenAI`](https://github.com/openai/openai-python#microsoft-azure-openai)
            client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
        http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
    """
    if openai_client is not None:
        assert azure_endpoint is None, 'Cannot provide both `openai_client` and `azure_endpoint`'
        assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
        assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
        self._base_url = str(openai_client.base_url)
        self._client = openai_client
    else:
        azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        if not azure_endpoint:
            raise UserError(
                'Must provide one of the `azure_endpoint` argument or the `AZURE_OPENAI_ENDPOINT` environment variable'
            )

        if not api_key and 'AZURE_OPENAI_API_KEY' not in os.environ:  # pragma: no cover
            raise UserError(
                'Must provide one of the `api_key` argument or the `AZURE_OPENAI_API_KEY` environment variable'
            )

        if not api_version and 'OPENAI_API_VERSION' not in os.environ:  # pragma: no cover
            raise UserError(
                'Must provide one of the `api_version` argument or the `OPENAI_API_VERSION` environment variable'
            )

        http_client = http_client or cached_async_http_client(provider='azure')
        self._client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            http_client=http_client,
        )
        self._base_url = str(self._client.base_url)

```

### CohereProvider

Bases: `Provider[AsyncClientV2]`

Provider for Cohere API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/cohere.py`

```python
class CohereProvider(Provider[AsyncClientV2]):
    """Provider for Cohere API."""

    @property
    def name(self) -> str:
        return 'cohere'

    @property
    def base_url(self) -> str:
        client_wrapper = self.client._client_wrapper  # type: ignore
        return str(client_wrapper.get_base_url())

    @property
    def client(self) -> AsyncClientV2:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return cohere_model_profile(model_name)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cohere_client: AsyncClientV2 | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Cohere provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `CO_API_KEY` environment variable
                will be used if available.
            cohere_client: An existing
                [AsyncClientV2](https://github.com/cohere-ai/cohere-python)
                client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if cohere_client is not None:
            assert http_client is None, 'Cannot provide both `cohere_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `cohere_client` and `api_key`'
            self._client = cohere_client
        else:
            api_key = api_key or os.getenv('CO_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `CO_API_KEY` environment variable or pass it via `CohereProvider(api_key=...)`'
                    'to use the Cohere provider.'
                )

            base_url = os.getenv('CO_BASE_URL')
            if http_client is not None:
                self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)
            else:
                http_client = cached_async_http_client(provider='cohere')
                self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)

```

#### __init__

```python
__init__(
    *,
    api_key: str | None = None,
    cohere_client: AsyncClientV2 | None = None,
    http_client: AsyncClient | None = None
) -> None

```

Create a new Cohere provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `api_key` | `str | None` | The API key to use for authentication, if not provided, the CO_API_KEY environment variable will be used if available. | `None` | | `cohere_client` | `AsyncClientV2 | None` | An existing AsyncClientV2 client to use. If provided, api_key and http_client must be None. | `None` | | `http_client` | `AsyncClient | None` | An existing httpx.AsyncClient to use for making HTTP requests. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/cohere.py`

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    cohere_client: AsyncClientV2 | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Create a new Cohere provider.

    Args:
        api_key: The API key to use for authentication, if not provided, the `CO_API_KEY` environment variable
            will be used if available.
        cohere_client: An existing
            [AsyncClientV2](https://github.com/cohere-ai/cohere-python)
            client to use. If provided, `api_key` and `http_client` must be `None`.
        http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
    """
    if cohere_client is not None:
        assert http_client is None, 'Cannot provide both `cohere_client` and `http_client`'
        assert api_key is None, 'Cannot provide both `cohere_client` and `api_key`'
        self._client = cohere_client
    else:
        api_key = api_key or os.getenv('CO_API_KEY')
        if not api_key:
            raise UserError(
                'Set the `CO_API_KEY` environment variable or pass it via `CohereProvider(api_key=...)`'
                'to use the Cohere provider.'
            )

        base_url = os.getenv('CO_BASE_URL')
        if http_client is not None:
            self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)
        else:
            http_client = cached_async_http_client(provider='cohere')
            self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)

```

Bases: `Provider[AsyncOpenAI]`

Provider for Cerebras API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/cerebras.py`

```python
class CerebrasProvider(Provider[AsyncOpenAI]):
    """Provider for Cerebras API."""

    @property
    def name(self) -> str:
        return 'cerebras'

    @property
    def base_url(self) -> str:
        return 'https://api.cerebras.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        prefix_to_profile = {'llama': meta_model_profile, 'qwen': qwen_model_profile, 'gpt-oss': harmony_model_profile}

        profile = None
        for prefix, profile_func in prefix_to_profile.items():
            model_name = model_name.lower()
            if model_name.startswith(prefix):
                profile = profile_func(model_name)

        # According to https://inference-docs.cerebras.ai/resources/openai#currently-unsupported-openai-features,
        # Cerebras doesn't support some model settings.
        unsupported_model_settings = (
            'frequency_penalty',
            'logit_bias',
            'presence_penalty',
            'parallel_tool_calls',
            'service_tier',
        )
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_unsupported_model_settings=unsupported_model_settings,
        ).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('CEREBRAS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `CEREBRAS_API_KEY` environment variable or pass it via `CerebrasProvider(api_key=...)` '
                'to use the Cerebras provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='cerebras')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

Bases: `Provider[Mistral]`

Provider for Mistral API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/mistral.py`

```python
class MistralProvider(Provider[Mistral]):
    """Provider for Mistral API."""

    @property
    def name(self) -> str:
        return 'mistral'

    @property
    def base_url(self) -> str:
        return self.client.sdk_configuration.get_server_details()[0]

    @property
    def client(self) -> Mistral:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return mistral_model_profile(model_name)

    @overload
    def __init__(self, *, mistral_client: Mistral | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str | None = None, http_client: httpx.AsyncClient | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mistral_client: Mistral | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Mistral provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `MISTRAL_API_KEY` environment variable
                will be used if available.
            mistral_client: An existing `Mistral` client to use, if provided, `api_key` and `http_client` must be `None`.
            base_url: The base url for the Mistral requests.
            http_client: An existing async client to use for making HTTP requests.
        """
        if mistral_client is not None:
            assert http_client is None, 'Cannot provide both `mistral_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `mistral_client` and `api_key`'
            assert base_url is None, 'Cannot provide both `mistral_client` and `base_url`'
            self._client = mistral_client
        else:
            api_key = api_key or os.getenv('MISTRAL_API_KEY')

            if not api_key:
                raise UserError(
                    'Set the `MISTRAL_API_KEY` environment variable or pass it via `MistralProvider(api_key=...)`'
                    'to use the Mistral provider.'
                )
            elif http_client is not None:
                self._client = Mistral(api_key=api_key, async_client=http_client, server_url=base_url)
            else:
                http_client = cached_async_http_client(provider='mistral')
                self._client = Mistral(api_key=api_key, async_client=http_client, server_url=base_url)

```

### __init__

```python
__init__(*, mistral_client: Mistral | None = None) -> None

```

```python
__init__(
    *,
    api_key: str | None = None,
    http_client: AsyncClient | None = None
) -> None

```

```python
__init__(
    *,
    api_key: str | None = None,
    mistral_client: Mistral | None = None,
    base_url: str | None = None,
    http_client: AsyncClient | None = None
) -> None

```

Create a new Mistral provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `api_key` | `str | None` | The API key to use for authentication, if not provided, the MISTRAL_API_KEY environment variable will be used if available. | `None` | | `mistral_client` | `Mistral | None` | An existing Mistral client to use, if provided, api_key and http_client must be None. | `None` | | `base_url` | `str | None` | The base url for the Mistral requests. | `None` | | `http_client` | `AsyncClient | None` | An existing async client to use for making HTTP requests. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/mistral.py`

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    mistral_client: Mistral | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Create a new Mistral provider.

    Args:
        api_key: The API key to use for authentication, if not provided, the `MISTRAL_API_KEY` environment variable
            will be used if available.
        mistral_client: An existing `Mistral` client to use, if provided, `api_key` and `http_client` must be `None`.
        base_url: The base url for the Mistral requests.
        http_client: An existing async client to use for making HTTP requests.
    """
    if mistral_client is not None:
        assert http_client is None, 'Cannot provide both `mistral_client` and `http_client`'
        assert api_key is None, 'Cannot provide both `mistral_client` and `api_key`'
        assert base_url is None, 'Cannot provide both `mistral_client` and `base_url`'
        self._client = mistral_client
    else:
        api_key = api_key or os.getenv('MISTRAL_API_KEY')

        if not api_key:
            raise UserError(
                'Set the `MISTRAL_API_KEY` environment variable or pass it via `MistralProvider(api_key=...)`'
                'to use the Mistral provider.'
            )
        elif http_client is not None:
            self._client = Mistral(api_key=api_key, async_client=http_client, server_url=base_url)
        else:
            http_client = cached_async_http_client(provider='mistral')
            self._client = Mistral(api_key=api_key, async_client=http_client, server_url=base_url)

```

Bases: `Provider[AsyncOpenAI]`

Provider for Fireworks AI API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/fireworks.py`

```python
class FireworksProvider(Provider[AsyncOpenAI]):
    """Provider for Fireworks AI API."""

    @property
    def name(self) -> str:
        return 'fireworks'

    @property
    def base_url(self) -> str:
        return 'https://api.fireworks.ai/inference/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        prefix_to_profile = {
            'llama': meta_model_profile,
            'qwen': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'gemma': google_model_profile,
        }

        prefix = 'accounts/fireworks/models/'

        profile = None
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix) :]
            for provider, profile_func in prefix_to_profile.items():
                if model_name.startswith(provider):
                    profile = profile_func(model_name)
                    break

        # As the Fireworks API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('FIREWORKS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `FIREWORKS_API_KEY` environment variable or pass it via `FireworksProvider(api_key=...)`'
                'to use the Fireworks AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='fireworks')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

Bases: `Provider[AsyncOpenAI]`

Provider for Grok API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/grok.py`

```python
class GrokProvider(Provider[AsyncOpenAI]):
    """Provider for Grok API."""

    @property
    def name(self) -> str:
        return 'grok'

    @property
    def base_url(self) -> str:
        return 'https://api.x.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = grok_model_profile(model_name)

        # As the Grok API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly.
        # Also, Grok does not support strict tool definitions: https://github.com/pydantic/pydantic-ai/issues/1846
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer, openai_supports_strict_tool_definition=False
        ).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('GROK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `GROK_API_KEY` environment variable or pass it via `GrokProvider(api_key=...)`'
                'to use the Grok provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='grok')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

Bases: `Provider[AsyncOpenAI]`

Provider for Together AI API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/together.py`

```python
class TogetherProvider(Provider[AsyncOpenAI]):
    """Provider for Together AI API."""

    @property
    def name(self) -> str:
        return 'together'

    @property
    def base_url(self) -> str:
        return 'https://api.together.xyz/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'deepseek-ai': deepseek_model_profile,
            'google': google_model_profile,
            'qwen': qwen_model_profile,
            'meta-llama': meta_model_profile,
            'mistralai': mistral_model_profile,
        }

        profile = None

        model_name = model_name.lower()
        provider, model_name = model_name.split('/', 1)
        if provider in provider_to_profile:
            profile = provider_to_profile[provider](model_name)

        # As the Together API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('TOGETHER_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `TOGETHER_API_KEY` environment variable or pass it via `TogetherProvider(api_key=...)`'
                'to use the Together AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='together')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

Bases: `Provider[AsyncOpenAI]`

Provider for Heroku API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/heroku.py`

```python
class HerokuProvider(Provider[AsyncOpenAI]):
    """Provider for Heroku API."""

    @property
    def name(self) -> str:
        return 'heroku'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        # As the Heroku API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer.
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            api_key = api_key or os.getenv('HEROKU_INFERENCE_KEY')
            if not api_key:
                raise UserError(
                    'Set the `HEROKU_INFERENCE_KEY` environment variable or pass it via `HerokuProvider(api_key=...)`'
                    'to use the Heroku provider.'
                )

            base_url = base_url or os.getenv('HEROKU_INFERENCE_URL', 'https://us.inference.heroku.com')
            base_url = base_url.rstrip('/') + '/v1'

            if http_client is not None:
                self._client = AsyncOpenAI(api_key=api_key, http_client=http_client, base_url=base_url)
            else:
                http_client = cached_async_http_client(provider='heroku')
                self._client = AsyncOpenAI(api_key=api_key, http_client=http_client, base_url=base_url)

```

Bases: `Provider[AsyncOpenAI]`

Provider for GitHub Models API.

GitHub Models provides access to various AI models through an OpenAI-compatible API. See <https://docs.github.com/en/github-models> for more information.

Source code in `pydantic_ai_slim/pydantic_ai/providers/github.py`

```python
class GitHubProvider(Provider[AsyncOpenAI]):
    """Provider for GitHub Models API.

    GitHub Models provides access to various AI models through an OpenAI-compatible API.
    See <https://docs.github.com/en/github-models> for more information.
    """

    @property
    def name(self) -> str:
        return 'github'

    @property
    def base_url(self) -> str:
        return 'https://models.github.ai/inference'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'xai': grok_model_profile,
            'meta': meta_model_profile,
            'microsoft': openai_model_profile,
            'mistral-ai': mistral_model_profile,
            'cohere': cohere_model_profile,
            'deepseek': deepseek_model_profile,
        }

        profile = None

        # If the model name does not contain a provider prefix, we assume it's an OpenAI model
        if '/' not in model_name:
            return openai_model_profile(model_name)

        provider, model_name = model_name.lower().split('/', 1)
        if provider in provider_to_profile:
            model_name, *_ = model_name.split(':', 1)  # drop tags
            profile = provider_to_profile[provider](model_name)

        # As GitHubProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new GitHub Models provider.

        Args:
            api_key: The GitHub token to use for authentication. If not provided, the `GITHUB_API_KEY`
                environment variable will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('GITHUB_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `GITHUB_API_KEY` environment variable or pass it via `GitHubProvider(api_key=...)`'
                ' to use the GitHub Models provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='github')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

### __init__

```python
__init__() -> None

```

```python
__init__(*, api_key: str) -> None

```

```python
__init__(*, api_key: str, http_client: AsyncClient) -> None

```

```python
__init__(
    *, openai_client: AsyncOpenAI | None = None
) -> None

```

```python
__init__(
    *,
    api_key: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: AsyncClient | None = None
) -> None

```

Create a new GitHub Models provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `api_key` | `str | None` | The GitHub token to use for authentication. If not provided, the GITHUB_API_KEY environment variable will be used if available. | `None` | | `openai_client` | `AsyncOpenAI | None` | An existing AsyncOpenAI client to use. If provided, api_key and http_client must be None. | `None` | | `http_client` | `AsyncClient | None` | An existing httpx.AsyncClient to use for making HTTP requests. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/github.py`

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Create a new GitHub Models provider.

    Args:
        api_key: The GitHub token to use for authentication. If not provided, the `GITHUB_API_KEY`
            environment variable will be used if available.
        openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client` must be `None`.
        http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
    """
    api_key = api_key or os.getenv('GITHUB_API_KEY')
    if not api_key and openai_client is None:
        raise UserError(
            'Set the `GITHUB_API_KEY` environment variable or pass it via `GitHubProvider(api_key=...)`'
            ' to use the GitHub Models provider.'
        )

    if openai_client is not None:
        self._client = openai_client
    elif http_client is not None:
        self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
    else:
        http_client = cached_async_http_client(provider='github')
        self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

Bases: `Provider[AsyncOpenAI]`

Provider for OpenRouter API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/openrouter.py`

```python
class OpenRouterProvider(Provider[AsyncOpenAI]):
    """Provider for OpenRouter API."""

    @property
    def name(self) -> str:
        return 'openrouter'

    @property
    def base_url(self) -> str:
        return 'https://openrouter.ai/api/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'google': google_model_profile,
            'openai': openai_model_profile,
            'anthropic': anthropic_model_profile,
            'mistralai': mistral_model_profile,
            'qwen': qwen_model_profile,
            'x-ai': grok_model_profile,
            'cohere': cohere_model_profile,
            'amazon': amazon_model_profile,
            'deepseek': deepseek_model_profile,
            'meta-llama': meta_model_profile,
            'moonshotai': moonshotai_model_profile,
        }

        profile = None

        provider, model_name = model_name.split('/', 1)
        if provider in provider_to_profile:
            model_name, *_ = model_name.split(':', 1)  # drop tags
            profile = provider_to_profile[provider](model_name)

        # As OpenRouterProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, api_key: str, app_url: str, app_title: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, app_url: str, app_title: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_url: str | None = None,
        app_title: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the provider with either an API key or prebuilt client.

        Args:
            api_key: OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``
                when omitted and required unless ``openai_client`` is provided.
            app_url: Optional url for app attribution. Falls back to
                ``OPENROUTER_APP_URL`` when omitted.
            app_title: Optional title for app attribution. Falls back to
                ``OPENROUTER_APP_TITLE`` when omitted.
            openai_client: Existing ``AsyncOpenAI`` client to reuse instead of
                creating one internally.
            http_client: Custom ``httpx.AsyncClient`` to pass into the
                ``AsyncOpenAI`` constructor when building a client.

        Raises:
            UserError: If no API key is available and no ``openai_client`` is
                provided.
        """
        api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `OPENROUTER_API_KEY` environment variable or pass it via `OpenRouterProvider(api_key=...)`'
                'to use the OpenRouter provider.'
            )

        attribution_headers: dict[str, str] = {}
        if http_referer := app_url or os.getenv('OPENROUTER_APP_URL'):
            attribution_headers['HTTP-Referer'] = http_referer
        if x_title := app_title or os.getenv('OPENROUTER_APP_TITLE'):
            attribution_headers['X-Title'] = x_title

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=attribution_headers
            )
        else:
            http_client = cached_async_http_client(provider='openrouter')
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=attribution_headers
            )

```

### __init__

```python
__init__() -> None

```

```python
__init__(*, api_key: str) -> None

```

```python
__init__(*, api_key: str, http_client: AsyncClient) -> None

```

```python
__init__(
    *, api_key: str, app_url: str, app_title: str
) -> None

```

```python
__init__(
    *,
    api_key: str,
    app_url: str,
    app_title: str,
    http_client: AsyncClient
) -> None

```

```python
__init__(*, http_client: AsyncClient) -> None

```

```python
__init__(
    *, openai_client: AsyncOpenAI | None = None
) -> None

```

```python
__init__(
    *,
    api_key: str | None = None,
    app_url: str | None = None,
    app_title: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: AsyncClient | None = None
) -> None

```

Configure the provider with either an API key or prebuilt client.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `api_key` | `str | None` | OpenRouter API key. Falls back to OPENROUTER_API_KEY when omitted and required unless openai_client is provided. | `None` | | `app_url` | `str | None` | Optional url for app attribution. Falls back to OPENROUTER_APP_URL when omitted. | `None` | | `app_title` | `str | None` | Optional title for app attribution. Falls back to OPENROUTER_APP_TITLE when omitted. | `None` | | `openai_client` | `AsyncOpenAI | None` | Existing AsyncOpenAI client to reuse instead of creating one internally. | `None` | | `http_client` | `AsyncClient | None` | Custom httpx.AsyncClient to pass into the AsyncOpenAI constructor when building a client. | `None` |

Raises:

| Type | Description | | --- | --- | | `UserError` | If no API key is available and no openai_client is provided. |

Source code in `pydantic_ai_slim/pydantic_ai/providers/openrouter.py`

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    app_url: str | None = None,
    app_title: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Configure the provider with either an API key or prebuilt client.

    Args:
        api_key: OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``
            when omitted and required unless ``openai_client`` is provided.
        app_url: Optional url for app attribution. Falls back to
            ``OPENROUTER_APP_URL`` when omitted.
        app_title: Optional title for app attribution. Falls back to
            ``OPENROUTER_APP_TITLE`` when omitted.
        openai_client: Existing ``AsyncOpenAI`` client to reuse instead of
            creating one internally.
        http_client: Custom ``httpx.AsyncClient`` to pass into the
            ``AsyncOpenAI`` constructor when building a client.

    Raises:
        UserError: If no API key is available and no ``openai_client`` is
            provided.
    """
    api_key = api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key and openai_client is None:
        raise UserError(
            'Set the `OPENROUTER_API_KEY` environment variable or pass it via `OpenRouterProvider(api_key=...)`'
            'to use the OpenRouter provider.'
        )

    attribution_headers: dict[str, str] = {}
    if http_referer := app_url or os.getenv('OPENROUTER_APP_URL'):
        attribution_headers['HTTP-Referer'] = http_referer
    if x_title := app_title or os.getenv('OPENROUTER_APP_TITLE'):
        attribution_headers['X-Title'] = x_title

    if openai_client is not None:
        self._client = openai_client
    elif http_client is not None:
        self._client = AsyncOpenAI(
            base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=attribution_headers
        )
    else:
        http_client = cached_async_http_client(provider='openrouter')
        self._client = AsyncOpenAI(
            base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=attribution_headers
        )

```

Bases: `Provider[AsyncOpenAI]`

Provider for Vercel AI Gateway API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/vercel.py`

```python
class VercelProvider(Provider[AsyncOpenAI]):
    """Provider for Vercel AI Gateway API."""

    @property
    def name(self) -> str:
        return 'vercel'

    @property
    def base_url(self) -> str:
        return 'https://ai-gateway.vercel.sh/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'bedrock': amazon_model_profile,
            'cohere': cohere_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'openai': openai_model_profile,
            'vertex': google_model_profile,
            'xai': grok_model_profile,
        }

        profile = None

        try:
            provider, model_name = model_name.split('/', 1)
        except ValueError:
            raise UserError(f"Model name must be in 'provider/model' format, got: {model_name!r}")

        if provider in provider_to_profile:
            profile = provider_to_profile[provider](model_name)

        # As VercelProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
        ).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # Support Vercel AI Gateway's standard environment variables
        api_key = api_key or os.getenv('VERCEL_AI_GATEWAY_API_KEY') or os.getenv('VERCEL_OIDC_TOKEN')

        if not api_key and openai_client is None:
            raise UserError(
                'Set the `VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN` environment variable '
                'or pass the API key via `VercelProvider(api_key=...)` to use the Vercel provider.'
            )

        default_headers = {'http-referer': 'https://ai.pydantic.dev/', 'x-title': 'pydantic-ai'}

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
        else:
            http_client = cached_async_http_client(provider='vercel')
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )

```

Bases: `Provider[AsyncInferenceClient]`

Provider for Hugging Face.

Source code in `pydantic_ai_slim/pydantic_ai/providers/huggingface.py`

```python
class HuggingFaceProvider(Provider[AsyncInferenceClient]):
    """Provider for Hugging Face."""

    @property
    def name(self) -> str:
        return 'huggingface'

    @property
    def base_url(self) -> str:
        return self.client.model  # type: ignore

    @property
    def client(self) -> AsyncInferenceClient:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'deepseek-ai': deepseek_model_profile,
            'google': google_model_profile,
            'qwen': qwen_model_profile,
            'meta-llama': meta_model_profile,
            'mistralai': mistral_model_profile,
            'moonshotai': moonshotai_model_profile,
        }

        if '/' not in model_name:
            return None

        model_name = model_name.lower()
        provider, model_name = model_name.split('/', 1)
        if provider in provider_to_profile:
            return provider_to_profile[provider](model_name)

        return None

    @overload
    def __init__(self, *, base_url: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, provider_name: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, hf_client: AsyncInferenceClient, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, hf_client: AsyncInferenceClient, base_url: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, hf_client: AsyncInferenceClient, provider_name: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, api_key: str | None = None) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        hf_client: AsyncInferenceClient | None = None,
        http_client: AsyncClient | None = None,
        provider_name: str | None = None,
    ) -> None:
        """Create a new Hugging Face provider.

        Args:
            base_url: The base url for the Hugging Face requests.
            api_key: The API key to use for authentication, if not provided, the `HF_TOKEN` environment variable
                will be used if available.
            hf_client: An existing
                [`AsyncInferenceClient`](https://huggingface.co/docs/huggingface_hub/v0.29.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)
                client to use. If not provided, a new instance will be created.
            http_client: (currently ignored) An existing `httpx.AsyncClient` to use for making HTTP requests.
            provider_name : Name of the provider to use for inference. available providers can be found in the [HF Inference Providers documentation](https://huggingface.co/docs/inference-providers/index#partners).
                defaults to "auto", which will select the first available provider for the model, the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers.
                If `base_url` is passed, then `provider_name` is not used.
        """
        api_key = api_key or os.getenv('HF_TOKEN')

        if api_key is None:
            raise UserError(
                'Set the `HF_TOKEN` environment variable or pass it via `HuggingFaceProvider(api_key=...)`'
                'to use the HuggingFace provider.'
            )

        if http_client is not None:
            raise ValueError('`http_client` is ignored for HuggingFace provider, please use `hf_client` instead.')

        if base_url is not None and provider_name is not None:
            raise ValueError('Cannot provide both `base_url` and `provider_name`.')

        if hf_client is None:
            self._client = AsyncInferenceClient(api_key=api_key, provider=provider_name, base_url=base_url)  # type: ignore
        else:
            self._client = hf_client

```

### __init__

```python
__init__(
    *, base_url: str, api_key: str | None = None
) -> None

```

```python
__init__(
    *, provider_name: str, api_key: str | None = None
) -> None

```

```python
__init__(
    *,
    hf_client: AsyncInferenceClient,
    api_key: str | None = None
) -> None

```

```python
__init__(
    *,
    hf_client: AsyncInferenceClient,
    base_url: str,
    api_key: str | None = None
) -> None

```

```python
__init__(
    *,
    hf_client: AsyncInferenceClient,
    provider_name: str,
    api_key: str | None = None
) -> None

```

```python
__init__(*, api_key: str | None = None) -> None

```

```python
__init__(
    base_url: str | None = None,
    api_key: str | None = None,
    hf_client: AsyncInferenceClient | None = None,
    http_client: AsyncClient | None = None,
    provider_name: str | None = None,
) -> None

```

Create a new Hugging Face provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `base_url` | `str | None` | The base url for the Hugging Face requests. | `None` | | `api_key` | `str | None` | The API key to use for authentication, if not provided, the HF_TOKEN environment variable will be used if available. | `None` | | `hf_client` | `AsyncInferenceClient | None` | An existing AsyncInferenceClient client to use. If not provided, a new instance will be created. | `None` | | `http_client` | `AsyncClient | None` | (currently ignored) An existing httpx.AsyncClient to use for making HTTP requests. | `None` | | `provider_name` | | Name of the provider to use for inference. available providers can be found in the HF Inference Providers documentation. defaults to "auto", which will select the first available provider for the model, the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers. If base_url is passed, then provider_name is not used. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/huggingface.py`

```python
def __init__(
    self,
    base_url: str | None = None,
    api_key: str | None = None,
    hf_client: AsyncInferenceClient | None = None,
    http_client: AsyncClient | None = None,
    provider_name: str | None = None,
) -> None:
    """Create a new Hugging Face provider.

    Args:
        base_url: The base url for the Hugging Face requests.
        api_key: The API key to use for authentication, if not provided, the `HF_TOKEN` environment variable
            will be used if available.
        hf_client: An existing
            [`AsyncInferenceClient`](https://huggingface.co/docs/huggingface_hub/v0.29.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)
            client to use. If not provided, a new instance will be created.
        http_client: (currently ignored) An existing `httpx.AsyncClient` to use for making HTTP requests.
        provider_name : Name of the provider to use for inference. available providers can be found in the [HF Inference Providers documentation](https://huggingface.co/docs/inference-providers/index#partners).
            defaults to "auto", which will select the first available provider for the model, the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers.
            If `base_url` is passed, then `provider_name` is not used.
    """
    api_key = api_key or os.getenv('HF_TOKEN')

    if api_key is None:
        raise UserError(
            'Set the `HF_TOKEN` environment variable or pass it via `HuggingFaceProvider(api_key=...)`'
            'to use the HuggingFace provider.'
        )

    if http_client is not None:
        raise ValueError('`http_client` is ignored for HuggingFace provider, please use `hf_client` instead.')

    if base_url is not None and provider_name is not None:
        raise ValueError('Cannot provide both `base_url` and `provider_name`.')

    if hf_client is None:
        self._client = AsyncInferenceClient(api_key=api_key, provider=provider_name, base_url=base_url)  # type: ignore
    else:
        self._client = hf_client

```

Bases: `Provider[AsyncOpenAI]`

Provider for MoonshotAI platform (Kimi models).

Source code in `pydantic_ai_slim/pydantic_ai/providers/moonshotai.py`

```python
class MoonshotAIProvider(Provider[AsyncOpenAI]):
    """Provider for MoonshotAI platform (Kimi models)."""

    @property
    def name(self) -> str:
        return 'moonshotai'

    @property
    def base_url(self) -> str:
        # OpenAI-compatible endpoint, see MoonshotAI docs
        return 'https://api.moonshot.ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = moonshotai_model_profile(model_name)

        # As the MoonshotAI API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly.
        # Also, MoonshotAI does not support strict tool definitions
        # https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-tool_choice
        # "Please note that the current version of Kimi API does not support the tool_choice=required parameter."
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_supports_tool_choice_required=False,
            supports_json_object_output=True,
        ).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('MOONSHOTAI_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `MOONSHOTAI_API_KEY` environment variable or pass it via '
                '`MoonshotAIProvider(api_key=...)` to use the MoonshotAI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='moonshotai')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

Bases: `Provider[AsyncOpenAI]`

Provider for local or remote Ollama API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/ollama.py`

```python
class OllamaProvider(Provider[AsyncOpenAI]):
    """Provider for local or remote Ollama API."""

    @property
    def name(self) -> str:
        return 'ollama'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        prefix_to_profile = {
            'llama': meta_model_profile,
            'gemma': google_model_profile,
            'qwen': qwen_model_profile,
            'qwq': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'command': cohere_model_profile,
            'gpt-oss': harmony_model_profile,
        }

        profile = None
        for prefix, profile_func in prefix_to_profile.items():
            model_name = model_name.lower()
            if model_name.startswith(prefix):
                profile = profile_func(model_name)

        # As OllamaProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Ollama provider.

        Args:
            base_url: The base url for the Ollama requests. If not provided, the `OLLAMA_BASE_URL` environment variable
                will be used if available.
            api_key: The API key to use for authentication, if not provided, the `OLLAMA_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            base_url = base_url or os.getenv('OLLAMA_BASE_URL')
            if not base_url:
                raise UserError(
                    'Set the `OLLAMA_BASE_URL` environment variable or pass it via `OllamaProvider(base_url=...)`'
                    'to use the Ollama provider.'
                )

            # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
            # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
            api_key = api_key or os.getenv('OLLAMA_API_KEY') or 'api-key-not-set'

            if http_client is not None:
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='ollama')
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

```

### __init__

```python
__init__(
    base_url: str | None = None,
    api_key: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: AsyncClient | None = None,
) -> None

```

Create a new Ollama provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `base_url` | `str | None` | The base url for the Ollama requests. If not provided, the OLLAMA_BASE_URL environment variable will be used if available. | `None` | | `api_key` | `str | None` | The API key to use for authentication, if not provided, the OLLAMA_API_KEY environment variable will be used if available. | `None` | | `openai_client` | `AsyncOpenAI | None` | An existing AsyncOpenAI client to use. If provided, base_url, api_key, and http_client must be None. | `None` | | `http_client` | `AsyncClient | None` | An existing httpx.AsyncClient to use for making HTTP requests. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/ollama.py`

```python
def __init__(
    self,
    base_url: str | None = None,
    api_key: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Create a new Ollama provider.

    Args:
        base_url: The base url for the Ollama requests. If not provided, the `OLLAMA_BASE_URL` environment variable
            will be used if available.
        api_key: The API key to use for authentication, if not provided, the `OLLAMA_API_KEY` environment variable
            will be used if available.
        openai_client: An existing
            [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
            client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
        http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
    """
    if openai_client is not None:
        assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
        assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
        assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
        self._client = openai_client
    else:
        base_url = base_url or os.getenv('OLLAMA_BASE_URL')
        if not base_url:
            raise UserError(
                'Set the `OLLAMA_BASE_URL` environment variable or pass it via `OllamaProvider(base_url=...)`'
                'to use the Ollama provider.'
            )

        # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
        # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
        api_key = api_key or os.getenv('OLLAMA_API_KEY') or 'api-key-not-set'

        if http_client is not None:
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='ollama')
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

```

Bases: `Provider[AsyncOpenAI]`

Provider for LiteLLM API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/litellm.py`

```python
class LiteLLMProvider(Provider[AsyncOpenAI]):
    """Provider for LiteLLM API."""

    @property
    def name(self) -> str:
        return 'litellm'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        # Map provider prefixes to their profile functions
        provider_to_profile = {
            'anthropic': anthropic_model_profile,
            'openai': openai_model_profile,
            'google': google_model_profile,
            'mistralai': mistral_model_profile,
            'mistral': mistral_model_profile,
            'cohere': cohere_model_profile,
            'amazon': amazon_model_profile,
            'bedrock': amazon_model_profile,
            'meta-llama': meta_model_profile,
            'meta': meta_model_profile,
            'groq': groq_model_profile,
            'deepseek': deepseek_model_profile,
            'moonshotai': moonshotai_model_profile,
            'x-ai': grok_model_profile,
            'qwen': qwen_model_profile,
        }

        profile = None

        # Check if model name contains a provider prefix (e.g., "anthropic/claude-3")
        if '/' in model_name:
            provider_prefix, model_suffix = model_name.split('/', 1)
            if provider_prefix in provider_to_profile:
                profile = provider_to_profile[provider_prefix](model_suffix)

        # If no profile found, default to OpenAI profile
        if profile is None:
            profile = openai_model_profile(model_name)

        # As LiteLLMProvider is used with OpenAIModel, which uses OpenAIJsonSchemaTransformer,
        # we maintain that behavior
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        http_client: AsyncHTTPClient,
    ) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Initialize a LiteLLM provider.

        Args:
            api_key: API key for the model provider. If None, LiteLLM will try to get it from environment variables.
            api_base: Base URL for the model provider. Use this for custom endpoints or self-hosted models.
            openai_client: Pre-configured OpenAI client. If provided, other parameters are ignored.
            http_client: Custom HTTP client to use.
        """
        if openai_client is not None:
            self._client = openai_client
            return

        # Create OpenAI client that will be used with LiteLLM's completion function
        # The actual API calls will be intercepted and routed through LiteLLM
        if http_client is not None:
            self._client = AsyncOpenAI(
                base_url=api_base, api_key=api_key or 'litellm-placeholder', http_client=http_client
            )
        else:
            http_client = cached_async_http_client(provider='litellm')
            self._client = AsyncOpenAI(
                base_url=api_base, api_key=api_key or 'litellm-placeholder', http_client=http_client
            )

```

### __init__

```python
__init__(
    *,
    api_key: str | None = None,
    api_base: str | None = None
) -> None

```

```python
__init__(
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    http_client: AsyncClient
) -> None

```

```python
__init__(*, openai_client: AsyncOpenAI) -> None

```

```python
__init__(
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: AsyncClient | None = None
) -> None

```

Initialize a LiteLLM provider.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `api_key` | `str | None` | API key for the model provider. If None, LiteLLM will try to get it from environment variables. | `None` | | `api_base` | `str | None` | Base URL for the model provider. Use this for custom endpoints or self-hosted models. | `None` | | `openai_client` | `AsyncOpenAI | None` | Pre-configured OpenAI client. If provided, other parameters are ignored. | `None` | | `http_client` | `AsyncClient | None` | Custom HTTP client to use. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/providers/litellm.py`

```python
def __init__(
    self,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: AsyncHTTPClient | None = None,
) -> None:
    """Initialize a LiteLLM provider.

    Args:
        api_key: API key for the model provider. If None, LiteLLM will try to get it from environment variables.
        api_base: Base URL for the model provider. Use this for custom endpoints or self-hosted models.
        openai_client: Pre-configured OpenAI client. If provided, other parameters are ignored.
        http_client: Custom HTTP client to use.
    """
    if openai_client is not None:
        self._client = openai_client
        return

    # Create OpenAI client that will be used with LiteLLM's completion function
    # The actual API calls will be intercepted and routed through LiteLLM
    if http_client is not None:
        self._client = AsyncOpenAI(
            base_url=api_base, api_key=api_key or 'litellm-placeholder', http_client=http_client
        )
    else:
        http_client = cached_async_http_client(provider='litellm')
        self._client = AsyncOpenAI(
            base_url=api_base, api_key=api_key or 'litellm-placeholder', http_client=http_client
        )

```

Bases: `Provider[AsyncOpenAI]`

Provider for Nebius AI Studio API.

Source code in `pydantic_ai_slim/pydantic_ai/providers/nebius.py`

```python
class NebiusProvider(Provider[AsyncOpenAI]):
    """Provider for Nebius AI Studio API."""

    @property
    def name(self) -> str:
        return 'nebius'

    @property
    def base_url(self) -> str:
        return 'https://api.studio.nebius.com/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        provider_to_profile = {
            'meta-llama': meta_model_profile,
            'deepseek-ai': deepseek_model_profile,
            'qwen': qwen_model_profile,
            'google': google_model_profile,
            'openai': harmony_model_profile,  # used for gpt-oss models on Nebius
            'mistralai': mistral_model_profile,
            'moonshotai': moonshotai_model_profile,
        }

        profile = None

        try:
            model_name = model_name.lower()
            provider, model_name = model_name.split('/', 1)
        except ValueError:
            raise UserError(f"Model name must be in 'provider/model' format, got: {model_name!r}")
        if provider in provider_to_profile:
            profile = provider_to_profile[provider](model_name)

        # As NebiusProvider is always used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('NEBIUS_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `NEBIUS_API_KEY` environment variable or pass it via '
                '`NebiusProvider(api_key=...)` to use the Nebius AI Studio provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='nebius')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

Bases: `Provider[AsyncOpenAI]`

Provider for OVHcloud AI Endpoints.

Source code in `pydantic_ai_slim/pydantic_ai/providers/ovhcloud.py`

```python
class OVHcloudProvider(Provider[AsyncOpenAI]):
    """Provider for OVHcloud AI Endpoints."""

    @property
    def name(self) -> str:
        return 'ovhcloud'

    @property
    def base_url(self) -> str:
        return 'https://oai.endpoints.kepler.ai.cloud.ovh.net/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        model_name = model_name.lower()

        prefix_to_profile = {
            'llama': meta_model_profile,
            'meta-': meta_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'gpt': harmony_model_profile,
            'qwen': qwen_model_profile,
        }

        profile = None
        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix):
                profile = profile_func(model_name)

        # As the OVHcloud AI Endpoints API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer.
        return OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('OVHCLOUD_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `OVHCLOUD_API_KEY` environment variable or pass it via '
                '`OVHcloudProvider(api_key=...)` to use OVHcloud AI Endpoints provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='ovhcloud')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```

