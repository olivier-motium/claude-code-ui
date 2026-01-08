<!-- Source: _complete-doc.md -->

# `pydantic_ai.models.bedrock`

## Setup

For details on how to set up authentication with this model, see [model configuration for Bedrock](../../../models/bedrock/).

### LatestBedrockModelNames

```python
LatestBedrockModelNames = Literal[
    "amazon.titan-tg1-large",
    "amazon.titan-text-lite-v1",
    "amazon.titan-text-express-v1",
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-micro-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-instant-v1",
    "anthropic.claude-v2:1",
    "anthropic.claude-v2",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "us.anthropic.claude-opus-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "eu.anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "cohere.command-text-v14",
    "cohere.command-r-v1:0",
    "cohere.command-r-plus-v1:0",
    "cohere.command-light-text-v14",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "us.meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "us.meta.llama3-2-11b-instruct-v1:0",
    "us.meta.llama3-2-90b-instruct-v1:0",
    "us.meta.llama3-2-1b-instruct-v1:0",
    "us.meta.llama3-2-3b-instruct-v1:0",
    "us.meta.llama3-3-70b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-large-2407-v1:0",
]

```

Latest Bedrock models.

### BedrockModelName

```python
BedrockModelName = str | LatestBedrockModelNames

```

Possible Bedrock model names.

Since Bedrock supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [the Bedrock docs](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for a full list.

### BedrockModelSettings

Bases: `ModelSettings`

Settings for Bedrock models.

See [the Bedrock Converse API docs](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax) for a full list. See [the boto3 implementation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html) of the Bedrock Converse API.

Source code in `pydantic_ai_slim/pydantic_ai/models/bedrock.py`

```python
class BedrockModelSettings(ModelSettings, total=False):
    """Settings for Bedrock models.

    See [the Bedrock Converse API docs](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax) for a full list.
    See [the boto3 implementation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html) of the Bedrock Converse API.
    """

    # ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    bedrock_guardrail_config: GuardrailConfigurationTypeDef
    """Content moderation and safety settings for Bedrock API requests.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailConfiguration.html>.
    """

    bedrock_performance_configuration: PerformanceConfigurationTypeDef
    """Performance optimization settings for model inference.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_PerformanceConfiguration.html>.
    """

    bedrock_request_metadata: dict[str, str]
    """Additional metadata to attach to Bedrock API requests.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax>.
    """

    bedrock_additional_model_response_fields_paths: list[str]
    """JSON paths to extract additional fields from model responses.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html>.
    """

    bedrock_prompt_variables: Mapping[str, PromptVariableValuesTypeDef]
    """Variables for substitution into prompt templates.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_PromptVariableValues.html>.
    """

    bedrock_additional_model_requests_fields: Mapping[str, Any]
    """Additional model-specific parameters to include in requests.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html>.
    """

```

#### bedrock_guardrail_config

```python
bedrock_guardrail_config: GuardrailConfigurationTypeDef

```

Content moderation and safety settings for Bedrock API requests.

See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailConfiguration.html>.

#### bedrock_performance_configuration

```python
bedrock_performance_configuration: (
    PerformanceConfigurationTypeDef
)

```

Performance optimization settings for model inference.

See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_PerformanceConfiguration.html>.

#### bedrock_request_metadata

```python
bedrock_request_metadata: dict[str, str]

```

Additional metadata to attach to Bedrock API requests.

See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax>.

#### bedrock_additional_model_response_fields_paths

```python
bedrock_additional_model_response_fields_paths: list[str]

```

JSON paths to extract additional fields from model responses.

See more about it on <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html>.

#### bedrock_prompt_variables

```python
bedrock_prompt_variables: Mapping[
    str, PromptVariableValuesTypeDef
]

```

Variables for substitution into prompt templates.

See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_PromptVariableValues.html>.

#### bedrock_additional_model_requests_fields

```python
bedrock_additional_model_requests_fields: Mapping[str, Any]

```

Additional model-specific parameters to include in requests.

See more about it on <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html>.

### BedrockConverseModel

Bases: `Model`

A model that uses the Bedrock Converse API.

Source code in `pydantic_ai_slim/pydantic_ai/models/bedrock.py`

```python
@dataclass(init=False)
class BedrockConverseModel(Model):
    """A model that uses the Bedrock Converse API."""

    client: BedrockRuntimeClient

    _model_name: BedrockModelName = field(repr=False)
    _provider: Provider[BaseClient] = field(repr=False)

    def __init__(
        self,
        model_name: BedrockModelName,
        *,
        provider: Literal['bedrock', 'gateway'] | Provider[BaseClient] = 'bedrock',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Bedrock model.

        Args:
            model_name: The name of the model to use.
            model_name: The name of the Bedrock model to use. List of model names available
                [here](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).
            provider: The provider to use for authentication and API access. Can be either the string
                'bedrock' or an instance of `Provider[BaseClient]`. If not provided, a new provider will be
                created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/bedrock' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = cast('BedrockRuntimeClient', provider.client)

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.meta.endpoint_url)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolTypeDef]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolTypeDef:
        tool_spec: ToolSpecificationTypeDef = {'name': f.name, 'inputSchema': {'json': f.parameters_json_schema}}

        if f.description:  # pragma: no branch
            tool_spec['description'] = f.description

        return {'toolSpec': tool_spec}

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
        settings = cast(BedrockModelSettings, model_settings or {})
        response = await self._messages_create(messages, False, settings, model_request_parameters)
        model_response = await self._process_response(response)
        return model_response

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        """Count the number of tokens, works with limited models.

        Check the actual supported models on <https://docs.aws.amazon.com/bedrock/latest/userguide/count-tokens.html>
        """
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        system_prompt, bedrock_messages = await self._map_messages(messages, model_request_parameters)
        params: CountTokensRequestTypeDef = {
            'modelId': self._remove_inference_geo_prefix(self.model_name),
            'input': {
                'converse': {
                    'messages': bedrock_messages,
                    'system': system_prompt,
                },
            },
        }
        try:
            response = await anyio.to_thread.run_sync(functools.partial(self.client.count_tokens, **params))
        except ClientError as e:
            status_code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            if isinstance(status_code, int):
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.response) from e
            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e
        return usage.RequestUsage(input_tokens=response['inputTokens'])

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
        settings = cast(BedrockModelSettings, model_settings or {})
        response = await self._messages_create(messages, True, settings, model_request_parameters)
        yield BedrockStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self.model_name,
            _event_stream=response['stream'],
            _provider_name=self._provider.name,
            _provider_response_id=response.get('ResponseMetadata', {}).get('RequestId', None),
        )

    async def _process_response(self, response: ConverseResponseTypeDef) -> ModelResponse:
        items: list[ModelResponsePart] = []
        if message := response['output'].get('message'):  # pragma: no branch
            for item in message['content']:
                if reasoning_content := item.get('reasoningContent'):
                    if redacted_content := reasoning_content.get('redactedContent'):
                        items.append(
                            ThinkingPart(
                                id='redacted_content',
                                content='',
                                signature=redacted_content.decode('utf-8'),
                                provider_name=self.system,
                            )
                        )
                    elif reasoning_text := reasoning_content.get('reasoningText'):  # pragma: no branch
                        signature = reasoning_text.get('signature')
                        items.append(
                            ThinkingPart(
                                content=reasoning_text['text'],
                                signature=signature,
                                provider_name=self.system if signature else None,
                            )
                        )
                if text := item.get('text'):
                    items.append(TextPart(content=text))
                elif tool_use := item.get('toolUse'):
                    items.append(
                        ToolCallPart(
                            tool_name=tool_use['name'],
                            args=tool_use['input'],
                            tool_call_id=tool_use['toolUseId'],
                        ),
                    )
        u = usage.RequestUsage(
            input_tokens=response['usage']['inputTokens'],
            output_tokens=response['usage']['outputTokens'],
        )
        response_id = response.get('ResponseMetadata', {}).get('RequestId', None)
        raw_finish_reason = response['stopReason']
        provider_details = {'finish_reason': raw_finish_reason}
        finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=items,
            usage=u,
            model_name=self.model_name,
            provider_response_id=response_id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: BedrockModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseStreamResponseTypeDef:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: BedrockModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseResponseTypeDef:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: BedrockModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseResponseTypeDef | ConverseStreamResponseTypeDef:
        system_prompt, bedrock_messages = await self._map_messages(messages, model_request_parameters)
        inference_config = self._map_inference_config(model_settings)

        params: ConverseRequestTypeDef = {
            'modelId': self.model_name,
            'messages': bedrock_messages,
            'system': system_prompt,
            'inferenceConfig': inference_config,
        }

        tool_config = self._map_tool_config(model_request_parameters)
        if tool_config:
            params['toolConfig'] = tool_config

        if model_request_parameters.builtin_tools:
            raise UserError('Bedrock does not support built-in tools')

        # Bedrock supports a set of specific extra parameters
        if model_settings:
            if guardrail_config := model_settings.get('bedrock_guardrail_config', None):
                params['guardrailConfig'] = guardrail_config
            if performance_configuration := model_settings.get('bedrock_performance_configuration', None):
                params['performanceConfig'] = performance_configuration
            if request_metadata := model_settings.get('bedrock_request_metadata', None):
                params['requestMetadata'] = request_metadata
            if additional_model_response_fields_paths := model_settings.get(
                'bedrock_additional_model_response_fields_paths', None
            ):
                params['additionalModelResponseFieldPaths'] = additional_model_response_fields_paths
            if additional_model_requests_fields := model_settings.get('bedrock_additional_model_requests_fields', None):
                params['additionalModelRequestFields'] = additional_model_requests_fields
            if prompt_variables := model_settings.get('bedrock_prompt_variables', None):
                params['promptVariables'] = prompt_variables

        try:
            if stream:
                model_response = await anyio.to_thread.run_sync(
                    functools.partial(self.client.converse_stream, **params)
                )
            else:
                model_response = await anyio.to_thread.run_sync(functools.partial(self.client.converse, **params))
        except ClientError as e:
            status_code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            if isinstance(status_code, int):
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.response) from e
            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e
        return model_response

    @staticmethod
    def _map_inference_config(
        model_settings: ModelSettings | None,
    ) -> InferenceConfigurationTypeDef:
        model_settings = model_settings or {}
        inference_config: InferenceConfigurationTypeDef = {}

        if max_tokens := model_settings.get('max_tokens'):
            inference_config['maxTokens'] = max_tokens
        if (temperature := model_settings.get('temperature')) is not None:
            inference_config['temperature'] = temperature
        if top_p := model_settings.get('top_p'):
            inference_config['topP'] = top_p
        if stop_sequences := model_settings.get('stop_sequences'):
            inference_config['stopSequences'] = stop_sequences

        return inference_config

    def _map_tool_config(self, model_request_parameters: ModelRequestParameters) -> ToolConfigurationTypeDef | None:
        tools = self._get_tools(model_request_parameters)
        if not tools:
            return None

        tool_choice: ToolChoiceTypeDef
        if not model_request_parameters.allow_text_output:
            tool_choice = {'any': {}}
        else:
            tool_choice = {'auto': {}}

        tool_config: ToolConfigurationTypeDef = {'tools': tools}
        if tool_choice and BedrockModelProfile.from_profile(self.profile).bedrock_supports_tool_choice:
            tool_config['toolChoice'] = tool_choice

        return tool_config

    async def _map_messages(  # noqa: C901
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> tuple[list[SystemContentBlockTypeDef], list[MessageUnionTypeDef]]:
        """Maps a `pydantic_ai.Message` to the Bedrock `MessageUnionTypeDef`.

        Groups consecutive ToolReturnPart objects into a single user message as required by Bedrock Claude/Nova models.
        """
        profile = BedrockModelProfile.from_profile(self.profile)
        system_prompt: list[SystemContentBlockTypeDef] = []
        bedrock_messages: list[MessageUnionTypeDef] = []
        document_count: Iterator[int] = count(1)
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart) and part.content:
                        system_prompt.append({'text': part.content})
                    elif isinstance(part, UserPromptPart):
                        bedrock_messages.extend(await self._map_user_prompt(part, document_count))
                    elif isinstance(part, ToolReturnPart):
                        assert part.tool_call_id is not None
                        bedrock_messages.append(
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        'toolResult': {
                                            'toolUseId': part.tool_call_id,
                                            'content': [
                                                {'text': part.model_response_str()}
                                                if profile.bedrock_tool_result_format == 'text'
                                                else {'json': part.model_response_object()}
                                            ],
                                            'status': 'success',
                                        }
                                    }
                                ],
                            }
                        )
                    elif isinstance(part, RetryPromptPart):
                        # TODO(Marcelo): We need to add a test here.
                        if part.tool_name is None:  # pragma: no cover
                            bedrock_messages.append({'role': 'user', 'content': [{'text': part.model_response()}]})
                        else:
                            assert part.tool_call_id is not None
                            bedrock_messages.append(
                                {
                                    'role': 'user',
                                    'content': [
                                        {
                                            'toolResult': {
                                                'toolUseId': part.tool_call_id,
                                                'content': [{'text': part.model_response()}],
                                                'status': 'error',
                                            }
                                        }
                                    ],
                                }
                            )
            elif isinstance(message, ModelResponse):
                content: list[ContentBlockOutputTypeDef] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        content.append({'text': item.content})
                    elif isinstance(item, ThinkingPart):
                        if (
                            item.provider_name == self.system
                            and item.signature
                            and BedrockModelProfile.from_profile(self.profile).bedrock_send_back_thinking_parts
                        ):
                            if item.id == 'redacted_content':
                                reasoning_content: ReasoningContentBlockOutputTypeDef = {
                                    'redactedContent': item.signature.encode('utf-8'),
                                }
                            else:
                                reasoning_content: ReasoningContentBlockOutputTypeDef = {
                                    'reasoningText': {
                                        'text': item.content,
                                        'signature': item.signature,
                                    }
                                }
                            content.append({'reasoningContent': reasoning_content})
                        else:
                            start_tag, end_tag = self.profile.thinking_tags
                            content.append({'text': '\n'.join([start_tag, item.content, end_tag])})
                    elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):
                        pass
                    else:
                        assert isinstance(item, ToolCallPart)
                        content.append(self._map_tool_call(item))
                bedrock_messages.append({'role': 'assistant', 'content': content})
            else:
                assert_never(message)

        # Merge together sequential user messages.
        processed_messages: list[MessageUnionTypeDef] = []
        last_message: dict[str, Any] | None = None
        for current_message in bedrock_messages:
            if (
                last_message is not None
                and current_message['role'] == last_message['role']
                and current_message['role'] == 'user'
            ):
                # Add the new user content onto the existing user message.
                last_content = list(last_message['content'])
                last_content.extend(current_message['content'])
                last_message['content'] = last_content
                continue

            # Add the entire message to the list of messages.
            processed_messages.append(current_message)
            last_message = cast(dict[str, Any], current_message)

        if instructions := self._get_instructions(messages, model_request_parameters):
            system_prompt.insert(0, {'text': instructions})

        return system_prompt, processed_messages

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart, document_count: Iterator[int]) -> list[MessageUnionTypeDef]:
        content: list[ContentBlockUnionTypeDef] = []
        if isinstance(part.content, str):
            content.append({'text': part.content})
        else:
            for item in part.content:
                if isinstance(item, str):
                    content.append({'text': item})
                elif isinstance(item, BinaryContent):
                    format = item.format
                    if item.is_document:
                        name = f'Document {next(document_count)}'
                        assert format in ('pdf', 'txt', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'md')
                        content.append({'document': {'name': name, 'format': format, 'source': {'bytes': item.data}}})
                    elif item.is_image:
                        assert format in ('jpeg', 'png', 'gif', 'webp')
                        content.append({'image': {'format': format, 'source': {'bytes': item.data}}})
                    elif item.is_video:
                        assert format in ('mkv', 'mov', 'mp4', 'webm', 'flv', 'mpeg', 'mpg', 'wmv', 'three_gp')
                        content.append({'video': {'format': format, 'source': {'bytes': item.data}}})
                    else:
                        raise NotImplementedError('Binary content is not supported yet.')
                elif isinstance(item, ImageUrl | DocumentUrl | VideoUrl):
                    downloaded_item = await download_item(item, data_format='bytes', type_format='extension')
                    format = downloaded_item['data_type']
                    if item.kind == 'image-url':
                        format = item.media_type.split('/')[1]
                        assert format in ('jpeg', 'png', 'gif', 'webp'), f'Unsupported image format: {format}'
                        image: ImageBlockTypeDef = {'format': format, 'source': {'bytes': downloaded_item['data']}}
                        content.append({'image': image})

                    elif item.kind == 'document-url':
                        name = f'Document {next(document_count)}'
                        document: DocumentBlockTypeDef = {
                            'name': name,
                            'format': item.format,
                            'source': {'bytes': downloaded_item['data']},
                        }
                        content.append({'document': document})

                    elif item.kind == 'video-url':  # pragma: no branch
                        format = item.media_type.split('/')[1]
                        assert format in (
                            'mkv',
                            'mov',
                            'mp4',
                            'webm',
                            'flv',
                            'mpeg',
                            'mpg',
                            'wmv',
                            'three_gp',
                        ), f'Unsupported video format: {format}'
                        video: VideoBlockTypeDef = {'format': format, 'source': {'bytes': downloaded_item['data']}}
                        content.append({'video': video})
                elif isinstance(item, AudioUrl):  # pragma: no cover
                    raise NotImplementedError('Audio is not supported yet.')
                elif isinstance(item, CachePoint):
                    # Bedrock support has not been implemented yet: https://github.com/pydantic/pydantic-ai/issues/3418
                    pass
                else:
                    assert_never(item)
        return [{'role': 'user', 'content': content}]

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ContentBlockOutputTypeDef:
        return {
            'toolUse': {'toolUseId': _utils.guard_tool_call_id(t=t), 'name': t.tool_name, 'input': t.args_as_dict()}
        }

    @staticmethod
    def _remove_inference_geo_prefix(model_name: BedrockModelName) -> BedrockModelName:
        """Remove inference geographic prefix from model ID if present."""
        for prefix in _AWS_BEDROCK_INFERENCE_GEO_PREFIXES:
            if model_name.startswith(prefix):
                return model_name.removeprefix(prefix)
        return model_name

```

#### __init__

```python
__init__(
    model_name: BedrockModelName,
    *,
    provider: (
        Literal["bedrock", "gateway"] | Provider[BaseClient]
    ) = "bedrock",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize a Bedrock model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model_name` | `BedrockModelName` | The name of the model to use. | *required* | | `model_name` | `BedrockModelName` | The name of the Bedrock model to use. List of model names available here. | *required* | | `provider` | `Literal['bedrock', 'gateway'] | Provider[BaseClient]` | The provider to use for authentication and API access. Can be either the string 'bedrock' or an instance of Provider[BaseClient]. If not provided, a new provider will be created using the other parameters. | `'bedrock'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider based on the model name. | `None` | | `settings` | `ModelSettings | None` | Model-specific settings that will be used as defaults for this model. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/bedrock.py`

```python
def __init__(
    self,
    model_name: BedrockModelName,
    *,
    provider: Literal['bedrock', 'gateway'] | Provider[BaseClient] = 'bedrock',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize a Bedrock model.

    Args:
        model_name: The name of the model to use.
        model_name: The name of the Bedrock model to use. List of model names available
            [here](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).
        provider: The provider to use for authentication and API access. Can be either the string
            'bedrock' or an instance of `Provider[BaseClient]`. If not provided, a new provider will be
            created using the other parameters.
        profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
        settings: Model-specific settings that will be used as defaults for this model.
    """
    self._model_name = model_name

    if isinstance(provider, str):
        provider = infer_provider('gateway/bedrock' if provider == 'gateway' else provider)
    self._provider = provider
    self.client = cast('BedrockRuntimeClient', provider.client)

    super().__init__(settings=settings, profile=profile or provider.model_profile)

```

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

#### count_tokens

```python
count_tokens(
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> RequestUsage

```

Count the number of tokens, works with limited models.

Check the actual supported models on <https://docs.aws.amazon.com/bedrock/latest/userguide/count-tokens.html>

Source code in `pydantic_ai_slim/pydantic_ai/models/bedrock.py`

```python
async def count_tokens(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> usage.RequestUsage:
    """Count the number of tokens, works with limited models.

    Check the actual supported models on <https://docs.aws.amazon.com/bedrock/latest/userguide/count-tokens.html>
    """
    model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
    system_prompt, bedrock_messages = await self._map_messages(messages, model_request_parameters)
    params: CountTokensRequestTypeDef = {
        'modelId': self._remove_inference_geo_prefix(self.model_name),
        'input': {
            'converse': {
                'messages': bedrock_messages,
                'system': system_prompt,
            },
        },
    }
    try:
        response = await anyio.to_thread.run_sync(functools.partial(self.client.count_tokens, **params))
    except ClientError as e:
        status_code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        if isinstance(status_code, int):
            raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.response) from e
        raise ModelAPIError(model_name=self.model_name, message=str(e)) from e
    return usage.RequestUsage(input_tokens=response['inputTokens'])

```

### BedrockStreamedResponse

Bases: `StreamedResponse`

Implementation of `StreamedResponse` for Bedrock models.

Source code in `pydantic_ai_slim/pydantic_ai/models/bedrock.py`

```python
@dataclass
class BedrockStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Bedrock models."""

    _model_name: BedrockModelName
    _event_stream: EventStream[ConverseStreamOutputTypeDef]
    _provider_name: str
    _timestamp: datetime = field(default_factory=_utils.now_utc)
    _provider_response_id: str | None = None

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        """Return an async iterator of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This method should be implemented by subclasses to translate the vendor-specific stream of events into
        pydantic_ai-format events.
        """
        if self._provider_response_id is not None:  # pragma: no cover
            self.provider_response_id = self._provider_response_id

        chunk: ConverseStreamOutputTypeDef
        tool_id: str | None = None
        async for chunk in _AsyncIteratorWrapper(self._event_stream):
            match chunk:
                case {'messageStart': _}:
                    continue
                case {'messageStop': message_stop}:
                    raw_finish_reason = message_stop['stopReason']
                    self.provider_details = {'finish_reason': raw_finish_reason}
                    self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)
                case {'metadata': metadata}:
                    if 'usage' in metadata:  # pragma: no branch
                        self._usage += self._map_usage(metadata)
                case {'contentBlockStart': content_block_start}:
                    index = content_block_start['contentBlockIndex']
                    start = content_block_start['start']
                    if 'toolUse' in start:  # pragma: no branch
                        tool_use_start = start['toolUse']
                        tool_id = tool_use_start['toolUseId']
                        tool_name = tool_use_start['name']
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=index,
                            tool_name=tool_name,
                            args=None,
                            tool_call_id=tool_id,
                        )
                        if maybe_event:  # pragma: no branch
                            yield maybe_event
                case {'contentBlockDelta': content_block_delta}:
                    index = content_block_delta['contentBlockIndex']
                    delta = content_block_delta['delta']
                    if 'reasoningContent' in delta:
                        if redacted_content := delta['reasoningContent'].get('redactedContent'):
                            yield self._parts_manager.handle_thinking_delta(
                                vendor_part_id=index,
                                id='redacted_content',
                                signature=redacted_content.decode('utf-8'),
                                provider_name=self.provider_name,
                            )
                        else:
                            signature = delta['reasoningContent'].get('signature')
                            yield self._parts_manager.handle_thinking_delta(
                                vendor_part_id=index,
                                content=delta['reasoningContent'].get('text'),
                                signature=signature,
                                provider_name=self.provider_name if signature else None,
                            )
                    if text := delta.get('text'):
                        maybe_event = self._parts_manager.handle_text_delta(vendor_part_id=index, content=text)
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    if 'toolUse' in delta:
                        tool_use = delta['toolUse']
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=index,
                            tool_name=tool_use.get('name'),
                            args=tool_use.get('input'),
                            tool_call_id=tool_id,
                        )
                        if maybe_event:  # pragma: no branch
                            yield maybe_event
                case _:
                    pass  # pyright wants match statements to be exhaustive

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
        return self._timestamp

    def _map_usage(self, metadata: ConverseStreamMetadataEventTypeDef) -> usage.RequestUsage:
        return usage.RequestUsage(
            input_tokens=metadata['usage']['inputTokens'],
            output_tokens=metadata['usage']['outputTokens'],
        )

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

# `pydantic_ai.models.cohere`

## Setup

For details on how to set up authentication with this model, see [model configuration for Cohere](../../../models/cohere/).

### LatestCohereModelNames

```python
LatestCohereModelNames = Literal[
    "c4ai-aya-expanse-32b",
    "c4ai-aya-expanse-8b",
    "command-nightly",
    "command-r-08-2024",
    "command-r-plus-08-2024",
    "command-r7b-12-2024",
]

```

Latest Cohere models.

### CohereModelName

```python
CohereModelName = str | LatestCohereModelNames

```

Possible Cohere model names.

Since Cohere supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [Cohere's docs](https://docs.cohere.com/v2/docs/models) for a list of all available models.

### CohereModelSettings

Bases: `ModelSettings`

Settings used for a Cohere model request.

Source code in `pydantic_ai_slim/pydantic_ai/models/cohere.py`

```python
class CohereModelSettings(ModelSettings, total=False):
    """Settings used for a Cohere model request."""

```

### CohereModel

Bases: `Model`

A model that uses the Cohere API.

Internally, this uses the [Cohere Python client](https://github.com/cohere-ai/cohere-python) to interact with the API.

Apart from `__init__`, all methods are private or match those of the base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/cohere.py`

```python
@dataclass(init=False)
class CohereModel(Model):
    """A model that uses the Cohere API.

    Internally, this uses the [Cohere Python client](
    https://github.com/cohere-ai/cohere-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncClientV2 = field(repr=False)

    _model_name: CohereModelName = field(repr=False)
    _provider: Provider[AsyncClientV2] = field(repr=False)

    def __init__(
        self,
        model_name: CohereModelName,
        *,
        provider: Literal['cohere'] | Provider[AsyncClientV2] = 'cohere',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Cohere model.

        Args:
            model_name: The name of the Cohere model to use. List of model names
                available [here](https://docs.cohere.com/docs/models#command).
            provider: The provider to use for authentication and API access. Can be either the string
                'cohere' or an instance of `Provider[AsyncClientV2]`. If not provided, a new provider will be
                created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        client_wrapper = self.client._client_wrapper  # type: ignore
        return str(client_wrapper.get_base_url())

    @property
    def model_name(self) -> CohereModelName:
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
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._chat(messages, cast(CohereModelSettings, model_settings or {}), model_request_parameters)
        model_response = self._process_response(response)
        return model_response

    async def _chat(
        self,
        messages: list[ModelMessage],
        model_settings: CohereModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> V2ChatResponse:
        tools = self._get_tools(model_request_parameters)

        if model_request_parameters.builtin_tools:
            raise UserError('Cohere does not support built-in tools')

        cohere_messages = self._map_messages(messages, model_request_parameters)
        try:
            return await self.client.chat(
                model=self._model_name,
                messages=cohere_messages,
                tools=tools or OMIT,
                max_tokens=model_settings.get('max_tokens', OMIT),
                stop_sequences=model_settings.get('stop_sequences', OMIT),
                temperature=model_settings.get('temperature', OMIT),
                p=model_settings.get('top_p', OMIT),
                seed=model_settings.get('seed', OMIT),
                presence_penalty=model_settings.get('presence_penalty', OMIT),
                frequency_penalty=model_settings.get('frequency_penalty', OMIT),
            )
        except ApiError as e:
            if (status_code := e.status_code) and status_code >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e

    def _process_response(self, response: V2ChatResponse) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        parts: list[ModelResponsePart] = []
        if response.message.content is not None:
            for content in response.message.content:
                if content.type == 'text':
                    parts.append(TextPart(content=content.text))
                elif content.type == 'thinking':  # pragma: no branch
                    parts.append(ThinkingPart(content=content.thinking))
        for c in response.message.tool_calls or []:
            if c.function and c.function.name and c.function.arguments:  # pragma: no branch
                parts.append(
                    ToolCallPart(
                        tool_name=c.function.name,
                        args=c.function.arguments,
                        tool_call_id=c.id or _generate_tool_call_id(),
                    )
                )

        raw_finish_reason = response.finish_reason
        provider_details = {'finish_reason': raw_finish_reason}
        finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=parts,
            usage=_map_usage(response),
            model_name=self._model_name,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[ChatMessageV2]:
        """Just maps a `pydantic_ai.Message` to a `cohere.ChatMessageV2`."""
        cohere_messages: list[ChatMessageV2] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                cohere_messages.extend(self._map_user_message(message))
            elif isinstance(message, ModelResponse):
                texts: list[str] = []
                thinking: list[str] = []
                tool_calls: list[ToolCallV2] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        texts.append(item.content)
                    elif isinstance(item, ThinkingPart):
                        thinking.append(item.content)
                    elif isinstance(item, ToolCallPart):
                        tool_calls.append(self._map_tool_call(item))
                    elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                        # This is currently never returned from cohere
                        pass
                    elif isinstance(item, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
                        pass
                    else:
                        assert_never(item)

                message_param = AssistantChatMessageV2(role='assistant')
                if texts or thinking:
                    contents: list[AssistantMessageV2ContentItem] = []
                    if thinking:
                        contents.append(ThinkingAssistantMessageV2ContentItem(thinking='\n\n'.join(thinking)))
                    if texts:  # pragma: no branch
                        contents.append(TextAssistantMessageV2ContentItem(text='\n\n'.join(texts)))
                    message_param.content = contents
                if tool_calls:
                    message_param.tool_calls = tool_calls
                cohere_messages.append(message_param)
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages, model_request_parameters):
            cohere_messages.insert(0, SystemChatMessageV2(role='system', content=instructions))
        return cohere_messages

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolV2]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ToolCallV2:
        return ToolCallV2(
            id=_guard_tool_call_id(t=t),
            type='function',
            function=ToolCallV2Function(
                name=t.tool_name,
                arguments=t.args_as_json_str(),
            ),
        )

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolV2:
        return ToolV2(
            type='function',
            function=ToolV2Function(
                name=f.name,
                description=f.description,
                parameters=f.parameters_json_schema,
            ),
        )

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[ChatMessageV2]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield SystemChatMessageV2(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    yield UserChatMessageV2(role='user', content=part.content)
                else:
                    raise RuntimeError('Cohere does not yet support multi-modal inputs.')
            elif isinstance(part, ToolReturnPart):
                yield ToolChatMessageV2(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield UserChatMessageV2(role='user', content=part.model_response())  # pragma: no cover
                else:
                    yield ToolChatMessageV2(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)

```

#### __init__

```python
__init__(
    model_name: CohereModelName,
    *,
    provider: (
        Literal["cohere"] | Provider[AsyncClientV2]
    ) = "cohere",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize an Cohere model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model_name` | `CohereModelName` | The name of the Cohere model to use. List of model names available here. | *required* | | `provider` | `Literal['cohere'] | Provider[AsyncClientV2]` | The provider to use for authentication and API access. Can be either the string 'cohere' or an instance of Provider[AsyncClientV2]. If not provided, a new provider will be created using the other parameters. | `'cohere'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider based on the model name. | `None` | | `settings` | `ModelSettings | None` | Model-specific settings that will be used as defaults for this model. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/cohere.py`

```python
def __init__(
    self,
    model_name: CohereModelName,
    *,
    provider: Literal['cohere'] | Provider[AsyncClientV2] = 'cohere',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize an Cohere model.

    Args:
        model_name: The name of the Cohere model to use. List of model names
            available [here](https://docs.cohere.com/docs/models#command).
        provider: The provider to use for authentication and API access. Can be either the string
            'cohere' or an instance of `Provider[AsyncClientV2]`. If not provided, a new provider will be
            created using the other parameters.
        profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
        settings: Model-specific settings that will be used as defaults for this model.
    """
    self._model_name = model_name

    if isinstance(provider, str):
        provider = infer_provider(provider)
    self._provider = provider
    self.client = provider.client

    super().__init__(settings=settings, profile=profile or provider.model_profile)

```

#### model_name

```python
model_name: CohereModelName

```

The model name.

#### system

```python
system: str

```

The model provider.

# pydantic_ai.models.fallback

### FallbackModel

Bases: `Model`

A model that uses one or more fallback models upon failure.

Apart from `__init__`, all methods are private or match those of the base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/fallback.py`

```python
@dataclass(init=False)
class FallbackModel(Model):
    """A model that uses one or more fallback models upon failure.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    models: list[Model]

    _model_name: str = field(repr=False)
    _fallback_on: Callable[[Exception], bool]

    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = (ModelAPIError,),
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
            fallback_on: A callable or tuple of exceptions that should trigger a fallback.
        """
        super().__init__()
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]

        if isinstance(fallback_on, tuple):
            self._fallback_on = _default_fallback_condition_factory(fallback_on)
        else:
            self._fallback_on = fallback_on

    @property
    def model_name(self) -> str:
        """The model name."""
        return f'fallback:{",".join(model.model_name for model in self.models)}'

    @property
    def system(self) -> str:
        return f'fallback:{",".join(model.system for model in self.models)}'

    @property
    def base_url(self) -> str | None:
        return self.models[0].base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Try each model in sequence until one succeeds.

        In case of failure, raise a FallbackExceptionGroup with all exceptions.
        """
        exceptions: list[Exception] = []

        for model in self.models:
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                response = await model.request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if self._fallback_on(exc):
                    exceptions.append(exc)
                    continue
                raise exc

            self._set_span_attributes(model, prepared_parameters)
            return response

        raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Try each model in sequence until one succeeds."""
        exceptions: list[Exception] = []

        for model in self.models:
            async with AsyncExitStack() as stack:
                try:
                    _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                    response = await stack.enter_async_context(
                        model.request_stream(messages, model_settings, model_request_parameters, run_context)
                    )
                except Exception as exc:
                    if self._fallback_on(exc):
                        exceptions.append(exc)
                        continue
                    raise exc  # pragma: no cover

                self._set_span_attributes(model, prepared_parameters)
                yield response
                return

        raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

    @cached_property
    def profile(self) -> ModelProfile:
        raise NotImplementedError('FallbackModel does not have its own model profile.')

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return model_request_parameters  # pragma: no cover

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return model_settings, model_request_parameters

    def _set_span_attributes(self, model: Model, model_request_parameters: ModelRequestParameters):
        with suppress(Exception):
            span = get_current_span()
            if span.is_recording():
                attributes = getattr(span, 'attributes', {})
                if attributes.get('gen_ai.request.model') == self.model_name:  # pragma: no branch
                    span.set_attributes(
                        {
                            **InstrumentedModel.model_attributes(model),
                            **InstrumentedModel.model_request_parameters_attributes(model_request_parameters),
                        }
                    )

```

#### __init__

```python
__init__(
    default_model: Model | KnownModelName | str,
    *fallback_models: Model | KnownModelName | str,
    fallback_on: (
        Callable[[Exception], bool]
        | tuple[type[Exception], ...]
    ) = (ModelAPIError,)
)

```

Initialize a fallback model instance.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `default_model` | `Model | KnownModelName | str` | The name or instance of the default model to use. | *required* | | `fallback_models` | `Model | KnownModelName | str` | The names or instances of the fallback models to use upon failure. | `()` | | `fallback_on` | `Callable[[Exception], bool] | tuple[type[Exception], ...]` | A callable or tuple of exceptions that should trigger a fallback. | `(ModelAPIError,)` |

Source code in `pydantic_ai_slim/pydantic_ai/models/fallback.py`

```python
def __init__(
    self,
    default_model: Model | KnownModelName | str,
    *fallback_models: Model | KnownModelName | str,
    fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = (ModelAPIError,),
):
    """Initialize a fallback model instance.

    Args:
        default_model: The name or instance of the default model to use.
        fallback_models: The names or instances of the fallback models to use upon failure.
        fallback_on: A callable or tuple of exceptions that should trigger a fallback.
    """
    super().__init__()
    self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]

    if isinstance(fallback_on, tuple):
        self._fallback_on = _default_fallback_condition_factory(fallback_on)
    else:
        self._fallback_on = fallback_on

```

#### model_name

```python
model_name: str

```

The model name.

#### request

```python
request(
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse

```

Try each model in sequence until one succeeds.

In case of failure, raise a FallbackExceptionGroup with all exceptions.

Source code in `pydantic_ai_slim/pydantic_ai/models/fallback.py`

```python
async def request(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse:
    """Try each model in sequence until one succeeds.

    In case of failure, raise a FallbackExceptionGroup with all exceptions.
    """
    exceptions: list[Exception] = []

    for model in self.models:
        try:
            _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
            response = await model.request(messages, model_settings, model_request_parameters)
        except Exception as exc:
            if self._fallback_on(exc):
                exceptions.append(exc)
                continue
            raise exc

        self._set_span_attributes(model, prepared_parameters)
        return response

    raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

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

Try each model in sequence until one succeeds.

Source code in `pydantic_ai_slim/pydantic_ai/models/fallback.py`

```python
@asynccontextmanager
async def request_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    run_context: RunContext[Any] | None = None,
) -> AsyncIterator[StreamedResponse]:
    """Try each model in sequence until one succeeds."""
    exceptions: list[Exception] = []

    for model in self.models:
        async with AsyncExitStack() as stack:
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                response = await stack.enter_async_context(
                    model.request_stream(messages, model_settings, model_request_parameters, run_context)
                )
            except Exception as exc:
                if self._fallback_on(exc):
                    exceptions.append(exc)
                    continue
                raise exc  # pragma: no cover

            self._set_span_attributes(model, prepared_parameters)
            yield response
            return

    raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

```

# `pydantic_ai.models.function`

A model controlled by a local function.

FunctionModel is similar to [`TestModel`](../test/), but allows greater control over the model's behavior.

Its primary use case is for more advanced unit testing than is possible with `TestModel`.

Here's a minimal example:

[Learn about Gateway](../../../gateway) function_model_usage.py

```python
from pydantic_ai import Agent
from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel, AgentInfo

my_agent = Agent('gateway/openai:gpt-5')


async def model_function(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    print(messages)
    """
    [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='Testing my agent...',
                    timestamp=datetime.datetime(...),
                )
            ],
            run_id='...',
        )
    ]
    """
    print(info)
    """
    AgentInfo(
        function_tools=[],
        allow_text_output=True,
        output_tools=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[], builtin_tools=[], output_tools=[]
        ),
        instructions=None,
    )
    """
    return ModelResponse(parts=[TextPart('hello world')])


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    with my_agent.override(model=FunctionModel(model_function)):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'hello world'

```

function_model_usage.py

```python
from pydantic_ai import Agent
from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel, AgentInfo

my_agent = Agent('openai:gpt-5')


async def model_function(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    print(messages)
    """
    [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='Testing my agent...',
                    timestamp=datetime.datetime(...),
                )
            ],
            run_id='...',
        )
    ]
    """
    print(info)
    """
    AgentInfo(
        function_tools=[],
        allow_text_output=True,
        output_tools=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[], builtin_tools=[], output_tools=[]
        ),
        instructions=None,
    )
    """
    return ModelResponse(parts=[TextPart('hello world')])


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    with my_agent.override(model=FunctionModel(model_function)):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'hello world'

```

See [Unit testing with `FunctionModel`](../../../testing/#unit-testing-with-functionmodel) for detailed documentation.

### FunctionModel

Bases: `Model`

A model controlled by a local function.

Apart from `__init__`, all methods are private or match those of the base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`

```python
@dataclass(init=False)
class FunctionModel(Model):
    """A model controlled by a local function.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    function: FunctionDef | None
    stream_function: StreamFunctionDef | None

    _model_name: str = field(repr=False)
    _system: str = field(default='function', repr=False)

    @overload
    def __init__(
        self,
        function: FunctionDef,
        *,
        model_name: str | None = None,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        stream_function: StreamFunctionDef,
        model_name: str | None = None,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        function: FunctionDef,
        *,
        stream_function: StreamFunctionDef,
        model_name: str | None = None,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    def __init__(
        self,
        function: FunctionDef | None = None,
        *,
        stream_function: StreamFunctionDef | None = None,
        model_name: str | None = None,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a `FunctionModel`.

        Either `function` or `stream_function` must be provided, providing both is allowed.

        Args:
            function: The function to call for non-streamed requests.
            stream_function: The function to call for streamed requests.
            model_name: The name of the model. If not provided, a name is generated from the function names.
            profile: The model profile to use.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        if function is None and stream_function is None:
            raise TypeError('Either `function` or `stream_function` must be provided')

        self.function = function
        self.stream_function = stream_function

        function_name = self.function.__name__ if self.function is not None else ''
        stream_function_name = self.stream_function.__name__ if self.stream_function is not None else ''
        self._model_name = model_name or f'function:{function_name}:{stream_function_name}'

        # Use a default profile that supports JSON schema and object output if none provided
        if profile is None:
            profile = ModelProfile(
                supports_json_schema_output=True,
                supports_json_object_output=True,
            )
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
        agent_info = AgentInfo(
            function_tools=model_request_parameters.function_tools,
            allow_text_output=model_request_parameters.allow_text_output,
            output_tools=model_request_parameters.output_tools,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
            instructions=self._get_instructions(messages, model_request_parameters),
        )

        assert self.function is not None, 'FunctionModel must receive a `function` to support non-streamed requests'

        if inspect.iscoroutinefunction(self.function):
            response = await self.function(messages, agent_info)
        else:
            response_ = await _utils.run_in_executor(self.function, messages, agent_info)
            assert isinstance(response_, ModelResponse), response_
            response = response_
        response.model_name = self._model_name
        # Add usage data if not already present
        if not response.usage.has_values():  # pragma: no branch
            response.usage = _estimate_usage(chain(messages, [response]))
        return response

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
        agent_info = AgentInfo(
            function_tools=model_request_parameters.function_tools,
            allow_text_output=model_request_parameters.allow_text_output,
            output_tools=model_request_parameters.output_tools,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
            instructions=self._get_instructions(messages, model_request_parameters),
        )

        assert self.stream_function is not None, (
            'FunctionModel must receive a `stream_function` to support streamed requests'
        )

        response_stream = PeekableAsyncStream(self.stream_function(messages, agent_info))

        first = await response_stream.peek()
        if isinstance(first, _utils.Unset):
            raise ValueError('Stream function must return at least one item')

        yield FunctionStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _iter=response_stream,
        )

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

```

#### __init__

```python
__init__(
    function: FunctionDef,
    *,
    model_name: str | None = None,
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
) -> None

```

```python
__init__(
    *,
    stream_function: StreamFunctionDef,
    model_name: str | None = None,
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
) -> None

```

```python
__init__(
    function: FunctionDef,
    *,
    stream_function: StreamFunctionDef,
    model_name: str | None = None,
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
) -> None

```

```python
__init__(
    function: FunctionDef | None = None,
    *,
    stream_function: StreamFunctionDef | None = None,
    model_name: str | None = None,
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize a `FunctionModel`.

Either `function` or `stream_function` must be provided, providing both is allowed.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `function` | `FunctionDef | None` | The function to call for non-streamed requests. | `None` | | `stream_function` | `StreamFunctionDef | None` | The function to call for streamed requests. | `None` | | `model_name` | `str | None` | The name of the model. If not provided, a name is generated from the function names. | `None` | | `profile` | `ModelProfileSpec | None` | The model profile to use. | `None` | | `settings` | `ModelSettings | None` | Model-specific settings that will be used as defaults for this model. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`

```python
def __init__(
    self,
    function: FunctionDef | None = None,
    *,
    stream_function: StreamFunctionDef | None = None,
    model_name: str | None = None,
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize a `FunctionModel`.

    Either `function` or `stream_function` must be provided, providing both is allowed.

    Args:
        function: The function to call for non-streamed requests.
        stream_function: The function to call for streamed requests.
        model_name: The name of the model. If not provided, a name is generated from the function names.
        profile: The model profile to use.
        settings: Model-specific settings that will be used as defaults for this model.
    """
    if function is None and stream_function is None:
        raise TypeError('Either `function` or `stream_function` must be provided')

    self.function = function
    self.stream_function = stream_function

    function_name = self.function.__name__ if self.function is not None else ''
    stream_function_name = self.stream_function.__name__ if self.stream_function is not None else ''
    self._model_name = model_name or f'function:{function_name}:{stream_function_name}'

    # Use a default profile that supports JSON schema and object output if none provided
    if profile is None:
        profile = ModelProfile(
            supports_json_schema_output=True,
            supports_json_object_output=True,
        )
    super().__init__(settings=settings, profile=profile)

```

#### model_name

```python
model_name: str

```

The model name.

#### system

```python
system: str

```

The system / model provider.

### AgentInfo

Information about an agent.

This is passed as the second to functions used within FunctionModel.

Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`

```python
@dataclass(frozen=True, kw_only=True)
class AgentInfo:
    """Information about an agent.

    This is passed as the second to functions used within [`FunctionModel`][pydantic_ai.models.function.FunctionModel].
    """

    function_tools: list[ToolDefinition]
    """The function tools available on this agent.

    These are the tools registered via the [`tool`][pydantic_ai.Agent.tool] and
    [`tool_plain`][pydantic_ai.Agent.tool_plain] decorators.
    """
    allow_text_output: bool
    """Whether a plain text output is allowed."""
    output_tools: list[ToolDefinition]
    """The tools that can called to produce the final output of the run."""
    model_settings: ModelSettings | None
    """The model settings passed to the run call."""
    model_request_parameters: ModelRequestParameters
    """The model request parameters passed to the run call."""
    instructions: str | None
    """The instructions passed to model."""

```

#### function_tools

```python
function_tools: list[ToolDefinition]

```

The function tools available on this agent.

These are the tools registered via the tool and tool_plain decorators.

#### allow_text_output

```python
allow_text_output: bool

```

Whether a plain text output is allowed.

#### output_tools

```python
output_tools: list[ToolDefinition]

```

The tools that can called to produce the final output of the run.

#### model_settings

```python
model_settings: ModelSettings | None

```

The model settings passed to the run call.

#### model_request_parameters

```python
model_request_parameters: ModelRequestParameters

```

The model request parameters passed to the run call.

#### instructions

```python
instructions: str | None

```

The instructions passed to model.

### DeltaToolCall

Incremental change to a tool call.

Used to describe a chunk when streaming structured responses.

Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`

```python
@dataclass
class DeltaToolCall:
    """Incremental change to a tool call.

    Used to describe a chunk when streaming structured responses.
    """

    name: str | None = None
    """Incremental change to the name of the tool."""

    json_args: str | None = None
    """Incremental change to the arguments as JSON"""

    _: KW_ONLY

    tool_call_id: str | None = None
    """Incremental change to the tool call ID."""

```

#### name

```python
name: str | None = None

```

Incremental change to the name of the tool.

#### json_args

```python
json_args: str | None = None

```

Incremental change to the arguments as JSON

#### tool_call_id

```python
tool_call_id: str | None = None

```

Incremental change to the tool call ID.

### DeltaThinkingPart

Incremental change to a thinking part.

Used to describe a chunk when streaming thinking responses.

Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`

```python
@dataclass(kw_only=True)
class DeltaThinkingPart:
    """Incremental change to a thinking part.

    Used to describe a chunk when streaming thinking responses.
    """

    content: str | None = None
    """Incremental change to the thinking content."""
    signature: str | None = None
    """Incremental change to the thinking signature."""

```

#### content

```python
content: str | None = None

```

Incremental change to the thinking content.

#### signature

```python
signature: str | None = None

```

Incremental change to the thinking signature.

### DeltaToolCalls

```python
DeltaToolCalls: TypeAlias = dict[int, DeltaToolCall]

```

A mapping of tool call IDs to incremental changes.

### DeltaThinkingCalls

```python
DeltaThinkingCalls: TypeAlias = dict[int, DeltaThinkingPart]

```

A mapping of thinking call IDs to incremental changes.

### FunctionDef

```python
FunctionDef: TypeAlias = Callable[
    [list[ModelMessage], AgentInfo],
    ModelResponse | Awaitable[ModelResponse],
]

```

A function used to generate a non-streamed response.

### StreamFunctionDef

```python
StreamFunctionDef: TypeAlias = Callable[
    [list[ModelMessage], AgentInfo],
    AsyncIterator[
        str
        | DeltaToolCalls
        | DeltaThinkingCalls
        | BuiltinToolCallsReturns
    ],
]

```

A function used to generate a streamed response.

While this is defined as having return type of `AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls | BuiltinTools]`, it should really be considered as `AsyncIterator[str] | AsyncIterator[DeltaToolCalls] | AsyncIterator[DeltaThinkingCalls]`,

E.g. you need to yield all text, all `DeltaToolCalls`, all `DeltaThinkingCalls`, or all `BuiltinToolCallsReturns`, not mix them.

### FunctionStreamedResponse

Bases: `StreamedResponse`

Implementation of `StreamedResponse` for FunctionModel.

Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`

```python
@dataclass
class FunctionStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for [FunctionModel][pydantic_ai.models.function.FunctionModel]."""

    _model_name: str
    _iter: AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls | BuiltinToolCallsReturns]
    _timestamp: datetime = field(default_factory=_utils.now_utc)

    def __post_init__(self):
        self._usage += _estimate_usage([])

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for item in self._iter:
            if isinstance(item, str):
                response_tokens = _estimate_string_tokens(item)
                self._usage += usage.RequestUsage(output_tokens=response_tokens)
                maybe_event = self._parts_manager.handle_text_delta(vendor_part_id='content', content=item)
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event
            elif isinstance(item, dict) and item:
                for dtc_index, delta in item.items():
                    if isinstance(delta, DeltaThinkingPart):
                        if delta.content:  # pragma: no branch
                            response_tokens = _estimate_string_tokens(delta.content)
                            self._usage += usage.RequestUsage(output_tokens=response_tokens)
                        yield self._parts_manager.handle_thinking_delta(
                            vendor_part_id=dtc_index,
                            content=delta.content,
                            signature=delta.signature,
                            provider_name='function' if delta.signature else None,
                        )
                    elif isinstance(delta, DeltaToolCall):
                        if delta.json_args:
                            response_tokens = _estimate_string_tokens(delta.json_args)
                            self._usage += usage.RequestUsage(output_tokens=response_tokens)
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=dtc_index,
                            tool_name=delta.name,
                            args=delta.json_args,
                            tool_call_id=delta.tool_call_id,
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    elif isinstance(delta, BuiltinToolCallPart):
                        if content := delta.args_as_json_str():  # pragma: no branch
                            response_tokens = _estimate_string_tokens(content)
                            self._usage += usage.RequestUsage(output_tokens=response_tokens)
                        yield self._parts_manager.handle_part(vendor_part_id=dtc_index, part=delta)
                    elif isinstance(delta, BuiltinToolReturnPart):
                        if content := delta.model_response_str():  # pragma: no branch
                            response_tokens = _estimate_string_tokens(content)
                            self._usage += usage.RequestUsage(output_tokens=response_tokens)
                        yield self._parts_manager.handle_part(vendor_part_id=dtc_index, part=delta)
                    else:
                        assert_never(delta)

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> None:
        """Get the provider name."""
        return None

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
provider_name: None

```

Get the provider name.

#### timestamp

```python
timestamp: datetime

```

Get the timestamp of the response.

# `pydantic_ai.models.google`

Interface that uses the [`google-genai`](https://pypi.org/project/google-genai/) package under the hood to access Google's Gemini models via both the Generative Language API and Vertex AI.

## Setup

For details on how to set up authentication with this model, see [model configuration for Google](../../../models/google/).

### LatestGoogleModelNames

```python
LatestGoogleModelNames = Literal[
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-3-pro-preview",
    "gemini-3-pro-preview-preview-09-2025",
    "gemini-3-pro-preview-image",
    "gemini-3-pro-preview-lite",
    "gemini-3-pro-preview-lite-preview-09-2025",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
]

```

Latest Gemini models.

### GoogleModelName

```python
GoogleModelName = str | LatestGoogleModelNames

```

Possible Gemini model names.

Since Gemini supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [the Gemini API docs](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations) for a full list.

### GoogleModelSettings

Bases: `ModelSettings`

Settings used for a Gemini model request.

Source code in `pydantic_ai_slim/pydantic_ai/models/google.py`

```python
class GoogleModelSettings(ModelSettings, total=False):
    """Settings used for a Gemini model request."""

    # ALL FIELDS MUST BE `gemini_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    google_safety_settings: list[SafetySettingDict]
    """The safety settings to use for the model.

    See <https://ai.google.dev/gemini-api/docs/safety-settings> for more information.
    """

    google_thinking_config: ThinkingConfigDict
    """The thinking configuration to use for the model.

    See <https://ai.google.dev/gemini-api/docs/thinking> for more information.
    """

    google_labels: dict[str, str]
    """User-defined metadata to break down billed charges. Only supported by the Vertex AI API.

    See the [Gemini API docs](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/add-labels-to-api-calls) for use cases and limitations.
    """

    google_video_resolution: MediaResolution
    """The video resolution to use for the model.

    See <https://ai.google.dev/api/generate-content#MediaResolution> for more information.
    """

    google_cached_content: str
    """The name of the cached content to use for the model.

    See <https://ai.google.dev/gemini-api/docs/caching> for more information.
    """

```

#### google_safety_settings

```python
google_safety_settings: list[SafetySettingDict]

```

The safety settings to use for the model.

See <https://ai.google.dev/gemini-api/docs/safety-settings> for more information.

#### google_thinking_config

```python
google_thinking_config: ThinkingConfigDict

```

The thinking configuration to use for the model.

See <https://ai.google.dev/gemini-api/docs/thinking> for more information.

#### google_labels

```python
google_labels: dict[str, str]

```

User-defined metadata to break down billed charges. Only supported by the Vertex AI API.

See the [Gemini API docs](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/add-labels-to-api-calls) for use cases and limitations.

#### google_video_resolution

```python
google_video_resolution: MediaResolution

```

The video resolution to use for the model.

See <https://ai.google.dev/api/generate-content#MediaResolution> for more information.

#### google_cached_content

```python
google_cached_content: str

```

The name of the cached content to use for the model.

See <https://ai.google.dev/gemini-api/docs/caching> for more information.

### GoogleModel

Bases: `Model`

A model that uses Gemini via `generativelanguage.googleapis.com` API.

This is implemented from scratch rather than using a dedicated SDK, good API documentation is available [here](https://ai.google.dev/api).

Apart from `__init__`, all methods are private or match those of the base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/google.py`

```python
@dataclass(init=False)
class GoogleModel(Model):
    """A model that uses Gemini via `generativelanguage.googleapis.com` API.

    This is implemented from scratch rather than using a dedicated SDK, good API documentation is
    available [here](https://ai.google.dev/api).

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: Client = field(repr=False)

    _model_name: GoogleModelName = field(repr=False)
    _provider: Provider[Client] = field(repr=False)

    def __init__(
        self,
        model_name: GoogleModelName,
        *,
        provider: Literal['google-gla', 'google-vertex', 'gateway'] | Provider[Client] = 'google-gla',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Gemini model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use for authentication and API access. Can be either the string
                'google-gla' or 'google-vertex' or an instance of `Provider[google.genai.AsyncClient]`.
                Defaults to 'google-gla'.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: The model settings to use. Defaults to None.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/google-vertex' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> GoogleModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        supports_native_output_with_builtin_tools = GoogleModelProfile.from_profile(
            self.profile
        ).google_supports_native_output_with_builtin_tools
        if model_request_parameters.builtin_tools and model_request_parameters.output_tools:
            if model_request_parameters.output_mode == 'auto':
                output_mode = 'native' if supports_native_output_with_builtin_tools else 'prompted'
                model_request_parameters = replace(model_request_parameters, output_mode=output_mode)
            else:
                output_mode = 'NativeOutput' if supports_native_output_with_builtin_tools else 'PromptedOutput'
                raise UserError(
                    f'Google does not support output tools and built-in tools at the same time. Use `output_type={output_mode}(...)` instead.'
                )
        return super().prepare_request(model_settings, model_request_parameters)

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
        model_settings = cast(GoogleModelSettings, model_settings or {})
        response = await self._generate_content(messages, False, model_settings, model_request_parameters)
        return self._process_response(response)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        model_settings = cast(GoogleModelSettings, model_settings or {})
        contents, generation_config = await self._build_content_and_config(
            messages, model_settings, model_request_parameters
        )

        # Annoyingly, the type of `GenerateContentConfigDict.get` is "partially `Unknown`" because `response_schema` includes `typing._UnionGenericAlias`,
        # so without this we'd need `pyright: ignore[reportUnknownMemberType]` on every line and wouldn't get type checking anyway.
        generation_config = cast(dict[str, Any], generation_config)

        config = CountTokensConfigDict(
            http_options=generation_config.get('http_options'),
        )
        if self._provider.name != 'google-gla':
            # The fields are not supported by the Gemini API per https://github.com/googleapis/python-genai/blob/7e4ec284dc6e521949626f3ed54028163ef9121d/google/genai/models.py#L1195-L1214
            config.update(  # pragma: lax no cover
                system_instruction=generation_config.get('system_instruction'),
                tools=cast(list[ToolDict], generation_config.get('tools')),
                # Annoyingly, GenerationConfigDict has fewer fields than GenerateContentConfigDict, and no extra fields are allowed.
                generation_config=GenerationConfigDict(
                    temperature=generation_config.get('temperature'),
                    top_p=generation_config.get('top_p'),
                    max_output_tokens=generation_config.get('max_output_tokens'),
                    stop_sequences=generation_config.get('stop_sequences'),
                    presence_penalty=generation_config.get('presence_penalty'),
                    frequency_penalty=generation_config.get('frequency_penalty'),
                    seed=generation_config.get('seed'),
                    thinking_config=generation_config.get('thinking_config'),
                    media_resolution=generation_config.get('media_resolution'),
                    response_mime_type=generation_config.get('response_mime_type'),
                    response_json_schema=generation_config.get('response_json_schema'),
                ),
            )

        response = await self.client.aio.models.count_tokens(
            model=self._model_name,
            contents=contents,
            config=config,
        )
        if response.total_tokens is None:
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Total tokens missing from Gemini response', str(response)
            )
        return usage.RequestUsage(
            input_tokens=response.total_tokens,
        )

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
        model_settings = cast(GoogleModelSettings, model_settings or {})
        response = await self._generate_content(messages, True, model_settings, model_request_parameters)
        yield await self._process_streamed_response(response, model_request_parameters)  # type: ignore

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolDict] | None:
        tools: list[ToolDict] = [
            ToolDict(function_declarations=[_function_declaration_from_tool(t)])
            for t in model_request_parameters.tool_defs.values()
        ]

        if model_request_parameters.builtin_tools:
            if model_request_parameters.function_tools:
                raise UserError('Google does not support function tools and built-in tools at the same time.')

            for tool in model_request_parameters.builtin_tools:
                if isinstance(tool, WebSearchTool):
                    tools.append(ToolDict(google_search=GoogleSearchDict()))
                elif isinstance(tool, UrlContextTool):
                    tools.append(ToolDict(url_context=UrlContextDict()))
                elif isinstance(tool, CodeExecutionTool):
                    tools.append(ToolDict(code_execution=ToolCodeExecutionDict()))
                elif isinstance(tool, ImageGenerationTool):  # pragma: no branch
                    if not self.profile.supports_image_output:
                        raise UserError(
                            "`ImageGenerationTool` is not supported by this model. Use a model with 'image' in the name instead."
                        )
                else:  # pragma: no cover
                    raise UserError(
                        f'`{tool.__class__.__name__}` is not supported by `GoogleModel`. If it should be, please file an issue.'
                    )
        return tools or None

    def _get_tool_config(
        self, model_request_parameters: ModelRequestParameters, tools: list[ToolDict] | None
    ) -> ToolConfigDict | None:
        if not model_request_parameters.allow_text_output and tools:
            names: list[str] = []
            for tool in tools:
                for function_declaration in tool.get('function_declarations') or []:
                    if name := function_declaration.get('name'):  # pragma: no branch
                        names.append(name)
            return _tool_config(names)
        else:
            return None

    @overload
    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> GenerateContentResponse: ...

    @overload
    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Awaitable[AsyncIterator[GenerateContentResponse]]: ...

    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> GenerateContentResponse | Awaitable[AsyncIterator[GenerateContentResponse]]:
        contents, config = await self._build_content_and_config(messages, model_settings, model_request_parameters)
        func = self.client.aio.models.generate_content_stream if stream else self.client.aio.models.generate_content
        try:
            return await func(model=self._model_name, contents=contents, config=config)  # type: ignore
        except errors.APIError as e:
            if (status_code := e.code) >= 400:
                raise ModelHTTPError(
                    status_code=status_code,
                    model_name=self._model_name,
                    body=cast(Any, e.details),  # pyright: ignore[reportUnknownMemberType]
                ) from e
            raise ModelAPIError(model_name=self._model_name, message=str(e)) from e

    async def _build_content_and_config(
        self,
        messages: list[ModelMessage],
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ContentUnionDict], GenerateContentConfigDict]:
        tools = self._get_tools(model_request_parameters)
        if tools and not self.profile.supports_tools:
            raise UserError('Tools are not supported by this model.')

        response_mime_type = None
        response_schema = None
        if model_request_parameters.output_mode == 'native':
            if model_request_parameters.function_tools:
                raise UserError(
                    'Google does not support `NativeOutput` and function tools at the same time. Use `output_type=ToolOutput(...)` instead.'
                )
            response_mime_type = 'application/json'
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_schema = self._map_response_schema(output_object)
        elif model_request_parameters.output_mode == 'prompted' and not tools:
            if not self.profile.supports_json_object_output:
                raise UserError('JSON output is not supported by this model.')
            response_mime_type = 'application/json'

        tool_config = self._get_tool_config(model_request_parameters, tools)
        system_instruction, contents = await self._map_messages(messages, model_request_parameters)

        modalities = [Modality.TEXT.value]
        if self.profile.supports_image_output:
            modalities.append(Modality.IMAGE.value)

        http_options: HttpOptionsDict = {
            'headers': {'Content-Type': 'application/json', 'User-Agent': get_user_agent()}
        }
        if timeout := model_settings.get('timeout'):
            if isinstance(timeout, int | float):
                http_options['timeout'] = int(1000 * timeout)
            else:
                raise UserError('Google does not support setting ModelSettings.timeout to a httpx.Timeout')

        config = GenerateContentConfigDict(
            http_options=http_options,
            system_instruction=system_instruction,
            temperature=model_settings.get('temperature'),
            top_p=model_settings.get('top_p'),
            max_output_tokens=model_settings.get('max_tokens'),
            stop_sequences=model_settings.get('stop_sequences'),
            presence_penalty=model_settings.get('presence_penalty'),
            frequency_penalty=model_settings.get('frequency_penalty'),
            seed=model_settings.get('seed'),
            safety_settings=model_settings.get('google_safety_settings'),
            thinking_config=model_settings.get('google_thinking_config'),
            labels=model_settings.get('google_labels'),
            media_resolution=model_settings.get('google_video_resolution'),
            cached_content=model_settings.get('google_cached_content'),
            tools=cast(ToolListUnionDict, tools),
            tool_config=tool_config,
            response_mime_type=response_mime_type,
            response_json_schema=response_schema,
            response_modalities=modalities,
        )
        return contents, config

    def _process_response(self, response: GenerateContentResponse) -> ModelResponse:
        if not response.candidates:
            raise UnexpectedModelBehavior('Expected at least one candidate in Gemini response')  # pragma: no cover

        candidate = response.candidates[0]

        vendor_id = response.response_id
        vendor_details: dict[str, Any] | None = None
        finish_reason: FinishReason | None = None
        raw_finish_reason = candidate.finish_reason
        if raw_finish_reason:  # pragma: no branch
            vendor_details = {'finish_reason': raw_finish_reason.value}
            finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        if candidate.content is None or candidate.content.parts is None:
            if finish_reason == 'content_filter' and raw_finish_reason:
                raise UnexpectedModelBehavior(
                    f'Content filter {raw_finish_reason.value!r} triggered', response.model_dump_json()
                )
            parts = []  # pragma: no cover
        else:
            parts = candidate.content.parts or []

        usage = _metadata_as_usage(response, provider=self._provider.name, provider_url=self._provider.base_url)
        return _process_response_from_parts(
            parts,
            candidate.grounding_metadata,
            response.model_version or self._model_name,
            self._provider.name,
            usage,
            vendor_id=vendor_id,
            vendor_details=vendor_details,
            finish_reason=finish_reason,
        )

    async def _process_streamed_response(
        self, response: AsyncIterator[GenerateContentResponse], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')  # pragma: no cover

        return GeminiStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.model_version or self._model_name,
            _response=peekable_response,
            _timestamp=first_chunk.create_time or _utils.now_utc(),
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
        )

    async def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> tuple[ContentDict | None, list[ContentUnionDict]]:
        contents: list[ContentUnionDict] = []
        system_parts: list[PartDict] = []

        for m in messages:
            if isinstance(m, ModelRequest):
                message_parts: list[PartDict] = []

                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        system_parts.append({'text': part.content})
                    elif isinstance(part, UserPromptPart):
                        message_parts.extend(await self._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        message_parts.append(
                            {
                                'function_response': {
                                    'name': part.tool_name,
                                    'response': part.model_response_object(),
                                    'id': part.tool_call_id,
                                }
                            }
                        )
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            message_parts.append({'text': part.model_response()})  # pragma: no cover
                        else:
                            message_parts.append(
                                {
                                    'function_response': {
                                        'name': part.tool_name,
                                        'response': {'call_error': part.model_response()},
                                        'id': part.tool_call_id,
                                    }
                                }
                            )
                    else:
                        assert_never(part)

                if message_parts:
                    contents.append({'role': 'user', 'parts': message_parts})
            elif isinstance(m, ModelResponse):
                maybe_content = _content_model_response(m, self.system)
                if maybe_content:
                    contents.append(maybe_content)
            else:
                assert_never(m)

        # Google GenAI requires at least one part in the message.
        if not contents:
            contents = [{'role': 'user', 'parts': [{'text': ''}]}]

        if instructions := self._get_instructions(messages, model_request_parameters):
            system_parts.insert(0, {'text': instructions})
        system_instruction = ContentDict(role='user', parts=system_parts) if system_parts else None

        return system_instruction, contents

    async def _map_user_prompt(self, part: UserPromptPart) -> list[PartDict]:
        if isinstance(part.content, str):
            return [{'text': part.content}]
        else:
            content: list[PartDict] = []
            for item in part.content:
                if isinstance(item, str):
                    content.append({'text': item})
                elif isinstance(item, BinaryContent):
                    inline_data_dict: BlobDict = {'data': item.data, 'mime_type': item.media_type}
                    part_dict: PartDict = {'inline_data': inline_data_dict}
                    if item.vendor_metadata:
                        part_dict['video_metadata'] = cast(VideoMetadataDict, item.vendor_metadata)
                    content.append(part_dict)
                elif isinstance(item, VideoUrl) and item.is_youtube:
                    file_data_dict: FileDataDict = {'file_uri': item.url, 'mime_type': item.media_type}
                    part_dict: PartDict = {'file_data': file_data_dict}
                    if item.vendor_metadata:  # pragma: no branch
                        part_dict['video_metadata'] = cast(VideoMetadataDict, item.vendor_metadata)
                    content.append(part_dict)
                elif isinstance(item, FileUrl):
                    if item.force_download or (
                        # google-gla does not support passing file urls directly, except for youtube videos
                        # (see above) and files uploaded to the file API (which cannot be downloaded anyway)
                        self.system == 'google-gla'
                        and not item.url.startswith(r'https://generativelanguage.googleapis.com/v1beta/files')
                    ):
                        downloaded_item = await download_item(item, data_format='bytes')
                        inline_data: BlobDict = {
                            'data': downloaded_item['data'],
                            'mime_type': downloaded_item['data_type'],
                        }
                        content.append({'inline_data': inline_data})
                    else:
                        file_data_dict: FileDataDict = {'file_uri': item.url, 'mime_type': item.media_type}
                        content.append({'file_data': file_data_dict})  # pragma: lax no cover
                elif isinstance(item, CachePoint):
                    # Google Gemini doesn't support prompt caching via CachePoint
                    pass
                else:
                    assert_never(item)
        return content

    def _map_response_schema(self, o: OutputObjectDefinition) -> dict[str, Any]:
        response_schema = o.json_schema.copy()
        if o.name:
            response_schema['title'] = o.name
        if o.description:
            response_schema['description'] = o.description

        return response_schema

```

#### __init__

```python
__init__(
    model_name: GoogleModelName,
    *,
    provider: (
        Literal["google-gla", "google-vertex", "gateway"]
        | Provider[Client]
    ) = "google-gla",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize a Gemini model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model_name` | `GoogleModelName` | The name of the model to use. | *required* | | `provider` | `Literal['google-gla', 'google-vertex', 'gateway'] | Provider[Client]` | The provider to use for authentication and API access. Can be either the string 'google-gla' or 'google-vertex' or an instance of Provider[google.genai.AsyncClient]. Defaults to 'google-gla'. | `'google-gla'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider based on the model name. | `None` | | `settings` | `ModelSettings | None` | The model settings to use. Defaults to None. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/google.py`

```python
def __init__(
    self,
    model_name: GoogleModelName,
    *,
    provider: Literal['google-gla', 'google-vertex', 'gateway'] | Provider[Client] = 'google-gla',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize a Gemini model.

    Args:
        model_name: The name of the model to use.
        provider: The provider to use for authentication and API access. Can be either the string
            'google-gla' or 'google-vertex' or an instance of `Provider[google.genai.AsyncClient]`.
            Defaults to 'google-gla'.
        profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
        settings: The model settings to use. Defaults to None.
    """
    self._model_name = model_name

    if isinstance(provider, str):
        provider = infer_provider('gateway/google-vertex' if provider == 'gateway' else provider)
    self._provider = provider
    self.client = provider.client

    super().__init__(settings=settings, profile=profile or provider.model_profile)

```

#### model_name

```python
model_name: GoogleModelName

```

The model name.

#### system

```python
system: str

```

The model provider.

### GeminiStreamedResponse

Bases: `StreamedResponse`

Implementation of `StreamedResponse` for the Gemini model.

Source code in `pydantic_ai_slim/pydantic_ai/models/google.py`

```python
@dataclass
class GeminiStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for the Gemini model."""

    _model_name: GoogleModelName
    _response: AsyncIterator[GenerateContentResponse]
    _timestamp: datetime
    _provider_name: str
    _provider_url: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        code_execution_tool_call_id: str | None = None
        async for chunk in self._response:
            self._usage = _metadata_as_usage(chunk, self._provider_name, self._provider_url)

            if not chunk.candidates:
                continue  # pragma: no cover

            candidate = chunk.candidates[0]

            if chunk.response_id:  # pragma: no branch
                self.provider_response_id = chunk.response_id

            raw_finish_reason = candidate.finish_reason
            if raw_finish_reason:
                self.provider_details = {'finish_reason': raw_finish_reason.value}
                self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

            # Google streams the grounding metadata (including the web search queries and results)
            # _after_ the text that was generated using it, so it would show up out of order in the stream,
            # and cause issues with the logic that doesn't consider text ahead of built-in tool calls as output.
            # If that gets fixed (or we have a workaround), we can uncomment this:
            # web_search_call, web_search_return = _map_grounding_metadata(
            #     candidate.grounding_metadata, self.provider_name
            # )
            # if web_search_call and web_search_return:
            #     yield self._parts_manager.handle_part(vendor_part_id=uuid4(), part=web_search_call)
            #     yield self._parts_manager.handle_part(
            #         vendor_part_id=uuid4(), part=web_search_return
            #     )

            if candidate.content is None or candidate.content.parts is None:
                if self.finish_reason == 'content_filter' and raw_finish_reason:  # pragma: no cover
                    raise UnexpectedModelBehavior(
                        f'Content filter {raw_finish_reason.value!r} triggered', chunk.model_dump_json()
                    )
                else:  # pragma: no cover
                    continue

            parts = candidate.content.parts
            if not parts:
                continue  # pragma: no cover

            for part in parts:
                provider_details: dict[str, Any] | None = None
                if part.thought_signature:
                    # Per https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#thought-signatures:
                    # - Always send the thought_signature back to the model inside its original Part.
                    # - Don't merge a Part containing a signature with one that does not. This breaks the positional context of the thought.
                    # - Don't combine two Parts that both contain signatures, as the signature strings cannot be merged.
                    thought_signature = base64.b64encode(part.thought_signature).decode('utf-8')
                    provider_details = {'thought_signature': thought_signature}

                if part.text is not None:
                    if len(part.text) == 0 and not provider_details:
                        continue
                    if part.thought:
                        yield self._parts_manager.handle_thinking_delta(
                            vendor_part_id=None, content=part.text, provider_details=provider_details
                        )
                    else:
                        maybe_event = self._parts_manager.handle_text_delta(
                            vendor_part_id=None, content=part.text, provider_details=provider_details
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                elif part.function_call:
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=uuid4(),
                        tool_name=part.function_call.name,
                        args=part.function_call.args,
                        tool_call_id=part.function_call.id,
                        provider_details=provider_details,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif part.inline_data is not None:
                    data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    assert data and mime_type, 'Inline data must have data and mime type'
                    content = BinaryContent(data=data, media_type=mime_type)
                    yield self._parts_manager.handle_part(
                        vendor_part_id=uuid4(),
                        part=FilePart(content=BinaryContent.narrow_type(content), provider_details=provider_details),
                    )
                elif part.executable_code is not None:
                    code_execution_tool_call_id = _utils.generate_tool_call_id()
                    part = _map_executable_code(part.executable_code, self.provider_name, code_execution_tool_call_id)
                    part.provider_details = provider_details
                    yield self._parts_manager.handle_part(vendor_part_id=uuid4(), part=part)
                elif part.code_execution_result is not None:
                    assert code_execution_tool_call_id is not None
                    part = _map_code_execution_result(
                        part.code_execution_result, self.provider_name, code_execution_tool_call_id
                    )
                    part.provider_details = provider_details
                    yield self._parts_manager.handle_part(vendor_part_id=uuid4(), part=part)
                else:
                    assert part.function_response is not None, f'Unexpected part: {part}'  # pragma: no cover

    @property
    def model_name(self) -> GoogleModelName:
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
model_name: GoogleModelName

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

# `pydantic_ai.models.groq`

## Setup

For details on how to set up authentication with this model, see [model configuration for Groq](../../../models/groq/).

### ProductionGroqModelNames

```python
ProductionGroqModelNames = Literal[
    "distil-whisper-large-v3-en",
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
]

```

Production Groq models from <https://console.groq.com/docs/models#production-models>.

### PreviewGroqModelNames

```python
PreviewGroqModelNames = Literal[
    "playai-tts",
    "playai-tts-arabic",
    "qwen-qwq-32b",
    "mistral-saba-24b",
    "qwen-2.5-coder-32b",
    "qwen-2.5-32b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-specdec",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "moonshotai/kimi-k2-instruct",
]

```

Preview Groq models from <https://console.groq.com/docs/models#preview-models>.

### GroqModelName

```python
GroqModelName = (
    str | ProductionGroqModelNames | PreviewGroqModelNames
)

```

Possible Groq model names.

Since Groq supports a variety of models and the list changes frequencly, we explicitly list the named models as of 2025-03-31 but allow any name in the type hints.

See <https://console.groq.com/docs/models> for an up to date date list of models and more details.

### GroqModelSettings

Bases: `ModelSettings`

Settings used for a Groq model request.

Source code in `pydantic_ai_slim/pydantic_ai/models/groq.py`

```python
class GroqModelSettings(ModelSettings, total=False):
    """Settings used for a Groq model request."""

    # ALL FIELDS MUST BE `groq_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    groq_reasoning_format: Literal['hidden', 'raw', 'parsed']
    """The format of the reasoning output.

    See [the Groq docs](https://console.groq.com/docs/reasoning#reasoning-format) for more details.
    """

```

#### groq_reasoning_format

```python
groq_reasoning_format: Literal['hidden', 'raw', 'parsed']

```

The format of the reasoning output.

See [the Groq docs](https://console.groq.com/docs/reasoning#reasoning-format) for more details.

### GroqModel

Bases: `Model`

A model that uses the Groq API.

Internally, this uses the [Groq Python client](https://github.com/groq/groq-python) to interact with the API.

Apart from `__init__`, all methods are private or match those of the base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/groq.py`

```python
@dataclass(init=False)
class GroqModel(Model):
    """A model that uses the Groq API.

    Internally, this uses the [Groq Python client](https://github.com/groq/groq-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncGroq = field(repr=False)

    _model_name: GroqModelName = field(repr=False)
    _provider: Provider[AsyncGroq] = field(repr=False)

    def __init__(
        self,
        model_name: GroqModelName,
        *,
        provider: Literal['groq', 'gateway'] | Provider[AsyncGroq] = 'groq',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Groq model.

        Args:
            model_name: The name of the Groq model to use. List of model names available
                [here](https://console.groq.com/docs/models).
            provider: The provider to use for authentication and API access. Can be either the string
                'groq' or an instance of `Provider[AsyncGroq]`. If not provided, a new provider will be
                created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/groq' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> GroqModelName:
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
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        try:
            response = await self._completions_create(
                messages, False, cast(GroqModelSettings, model_settings or {}), model_request_parameters
            )
        except ModelHTTPError as e:
            if isinstance(e.body, dict):  # pragma: no branch
                # The Groq SDK tries to be helpful by raising an exception when generated tool arguments don't match the schema,
                # but we'd rather handle it ourselves so we can tell the model to retry the tool call.
                try:
                    error = _GroqToolUseFailedError.model_validate(e.body)  # pyright: ignore[reportUnknownMemberType]
                    tool_call_part = ToolCallPart(
                        tool_name=error.error.failed_generation.name,
                        args=error.error.failed_generation.arguments,
                    )
                    return ModelResponse(
                        parts=[tool_call_part],
                        model_name=e.model_name,
                        timestamp=_utils.now_utc(),
                        provider_name=self._provider.name,
                        finish_reason='error',
                    )
                except ValidationError:
                    pass
            raise
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
            messages, True, cast(GroqModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: GroqModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[chat.ChatCompletionChunk]:
        pass

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: GroqModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion:
        pass

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: GroqModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[chat.ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)
        tools += self._get_builtin_tools(model_request_parameters)
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        groq_messages = self._map_messages(messages, model_request_parameters)

        response_format: chat.completion_create_params.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == 'prompted'
            and not tools
            and self.profile.supports_json_object_output
        ):  # pragma: no branch
            response_format = {'type': 'json_object'}

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=groq_messages,
                n=1,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stop=model_settings.get('stop_sequences', NOT_GIVEN),
                stream=stream,
                response_format=response_format or NOT_GIVEN,
                max_tokens=model_settings.get('max_tokens', NOT_GIVEN),
                temperature=model_settings.get('temperature', NOT_GIVEN),
                top_p=model_settings.get('top_p', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                seed=model_settings.get('seed', NOT_GIVEN),
                presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
                reasoning_format=model_settings.get('groq_reasoning_format', NOT_GIVEN),
                frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
                logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e  # pragma: no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = number_to_datetime(response.created)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []
        if choice.message.reasoning is not None:
            # NOTE: The `reasoning` field is only present if `groq_reasoning_format` is set to `parsed`.
            items.append(ThinkingPart(content=choice.message.reasoning))
        if choice.message.executed_tools:
            for tool in choice.message.executed_tools:
                call_part, return_part = _map_executed_tool(tool, self.system)
                if call_part and return_part:  # pragma: no branch
                    items.append(call_part)
                    items.append(return_part)
        if choice.message.content:
            # NOTE: The `<think>` tag is only present if `groq_reasoning_format` is set to `raw`.
            items.extend(split_content_into_text_and_thinking(choice.message.content, self.profile.thinking_tags))
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(ToolCallPart(tool_name=c.function.name, args=c.function.arguments, tool_call_id=c.id))

        raw_finish_reason = choice.finish_reason
        provider_details = {'finish_reason': raw_finish_reason}
        finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)
        return ModelResponse(
            parts=items,
            usage=_map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self, response: AsyncStream[chat.ChatCompletionChunk], model_request_parameters: ModelRequestParameters
    ) -> GroqStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        return GroqStreamedResponse(
            model_request_parameters=model_request_parameters,
            _response=peekable_response,
            _model_name=first_chunk.model,
            _model_profile=self.profile,
            _timestamp=number_to_datetime(first_chunk.created),
            _provider_name=self._provider.name,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_builtin_tools(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[chat.ChatCompletionToolParam]:
        tools: list[chat.ChatCompletionToolParam] = []
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                if not GroqModelProfile.from_profile(self.profile).groq_always_has_web_search_builtin_tool:
                    raise UserError('`WebSearchTool` is not supported by Groq')  # pragma: no cover
            else:
                raise UserError(
                    f'`{tool.__class__.__name__}` is not supported by `GroqModel`. If it should be, please file an issue.'
                )
        return tools

    def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `groq.types.ChatCompletionMessageParam`."""
        groq_messages: list[chat.ChatCompletionMessageParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                groq_messages.extend(self._map_user_message(message))
            elif isinstance(message, ModelResponse):
                texts: list[str] = []
                tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        texts.append(item.content)
                    elif isinstance(item, ToolCallPart):
                        tool_calls.append(self._map_tool_call(item))
                    elif isinstance(item, ThinkingPart):
                        start_tag, end_tag = self.profile.thinking_tags
                        texts.append('\n'.join([start_tag, item.content, end_tag]))
                    elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                        # These are not currently sent back
                        pass
                    elif isinstance(item, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
                        pass
                    else:
                        assert_never(item)
                message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
                if texts:
                    # Note: model responses from this model should only have one text item, so the following
                    # shouldn't merge multiple texts into one unless you switch models between runs:
                    message_param['content'] = '\n\n'.join(texts)
                if tool_calls:
                    message_param['tool_calls'] = tool_calls
                groq_messages.append(message_param)
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages, model_request_parameters):
            groq_messages.insert(0, chat.ChatCompletionSystemMessageParam(role='system', content=instructions))
        return groq_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
        return chat.ChatCompletionMessageToolCallParam(
            id=_guard_tool_call_id(t=t),
            type='function',
            function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
        )

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> chat.ChatCompletionToolParam:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description or '',
                'parameters': f.parameters_json_schema,
            },
        }

    def _map_json_schema(self, o: OutputObjectDefinition) -> chat.completion_create_params.ResponseFormat:
        response_format_param: chat.completion_create_params.ResponseFormatResponseFormatJsonSchema = {
            'type': 'json_schema',
            'json_schema': {
                'name': o.name or DEFAULT_OUTPUT_TOOL_NAME,
                'schema': o.json_schema,
                'strict': o.strict,
            },
        }
        if o.description:  # pragma: no branch
            response_format_param['json_schema']['description'] = o.description
        return response_format_param

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield cls._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):  # pragma: no branch
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(  # pragma: no cover
                        role='user', content=part.model_response()
                    )
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )

    @staticmethod
    def _map_user_prompt(part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
        content: str | list[chat.ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(chat.ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url = ImageURL(url=item.url)
                    content.append(chat.ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        image_url = ImageURL(url=item.data_uri)
                        content.append(chat.ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                    else:
                        raise RuntimeError('Only images are supported for binary content in Groq.')
                elif isinstance(item, DocumentUrl):  # pragma: no cover
                    raise RuntimeError('DocumentUrl is not supported in Groq.')
                else:  # pragma: no cover
                    raise RuntimeError(f'Unsupported content type: {type(item)}')

        return chat.ChatCompletionUserMessageParam(role='user', content=content)

```

#### __init__

```python
__init__(
    model_name: GroqModelName,
    *,
    provider: (
        Literal["groq", "gateway"] | Provider[AsyncGroq]
    ) = "groq",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize a Groq model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model_name` | `GroqModelName` | The name of the Groq model to use. List of model names available here. | *required* | | `provider` | `Literal['groq', 'gateway'] | Provider[AsyncGroq]` | The provider to use for authentication and API access. Can be either the string 'groq' or an instance of Provider[AsyncGroq]. If not provided, a new provider will be created using the other parameters. | `'groq'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider based on the model name. | `None` | | `settings` | `ModelSettings | None` | Model-specific settings that will be used as defaults for this model. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/groq.py`

```python
def __init__(
    self,
    model_name: GroqModelName,
    *,
    provider: Literal['groq', 'gateway'] | Provider[AsyncGroq] = 'groq',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize a Groq model.

    Args:
        model_name: The name of the Groq model to use. List of model names available
            [here](https://console.groq.com/docs/models).
        provider: The provider to use for authentication and API access. Can be either the string
            'groq' or an instance of `Provider[AsyncGroq]`. If not provided, a new provider will be
            created using the other parameters.
        profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
        settings: Model-specific settings that will be used as defaults for this model.
    """
    self._model_name = model_name

    if isinstance(provider, str):
        provider = infer_provider('gateway/groq' if provider == 'gateway' else provider)
    self._provider = provider
    self.client = provider.client

    super().__init__(settings=settings, profile=profile or provider.model_profile)

```

#### model_name

```python
model_name: GroqModelName

```

The model name.

#### system

```python
system: str

```

The model provider.

### GroqStreamedResponse

Bases: `StreamedResponse`

Implementation of `StreamedResponse` for Groq models.

Source code in `pydantic_ai_slim/pydantic_ai/models/groq.py`

```python
@dataclass
class GroqStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Groq models."""

    _model_name: GroqModelName
    _model_profile: ModelProfile
    _response: AsyncIterable[chat.ChatCompletionChunk]
    _timestamp: datetime
    _provider_name: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        try:
            executed_tool_call_id: str | None = None
            reasoning_index = 0
            reasoning = False
            async for chunk in self._response:
                self._usage += _map_usage(chunk)

                if chunk.id:  # pragma: no branch
                    self.provider_response_id = chunk.id

                try:
                    choice = chunk.choices[0]
                except IndexError:
                    continue

                if raw_finish_reason := choice.finish_reason:
                    self.provider_details = {'finish_reason': raw_finish_reason}
                    self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

                if choice.delta.reasoning is not None:
                    if not reasoning:
                        reasoning_index += 1
                        reasoning = True

                    # NOTE: The `reasoning` field is only present if `groq_reasoning_format` is set to `parsed`.
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=f'reasoning-{reasoning_index}', content=choice.delta.reasoning
                    )
                else:
                    reasoning = False

                if choice.delta.executed_tools:
                    for tool in choice.delta.executed_tools:
                        call_part, return_part = _map_executed_tool(
                            tool, self.provider_name, streaming=True, tool_call_id=executed_tool_call_id
                        )
                        if call_part:
                            executed_tool_call_id = call_part.tool_call_id
                            yield self._parts_manager.handle_part(
                                vendor_part_id=f'executed_tools-{tool.index}-call', part=call_part
                            )
                        if return_part:
                            executed_tool_call_id = None
                            yield self._parts_manager.handle_part(
                                vendor_part_id=f'executed_tools-{tool.index}-return', part=return_part
                            )

                # Handle the text part of the response
                content = choice.delta.content
                if content:
                    maybe_event = self._parts_manager.handle_text_delta(
                        vendor_part_id='content',
                        content=content,
                        thinking_tags=self._model_profile.thinking_tags,
                        ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

                # Handle the tool calls
                for dtc in choice.delta.tool_calls or []:
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=dtc.index,
                        tool_name=dtc.function and dtc.function.name,
                        args=dtc.function and dtc.function.arguments,
                        tool_call_id=dtc.id,
                    )
                    if maybe_event is not None:
                        yield maybe_event
        except APIError as e:
            if isinstance(e.body, dict):  # pragma: no branch
                # The Groq SDK tries to be helpful by raising an exception when generated tool arguments don't match the schema,
                # but we'd rather handle it ourselves so we can tell the model to retry the tool call
                try:
                    error = _GroqToolUseFailedInnerError.model_validate(e.body)  # pyright: ignore[reportUnknownMemberType]
                    yield self._parts_manager.handle_tool_call_part(
                        vendor_part_id='tool_use_failed',
                        tool_name=error.failed_generation.name,
                        args=error.failed_generation.arguments,
                    )
                    return
                except ValidationError as e:  # pragma: no cover
                    pass
            raise  # pragma: no cover

    @property
    def model_name(self) -> GroqModelName:
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
model_name: GroqModelName

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

# `pydantic_ai.models.huggingface`

## Setup

For details on how to set up authentication with this model, see [model configuration for Hugging Face](../../../models/huggingface/).

### HuggingFaceModelSettings

Bases: `ModelSettings`

Settings used for a Hugging Face model request.

Source code in `pydantic_ai_slim/pydantic_ai/models/huggingface.py`

```python
class HuggingFaceModelSettings(ModelSettings, total=False):
    """Settings used for a Hugging Face model request."""

```

### HuggingFaceModel

Bases: `Model`

A model that uses Hugging Face Inference Providers.

Internally, this uses the [HF Python client](https://github.com/huggingface/huggingface_hub) to interact with the API.

Apart from `__init__`, all methods are private or match those of the base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/huggingface.py`

```python
@dataclass(init=False)
class HuggingFaceModel(Model):
    """A model that uses Hugging Face Inference Providers.

    Internally, this uses the [HF Python client](https://github.com/huggingface/huggingface_hub) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncInferenceClient = field(repr=False)

    _model_name: str = field(repr=False)
    _provider: Provider[AsyncInferenceClient] = field(repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['huggingface'] | Provider[AsyncInferenceClient] = 'huggingface',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Hugging Face model.

        Args:
            model_name: The name of the Model to use. You can browse available models [here](https://huggingface.co/models?pipeline_tag=text-generation&inference_provider=all&sort=trending).
            provider: The provider to use for Hugging Face Inference Providers. Can be either the string 'huggingface' or an
                instance of `Provider[AsyncInferenceClient]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def model_name(self) -> HuggingFaceModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._provider.name

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
            messages, False, cast(HuggingFaceModelSettings, model_settings or {}), model_request_parameters
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
            messages, True, cast(HuggingFaceModelSettings, model_settings or {}), model_request_parameters
        )
        yield await self._process_streamed_response(response, model_request_parameters)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: HuggingFaceModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterable[ChatCompletionStreamOutput]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: HuggingFaceModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ChatCompletionOutput: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: HuggingFaceModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ChatCompletionOutput | AsyncIterable[ChatCompletionStreamOutput]:
        tools = self._get_tools(model_request_parameters)

        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        if model_request_parameters.builtin_tools:
            raise UserError('HuggingFace does not support built-in tools')

        hf_messages = await self._map_messages(messages, model_request_parameters)

        try:
            return await self.client.chat.completions.create(  # type: ignore
                model=self._model_name,
                messages=hf_messages,  # type: ignore
                tools=tools,
                tool_choice=tool_choice or None,
                stream=stream,
                stop=model_settings.get('stop_sequences', None),
                temperature=model_settings.get('temperature', None),
                top_p=model_settings.get('top_p', None),
                seed=model_settings.get('seed', None),
                presence_penalty=model_settings.get('presence_penalty', None),
                frequency_penalty=model_settings.get('frequency_penalty', None),
                logit_bias=model_settings.get('logit_bias', None),  # type: ignore
                logprobs=model_settings.get('logprobs', None),
                top_logprobs=model_settings.get('top_logprobs', None),
                extra_body=model_settings.get('extra_body'),  # type: ignore
            )
        except aiohttp.ClientResponseError as e:
            raise ModelHTTPError(
                status_code=e.status,
                model_name=self.model_name,
                body=e.response_error_payload,  # type: ignore
            ) from e
        except HfHubHTTPError as e:
            raise ModelHTTPError(
                status_code=e.response.status_code,
                model_name=self.model_name,
                body=e.response.content,
            ) from e

    def _process_response(self, response: ChatCompletionOutput) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        if response.created:
            timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        else:
            timestamp = _now_utc()

        choice = response.choices[0]
        content = choice.message.content
        tool_calls = choice.message.tool_calls

        items: list[ModelResponsePart] = []

        if content:
            items.extend(split_content_into_text_and_thinking(content, self.profile.thinking_tags))
        if tool_calls is not None:
            for c in tool_calls:
                items.append(ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id))

        raw_finish_reason = choice.finish_reason
        provider_details = {'finish_reason': raw_finish_reason}
        finish_reason = _FINISH_REASON_MAP.get(cast(TextGenerationOutputFinishReason, raw_finish_reason), None)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self, response: AsyncIterable[ChatCompletionStreamOutput], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        return HuggingFaceStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.model,
            _model_profile=self.profile,
            _response=peekable_response,
            _timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
            _provider_name=self._provider.name,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ChatCompletionInputTool]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    async def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[ChatCompletionInputMessage | ChatCompletionOutputMessage]:
        """Just maps a `pydantic_ai.Message` to a `huggingface_hub.ChatCompletionInputMessage`."""
        hf_messages: list[ChatCompletionInputMessage | ChatCompletionOutputMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                async for item in self._map_user_message(message):
                    hf_messages.append(item)
            elif isinstance(message, ModelResponse):
                texts: list[str] = []
                tool_calls: list[ChatCompletionInputToolCall] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        texts.append(item.content)
                    elif isinstance(item, ToolCallPart):
                        tool_calls.append(self._map_tool_call(item))
                    elif isinstance(item, ThinkingPart):
                        start_tag, end_tag = self.profile.thinking_tags
                        texts.append('\n'.join([start_tag, item.content, end_tag]))
                    elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                        # This is currently never returned from huggingface
                        pass
                    elif isinstance(item, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
                        pass
                    else:
                        assert_never(item)
                message_param = ChatCompletionInputMessage(role='assistant')  # type: ignore
                if texts:
                    # Note: model responses from this model should only have one text item, so the following
                    # shouldn't merge multiple texts into one unless you switch models between runs:
                    message_param['content'] = '\n\n'.join(texts)
                if tool_calls:
                    message_param['tool_calls'] = tool_calls
                hf_messages.append(message_param)
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages, model_request_parameters):
            hf_messages.insert(0, ChatCompletionInputMessage(content=instructions, role='system'))  # type: ignore
        return hf_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ChatCompletionInputToolCall:
        return ChatCompletionInputToolCall.parse_obj_as_instance(  # type: ignore
            {
                'id': _guard_tool_call_id(t=t),
                'type': 'function',
                'function': {
                    'name': t.tool_name,
                    'arguments': t.args_as_json_str(),
                },
            }
        )

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ChatCompletionInputTool:
        tool_param: ChatCompletionInputTool = ChatCompletionInputTool.parse_obj_as_instance(  # type: ignore
            {
                'type': 'function',
                'function': {
                    'name': f.name,
                    'description': f.description,
                    'parameters': f.parameters_json_schema,
                },
            }
        )
        return tool_param

    async def _map_user_message(
        self, message: ModelRequest
    ) -> AsyncIterable[ChatCompletionInputMessage | ChatCompletionOutputMessage]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield ChatCompletionInputMessage.parse_obj_as_instance({'role': 'system', 'content': part.content})  # type: ignore
            elif isinstance(part, UserPromptPart):
                yield await self._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield ChatCompletionOutputMessage.parse_obj_as_instance(  # type: ignore
                    {
                        'role': 'tool',
                        'tool_call_id': _guard_tool_call_id(t=part),
                        'content': part.model_response_str(),
                    }
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield ChatCompletionInputMessage.parse_obj_as_instance(  # type: ignore
                        {'role': 'user', 'content': part.model_response()}
                    )
                else:
                    yield ChatCompletionInputMessage.parse_obj_as_instance(  # type: ignore
                        {
                            'role': 'tool',
                            'tool_call_id': _guard_tool_call_id(t=part),
                            'content': part.model_response(),
                        }
                    )
            else:
                assert_never(part)

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> ChatCompletionInputMessage:
        content: str | list[ChatCompletionInputMessage]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(ChatCompletionInputMessageChunk(type='text', text=item))  # type: ignore
                elif isinstance(item, ImageUrl):
                    url = ChatCompletionInputURL(url=item.url)  # type: ignore
                    content.append(ChatCompletionInputMessageChunk(type='image_url', image_url=url))  # type: ignore
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        url = ChatCompletionInputURL(url=item.data_uri)  # type: ignore
                        content.append(ChatCompletionInputMessageChunk(type='image_url', image_url=url))  # type: ignore
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):
                    raise NotImplementedError('AudioUrl is not supported for Hugging Face')
                elif isinstance(item, DocumentUrl):
                    raise NotImplementedError('DocumentUrl is not supported for Hugging Face')
                elif isinstance(item, VideoUrl):
                    raise NotImplementedError('VideoUrl is not supported for Hugging Face')
                elif isinstance(item, CachePoint):
                    # Hugging Face doesn't support prompt caching via CachePoint
                    pass
                else:
                    assert_never(item)
        return ChatCompletionInputMessage(role='user', content=content)  # type: ignore

```

#### __init__

```python
__init__(
    model_name: str,
    *,
    provider: (
        Literal["huggingface"]
        | Provider[AsyncInferenceClient]
    ) = "huggingface",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize a Hugging Face model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model_name` | `str` | The name of the Model to use. You can browse available models here. | *required* | | `provider` | `Literal['huggingface'] | Provider[AsyncInferenceClient]` | The provider to use for Hugging Face Inference Providers. Can be either the string 'huggingface' or an instance of Provider[AsyncInferenceClient]. If not provided, the other parameters will be used. | `'huggingface'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider based on the model name. | `None` | | `settings` | `ModelSettings | None` | Model-specific settings that will be used as defaults for this model. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/huggingface.py`

```python
def __init__(
    self,
    model_name: str,
    *,
    provider: Literal['huggingface'] | Provider[AsyncInferenceClient] = 'huggingface',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize a Hugging Face model.

    Args:
        model_name: The name of the Model to use. You can browse available models [here](https://huggingface.co/models?pipeline_tag=text-generation&inference_provider=all&sort=trending).
        provider: The provider to use for Hugging Face Inference Providers. Can be either the string 'huggingface' or an
            instance of `Provider[AsyncInferenceClient]`. If not provided, the other parameters will be used.
        profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
        settings: Model-specific settings that will be used as defaults for this model.
    """
    self._model_name = model_name
    if isinstance(provider, str):
        provider = infer_provider(provider)
    self._provider = provider
    self.client = provider.client

    super().__init__(settings=settings, profile=profile or provider.model_profile)

```

#### model_name

```python
model_name: HuggingFaceModelName

```

The model name.

#### system

```python
system: str

```

The system / model provider.

# pydantic_ai.models.instrumented

### instrument_model

```python
instrument_model(
    model: Model, instrument: InstrumentationSettings | bool
) -> Model

```

Instrument a model with OpenTelemetry/logfire.

Source code in `pydantic_ai_slim/pydantic_ai/models/instrumented.py`

```python
def instrument_model(model: Model, instrument: InstrumentationSettings | bool) -> Model:
    """Instrument a model with OpenTelemetry/logfire."""
    if instrument and not isinstance(model, InstrumentedModel):
        if instrument is True:
            instrument = InstrumentationSettings()

        model = InstrumentedModel(model, instrument)

    return model

```

### InstrumentationSettings

Options for instrumenting models and agents with OpenTelemetry.

Used in:

- `Agent(instrument=...)`
- Agent.instrument_all()
- InstrumentedModel

See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.

Source code in `pydantic_ai_slim/pydantic_ai/models/instrumented.py`

```python
@dataclass(init=False)
class InstrumentationSettings:
    """Options for instrumenting models and agents with OpenTelemetry.

    Used in:

    - `Agent(instrument=...)`
    - [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all]
    - [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel]

    See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
    """

    tracer: Tracer = field(repr=False)
    event_logger: EventLogger = field(repr=False)
    event_mode: Literal['attributes', 'logs'] = 'attributes'
    include_binary_content: bool = True
    include_content: bool = True
    version: Literal[1, 2, 3] = DEFAULT_INSTRUMENTATION_VERSION

    def __init__(
        self,
        *,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        include_binary_content: bool = True,
        include_content: bool = True,
        version: Literal[1, 2, 3] = DEFAULT_INSTRUMENTATION_VERSION,
        event_mode: Literal['attributes', 'logs'] = 'attributes',
        event_logger_provider: EventLoggerProvider | None = None,
    ):
        """Create instrumentation options.

        Args:
            tracer_provider: The OpenTelemetry tracer provider to use.
                If not provided, the global tracer provider is used.
                Calling `logfire.configure()` sets the global tracer provider, so most users don't need this.
            meter_provider: The OpenTelemetry meter provider to use.
                If not provided, the global meter provider is used.
                Calling `logfire.configure()` sets the global meter provider, so most users don't need this.
            include_binary_content: Whether to include binary content in the instrumentation events.
            include_content: Whether to include prompts, completions, and tool call arguments and responses
                in the instrumentation events.
            version: Version of the data format. This is unrelated to the Pydantic AI package version.
                Version 1 is based on the legacy event-based OpenTelemetry GenAI spec
                    and will be removed in a future release.
                    The parameters `event_mode` and `event_logger_provider` are only relevant for version 1.
                Version 2 uses the newer OpenTelemetry GenAI spec and stores messages in the following attributes:
                    - `gen_ai.system_instructions` for instructions passed to the agent.
                    - `gen_ai.input.messages` and `gen_ai.output.messages` on model request spans.
                    - `pydantic_ai.all_messages` on agent run spans.
            event_mode: The mode for emitting events in version 1.
                If `'attributes'`, events are attached to the span as attributes.
                If `'logs'`, events are emitted as OpenTelemetry log-based events.
            event_logger_provider: The OpenTelemetry event logger provider to use.
                If not provided, the global event logger provider is used.
                Calling `logfire.configure()` sets the global event logger provider, so most users don't need this.
                This is only used if `event_mode='logs'` and `version=1`.
        """
        from pydantic_ai import __version__

        tracer_provider = tracer_provider or get_tracer_provider()
        meter_provider = meter_provider or get_meter_provider()
        event_logger_provider = event_logger_provider or get_event_logger_provider()
        scope_name = 'pydantic-ai'
        self.tracer = tracer_provider.get_tracer(scope_name, __version__)
        self.meter = meter_provider.get_meter(scope_name, __version__)
        self.event_logger = event_logger_provider.get_event_logger(scope_name, __version__)
        self.event_mode = event_mode
        self.include_binary_content = include_binary_content
        self.include_content = include_content

        if event_mode == 'logs' and version != 1:
            warnings.warn(
                'event_mode is only relevant for version=1 which is deprecated and will be removed in a future release.',
                stacklevel=2,
            )
            version = 1

        self.version = version

        # As specified in the OpenTelemetry GenAI metrics spec:
        # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/#metric-gen_aiclienttokenusage
        tokens_histogram_kwargs = dict(
            name='gen_ai.client.token.usage',
            unit='{token}',
            description='Measures number of input and output tokens used',
        )
        try:
            self.tokens_histogram = self.meter.create_histogram(
                **tokens_histogram_kwargs,
                explicit_bucket_boundaries_advisory=TOKEN_HISTOGRAM_BOUNDARIES,
            )
        except TypeError:  # pragma: lax no cover
            # Older OTel/logfire versions don't support explicit_bucket_boundaries_advisory
            self.tokens_histogram = self.meter.create_histogram(
                **tokens_histogram_kwargs,  # pyright: ignore
            )
        self.cost_histogram = self.meter.create_histogram(
            'operation.cost',
            unit='{USD}',
            description='Monetary cost',
        )

    def messages_to_otel_events(
        self, messages: list[ModelMessage], parameters: ModelRequestParameters | None = None
    ) -> list[Event]:
        """Convert a list of model messages to OpenTelemetry events.

        Args:
            messages: The messages to convert.
            parameters: The model request parameters.

        Returns:
            A list of OpenTelemetry events.
        """
        events: list[Event] = []
        instructions = InstrumentedModel._get_instructions(messages, parameters)  # pyright: ignore [reportPrivateUsage]
        if instructions is not None:
            events.append(
                Event(
                    'gen_ai.system.message',
                    body={**({'content': instructions} if self.include_content else {}), 'role': 'system'},
                )
            )

        for message_index, message in enumerate(messages):
            message_events: list[Event] = []
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if hasattr(part, 'otel_event'):
                        message_events.append(part.otel_event(self))
            elif isinstance(message, ModelResponse):  # pragma: no branch
                message_events = message.otel_events(self)
            for event in message_events:
                event.attributes = {
                    'gen_ai.message.index': message_index,
                    **(event.attributes or {}),
                }
            events.extend(message_events)

        for event in events:
            event.body = InstrumentedModel.serialize_any(event.body)
        return events

    def messages_to_otel_messages(self, messages: list[ModelMessage]) -> list[_otel_messages.ChatMessage]:
        result: list[_otel_messages.ChatMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for is_system, group in itertools.groupby(message.parts, key=lambda p: isinstance(p, SystemPromptPart)):
                    message_parts: list[_otel_messages.MessagePart] = []
                    for part in group:
                        if hasattr(part, 'otel_message_parts'):
                            message_parts.extend(part.otel_message_parts(self))
                    result.append(
                        _otel_messages.ChatMessage(role='system' if is_system else 'user', parts=message_parts)
                    )
            elif isinstance(message, ModelResponse):  # pragma: no branch
                otel_message = _otel_messages.OutputMessage(role='assistant', parts=message.otel_message_parts(self))
                if message.finish_reason is not None:
                    otel_message['finish_reason'] = message.finish_reason
                result.append(otel_message)
        return result

    def handle_messages(
        self,
        input_messages: list[ModelMessage],
        response: ModelResponse,
        system: str,
        span: Span,
        parameters: ModelRequestParameters | None = None,
    ):
        if self.version == 1:
            events = self.messages_to_otel_events(input_messages, parameters)
            for event in self.messages_to_otel_events([response], parameters):
                events.append(
                    Event(
                        'gen_ai.choice',
                        body={
                            'index': 0,
                            'message': event.body,
                        },
                    )
                )
            for event in events:
                event.attributes = {
                    GEN_AI_SYSTEM_ATTRIBUTE: system,
                    **(event.attributes or {}),
                }
            self._emit_events(span, events)
        else:
            output_messages = self.messages_to_otel_messages([response])
            assert len(output_messages) == 1
            output_message = output_messages[0]
            instructions = InstrumentedModel._get_instructions(input_messages, parameters)  # pyright: ignore [reportPrivateUsage]
            system_instructions_attributes = self.system_instructions_attributes(instructions)
            attributes: dict[str, AttributeValue] = {
                'gen_ai.input.messages': json.dumps(self.messages_to_otel_messages(input_messages)),
                'gen_ai.output.messages': json.dumps([output_message]),
                **system_instructions_attributes,
                'logfire.json_schema': json.dumps(
                    {
                        'type': 'object',
                        'properties': {
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            **(
                                {'gen_ai.system_instructions': {'type': 'array'}}
                                if system_instructions_attributes
                                else {}
                            ),
                            'model_request_parameters': {'type': 'object'},
                        },
                    }
                ),
            }
            span.set_attributes(attributes)

    def system_instructions_attributes(self, instructions: str | None) -> dict[str, str]:
        if instructions and self.include_content:
            return {
                'gen_ai.system_instructions': json.dumps([_otel_messages.TextPart(type='text', content=instructions)]),
            }
        return {}

    def _emit_events(self, span: Span, events: list[Event]) -> None:
        if self.event_mode == 'logs':
            for event in events:
                self.event_logger.emit(event)
        else:
            attr_name = 'events'
            span.set_attributes(
                {
                    attr_name: json.dumps([InstrumentedModel.event_to_dict(event) for event in events]),
                    'logfire.json_schema': json.dumps(
                        {
                            'type': 'object',
                            'properties': {
                                attr_name: {'type': 'array'},
                                'model_request_parameters': {'type': 'object'},
                            },
                        }
                    ),
                }
            )

    def record_metrics(
        self,
        response: ModelResponse,
        price_calculation: PriceCalculation | None,
        attributes: dict[str, AttributeValue],
    ):
        for typ in ['input', 'output']:
            if not (tokens := getattr(response.usage, f'{typ}_tokens', 0)):  # pragma: no cover
                continue
            token_attributes = {**attributes, 'gen_ai.token.type': typ}
            self.tokens_histogram.record(tokens, token_attributes)
            if price_calculation:
                cost = float(getattr(price_calculation, f'{typ}_price'))
                self.cost_histogram.record(cost, token_attributes)

```

#### __init__

```python
__init__(
    *,
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
    include_binary_content: bool = True,
    include_content: bool = True,
    version: Literal[
        1, 2, 3
    ] = DEFAULT_INSTRUMENTATION_VERSION,
    event_mode: Literal[
        "attributes", "logs"
    ] = "attributes",
    event_logger_provider: EventLoggerProvider | None = None
)

```

Create instrumentation options.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tracer_provider` | `TracerProvider | None` | The OpenTelemetry tracer provider to use. If not provided, the global tracer provider is used. Calling logfire.configure() sets the global tracer provider, so most users don't need this. | `None` | | `meter_provider` | `MeterProvider | None` | The OpenTelemetry meter provider to use. If not provided, the global meter provider is used. Calling logfire.configure() sets the global meter provider, so most users don't need this. | `None` | | `include_binary_content` | `bool` | Whether to include binary content in the instrumentation events. | `True` | | `include_content` | `bool` | Whether to include prompts, completions, and tool call arguments and responses in the instrumentation events. | `True` | | `version` | `Literal[1, 2, 3]` | Version of the data format. This is unrelated to the Pydantic AI package version. Version 1 is based on the legacy event-based OpenTelemetry GenAI spec and will be removed in a future release. The parameters event_mode and event_logger_provider are only relevant for version 1. Version 2 uses the newer OpenTelemetry GenAI spec and stores messages in the following attributes: - gen_ai.system_instructions for instructions passed to the agent. - gen_ai.input.messages and gen_ai.output.messages on model request spans. - pydantic_ai.all_messages on agent run spans. | `DEFAULT_INSTRUMENTATION_VERSION` | | `event_mode` | `Literal['attributes', 'logs']` | The mode for emitting events in version 1. If 'attributes', events are attached to the span as attributes. If 'logs', events are emitted as OpenTelemetry log-based events. | `'attributes'` | | `event_logger_provider` | `EventLoggerProvider | None` | The OpenTelemetry event logger provider to use. If not provided, the global event logger provider is used. Calling logfire.configure() sets the global event logger provider, so most users don't need this. This is only used if event_mode='logs' and version=1. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/instrumented.py`

```python
def __init__(
    self,
    *,
    tracer_provider: TracerProvider | None = None,
    meter_provider: MeterProvider | None = None,
    include_binary_content: bool = True,
    include_content: bool = True,
    version: Literal[1, 2, 3] = DEFAULT_INSTRUMENTATION_VERSION,
    event_mode: Literal['attributes', 'logs'] = 'attributes',
    event_logger_provider: EventLoggerProvider | None = None,
):
    """Create instrumentation options.

    Args:
        tracer_provider: The OpenTelemetry tracer provider to use.
            If not provided, the global tracer provider is used.
            Calling `logfire.configure()` sets the global tracer provider, so most users don't need this.
        meter_provider: The OpenTelemetry meter provider to use.
            If not provided, the global meter provider is used.
            Calling `logfire.configure()` sets the global meter provider, so most users don't need this.
        include_binary_content: Whether to include binary content in the instrumentation events.
        include_content: Whether to include prompts, completions, and tool call arguments and responses
            in the instrumentation events.
        version: Version of the data format. This is unrelated to the Pydantic AI package version.
            Version 1 is based on the legacy event-based OpenTelemetry GenAI spec
                and will be removed in a future release.
                The parameters `event_mode` and `event_logger_provider` are only relevant for version 1.
            Version 2 uses the newer OpenTelemetry GenAI spec and stores messages in the following attributes:
                - `gen_ai.system_instructions` for instructions passed to the agent.
                - `gen_ai.input.messages` and `gen_ai.output.messages` on model request spans.
                - `pydantic_ai.all_messages` on agent run spans.
        event_mode: The mode for emitting events in version 1.
            If `'attributes'`, events are attached to the span as attributes.
            If `'logs'`, events are emitted as OpenTelemetry log-based events.
        event_logger_provider: The OpenTelemetry event logger provider to use.
            If not provided, the global event logger provider is used.
            Calling `logfire.configure()` sets the global event logger provider, so most users don't need this.
            This is only used if `event_mode='logs'` and `version=1`.
    """
    from pydantic_ai import __version__

    tracer_provider = tracer_provider or get_tracer_provider()
    meter_provider = meter_provider or get_meter_provider()
    event_logger_provider = event_logger_provider or get_event_logger_provider()
    scope_name = 'pydantic-ai'
    self.tracer = tracer_provider.get_tracer(scope_name, __version__)
    self.meter = meter_provider.get_meter(scope_name, __version__)
    self.event_logger = event_logger_provider.get_event_logger(scope_name, __version__)
    self.event_mode = event_mode
    self.include_binary_content = include_binary_content
    self.include_content = include_content

    if event_mode == 'logs' and version != 1:
        warnings.warn(
            'event_mode is only relevant for version=1 which is deprecated and will be removed in a future release.',
            stacklevel=2,
        )
        version = 1

    self.version = version

    # As specified in the OpenTelemetry GenAI metrics spec:
    # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/#metric-gen_aiclienttokenusage
    tokens_histogram_kwargs = dict(
        name='gen_ai.client.token.usage',
        unit='{token}',
        description='Measures number of input and output tokens used',
    )
    try:
        self.tokens_histogram = self.meter.create_histogram(
            **tokens_histogram_kwargs,
            explicit_bucket_boundaries_advisory=TOKEN_HISTOGRAM_BOUNDARIES,
        )
    except TypeError:  # pragma: lax no cover
        # Older OTel/logfire versions don't support explicit_bucket_boundaries_advisory
        self.tokens_histogram = self.meter.create_histogram(
            **tokens_histogram_kwargs,  # pyright: ignore
        )
    self.cost_histogram = self.meter.create_histogram(
        'operation.cost',
        unit='{USD}',
        description='Monetary cost',
    )

```

#### messages_to_otel_events

```python
messages_to_otel_events(
    messages: list[ModelMessage],
    parameters: ModelRequestParameters | None = None,
) -> list[Event]

```

Convert a list of model messages to OpenTelemetry events.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `messages` | `list[ModelMessage]` | The messages to convert. | *required* | | `parameters` | `ModelRequestParameters | None` | The model request parameters. | `None` |

Returns:

| Type | Description | | --- | --- | | `list[Event]` | A list of OpenTelemetry events. |

Source code in `pydantic_ai_slim/pydantic_ai/models/instrumented.py`

```python
def messages_to_otel_events(
    self, messages: list[ModelMessage], parameters: ModelRequestParameters | None = None
) -> list[Event]:
    """Convert a list of model messages to OpenTelemetry events.

    Args:
        messages: The messages to convert.
        parameters: The model request parameters.

    Returns:
        A list of OpenTelemetry events.
    """
    events: list[Event] = []
    instructions = InstrumentedModel._get_instructions(messages, parameters)  # pyright: ignore [reportPrivateUsage]
    if instructions is not None:
        events.append(
            Event(
                'gen_ai.system.message',
                body={**({'content': instructions} if self.include_content else {}), 'role': 'system'},
            )
        )

    for message_index, message in enumerate(messages):
        message_events: list[Event] = []
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if hasattr(part, 'otel_event'):
                    message_events.append(part.otel_event(self))
        elif isinstance(message, ModelResponse):  # pragma: no branch
            message_events = message.otel_events(self)
        for event in message_events:
            event.attributes = {
                'gen_ai.message.index': message_index,
                **(event.attributes or {}),
            }
        events.extend(message_events)

    for event in events:
        event.body = InstrumentedModel.serialize_any(event.body)
    return events

```

### InstrumentedModel

Bases: `WrapperModel`

Model which wraps another model so that requests are instrumented with OpenTelemetry.

See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.

Source code in `pydantic_ai_slim/pydantic_ai/models/instrumented.py`

```python
@dataclass(init=False)
class InstrumentedModel(WrapperModel):
    """Model which wraps another model so that requests are instrumented with OpenTelemetry.

    See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
    """

    instrumentation_settings: InstrumentationSettings
    """Instrumentation settings for this model."""

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        options: InstrumentationSettings | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.instrumentation_settings = options or InstrumentationSettings()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        prepared_settings, prepared_parameters = self.wrapped.prepare_request(
            model_settings,
            model_request_parameters,
        )
        with self._instrument(messages, prepared_settings, prepared_parameters) as finish:
            response = await self.wrapped.request(messages, model_settings, model_request_parameters)
            finish(response, prepared_parameters)
            return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        prepared_settings, prepared_parameters = self.wrapped.prepare_request(
            model_settings,
            model_request_parameters,
        )
        with self._instrument(messages, prepared_settings, prepared_parameters) as finish:
            response_stream: StreamedResponse | None = None
            try:
                async with self.wrapped.request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as response_stream:
                    yield response_stream
            finally:
                if response_stream:  # pragma: no branch
                    finish(response_stream.get(), prepared_parameters)

    @contextmanager
    def _instrument(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Iterator[Callable[[ModelResponse, ModelRequestParameters], None]]:
        operation = 'chat'
        span_name = f'{operation} {self.model_name}'
        # TODO Missing attributes:
        #  - error.type: unclear if we should do something here or just always rely on span exceptions
        #  - gen_ai.request.stop_sequences/top_k: model_settings doesn't include these
        attributes: dict[str, AttributeValue] = {
            'gen_ai.operation.name': operation,
            **self.model_attributes(self.wrapped),
            **self.model_request_parameters_attributes(model_request_parameters),
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {'model_request_parameters': {'type': 'object'}},
                }
            ),
        }

        if model_settings:
            for key in MODEL_SETTING_ATTRIBUTES:
                if isinstance(value := model_settings.get(key), float | int):
                    attributes[f'gen_ai.request.{key}'] = value

        record_metrics: Callable[[], None] | None = None
        try:
            with self.instrumentation_settings.tracer.start_as_current_span(span_name, attributes=attributes) as span:

                def finish(response: ModelResponse, parameters: ModelRequestParameters):
                    # FallbackModel updates these span attributes.
                    attributes.update(getattr(span, 'attributes', {}))
                    request_model = attributes[GEN_AI_REQUEST_MODEL_ATTRIBUTE]
                    system = cast(str, attributes[GEN_AI_SYSTEM_ATTRIBUTE])

                    response_model = response.model_name or request_model
                    price_calculation = None

                    def _record_metrics():
                        metric_attributes = {
                            GEN_AI_SYSTEM_ATTRIBUTE: system,
                            'gen_ai.operation.name': operation,
                            'gen_ai.request.model': request_model,
                            'gen_ai.response.model': response_model,
                        }
                        self.instrumentation_settings.record_metrics(response, price_calculation, metric_attributes)

                    nonlocal record_metrics
                    record_metrics = _record_metrics

                    if not span.is_recording():
                        return

                    self.instrumentation_settings.handle_messages(messages, response, system, span, parameters)

                    attributes_to_set = {
                        **response.usage.opentelemetry_attributes(),
                        'gen_ai.response.model': response_model,
                    }
                    try:
                        price_calculation = response.cost()
                    except LookupError:
                        # The cost of this provider/model is unknown, which is common.
                        pass
                    except Exception as e:
                        warnings.warn(
                            f'Failed to get cost from response: {type(e).__name__}: {e}', CostCalculationFailedWarning
                        )
                    else:
                        attributes_to_set['operation.cost'] = float(price_calculation.total_price)

                    if response.provider_response_id is not None:
                        attributes_to_set['gen_ai.response.id'] = response.provider_response_id
                    if response.finish_reason is not None:
                        attributes_to_set['gen_ai.response.finish_reasons'] = [response.finish_reason]
                    span.set_attributes(attributes_to_set)
                    span.update_name(f'{operation} {request_model}')

                yield finish
        finally:
            if record_metrics:
                # We only want to record metrics after the span is finished,
                # to prevent them from being redundantly recorded in the span itself by logfire.
                record_metrics()

    @staticmethod
    def model_attributes(model: Model) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {
            GEN_AI_SYSTEM_ATTRIBUTE: model.system,
            GEN_AI_REQUEST_MODEL_ATTRIBUTE: model.model_name,
        }
        if base_url := model.base_url:
            try:
                parsed = urlparse(base_url)
            except Exception:  # pragma: no cover
                pass
            else:
                if parsed.hostname:  # pragma: no branch
                    attributes['server.address'] = parsed.hostname
                if parsed.port:  # pragma: no branch
                    attributes['server.port'] = parsed.port

        return attributes

    @staticmethod
    def model_request_parameters_attributes(
        model_request_parameters: ModelRequestParameters,
    ) -> dict[str, AttributeValue]:
        return {'model_request_parameters': json.dumps(InstrumentedModel.serialize_any(model_request_parameters))}

    @staticmethod
    def event_to_dict(event: Event) -> dict[str, Any]:
        if not event.body:
            body = {}  # pragma: no cover
        elif isinstance(event.body, Mapping):
            body = event.body  # type: ignore
        else:
            body = {'body': event.body}
        return {**body, **(event.attributes or {})}

    @staticmethod
    def serialize_any(value: Any) -> str:
        try:
            return ANY_ADAPTER.dump_python(value, mode='json')
        except Exception:
            try:
                return str(value)
            except Exception as e:
                return f'Unable to serialize: {e}'

```

#### instrumentation_settings

```python
instrumentation_settings: InstrumentationSettings = (
    options or InstrumentationSettings()
)

```

Instrumentation settings for this model.

