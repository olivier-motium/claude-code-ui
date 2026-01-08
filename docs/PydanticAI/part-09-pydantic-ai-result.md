<!-- Source: _complete-doc.md -->

# `pydantic_ai.result`

### StreamedRunResult

Bases: `Generic[AgentDepsT, OutputDataT]`

Result of a streamed run that returns structured data via a tool call.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
@dataclass(init=False)
class StreamedRunResult(Generic[AgentDepsT, OutputDataT]):
    """Result of a streamed run that returns structured data via a tool call."""

    _all_messages: list[_messages.ModelMessage]
    _new_message_index: int

    _stream_response: AgentStream[AgentDepsT, OutputDataT] | None = None
    _on_complete: Callable[[], Awaitable[None]] | None = None

    _run_result: AgentRunResult[OutputDataT] | None = None

    is_complete: bool = field(default=False, init=False)
    """Whether the stream has all been received.

    This is set to `True` when one of
    [`stream_output`][pydantic_ai.result.StreamedRunResult.stream_output],
    [`stream_text`][pydantic_ai.result.StreamedRunResult.stream_text],
    [`stream_responses`][pydantic_ai.result.StreamedRunResult.stream_responses] or
    [`get_output`][pydantic_ai.result.StreamedRunResult.get_output] completes.
    """

    @overload
    def __init__(
        self,
        all_messages: list[_messages.ModelMessage],
        new_message_index: int,
        stream_response: AgentStream[AgentDepsT, OutputDataT] | None,
        on_complete: Callable[[], Awaitable[None]] | None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        all_messages: list[_messages.ModelMessage],
        new_message_index: int,
        *,
        run_result: AgentRunResult[OutputDataT],
    ) -> None: ...

    def __init__(
        self,
        all_messages: list[_messages.ModelMessage],
        new_message_index: int,
        stream_response: AgentStream[AgentDepsT, OutputDataT] | None = None,
        on_complete: Callable[[], Awaitable[None]] | None = None,
        run_result: AgentRunResult[OutputDataT] | None = None,
    ) -> None:
        self._all_messages = all_messages
        self._new_message_index = new_message_index

        self._stream_response = stream_response
        self._on_complete = on_complete
        self._run_result = run_result

    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        # this is a method to be consistent with the other methods
        if output_tool_return_content is not None:
            raise NotImplementedError('Setting output tool return content is not supported for this result type.')
        return self._all_messages

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResult.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(output_tool_return_content=output_tool_return_content)
        )

    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self.all_messages(output_tool_return_content=output_tool_return_content)[self._new_message_index :]

    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResult.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.new_messages(output_tool_return_content=output_tool_return_content)
        )

    @deprecated('`StreamedRunResult.stream` is deprecated, use `stream_output` instead.')
    async def stream(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        async for output in self.stream_output(debounce_by=debounce_by):
            yield output

    async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        """Stream the output as an async iterable.

        The pydantic validator for structured data will be called in
        [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
        on each iteration.

        Args:
            debounce_by: by how much (if at all) to debounce/group the output chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured outputs to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An async iterable of the response data.
        """
        if self._run_result is not None:
            yield self._run_result.output
            await self._marked_completed()
        elif self._stream_response is not None:
            async for output in self._stream_response.stream_output(debounce_by=debounce_by):
                yield output
            await self._marked_completed(self.response)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    async def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]:
        """Stream the text result as an async iterable.

        !!! note
            Result validators will NOT be called on the text result if `delta=True`.

        Args:
            delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
                up to the current point.
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.
        """
        if self._run_result is not None:  # pragma: no cover
            # We can't really get here, as `_run_result` is only set in `run_stream` when `CallToolsNode` produces `DeferredToolRequests` output
            # as a result of a tool function raising `CallDeferred` or `ApprovalRequired`.
            # That'll change if we ever support something like `raise EndRun(output: OutputT)` where `OutputT` could be `str`.
            if not isinstance(self._run_result.output, str):
                raise exceptions.UserError('stream_text() can only be used with text responses')
            yield self._run_result.output
            await self._marked_completed()
        elif self._stream_response is not None:
            async for text in self._stream_response.stream_text(delta=delta, debounce_by=debounce_by):
                yield text
            await self._marked_completed(self.response)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @deprecated('`StreamedRunResult.stream_structured` is deprecated, use `stream_responses` instead.')
    async def stream_structured(
        self, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
        async for msg, last in self.stream_responses(debounce_by=debounce_by):
            yield msg, last

    async def stream_responses(
        self, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
        """Stream the response as an async iterable of Structured LLM Messages.

        Args:
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An async iterable of the structured response message and whether that is the last message.
        """
        if self._run_result is not None:
            yield self.response, True
            await self._marked_completed()
        elif self._stream_response is not None:
            # if the message currently has any parts with content, yield before streaming
            async for msg in self._stream_response.stream_responses(debounce_by=debounce_by):
                yield msg, False

            msg = self.response
            yield msg, True

            await self._marked_completed(msg)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    async def get_output(self) -> OutputDataT:
        """Stream the whole response, validate and return it."""
        if self._run_result is not None:
            output = self._run_result.output
            await self._marked_completed()
            return output
        elif self._stream_response is not None:
            output = await self._stream_response.get_output()
            await self._marked_completed(self.response)
            return output
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @property
    def response(self) -> _messages.ModelResponse:
        """Return the current state of the response."""
        if self._run_result is not None:
            return self._run_result.response
        elif self._stream_response is not None:
            return self._stream_response.get()
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    # TODO (v2): Make this a property
    def usage(self) -> RunUsage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        if self._run_result is not None:
            return self._run_result.usage()
        elif self._stream_response is not None:
            return self._stream_response.usage()
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    # TODO (v2): Make this a property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        if self._run_result is not None:
            return self._run_result.timestamp()
        elif self._stream_response is not None:
            return self._stream_response.timestamp()
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        if self._run_result is not None:
            return self._run_result.run_id
        elif self._stream_response is not None:
            return self._stream_response.run_id
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @deprecated('`validate_structured_output` is deprecated, use `validate_response_output` instead.')
    async def validate_structured_output(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        return await self.validate_response_output(message, allow_partial=allow_partial)

    async def validate_response_output(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        """Validate a structured result message."""
        if self._run_result is not None:
            return self._run_result.output
        elif self._stream_response is not None:
            return await self._stream_response.validate_response_output(message, allow_partial=allow_partial)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    async def _marked_completed(self, message: _messages.ModelResponse | None = None) -> None:
        self.is_complete = True
        if message is not None:
            if self._stream_response:  # pragma: no branch
                message.run_id = self._stream_response.run_id
            self._all_messages.append(message)
        if self._on_complete is not None:
            await self._on_complete()

```

#### is_complete

```python
is_complete: bool = field(default=False, init=False)

```

Whether the stream has all been received.

This is set to `True` when one of stream_output, stream_text, stream_responses or get_output completes.

#### all_messages

```python
all_messages(
    *, output_tool_return_content: str | None = None
) -> list[ModelMessage]

```

Return the history of \_messages.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `list[ModelMessage]` | List of messages. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
    """Return the history of _messages.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        List of messages.
    """
    # this is a method to be consistent with the other methods
    if output_tool_return_content is not None:
        raise NotImplementedError('Setting output tool return content is not supported for this result type.')
    return self._all_messages

```

#### all_messages_json

```python
all_messages_json(
    *, output_tool_return_content: str | None = None
) -> bytes

```

Return all messages from all_messages as JSON bytes.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `bytes` | JSON bytes representing the messages. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
    """Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResult.all_messages] as JSON bytes.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        JSON bytes representing the messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(
        self.all_messages(output_tool_return_content=output_tool_return_content)
    )

```

#### new_messages

```python
new_messages(
    *, output_tool_return_content: str | None = None
) -> list[ModelMessage]

```

Return new messages associated with this run.

Messages from older runs are excluded.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `list[ModelMessage]` | List of new messages. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
    """Return new messages associated with this run.

    Messages from older runs are excluded.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        List of new messages.
    """
    return self.all_messages(output_tool_return_content=output_tool_return_content)[self._new_message_index :]

```

#### new_messages_json

```python
new_messages_json(
    *, output_tool_return_content: str | None = None
) -> bytes

```

Return new messages from new_messages as JSON bytes.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `bytes` | JSON bytes representing the new messages. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
    """Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResult.new_messages] as JSON bytes.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        JSON bytes representing the new messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(
        self.new_messages(output_tool_return_content=output_tool_return_content)
    )

```

#### stream

```python
stream(
    *, debounce_by: float | None = 0.1
) -> AsyncIterator[OutputDataT]

```

Deprecated

`StreamedRunResult.stream` is deprecated, use `stream_output` instead.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
@deprecated('`StreamedRunResult.stream` is deprecated, use `stream_output` instead.')
async def stream(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
    async for output in self.stream_output(debounce_by=debounce_by):
        yield output

```

#### stream_output

```python
stream_output(
    *, debounce_by: float | None = 0.1
) -> AsyncIterator[OutputDataT]

```

Stream the output as an async iterable.

The pydantic validator for structured data will be called in [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation) on each iteration.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `debounce_by` | `float | None` | by how much (if at all) to debounce/group the output chunks by. None means no debouncing. Debouncing is particularly important for long structured outputs to reduce the overhead of performing validation as each token is received. | `0.1` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[OutputDataT]` | An async iterable of the response data. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
    """Stream the output as an async iterable.

    The pydantic validator for structured data will be called in
    [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
    on each iteration.

    Args:
        debounce_by: by how much (if at all) to debounce/group the output chunks by. `None` means no debouncing.
            Debouncing is particularly important for long structured outputs to reduce the overhead of
            performing validation as each token is received.

    Returns:
        An async iterable of the response data.
    """
    if self._run_result is not None:
        yield self._run_result.output
        await self._marked_completed()
    elif self._stream_response is not None:
        async for output in self._stream_response.stream_output(debounce_by=debounce_by):
            yield output
        await self._marked_completed(self.response)
    else:
        raise ValueError('No stream response or run result provided')  # pragma: no cover

```

#### stream_text

```python
stream_text(
    *, delta: bool = False, debounce_by: float | None = 0.1
) -> AsyncIterator[str]

```

Stream the text result as an async iterable.

Note

Result validators will NOT be called on the text result if `delta=True`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `delta` | `bool` | if True, yield each chunk of text as it is received, if False (default), yield the full text up to the current point. | `False` | | `debounce_by` | `float | None` | by how much (if at all) to debounce/group the response chunks by. None means no debouncing. Debouncing is particularly important for long structured responses to reduce the overhead of performing validation as each token is received. | `0.1` |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
async def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]:
    """Stream the text result as an async iterable.

    !!! note
        Result validators will NOT be called on the text result if `delta=True`.

    Args:
        delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
            up to the current point.
        debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
            Debouncing is particularly important for long structured responses to reduce the overhead of
            performing validation as each token is received.
    """
    if self._run_result is not None:  # pragma: no cover
        # We can't really get here, as `_run_result` is only set in `run_stream` when `CallToolsNode` produces `DeferredToolRequests` output
        # as a result of a tool function raising `CallDeferred` or `ApprovalRequired`.
        # That'll change if we ever support something like `raise EndRun(output: OutputT)` where `OutputT` could be `str`.
        if not isinstance(self._run_result.output, str):
            raise exceptions.UserError('stream_text() can only be used with text responses')
        yield self._run_result.output
        await self._marked_completed()
    elif self._stream_response is not None:
        async for text in self._stream_response.stream_text(delta=delta, debounce_by=debounce_by):
            yield text
        await self._marked_completed(self.response)
    else:
        raise ValueError('No stream response or run result provided')  # pragma: no cover

```

#### stream_structured

```python
stream_structured(
    *, debounce_by: float | None = 0.1
) -> AsyncIterator[tuple[ModelResponse, bool]]

```

Deprecated

`StreamedRunResult.stream_structured` is deprecated, use `stream_responses` instead.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
@deprecated('`StreamedRunResult.stream_structured` is deprecated, use `stream_responses` instead.')
async def stream_structured(
    self, *, debounce_by: float | None = 0.1
) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
    async for msg, last in self.stream_responses(debounce_by=debounce_by):
        yield msg, last

```

#### stream_responses

```python
stream_responses(
    *, debounce_by: float | None = 0.1
) -> AsyncIterator[tuple[ModelResponse, bool]]

```

Stream the response as an async iterable of Structured LLM Messages.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `debounce_by` | `float | None` | by how much (if at all) to debounce/group the response chunks by. None means no debouncing. Debouncing is particularly important for long structured responses to reduce the overhead of performing validation as each token is received. | `0.1` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[tuple[ModelResponse, bool]]` | An async iterable of the structured response message and whether that is the last message. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
async def stream_responses(
    self, *, debounce_by: float | None = 0.1
) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
    """Stream the response as an async iterable of Structured LLM Messages.

    Args:
        debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
            Debouncing is particularly important for long structured responses to reduce the overhead of
            performing validation as each token is received.

    Returns:
        An async iterable of the structured response message and whether that is the last message.
    """
    if self._run_result is not None:
        yield self.response, True
        await self._marked_completed()
    elif self._stream_response is not None:
        # if the message currently has any parts with content, yield before streaming
        async for msg in self._stream_response.stream_responses(debounce_by=debounce_by):
            yield msg, False

        msg = self.response
        yield msg, True

        await self._marked_completed(msg)
    else:
        raise ValueError('No stream response or run result provided')  # pragma: no cover

```

#### get_output

```python
get_output() -> OutputDataT

```

Stream the whole response, validate and return it.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
async def get_output(self) -> OutputDataT:
    """Stream the whole response, validate and return it."""
    if self._run_result is not None:
        output = self._run_result.output
        await self._marked_completed()
        return output
    elif self._stream_response is not None:
        output = await self._stream_response.get_output()
        await self._marked_completed(self.response)
        return output
    else:
        raise ValueError('No stream response or run result provided')  # pragma: no cover

```

#### response

```python
response: ModelResponse

```

Return the current state of the response.

#### usage

```python
usage() -> RunUsage

```

Return the usage of the whole run.

Note

This won't return the full usage until the stream is finished.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def usage(self) -> RunUsage:
    """Return the usage of the whole run.

    !!! note
        This won't return the full usage until the stream is finished.
    """
    if self._run_result is not None:
        return self._run_result.usage()
    elif self._stream_response is not None:
        return self._stream_response.usage()
    else:
        raise ValueError('No stream response or run result provided')  # pragma: no cover

```

#### timestamp

```python
timestamp() -> datetime

```

Get the timestamp of the response.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def timestamp(self) -> datetime:
    """Get the timestamp of the response."""
    if self._run_result is not None:
        return self._run_result.timestamp()
    elif self._stream_response is not None:
        return self._stream_response.timestamp()
    else:
        raise ValueError('No stream response or run result provided')  # pragma: no cover

```

#### run_id

```python
run_id: str

```

The unique identifier for the agent run.

#### validate_structured_output

```python
validate_structured_output(
    message: ModelResponse, *, allow_partial: bool = False
) -> OutputDataT

```

Deprecated

`validate_structured_output` is deprecated, use `validate_response_output` instead.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
@deprecated('`validate_structured_output` is deprecated, use `validate_response_output` instead.')
async def validate_structured_output(
    self, message: _messages.ModelResponse, *, allow_partial: bool = False
) -> OutputDataT:
    return await self.validate_response_output(message, allow_partial=allow_partial)

```

#### validate_response_output

```python
validate_response_output(
    message: ModelResponse, *, allow_partial: bool = False
) -> OutputDataT

```

Validate a structured result message.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
async def validate_response_output(
    self, message: _messages.ModelResponse, *, allow_partial: bool = False
) -> OutputDataT:
    """Validate a structured result message."""
    if self._run_result is not None:
        return self._run_result.output
    elif self._stream_response is not None:
        return await self._stream_response.validate_response_output(message, allow_partial=allow_partial)
    else:
        raise ValueError('No stream response or run result provided')  # pragma: no cover

```

### StreamedRunResultSync

Bases: `Generic[AgentDepsT, OutputDataT]`

Synchronous wrapper for StreamedRunResult that only exposes sync methods.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
@dataclass(init=False)
class StreamedRunResultSync(Generic[AgentDepsT, OutputDataT]):
    """Synchronous wrapper for [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult] that only exposes sync methods."""

    _streamed_run_result: StreamedRunResult[AgentDepsT, OutputDataT]

    def __init__(self, streamed_run_result: StreamedRunResult[AgentDepsT, OutputDataT]) -> None:
        self._streamed_run_result = streamed_run_result

    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        return self._streamed_run_result.all_messages(output_tool_return_content=output_tool_return_content)

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResultSync.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return self._streamed_run_result.all_messages_json(output_tool_return_content=output_tool_return_content)

    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self._streamed_run_result.new_messages(output_tool_return_content=output_tool_return_content)

    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResultSync.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return self._streamed_run_result.new_messages_json(output_tool_return_content=output_tool_return_content)

    def stream_output(self, *, debounce_by: float | None = 0.1) -> Iterator[OutputDataT]:
        """Stream the output as an iterable.

        The pydantic validator for structured data will be called in
        [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
        on each iteration.

        Args:
            debounce_by: by how much (if at all) to debounce/group the output chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured outputs to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An iterable of the response data.
        """
        return _utils.sync_async_iterator(self._streamed_run_result.stream_output(debounce_by=debounce_by))

    def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> Iterator[str]:
        """Stream the text result as an iterable.

        !!! note
            Result validators will NOT be called on the text result if `delta=True`.

        Args:
            delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
                up to the current point.
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.
        """
        return _utils.sync_async_iterator(self._streamed_run_result.stream_text(delta=delta, debounce_by=debounce_by))

    def stream_responses(self, *, debounce_by: float | None = 0.1) -> Iterator[tuple[_messages.ModelResponse, bool]]:
        """Stream the response as an iterable of Structured LLM Messages.

        Args:
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An iterable of the structured response message and whether that is the last message.
        """
        return _utils.sync_async_iterator(self._streamed_run_result.stream_responses(debounce_by=debounce_by))

    def get_output(self) -> OutputDataT:
        """Stream the whole response, validate and return it."""
        return _utils.get_event_loop().run_until_complete(self._streamed_run_result.get_output())

    @property
    def response(self) -> _messages.ModelResponse:
        """Return the current state of the response."""
        return self._streamed_run_result.response

    def usage(self) -> RunUsage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        return self._streamed_run_result.usage()

    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._streamed_run_result.timestamp()

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        return self._streamed_run_result.run_id

    def validate_response_output(self, message: _messages.ModelResponse, *, allow_partial: bool = False) -> OutputDataT:
        """Validate a structured result message."""
        return _utils.get_event_loop().run_until_complete(
            self._streamed_run_result.validate_response_output(message, allow_partial=allow_partial)
        )

    @property
    def is_complete(self) -> bool:
        """Whether the stream has all been received.

        This is set to `True` when one of
        [`stream_output`][pydantic_ai.result.StreamedRunResultSync.stream_output],
        [`stream_text`][pydantic_ai.result.StreamedRunResultSync.stream_text],
        [`stream_responses`][pydantic_ai.result.StreamedRunResultSync.stream_responses] or
        [`get_output`][pydantic_ai.result.StreamedRunResultSync.get_output] completes.
        """
        return self._streamed_run_result.is_complete

```

#### all_messages

```python
all_messages(
    *, output_tool_return_content: str | None = None
) -> list[ModelMessage]

```

Return the history of messages.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `list[ModelMessage]` | List of messages. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
    """Return the history of messages.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        List of messages.
    """
    return self._streamed_run_result.all_messages(output_tool_return_content=output_tool_return_content)

```

#### all_messages_json

```python
all_messages_json(
    *, output_tool_return_content: str | None = None
) -> bytes

```

Return all messages from all_messages as JSON bytes.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `bytes` | JSON bytes representing the messages. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
    """Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResultSync.all_messages] as JSON bytes.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        JSON bytes representing the messages.
    """
    return self._streamed_run_result.all_messages_json(output_tool_return_content=output_tool_return_content)

```

#### new_messages

```python
new_messages(
    *, output_tool_return_content: str | None = None
) -> list[ModelMessage]

```

Return new messages associated with this run.

Messages from older runs are excluded.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `list[ModelMessage]` | List of new messages. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
    """Return new messages associated with this run.

    Messages from older runs are excluded.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        List of new messages.
    """
    return self._streamed_run_result.new_messages(output_tool_return_content=output_tool_return_content)

```

#### new_messages_json

```python
new_messages_json(
    *, output_tool_return_content: str | None = None
) -> bytes

```

Return new messages from new_messages as JSON bytes.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `bytes` | JSON bytes representing the new messages. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
    """Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResultSync.new_messages] as JSON bytes.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        JSON bytes representing the new messages.
    """
    return self._streamed_run_result.new_messages_json(output_tool_return_content=output_tool_return_content)

```

#### stream_output

```python
stream_output(
    *, debounce_by: float | None = 0.1
) -> Iterator[OutputDataT]

```

Stream the output as an iterable.

The pydantic validator for structured data will be called in [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation) on each iteration.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `debounce_by` | `float | None` | by how much (if at all) to debounce/group the output chunks by. None means no debouncing. Debouncing is particularly important for long structured outputs to reduce the overhead of performing validation as each token is received. | `0.1` |

Returns:

| Type | Description | | --- | --- | | `Iterator[OutputDataT]` | An iterable of the response data. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def stream_output(self, *, debounce_by: float | None = 0.1) -> Iterator[OutputDataT]:
    """Stream the output as an iterable.

    The pydantic validator for structured data will be called in
    [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
    on each iteration.

    Args:
        debounce_by: by how much (if at all) to debounce/group the output chunks by. `None` means no debouncing.
            Debouncing is particularly important for long structured outputs to reduce the overhead of
            performing validation as each token is received.

    Returns:
        An iterable of the response data.
    """
    return _utils.sync_async_iterator(self._streamed_run_result.stream_output(debounce_by=debounce_by))

```

#### stream_text

```python
stream_text(
    *, delta: bool = False, debounce_by: float | None = 0.1
) -> Iterator[str]

```

Stream the text result as an iterable.

Note

Result validators will NOT be called on the text result if `delta=True`.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `delta` | `bool` | if True, yield each chunk of text as it is received, if False (default), yield the full text up to the current point. | `False` | | `debounce_by` | `float | None` | by how much (if at all) to debounce/group the response chunks by. None means no debouncing. Debouncing is particularly important for long structured responses to reduce the overhead of performing validation as each token is received. | `0.1` |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> Iterator[str]:
    """Stream the text result as an iterable.

    !!! note
        Result validators will NOT be called on the text result if `delta=True`.

    Args:
        delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
            up to the current point.
        debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
            Debouncing is particularly important for long structured responses to reduce the overhead of
            performing validation as each token is received.
    """
    return _utils.sync_async_iterator(self._streamed_run_result.stream_text(delta=delta, debounce_by=debounce_by))

```

#### stream_responses

```python
stream_responses(
    *, debounce_by: float | None = 0.1
) -> Iterator[tuple[ModelResponse, bool]]

```

Stream the response as an iterable of Structured LLM Messages.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `debounce_by` | `float | None` | by how much (if at all) to debounce/group the response chunks by. None means no debouncing. Debouncing is particularly important for long structured responses to reduce the overhead of performing validation as each token is received. | `0.1` |

Returns:

| Type | Description | | --- | --- | | `Iterator[tuple[ModelResponse, bool]]` | An iterable of the structured response message and whether that is the last message. |

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def stream_responses(self, *, debounce_by: float | None = 0.1) -> Iterator[tuple[_messages.ModelResponse, bool]]:
    """Stream the response as an iterable of Structured LLM Messages.

    Args:
        debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
            Debouncing is particularly important for long structured responses to reduce the overhead of
            performing validation as each token is received.

    Returns:
        An iterable of the structured response message and whether that is the last message.
    """
    return _utils.sync_async_iterator(self._streamed_run_result.stream_responses(debounce_by=debounce_by))

```

#### get_output

```python
get_output() -> OutputDataT

```

Stream the whole response, validate and return it.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def get_output(self) -> OutputDataT:
    """Stream the whole response, validate and return it."""
    return _utils.get_event_loop().run_until_complete(self._streamed_run_result.get_output())

```

#### response

```python
response: ModelResponse

```

Return the current state of the response.

#### usage

```python
usage() -> RunUsage

```

Return the usage of the whole run.

Note

This won't return the full usage until the stream is finished.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def usage(self) -> RunUsage:
    """Return the usage of the whole run.

    !!! note
        This won't return the full usage until the stream is finished.
    """
    return self._streamed_run_result.usage()

```

#### timestamp

```python
timestamp() -> datetime

```

Get the timestamp of the response.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def timestamp(self) -> datetime:
    """Get the timestamp of the response."""
    return self._streamed_run_result.timestamp()

```

#### run_id

```python
run_id: str

```

The unique identifier for the agent run.

#### validate_response_output

```python
validate_response_output(
    message: ModelResponse, *, allow_partial: bool = False
) -> OutputDataT

```

Validate a structured result message.

Source code in `pydantic_ai_slim/pydantic_ai/result.py`

```python
def validate_response_output(self, message: _messages.ModelResponse, *, allow_partial: bool = False) -> OutputDataT:
    """Validate a structured result message."""
    return _utils.get_event_loop().run_until_complete(
        self._streamed_run_result.validate_response_output(message, allow_partial=allow_partial)
    )

```

#### is_complete

```python
is_complete: bool

```

Whether the stream has all been received.

This is set to `True` when one of stream_output, stream_text, stream_responses or get_output completes.

# `pydantic_ai.retries`

Retries utilities based on tenacity, especially for HTTP requests.

This module provides HTTP transport wrappers and wait strategies that integrate with the tenacity library to add retry capabilities to HTTP requests. The transports can be used with HTTP clients that support custom transports (such as httpx), while the wait strategies can be used with any tenacity retry decorator.

The module includes:

- TenacityTransport: Synchronous HTTP transport with retry capabilities
- AsyncTenacityTransport: Asynchronous HTTP transport with retry capabilities
- wait_retry_after: Wait strategy that respects HTTP Retry-After headers

### RetryConfig

Bases: `TypedDict`

The configuration for tenacity-based retrying.

These are precisely the arguments to the tenacity `retry` decorator, and they are generally used internally by passing them to that decorator via `@retry(**config)` or similar.

All fields are optional, and if not provided, the default values from the `tenacity.retry` decorator will be used.

Source code in `pydantic_ai_slim/pydantic_ai/retries.py`

```python
class RetryConfig(TypedDict, total=False):
    """The configuration for tenacity-based retrying.

    These are precisely the arguments to the tenacity `retry` decorator, and they are generally
    used internally by passing them to that decorator via `@retry(**config)` or similar.

    All fields are optional, and if not provided, the default values from the `tenacity.retry` decorator will be used.
    """

    sleep: Callable[[int | float], None | Awaitable[None]]
    """A sleep strategy to use for sleeping between retries.

    Tenacity's default for this argument is `tenacity.nap.sleep`."""

    stop: StopBaseT
    """
    A stop strategy to determine when to stop retrying.

    Tenacity's default for this argument is `tenacity.stop.stop_never`."""

    wait: WaitBaseT
    """
    A wait strategy to determine how long to wait between retries.

    Tenacity's default for this argument is `tenacity.wait.wait_none`."""

    retry: SyncRetryBaseT | RetryBaseT
    """A retry strategy to determine which exceptions should trigger a retry.

    Tenacity's default for this argument is `tenacity.retry.retry_if_exception_type()`."""

    before: Callable[[RetryCallState], None | Awaitable[None]]
    """
    A callable that is called before each retry attempt.

    Tenacity's default for this argument is `tenacity.before.before_nothing`."""

    after: Callable[[RetryCallState], None | Awaitable[None]]
    """
    A callable that is called after each retry attempt.

    Tenacity's default for this argument is `tenacity.after.after_nothing`."""

    before_sleep: Callable[[RetryCallState], None | Awaitable[None]] | None
    """
    An optional callable that is called before sleeping between retries.

    Tenacity's default for this argument is `None`."""

    reraise: bool
    """Whether to reraise the last exception if the retry attempts are exhausted, or raise a RetryError instead.

    Tenacity's default for this argument is `False`."""

    retry_error_cls: type[RetryError]
    """The exception class to raise when the retry attempts are exhausted and `reraise` is False.

    Tenacity's default for this argument is `tenacity.RetryError`."""

    retry_error_callback: Callable[[RetryCallState], Any | Awaitable[Any]] | None
    """An optional callable that is called when the retry attempts are exhausted and `reraise` is False.

    Tenacity's default for this argument is `None`."""

```

#### sleep

```python
sleep: Callable[[int | float], None | Awaitable[None]]

```

A sleep strategy to use for sleeping between retries.

Tenacity's default for this argument is `tenacity.nap.sleep`.

#### stop

```python
stop: StopBaseT

```

A stop strategy to determine when to stop retrying.

Tenacity's default for this argument is `tenacity.stop.stop_never`.

#### wait

```python
wait: WaitBaseT

```

A wait strategy to determine how long to wait between retries.

Tenacity's default for this argument is `tenacity.wait.wait_none`.

#### retry

```python
retry: RetryBaseT | RetryBaseT

```

A retry strategy to determine which exceptions should trigger a retry.

Tenacity's default for this argument is `tenacity.retry.retry_if_exception_type()`.

#### before

```python
before: Callable[[RetryCallState], None | Awaitable[None]]

```

A callable that is called before each retry attempt.

Tenacity's default for this argument is `tenacity.before.before_nothing`.

#### after

```python
after: Callable[[RetryCallState], None | Awaitable[None]]

```

A callable that is called after each retry attempt.

Tenacity's default for this argument is `tenacity.after.after_nothing`.

#### before_sleep

```python
before_sleep: (
    Callable[[RetryCallState], None | Awaitable[None]]
    | None
)

```

An optional callable that is called before sleeping between retries.

Tenacity's default for this argument is `None`.

#### reraise

```python
reraise: bool

```

Whether to reraise the last exception if the retry attempts are exhausted, or raise a RetryError instead.

Tenacity's default for this argument is `False`.

#### retry_error_cls

```python
retry_error_cls: type[RetryError]

```

The exception class to raise when the retry attempts are exhausted and `reraise` is False.

Tenacity's default for this argument is `tenacity.RetryError`.

#### retry_error_callback

```python
retry_error_callback: (
    Callable[[RetryCallState], Any | Awaitable[Any]] | None
)

```

An optional callable that is called when the retry attempts are exhausted and `reraise` is False.

Tenacity's default for this argument is `None`.

### TenacityTransport

Bases: `BaseTransport`

Synchronous HTTP transport with tenacity-based retry functionality.

This transport wraps another BaseTransport and adds retry capabilities using the tenacity library. It can be configured to retry requests based on various conditions such as specific exception types, response status codes, or custom validation logic.

The transport works by intercepting HTTP requests and responses, allowing the tenacity controller to determine when and how to retry failed requests. The validate_response function can be used to convert HTTP responses into exceptions that trigger retries.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `wrapped` | `BaseTransport | None` | The underlying transport to wrap and add retry functionality to. | `None` | | `config` | `RetryConfig` | The arguments to use for the tenacity retry decorator, including retry conditions, wait strategy, stop conditions, etc. See the tenacity docs for more info. | *required* | | `validate_response` | `Callable[[Response], Any] | None` | Optional callable that takes a Response and can raise an exception to be handled by the controller if the response should trigger a retry. Common use case is to raise exceptions for certain HTTP status codes. If None, no response validation is performed. | `None` |

Example

```python
from httpx import Client, HTTPStatusError, HTTPTransport
from tenacity import retry_if_exception_type, stop_after_attempt

from pydantic_ai.retries import RetryConfig, TenacityTransport, wait_retry_after

transport = TenacityTransport(
    RetryConfig(
        retry=retry_if_exception_type(HTTPStatusError),
        wait=wait_retry_after(max_wait=300),
        stop=stop_after_attempt(5),
        reraise=True
    ),
    HTTPTransport(),
    validate_response=lambda r: r.raise_for_status()
)
client = Client(transport=transport)

```

Source code in `pydantic_ai_slim/pydantic_ai/retries.py`

````python
class TenacityTransport(BaseTransport):
    """Synchronous HTTP transport with tenacity-based retry functionality.

    This transport wraps another BaseTransport and adds retry capabilities using the tenacity library.
    It can be configured to retry requests based on various conditions such as specific exception types,
    response status codes, or custom validation logic.

    The transport works by intercepting HTTP requests and responses, allowing the tenacity controller
    to determine when and how to retry failed requests. The validate_response function can be used
    to convert HTTP responses into exceptions that trigger retries.

    Args:
        wrapped: The underlying transport to wrap and add retry functionality to.
        config: The arguments to use for the tenacity `retry` decorator, including retry conditions,
            wait strategy, stop conditions, etc. See the tenacity docs for more info.
        validate_response: Optional callable that takes a Response and can raise an exception
            to be handled by the controller if the response should trigger a retry.
            Common use case is to raise exceptions for certain HTTP status codes.
            If None, no response validation is performed.

    Example:
        ```python
        from httpx import Client, HTTPStatusError, HTTPTransport
        from tenacity import retry_if_exception_type, stop_after_attempt

        from pydantic_ai.retries import RetryConfig, TenacityTransport, wait_retry_after

        transport = TenacityTransport(
            RetryConfig(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=300),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            HTTPTransport(),
            validate_response=lambda r: r.raise_for_status()
        )
        client = Client(transport=transport)
        ```
    """

    def __init__(
        self,
        config: RetryConfig,
        wrapped: BaseTransport | None = None,
        validate_response: Callable[[Response], Any] | None = None,
    ):
        self.config = config
        self.wrapped = wrapped or HTTPTransport()
        self.validate_response = validate_response

    def handle_request(self, request: Request) -> Response:
        """Handle an HTTP request with retry logic.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.

        Raises:
            RuntimeError: If the retry controller did not make any attempts.
            Exception: Any exception raised by the wrapped transport or validation function.
        """

        @retry(**self.config)
        def handle_request(req: Request) -> Response:
            response = self.wrapped.handle_request(req)

            # this is normally set by httpx _after_ calling this function, but we want the request in the validator:
            response.request = req

            if self.validate_response:
                try:
                    self.validate_response(response)
                except Exception:
                    response.close()
                    raise
            return response

        return handle_request(request)

    def __enter__(self) -> TenacityTransport:
        self.wrapped.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        self.wrapped.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        self.wrapped.close()  # pragma: no cover

````

#### handle_request

```python
handle_request(request: Request) -> Response

```

Handle an HTTP request with retry logic.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `request` | `Request` | The HTTP request to handle. | *required* |

Returns:

| Type | Description | | --- | --- | | `Response` | The HTTP response. |

Raises:

| Type | Description | | --- | --- | | `RuntimeError` | If the retry controller did not make any attempts. | | `Exception` | Any exception raised by the wrapped transport or validation function. |

Source code in `pydantic_ai_slim/pydantic_ai/retries.py`

```python
def handle_request(self, request: Request) -> Response:
    """Handle an HTTP request with retry logic.

    Args:
        request: The HTTP request to handle.

    Returns:
        The HTTP response.

    Raises:
        RuntimeError: If the retry controller did not make any attempts.
        Exception: Any exception raised by the wrapped transport or validation function.
    """

    @retry(**self.config)
    def handle_request(req: Request) -> Response:
        response = self.wrapped.handle_request(req)

        # this is normally set by httpx _after_ calling this function, but we want the request in the validator:
        response.request = req

        if self.validate_response:
            try:
                self.validate_response(response)
            except Exception:
                response.close()
                raise
        return response

    return handle_request(request)

```

### AsyncTenacityTransport

Bases: `AsyncBaseTransport`

Asynchronous HTTP transport with tenacity-based retry functionality.

This transport wraps another AsyncBaseTransport and adds retry capabilities using the tenacity library. It can be configured to retry requests based on various conditions such as specific exception types, response status codes, or custom validation logic.

The transport works by intercepting HTTP requests and responses, allowing the tenacity controller to determine when and how to retry failed requests. The validate_response function can be used to convert HTTP responses into exceptions that trigger retries.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `wrapped` | `AsyncBaseTransport | None` | The underlying async transport to wrap and add retry functionality to. | `None` | | `config` | `RetryConfig` | The arguments to use for the tenacity retry decorator, including retry conditions, wait strategy, stop conditions, etc. See the tenacity docs for more info. | *required* | | `validate_response` | `Callable[[Response], Any] | None` | Optional callable that takes a Response and can raise an exception to be handled by the controller if the response should trigger a retry. Common use case is to raise exceptions for certain HTTP status codes. If None, no response validation is performed. | `None` |

Example

```python
from httpx import AsyncClient, HTTPStatusError
from tenacity import retry_if_exception_type, stop_after_attempt

from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

transport = AsyncTenacityTransport(
    RetryConfig(
        retry=retry_if_exception_type(HTTPStatusError),
        wait=wait_retry_after(max_wait=300),
        stop=stop_after_attempt(5),
        reraise=True
    ),
    validate_response=lambda r: r.raise_for_status()
)
client = AsyncClient(transport=transport)

```

Source code in `pydantic_ai_slim/pydantic_ai/retries.py`

````python
class AsyncTenacityTransport(AsyncBaseTransport):
    """Asynchronous HTTP transport with tenacity-based retry functionality.

    This transport wraps another AsyncBaseTransport and adds retry capabilities using the tenacity library.
    It can be configured to retry requests based on various conditions such as specific exception types,
    response status codes, or custom validation logic.

    The transport works by intercepting HTTP requests and responses, allowing the tenacity controller
    to determine when and how to retry failed requests. The validate_response function can be used
    to convert HTTP responses into exceptions that trigger retries.

    Args:
        wrapped: The underlying async transport to wrap and add retry functionality to.
        config: The arguments to use for the tenacity `retry` decorator, including retry conditions,
            wait strategy, stop conditions, etc. See the tenacity docs for more info.
        validate_response: Optional callable that takes a Response and can raise an exception
            to be handled by the controller if the response should trigger a retry.
            Common use case is to raise exceptions for certain HTTP status codes.
            If None, no response validation is performed.

    Example:
        ```python
        from httpx import AsyncClient, HTTPStatusError
        from tenacity import retry_if_exception_type, stop_after_attempt

        from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

        transport = AsyncTenacityTransport(
            RetryConfig(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=300),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            validate_response=lambda r: r.raise_for_status()
        )
        client = AsyncClient(transport=transport)
        ```
    """

    def __init__(
        self,
        config: RetryConfig,
        wrapped: AsyncBaseTransport | None = None,
        validate_response: Callable[[Response], Any] | None = None,
    ):
        self.config = config
        self.wrapped = wrapped or AsyncHTTPTransport()
        self.validate_response = validate_response

    async def handle_async_request(self, request: Request) -> Response:
        """Handle an async HTTP request with retry logic.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.

        Raises:
            RuntimeError: If the retry controller did not make any attempts.
            Exception: Any exception raised by the wrapped transport or validation function.
        """

        @retry(**self.config)
        async def handle_async_request(req: Request) -> Response:
            response = await self.wrapped.handle_async_request(req)

            # this is normally set by httpx _after_ calling this function, but we want the request in the validator:
            response.request = req

            if self.validate_response:
                try:
                    self.validate_response(response)
                except Exception:
                    await response.aclose()
                    raise
            return response

        return await handle_async_request(request)

    async def __aenter__(self) -> AsyncTenacityTransport:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        await self.wrapped.__aexit__(exc_type, exc_value, traceback)

    async def aclose(self) -> None:
        await self.wrapped.aclose()

````

#### handle_async_request

```python
handle_async_request(request: Request) -> Response

```

Handle an async HTTP request with retry logic.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `request` | `Request` | The HTTP request to handle. | *required* |

Returns:

| Type | Description | | --- | --- | | `Response` | The HTTP response. |

Raises:

| Type | Description | | --- | --- | | `RuntimeError` | If the retry controller did not make any attempts. | | `Exception` | Any exception raised by the wrapped transport or validation function. |

Source code in `pydantic_ai_slim/pydantic_ai/retries.py`

```python
async def handle_async_request(self, request: Request) -> Response:
    """Handle an async HTTP request with retry logic.

    Args:
        request: The HTTP request to handle.

    Returns:
        The HTTP response.

    Raises:
        RuntimeError: If the retry controller did not make any attempts.
        Exception: Any exception raised by the wrapped transport or validation function.
    """

    @retry(**self.config)
    async def handle_async_request(req: Request) -> Response:
        response = await self.wrapped.handle_async_request(req)

        # this is normally set by httpx _after_ calling this function, but we want the request in the validator:
        response.request = req

        if self.validate_response:
            try:
                self.validate_response(response)
            except Exception:
                await response.aclose()
                raise
        return response

    return await handle_async_request(request)

```

### wait_retry_after

```python
wait_retry_after(
    fallback_strategy: (
        Callable[[RetryCallState], float] | None
    ) = None,
    max_wait: float = 300,
) -> Callable[[RetryCallState], float]

```

Create a tenacity-compatible wait strategy that respects HTTP Retry-After headers.

This wait strategy checks if the exception contains an HTTPStatusError with a Retry-After header, and if so, waits for the time specified in the header. If no header is present or parsing fails, it falls back to the provided strategy.

The Retry-After header can be in two formats:

- An integer representing seconds to wait
- An HTTP date string representing when to retry

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `fallback_strategy` | `Callable[[RetryCallState], float] | None` | Wait strategy to use when no Retry-After header is present or parsing fails. Defaults to exponential backoff with max 60s. | `None` | | `max_wait` | `float` | Maximum time to wait in seconds, regardless of header value. Defaults to 300 (5 minutes). | `300` |

Returns:

| Type | Description | | --- | --- | | `Callable[[RetryCallState], float]` | A wait function that can be used with tenacity retry decorators. |

Example

```python
from httpx import AsyncClient, HTTPStatusError
from tenacity import retry_if_exception_type, stop_after_attempt

from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

transport = AsyncTenacityTransport(
    RetryConfig(
        retry=retry_if_exception_type(HTTPStatusError),
        wait=wait_retry_after(max_wait=120),
        stop=stop_after_attempt(5),
        reraise=True
    ),
    validate_response=lambda r: r.raise_for_status()
)
client = AsyncClient(transport=transport)

```

Source code in `pydantic_ai_slim/pydantic_ai/retries.py`

````python
def wait_retry_after(
    fallback_strategy: Callable[[RetryCallState], float] | None = None, max_wait: float = 300
) -> Callable[[RetryCallState], float]:
    """Create a tenacity-compatible wait strategy that respects HTTP Retry-After headers.

    This wait strategy checks if the exception contains an HTTPStatusError with a
    Retry-After header, and if so, waits for the time specified in the header.
    If no header is present or parsing fails, it falls back to the provided strategy.

    The Retry-After header can be in two formats:
    - An integer representing seconds to wait
    - An HTTP date string representing when to retry

    Args:
        fallback_strategy: Wait strategy to use when no Retry-After header is present
                          or parsing fails. Defaults to exponential backoff with max 60s.
        max_wait: Maximum time to wait in seconds, regardless of header value.
                 Defaults to 300 (5 minutes).

    Returns:
        A wait function that can be used with tenacity retry decorators.

    Example:
        ```python
        from httpx import AsyncClient, HTTPStatusError
        from tenacity import retry_if_exception_type, stop_after_attempt

        from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

        transport = AsyncTenacityTransport(
            RetryConfig(
                retry=retry_if_exception_type(HTTPStatusError),
                wait=wait_retry_after(max_wait=120),
                stop=stop_after_attempt(5),
                reraise=True
            ),
            validate_response=lambda r: r.raise_for_status()
        )
        client = AsyncClient(transport=transport)
        ```
    """
    if fallback_strategy is None:
        fallback_strategy = wait_exponential(multiplier=1, max=60)

    def wait_func(state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, HTTPStatusError):
            retry_after = exc.response.headers.get('retry-after')
            if retry_after:
                try:
                    # Try parsing as seconds first
                    wait_seconds = int(retry_after)
                    return min(float(wait_seconds), max_wait)
                except ValueError:
                    # Try parsing as HTTP date
                    try:
                        retry_time = cast(datetime, parsedate_to_datetime(retry_after))
                        assert isinstance(retry_time, datetime)
                        now = datetime.now(timezone.utc)
                        wait_seconds = (retry_time - now).total_seconds()

                        if wait_seconds > 0:
                            return min(wait_seconds, max_wait)
                    except (ValueError, TypeError, AssertionError):
                        # If date parsing fails, fall back to fallback strategy
                        pass

        # Use fallback strategy
        return fallback_strategy(state)

    return wait_func

````

# `pydantic_ai.run`

### AgentRun

Bases: `Generic[AgentDepsT, OutputDataT]`

A stateful, async-iterable run of an Agent.

You generally obtain an `AgentRun` instance by calling `async with my_agent.iter(...) as agent_run:`.

Once you have an instance, you can use it to iterate through the run's nodes as they execute. When an End is reached, the run finishes and result becomes available.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    nodes = []
    # Iterate through the run, recording each node along the way:
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

You can also manually drive the iteration using the next method for more granular control.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

````python
@dataclasses.dataclass(repr=False)
class AgentRun(Generic[AgentDepsT, OutputDataT]):
    """A stateful, async-iterable run of an [`Agent`][pydantic_ai.agent.Agent].

    You generally obtain an `AgentRun` instance by calling `async with my_agent.iter(...) as agent_run:`.

    Once you have an instance, you can use it to iterate through the run's nodes as they execute. When an
    [`End`][pydantic_graph.nodes.End] is reached, the run finishes and [`result`][pydantic_ai.agent.AgentRun.result]
    becomes available.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
        # Iterate through the run, recording each node along the way:
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

    You can also manually drive the iteration using the [`next`][pydantic_ai.agent.AgentRun.next] method for
    more granular control.
    """

    _graph_run: GraphRun[
        _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[OutputDataT]
    ]

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        traceparent = self._graph_run._traceparent(required=False)  # type: ignore[reportPrivateUsage]
        if traceparent is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return traceparent

    @property
    def ctx(self) -> GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]]:
        """The current context of the agent run."""
        return GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]](
            state=self._graph_run.state, deps=self._graph_run.deps
        )

    @property
    def next_node(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """The next node that will be run in the agent graph.

        This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
        """
        task = self._graph_run.next_task
        return self._task_to_node(task)

    @property
    def result(self) -> AgentRunResult[OutputDataT] | None:
        """The final result of the run if it has ended, otherwise `None`.

        Once the run returns an [`End`][pydantic_graph.nodes.End] node, `result` is populated
        with an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
        """
        graph_run_output = self._graph_run.output
        if graph_run_output is None:
            return None
        return AgentRunResult(
            graph_run_output.output,
            graph_run_output.tool_name,
            self._graph_run.state,
            self._graph_run.deps.new_message_index,
            self._traceparent(required=False),
        )

    def all_messages(self) -> list[_messages.ModelMessage]:
        """Return all messages for the run so far.

        Messages from older runs are included.
        """
        return self.ctx.state.message_history

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRun.all_messages] as JSON bytes.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(self.all_messages())

    def new_messages(self) -> list[_messages.ModelMessage]:
        """Return new messages for the run so far.

        Messages from older runs are excluded.
        """
        return self.all_messages()[self.ctx.deps.new_message_index :]

    def new_messages_json(self) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRun.new_messages] as JSON bytes.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(self.new_messages())

    def __aiter__(
        self,
    ) -> AsyncIterator[_agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]]:
        """Provide async-iteration over the nodes in the agent run."""
        return self

    async def __anext__(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Advance to the next node automatically based on the last returned node."""
        task = await anext(self._graph_run)
        return self._task_to_node(task)

    def _task_to_node(
        self, task: EndMarker[FinalResult[OutputDataT]] | JoinItem | Sequence[GraphTaskRequest]
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        if isinstance(task, Sequence) and len(task) == 1:
            first_task = task[0]
            if isinstance(first_task.inputs, BaseNode):  # pragma: no branch
                base_node: BaseNode[
                    _agent_graph.GraphAgentState,
                    _agent_graph.GraphAgentDeps[AgentDepsT, OutputDataT],
                    FinalResult[OutputDataT],
                ] = first_task.inputs  # type: ignore[reportUnknownMemberType]
                if _agent_graph.is_agent_node(node=base_node):  # pragma: no branch
                    return base_node
        if isinstance(task, EndMarker):
            return End(task.value)
        raise exceptions.AgentRunError(f'Unexpected node: {task}')  # pragma: no cover

    def _node_to_task(self, node: _agent_graph.AgentNode[AgentDepsT, OutputDataT]) -> GraphTaskRequest:
        return GraphTaskRequest(NodeStep(type(node)).id, inputs=node, fork_stack=())

    async def next(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, OutputDataT],
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Manually drive the agent run by passing in the node you want to run next.

        This lets you inspect or mutate the node before continuing execution, or skip certain nodes
        under dynamic conditions. The agent run should be stopped when you return an [`End`][pydantic_graph.nodes.End]
        node.

        Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_graph import End

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.iter('What is the capital of France?') as agent_run:
                next_node = agent_run.next_node  # start with the first node
                nodes = [next_node]
                while not isinstance(next_node, End):
                    next_node = await agent_run.next(next_node)
                    nodes.append(next_node)
                # Once `next_node` is an End, we've finished:
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
                print('Final result:', agent_run.result.output)
                #> Final result: The capital of France is Paris.
        ```

        Args:
            node: The node to run next in the graph.

        Returns:
            The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
            the run has completed.
        """
        # Note: It might be nice to expose a synchronous interface for iteration, but we shouldn't do it
        # on this class, or else IDEs won't warn you if you accidentally use `for` instead of `async for` to iterate.
        task = [self._node_to_task(node)]
        try:
            task = await self._graph_run.next(task)
        except StopAsyncIteration:
            pass
        return self._task_to_node(task)

    # TODO (v2): Make this a property
    def usage(self) -> _usage.RunUsage:
        """Get usage statistics for the run so far, including token usage, model requests, and so on."""
        return self._graph_run.state.usage

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        return self._graph_run.state.run_id

    def __repr__(self) -> str:  # pragma: no cover
        result = self._graph_run.output
        result_repr = '<run not finished>' if result is None else repr(result.output)
        return f'<{type(self).__name__} result={result_repr} usage={self.usage()}>'

````

#### ctx

```python
ctx: GraphRunContext[
    GraphAgentState, GraphAgentDeps[AgentDepsT, Any]
]

```

The current context of the agent run.

#### next_node

```python
next_node: (
    AgentNode[AgentDepsT, OutputDataT]
    | End[FinalResult[OutputDataT]]
)

```

The next node that will be run in the agent graph.

This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.

#### result

```python
result: AgentRunResult[OutputDataT] | None

```

The final result of the run if it has ended, otherwise `None`.

Once the run returns an End node, `result` is populated with an AgentRunResult.

#### all_messages

```python
all_messages() -> list[ModelMessage]

```

Return all messages for the run so far.

Messages from older runs are included.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def all_messages(self) -> list[_messages.ModelMessage]:
    """Return all messages for the run so far.

    Messages from older runs are included.
    """
    return self.ctx.state.message_history

```

#### all_messages_json

```python
all_messages_json(
    *, output_tool_return_content: str | None = None
) -> bytes

```

Return all messages from all_messages as JSON bytes.

Returns:

| Type | Description | | --- | --- | | `bytes` | JSON bytes representing the messages. |

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
    """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRun.all_messages] as JSON bytes.

    Returns:
        JSON bytes representing the messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(self.all_messages())

```

#### new_messages

```python
new_messages() -> list[ModelMessage]

```

Return new messages for the run so far.

Messages from older runs are excluded.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def new_messages(self) -> list[_messages.ModelMessage]:
    """Return new messages for the run so far.

    Messages from older runs are excluded.
    """
    return self.all_messages()[self.ctx.deps.new_message_index :]

```

#### new_messages_json

```python
new_messages_json() -> bytes

```

Return new messages from new_messages as JSON bytes.

Returns:

| Type | Description | | --- | --- | | `bytes` | JSON bytes representing the new messages. |

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def new_messages_json(self) -> bytes:
    """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRun.new_messages] as JSON bytes.

    Returns:
        JSON bytes representing the new messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(self.new_messages())

```

#### __aiter__

```python
__aiter__() -> (
    AsyncIterator[
        AgentNode[AgentDepsT, OutputDataT]
        | End[FinalResult[OutputDataT]]
    ]
)

```

Provide async-iteration over the nodes in the agent run.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def __aiter__(
    self,
) -> AsyncIterator[_agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]]:
    """Provide async-iteration over the nodes in the agent run."""
    return self

```

#### __anext__

```python
__anext__() -> (
    AgentNode[AgentDepsT, OutputDataT]
    | End[FinalResult[OutputDataT]]
)

```

Advance to the next node automatically based on the last returned node.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
async def __anext__(
    self,
) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
    """Advance to the next node automatically based on the last returned node."""
    task = await anext(self._graph_run)
    return self._task_to_node(task)

```

#### next

```python
next(
    node: AgentNode[AgentDepsT, OutputDataT],
) -> (
    AgentNode[AgentDepsT, OutputDataT]
    | End[FinalResult[OutputDataT]]
)

```

Manually drive the agent run by passing in the node you want to run next.

This lets you inspect or mutate the node before continuing execution, or skip certain nodes under dynamic conditions. The agent run should be stopped when you return an End node.

Example:

```python
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.iter('What is the capital of France?') as agent_run:
        next_node = agent_run.next_node  # start with the first node
        nodes = [next_node]
        while not isinstance(next_node, End):
            next_node = await agent_run.next(next_node)
            nodes.append(next_node)
        # Once `next_node` is an End, we've finished:
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
        print('Final result:', agent_run.result.output)
        #> Final result: The capital of France is Paris.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `node` | `AgentNode[AgentDepsT, OutputDataT]` | The node to run next in the graph. | *required* |

Returns:

| Type | Description | | --- | --- | | `AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]` | The next node returned by the graph logic, or an End node if | | `AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]` | the run has completed. |

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

````python
async def next(
    self,
    node: _agent_graph.AgentNode[AgentDepsT, OutputDataT],
) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
    """Manually drive the agent run by passing in the node you want to run next.

    This lets you inspect or mutate the node before continuing execution, or skip certain nodes
    under dynamic conditions. The agent run should be stopped when you return an [`End`][pydantic_graph.nodes.End]
    node.

    Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_graph import End

    agent = Agent('openai:gpt-4o')

    async def main():
        async with agent.iter('What is the capital of France?') as agent_run:
            next_node = agent_run.next_node  # start with the first node
            nodes = [next_node]
            while not isinstance(next_node, End):
                next_node = await agent_run.next(next_node)
                nodes.append(next_node)
            # Once `next_node` is an End, we've finished:
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
            print('Final result:', agent_run.result.output)
            #> Final result: The capital of France is Paris.
    ```

    Args:
        node: The node to run next in the graph.

    Returns:
        The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
        the run has completed.
    """
    # Note: It might be nice to expose a synchronous interface for iteration, but we shouldn't do it
    # on this class, or else IDEs won't warn you if you accidentally use `for` instead of `async for` to iterate.
    task = [self._node_to_task(node)]
    try:
        task = await self._graph_run.next(task)
    except StopAsyncIteration:
        pass
    return self._task_to_node(task)

````

#### usage

```python
usage() -> RunUsage

```

Get usage statistics for the run so far, including token usage, model requests, and so on.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def usage(self) -> _usage.RunUsage:
    """Get usage statistics for the run so far, including token usage, model requests, and so on."""
    return self._graph_run.state.usage

```

#### run_id

```python
run_id: str

```

The unique identifier for the agent run.

### AgentRunResult

Bases: `Generic[OutputDataT]`

The final result of an agent run.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
@dataclasses.dataclass
class AgentRunResult(Generic[OutputDataT]):
    """The final result of an agent run."""

    output: OutputDataT
    """The output data from the agent run."""

    _output_tool_name: str | None = dataclasses.field(repr=False, compare=False, default=None)
    _state: _agent_graph.GraphAgentState = dataclasses.field(
        repr=False, compare=False, default_factory=_agent_graph.GraphAgentState
    )
    _new_message_index: int = dataclasses.field(repr=False, compare=False, default=0)
    _traceparent_value: str | None = dataclasses.field(repr=False, compare=False, default=None)

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        if self._traceparent_value is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return self._traceparent_value

    def _set_output_tool_return(self, return_content: str) -> list[_messages.ModelMessage]:
        """Set return content for the output tool.

        Useful if you want to continue the conversation and want to set the response to the output tool call.
        """
        if not self._output_tool_name:
            raise ValueError('Cannot set output tool return content when the return type is `str`.')

        messages = self._state.message_history
        last_message = messages[-1]
        for idx, part in enumerate(last_message.parts):
            if isinstance(part, _messages.ToolReturnPart) and part.tool_name == self._output_tool_name:
                # Only do deepcopy when we have to modify
                copied_messages = list(messages)
                copied_last = deepcopy(last_message)
                copied_last.parts[idx].content = return_content  # type: ignore[misc]
                copied_messages[-1] = copied_last
                return copied_messages

        raise LookupError(f'No tool call found with tool name {self._output_tool_name!r}.')

    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        if output_tool_return_content is not None:
            return self._set_output_tool_return(output_tool_return_content)
        else:
            return self._state.message_history

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRunResult.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(output_tool_return_content=output_tool_return_content)
        )

    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self.all_messages(output_tool_return_content=output_tool_return_content)[self._new_message_index :]

    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRunResult.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.new_messages(output_tool_return_content=output_tool_return_content)
        )

    @property
    def response(self) -> _messages.ModelResponse:
        """Return the last response from the message history."""
        # The response may not be the very last item if it contained an output tool call. See `CallToolsNode._handle_final_result`.
        for message in reversed(self.all_messages()):
            if isinstance(message, _messages.ModelResponse):
                return message
        raise ValueError('No response found in the message history')  # pragma: no cover

    # TODO (v2): Make this a property
    def usage(self) -> _usage.RunUsage:
        """Return the usage of the whole run."""
        return self._state.usage

    # TODO (v2): Make this a property
    def timestamp(self) -> datetime:
        """Return the timestamp of last response."""
        return self.response.timestamp

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        return self._state.run_id

```

#### output

```python
output: OutputDataT

```

The output data from the agent run.

#### all_messages

```python
all_messages(
    *, output_tool_return_content: str | None = None
) -> list[ModelMessage]

```

Return the history of \_messages.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `list[ModelMessage]` | List of messages. |

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
    """Return the history of _messages.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        List of messages.
    """
    if output_tool_return_content is not None:
        return self._set_output_tool_return(output_tool_return_content)
    else:
        return self._state.message_history

```

#### all_messages_json

```python
all_messages_json(
    *, output_tool_return_content: str | None = None
) -> bytes

```

Return all messages from all_messages as JSON bytes.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `bytes` | JSON bytes representing the messages. |

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
    """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRunResult.all_messages] as JSON bytes.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        JSON bytes representing the messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(
        self.all_messages(output_tool_return_content=output_tool_return_content)
    )

```

#### new_messages

```python
new_messages(
    *, output_tool_return_content: str | None = None
) -> list[ModelMessage]

```

Return new messages associated with this run.

Messages from older runs are excluded.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `list[ModelMessage]` | List of new messages. |

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
    """Return new messages associated with this run.

    Messages from older runs are excluded.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        List of new messages.
    """
    return self.all_messages(output_tool_return_content=output_tool_return_content)[self._new_message_index :]

```

#### new_messages_json

```python
new_messages_json(
    *, output_tool_return_content: str | None = None
) -> bytes

```

Return new messages from new_messages as JSON bytes.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_tool_return_content` | `str | None` | The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the output tool call if you want to continue the conversation and want to set the response to the output tool call. If None, the last message will not be modified. | `None` |

Returns:

| Type | Description | | --- | --- | | `bytes` | JSON bytes representing the new messages. |

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
    """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRunResult.new_messages] as JSON bytes.

    Args:
        output_tool_return_content: The return content of the tool call to set in the last message.
            This provides a convenient way to modify the content of the output tool call if you want to continue
            the conversation and want to set the response to the output tool call. If `None`, the last message will
            not be modified.

    Returns:
        JSON bytes representing the new messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(
        self.new_messages(output_tool_return_content=output_tool_return_content)
    )

```

#### response

```python
response: ModelResponse

```

Return the last response from the message history.

#### usage

```python
usage() -> RunUsage

```

Return the usage of the whole run.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def usage(self) -> _usage.RunUsage:
    """Return the usage of the whole run."""
    return self._state.usage

```

#### timestamp

```python
timestamp() -> datetime

```

Return the timestamp of last response.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
def timestamp(self) -> datetime:
    """Return the timestamp of last response."""
    return self.response.timestamp

```

#### run_id

```python
run_id: str

```

The unique identifier for the agent run.

### AgentRunResultEvent

Bases: `Generic[OutputDataT]`

An event indicating the agent run ended and containing the final result of the agent run.

Source code in `pydantic_ai_slim/pydantic_ai/run.py`

```python
@dataclasses.dataclass(repr=False)
class AgentRunResultEvent(Generic[OutputDataT]):
    """An event indicating the agent run ended and containing the final result of the agent run."""

    result: AgentRunResult[OutputDataT]
    """The result of the run."""

    _: dataclasses.KW_ONLY

    event_kind: Literal['agent_run_result'] = 'agent_run_result'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr

```

#### result

```python
result: AgentRunResult[OutputDataT]

```

The result of the run.

#### event_kind

```python
event_kind: Literal["agent_run_result"] = "agent_run_result"

```

Event type identifier, used as a discriminator.

# `pydantic_ai.settings`

### ModelSettings

Bases: `TypedDict`

Settings to configure an LLM.

Here we include only settings which apply to multiple models / model providers, though not all of these settings are supported by all models.

Source code in `pydantic_ai_slim/pydantic_ai/settings.py`

```python
class ModelSettings(TypedDict, total=False):
    """Settings to configure an LLM.

    Here we include only settings which apply to multiple models / model providers,
    though not all of these settings are supported by all models.
    """

    max_tokens: int
    """The maximum number of tokens to generate before stopping.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * MCP Sampling
    * Outlines (all providers)
    """

    temperature: float
    """Amount of randomness injected into the response.

    Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to a model's
    maximum `temperature` for creative and generative tasks.

    Note that even with `temperature` of `0.0`, the results will not be fully deterministic.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * Outlines (Transformers, LlamaCpp, SgLang, VLLMOffline)
    """

    top_p: float
    """An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.

    So 0.1 means only the tokens comprising the top 10% probability mass are considered.

    You should either alter `temperature` or `top_p`, but not both.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    * Outlines (Transformers, LlamaCpp, SgLang, VLLMOffline)
    """

    timeout: float | Timeout
    """Override the client-level default timeout for a request, in seconds.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Mistral
    """

    parallel_tool_calls: bool
    """Whether to allow parallel tool calls.

    Supported by:

    * OpenAI (some models, not o1)
    * Groq
    * Anthropic
    """

    seed: int
    """The random seed to use for the model, theoretically allowing for deterministic results.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Gemini
    * Outlines (LlamaCpp, VLLMOffline)
    """

    presence_penalty: float
    """Penalize new tokens based on whether they have appeared in the text so far.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    * Outlines (LlamaCpp, SgLang, VLLMOffline)
    """

    frequency_penalty: float
    """Penalize new tokens based on their existing frequency in the text so far.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    * Outlines (LlamaCpp, SgLang, VLLMOffline)
    """

    logit_bias: dict[str, int]
    """Modify the likelihood of specified tokens appearing in the completion.

    Supported by:

    * OpenAI
    * Groq
    * Outlines (Transformers, LlamaCpp, VLLMOffline)
    """

    stop_sequences: list[str]
    """Sequences that will cause the model to stop generating.

    Supported by:

    * OpenAI
    * Anthropic
    * Bedrock
    * Mistral
    * Groq
    * Cohere
    * Google
    """

    extra_headers: dict[str, str]
    """Extra headers to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    """

    extra_body: object
    """Extra body to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    * Outlines (all providers)
    """

```

#### max_tokens

```python
max_tokens: int

```

The maximum number of tokens to generate before stopping.

Supported by:

- Gemini
- Anthropic
- OpenAI
- Groq
- Cohere
- Mistral
- Bedrock
- MCP Sampling
- Outlines (all providers)

#### temperature

```python
temperature: float

```

Amount of randomness injected into the response.

Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to a model's maximum `temperature` for creative and generative tasks.

Note that even with `temperature` of `0.0`, the results will not be fully deterministic.

Supported by:

- Gemini
- Anthropic
- OpenAI
- Groq
- Cohere
- Mistral
- Bedrock
- Outlines (Transformers, LlamaCpp, SgLang, VLLMOffline)

#### top_p

```python
top_p: float

```

An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.

So 0.1 means only the tokens comprising the top 10% probability mass are considered.

You should either alter `temperature` or `top_p`, but not both.

Supported by:

- Gemini
- Anthropic
- OpenAI
- Groq
- Cohere
- Mistral
- Bedrock
- Outlines (Transformers, LlamaCpp, SgLang, VLLMOffline)

#### timeout

```python
timeout: float | Timeout

```

Override the client-level default timeout for a request, in seconds.

Supported by:

- Gemini
- Anthropic
- OpenAI
- Groq
- Mistral

#### parallel_tool_calls

```python
parallel_tool_calls: bool

```

Whether to allow parallel tool calls.

Supported by:

- OpenAI (some models, not o1)
- Groq
- Anthropic

#### seed

```python
seed: int

```

The random seed to use for the model, theoretically allowing for deterministic results.

Supported by:

- OpenAI
- Groq
- Cohere
- Mistral
- Gemini
- Outlines (LlamaCpp, VLLMOffline)

#### presence_penalty

```python
presence_penalty: float

```

Penalize new tokens based on whether they have appeared in the text so far.

Supported by:

- OpenAI
- Groq
- Cohere
- Gemini
- Mistral
- Outlines (LlamaCpp, SgLang, VLLMOffline)

#### frequency_penalty

```python
frequency_penalty: float

```

Penalize new tokens based on their existing frequency in the text so far.

Supported by:

- OpenAI
- Groq
- Cohere
- Gemini
- Mistral
- Outlines (LlamaCpp, SgLang, VLLMOffline)

#### logit_bias

```python
logit_bias: dict[str, int]

```

Modify the likelihood of specified tokens appearing in the completion.

Supported by:

- OpenAI
- Groq
- Outlines (Transformers, LlamaCpp, VLLMOffline)

#### stop_sequences

```python
stop_sequences: list[str]

```

Sequences that will cause the model to stop generating.

Supported by:

- OpenAI
- Anthropic
- Bedrock
- Mistral
- Groq
- Cohere
- Google

#### extra_headers

```python
extra_headers: dict[str, str]

```

Extra headers to send to the model.

Supported by:

- OpenAI
- Anthropic
- Groq

#### extra_body

```python
extra_body: object

```

Extra body to send to the model.

Supported by:

- OpenAI
- Anthropic
- Groq
- Outlines (all providers)

# `pydantic_ai.tools`

### AgentDepsT

```python
AgentDepsT = TypeVar(
    "AgentDepsT", default=None, contravariant=True
)

```

Type variable for agent dependencies.

### RunContext

Bases: `Generic[RunContextAgentDepsT]`

Information about the current call.

Source code in `pydantic_ai_slim/pydantic_ai/_run_context.py`

```python
@dataclasses.dataclass(repr=False, kw_only=True)
class RunContext(Generic[RunContextAgentDepsT]):
    """Information about the current call."""

    deps: RunContextAgentDepsT
    """Dependencies for the agent."""
    model: Model
    """The model used in this run."""
    usage: RunUsage
    """LLM usage associated with the run."""
    prompt: str | Sequence[_messages.UserContent] | None = None
    """The original user prompt passed to the run."""
    messages: list[_messages.ModelMessage] = field(default_factory=list)
    """Messages exchanged in the conversation so far."""
    tracer: Tracer = field(default_factory=NoOpTracer)
    """The tracer to use for tracing the run."""
    trace_include_content: bool = False
    """Whether to include the content of the messages in the trace."""
    instrumentation_version: int = DEFAULT_INSTRUMENTATION_VERSION
    """Instrumentation settings version, if instrumentation is enabled."""
    retries: dict[str, int] = field(default_factory=dict)
    """Number of retries for each tool so far."""
    tool_call_id: str | None = None
    """The ID of the tool call."""
    tool_name: str | None = None
    """Name of the tool being called."""
    retry: int = 0
    """Number of retries of this tool so far."""
    max_retries: int = 0
    """The maximum number of retries of this tool."""
    run_step: int = 0
    """The current step in the run."""
    tool_call_approved: bool = False
    """Whether a tool call that required approval has now been approved."""
    partial_output: bool = False
    """Whether the output passed to an output validator is partial."""
    run_id: str | None = None
    """"Unique identifier for the agent run."""

    @property
    def last_attempt(self) -> bool:
        """Whether this is the last attempt at running this tool before an error is raised."""
        return self.retry == self.max_retries

    __repr__ = _utils.dataclasses_no_defaults_repr

```

#### deps

```python
deps: RunContextAgentDepsT

```

Dependencies for the agent.

#### model

```python
model: Model

```

The model used in this run.

#### usage

```python
usage: RunUsage

```

LLM usage associated with the run.

#### prompt

```python
prompt: str | Sequence[UserContent] | None = None

```

The original user prompt passed to the run.

#### messages

```python
messages: list[ModelMessage] = field(default_factory=list)

```

Messages exchanged in the conversation so far.

#### tracer

```python
tracer: Tracer = field(default_factory=NoOpTracer)

```

The tracer to use for tracing the run.

#### trace_include_content

```python
trace_include_content: bool = False

```

Whether to include the content of the messages in the trace.

#### instrumentation_version

```python
instrumentation_version: int = (
    DEFAULT_INSTRUMENTATION_VERSION
)

```

Instrumentation settings version, if instrumentation is enabled.

#### retries

```python
retries: dict[str, int] = field(default_factory=dict)

```

Number of retries for each tool so far.

#### tool_call_id

```python
tool_call_id: str | None = None

```

The ID of the tool call.

#### tool_name

```python
tool_name: str | None = None

```

Name of the tool being called.

#### retry

```python
retry: int = 0

```

Number of retries of this tool so far.

#### max_retries

```python
max_retries: int = 0

```

The maximum number of retries of this tool.

#### run_step

```python
run_step: int = 0

```

The current step in the run.

#### tool_call_approved

```python
tool_call_approved: bool = False

```

Whether a tool call that required approval has now been approved.

#### partial_output

```python
partial_output: bool = False

```

Whether the output passed to an output validator is partial.

#### run_id

```python
run_id: str | None = None

```

"Unique identifier for the agent run.

#### last_attempt

```python
last_attempt: bool

```

Whether this is the last attempt at running this tool before an error is raised.

### ToolParams

```python
ToolParams = ParamSpec('ToolParams', default=...)

```

Retrieval function param spec.

### SystemPromptFunc

```python
SystemPromptFunc: TypeAlias = (
    Callable[[RunContext[AgentDepsT]], str]
    | Callable[[RunContext[AgentDepsT]], Awaitable[str]]
    | Callable[[], str]
    | Callable[[], Awaitable[str]]
)

```

A function that may or maybe not take `RunContext` as an argument, and may or may not be async.

Usage `SystemPromptFunc[AgentDepsT]`.

### ToolFuncContext

```python
ToolFuncContext: TypeAlias = Callable[
    Concatenate[RunContext[AgentDepsT], ToolParams], Any
]

```

A tool function that takes `RunContext` as the first argument.

Usage `ToolContextFunc[AgentDepsT, ToolParams]`.

### ToolFuncPlain

```python
ToolFuncPlain: TypeAlias = Callable[ToolParams, Any]

```

A tool function that does not take `RunContext` as the first argument.

Usage `ToolPlainFunc[ToolParams]`.

### ToolFuncEither

```python
ToolFuncEither: TypeAlias = (
    ToolFuncContext[AgentDepsT, ToolParams]
    | ToolFuncPlain[ToolParams]
)

```

Either kind of tool function.

This is just a union of ToolFuncContext and ToolFuncPlain.

Usage `ToolFuncEither[AgentDepsT, ToolParams]`.

### ToolPrepareFunc

```python
ToolPrepareFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT], "ToolDefinition"],
    Awaitable["ToolDefinition | None"],
]

```

Definition of a function that can prepare a tool definition at call time.

See [tool docs](../../tools-advanced/#tool-prepare) for more information.

Example â€” here `only_if_42` is valid as a `ToolPrepareFunc`:

```python
from pydantic_ai import RunContext, Tool
from pydantic_ai.tools import ToolDefinition

async def only_if_42(
    ctx: RunContext[int], tool_def: ToolDefinition
) -> ToolDefinition | None:
    if ctx.deps == 42:
        return tool_def

def hitchhiker(ctx: RunContext[int], answer: str) -> str:
    return f'{ctx.deps} {answer}'

hitchhiker = Tool(hitchhiker, prepare=only_if_42)

```

Usage `ToolPrepareFunc[AgentDepsT]`.

### ToolsPrepareFunc

```python
ToolsPrepareFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT], list["ToolDefinition"]],
    Awaitable["list[ToolDefinition] | None"],
]

```

Definition of a function that can prepare the tool definition of all tools for each step. This is useful if you want to customize the definition of multiple tools or you want to register a subset of tools for a given step.

Example â€” here `turn_on_strict_if_openai` is valid as a `ToolsPrepareFunc`:

```python
from dataclasses import replace

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition


async def turn_on_strict_if_openai(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    if ctx.model.system == 'openai':
        return [replace(tool_def, strict=True) for tool_def in tool_defs]
    return tool_defs

agent = Agent('openai:gpt-4o', prepare_tools=turn_on_strict_if_openai)

```

Usage `ToolsPrepareFunc[AgentDepsT]`.

### DocstringFormat

```python
DocstringFormat: TypeAlias = Literal[
    "google", "numpy", "sphinx", "auto"
]

```

Supported docstring formats.

- `'google'` â€” [Google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings.
- `'numpy'` â€” [Numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings.
- `'sphinx'` â€” [Sphinx-style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format) docstrings.
- `'auto'` â€” Automatically infer the format based on the structure of the docstring.

### DeferredToolRequests

Tool calls that require approval or external execution.

This can be used as an agent's `output_type` and will be used as the output of the agent run if the model called any deferred tools.

Results can be passed to the next agent run using a DeferredToolResults object with the same tool call IDs.

See [deferred tools docs](../../deferred-tools/#deferred-tools) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/tools.py`

```python
@dataclass(kw_only=True)
class DeferredToolRequests:
    """Tool calls that require approval or external execution.

    This can be used as an agent's `output_type` and will be used as the output of the agent run if the model called any deferred tools.

    Results can be passed to the next agent run using a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] object with the same tool call IDs.

    See [deferred tools docs](../deferred-tools.md#deferred-tools) for more information.
    """

    calls: list[ToolCallPart] = field(default_factory=list)
    """Tool calls that require external execution."""
    approvals: list[ToolCallPart] = field(default_factory=list)
    """Tool calls that require human-in-the-loop approval."""
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Metadata for deferred tool calls, keyed by `tool_call_id`."""

```

#### calls

```python
calls: list[ToolCallPart] = field(default_factory=list)

```

Tool calls that require external execution.

#### approvals

```python
approvals: list[ToolCallPart] = field(default_factory=list)

```

Tool calls that require human-in-the-loop approval.

#### metadata

```python
metadata: dict[str, dict[str, Any]] = field(
    default_factory=dict
)

```

Metadata for deferred tool calls, keyed by `tool_call_id`.

### ToolApproved

Indicates that a tool call has been approved and that the tool function should be executed.

Source code in `pydantic_ai_slim/pydantic_ai/tools.py`

```python
@dataclass(kw_only=True)
class ToolApproved:
    """Indicates that a tool call has been approved and that the tool function should be executed."""

    override_args: dict[str, Any] | None = None
    """Optional tool call arguments to use instead of the original arguments."""

    kind: Literal['tool-approved'] = 'tool-approved'

```

#### override_args

```python
override_args: dict[str, Any] | None = None

```

Optional tool call arguments to use instead of the original arguments.

### ToolDenied

Indicates that a tool call has been denied and that a denial message should be returned to the model.

Source code in `pydantic_ai_slim/pydantic_ai/tools.py`

```python
@dataclass
class ToolDenied:
    """Indicates that a tool call has been denied and that a denial message should be returned to the model."""

    message: str = 'The tool call was denied.'
    """The message to return to the model."""

    _: KW_ONLY

    kind: Literal['tool-denied'] = 'tool-denied'

```

#### message

```python
message: str = 'The tool call was denied.'

```

The message to return to the model.

### DeferredToolResults

Results for deferred tool calls from a previous run that required approval or external execution.

The tool call IDs need to match those from the DeferredToolRequests output object from the previous run.

See [deferred tools docs](../../deferred-tools/#deferred-tools) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/tools.py`

```python
@dataclass(kw_only=True)
class DeferredToolResults:
    """Results for deferred tool calls from a previous run that required approval or external execution.

    The tool call IDs need to match those from the [`DeferredToolRequests`][pydantic_ai.output.DeferredToolRequests] output object from the previous run.

    See [deferred tools docs](../deferred-tools.md#deferred-tools) for more information.
    """

    calls: dict[str, DeferredToolCallResult | Any] = field(default_factory=dict)
    """Map of tool call IDs to results for tool calls that required external execution."""
    approvals: dict[str, bool | DeferredToolApprovalResult] = field(default_factory=dict)
    """Map of tool call IDs to results for tool calls that required human-in-the-loop approval."""

```

#### calls

```python
calls: dict[str, DeferredToolCallResult | Any] = field(
    default_factory=dict
)

```

Map of tool call IDs to results for tool calls that required external execution.

#### approvals

```python
approvals: dict[str, bool | DeferredToolApprovalResult] = (
    field(default_factory=dict)
)

```

Map of tool call IDs to results for tool calls that required human-in-the-loop approval.

### Tool

Bases: `Generic[ToolAgentDepsT]`

A tool function for an agent.

Source code in `pydantic_ai_slim/pydantic_ai/tools.py`

````python
@dataclass(init=False)
class Tool(Generic[ToolAgentDepsT]):
    """A tool function for an agent."""

    function: ToolFuncEither[ToolAgentDepsT]
    takes_ctx: bool
    max_retries: int | None
    name: str
    description: str | None
    prepare: ToolPrepareFunc[ToolAgentDepsT] | None
    docstring_format: DocstringFormat
    require_parameter_descriptions: bool
    strict: bool | None
    sequential: bool
    requires_approval: bool
    metadata: dict[str, Any] | None
    function_schema: _function_schema.FunctionSchema
    """
    The base JSON schema for the tool's parameters.

    This schema may be modified by the `prepare` function or by the Model class prior to including it in an API request.
    """

    def __init__(
        self,
        function: ToolFuncEither[ToolAgentDepsT],
        *,
        takes_ctx: bool | None = None,
        max_retries: int | None = None,
        name: str | None = None,
        description: str | None = None,
        prepare: ToolPrepareFunc[ToolAgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        function_schema: _function_schema.FunctionSchema | None = None,
    ):
        """Create a new tool instance.

        Example usage:

        ```python {noqa="I001"}
        from pydantic_ai import Agent, RunContext, Tool

        async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
            return f'{ctx.deps} {x} {y}'

        agent = Agent('test', tools=[Tool(my_tool)])
        ```

        or with a custom prepare method:

        ```python {noqa="I001"}

        from pydantic_ai import Agent, RunContext, Tool
        from pydantic_ai.tools import ToolDefinition

        async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
            return f'{ctx.deps} {x} {y}'

        async def prep_my_tool(
            ctx: RunContext[int], tool_def: ToolDefinition
        ) -> ToolDefinition | None:
            # only register the tool if `deps == 42`
            if ctx.deps == 42:
                return tool_def

        agent = Agent('test', tools=[Tool(my_tool, prepare=prep_my_tool)])
        ```


        Args:
            function: The Python function to call as the tool.
            takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] first argument,
                this is inferred if unset.
            max_retries: Maximum number of retries allowed for this tool, set to the agent default if `None`.
            name: Name of the tool, inferred from the function if `None`.
            description: Description of the tool, inferred from the function if `None`.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
            function_schema: The function schema to use for the tool. If not provided, it will be generated.
        """
        self.function = function
        self.function_schema = function_schema or _function_schema.function_schema(
            function,
            schema_generator,
            takes_ctx=takes_ctx,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
        )
        self.takes_ctx = self.function_schema.takes_ctx
        self.max_retries = max_retries
        self.name = name or function.__name__
        self.description = description or self.function_schema.description
        self.prepare = prepare
        self.docstring_format = docstring_format
        self.require_parameter_descriptions = require_parameter_descriptions
        self.strict = strict
        self.sequential = sequential
        self.requires_approval = requires_approval
        self.metadata = metadata

    @classmethod
    def from_schema(
        cls,
        function: Callable[..., Any],
        name: str,
        description: str | None,
        json_schema: JsonSchemaValue,
        takes_ctx: bool = False,
        sequential: bool = False,
    ) -> Self:
        """Creates a Pydantic tool from a function and a JSON schema.

        Args:
            function: The function to call.
                This will be called with keywords only, and no validation of
                the arguments will be performed.
            name: The unique name of the tool that clearly communicates its purpose
            description: Used to tell the model how/when/why to use the tool.
                You can provide few-shot examples as a part of the description.
            json_schema: The schema for the function arguments
            takes_ctx: An optional boolean parameter indicating whether the function
                accepts the context object as an argument.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.

        Returns:
            A Pydantic tool that calls the function
        """
        function_schema = _function_schema.FunctionSchema(
            function=function,
            description=description,
            validator=SchemaValidator(schema=core_schema.any_schema()),
            json_schema=json_schema,
            takes_ctx=takes_ctx,
            is_async=_utils.is_async_callable(function),
        )

        return cls(
            function,
            takes_ctx=takes_ctx,
            name=name,
            description=description,
            function_schema=function_schema,
            sequential=sequential,
        )

    @property
    def tool_def(self):
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.function_schema.json_schema,
            strict=self.strict,
            sequential=self.sequential,
            metadata=self.metadata,
            kind='unapproved' if self.requires_approval else 'function',
        )

    async def prepare_tool_def(self, ctx: RunContext[ToolAgentDepsT]) -> ToolDefinition | None:
        """Get the tool definition.

        By default, this method creates a tool definition, then either returns it, or calls `self.prepare`
        if it's set.

        Returns:
            return a `ToolDefinition` or `None` if the tools should not be registered for this run.
        """
        base_tool_def = self.tool_def

        if self.prepare is not None:
            return await self.prepare(ctx, base_tool_def)
        else:
            return base_tool_def

````

#### __init__

```python
__init__(
    function: ToolFuncEither[ToolAgentDepsT],
    *,
    takes_ctx: bool | None = None,
    max_retries: int | None = None,
    name: str | None = None,
    description: str | None = None,
    prepare: ToolPrepareFunc[ToolAgentDepsT] | None = None,
    docstring_format: DocstringFormat = "auto",
    require_parameter_descriptions: bool = False,
    schema_generator: type[
        GenerateJsonSchema
    ] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
    function_schema: FunctionSchema | None = None
)

```

Create a new tool instance.

Example usage:

```python
from pydantic_ai import Agent, RunContext, Tool

async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
    return f'{ctx.deps} {x} {y}'

agent = Agent('test', tools=[Tool(my_tool)])

```

or with a custom prepare method:

```python
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.tools import ToolDefinition

async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
    return f'{ctx.deps} {x} {y}'

async def prep_my_tool(
    ctx: RunContext[int], tool_def: ToolDefinition
) -> ToolDefinition | None:
    # only register the tool if `deps == 42`
    if ctx.deps == 42:
        return tool_def

agent = Agent('test', tools=[Tool(my_tool, prepare=prep_my_tool)])

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `function` | `ToolFuncEither[ToolAgentDepsT]` | The Python function to call as the tool. | *required* | | `takes_ctx` | `bool | None` | Whether the function takes a RunContext first argument, this is inferred if unset. | `None` | | `max_retries` | `int | None` | Maximum number of retries allowed for this tool, set to the agent default if None. | `None` | | `name` | `str | None` | Name of the tool, inferred from the function if None. | `None` | | `description` | `str | None` | Description of the tool, inferred from the function if None. | `None` | | `prepare` | `ToolPrepareFunc[ToolAgentDepsT] | None` | custom method to prepare the tool definition for each step, return None to omit this tool from a given step. This is useful if you want to customise a tool at call time, or omit it completely from a step. See ToolPrepareFunc. | `None` | | `docstring_format` | `DocstringFormat` | The format of the docstring, see DocstringFormat. Defaults to 'auto', such that the format is inferred from the structure of the docstring. | `'auto'` | | `require_parameter_descriptions` | `bool` | If True, raise an error if a parameter description is missing. Defaults to False. | `False` | | `schema_generator` | `type[GenerateJsonSchema]` | The JSON schema generator class to use. Defaults to GenerateToolJsonSchema. | `GenerateToolJsonSchema` | | `strict` | `bool | None` | Whether to enforce JSON schema compliance (only affects OpenAI). See ToolDefinition for more info. | `None` | | `sequential` | `bool` | Whether the function requires a sequential/serial execution environment. Defaults to False. | `False` | | `requires_approval` | `bool` | Whether this tool requires human-in-the-loop approval. Defaults to False. See the tools documentation for more info. | `False` | | `metadata` | `dict[str, Any] | None` | Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization. | `None` | | `function_schema` | `FunctionSchema | None` | The function schema to use for the tool. If not provided, it will be generated. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/tools.py`

````python
def __init__(
    self,
    function: ToolFuncEither[ToolAgentDepsT],
    *,
    takes_ctx: bool | None = None,
    max_retries: int | None = None,
    name: str | None = None,
    description: str | None = None,
    prepare: ToolPrepareFunc[ToolAgentDepsT] | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
    function_schema: _function_schema.FunctionSchema | None = None,
):
    """Create a new tool instance.

    Example usage:

    ```python {noqa="I001"}
    from pydantic_ai import Agent, RunContext, Tool

    async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
        return f'{ctx.deps} {x} {y}'

    agent = Agent('test', tools=[Tool(my_tool)])
    ```

    or with a custom prepare method:

    ```python {noqa="I001"}

    from pydantic_ai import Agent, RunContext, Tool
    from pydantic_ai.tools import ToolDefinition

    async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
        return f'{ctx.deps} {x} {y}'

    async def prep_my_tool(
        ctx: RunContext[int], tool_def: ToolDefinition
    ) -> ToolDefinition | None:
        # only register the tool if `deps == 42`
        if ctx.deps == 42:
            return tool_def

    agent = Agent('test', tools=[Tool(my_tool, prepare=prep_my_tool)])
    ```


    Args:
        function: The Python function to call as the tool.
        takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] first argument,
            this is inferred if unset.
        max_retries: Maximum number of retries allowed for this tool, set to the agent default if `None`.
        name: Name of the tool, inferred from the function if `None`.
        description: Description of the tool, inferred from the function if `None`.
        prepare: custom method to prepare the tool definition for each step, return `None` to omit this
            tool from a given step. This is useful if you want to customise a tool at call time,
            or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
        docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
            Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
        require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
        schema_generator: The JSON schema generator class to use. Defaults to `GenerateToolJsonSchema`.
        strict: Whether to enforce JSON schema compliance (only affects OpenAI).
            See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
        sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
        requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
            See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
        metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
        function_schema: The function schema to use for the tool. If not provided, it will be generated.
    """
    self.function = function
    self.function_schema = function_schema or _function_schema.function_schema(
        function,
        schema_generator,
        takes_ctx=takes_ctx,
        docstring_format=docstring_format,
        require_parameter_descriptions=require_parameter_descriptions,
    )
    self.takes_ctx = self.function_schema.takes_ctx
    self.max_retries = max_retries
    self.name = name or function.__name__
    self.description = description or self.function_schema.description
    self.prepare = prepare
    self.docstring_format = docstring_format
    self.require_parameter_descriptions = require_parameter_descriptions
    self.strict = strict
    self.sequential = sequential
    self.requires_approval = requires_approval
    self.metadata = metadata

````

#### function_schema

```python
function_schema: FunctionSchema = (
    function_schema
    or function_schema(
        function,
        schema_generator,
        takes_ctx=takes_ctx,
        docstring_format=docstring_format,
        require_parameter_descriptions=require_parameter_descriptions,
    )
)

```

The base JSON schema for the tool's parameters.

This schema may be modified by the `prepare` function or by the Model class prior to including it in an API request.

#### from_schema

```python
from_schema(
    function: Callable[..., Any],
    name: str,
    description: str | None,
    json_schema: JsonSchemaValue,
    takes_ctx: bool = False,
    sequential: bool = False,
) -> Self

```

Creates a Pydantic tool from a function and a JSON schema.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `function` | `Callable[..., Any]` | The function to call. This will be called with keywords only, and no validation of the arguments will be performed. | *required* | | `name` | `str` | The unique name of the tool that clearly communicates its purpose | *required* | | `description` | `str | None` | Used to tell the model how/when/why to use the tool. You can provide few-shot examples as a part of the description. | *required* | | `json_schema` | `JsonSchemaValue` | The schema for the function arguments | *required* | | `takes_ctx` | `bool` | An optional boolean parameter indicating whether the function accepts the context object as an argument. | `False` | | `sequential` | `bool` | Whether the function requires a sequential/serial execution environment. Defaults to False. | `False` |

Returns:

| Type | Description | | --- | --- | | `Self` | A Pydantic tool that calls the function |

Source code in `pydantic_ai_slim/pydantic_ai/tools.py`

```python
@classmethod
def from_schema(
    cls,
    function: Callable[..., Any],
    name: str,
    description: str | None,
    json_schema: JsonSchemaValue,
    takes_ctx: bool = False,
    sequential: bool = False,
) -> Self:
    """Creates a Pydantic tool from a function and a JSON schema.

    Args:
        function: The function to call.
            This will be called with keywords only, and no validation of
            the arguments will be performed.
        name: The unique name of the tool that clearly communicates its purpose
        description: Used to tell the model how/when/why to use the tool.
            You can provide few-shot examples as a part of the description.
        json_schema: The schema for the function arguments
        takes_ctx: An optional boolean parameter indicating whether the function
            accepts the context object as an argument.
        sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.

    Returns:
        A Pydantic tool that calls the function
    """
    function_schema = _function_schema.FunctionSchema(
        function=function,
        description=description,
        validator=SchemaValidator(schema=core_schema.any_schema()),
        json_schema=json_schema,
        takes_ctx=takes_ctx,
        is_async=_utils.is_async_callable(function),
    )

    return cls(
        function,
        takes_ctx=takes_ctx,
        name=name,
        description=description,
        function_schema=function_schema,
        sequential=sequential,
    )

```

#### prepare_tool_def

```python
prepare_tool_def(
    ctx: RunContext[ToolAgentDepsT],
) -> ToolDefinition | None

```

Get the tool definition.

By default, this method creates a tool definition, then either returns it, or calls `self.prepare` if it's set.

Returns:

| Type | Description | | --- | --- | | `ToolDefinition | None` | return a ToolDefinition or None if the tools should not be registered for this run. |

Source code in `pydantic_ai_slim/pydantic_ai/tools.py`

```python
async def prepare_tool_def(self, ctx: RunContext[ToolAgentDepsT]) -> ToolDefinition | None:
    """Get the tool definition.

    By default, this method creates a tool definition, then either returns it, or calls `self.prepare`
    if it's set.

    Returns:
        return a `ToolDefinition` or `None` if the tools should not be registered for this run.
    """
    base_tool_def = self.tool_def

    if self.prepare is not None:
        return await self.prepare(ctx, base_tool_def)
    else:
        return base_tool_def

```

### ObjectJsonSchema

```python
ObjectJsonSchema: TypeAlias = dict[str, Any]

```

Type representing JSON schema of an object, e.g. where `"type": "object"`.

This type is used to define tools parameters (aka arguments) in ToolDefinition.

With PEP-728 this should be a TypedDict with `type: Literal['object']`, and `extra_parts=Any`

### ToolDefinition

Definition of a tool passed to a model.

This is used for both function tools and output tools.

Source code in `pydantic_ai_slim/pydantic_ai/tools.py`

```python
@dataclass(repr=False, kw_only=True)
class ToolDefinition:
    """Definition of a tool passed to a model.

    This is used for both function tools and output tools.
    """

    name: str
    """The name of the tool."""

    parameters_json_schema: ObjectJsonSchema = field(default_factory=lambda: {'type': 'object', 'properties': {}})
    """The JSON schema for the tool's parameters."""

    description: str | None = None
    """The description of the tool."""

    outer_typed_dict_key: str | None = None
    """The key in the outer [TypedDict] that wraps an output tool.

    This will only be set for output tools which don't have an `object` JSON schema.
    """

    strict: bool | None = None
    """Whether to enforce (vendor-specific) strict JSON schema validation for tool calls.

    Setting this to `True` while using a supported model generally imposes some restrictions on the tool's JSON schema
    in exchange for guaranteeing the API responses strictly match that schema.

    When `False`, the model may be free to generate other properties or types (depending on the vendor).
    When `None` (the default), the value will be inferred based on the compatibility of the parameters_json_schema.

    Note: this is currently only supported by OpenAI models.
    """

    sequential: bool = False
    """Whether this tool requires a sequential/serial execution environment."""

    kind: ToolKind = field(default='function')
    """The kind of tool:

    - `'function'`: a tool that will be executed by Pydantic AI during an agent run and has its result returned to the model
    - `'output'`: a tool that passes through an output value that ends the run
    - `'external'`: a tool whose result will be produced outside of the Pydantic AI agent run in which it was called, because it depends on an upstream service (or user) or could take longer to generate than it's reasonable to keep the agent process running.
        See the [tools documentation](../deferred-tools.md#deferred-tools) for more info.
    - `'unapproved'`: a tool that requires human-in-the-loop approval.
        See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
    """

    metadata: dict[str, Any] | None = None
    """Tool metadata that can be set by the toolset this tool came from. It is not sent to the model, but can be used for filtering and tool behavior customization.

    For MCP tools, this contains the `meta`, `annotations`, and `output_schema` fields from the tool definition.
    """

    @property
    def defer(self) -> bool:
        """Whether calls to this tool will be deferred.

        See the [tools documentation](../deferred-tools.md#deferred-tools) for more info.
        """
        return self.kind in ('external', 'unapproved')

    __repr__ = _utils.dataclasses_no_defaults_repr

```

#### name

```python
name: str

```

The name of the tool.

#### parameters_json_schema

```python
parameters_json_schema: ObjectJsonSchema = field(
    default_factory=lambda: {
        "type": "object",
        "properties": {},
    }
)

```

The JSON schema for the tool's parameters.

#### description

```python
description: str | None = None

```

The description of the tool.

#### outer_typed_dict_key

```python
outer_typed_dict_key: str | None = None

```

The key in the outer [TypedDict] that wraps an output tool.

This will only be set for output tools which don't have an `object` JSON schema.

#### strict

```python
strict: bool | None = None

```

Whether to enforce (vendor-specific) strict JSON schema validation for tool calls.

Setting this to `True` while using a supported model generally imposes some restrictions on the tool's JSON schema in exchange for guaranteeing the API responses strictly match that schema.

When `False`, the model may be free to generate other properties or types (depending on the vendor). When `None` (the default), the value will be inferred based on the compatibility of the parameters_json_schema.

Note: this is currently only supported by OpenAI models.

#### sequential

```python
sequential: bool = False

```

Whether this tool requires a sequential/serial execution environment.

#### kind

```python
kind: ToolKind = field(default='function')

```

The kind of tool:

- `'function'`: a tool that will be executed by Pydantic AI during an agent run and has its result returned to the model
- `'output'`: a tool that passes through an output value that ends the run
- `'external'`: a tool whose result will be produced outside of the Pydantic AI agent run in which it was called, because it depends on an upstream service (or user) or could take longer to generate than it's reasonable to keep the agent process running. See the [tools documentation](../../deferred-tools/#deferred-tools) for more info.
- `'unapproved'`: a tool that requires human-in-the-loop approval. See the [tools documentation](../../deferred-tools/#human-in-the-loop-tool-approval) for more info.

#### metadata

```python
metadata: dict[str, Any] | None = None

```

Tool metadata that can be set by the toolset this tool came from. It is not sent to the model, but can be used for filtering and tool behavior customization.

For MCP tools, this contains the `meta`, `annotations`, and `output_schema` fields from the tool definition.

#### defer

```python
defer: bool

```

Whether calls to this tool will be deferred.

See the [tools documentation](../../deferred-tools/#deferred-tools) for more info.

