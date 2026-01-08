<!-- Source: _complete-doc.md -->

# `pydantic_ai.toolsets`

### AbstractToolset

Bases: `ABC`, `Generic[AgentDepsT]`

A toolset is a collection of tools that can be used by an agent.

It is responsible for:

- Listing the tools it contains
- Validating the arguments of the tools
- Calling the tools

See [toolset docs](../../toolsets/) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
class AbstractToolset(ABC, Generic[AgentDepsT]):
    """A toolset is a collection of tools that can be used by an agent.

    It is responsible for:

    - Listing the tools it contains
    - Validating the arguments of the tools
    - Calling the tools

    See [toolset docs](../toolsets.md) for more information.
    """

    @property
    @abstractmethod
    def id(self) -> str | None:
        """An ID for the toolset that is unique among all toolsets registered with the same agent.

        If you're implementing a concrete implementation that users can instantiate more than once, you should let them optionally pass a custom ID to the constructor and return that here.

        A toolset needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the toolset's activities within the workflow.
        """
        raise NotImplementedError()

    @property
    def label(self) -> str:
        """The name of the toolset for use in error messages."""
        label = self.__class__.__name__
        if self.id:  # pragma: no branch
            label += f' {self.id!r}'
        return label

    @property
    def tool_name_conflict_hint(self) -> str:
        """A hint for how to avoid name conflicts with other toolsets for use in error messages."""
        return 'Rename the tool or wrap the toolset in a `PrefixedToolset` to avoid name conflicts.'

    async def __aenter__(self) -> Self:
        """Enter the toolset context.

        This is where you can set up network connections in a concrete implementation.
        """
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """Exit the toolset context.

        This is where you can tear down network connections in a concrete implementation.
        """
        return None

    @abstractmethod
    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """The tools that are available in this toolset."""
        raise NotImplementedError()

    @abstractmethod
    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Call a tool with the given arguments.

        Args:
            name: The name of the tool to call.
            tool_args: The arguments to pass to the tool.
            ctx: The run context.
            tool: The tool definition returned by [`get_tools`][pydantic_ai.toolsets.AbstractToolset.get_tools] that was called.
        """
        raise NotImplementedError()

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        """Run a visitor function on all "leaf" toolsets (i.e. those that implement their own tool listing and calling)."""
        visitor(self)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        """Run a visitor function on all "leaf" toolsets (i.e. those that implement their own tool listing and calling) and replace them in the hierarchy with the result of the function."""
        return visitor(self)

    def filtered(
        self, filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool]
    ) -> FilteredToolset[AgentDepsT]:
        """Returns a new toolset that filters this toolset's tools using a filter function that takes the agent context and the tool definition.

        See [toolset docs](../toolsets.md#filtering-tools) for more information.
        """
        from .filtered import FilteredToolset

        return FilteredToolset(self, filter_func)

    def prefixed(self, prefix: str) -> PrefixedToolset[AgentDepsT]:
        """Returns a new toolset that prefixes the names of this toolset's tools.

        See [toolset docs](../toolsets.md#prefixing-tool-names) for more information.
        """
        from .prefixed import PrefixedToolset

        return PrefixedToolset(self, prefix)

    def prepared(self, prepare_func: ToolsPrepareFunc[AgentDepsT]) -> PreparedToolset[AgentDepsT]:
        """Returns a new toolset that prepares this toolset's tools using a prepare function that takes the agent context and the original tool definitions.

        See [toolset docs](../toolsets.md#preparing-tool-definitions) for more information.
        """
        from .prepared import PreparedToolset

        return PreparedToolset(self, prepare_func)

    def renamed(self, name_map: dict[str, str]) -> RenamedToolset[AgentDepsT]:
        """Returns a new toolset that renames this toolset's tools using a dictionary mapping new names to original names.

        See [toolset docs](../toolsets.md#renaming-tools) for more information.
        """
        from .renamed import RenamedToolset

        return RenamedToolset(self, name_map)

    def approval_required(
        self,
        approval_required_func: Callable[[RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool] = (
            lambda ctx, tool_def, tool_args: True
        ),
    ) -> ApprovalRequiredToolset[AgentDepsT]:
        """Returns a new toolset that requires (some) calls to tools it contains to be approved.

        See [toolset docs](../toolsets.md#requiring-tool-approval) for more information.
        """
        from .approval_required import ApprovalRequiredToolset

        return ApprovalRequiredToolset(self, approval_required_func)

```

#### id

```python
id: str | None

```

An ID for the toolset that is unique among all toolsets registered with the same agent.

If you're implementing a concrete implementation that users can instantiate more than once, you should let them optionally pass a custom ID to the constructor and return that here.

A toolset needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the toolset's activities within the workflow.

#### label

```python
label: str

```

The name of the toolset for use in error messages.

#### tool_name_conflict_hint

```python
tool_name_conflict_hint: str

```

A hint for how to avoid name conflicts with other toolsets for use in error messages.

#### __aenter__

```python
__aenter__() -> Self

```

Enter the toolset context.

This is where you can set up network connections in a concrete implementation.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
async def __aenter__(self) -> Self:
    """Enter the toolset context.

    This is where you can set up network connections in a concrete implementation.
    """
    return self

```

#### __aexit__

```python
__aexit__(*args: Any) -> bool | None

```

Exit the toolset context.

This is where you can tear down network connections in a concrete implementation.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
async def __aexit__(self, *args: Any) -> bool | None:
    """Exit the toolset context.

    This is where you can tear down network connections in a concrete implementation.
    """
    return None

```

#### get_tools

```python
get_tools(
    ctx: RunContext[AgentDepsT],
) -> dict[str, ToolsetTool[AgentDepsT]]

```

The tools that are available in this toolset.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
@abstractmethod
async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
    """The tools that are available in this toolset."""
    raise NotImplementedError()

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

Call a tool with the given arguments.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of the tool to call. | *required* | | `tool_args` | `dict[str, Any]` | The arguments to pass to the tool. | *required* | | `ctx` | `RunContext[AgentDepsT]` | The run context. | *required* | | `tool` | `ToolsetTool[AgentDepsT]` | The tool definition returned by get_tools that was called. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
@abstractmethod
async def call_tool(
    self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
) -> Any:
    """Call a tool with the given arguments.

    Args:
        name: The name of the tool to call.
        tool_args: The arguments to pass to the tool.
        ctx: The run context.
        tool: The tool definition returned by [`get_tools`][pydantic_ai.toolsets.AbstractToolset.get_tools] that was called.
    """
    raise NotImplementedError()

```

#### apply

```python
apply(
    visitor: Callable[[AbstractToolset[AgentDepsT]], None],
) -> None

```

Run a visitor function on all "leaf" toolsets (i.e. those that implement their own tool listing and calling).

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
    """Run a visitor function on all "leaf" toolsets (i.e. those that implement their own tool listing and calling)."""
    visitor(self)

```

#### visit_and_replace

```python
visit_and_replace(
    visitor: Callable[
        [AbstractToolset[AgentDepsT]],
        AbstractToolset[AgentDepsT],
    ],
) -> AbstractToolset[AgentDepsT]

```

Run a visitor function on all "leaf" toolsets (i.e. those that implement their own tool listing and calling) and replace them in the hierarchy with the result of the function.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
def visit_and_replace(
    self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
) -> AbstractToolset[AgentDepsT]:
    """Run a visitor function on all "leaf" toolsets (i.e. those that implement their own tool listing and calling) and replace them in the hierarchy with the result of the function."""
    return visitor(self)

```

#### filtered

```python
filtered(
    filter_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition], bool
    ],
) -> FilteredToolset[AgentDepsT]

```

Returns a new toolset that filters this toolset's tools using a filter function that takes the agent context and the tool definition.

See [toolset docs](../../toolsets/#filtering-tools) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
def filtered(
    self, filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool]
) -> FilteredToolset[AgentDepsT]:
    """Returns a new toolset that filters this toolset's tools using a filter function that takes the agent context and the tool definition.

    See [toolset docs](../toolsets.md#filtering-tools) for more information.
    """
    from .filtered import FilteredToolset

    return FilteredToolset(self, filter_func)

```

#### prefixed

```python
prefixed(prefix: str) -> PrefixedToolset[AgentDepsT]

```

Returns a new toolset that prefixes the names of this toolset's tools.

See [toolset docs](../../toolsets/#prefixing-tool-names) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
def prefixed(self, prefix: str) -> PrefixedToolset[AgentDepsT]:
    """Returns a new toolset that prefixes the names of this toolset's tools.

    See [toolset docs](../toolsets.md#prefixing-tool-names) for more information.
    """
    from .prefixed import PrefixedToolset

    return PrefixedToolset(self, prefix)

```

#### prepared

```python
prepared(
    prepare_func: ToolsPrepareFunc[AgentDepsT],
) -> PreparedToolset[AgentDepsT]

```

Returns a new toolset that prepares this toolset's tools using a prepare function that takes the agent context and the original tool definitions.

See [toolset docs](../../toolsets/#preparing-tool-definitions) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
def prepared(self, prepare_func: ToolsPrepareFunc[AgentDepsT]) -> PreparedToolset[AgentDepsT]:
    """Returns a new toolset that prepares this toolset's tools using a prepare function that takes the agent context and the original tool definitions.

    See [toolset docs](../toolsets.md#preparing-tool-definitions) for more information.
    """
    from .prepared import PreparedToolset

    return PreparedToolset(self, prepare_func)

```

#### renamed

```python
renamed(
    name_map: dict[str, str],
) -> RenamedToolset[AgentDepsT]

```

Returns a new toolset that renames this toolset's tools using a dictionary mapping new names to original names.

See [toolset docs](../../toolsets/#renaming-tools) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
def renamed(self, name_map: dict[str, str]) -> RenamedToolset[AgentDepsT]:
    """Returns a new toolset that renames this toolset's tools using a dictionary mapping new names to original names.

    See [toolset docs](../toolsets.md#renaming-tools) for more information.
    """
    from .renamed import RenamedToolset

    return RenamedToolset(self, name_map)

```

#### approval_required

```python
approval_required(
    approval_required_func: Callable[
        [
            RunContext[AgentDepsT],
            ToolDefinition,
            dict[str, Any],
        ],
        bool,
    ] = lambda ctx, tool_def, tool_args: True
) -> ApprovalRequiredToolset[AgentDepsT]

```

Returns a new toolset that requires (some) calls to tools it contains to be approved.

See [toolset docs](../../toolsets/#requiring-tool-approval) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/abstract.py`

```python
def approval_required(
    self,
    approval_required_func: Callable[[RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool] = (
        lambda ctx, tool_def, tool_args: True
    ),
) -> ApprovalRequiredToolset[AgentDepsT]:
    """Returns a new toolset that requires (some) calls to tools it contains to be approved.

    See [toolset docs](../toolsets.md#requiring-tool-approval) for more information.
    """
    from .approval_required import ApprovalRequiredToolset

    return ApprovalRequiredToolset(self, approval_required_func)

```

### CombinedToolset

Bases: `AbstractToolset[AgentDepsT]`

A toolset that combines multiple toolsets.

See [toolset docs](../../toolsets/#combining-toolsets) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/combined.py`

```python
@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets.

    See [toolset docs](../toolsets.md#combining-toolsets) for more information.
    """

    toolsets: Sequence[AbstractToolset[AgentDepsT]]

    _enter_lock: Lock = field(compare=False, init=False, default_factory=Lock)
    _entered_count: int = field(init=False, default=0)
    _exit_stack: AsyncExitStack | None = field(init=False, default=None)

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({", ".join(toolset.label for toolset in self.toolsets)})'  # pragma: no cover

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._entered_count == 0:
                async with AsyncExitStack() as exit_stack:
                    for toolset in self.toolsets:
                        await exit_stack.enter_async_context(toolset)
                    self._exit_stack = exit_stack.pop_all()
            self._entered_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._entered_count -= 1
            if self._entered_count == 0 and self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        toolsets_tools = await asyncio.gather(*(toolset.get_tools(ctx) for toolset in self.toolsets))
        all_tools: dict[str, ToolsetTool[AgentDepsT]] = {}

        for toolset, tools in zip(self.toolsets, toolsets_tools):
            for name, tool in tools.items():
                tool_toolset = tool.toolset
                if existing_tool := all_tools.get(name):
                    capitalized_toolset_label = tool_toolset.label[0].upper() + tool_toolset.label[1:]
                    raise UserError(
                        f'{capitalized_toolset_label} defines a tool whose name conflicts with existing tool from {existing_tool.toolset.label}: {name!r}. {toolset.tool_name_conflict_hint}'
                    )

                all_tools[name] = _CombinedToolsetTool(
                    toolset=tool_toolset,
                    tool_def=tool.tool_def,
                    max_retries=tool.max_retries,
                    args_validator=tool.args_validator,
                    source_toolset=toolset,
                    source_tool=tool,
                )
        return all_tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert isinstance(tool, _CombinedToolsetTool)
        return await tool.source_toolset.call_tool(name, tool_args, ctx, tool.source_tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        for toolset in self.toolsets:
            toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return replace(self, toolsets=[toolset.visit_and_replace(visitor) for toolset in self.toolsets])

```

### ExternalToolset

Bases: `AbstractToolset[AgentDepsT]`

A toolset that holds tools whose results will be produced outside of the Pydantic AI agent run in which they were called.

See [toolset docs](../../toolsets/#external-toolset) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/external.py`

```python
class ExternalToolset(AbstractToolset[AgentDepsT]):
    """A toolset that holds tools whose results will be produced outside of the Pydantic AI agent run in which they were called.

    See [toolset docs](../toolsets.md#external-toolset) for more information.
    """

    tool_defs: list[ToolDefinition]
    _id: str | None

    def __init__(self, tool_defs: list[ToolDefinition], *, id: str | None = None):
        self.tool_defs = tool_defs
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=replace(tool_def, kind='external'),
                max_retries=0,
                args_validator=TOOL_SCHEMA_VALIDATOR,
            )
            for tool_def in self.tool_defs
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        raise NotImplementedError('External tools cannot be called directly')

```

### ApprovalRequiredToolset

Bases: `WrapperToolset[AgentDepsT]`

A toolset that requires (some) calls to tools it contains to be approved.

See [toolset docs](../../toolsets/#requiring-tool-approval) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/approval_required.py`

```python
@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    """A toolset that requires (some) calls to tools it contains to be approved.

    See [toolset docs](../toolsets.md#requiring-tool-approval) for more information.
    """

    approval_required_func: Callable[[RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool] = (
        lambda ctx, tool_def, tool_args: True
    )

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not ctx.tool_call_approved and self.approval_required_func(ctx, tool.tool_def, tool_args):
            raise ApprovalRequired

        return await super().call_tool(name, tool_args, ctx, tool)

```

### FilteredToolset

Bases: `WrapperToolset[AgentDepsT]`

A toolset that filters the tools it contains using a filter function that takes the agent context and the tool definition.

See [toolset docs](../../toolsets/#filtering-tools) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/filtered.py`

```python
@dataclass
class FilteredToolset(WrapperToolset[AgentDepsT]):
    """A toolset that filters the tools it contains using a filter function that takes the agent context and the tool definition.

    See [toolset docs](../toolsets.md#filtering-tools) for more information.
    """

    filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            name: tool for name, tool in (await super().get_tools(ctx)).items() if self.filter_func(ctx, tool.tool_def)
        }

```

### FunctionToolset

Bases: `AbstractToolset[AgentDepsT]`

A toolset that lets Python functions be used as tools.

See [toolset docs](../../toolsets/#function-toolset) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/function.py`

````python
class FunctionToolset(AbstractToolset[AgentDepsT]):
    """A toolset that lets Python functions be used as tools.

    See [toolset docs](../toolsets.md#function-toolset) for more information.
    """

    tools: dict[str, Tool[Any]]
    max_retries: int
    _id: str | None
    docstring_format: DocstringFormat
    require_parameter_descriptions: bool
    schema_generator: type[GenerateJsonSchema]

    def __init__(
        self,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = [],
        *,
        max_retries: int = 1,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
    ):
        """Build a new function toolset.

        Args:
            tools: The tools to add to the toolset.
            max_retries: The maximum number of retries for each tool during a run.
                Applies to all tools, unless overridden when adding a tool.
            docstring_format: Format of tool docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
                Applies to all tools, unless overridden when adding a tool.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
                Applies to all tools, unless overridden when adding a tool.
            schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
                Applies to all tools, unless overridden when adding a tool.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
                Applies to all tools, unless overridden when adding a tool.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
                Applies to all tools, unless overridden when adding a tool.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
                Applies to all tools, unless overridden when adding a tool, which will be merged with the toolset's metadata.
            id: An optional unique ID for the toolset. A toolset needs to have an ID in order to be used in a durable execution environment like Temporal,
                in which case the ID will be used to identify the toolset's activities within the workflow.
        """
        self.max_retries = max_retries
        self._id = id
        self.docstring_format = docstring_format
        self.require_parameter_descriptions = require_parameter_descriptions
        self.schema_generator = schema_generator
        self.strict = strict
        self.sequential = sequential
        self.requires_approval = requires_approval
        self.metadata = metadata

        self.tools = {}
        for tool in tools:
            if isinstance(tool, Tool):
                self.add_tool(tool)
            else:
                self.add_function(tool)

    @property
    def id(self) -> str | None:
        return self._id

    @overload
    def tool(self, func: ToolFuncEither[AgentDepsT, ToolParams], /) -> ToolFuncEither[AgentDepsT, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat | None = None,
        require_parameter_descriptions: bool | None = None,
        schema_generator: type[GenerateJsonSchema] | None = None,
        strict: bool | None = None,
        sequential: bool | None = None,
        requires_approval: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[ToolFuncEither[AgentDepsT, ToolParams]], ToolFuncEither[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncEither[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat | None = None,
        require_parameter_descriptions: bool | None = None,
        schema_generator: type[GenerateJsonSchema] | None = None,
        strict: bool | None = None,
        sequential: bool | None = None,
        requires_approval: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@toolset.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, FunctionToolset, RunContext

        toolset = FunctionToolset()

        @toolset.tool
        def foobar(ctx: RunContext[int], x: int) -> int:
            return ctx.deps + x

        @toolset.tool(retries=2)
        async def spam(ctx: RunContext[str], y: float) -> float:
            return ctx.deps + y

        agent = Agent('test', toolsets=[toolset], deps_type=int)
        result = agent.run_sync('foobar', deps=1)
        print(result.output)
        #> {"foobar":1,"spam":1.0}
        ```

        Args:
            func: The tool function to register.
            name: The name of the tool, defaults to the function name.
            description: The description of the tool,defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                If `None`, the default value is determined by the toolset.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing.
                If `None`, the default value is determined by the toolset.
            schema_generator: The JSON schema generator class to use for this tool.
                If `None`, the default value is determined by the toolset.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
                If `None`, the default value is determined by the toolset.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
                If `None`, the default value is determined by the toolset.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
                If `None`, the default value is determined by the toolset.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
                If `None`, the default value is determined by the toolset. If provided, it will be merged with the toolset's metadata.
        """

        def tool_decorator(
            func_: ToolFuncEither[AgentDepsT, ToolParams],
        ) -> ToolFuncEither[AgentDepsT, ToolParams]:
            # noinspection PyTypeChecker
            self.add_function(
                func=func_,
                takes_ctx=None,
                name=name,
                description=description,
                retries=retries,
                prepare=prepare,
                docstring_format=docstring_format,
                require_parameter_descriptions=require_parameter_descriptions,
                schema_generator=schema_generator,
                strict=strict,
                sequential=sequential,
                requires_approval=requires_approval,
                metadata=metadata,
            )
            return func_

        return tool_decorator if func is None else tool_decorator(func)

    def add_function(
        self,
        func: ToolFuncEither[AgentDepsT, ToolParams],
        takes_ctx: bool | None = None,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat | None = None,
        require_parameter_descriptions: bool | None = None,
        schema_generator: type[GenerateJsonSchema] | None = None,
        strict: bool | None = None,
        sequential: bool | None = None,
        requires_approval: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a function as a tool to the toolset.

        Can take a sync or async function.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        Args:
            func: The tool function to register.
            takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] as its first argument. If `None`, this is inferred from the function signature.
            name: The name of the tool, defaults to the function name.
            description: The description of the tool, defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                If `None`, the default value is determined by the toolset.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing.
                If `None`, the default value is determined by the toolset.
            schema_generator: The JSON schema generator class to use for this tool.
                If `None`, the default value is determined by the toolset.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
                If `None`, the default value is determined by the toolset.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
                If `None`, the default value is determined by the toolset.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
                If `None`, the default value is determined by the toolset.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
                If `None`, the default value is determined by the toolset. If provided, it will be merged with the toolset's metadata.
        """
        if docstring_format is None:
            docstring_format = self.docstring_format
        if require_parameter_descriptions is None:
            require_parameter_descriptions = self.require_parameter_descriptions
        if schema_generator is None:
            schema_generator = self.schema_generator
        if strict is None:
            strict = self.strict
        if sequential is None:
            sequential = self.sequential
        if requires_approval is None:
            requires_approval = self.requires_approval

        tool = Tool[AgentDepsT](
            func,
            takes_ctx=takes_ctx,
            name=name,
            description=description,
            max_retries=retries,
            prepare=prepare,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            schema_generator=schema_generator,
            strict=strict,
            sequential=sequential,
            requires_approval=requires_approval,
            metadata=metadata,
        )
        self.add_tool(tool)

    def add_tool(self, tool: Tool[AgentDepsT]) -> None:
        """Add a tool to the toolset.

        Args:
            tool: The tool to add.
        """
        if tool.name in self.tools:
            raise UserError(f'Tool name conflicts with existing tool: {tool.name!r}')
        if tool.max_retries is None:
            tool.max_retries = self.max_retries
        if self.metadata is not None:
            tool.metadata = self.metadata | (tool.metadata or {})
        self.tools[tool.name] = tool

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        for original_name, tool in self.tools.items():
            max_retries = tool.max_retries if tool.max_retries is not None else self.max_retries
            run_context = replace(
                ctx,
                tool_name=original_name,
                retry=ctx.retries.get(original_name, 0),
                max_retries=max_retries,
            )
            tool_def = await tool.prepare_tool_def(run_context)
            if not tool_def:
                continue

            new_name = tool_def.name
            if new_name in tools:
                if new_name != original_name:
                    raise UserError(f'Renaming tool {original_name!r} to {new_name!r} conflicts with existing tool.')
                else:
                    raise UserError(f'Tool name conflicts with previously renamed tool: {new_name!r}.')

            tools[new_name] = FunctionToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=max_retries,
                args_validator=tool.function_schema.validator,
                call_func=tool.function_schema.call,
                is_async=tool.function_schema.is_async,
            )
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert isinstance(tool, FunctionToolsetTool)
        return await tool.call_func(tool_args, ctx)

````

#### __init__

```python
__init__(
    tools: Sequence[
        Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]
    ] = [],
    *,
    max_retries: int = 1,
    docstring_format: DocstringFormat = "auto",
    require_parameter_descriptions: bool = False,
    schema_generator: type[
        GenerateJsonSchema
    ] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
    id: str | None = None
)

```

Build a new function toolset.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tools` | `Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]]` | The tools to add to the toolset. | `[]` | | `max_retries` | `int` | The maximum number of retries for each tool during a run. Applies to all tools, unless overridden when adding a tool. | `1` | | `docstring_format` | `DocstringFormat` | Format of tool docstring, see DocstringFormat. Defaults to 'auto', such that the format is inferred from the structure of the docstring. Applies to all tools, unless overridden when adding a tool. | `'auto'` | | `require_parameter_descriptions` | `bool` | If True, raise an error if a parameter description is missing. Defaults to False. Applies to all tools, unless overridden when adding a tool. | `False` | | `schema_generator` | `type[GenerateJsonSchema]` | The JSON schema generator class to use for this tool. Defaults to GenerateToolJsonSchema. Applies to all tools, unless overridden when adding a tool. | `GenerateToolJsonSchema` | | `strict` | `bool | None` | Whether to enforce JSON schema compliance (only affects OpenAI). See ToolDefinition for more info. | `None` | | `sequential` | `bool` | Whether the function requires a sequential/serial execution environment. Defaults to False. Applies to all tools, unless overridden when adding a tool. | `False` | | `requires_approval` | `bool` | Whether this tool requires human-in-the-loop approval. Defaults to False. See the tools documentation for more info. Applies to all tools, unless overridden when adding a tool. | `False` | | `metadata` | `dict[str, Any] | None` | Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization. Applies to all tools, unless overridden when adding a tool, which will be merged with the toolset's metadata. | `None` | | `id` | `str | None` | An optional unique ID for the toolset. A toolset needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the toolset's activities within the workflow. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/function.py`

```python
def __init__(
    self,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = [],
    *,
    max_retries: int = 1,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
    id: str | None = None,
):
    """Build a new function toolset.

    Args:
        tools: The tools to add to the toolset.
        max_retries: The maximum number of retries for each tool during a run.
            Applies to all tools, unless overridden when adding a tool.
        docstring_format: Format of tool docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
            Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            Applies to all tools, unless overridden when adding a tool.
        require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            Applies to all tools, unless overridden when adding a tool.
        schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
            Applies to all tools, unless overridden when adding a tool.
        strict: Whether to enforce JSON schema compliance (only affects OpenAI).
            See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
        sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
            Applies to all tools, unless overridden when adding a tool.
        requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
            See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            Applies to all tools, unless overridden when adding a tool.
        metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
            Applies to all tools, unless overridden when adding a tool, which will be merged with the toolset's metadata.
        id: An optional unique ID for the toolset. A toolset needs to have an ID in order to be used in a durable execution environment like Temporal,
            in which case the ID will be used to identify the toolset's activities within the workflow.
    """
    self.max_retries = max_retries
    self._id = id
    self.docstring_format = docstring_format
    self.require_parameter_descriptions = require_parameter_descriptions
    self.schema_generator = schema_generator
    self.strict = strict
    self.sequential = sequential
    self.requires_approval = requires_approval
    self.metadata = metadata

    self.tools = {}
    for tool in tools:
        if isinstance(tool, Tool):
            self.add_tool(tool)
        else:
            self.add_function(tool)

```

#### tool

```python
tool(
    func: ToolFuncEither[AgentDepsT, ToolParams],
) -> ToolFuncEither[AgentDepsT, ToolParams]

```

```python
tool(
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat | None = None,
    require_parameter_descriptions: bool | None = None,
    schema_generator: (
        type[GenerateJsonSchema] | None
    ) = None,
    strict: bool | None = None,
    sequential: bool | None = None,
    requires_approval: bool | None = None,
    metadata: dict[str, Any] | None = None
) -> Callable[
    [ToolFuncEither[AgentDepsT, ToolParams]],
    ToolFuncEither[AgentDepsT, ToolParams],
]

```

```python
tool(
    func: (
        ToolFuncEither[AgentDepsT, ToolParams] | None
    ) = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat | None = None,
    require_parameter_descriptions: bool | None = None,
    schema_generator: (
        type[GenerateJsonSchema] | None
    ) = None,
    strict: bool | None = None,
    sequential: bool | None = None,
    requires_approval: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any

```

Decorator to register a tool function which takes RunContext as its first argument.

Can decorate a sync or async functions.

The docstring is inspected to extract both the tool description and description of each parameter, [learn more](../../tools/#function-tools-and-schema).

We can't add overloads for every possible signature of tool, since the return type is a recursive union so the signature of functions decorated with `@toolset.tool` is obscured.

Example:

```python
from pydantic_ai import Agent, FunctionToolset, RunContext

toolset = FunctionToolset()

@toolset.tool
def foobar(ctx: RunContext[int], x: int) -> int:
    return ctx.deps + x

@toolset.tool(retries=2)
async def spam(ctx: RunContext[str], y: float) -> float:
    return ctx.deps + y

agent = Agent('test', toolsets=[toolset], deps_type=int)
result = agent.run_sync('foobar', deps=1)
print(result.output)
#> {"foobar":1,"spam":1.0}

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `func` | `ToolFuncEither[AgentDepsT, ToolParams] | None` | The tool function to register. | `None` | | `name` | `str | None` | The name of the tool, defaults to the function name. | `None` | | `description` | `str | None` | The description of the tool,defaults to the function docstring. | `None` | | `retries` | `int | None` | The number of retries to allow for this tool, defaults to the agent's default retries, which defaults to 1. | `None` | | `prepare` | `ToolPrepareFunc[AgentDepsT] | None` | custom method to prepare the tool definition for each step, return None to omit this tool from a given step. This is useful if you want to customise a tool at call time, or omit it completely from a step. See ToolPrepareFunc. | `None` | | `docstring_format` | `DocstringFormat | None` | The format of the docstring, see DocstringFormat. If None, the default value is determined by the toolset. | `None` | | `require_parameter_descriptions` | `bool | None` | If True, raise an error if a parameter description is missing. If None, the default value is determined by the toolset. | `None` | | `schema_generator` | `type[GenerateJsonSchema] | None` | The JSON schema generator class to use for this tool. If None, the default value is determined by the toolset. | `None` | | `strict` | `bool | None` | Whether to enforce JSON schema compliance (only affects OpenAI). See ToolDefinition for more info. If None, the default value is determined by the toolset. | `None` | | `sequential` | `bool | None` | Whether the function requires a sequential/serial execution environment. Defaults to False. If None, the default value is determined by the toolset. | `None` | | `requires_approval` | `bool | None` | Whether this tool requires human-in-the-loop approval. Defaults to False. See the tools documentation for more info. If None, the default value is determined by the toolset. | `None` | | `metadata` | `dict[str, Any] | None` | Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization. If None, the default value is determined by the toolset. If provided, it will be merged with the toolset's metadata. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/function.py`

````python
def tool(
    self,
    func: ToolFuncEither[AgentDepsT, ToolParams] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat | None = None,
    require_parameter_descriptions: bool | None = None,
    schema_generator: type[GenerateJsonSchema] | None = None,
    strict: bool | None = None,
    sequential: bool | None = None,
    requires_approval: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

    Can decorate a sync or async functions.

    The docstring is inspected to extract both the tool description and description of each parameter,
    [learn more](../tools.md#function-tools-and-schema).

    We can't add overloads for every possible signature of tool, since the return type is a recursive union
    so the signature of functions decorated with `@toolset.tool` is obscured.

    Example:
    ```python
    from pydantic_ai import Agent, FunctionToolset, RunContext

    toolset = FunctionToolset()

    @toolset.tool
    def foobar(ctx: RunContext[int], x: int) -> int:
        return ctx.deps + x

    @toolset.tool(retries=2)
    async def spam(ctx: RunContext[str], y: float) -> float:
        return ctx.deps + y

    agent = Agent('test', toolsets=[toolset], deps_type=int)
    result = agent.run_sync('foobar', deps=1)
    print(result.output)
    #> {"foobar":1,"spam":1.0}
    ```

    Args:
        func: The tool function to register.
        name: The name of the tool, defaults to the function name.
        description: The description of the tool,defaults to the function docstring.
        retries: The number of retries to allow for this tool, defaults to the agent's default retries,
            which defaults to 1.
        prepare: custom method to prepare the tool definition for each step, return `None` to omit this
            tool from a given step. This is useful if you want to customise a tool at call time,
            or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
        docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
            If `None`, the default value is determined by the toolset.
        require_parameter_descriptions: If True, raise an error if a parameter description is missing.
            If `None`, the default value is determined by the toolset.
        schema_generator: The JSON schema generator class to use for this tool.
            If `None`, the default value is determined by the toolset.
        strict: Whether to enforce JSON schema compliance (only affects OpenAI).
            See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            If `None`, the default value is determined by the toolset.
        sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
            If `None`, the default value is determined by the toolset.
        requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
            See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            If `None`, the default value is determined by the toolset.
        metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
            If `None`, the default value is determined by the toolset. If provided, it will be merged with the toolset's metadata.
    """

    def tool_decorator(
        func_: ToolFuncEither[AgentDepsT, ToolParams],
    ) -> ToolFuncEither[AgentDepsT, ToolParams]:
        # noinspection PyTypeChecker
        self.add_function(
            func=func_,
            takes_ctx=None,
            name=name,
            description=description,
            retries=retries,
            prepare=prepare,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            schema_generator=schema_generator,
            strict=strict,
            sequential=sequential,
            requires_approval=requires_approval,
            metadata=metadata,
        )
        return func_

    return tool_decorator if func is None else tool_decorator(func)

````

#### add_function

```python
add_function(
    func: ToolFuncEither[AgentDepsT, ToolParams],
    takes_ctx: bool | None = None,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat | None = None,
    require_parameter_descriptions: bool | None = None,
    schema_generator: (
        type[GenerateJsonSchema] | None
    ) = None,
    strict: bool | None = None,
    sequential: bool | None = None,
    requires_approval: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> None

```

Add a function as a tool to the toolset.

Can take a sync or async function.

The docstring is inspected to extract both the tool description and description of each parameter, [learn more](../../tools/#function-tools-and-schema).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `func` | `ToolFuncEither[AgentDepsT, ToolParams]` | The tool function to register. | *required* | | `takes_ctx` | `bool | None` | Whether the function takes a RunContext as its first argument. If None, this is inferred from the function signature. | `None` | | `name` | `str | None` | The name of the tool, defaults to the function name. | `None` | | `description` | `str | None` | The description of the tool, defaults to the function docstring. | `None` | | `retries` | `int | None` | The number of retries to allow for this tool, defaults to the agent's default retries, which defaults to 1. | `None` | | `prepare` | `ToolPrepareFunc[AgentDepsT] | None` | custom method to prepare the tool definition for each step, return None to omit this tool from a given step. This is useful if you want to customise a tool at call time, or omit it completely from a step. See ToolPrepareFunc. | `None` | | `docstring_format` | `DocstringFormat | None` | The format of the docstring, see DocstringFormat. If None, the default value is determined by the toolset. | `None` | | `require_parameter_descriptions` | `bool | None` | If True, raise an error if a parameter description is missing. If None, the default value is determined by the toolset. | `None` | | `schema_generator` | `type[GenerateJsonSchema] | None` | The JSON schema generator class to use for this tool. If None, the default value is determined by the toolset. | `None` | | `strict` | `bool | None` | Whether to enforce JSON schema compliance (only affects OpenAI). See ToolDefinition for more info. If None, the default value is determined by the toolset. | `None` | | `sequential` | `bool | None` | Whether the function requires a sequential/serial execution environment. Defaults to False. If None, the default value is determined by the toolset. | `None` | | `requires_approval` | `bool | None` | Whether this tool requires human-in-the-loop approval. Defaults to False. See the tools documentation for more info. If None, the default value is determined by the toolset. | `None` | | `metadata` | `dict[str, Any] | None` | Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization. If None, the default value is determined by the toolset. If provided, it will be merged with the toolset's metadata. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/function.py`

```python
def add_function(
    self,
    func: ToolFuncEither[AgentDepsT, ToolParams],
    takes_ctx: bool | None = None,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat | None = None,
    require_parameter_descriptions: bool | None = None,
    schema_generator: type[GenerateJsonSchema] | None = None,
    strict: bool | None = None,
    sequential: bool | None = None,
    requires_approval: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Add a function as a tool to the toolset.

    Can take a sync or async function.

    The docstring is inspected to extract both the tool description and description of each parameter,
    [learn more](../tools.md#function-tools-and-schema).

    Args:
        func: The tool function to register.
        takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] as its first argument. If `None`, this is inferred from the function signature.
        name: The name of the tool, defaults to the function name.
        description: The description of the tool, defaults to the function docstring.
        retries: The number of retries to allow for this tool, defaults to the agent's default retries,
            which defaults to 1.
        prepare: custom method to prepare the tool definition for each step, return `None` to omit this
            tool from a given step. This is useful if you want to customise a tool at call time,
            or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
        docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
            If `None`, the default value is determined by the toolset.
        require_parameter_descriptions: If True, raise an error if a parameter description is missing.
            If `None`, the default value is determined by the toolset.
        schema_generator: The JSON schema generator class to use for this tool.
            If `None`, the default value is determined by the toolset.
        strict: Whether to enforce JSON schema compliance (only affects OpenAI).
            See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            If `None`, the default value is determined by the toolset.
        sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
            If `None`, the default value is determined by the toolset.
        requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
            See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            If `None`, the default value is determined by the toolset.
        metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
            If `None`, the default value is determined by the toolset. If provided, it will be merged with the toolset's metadata.
    """
    if docstring_format is None:
        docstring_format = self.docstring_format
    if require_parameter_descriptions is None:
        require_parameter_descriptions = self.require_parameter_descriptions
    if schema_generator is None:
        schema_generator = self.schema_generator
    if strict is None:
        strict = self.strict
    if sequential is None:
        sequential = self.sequential
    if requires_approval is None:
        requires_approval = self.requires_approval

    tool = Tool[AgentDepsT](
        func,
        takes_ctx=takes_ctx,
        name=name,
        description=description,
        max_retries=retries,
        prepare=prepare,
        docstring_format=docstring_format,
        require_parameter_descriptions=require_parameter_descriptions,
        schema_generator=schema_generator,
        strict=strict,
        sequential=sequential,
        requires_approval=requires_approval,
        metadata=metadata,
    )
    self.add_tool(tool)

```

#### add_tool

```python
add_tool(tool: Tool[AgentDepsT]) -> None

```

Add a tool to the toolset.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tool` | `Tool[AgentDepsT]` | The tool to add. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/function.py`

```python
def add_tool(self, tool: Tool[AgentDepsT]) -> None:
    """Add a tool to the toolset.

    Args:
        tool: The tool to add.
    """
    if tool.name in self.tools:
        raise UserError(f'Tool name conflicts with existing tool: {tool.name!r}')
    if tool.max_retries is None:
        tool.max_retries = self.max_retries
    if self.metadata is not None:
        tool.metadata = self.metadata | (tool.metadata or {})
    self.tools[tool.name] = tool

```

### PrefixedToolset

Bases: `WrapperToolset[AgentDepsT]`

A toolset that prefixes the names of the tools it contains.

See [toolset docs](../../toolsets/#prefixing-tool-names) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/prefixed.py`

```python
@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains.

    See [toolset docs](../toolsets.md#prefixing-tool-names) for more information.
    """

    prefix: str

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Change the `prefix` attribute to avoid name conflicts.'

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            new_name: replace(
                tool,
                toolset=self,
                tool_def=replace(tool.tool_def, name=new_name),
            )
            for name, tool in (await super().get_tools(ctx)).items()
            if (new_name := f'{self.prefix}_{name}')
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        original_name = name.removeprefix(self.prefix + '_')
        ctx = replace(ctx, tool_name=original_name)
        tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
        return await super().call_tool(original_name, tool_args, ctx, tool)

```

### RenamedToolset

Bases: `WrapperToolset[AgentDepsT]`

A toolset that renames the tools it contains using a dictionary mapping new names to original names.

See [toolset docs](../../toolsets/#renaming-tools) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/renamed.py`

```python
@dataclass
class RenamedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that renames the tools it contains using a dictionary mapping new names to original names.

    See [toolset docs](../toolsets.md#renaming-tools) for more information.
    """

    name_map: dict[str, str]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_to_new_name_map = {v: k for k, v in self.name_map.items()}
        original_tools = await super().get_tools(ctx)
        tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        for original_name, tool in original_tools.items():
            new_name = original_to_new_name_map.get(original_name, None)
            if new_name:
                tools[new_name] = replace(
                    tool,
                    toolset=self,
                    tool_def=replace(tool.tool_def, name=new_name),
                )
            else:
                tools[original_name] = tool
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        original_name = self.name_map.get(name, name)
        ctx = replace(ctx, tool_name=original_name)
        tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
        return await super().call_tool(original_name, tool_args, ctx, tool)

```

### PreparedToolset

Bases: `WrapperToolset[AgentDepsT]`

A toolset that prepares the tools it contains using a prepare function that takes the agent context and the original tool definitions.

See [toolset docs](../../toolsets/#preparing-tool-definitions) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/prepared.py`

```python
@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a prepare function that takes the agent context and the original tool definitions.

    See [toolset docs](../toolsets.md#preparing-tool-definitions) for more information.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        original_tools = await super().get_tools(ctx)
        original_tool_defs = [tool.tool_def for tool in original_tools.values()]
        prepared_tool_defs_by_name = {
            tool_def.name: tool_def for tool_def in (await self.prepare_func(ctx, original_tool_defs) or [])
        }

        if len(prepared_tool_defs_by_name.keys() - original_tools.keys()) > 0:
            raise UserError(
                'Prepare function cannot add or rename tools. Use `FunctionToolset.add_function()` or `RenamedToolset` instead.'
            )

        return {
            name: replace(original_tools[name], tool_def=tool_def)
            for name, tool_def in prepared_tool_defs_by_name.items()
        }

```

### WrapperToolset

Bases: `AbstractToolset[AgentDepsT]`

A toolset that wraps another toolset and delegates to it.

See [toolset docs](../../toolsets/#wrapping-a-toolset) for more information.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/wrapper.py`

```python
@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    """A toolset that wraps another toolset and delegates to it.

    See [toolset docs](../toolsets.md#wrapping-a-toolset) for more information.
    """

    wrapped: AbstractToolset[AgentDepsT]

    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({self.wrapped.label})'

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return await self.wrapped.get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return replace(self, wrapped=self.wrapped.visit_and_replace(visitor))

```

### ToolsetFunc

```python
ToolsetFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    AbstractToolset[AgentDepsT]
    | None
    | Awaitable[AbstractToolset[AgentDepsT] | None],
]

```

A sync/async function which takes a run context and returns a toolset.

### FastMCPToolset

Bases: `AbstractToolset[AgentDepsT]`

A FastMCP Toolset that uses the FastMCP Client to call tools from a local or remote MCP Server.

The Toolset can accept a FastMCP Client, a FastMCP Transport, or any other object which a FastMCP Transport can be created from.

See https://gofastmcp.com/clients/transports for a full list of transports available.

Source code in `pydantic_ai_slim/pydantic_ai/toolsets/fastmcp.py`

```python
@dataclass(init=False)
class FastMCPToolset(AbstractToolset[AgentDepsT]):
    """A FastMCP Toolset that uses the FastMCP Client to call tools from a local or remote MCP Server.

    The Toolset can accept a FastMCP Client, a FastMCP Transport, or any other object which a FastMCP Transport can be created from.

    See https://gofastmcp.com/clients/transports for a full list of transports available.
    """

    client: Client[Any]
    """The FastMCP client to use."""

    _: KW_ONLY

    tool_error_behavior: Literal['model_retry', 'error']
    """The behavior to take when a tool error occurs."""

    max_retries: int
    """The maximum number of retries to attempt if a tool call fails."""

    _id: str | None

    def __init__(
        self,
        client: Client[Any]
        | ClientTransport
        | FastMCP
        | FastMCP1Server
        | AnyUrl
        | Path
        | MCPConfig
        | dict[str, Any]
        | str,
        *,
        max_retries: int = 1,
        tool_error_behavior: Literal['model_retry', 'error'] = 'model_retry',
        id: str | None = None,
    ) -> None:
        if isinstance(client, Client):
            self.client = client
        else:
            self.client = Client[Any](transport=client)

        self._id = id
        self.max_retries = max_retries
        self.tool_error_behavior = tool_error_behavior

        self._enter_lock: Lock = Lock()
        self._running_count: int = 0
        self._exit_stack: AsyncExitStack | None = None

    @property
    def id(self) -> str | None:
        return self._id

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._running_count == 0:
                self._exit_stack = AsyncExitStack()
                await self._exit_stack.enter_async_context(self.client)

            self._running_count += 1

        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._running_count -= 1
            if self._running_count == 0 and self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None

        return None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async with self:
            return {
                mcp_tool.name: self.tool_for_tool_def(
                    ToolDefinition(
                        name=mcp_tool.name,
                        description=mcp_tool.description,
                        parameters_json_schema=mcp_tool.inputSchema,
                        metadata={
                            'meta': mcp_tool.meta,
                            'annotations': mcp_tool.annotations.model_dump() if mcp_tool.annotations else None,
                            'output_schema': mcp_tool.outputSchema or None,
                        },
                    )
                )
                for mcp_tool in await self.client.list_tools()
            }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        async with self:
            try:
                call_tool_result: CallToolResult = await self.client.call_tool(name=name, arguments=tool_args)
            except ToolError as e:
                if self.tool_error_behavior == 'model_retry':
                    raise ModelRetry(message=str(e)) from e
                else:
                    raise e

        # If we have structured content, return that
        if call_tool_result.structured_content:
            return call_tool_result.structured_content

        # Otherwise, return the content
        return _map_fastmcp_tool_results(parts=call_tool_result.content)

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return ToolsetTool[AgentDepsT](
            tool_def=tool_def,
            toolset=self,
            max_retries=self.max_retries,
            args_validator=TOOL_SCHEMA_VALIDATOR,
        )

```

#### client

```python
client: Client[Any]

```

The FastMCP client to use.

#### max_retries

```python
max_retries: int = max_retries

```

The maximum number of retries to attempt if a tool call fails.

#### tool_error_behavior

```python
tool_error_behavior: Literal["model_retry", "error"] = (
    tool_error_behavior
)

```

The behavior to take when a tool error occurs.

# `pydantic_ai.usage`

### RequestUsage

Bases: `UsageBase`

LLM usage associated with a single request.

This is an implementation of `genai_prices.types.AbstractUsage` so it can be used to calculate the price of the request using [genai-prices](https://github.com/pydantic/genai-prices).

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
@dataclass(repr=False, kw_only=True)
class RequestUsage(UsageBase):
    """LLM usage associated with a single request.

    This is an implementation of `genai_prices.types.AbstractUsage` so it can be used to calculate the price of the
    request using [genai-prices](https://github.com/pydantic/genai-prices).
    """

    @property
    def requests(self):
        return 1

    def incr(self, incr_usage: RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        return _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: RequestUsage) -> RequestUsage:
        """Add two RequestUsages together.

        This is provided so it's trivial to sum usage information from multiple parts of a response.

        **WARNING:** this CANNOT be used to sum multiple requests without breaking some pricing calculations.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage

    @classmethod
    def extract(
        cls,
        data: Any,
        *,
        provider: str,
        provider_url: str,
        provider_fallback: str,
        api_flavor: str = 'default',
        details: dict[str, Any] | None = None,
    ) -> RequestUsage:
        """Extract usage information from the response data using genai-prices.

        Args:
            data: The response data from the model API.
            provider: The actual provider ID
            provider_url: The provider base_url
            provider_fallback: The fallback provider ID to use if the actual provider is not found in genai-prices.
                For example, an OpenAI model should set this to "openai" in case it has an obscure provider ID.
            api_flavor: The API flavor to use when extracting usage information,
                e.g. 'chat' or 'responses' for OpenAI.
            details: Becomes the `details` field on the returned `RequestUsage` for convenience.
        """
        details = details or {}
        for provider_id, provider_api_url in [(None, provider_url), (provider, None), (provider_fallback, None)]:
            try:
                provider_obj = get_snapshot().find_provider(None, provider_id, provider_api_url)
                _model_ref, extracted_usage = provider_obj.extract_usage(data, api_flavor=api_flavor)
                return cls(**{k: v for k, v in extracted_usage.__dict__.items() if v is not None}, details=details)
            except Exception:
                pass
        return cls(details=details)

```

#### incr

```python
incr(incr_usage: RequestUsage) -> None

```

Increment the usage in place.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `incr_usage` | `RequestUsage` | The usage to increment by. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
def incr(self, incr_usage: RequestUsage) -> None:
    """Increment the usage in place.

    Args:
        incr_usage: The usage to increment by.
    """
    return _incr_usage_tokens(self, incr_usage)

```

#### __add__

```python
__add__(other: RequestUsage) -> RequestUsage

```

Add two RequestUsages together.

This is provided so it's trivial to sum usage information from multiple parts of a response.

**WARNING:** this CANNOT be used to sum multiple requests without breaking some pricing calculations.

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
def __add__(self, other: RequestUsage) -> RequestUsage:
    """Add two RequestUsages together.

    This is provided so it's trivial to sum usage information from multiple parts of a response.

    **WARNING:** this CANNOT be used to sum multiple requests without breaking some pricing calculations.
    """
    new_usage = copy(self)
    new_usage.incr(other)
    return new_usage

```

#### extract

```python
extract(
    data: Any,
    *,
    provider: str,
    provider_url: str,
    provider_fallback: str,
    api_flavor: str = "default",
    details: dict[str, Any] | None = None
) -> RequestUsage

```

Extract usage information from the response data using genai-prices.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `data` | `Any` | The response data from the model API. | *required* | | `provider` | `str` | The actual provider ID | *required* | | `provider_url` | `str` | The provider base_url | *required* | | `provider_fallback` | `str` | The fallback provider ID to use if the actual provider is not found in genai-prices. For example, an OpenAI model should set this to "openai" in case it has an obscure provider ID. | *required* | | `api_flavor` | `str` | The API flavor to use when extracting usage information, e.g. 'chat' or 'responses' for OpenAI. | `'default'` | | `details` | `dict[str, Any] | None` | Becomes the details field on the returned RequestUsage for convenience. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
@classmethod
def extract(
    cls,
    data: Any,
    *,
    provider: str,
    provider_url: str,
    provider_fallback: str,
    api_flavor: str = 'default',
    details: dict[str, Any] | None = None,
) -> RequestUsage:
    """Extract usage information from the response data using genai-prices.

    Args:
        data: The response data from the model API.
        provider: The actual provider ID
        provider_url: The provider base_url
        provider_fallback: The fallback provider ID to use if the actual provider is not found in genai-prices.
            For example, an OpenAI model should set this to "openai" in case it has an obscure provider ID.
        api_flavor: The API flavor to use when extracting usage information,
            e.g. 'chat' or 'responses' for OpenAI.
        details: Becomes the `details` field on the returned `RequestUsage` for convenience.
    """
    details = details or {}
    for provider_id, provider_api_url in [(None, provider_url), (provider, None), (provider_fallback, None)]:
        try:
            provider_obj = get_snapshot().find_provider(None, provider_id, provider_api_url)
            _model_ref, extracted_usage = provider_obj.extract_usage(data, api_flavor=api_flavor)
            return cls(**{k: v for k, v in extracted_usage.__dict__.items() if v is not None}, details=details)
        except Exception:
            pass
    return cls(details=details)

```

### RunUsage

Bases: `UsageBase`

LLM usage associated with an agent run.

Responsibility for calculating request usage is on the model; Pydantic AI simply sums the usage information across requests.

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
@dataclass(repr=False, kw_only=True)
class RunUsage(UsageBase):
    """LLM usage associated with an agent run.

    Responsibility for calculating request usage is on the model; Pydantic AI simply sums the usage information across requests.
    """

    requests: int = 0
    """Number of requests made to the LLM API."""

    tool_calls: int = 0
    """Number of successful tool calls executed during the run."""

    input_tokens: int = 0
    """Total number of input/prompt tokens."""

    cache_write_tokens: int = 0
    """Total number of tokens written to the cache."""

    cache_read_tokens: int = 0
    """Total number of tokens read from the cache."""

    input_audio_tokens: int = 0
    """Total number of audio input tokens."""

    cache_audio_read_tokens: int = 0
    """Total number of audio tokens read from the cache."""

    output_tokens: int = 0
    """Total number of output/completion tokens."""

    details: dict[str, int] = dataclasses.field(default_factory=dict)
    """Any extra details returned by the model."""

    def incr(self, incr_usage: RunUsage | RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        if isinstance(incr_usage, RunUsage):
            self.requests += incr_usage.requests
            self.tool_calls += incr_usage.tool_calls
        return _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: RunUsage | RequestUsage) -> RunUsage:
        """Add two RunUsages together.

        This is provided so it's trivial to sum usage information from multiple runs.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage

```

#### requests

```python
requests: int = 0

```

Number of requests made to the LLM API.

#### tool_calls

```python
tool_calls: int = 0

```

Number of successful tool calls executed during the run.

#### input_tokens

```python
input_tokens: int = 0

```

Total number of input/prompt tokens.

#### cache_write_tokens

```python
cache_write_tokens: int = 0

```

Total number of tokens written to the cache.

#### cache_read_tokens

```python
cache_read_tokens: int = 0

```

Total number of tokens read from the cache.

#### input_audio_tokens

```python
input_audio_tokens: int = 0

```

Total number of audio input tokens.

#### cache_audio_read_tokens

```python
cache_audio_read_tokens: int = 0

```

Total number of audio tokens read from the cache.

#### output_tokens

```python
output_tokens: int = 0

```

Total number of output/completion tokens.

#### details

```python
details: dict[str, int] = field(default_factory=dict)

```

Any extra details returned by the model.

#### incr

```python
incr(incr_usage: RunUsage | RequestUsage) -> None

```

Increment the usage in place.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `incr_usage` | `RunUsage | RequestUsage` | The usage to increment by. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
def incr(self, incr_usage: RunUsage | RequestUsage) -> None:
    """Increment the usage in place.

    Args:
        incr_usage: The usage to increment by.
    """
    if isinstance(incr_usage, RunUsage):
        self.requests += incr_usage.requests
        self.tool_calls += incr_usage.tool_calls
    return _incr_usage_tokens(self, incr_usage)

```

#### __add__

```python
__add__(other: RunUsage | RequestUsage) -> RunUsage

```

Add two RunUsages together.

This is provided so it's trivial to sum usage information from multiple runs.

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
def __add__(self, other: RunUsage | RequestUsage) -> RunUsage:
    """Add two RunUsages together.

    This is provided so it's trivial to sum usage information from multiple runs.
    """
    new_usage = copy(self)
    new_usage.incr(other)
    return new_usage

```

### Usage

Bases: `RunUsage`

Deprecated

`Usage` is deprecated, use `RunUsage` instead

Deprecated alias for `RunUsage`.

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
@dataclass(repr=False, kw_only=True)
@deprecated('`Usage` is deprecated, use `RunUsage` instead')
class Usage(RunUsage):
    """Deprecated alias for `RunUsage`."""

```

### UsageLimits

Limits on model usage.

The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model. Token counts are provided in responses from the model, and the token limits are checked after each response.

Each of the limits can be set to `None` to disable that limit.

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
@dataclass(repr=False, kw_only=True)
class UsageLimits:
    """Limits on model usage.

    The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model.
    Token counts are provided in responses from the model, and the token limits are checked after each response.

    Each of the limits can be set to `None` to disable that limit.
    """

    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""
    tool_calls_limit: int | None = None
    """The maximum number of successful tool calls allowed to be executed."""
    input_tokens_limit: int | None = None
    """The maximum number of input/prompt tokens allowed."""
    output_tokens_limit: int | None = None
    """The maximum number of output/response tokens allowed."""
    total_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests and responses combined."""
    count_tokens_before_request: bool = False
    """If True, perform a token counting pass before sending the request to the model,
    to enforce `request_tokens_limit` ahead of time. This may incur additional overhead
    (from calling the model's `count_tokens` API before making the actual request) and is disabled by default."""

    @property
    @deprecated('`request_tokens_limit` is deprecated, use `input_tokens_limit` instead')
    def request_tokens_limit(self) -> int | None:
        return self.input_tokens_limit

    @property
    @deprecated('`response_tokens_limit` is deprecated, use `output_tokens_limit` instead')
    def response_tokens_limit(self) -> int | None:
        return self.output_tokens_limit

    @overload
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
    ) -> None:
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = input_tokens_limit
        self.output_tokens_limit = output_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request

    @overload
    @deprecated(
        'Use `input_tokens_limit` instead of `request_tokens_limit` and `output_tokens_limit` and `total_tokens_limit`'
    )
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
    ) -> None:
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = request_tokens_limit
        self.output_tokens_limit = response_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request

    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
        # deprecated:
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
    ):
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = input_tokens_limit or request_tokens_limit
        self.output_tokens_limit = output_tokens_limit or response_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request

    def has_token_limits(self) -> bool:
        """Returns `True` if this instance places any limits on token counts.

        If this returns `False`, the `check_tokens` method will never raise an error.

        This is useful because if we have token limits, we need to check them after receiving each streamed message.
        If there are no limits, we can skip that processing in the streaming response iterator.
        """
        return any(
            limit is not None for limit in (self.input_tokens_limit, self.output_tokens_limit, self.total_tokens_limit)
        )

    def check_before_request(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next request would exceed any of the limits."""
        request_limit = self.request_limit
        if request_limit is not None and usage.requests >= request_limit:
            raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(
                f'The next request would exceed the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(  # pragma: lax no cover
                f'The next request would exceed the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})'
            )

    def check_tokens(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})')

        output_tokens = usage.output_tokens
        if self.output_tokens_limit is not None and output_tokens > self.output_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the output_tokens_limit of {self.output_tokens_limit} ({output_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

    def check_before_tool_call(self, projected_usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next tool call(s) would exceed the tool call limit."""
        tool_calls_limit = self.tool_calls_limit
        tool_calls = projected_usage.tool_calls
        if tool_calls_limit is not None and tool_calls > tool_calls_limit:
            raise UsageLimitExceeded(
                f'The next tool call(s) would exceed the tool_calls_limit of {tool_calls_limit} ({tool_calls=}).'
            )

    __repr__ = _utils.dataclasses_no_defaults_repr

```

#### request_limit

```python
request_limit: int | None = request_limit

```

The maximum number of requests allowed to the model.

#### tool_calls_limit

```python
tool_calls_limit: int | None = tool_calls_limit

```

The maximum number of successful tool calls allowed to be executed.

#### input_tokens_limit

```python
input_tokens_limit: int | None = (
    input_tokens_limit or request_tokens_limit
)

```

The maximum number of input/prompt tokens allowed.

#### output_tokens_limit

```python
output_tokens_limit: int | None = (
    output_tokens_limit or response_tokens_limit
)

```

The maximum number of output/response tokens allowed.

#### total_tokens_limit

```python
total_tokens_limit: int | None = total_tokens_limit

```

The maximum number of tokens allowed in requests and responses combined.

#### count_tokens_before_request

```python
count_tokens_before_request: bool = (
    count_tokens_before_request
)

```

If True, perform a token counting pass before sending the request to the model, to enforce `request_tokens_limit` ahead of time. This may incur additional overhead (from calling the model's `count_tokens` API before making the actual request) and is disabled by default.

#### has_token_limits

```python
has_token_limits() -> bool

```

Returns `True` if this instance places any limits on token counts.

If this returns `False`, the `check_tokens` method will never raise an error.

This is useful because if we have token limits, we need to check them after receiving each streamed message. If there are no limits, we can skip that processing in the streaming response iterator.

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
def has_token_limits(self) -> bool:
    """Returns `True` if this instance places any limits on token counts.

    If this returns `False`, the `check_tokens` method will never raise an error.

    This is useful because if we have token limits, we need to check them after receiving each streamed message.
    If there are no limits, we can skip that processing in the streaming response iterator.
    """
    return any(
        limit is not None for limit in (self.input_tokens_limit, self.output_tokens_limit, self.total_tokens_limit)
    )

```

#### check_before_request

```python
check_before_request(usage: RunUsage) -> None

```

Raises a `UsageLimitExceeded` exception if the next request would exceed any of the limits.

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
def check_before_request(self, usage: RunUsage) -> None:
    """Raises a `UsageLimitExceeded` exception if the next request would exceed any of the limits."""
    request_limit = self.request_limit
    if request_limit is not None and usage.requests >= request_limit:
        raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

    input_tokens = usage.input_tokens
    if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
        raise UsageLimitExceeded(
            f'The next request would exceed the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})'
        )

    total_tokens = usage.total_tokens
    if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
        raise UsageLimitExceeded(  # pragma: lax no cover
            f'The next request would exceed the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})'
        )

```

#### check_tokens

```python
check_tokens(usage: RunUsage) -> None

```

Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits.

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
def check_tokens(self, usage: RunUsage) -> None:
    """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
    input_tokens = usage.input_tokens
    if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
        raise UsageLimitExceeded(f'Exceeded the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})')

    output_tokens = usage.output_tokens
    if self.output_tokens_limit is not None and output_tokens > self.output_tokens_limit:
        raise UsageLimitExceeded(
            f'Exceeded the output_tokens_limit of {self.output_tokens_limit} ({output_tokens=})'
        )

    total_tokens = usage.total_tokens
    if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
        raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

```

#### check_before_tool_call

```python
check_before_tool_call(projected_usage: RunUsage) -> None

```

Raises a `UsageLimitExceeded` exception if the next tool call(s) would exceed the tool call limit.

Source code in `pydantic_ai_slim/pydantic_ai/usage.py`

```python
def check_before_tool_call(self, projected_usage: RunUsage) -> None:
    """Raises a `UsageLimitExceeded` exception if the next tool call(s) would exceed the tool call limit."""
    tool_calls_limit = self.tool_calls_limit
    tool_calls = projected_usage.tool_calls
    if tool_calls_limit is not None and tool_calls > tool_calls_limit:
        raise UsageLimitExceeded(
            f'The next tool call(s) would exceed the tool_calls_limit of {tool_calls_limit} ({tool_calls=}).'
        )

```

# `pydantic_ai.models.anthropic`

## Setup

For details on how to set up authentication with this model, see [model configuration for Anthropic](../../../models/anthropic/).

### LatestAnthropicModelNames

```python
LatestAnthropicModelNames = ModelParam

```

Latest Anthropic models.

### AnthropicModelName

```python
AnthropicModelName = str | LatestAnthropicModelNames

```

Possible Anthropic model names.

Since Anthropic supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [the Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models) for a full list.

### AnthropicModelSettings

Bases: `ModelSettings`

Settings used for an Anthropic model request.

Source code in `pydantic_ai_slim/pydantic_ai/models/anthropic.py`

```python
class AnthropicModelSettings(ModelSettings, total=False):
    """Settings used for an Anthropic model request."""

    # ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    anthropic_metadata: BetaMetadataParam
    """An object describing metadata about the request.

    Contains `user_id`, an external identifier for the user who is associated with the request.
    """

    anthropic_thinking: BetaThinkingConfigParam
    """Determine whether the model should generate a thinking block.

    See [the Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) for more information.
    """

    anthropic_cache_tool_definitions: bool | Literal['5m', '1h']
    """Whether to add `cache_control` to the last tool definition.

    When enabled, the last tool in the `tools` array will have `cache_control` set,
    allowing Anthropic to cache tool definitions and reduce costs.
    If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly.
    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
    """

    anthropic_cache_instructions: bool | Literal['5m', '1h']
    """Whether to add `cache_control` to the last system prompt block.

    When enabled, the last system prompt will have `cache_control` set,
    allowing Anthropic to cache system instructions and reduce costs.
    If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly.
    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
    """

```

#### anthropic_metadata

```python
anthropic_metadata: BetaMetadataParam

```

An object describing metadata about the request.

Contains `user_id`, an external identifier for the user who is associated with the request.

#### anthropic_thinking

```python
anthropic_thinking: BetaThinkingConfigParam

```

Determine whether the model should generate a thinking block.

See [the Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) for more information.

#### anthropic_cache_tool_definitions

```python
anthropic_cache_tool_definitions: bool | Literal["5m", "1h"]

```

Whether to add `cache_control` to the last tool definition.

When enabled, the last tool in the `tools` array will have `cache_control` set, allowing Anthropic to cache tool definitions and reduce costs. If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly. See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.

#### anthropic_cache_instructions

```python
anthropic_cache_instructions: bool | Literal['5m', '1h']

```

Whether to add `cache_control` to the last system prompt block.

When enabled, the last system prompt will have `cache_control` set, allowing Anthropic to cache system instructions and reduce costs. If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly. See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.

### AnthropicModel

Bases: `Model`

A model that uses the Anthropic API.

Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.

Apart from `__init__`, all methods are private or match those of the base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/anthropic.py`

```python
@dataclass(init=False)
class AnthropicModel(Model):
    """A model that uses the Anthropic API.

    Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncAnthropicClient = field(repr=False)

    _model_name: AnthropicModelName = field(repr=False)
    _provider: Provider[AsyncAnthropicClient] = field(repr=False)

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        provider: Literal['anthropic', 'gateway'] | Provider[AsyncAnthropicClient] = 'anthropic',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use. List of model names available
                [here](https://docs.anthropic.com/en/docs/about-claude/models).
            provider: The provider to use for the Anthropic API. Can be either the string 'anthropic' or an
                instance of `Provider[AsyncAnthropicClient]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/anthropic' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> AnthropicModelName:
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
        response = await self._messages_create(
            messages, False, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        return model_response

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )

        response = await self._messages_count_tokens(
            messages, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )

        return usage.RequestUsage(input_tokens=response.input_tokens)

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
        response = await self._messages_create(
            messages, True, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        settings = merge_model_settings(self.settings, model_settings)
        if (
            model_request_parameters.output_tools
            and settings
            and (thinking := settings.get('anthropic_thinking'))
            and thinking.get('type') == 'enabled'
        ):
            if model_request_parameters.output_mode == 'auto':
                model_request_parameters = replace(model_request_parameters, output_mode='prompted')
            elif (
                model_request_parameters.output_mode == 'tool' and not model_request_parameters.allow_text_output
            ):  # pragma: no branch
                # This would result in `tool_choice=required`, which Anthropic does not support with thinking.
                raise UserError(
                    'Anthropic does not support thinking and output tools at the same time. Use `output_type=PromptedOutput(...)` instead.'
                )
        return super().prepare_request(model_settings, model_request_parameters)

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[BetaRawMessageStreamEvent]:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessage:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessage | AsyncStream[BetaRawMessageStreamEvent]:
        # standalone function to make it easier to override
        tools = self._get_tools(model_request_parameters, model_settings)
        tools, mcp_servers, beta_features = self._add_builtin_tools(tools, model_request_parameters)

        tool_choice = self._infer_tool_choice(tools, model_settings, model_request_parameters)

        system_prompt, anthropic_messages = await self._map_message(messages, model_request_parameters, model_settings)

        try:
            extra_headers = self._map_extra_headers(beta_features, model_settings)

            return await self.client.beta.messages.create(
                max_tokens=model_settings.get('max_tokens', 4096),
                system=system_prompt or OMIT,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                mcp_servers=mcp_servers or OMIT,
                stream=stream,
                thinking=model_settings.get('anthropic_thinking', OMIT),
                stop_sequences=model_settings.get('stop_sequences', OMIT),
                temperature=model_settings.get('temperature', OMIT),
                top_p=model_settings.get('top_p', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                metadata=model_settings.get('anthropic_metadata', OMIT),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e  # pragma: lax no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

    async def _messages_count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessageTokensCount:
        if isinstance(self.client, AsyncAnthropicBedrock):
            raise UserError('AsyncAnthropicBedrock client does not support `count_tokens` api.')

        # standalone function to make it easier to override
        tools = self._get_tools(model_request_parameters, model_settings)
        tools, mcp_servers, beta_features = self._add_builtin_tools(tools, model_request_parameters)

        tool_choice = self._infer_tool_choice(tools, model_settings, model_request_parameters)

        system_prompt, anthropic_messages = await self._map_message(messages, model_request_parameters, model_settings)

        try:
            extra_headers = self._map_extra_headers(beta_features, model_settings)

            return await self.client.beta.messages.count_tokens(
                system=system_prompt or OMIT,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                mcp_servers=mcp_servers or OMIT,
                thinking=model_settings.get('anthropic_thinking', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e  # pragma: lax no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

    def _process_response(self, response: BetaMessage) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        items: list[ModelResponsePart] = []
        builtin_tool_calls: dict[str, BuiltinToolCallPart] = {}
        for item in response.content:
            if isinstance(item, BetaTextBlock):
                items.append(TextPart(content=item.text))
            elif isinstance(item, BetaServerToolUseBlock):
                call_part = _map_server_tool_use_block(item, self.system)
                builtin_tool_calls[call_part.tool_call_id] = call_part
                items.append(call_part)
            elif isinstance(item, BetaWebSearchToolResultBlock):
                items.append(_map_web_search_tool_result_block(item, self.system))
            elif isinstance(item, BetaCodeExecutionToolResultBlock):
                items.append(_map_code_execution_tool_result_block(item, self.system))
            elif isinstance(item, BetaRedactedThinkingBlock):
                items.append(
                    ThinkingPart(id='redacted_thinking', content='', signature=item.data, provider_name=self.system)
                )
            elif isinstance(item, BetaThinkingBlock):
                items.append(ThinkingPart(content=item.thinking, signature=item.signature, provider_name=self.system))
            elif isinstance(item, BetaMCPToolUseBlock):
                call_part = _map_mcp_server_use_block(item, self.system)
                builtin_tool_calls[call_part.tool_call_id] = call_part
                items.append(call_part)
            elif isinstance(item, BetaMCPToolResultBlock):
                call_part = builtin_tool_calls.get(item.tool_use_id)
                items.append(_map_mcp_server_result_block(item, call_part, self.system))
            else:
                assert isinstance(item, BetaToolUseBlock), f'unexpected item type {type(item)}'
                items.append(
                    ToolCallPart(
                        tool_name=item.name,
                        args=cast(dict[str, Any], item.input),
                        tool_call_id=item.id,
                    )
                )

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] | None = None
        if raw_finish_reason := response.stop_reason:  # pragma: no branch
            provider_details = {'finish_reason': raw_finish_reason}
            finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response, self._provider.name, self._provider.base_url, self._model_name),
            model_name=response.model,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self, response: AsyncStream[BetaRawMessageStreamEvent], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')  # pragma: no cover

        assert isinstance(first_chunk, BetaRawMessageStartEvent)

        return AnthropicStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.message.model,
            _response=peekable_response,
            _timestamp=_utils.now_utc(),
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
        )

    def _get_tools(
        self, model_request_parameters: ModelRequestParameters, model_settings: AnthropicModelSettings
    ) -> list[BetaToolUnionParam]:
        tools: list[BetaToolUnionParam] = [
            self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()
        ]

        # Add cache_control to the last tool if enabled
        if tools and (cache_tool_defs := model_settings.get('anthropic_cache_tool_definitions')):
            # If True, use '5m'; otherwise use the specified ttl value
            ttl: Literal['5m', '1h'] = '5m' if cache_tool_defs is True else cache_tool_defs
            last_tool = tools[-1]
            last_tool['cache_control'] = BetaCacheControlEphemeralParam(type='ephemeral', ttl=ttl)

        return tools

    def _add_builtin_tools(
        self, tools: list[BetaToolUnionParam], model_request_parameters: ModelRequestParameters
    ) -> tuple[list[BetaToolUnionParam], list[BetaRequestMCPServerURLDefinitionParam], list[str]]:
        beta_features: list[str] = []
        mcp_servers: list[BetaRequestMCPServerURLDefinitionParam] = []
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                user_location = UserLocation(type='approximate', **tool.user_location) if tool.user_location else None
                tools.append(
                    BetaWebSearchTool20250305Param(
                        name='web_search',
                        type='web_search_20250305',
                        max_uses=tool.max_uses,
                        allowed_domains=tool.allowed_domains,
                        blocked_domains=tool.blocked_domains,
                        user_location=user_location,
                    )
                )
            elif isinstance(tool, CodeExecutionTool):  # pragma: no branch
                tools.append(BetaCodeExecutionTool20250522Param(name='code_execution', type='code_execution_20250522'))
                beta_features.append('code-execution-2025-05-22')
            elif isinstance(tool, MemoryTool):  # pragma: no branch
                if 'memory' not in model_request_parameters.tool_defs:
                    raise UserError("Built-in `MemoryTool` requires a 'memory' tool to be defined.")
                # Replace the memory tool definition with the built-in memory tool
                tools = [tool for tool in tools if tool['name'] != 'memory']
                tools.append(BetaMemoryTool20250818Param(name='memory', type='memory_20250818'))
                beta_features.append('context-management-2025-06-27')
            elif isinstance(tool, MCPServerTool) and tool.url:
                mcp_server_url_definition_param = BetaRequestMCPServerURLDefinitionParam(
                    type='url',
                    name=tool.id,
                    url=tool.url,
                )
                if tool.allowed_tools is not None:  # pragma: no branch
                    mcp_server_url_definition_param['tool_configuration'] = BetaRequestMCPServerToolConfigurationParam(
                        enabled=bool(tool.allowed_tools),
                        allowed_tools=tool.allowed_tools,
                    )
                if tool.authorization_token:  # pragma: no cover
                    mcp_server_url_definition_param['authorization_token'] = tool.authorization_token
                mcp_servers.append(mcp_server_url_definition_param)
                beta_features.append('mcp-client-2025-04-04')
            else:  # pragma: no cover
                raise UserError(
                    f'`{tool.__class__.__name__}` is not supported by `AnthropicModel`. If it should be, please file an issue.'
                )
        return tools, mcp_servers, beta_features

    def _infer_tool_choice(
        self,
        tools: list[BetaToolUnionParam],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaToolChoiceParam | None:
        if not tools:
            return None
        else:
            tool_choice: BetaToolChoiceParam

            if not model_request_parameters.allow_text_output:
                tool_choice = {'type': 'any'}
            else:
                tool_choice = {'type': 'auto'}

            if 'parallel_tool_calls' in model_settings:
                tool_choice['disable_parallel_tool_use'] = not model_settings['parallel_tool_calls']

            return tool_choice

    def _map_extra_headers(self, beta_features: list[str], model_settings: AnthropicModelSettings) -> dict[str, str]:
        """Apply beta_features to extra_headers in model_settings."""
        extra_headers = model_settings.get('extra_headers', {})
        extra_headers.setdefault('User-Agent', get_user_agent())
        if beta_features:
            if 'anthropic-beta' in extra_headers:
                beta_features.insert(0, extra_headers['anthropic-beta'])
            extra_headers['anthropic-beta'] = ','.join(beta_features)
        return extra_headers

    async def _map_message(  # noqa: C901
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> tuple[str | list[BetaTextBlockParam], list[BetaMessageParam]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        system_prompt_parts: list[str] = []
        anthropic_messages: list[BetaMessageParam] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                user_content_params: list[BetaContentBlockParam] = []
                for request_part in m.parts:
                    if isinstance(request_part, SystemPromptPart):
                        system_prompt_parts.append(request_part.content)
                    elif isinstance(request_part, UserPromptPart):
                        async for content in self._map_user_prompt(request_part):
                            if isinstance(content, CachePoint):
                                self._add_cache_control_to_last_param(user_content_params, ttl=content.ttl)
                            else:
                                user_content_params.append(content)
                    elif isinstance(request_part, ToolReturnPart):
                        tool_result_block_param = BetaToolResultBlockParam(
                            tool_use_id=_guard_tool_call_id(t=request_part),
                            type='tool_result',
                            content=request_part.model_response_str(),
                            is_error=False,
                        )
                        user_content_params.append(tool_result_block_param)
                    elif isinstance(request_part, RetryPromptPart):  # pragma: no branch
                        if request_part.tool_name is None:
                            text = request_part.model_response()  # pragma: no cover
                            retry_param = BetaTextBlockParam(type='text', text=text)  # pragma: no cover
                        else:
                            retry_param = BetaToolResultBlockParam(
                                tool_use_id=_guard_tool_call_id(t=request_part),
                                type='tool_result',
                                content=request_part.model_response(),
                                is_error=True,
                            )
                        user_content_params.append(retry_param)
                if len(user_content_params) > 0:
                    anthropic_messages.append(BetaMessageParam(role='user', content=user_content_params))
            elif isinstance(m, ModelResponse):
                assistant_content_params: list[
                    BetaTextBlockParam
                    | BetaToolUseBlockParam
                    | BetaServerToolUseBlockParam
                    | BetaWebSearchToolResultBlockParam
                    | BetaCodeExecutionToolResultBlockParam
                    | BetaThinkingBlockParam
                    | BetaRedactedThinkingBlockParam
                    | BetaMCPToolUseBlockParam
                    | BetaMCPToolResultBlock
                ] = []
                for response_part in m.parts:
                    if isinstance(response_part, TextPart):
                        if response_part.content:
                            assistant_content_params.append(BetaTextBlockParam(text=response_part.content, type='text'))
                    elif isinstance(response_part, ToolCallPart):
                        tool_use_block_param = BetaToolUseBlockParam(
                            id=_guard_tool_call_id(t=response_part),
                            type='tool_use',
                            name=response_part.tool_name,
                            input=response_part.args_as_dict(),
                        )
                        assistant_content_params.append(tool_use_block_param)
                    elif isinstance(response_part, ThinkingPart):
                        if (
                            response_part.provider_name == self.system and response_part.signature is not None
                        ):  # pragma: no branch
                            if response_part.id == 'redacted_thinking':
                                assistant_content_params.append(
                                    BetaRedactedThinkingBlockParam(
                                        data=response_part.signature,
                                        type='redacted_thinking',
                                    )
                                )
                            else:
                                assistant_content_params.append(
                                    BetaThinkingBlockParam(
                                        thinking=response_part.content,
                                        signature=response_part.signature,
                                        type='thinking',
                                    )
                                )
                        elif response_part.content:  # pragma: no branch
                            start_tag, end_tag = self.profile.thinking_tags
                            assistant_content_params.append(
                                BetaTextBlockParam(
                                    text='\n'.join([start_tag, response_part.content, end_tag]), type='text'
                                )
                            )
                    elif isinstance(response_part, BuiltinToolCallPart):
                        if response_part.provider_name == self.system:
                            tool_use_id = _guard_tool_call_id(t=response_part)
                            if response_part.tool_name == WebSearchTool.kind:
                                server_tool_use_block_param = BetaServerToolUseBlockParam(
                                    id=tool_use_id,
                                    type='server_tool_use',
                                    name='web_search',
                                    input=response_part.args_as_dict(),
                                )
                                assistant_content_params.append(server_tool_use_block_param)
                            elif response_part.tool_name == CodeExecutionTool.kind:
                                server_tool_use_block_param = BetaServerToolUseBlockParam(
                                    id=tool_use_id,
                                    type='server_tool_use',
                                    name='code_execution',
                                    input=response_part.args_as_dict(),
                                )
                                assistant_content_params.append(server_tool_use_block_param)
                            elif (
                                response_part.tool_name.startswith(MCPServerTool.kind)
                                and (server_id := response_part.tool_name.split(':', 1)[1])
                                and (args := response_part.args_as_dict())
                                and (tool_name := args.get('tool_name'))
                                and (tool_args := args.get('tool_args'))
                            ):  # pragma: no branch
                                mcp_tool_use_block_param = BetaMCPToolUseBlockParam(
                                    id=tool_use_id,
                                    type='mcp_tool_use',
                                    server_name=server_id,
                                    name=tool_name,
                                    input=tool_args,
                                )
                                assistant_content_params.append(mcp_tool_use_block_param)
                    elif isinstance(response_part, BuiltinToolReturnPart):
                        if response_part.provider_name == self.system:
                            tool_use_id = _guard_tool_call_id(t=response_part)
                            if response_part.tool_name in (
                                WebSearchTool.kind,
                                'web_search_tool_result',  # Backward compatibility
                            ) and isinstance(response_part.content, dict | list):
                                assistant_content_params.append(
                                    BetaWebSearchToolResultBlockParam(
                                        tool_use_id=tool_use_id,
                                        type='web_search_tool_result',
                                        content=cast(
                                            BetaWebSearchToolResultBlockParamContentParam,
                                            response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                        ),
                                    )
                                )
                            elif response_part.tool_name in (  # pragma: no branch
                                CodeExecutionTool.kind,
                                'code_execution_tool_result',  # Backward compatibility
                            ) and isinstance(response_part.content, dict):
                                assistant_content_params.append(
                                    BetaCodeExecutionToolResultBlockParam(
                                        tool_use_id=tool_use_id,
                                        type='code_execution_tool_result',
                                        content=cast(
                                            BetaCodeExecutionToolResultBlockParamContentParam,
                                            response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                        ),
                                    )
                                )
                            elif response_part.tool_name.startswith(MCPServerTool.kind) and isinstance(
                                response_part.content, dict
                            ):  # pragma: no branch
                                assistant_content_params.append(
                                    BetaMCPToolResultBlock(
                                        tool_use_id=tool_use_id,
                                        type='mcp_tool_result',
                                        **cast(dict[str, Any], response_part.content),  # pyright: ignore[reportUnknownMemberType]
                                    )
                                )
                    elif isinstance(response_part, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
                        pass
                    else:
                        assert_never(response_part)
                if len(assistant_content_params) > 0:
                    anthropic_messages.append(BetaMessageParam(role='assistant', content=assistant_content_params))
            else:
                assert_never(m)
        if instructions := self._get_instructions(messages, model_request_parameters):
            system_prompt_parts.insert(0, instructions)
        system_prompt = '\n\n'.join(system_prompt_parts)

        # If anthropic_cache_instructions is enabled, return system prompt as a list with cache_control
        if system_prompt and (cache_instructions := model_settings.get('anthropic_cache_instructions')):
            # If True, use '5m'; otherwise use the specified ttl value
            ttl: Literal['5m', '1h'] = '5m' if cache_instructions is True else cache_instructions
            system_prompt_blocks = [
                BetaTextBlockParam(
                    type='text',
                    text=system_prompt,
                    cache_control=BetaCacheControlEphemeralParam(type='ephemeral', ttl=ttl),
                )
            ]
            return system_prompt_blocks, anthropic_messages

        return system_prompt, anthropic_messages

    @staticmethod
    def _add_cache_control_to_last_param(params: list[BetaContentBlockParam], ttl: Literal['5m', '1h'] = '5m') -> None:
        """Add cache control to the last content block param.

        See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
        """
        if not params:
            raise UserError(
                'CachePoint cannot be the first content in a user message - there must be previous content to attach the CachePoint to. '
                'To cache system instructions or tool definitions, use the `anthropic_cache_instructions` or `anthropic_cache_tool_definitions` settings instead.'
            )

        # Only certain types support cache_control
        # See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#what-can-be-cached
        cacheable_types = {'text', 'tool_use', 'server_tool_use', 'image', 'tool_result', 'document'}
        # Cast needed because BetaContentBlockParam is a union including response Block types (Pydantic models)
        # that don't support dict operations, even though at runtime we only have request Param types (TypedDicts).
        last_param = cast(dict[str, Any], params[-1])
        if last_param['type'] not in cacheable_types:
            raise UserError(f'Cache control not supported for param type: {last_param["type"]}')

        # Add cache_control to the last param
        last_param['cache_control'] = BetaCacheControlEphemeralParam(type='ephemeral', ttl=ttl)

    @staticmethod
    async def _map_user_prompt(
        part: UserPromptPart,
    ) -> AsyncGenerator[BetaContentBlockParam | CachePoint]:
        if isinstance(part.content, str):
            if part.content:  # Only yield non-empty text
                yield BetaTextBlockParam(text=part.content, type='text')
        else:
            for item in part.content:
                if isinstance(item, str):
                    if item:  # Only yield non-empty text
                        yield BetaTextBlockParam(text=item, type='text')
                elif isinstance(item, CachePoint):
                    yield item
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        yield BetaImageBlockParam(
                            source={'data': io.BytesIO(item.data), 'media_type': item.media_type, 'type': 'base64'},  # type: ignore
                            type='image',
                        )
                    elif item.media_type == 'application/pdf':
                        yield BetaBase64PDFBlockParam(
                            source=BetaBase64PDFSourceParam(
                                data=io.BytesIO(item.data),
                                media_type='application/pdf',
                                type='base64',
                            ),
                            type='document',
                        )
                    else:
                        raise RuntimeError('Only images and PDFs are supported for binary content')
                elif isinstance(item, ImageUrl):
                    yield BetaImageBlockParam(source={'type': 'url', 'url': item.url}, type='image')
                elif isinstance(item, DocumentUrl):
                    if item.media_type == 'application/pdf':
                        yield BetaBase64PDFBlockParam(source={'url': item.url, 'type': 'url'}, type='document')
                    elif item.media_type == 'text/plain':
                        downloaded_item = await download_item(item, data_format='text')
                        yield BetaBase64PDFBlockParam(
                            source=BetaPlainTextSourceParam(
                                data=downloaded_item['data'], media_type=item.media_type, type='text'
                            ),
                            type='document',
                        )
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported media type: {item.media_type}')
                else:
                    raise RuntimeError(f'Unsupported content type: {type(item)}')  # pragma: no cover

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> BetaToolParam:
        return {
            'name': f.name,
            'description': f.description or '',
            'input_schema': f.parameters_json_schema,
        }

```

#### __init__

```python
__init__(
    model_name: AnthropicModelName,
    *,
    provider: (
        Literal["anthropic", "gateway"]
        | Provider[AsyncAnthropicClient]
    ) = "anthropic",
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None
)

```

Initialize an Anthropic model.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model_name` | `AnthropicModelName` | The name of the Anthropic model to use. List of model names available here. | *required* | | `provider` | `Literal['anthropic', 'gateway'] | Provider[AsyncAnthropicClient]` | The provider to use for the Anthropic API. Can be either the string 'anthropic' or an instance of Provider[AsyncAnthropicClient]. If not provided, the other parameters will be used. | `'anthropic'` | | `profile` | `ModelProfileSpec | None` | The model profile to use. Defaults to a profile picked by the provider based on the model name. | `None` | | `settings` | `ModelSettings | None` | Default model settings for this model instance. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/anthropic.py`

```python
def __init__(
    self,
    model_name: AnthropicModelName,
    *,
    provider: Literal['anthropic', 'gateway'] | Provider[AsyncAnthropicClient] = 'anthropic',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
):
    """Initialize an Anthropic model.

    Args:
        model_name: The name of the Anthropic model to use. List of model names available
            [here](https://docs.anthropic.com/en/docs/about-claude/models).
        provider: The provider to use for the Anthropic API. Can be either the string 'anthropic' or an
            instance of `Provider[AsyncAnthropicClient]`. If not provided, the other parameters will be used.
        profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
        settings: Default model settings for this model instance.
    """
    self._model_name = model_name

    if isinstance(provider, str):
        provider = infer_provider('gateway/anthropic' if provider == 'gateway' else provider)
    self._provider = provider
    self.client = provider.client

    super().__init__(settings=settings, profile=profile or provider.model_profile)

```

#### model_name

```python
model_name: AnthropicModelName

```

The model name.

#### system

```python
system: str

```

The model provider.

### AnthropicStreamedResponse

Bases: `StreamedResponse`

Implementation of `StreamedResponse` for Anthropic models.

Source code in `pydantic_ai_slim/pydantic_ai/models/anthropic.py`

```python
@dataclass
class AnthropicStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Anthropic models."""

    _model_name: AnthropicModelName
    _response: AsyncIterable[BetaRawMessageStreamEvent]
    _timestamp: datetime
    _provider_name: str
    _provider_url: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        current_block: BetaContentBlock | None = None

        builtin_tool_calls: dict[str, BuiltinToolCallPart] = {}
        async for event in self._response:
            if isinstance(event, BetaRawMessageStartEvent):
                self._usage = _map_usage(event, self._provider_name, self._provider_url, self._model_name)
                self.provider_response_id = event.message.id

            elif isinstance(event, BetaRawContentBlockStartEvent):
                current_block = event.content_block
                if isinstance(current_block, BetaTextBlock) and current_block.text:
                    maybe_event = self._parts_manager.handle_text_delta(
                        vendor_part_id=event.index, content=current_block.text
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(current_block, BetaThinkingBlock):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        content=current_block.thinking,
                        signature=current_block.signature,
                        provider_name=self.provider_name,
                    )
                elif isinstance(current_block, BetaRedactedThinkingBlock):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        id='redacted_thinking',
                        signature=current_block.data,
                        provider_name=self.provider_name,
                    )
                elif isinstance(current_block, BetaToolUseBlock):
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=event.index,
                        tool_name=current_block.name,
                        args=cast(dict[str, Any], current_block.input) or None,
                        tool_call_id=current_block.id,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(current_block, BetaServerToolUseBlock):
                    call_part = _map_server_tool_use_block(current_block, self.provider_name)
                    builtin_tool_calls[call_part.tool_call_id] = call_part
                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index,
                        part=call_part,
                    )
                elif isinstance(current_block, BetaWebSearchToolResultBlock):
                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index,
                        part=_map_web_search_tool_result_block(current_block, self.provider_name),
                    )
                elif isinstance(current_block, BetaCodeExecutionToolResultBlock):
                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index,
                        part=_map_code_execution_tool_result_block(current_block, self.provider_name),
                    )
                elif isinstance(current_block, BetaMCPToolUseBlock):
                    call_part = _map_mcp_server_use_block(current_block, self.provider_name)
                    builtin_tool_calls[call_part.tool_call_id] = call_part

                    args_json = call_part.args_as_json_str()
                    # Drop the final `{}}` so that we can add tool args deltas
                    args_json_delta = args_json[:-3]
                    assert args_json_delta.endswith('"tool_args":'), (
                        f'Expected {args_json_delta!r} to end in `"tool_args":`'
                    )

                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index, part=replace(call_part, args=None)
                    )
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=event.index,
                        args=args_json_delta,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(current_block, BetaMCPToolResultBlock):
                    call_part = builtin_tool_calls.get(current_block.tool_use_id)
                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index,
                        part=_map_mcp_server_result_block(current_block, call_part, self.provider_name),
                    )

            elif isinstance(event, BetaRawContentBlockDeltaEvent):
                if isinstance(event.delta, BetaTextDelta):
                    maybe_event = self._parts_manager.handle_text_delta(
                        vendor_part_id=event.index, content=event.delta.text
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(event.delta, BetaThinkingDelta):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        content=event.delta.thinking,
                        provider_name=self.provider_name,
                    )
                elif isinstance(event.delta, BetaSignatureDelta):
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=event.index,
                        signature=event.delta.signature,
                        provider_name=self.provider_name,
                    )
                elif isinstance(event.delta, BetaInputJSONDelta):
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=event.index,
                        args=event.delta.partial_json,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                # TODO(Marcelo): We need to handle citations.
                elif isinstance(event.delta, BetaCitationsDelta):
                    pass

            elif isinstance(event, BetaRawMessageDeltaEvent):
                self._usage = _map_usage(event, self._provider_name, self._provider_url, self._model_name, self._usage)
                if raw_finish_reason := event.delta.stop_reason:  # pragma: no branch
                    self.provider_details = {'finish_reason': raw_finish_reason}
                    self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

            elif isinstance(event, BetaRawContentBlockStopEvent):  # pragma: no branch
                if isinstance(current_block, BetaMCPToolUseBlock):
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=event.index,
                        args='}',
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                current_block = None
            elif isinstance(event, BetaRawMessageStopEvent):  # pragma: no branch
                current_block = None

    @property
    def model_name(self) -> AnthropicModelName:
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
model_name: AnthropicModelName

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

# `pydantic_ai.models`

Logic related to making requests to an LLM.

The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the specific LLM being used.

### KnownModelName

```python
KnownModelName = TypeAliasType(
    "KnownModelName",
    Literal[
        "anthropic:claude-3-5-haiku-20241022",
        "anthropic:claude-3-5-haiku-latest",
        "anthropic:claude-3-5-sonnet-20240620",
        "anthropic:claude-3-5-sonnet-20241022",
        "anthropic:claude-3-5-sonnet-latest",
        "anthropic:claude-3-7-sonnet-20250219",
        "anthropic:claude-3-7-sonnet-latest",
        "anthropic:claude-3-haiku-20240307",
        "anthropic:claude-3-opus-20240229",
        "anthropic:claude-3-opus-latest",
        "anthropic:claude-4-opus-20250514",
        "anthropic:claude-4-sonnet-20250514",
        "anthropic:claude-haiku-4-5",
        "anthropic:claude-haiku-4-5-20251001",
        "anthropic:claude-opus-4-0",
        "anthropic:claude-opus-4-1-20250805",
        "anthropic:claude-opus-4-20250514",
        "anthropic:claude-sonnet-4-0",
        "anthropic:claude-sonnet-4-20250514",
        "anthropic:claude-sonnet-4-5",
        "anthropic:claude-sonnet-4-5-20250929",
        "bedrock:amazon.titan-text-express-v1",
        "bedrock:amazon.titan-text-lite-v1",
        "bedrock:amazon.titan-tg1-large",
        "bedrock:anthropic.claude-3-5-haiku-20241022-v1:0",
        "bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0",
        "bedrock:anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock:anthropic.claude-3-opus-20240229-v1:0",
        "bedrock:anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock:anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock:anthropic.claude-instant-v1",
        "bedrock:anthropic.claude-opus-4-20250514-v1:0",
        "bedrock:anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock:anthropic.claude-v2",
        "bedrock:anthropic.claude-v2:1",
        "bedrock:cohere.command-light-text-v14",
        "bedrock:cohere.command-r-plus-v1:0",
        "bedrock:cohere.command-r-v1:0",
        "bedrock:cohere.command-text-v14",
        "bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock:eu.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock:eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock:meta.llama3-1-405b-instruct-v1:0",
        "bedrock:meta.llama3-1-70b-instruct-v1:0",
        "bedrock:meta.llama3-1-8b-instruct-v1:0",
        "bedrock:meta.llama3-70b-instruct-v1:0",
        "bedrock:meta.llama3-8b-instruct-v1:0",
        "bedrock:mistral.mistral-7b-instruct-v0:2",
        "bedrock:mistral.mistral-large-2402-v1:0",
        "bedrock:mistral.mistral-large-2407-v1:0",
        "bedrock:mistral.mixtral-8x7b-instruct-v0:1",
        "bedrock:us.amazon.nova-lite-v1:0",
        "bedrock:us.amazon.nova-micro-v1:0",
        "bedrock:us.amazon.nova-pro-v1:0",
        "bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "bedrock:us.anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock:us.anthropic.claude-3-opus-20240229-v1:0",
        "bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock:us.anthropic.claude-opus-4-20250514-v1:0",
        "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock:us.meta.llama3-1-70b-instruct-v1:0",
        "bedrock:us.meta.llama3-1-8b-instruct-v1:0",
        "bedrock:us.meta.llama3-2-11b-instruct-v1:0",
        "bedrock:us.meta.llama3-2-1b-instruct-v1:0",
        "bedrock:us.meta.llama3-2-3b-instruct-v1:0",
        "bedrock:us.meta.llama3-2-90b-instruct-v1:0",
        "bedrock:us.meta.llama3-3-70b-instruct-v1:0",
        "cerebras:gpt-oss-120b",
        "cerebras:llama-3.3-70b",
        "cerebras:llama3.1-8b",
        "cerebras:qwen-3-235b-a22b-instruct-2507",
        "cerebras:qwen-3-235b-a22b-thinking-2507",
        "cerebras:qwen-3-32b",
        "cerebras:zai-glm-4.6",
        "cohere:c4ai-aya-expanse-32b",
        "cohere:c4ai-aya-expanse-8b",
        "cohere:command-nightly",
        "cohere:command-r-08-2024",
        "cohere:command-r-plus-08-2024",
        "cohere:command-r7b-12-2024",
        "deepseek:deepseek-chat",
        "deepseek:deepseek-reasoner",
        "google-gla:gemini-flash-latest",
        "google-gla:gemini-flash-lite-latest",
        "google-gla:gemini-2.0-flash",
        "google-gla:gemini-2.0-flash-lite",
        "google-gla:gemini-3-pro-preview",
        "google-gla:gemini-3-pro-preview-preview-09-2025",
        "google-gla:gemini-3-pro-preview-image",
        "google-gla:gemini-3-pro-preview-lite",
        "google-gla:gemini-3-pro-preview-lite-preview-09-2025",
        "google-gla:gemini-2.5-pro",
        "google-gla:gemini-3-pro-preview",
        "google-vertex:gemini-flash-latest",
        "google-vertex:gemini-flash-lite-latest",
        "google-vertex:gemini-2.0-flash",
        "google-vertex:gemini-2.0-flash-lite",
        "google-vertex:gemini-3-pro-preview",
        "google-vertex:gemini-3-pro-preview-preview-09-2025",
        "google-vertex:gemini-3-pro-preview-image",
        "google-vertex:gemini-3-pro-preview-lite",
        "google-vertex:gemini-3-pro-preview-lite-preview-09-2025",
        "google-vertex:gemini-2.5-pro",
        "google-vertex:gemini-3-pro-preview",
        "grok:grok-2-image-1212",
        "grok:grok-2-vision-1212",
        "grok:grok-3",
        "grok:grok-3-fast",
        "grok:grok-3-mini",
        "grok:grok-3-mini-fast",
        "grok:grok-4",
        "grok:grok-4-0709",
        "groq:deepseek-r1-distill-llama-70b",
        "groq:deepseek-r1-distill-qwen-32b",
        "groq:distil-whisper-large-v3-en",
        "groq:gemma2-9b-it",
        "groq:llama-3.1-8b-instant",
        "groq:llama-3.2-11b-vision-preview",
        "groq:llama-3.2-1b-preview",
        "groq:llama-3.2-3b-preview",
        "groq:llama-3.2-90b-vision-preview",
        "groq:llama-3.3-70b-specdec",
        "groq:llama-3.3-70b-versatile",
        "groq:llama-guard-3-8b",
        "groq:llama3-70b-8192",
        "groq:llama3-8b-8192",
        "groq:mistral-saba-24b",
        "groq:moonshotai/kimi-k2-instruct",
        "groq:playai-tts",
        "groq:playai-tts-arabic",
        "groq:qwen-2.5-32b",
        "groq:qwen-2.5-coder-32b",
        "groq:qwen-qwq-32b",
        "groq:whisper-large-v3",
        "groq:whisper-large-v3-turbo",
        "heroku:amazon-rerank-1-0",
        "heroku:claude-3-5-haiku",
        "heroku:claude-3-5-sonnet-latest",
        "heroku:claude-3-7-sonnet",
        "heroku:claude-3-haiku",
        "heroku:claude-4-5-haiku",
        "heroku:claude-4-5-sonnet",
        "heroku:claude-4-sonnet",
        "heroku:cohere-rerank-3-5",
        "heroku:gpt-oss-120b",
        "heroku:nova-lite",
        "heroku:nova-pro",
        "huggingface:Qwen/QwQ-32B",
        "huggingface:Qwen/Qwen2.5-72B-Instruct",
        "huggingface:Qwen/Qwen3-235B-A22B",
        "huggingface:Qwen/Qwen3-32B",
        "huggingface:deepseek-ai/DeepSeek-R1",
        "huggingface:meta-llama/Llama-3.3-70B-Instruct",
        "huggingface:meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "huggingface:meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "mistral:codestral-latest",
        "mistral:mistral-large-latest",
        "mistral:mistral-moderation-latest",
        "mistral:mistral-small-latest",
        "moonshotai:kimi-k2-0711-preview",
        "moonshotai:kimi-latest",
        "moonshotai:kimi-thinking-preview",
        "moonshotai:moonshot-v1-128k",
        "moonshotai:moonshot-v1-128k-vision-preview",
        "moonshotai:moonshot-v1-32k",
        "moonshotai:moonshot-v1-32k-vision-preview",
        "moonshotai:moonshot-v1-8k",
        "moonshotai:moonshot-v1-8k-vision-preview",
        "openai:chatgpt-4o-latest",
        "openai:codex-mini-latest",
        "openai:computer-use-preview",
        "openai:computer-use-preview-2025-03-11",
        "openai:gpt-3.5-turbo",
        "openai:gpt-3.5-turbo-0125",
        "openai:gpt-3.5-turbo-0301",
        "openai:gpt-3.5-turbo-0613",
        "openai:gpt-3.5-turbo-1106",
        "openai:gpt-3.5-turbo-16k",
        "openai:gpt-3.5-turbo-16k-0613",
        "openai:gpt-4",
        "openai:gpt-4-0125-preview",
        "openai:gpt-4-0314",
        "openai:gpt-4-0613",
        "openai:gpt-4-1106-preview",
        "openai:gpt-4-32k",
        "openai:gpt-4-32k-0314",
        "openai:gpt-4-32k-0613",
        "openai:gpt-4-turbo",
        "openai:gpt-4-turbo-2024-04-09",
        "openai:gpt-4-turbo-preview",
        "openai:gpt-4-vision-preview",
        "openai:gpt-4.1",
        "openai:gpt-4.1-2025-04-14",
        "openai:gpt-4.1-mini",
        "openai:gpt-4.1-mini-2025-04-14",
        "openai:gpt-4.1-nano",
        "openai:gpt-4.1-nano-2025-04-14",
        "openai:gpt-4o",
        "openai:gpt-4o-2024-05-13",
        "openai:gpt-4o-2024-08-06",
        "openai:gpt-4o-2024-11-20",
        "openai:gpt-4o-audio-preview",
        "openai:gpt-4o-audio-preview-2024-10-01",
        "openai:gpt-4o-audio-preview-2024-12-17",
        "openai:gpt-4o-audio-preview-2025-06-03",
        "openai:gpt-4o-mini",
        "openai:gpt-4o-mini-2024-07-18",
        "openai:gpt-4o-mini-audio-preview",
        "openai:gpt-4o-mini-audio-preview-2024-12-17",
        "openai:gpt-4o-mini-search-preview",
        "openai:gpt-4o-mini-search-preview-2025-03-11",
        "openai:gpt-4o-search-preview",
        "openai:gpt-4o-search-preview-2025-03-11",
        "openai:gpt-5",
        "openai:gpt-5-2025-08-07",
        "openai:gpt-5-chat-latest",
        "openai:gpt-5-codex",
        "openai:gpt-5-mini",
        "openai:gpt-5-mini-2025-08-07",
        "openai:gpt-5-nano",
        "openai:gpt-5-nano-2025-08-07",
        "openai:gpt-5-pro",
        "openai:gpt-5-pro-2025-10-06",
        "openai:gpt-5.1",
        "openai:gpt-5.1-2025-11-13",
        "openai:gpt-5.1-chat-latest",
        "openai:gpt-5.1-codex",
        "openai:gpt-5.1-mini",
        "openai:o1",
        "openai:o1-2024-12-17",
        "openai:o1-mini",
        "openai:o1-mini-2024-09-12",
        "openai:o1-preview",
        "openai:o1-preview-2024-09-12",
        "openai:o1-pro",
        "openai:o1-pro-2025-03-19",
        "openai:o3",
        "openai:o3-2025-04-16",
        "openai:o3-deep-research",
        "openai:o3-deep-research-2025-06-26",
        "openai:o3-mini",
        "openai:o3-mini-2025-01-31",
        "openai:o3-pro",
        "openai:o3-pro-2025-06-10",
        "openai:o4-mini",
        "openai:o4-mini-2025-04-16",
        "openai:o4-mini-deep-research",
        "openai:o4-mini-deep-research-2025-06-26",
        "test",
    ],
)

```

Known model names that can be used with the `model` parameter of Agent.

`KnownModelName` is provided as a concise way to specify a model.

### ModelRequestParameters

Configuration for an agent's request to a model, specifically related to tools and output handling.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
@dataclass(repr=False, kw_only=True)
class ModelRequestParameters:
    """Configuration for an agent's request to a model, specifically related to tools and output handling."""

    function_tools: list[ToolDefinition] = field(default_factory=list)
    builtin_tools: list[AbstractBuiltinTool] = field(default_factory=list)

    output_mode: OutputMode = 'text'
    output_object: OutputObjectDefinition | None = None
    output_tools: list[ToolDefinition] = field(default_factory=list)
    prompted_output_template: str | None = None
    allow_text_output: bool = True
    allow_image_output: bool = False

    @cached_property
    def tool_defs(self) -> dict[str, ToolDefinition]:
        return {tool_def.name: tool_def for tool_def in [*self.function_tools, *self.output_tools]}

    @cached_property
    def prompted_output_instructions(self) -> str | None:
        if self.output_mode == 'prompted' and self.prompted_output_template and self.output_object:
            return PromptedOutputSchema.build_instructions(self.prompted_output_template, self.output_object)
        return None

    __repr__ = _utils.dataclasses_no_defaults_repr

```

### Model

Bases: `ABC`

Abstract class for a model.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
class Model(ABC):
    """Abstract class for a model."""

    _profile: ModelProfileSpec | None = None
    _settings: ModelSettings | None = None

    def __init__(
        self,
        *,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        """Initialize the model with optional settings and profile.

        Args:
            settings: Model-specific settings that will be used as defaults for this model.
            profile: The model profile to use.
        """
        self._settings = settings
        self._profile = profile

    @property
    def settings(self) -> ModelSettings | None:
        """Get the model settings."""
        return self._settings

    @abstractmethod
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model."""
        raise NotImplementedError()

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        """Make a request to the model for counting tokens."""
        # This method is not required, but you need to implement it if you want to support `UsageLimits.count_tokens_before_request`.
        raise NotImplementedError(f'Token counting ahead of the request is not supported by {self.__class__.__name__}')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a request to the model and return a streaming response."""
        # This method is not required, but you need to implement it if you want to support streamed responses
        raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
        # yield is required to make this a generator for type checking
        # noinspection PyUnreachableCode
        yield  # pragma: no cover

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        """Customize the request parameters for the model.

        This method can be overridden by subclasses to modify the request parameters before sending them to the model.
        In particular, this method can be used to make modifications to the generated tool JSON schemas if necessary
        for vendor/model-specific reasons.
        """
        if transformer := self.profile.json_schema_transformer:
            model_request_parameters = replace(
                model_request_parameters,
                function_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.function_tools],
                output_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.output_tools],
            )
            if output_object := model_request_parameters.output_object:
                model_request_parameters = replace(
                    model_request_parameters,
                    output_object=_customize_output_object(transformer, output_object),
                )

        return model_request_parameters

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        """Prepare request inputs before they are passed to the provider.

        This merges the given `model_settings` with the model's own `settings` attribute and ensures
        `customize_request_parameters` is applied to the resolved
        [`ModelRequestParameters`][pydantic_ai.models.ModelRequestParameters]. Subclasses can override this method if
        they need to customize the preparation flow further, but most implementations should simply call
        `self.prepare_request(...)` at the start of their `request` (and related) methods.
        """
        model_settings = merge_model_settings(self.settings, model_settings)

        params = self.customize_request_parameters(model_request_parameters)

        if builtin_tools := params.builtin_tools:
            # Deduplicate builtin tools
            params = replace(
                params,
                builtin_tools=list({tool.unique_id: tool for tool in builtin_tools}.values()),
            )

        if params.output_mode == 'auto':
            output_mode = self.profile.default_structured_output_mode
            params = replace(
                params,
                output_mode=output_mode,
                allow_text_output=output_mode in ('native', 'prompted'),
            )

        # Reset irrelevant fields
        if params.output_tools and params.output_mode != 'tool':
            params = replace(params, output_tools=[])
        if params.output_object and params.output_mode not in ('native', 'prompted'):
            params = replace(params, output_object=None)
        if params.prompted_output_template and params.output_mode != 'prompted':
            params = replace(params, prompted_output_template=None)  # pragma: no cover

        # Set default prompted output template
        if params.output_mode == 'prompted' and not params.prompted_output_template:
            params = replace(params, prompted_output_template=self.profile.prompted_output_template)

        # Check if output mode is supported
        if params.output_mode == 'native' and not self.profile.supports_json_schema_output:
            raise UserError('Native structured output is not supported by this model.')
        if params.output_mode == 'tool' and not self.profile.supports_tools:
            raise UserError('Tool output is not supported by this model.')
        if params.allow_image_output and not self.profile.supports_image_output:
            raise UserError('Image output is not supported by this model.')

        return model_settings, params

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name."""
        raise NotImplementedError()

    @cached_property
    def profile(self) -> ModelProfile:
        """The model profile."""
        _profile = self._profile
        if callable(_profile):
            _profile = _profile(self.model_name)

        if _profile is None:
            return DEFAULT_PROFILE

        return _profile

    @property
    @abstractmethod
    def system(self) -> str:
        """The model provider, ex: openai.

        Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute,
        so should use well-known values listed in
        https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system
        when applicable.
        """
        raise NotImplementedError()

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @staticmethod
    def _get_instructions(
        messages: list[ModelMessage], model_request_parameters: ModelRequestParameters | None = None
    ) -> str | None:
        """Get instructions from the first ModelRequest found when iterating messages in reverse.

        In the case that a "mock" request was generated to include a tool-return part for a result tool,
        we want to use the instructions from the second-to-most-recent request (which should correspond to the
        original request that generated the response that resulted in the tool-return part).
        """
        instructions = None

        last_two_requests: list[ModelRequest] = []
        for message in reversed(messages):
            if isinstance(message, ModelRequest):
                last_two_requests.append(message)
                if len(last_two_requests) == 2:
                    break
                if message.instructions is not None:
                    instructions = message.instructions
                    break

        # If we don't have two requests, and we didn't already return instructions, there are definitely not any:
        if instructions is None and len(last_two_requests) == 2:
            most_recent_request = last_two_requests[0]
            second_most_recent_request = last_two_requests[1]

            # If we've gotten this far and the most recent request consists of only tool-return parts or retry-prompt parts,
            # we use the instructions from the second-to-most-recent request. This is necessary because when handling
            # result tools, we generate a "mock" ModelRequest with a tool-return part for it, and that ModelRequest will not
            # have the relevant instructions from the agent.

            # While it's possible that you could have a message history where the most recent request has only tool returns,
            # I believe there is no way to achieve that would _change_ the instructions without manually crafting the most
            # recent message. That might make sense in principle for some usage pattern, but it's enough of an edge case
            # that I think it's not worth worrying about, since you can work around this by inserting another ModelRequest
            # with no parts at all immediately before the request that has the tool calls (that works because we only look
            # at the two most recent ModelRequests here).

            # If you have a use case where this causes pain, please open a GitHub issue and we can discuss alternatives.

            if all(p.part_kind == 'tool-return' or p.part_kind == 'retry-prompt' for p in most_recent_request.parts):
                instructions = second_most_recent_request.instructions

        if model_request_parameters and (output_instructions := model_request_parameters.prompted_output_instructions):
            if instructions:
                instructions = '\n\n'.join([instructions, output_instructions])
            else:
                instructions = output_instructions

        return instructions

```

#### __init__

```python
__init__(
    *,
    settings: ModelSettings | None = None,
    profile: ModelProfileSpec | None = None
) -> None

```

Initialize the model with optional settings and profile.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `settings` | `ModelSettings | None` | Model-specific settings that will be used as defaults for this model. | `None` | | `profile` | `ModelProfileSpec | None` | The model profile to use. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
def __init__(
    self,
    *,
    settings: ModelSettings | None = None,
    profile: ModelProfileSpec | None = None,
) -> None:
    """Initialize the model with optional settings and profile.

    Args:
        settings: Model-specific settings that will be used as defaults for this model.
        profile: The model profile to use.
    """
    self._settings = settings
    self._profile = profile

```

#### settings

```python
settings: ModelSettings | None

```

Get the model settings.

#### request

```python
request(
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse

```

Make a request to the model.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
@abstractmethod
async def request(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse:
    """Make a request to the model."""
    raise NotImplementedError()

```

#### count_tokens

```python
count_tokens(
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> RequestUsage

```

Make a request to the model for counting tokens.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
async def count_tokens(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> RequestUsage:
    """Make a request to the model for counting tokens."""
    # This method is not required, but you need to implement it if you want to support `UsageLimits.count_tokens_before_request`.
    raise NotImplementedError(f'Token counting ahead of the request is not supported by {self.__class__.__name__}')

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

Make a request to the model and return a streaming response.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
@asynccontextmanager
async def request_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    run_context: RunContext[Any] | None = None,
) -> AsyncIterator[StreamedResponse]:
    """Make a request to the model and return a streaming response."""
    # This method is not required, but you need to implement it if you want to support streamed responses
    raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
    # yield is required to make this a generator for type checking
    # noinspection PyUnreachableCode
    yield  # pragma: no cover

```

#### customize_request_parameters

```python
customize_request_parameters(
    model_request_parameters: ModelRequestParameters,
) -> ModelRequestParameters

```

Customize the request parameters for the model.

This method can be overridden by subclasses to modify the request parameters before sending them to the model. In particular, this method can be used to make modifications to the generated tool JSON schemas if necessary for vendor/model-specific reasons.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
    """Customize the request parameters for the model.

    This method can be overridden by subclasses to modify the request parameters before sending them to the model.
    In particular, this method can be used to make modifications to the generated tool JSON schemas if necessary
    for vendor/model-specific reasons.
    """
    if transformer := self.profile.json_schema_transformer:
        model_request_parameters = replace(
            model_request_parameters,
            function_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.function_tools],
            output_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.output_tools],
        )
        if output_object := model_request_parameters.output_object:
            model_request_parameters = replace(
                model_request_parameters,
                output_object=_customize_output_object(transformer, output_object),
            )

    return model_request_parameters

```

#### prepare_request

```python
prepare_request(
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> tuple[ModelSettings | None, ModelRequestParameters]

```

Prepare request inputs before they are passed to the provider.

This merges the given `model_settings` with the model's own `settings` attribute and ensures `customize_request_parameters` is applied to the resolved ModelRequestParameters. Subclasses can override this method if they need to customize the preparation flow further, but most implementations should simply call `self.prepare_request(...)` at the start of their `request` (and related) methods.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
def prepare_request(
    self,
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> tuple[ModelSettings | None, ModelRequestParameters]:
    """Prepare request inputs before they are passed to the provider.

    This merges the given `model_settings` with the model's own `settings` attribute and ensures
    `customize_request_parameters` is applied to the resolved
    [`ModelRequestParameters`][pydantic_ai.models.ModelRequestParameters]. Subclasses can override this method if
    they need to customize the preparation flow further, but most implementations should simply call
    `self.prepare_request(...)` at the start of their `request` (and related) methods.
    """
    model_settings = merge_model_settings(self.settings, model_settings)

    params = self.customize_request_parameters(model_request_parameters)

    if builtin_tools := params.builtin_tools:
        # Deduplicate builtin tools
        params = replace(
            params,
            builtin_tools=list({tool.unique_id: tool for tool in builtin_tools}.values()),
        )

    if params.output_mode == 'auto':
        output_mode = self.profile.default_structured_output_mode
        params = replace(
            params,
            output_mode=output_mode,
            allow_text_output=output_mode in ('native', 'prompted'),
        )

    # Reset irrelevant fields
    if params.output_tools and params.output_mode != 'tool':
        params = replace(params, output_tools=[])
    if params.output_object and params.output_mode not in ('native', 'prompted'):
        params = replace(params, output_object=None)
    if params.prompted_output_template and params.output_mode != 'prompted':
        params = replace(params, prompted_output_template=None)  # pragma: no cover

    # Set default prompted output template
    if params.output_mode == 'prompted' and not params.prompted_output_template:
        params = replace(params, prompted_output_template=self.profile.prompted_output_template)

    # Check if output mode is supported
    if params.output_mode == 'native' and not self.profile.supports_json_schema_output:
        raise UserError('Native structured output is not supported by this model.')
    if params.output_mode == 'tool' and not self.profile.supports_tools:
        raise UserError('Tool output is not supported by this model.')
    if params.allow_image_output and not self.profile.supports_image_output:
        raise UserError('Image output is not supported by this model.')

    return model_settings, params

```

#### model_name

```python
model_name: str

```

The model name.

#### profile

```python
profile: ModelProfile

```

The model profile.

#### system

```python
system: str

```

The model provider, ex: openai.

Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute, so should use well-known values listed in https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system when applicable.

#### base_url

```python
base_url: str | None

```

The base URL for the provider API, if available.

### StreamedResponse

Bases: `ABC`

Streamed response from an LLM when calling a tool.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
@dataclass
class StreamedResponse(ABC):
    """Streamed response from an LLM when calling a tool."""

    model_request_parameters: ModelRequestParameters

    final_result_event: FinalResultEvent | None = field(default=None, init=False)

    provider_response_id: str | None = field(default=None, init=False)
    provider_details: dict[str, Any] | None = field(default=None, init=False)
    finish_reason: FinishReason | None = field(default=None, init=False)

    _parts_manager: ModelResponsePartsManager = field(default_factory=ModelResponsePartsManager, init=False)
    _event_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)
    _usage: RequestUsage = field(default_factory=RequestUsage, init=False)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream the response as an async iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This proxies the `_event_iterator()` and emits all events, while also checking for matches
        on the result schema and emitting a [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] if/when the
        first match is found.
        """
        if self._event_iterator is None:

            async def iterator_with_final_event(
                iterator: AsyncIterator[ModelResponseStreamEvent],
            ) -> AsyncIterator[ModelResponseStreamEvent]:
                async for event in iterator:
                    yield event
                    if (
                        final_result_event := _get_final_result_event(event, self.model_request_parameters)
                    ) is not None:
                        self.final_result_event = final_result_event
                        yield final_result_event
                        break

                # If we broke out of the above loop, we need to yield the rest of the events
                # If we didn't, this will just be a no-op
                async for event in iterator:
                    yield event

            async def iterator_with_part_end(
                iterator: AsyncIterator[ModelResponseStreamEvent],
            ) -> AsyncIterator[ModelResponseStreamEvent]:
                last_start_event: PartStartEvent | None = None

                def part_end_event(next_part: ModelResponsePart | None = None) -> PartEndEvent | None:
                    if not last_start_event:
                        return None

                    index = last_start_event.index
                    part = self._parts_manager.get_parts()[index]
                    if not isinstance(part, TextPart | ThinkingPart | BaseToolCallPart):
                        # Parts other than these 3 don't have deltas, so don't need an end part.
                        return None

                    return PartEndEvent(
                        index=index,
                        part=part,
                        next_part_kind=next_part.part_kind if next_part else None,
                    )

                async for event in iterator:
                    if isinstance(event, PartStartEvent):
                        if last_start_event:
                            end_event = part_end_event(event.part)
                            if end_event:
                                yield end_event

                            event.previous_part_kind = last_start_event.part.part_kind
                        last_start_event = event

                    yield event

                end_event = part_end_event()
                if end_event:
                    yield end_event

            self._event_iterator = iterator_with_part_end(iterator_with_final_event(self._get_event_iterator()))
        return self._event_iterator

    @abstractmethod
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Return an async iterator of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This method should be implemented by subclasses to translate the vendor-specific stream of events into
        pydantic_ai-format events.

        It should use the `_parts_manager` to handle deltas, and should update the `_usage` attributes as it goes.
        """
        raise NotImplementedError()
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        """Build a [`ModelResponse`][pydantic_ai.messages.ModelResponse] from the data received from the stream so far."""
        return ModelResponse(
            parts=self._parts_manager.get_parts(),
            model_name=self.model_name,
            timestamp=self.timestamp,
            usage=self.usage(),
            provider_name=self.provider_name,
            provider_response_id=self.provider_response_id,
            provider_details=self.provider_details,
            finish_reason=self.finish_reason,
        )

    # TODO (v2): Make this a property
    def usage(self) -> RequestUsage:
        """Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
        return self._usage

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name of the response."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def provider_name(self) -> str | None:
        """Get the provider name."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        raise NotImplementedError()

```

#### __aiter__

```python
__aiter__() -> AsyncIterator[ModelResponseStreamEvent]

```

Stream the response as an async iterable of ModelResponseStreamEvents.

This proxies the `_event_iterator()` and emits all events, while also checking for matches on the result schema and emitting a FinalResultEvent if/when the first match is found.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
    """Stream the response as an async iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

    This proxies the `_event_iterator()` and emits all events, while also checking for matches
    on the result schema and emitting a [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] if/when the
    first match is found.
    """
    if self._event_iterator is None:

        async def iterator_with_final_event(
            iterator: AsyncIterator[ModelResponseStreamEvent],
        ) -> AsyncIterator[ModelResponseStreamEvent]:
            async for event in iterator:
                yield event
                if (
                    final_result_event := _get_final_result_event(event, self.model_request_parameters)
                ) is not None:
                    self.final_result_event = final_result_event
                    yield final_result_event
                    break

            # If we broke out of the above loop, we need to yield the rest of the events
            # If we didn't, this will just be a no-op
            async for event in iterator:
                yield event

        async def iterator_with_part_end(
            iterator: AsyncIterator[ModelResponseStreamEvent],
        ) -> AsyncIterator[ModelResponseStreamEvent]:
            last_start_event: PartStartEvent | None = None

            def part_end_event(next_part: ModelResponsePart | None = None) -> PartEndEvent | None:
                if not last_start_event:
                    return None

                index = last_start_event.index
                part = self._parts_manager.get_parts()[index]
                if not isinstance(part, TextPart | ThinkingPart | BaseToolCallPart):
                    # Parts other than these 3 don't have deltas, so don't need an end part.
                    return None

                return PartEndEvent(
                    index=index,
                    part=part,
                    next_part_kind=next_part.part_kind if next_part else None,
                )

            async for event in iterator:
                if isinstance(event, PartStartEvent):
                    if last_start_event:
                        end_event = part_end_event(event.part)
                        if end_event:
                            yield end_event

                        event.previous_part_kind = last_start_event.part.part_kind
                    last_start_event = event

                yield event

            end_event = part_end_event()
            if end_event:
                yield end_event

        self._event_iterator = iterator_with_part_end(iterator_with_final_event(self._get_event_iterator()))
    return self._event_iterator

```

#### get

```python
get() -> ModelResponse

```

Build a ModelResponse from the data received from the stream so far.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
def get(self) -> ModelResponse:
    """Build a [`ModelResponse`][pydantic_ai.messages.ModelResponse] from the data received from the stream so far."""
    return ModelResponse(
        parts=self._parts_manager.get_parts(),
        model_name=self.model_name,
        timestamp=self.timestamp,
        usage=self.usage(),
        provider_name=self.provider_name,
        provider_response_id=self.provider_response_id,
        provider_details=self.provider_details,
        finish_reason=self.finish_reason,
    )

```

#### usage

```python
usage() -> RequestUsage

```

Get the usage of the response so far. This will not be the final usage until the stream is exhausted.

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
def usage(self) -> RequestUsage:
    """Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
    return self._usage

```

#### model_name

```python
model_name: str

```

Get the model name of the response.

#### provider_name

```python
provider_name: str | None

```

Get the provider name.

#### timestamp

```python
timestamp: datetime

```

Get the timestamp of the response.

### ALLOW_MODEL_REQUESTS

```python
ALLOW_MODEL_REQUESTS = True

```

Whether to allow requests to models.

This global setting allows you to disable request to most models, e.g. to make sure you don't accidentally make costly requests to a model during tests.

The testing models TestModel and FunctionModel are no affected by this setting.

### check_allow_model_requests

```python
check_allow_model_requests() -> None

```

Check if model requests are allowed.

If you're defining your own models that have costs or latency associated with their use, you should call this in Model.request and Model.request_stream.

Raises:

| Type | Description | | --- | --- | | `RuntimeError` | If model requests are not allowed. |

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
def check_allow_model_requests() -> None:
    """Check if model requests are allowed.

    If you're defining your own models that have costs or latency associated with their use, you should call this in
    [`Model.request`][pydantic_ai.models.Model.request] and [`Model.request_stream`][pydantic_ai.models.Model.request_stream].

    Raises:
        RuntimeError: If model requests are not allowed.
    """
    if not ALLOW_MODEL_REQUESTS:
        raise RuntimeError('Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False')

```

### override_allow_model_requests

```python
override_allow_model_requests(
    allow_model_requests: bool,
) -> Iterator[None]

```

Context manager to temporarily override ALLOW_MODEL_REQUESTS.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `allow_model_requests` | `bool` | Whether to allow model requests within the context. | *required* |

Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`

```python
@contextmanager
def override_allow_model_requests(allow_model_requests: bool) -> Iterator[None]:
    """Context manager to temporarily override [`ALLOW_MODEL_REQUESTS`][pydantic_ai.models.ALLOW_MODEL_REQUESTS].

    Args:
        allow_model_requests: Whether to allow model requests within the context.
    """
    global ALLOW_MODEL_REQUESTS
    old_value = ALLOW_MODEL_REQUESTS
    ALLOW_MODEL_REQUESTS = allow_model_requests  # pyright: ignore[reportConstantRedefinition]
    try:
        yield
    finally:
        ALLOW_MODEL_REQUESTS = old_value  # pyright: ignore[reportConstantRedefinition]

```

