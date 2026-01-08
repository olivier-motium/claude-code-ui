<!-- Source: _complete-doc.md -->

# `pydantic_graph.beta.step`

Step-based graph execution components.

This module provides the core abstractions for step-based graph execution, including step contexts, step functions, and step nodes that bridge between the v1 and v2 graph execution systems.

### StepContext

Bases: `Generic[StateT, DepsT, InputT]`

Context information passed to step functions during graph execution.

The step context provides access to the current graph state, dependencies, and input data for a step.

Type Parameters

StateT: The type of the graph state DepsT: The type of the dependencies InputT: The type of the input data

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
@dataclass(init=False)
class StepContext(Generic[StateT, DepsT, InputT]):
    """Context information passed to step functions during graph execution.

    The step context provides access to the current graph state, dependencies, and input data for a step.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
    """

    _state: StateT
    """The current graph state."""
    _deps: DepsT
    """The graph run dependencies."""
    _inputs: InputT
    """The input data for this step."""

    def __init__(self, *, state: StateT, deps: DepsT, inputs: InputT):
        self._state = state
        self._deps = deps
        self._inputs = inputs

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def deps(self) -> DepsT:
        return self._deps

    @property
    def inputs(self) -> InputT:
        """The input data for this step.

        This must be a property to ensure correct variance behavior
        """
        return self._inputs

```

#### inputs

```python
inputs: InputT

```

The input data for this step.

This must be a property to ensure correct variance behavior

### StepFunction

Bases: `Protocol[StateT, DepsT, InputT, OutputT]`

Protocol for step functions that can be executed in the graph.

Step functions are async callables that receive a step context and return a result.

Type Parameters

StateT: The type of the graph state DepsT: The type of the dependencies InputT: The type of the input data OutputT: The type of the output data

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
class StepFunction(Protocol[StateT, DepsT, InputT, OutputT]):
    """Protocol for step functions that can be executed in the graph.

    Step functions are async callables that receive a step context and return a result.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> Awaitable[OutputT]:
        """Execute the step function with the given context.

        Args:
            ctx: The step context containing state, dependencies, and inputs

        Returns:
            An awaitable that resolves to the step's output
        """
        raise NotImplementedError

```

#### __call__

```python
__call__(
    ctx: StepContext[StateT, DepsT, InputT],
) -> Awaitable[OutputT]

```

Execute the step function with the given context.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `StepContext[StateT, DepsT, InputT]` | The step context containing state, dependencies, and inputs | *required* |

Returns:

| Type | Description | | --- | --- | | `Awaitable[OutputT]` | An awaitable that resolves to the step's output |

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> Awaitable[OutputT]:
    """Execute the step function with the given context.

    Args:
        ctx: The step context containing state, dependencies, and inputs

    Returns:
        An awaitable that resolves to the step's output
    """
    raise NotImplementedError

```

### StreamFunction

Bases: `Protocol[StateT, DepsT, InputT, OutputT]`

Protocol for stream functions that can be executed in the graph.

Stream functions are async callables that receive a step context and return an async iterator.

Type Parameters

StateT: The type of the graph state DepsT: The type of the dependencies InputT: The type of the input data OutputT: The type of the output data

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
class StreamFunction(Protocol[StateT, DepsT, InputT, OutputT]):
    """Protocol for stream functions that can be executed in the graph.

    Stream functions are async callables that receive a step context and return an async iterator.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> AsyncIterator[OutputT]:
        """Execute the stream function with the given context.

        Args:
            ctx: The step context containing state, dependencies, and inputs

        Returns:
            An async iterator yielding the streamed output
        """
        raise NotImplementedError
        yield

```

#### __call__

```python
__call__(
    ctx: StepContext[StateT, DepsT, InputT],
) -> AsyncIterator[OutputT]

```

Execute the stream function with the given context.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `StepContext[StateT, DepsT, InputT]` | The step context containing state, dependencies, and inputs | *required* |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[OutputT]` | An async iterator yielding the streamed output |

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> AsyncIterator[OutputT]:
    """Execute the stream function with the given context.

    Args:
        ctx: The step context containing state, dependencies, and inputs

    Returns:
        An async iterator yielding the streamed output
    """
    raise NotImplementedError
    yield

```

### AnyStepFunction

```python
AnyStepFunction = StepFunction[Any, Any, Any, Any]

```

Type alias for a step function with any type parameters.

### Step

Bases: `Generic[StateT, DepsT, InputT, OutputT]`

A step in the graph execution that wraps a step function.

Steps represent individual units of execution in the graph, encapsulating a step function along with metadata like ID and label.

Type Parameters

StateT: The type of the graph state DepsT: The type of the dependencies InputT: The type of the input data OutputT: The type of the output data

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
@dataclass(init=False)
class Step(Generic[StateT, DepsT, InputT, OutputT]):
    """A step in the graph execution that wraps a step function.

    Steps represent individual units of execution in the graph, encapsulating
    a step function along with metadata like ID and label.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    id: NodeID
    """Unique identifier for this step."""
    _call: StepFunction[StateT, DepsT, InputT, OutputT]
    """The step function to execute."""
    label: str | None
    """Optional human-readable label for this step."""

    def __init__(self, *, id: NodeID, call: StepFunction[StateT, DepsT, InputT, OutputT], label: str | None = None):
        self.id = id
        self._call = call
        self.label = label

    @property
    def call(self) -> StepFunction[StateT, DepsT, InputT, OutputT]:
        """The step function to execute. This needs to be a property for proper variance inference."""
        return self._call

    @overload
    def as_node(self, inputs: None = None) -> StepNode[StateT, DepsT]: ...

    @overload
    def as_node(self, inputs: InputT) -> StepNode[StateT, DepsT]: ...

    def as_node(self, inputs: InputT | None = None) -> StepNode[StateT, DepsT]:
        """Create a step node with bound inputs.

        Args:
            inputs: The input data to bind to this step, or None

        Returns:
            A [`StepNode`][pydantic_graph.beta.step.StepNode] with this step and the bound inputs
        """
        return StepNode(self, inputs)

```

#### id

```python
id: NodeID = id

```

Unique identifier for this step.

#### label

```python
label: str | None = label

```

Optional human-readable label for this step.

#### call

```python
call: StepFunction[StateT, DepsT, InputT, OutputT]

```

The step function to execute. This needs to be a property for proper variance inference.

#### as_node

```python
as_node(inputs: None = None) -> StepNode[StateT, DepsT]

```

```python
as_node(inputs: InputT) -> StepNode[StateT, DepsT]

```

```python
as_node(
    inputs: InputT | None = None,
) -> StepNode[StateT, DepsT]

```

Create a step node with bound inputs.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `inputs` | `InputT | None` | The input data to bind to this step, or None | `None` |

Returns:

| Type | Description | | --- | --- | | `StepNode[StateT, DepsT]` | A StepNode with this step and the bound inputs |

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
def as_node(self, inputs: InputT | None = None) -> StepNode[StateT, DepsT]:
    """Create a step node with bound inputs.

    Args:
        inputs: The input data to bind to this step, or None

    Returns:
        A [`StepNode`][pydantic_graph.beta.step.StepNode] with this step and the bound inputs
    """
    return StepNode(self, inputs)

```

### StepNode

Bases: `BaseNode[StateT, DepsT, Any]`

A base node that represents a step with bound inputs.

StepNode bridges between the v1 and v2 graph execution systems by wrapping a Step with bound inputs in a BaseNode interface. It is not meant to be run directly but rather used to indicate transitions to v2-style steps.

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
@dataclass
class StepNode(BaseNode[StateT, DepsT, Any]):
    """A base node that represents a step with bound inputs.

    StepNode bridges between the v1 and v2 graph execution systems by wrapping
    a [`Step`][pydantic_graph.beta.step.Step] with bound inputs in a BaseNode interface.
    It is not meant to be run directly but rather used to indicate transitions
    to v2-style steps.
    """

    step: Step[StateT, DepsT, Any, Any]
    """The step to execute."""

    inputs: Any
    """The inputs bound to this step."""

    async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
        """Attempt to run the step node.

        Args:
            ctx: The graph execution context

        Returns:
            The result of step execution

        Raises:
            NotImplementedError: Always raised as StepNode is not meant to be run directly
        """
        raise NotImplementedError(
            '`StepNode` is not meant to be run directly, it is meant to be used in `BaseNode` subclasses to indicate a transition to v2-style steps.'
        )

```

#### step

```python
step: Step[StateT, DepsT, Any, Any]

```

The step to execute.

#### inputs

```python
inputs: Any

```

The inputs bound to this step.

#### run

```python
run(
    ctx: GraphRunContext[StateT, DepsT],
) -> BaseNode[StateT, DepsT, Any] | End[Any]

```

Attempt to run the step node.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `GraphRunContext[StateT, DepsT]` | The graph execution context | *required* |

Returns:

| Type | Description | | --- | --- | | `BaseNode[StateT, DepsT, Any] | End[Any]` | The result of step execution |

Raises:

| Type | Description | | --- | --- | | `NotImplementedError` | Always raised as StepNode is not meant to be run directly |

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
    """Attempt to run the step node.

    Args:
        ctx: The graph execution context

    Returns:
        The result of step execution

    Raises:
        NotImplementedError: Always raised as StepNode is not meant to be run directly
    """
    raise NotImplementedError(
        '`StepNode` is not meant to be run directly, it is meant to be used in `BaseNode` subclasses to indicate a transition to v2-style steps.'
    )

```

### NodeStep

Bases: `Step[StateT, DepsT, Any, BaseNode[StateT, DepsT, Any] | End[Any]]`

A step that wraps a BaseNode type for execution.

NodeStep allows v1-style BaseNode classes to be used as steps in the v2 graph execution system. It validates that the input is of the expected node type and runs it with the appropriate graph context.

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
class NodeStep(Step[StateT, DepsT, Any, BaseNode[StateT, DepsT, Any] | End[Any]]):
    """A step that wraps a BaseNode type for execution.

    NodeStep allows v1-style BaseNode classes to be used as steps in the
    v2 graph execution system. It validates that the input is of the expected
    node type and runs it with the appropriate graph context.
    """

    node_type: type[BaseNode[StateT, DepsT, Any]]
    """The BaseNode type this step executes."""

    def __init__(
        self,
        node_type: type[BaseNode[StateT, DepsT, Any]],
        *,
        id: NodeID | None = None,
        label: str | None = None,
    ):
        """Initialize a node step.

        Args:
            node_type: The BaseNode class this step will execute
            id: Optional unique identifier, defaults to the node's get_node_id()
            label: Optional human-readable label for this step
        """
        super().__init__(
            id=id or NodeID(node_type.get_node_id()),
            call=self._call_node,
            label=label,
        )
        # `type[BaseNode[StateT, DepsT, Any]]` could actually be a `typing._GenericAlias` like `pydantic_ai._agent_graph.UserPromptNode[~DepsT, ~OutputT]`,
        # so we get the origin to get to the actual class
        self.node_type = get_origin(node_type) or node_type

    async def _call_node(self, ctx: StepContext[StateT, DepsT, Any]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
        """Execute the wrapped node with the step context.

        Args:
            ctx: The step context containing the node instance to run

        Returns:
            The result of running the node, either another BaseNode or End

        Raises:
            ValueError: If the input node is not of the expected type
        """
        node = ctx.inputs
        if not isinstance(node, self.node_type):
            raise ValueError(f'Node {node} is not of type {self.node_type}')  # pragma: no cover
        node = cast(BaseNode[StateT, DepsT, Any], node)
        return await node.run(GraphRunContext(state=ctx.state, deps=ctx.deps))

```

#### __init__

```python
__init__(
    node_type: type[BaseNode[StateT, DepsT, Any]],
    *,
    id: NodeID | None = None,
    label: str | None = None
)

```

Initialize a node step.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `node_type` | `type[BaseNode[StateT, DepsT, Any]]` | The BaseNode class this step will execute | *required* | | `id` | `NodeID | None` | Optional unique identifier, defaults to the node's get_node_id() | `None` | | `label` | `str | None` | Optional human-readable label for this step | `None` |

Source code in `pydantic_graph/pydantic_graph/beta/step.py`

```python
def __init__(
    self,
    node_type: type[BaseNode[StateT, DepsT, Any]],
    *,
    id: NodeID | None = None,
    label: str | None = None,
):
    """Initialize a node step.

    Args:
        node_type: The BaseNode class this step will execute
        id: Optional unique identifier, defaults to the node's get_node_id()
        label: Optional human-readable label for this step
    """
    super().__init__(
        id=id or NodeID(node_type.get_node_id()),
        call=self._call_node,
        label=label,
    )
    # `type[BaseNode[StateT, DepsT, Any]]` could actually be a `typing._GenericAlias` like `pydantic_ai._agent_graph.UserPromptNode[~DepsT, ~OutputT]`,
    # so we get the origin to get to the actual class
    self.node_type = get_origin(node_type) or node_type

```

#### node_type

```python
node_type: type[BaseNode[StateT, DepsT, Any]] = (
    get_origin(node_type) or node_type
)

```

The BaseNode type this step executes.

# `pydantic_graph.exceptions`

### GraphSetupError

Bases: `TypeError`

Error caused by an incorrectly configured graph.

Source code in `pydantic_graph/pydantic_graph/exceptions.py`

```python
class GraphSetupError(TypeError):
    """Error caused by an incorrectly configured graph."""

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

### GraphBuildingError

Bases: `ValueError`

An error raised during graph-building.

Source code in `pydantic_graph/pydantic_graph/exceptions.py`

```python
class GraphBuildingError(ValueError):
    """An error raised during graph-building."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

```

#### message

```python
message: str = message

```

The error message.

### GraphValidationError

Bases: `ValueError`

An error raised during graph validation.

Source code in `pydantic_graph/pydantic_graph/exceptions.py`

```python
class GraphValidationError(ValueError):
    """An error raised during graph validation."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

```

#### message

```python
message: str = message

```

The error message.

### GraphRuntimeError

Bases: `RuntimeError`

Error caused by an issue during graph execution.

Source code in `pydantic_graph/pydantic_graph/exceptions.py`

```python
class GraphRuntimeError(RuntimeError):
    """Error caused by an issue during graph execution."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

```

#### message

```python
message: str = message

```

The error message.

### GraphNodeStatusError

Bases: `GraphRuntimeError`

Error caused by trying to run a node that already has status `'running'`, `'success'`, or `'error'`.

Source code in `pydantic_graph/pydantic_graph/exceptions.py`

```python
class GraphNodeStatusError(GraphRuntimeError):
    """Error caused by trying to run a node that already has status `'running'`, `'success'`, or `'error'`."""

    def __init__(self, actual_status: 'SnapshotStatus'):
        self.actual_status = actual_status
        super().__init__(f"Incorrect snapshot status {actual_status!r}, must be 'created' or 'pending'.")

    @classmethod
    def check(cls, status: 'SnapshotStatus') -> None:
        """Check if the status is valid."""
        if status not in {'created', 'pending'}:
            raise cls(status)

```

#### check

```python
check(status: SnapshotStatus) -> None

```

Check if the status is valid.

Source code in `pydantic_graph/pydantic_graph/exceptions.py`

```python
@classmethod
def check(cls, status: 'SnapshotStatus') -> None:
    """Check if the status is valid."""
    if status not in {'created', 'pending'}:
        raise cls(status)

```

# `pydantic_graph`

### Graph

Bases: `Generic[StateT, DepsT, RunEndT]`

Definition of a graph.

In `pydantic-graph`, a graph is a collection of nodes that can be run in sequence. The nodes define their outgoing edges â€” e.g. which nodes may be run next, and thereby the structure of the graph.

Here's a very simple example of a graph which increments a number by 1, but makes sure the number is never 42 at the end.

never_42.py

```py
from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass
class MyState:
    number: int

@dataclass
class Increment(BaseNode[MyState]):
    async def run(self, ctx: GraphRunContext) -> Check42:
        ctx.state.number += 1
        return Check42()

@dataclass
class Check42(BaseNode[MyState, None, int]):
    async def run(self, ctx: GraphRunContext) -> Increment | End[int]:
        if ctx.state.number == 42:
            return Increment()
        else:
            return End(ctx.state.number)

never_42_graph = Graph(nodes=(Increment, Check42))

```

*(This example is complete, it can be run "as is")*

See run For an example of running graph, and mermaid_code for an example of generating a mermaid diagram from the graph.

Source code in `pydantic_graph/pydantic_graph/graph.py`

````python
@dataclass(init=False)
class Graph(Generic[StateT, DepsT, RunEndT]):
    """Definition of a graph.

    In `pydantic-graph`, a graph is a collection of nodes that can be run in sequence. The nodes define
    their outgoing edges â€” e.g. which nodes may be run next, and thereby the structure of the graph.

    Here's a very simple example of a graph which increments a number by 1, but makes sure the number is never
    42 at the end.

    ```py {title="never_42.py" noqa="I001"}
    from __future__ import annotations

    from dataclasses import dataclass

    from pydantic_graph import BaseNode, End, Graph, GraphRunContext

    @dataclass
    class MyState:
        number: int

    @dataclass
    class Increment(BaseNode[MyState]):
        async def run(self, ctx: GraphRunContext) -> Check42:
            ctx.state.number += 1
            return Check42()

    @dataclass
    class Check42(BaseNode[MyState, None, int]):
        async def run(self, ctx: GraphRunContext) -> Increment | End[int]:
            if ctx.state.number == 42:
                return Increment()
            else:
                return End(ctx.state.number)

    never_42_graph = Graph(nodes=(Increment, Check42))
    ```
    _(This example is complete, it can be run "as is")_

    See [`run`][pydantic_graph.graph.Graph.run] For an example of running graph, and
    [`mermaid_code`][pydantic_graph.graph.Graph.mermaid_code] for an example of generating a mermaid diagram
    from the graph.
    """

    name: str | None
    node_defs: dict[str, NodeDef[StateT, DepsT, RunEndT]]
    _state_type: type[StateT] | _utils.Unset = field(repr=False)
    _run_end_type: type[RunEndT] | _utils.Unset = field(repr=False)
    auto_instrument: bool = field(repr=False)

    def __init__(
        self,
        *,
        nodes: Sequence[type[BaseNode[StateT, DepsT, RunEndT]]],
        name: str | None = None,
        state_type: type[StateT] | _utils.Unset = _utils.UNSET,
        run_end_type: type[RunEndT] | _utils.Unset = _utils.UNSET,
        auto_instrument: bool = True,
    ):
        """Create a graph from a sequence of nodes.

        Args:
            nodes: The nodes which make up the graph, nodes need to be unique and all be generic in the same
                state type.
            name: Optional name for the graph, if not provided the name will be inferred from the calling frame
                on the first call to a graph method.
            state_type: The type of the state for the graph, this can generally be inferred from `nodes`.
            run_end_type: The type of the result of running the graph, this can generally be inferred from `nodes`.
            auto_instrument: Whether to create a span for the graph run and the execution of each node's run method.
        """
        self.name = name
        self._state_type = state_type
        self._run_end_type = run_end_type
        self.auto_instrument = auto_instrument

        parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
        self.node_defs = {}
        for node in nodes:
            self._register_node(node, parent_namespace)

        self._validate_edges()

    async def run(
        self,
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        state: StateT = None,
        deps: DepsT = None,
        persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
        infer_name: bool = True,
    ) -> GraphRunResult[StateT, RunEndT]:
        """Run the graph from a starting node until it ends.

        Args:
            start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            A `GraphRunResult` containing information about the run, including its final result.

        Here's an example of running the graph from [above][pydantic_graph.graph.Graph]:

        ```py {title="run_never_42.py" noqa="I001" requires="never_42.py"}
        from never_42 import Increment, MyState, never_42_graph

        async def main():
            state = MyState(1)
            await never_42_graph.run(Increment(), state=state)
            print(state)
            #> MyState(number=2)

            state = MyState(41)
            await never_42_graph.run(Increment(), state=state)
            print(state)
            #> MyState(number=43)
        ```
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        async with self.iter(
            start_node, state=state, deps=deps, persistence=persistence, infer_name=False
        ) as graph_run:
            async for _node in graph_run:
                pass

        result = graph_run.result
        assert result is not None, 'GraphRun should have a result'
        return result

    def run_sync(
        self,
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        state: StateT = None,
        deps: DepsT = None,
        persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
        infer_name: bool = True,
    ) -> GraphRunResult[StateT, RunEndT]:
        """Synchronously run the graph.

        This is a convenience method that wraps [`self.run`][pydantic_graph.Graph.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Args:
            start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The result type from ending the run and the history of the run.
        """
        if infer_name and self.name is None:  # pragma: no branch
            self._infer_name(inspect.currentframe())

        return _utils.get_event_loop().run_until_complete(
            self.run(start_node, state=state, deps=deps, persistence=persistence, infer_name=False)
        )

    @asynccontextmanager
    async def iter(
        self,
        start_node: BaseNode[StateT, DepsT, RunEndT],
        *,
        state: StateT = None,
        deps: DepsT = None,
        persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
        span: AbstractContextManager[AbstractSpan] | None = None,
        infer_name: bool = True,
    ) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
        """A contextmanager which can be used to iterate over the graph's nodes as they are executed.

        This method returns a `GraphRun` object which can be used to async-iterate over the nodes of this `Graph` as
        they are executed. This is the API to use if you want to record or interact with the nodes as the graph
        execution unfolds.

        The `GraphRun` can also be used to manually drive the graph execution by calling
        [`GraphRun.next`][pydantic_graph.graph.GraphRun.next].

        The `GraphRun` provides access to the full run history, state, deps, and the final result of the run once
        it has completed.

        For more details, see the API documentation of [`GraphRun`][pydantic_graph.graph.GraphRun].

        Args:
            start_node: the first node to run. Since the graph definition doesn't define the entry point in the graph,
                you need to provide the starting node.
            state: The initial state of the graph.
            deps: The dependencies of the graph.
            persistence: State persistence interface, defaults to
                [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
            span: The span to use for the graph run. If not provided, a new span will be created.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns: A GraphRun that can be async iterated over to drive the graph to completion.
        """
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        if persistence is None:
            persistence = SimpleStatePersistence()
        persistence.set_graph_types(self)

        with ExitStack() as stack:
            entered_span: AbstractSpan | None = None
            if span is None:
                if self.auto_instrument:  # pragma: no branch
                    # Separate variable because we actually don't want logfire's f-string magic here,
                    # we want the span_name to be preformatted for other backends
                    # as requested in https://github.com/pydantic/pydantic-ai/issues/3173.
                    span_name = f'run graph {self.name}'
                    entered_span = stack.enter_context(logfire_span(span_name, graph=self))
            else:
                entered_span = stack.enter_context(span)
            traceparent = None if entered_span is None else get_traceparent(entered_span)
            yield GraphRun[StateT, DepsT, RunEndT](
                graph=self,
                start_node=start_node,
                persistence=persistence,
                state=state,
                deps=deps,
                traceparent=traceparent,
            )

    @asynccontextmanager
    async def iter_from_persistence(
        self,
        persistence: BaseStatePersistence[StateT, RunEndT],
        *,
        deps: DepsT = None,
        span: AbstractContextManager[AbstractSpan] | None = None,
        infer_name: bool = True,
    ) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
        """A contextmanager to iterate over the graph's nodes as they are executed, created from a persistence object.

        This method has similar functionality to [`iter`][pydantic_graph.graph.Graph.iter],
        but instead of passing the node to run, it will restore the node and state from state persistence.

        Args:
            persistence: The state persistence interface to use.
            deps: The dependencies of the graph.
            span: The span to use for the graph run. If not provided, a new span will be created.
            infer_name: Whether to infer the graph name from the calling frame.

        Returns: A GraphRun that can be async iterated over to drive the graph to completion.
        """
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        persistence.set_graph_types(self)

        snapshot = await persistence.load_next()
        if snapshot is None:
            raise exceptions.GraphRuntimeError('Unable to restore snapshot from state persistence.')

        snapshot.node.set_snapshot_id(snapshot.id)

        if self.auto_instrument and span is None:  # pragma: no branch
            span = logfire_span('run graph {graph.name}', graph=self)

        with ExitStack() as stack:
            entered_span = None if span is None else stack.enter_context(span)
            traceparent = None if entered_span is None else get_traceparent(entered_span)
            yield GraphRun[StateT, DepsT, RunEndT](
                graph=self,
                start_node=snapshot.node,
                persistence=persistence,
                state=snapshot.state,
                deps=deps,
                snapshot_id=snapshot.id,
                traceparent=traceparent,
            )

    async def initialize(
        self,
        node: BaseNode[StateT, DepsT, RunEndT],
        persistence: BaseStatePersistence[StateT, RunEndT],
        *,
        state: StateT = None,
        infer_name: bool = True,
    ) -> None:
        """Initialize a new graph run in persistence without running it.

        This is useful if you want to set up a graph run to be run later, e.g. via
        [`iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence].

        Args:
            node: The node to run first.
            persistence: State persistence interface.
            state: The start state of the graph.
            infer_name: Whether to infer the graph name from the calling frame.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        persistence.set_graph_types(self)
        await persistence.snapshot_node(state, node)

    def mermaid_code(
        self,
        *,
        start_node: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
        title: str | None | typing_extensions.Literal[False] = None,
        edge_labels: bool = True,
        notes: bool = True,
        highlighted_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
        highlight_css: str = mermaid.DEFAULT_HIGHLIGHT_CSS,
        infer_name: bool = True,
        direction: mermaid.StateDiagramDirection | None = None,
    ) -> str:
        """Generate a diagram representing the graph as [mermaid](https://mermaid.js.org/) diagram.

        This method calls [`pydantic_graph.mermaid.generate_code`][pydantic_graph.mermaid.generate_code].

        Args:
            start_node: The node or nodes which can start the graph.
            title: The title of the diagram, use `False` to not include a title.
            edge_labels: Whether to include edge labels.
            notes: Whether to include notes on each node.
            highlighted_nodes: Optional node or nodes to highlight.
            highlight_css: The CSS to use for highlighting nodes.
            infer_name: Whether to infer the graph name from the calling frame.
            direction: The direction of flow.

        Returns:
            The mermaid code for the graph, which can then be rendered as a diagram.

        Here's an example of generating a diagram for the graph from [above][pydantic_graph.graph.Graph]:

        ```py {title="mermaid_never_42.py" requires="never_42.py"}
        from never_42 import Increment, never_42_graph

        print(never_42_graph.mermaid_code(start_node=Increment))
        '''
        ---
        title: never_42_graph
        ---
        stateDiagram-v2
          [*] --> Increment
          Increment --> Check42
          Check42 --> Increment
          Check42 --> [*]
        '''
        ```

        The rendered diagram will look like this:

        ```mermaid
        ---
        title: never_42_graph
        ---
        stateDiagram-v2
          [*] --> Increment
          Increment --> Check42
          Check42 --> Increment
          Check42 --> [*]
        ```
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        if title is None and self.name:
            title = self.name
        return mermaid.generate_code(
            self,
            start_node=start_node,
            highlighted_nodes=highlighted_nodes,
            highlight_css=highlight_css,
            title=title or None,
            edge_labels=edge_labels,
            notes=notes,
            direction=direction,
        )

    def mermaid_image(
        self, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
    ) -> bytes:
        """Generate a diagram representing the graph as an image.

        The format and diagram can be customized using `kwargs`,
        see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].

        !!! note "Uses external service"
            This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
            is a free service not affiliated with Pydantic.

        Args:
            infer_name: Whether to infer the graph name from the calling frame.
            **kwargs: Additional arguments to pass to `mermaid.request_image`.

        Returns:
            The image bytes.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        if 'title' not in kwargs and self.name:
            kwargs['title'] = self.name
        return mermaid.request_image(self, **kwargs)

    def mermaid_save(
        self, path: Path | str, /, *, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
    ) -> None:
        """Generate a diagram representing the graph and save it as an image.

        The format and diagram can be customized using `kwargs`,
        see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].

        !!! note "Uses external service"
            This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
            is a free service not affiliated with Pydantic.

        Args:
            path: The path to save the image to.
            infer_name: Whether to infer the graph name from the calling frame.
            **kwargs: Additional arguments to pass to `mermaid.save_image`.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        if 'title' not in kwargs and self.name:
            kwargs['title'] = self.name
        mermaid.save_image(path, self, **kwargs)

    def get_nodes(self) -> Sequence[type[BaseNode[StateT, DepsT, RunEndT]]]:
        """Get the nodes in the graph."""
        return [node_def.node for node_def in self.node_defs.values()]

    @cached_property
    def inferred_types(self) -> tuple[type[StateT], type[RunEndT]]:
        # Get the types of the state and run end from the graph.
        if _utils.is_set(self._state_type) and _utils.is_set(self._run_end_type):
            return self._state_type, self._run_end_type

        state_type = self._state_type
        run_end_type = self._run_end_type

        for node_def in self.node_defs.values():
            for base in typing_extensions.get_original_bases(node_def.node):
                if typing_extensions.get_origin(base) is BaseNode:
                    args = typing_extensions.get_args(base)
                    if not _utils.is_set(state_type) and args:
                        state_type = args[0]

                    if not _utils.is_set(run_end_type) and len(args) == 3:
                        t = args[2]
                        if not typing_objects.is_never(t):
                            run_end_type = t
                    if _utils.is_set(state_type) and _utils.is_set(run_end_type):
                        return state_type, run_end_type  # pyright: ignore[reportReturnType]
                    # break the inner (bases) loop
                    break

        if not _utils.is_set(state_type):  # pragma: no branch
            # state defaults to None, so use that if we can't infer it
            state_type = None
        if not _utils.is_set(run_end_type):
            # this happens if a graph has no return nodes, use None so any downstream errors are clear
            run_end_type = None
        return state_type, run_end_type  # pyright: ignore[reportReturnType]

    def _register_node(
        self,
        node: type[BaseNode[StateT, DepsT, RunEndT]],
        parent_namespace: dict[str, Any] | None,
    ) -> None:
        node_id = node.get_node_id()
        if existing_node := self.node_defs.get(node_id):
            raise exceptions.GraphSetupError(
                f'Node ID `{node_id}` is not unique â€” found on {existing_node.node} and {node}'
            )
        else:
            self.node_defs[node_id] = node.get_node_def(parent_namespace)

    def _validate_edges(self):
        known_node_ids = self.node_defs.keys()
        bad_edges: dict[str, list[str]] = {}

        for node_id, node_def in self.node_defs.items():
            for edge in node_def.next_node_edges.keys():
                if edge not in known_node_ids:
                    bad_edges.setdefault(edge, []).append(f'`{node_id}`')

        if bad_edges:
            bad_edges_list = [f'`{k}` is referenced by {_utils.comma_and(v)}' for k, v in bad_edges.items()]
            if len(bad_edges_list) == 1:
                raise exceptions.GraphSetupError(f'{bad_edges_list[0]} but not included in the graph.')
            else:
                b = '\n'.join(f' {be}' for be in bad_edges_list)
                raise exceptions.GraphSetupError(
                    f'Nodes are referenced in the graph but not included in the graph:\n{b}'
                )

    def _infer_name(self, function_frame: types.FrameType | None) -> None:
        """Infer the agent name from the call frame.

        Usage should be `self._infer_name(inspect.currentframe())`.

        Copied from `Agent`.
        """
        assert self.name is None, 'Name already set'
        if function_frame is not None and (parent_frame := function_frame.f_back):  # pragma: no branch
            for name, item in parent_frame.f_locals.items():
                if item is self:
                    self.name = name
                    return
            if parent_frame.f_locals != parent_frame.f_globals:  # pragma: no branch
                # if we couldn't find the agent in locals and globals are a different dict, try globals
                for name, item in parent_frame.f_globals.items():  # pragma: no branch
                    if item is self:
                        self.name = name
                        return

````

#### __init__

```python
__init__(
    *,
    nodes: Sequence[type[BaseNode[StateT, DepsT, RunEndT]]],
    name: str | None = None,
    state_type: type[StateT] | Unset = UNSET,
    run_end_type: type[RunEndT] | Unset = UNSET,
    auto_instrument: bool = True
)

```

Create a graph from a sequence of nodes.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `nodes` | `Sequence[type[BaseNode[StateT, DepsT, RunEndT]]]` | The nodes which make up the graph, nodes need to be unique and all be generic in the same state type. | *required* | | `name` | `str | None` | Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method. | `None` | | `state_type` | `type[StateT] | Unset` | The type of the state for the graph, this can generally be inferred from nodes. | `UNSET` | | `run_end_type` | `type[RunEndT] | Unset` | The type of the result of running the graph, this can generally be inferred from nodes. | `UNSET` | | `auto_instrument` | `bool` | Whether to create a span for the graph run and the execution of each node's run method. | `True` |

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
def __init__(
    self,
    *,
    nodes: Sequence[type[BaseNode[StateT, DepsT, RunEndT]]],
    name: str | None = None,
    state_type: type[StateT] | _utils.Unset = _utils.UNSET,
    run_end_type: type[RunEndT] | _utils.Unset = _utils.UNSET,
    auto_instrument: bool = True,
):
    """Create a graph from a sequence of nodes.

    Args:
        nodes: The nodes which make up the graph, nodes need to be unique and all be generic in the same
            state type.
        name: Optional name for the graph, if not provided the name will be inferred from the calling frame
            on the first call to a graph method.
        state_type: The type of the state for the graph, this can generally be inferred from `nodes`.
        run_end_type: The type of the result of running the graph, this can generally be inferred from `nodes`.
        auto_instrument: Whether to create a span for the graph run and the execution of each node's run method.
    """
    self.name = name
    self._state_type = state_type
    self._run_end_type = run_end_type
    self.auto_instrument = auto_instrument

    parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
    self.node_defs = {}
    for node in nodes:
        self._register_node(node, parent_namespace)

    self._validate_edges()

```

#### run

```python
run(
    start_node: BaseNode[StateT, DepsT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    persistence: (
        BaseStatePersistence[StateT, RunEndT] | None
    ) = None,
    infer_name: bool = True
) -> GraphRunResult[StateT, RunEndT]

```

Run the graph from a starting node until it ends.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `start_node` | `BaseNode[StateT, DepsT, RunEndT]` | the first node to run, since the graph definition doesn't define the entry point in the graph, you need to provide the starting node. | *required* | | `state` | `StateT` | The initial state of the graph. | `None` | | `deps` | `DepsT` | The dependencies of the graph. | `None` | | `persistence` | `BaseStatePersistence[StateT, RunEndT] | None` | State persistence interface, defaults to SimpleStatePersistence if None. | `None` | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` |

Returns:

| Type | Description | | --- | --- | | `GraphRunResult[StateT, RunEndT]` | A GraphRunResult containing information about the run, including its final result. |

Here's an example of running the graph from above:

run_never_42.py

```py
from never_42 import Increment, MyState, never_42_graph

async def main():
    state = MyState(1)
    await never_42_graph.run(Increment(), state=state)
    print(state)
    #> MyState(number=2)

    state = MyState(41)
    await never_42_graph.run(Increment(), state=state)
    print(state)
    #> MyState(number=43)

```

Source code in `pydantic_graph/pydantic_graph/graph.py`

````python
async def run(
    self,
    start_node: BaseNode[StateT, DepsT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
    infer_name: bool = True,
) -> GraphRunResult[StateT, RunEndT]:
    """Run the graph from a starting node until it ends.

    Args:
        start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
            you need to provide the starting node.
        state: The initial state of the graph.
        deps: The dependencies of the graph.
        persistence: State persistence interface, defaults to
            [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
        infer_name: Whether to infer the graph name from the calling frame.

    Returns:
        A `GraphRunResult` containing information about the run, including its final result.

    Here's an example of running the graph from [above][pydantic_graph.graph.Graph]:

    ```py {title="run_never_42.py" noqa="I001" requires="never_42.py"}
    from never_42 import Increment, MyState, never_42_graph

    async def main():
        state = MyState(1)
        await never_42_graph.run(Increment(), state=state)
        print(state)
        #> MyState(number=2)

        state = MyState(41)
        await never_42_graph.run(Increment(), state=state)
        print(state)
        #> MyState(number=43)
    ```
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())

    async with self.iter(
        start_node, state=state, deps=deps, persistence=persistence, infer_name=False
    ) as graph_run:
        async for _node in graph_run:
            pass

    result = graph_run.result
    assert result is not None, 'GraphRun should have a result'
    return result

````

#### run_sync

```python
run_sync(
    start_node: BaseNode[StateT, DepsT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    persistence: (
        BaseStatePersistence[StateT, RunEndT] | None
    ) = None,
    infer_name: bool = True
) -> GraphRunResult[StateT, RunEndT]

```

Synchronously run the graph.

This is a convenience method that wraps self.run with `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `start_node` | `BaseNode[StateT, DepsT, RunEndT]` | the first node to run, since the graph definition doesn't define the entry point in the graph, you need to provide the starting node. | *required* | | `state` | `StateT` | The initial state of the graph. | `None` | | `deps` | `DepsT` | The dependencies of the graph. | `None` | | `persistence` | `BaseStatePersistence[StateT, RunEndT] | None` | State persistence interface, defaults to SimpleStatePersistence if None. | `None` | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` |

Returns:

| Type | Description | | --- | --- | | `GraphRunResult[StateT, RunEndT]` | The result type from ending the run and the history of the run. |

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
def run_sync(
    self,
    start_node: BaseNode[StateT, DepsT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
    infer_name: bool = True,
) -> GraphRunResult[StateT, RunEndT]:
    """Synchronously run the graph.

    This is a convenience method that wraps [`self.run`][pydantic_graph.Graph.run] with `loop.run_until_complete(...)`.
    You therefore can't use this method inside async code or if there's an active event loop.

    Args:
        start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
            you need to provide the starting node.
        state: The initial state of the graph.
        deps: The dependencies of the graph.
        persistence: State persistence interface, defaults to
            [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
        infer_name: Whether to infer the graph name from the calling frame.

    Returns:
        The result type from ending the run and the history of the run.
    """
    if infer_name and self.name is None:  # pragma: no branch
        self._infer_name(inspect.currentframe())

    return _utils.get_event_loop().run_until_complete(
        self.run(start_node, state=state, deps=deps, persistence=persistence, infer_name=False)
    )

```

#### iter

```python
iter(
    start_node: BaseNode[StateT, DepsT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    persistence: (
        BaseStatePersistence[StateT, RunEndT] | None
    ) = None,
    span: (
        AbstractContextManager[AbstractSpan] | None
    ) = None,
    infer_name: bool = True
) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]

```

A contextmanager which can be used to iterate over the graph's nodes as they are executed.

This method returns a `GraphRun` object which can be used to async-iterate over the nodes of this `Graph` as they are executed. This is the API to use if you want to record or interact with the nodes as the graph execution unfolds.

The `GraphRun` can also be used to manually drive the graph execution by calling GraphRun.next.

The `GraphRun` provides access to the full run history, state, deps, and the final result of the run once it has completed.

For more details, see the API documentation of GraphRun.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `start_node` | `BaseNode[StateT, DepsT, RunEndT]` | the first node to run. Since the graph definition doesn't define the entry point in the graph, you need to provide the starting node. | *required* | | `state` | `StateT` | The initial state of the graph. | `None` | | `deps` | `DepsT` | The dependencies of the graph. | `None` | | `persistence` | `BaseStatePersistence[StateT, RunEndT] | None` | State persistence interface, defaults to SimpleStatePersistence if None. | `None` | | `span` | `AbstractContextManager[AbstractSpan] | None` | The span to use for the graph run. If not provided, a new span will be created. | `None` | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` |

Returns: A GraphRun that can be async iterated over to drive the graph to completion.

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
@asynccontextmanager
async def iter(
    self,
    start_node: BaseNode[StateT, DepsT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
    span: AbstractContextManager[AbstractSpan] | None = None,
    infer_name: bool = True,
) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
    """A contextmanager which can be used to iterate over the graph's nodes as they are executed.

    This method returns a `GraphRun` object which can be used to async-iterate over the nodes of this `Graph` as
    they are executed. This is the API to use if you want to record or interact with the nodes as the graph
    execution unfolds.

    The `GraphRun` can also be used to manually drive the graph execution by calling
    [`GraphRun.next`][pydantic_graph.graph.GraphRun.next].

    The `GraphRun` provides access to the full run history, state, deps, and the final result of the run once
    it has completed.

    For more details, see the API documentation of [`GraphRun`][pydantic_graph.graph.GraphRun].

    Args:
        start_node: the first node to run. Since the graph definition doesn't define the entry point in the graph,
            you need to provide the starting node.
        state: The initial state of the graph.
        deps: The dependencies of the graph.
        persistence: State persistence interface, defaults to
            [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
        span: The span to use for the graph run. If not provided, a new span will be created.
        infer_name: Whether to infer the graph name from the calling frame.

    Returns: A GraphRun that can be async iterated over to drive the graph to completion.
    """
    if infer_name and self.name is None:
        # f_back because `asynccontextmanager` adds one frame
        if frame := inspect.currentframe():  # pragma: no branch
            self._infer_name(frame.f_back)

    if persistence is None:
        persistence = SimpleStatePersistence()
    persistence.set_graph_types(self)

    with ExitStack() as stack:
        entered_span: AbstractSpan | None = None
        if span is None:
            if self.auto_instrument:  # pragma: no branch
                # Separate variable because we actually don't want logfire's f-string magic here,
                # we want the span_name to be preformatted for other backends
                # as requested in https://github.com/pydantic/pydantic-ai/issues/3173.
                span_name = f'run graph {self.name}'
                entered_span = stack.enter_context(logfire_span(span_name, graph=self))
        else:
            entered_span = stack.enter_context(span)
        traceparent = None if entered_span is None else get_traceparent(entered_span)
        yield GraphRun[StateT, DepsT, RunEndT](
            graph=self,
            start_node=start_node,
            persistence=persistence,
            state=state,
            deps=deps,
            traceparent=traceparent,
        )

```

#### iter_from_persistence

```python
iter_from_persistence(
    persistence: BaseStatePersistence[StateT, RunEndT],
    *,
    deps: DepsT = None,
    span: (
        AbstractContextManager[AbstractSpan] | None
    ) = None,
    infer_name: bool = True
) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]

```

A contextmanager to iterate over the graph's nodes as they are executed, created from a persistence object.

This method has similar functionality to iter, but instead of passing the node to run, it will restore the node and state from state persistence.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `persistence` | `BaseStatePersistence[StateT, RunEndT]` | The state persistence interface to use. | *required* | | `deps` | `DepsT` | The dependencies of the graph. | `None` | | `span` | `AbstractContextManager[AbstractSpan] | None` | The span to use for the graph run. If not provided, a new span will be created. | `None` | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` |

Returns: A GraphRun that can be async iterated over to drive the graph to completion.

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
@asynccontextmanager
async def iter_from_persistence(
    self,
    persistence: BaseStatePersistence[StateT, RunEndT],
    *,
    deps: DepsT = None,
    span: AbstractContextManager[AbstractSpan] | None = None,
    infer_name: bool = True,
) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
    """A contextmanager to iterate over the graph's nodes as they are executed, created from a persistence object.

    This method has similar functionality to [`iter`][pydantic_graph.graph.Graph.iter],
    but instead of passing the node to run, it will restore the node and state from state persistence.

    Args:
        persistence: The state persistence interface to use.
        deps: The dependencies of the graph.
        span: The span to use for the graph run. If not provided, a new span will be created.
        infer_name: Whether to infer the graph name from the calling frame.

    Returns: A GraphRun that can be async iterated over to drive the graph to completion.
    """
    if infer_name and self.name is None:
        # f_back because `asynccontextmanager` adds one frame
        if frame := inspect.currentframe():  # pragma: no branch
            self._infer_name(frame.f_back)

    persistence.set_graph_types(self)

    snapshot = await persistence.load_next()
    if snapshot is None:
        raise exceptions.GraphRuntimeError('Unable to restore snapshot from state persistence.')

    snapshot.node.set_snapshot_id(snapshot.id)

    if self.auto_instrument and span is None:  # pragma: no branch
        span = logfire_span('run graph {graph.name}', graph=self)

    with ExitStack() as stack:
        entered_span = None if span is None else stack.enter_context(span)
        traceparent = None if entered_span is None else get_traceparent(entered_span)
        yield GraphRun[StateT, DepsT, RunEndT](
            graph=self,
            start_node=snapshot.node,
            persistence=persistence,
            state=snapshot.state,
            deps=deps,
            snapshot_id=snapshot.id,
            traceparent=traceparent,
        )

```

#### initialize

```python
initialize(
    node: BaseNode[StateT, DepsT, RunEndT],
    persistence: BaseStatePersistence[StateT, RunEndT],
    *,
    state: StateT = None,
    infer_name: bool = True
) -> None

```

Initialize a new graph run in persistence without running it.

This is useful if you want to set up a graph run to be run later, e.g. via iter_from_persistence.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `node` | `BaseNode[StateT, DepsT, RunEndT]` | The node to run first. | *required* | | `persistence` | `BaseStatePersistence[StateT, RunEndT]` | State persistence interface. | *required* | | `state` | `StateT` | The start state of the graph. | `None` | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` |

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
async def initialize(
    self,
    node: BaseNode[StateT, DepsT, RunEndT],
    persistence: BaseStatePersistence[StateT, RunEndT],
    *,
    state: StateT = None,
    infer_name: bool = True,
) -> None:
    """Initialize a new graph run in persistence without running it.

    This is useful if you want to set up a graph run to be run later, e.g. via
    [`iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence].

    Args:
        node: The node to run first.
        persistence: State persistence interface.
        state: The start state of the graph.
        infer_name: Whether to infer the graph name from the calling frame.
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())

    persistence.set_graph_types(self)
    await persistence.snapshot_node(state, node)

```

#### mermaid_code

```python
mermaid_code(
    *,
    start_node: (
        Sequence[NodeIdent] | NodeIdent | None
    ) = None,
    title: str | None | Literal[False] = None,
    edge_labels: bool = True,
    notes: bool = True,
    highlighted_nodes: (
        Sequence[NodeIdent] | NodeIdent | None
    ) = None,
    highlight_css: str = DEFAULT_HIGHLIGHT_CSS,
    infer_name: bool = True,
    direction: StateDiagramDirection | None = None
) -> str

```

Generate a diagram representing the graph as [mermaid](https://mermaid.js.org/) diagram.

This method calls pydantic_graph.mermaid.generate_code.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `start_node` | `Sequence[NodeIdent] | NodeIdent | None` | The node or nodes which can start the graph. | `None` | | `title` | `str | None | Literal[False]` | The title of the diagram, use False to not include a title. | `None` | | `edge_labels` | `bool` | Whether to include edge labels. | `True` | | `notes` | `bool` | Whether to include notes on each node. | `True` | | `highlighted_nodes` | `Sequence[NodeIdent] | NodeIdent | None` | Optional node or nodes to highlight. | `None` | | `highlight_css` | `str` | The CSS to use for highlighting nodes. | `DEFAULT_HIGHLIGHT_CSS` | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` | | `direction` | `StateDiagramDirection | None` | The direction of flow. | `None` |

Returns:

| Type | Description | | --- | --- | | `str` | The mermaid code for the graph, which can then be rendered as a diagram. |

Here's an example of generating a diagram for the graph from above:

mermaid_never_42.py

```py
from never_42 import Increment, never_42_graph

print(never_42_graph.mermaid_code(start_node=Increment))
'''
---
title: never_42_graph
---
stateDiagram-v2
  [*] --> Increment
  Increment --> Check42
  Check42 --> Increment
  Check42 --> [*]
'''

```

The rendered diagram will look like this:

```
---
title: never_42_graph
---
stateDiagram-v2
  [*] --> Increment
  Increment --> Check42
  Check42 --> Increment
  Check42 --> [*]
```

Source code in `pydantic_graph/pydantic_graph/graph.py`

````python
def mermaid_code(
    self,
    *,
    start_node: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
    title: str | None | typing_extensions.Literal[False] = None,
    edge_labels: bool = True,
    notes: bool = True,
    highlighted_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
    highlight_css: str = mermaid.DEFAULT_HIGHLIGHT_CSS,
    infer_name: bool = True,
    direction: mermaid.StateDiagramDirection | None = None,
) -> str:
    """Generate a diagram representing the graph as [mermaid](https://mermaid.js.org/) diagram.

    This method calls [`pydantic_graph.mermaid.generate_code`][pydantic_graph.mermaid.generate_code].

    Args:
        start_node: The node or nodes which can start the graph.
        title: The title of the diagram, use `False` to not include a title.
        edge_labels: Whether to include edge labels.
        notes: Whether to include notes on each node.
        highlighted_nodes: Optional node or nodes to highlight.
        highlight_css: The CSS to use for highlighting nodes.
        infer_name: Whether to infer the graph name from the calling frame.
        direction: The direction of flow.

    Returns:
        The mermaid code for the graph, which can then be rendered as a diagram.

    Here's an example of generating a diagram for the graph from [above][pydantic_graph.graph.Graph]:

    ```py {title="mermaid_never_42.py" requires="never_42.py"}
    from never_42 import Increment, never_42_graph

    print(never_42_graph.mermaid_code(start_node=Increment))
    '''
    ---
    title: never_42_graph
    ---
    stateDiagram-v2
      [*] --> Increment
      Increment --> Check42
      Check42 --> Increment
      Check42 --> [*]
    '''
    ```

    The rendered diagram will look like this:

    ```mermaid
    ---
    title: never_42_graph
    ---
    stateDiagram-v2
      [*] --> Increment
      Increment --> Check42
      Check42 --> Increment
      Check42 --> [*]
    ```
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())
    if title is None and self.name:
        title = self.name
    return mermaid.generate_code(
        self,
        start_node=start_node,
        highlighted_nodes=highlighted_nodes,
        highlight_css=highlight_css,
        title=title or None,
        edge_labels=edge_labels,
        notes=notes,
        direction=direction,
    )

````

#### mermaid_image

```python
mermaid_image(
    infer_name: bool = True, **kwargs: Unpack[MermaidConfig]
) -> bytes

```

Generate a diagram representing the graph as an image.

The format and diagram can be customized using `kwargs`, see pydantic_graph.mermaid.MermaidConfig.

Uses external service

This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink` is a free service not affiliated with Pydantic.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` | | `**kwargs` | `Unpack[MermaidConfig]` | Additional arguments to pass to mermaid.request_image. | `{}` |

Returns:

| Type | Description | | --- | --- | | `bytes` | The image bytes. |

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
def mermaid_image(
    self, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
) -> bytes:
    """Generate a diagram representing the graph as an image.

    The format and diagram can be customized using `kwargs`,
    see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].

    !!! note "Uses external service"
        This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
        is a free service not affiliated with Pydantic.

    Args:
        infer_name: Whether to infer the graph name from the calling frame.
        **kwargs: Additional arguments to pass to `mermaid.request_image`.

    Returns:
        The image bytes.
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())
    if 'title' not in kwargs and self.name:
        kwargs['title'] = self.name
    return mermaid.request_image(self, **kwargs)

```

#### mermaid_save

```python
mermaid_save(
    path: Path | str,
    /,
    *,
    infer_name: bool = True,
    **kwargs: Unpack[MermaidConfig],
) -> None

```

Generate a diagram representing the graph and save it as an image.

The format and diagram can be customized using `kwargs`, see pydantic_graph.mermaid.MermaidConfig.

Uses external service

This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink` is a free service not affiliated with Pydantic.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `path` | `Path | str` | The path to save the image to. | *required* | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` | | `**kwargs` | `Unpack[MermaidConfig]` | Additional arguments to pass to mermaid.save_image. | `{}` |

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
def mermaid_save(
    self, path: Path | str, /, *, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
) -> None:
    """Generate a diagram representing the graph and save it as an image.

    The format and diagram can be customized using `kwargs`,
    see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].

    !!! note "Uses external service"
        This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
        is a free service not affiliated with Pydantic.

    Args:
        path: The path to save the image to.
        infer_name: Whether to infer the graph name from the calling frame.
        **kwargs: Additional arguments to pass to `mermaid.save_image`.
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())
    if 'title' not in kwargs and self.name:
        kwargs['title'] = self.name
    mermaid.save_image(path, self, **kwargs)

```

#### get_nodes

```python
get_nodes() -> (
    Sequence[type[BaseNode[StateT, DepsT, RunEndT]]]
)

```

Get the nodes in the graph.

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
def get_nodes(self) -> Sequence[type[BaseNode[StateT, DepsT, RunEndT]]]:
    """Get the nodes in the graph."""
    return [node_def.node for node_def in self.node_defs.values()]

```

### GraphRun

Bases: `Generic[StateT, DepsT, RunEndT]`

A stateful, async-iterable run of a Graph.

You typically get a `GraphRun` instance from calling `async with [my_graph.iter(...)][pydantic_graph.graph.Graph.iter] as graph_run:`. That gives you the ability to iterate through nodes as they run, either by `async for` iteration or by repeatedly calling `.next(...)`.

Here's an example of iterating over the graph from above: iter_never_42.py

```py
from copy import deepcopy
from never_42 import Increment, MyState, never_42_graph

async def main():
    state = MyState(1)
    async with never_42_graph.iter(Increment(), state=state) as graph_run:
        node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
        async for node in graph_run:
            node_states.append((node, deepcopy(graph_run.state)))
        print(node_states)
        '''
        [
            (Increment(), MyState(number=1)),
            (Increment(), MyState(number=1)),
            (Check42(), MyState(number=2)),
            (End(data=2), MyState(number=2)),
        ]
        '''

    state = MyState(41)
    async with never_42_graph.iter(Increment(), state=state) as graph_run:
        node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
        async for node in graph_run:
            node_states.append((node, deepcopy(graph_run.state)))
        print(node_states)
        '''
        [
            (Increment(), MyState(number=41)),
            (Increment(), MyState(number=41)),
            (Check42(), MyState(number=42)),
            (Increment(), MyState(number=42)),
            (Check42(), MyState(number=43)),
            (End(data=43), MyState(number=43)),
        ]
        '''

```

See the GraphRun.next documentation for an example of how to manually drive the graph run.

Source code in `pydantic_graph/pydantic_graph/graph.py`

````python
class GraphRun(Generic[StateT, DepsT, RunEndT]):
    """A stateful, async-iterable run of a [`Graph`][pydantic_graph.graph.Graph].

    You typically get a `GraphRun` instance from calling
    `async with [my_graph.iter(...)][pydantic_graph.graph.Graph.iter] as graph_run:`. That gives you the ability to iterate
    through nodes as they run, either by `async for` iteration or by repeatedly calling `.next(...)`.

    Here's an example of iterating over the graph from [above][pydantic_graph.graph.Graph]:
    ```py {title="iter_never_42.py" noqa="I001" requires="never_42.py"}
    from copy import deepcopy
    from never_42 import Increment, MyState, never_42_graph

    async def main():
        state = MyState(1)
        async with never_42_graph.iter(Increment(), state=state) as graph_run:
            node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
            async for node in graph_run:
                node_states.append((node, deepcopy(graph_run.state)))
            print(node_states)
            '''
            [
                (Increment(), MyState(number=1)),
                (Increment(), MyState(number=1)),
                (Check42(), MyState(number=2)),
                (End(data=2), MyState(number=2)),
            ]
            '''

        state = MyState(41)
        async with never_42_graph.iter(Increment(), state=state) as graph_run:
            node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
            async for node in graph_run:
                node_states.append((node, deepcopy(graph_run.state)))
            print(node_states)
            '''
            [
                (Increment(), MyState(number=41)),
                (Increment(), MyState(number=41)),
                (Check42(), MyState(number=42)),
                (Increment(), MyState(number=42)),
                (Check42(), MyState(number=43)),
                (End(data=43), MyState(number=43)),
            ]
            '''
    ```

    See the [`GraphRun.next` documentation][pydantic_graph.graph.GraphRun.next] for an example of how to manually
    drive the graph run.
    """

    def __init__(
        self,
        *,
        graph: Graph[StateT, DepsT, RunEndT],
        start_node: BaseNode[StateT, DepsT, RunEndT],
        persistence: BaseStatePersistence[StateT, RunEndT],
        state: StateT,
        deps: DepsT,
        traceparent: str | None,
        snapshot_id: str | None = None,
    ):
        """Create a new run for a given graph, starting at the specified node.

        Typically, you'll use [`Graph.iter`][pydantic_graph.graph.Graph.iter] rather than calling this directly.

        Args:
            graph: The [`Graph`][pydantic_graph.graph.Graph] to run.
            start_node: The node where execution will begin.
            persistence: State persistence interface.
            state: A shared state object or primitive (like a counter, dataclass, etc.) that is available
                to all nodes via `ctx.state`.
            deps: Optional dependencies that each node can access via `ctx.deps`, e.g. database connections,
                configuration, or logging clients.
            traceparent: The traceparent for the span used for the graph run.
            snapshot_id: The ID of the snapshot the node came from.
        """
        self.graph = graph
        self.persistence = persistence
        self._snapshot_id: str | None = snapshot_id
        self.state = state
        self.deps = deps

        self.__traceparent = traceparent
        self._next_node: BaseNode[StateT, DepsT, RunEndT] | End[RunEndT] = start_node
        self._is_started: bool = False

    @overload
    def _traceparent(self, *, required: typing_extensions.Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        if self.__traceparent is None and required:  # pragma: no cover
            raise exceptions.GraphRuntimeError('No span was created for this graph run')
        return self.__traceparent

    @property
    def next_node(self) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
        """The next node that will be run in the graph.

        This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
        """
        return self._next_node

    @property
    def result(self) -> GraphRunResult[StateT, RunEndT] | None:
        """The final result of the graph run if the run is completed, otherwise `None`."""
        if not isinstance(self._next_node, End):
            return None  # The GraphRun has not finished running
        return GraphRunResult[StateT, RunEndT](
            self._next_node.data,
            state=self.state,
            persistence=self.persistence,
            traceparent=self._traceparent(required=False),
        )

    async def next(
        self, node: BaseNode[StateT, DepsT, RunEndT] | None = None
    ) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
        """Manually drive the graph run by passing in the node you want to run next.

        This lets you inspect or mutate the node before continuing execution, or skip certain nodes
        under dynamic conditions. The graph run should stop when you return an [`End`][pydantic_graph.nodes.End] node.

        Here's an example of using `next` to drive the graph from [above][pydantic_graph.graph.Graph]:
        ```py {title="next_never_42.py" noqa="I001" requires="never_42.py"}
        from copy import deepcopy
        from pydantic_graph import End
        from never_42 import Increment, MyState, never_42_graph

        async def main():
            state = MyState(48)
            async with never_42_graph.iter(Increment(), state=state) as graph_run:
                next_node = graph_run.next_node  # start with the first node
                node_states = [(next_node, deepcopy(graph_run.state))]

                while not isinstance(next_node, End):
                    if graph_run.state.number == 50:
                        graph_run.state.number = 42
                    next_node = await graph_run.next(next_node)
                    node_states.append((next_node, deepcopy(graph_run.state)))

                print(node_states)
                '''
                [
                    (Increment(), MyState(number=48)),
                    (Check42(), MyState(number=49)),
                    (End(data=49), MyState(number=49)),
                ]
                '''
        ```

        Args:
            node: The node to run next in the graph. If not specified, uses `self.next_node`, which is initialized to
                the `start_node` of the run and updated each time a new node is returned.

        Returns:
            The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
            the run has completed.
        """
        if node is None:
            # This cast is necessary because self._next_node could be an `End`. You'll get a runtime error if that's
            # the case, but if it is, the only way to get there would be to have tried calling next manually after
            # the run finished. Either way, maybe it would be better to not do this cast...
            node = cast(BaseNode[StateT, DepsT, RunEndT], self._next_node)
            node_snapshot_id = node.get_snapshot_id()
        else:
            node_snapshot_id = node.get_snapshot_id()

        if node_snapshot_id != self._snapshot_id:
            await self.persistence.snapshot_node_if_new(node_snapshot_id, self.state, node)
            self._snapshot_id = node_snapshot_id

        if not isinstance(node, BaseNode):
            # While technically this is not compatible with the documented method signature, it's an easy mistake to
            # make, and we should eagerly provide a more helpful error message than you'd get otherwise.
            raise TypeError(f'`next` must be called with a `BaseNode` instance, got {node!r}.')

        node_id = node.get_node_id()
        if node_id not in self.graph.node_defs:
            raise exceptions.GraphRuntimeError(f'Node `{node}` is not in the graph.')

        with ExitStack() as stack:
            if self.graph.auto_instrument:  # pragma: no branch
                # Separate variable because we actually don't want logfire's f-string magic here,
                # we want the span_name to be preformatted for other backends
                # as requested in https://github.com/pydantic/pydantic-ai/issues/3173.
                span_name = f'run node {node_id}'
                stack.enter_context(logfire_span(span_name, node_id=node_id, node=node))

            async with self.persistence.record_run(node_snapshot_id):
                ctx = GraphRunContext(state=self.state, deps=self.deps)
                self._next_node = await node.run(ctx)

        if isinstance(self._next_node, End):
            self._snapshot_id = self._next_node.get_snapshot_id()
            await self.persistence.snapshot_end(self.state, self._next_node)
        elif isinstance(self._next_node, BaseNode):
            self._snapshot_id = self._next_node.get_snapshot_id()
            await self.persistence.snapshot_node(self.state, self._next_node)
        else:
            raise exceptions.GraphRuntimeError(
                f'Invalid node return type: `{type(self._next_node).__name__}`. Expected `BaseNode` or `End`.'
            )

        return self._next_node

    def __aiter__(self) -> AsyncIterator[BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]]:
        return self

    async def __anext__(self) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
        """Use the last returned node as the input to `Graph.next`."""
        if not self._is_started:
            self._is_started = True
            return self._next_node

        if isinstance(self._next_node, End):
            raise StopAsyncIteration

        return await self.next(self._next_node)

    def __repr__(self) -> str:
        return f'<GraphRun graph={self.graph.name or "[unnamed]"}>'

````

#### __init__

```python
__init__(
    *,
    graph: Graph[StateT, DepsT, RunEndT],
    start_node: BaseNode[StateT, DepsT, RunEndT],
    persistence: BaseStatePersistence[StateT, RunEndT],
    state: StateT,
    deps: DepsT,
    traceparent: str | None,
    snapshot_id: str | None = None
)

```

Create a new run for a given graph, starting at the specified node.

Typically, you'll use Graph.iter rather than calling this directly.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `graph` | `Graph[StateT, DepsT, RunEndT]` | The Graph to run. | *required* | | `start_node` | `BaseNode[StateT, DepsT, RunEndT]` | The node where execution will begin. | *required* | | `persistence` | `BaseStatePersistence[StateT, RunEndT]` | State persistence interface. | *required* | | `state` | `StateT` | A shared state object or primitive (like a counter, dataclass, etc.) that is available to all nodes via ctx.state. | *required* | | `deps` | `DepsT` | Optional dependencies that each node can access via ctx.deps, e.g. database connections, configuration, or logging clients. | *required* | | `traceparent` | `str | None` | The traceparent for the span used for the graph run. | *required* | | `snapshot_id` | `str | None` | The ID of the snapshot the node came from. | `None` |

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
def __init__(
    self,
    *,
    graph: Graph[StateT, DepsT, RunEndT],
    start_node: BaseNode[StateT, DepsT, RunEndT],
    persistence: BaseStatePersistence[StateT, RunEndT],
    state: StateT,
    deps: DepsT,
    traceparent: str | None,
    snapshot_id: str | None = None,
):
    """Create a new run for a given graph, starting at the specified node.

    Typically, you'll use [`Graph.iter`][pydantic_graph.graph.Graph.iter] rather than calling this directly.

    Args:
        graph: The [`Graph`][pydantic_graph.graph.Graph] to run.
        start_node: The node where execution will begin.
        persistence: State persistence interface.
        state: A shared state object or primitive (like a counter, dataclass, etc.) that is available
            to all nodes via `ctx.state`.
        deps: Optional dependencies that each node can access via `ctx.deps`, e.g. database connections,
            configuration, or logging clients.
        traceparent: The traceparent for the span used for the graph run.
        snapshot_id: The ID of the snapshot the node came from.
    """
    self.graph = graph
    self.persistence = persistence
    self._snapshot_id: str | None = snapshot_id
    self.state = state
    self.deps = deps

    self.__traceparent = traceparent
    self._next_node: BaseNode[StateT, DepsT, RunEndT] | End[RunEndT] = start_node
    self._is_started: bool = False

```

#### next_node

```python
next_node: BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]

```

The next node that will be run in the graph.

This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.

#### result

```python
result: GraphRunResult[StateT, RunEndT] | None

```

The final result of the graph run if the run is completed, otherwise `None`.

#### next

```python
next(
    node: BaseNode[StateT, DepsT, RunEndT] | None = None,
) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]

```

Manually drive the graph run by passing in the node you want to run next.

This lets you inspect or mutate the node before continuing execution, or skip certain nodes under dynamic conditions. The graph run should stop when you return an End node.

Here's an example of using `next` to drive the graph from above: next_never_42.py

```py
from copy import deepcopy
from pydantic_graph import End
from never_42 import Increment, MyState, never_42_graph

async def main():
    state = MyState(48)
    async with never_42_graph.iter(Increment(), state=state) as graph_run:
        next_node = graph_run.next_node  # start with the first node
        node_states = [(next_node, deepcopy(graph_run.state))]

        while not isinstance(next_node, End):
            if graph_run.state.number == 50:
                graph_run.state.number = 42
            next_node = await graph_run.next(next_node)
            node_states.append((next_node, deepcopy(graph_run.state)))

        print(node_states)
        '''
        [
            (Increment(), MyState(number=48)),
            (Check42(), MyState(number=49)),
            (End(data=49), MyState(number=49)),
        ]
        '''

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `node` | `BaseNode[StateT, DepsT, RunEndT] | None` | The node to run next in the graph. If not specified, uses self.next_node, which is initialized to the start_node of the run and updated each time a new node is returned. | `None` |

Returns:

| Type | Description | | --- | --- | | `BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]` | The next node returned by the graph logic, or an End node if | | `BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]` | the run has completed. |

Source code in `pydantic_graph/pydantic_graph/graph.py`

````python
async def next(
    self, node: BaseNode[StateT, DepsT, RunEndT] | None = None
) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
    """Manually drive the graph run by passing in the node you want to run next.

    This lets you inspect or mutate the node before continuing execution, or skip certain nodes
    under dynamic conditions. The graph run should stop when you return an [`End`][pydantic_graph.nodes.End] node.

    Here's an example of using `next` to drive the graph from [above][pydantic_graph.graph.Graph]:
    ```py {title="next_never_42.py" noqa="I001" requires="never_42.py"}
    from copy import deepcopy
    from pydantic_graph import End
    from never_42 import Increment, MyState, never_42_graph

    async def main():
        state = MyState(48)
        async with never_42_graph.iter(Increment(), state=state) as graph_run:
            next_node = graph_run.next_node  # start with the first node
            node_states = [(next_node, deepcopy(graph_run.state))]

            while not isinstance(next_node, End):
                if graph_run.state.number == 50:
                    graph_run.state.number = 42
                next_node = await graph_run.next(next_node)
                node_states.append((next_node, deepcopy(graph_run.state)))

            print(node_states)
            '''
            [
                (Increment(), MyState(number=48)),
                (Check42(), MyState(number=49)),
                (End(data=49), MyState(number=49)),
            ]
            '''
    ```

    Args:
        node: The node to run next in the graph. If not specified, uses `self.next_node`, which is initialized to
            the `start_node` of the run and updated each time a new node is returned.

    Returns:
        The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
        the run has completed.
    """
    if node is None:
        # This cast is necessary because self._next_node could be an `End`. You'll get a runtime error if that's
        # the case, but if it is, the only way to get there would be to have tried calling next manually after
        # the run finished. Either way, maybe it would be better to not do this cast...
        node = cast(BaseNode[StateT, DepsT, RunEndT], self._next_node)
        node_snapshot_id = node.get_snapshot_id()
    else:
        node_snapshot_id = node.get_snapshot_id()

    if node_snapshot_id != self._snapshot_id:
        await self.persistence.snapshot_node_if_new(node_snapshot_id, self.state, node)
        self._snapshot_id = node_snapshot_id

    if not isinstance(node, BaseNode):
        # While technically this is not compatible with the documented method signature, it's an easy mistake to
        # make, and we should eagerly provide a more helpful error message than you'd get otherwise.
        raise TypeError(f'`next` must be called with a `BaseNode` instance, got {node!r}.')

    node_id = node.get_node_id()
    if node_id not in self.graph.node_defs:
        raise exceptions.GraphRuntimeError(f'Node `{node}` is not in the graph.')

    with ExitStack() as stack:
        if self.graph.auto_instrument:  # pragma: no branch
            # Separate variable because we actually don't want logfire's f-string magic here,
            # we want the span_name to be preformatted for other backends
            # as requested in https://github.com/pydantic/pydantic-ai/issues/3173.
            span_name = f'run node {node_id}'
            stack.enter_context(logfire_span(span_name, node_id=node_id, node=node))

        async with self.persistence.record_run(node_snapshot_id):
            ctx = GraphRunContext(state=self.state, deps=self.deps)
            self._next_node = await node.run(ctx)

    if isinstance(self._next_node, End):
        self._snapshot_id = self._next_node.get_snapshot_id()
        await self.persistence.snapshot_end(self.state, self._next_node)
    elif isinstance(self._next_node, BaseNode):
        self._snapshot_id = self._next_node.get_snapshot_id()
        await self.persistence.snapshot_node(self.state, self._next_node)
    else:
        raise exceptions.GraphRuntimeError(
            f'Invalid node return type: `{type(self._next_node).__name__}`. Expected `BaseNode` or `End`.'
        )

    return self._next_node

````

#### __anext__

```python
__anext__() -> (
    BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]
)

```

Use the last returned node as the input to `Graph.next`.

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
async def __anext__(self) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
    """Use the last returned node as the input to `Graph.next`."""
    if not self._is_started:
        self._is_started = True
        return self._next_node

    if isinstance(self._next_node, End):
        raise StopAsyncIteration

    return await self.next(self._next_node)

```

### GraphRunResult

Bases: `Generic[StateT, RunEndT]`

The final result of running a graph.

Source code in `pydantic_graph/pydantic_graph/graph.py`

```python
@dataclass(init=False)
class GraphRunResult(Generic[StateT, RunEndT]):
    """The final result of running a graph."""

    output: RunEndT
    state: StateT
    persistence: BaseStatePersistence[StateT, RunEndT] = field(repr=False)

    def __init__(
        self,
        output: RunEndT,
        state: StateT,
        persistence: BaseStatePersistence[StateT, RunEndT],
        traceparent: str | None = None,
    ):
        self.output = output
        self.state = state
        self.persistence = persistence
        self.__traceparent = traceparent

    @overload
    def _traceparent(self, *, required: typing_extensions.Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:  # pragma: no cover
        if self.__traceparent is None and required:
            raise exceptions.GraphRuntimeError('No span was created for this graph run.')
        return self.__traceparent

```

# `pydantic_graph.mermaid`

### DEFAULT_HIGHLIGHT_CSS

```python
DEFAULT_HIGHLIGHT_CSS = 'fill:#fdff32'

```

The default CSS to use for highlighting nodes.

### StateDiagramDirection

```python
StateDiagramDirection = Literal['TB', 'LR', 'RL', 'BT']

```

Used to specify the direction of the state diagram generated by mermaid.

- `'TB'`: Top to bottom, this is the default for mermaid charts.
- `'LR'`: Left to right
- `'RL'`: Right to left
- `'BT'`: Bottom to top

### generate_code

```python
generate_code(
    graph: Graph[Any, Any, Any],
    /,
    *,
    start_node: (
        Sequence[NodeIdent] | NodeIdent | None
    ) = None,
    highlighted_nodes: (
        Sequence[NodeIdent] | NodeIdent | None
    ) = None,
    highlight_css: str = DEFAULT_HIGHLIGHT_CSS,
    title: str | None = None,
    edge_labels: bool = True,
    notes: bool = True,
    direction: StateDiagramDirection | None,
) -> str

```

Generate [Mermaid state diagram](https://mermaid.js.org/syntax/stateDiagram.html) code for a graph.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `graph` | `Graph[Any, Any, Any]` | The graph to generate the image for. | *required* | | `start_node` | `Sequence[NodeIdent] | NodeIdent | None` | Identifiers of nodes that start the graph. | `None` | | `highlighted_nodes` | `Sequence[NodeIdent] | NodeIdent | None` | Identifiers of nodes to highlight. | `None` | | `highlight_css` | `str` | CSS to use for highlighting nodes. | `DEFAULT_HIGHLIGHT_CSS` | | `title` | `str | None` | The title of the diagram. | `None` | | `edge_labels` | `bool` | Whether to include edge labels in the diagram. | `True` | | `notes` | `bool` | Whether to include notes in the diagram. | `True` | | `direction` | `StateDiagramDirection | None` | The direction of flow. | *required* |

Returns:

| Type | Description | | --- | --- | | `str` | The Mermaid code for the graph. |

Source code in `pydantic_graph/pydantic_graph/mermaid.py`

```python
def generate_code(  # noqa: C901
    graph: Graph[Any, Any, Any],
    /,
    *,
    start_node: Sequence[NodeIdent] | NodeIdent | None = None,
    highlighted_nodes: Sequence[NodeIdent] | NodeIdent | None = None,
    highlight_css: str = DEFAULT_HIGHLIGHT_CSS,
    title: str | None = None,
    edge_labels: bool = True,
    notes: bool = True,
    direction: StateDiagramDirection | None,
) -> str:
    """Generate [Mermaid state diagram](https://mermaid.js.org/syntax/stateDiagram.html) code for a graph.

    Args:
        graph: The graph to generate the image for.
        start_node: Identifiers of nodes that start the graph.
        highlighted_nodes: Identifiers of nodes to highlight.
        highlight_css: CSS to use for highlighting nodes.
        title: The title of the diagram.
        edge_labels: Whether to include edge labels in the diagram.
        notes: Whether to include notes in the diagram.
        direction: The direction of flow.


    Returns:
        The Mermaid code for the graph.
    """
    start_node_ids = set(_node_ids(start_node or ()))
    for node_id in start_node_ids:
        if node_id not in graph.node_defs:
            raise LookupError(f'Start node "{node_id}" is not in the graph.')

    lines: list[str] = []
    if title:
        lines = ['---', f'title: {title}', '---']
    lines.append('stateDiagram-v2')
    if direction is not None:
        lines.append(f'  direction {direction}')
    for node_id, node_def in graph.node_defs.items():
        # we use round brackets (rounded box) for nodes other than the start and end
        if node_id in start_node_ids:
            lines.append(f'  [*] --> {node_id}')
        if node_def.returns_base_node:
            for next_node_id in graph.node_defs:
                lines.append(f'  {node_id} --> {next_node_id}')
        else:
            for next_node_id, edge in node_def.next_node_edges.items():
                line = f'  {node_id} --> {next_node_id}'
                if edge_labels and edge.label:
                    line += f': {edge.label}'
                lines.append(line)
        if end_edge := node_def.end_edge:
            line = f'  {node_id} --> [*]'
            if edge_labels and end_edge.label:
                line += f': {end_edge.label}'
            lines.append(line)

        if notes and node_def.note:
            lines.append(f'  note right of {node_id}')
            # mermaid doesn't like multiple paragraphs in a note, and shows if so
            clean_docs = re.sub('\n{2,}', '\n', node_def.note)
            lines.append(indent(clean_docs, '    '))
            lines.append('  end note')

    if highlighted_nodes:
        lines.append('')
        lines.append(f'classDef highlighted {highlight_css}')
        for node_id in _node_ids(highlighted_nodes):
            if node_id not in graph.node_defs:
                raise LookupError(f'Highlighted node "{node_id}" is not in the graph.')
            lines.append(f'class {node_id} highlighted')

    return '\n'.join(lines)

```

### request_image

```python
request_image(
    graph: Graph[Any, Any, Any],
    /,
    **kwargs: Unpack[MermaidConfig],
) -> bytes

```

Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `graph` | `Graph[Any, Any, Any]` | The graph to generate the image for. | *required* | | `**kwargs` | `Unpack[MermaidConfig]` | Additional parameters to configure mermaid chart generation. | `{}` |

Returns:

| Type | Description | | --- | --- | | `bytes` | The image data. |

Source code in `pydantic_graph/pydantic_graph/mermaid.py`

```python
def request_image(
    graph: Graph[Any, Any, Any],
    /,
    **kwargs: Unpack[MermaidConfig],
) -> bytes:
    """Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink).

    Args:
        graph: The graph to generate the image for.
        **kwargs: Additional parameters to configure mermaid chart generation.

    Returns:
        The image data.
    """
    code = generate_code(
        graph,
        start_node=kwargs.get('start_node'),
        highlighted_nodes=kwargs.get('highlighted_nodes'),
        highlight_css=kwargs.get('highlight_css', DEFAULT_HIGHLIGHT_CSS),
        title=kwargs.get('title'),
        edge_labels=kwargs.get('edge_labels', True),
        notes=kwargs.get('notes', True),
        direction=kwargs.get('direction'),
    )
    code_base64 = base64.b64encode(code.encode()).decode()

    params: dict[str, str | float] = {}
    if kwargs.get('image_type') == 'pdf':
        url = f'https://mermaid.ink/pdf/{code_base64}'
        if kwargs.get('pdf_fit'):
            params['fit'] = ''
        if kwargs.get('pdf_landscape'):
            params['landscape'] = ''
        if pdf_paper := kwargs.get('pdf_paper'):
            params['paper'] = pdf_paper
    elif kwargs.get('image_type') == 'svg':
        url = f'https://mermaid.ink/svg/{code_base64}'
    else:
        url = f'https://mermaid.ink/img/{code_base64}'

        if image_type := kwargs.get('image_type'):
            params['type'] = image_type

    if background_color := kwargs.get('background_color'):
        params['bgColor'] = background_color
    if theme := kwargs.get('theme'):
        params['theme'] = theme
    if width := kwargs.get('width'):
        params['width'] = width
    if height := kwargs.get('height'):
        params['height'] = height
    if scale := kwargs.get('scale'):
        params['scale'] = scale

    httpx_client = kwargs.get('httpx_client') or httpx.Client()
    response = httpx_client.get(url, params=params)
    if not response.is_success:
        raise httpx.HTTPStatusError(
            f'{response.status_code} error generating image:\n{response.text}',
            request=response.request,
            response=response,
        )
    return response.content

```

### save_image

```python
save_image(
    path: Path | str,
    graph: Graph[Any, Any, Any],
    /,
    **kwargs: Unpack[MermaidConfig],
) -> None

```

Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink) and save it to a local file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `path` | `Path | str` | The path to save the image to. | *required* | | `graph` | `Graph[Any, Any, Any]` | The graph to generate the image for. | *required* | | `**kwargs` | `Unpack[MermaidConfig]` | Additional parameters to configure mermaid chart generation. | `{}` |

Source code in `pydantic_graph/pydantic_graph/mermaid.py`

```python
def save_image(
    path: Path | str,
    graph: Graph[Any, Any, Any],
    /,
    **kwargs: Unpack[MermaidConfig],
) -> None:
    """Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink) and save it to a local file.

    Args:
        path: The path to save the image to.
        graph: The graph to generate the image for.
        **kwargs: Additional parameters to configure mermaid chart generation.
    """
    if isinstance(path, str):
        path = Path(path)

    if 'image_type' not in kwargs:
        ext = path.suffix.lower()[1:]
        # no need to check for .jpeg/.jpg, as it is the default
        if ext in ('png', 'webp', 'svg', 'pdf'):
            kwargs['image_type'] = ext

    image_data = request_image(graph, **kwargs)
    path.write_bytes(image_data)

```

### MermaidConfig

Bases: `TypedDict`

Parameters to configure mermaid chart generation.

Source code in `pydantic_graph/pydantic_graph/mermaid.py`

```python
class MermaidConfig(TypedDict, total=False):
    """Parameters to configure mermaid chart generation."""

    start_node: Sequence[NodeIdent] | NodeIdent
    """Identifiers of nodes that start the graph."""
    highlighted_nodes: Sequence[NodeIdent] | NodeIdent
    """Identifiers of nodes to highlight."""
    highlight_css: str
    """CSS to use for highlighting nodes."""
    title: str | None
    """The title of the diagram."""
    edge_labels: bool
    """Whether to include edge labels in the diagram."""
    notes: bool
    """Whether to include notes on nodes in the diagram, defaults to true."""
    image_type: Literal['jpeg', 'png', 'webp', 'svg', 'pdf']
    """The image type to generate. If unspecified, the default behavior is `'jpeg'`."""
    pdf_fit: bool
    """When using image_type='pdf', whether to fit the diagram to the PDF page."""
    pdf_landscape: bool
    """When using image_type='pdf', whether to use landscape orientation for the PDF.

    This has no effect if using `pdf_fit`.
    """
    pdf_paper: Literal['letter', 'legal', 'tabloid', 'ledger', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    """When using image_type='pdf', the paper size of the PDF."""
    background_color: str
    """The background color of the diagram.

    If None, the default transparent background is used. The color value is interpreted as a hexadecimal color
    code by default (and should not have a leading '#'), but you can also use named colors by prefixing the
    value with `'!'`. For example, valid choices include `background_color='!white'` or `background_color='FF0000'`.
    """
    theme: Literal['default', 'neutral', 'dark', 'forest']
    """The theme of the diagram. Defaults to 'default'."""
    width: int
    """The width of the diagram."""
    height: int
    """The height of the diagram."""
    scale: Annotated[float, Ge(1), Le(3)]
    """The scale of the diagram.

    The scale must be a number between 1 and 3, and you can only set a scale if one or both of width and height are set.
    """
    httpx_client: httpx.Client
    """An HTTPX client to use for requests, mostly for testing purposes."""
    direction: StateDiagramDirection
    """The direction of the state diagram."""

```

#### start_node

```python
start_node: Sequence[NodeIdent] | NodeIdent

```

Identifiers of nodes that start the graph.

#### highlighted_nodes

```python
highlighted_nodes: Sequence[NodeIdent] | NodeIdent

```

Identifiers of nodes to highlight.

#### highlight_css

```python
highlight_css: str

```

CSS to use for highlighting nodes.

#### title

```python
title: str | None

```

The title of the diagram.

#### edge_labels

```python
edge_labels: bool

```

Whether to include edge labels in the diagram.

#### notes

```python
notes: bool

```

Whether to include notes on nodes in the diagram, defaults to true.

#### image_type

```python
image_type: Literal['jpeg', 'png', 'webp', 'svg', 'pdf']

```

The image type to generate. If unspecified, the default behavior is `'jpeg'`.

#### pdf_fit

```python
pdf_fit: bool

```

When using image_type='pdf', whether to fit the diagram to the PDF page.

#### pdf_landscape

```python
pdf_landscape: bool

```

When using image_type='pdf', whether to use landscape orientation for the PDF.

This has no effect if using `pdf_fit`.

#### pdf_paper

```python
pdf_paper: Literal[
    "letter",
    "legal",
    "tabloid",
    "ledger",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
]

```

When using image_type='pdf', the paper size of the PDF.

#### background_color

```python
background_color: str

```

The background color of the diagram.

If None, the default transparent background is used. The color value is interpreted as a hexadecimal color code by default (and should not have a leading '#'), but you can also use named colors by prefixing the value with `'!'`. For example, valid choices include `background_color='!white'` or `background_color='FF0000'`.

#### theme

```python
theme: Literal['default', 'neutral', 'dark', 'forest']

```

The theme of the diagram. Defaults to 'default'.

#### width

```python
width: int

```

The width of the diagram.

#### height

```python
height: int

```

The height of the diagram.

#### scale

```python
scale: Annotated[float, Ge(1), Le(3)]

```

The scale of the diagram.

The scale must be a number between 1 and 3, and you can only set a scale if one or both of width and height are set.

#### httpx_client

```python
httpx_client: Client

```

An HTTPX client to use for requests, mostly for testing purposes.

#### direction

```python
direction: StateDiagramDirection

```

The direction of the state diagram.

### NodeIdent

```python
NodeIdent: TypeAlias = (
    "type[BaseNode[Any, Any, Any]] | BaseNode[Any, Any, Any] | str"
)

```

A type alias for a node identifier.

This can be:

- A node instance (instance of a subclass of BaseNode).
- A node class (subclass of BaseNode).
- A string representing the node ID.

# `pydantic_graph.nodes`

### StateT

```python
StateT = TypeVar('StateT', default=None)

```

Type variable for the state in a graph.

### GraphRunContext

Bases: `Generic[StateT, DepsT]`

Context for a graph.

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
@dataclass(kw_only=True)
class GraphRunContext(Generic[StateT, DepsT]):
    """Context for a graph."""

    state: StateT
    """The state of the graph."""
    deps: DepsT
    """Dependencies for the graph."""

```

#### state

```python
state: StateT

```

The state of the graph.

#### deps

```python
deps: DepsT

```

Dependencies for the graph.

### BaseNode

Bases: `ABC`, `Generic[StateT, DepsT, NodeRunEndT]`

Base class for a node.

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
class BaseNode(ABC, Generic[StateT, DepsT, NodeRunEndT]):
    """Base class for a node."""

    docstring_notes: ClassVar[bool] = False
    """Set to `True` to generate mermaid diagram notes from the class's docstring.

    While this can add valuable information to the diagram, it can make diagrams harder to view, hence
    it is disabled by default. You can also customise notes overriding the
    [`get_note`][pydantic_graph.nodes.BaseNode.get_note] method.
    """

    @abstractmethod
    async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[NodeRunEndT]:
        """Run the node.

        This is an abstract method that must be implemented by subclasses.

        !!! note "Return types used at runtime"
            The return type of this method are read by `pydantic_graph` at runtime and used to define which
            nodes can be called next in the graph. This is displayed in [mermaid diagrams](mermaid.md)
            and enforced when running the graph.

        Args:
            ctx: The graph context.

        Returns:
            The next node to run or [`End`][pydantic_graph.nodes.End] to signal the end of the graph.
        """
        ...

    def get_snapshot_id(self) -> str:
        if snapshot_id := getattr(self, '__snapshot_id', None):
            return snapshot_id
        else:
            self.__dict__['__snapshot_id'] = snapshot_id = generate_snapshot_id(self.get_node_id())
            return snapshot_id

    def set_snapshot_id(self, snapshot_id: str) -> None:
        self.__dict__['__snapshot_id'] = snapshot_id

    @classmethod
    @cache
    def get_node_id(cls) -> str:
        """Get the ID of the node."""
        return cls.__name__

    @classmethod
    def get_note(cls) -> str | None:
        """Get a note about the node to render on mermaid charts.

        By default, this returns a note only if [`docstring_notes`][pydantic_graph.nodes.BaseNode.docstring_notes]
        is `True`. You can override this method to customise the node notes.
        """
        if not cls.docstring_notes:
            return None
        docstring = cls.__doc__
        # dataclasses get an automatic docstring which is just their signature, we don't want that
        if docstring and is_dataclass(cls) and docstring.startswith(f'{cls.__name__}('):
            docstring = None  # pragma: no cover
        if docstring:  # pragma: no branch
            # remove indentation from docstring
            import inspect

            docstring = inspect.cleandoc(docstring)
        return docstring

    @classmethod
    def get_node_def(cls, local_ns: dict[str, Any] | None) -> NodeDef[StateT, DepsT, NodeRunEndT]:
        """Get the node definition."""
        type_hints = get_type_hints(cls.run, localns=local_ns, include_extras=True)
        try:
            return_hint = type_hints['return']
        except KeyError as e:
            raise exceptions.GraphSetupError(f'Node {cls} is missing a return type hint on its `run` method') from e

        next_node_edges: dict[str, Edge] = {}
        end_edge: Edge | None = None
        returns_base_node: bool = False
        for return_type in _utils.get_union_args(return_hint):
            return_type, annotations = _utils.unpack_annotated(return_type)
            edge = next((a for a in annotations if isinstance(a, Edge)), Edge(None))
            return_type_origin = get_origin(return_type) or return_type
            if return_type_origin is End:
                end_edge = edge
            elif return_type_origin is BaseNode:
                returns_base_node = True
            elif issubclass(return_type_origin, BaseNode):
                next_node_edges[return_type.get_node_id()] = edge
            else:
                raise exceptions.GraphSetupError(f'Invalid return type: {return_type}')

        return NodeDef(
            node=cls,
            node_id=cls.get_node_id(),
            note=cls.get_note(),
            next_node_edges=next_node_edges,
            end_edge=end_edge,
            returns_base_node=returns_base_node,
        )

    def deep_copy(self) -> Self:
        """Returns a deep copy of the node."""
        return copy.deepcopy(self)

```

#### docstring_notes

```python
docstring_notes: bool = False

```

Set to `True` to generate mermaid diagram notes from the class's docstring.

While this can add valuable information to the diagram, it can make diagrams harder to view, hence it is disabled by default. You can also customise notes overriding the get_note method.

#### run

```python
run(
    ctx: GraphRunContext[StateT, DepsT],
) -> BaseNode[StateT, DepsT, Any] | End[NodeRunEndT]

```

Run the node.

This is an abstract method that must be implemented by subclasses.

Return types used at runtime

The return type of this method are read by `pydantic_graph` at runtime and used to define which nodes can be called next in the graph. This is displayed in [mermaid diagrams](../mermaid/) and enforced when running the graph.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `GraphRunContext[StateT, DepsT]` | The graph context. | *required* |

Returns:

| Type | Description | | --- | --- | | `BaseNode[StateT, DepsT, Any] | End[NodeRunEndT]` | The next node to run or End to signal the end of the graph. |

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
@abstractmethod
async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[NodeRunEndT]:
    """Run the node.

    This is an abstract method that must be implemented by subclasses.

    !!! note "Return types used at runtime"
        The return type of this method are read by `pydantic_graph` at runtime and used to define which
        nodes can be called next in the graph. This is displayed in [mermaid diagrams](mermaid.md)
        and enforced when running the graph.

    Args:
        ctx: The graph context.

    Returns:
        The next node to run or [`End`][pydantic_graph.nodes.End] to signal the end of the graph.
    """
    ...

```

#### get_node_id

```python
get_node_id() -> str

```

Get the ID of the node.

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
@classmethod
@cache
def get_node_id(cls) -> str:
    """Get the ID of the node."""
    return cls.__name__

```

#### get_note

```python
get_note() -> str | None

```

Get a note about the node to render on mermaid charts.

By default, this returns a note only if docstring_notes is `True`. You can override this method to customise the node notes.

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
@classmethod
def get_note(cls) -> str | None:
    """Get a note about the node to render on mermaid charts.

    By default, this returns a note only if [`docstring_notes`][pydantic_graph.nodes.BaseNode.docstring_notes]
    is `True`. You can override this method to customise the node notes.
    """
    if not cls.docstring_notes:
        return None
    docstring = cls.__doc__
    # dataclasses get an automatic docstring which is just their signature, we don't want that
    if docstring and is_dataclass(cls) and docstring.startswith(f'{cls.__name__}('):
        docstring = None  # pragma: no cover
    if docstring:  # pragma: no branch
        # remove indentation from docstring
        import inspect

        docstring = inspect.cleandoc(docstring)
    return docstring

```

#### get_node_def

```python
get_node_def(
    local_ns: dict[str, Any] | None,
) -> NodeDef[StateT, DepsT, NodeRunEndT]

```

Get the node definition.

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
@classmethod
def get_node_def(cls, local_ns: dict[str, Any] | None) -> NodeDef[StateT, DepsT, NodeRunEndT]:
    """Get the node definition."""
    type_hints = get_type_hints(cls.run, localns=local_ns, include_extras=True)
    try:
        return_hint = type_hints['return']
    except KeyError as e:
        raise exceptions.GraphSetupError(f'Node {cls} is missing a return type hint on its `run` method') from e

    next_node_edges: dict[str, Edge] = {}
    end_edge: Edge | None = None
    returns_base_node: bool = False
    for return_type in _utils.get_union_args(return_hint):
        return_type, annotations = _utils.unpack_annotated(return_type)
        edge = next((a for a in annotations if isinstance(a, Edge)), Edge(None))
        return_type_origin = get_origin(return_type) or return_type
        if return_type_origin is End:
            end_edge = edge
        elif return_type_origin is BaseNode:
            returns_base_node = True
        elif issubclass(return_type_origin, BaseNode):
            next_node_edges[return_type.get_node_id()] = edge
        else:
            raise exceptions.GraphSetupError(f'Invalid return type: {return_type}')

    return NodeDef(
        node=cls,
        node_id=cls.get_node_id(),
        note=cls.get_note(),
        next_node_edges=next_node_edges,
        end_edge=end_edge,
        returns_base_node=returns_base_node,
    )

```

#### deep_copy

```python
deep_copy() -> Self

```

Returns a deep copy of the node.

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
def deep_copy(self) -> Self:
    """Returns a deep copy of the node."""
    return copy.deepcopy(self)

```

### End

Bases: `Generic[RunEndT]`

Type to return from a node to signal the end of the graph.

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
@dataclass
class End(Generic[RunEndT]):
    """Type to return from a node to signal the end of the graph."""

    data: RunEndT
    """Data to return from the graph."""

    def deep_copy_data(self) -> End[RunEndT]:
        """Returns a deep copy of the end of the run."""
        if self.data is None:
            return self
        else:
            end = End(copy.deepcopy(self.data))
            end.set_snapshot_id(self.get_snapshot_id())
            return end

    def get_snapshot_id(self) -> str:
        if snapshot_id := getattr(self, '__snapshot_id', None):
            return snapshot_id
        else:
            self.__dict__['__snapshot_id'] = snapshot_id = generate_snapshot_id('end')
            return snapshot_id

    def set_snapshot_id(self, set_id: str) -> None:
        self.__dict__['__snapshot_id'] = set_id

```

#### data

```python
data: RunEndT

```

Data to return from the graph.

#### deep_copy_data

```python
deep_copy_data() -> End[RunEndT]

```

Returns a deep copy of the end of the run.

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
def deep_copy_data(self) -> End[RunEndT]:
    """Returns a deep copy of the end of the run."""
    if self.data is None:
        return self
    else:
        end = End(copy.deepcopy(self.data))
        end.set_snapshot_id(self.get_snapshot_id())
        return end

```

### Edge

Annotation to apply a label to an edge in a graph.

Source code in `pydantic_graph/pydantic_graph/nodes.py`

```python
@dataclass(frozen=True)
class Edge:
    """Annotation to apply a label to an edge in a graph."""

    label: str | None
    """Label for the edge."""

```

#### label

```python
label: str | None

```

Label for the edge.

### DepsT

```python
DepsT = TypeVar('DepsT', default=None, contravariant=True)

```

Type variable for the dependencies of a graph and node.

### RunEndT

```python
RunEndT = TypeVar('RunEndT', covariant=True, default=None)

```

Covariant type variable for the return type of a graph run.

### NodeRunEndT

```python
NodeRunEndT = TypeVar(
    "NodeRunEndT", covariant=True, default=Never
)

```

Covariant type variable for the return type of a node run.

# `pydantic_graph.persistence`

### SnapshotStatus

```python
SnapshotStatus = Literal[
    "created", "pending", "running", "success", "error"
]

```

The status of a snapshot.

- `'created'`: The snapshot has been created but not yet run.
- `'pending'`: The snapshot has been retrieved with load_next but not yet run.
- `'running'`: The snapshot is currently running.
- `'success'`: The snapshot has been run successfully.
- `'error'`: The snapshot has been run but an error occurred.

### NodeSnapshot

Bases: `Generic[StateT, RunEndT]`

History step describing the execution of a node in a graph.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
@dataclass(kw_only=True)
class NodeSnapshot(Generic[StateT, RunEndT]):
    """History step describing the execution of a node in a graph."""

    state: StateT
    """The state of the graph before the node is run."""
    node: Annotated[BaseNode[StateT, Any, RunEndT], _utils.CustomNodeSchema()]
    """The node to run next."""
    start_ts: datetime | None = None
    """The timestamp when the node started running, `None` until the run starts."""
    duration: float | None = None
    """The duration of the node run in seconds, if the node has been run."""
    status: SnapshotStatus = 'created'
    """The status of the snapshot."""
    kind: Literal['node'] = 'node'
    """The kind of history step, can be used as a discriminator when deserializing history."""

    id: str = UNSET_SNAPSHOT_ID
    """Unique ID of the snapshot."""

    def __post_init__(self) -> None:
        if self.id == UNSET_SNAPSHOT_ID:
            self.id = self.node.get_snapshot_id()

```

#### state

```python
state: StateT

```

The state of the graph before the node is run.

#### node

```python
node: Annotated[
    BaseNode[StateT, Any, RunEndT], CustomNodeSchema()
]

```

The node to run next.

#### start_ts

```python
start_ts: datetime | None = None

```

The timestamp when the node started running, `None` until the run starts.

#### duration

```python
duration: float | None = None

```

The duration of the node run in seconds, if the node has been run.

#### status

```python
status: SnapshotStatus = 'created'

```

The status of the snapshot.

#### kind

```python
kind: Literal['node'] = 'node'

```

The kind of history step, can be used as a discriminator when deserializing history.

#### id

```python
id: str = UNSET_SNAPSHOT_ID

```

Unique ID of the snapshot.

### EndSnapshot

Bases: `Generic[StateT, RunEndT]`

History step describing the end of a graph run.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
@dataclass(kw_only=True)
class EndSnapshot(Generic[StateT, RunEndT]):
    """History step describing the end of a graph run."""

    state: StateT
    """The state of the graph at the end of the run."""
    result: End[RunEndT]
    """The result of the graph run."""
    ts: datetime = field(default_factory=_utils.now_utc)
    """The timestamp when the graph run ended."""
    kind: Literal['end'] = 'end'
    """The kind of history step, can be used as a discriminator when deserializing history."""

    id: str = UNSET_SNAPSHOT_ID
    """Unique ID of the snapshot."""

    def __post_init__(self) -> None:
        if self.id == UNSET_SNAPSHOT_ID:
            self.id = self.node.get_snapshot_id()

    @property
    def node(self) -> End[RunEndT]:
        """Shim to get the [`result`][pydantic_graph.persistence.EndSnapshot.result].

        Useful to allow `[snapshot.node for snapshot in persistence.history]`.
        """
        return self.result

```

#### state

```python
state: StateT

```

The state of the graph at the end of the run.

#### result

```python
result: End[RunEndT]

```

The result of the graph run.

#### ts

```python
ts: datetime = field(default_factory=now_utc)

```

The timestamp when the graph run ended.

#### kind

```python
kind: Literal['end'] = 'end'

```

The kind of history step, can be used as a discriminator when deserializing history.

#### id

```python
id: str = UNSET_SNAPSHOT_ID

```

Unique ID of the snapshot.

#### node

```python
node: End[RunEndT]

```

Shim to get the result.

Useful to allow `[snapshot.node for snapshot in persistence.history]`.

### Snapshot

```python
Snapshot = (
    NodeSnapshot[StateT, RunEndT]
    | EndSnapshot[StateT, RunEndT]
)

```

A step in the history of a graph run.

Graph.run returns a list of these steps describing the execution of the graph, together with the run return value.

### BaseStatePersistence

Bases: `ABC`, `Generic[StateT, RunEndT]`

Abstract base class for storing the state of a graph run.

Each instance of a `BaseStatePersistence` subclass should be used for a single graph run.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
class BaseStatePersistence(ABC, Generic[StateT, RunEndT]):
    """Abstract base class for storing the state of a graph run.

    Each instance of a `BaseStatePersistence` subclass should be used for a single graph run.
    """

    @abstractmethod
    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        """Snapshot the state of a graph, when the next step is to run a node.

        This method should add a [`NodeSnapshot`][pydantic_graph.persistence.NodeSnapshot] to persistence.

        Args:
            state: The state of the graph.
            next_node: The next node to run.
        """
        raise NotImplementedError

    @abstractmethod
    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
    ) -> None:
        """Snapshot the state of a graph if the snapshot ID doesn't already exist in persistence.

        This method will generally call [`snapshot_node`][pydantic_graph.persistence.BaseStatePersistence.snapshot_node]
        but should do so in an atomic way.

        Args:
            snapshot_id: The ID of the snapshot to check.
            state: The state of the graph.
            next_node: The next node to run.
        """
        raise NotImplementedError

    @abstractmethod
    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        """Snapshot the state of a graph when the graph has ended.

        This method should add an [`EndSnapshot`][pydantic_graph.persistence.EndSnapshot] to persistence.

        Args:
            state: The state of the graph.
            end: data from the end of the run.
        """
        raise NotImplementedError

    @abstractmethod
    def record_run(self, snapshot_id: str) -> AbstractAsyncContextManager[None]:
        """Record the run of the node, or error if the node is already running.

        Args:
            snapshot_id: The ID of the snapshot to record.

        Raises:
            GraphNodeRunningError: if the node status it not `'created'` or `'pending'`.
            LookupError: if the snapshot ID is not found in persistence.

        Returns:
            An async context manager that records the run of the node.

        In particular this should set:

        - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'running'` and
          [`NodeSnapshot.start_ts`][pydantic_graph.persistence.NodeSnapshot.start_ts] when the run starts.
        - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'success'` or `'error'` and
          [`NodeSnapshot.duration`][pydantic_graph.persistence.NodeSnapshot.duration] when the run finishes.
        """
        raise NotImplementedError

    @abstractmethod
    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        """Retrieve a node snapshot with status `'created`' and set its status to `'pending'`.

        This is used by [`Graph.iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence]
        to get the next node to run.

        Returns: The snapshot, or `None` if no snapshot with status `'created`' exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        """Load the entire history of snapshots.

        `load_all` is not used by pydantic-graph itself, instead it's provided to make it convenient to
        get all [snapshots][pydantic_graph.persistence.Snapshot] from persistence.

        Returns: The list of snapshots.
        """
        raise NotImplementedError

    def set_graph_types(self, graph: Graph[StateT, Any, RunEndT]) -> None:
        """Set the types of the state and run end from a graph.

        You generally won't need to customise this method, instead implement
        [`set_types`][pydantic_graph.persistence.BaseStatePersistence.set_types] and
        [`should_set_types`][pydantic_graph.persistence.BaseStatePersistence.should_set_types].
        """
        if self.should_set_types():
            with _utils.set_nodes_type_context(graph.get_nodes()):
                self.set_types(*graph.inferred_types)

    def should_set_types(self) -> bool:
        """Whether types need to be set.

        Implementations should override this method to return `True` when types have not been set if they are needed.
        """
        return False

    def set_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
        """Set the types of the state and run end.

        This can be used to create [type adapters][pydantic.TypeAdapter] for serializing and deserializing snapshots,
        e.g. with [`build_snapshot_list_type_adapter`][pydantic_graph.persistence.build_snapshot_list_type_adapter].

        Args:
            state_type: The state type.
            run_end_type: The run end type.
        """
        pass

```

#### snapshot_node

```python
snapshot_node(
    state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
) -> None

```

Snapshot the state of a graph, when the next step is to run a node.

This method should add a NodeSnapshot to persistence.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `state` | `StateT` | The state of the graph. | *required* | | `next_node` | `BaseNode[StateT, Any, RunEndT]` | The next node to run. | *required* |

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
@abstractmethod
async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
    """Snapshot the state of a graph, when the next step is to run a node.

    This method should add a [`NodeSnapshot`][pydantic_graph.persistence.NodeSnapshot] to persistence.

    Args:
        state: The state of the graph.
        next_node: The next node to run.
    """
    raise NotImplementedError

```

#### snapshot_node_if_new

```python
snapshot_node_if_new(
    snapshot_id: str,
    state: StateT,
    next_node: BaseNode[StateT, Any, RunEndT],
) -> None

```

Snapshot the state of a graph if the snapshot ID doesn't already exist in persistence.

This method will generally call snapshot_node but should do so in an atomic way.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `snapshot_id` | `str` | The ID of the snapshot to check. | *required* | | `state` | `StateT` | The state of the graph. | *required* | | `next_node` | `BaseNode[StateT, Any, RunEndT]` | The next node to run. | *required* |

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
@abstractmethod
async def snapshot_node_if_new(
    self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
) -> None:
    """Snapshot the state of a graph if the snapshot ID doesn't already exist in persistence.

    This method will generally call [`snapshot_node`][pydantic_graph.persistence.BaseStatePersistence.snapshot_node]
    but should do so in an atomic way.

    Args:
        snapshot_id: The ID of the snapshot to check.
        state: The state of the graph.
        next_node: The next node to run.
    """
    raise NotImplementedError

```

#### snapshot_end

```python
snapshot_end(state: StateT, end: End[RunEndT]) -> None

```

Snapshot the state of a graph when the graph has ended.

This method should add an EndSnapshot to persistence.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `state` | `StateT` | The state of the graph. | *required* | | `end` | `End[RunEndT]` | data from the end of the run. | *required* |

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
@abstractmethod
async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
    """Snapshot the state of a graph when the graph has ended.

    This method should add an [`EndSnapshot`][pydantic_graph.persistence.EndSnapshot] to persistence.

    Args:
        state: The state of the graph.
        end: data from the end of the run.
    """
    raise NotImplementedError

```

#### record_run

```python
record_run(
    snapshot_id: str,
) -> AbstractAsyncContextManager[None]

```

Record the run of the node, or error if the node is already running.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `snapshot_id` | `str` | The ID of the snapshot to record. | *required* |

Raises:

| Type | Description | | --- | --- | | `GraphNodeRunningError` | if the node status it not 'created' or 'pending'. | | `LookupError` | if the snapshot ID is not found in persistence. |

Returns:

| Type | Description | | --- | --- | | `AbstractAsyncContextManager[None]` | An async context manager that records the run of the node. |

In particular this should set:

- NodeSnapshot.status to `'running'` and NodeSnapshot.start_ts when the run starts.
- NodeSnapshot.status to `'success'` or `'error'` and NodeSnapshot.duration when the run finishes.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
@abstractmethod
def record_run(self, snapshot_id: str) -> AbstractAsyncContextManager[None]:
    """Record the run of the node, or error if the node is already running.

    Args:
        snapshot_id: The ID of the snapshot to record.

    Raises:
        GraphNodeRunningError: if the node status it not `'created'` or `'pending'`.
        LookupError: if the snapshot ID is not found in persistence.

    Returns:
        An async context manager that records the run of the node.

    In particular this should set:

    - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'running'` and
      [`NodeSnapshot.start_ts`][pydantic_graph.persistence.NodeSnapshot.start_ts] when the run starts.
    - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'success'` or `'error'` and
      [`NodeSnapshot.duration`][pydantic_graph.persistence.NodeSnapshot.duration] when the run finishes.
    """
    raise NotImplementedError

```

#### load_next

```python
load_next() -> NodeSnapshot[StateT, RunEndT] | None

```

Retrieve a node snapshot with status `'created`' and set its status to `'pending'`.

This is used by Graph.iter_from_persistence to get the next node to run.

Returns: The snapshot, or `None` if no snapshot with status `'created`' exists.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
@abstractmethod
async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
    """Retrieve a node snapshot with status `'created`' and set its status to `'pending'`.

    This is used by [`Graph.iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence]
    to get the next node to run.

    Returns: The snapshot, or `None` if no snapshot with status `'created`' exists.
    """
    raise NotImplementedError

```

#### load_all

```python
load_all() -> list[Snapshot[StateT, RunEndT]]

```

Load the entire history of snapshots.

`load_all` is not used by pydantic-graph itself, instead it's provided to make it convenient to get all snapshots from persistence.

Returns: The list of snapshots.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
@abstractmethod
async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
    """Load the entire history of snapshots.

    `load_all` is not used by pydantic-graph itself, instead it's provided to make it convenient to
    get all [snapshots][pydantic_graph.persistence.Snapshot] from persistence.

    Returns: The list of snapshots.
    """
    raise NotImplementedError

```

#### set_graph_types

```python
set_graph_types(graph: Graph[StateT, Any, RunEndT]) -> None

```

Set the types of the state and run end from a graph.

You generally won't need to customise this method, instead implement set_types and should_set_types.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
def set_graph_types(self, graph: Graph[StateT, Any, RunEndT]) -> None:
    """Set the types of the state and run end from a graph.

    You generally won't need to customise this method, instead implement
    [`set_types`][pydantic_graph.persistence.BaseStatePersistence.set_types] and
    [`should_set_types`][pydantic_graph.persistence.BaseStatePersistence.should_set_types].
    """
    if self.should_set_types():
        with _utils.set_nodes_type_context(graph.get_nodes()):
            self.set_types(*graph.inferred_types)

```

#### should_set_types

```python
should_set_types() -> bool

```

Whether types need to be set.

Implementations should override this method to return `True` when types have not been set if they are needed.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
def should_set_types(self) -> bool:
    """Whether types need to be set.

    Implementations should override this method to return `True` when types have not been set if they are needed.
    """
    return False

```

#### set_types

```python
set_types(
    state_type: type[StateT], run_end_type: type[RunEndT]
) -> None

```

Set the types of the state and run end.

This can be used to create type adapters for serializing and deserializing snapshots, e.g. with build_snapshot_list_type_adapter.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `state_type` | `type[StateT]` | The state type. | *required* | | `run_end_type` | `type[RunEndT]` | The run end type. | *required* |

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
def set_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
    """Set the types of the state and run end.

    This can be used to create [type adapters][pydantic.TypeAdapter] for serializing and deserializing snapshots,
    e.g. with [`build_snapshot_list_type_adapter`][pydantic_graph.persistence.build_snapshot_list_type_adapter].

    Args:
        state_type: The state type.
        run_end_type: The run end type.
    """
    pass

```

### build_snapshot_list_type_adapter

```python
build_snapshot_list_type_adapter(
    state_t: type[StateT], run_end_t: type[RunEndT]
) -> TypeAdapter[list[Snapshot[StateT, RunEndT]]]

```

Build a type adapter for a list of snapshots.

This method should be called from within set_types where context variables will be set such that Pydantic can create a schema for NodeSnapshot.node.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`

```python
def build_snapshot_list_type_adapter(
    state_t: type[StateT], run_end_t: type[RunEndT]
) -> pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]]:
    """Build a type adapter for a list of snapshots.

    This method should be called from within
    [`set_types`][pydantic_graph.persistence.BaseStatePersistence.set_types]
    where context variables will be set such that Pydantic can create a schema for
    [`NodeSnapshot.node`][pydantic_graph.persistence.NodeSnapshot.node].
    """
    return pydantic.TypeAdapter(list[Annotated[Snapshot[state_t, run_end_t], pydantic.Discriminator('kind')]])

```

In memory state persistence.

This module provides simple in memory state persistence for graphs.

### SimpleStatePersistence

Bases: `BaseStatePersistence[StateT, RunEndT]`

Simple in memory state persistence that just hold the latest snapshot.

If no state persistence implementation is provided when running a graph, this is used by default.

Source code in `pydantic_graph/pydantic_graph/persistence/in_mem.py`

```python
@dataclass
class SimpleStatePersistence(BaseStatePersistence[StateT, RunEndT]):
    """Simple in memory state persistence that just hold the latest snapshot.

    If no state persistence implementation is provided when running a graph, this is used by default.
    """

    last_snapshot: Snapshot[StateT, RunEndT] | None = None
    """The last snapshot."""

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        self.last_snapshot = NodeSnapshot(state=state, node=next_node)

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
    ) -> None:
        if self.last_snapshot and self.last_snapshot.id == snapshot_id:
            return  # pragma: no cover
        else:
            await self.snapshot_node(state, next_node)

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        self.last_snapshot = EndSnapshot(state=state, result=end)

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        if self.last_snapshot is None or snapshot_id != self.last_snapshot.id:
            raise LookupError(f'No snapshot found with id={snapshot_id!r}')

        assert isinstance(self.last_snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
        exceptions.GraphNodeStatusError.check(self.last_snapshot.status)
        self.last_snapshot.status = 'running'
        self.last_snapshot.start_ts = _utils.now_utc()

        start = perf_counter()
        try:
            yield
        except Exception:  # pragma: no cover
            self.last_snapshot.duration = perf_counter() - start
            self.last_snapshot.status = 'error'
            raise
        else:
            self.last_snapshot.duration = perf_counter() - start
            self.last_snapshot.status = 'success'

    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        if isinstance(self.last_snapshot, NodeSnapshot) and self.last_snapshot.status == 'created':
            self.last_snapshot.status = 'pending'
            return copy.deepcopy(self.last_snapshot)

    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        raise NotImplementedError('load is not supported for SimpleStatePersistence')

```

#### last_snapshot

```python
last_snapshot: Snapshot[StateT, RunEndT] | None = None

```

The last snapshot.

### FullStatePersistence

Bases: `BaseStatePersistence[StateT, RunEndT]`

In memory state persistence that hold a list of snapshots.

Source code in `pydantic_graph/pydantic_graph/persistence/in_mem.py`

```python
@dataclass
class FullStatePersistence(BaseStatePersistence[StateT, RunEndT]):
    """In memory state persistence that hold a list of snapshots."""

    deep_copy: bool = True
    """Whether to deep copy the state and nodes when storing them.

    Defaults to `True` so even if nodes or state are modified after the snapshot is taken,
    the persistence history will record the value at the time of the snapshot.
    """
    history: list[Snapshot[StateT, RunEndT]] = field(default_factory=list)
    """List of snapshots taken during the graph run."""
    _snapshots_type_adapter: pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = field(
        default=None, init=False, repr=False
    )

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        snapshot = NodeSnapshot(
            state=self._prep_state(state),
            node=next_node.deep_copy() if self.deep_copy else next_node,
        )
        self.history.append(snapshot)

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
    ) -> None:
        if not any(s.id == snapshot_id for s in self.history):
            await self.snapshot_node(state, next_node)

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        snapshot = EndSnapshot(
            state=self._prep_state(state),
            result=end.deep_copy_data() if self.deep_copy else end,
        )
        self.history.append(snapshot)

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        try:
            snapshot = next(s for s in self.history if s.id == snapshot_id)
        except StopIteration as e:
            raise LookupError(f'No snapshot found with id={snapshot_id!r}') from e

        assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
        exceptions.GraphNodeStatusError.check(snapshot.status)
        snapshot.status = 'running'
        snapshot.start_ts = _utils.now_utc()
        start = perf_counter()
        try:
            yield
        except Exception:
            snapshot.duration = perf_counter() - start
            snapshot.status = 'error'
            raise
        else:
            snapshot.duration = perf_counter() - start
            snapshot.status = 'success'

    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        if snapshot := next((s for s in self.history if isinstance(s, NodeSnapshot) and s.status == 'created'), None):
            snapshot.status = 'pending'
            return copy.deepcopy(snapshot)

    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        return self.history

    def should_set_types(self) -> bool:
        return self._snapshots_type_adapter is None

    def set_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
        self._snapshots_type_adapter = build_snapshot_list_type_adapter(state_type, run_end_type)

    def dump_json(self, *, indent: int | None = None) -> bytes:
        """Dump the history to JSON bytes."""
        assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `dump_json`'
        return self._snapshots_type_adapter.dump_json(self.history, indent=indent)

    def load_json(self, json_data: str | bytes | bytearray) -> None:
        """Load the history from JSON."""
        assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `load_json`'
        self.history = self._snapshots_type_adapter.validate_json(json_data)

    def _prep_state(self, state: StateT) -> StateT:
        """Prepare state for snapshot, uses [`copy.deepcopy`][copy.deepcopy] by default."""
        if not self.deep_copy or state is None:
            return state
        else:
            return copy.deepcopy(state)

```

#### deep_copy

```python
deep_copy: bool = True

```

Whether to deep copy the state and nodes when storing them.

Defaults to `True` so even if nodes or state are modified after the snapshot is taken, the persistence history will record the value at the time of the snapshot.

#### history

```python
history: list[Snapshot[StateT, RunEndT]] = field(
    default_factory=list
)

```

List of snapshots taken during the graph run.

#### dump_json

```python
dump_json(*, indent: int | None = None) -> bytes

```

Dump the history to JSON bytes.

Source code in `pydantic_graph/pydantic_graph/persistence/in_mem.py`

```python
def dump_json(self, *, indent: int | None = None) -> bytes:
    """Dump the history to JSON bytes."""
    assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `dump_json`'
    return self._snapshots_type_adapter.dump_json(self.history, indent=indent)

```

#### load_json

```python
load_json(json_data: str | bytes | bytearray) -> None

```

Load the history from JSON.

Source code in `pydantic_graph/pydantic_graph/persistence/in_mem.py`

```python
def load_json(self, json_data: str | bytes | bytearray) -> None:
    """Load the history from JSON."""
    assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `load_json`'
    self.history = self._snapshots_type_adapter.validate_json(json_data)

```

### FileStatePersistence

Bases: `BaseStatePersistence[StateT, RunEndT]`

File based state persistence that hold graph run state in a JSON file.

Source code in `pydantic_graph/pydantic_graph/persistence/file.py`

````python
@dataclass
class FileStatePersistence(BaseStatePersistence[StateT, RunEndT]):
    """File based state persistence that hold graph run state in a JSON file."""

    json_file: Path
    """Path to the JSON file where the snapshots are stored.

    You should use a different file for each graph run, but a single file should be reused for multiple
    steps of the same run.

    For example if you have a run ID of the form `run_123abc`, you might create a `FileStatePersistence` thus:

    ```py
    from pathlib import Path

    from pydantic_graph import FullStatePersistence

    run_id = 'run_123abc'
    persistence = FullStatePersistence(Path('runs') / f'{run_id}.json')
    ```
    """
    _snapshots_type_adapter: pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = field(
        default=None, init=False, repr=False
    )

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        await self._append_save(NodeSnapshot(state=state, node=next_node))

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
    ) -> None:
        async with self._lock():
            snapshots = await self.load_all()
            if not any(s.id == snapshot_id for s in snapshots):  # pragma: no branch
                await self._append_save(NodeSnapshot(state=state, node=next_node), lock=False)

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        await self._append_save(EndSnapshot(state=state, result=end))

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        async with self._lock():
            snapshots = await self.load_all()
            try:
                snapshot = next(s for s in snapshots if s.id == snapshot_id)
            except StopIteration as e:
                raise LookupError(f'No snapshot found with id={snapshot_id!r}') from e

            assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
            exceptions.GraphNodeStatusError.check(snapshot.status)
            snapshot.status = 'running'
            snapshot.start_ts = _utils.now_utc()
            await self._save(snapshots)

        start = perf_counter()
        try:
            yield
        except Exception:
            duration = perf_counter() - start
            async with self._lock():
                await _graph_utils.run_in_executor(self._after_run_sync, snapshot_id, duration, 'error')
            raise
        else:
            snapshot.duration = perf_counter() - start
            async with self._lock():
                await _graph_utils.run_in_executor(self._after_run_sync, snapshot_id, snapshot.duration, 'success')

    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        async with self._lock():
            snapshots = await self.load_all()
            if snapshot := next((s for s in snapshots if isinstance(s, NodeSnapshot) and s.status == 'created'), None):
                snapshot.status = 'pending'
                await self._save(snapshots)
                return snapshot

    def should_set_types(self) -> bool:
        """Whether types need to be set."""
        return self._snapshots_type_adapter is None

    def set_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
        self._snapshots_type_adapter = build_snapshot_list_type_adapter(state_type, run_end_type)

    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        return await _graph_utils.run_in_executor(self._load_sync)

    def _load_sync(self) -> list[Snapshot[StateT, RunEndT]]:
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
        try:
            content = self.json_file.read_bytes()
        except FileNotFoundError:
            return []
        else:
            return self._snapshots_type_adapter.validate_json(content)

    def _after_run_sync(self, snapshot_id: str, duration: float, status: SnapshotStatus) -> None:
        snapshots = self._load_sync()
        snapshot = next(s for s in snapshots if s.id == snapshot_id)
        assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
        snapshot.duration = duration
        snapshot.status = status
        self._save_sync(snapshots)

    async def _save(self, snapshots: list[Snapshot[StateT, RunEndT]]) -> None:
        await _graph_utils.run_in_executor(self._save_sync, snapshots)

    def _save_sync(self, snapshots: list[Snapshot[StateT, RunEndT]]) -> None:
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
        self.json_file.write_bytes(self._snapshots_type_adapter.dump_json(snapshots, indent=2))

    async def _append_save(self, snapshot: Snapshot[StateT, RunEndT], *, lock: bool = True) -> None:
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
        async with AsyncExitStack() as stack:
            if lock:
                await stack.enter_async_context(self._lock())
            snapshots = await self.load_all()
            snapshots.append(snapshot)
            await self._save(snapshots)

    @asynccontextmanager
    async def _lock(self, *, timeout: float = 1.0) -> AsyncIterator[None]:
        """Lock a file by checking and writing a `.pydantic-graph-persistence-lock` to it.

        Args:
            timeout: how long to wait for the lock

        Returns: an async context manager that holds the lock
        """
        lock_file = self.json_file.parent / f'{self.json_file.name}.pydantic-graph-persistence-lock'
        lock_id = secrets.token_urlsafe().encode()

        with anyio.fail_after(timeout):
            while not await _file_append_check(lock_file, lock_id):
                await anyio.sleep(0.01)

        try:
            yield
        finally:
            await _graph_utils.run_in_executor(lock_file.unlink, missing_ok=True)

````

#### json_file

```python
json_file: Path

```

Path to the JSON file where the snapshots are stored.

You should use a different file for each graph run, but a single file should be reused for multiple steps of the same run.

For example if you have a run ID of the form `run_123abc`, you might create a `FileStatePersistence` thus:

```py
from pathlib import Path

from pydantic_graph import FullStatePersistence

run_id = 'run_123abc'
persistence = FullStatePersistence(Path('runs') / f'{run_id}.json')

```

#### should_set_types

```python
should_set_types() -> bool

```

Whether types need to be set.

Source code in `pydantic_graph/pydantic_graph/persistence/file.py`

```python
def should_set_types(self) -> bool:
    """Whether types need to be set."""
    return self._snapshots_type_adapter is None

```

