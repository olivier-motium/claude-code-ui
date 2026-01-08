<!-- Source: _complete-doc.md -->

# `pydantic_graph.beta.decision`

Decision node implementation for conditional branching in graph execution.

This module provides the Decision node type and related classes for implementing conditional branching logic in parallel control flow graphs. Decision nodes allow the graph to choose different execution paths based on runtime conditions.

### StateT

```python
StateT = TypeVar('StateT', infer_variance=True)

```

Type variable for graph state.

### DepsT

```python
DepsT = TypeVar('DepsT', infer_variance=True)

```

Type variable for graph dependencies.

### HandledT

```python
HandledT = TypeVar('HandledT', infer_variance=True)

```

Type variable used to track types handled by the branches of a Decision.

### T

```python
T = TypeVar('T', infer_variance=True)

```

Generic type variable.

### Decision

Bases: `Generic[StateT, DepsT, HandledT]`

Decision node for conditional branching in graph execution.

A Decision node evaluates conditions and routes execution to different branches based on the input data type or custom matching logic.

Source code in `pydantic_graph/pydantic_graph/beta/decision.py`

```python
@dataclass(kw_only=True)
class Decision(Generic[StateT, DepsT, HandledT]):
    """Decision node for conditional branching in graph execution.

    A Decision node evaluates conditions and routes execution to different
    branches based on the input data type or custom matching logic.
    """

    id: NodeID
    """Unique identifier for this decision node."""

    branches: list[DecisionBranch[Any]]
    """List of branches that can be taken from this decision."""

    note: str | None
    """Optional documentation note for this decision."""

    def branch(self, branch: DecisionBranch[T]) -> Decision[StateT, DepsT, HandledT | T]:
        """Add a new branch to this decision.

        Args:
            branch: The branch to add to this decision.

        Returns:
            A new Decision with the additional branch.
        """
        return Decision(id=self.id, branches=self.branches + [branch], note=self.note)

    def _force_handled_contravariant(self, inputs: HandledT) -> Never:  # pragma: no cover
        """Forces this type to be contravariant in the HandledT type variable.

        This is an implementation detail of how we can type-check that all possible input types have
        been exhaustively covered.

        Args:
            inputs: Input data of handled types.

        Raises:
            RuntimeError: Always, as this method should never be executed.
        """
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')

```

#### id

```python
id: NodeID

```

Unique identifier for this decision node.

#### branches

```python
branches: list[DecisionBranch[Any]]

```

List of branches that can be taken from this decision.

#### note

```python
note: str | None

```

Optional documentation note for this decision.

#### branch

```python
branch(
    branch: DecisionBranch[T],
) -> Decision[StateT, DepsT, HandledT | T]

```

Add a new branch to this decision.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `branch` | `DecisionBranch[T]` | The branch to add to this decision. | *required* |

Returns:

| Type | Description | | --- | --- | | `Decision[StateT, DepsT, HandledT | T]` | A new Decision with the additional branch. |

Source code in `pydantic_graph/pydantic_graph/beta/decision.py`

```python
def branch(self, branch: DecisionBranch[T]) -> Decision[StateT, DepsT, HandledT | T]:
    """Add a new branch to this decision.

    Args:
        branch: The branch to add to this decision.

    Returns:
        A new Decision with the additional branch.
    """
    return Decision(id=self.id, branches=self.branches + [branch], note=self.note)

```

### SourceT

```python
SourceT = TypeVar('SourceT', infer_variance=True)

```

Type variable for source data for a DecisionBranch.

### DecisionBranch

Bases: `Generic[SourceT]`

Represents a single branch within a decision node.

Each branch defines the conditions under which it should be taken and the path to follow when those conditions are met.

Note: with the current design, it is actually *critical* that this class is invariant in SourceT for the sake of type-checking that inputs to a Decision are actually handled. See the `# type: ignore` comment in `tests.graph.beta.test_graph_edge_cases.test_decision_no_matching_branch` for an example of how this works.

Source code in `pydantic_graph/pydantic_graph/beta/decision.py`

```python
@dataclass
class DecisionBranch(Generic[SourceT]):
    """Represents a single branch within a decision node.

    Each branch defines the conditions under which it should be taken
    and the path to follow when those conditions are met.

    Note: with the current design, it is actually _critical_ that this class is invariant in SourceT for the sake
    of type-checking that inputs to a Decision are actually handled. See the `# type: ignore` comment in
    `tests.graph.beta.test_graph_edge_cases.test_decision_no_matching_branch` for an example of how this works.
    """

    source: TypeOrTypeExpression[SourceT]
    """The expected type of data for this branch.

    This is necessary for exhaustiveness-checking when handling the inputs to a decision node."""

    matches: Callable[[Any], bool] | None
    """An optional predicate function used to determine whether input data matches this branch.

    If `None`, default logic is used which attempts to check the value for type-compatibility with the `source` type:
    * If `source` is `Any` or `object`, the branch will always match
    * If `source` is a `Literal` type, this branch will match if the value is one of the parametrizing literal values
    * If `source` is any other type, the value will be checked for matching using `isinstance`

    Inputs are tested against each branch of a decision node in order, and the path of the first matching branch is
    used to handle the input value.
    """

    path: Path
    """The execution path to follow when an input value matches this branch of a decision node.

    This can include transforming, mapping, and broadcasting the output before sending to the next node or nodes.

    The path can also include position-aware labels which are used when generating mermaid diagrams."""

    destinations: list[AnyDestinationNode]
    """The destination nodes that can be referenced by DestinationMarker in the path."""

```

#### source

```python
source: TypeOrTypeExpression[SourceT]

```

The expected type of data for this branch.

This is necessary for exhaustiveness-checking when handling the inputs to a decision node.

#### matches

```python
matches: Callable[[Any], bool] | None

```

An optional predicate function used to determine whether input data matches this branch.

If `None`, default logic is used which attempts to check the value for type-compatibility with the `source` type: * If `source` is `Any` or `object`, the branch will always match * If `source` is a `Literal` type, this branch will match if the value is one of the parametrizing literal values * If `source` is any other type, the value will be checked for matching using `isinstance`

Inputs are tested against each branch of a decision node in order, and the path of the first matching branch is used to handle the input value.

#### path

```python
path: Path

```

The execution path to follow when an input value matches this branch of a decision node.

This can include transforming, mapping, and broadcasting the output before sending to the next node or nodes.

The path can also include position-aware labels which are used when generating mermaid diagrams.

#### destinations

```python
destinations: list[AnyDestinationNode]

```

The destination nodes that can be referenced by DestinationMarker in the path.

### OutputT

```python
OutputT = TypeVar('OutputT', infer_variance=True)

```

Type variable for the output data of a node.

### NewOutputT

```python
NewOutputT = TypeVar('NewOutputT', infer_variance=True)

```

Type variable for transformed output.

### DecisionBranchBuilder

Bases: `Generic[StateT, DepsT, OutputT, SourceT, HandledT]`

Builder for constructing decision branches with fluent API.

This builder provides methods to configure branches with destinations, forks, and transformations in a type-safe manner.

Instances of this class should be created using GraphBuilder.match, not created directly.

Source code in `pydantic_graph/pydantic_graph/beta/decision.py`

```python
@dataclass(init=False)
class DecisionBranchBuilder(Generic[StateT, DepsT, OutputT, SourceT, HandledT]):
    """Builder for constructing decision branches with fluent API.

    This builder provides methods to configure branches with destinations,
    forks, and transformations in a type-safe manner.

    Instances of this class should be created using [`GraphBuilder.match`][pydantic_graph.beta.graph_builder.GraphBuilder],
    not created directly.
    """

    _decision: Decision[StateT, DepsT, HandledT]
    """The parent decision node."""
    _source: TypeOrTypeExpression[SourceT]
    """The expected source type for this branch."""
    _matches: Callable[[Any], bool] | None
    """Optional matching predicate."""

    _path_builder: PathBuilder[StateT, DepsT, OutputT]
    """Builder for the execution path."""

    def __init__(
        self,
        *,
        decision: Decision[StateT, DepsT, HandledT],
        source: TypeOrTypeExpression[SourceT],
        matches: Callable[[Any], bool] | None,
        path_builder: PathBuilder[StateT, DepsT, OutputT],
    ):
        # This manually-defined initializer is necessary due to https://github.com/python/mypy/issues/17623.
        self._decision = decision
        self._source = source
        self._matches = matches
        self._path_builder = path_builder

    def to(
        self,
        destination: DestinationNode[StateT, DepsT, OutputT] | type[BaseNode[StateT, DepsT, Any]],
        /,
        *extra_destinations: DestinationNode[StateT, DepsT, OutputT] | type[BaseNode[StateT, DepsT, Any]],
        fork_id: str | None = None,
    ) -> DecisionBranch[SourceT]:
        """Set the destination(s) for this branch.

        Args:
            destination: The primary destination node.
            *extra_destinations: Additional destination nodes.
            fork_id: Optional node ID to use for the resulting broadcast fork if multiple destinations are provided.

        Returns:
            A completed DecisionBranch with the specified destinations.
        """
        destination = get_origin(destination) or destination
        extra_destinations = tuple(get_origin(d) or d for d in extra_destinations)
        destinations = [(NodeStep(d) if inspect.isclass(d) else d) for d in (destination, *extra_destinations)]
        return DecisionBranch(
            source=self._source,
            matches=self._matches,
            path=self._path_builder.to(*destinations, fork_id=fork_id),
            destinations=destinations,
        )

    def broadcast(
        self, get_forks: Callable[[Self], Sequence[DecisionBranch[SourceT]]], /, *, fork_id: str | None = None
    ) -> DecisionBranch[SourceT]:
        """Broadcast this decision branch into multiple destinations.

        Args:
            get_forks: The callback that will return a sequence of decision branches to broadcast to.
            fork_id: Optional node ID to use for the resulting broadcast fork.

        Returns:
            A completed DecisionBranch with the specified destinations.
        """
        fork_decision_branches = get_forks(self)
        new_paths = [b.path for b in fork_decision_branches]
        if not new_paths:
            raise GraphBuildingError(f'The call to {get_forks} returned no branches, but must return at least one.')
        path = self._path_builder.broadcast(new_paths, fork_id=fork_id)
        destinations = [d for fdp in fork_decision_branches for d in fdp.destinations]
        return DecisionBranch(source=self._source, matches=self._matches, path=path, destinations=destinations)

    def transform(
        self, func: TransformFunction[StateT, DepsT, OutputT, NewOutputT], /
    ) -> DecisionBranchBuilder[StateT, DepsT, NewOutputT, SourceT, HandledT]:
        """Apply a transformation to the branch's output.

        Args:
            func: Transformation function to apply.

        Returns:
            A new DecisionBranchBuilder where the provided transform is applied prior to generating the final output.
        """
        return DecisionBranchBuilder(
            decision=self._decision,
            source=self._source,
            matches=self._matches,
            path_builder=self._path_builder.transform(func),
        )

    def map(
        self: DecisionBranchBuilder[StateT, DepsT, Iterable[T], SourceT, HandledT]
        | DecisionBranchBuilder[StateT, DepsT, AsyncIterable[T], SourceT, HandledT],
        *,
        fork_id: str | None = None,
        downstream_join_id: str | None = None,
    ) -> DecisionBranchBuilder[StateT, DepsT, T, SourceT, HandledT]:
        """Spread the branch's output.

        To do this, the current output must be iterable, and any subsequent steps in the path being built for this
        branch will be applied to each item of the current output in parallel.

        Args:
            fork_id: Optional ID for the fork, defaults to a generated value
            downstream_join_id: Optional ID of a downstream join node which is involved when mapping empty iterables

        Returns:
            A new DecisionBranchBuilder where mapping is performed prior to generating the final output.
        """
        return DecisionBranchBuilder(
            decision=self._decision,
            source=self._source,
            matches=self._matches,
            path_builder=self._path_builder.map(fork_id=fork_id, downstream_join_id=downstream_join_id),
        )

    def label(self, label: str) -> DecisionBranchBuilder[StateT, DepsT, OutputT, SourceT, HandledT]:
        """Apply a label to the branch at the current point in the path being built.

        These labels are only used in generated mermaid diagrams.

        Args:
            label: The label to apply.

        Returns:
            A new DecisionBranchBuilder where the label has been applied at the end of the current path being built.
        """
        return DecisionBranchBuilder(
            decision=self._decision,
            source=self._source,
            matches=self._matches,
            path_builder=self._path_builder.label(label),
        )

```

#### to

```python
to(
    destination: (
        DestinationNode[StateT, DepsT, OutputT]
        | type[BaseNode[StateT, DepsT, Any]]
    ),
    /,
    *extra_destinations: DestinationNode[
        StateT, DepsT, OutputT
    ]
    | type[BaseNode[StateT, DepsT, Any]],
    fork_id: str | None = None,
) -> DecisionBranch[SourceT]

```

Set the destination(s) for this branch.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `destination` | `DestinationNode[StateT, DepsT, OutputT] | type[BaseNode[StateT, DepsT, Any]]` | The primary destination node. | *required* | | `*extra_destinations` | `DestinationNode[StateT, DepsT, OutputT] | type[BaseNode[StateT, DepsT, Any]]` | Additional destination nodes. | `()` | | `fork_id` | `str | None` | Optional node ID to use for the resulting broadcast fork if multiple destinations are provided. | `None` |

Returns:

| Type | Description | | --- | --- | | `DecisionBranch[SourceT]` | A completed DecisionBranch with the specified destinations. |

Source code in `pydantic_graph/pydantic_graph/beta/decision.py`

```python
def to(
    self,
    destination: DestinationNode[StateT, DepsT, OutputT] | type[BaseNode[StateT, DepsT, Any]],
    /,
    *extra_destinations: DestinationNode[StateT, DepsT, OutputT] | type[BaseNode[StateT, DepsT, Any]],
    fork_id: str | None = None,
) -> DecisionBranch[SourceT]:
    """Set the destination(s) for this branch.

    Args:
        destination: The primary destination node.
        *extra_destinations: Additional destination nodes.
        fork_id: Optional node ID to use for the resulting broadcast fork if multiple destinations are provided.

    Returns:
        A completed DecisionBranch with the specified destinations.
    """
    destination = get_origin(destination) or destination
    extra_destinations = tuple(get_origin(d) or d for d in extra_destinations)
    destinations = [(NodeStep(d) if inspect.isclass(d) else d) for d in (destination, *extra_destinations)]
    return DecisionBranch(
        source=self._source,
        matches=self._matches,
        path=self._path_builder.to(*destinations, fork_id=fork_id),
        destinations=destinations,
    )

```

#### broadcast

```python
broadcast(
    get_forks: Callable[
        [Self], Sequence[DecisionBranch[SourceT]]
    ],
    /,
    *,
    fork_id: str | None = None,
) -> DecisionBranch[SourceT]

```

Broadcast this decision branch into multiple destinations.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `get_forks` | `Callable[[Self], Sequence[DecisionBranch[SourceT]]]` | The callback that will return a sequence of decision branches to broadcast to. | *required* | | `fork_id` | `str | None` | Optional node ID to use for the resulting broadcast fork. | `None` |

Returns:

| Type | Description | | --- | --- | | `DecisionBranch[SourceT]` | A completed DecisionBranch with the specified destinations. |

Source code in `pydantic_graph/pydantic_graph/beta/decision.py`

```python
def broadcast(
    self, get_forks: Callable[[Self], Sequence[DecisionBranch[SourceT]]], /, *, fork_id: str | None = None
) -> DecisionBranch[SourceT]:
    """Broadcast this decision branch into multiple destinations.

    Args:
        get_forks: The callback that will return a sequence of decision branches to broadcast to.
        fork_id: Optional node ID to use for the resulting broadcast fork.

    Returns:
        A completed DecisionBranch with the specified destinations.
    """
    fork_decision_branches = get_forks(self)
    new_paths = [b.path for b in fork_decision_branches]
    if not new_paths:
        raise GraphBuildingError(f'The call to {get_forks} returned no branches, but must return at least one.')
    path = self._path_builder.broadcast(new_paths, fork_id=fork_id)
    destinations = [d for fdp in fork_decision_branches for d in fdp.destinations]
    return DecisionBranch(source=self._source, matches=self._matches, path=path, destinations=destinations)

```

#### transform

```python
transform(
    func: TransformFunction[
        StateT, DepsT, OutputT, NewOutputT
    ],
) -> DecisionBranchBuilder[
    StateT, DepsT, NewOutputT, SourceT, HandledT
]

```

Apply a transformation to the branch's output.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `func` | `TransformFunction[StateT, DepsT, OutputT, NewOutputT]` | Transformation function to apply. | *required* |

Returns:

| Type | Description | | --- | --- | | `DecisionBranchBuilder[StateT, DepsT, NewOutputT, SourceT, HandledT]` | A new DecisionBranchBuilder where the provided transform is applied prior to generating the final output. |

Source code in `pydantic_graph/pydantic_graph/beta/decision.py`

```python
def transform(
    self, func: TransformFunction[StateT, DepsT, OutputT, NewOutputT], /
) -> DecisionBranchBuilder[StateT, DepsT, NewOutputT, SourceT, HandledT]:
    """Apply a transformation to the branch's output.

    Args:
        func: Transformation function to apply.

    Returns:
        A new DecisionBranchBuilder where the provided transform is applied prior to generating the final output.
    """
    return DecisionBranchBuilder(
        decision=self._decision,
        source=self._source,
        matches=self._matches,
        path_builder=self._path_builder.transform(func),
    )

```

#### map

```python
map(
    *,
    fork_id: str | None = None,
    downstream_join_id: str | None = None
) -> DecisionBranchBuilder[
    StateT, DepsT, T, SourceT, HandledT
]

```

Spread the branch's output.

To do this, the current output must be iterable, and any subsequent steps in the path being built for this branch will be applied to each item of the current output in parallel.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `fork_id` | `str | None` | Optional ID for the fork, defaults to a generated value | `None` | | `downstream_join_id` | `str | None` | Optional ID of a downstream join node which is involved when mapping empty iterables | `None` |

Returns:

| Type | Description | | --- | --- | | `DecisionBranchBuilder[StateT, DepsT, T, SourceT, HandledT]` | A new DecisionBranchBuilder where mapping is performed prior to generating the final output. |

Source code in `pydantic_graph/pydantic_graph/beta/decision.py`

```python
def map(
    self: DecisionBranchBuilder[StateT, DepsT, Iterable[T], SourceT, HandledT]
    | DecisionBranchBuilder[StateT, DepsT, AsyncIterable[T], SourceT, HandledT],
    *,
    fork_id: str | None = None,
    downstream_join_id: str | None = None,
) -> DecisionBranchBuilder[StateT, DepsT, T, SourceT, HandledT]:
    """Spread the branch's output.

    To do this, the current output must be iterable, and any subsequent steps in the path being built for this
    branch will be applied to each item of the current output in parallel.

    Args:
        fork_id: Optional ID for the fork, defaults to a generated value
        downstream_join_id: Optional ID of a downstream join node which is involved when mapping empty iterables

    Returns:
        A new DecisionBranchBuilder where mapping is performed prior to generating the final output.
    """
    return DecisionBranchBuilder(
        decision=self._decision,
        source=self._source,
        matches=self._matches,
        path_builder=self._path_builder.map(fork_id=fork_id, downstream_join_id=downstream_join_id),
    )

```

#### label

```python
label(
    label: str,
) -> DecisionBranchBuilder[
    StateT, DepsT, OutputT, SourceT, HandledT
]

```

Apply a label to the branch at the current point in the path being built.

These labels are only used in generated mermaid diagrams.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `label` | `str` | The label to apply. | *required* |

Returns:

| Type | Description | | --- | --- | | `DecisionBranchBuilder[StateT, DepsT, OutputT, SourceT, HandledT]` | A new DecisionBranchBuilder where the label has been applied at the end of the current path being built. |

Source code in `pydantic_graph/pydantic_graph/beta/decision.py`

```python
def label(self, label: str) -> DecisionBranchBuilder[StateT, DepsT, OutputT, SourceT, HandledT]:
    """Apply a label to the branch at the current point in the path being built.

    These labels are only used in generated mermaid diagrams.

    Args:
        label: The label to apply.

    Returns:
        A new DecisionBranchBuilder where the label has been applied at the end of the current path being built.
    """
    return DecisionBranchBuilder(
        decision=self._decision,
        source=self._source,
        matches=self._matches,
        path_builder=self._path_builder.label(label),
    )

```

# `pydantic_graph.beta.graph`

Core graph execution engine for the next version of the pydantic-graph library.

This module provides the main `Graph` class and `GraphRun` execution engine that handles the orchestration of nodes, edges, and parallel execution paths in the graph-based workflow system.

### StateT

```python
StateT = TypeVar('StateT', infer_variance=True)

```

Type variable for graph state.

### DepsT

```python
DepsT = TypeVar('DepsT', infer_variance=True)

```

Type variable for graph dependencies.

### InputT

```python
InputT = TypeVar('InputT', infer_variance=True)

```

Type variable for graph inputs.

### OutputT

```python
OutputT = TypeVar('OutputT', infer_variance=True)

```

Type variable for graph outputs.

### EndMarker

Bases: `Generic[OutputT]`

A marker indicating the end of graph execution with a final value.

EndMarker is used internally to signal that the graph has completed execution and carries the final output value.

Type Parameters

OutputT: The type of the final output value

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
@dataclass(init=False)
class EndMarker(Generic[OutputT]):
    """A marker indicating the end of graph execution with a final value.

    EndMarker is used internally to signal that the graph has completed
    execution and carries the final output value.

    Type Parameters:
        OutputT: The type of the final output value
    """

    _value: OutputT
    """The final output value from the graph execution."""

    def __init__(self, value: OutputT):
        # This manually-defined initializer is necessary due to https://github.com/python/mypy/issues/17623.
        self._value = value

    @property
    def value(self) -> OutputT:
        return self._value

```

### JoinItem

An item representing data flowing into a join operation.

JoinItem carries input data from a parallel execution path to a join node, along with metadata about which execution 'fork' it originated from.

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
@dataclass
class JoinItem:
    """An item representing data flowing into a join operation.

    JoinItem carries input data from a parallel execution path to a join
    node, along with metadata about which execution 'fork' it originated from.
    """

    join_id: JoinID
    """The ID of the join node this item is targeting."""

    inputs: Any
    """The input data for the join operation."""

    fork_stack: ForkStack
    """The stack of ForkStackItems that led to producing this join item."""

```

#### join_id

```python
join_id: JoinID

```

The ID of the join node this item is targeting.

#### inputs

```python
inputs: Any

```

The input data for the join operation.

#### fork_stack

```python
fork_stack: ForkStack

```

The stack of ForkStackItems that led to producing this join item.

### Graph

Bases: `Generic[StateT, DepsT, InputT, OutputT]`

A complete graph definition ready for execution.

The Graph class represents a complete workflow graph with typed inputs, outputs, state, and dependencies. It contains all nodes, edges, and metadata needed for execution.

Type Parameters

StateT: The type of the graph state DepsT: The type of the dependencies InputT: The type of the input data OutputT: The type of the output data

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
@dataclass(repr=False)
class Graph(Generic[StateT, DepsT, InputT, OutputT]):
    """A complete graph definition ready for execution.

    The Graph class represents a complete workflow graph with typed inputs,
    outputs, state, and dependencies. It contains all nodes, edges, and
    metadata needed for execution.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of the input data
        OutputT: The type of the output data
    """

    name: str | None
    """Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method."""

    state_type: type[StateT]
    """The type of the graph state."""

    deps_type: type[DepsT]
    """The type of the dependencies."""

    input_type: type[InputT]
    """The type of the input data."""

    output_type: type[OutputT]
    """The type of the output data."""

    auto_instrument: bool
    """Whether to automatically create instrumentation spans."""

    nodes: dict[NodeID, AnyNode]
    """All nodes in the graph indexed by their ID."""

    edges_by_source: dict[NodeID, list[Path]]
    """Outgoing paths from each source node."""

    parent_forks: dict[JoinID, ParentFork[NodeID]]
    """Parent fork information for each join node."""

    intermediate_join_nodes: dict[JoinID, set[JoinID]]
    """For each join, the set of other joins that appear between it and its parent fork.

    Used to determine which joins are "final" (have no other joins as intermediates) and
    which joins should preserve fork stacks when proceeding downstream."""

    def get_parent_fork(self, join_id: JoinID) -> ParentFork[NodeID]:
        """Get the parent fork information for a join node.

        Args:
            join_id: The ID of the join node

        Returns:
            The parent fork information for the join

        Raises:
            RuntimeError: If the join ID is not found or has no parent fork
        """
        result = self.parent_forks.get(join_id)
        if result is None:
            raise RuntimeError(f'Node {join_id} is not a join node or did not have a dominating fork (this is a bug)')
        return result

    def is_final_join(self, join_id: JoinID) -> bool:
        """Check if a join is 'final' (has no downstream joins with the same parent fork).

        A join is non-final if it appears as an intermediate node for another join
        with the same parent fork.

        Args:
            join_id: The ID of the join node

        Returns:
            True if the join is final, False if it's non-final
        """
        # Check if this join appears in any other join's intermediate_join_nodes
        for intermediate_joins in self.intermediate_join_nodes.values():
            if join_id in intermediate_joins:
                return False
        return True

    async def run(
        self,
        *,
        state: StateT = None,
        deps: DepsT = None,
        inputs: InputT = None,
        span: AbstractContextManager[AbstractSpan] | None = None,
        infer_name: bool = True,
    ) -> OutputT:
        """Execute the graph and return the final output.

        This is the main entry point for graph execution. It runs the graph
        to completion and returns the final output value.

        Args:
            state: The graph state instance
            deps: The dependencies instance
            inputs: The input data for the graph
            span: Optional span for tracing/instrumentation
            infer_name: Whether to infer the graph name from the calling frame.

        Returns:
            The final output from the graph execution
        """
        if infer_name and self.name is None:
            inferred_name = infer_obj_name(self, depth=2)
            if inferred_name is not None:  # pragma: no branch
                self.name = inferred_name

        async with self.iter(state=state, deps=deps, inputs=inputs, span=span, infer_name=False) as graph_run:
            # Note: This would probably be better using `async for _ in graph_run`, but this tests the `next` method,
            # which I'm less confident will be implemented correctly if not used on the critical path. We can change it
            # once we have tests, etc.
            event: Any = None
            while True:
                try:
                    event = await graph_run.next(event)
                except StopAsyncIteration:
                    assert isinstance(event, EndMarker), 'Graph run should end with an EndMarker.'
                    return cast(EndMarker[OutputT], event).value

    @asynccontextmanager
    async def iter(
        self,
        *,
        state: StateT = None,
        deps: DepsT = None,
        inputs: InputT = None,
        span: AbstractContextManager[AbstractSpan] | None = None,
        infer_name: bool = True,
    ) -> AsyncIterator[GraphRun[StateT, DepsT, OutputT]]:
        """Create an iterator for step-by-step graph execution.

        This method allows for more fine-grained control over graph execution,
        enabling inspection of intermediate states and results.

        Args:
            state: The graph state instance
            deps: The dependencies instance
            inputs: The input data for the graph
            span: Optional span for tracing/instrumentation
            infer_name: Whether to infer the graph name from the calling frame.

        Yields:
            A GraphRun instance that can be iterated for step-by-step execution
        """
        if infer_name and self.name is None:
            inferred_name = infer_obj_name(self, depth=3)  # depth=3 because asynccontextmanager adds one
            if inferred_name is not None:  # pragma: no branch
                self.name = inferred_name

        with ExitStack() as stack:
            entered_span: AbstractSpan | None = None
            if span is None:
                if self.auto_instrument:
                    entered_span = stack.enter_context(logfire_span('run graph {graph.name}', graph=self))
            else:
                entered_span = stack.enter_context(span)
            traceparent = None if entered_span is None else get_traceparent(entered_span)
            async with GraphRun[StateT, DepsT, OutputT](
                graph=self,
                state=state,
                deps=deps,
                inputs=inputs,
                traceparent=traceparent,
            ) as graph_run:
                yield graph_run

    def render(self, *, title: str | None = None, direction: StateDiagramDirection | None = None) -> str:
        """Render the graph as a Mermaid diagram string.

        Args:
            title: Optional title for the diagram
            direction: Optional direction for the diagram layout

        Returns:
            A string containing the Mermaid diagram representation
        """
        from pydantic_graph.beta.mermaid import build_mermaid_graph

        return build_mermaid_graph(self.nodes, self.edges_by_source).render(title=title, direction=direction)

    def __repr__(self) -> str:
        super_repr = super().__repr__()  # include class and memory address
        # Insert the result of calling `__str__` before the final '>' in the repr
        return f'{super_repr[:-1]}\n{self}\n{super_repr[-1]}'

    def __str__(self) -> str:
        """Return a Mermaid diagram representation of the graph.

        Returns:
            A string containing the Mermaid diagram of the graph
        """
        return self.render()

```

#### name

```python
name: str | None

```

Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method.

#### state_type

```python
state_type: type[StateT]

```

The type of the graph state.

#### deps_type

```python
deps_type: type[DepsT]

```

The type of the dependencies.

#### input_type

```python
input_type: type[InputT]

```

The type of the input data.

#### output_type

```python
output_type: type[OutputT]

```

The type of the output data.

#### auto_instrument

```python
auto_instrument: bool

```

Whether to automatically create instrumentation spans.

#### nodes

```python
nodes: dict[NodeID, AnyNode]

```

All nodes in the graph indexed by their ID.

#### edges_by_source

```python
edges_by_source: dict[NodeID, list[Path]]

```

Outgoing paths from each source node.

#### parent_forks

```python
parent_forks: dict[JoinID, ParentFork[NodeID]]

```

Parent fork information for each join node.

#### intermediate_join_nodes

```python
intermediate_join_nodes: dict[JoinID, set[JoinID]]

```

For each join, the set of other joins that appear between it and its parent fork.

Used to determine which joins are "final" (have no other joins as intermediates) and which joins should preserve fork stacks when proceeding downstream.

#### get_parent_fork

```python
get_parent_fork(join_id: JoinID) -> ParentFork[NodeID]

```

Get the parent fork information for a join node.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `join_id` | `JoinID` | The ID of the join node | *required* |

Returns:

| Type | Description | | --- | --- | | `ParentFork[NodeID]` | The parent fork information for the join |

Raises:

| Type | Description | | --- | --- | | `RuntimeError` | If the join ID is not found or has no parent fork |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
def get_parent_fork(self, join_id: JoinID) -> ParentFork[NodeID]:
    """Get the parent fork information for a join node.

    Args:
        join_id: The ID of the join node

    Returns:
        The parent fork information for the join

    Raises:
        RuntimeError: If the join ID is not found or has no parent fork
    """
    result = self.parent_forks.get(join_id)
    if result is None:
        raise RuntimeError(f'Node {join_id} is not a join node or did not have a dominating fork (this is a bug)')
    return result

```

#### is_final_join

```python
is_final_join(join_id: JoinID) -> bool

```

Check if a join is 'final' (has no downstream joins with the same parent fork).

A join is non-final if it appears as an intermediate node for another join with the same parent fork.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `join_id` | `JoinID` | The ID of the join node | *required* |

Returns:

| Type | Description | | --- | --- | | `bool` | True if the join is final, False if it's non-final |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
def is_final_join(self, join_id: JoinID) -> bool:
    """Check if a join is 'final' (has no downstream joins with the same parent fork).

    A join is non-final if it appears as an intermediate node for another join
    with the same parent fork.

    Args:
        join_id: The ID of the join node

    Returns:
        True if the join is final, False if it's non-final
    """
    # Check if this join appears in any other join's intermediate_join_nodes
    for intermediate_joins in self.intermediate_join_nodes.values():
        if join_id in intermediate_joins:
            return False
    return True

```

#### run

```python
run(
    *,
    state: StateT = None,
    deps: DepsT = None,
    inputs: InputT = None,
    span: (
        AbstractContextManager[AbstractSpan] | None
    ) = None,
    infer_name: bool = True
) -> OutputT

```

Execute the graph and return the final output.

This is the main entry point for graph execution. It runs the graph to completion and returns the final output value.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `state` | `StateT` | The graph state instance | `None` | | `deps` | `DepsT` | The dependencies instance | `None` | | `inputs` | `InputT` | The input data for the graph | `None` | | `span` | `AbstractContextManager[AbstractSpan] | None` | Optional span for tracing/instrumentation | `None` | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` |

Returns:

| Type | Description | | --- | --- | | `OutputT` | The final output from the graph execution |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
async def run(
    self,
    *,
    state: StateT = None,
    deps: DepsT = None,
    inputs: InputT = None,
    span: AbstractContextManager[AbstractSpan] | None = None,
    infer_name: bool = True,
) -> OutputT:
    """Execute the graph and return the final output.

    This is the main entry point for graph execution. It runs the graph
    to completion and returns the final output value.

    Args:
        state: The graph state instance
        deps: The dependencies instance
        inputs: The input data for the graph
        span: Optional span for tracing/instrumentation
        infer_name: Whether to infer the graph name from the calling frame.

    Returns:
        The final output from the graph execution
    """
    if infer_name and self.name is None:
        inferred_name = infer_obj_name(self, depth=2)
        if inferred_name is not None:  # pragma: no branch
            self.name = inferred_name

    async with self.iter(state=state, deps=deps, inputs=inputs, span=span, infer_name=False) as graph_run:
        # Note: This would probably be better using `async for _ in graph_run`, but this tests the `next` method,
        # which I'm less confident will be implemented correctly if not used on the critical path. We can change it
        # once we have tests, etc.
        event: Any = None
        while True:
            try:
                event = await graph_run.next(event)
            except StopAsyncIteration:
                assert isinstance(event, EndMarker), 'Graph run should end with an EndMarker.'
                return cast(EndMarker[OutputT], event).value

```

#### iter

```python
iter(
    *,
    state: StateT = None,
    deps: DepsT = None,
    inputs: InputT = None,
    span: (
        AbstractContextManager[AbstractSpan] | None
    ) = None,
    infer_name: bool = True
) -> AsyncIterator[GraphRun[StateT, DepsT, OutputT]]

```

Create an iterator for step-by-step graph execution.

This method allows for more fine-grained control over graph execution, enabling inspection of intermediate states and results.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `state` | `StateT` | The graph state instance | `None` | | `deps` | `DepsT` | The dependencies instance | `None` | | `inputs` | `InputT` | The input data for the graph | `None` | | `span` | `AbstractContextManager[AbstractSpan] | None` | Optional span for tracing/instrumentation | `None` | | `infer_name` | `bool` | Whether to infer the graph name from the calling frame. | `True` |

Yields:

| Type | Description | | --- | --- | | `AsyncIterator[GraphRun[StateT, DepsT, OutputT]]` | A GraphRun instance that can be iterated for step-by-step execution |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
@asynccontextmanager
async def iter(
    self,
    *,
    state: StateT = None,
    deps: DepsT = None,
    inputs: InputT = None,
    span: AbstractContextManager[AbstractSpan] | None = None,
    infer_name: bool = True,
) -> AsyncIterator[GraphRun[StateT, DepsT, OutputT]]:
    """Create an iterator for step-by-step graph execution.

    This method allows for more fine-grained control over graph execution,
    enabling inspection of intermediate states and results.

    Args:
        state: The graph state instance
        deps: The dependencies instance
        inputs: The input data for the graph
        span: Optional span for tracing/instrumentation
        infer_name: Whether to infer the graph name from the calling frame.

    Yields:
        A GraphRun instance that can be iterated for step-by-step execution
    """
    if infer_name and self.name is None:
        inferred_name = infer_obj_name(self, depth=3)  # depth=3 because asynccontextmanager adds one
        if inferred_name is not None:  # pragma: no branch
            self.name = inferred_name

    with ExitStack() as stack:
        entered_span: AbstractSpan | None = None
        if span is None:
            if self.auto_instrument:
                entered_span = stack.enter_context(logfire_span('run graph {graph.name}', graph=self))
        else:
            entered_span = stack.enter_context(span)
        traceparent = None if entered_span is None else get_traceparent(entered_span)
        async with GraphRun[StateT, DepsT, OutputT](
            graph=self,
            state=state,
            deps=deps,
            inputs=inputs,
            traceparent=traceparent,
        ) as graph_run:
            yield graph_run

```

#### render

```python
render(
    *,
    title: str | None = None,
    direction: StateDiagramDirection | None = None
) -> str

```

Render the graph as a Mermaid diagram string.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `title` | `str | None` | Optional title for the diagram | `None` | | `direction` | `StateDiagramDirection | None` | Optional direction for the diagram layout | `None` |

Returns:

| Type | Description | | --- | --- | | `str` | A string containing the Mermaid diagram representation |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
def render(self, *, title: str | None = None, direction: StateDiagramDirection | None = None) -> str:
    """Render the graph as a Mermaid diagram string.

    Args:
        title: Optional title for the diagram
        direction: Optional direction for the diagram layout

    Returns:
        A string containing the Mermaid diagram representation
    """
    from pydantic_graph.beta.mermaid import build_mermaid_graph

    return build_mermaid_graph(self.nodes, self.edges_by_source).render(title=title, direction=direction)

```

#### __str__

```python
__str__() -> str

```

Return a Mermaid diagram representation of the graph.

Returns:

| Type | Description | | --- | --- | | `str` | A string containing the Mermaid diagram of the graph |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
def __str__(self) -> str:
    """Return a Mermaid diagram representation of the graph.

    Returns:
        A string containing the Mermaid diagram of the graph
    """
    return self.render()

```

### GraphTaskRequest

A request to run a task representing the execution of a node in the graph.

GraphTaskRequest encapsulates all the information needed to execute a specific node, including its inputs and the fork context it's executing within.

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
@dataclass
class GraphTaskRequest:
    """A request to run a task representing the execution of a node in the graph.

    GraphTaskRequest encapsulates all the information needed to execute a specific
    node, including its inputs and the fork context it's executing within.
    """

    node_id: NodeID
    """The ID of the node to execute."""

    inputs: Any
    """The input data for the node."""

    fork_stack: ForkStack = field(repr=False)
    """Stack of forks that have been entered.

    Used by the GraphRun to decide when to proceed through joins.
    """

```

#### node_id

```python
node_id: NodeID

```

The ID of the node to execute.

#### inputs

```python
inputs: Any

```

The input data for the node.

#### fork_stack

```python
fork_stack: ForkStack = field(repr=False)

```

Stack of forks that have been entered.

Used by the GraphRun to decide when to proceed through joins.

### GraphTask

Bases: `GraphTaskRequest`

A task representing the execution of a node in the graph.

GraphTask encapsulates all the information needed to execute a specific node, including its inputs and the fork context it's executing within, and has a unique ID to identify the task within the graph run.

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
@dataclass
class GraphTask(GraphTaskRequest):
    """A task representing the execution of a node in the graph.

    GraphTask encapsulates all the information needed to execute a specific
    node, including its inputs and the fork context it's executing within,
    and has a unique ID to identify the task within the graph run.
    """

    task_id: TaskID = field(repr=False)
    """Unique identifier for this task."""

    @staticmethod
    def from_request(request: GraphTaskRequest, get_task_id: Callable[[], TaskID]) -> GraphTask:
        # Don't call the get_task_id callable, this is already a task
        if isinstance(request, GraphTask):
            return request
        return GraphTask(request.node_id, request.inputs, request.fork_stack, get_task_id())

```

#### task_id

```python
task_id: TaskID = field(repr=False)

```

Unique identifier for this task.

### GraphRun

Bases: `Generic[StateT, DepsT, OutputT]`

A single execution instance of a graph.

GraphRun manages the execution state for a single run of a graph, including task scheduling, fork/join coordination, and result tracking.

Type Parameters

StateT: The type of the graph state DepsT: The type of the dependencies OutputT: The type of the output data

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
class GraphRun(Generic[StateT, DepsT, OutputT]):
    """A single execution instance of a graph.

    GraphRun manages the execution state for a single run of a graph,
    including task scheduling, fork/join coordination, and result tracking.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        OutputT: The type of the output data
    """

    def __init__(
        self,
        graph: Graph[StateT, DepsT, InputT, OutputT],
        *,
        state: StateT,
        deps: DepsT,
        inputs: InputT,
        traceparent: str | None,
    ):
        """Initialize a graph run.

        Args:
            graph: The graph to execute
            state: The graph state instance
            deps: The dependencies instance
            inputs: The input data for the graph
            traceparent: Optional trace parent for instrumentation
        """
        self.graph = graph
        """The graph being executed."""

        self.state = state
        """The graph state instance."""

        self.deps = deps
        """The dependencies instance."""

        self.inputs = inputs
        """The initial input data."""

        self._active_reducers: dict[tuple[JoinID, NodeRunID], JoinState] = {}
        """Active reducers for join operations."""

        self._next: EndMarker[OutputT] | Sequence[GraphTask] | None = None
        """The next item to be processed."""

        self._next_task_id = 0
        self._next_node_run_id = 0
        initial_fork_stack: ForkStack = (ForkStackItem(StartNode.id, self._get_next_node_run_id(), 0),)
        self._first_task = GraphTask(
            node_id=StartNode.id, inputs=inputs, fork_stack=initial_fork_stack, task_id=self._get_next_task_id()
        )
        self._iterator_task_group = create_task_group()
        self._iterator_instance = _GraphIterator[StateT, DepsT, OutputT](
            self.graph,
            self.state,
            self.deps,
            self._iterator_task_group,
            self._get_next_node_run_id,
            self._get_next_task_id,
        )
        self._iterator = self._iterator_instance.iter_graph(self._first_task)

        self.__traceparent = traceparent
        self._async_exit_stack = AsyncExitStack()

    async def __aenter__(self):
        self._async_exit_stack.enter_context(_unwrap_exception_groups())
        await self._async_exit_stack.enter_async_context(self._iterator_task_group)
        await self._async_exit_stack.enter_async_context(self._iterator_context())
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    @asynccontextmanager
    async def _iterator_context(self):
        try:
            yield
        finally:
            self._iterator_instance.iter_stream_sender.close()
            self._iterator_instance.iter_stream_receiver.close()
            await self._iterator.aclose()

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        """Get the trace parent for instrumentation.

        Args:
            required: Whether to raise an error if no traceparent exists

        Returns:
            The traceparent string, or None if not required and not set

        Raises:
            GraphRuntimeError: If required is True and no traceparent exists
        """
        if self.__traceparent is None and required:  # pragma: no cover
            raise exceptions.GraphRuntimeError('No span was created for this graph run')
        return self.__traceparent

    def __aiter__(self) -> AsyncIterator[EndMarker[OutputT] | Sequence[GraphTask]]:
        """Return self as an async iterator.

        Returns:
            Self for async iteration
        """
        return self

    async def __anext__(self) -> EndMarker[OutputT] | Sequence[GraphTask]:
        """Get the next item in the async iteration.

        Returns:
            The next execution result from the graph
        """
        if self._next is None:
            self._next = await anext(self._iterator)
        else:
            self._next = await self._iterator.asend(self._next)
        return self._next

    async def next(
        self, value: EndMarker[OutputT] | Sequence[GraphTaskRequest] | None = None
    ) -> EndMarker[OutputT] | Sequence[GraphTask]:
        """Advance the graph execution by one step.

        This method allows for sending a value to the iterator, which is useful
        for resuming iteration or overriding intermediate results.

        Args:
            value: Optional value to send to the iterator

        Returns:
            The next execution result: either an EndMarker, or sequence of GraphTasks
        """
        if self._next is None:
            # Prevent `TypeError: can't send non-None value to a just-started async generator`
            # if `next` is called before the `first_node` has run.
            await anext(self)
        if value is not None:
            if isinstance(value, EndMarker):
                self._next = value
            else:
                self._next = [GraphTask.from_request(gtr, self._get_next_task_id) for gtr in value]
        return await anext(self)

    @property
    def next_task(self) -> EndMarker[OutputT] | Sequence[GraphTask]:
        """Get the next task(s) to be executed.

        Returns:
            The next execution item, or the initial task if none is set
        """
        return self._next or [self._first_task]

    @property
    def output(self) -> OutputT | None:
        """Get the final output if the graph has completed.

        Returns:
            The output value if execution is complete, None otherwise
        """
        if isinstance(self._next, EndMarker):
            return self._next.value
        return None

    def _get_next_task_id(self) -> TaskID:
        next_id = TaskID(f'task:{self._next_task_id}')
        self._next_task_id += 1
        return next_id

    def _get_next_node_run_id(self) -> NodeRunID:
        next_id = NodeRunID(f'task:{self._next_node_run_id}')
        self._next_node_run_id += 1
        return next_id

```

#### __init__

```python
__init__(
    graph: Graph[StateT, DepsT, InputT, OutputT],
    *,
    state: StateT,
    deps: DepsT,
    inputs: InputT,
    traceparent: str | None
)

```

Initialize a graph run.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `graph` | `Graph[StateT, DepsT, InputT, OutputT]` | The graph to execute | *required* | | `state` | `StateT` | The graph state instance | *required* | | `deps` | `DepsT` | The dependencies instance | *required* | | `inputs` | `InputT` | The input data for the graph | *required* | | `traceparent` | `str | None` | Optional trace parent for instrumentation | *required* |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
def __init__(
    self,
    graph: Graph[StateT, DepsT, InputT, OutputT],
    *,
    state: StateT,
    deps: DepsT,
    inputs: InputT,
    traceparent: str | None,
):
    """Initialize a graph run.

    Args:
        graph: The graph to execute
        state: The graph state instance
        deps: The dependencies instance
        inputs: The input data for the graph
        traceparent: Optional trace parent for instrumentation
    """
    self.graph = graph
    """The graph being executed."""

    self.state = state
    """The graph state instance."""

    self.deps = deps
    """The dependencies instance."""

    self.inputs = inputs
    """The initial input data."""

    self._active_reducers: dict[tuple[JoinID, NodeRunID], JoinState] = {}
    """Active reducers for join operations."""

    self._next: EndMarker[OutputT] | Sequence[GraphTask] | None = None
    """The next item to be processed."""

    self._next_task_id = 0
    self._next_node_run_id = 0
    initial_fork_stack: ForkStack = (ForkStackItem(StartNode.id, self._get_next_node_run_id(), 0),)
    self._first_task = GraphTask(
        node_id=StartNode.id, inputs=inputs, fork_stack=initial_fork_stack, task_id=self._get_next_task_id()
    )
    self._iterator_task_group = create_task_group()
    self._iterator_instance = _GraphIterator[StateT, DepsT, OutputT](
        self.graph,
        self.state,
        self.deps,
        self._iterator_task_group,
        self._get_next_node_run_id,
        self._get_next_task_id,
    )
    self._iterator = self._iterator_instance.iter_graph(self._first_task)

    self.__traceparent = traceparent
    self._async_exit_stack = AsyncExitStack()

```

#### graph

```python
graph = graph

```

The graph being executed.

#### state

```python
state = state

```

The graph state instance.

#### deps

```python
deps = deps

```

The dependencies instance.

#### inputs

```python
inputs = inputs

```

The initial input data.

#### __aiter__

```python
__aiter__() -> (
    AsyncIterator[EndMarker[OutputT] | Sequence[GraphTask]]
)

```

Return self as an async iterator.

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[EndMarker[OutputT] | Sequence[GraphTask]]` | Self for async iteration |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
def __aiter__(self) -> AsyncIterator[EndMarker[OutputT] | Sequence[GraphTask]]:
    """Return self as an async iterator.

    Returns:
        Self for async iteration
    """
    return self

```

#### __anext__

```python
__anext__() -> EndMarker[OutputT] | Sequence[GraphTask]

```

Get the next item in the async iteration.

Returns:

| Type | Description | | --- | --- | | `EndMarker[OutputT] | Sequence[GraphTask]` | The next execution result from the graph |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
async def __anext__(self) -> EndMarker[OutputT] | Sequence[GraphTask]:
    """Get the next item in the async iteration.

    Returns:
        The next execution result from the graph
    """
    if self._next is None:
        self._next = await anext(self._iterator)
    else:
        self._next = await self._iterator.asend(self._next)
    return self._next

```

#### next

```python
next(
    value: (
        EndMarker[OutputT]
        | Sequence[GraphTaskRequest]
        | None
    ) = None,
) -> EndMarker[OutputT] | Sequence[GraphTask]

```

Advance the graph execution by one step.

This method allows for sending a value to the iterator, which is useful for resuming iteration or overriding intermediate results.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `value` | `EndMarker[OutputT] | Sequence[GraphTaskRequest] | None` | Optional value to send to the iterator | `None` |

Returns:

| Type | Description | | --- | --- | | `EndMarker[OutputT] | Sequence[GraphTask]` | The next execution result: either an EndMarker, or sequence of GraphTasks |

Source code in `pydantic_graph/pydantic_graph/beta/graph.py`

```python
async def next(
    self, value: EndMarker[OutputT] | Sequence[GraphTaskRequest] | None = None
) -> EndMarker[OutputT] | Sequence[GraphTask]:
    """Advance the graph execution by one step.

    This method allows for sending a value to the iterator, which is useful
    for resuming iteration or overriding intermediate results.

    Args:
        value: Optional value to send to the iterator

    Returns:
        The next execution result: either an EndMarker, or sequence of GraphTasks
    """
    if self._next is None:
        # Prevent `TypeError: can't send non-None value to a just-started async generator`
        # if `next` is called before the `first_node` has run.
        await anext(self)
    if value is not None:
        if isinstance(value, EndMarker):
            self._next = value
        else:
            self._next = [GraphTask.from_request(gtr, self._get_next_task_id) for gtr in value]
    return await anext(self)

```

#### next_task

```python
next_task: EndMarker[OutputT] | Sequence[GraphTask]

```

Get the next task(s) to be executed.

Returns:

| Type | Description | | --- | --- | | `EndMarker[OutputT] | Sequence[GraphTask]` | The next execution item, or the initial task if none is set |

#### output

```python
output: OutputT | None

```

Get the final output if the graph has completed.

Returns:

| Type | Description | | --- | --- | | `OutputT | None` | The output value if execution is complete, None otherwise |

# `pydantic_graph.beta.graph_builder`

Graph builder for constructing executable graph definitions.

This module provides the GraphBuilder class and related utilities for constructing typed, executable graph definitions with steps, joins, decisions, and edge routing.

### GraphBuilder

Bases: `Generic[StateT, DepsT, GraphInputT, GraphOutputT]`

A builder for constructing executable graph definitions.

GraphBuilder provides a fluent interface for defining nodes, edges, and routing in a graph workflow. It supports typed state, dependencies, and input/output validation.

Type Parameters

StateT: The type of the graph state DepsT: The type of the dependencies GraphInputT: The type of the graph input data GraphOutputT: The type of the graph output data

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
@dataclass(init=False)
class GraphBuilder(Generic[StateT, DepsT, GraphInputT, GraphOutputT]):
    """A builder for constructing executable graph definitions.

    GraphBuilder provides a fluent interface for defining nodes, edges, and
    routing in a graph workflow. It supports typed state, dependencies, and
    input/output validation.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        GraphInputT: The type of the graph input data
        GraphOutputT: The type of the graph output data
    """

    name: str | None
    """Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method."""

    state_type: TypeOrTypeExpression[StateT]
    """The type of the graph state."""

    deps_type: TypeOrTypeExpression[DepsT]
    """The type of the dependencies."""

    input_type: TypeOrTypeExpression[GraphInputT]
    """The type of the graph input data."""

    output_type: TypeOrTypeExpression[GraphOutputT]
    """The type of the graph output data."""

    auto_instrument: bool
    """Whether to automatically create instrumentation spans."""

    _nodes: dict[NodeID, AnyNode]
    """Internal storage for nodes in the graph."""

    _edges_by_source: dict[NodeID, list[Path]]
    """Internal storage for edges by source node."""

    _decision_index: int
    """Counter for generating unique decision node IDs."""

    Source = TypeAliasType('Source', SourceNode[StateT, DepsT, OutputT], type_params=(OutputT,))
    Destination = TypeAliasType('Destination', DestinationNode[StateT, DepsT, InputT], type_params=(InputT,))

    def __init__(
        self,
        *,
        name: str | None = None,
        state_type: TypeOrTypeExpression[StateT] = NoneType,
        deps_type: TypeOrTypeExpression[DepsT] = NoneType,
        input_type: TypeOrTypeExpression[GraphInputT] = NoneType,
        output_type: TypeOrTypeExpression[GraphOutputT] = NoneType,
        auto_instrument: bool = True,
    ):
        """Initialize a graph builder.

        Args:
            name: Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method.
            state_type: The type of the graph state
            deps_type: The type of the dependencies
            input_type: The type of the graph input data
            output_type: The type of the graph output data
            auto_instrument: Whether to automatically create instrumentation spans
        """
        self.name = name

        self.state_type = state_type
        self.deps_type = deps_type
        self.input_type = input_type
        self.output_type = output_type

        self.auto_instrument = auto_instrument

        self._nodes = {}
        self._edges_by_source = defaultdict(list)
        self._decision_index = 1

        self._start_node = StartNode[GraphInputT]()
        self._end_node = EndNode[GraphOutputT]()

    # Node building
    @property
    def start_node(self) -> StartNode[GraphInputT]:
        """Get the start node for the graph.

        Returns:
            The start node that receives the initial graph input
        """
        return self._start_node

    @property
    def end_node(self) -> EndNode[GraphOutputT]:
        """Get the end node for the graph.

        Returns:
            The end node that produces the final graph output
        """
        return self._end_node

    @overload
    def step(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]: ...
    @overload
    def step(
        self,
        call: StepFunction[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, OutputT]: ...
    def step(
        self,
        call: StepFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, OutputT]
        | Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]
    ):
        """Create a step from a step function.

        This method can be used as a decorator or called directly to create
        a step node from an async function.

        Args:
            call: The step function to wrap
            node_id: Optional ID for the node
            label: Optional human-readable label

        Returns:
            Either a Step instance or a decorator function
        """
        if call is None:

            def decorator(
                func: StepFunction[StateT, DepsT, InputT, OutputT],
            ) -> Step[StateT, DepsT, InputT, OutputT]:
                return self.step(call=func, node_id=node_id, label=label)

            return decorator

        node_id = node_id or get_callable_name(call)

        step = Step[StateT, DepsT, InputT, OutputT](id=NodeID(node_id), call=call, label=label)

        return step

    @overload
    def stream(
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Callable[
        [StreamFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
    ]: ...
    @overload
    def stream(
        self,
        call: StreamFunction[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]: ...
    @overload
    def stream(
        self,
        call: StreamFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
        | Callable[
            [StreamFunction[StateT, DepsT, InputT, OutputT]],
            Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
        ]
    ): ...
    def stream(
        self,
        call: StreamFunction[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
        | Callable[
            [StreamFunction[StateT, DepsT, InputT, OutputT]],
            Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
        ]
    ):
        """Create a step from an async iterator (which functions like a "stream").

        This method can be used as a decorator or called directly to create
        a step node from an async function.

        Args:
            call: The step function to wrap
            node_id: Optional ID for the node
            label: Optional human-readable label

        Returns:
            Either a Step instance or a decorator function
        """
        if call is None:

            def decorator(
                func: StreamFunction[StateT, DepsT, InputT, OutputT],
            ) -> Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]:
                return self.stream(call=func, node_id=node_id, label=label)

            return decorator

        # We need to wrap the call so that we can call `await` even though the result is an async iterator
        async def wrapper(ctx: StepContext[StateT, DepsT, InputT]):
            return call(ctx)

        return self.step(call=wrapper, node_id=node_id, label=label)

    @overload
    def join(
        self,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        *,
        initial: OutputT,
        node_id: str | None = None,
        parent_fork_id: str | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ) -> Join[StateT, DepsT, InputT, OutputT]: ...
    @overload
    def join(
        self,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        *,
        initial_factory: Callable[[], OutputT],
        node_id: str | None = None,
        parent_fork_id: str | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ) -> Join[StateT, DepsT, InputT, OutputT]: ...

    def join(
        self,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        *,
        initial: OutputT | Unset = UNSET,
        initial_factory: Callable[[], OutputT] | Unset = UNSET,
        node_id: str | None = None,
        parent_fork_id: str | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ) -> Join[StateT, DepsT, InputT, OutputT]:
        if initial_factory is UNSET:
            initial_factory = lambda: initial  # pyright: ignore[reportAssignmentType]  # noqa E731

        return Join[StateT, DepsT, InputT, OutputT](
            id=JoinID(NodeID(node_id or generate_placeholder_node_id(get_callable_name(reducer)))),
            reducer=reducer,
            initial_factory=cast(Callable[[], OutputT], initial_factory),
            parent_fork_id=ForkID(parent_fork_id) if parent_fork_id is not None else None,
            preferred_parent_fork=preferred_parent_fork,
        )

    # Edge building
    def add(self, *edges: EdgePath[StateT, DepsT]) -> None:  # noqa C901
        """Add one or more edge paths to the graph.

        This method processes edge paths and automatically creates any necessary
        fork nodes for broadcasts and maps.

        Args:
            *edges: The edge paths to add to the graph
        """

        def _handle_path(p: Path):
            """Process a path and create necessary fork nodes.

            Args:
                p: The path to process
            """
            for item in p.items:
                if isinstance(item, BroadcastMarker):
                    new_node = Fork[Any, Any](id=item.fork_id, is_map=False, downstream_join_id=None)
                    self._insert_node(new_node)
                    for path in item.paths:
                        _handle_path(Path(items=[*path.items]))
                elif isinstance(item, MapMarker):
                    new_node = Fork[Any, Any](id=item.fork_id, is_map=True, downstream_join_id=item.downstream_join_id)
                    self._insert_node(new_node)
                elif isinstance(item, DestinationMarker):
                    pass

        def _handle_destination_node(d: AnyDestinationNode):
            if id(d) in destination_ids:
                return  # prevent infinite recursion if there is a cycle of decisions

            destination_ids.add(id(d))
            destinations.append(d)
            self._insert_node(d)
            if isinstance(d, Decision):
                for branch in d.branches:
                    _handle_path(branch.path)
                    for d2 in branch.destinations:
                        _handle_destination_node(d2)

        destination_ids = set[int]()
        destinations: list[AnyDestinationNode] = []
        for edge in edges:
            for source_node in edge.sources:
                self._insert_node(source_node)
                self._edges_by_source[source_node.id].append(edge.path)
            for destination_node in edge.destinations:
                _handle_destination_node(destination_node)
            _handle_path(edge.path)

        # Automatically create edges from step function return hints including `BaseNode`s
        for destination in destinations:
            if not isinstance(destination, Step) or isinstance(destination, NodeStep):
                continue
            parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
            type_hints = get_type_hints(destination.call, localns=parent_namespace, include_extras=True)
            try:
                return_hint = type_hints['return']
            except KeyError:
                pass
            else:
                edge = self._edge_from_return_hint(destination, return_hint)
                if edge is not None:
                    self.add(edge)

    def add_edge(self, source: Source[T], destination: Destination[T], *, label: str | None = None) -> None:
        """Add a simple edge between two nodes.

        Args:
            source: The source node
            destination: The destination node
            label: Optional label for the edge
        """
        builder = self.edge_from(source)
        if label is not None:
            builder = builder.label(label)
        self.add(builder.to(destination))

    def add_mapping_edge(
        self,
        source: Source[Iterable[T]],
        map_to: Destination[T],
        *,
        pre_map_label: str | None = None,
        post_map_label: str | None = None,
        fork_id: ForkID | None = None,
        downstream_join_id: JoinID | None = None,
    ) -> None:
        """Add an edge that maps iterable data across parallel paths.

        Args:
            source: The source node that produces iterable data
            map_to: The destination node that receives individual items
            pre_map_label: Optional label before the map operation
            post_map_label: Optional label after the map operation
            fork_id: Optional ID for the fork node produced for this map operation
            downstream_join_id: Optional ID of a join node that will always be downstream of this map.
                Specifying this ensures correct handling if you try to map an empty iterable.
        """
        builder = self.edge_from(source)
        if pre_map_label is not None:
            builder = builder.label(pre_map_label)
        builder = builder.map(fork_id=fork_id, downstream_join_id=downstream_join_id)
        if post_map_label is not None:
            builder = builder.label(post_map_label)
        self.add(builder.to(map_to))

    # TODO(DavidM): Support adding subgraphs; I think this behaves like a step with the same inputs/outputs but gets rendered as a subgraph in mermaid

    def edge_from(self, *sources: Source[SourceOutputT]) -> EdgePathBuilder[StateT, DepsT, SourceOutputT]:
        """Create an edge path builder starting from the given source nodes.

        Args:
            *sources: The source nodes to start the edge path from

        Returns:
            An EdgePathBuilder for constructing the complete edge path
        """
        return EdgePathBuilder[StateT, DepsT, SourceOutputT](
            sources=sources, path_builder=PathBuilder(working_items=[])
        )

    def decision(self, *, note: str | None = None, node_id: str | None = None) -> Decision[StateT, DepsT, Never]:
        """Create a new decision node.

        Args:
            note: Optional note to describe the decision logic
            node_id: Optional ID for the node produced for this decision logic

        Returns:
            A new Decision node with no branches
        """
        return Decision(id=NodeID(node_id or generate_placeholder_node_id('decision')), branches=[], note=note)

    def match(
        self,
        source: TypeOrTypeExpression[SourceT],
        *,
        matches: Callable[[Any], bool] | None = None,
    ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, SourceT, Never]:
        """Create a decision branch matcher.

        Args:
            source: The type or type expression to match against
            matches: Optional custom matching function

        Returns:
            A DecisionBranchBuilder for constructing the branch
        """
        # Note, the following node_id really is just a placeholder and shouldn't end up in the final graph
        # This is why we don't expose a way for end users to override the value used here.
        node_id = NodeID(generate_placeholder_node_id('match_decision'))
        decision = Decision[StateT, DepsT, Never](id=node_id, branches=[], note=None)
        new_path_builder = PathBuilder[StateT, DepsT, SourceT](working_items=[])
        return DecisionBranchBuilder(decision=decision, source=source, matches=matches, path_builder=new_path_builder)

    def match_node(
        self,
        source: type[SourceNodeT],
        *,
        matches: Callable[[Any], bool] | None = None,
    ) -> DecisionBranch[SourceNodeT]:
        """Create a decision branch for BaseNode subclasses.

        This is similar to match() but specifically designed for matching
        against BaseNode types from the v1 system.

        Args:
            source: The BaseNode subclass to match against
            matches: Optional custom matching function

        Returns:
            A DecisionBranch for the BaseNode type
        """
        node = NodeStep(source)
        path = Path(items=[DestinationMarker(node.id)])
        return DecisionBranch(source=source, matches=matches, path=path, destinations=[node])

    def node(
        self,
        node_type: type[BaseNode[StateT, DepsT, GraphOutputT]],
    ) -> EdgePath[StateT, DepsT]:
        """Create an edge path from a BaseNode class.

        This method integrates v1-style BaseNode classes into the v2 graph
        system by analyzing their type hints and creating appropriate edges.

        Args:
            node_type: The BaseNode subclass to integrate

        Returns:
            An EdgePath representing the node and its connections

        Raises:
            GraphSetupError: If the node type is missing required type hints
        """
        parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
        type_hints = get_type_hints(node_type.run, localns=parent_namespace, include_extras=True)
        try:
            return_hint = type_hints['return']
        except KeyError as e:  # pragma: no cover
            raise exceptions.GraphSetupError(
                f'Node {node_type} is missing a return type hint on its `run` method'
            ) from e

        node = NodeStep(node_type)

        edge = self._edge_from_return_hint(node, return_hint)
        if not edge:  # pragma: no cover
            raise exceptions.GraphSetupError(f'Node {node_type} is missing a return type hint on its `run` method')

        return edge

    # Helpers
    def _insert_node(self, node: AnyNode) -> None:
        """Insert a node into the graph, checking for ID conflicts.

        Args:
            node: The node to insert

        Raises:
            ValueError: If a different node with the same ID already exists
        """
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
        elif isinstance(existing, NodeStep) and isinstance(node, NodeStep) and existing.node_type is node.node_type:
            pass
        elif existing is not node:
            raise GraphBuildingError(
                f'All nodes must have unique node IDs. {node.id!r} was the ID for {existing} and {node}'
            )

    def _edge_from_return_hint(
        self, node: SourceNode[StateT, DepsT, Any], return_hint: TypeOrTypeExpression[Any]
    ) -> EdgePath[StateT, DepsT] | None:
        """Create edges from a return type hint.

        This method analyzes return type hints from step functions or node methods
        to automatically create appropriate edges in the graph.

        Args:
            node: The source node
            return_hint: The return type hint to analyze

        Returns:
            An EdgePath if edges can be inferred, None otherwise

        Raises:
            GraphSetupError: If the return type hint is invalid or incomplete
        """
        destinations: list[AnyDestinationNode] = []
        union_args = _utils.get_union_args(return_hint)
        for return_type in union_args:
            return_type, annotations = _utils.unpack_annotated(return_type)
            return_type_origin = get_origin(return_type) or return_type
            if return_type_origin is End:
                destinations.append(self.end_node)
            elif return_type_origin is BaseNode:
                raise exceptions.GraphSetupError(  # pragma: no cover
                    f'Node {node} return type hint includes a plain `BaseNode`. '
                    'Edge inference requires each possible returned `BaseNode` subclass to be listed explicitly.'
                )
            elif return_type_origin is StepNode:
                step = cast(
                    Step[StateT, DepsT, Any, Any] | None,
                    next((a for a in annotations if isinstance(a, Step)), None),  # pyright: ignore[reportUnknownArgumentType]
                )
                if step is None:
                    raise exceptions.GraphSetupError(  # pragma: no cover
                        f'Node {node} return type hint includes a `StepNode` without a `Step` annotation. '
                        'When returning `my_step.as_node()`, use `Annotated[StepNode[StateT, DepsT], my_step]` as the return type hint.'
                    )
                destinations.append(step)
            elif return_type_origin is JoinNode:
                join = cast(
                    Join[StateT, DepsT, Any, Any] | None,
                    next((a for a in annotations if isinstance(a, Join)), None),  # pyright: ignore[reportUnknownArgumentType]
                )
                if join is None:
                    raise exceptions.GraphSetupError(  # pragma: no cover
                        f'Node {node} return type hint includes a `JoinNode` without a `Join` annotation. '
                        'When returning `my_join.as_node()`, use `Annotated[JoinNode[StateT, DepsT], my_join]` as the return type hint.'
                    )
                destinations.append(join)
            elif inspect.isclass(return_type_origin) and issubclass(return_type_origin, BaseNode):
                destinations.append(NodeStep(return_type))

        if len(destinations) < len(union_args):
            # Only build edges if all the return types are nodes
            return None

        edge = self.edge_from(node)
        if len(destinations) == 1:
            return edge.to(destinations[0])
        else:
            decision = self.decision()
            for destination in destinations:
                # We don't actually use this decision mechanism, but we need to build the edges for parent-fork finding
                decision = decision.branch(self.match(NoneType).to(destination))
            return edge.to(decision)

    # Graph building
    def build(self, validate_graph_structure: bool = True) -> Graph[StateT, DepsT, GraphInputT, GraphOutputT]:
        """Build the final executable graph from the accumulated nodes and edges.

        This method performs validation, normalization, and analysis of the graph
        structure to create a complete, executable graph instance.

        Args:
            validate_graph_structure: whether to perform validation of the graph structure
                See the docstring of _validate_graph_structure below for more details.

        Returns:
            A complete Graph instance ready for execution

        Raises:
            ValueError: If the graph structure is invalid (e.g., join without parent fork)
        """
        nodes = self._nodes
        edges_by_source = self._edges_by_source

        nodes, edges_by_source = _replace_placeholder_node_ids(nodes, edges_by_source)
        nodes, edges_by_source = _flatten_paths(nodes, edges_by_source)
        nodes, edges_by_source = _normalize_forks(nodes, edges_by_source)
        if validate_graph_structure:
            _validate_graph_structure(nodes, edges_by_source)
        parent_forks = _collect_dominating_forks(nodes, edges_by_source)
        intermediate_join_nodes = _compute_intermediate_join_nodes(nodes, parent_forks)

        return Graph[StateT, DepsT, GraphInputT, GraphOutputT](
            name=self.name,
            state_type=unpack_type_expression(self.state_type),
            deps_type=unpack_type_expression(self.deps_type),
            input_type=unpack_type_expression(self.input_type),
            output_type=unpack_type_expression(self.output_type),
            nodes=nodes,
            edges_by_source=edges_by_source,
            parent_forks=parent_forks,
            intermediate_join_nodes=intermediate_join_nodes,
            auto_instrument=self.auto_instrument,
        )

```

#### __init__

```python
__init__(
    *,
    name: str | None = None,
    state_type: TypeOrTypeExpression[StateT] = NoneType,
    deps_type: TypeOrTypeExpression[DepsT] = NoneType,
    input_type: TypeOrTypeExpression[
        GraphInputT
    ] = NoneType,
    output_type: TypeOrTypeExpression[
        GraphOutputT
    ] = NoneType,
    auto_instrument: bool = True
)

```

Initialize a graph builder.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method. | `None` | | `state_type` | `TypeOrTypeExpression[StateT]` | The type of the graph state | `NoneType` | | `deps_type` | `TypeOrTypeExpression[DepsT]` | The type of the dependencies | `NoneType` | | `input_type` | `TypeOrTypeExpression[GraphInputT]` | The type of the graph input data | `NoneType` | | `output_type` | `TypeOrTypeExpression[GraphOutputT]` | The type of the graph output data | `NoneType` | | `auto_instrument` | `bool` | Whether to automatically create instrumentation spans | `True` |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def __init__(
    self,
    *,
    name: str | None = None,
    state_type: TypeOrTypeExpression[StateT] = NoneType,
    deps_type: TypeOrTypeExpression[DepsT] = NoneType,
    input_type: TypeOrTypeExpression[GraphInputT] = NoneType,
    output_type: TypeOrTypeExpression[GraphOutputT] = NoneType,
    auto_instrument: bool = True,
):
    """Initialize a graph builder.

    Args:
        name: Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method.
        state_type: The type of the graph state
        deps_type: The type of the dependencies
        input_type: The type of the graph input data
        output_type: The type of the graph output data
        auto_instrument: Whether to automatically create instrumentation spans
    """
    self.name = name

    self.state_type = state_type
    self.deps_type = deps_type
    self.input_type = input_type
    self.output_type = output_type

    self.auto_instrument = auto_instrument

    self._nodes = {}
    self._edges_by_source = defaultdict(list)
    self._decision_index = 1

    self._start_node = StartNode[GraphInputT]()
    self._end_node = EndNode[GraphOutputT]()

```

#### name

```python
name: str | None = name

```

Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method.

#### state_type

```python
state_type: TypeOrTypeExpression[StateT] = state_type

```

The type of the graph state.

#### deps_type

```python
deps_type: TypeOrTypeExpression[DepsT] = deps_type

```

The type of the dependencies.

#### input_type

```python
input_type: TypeOrTypeExpression[GraphInputT] = input_type

```

The type of the graph input data.

#### output_type

```python
output_type: TypeOrTypeExpression[GraphOutputT] = (
    output_type
)

```

The type of the graph output data.

#### auto_instrument

```python
auto_instrument: bool = auto_instrument

```

Whether to automatically create instrumentation spans.

#### start_node

```python
start_node: StartNode[GraphInputT]

```

Get the start node for the graph.

Returns:

| Type | Description | | --- | --- | | `StartNode[GraphInputT]` | The start node that receives the initial graph input |

#### end_node

```python
end_node: EndNode[GraphOutputT]

```

Get the end node for the graph.

Returns:

| Type | Description | | --- | --- | | `EndNode[GraphOutputT]` | The end node that produces the final graph output |

#### step

```python
step(
    *, node_id: str | None = None, label: str | None = None
) -> Callable[
    [StepFunction[StateT, DepsT, InputT, OutputT]],
    Step[StateT, DepsT, InputT, OutputT],
]

```

```python
step(
    call: StepFunction[StateT, DepsT, InputT, OutputT],
    *,
    node_id: str | None = None,
    label: str | None = None
) -> Step[StateT, DepsT, InputT, OutputT]

```

```python
step(
    call: (
        StepFunction[StateT, DepsT, InputT, OutputT] | None
    ) = None,
    *,
    node_id: str | None = None,
    label: str | None = None
) -> (
    Step[StateT, DepsT, InputT, OutputT]
    | Callable[
        [StepFunction[StateT, DepsT, InputT, OutputT]],
        Step[StateT, DepsT, InputT, OutputT],
    ]
)

```

Create a step from a step function.

This method can be used as a decorator or called directly to create a step node from an async function.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `call` | `StepFunction[StateT, DepsT, InputT, OutputT] | None` | The step function to wrap | `None` | | `node_id` | `str | None` | Optional ID for the node | `None` | | `label` | `str | None` | Optional human-readable label | `None` |

Returns:

| Type | Description | | --- | --- | | `Step[StateT, DepsT, InputT, OutputT] | Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]` | Either a Step instance or a decorator function |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def step(
    self,
    call: StepFunction[StateT, DepsT, InputT, OutputT] | None = None,
    *,
    node_id: str | None = None,
    label: str | None = None,
) -> (
    Step[StateT, DepsT, InputT, OutputT]
    | Callable[[StepFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]
):
    """Create a step from a step function.

    This method can be used as a decorator or called directly to create
    a step node from an async function.

    Args:
        call: The step function to wrap
        node_id: Optional ID for the node
        label: Optional human-readable label

    Returns:
        Either a Step instance or a decorator function
    """
    if call is None:

        def decorator(
            func: StepFunction[StateT, DepsT, InputT, OutputT],
        ) -> Step[StateT, DepsT, InputT, OutputT]:
            return self.step(call=func, node_id=node_id, label=label)

        return decorator

    node_id = node_id or get_callable_name(call)

    step = Step[StateT, DepsT, InputT, OutputT](id=NodeID(node_id), call=call, label=label)

    return step

```

#### stream

```python
stream(
    *, node_id: str | None = None, label: str | None = None
) -> Callable[
    [StreamFunction[StateT, DepsT, InputT, OutputT]],
    Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
]

```

```python
stream(
    call: StreamFunction[StateT, DepsT, InputT, OutputT],
    *,
    node_id: str | None = None,
    label: str | None = None
) -> Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]

```

```python
stream(
    call: (
        StreamFunction[StateT, DepsT, InputT, OutputT]
        | None
    ) = None,
    *,
    node_id: str | None = None,
    label: str | None = None
) -> (
    Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
    | Callable[
        [StreamFunction[StateT, DepsT, InputT, OutputT]],
        Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
    ]
)

```

```python
stream(
    call: (
        StreamFunction[StateT, DepsT, InputT, OutputT]
        | None
    ) = None,
    *,
    node_id: str | None = None,
    label: str | None = None
) -> (
    Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
    | Callable[
        [StreamFunction[StateT, DepsT, InputT, OutputT]],
        Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
    ]
)

```

Create a step from an async iterator (which functions like a "stream").

This method can be used as a decorator or called directly to create a step node from an async function.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `call` | `StreamFunction[StateT, DepsT, InputT, OutputT] | None` | The step function to wrap | `None` | | `node_id` | `str | None` | Optional ID for the node | `None` | | `label` | `str | None` | Optional human-readable label | `None` |

Returns:

| Type | Description | | --- | --- | | `Step[StateT, DepsT, InputT, AsyncIterable[OutputT]] | Callable[[StreamFunction[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]]` | Either a Step instance or a decorator function |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def stream(
    self,
    call: StreamFunction[StateT, DepsT, InputT, OutputT] | None = None,
    *,
    node_id: str | None = None,
    label: str | None = None,
) -> (
    Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]
    | Callable[
        [StreamFunction[StateT, DepsT, InputT, OutputT]],
        Step[StateT, DepsT, InputT, AsyncIterable[OutputT]],
    ]
):
    """Create a step from an async iterator (which functions like a "stream").

    This method can be used as a decorator or called directly to create
    a step node from an async function.

    Args:
        call: The step function to wrap
        node_id: Optional ID for the node
        label: Optional human-readable label

    Returns:
        Either a Step instance or a decorator function
    """
    if call is None:

        def decorator(
            func: StreamFunction[StateT, DepsT, InputT, OutputT],
        ) -> Step[StateT, DepsT, InputT, AsyncIterable[OutputT]]:
            return self.stream(call=func, node_id=node_id, label=label)

        return decorator

    # We need to wrap the call so that we can call `await` even though the result is an async iterator
    async def wrapper(ctx: StepContext[StateT, DepsT, InputT]):
        return call(ctx)

    return self.step(call=wrapper, node_id=node_id, label=label)

```

#### add

```python
add(*edges: EdgePath[StateT, DepsT]) -> None

```

Add one or more edge paths to the graph.

This method processes edge paths and automatically creates any necessary fork nodes for broadcasts and maps.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `*edges` | `EdgePath[StateT, DepsT]` | The edge paths to add to the graph | `()` |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def add(self, *edges: EdgePath[StateT, DepsT]) -> None:  # noqa C901
    """Add one or more edge paths to the graph.

    This method processes edge paths and automatically creates any necessary
    fork nodes for broadcasts and maps.

    Args:
        *edges: The edge paths to add to the graph
    """

    def _handle_path(p: Path):
        """Process a path and create necessary fork nodes.

        Args:
            p: The path to process
        """
        for item in p.items:
            if isinstance(item, BroadcastMarker):
                new_node = Fork[Any, Any](id=item.fork_id, is_map=False, downstream_join_id=None)
                self._insert_node(new_node)
                for path in item.paths:
                    _handle_path(Path(items=[*path.items]))
            elif isinstance(item, MapMarker):
                new_node = Fork[Any, Any](id=item.fork_id, is_map=True, downstream_join_id=item.downstream_join_id)
                self._insert_node(new_node)
            elif isinstance(item, DestinationMarker):
                pass

    def _handle_destination_node(d: AnyDestinationNode):
        if id(d) in destination_ids:
            return  # prevent infinite recursion if there is a cycle of decisions

        destination_ids.add(id(d))
        destinations.append(d)
        self._insert_node(d)
        if isinstance(d, Decision):
            for branch in d.branches:
                _handle_path(branch.path)
                for d2 in branch.destinations:
                    _handle_destination_node(d2)

    destination_ids = set[int]()
    destinations: list[AnyDestinationNode] = []
    for edge in edges:
        for source_node in edge.sources:
            self._insert_node(source_node)
            self._edges_by_source[source_node.id].append(edge.path)
        for destination_node in edge.destinations:
            _handle_destination_node(destination_node)
        _handle_path(edge.path)

    # Automatically create edges from step function return hints including `BaseNode`s
    for destination in destinations:
        if not isinstance(destination, Step) or isinstance(destination, NodeStep):
            continue
        parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
        type_hints = get_type_hints(destination.call, localns=parent_namespace, include_extras=True)
        try:
            return_hint = type_hints['return']
        except KeyError:
            pass
        else:
            edge = self._edge_from_return_hint(destination, return_hint)
            if edge is not None:
                self.add(edge)

```

#### add_edge

```python
add_edge(
    source: Source[T],
    destination: Destination[T],
    *,
    label: str | None = None
) -> None

```

Add a simple edge between two nodes.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `source` | `Source[T]` | The source node | *required* | | `destination` | `Destination[T]` | The destination node | *required* | | `label` | `str | None` | Optional label for the edge | `None` |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def add_edge(self, source: Source[T], destination: Destination[T], *, label: str | None = None) -> None:
    """Add a simple edge between two nodes.

    Args:
        source: The source node
        destination: The destination node
        label: Optional label for the edge
    """
    builder = self.edge_from(source)
    if label is not None:
        builder = builder.label(label)
    self.add(builder.to(destination))

```

#### add_mapping_edge

```python
add_mapping_edge(
    source: Source[Iterable[T]],
    map_to: Destination[T],
    *,
    pre_map_label: str | None = None,
    post_map_label: str | None = None,
    fork_id: ForkID | None = None,
    downstream_join_id: JoinID | None = None
) -> None

```

Add an edge that maps iterable data across parallel paths.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `source` | `Source[Iterable[T]]` | The source node that produces iterable data | *required* | | `map_to` | `Destination[T]` | The destination node that receives individual items | *required* | | `pre_map_label` | `str | None` | Optional label before the map operation | `None` | | `post_map_label` | `str | None` | Optional label after the map operation | `None` | | `fork_id` | `ForkID | None` | Optional ID for the fork node produced for this map operation | `None` | | `downstream_join_id` | `JoinID | None` | Optional ID of a join node that will always be downstream of this map. Specifying this ensures correct handling if you try to map an empty iterable. | `None` |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def add_mapping_edge(
    self,
    source: Source[Iterable[T]],
    map_to: Destination[T],
    *,
    pre_map_label: str | None = None,
    post_map_label: str | None = None,
    fork_id: ForkID | None = None,
    downstream_join_id: JoinID | None = None,
) -> None:
    """Add an edge that maps iterable data across parallel paths.

    Args:
        source: The source node that produces iterable data
        map_to: The destination node that receives individual items
        pre_map_label: Optional label before the map operation
        post_map_label: Optional label after the map operation
        fork_id: Optional ID for the fork node produced for this map operation
        downstream_join_id: Optional ID of a join node that will always be downstream of this map.
            Specifying this ensures correct handling if you try to map an empty iterable.
    """
    builder = self.edge_from(source)
    if pre_map_label is not None:
        builder = builder.label(pre_map_label)
    builder = builder.map(fork_id=fork_id, downstream_join_id=downstream_join_id)
    if post_map_label is not None:
        builder = builder.label(post_map_label)
    self.add(builder.to(map_to))

```

#### edge_from

```python
edge_from(
    *sources: Source[SourceOutputT],
) -> EdgePathBuilder[StateT, DepsT, SourceOutputT]

```

Create an edge path builder starting from the given source nodes.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `*sources` | `Source[SourceOutputT]` | The source nodes to start the edge path from | `()` |

Returns:

| Type | Description | | --- | --- | | `EdgePathBuilder[StateT, DepsT, SourceOutputT]` | An EdgePathBuilder for constructing the complete edge path |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def edge_from(self, *sources: Source[SourceOutputT]) -> EdgePathBuilder[StateT, DepsT, SourceOutputT]:
    """Create an edge path builder starting from the given source nodes.

    Args:
        *sources: The source nodes to start the edge path from

    Returns:
        An EdgePathBuilder for constructing the complete edge path
    """
    return EdgePathBuilder[StateT, DepsT, SourceOutputT](
        sources=sources, path_builder=PathBuilder(working_items=[])
    )

```

#### decision

```python
decision(
    *, note: str | None = None, node_id: str | None = None
) -> Decision[StateT, DepsT, Never]

```

Create a new decision node.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `note` | `str | None` | Optional note to describe the decision logic | `None` | | `node_id` | `str | None` | Optional ID for the node produced for this decision logic | `None` |

Returns:

| Type | Description | | --- | --- | | `Decision[StateT, DepsT, Never]` | A new Decision node with no branches |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def decision(self, *, note: str | None = None, node_id: str | None = None) -> Decision[StateT, DepsT, Never]:
    """Create a new decision node.

    Args:
        note: Optional note to describe the decision logic
        node_id: Optional ID for the node produced for this decision logic

    Returns:
        A new Decision node with no branches
    """
    return Decision(id=NodeID(node_id or generate_placeholder_node_id('decision')), branches=[], note=note)

```

#### match

```python
match(
    source: TypeOrTypeExpression[SourceT],
    *,
    matches: Callable[[Any], bool] | None = None
) -> DecisionBranchBuilder[
    StateT, DepsT, SourceT, SourceT, Never
]

```

Create a decision branch matcher.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `source` | `TypeOrTypeExpression[SourceT]` | The type or type expression to match against | *required* | | `matches` | `Callable[[Any], bool] | None` | Optional custom matching function | `None` |

Returns:

| Type | Description | | --- | --- | | `DecisionBranchBuilder[StateT, DepsT, SourceT, SourceT, Never]` | A DecisionBranchBuilder for constructing the branch |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def match(
    self,
    source: TypeOrTypeExpression[SourceT],
    *,
    matches: Callable[[Any], bool] | None = None,
) -> DecisionBranchBuilder[StateT, DepsT, SourceT, SourceT, Never]:
    """Create a decision branch matcher.

    Args:
        source: The type or type expression to match against
        matches: Optional custom matching function

    Returns:
        A DecisionBranchBuilder for constructing the branch
    """
    # Note, the following node_id really is just a placeholder and shouldn't end up in the final graph
    # This is why we don't expose a way for end users to override the value used here.
    node_id = NodeID(generate_placeholder_node_id('match_decision'))
    decision = Decision[StateT, DepsT, Never](id=node_id, branches=[], note=None)
    new_path_builder = PathBuilder[StateT, DepsT, SourceT](working_items=[])
    return DecisionBranchBuilder(decision=decision, source=source, matches=matches, path_builder=new_path_builder)

```

#### match_node

```python
match_node(
    source: type[SourceNodeT],
    *,
    matches: Callable[[Any], bool] | None = None
) -> DecisionBranch[SourceNodeT]

```

Create a decision branch for BaseNode subclasses.

This is similar to match() but specifically designed for matching against BaseNode types from the v1 system.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `source` | `type[SourceNodeT]` | The BaseNode subclass to match against | *required* | | `matches` | `Callable[[Any], bool] | None` | Optional custom matching function | `None` |

Returns:

| Type | Description | | --- | --- | | `DecisionBranch[SourceNodeT]` | A DecisionBranch for the BaseNode type |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def match_node(
    self,
    source: type[SourceNodeT],
    *,
    matches: Callable[[Any], bool] | None = None,
) -> DecisionBranch[SourceNodeT]:
    """Create a decision branch for BaseNode subclasses.

    This is similar to match() but specifically designed for matching
    against BaseNode types from the v1 system.

    Args:
        source: The BaseNode subclass to match against
        matches: Optional custom matching function

    Returns:
        A DecisionBranch for the BaseNode type
    """
    node = NodeStep(source)
    path = Path(items=[DestinationMarker(node.id)])
    return DecisionBranch(source=source, matches=matches, path=path, destinations=[node])

```

#### node

```python
node(
    node_type: type[BaseNode[StateT, DepsT, GraphOutputT]],
) -> EdgePath[StateT, DepsT]

```

Create an edge path from a BaseNode class.

This method integrates v1-style BaseNode classes into the v2 graph system by analyzing their type hints and creating appropriate edges.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `node_type` | `type[BaseNode[StateT, DepsT, GraphOutputT]]` | The BaseNode subclass to integrate | *required* |

Returns:

| Type | Description | | --- | --- | | `EdgePath[StateT, DepsT]` | An EdgePath representing the node and its connections |

Raises:

| Type | Description | | --- | --- | | `GraphSetupError` | If the node type is missing required type hints |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def node(
    self,
    node_type: type[BaseNode[StateT, DepsT, GraphOutputT]],
) -> EdgePath[StateT, DepsT]:
    """Create an edge path from a BaseNode class.

    This method integrates v1-style BaseNode classes into the v2 graph
    system by analyzing their type hints and creating appropriate edges.

    Args:
        node_type: The BaseNode subclass to integrate

    Returns:
        An EdgePath representing the node and its connections

    Raises:
        GraphSetupError: If the node type is missing required type hints
    """
    parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
    type_hints = get_type_hints(node_type.run, localns=parent_namespace, include_extras=True)
    try:
        return_hint = type_hints['return']
    except KeyError as e:  # pragma: no cover
        raise exceptions.GraphSetupError(
            f'Node {node_type} is missing a return type hint on its `run` method'
        ) from e

    node = NodeStep(node_type)

    edge = self._edge_from_return_hint(node, return_hint)
    if not edge:  # pragma: no cover
        raise exceptions.GraphSetupError(f'Node {node_type} is missing a return type hint on its `run` method')

    return edge

```

#### build

```python
build(
    validate_graph_structure: bool = True,
) -> Graph[StateT, DepsT, GraphInputT, GraphOutputT]

```

Build the final executable graph from the accumulated nodes and edges.

This method performs validation, normalization, and analysis of the graph structure to create a complete, executable graph instance.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `validate_graph_structure` | `bool` | whether to perform validation of the graph structure See the docstring of \_validate_graph_structure below for more details. | `True` |

Returns:

| Type | Description | | --- | --- | | `Graph[StateT, DepsT, GraphInputT, GraphOutputT]` | A complete Graph instance ready for execution |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If the graph structure is invalid (e.g., join without parent fork) |

Source code in `pydantic_graph/pydantic_graph/beta/graph_builder.py`

```python
def build(self, validate_graph_structure: bool = True) -> Graph[StateT, DepsT, GraphInputT, GraphOutputT]:
    """Build the final executable graph from the accumulated nodes and edges.

    This method performs validation, normalization, and analysis of the graph
    structure to create a complete, executable graph instance.

    Args:
        validate_graph_structure: whether to perform validation of the graph structure
            See the docstring of _validate_graph_structure below for more details.

    Returns:
        A complete Graph instance ready for execution

    Raises:
        ValueError: If the graph structure is invalid (e.g., join without parent fork)
    """
    nodes = self._nodes
    edges_by_source = self._edges_by_source

    nodes, edges_by_source = _replace_placeholder_node_ids(nodes, edges_by_source)
    nodes, edges_by_source = _flatten_paths(nodes, edges_by_source)
    nodes, edges_by_source = _normalize_forks(nodes, edges_by_source)
    if validate_graph_structure:
        _validate_graph_structure(nodes, edges_by_source)
    parent_forks = _collect_dominating_forks(nodes, edges_by_source)
    intermediate_join_nodes = _compute_intermediate_join_nodes(nodes, parent_forks)

    return Graph[StateT, DepsT, GraphInputT, GraphOutputT](
        name=self.name,
        state_type=unpack_type_expression(self.state_type),
        deps_type=unpack_type_expression(self.deps_type),
        input_type=unpack_type_expression(self.input_type),
        output_type=unpack_type_expression(self.output_type),
        nodes=nodes,
        edges_by_source=edges_by_source,
        parent_forks=parent_forks,
        intermediate_join_nodes=intermediate_join_nodes,
        auto_instrument=self.auto_instrument,
    )

```

# `pydantic_graph.beta.join`

Join operations and reducers for graph execution.

This module provides the core components for joining parallel execution paths in a graph, including various reducer types that aggregate data from multiple sources into a single output.

### JoinState

The state of a join during graph execution associated to a particular fork run.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
@dataclass
class JoinState:
    """The state of a join during graph execution associated to a particular fork run."""

    current: Any
    downstream_fork_stack: ForkStack
    cancelled_sibling_tasks: bool = False

```

### ReducerContext

Bases: `Generic[StateT, DepsT]`

Context information passed to reducer functions during graph execution.

The reducer context provides access to the current graph state and dependencies.

Type Parameters

StateT: The type of the graph state DepsT: The type of the dependencies

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
@dataclass(init=False)
class ReducerContext(Generic[StateT, DepsT]):
    """Context information passed to reducer functions during graph execution.

    The reducer context provides access to the current graph state and dependencies.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
    """

    _state: StateT
    """The current graph state."""
    _deps: DepsT
    """The dependencies of the current graph run."""
    _join_state: JoinState
    """The JoinState for this reducer context."""

    def __init__(self, *, state: StateT, deps: DepsT, join_state: JoinState):
        self._state = state
        self._deps = deps
        self._join_state = join_state

    @property
    def state(self) -> StateT:
        """The state of the graph run."""
        return self._state

    @property
    def deps(self) -> DepsT:
        """The deps for the graph run."""
        return self._deps

    def cancel_sibling_tasks(self):
        """Cancel all sibling tasks created from the same fork.

        You can call this if you want your join to have early-stopping behavior.
        """
        self._join_state.cancelled_sibling_tasks = True

```

#### state

```python
state: StateT

```

The state of the graph run.

#### deps

```python
deps: DepsT

```

The deps for the graph run.

#### cancel_sibling_tasks

```python
cancel_sibling_tasks()

```

Cancel all sibling tasks created from the same fork.

You can call this if you want your join to have early-stopping behavior.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
def cancel_sibling_tasks(self):
    """Cancel all sibling tasks created from the same fork.

    You can call this if you want your join to have early-stopping behavior.
    """
    self._join_state.cancelled_sibling_tasks = True

```

### ReducerFunction

```python
ReducerFunction = TypeAliasType(
    "ReducerFunction",
    ContextReducerFunction[StateT, DepsT, InputT, OutputT]
    | PlainReducerFunction[InputT, OutputT],
    type_params=(StateT, DepsT, InputT, OutputT),
)

```

A function used for reducing inputs to a join node.

### reduce_null

```python
reduce_null(current: None, inputs: Any) -> None

```

A reducer that discards all input data and returns None.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
def reduce_null(current: None, inputs: Any) -> None:
    """A reducer that discards all input data and returns None."""
    return None

```

### reduce_list_append

```python
reduce_list_append(
    current: list[T], inputs: T
) -> list[T]

```

A reducer that appends to a list.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
def reduce_list_append(current: list[T], inputs: T) -> list[T]:
    """A reducer that appends to a list."""
    current.append(inputs)
    return current

```

### reduce_list_extend

```python
reduce_list_extend(
    current: list[T], inputs: Iterable[T]
) -> list[T]

```

A reducer that extends a list.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
def reduce_list_extend(current: list[T], inputs: Iterable[T]) -> list[T]:
    """A reducer that extends a list."""
    current.extend(inputs)
    return current

```

### reduce_dict_update

```python
reduce_dict_update(
    current: dict[K, V], inputs: Mapping[K, V]
) -> dict[K, V]

```

A reducer that updates a dict.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
def reduce_dict_update(current: dict[K, V], inputs: Mapping[K, V]) -> dict[K, V]:
    """A reducer that updates a dict."""
    current.update(inputs)
    return current

```

### SupportsSum

Bases: `Protocol`

A protocol for a type that supports adding to itself.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
class SupportsSum(Protocol):
    """A protocol for a type that supports adding to itself."""

    @abstractmethod
    def __add__(self, other: Self, /) -> Self:
        pass

```

### reduce_sum

```python
reduce_sum(current: NumericT, inputs: NumericT) -> NumericT

```

A reducer that sums numbers.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
def reduce_sum(current: NumericT, inputs: NumericT) -> NumericT:
    """A reducer that sums numbers."""
    return current + inputs

```

### ReduceFirstValue

Bases: `Generic[T]`

A reducer that returns the first value it encounters, and cancels all other tasks.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
@dataclass
class ReduceFirstValue(Generic[T]):
    """A reducer that returns the first value it encounters, and cancels all other tasks."""

    def __call__(self, ctx: ReducerContext[object, object], current: T, inputs: T) -> T:
        """The reducer function."""
        ctx.cancel_sibling_tasks()
        return inputs

```

#### __call__

```python
__call__(
    ctx: ReducerContext[object, object],
    current: T,
    inputs: T,
) -> T

```

The reducer function.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
def __call__(self, ctx: ReducerContext[object, object], current: T, inputs: T) -> T:
    """The reducer function."""
    ctx.cancel_sibling_tasks()
    return inputs

```

### Join

Bases: `Generic[StateT, DepsT, InputT, OutputT]`

A join operation that synchronizes and aggregates parallel execution paths.

A join defines how to combine outputs from multiple parallel execution paths using a ReducerFunction. It specifies which fork it joins (if any) and manages the initialization of reducers.

Type Parameters

StateT: The type of the graph state DepsT: The type of the dependencies InputT: The type of input data to join OutputT: The type of the final joined output

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
@dataclass(init=False)
class Join(Generic[StateT, DepsT, InputT, OutputT]):
    """A join operation that synchronizes and aggregates parallel execution paths.

    A join defines how to combine outputs from multiple parallel execution paths
    using a [`ReducerFunction`][pydantic_graph.beta.join.ReducerFunction]. It specifies which fork
    it joins (if any) and manages the initialization of reducers.

    Type Parameters:
        StateT: The type of the graph state
        DepsT: The type of the dependencies
        InputT: The type of input data to join
        OutputT: The type of the final joined output
    """

    id: JoinID
    _reducer: ReducerFunction[StateT, DepsT, InputT, OutputT]
    _initial_factory: Callable[[], OutputT]
    parent_fork_id: ForkID | None
    preferred_parent_fork: Literal['closest', 'farthest']

    def __init__(
        self,
        *,
        id: JoinID,
        reducer: ReducerFunction[StateT, DepsT, InputT, OutputT],
        initial_factory: Callable[[], OutputT],
        parent_fork_id: ForkID | None = None,
        preferred_parent_fork: Literal['farthest', 'closest'] = 'farthest',
    ):
        self.id = id
        self._reducer = reducer
        self._initial_factory = initial_factory
        self.parent_fork_id = parent_fork_id
        self.preferred_parent_fork = preferred_parent_fork

    @property
    def reducer(self):
        return self._reducer

    @property
    def initial_factory(self):
        return self._initial_factory

    def reduce(self, ctx: ReducerContext[StateT, DepsT], current: OutputT, inputs: InputT) -> OutputT:
        n_parameters = len(inspect.signature(self.reducer).parameters)
        if n_parameters == 2:
            return cast(PlainReducerFunction[InputT, OutputT], self.reducer)(current, inputs)
        else:
            return cast(ContextReducerFunction[StateT, DepsT, InputT, OutputT], self.reducer)(ctx, current, inputs)

    @overload
    def as_node(self, inputs: None = None) -> JoinNode[StateT, DepsT]: ...

    @overload
    def as_node(self, inputs: InputT) -> JoinNode[StateT, DepsT]: ...

    def as_node(self, inputs: InputT | None = None) -> JoinNode[StateT, DepsT]:
        """Create a step node with bound inputs.

        Args:
            inputs: The input data to bind to this step, or None

        Returns:
            A [`StepNode`][pydantic_graph.beta.step.StepNode] with this step and the bound inputs
        """
        return JoinNode(self, inputs)

```

#### as_node

```python
as_node(inputs: None = None) -> JoinNode[StateT, DepsT]

```

```python
as_node(inputs: InputT) -> JoinNode[StateT, DepsT]

```

```python
as_node(
    inputs: InputT | None = None,
) -> JoinNode[StateT, DepsT]

```

Create a step node with bound inputs.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `inputs` | `InputT | None` | The input data to bind to this step, or None | `None` |

Returns:

| Type | Description | | --- | --- | | `JoinNode[StateT, DepsT]` | A StepNode with this step and the bound inputs |

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
def as_node(self, inputs: InputT | None = None) -> JoinNode[StateT, DepsT]:
    """Create a step node with bound inputs.

    Args:
        inputs: The input data to bind to this step, or None

    Returns:
        A [`StepNode`][pydantic_graph.beta.step.StepNode] with this step and the bound inputs
    """
    return JoinNode(self, inputs)

```

### JoinNode

Bases: `BaseNode[StateT, DepsT, Any]`

A base node that represents a join item with bound inputs.

JoinNode bridges between the v1 and v2 graph execution systems by wrapping a Join with bound inputs in a BaseNode interface. It is not meant to be run directly but rather used to indicate transitions to v2-style steps.

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
@dataclass
class JoinNode(BaseNode[StateT, DepsT, Any]):
    """A base node that represents a join item with bound inputs.

    JoinNode bridges between the v1 and v2 graph execution systems by wrapping
    a [`Join`][pydantic_graph.beta.join.Join] with bound inputs in a BaseNode interface.
    It is not meant to be run directly but rather used to indicate transitions
    to v2-style steps.
    """

    join: Join[StateT, DepsT, Any, Any]
    """The step to execute."""

    inputs: Any
    """The inputs bound to this step."""

    async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
        """Attempt to run the join node.

        Args:
            ctx: The graph execution context

        Returns:
            The result of step execution

        Raises:
            NotImplementedError: Always raised as StepNode is not meant to be run directly
        """
        raise NotImplementedError(
            '`JoinNode` is not meant to be run directly, it is meant to be used in `BaseNode` subclasses to indicate a transition to v2-style steps.'
        )

```

#### join

```python
join: Join[StateT, DepsT, Any, Any]

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

Attempt to run the join node.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `GraphRunContext[StateT, DepsT]` | The graph execution context | *required* |

Returns:

| Type | Description | | --- | --- | | `BaseNode[StateT, DepsT, Any] | End[Any]` | The result of step execution |

Raises:

| Type | Description | | --- | --- | | `NotImplementedError` | Always raised as StepNode is not meant to be run directly |

Source code in `pydantic_graph/pydantic_graph/beta/join.py`

```python
async def run(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[Any]:
    """Attempt to run the join node.

    Args:
        ctx: The graph execution context

    Returns:
        The result of step execution

    Raises:
        NotImplementedError: Always raised as StepNode is not meant to be run directly
    """
    raise NotImplementedError(
        '`JoinNode` is not meant to be run directly, it is meant to be used in `BaseNode` subclasses to indicate a transition to v2-style steps.'
    )

```

# `pydantic_graph.beta.node`

Core node types for graph construction and execution.

This module defines the fundamental node types used to build execution graphs, including start/end nodes and fork nodes for parallel execution.

### StateT

```python
StateT = TypeVar('StateT', infer_variance=True)

```

Type variable for graph state.

### OutputT

```python
OutputT = TypeVar('OutputT', infer_variance=True)

```

Type variable for node output data.

### InputT

```python
InputT = TypeVar('InputT', infer_variance=True)

```

Type variable for node input data.

### StartNode

Bases: `Generic[OutputT]`

Entry point node for graph execution.

The StartNode represents the beginning of a graph execution flow.

Source code in `pydantic_graph/pydantic_graph/beta/node.py`

```python
class StartNode(Generic[OutputT]):
    """Entry point node for graph execution.

    The StartNode represents the beginning of a graph execution flow.
    """

    id = NodeID('__start__')
    """Fixed identifier for the start node."""

```

#### id

```python
id = NodeID('__start__')

```

Fixed identifier for the start node.

### EndNode

Bases: `Generic[InputT]`

Terminal node representing the completion of graph execution.

The EndNode marks the successful completion of a graph execution flow and can collect the final output data.

Source code in `pydantic_graph/pydantic_graph/beta/node.py`

```python
class EndNode(Generic[InputT]):
    """Terminal node representing the completion of graph execution.

    The EndNode marks the successful completion of a graph execution flow
    and can collect the final output data.
    """

    id = NodeID('__end__')
    """Fixed identifier for the end node."""

    def _force_variance(self, inputs: InputT) -> None:  # pragma: no cover
        """Force type variance for proper generic typing.

        This method exists solely for type checking purposes and should never be called.

        Args:
            inputs: Input data of type InputT.

        Raises:
            RuntimeError: Always, as this method should never be executed.
        """
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')

```

#### id

```python
id = NodeID('__end__')

```

Fixed identifier for the end node.

### Fork

Bases: `Generic[InputT, OutputT]`

Fork node that creates parallel execution branches.

A Fork node splits the execution flow into multiple parallel branches, enabling concurrent execution of downstream nodes. It can either map a sequence across multiple branches or duplicate data to each branch.

Source code in `pydantic_graph/pydantic_graph/beta/node.py`

```python
@dataclass
class Fork(Generic[InputT, OutputT]):
    """Fork node that creates parallel execution branches.

    A Fork node splits the execution flow into multiple parallel branches,
    enabling concurrent execution of downstream nodes. It can either map
    a sequence across multiple branches or duplicate data to each branch.
    """

    id: ForkID
    """Unique identifier for this fork node."""

    is_map: bool
    """Determines fork behavior.

    If True, InputT must be Sequence[OutputT] and each element is sent to a separate branch.
    If False, InputT must be OutputT and the same data is sent to all branches.
    """
    downstream_join_id: JoinID | None
    """Optional identifier of a downstream join node that should be jumped to if mapping an empty iterable."""

    def _force_variance(self, inputs: InputT) -> OutputT:  # pragma: no cover
        """Force type variance for proper generic typing.

        This method exists solely for type checking purposes and should never be called.

        Args:
            inputs: Input data to be forked.

        Returns:
            Output data type (never actually returned).

        Raises:
            RuntimeError: Always, as this method should never be executed.
        """
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')

```

#### id

```python
id: ForkID

```

Unique identifier for this fork node.

#### is_map

```python
is_map: bool

```

Determines fork behavior.

If True, InputT must be Sequence[OutputT] and each element is sent to a separate branch. If False, InputT must be OutputT and the same data is sent to all branches.

#### downstream_join_id

```python
downstream_join_id: JoinID | None

```

Optional identifier of a downstream join node that should be jumped to if mapping an empty iterable.

