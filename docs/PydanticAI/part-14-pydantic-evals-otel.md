<!-- Source: _complete-doc.md -->

# `pydantic_evals.otel`

### SpanNode

A node in the span tree; provides references to parents/children for easy traversal and queries.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
@dataclass(repr=False, kw_only=True)
class SpanNode:
    """A node in the span tree; provides references to parents/children for easy traversal and queries."""

    name: str
    trace_id: int
    span_id: int
    parent_span_id: int | None
    start_timestamp: datetime
    end_timestamp: datetime
    attributes: dict[str, AttributeValue]

    @property
    def duration(self) -> timedelta:
        """Return the span's duration as a timedelta, or None if start/end not set."""
        return self.end_timestamp - self.start_timestamp

    @property
    def children(self) -> list[SpanNode]:
        return list(self.children_by_id.values())

    @property
    def descendants(self) -> list[SpanNode]:
        """Return all descendants of this node in DFS order."""
        return self.find_descendants(lambda _: True)

    @property
    def ancestors(self) -> list[SpanNode]:
        """Return all ancestors of this node."""
        return self.find_ancestors(lambda _: True)

    @property
    def node_key(self) -> str:
        return f'{self.trace_id:032x}:{self.span_id:016x}'

    @property
    def parent_node_key(self) -> str | None:
        return None if self.parent_span_id is None else f'{self.trace_id:032x}:{self.parent_span_id:016x}'

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.parent: SpanNode | None = None
        self.children_by_id: dict[str, SpanNode] = {}

    @staticmethod
    def from_readable_span(span: ReadableSpan) -> SpanNode:
        assert span.context is not None, 'Span has no context'
        assert span.start_time is not None, 'Span has no start time'
        assert span.end_time is not None, 'Span has no end time'
        return SpanNode(
            name=span.name,
            trace_id=span.context.trace_id,
            span_id=span.context.span_id,
            parent_span_id=span.parent.span_id if span.parent else None,
            start_timestamp=datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc),
            end_timestamp=datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc),
            attributes=dict(span.attributes or {}),
        )

    def add_child(self, child: SpanNode) -> None:
        """Attach a child node to this node's list of children."""
        assert child.trace_id == self.trace_id, f"traces don't match: {child.trace_id:032x} != {self.trace_id:032x}"
        assert child.parent_span_id == self.span_id, (
            f'parent span mismatch: {child.parent_span_id:016x} != {self.span_id:016x}'
        )
        self.children_by_id[child.node_key] = child
        child.parent = self

    # -------------------------------------------------------------------------
    # Child queries
    # -------------------------------------------------------------------------
    def find_children(self, predicate: SpanQuery | SpanPredicate) -> list[SpanNode]:
        """Return all immediate children that satisfy the given predicate."""
        return list(self._filter_children(predicate))

    def first_child(self, predicate: SpanQuery | SpanPredicate) -> SpanNode | None:
        """Return the first immediate child that satisfies the given predicate, or None if none match."""
        return next(self._filter_children(predicate), None)

    def any_child(self, predicate: SpanQuery | SpanPredicate) -> bool:
        """Returns True if there is at least one child that satisfies the predicate."""
        return self.first_child(predicate) is not None

    def _filter_children(self, predicate: SpanQuery | SpanPredicate) -> Iterator[SpanNode]:
        return (child for child in self.children if child.matches(predicate))

    # -------------------------------------------------------------------------
    # Descendant queries (DFS)
    # -------------------------------------------------------------------------
    def find_descendants(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> list[SpanNode]:
        """Return all descendant nodes that satisfy the given predicate in DFS order."""
        return list(self._filter_descendants(predicate, stop_recursing_when))

    def first_descendant(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> SpanNode | None:
        """DFS: Return the first descendant (in DFS order) that satisfies the given predicate, or `None` if none match."""
        return next(self._filter_descendants(predicate, stop_recursing_when), None)

    def any_descendant(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> bool:
        """Returns `True` if there is at least one descendant that satisfies the predicate."""
        return self.first_descendant(predicate, stop_recursing_when) is not None

    def _filter_descendants(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None
    ) -> Iterator[SpanNode]:
        stack = list(self.children)
        while stack:
            node = stack.pop()
            if node.matches(predicate):
                yield node
            if stop_recursing_when is not None and node.matches(stop_recursing_when):
                continue
            stack.extend(node.children)

    # -------------------------------------------------------------------------
    # Ancestor queries (DFS "up" the chain)
    # -------------------------------------------------------------------------
    def find_ancestors(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> list[SpanNode]:
        """Return all ancestors that satisfy the given predicate."""
        return list(self._filter_ancestors(predicate, stop_recursing_when))

    def first_ancestor(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> SpanNode | None:
        """Return the closest ancestor that satisfies the given predicate, or `None` if none match."""
        return next(self._filter_ancestors(predicate, stop_recursing_when), None)

    def any_ancestor(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
    ) -> bool:
        """Returns True if any ancestor satisfies the predicate."""
        return self.first_ancestor(predicate, stop_recursing_when) is not None

    def _filter_ancestors(
        self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None
    ) -> Iterator[SpanNode]:
        node = self.parent
        while node:
            if node.matches(predicate):
                yield node
            if stop_recursing_when is not None and node.matches(stop_recursing_when):
                break
            node = node.parent

    # -------------------------------------------------------------------------
    # Query matching
    # -------------------------------------------------------------------------
    def matches(self, query: SpanQuery | SpanPredicate) -> bool:
        """Check if the span node matches the query conditions or predicate."""
        if callable(query):
            return query(self)

        return self._matches_query(query)

    def _matches_query(self, query: SpanQuery) -> bool:  # noqa C901
        """Check if the span matches the query conditions."""
        # Logical combinations
        if or_ := query.get('or_'):
            if len(query) > 1:
                raise ValueError("Cannot combine 'or_' conditions with other conditions at the same level")
            return any(self._matches_query(q) for q in or_)
        if not_ := query.get('not_'):
            if self._matches_query(not_):
                return False
        if and_ := query.get('and_'):
            results = [self._matches_query(q) for q in and_]
            if not all(results):
                return False
        # At this point, all existing ANDs and no existing ORs have passed, so it comes down to this condition

        # Name conditions
        if (name_equals := query.get('name_equals')) and self.name != name_equals:
            return False
        if (name_contains := query.get('name_contains')) and name_contains not in self.name:
            return False
        if (name_matches_regex := query.get('name_matches_regex')) and not re.match(name_matches_regex, self.name):
            return False

        # Attribute conditions
        if (has_attributes := query.get('has_attributes')) and not all(
            self.attributes.get(key) == value for key, value in has_attributes.items()
        ):
            return False
        if (has_attributes_keys := query.get('has_attribute_keys')) and not all(
            key in self.attributes for key in has_attributes_keys
        ):
            return False

        # Timing conditions
        if (min_duration := query.get('min_duration')) is not None:
            if not isinstance(min_duration, timedelta):
                min_duration = timedelta(seconds=min_duration)
            if self.duration < min_duration:
                return False
        if (max_duration := query.get('max_duration')) is not None:
            if not isinstance(max_duration, timedelta):
                max_duration = timedelta(seconds=max_duration)
            if self.duration > max_duration:
                return False

        # Children conditions
        if (min_child_count := query.get('min_child_count')) and len(self.children) < min_child_count:
            return False
        if (max_child_count := query.get('max_child_count')) and len(self.children) > max_child_count:
            return False
        if (some_child_has := query.get('some_child_has')) and not any(
            child._matches_query(some_child_has) for child in self.children
        ):
            return False
        if (all_children_have := query.get('all_children_have')) and not all(
            child._matches_query(all_children_have) for child in self.children
        ):
            return False
        if (no_child_has := query.get('no_child_has')) and any(
            child._matches_query(no_child_has) for child in self.children
        ):
            return False

        # Descendant conditions
        # The following local functions with cache decorators are used to avoid repeatedly evaluating these properties
        @cache
        def descendants():
            return self.descendants

        @cache
        def pruned_descendants():
            stop_recursing_when = query.get('stop_recursing_when')
            return (
                self._filter_descendants(lambda _: True, stop_recursing_when) if stop_recursing_when else descendants()
            )

        if (min_descendant_count := query.get('min_descendant_count')) and len(descendants()) < min_descendant_count:
            return False
        if (max_descendant_count := query.get('max_descendant_count')) and len(descendants()) > max_descendant_count:
            return False
        if (some_descendant_has := query.get('some_descendant_has')) and not any(
            descendant._matches_query(some_descendant_has) for descendant in pruned_descendants()
        ):
            return False
        if (all_descendants_have := query.get('all_descendants_have')) and not all(
            descendant._matches_query(all_descendants_have) for descendant in pruned_descendants()
        ):
            return False
        if (no_descendant_has := query.get('no_descendant_has')) and any(
            descendant._matches_query(no_descendant_has) for descendant in pruned_descendants()
        ):
            return False

        # Ancestor conditions
        # The following local functions with cache decorators are used to avoid repeatedly evaluating these properties
        @cache
        def ancestors():
            return self.ancestors

        @cache
        def pruned_ancestors():
            stop_recursing_when = query.get('stop_recursing_when')
            return self._filter_ancestors(lambda _: True, stop_recursing_when) if stop_recursing_when else ancestors()

        if (min_depth := query.get('min_depth')) and len(ancestors()) < min_depth:
            return False
        if (max_depth := query.get('max_depth')) and len(ancestors()) > max_depth:
            return False
        if (some_ancestor_has := query.get('some_ancestor_has')) and not any(
            ancestor._matches_query(some_ancestor_has) for ancestor in pruned_ancestors()
        ):
            return False
        if (all_ancestors_have := query.get('all_ancestors_have')) and not all(
            ancestor._matches_query(all_ancestors_have) for ancestor in pruned_ancestors()
        ):
            return False
        if (no_ancestor_has := query.get('no_ancestor_has')) and any(
            ancestor._matches_query(no_ancestor_has) for ancestor in pruned_ancestors()
        ):
            return False

        return True

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------
    def repr_xml(
        self,
        include_children: bool = True,
        include_trace_id: bool = False,
        include_span_id: bool = False,
        include_start_timestamp: bool = False,
        include_duration: bool = False,
    ) -> str:
        """Return an XML-like string representation of the node.

        Optionally includes children, trace_id, span_id, start_timestamp, and duration.
        """
        first_line_parts = [f'<SpanNode name={self.name!r}']
        if include_trace_id:
            first_line_parts.append(f"trace_id='{self.trace_id:032x}'")
        if include_span_id:
            first_line_parts.append(f"span_id='{self.span_id:016x}'")
        if include_start_timestamp:
            first_line_parts.append(f'start_timestamp={self.start_timestamp.isoformat()!r}')
        if include_duration:
            first_line_parts.append(f"duration='{self.duration}'")

        extra_lines: list[str] = []
        if include_children and self.children:
            first_line_parts.append('>')
            for child in self.children:
                extra_lines.append(
                    indent(
                        child.repr_xml(
                            include_children=include_children,
                            include_trace_id=include_trace_id,
                            include_span_id=include_span_id,
                            include_start_timestamp=include_start_timestamp,
                            include_duration=include_duration,
                        ),
                        '  ',
                    )
                )
            extra_lines.append('</SpanNode>')
        else:
            if self.children:
                first_line_parts.append('children=...')
            first_line_parts.append('/>')
        return '\n'.join([' '.join(first_line_parts), *extra_lines])

    def __str__(self) -> str:
        if self.children:
            return f"<SpanNode name={self.name!r} span_id='{self.span_id:016x}'>...</SpanNode>"
        else:
            return f"<SpanNode name={self.name!r} span_id='{self.span_id:016x}' />"

    def __repr__(self) -> str:
        return self.repr_xml()

```

#### duration

```python
duration: timedelta

```

Return the span's duration as a timedelta, or None if start/end not set.

#### descendants

```python
descendants: list[SpanNode]

```

Return all descendants of this node in DFS order.

#### ancestors

```python
ancestors: list[SpanNode]

```

Return all ancestors of this node.

#### add_child

```python
add_child(child: SpanNode) -> None

```

Attach a child node to this node's list of children.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def add_child(self, child: SpanNode) -> None:
    """Attach a child node to this node's list of children."""
    assert child.trace_id == self.trace_id, f"traces don't match: {child.trace_id:032x} != {self.trace_id:032x}"
    assert child.parent_span_id == self.span_id, (
        f'parent span mismatch: {child.parent_span_id:016x} != {self.span_id:016x}'
    )
    self.children_by_id[child.node_key] = child
    child.parent = self

```

#### find_children

```python
find_children(
    predicate: SpanQuery | SpanPredicate,
) -> list[SpanNode]

```

Return all immediate children that satisfy the given predicate.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def find_children(self, predicate: SpanQuery | SpanPredicate) -> list[SpanNode]:
    """Return all immediate children that satisfy the given predicate."""
    return list(self._filter_children(predicate))

```

#### first_child

```python
first_child(
    predicate: SpanQuery | SpanPredicate,
) -> SpanNode | None

```

Return the first immediate child that satisfies the given predicate, or None if none match.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def first_child(self, predicate: SpanQuery | SpanPredicate) -> SpanNode | None:
    """Return the first immediate child that satisfies the given predicate, or None if none match."""
    return next(self._filter_children(predicate), None)

```

#### any_child

```python
any_child(predicate: SpanQuery | SpanPredicate) -> bool

```

Returns True if there is at least one child that satisfies the predicate.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def any_child(self, predicate: SpanQuery | SpanPredicate) -> bool:
    """Returns True if there is at least one child that satisfies the predicate."""
    return self.first_child(predicate) is not None

```

#### find_descendants

```python
find_descendants(
    predicate: SpanQuery | SpanPredicate,
    stop_recursing_when: (
        SpanQuery | SpanPredicate | None
    ) = None,
) -> list[SpanNode]

```

Return all descendant nodes that satisfy the given predicate in DFS order.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def find_descendants(
    self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
) -> list[SpanNode]:
    """Return all descendant nodes that satisfy the given predicate in DFS order."""
    return list(self._filter_descendants(predicate, stop_recursing_when))

```

#### first_descendant

```python
first_descendant(
    predicate: SpanQuery | SpanPredicate,
    stop_recursing_when: (
        SpanQuery | SpanPredicate | None
    ) = None,
) -> SpanNode | None

```

DFS: Return the first descendant (in DFS order) that satisfies the given predicate, or `None` if none match.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def first_descendant(
    self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
) -> SpanNode | None:
    """DFS: Return the first descendant (in DFS order) that satisfies the given predicate, or `None` if none match."""
    return next(self._filter_descendants(predicate, stop_recursing_when), None)

```

#### any_descendant

```python
any_descendant(
    predicate: SpanQuery | SpanPredicate,
    stop_recursing_when: (
        SpanQuery | SpanPredicate | None
    ) = None,
) -> bool

```

Returns `True` if there is at least one descendant that satisfies the predicate.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def any_descendant(
    self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
) -> bool:
    """Returns `True` if there is at least one descendant that satisfies the predicate."""
    return self.first_descendant(predicate, stop_recursing_when) is not None

```

#### find_ancestors

```python
find_ancestors(
    predicate: SpanQuery | SpanPredicate,
    stop_recursing_when: (
        SpanQuery | SpanPredicate | None
    ) = None,
) -> list[SpanNode]

```

Return all ancestors that satisfy the given predicate.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def find_ancestors(
    self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
) -> list[SpanNode]:
    """Return all ancestors that satisfy the given predicate."""
    return list(self._filter_ancestors(predicate, stop_recursing_when))

```

#### first_ancestor

```python
first_ancestor(
    predicate: SpanQuery | SpanPredicate,
    stop_recursing_when: (
        SpanQuery | SpanPredicate | None
    ) = None,
) -> SpanNode | None

```

Return the closest ancestor that satisfies the given predicate, or `None` if none match.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def first_ancestor(
    self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
) -> SpanNode | None:
    """Return the closest ancestor that satisfies the given predicate, or `None` if none match."""
    return next(self._filter_ancestors(predicate, stop_recursing_when), None)

```

#### any_ancestor

```python
any_ancestor(
    predicate: SpanQuery | SpanPredicate,
    stop_recursing_when: (
        SpanQuery | SpanPredicate | None
    ) = None,
) -> bool

```

Returns True if any ancestor satisfies the predicate.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def any_ancestor(
    self, predicate: SpanQuery | SpanPredicate, stop_recursing_when: SpanQuery | SpanPredicate | None = None
) -> bool:
    """Returns True if any ancestor satisfies the predicate."""
    return self.first_ancestor(predicate, stop_recursing_when) is not None

```

#### matches

```python
matches(query: SpanQuery | SpanPredicate) -> bool

```

Check if the span node matches the query conditions or predicate.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def matches(self, query: SpanQuery | SpanPredicate) -> bool:
    """Check if the span node matches the query conditions or predicate."""
    if callable(query):
        return query(self)

    return self._matches_query(query)

```

#### repr_xml

```python
repr_xml(
    include_children: bool = True,
    include_trace_id: bool = False,
    include_span_id: bool = False,
    include_start_timestamp: bool = False,
    include_duration: bool = False,
) -> str

```

Return an XML-like string representation of the node.

Optionally includes children, trace_id, span_id, start_timestamp, and duration.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def repr_xml(
    self,
    include_children: bool = True,
    include_trace_id: bool = False,
    include_span_id: bool = False,
    include_start_timestamp: bool = False,
    include_duration: bool = False,
) -> str:
    """Return an XML-like string representation of the node.

    Optionally includes children, trace_id, span_id, start_timestamp, and duration.
    """
    first_line_parts = [f'<SpanNode name={self.name!r}']
    if include_trace_id:
        first_line_parts.append(f"trace_id='{self.trace_id:032x}'")
    if include_span_id:
        first_line_parts.append(f"span_id='{self.span_id:016x}'")
    if include_start_timestamp:
        first_line_parts.append(f'start_timestamp={self.start_timestamp.isoformat()!r}')
    if include_duration:
        first_line_parts.append(f"duration='{self.duration}'")

    extra_lines: list[str] = []
    if include_children and self.children:
        first_line_parts.append('>')
        for child in self.children:
            extra_lines.append(
                indent(
                    child.repr_xml(
                        include_children=include_children,
                        include_trace_id=include_trace_id,
                        include_span_id=include_span_id,
                        include_start_timestamp=include_start_timestamp,
                        include_duration=include_duration,
                    ),
                    '  ',
                )
            )
        extra_lines.append('</SpanNode>')
    else:
        if self.children:
            first_line_parts.append('children=...')
        first_line_parts.append('/>')
    return '\n'.join([' '.join(first_line_parts), *extra_lines])

```

### SpanQuery

Bases: `TypedDict`

A serializable query for filtering SpanNodes based on various conditions.

All fields are optional and combined with AND logic by default.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
class SpanQuery(TypedDict, total=False):
    """A serializable query for filtering SpanNodes based on various conditions.

    All fields are optional and combined with AND logic by default.
    """

    # These fields are ordered to match the implementation of SpanNode.matches_query for easy review.
    # * Individual span conditions come first because these are generally the cheapest to evaluate
    # * Logical combinations come next because they may just be combinations of individual span conditions
    # * Related-span conditions come last because they may require the most work to evaluate

    # Individual span conditions
    ## Name conditions
    name_equals: str
    name_contains: str
    name_matches_regex: str  # regex pattern

    ## Attribute conditions
    has_attributes: dict[str, Any]
    has_attribute_keys: list[str]

    ## Timing conditions
    min_duration: timedelta | float
    max_duration: timedelta | float

    # Logical combinations of conditions
    not_: SpanQuery
    and_: list[SpanQuery]
    or_: list[SpanQuery]

    # Child conditions
    min_child_count: int
    max_child_count: int
    some_child_has: SpanQuery
    all_children_have: SpanQuery
    no_child_has: SpanQuery

    # Recursive conditions
    stop_recursing_when: SpanQuery
    """If present, stop recursing through ancestors or descendants at nodes that match this condition."""

    ## Descendant conditions
    min_descendant_count: int
    max_descendant_count: int
    some_descendant_has: SpanQuery
    all_descendants_have: SpanQuery
    no_descendant_has: SpanQuery

    ## Ancestor conditions
    min_depth: int  # depth is equivalent to ancestor count; roots have depth 0
    max_depth: int
    some_ancestor_has: SpanQuery
    all_ancestors_have: SpanQuery
    no_ancestor_has: SpanQuery

```

#### stop_recursing_when

```python
stop_recursing_when: SpanQuery

```

If present, stop recursing through ancestors or descendants at nodes that match this condition.

### SpanTree

A container that builds a hierarchy of SpanNode objects from a list of finished spans.

You can then search or iterate the tree to make your assertions (using DFS for traversal).

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
@dataclass(repr=False, kw_only=True)
class SpanTree:
    """A container that builds a hierarchy of SpanNode objects from a list of finished spans.

    You can then search or iterate the tree to make your assertions (using DFS for traversal).
    """

    roots: list[SpanNode] = field(default_factory=list)
    nodes_by_id: dict[str, SpanNode] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self._rebuild_tree()

    def add_spans(self, spans: list[SpanNode]) -> None:
        """Add a list of spans to the tree, rebuilding the tree structure."""
        for span in spans:
            self.nodes_by_id[span.node_key] = span
        self._rebuild_tree()

    def add_readable_spans(self, readable_spans: list[ReadableSpan]):
        self.add_spans([SpanNode.from_readable_span(span) for span in readable_spans])

    def _rebuild_tree(self):
        # Ensure spans are ordered by start_timestamp so that roots and children end up in the right order
        nodes = list(self.nodes_by_id.values())
        nodes.sort(key=lambda node: node.start_timestamp or datetime.min)
        self.nodes_by_id = {node.node_key: node for node in nodes}

        # Build the parent/child relationships
        for node in self.nodes_by_id.values():
            parent_node_key = node.parent_node_key
            if parent_node_key is not None:
                parent_node = self.nodes_by_id.get(parent_node_key)
                if parent_node is not None:
                    parent_node.add_child(node)

        # Determine the roots
        # A node is a "root" if its parent is None or if its parent's span_id is not in the current set of spans.
        self.roots = []
        for node in self.nodes_by_id.values():
            parent_node_key = node.parent_node_key
            if parent_node_key is None or parent_node_key not in self.nodes_by_id:
                self.roots.append(node)

    # -------------------------------------------------------------------------
    # Node filtering and iteration
    # -------------------------------------------------------------------------
    def find(self, predicate: SpanQuery | SpanPredicate) -> list[SpanNode]:
        """Find all nodes in the entire tree that match the predicate, scanning from each root in DFS order."""
        return list(self._filter(predicate))

    def first(self, predicate: SpanQuery | SpanPredicate) -> SpanNode | None:
        """Find the first node that matches a predicate, scanning from each root in DFS order. Returns `None` if not found."""
        return next(self._filter(predicate), None)

    def any(self, predicate: SpanQuery | SpanPredicate) -> bool:
        """Returns True if any node in the tree matches the predicate."""
        return self.first(predicate) is not None

    def _filter(self, predicate: SpanQuery | SpanPredicate) -> Iterator[SpanNode]:
        for node in self:
            if node.matches(predicate):
                yield node

    def __iter__(self) -> Iterator[SpanNode]:
        """Return an iterator over all nodes in the tree."""
        return iter(self.nodes_by_id.values())

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------
    def repr_xml(
        self,
        include_children: bool = True,
        include_trace_id: bool = False,
        include_span_id: bool = False,
        include_start_timestamp: bool = False,
        include_duration: bool = False,
    ) -> str:
        """Return an XML-like string representation of the tree, optionally including children, trace_id, span_id, duration, and timestamps."""
        if not self.roots:
            return '<SpanTree />'
        repr_parts = [
            '<SpanTree>',
            *[
                indent(
                    root.repr_xml(
                        include_children=include_children,
                        include_trace_id=include_trace_id,
                        include_span_id=include_span_id,
                        include_start_timestamp=include_start_timestamp,
                        include_duration=include_duration,
                    ),
                    '  ',
                )
                for root in self.roots
            ],
            '</SpanTree>',
        ]
        return '\n'.join(repr_parts)

    def __str__(self):
        return f'<SpanTree num_roots={len(self.roots)} total_spans={len(self.nodes_by_id)} />'

    def __repr__(self):
        return self.repr_xml()

```

#### add_spans

```python
add_spans(spans: list[SpanNode]) -> None

```

Add a list of spans to the tree, rebuilding the tree structure.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def add_spans(self, spans: list[SpanNode]) -> None:
    """Add a list of spans to the tree, rebuilding the tree structure."""
    for span in spans:
        self.nodes_by_id[span.node_key] = span
    self._rebuild_tree()

```

#### find

```python
find(
    predicate: SpanQuery | SpanPredicate,
) -> list[SpanNode]

```

Find all nodes in the entire tree that match the predicate, scanning from each root in DFS order.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def find(self, predicate: SpanQuery | SpanPredicate) -> list[SpanNode]:
    """Find all nodes in the entire tree that match the predicate, scanning from each root in DFS order."""
    return list(self._filter(predicate))

```

#### first

```python
first(
    predicate: SpanQuery | SpanPredicate,
) -> SpanNode | None

```

Find the first node that matches a predicate, scanning from each root in DFS order. Returns `None` if not found.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def first(self, predicate: SpanQuery | SpanPredicate) -> SpanNode | None:
    """Find the first node that matches a predicate, scanning from each root in DFS order. Returns `None` if not found."""
    return next(self._filter(predicate), None)

```

#### any

```python
any(predicate: SpanQuery | SpanPredicate) -> bool

```

Returns True if any node in the tree matches the predicate.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def any(self, predicate: SpanQuery | SpanPredicate) -> bool:
    """Returns True if any node in the tree matches the predicate."""
    return self.first(predicate) is not None

```

#### __iter__

```python
__iter__() -> Iterator[SpanNode]

```

Return an iterator over all nodes in the tree.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def __iter__(self) -> Iterator[SpanNode]:
    """Return an iterator over all nodes in the tree."""
    return iter(self.nodes_by_id.values())

```

#### repr_xml

```python
repr_xml(
    include_children: bool = True,
    include_trace_id: bool = False,
    include_span_id: bool = False,
    include_start_timestamp: bool = False,
    include_duration: bool = False,
) -> str

```

Return an XML-like string representation of the tree, optionally including children, trace_id, span_id, duration, and timestamps.

Source code in `pydantic_evals/pydantic_evals/otel/span_tree.py`

```python
def repr_xml(
    self,
    include_children: bool = True,
    include_trace_id: bool = False,
    include_span_id: bool = False,
    include_start_timestamp: bool = False,
    include_duration: bool = False,
) -> str:
    """Return an XML-like string representation of the tree, optionally including children, trace_id, span_id, duration, and timestamps."""
    if not self.roots:
        return '<SpanTree />'
    repr_parts = [
        '<SpanTree>',
        *[
            indent(
                root.repr_xml(
                    include_children=include_children,
                    include_trace_id=include_trace_id,
                    include_span_id=include_span_id,
                    include_start_timestamp=include_start_timestamp,
                    include_duration=include_duration,
                ),
                '  ',
            )
            for root in self.roots
        ],
        '</SpanTree>',
    ]
    return '\n'.join(repr_parts)

```

# `pydantic_evals.reporting`

### ReportCase

Bases: `Generic[InputsT, OutputT, MetadataT]`

A single case in an evaluation report.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
@dataclass(kw_only=True)
class ReportCase(Generic[InputsT, OutputT, MetadataT]):
    """A single case in an evaluation report."""

    name: str
    """The name of the [case][pydantic_evals.Case]."""
    inputs: InputsT
    """The inputs to the task, from [`Case.inputs`][pydantic_evals.Case.inputs]."""
    metadata: MetadataT | None
    """Any metadata associated with the case, from [`Case.metadata`][pydantic_evals.Case.metadata]."""
    expected_output: OutputT | None
    """The expected output of the task, from [`Case.expected_output`][pydantic_evals.Case.expected_output]."""
    output: OutputT
    """The output of the task execution."""

    metrics: dict[str, float | int]
    attributes: dict[str, Any]

    scores: dict[str, EvaluationResult[int | float]]
    labels: dict[str, EvaluationResult[str]]
    assertions: dict[str, EvaluationResult[bool]]

    task_duration: float
    total_duration: float  # includes evaluator execution time

    trace_id: str | None = None
    """The trace ID of the case span."""
    span_id: str | None = None
    """The span ID of the case span."""

    evaluator_failures: list[EvaluatorFailure] = field(default_factory=list)

```

#### name

```python
name: str

```

The name of the case.

#### inputs

```python
inputs: InputsT

```

The inputs to the task, from Case.inputs.

#### metadata

```python
metadata: MetadataT | None

```

Any metadata associated with the case, from Case.metadata.

#### expected_output

```python
expected_output: OutputT | None

```

The expected output of the task, from Case.expected_output.

#### output

```python
output: OutputT

```

The output of the task execution.

#### trace_id

```python
trace_id: str | None = None

```

The trace ID of the case span.

#### span_id

```python
span_id: str | None = None

```

The span ID of the case span.

### ReportCaseFailure

Bases: `Generic[InputsT, OutputT, MetadataT]`

A single case in an evaluation report that failed due to an error during task execution.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
@dataclass(kw_only=True)
class ReportCaseFailure(Generic[InputsT, OutputT, MetadataT]):
    """A single case in an evaluation report that failed due to an error during task execution."""

    name: str
    """The name of the [case][pydantic_evals.Case]."""
    inputs: InputsT
    """The inputs to the task, from [`Case.inputs`][pydantic_evals.Case.inputs]."""
    metadata: MetadataT | None
    """Any metadata associated with the case, from [`Case.metadata`][pydantic_evals.Case.metadata]."""
    expected_output: OutputT | None
    """The expected output of the task, from [`Case.expected_output`][pydantic_evals.Case.expected_output]."""

    error_message: str
    """The message of the exception that caused the failure."""
    error_stacktrace: str
    """The stacktrace of the exception that caused the failure."""

    trace_id: str | None = None
    """The trace ID of the case span."""
    span_id: str | None = None
    """The span ID of the case span."""

```

#### name

```python
name: str

```

The name of the case.

#### inputs

```python
inputs: InputsT

```

The inputs to the task, from Case.inputs.

#### metadata

```python
metadata: MetadataT | None

```

Any metadata associated with the case, from Case.metadata.

#### expected_output

```python
expected_output: OutputT | None

```

The expected output of the task, from Case.expected_output.

#### error_message

```python
error_message: str

```

The message of the exception that caused the failure.

#### error_stacktrace

```python
error_stacktrace: str

```

The stacktrace of the exception that caused the failure.

#### trace_id

```python
trace_id: str | None = None

```

The trace ID of the case span.

#### span_id

```python
span_id: str | None = None

```

The span ID of the case span.

### ReportCaseAggregate

Bases: `BaseModel`

A synthetic case that summarizes a set of cases.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
class ReportCaseAggregate(BaseModel):
    """A synthetic case that summarizes a set of cases."""

    name: str

    scores: dict[str, float | int]
    labels: dict[str, dict[str, float]]
    metrics: dict[str, float | int]
    assertions: float | None
    task_duration: float
    total_duration: float

    @staticmethod
    def average(cases: list[ReportCase]) -> ReportCaseAggregate:
        """Produce a synthetic "summary" case by averaging quantitative attributes."""
        num_cases = len(cases)
        if num_cases == 0:
            return ReportCaseAggregate(
                name='Averages',
                scores={},
                labels={},
                metrics={},
                assertions=None,
                task_duration=0.0,
                total_duration=0.0,
            )

        def _scores_averages(scores_by_name: list[dict[str, int | float | bool]]) -> dict[str, float]:
            counts_by_name: dict[str, int] = defaultdict(int)
            sums_by_name: dict[str, float] = defaultdict(float)
            for sbn in scores_by_name:
                for name, score in sbn.items():
                    counts_by_name[name] += 1
                    sums_by_name[name] += score
            return {name: sums_by_name[name] / counts_by_name[name] for name in sums_by_name}

        def _labels_averages(labels_by_name: list[dict[str, str]]) -> dict[str, dict[str, float]]:
            counts_by_name: dict[str, int] = defaultdict(int)
            sums_by_name: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            for lbn in labels_by_name:
                for name, label in lbn.items():
                    counts_by_name[name] += 1
                    sums_by_name[name][label] += 1
            return {
                name: {value: count / counts_by_name[name] for value, count in sums_by_name[name].items()}
                for name in sums_by_name
            }

        average_task_duration = sum(case.task_duration for case in cases) / num_cases
        average_total_duration = sum(case.total_duration for case in cases) / num_cases

        # average_assertions: dict[str, float] = _scores_averages([{k: v.value for k, v in case.scores.items()} for case in cases])
        average_scores: dict[str, float] = _scores_averages(
            [{k: v.value for k, v in case.scores.items()} for case in cases]
        )
        average_labels: dict[str, dict[str, float]] = _labels_averages(
            [{k: v.value for k, v in case.labels.items()} for case in cases]
        )
        average_metrics: dict[str, float] = _scores_averages([case.metrics for case in cases])

        average_assertions: float | None = None
        n_assertions = sum(len(case.assertions) for case in cases)
        if n_assertions > 0:
            n_passing = sum(1 for case in cases for assertion in case.assertions.values() if assertion.value)
            average_assertions = n_passing / n_assertions

        return ReportCaseAggregate(
            name='Averages',
            scores=average_scores,
            labels=average_labels,
            metrics=average_metrics,
            assertions=average_assertions,
            task_duration=average_task_duration,
            total_duration=average_total_duration,
        )

```

#### average

```python
average(cases: list[ReportCase]) -> ReportCaseAggregate

```

Produce a synthetic "summary" case by averaging quantitative attributes.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
@staticmethod
def average(cases: list[ReportCase]) -> ReportCaseAggregate:
    """Produce a synthetic "summary" case by averaging quantitative attributes."""
    num_cases = len(cases)
    if num_cases == 0:
        return ReportCaseAggregate(
            name='Averages',
            scores={},
            labels={},
            metrics={},
            assertions=None,
            task_duration=0.0,
            total_duration=0.0,
        )

    def _scores_averages(scores_by_name: list[dict[str, int | float | bool]]) -> dict[str, float]:
        counts_by_name: dict[str, int] = defaultdict(int)
        sums_by_name: dict[str, float] = defaultdict(float)
        for sbn in scores_by_name:
            for name, score in sbn.items():
                counts_by_name[name] += 1
                sums_by_name[name] += score
        return {name: sums_by_name[name] / counts_by_name[name] for name in sums_by_name}

    def _labels_averages(labels_by_name: list[dict[str, str]]) -> dict[str, dict[str, float]]:
        counts_by_name: dict[str, int] = defaultdict(int)
        sums_by_name: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for lbn in labels_by_name:
            for name, label in lbn.items():
                counts_by_name[name] += 1
                sums_by_name[name][label] += 1
        return {
            name: {value: count / counts_by_name[name] for value, count in sums_by_name[name].items()}
            for name in sums_by_name
        }

    average_task_duration = sum(case.task_duration for case in cases) / num_cases
    average_total_duration = sum(case.total_duration for case in cases) / num_cases

    # average_assertions: dict[str, float] = _scores_averages([{k: v.value for k, v in case.scores.items()} for case in cases])
    average_scores: dict[str, float] = _scores_averages(
        [{k: v.value for k, v in case.scores.items()} for case in cases]
    )
    average_labels: dict[str, dict[str, float]] = _labels_averages(
        [{k: v.value for k, v in case.labels.items()} for case in cases]
    )
    average_metrics: dict[str, float] = _scores_averages([case.metrics for case in cases])

    average_assertions: float | None = None
    n_assertions = sum(len(case.assertions) for case in cases)
    if n_assertions > 0:
        n_passing = sum(1 for case in cases for assertion in case.assertions.values() if assertion.value)
        average_assertions = n_passing / n_assertions

    return ReportCaseAggregate(
        name='Averages',
        scores=average_scores,
        labels=average_labels,
        metrics=average_metrics,
        assertions=average_assertions,
        task_duration=average_task_duration,
        total_duration=average_total_duration,
    )

```

### EvaluationReport

Bases: `Generic[InputsT, OutputT, MetadataT]`

A report of the results of evaluating a model on a set of cases.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
@dataclass(kw_only=True)
class EvaluationReport(Generic[InputsT, OutputT, MetadataT]):
    """A report of the results of evaluating a model on a set of cases."""

    name: str
    """The name of the report."""

    cases: list[ReportCase[InputsT, OutputT, MetadataT]]
    """The cases in the report."""
    failures: list[ReportCaseFailure[InputsT, OutputT, MetadataT]] = field(default_factory=list)
    """The failures in the report. These are cases where task execution raised an exception."""

    experiment_metadata: dict[str, Any] | None = None
    """Metadata associated with the specific experiment represented by this report."""
    trace_id: str | None = None
    """The trace ID of the evaluation."""
    span_id: str | None = None
    """The span ID of the evaluation."""

    def averages(self) -> ReportCaseAggregate | None:
        if self.cases:
            return ReportCaseAggregate.average(self.cases)
        return None

    def render(
        self,
        width: int | None = None,
        baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None,
        *,
        include_input: bool = False,
        include_metadata: bool = False,
        include_expected_output: bool = False,
        include_output: bool = False,
        include_durations: bool = True,
        include_total_duration: bool = False,
        include_removed_cases: bool = False,
        include_averages: bool = True,
        include_errors: bool = True,
        include_error_stacktrace: bool = False,
        include_evaluator_failures: bool = True,
        input_config: RenderValueConfig | None = None,
        metadata_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
        include_reasons: bool = False,
    ) -> str:
        """Render this report to a nicely-formatted string, optionally comparing it to a baseline report.

        If you want more control over the output, use `console_table` instead and pass it to `rich.Console.print`.
        """
        io_file = StringIO()
        console = Console(width=width, file=io_file)
        self.print(
            width=width,
            baseline=baseline,
            console=console,
            include_input=include_input,
            include_metadata=include_metadata,
            include_expected_output=include_expected_output,
            include_output=include_output,
            include_durations=include_durations,
            include_total_duration=include_total_duration,
            include_removed_cases=include_removed_cases,
            include_averages=include_averages,
            include_errors=include_errors,
            include_error_stacktrace=include_error_stacktrace,
            include_evaluator_failures=include_evaluator_failures,
            input_config=input_config,
            metadata_config=metadata_config,
            output_config=output_config,
            score_configs=score_configs,
            label_configs=label_configs,
            metric_configs=metric_configs,
            duration_config=duration_config,
            include_reasons=include_reasons,
        )
        return io_file.getvalue()

    def print(
        self,
        width: int | None = None,
        baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None,
        *,
        console: Console | None = None,
        include_input: bool = False,
        include_metadata: bool = False,
        include_expected_output: bool = False,
        include_output: bool = False,
        include_durations: bool = True,
        include_total_duration: bool = False,
        include_removed_cases: bool = False,
        include_averages: bool = True,
        include_errors: bool = True,
        include_error_stacktrace: bool = False,
        include_evaluator_failures: bool = True,
        input_config: RenderValueConfig | None = None,
        metadata_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
        include_reasons: bool = False,
    ) -> None:
        """Print this report to the console, optionally comparing it to a baseline report.

        If you want more control over the output, use `console_table` instead and pass it to `rich.Console.print`.
        """
        if console is None:  # pragma: no branch
            console = Console(width=width)

        metadata_panel = self._metadata_panel(baseline=baseline)
        renderable: RenderableType = self.console_table(
            baseline=baseline,
            include_input=include_input,
            include_metadata=include_metadata,
            include_expected_output=include_expected_output,
            include_output=include_output,
            include_durations=include_durations,
            include_total_duration=include_total_duration,
            include_removed_cases=include_removed_cases,
            include_averages=include_averages,
            include_evaluator_failures=include_evaluator_failures,
            input_config=input_config,
            metadata_config=metadata_config,
            output_config=output_config,
            score_configs=score_configs,
            label_configs=label_configs,
            metric_configs=metric_configs,
            duration_config=duration_config,
            include_reasons=include_reasons,
            with_title=not metadata_panel,
        )
        # Wrap table with experiment metadata panel if present
        if metadata_panel:
            renderable = Group(metadata_panel, renderable)
        console.print(renderable)
        if include_errors and self.failures:  # pragma: no cover
            failures_table = self.failures_table(
                include_input=include_input,
                include_metadata=include_metadata,
                include_expected_output=include_expected_output,
                include_error_message=True,
                include_error_stacktrace=include_error_stacktrace,
                input_config=input_config,
                metadata_config=metadata_config,
            )
            console.print(failures_table, style='red')

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def console_table(
        self,
        baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None,
        *,
        include_input: bool = False,
        include_metadata: bool = False,
        include_expected_output: bool = False,
        include_output: bool = False,
        include_durations: bool = True,
        include_total_duration: bool = False,
        include_removed_cases: bool = False,
        include_averages: bool = True,
        include_evaluator_failures: bool = True,
        input_config: RenderValueConfig | None = None,
        metadata_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
        include_reasons: bool = False,
        with_title: bool = True,
    ) -> Table:
        """Return a table containing the data from this report.

        If a baseline is provided, returns a diff between this report and the baseline report.
        Optionally include input and output details.
        """
        renderer = EvaluationRenderer(
            include_input=include_input,
            include_metadata=include_metadata,
            include_expected_output=include_expected_output,
            include_output=include_output,
            include_durations=include_durations,
            include_total_duration=include_total_duration,
            include_removed_cases=include_removed_cases,
            include_averages=include_averages,
            include_error_message=False,
            include_error_stacktrace=False,
            include_evaluator_failures=include_evaluator_failures,
            input_config={**_DEFAULT_VALUE_CONFIG, **(input_config or {})},
            metadata_config={**_DEFAULT_VALUE_CONFIG, **(metadata_config or {})},
            output_config=output_config or _DEFAULT_VALUE_CONFIG,
            score_configs=score_configs or {},
            label_configs=label_configs or {},
            metric_configs=metric_configs or {},
            duration_config=duration_config or _DEFAULT_DURATION_CONFIG,
            include_reasons=include_reasons,
        )
        if baseline is None:
            return renderer.build_table(self, with_title=with_title)
        else:
            return renderer.build_diff_table(self, baseline, with_title=with_title)

    def _metadata_panel(
        self, baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None
    ) -> RenderableType | None:
        """Wrap a table with an experiment metadata panel if metadata exists.

        Args:
            table: The table to wrap
            baseline: Optional baseline report for diff metadata

        Returns:
            Either the table unchanged or a Group with Panel and Table
        """
        if baseline is None:
            # Single report - show metadata if present
            if self.experiment_metadata:
                metadata_text = Text()
                items = list(self.experiment_metadata.items())
                for i, (key, value) in enumerate(items):
                    metadata_text.append(f'{key}: {value}', style='dim')
                    if i < len(items) - 1:
                        metadata_text.append('\n')
                return Panel(
                    metadata_text,
                    title=f'Evaluation Summary: {self.name}',
                    title_align='left',
                    border_style='dim',
                    padding=(0, 1),
                    expand=False,
                )
        else:
            # Diff report - show metadata diff if either has metadata
            if self.experiment_metadata or baseline.experiment_metadata:
                diff_name = baseline.name if baseline.name == self.name else f'{baseline.name} â†’ {self.name}'
                metadata_text = Text()
                lines_styles: list[tuple[str, str]] = []
                if baseline.experiment_metadata and self.experiment_metadata:
                    # Collect all keys from both
                    all_keys = sorted(set(baseline.experiment_metadata.keys()) | set(self.experiment_metadata.keys()))
                    for key in all_keys:
                        baseline_val = baseline.experiment_metadata.get(key)
                        report_val = self.experiment_metadata.get(key)
                        if baseline_val == report_val:
                            lines_styles.append((f'{key}: {report_val}', 'dim'))
                        elif baseline_val is None:
                            lines_styles.append((f'+ {key}: {report_val}', 'green'))
                        elif report_val is None:
                            lines_styles.append((f'- {key}: {baseline_val}', 'red'))
                        else:
                            lines_styles.append((f'{key}: {baseline_val} â†’ {report_val}', 'yellow'))
                elif self.experiment_metadata:
                    lines_styles = [(f'+ {k}: {v}', 'green') for k, v in self.experiment_metadata.items()]
                else:  # baseline.experiment_metadata only
                    assert baseline.experiment_metadata is not None
                    lines_styles = [(f'- {k}: {v}', 'red') for k, v in baseline.experiment_metadata.items()]

                for i, (line, style) in enumerate(lines_styles):
                    metadata_text.append(line, style=style)
                    if i < len(lines_styles) - 1:
                        metadata_text.append('\n')

                return Panel(
                    metadata_text,
                    title=f'Evaluation Diff: {diff_name}',
                    title_align='left',
                    border_style='dim',
                    padding=(0, 1),
                    expand=False,
                )

        return None

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def failures_table(
        self,
        *,
        include_input: bool = False,
        include_metadata: bool = False,
        include_expected_output: bool = False,
        include_error_message: bool = True,
        include_error_stacktrace: bool = True,
        input_config: RenderValueConfig | None = None,
        metadata_config: RenderValueConfig | None = None,
    ) -> Table:
        """Return a table containing the failures in this report."""
        renderer = EvaluationRenderer(
            include_input=include_input,
            include_metadata=include_metadata,
            include_expected_output=include_expected_output,
            include_output=False,
            include_durations=False,
            include_total_duration=False,
            include_removed_cases=False,
            include_averages=False,
            input_config={**_DEFAULT_VALUE_CONFIG, **(input_config or {})},
            metadata_config={**_DEFAULT_VALUE_CONFIG, **(metadata_config or {})},
            output_config=_DEFAULT_VALUE_CONFIG,
            score_configs={},
            label_configs={},
            metric_configs={},
            duration_config=_DEFAULT_DURATION_CONFIG,
            include_reasons=False,
            include_error_message=include_error_message,
            include_error_stacktrace=include_error_stacktrace,
            include_evaluator_failures=False,  # Not applicable for failures table
        )
        return renderer.build_failures_table(self)

    def __str__(self) -> str:  # pragma: lax no cover
        """Return a string representation of the report."""
        return self.render()

```

#### name

```python
name: str

```

The name of the report.

#### cases

```python
cases: list[ReportCase[InputsT, OutputT, MetadataT]]

```

The cases in the report.

#### failures

```python
failures: list[
    ReportCaseFailure[InputsT, OutputT, MetadataT]
] = field(default_factory=list)

```

The failures in the report. These are cases where task execution raised an exception.

#### experiment_metadata

```python
experiment_metadata: dict[str, Any] | None = None

```

Metadata associated with the specific experiment represented by this report.

#### trace_id

```python
trace_id: str | None = None

```

The trace ID of the evaluation.

#### span_id

```python
span_id: str | None = None

```

The span ID of the evaluation.

#### render

```python
render(
    width: int | None = None,
    baseline: (
        EvaluationReport[InputsT, OutputT, MetadataT] | None
    ) = None,
    *,
    include_input: bool = False,
    include_metadata: bool = False,
    include_expected_output: bool = False,
    include_output: bool = False,
    include_durations: bool = True,
    include_total_duration: bool = False,
    include_removed_cases: bool = False,
    include_averages: bool = True,
    include_errors: bool = True,
    include_error_stacktrace: bool = False,
    include_evaluator_failures: bool = True,
    input_config: RenderValueConfig | None = None,
    metadata_config: RenderValueConfig | None = None,
    output_config: RenderValueConfig | None = None,
    score_configs: (
        dict[str, RenderNumberConfig] | None
    ) = None,
    label_configs: (
        dict[str, RenderValueConfig] | None
    ) = None,
    metric_configs: (
        dict[str, RenderNumberConfig] | None
    ) = None,
    duration_config: RenderNumberConfig | None = None,
    include_reasons: bool = False
) -> str

```

Render this report to a nicely-formatted string, optionally comparing it to a baseline report.

If you want more control over the output, use `console_table` instead and pass it to `rich.Console.print`.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
def render(
    self,
    width: int | None = None,
    baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None,
    *,
    include_input: bool = False,
    include_metadata: bool = False,
    include_expected_output: bool = False,
    include_output: bool = False,
    include_durations: bool = True,
    include_total_duration: bool = False,
    include_removed_cases: bool = False,
    include_averages: bool = True,
    include_errors: bool = True,
    include_error_stacktrace: bool = False,
    include_evaluator_failures: bool = True,
    input_config: RenderValueConfig | None = None,
    metadata_config: RenderValueConfig | None = None,
    output_config: RenderValueConfig | None = None,
    score_configs: dict[str, RenderNumberConfig] | None = None,
    label_configs: dict[str, RenderValueConfig] | None = None,
    metric_configs: dict[str, RenderNumberConfig] | None = None,
    duration_config: RenderNumberConfig | None = None,
    include_reasons: bool = False,
) -> str:
    """Render this report to a nicely-formatted string, optionally comparing it to a baseline report.

    If you want more control over the output, use `console_table` instead and pass it to `rich.Console.print`.
    """
    io_file = StringIO()
    console = Console(width=width, file=io_file)
    self.print(
        width=width,
        baseline=baseline,
        console=console,
        include_input=include_input,
        include_metadata=include_metadata,
        include_expected_output=include_expected_output,
        include_output=include_output,
        include_durations=include_durations,
        include_total_duration=include_total_duration,
        include_removed_cases=include_removed_cases,
        include_averages=include_averages,
        include_errors=include_errors,
        include_error_stacktrace=include_error_stacktrace,
        include_evaluator_failures=include_evaluator_failures,
        input_config=input_config,
        metadata_config=metadata_config,
        output_config=output_config,
        score_configs=score_configs,
        label_configs=label_configs,
        metric_configs=metric_configs,
        duration_config=duration_config,
        include_reasons=include_reasons,
    )
    return io_file.getvalue()

```

#### print

```python
print(
    width: int | None = None,
    baseline: (
        EvaluationReport[InputsT, OutputT, MetadataT] | None
    ) = None,
    *,
    console: Console | None = None,
    include_input: bool = False,
    include_metadata: bool = False,
    include_expected_output: bool = False,
    include_output: bool = False,
    include_durations: bool = True,
    include_total_duration: bool = False,
    include_removed_cases: bool = False,
    include_averages: bool = True,
    include_errors: bool = True,
    include_error_stacktrace: bool = False,
    include_evaluator_failures: bool = True,
    input_config: RenderValueConfig | None = None,
    metadata_config: RenderValueConfig | None = None,
    output_config: RenderValueConfig | None = None,
    score_configs: (
        dict[str, RenderNumberConfig] | None
    ) = None,
    label_configs: (
        dict[str, RenderValueConfig] | None
    ) = None,
    metric_configs: (
        dict[str, RenderNumberConfig] | None
    ) = None,
    duration_config: RenderNumberConfig | None = None,
    include_reasons: bool = False
) -> None

```

Print this report to the console, optionally comparing it to a baseline report.

If you want more control over the output, use `console_table` instead and pass it to `rich.Console.print`.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
def print(
    self,
    width: int | None = None,
    baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None,
    *,
    console: Console | None = None,
    include_input: bool = False,
    include_metadata: bool = False,
    include_expected_output: bool = False,
    include_output: bool = False,
    include_durations: bool = True,
    include_total_duration: bool = False,
    include_removed_cases: bool = False,
    include_averages: bool = True,
    include_errors: bool = True,
    include_error_stacktrace: bool = False,
    include_evaluator_failures: bool = True,
    input_config: RenderValueConfig | None = None,
    metadata_config: RenderValueConfig | None = None,
    output_config: RenderValueConfig | None = None,
    score_configs: dict[str, RenderNumberConfig] | None = None,
    label_configs: dict[str, RenderValueConfig] | None = None,
    metric_configs: dict[str, RenderNumberConfig] | None = None,
    duration_config: RenderNumberConfig | None = None,
    include_reasons: bool = False,
) -> None:
    """Print this report to the console, optionally comparing it to a baseline report.

    If you want more control over the output, use `console_table` instead and pass it to `rich.Console.print`.
    """
    if console is None:  # pragma: no branch
        console = Console(width=width)

    metadata_panel = self._metadata_panel(baseline=baseline)
    renderable: RenderableType = self.console_table(
        baseline=baseline,
        include_input=include_input,
        include_metadata=include_metadata,
        include_expected_output=include_expected_output,
        include_output=include_output,
        include_durations=include_durations,
        include_total_duration=include_total_duration,
        include_removed_cases=include_removed_cases,
        include_averages=include_averages,
        include_evaluator_failures=include_evaluator_failures,
        input_config=input_config,
        metadata_config=metadata_config,
        output_config=output_config,
        score_configs=score_configs,
        label_configs=label_configs,
        metric_configs=metric_configs,
        duration_config=duration_config,
        include_reasons=include_reasons,
        with_title=not metadata_panel,
    )
    # Wrap table with experiment metadata panel if present
    if metadata_panel:
        renderable = Group(metadata_panel, renderable)
    console.print(renderable)
    if include_errors and self.failures:  # pragma: no cover
        failures_table = self.failures_table(
            include_input=include_input,
            include_metadata=include_metadata,
            include_expected_output=include_expected_output,
            include_error_message=True,
            include_error_stacktrace=include_error_stacktrace,
            input_config=input_config,
            metadata_config=metadata_config,
        )
        console.print(failures_table, style='red')

```

#### console_table

```python
console_table(
    baseline: (
        EvaluationReport[InputsT, OutputT, MetadataT] | None
    ) = None,
    *,
    include_input: bool = False,
    include_metadata: bool = False,
    include_expected_output: bool = False,
    include_output: bool = False,
    include_durations: bool = True,
    include_total_duration: bool = False,
    include_removed_cases: bool = False,
    include_averages: bool = True,
    include_evaluator_failures: bool = True,
    input_config: RenderValueConfig | None = None,
    metadata_config: RenderValueConfig | None = None,
    output_config: RenderValueConfig | None = None,
    score_configs: (
        dict[str, RenderNumberConfig] | None
    ) = None,
    label_configs: (
        dict[str, RenderValueConfig] | None
    ) = None,
    metric_configs: (
        dict[str, RenderNumberConfig] | None
    ) = None,
    duration_config: RenderNumberConfig | None = None,
    include_reasons: bool = False,
    with_title: bool = True
) -> Table

```

Return a table containing the data from this report.

If a baseline is provided, returns a diff between this report and the baseline report. Optionally include input and output details.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
def console_table(
    self,
    baseline: EvaluationReport[InputsT, OutputT, MetadataT] | None = None,
    *,
    include_input: bool = False,
    include_metadata: bool = False,
    include_expected_output: bool = False,
    include_output: bool = False,
    include_durations: bool = True,
    include_total_duration: bool = False,
    include_removed_cases: bool = False,
    include_averages: bool = True,
    include_evaluator_failures: bool = True,
    input_config: RenderValueConfig | None = None,
    metadata_config: RenderValueConfig | None = None,
    output_config: RenderValueConfig | None = None,
    score_configs: dict[str, RenderNumberConfig] | None = None,
    label_configs: dict[str, RenderValueConfig] | None = None,
    metric_configs: dict[str, RenderNumberConfig] | None = None,
    duration_config: RenderNumberConfig | None = None,
    include_reasons: bool = False,
    with_title: bool = True,
) -> Table:
    """Return a table containing the data from this report.

    If a baseline is provided, returns a diff between this report and the baseline report.
    Optionally include input and output details.
    """
    renderer = EvaluationRenderer(
        include_input=include_input,
        include_metadata=include_metadata,
        include_expected_output=include_expected_output,
        include_output=include_output,
        include_durations=include_durations,
        include_total_duration=include_total_duration,
        include_removed_cases=include_removed_cases,
        include_averages=include_averages,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=include_evaluator_failures,
        input_config={**_DEFAULT_VALUE_CONFIG, **(input_config or {})},
        metadata_config={**_DEFAULT_VALUE_CONFIG, **(metadata_config or {})},
        output_config=output_config or _DEFAULT_VALUE_CONFIG,
        score_configs=score_configs or {},
        label_configs=label_configs or {},
        metric_configs=metric_configs or {},
        duration_config=duration_config or _DEFAULT_DURATION_CONFIG,
        include_reasons=include_reasons,
    )
    if baseline is None:
        return renderer.build_table(self, with_title=with_title)
    else:
        return renderer.build_diff_table(self, baseline, with_title=with_title)

```

#### failures_table

```python
failures_table(
    *,
    include_input: bool = False,
    include_metadata: bool = False,
    include_expected_output: bool = False,
    include_error_message: bool = True,
    include_error_stacktrace: bool = True,
    input_config: RenderValueConfig | None = None,
    metadata_config: RenderValueConfig | None = None
) -> Table

```

Return a table containing the failures in this report.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
def failures_table(
    self,
    *,
    include_input: bool = False,
    include_metadata: bool = False,
    include_expected_output: bool = False,
    include_error_message: bool = True,
    include_error_stacktrace: bool = True,
    input_config: RenderValueConfig | None = None,
    metadata_config: RenderValueConfig | None = None,
) -> Table:
    """Return a table containing the failures in this report."""
    renderer = EvaluationRenderer(
        include_input=include_input,
        include_metadata=include_metadata,
        include_expected_output=include_expected_output,
        include_output=False,
        include_durations=False,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=False,
        input_config={**_DEFAULT_VALUE_CONFIG, **(input_config or {})},
        metadata_config={**_DEFAULT_VALUE_CONFIG, **(metadata_config or {})},
        output_config=_DEFAULT_VALUE_CONFIG,
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config=_DEFAULT_DURATION_CONFIG,
        include_reasons=False,
        include_error_message=include_error_message,
        include_error_stacktrace=include_error_stacktrace,
        include_evaluator_failures=False,  # Not applicable for failures table
    )
    return renderer.build_failures_table(self)

```

#### __str__

```python
__str__() -> str

```

Return a string representation of the report.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
def __str__(self) -> str:  # pragma: lax no cover
    """Return a string representation of the report."""
    return self.render()

```

### RenderValueConfig

Bases: `TypedDict`

A configuration for rendering a values in an Evaluation report.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
class RenderValueConfig(TypedDict, total=False):
    """A configuration for rendering a values in an Evaluation report."""

    value_formatter: str | Callable[[Any], str]
    diff_checker: Callable[[Any, Any], bool] | None
    diff_formatter: Callable[[Any, Any], str | None] | None
    diff_style: str

```

### RenderNumberConfig

Bases: `TypedDict`

A configuration for rendering a particular score or metric in an Evaluation report.

See the implementation of `_RenderNumber` for more clarity on how these parameters affect the rendering.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
class RenderNumberConfig(TypedDict, total=False):
    """A configuration for rendering a particular score or metric in an Evaluation report.

    See the implementation of `_RenderNumber` for more clarity on how these parameters affect the rendering.
    """

    value_formatter: str | Callable[[float | int], str]
    """The logic to use for formatting values.

    * If not provided, format as ints if all values are ints, otherwise at least one decimal place and at least four significant figures.
    * You can also use a custom string format spec, e.g. '{:.3f}'
    * You can also use a custom function, e.g. lambda x: f'{x:.3f}'
    """
    diff_formatter: str | Callable[[float | int, float | int], str | None] | None
    """The logic to use for formatting details about the diff.

    The strings produced by the value_formatter will always be included in the reports, but the diff_formatter is
    used to produce additional text about the difference between the old and new values, such as the absolute or
    relative difference.

    * If not provided, format as ints if all values are ints, otherwise at least one decimal place and at least four
        significant figures, and will include the percentage change.
    * You can also use a custom string format spec, e.g. '{:+.3f}'
    * You can also use a custom function, e.g. lambda x: f'{x:+.3f}'.
        If this function returns None, no extra diff text will be added.
    * You can also use None to never generate extra diff text.
    """
    diff_atol: float
    """The absolute tolerance for considering a difference "significant".

    A difference is "significant" if `abs(new - old) < self.diff_atol + self.diff_rtol * abs(old)`.

    If a difference is not significant, it will not have the diff styles applied. Note that we still show
    both the rendered before and after values in the diff any time they differ, even if the difference is not
    significant. (If the rendered values are exactly the same, we only show the value once.)

    If not provided, use 1e-6.
    """
    diff_rtol: float
    """The relative tolerance for considering a difference "significant".

    See the description of `diff_atol` for more details about what makes a difference "significant".

    If not provided, use 0.001 if all values are ints, otherwise 0.05.
    """
    diff_increase_style: str
    """The style to apply to diffed values that have a significant increase.

    See the description of `diff_atol` for more details about what makes a difference "significant".

    If not provided, use green for scores and red for metrics. You can also use arbitrary `rich` styles, such as "bold red".
    """
    diff_decrease_style: str
    """The style to apply to diffed values that have significant decrease.

    See the description of `diff_atol` for more details about what makes a difference "significant".

    If not provided, use red for scores and green for metrics. You can also use arbitrary `rich` styles, such as "bold red".
    """

```

#### value_formatter

```python
value_formatter: str | Callable[[float | int], str]

```

The logic to use for formatting values.

- If not provided, format as ints if all values are ints, otherwise at least one decimal place and at least four significant figures.
- You can also use a custom string format spec, e.g. '{:.3f}'
- You can also use a custom function, e.g. lambda x: f'{x:.3f}'

#### diff_formatter

```python
diff_formatter: (
    str
    | Callable[[float | int, float | int], str | None]
    | None
)

```

The logic to use for formatting details about the diff.

The strings produced by the value_formatter will always be included in the reports, but the diff_formatter is used to produce additional text about the difference between the old and new values, such as the absolute or relative difference.

- If not provided, format as ints if all values are ints, otherwise at least one decimal place and at least four significant figures, and will include the percentage change.
- You can also use a custom string format spec, e.g. '{:+.3f}'
- You can also use a custom function, e.g. lambda x: f'{x:+.3f}'. If this function returns None, no extra diff text will be added.
- You can also use None to never generate extra diff text.

#### diff_atol

```python
diff_atol: float

```

The absolute tolerance for considering a difference "significant".

A difference is "significant" if `abs(new - old) < self.diff_atol + self.diff_rtol * abs(old)`.

If a difference is not significant, it will not have the diff styles applied. Note that we still show both the rendered before and after values in the diff any time they differ, even if the difference is not significant. (If the rendered values are exactly the same, we only show the value once.)

If not provided, use 1e-6.

#### diff_rtol

```python
diff_rtol: float

```

The relative tolerance for considering a difference "significant".

See the description of `diff_atol` for more details about what makes a difference "significant".

If not provided, use 0.001 if all values are ints, otherwise 0.05.

#### diff_increase_style

```python
diff_increase_style: str

```

The style to apply to diffed values that have a significant increase.

See the description of `diff_atol` for more details about what makes a difference "significant".

If not provided, use green for scores and red for metrics. You can also use arbitrary `rich` styles, such as "bold red".

#### diff_decrease_style

```python
diff_decrease_style: str

```

The style to apply to diffed values that have significant decrease.

See the description of `diff_atol` for more details about what makes a difference "significant".

If not provided, use red for scores and green for metrics. You can also use arbitrary `rich` styles, such as "bold red".

### EvaluationRenderer

A class for rendering an EvalReport or the diff between two EvalReports.

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
@dataclass(kw_only=True)
class EvaluationRenderer:
    """A class for rendering an EvalReport or the diff between two EvalReports."""

    # Columns to include
    include_input: bool
    include_metadata: bool
    include_expected_output: bool
    include_output: bool
    include_durations: bool
    include_total_duration: bool

    # Rows to include
    include_removed_cases: bool
    include_averages: bool

    input_config: RenderValueConfig
    metadata_config: RenderValueConfig
    output_config: RenderValueConfig
    score_configs: dict[str, RenderNumberConfig]
    label_configs: dict[str, RenderValueConfig]
    metric_configs: dict[str, RenderNumberConfig]
    duration_config: RenderNumberConfig

    # Data to include
    include_reasons: bool  # only applies to reports, not to diffs

    include_error_message: bool
    include_error_stacktrace: bool
    include_evaluator_failures: bool

    def include_scores(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return any(case.scores for case in self._all_cases(report, baseline))

    def include_labels(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return any(case.labels for case in self._all_cases(report, baseline))

    def include_metrics(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return any(case.metrics for case in self._all_cases(report, baseline))

    def include_assertions(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return any(case.assertions for case in self._all_cases(report, baseline))

    def include_evaluator_failures_column(self, report: EvaluationReport, baseline: EvaluationReport | None = None):
        return self.include_evaluator_failures and any(
            case.evaluator_failures for case in self._all_cases(report, baseline)
        )

    def _all_cases(self, report: EvaluationReport, baseline: EvaluationReport | None) -> list[ReportCase]:
        if not baseline:
            return report.cases
        else:
            return report.cases + self._baseline_cases_to_include(report, baseline)

    def _baseline_cases_to_include(self, report: EvaluationReport, baseline: EvaluationReport) -> list[ReportCase]:
        if self.include_removed_cases:
            return baseline.cases
        report_case_names = {case.name for case in report.cases}
        return [case for case in baseline.cases if case.name in report_case_names]

    def _get_case_renderer(
        self, report: EvaluationReport, baseline: EvaluationReport | None = None
    ) -> ReportCaseRenderer:
        input_renderer = _ValueRenderer.from_config(self.input_config)
        metadata_renderer = _ValueRenderer.from_config(self.metadata_config)
        output_renderer = _ValueRenderer.from_config(self.output_config)
        score_renderers = self._infer_score_renderers(report, baseline)
        label_renderers = self._infer_label_renderers(report, baseline)
        metric_renderers = self._infer_metric_renderers(report, baseline)
        duration_renderer = _NumberRenderer.infer_from_config(
            self.duration_config, 'duration', [x.task_duration for x in self._all_cases(report, baseline)]
        )

        return ReportCaseRenderer(
            include_input=self.include_input,
            include_metadata=self.include_metadata,
            include_expected_output=self.include_expected_output,
            include_output=self.include_output,
            include_scores=self.include_scores(report, baseline),
            include_labels=self.include_labels(report, baseline),
            include_metrics=self.include_metrics(report, baseline),
            include_assertions=self.include_assertions(report, baseline),
            include_reasons=self.include_reasons,
            include_durations=self.include_durations,
            include_total_duration=self.include_total_duration,
            include_error_message=self.include_error_message,
            include_error_stacktrace=self.include_error_stacktrace,
            include_evaluator_failures=self.include_evaluator_failures_column(report, baseline),
            input_renderer=input_renderer,
            metadata_renderer=metadata_renderer,
            output_renderer=output_renderer,
            score_renderers=score_renderers,
            label_renderers=label_renderers,
            metric_renderers=metric_renderers,
            duration_renderer=duration_renderer,
        )

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def build_table(self, report: EvaluationReport, *, with_title: bool = True) -> Table:
        """Build a table for the report.

        Args:
            report: The evaluation report to render
            with_title: Whether to include the title in the table (default True)

        Returns:
            A Rich Table object
        """
        case_renderer = self._get_case_renderer(report)

        title = f'Evaluation Summary: {report.name}' if with_title else ''
        table = case_renderer.build_base_table(title)

        for case in report.cases:
            table.add_row(*case_renderer.build_row(case))

        if self.include_averages:  # pragma: no branch
            average = report.averages()
            if average:  # pragma: no branch
                table.add_row(*case_renderer.build_aggregate_row(average))

        return table

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def build_diff_table(
        self, report: EvaluationReport, baseline: EvaluationReport, *, with_title: bool = True
    ) -> Table:
        """Build a diff table comparing report to baseline.

        Args:
            report: The evaluation report to compare
            baseline: The baseline report to compare against
            with_title: Whether to include the title in the table (default True)

        Returns:
            A Rich Table object
        """
        report_cases = report.cases
        baseline_cases = self._baseline_cases_to_include(report, baseline)

        report_cases_by_id = {case.name: case for case in report_cases}
        baseline_cases_by_id = {case.name: case for case in baseline_cases}

        diff_cases: list[tuple[ReportCase, ReportCase]] = []
        removed_cases: list[ReportCase] = []
        added_cases: list[ReportCase] = []

        for case_id in sorted(set(baseline_cases_by_id.keys()) | set(report_cases_by_id.keys())):
            maybe_baseline_case = baseline_cases_by_id.get(case_id)
            maybe_report_case = report_cases_by_id.get(case_id)
            if maybe_baseline_case and maybe_report_case:
                diff_cases.append((maybe_baseline_case, maybe_report_case))
            elif maybe_baseline_case:
                removed_cases.append(maybe_baseline_case)
            elif maybe_report_case:
                added_cases.append(maybe_report_case)
            else:  # pragma: no cover
                assert False, 'This should be unreachable'

        case_renderer = self._get_case_renderer(report, baseline)
        diff_name = baseline.name if baseline.name == report.name else f'{baseline.name} â†’ {report.name}'

        title = f'Evaluation Diff: {diff_name}' if with_title else ''
        table = case_renderer.build_base_table(title)

        for baseline_case, new_case in diff_cases:
            table.add_row(*case_renderer.build_diff_row(new_case, baseline_case))
        for case in added_cases:
            row = case_renderer.build_row(case)
            row[0] = f'[green]+ Added Case[/]\n{row[0]}'
            table.add_row(*row)
        for case in removed_cases:
            row = case_renderer.build_row(case)
            row[0] = f'[red]- Removed Case[/]\n{row[0]}'
            table.add_row(*row)

        if self.include_averages:  # pragma: no branch
            report_average = ReportCaseAggregate.average(report_cases)
            baseline_average = ReportCaseAggregate.average(baseline_cases)
            table.add_row(*case_renderer.build_diff_aggregate_row(report_average, baseline_average))

        return table

    # TODO(DavidM): in v2, change the return type here to RenderableType
    def build_failures_table(self, report: EvaluationReport) -> Table:
        case_renderer = self._get_case_renderer(report)
        table = case_renderer.build_failures_table('Case Failures')
        for case in report.failures:
            table.add_row(*case_renderer.build_failure_row(case))

        return table

    def _infer_score_renderers(
        self, report: EvaluationReport, baseline: EvaluationReport | None
    ) -> dict[str, _NumberRenderer]:
        all_cases = self._all_cases(report, baseline)

        values_by_name: dict[str, list[float | int]] = {}
        for case in all_cases:
            for k, score in case.scores.items():
                values_by_name.setdefault(k, []).append(score.value)

        all_renderers: dict[str, _NumberRenderer] = {}
        for name, values in values_by_name.items():
            merged_config = _DEFAULT_NUMBER_CONFIG.copy()
            merged_config.update(self.score_configs.get(name, {}))
            all_renderers[name] = _NumberRenderer.infer_from_config(merged_config, 'score', values)
        return all_renderers

    def _infer_label_renderers(
        self, report: EvaluationReport, baseline: EvaluationReport | None
    ) -> dict[str, _ValueRenderer]:
        all_cases = self._all_cases(report, baseline)
        all_names: set[str] = set()
        for case in all_cases:
            for k in case.labels:
                all_names.add(k)

        all_renderers: dict[str, _ValueRenderer] = {}
        for name in all_names:
            merged_config = _DEFAULT_VALUE_CONFIG.copy()
            merged_config.update(self.label_configs.get(name, {}))
            all_renderers[name] = _ValueRenderer.from_config(merged_config)
        return all_renderers

    def _infer_metric_renderers(
        self, report: EvaluationReport, baseline: EvaluationReport | None
    ) -> dict[str, _NumberRenderer]:
        all_cases = self._all_cases(report, baseline)

        values_by_name: dict[str, list[float | int]] = {}
        for case in all_cases:
            for k, v in case.metrics.items():
                values_by_name.setdefault(k, []).append(v)

        all_renderers: dict[str, _NumberRenderer] = {}
        for name, values in values_by_name.items():
            merged_config = _DEFAULT_NUMBER_CONFIG.copy()
            merged_config.update(self.metric_configs.get(name, {}))
            all_renderers[name] = _NumberRenderer.infer_from_config(merged_config, 'metric', values)
        return all_renderers

    def _infer_duration_renderer(
        self, report: EvaluationReport, baseline: EvaluationReport | None
    ) -> _NumberRenderer:  # pragma: no cover
        all_cases = self._all_cases(report, baseline)
        all_durations = [x.task_duration for x in all_cases]
        if self.include_total_duration:
            all_durations += [x.total_duration for x in all_cases]
        return _NumberRenderer.infer_from_config(self.duration_config, 'duration', all_durations)

```

#### build_table

```python
build_table(
    report: EvaluationReport, *, with_title: bool = True
) -> Table

```

Build a table for the report.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `report` | `EvaluationReport` | The evaluation report to render | *required* | | `with_title` | `bool` | Whether to include the title in the table (default True) | `True` |

Returns:

| Type | Description | | --- | --- | | `Table` | A Rich Table object |

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
def build_table(self, report: EvaluationReport, *, with_title: bool = True) -> Table:
    """Build a table for the report.

    Args:
        report: The evaluation report to render
        with_title: Whether to include the title in the table (default True)

    Returns:
        A Rich Table object
    """
    case_renderer = self._get_case_renderer(report)

    title = f'Evaluation Summary: {report.name}' if with_title else ''
    table = case_renderer.build_base_table(title)

    for case in report.cases:
        table.add_row(*case_renderer.build_row(case))

    if self.include_averages:  # pragma: no branch
        average = report.averages()
        if average:  # pragma: no branch
            table.add_row(*case_renderer.build_aggregate_row(average))

    return table

```

#### build_diff_table

```python
build_diff_table(
    report: EvaluationReport,
    baseline: EvaluationReport,
    *,
    with_title: bool = True
) -> Table

```

Build a diff table comparing report to baseline.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `report` | `EvaluationReport` | The evaluation report to compare | *required* | | `baseline` | `EvaluationReport` | The baseline report to compare against | *required* | | `with_title` | `bool` | Whether to include the title in the table (default True) | `True` |

Returns:

| Type | Description | | --- | --- | | `Table` | A Rich Table object |

Source code in `pydantic_evals/pydantic_evals/reporting/__init__.py`

```python
def build_diff_table(
    self, report: EvaluationReport, baseline: EvaluationReport, *, with_title: bool = True
) -> Table:
    """Build a diff table comparing report to baseline.

    Args:
        report: The evaluation report to compare
        baseline: The baseline report to compare against
        with_title: Whether to include the title in the table (default True)

    Returns:
        A Rich Table object
    """
    report_cases = report.cases
    baseline_cases = self._baseline_cases_to_include(report, baseline)

    report_cases_by_id = {case.name: case for case in report_cases}
    baseline_cases_by_id = {case.name: case for case in baseline_cases}

    diff_cases: list[tuple[ReportCase, ReportCase]] = []
    removed_cases: list[ReportCase] = []
    added_cases: list[ReportCase] = []

    for case_id in sorted(set(baseline_cases_by_id.keys()) | set(report_cases_by_id.keys())):
        maybe_baseline_case = baseline_cases_by_id.get(case_id)
        maybe_report_case = report_cases_by_id.get(case_id)
        if maybe_baseline_case and maybe_report_case:
            diff_cases.append((maybe_baseline_case, maybe_report_case))
        elif maybe_baseline_case:
            removed_cases.append(maybe_baseline_case)
        elif maybe_report_case:
            added_cases.append(maybe_report_case)
        else:  # pragma: no cover
            assert False, 'This should be unreachable'

    case_renderer = self._get_case_renderer(report, baseline)
    diff_name = baseline.name if baseline.name == report.name else f'{baseline.name} â†’ {report.name}'

    title = f'Evaluation Diff: {diff_name}' if with_title else ''
    table = case_renderer.build_base_table(title)

    for baseline_case, new_case in diff_cases:
        table.add_row(*case_renderer.build_diff_row(new_case, baseline_case))
    for case in added_cases:
        row = case_renderer.build_row(case)
        row[0] = f'[green]+ Added Case[/]\n{row[0]}'
        table.add_row(*row)
    for case in removed_cases:
        row = case_renderer.build_row(case)
        row[0] = f'[red]- Removed Case[/]\n{row[0]}'
        table.add_row(*row)

    if self.include_averages:  # pragma: no branch
        report_average = ReportCaseAggregate.average(report_cases)
        baseline_average = ReportCaseAggregate.average(baseline_cases)
        table.add_row(*case_renderer.build_diff_aggregate_row(report_average, baseline_average))

    return table

```

# `pydantic_graph.beta`

The next version of the pydantic-graph framework with enhanced graph execution capabilities.

This module provides a parallel control flow graph execution framework with support for:

- 'Step' nodes for task execution
- 'Decision' nodes for conditional branching
- 'Fork' nodes for parallel execution coordination
- 'Join' nodes and 'Reducer's for re-joining parallel executions
- Mermaid diagram generation for graph visualization

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

### TypeExpression

Bases: `Generic[T]`

A workaround for type checker limitations when using complex type expressions.

```text
This class serves as a wrapper for types that cannot normally be used in positions

```

requiring `type[T]`, such as `Any`, `Union[...]`, or `Literal[...]`. It provides a way to pass these complex type expressions to functions expecting concrete types.

Example

Instead of `output_type=Union[str, int]` (which may cause type errors), use `output_type=TypeExpression[Union[str, int]]`.

Note

This is a workaround for the lack of TypeForm in the Python type system.

Source code in `pydantic_graph/pydantic_graph/beta/util.py`

```python
class TypeExpression(Generic[T]):
    """A workaround for type checker limitations when using complex type expressions.

        This class serves as a wrapper for types that cannot normally be used in positions
    requiring `type[T]`, such as `Any`, `Union[...]`, or `Literal[...]`. It provides a
        way to pass these complex type expressions to functions expecting concrete types.

    Example:
            Instead of `output_type=Union[str, int]` (which may cause type errors),
            use `output_type=TypeExpression[Union[str, int]]`.

    Note:
            This is a workaround for the lack of TypeForm in the Python type system.
    """

    pass

```

