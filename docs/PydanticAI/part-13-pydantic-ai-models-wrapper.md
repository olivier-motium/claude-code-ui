<!-- Source: _complete-doc.md -->

# pydantic_ai.models.wrapper

### WrapperModel

Bases: `Model`

Model which wraps another model.

Does nothing on its own, used as a base class.

Source code in `pydantic_ai_slim/pydantic_ai/models/wrapper.py`

```python
@dataclass(init=False)
class WrapperModel(Model):
    """Model which wraps another model.

    Does nothing on its own, used as a base class.
    """

    wrapped: Model
    """The underlying model being wrapped."""

    def __init__(self, wrapped: Model | KnownModelName):
        super().__init__()
        self.wrapped = infer_model(wrapped)

    async def request(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return await self.wrapped.request(*args, **kwargs)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response_stream:
            yield response_stream

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return self.wrapped.customize_request_parameters(model_request_parameters)  # pragma: no cover

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return self.wrapped.prepare_request(model_settings, model_request_parameters)

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def system(self) -> str:
        return self.wrapped.system

    @cached_property
    def profile(self) -> ModelProfile:
        return self.wrapped.profile

    @property
    def settings(self) -> ModelSettings | None:
        """Get the settings from the wrapped model."""
        return self.wrapped.settings

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)

```

#### wrapped

```python
wrapped: Model = infer_model(wrapped)

```

The underlying model being wrapped.

#### settings

```python
settings: ModelSettings | None

```

Get the settings from the wrapped model.

# `pydantic_evals.dataset`

Dataset management for pydantic evals.

This module provides functionality for creating, loading, saving, and evaluating datasets of test cases. Each case must have inputs, and can optionally have a name, expected output, metadata, and case-specific evaluators.

Datasets can be loaded from and saved to YAML or JSON files, and can be evaluated against a task function to produce an evaluation report.

### Case

Bases: `Generic[InputsT, OutputT, MetadataT]`

A single row of a Dataset.

Each case represents a single test scenario with inputs to test. A case may optionally specify a name, expected outputs to compare against, and arbitrary metadata.

Cases can also have their own specific evaluators which are run in addition to dataset-level evaluators.

Example:

```python
from pydantic_evals import Case

case = Case(
    name='Simple addition',
    inputs={'a': 1, 'b': 2},
    expected_output=3,
    metadata={'description': 'Tests basic addition'},
)

```

Source code in `pydantic_evals/pydantic_evals/dataset.py`

````python
@dataclass(init=False)
class Case(Generic[InputsT, OutputT, MetadataT]):
    """A single row of a [`Dataset`][pydantic_evals.Dataset].

    Each case represents a single test scenario with inputs to test. A case may optionally specify a name, expected
    outputs to compare against, and arbitrary metadata.

    Cases can also have their own specific evaluators which are run in addition to dataset-level evaluators.

    Example:
    ```python
    from pydantic_evals import Case

    case = Case(
        name='Simple addition',
        inputs={'a': 1, 'b': 2},
        expected_output=3,
        metadata={'description': 'Tests basic addition'},
    )
    ```
    """

    name: str | None
    """Name of the case. This is used to identify the case in the report and can be used to filter cases."""
    inputs: InputsT
    """Inputs to the task. This is the input to the task that will be evaluated."""
    metadata: MetadataT | None = None
    """Metadata to be used in the evaluation.

    This can be used to provide additional information about the case to the evaluators.
    """
    expected_output: OutputT | None = None
    """Expected output of the task. This is the expected output of the task that will be evaluated."""
    evaluators: list[Evaluator[InputsT, OutputT, MetadataT]] = field(default_factory=list)
    """Evaluators to be used just on this case."""

    def __init__(
        self,
        *,
        name: str | None = None,
        inputs: InputsT,
        metadata: MetadataT | None = None,
        expected_output: OutputT | None = None,
        evaluators: tuple[Evaluator[InputsT, OutputT, MetadataT], ...] = (),
    ):
        """Initialize a new test case.

        Args:
            name: Optional name for the case. If not provided, a generic name will be assigned when added to a dataset.
            inputs: The inputs to the task being evaluated.
            metadata: Optional metadata for the case, which can be used by evaluators.
            expected_output: Optional expected output of the task, used for comparison in evaluators.
            evaluators: Tuple of evaluators specific to this case. These are in addition to any
                dataset-level evaluators.

        """
        # Note: `evaluators` must be a tuple instead of Sequence due to misbehavior with pyright's generic parameter
        # inference if it has type `Sequence`
        self.name = name
        self.inputs = inputs
        self.metadata = metadata
        self.expected_output = expected_output
        self.evaluators = list(evaluators)

````

#### __init__

```python
__init__(
    *,
    name: str | None = None,
    inputs: InputsT,
    metadata: MetadataT | None = None,
    expected_output: OutputT | None = None,
    evaluators: tuple[
        Evaluator[InputsT, OutputT, MetadataT], ...
    ] = ()
)

```

Initialize a new test case.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | Optional name for the case. If not provided, a generic name will be assigned when added to a dataset. | `None` | | `inputs` | `InputsT` | The inputs to the task being evaluated. | *required* | | `metadata` | `MetadataT | None` | Optional metadata for the case, which can be used by evaluators. | `None` | | `expected_output` | `OutputT | None` | Optional expected output of the task, used for comparison in evaluators. | `None` | | `evaluators` | `tuple[Evaluator[InputsT, OutputT, MetadataT], ...]` | Tuple of evaluators specific to this case. These are in addition to any dataset-level evaluators. | `()` |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
def __init__(
    self,
    *,
    name: str | None = None,
    inputs: InputsT,
    metadata: MetadataT | None = None,
    expected_output: OutputT | None = None,
    evaluators: tuple[Evaluator[InputsT, OutputT, MetadataT], ...] = (),
):
    """Initialize a new test case.

    Args:
        name: Optional name for the case. If not provided, a generic name will be assigned when added to a dataset.
        inputs: The inputs to the task being evaluated.
        metadata: Optional metadata for the case, which can be used by evaluators.
        expected_output: Optional expected output of the task, used for comparison in evaluators.
        evaluators: Tuple of evaluators specific to this case. These are in addition to any
            dataset-level evaluators.

    """
    # Note: `evaluators` must be a tuple instead of Sequence due to misbehavior with pyright's generic parameter
    # inference if it has type `Sequence`
    self.name = name
    self.inputs = inputs
    self.metadata = metadata
    self.expected_output = expected_output
    self.evaluators = list(evaluators)

```

#### name

```python
name: str | None = name

```

Name of the case. This is used to identify the case in the report and can be used to filter cases.

#### inputs

```python
inputs: InputsT = inputs

```

Inputs to the task. This is the input to the task that will be evaluated.

#### metadata

```python
metadata: MetadataT | None = metadata

```

Metadata to be used in the evaluation.

This can be used to provide additional information about the case to the evaluators.

#### expected_output

```python
expected_output: OutputT | None = expected_output

```

Expected output of the task. This is the expected output of the task that will be evaluated.

#### evaluators

```python
evaluators: list[Evaluator[InputsT, OutputT, MetadataT]] = (
    list(evaluators)
)

```

Evaluators to be used just on this case.

### Dataset

Bases: `BaseModel`, `Generic[InputsT, OutputT, MetadataT]`

A dataset of test cases.

Datasets allow you to organize a collection of test cases and evaluate them against a task function. They can be loaded from and saved to YAML or JSON files, and can have dataset-level evaluators that apply to all cases.

Example:

```python
# Create a dataset with two test cases
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ExactMatch(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output == ctx.expected_output

dataset = Dataset(
    cases=[
        Case(name='test1', inputs={'text': 'Hello'}, expected_output='HELLO'),
        Case(name='test2', inputs={'text': 'World'}, expected_output='WORLD'),
    ],
    evaluators=[ExactMatch()],
)

# Evaluate the dataset against a task function
async def uppercase(inputs: dict) -> str:
    return inputs['text'].upper()

async def main():
    report = await dataset.evaluate(uppercase)
    report.print()
'''
   Evaluation Summary: uppercase
â”â”â”â”â”â”â”â”â”â”â”â”³â”â”â”â”â”â”â”â”â”â”â”â”â”³â”â”â”â”â”â”â”â”â”â”â”“
â”ƒ Case ID  â”ƒ Assertions â”ƒ Duration â”ƒ
â”¡â”â”â”â”â”â”â”â”â”â”â•‡â”â”â”â”â”â”â”â”â”â”â”â”â•‡â”â”â”â”â”â”â”â”â”â”â”©
â”‚ test1    â”‚ âœ”          â”‚     10ms â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ test2    â”‚ âœ”          â”‚     10ms â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Averages â”‚ 100.0% âœ”   â”‚     10ms â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
'''

```

Source code in `pydantic_evals/pydantic_evals/dataset.py`

````python
class Dataset(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid', arbitrary_types_allowed=True):
    """A dataset of test [cases][pydantic_evals.Case].

    Datasets allow you to organize a collection of test cases and evaluate them against a task function.
    They can be loaded from and saved to YAML or JSON files, and can have dataset-level evaluators that
    apply to all cases.

    Example:
    ```python
    # Create a dataset with two test cases
    from dataclasses import dataclass

    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext


    @dataclass
    class ExactMatch(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return ctx.output == ctx.expected_output

    dataset = Dataset(
        cases=[
            Case(name='test1', inputs={'text': 'Hello'}, expected_output='HELLO'),
            Case(name='test2', inputs={'text': 'World'}, expected_output='WORLD'),
        ],
        evaluators=[ExactMatch()],
    )

    # Evaluate the dataset against a task function
    async def uppercase(inputs: dict) -> str:
        return inputs['text'].upper()

    async def main():
        report = await dataset.evaluate(uppercase)
        report.print()
    '''
       Evaluation Summary: uppercase
    â”â”â”â”â”â”â”â”â”â”â”â”³â”â”â”â”â”â”â”â”â”â”â”â”â”³â”â”â”â”â”â”â”â”â”â”â”“
    â”ƒ Case ID  â”ƒ Assertions â”ƒ Duration â”ƒ
    â”¡â”â”â”â”â”â”â”â”â”â”â•‡â”â”â”â”â”â”â”â”â”â”â”â”â•‡â”â”â”â”â”â”â”â”â”â”â”©
    â”‚ test1    â”‚ âœ”          â”‚     10ms â”‚
    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    â”‚ test2    â”‚ âœ”          â”‚     10ms â”‚
    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    â”‚ Averages â”‚ 100.0% âœ”   â”‚     10ms â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    '''
    ```
    """

    name: str | None = None
    """Optional name of the dataset."""
    cases: list[Case[InputsT, OutputT, MetadataT]]
    """List of test cases in the dataset."""
    evaluators: list[Evaluator[InputsT, OutputT, MetadataT]] = []
    """List of evaluators to be used on all cases in the dataset."""

    def __init__(
        self,
        *,
        name: str | None = None,
        cases: Sequence[Case[InputsT, OutputT, MetadataT]],
        evaluators: Sequence[Evaluator[InputsT, OutputT, MetadataT]] = (),
    ):
        """Initialize a new dataset with test cases and optional evaluators.

        Args:
            name: Optional name for the dataset.
            cases: Sequence of test cases to include in the dataset.
            evaluators: Optional sequence of evaluators to apply to all cases in the dataset.
        """
        case_names = set[str]()
        for case in cases:
            if case.name is None:
                continue
            if case.name in case_names:
                raise ValueError(f'Duplicate case name: {case.name!r}')
            case_names.add(case.name)

        super().__init__(
            name=name,
            cases=cases,
            evaluators=list(evaluators),
        )

    # TODO in v2: Make everything not required keyword-only
    async def evaluate(
        self,
        task: Callable[[InputsT], Awaitable[OutputT]] | Callable[[InputsT], OutputT],
        name: str | None = None,
        max_concurrency: int | None = None,
        progress: bool = True,
        retry_task: RetryConfig | None = None,
        retry_evaluators: RetryConfig | None = None,
        *,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport[InputsT, OutputT, MetadataT]:
        """Evaluates the test cases in the dataset using the given task.

        This method runs the task on each case in the dataset, applies evaluators,
        and collects results into a report. Cases are run concurrently, limited by `max_concurrency` if specified.

        Args:
            task: The task to evaluate. This should be a callable that takes the inputs of the case
                and returns the output.
            name: The name of the experiment being run, this is used to identify the experiment in the report.
                If omitted, the task_name will be used; if that is not specified, the name of the task function is used.
            max_concurrency: The maximum number of concurrent evaluations of the task to allow.
                If None, all cases will be evaluated concurrently.
            progress: Whether to show a progress bar for the evaluation. Defaults to `True`.
            retry_task: Optional retry configuration for the task execution.
            retry_evaluators: Optional retry configuration for evaluator execution.
            task_name: Optional override to the name of the task being executed, otherwise the name of the task
                function will be used.
            metadata: Optional dict of experiment metadata.

        Returns:
            A report containing the results of the evaluation.
        """
        task_name = task_name or get_unwrapped_function_name(task)
        name = name or task_name
        total_cases = len(self.cases)
        progress_bar = Progress() if progress else None

        limiter = anyio.Semaphore(max_concurrency) if max_concurrency is not None else AsyncExitStack()

        extra_attributes: dict[str, Any] = {'gen_ai.operation.name': 'experiment'}
        if metadata is not None:
            extra_attributes['metadata'] = metadata
        with (
            logfire_span(
                'evaluate {name}',
                name=name,
                task_name=task_name,
                dataset_name=self.name,
                n_cases=len(self.cases),
                **extra_attributes,
            ) as eval_span,
            progress_bar or nullcontext(),
        ):
            task_id = progress_bar.add_task(f'Evaluating {task_name}', total=total_cases) if progress_bar else None

            async def _handle_case(case: Case[InputsT, OutputT, MetadataT], report_case_name: str):
                async with limiter:
                    result = await _run_task_and_evaluators(
                        task, case, report_case_name, self.evaluators, retry_task, retry_evaluators
                    )
                    if progress_bar and task_id is not None:  # pragma: no branch
                        progress_bar.update(task_id, advance=1)
                    return result

            if (context := eval_span.context) is None:  # pragma: no cover
                trace_id = None
                span_id = None
            else:
                trace_id = f'{context.trace_id:032x}'
                span_id = f'{context.span_id:016x}'
            cases_and_failures = await task_group_gather(
                [
                    lambda case=case, i=i: _handle_case(case, case.name or f'Case {i}')
                    for i, case in enumerate(self.cases, 1)
                ]
            )
            cases: list[ReportCase] = []
            failures: list[ReportCaseFailure] = []
            for item in cases_and_failures:
                if isinstance(item, ReportCase):
                    cases.append(item)
                else:
                    failures.append(item)
            report = EvaluationReport(
                name=name,
                cases=cases,
                failures=failures,
                experiment_metadata=metadata,
                span_id=span_id,
                trace_id=trace_id,
            )
            full_experiment_metadata: dict[str, Any] = {'n_cases': len(self.cases)}
            if metadata is not None:
                full_experiment_metadata['metadata'] = metadata
            if (averages := report.averages()) is not None:
                full_experiment_metadata['averages'] = averages
                if averages.assertions is not None:
                    eval_span.set_attribute('assertion_pass_rate', averages.assertions)
            eval_span.set_attribute('logfire.experiment.metadata', full_experiment_metadata)
        return report

    def evaluate_sync(
        self,
        task: Callable[[InputsT], Awaitable[OutputT]] | Callable[[InputsT], OutputT],
        name: str | None = None,
        max_concurrency: int | None = None,
        progress: bool = True,
        retry_task: RetryConfig | None = None,
        retry_evaluators: RetryConfig | None = None,
        *,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport[InputsT, OutputT, MetadataT]:
        """Evaluates the test cases in the dataset using the given task.

        This is a synchronous wrapper around [`evaluate`][pydantic_evals.Dataset.evaluate] provided for convenience.

        Args:
            task: The task to evaluate. This should be a callable that takes the inputs of the case
                and returns the output.
            name: The name of the experiment being run, this is used to identify the experiment in the report.
                If omitted, the task_name will be used; if that is not specified, the name of the task function is used.
            max_concurrency: The maximum number of concurrent evaluations of the task to allow.
                If None, all cases will be evaluated concurrently.
            progress: Whether to show a progress bar for the evaluation. Defaults to `True`.
            retry_task: Optional retry configuration for the task execution.
            retry_evaluators: Optional retry configuration for evaluator execution.
            task_name: Optional override to the name of the task being executed, otherwise the name of the task
                function will be used.
            metadata: Optional dict of experiment metadata.

        Returns:
            A report containing the results of the evaluation.
        """
        return get_event_loop().run_until_complete(
            self.evaluate(
                task,
                name=name,
                max_concurrency=max_concurrency,
                progress=progress,
                retry_task=retry_task,
                retry_evaluators=retry_evaluators,
                task_name=task_name,
                metadata=metadata,
            )
        )

    def add_case(
        self,
        *,
        name: str | None = None,
        inputs: InputsT,
        metadata: MetadataT | None = None,
        expected_output: OutputT | None = None,
        evaluators: tuple[Evaluator[InputsT, OutputT, MetadataT], ...] = (),
    ) -> None:
        """Adds a case to the dataset.

        This is a convenience method for creating a [`Case`][pydantic_evals.Case] and adding it to the dataset.

        Args:
            name: Optional name for the case. If not provided, a generic name will be assigned.
            inputs: The inputs to the task being evaluated.
            metadata: Optional metadata for the case, which can be used by evaluators.
            expected_output: The expected output of the task, used for comparison in evaluators.
            evaluators: Tuple of evaluators specific to this case, in addition to dataset-level evaluators.
        """
        if name in {case.name for case in self.cases}:
            raise ValueError(f'Duplicate case name: {name!r}')

        case = Case[InputsT, OutputT, MetadataT](
            name=name,
            inputs=inputs,
            metadata=metadata,
            expected_output=expected_output,
            evaluators=evaluators,
        )
        self.cases.append(case)

    def add_evaluator(
        self,
        evaluator: Evaluator[InputsT, OutputT, MetadataT],
        specific_case: str | None = None,
    ) -> None:
        """Adds an evaluator to the dataset or a specific case.

        Args:
            evaluator: The evaluator to add.
            specific_case: If provided, the evaluator will only be added to the case with this name.
                If None, the evaluator will be added to all cases in the dataset.

        Raises:
            ValueError: If `specific_case` is provided but no case with that name exists in the dataset.
        """
        if specific_case is None:
            self.evaluators.append(evaluator)
        else:
            # If this is too slow, we could try to add a case lookup dict.
            # Note that if we do that, we'd need to make the cases list private to prevent modification.
            added = False
            for case in self.cases:
                if case.name == specific_case:
                    case.evaluators.append(evaluator)
                    added = True
            if not added:
                raise ValueError(f'Case {specific_case!r} not found in the dataset')

    @classmethod
    @functools.cache
    def _params(cls) -> tuple[type[InputsT], type[OutputT], type[MetadataT]]:
        """Get the type parameters for the Dataset class.

        Returns:
            A tuple of (InputsT, OutputT, MetadataT) types.
        """
        for c in cls.__mro__:
            metadata = getattr(c, '__pydantic_generic_metadata__', {})
            if len(args := (metadata.get('args', ()) or getattr(c, '__args__', ()))) == 3:  # pragma: no branch
                return args
        else:  # pragma: no cover
            warnings.warn(
                f'Could not determine the generic parameters for {cls}; using `Any` for each.'
                f' You should explicitly set the generic parameters via `Dataset[MyInputs, MyOutput, MyMetadata]`'
                f' when serializing or deserializing.',
                UserWarning,
            )
            return Any, Any, Any  # type: ignore

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        fmt: Literal['yaml', 'json'] | None = None,
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    ) -> Self:
        """Load a dataset from a file.

        Args:
            path: Path to the file to load.
            fmt: Format of the file. If None, the format will be inferred from the file extension.
                Must be either 'yaml' or 'json'.
            custom_evaluator_types: Custom evaluator classes to use when deserializing the dataset.
                These are additional evaluators beyond the default ones.

        Returns:
            A new Dataset instance loaded from the file.

        Raises:
            ValidationError: If the file cannot be parsed as a valid dataset.
            ValueError: If the format cannot be inferred from the file extension.
        """
        path = Path(path)
        fmt = cls._infer_fmt(path, fmt)

        raw = Path(path).read_text()
        try:
            return cls.from_text(raw, fmt=fmt, custom_evaluator_types=custom_evaluator_types, default_name=path.stem)
        except ValidationError as e:  # pragma: no cover
            raise ValueError(f'{path} contains data that does not match the schema for {cls.__name__}:\n{e}.') from e

    @classmethod
    def from_text(
        cls,
        contents: str,
        fmt: Literal['yaml', 'json'] = 'yaml',
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
        *,
        default_name: str | None = None,
    ) -> Self:
        """Load a dataset from a string.

        Args:
            contents: The string content to parse.
            fmt: Format of the content. Must be either 'yaml' or 'json'.
            custom_evaluator_types: Custom evaluator classes to use when deserializing the dataset.
                These are additional evaluators beyond the default ones.
            default_name: Default name of the dataset, to be used if not specified in the serialized contents.

        Returns:
            A new Dataset instance parsed from the string.

        Raises:
            ValidationError: If the content cannot be parsed as a valid dataset.
        """
        if fmt == 'yaml':
            loaded = yaml.safe_load(contents)
            return cls.from_dict(loaded, custom_evaluator_types, default_name=default_name)
        else:
            dataset_model_type = cls._serialization_type()
            dataset_model = dataset_model_type.model_validate_json(contents)
            return cls._from_dataset_model(dataset_model, custom_evaluator_types, default_name)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
        *,
        default_name: str | None = None,
    ) -> Self:
        """Load a dataset from a dictionary.

        Args:
            data: Dictionary representation of the dataset.
            custom_evaluator_types: Custom evaluator classes to use when deserializing the dataset.
                These are additional evaluators beyond the default ones.
            default_name: Default name of the dataset, to be used if not specified in the data.

        Returns:
            A new Dataset instance created from the dictionary.

        Raises:
            ValidationError: If the dictionary cannot be converted to a valid dataset.
        """
        dataset_model_type = cls._serialization_type()
        dataset_model = dataset_model_type.model_validate(data)
        return cls._from_dataset_model(dataset_model, custom_evaluator_types, default_name)

    @classmethod
    def _from_dataset_model(
        cls,
        dataset_model: _DatasetModel[InputsT, OutputT, MetadataT],
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
        default_name: str | None = None,
    ) -> Self:
        """Create a Dataset from a _DatasetModel.

        Args:
            dataset_model: The _DatasetModel to convert.
            custom_evaluator_types: Custom evaluator classes to register for deserialization.
            default_name: Default name of the dataset, to be used if the value is `None` in the provided model.

        Returns:
            A new Dataset instance created from the _DatasetModel.
        """
        registry = _get_registry(custom_evaluator_types)

        cases: list[Case[InputsT, OutputT, MetadataT]] = []
        errors: list[ValueError] = []
        dataset_evaluators: list[Evaluator] = []
        for spec in dataset_model.evaluators:
            try:
                dataset_evaluator = _load_evaluator_from_registry(registry, None, spec)
            except ValueError as e:
                errors.append(e)
                continue
            dataset_evaluators.append(dataset_evaluator)

        for row in dataset_model.cases:
            evaluators: list[Evaluator] = []
            for spec in row.evaluators:
                try:
                    evaluator = _load_evaluator_from_registry(registry, row.name, spec)
                except ValueError as e:
                    errors.append(e)
                    continue
                evaluators.append(evaluator)
            row = Case[InputsT, OutputT, MetadataT](
                name=row.name,
                inputs=row.inputs,
                metadata=row.metadata,
                expected_output=row.expected_output,
            )
            row.evaluators = evaluators
            cases.append(row)
        if errors:
            raise ExceptionGroup(f'{len(errors)} error(s) loading evaluators from registry', errors[:3])
        result = cls(name=dataset_model.name, cases=cases)
        if result.name is None:
            result.name = default_name
        result.evaluators = dataset_evaluators
        return result

    def to_file(
        self,
        path: Path | str,
        fmt: Literal['yaml', 'json'] | None = None,
        schema_path: Path | str | None = DEFAULT_SCHEMA_PATH_TEMPLATE,
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    ):
        """Save the dataset to a file.

        Args:
            path: Path to save the dataset to.
            fmt: Format to use. If None, the format will be inferred from the file extension.
                Must be either 'yaml' or 'json'.
            schema_path: Path to save the JSON schema to. If None, no schema will be saved.
                Can be a string template with {stem} which will be replaced with the dataset filename stem.
            custom_evaluator_types: Custom evaluator classes to include in the schema.
        """
        path = Path(path)
        fmt = self._infer_fmt(path, fmt)

        schema_ref: str | None = None
        if schema_path is not None:  # pragma: no branch
            if isinstance(schema_path, str):  # pragma: no branch
                schema_path = Path(schema_path.format(stem=path.stem))

            if not schema_path.is_absolute():
                schema_ref = str(schema_path)
                schema_path = path.parent / schema_path
            elif schema_path.is_relative_to(path):  # pragma: no cover
                schema_ref = str(_get_relative_path_reference(schema_path, path))
            else:  # pragma: no cover
                schema_ref = str(schema_path)
            self._save_schema(schema_path, custom_evaluator_types)

        context: dict[str, Any] = {'use_short_form': True}
        if fmt == 'yaml':
            dumped_data = self.model_dump(mode='json', by_alias=True, context=context)
            content = yaml.dump(dumped_data, sort_keys=False)
            if schema_ref:  # pragma: no branch
                yaml_language_server_line = f'{_YAML_SCHEMA_LINE_PREFIX}{schema_ref}'
                content = f'{yaml_language_server_line}\n{content}'
            path.write_text(content)
        else:
            context['$schema'] = schema_ref
            json_data = self.model_dump_json(indent=2, by_alias=True, context=context)
            path.write_text(json_data + '\n')

    @classmethod
    def model_json_schema_with_evaluators(
        cls,
        custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    ) -> dict[str, Any]:
        """Generate a JSON schema for this dataset type, including evaluator details.

        This is useful for generating a schema that can be used to validate YAML-format dataset files.

        Args:
            custom_evaluator_types: Custom evaluator classes to include in the schema.

        Returns:
            A dictionary representing the JSON schema.
        """
        # Note: this function could maybe be simplified now that Evaluators are always dataclasses
        registry = _get_registry(custom_evaluator_types)

        evaluator_schema_types: list[Any] = []
        for name, evaluator_class in registry.items():
            type_hints = _typing_extra.get_function_type_hints(evaluator_class)
            type_hints.pop('return', None)
            required_type_hints: dict[str, Any] = {}

            for p in inspect.signature(evaluator_class).parameters.values():
                type_hints.setdefault(p.name, Any)
                if p.default is not p.empty:
                    type_hints[p.name] = NotRequired[type_hints[p.name]]
                else:
                    required_type_hints[p.name] = type_hints[p.name]

            def _make_typed_dict(cls_name_prefix: str, fields: dict[str, Any]) -> Any:
                td = TypedDict(f'{cls_name_prefix}_{name}', fields)  # pyright: ignore[reportArgumentType]
                config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
                # TODO: Replace with pydantic.with_config once pydantic 2.11 is the min supported version
                td.__pydantic_config__ = config  # pyright: ignore[reportAttributeAccessIssue]
                return td

            # Shortest form: just the call name
            if len(type_hints) == 0 or not required_type_hints:
                evaluator_schema_types.append(Literal[name])

            # Short form: can be called with only one parameter
            if len(type_hints) == 1:
                [type_hint_type] = type_hints.values()
                evaluator_schema_types.append(_make_typed_dict('short_evaluator', {name: type_hint_type}))
            elif len(required_type_hints) == 1:  # pragma: no branch
                [type_hint_type] = required_type_hints.values()
                evaluator_schema_types.append(_make_typed_dict('short_evaluator', {name: type_hint_type}))

            # Long form: multiple parameters, possibly required
            if len(type_hints) > 1:
                params_td = _make_typed_dict('evaluator_params', type_hints)
                evaluator_schema_types.append(_make_typed_dict('evaluator', {name: params_td}))

        in_type, out_type, meta_type = cls._params()

        # Note: we shadow the `Case` and `Dataset` class names here to generate a clean JSON schema
        class Case(BaseModel, extra='forbid'):  # pyright: ignore[reportUnusedClass]  # this _is_ used below, but pyright doesn't seem to notice..
            name: str | None = None
            inputs: in_type  # pyright: ignore[reportInvalidTypeForm]
            metadata: meta_type | None = None  # pyright: ignore[reportInvalidTypeForm,reportUnknownVariableType]
            expected_output: out_type | None = None  # pyright: ignore[reportInvalidTypeForm,reportUnknownVariableType]
            if evaluator_schema_types:  # pragma: no branch
                evaluators: list[Union[tuple(evaluator_schema_types)]] = []  # pyright: ignore  # noqa UP007

        class Dataset(BaseModel, extra='forbid'):
            name: str | None = None
            cases: list[Case]
            if evaluator_schema_types:  # pragma: no branch
                evaluators: list[Union[tuple(evaluator_schema_types)]] = []  # pyright: ignore  # noqa UP007

        json_schema = Dataset.model_json_schema()
        # See `_add_json_schema` below, since `$schema` is added to the JSON, it has to be supported in the JSON
        json_schema['properties']['$schema'] = {'type': 'string'}
        return json_schema

    @classmethod
    def _save_schema(
        cls, path: Path | str, custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = ()
    ):
        """Save the JSON schema for this dataset type to a file.

        Args:
            path: Path to save the schema to.
            custom_evaluator_types: Custom evaluator classes to include in the schema.
        """
        path = Path(path)
        json_schema = cls.model_json_schema_with_evaluators(custom_evaluator_types)
        schema_content = to_json(json_schema, indent=2).decode() + '\n'
        if not path.exists() or path.read_text() != schema_content:  # pragma: no branch
            path.write_text(schema_content)

    @classmethod
    @functools.cache
    def _serialization_type(cls) -> type[_DatasetModel[InputsT, OutputT, MetadataT]]:
        """Get the serialization type for this dataset class.

        Returns:
            A _DatasetModel type with the same generic parameters as this Dataset class.
        """
        input_type, output_type, metadata_type = cls._params()
        return _DatasetModel[input_type, output_type, metadata_type]

    @classmethod
    def _infer_fmt(cls, path: Path, fmt: Literal['yaml', 'json'] | None) -> Literal['yaml', 'json']:
        """Infer the format to use for a file based on its extension.

        Args:
            path: The path to infer the format for.
            fmt: The explicitly provided format, if any.

        Returns:
            The inferred format ('yaml' or 'json').

        Raises:
            ValueError: If the format cannot be inferred from the file extension.
        """
        if fmt is not None:
            return fmt
        suffix = path.suffix.lower()
        if suffix in {'.yaml', '.yml'}:
            return 'yaml'
        elif suffix == '.json':
            return 'json'
        raise ValueError(
            f'Could not infer format for filename {path.name!r}. Use the `fmt` argument to specify the format.'
        )

    @model_serializer(mode='wrap')
    def _add_json_schema(self, nxt: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict[str, Any]:
        """Add the JSON schema path to the serialized output.

        See <https://github.com/json-schema-org/json-schema-spec/issues/828> for context, that seems to be the nearest
        there is to a spec for this.
        """
        context = cast(dict[str, Any] | None, info.context)
        if isinstance(context, dict) and (schema := context.get('$schema')):
            return {'$schema': schema} | nxt(self)
        else:
            return nxt(self)

````

#### name

```python
name: str | None = None

```

Optional name of the dataset.

#### cases

```python
cases: list[Case[InputsT, OutputT, MetadataT]]

```

List of test cases in the dataset.

#### evaluators

```python
evaluators: list[Evaluator[InputsT, OutputT, MetadataT]] = (
    []
)

```

List of evaluators to be used on all cases in the dataset.

#### __init__

```python
__init__(
    *,
    name: str | None = None,
    cases: Sequence[Case[InputsT, OutputT, MetadataT]],
    evaluators: Sequence[
        Evaluator[InputsT, OutputT, MetadataT]
    ] = ()
)

```

Initialize a new dataset with test cases and optional evaluators.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | Optional name for the dataset. | `None` | | `cases` | `Sequence[Case[InputsT, OutputT, MetadataT]]` | Sequence of test cases to include in the dataset. | *required* | | `evaluators` | `Sequence[Evaluator[InputsT, OutputT, MetadataT]]` | Optional sequence of evaluators to apply to all cases in the dataset. | `()` |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
def __init__(
    self,
    *,
    name: str | None = None,
    cases: Sequence[Case[InputsT, OutputT, MetadataT]],
    evaluators: Sequence[Evaluator[InputsT, OutputT, MetadataT]] = (),
):
    """Initialize a new dataset with test cases and optional evaluators.

    Args:
        name: Optional name for the dataset.
        cases: Sequence of test cases to include in the dataset.
        evaluators: Optional sequence of evaluators to apply to all cases in the dataset.
    """
    case_names = set[str]()
    for case in cases:
        if case.name is None:
            continue
        if case.name in case_names:
            raise ValueError(f'Duplicate case name: {case.name!r}')
        case_names.add(case.name)

    super().__init__(
        name=name,
        cases=cases,
        evaluators=list(evaluators),
    )

```

#### evaluate

```python
evaluate(
    task: (
        Callable[[InputsT], Awaitable[OutputT]]
        | Callable[[InputsT], OutputT]
    ),
    name: str | None = None,
    max_concurrency: int | None = None,
    progress: bool = True,
    retry_task: RetryConfig | None = None,
    retry_evaluators: RetryConfig | None = None,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None
) -> EvaluationReport[InputsT, OutputT, MetadataT]

```

Evaluates the test cases in the dataset using the given task.

This method runs the task on each case in the dataset, applies evaluators, and collects results into a report. Cases are run concurrently, limited by `max_concurrency` if specified.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `task` | `Callable[[InputsT], Awaitable[OutputT]] | Callable[[InputsT], OutputT]` | The task to evaluate. This should be a callable that takes the inputs of the case and returns the output. | *required* | | `name` | `str | None` | The name of the experiment being run, this is used to identify the experiment in the report. If omitted, the task_name will be used; if that is not specified, the name of the task function is used. | `None` | | `max_concurrency` | `int | None` | The maximum number of concurrent evaluations of the task to allow. If None, all cases will be evaluated concurrently. | `None` | | `progress` | `bool` | Whether to show a progress bar for the evaluation. Defaults to True. | `True` | | `retry_task` | `RetryConfig | None` | Optional retry configuration for the task execution. | `None` | | `retry_evaluators` | `RetryConfig | None` | Optional retry configuration for evaluator execution. | `None` | | `task_name` | `str | None` | Optional override to the name of the task being executed, otherwise the name of the task function will be used. | `None` | | `metadata` | `dict[str, Any] | None` | Optional dict of experiment metadata. | `None` |

Returns:

| Type | Description | | --- | --- | | `EvaluationReport[InputsT, OutputT, MetadataT]` | A report containing the results of the evaluation. |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
async def evaluate(
    self,
    task: Callable[[InputsT], Awaitable[OutputT]] | Callable[[InputsT], OutputT],
    name: str | None = None,
    max_concurrency: int | None = None,
    progress: bool = True,
    retry_task: RetryConfig | None = None,
    retry_evaluators: RetryConfig | None = None,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport[InputsT, OutputT, MetadataT]:
    """Evaluates the test cases in the dataset using the given task.

    This method runs the task on each case in the dataset, applies evaluators,
    and collects results into a report. Cases are run concurrently, limited by `max_concurrency` if specified.

    Args:
        task: The task to evaluate. This should be a callable that takes the inputs of the case
            and returns the output.
        name: The name of the experiment being run, this is used to identify the experiment in the report.
            If omitted, the task_name will be used; if that is not specified, the name of the task function is used.
        max_concurrency: The maximum number of concurrent evaluations of the task to allow.
            If None, all cases will be evaluated concurrently.
        progress: Whether to show a progress bar for the evaluation. Defaults to `True`.
        retry_task: Optional retry configuration for the task execution.
        retry_evaluators: Optional retry configuration for evaluator execution.
        task_name: Optional override to the name of the task being executed, otherwise the name of the task
            function will be used.
        metadata: Optional dict of experiment metadata.

    Returns:
        A report containing the results of the evaluation.
    """
    task_name = task_name or get_unwrapped_function_name(task)
    name = name or task_name
    total_cases = len(self.cases)
    progress_bar = Progress() if progress else None

    limiter = anyio.Semaphore(max_concurrency) if max_concurrency is not None else AsyncExitStack()

    extra_attributes: dict[str, Any] = {'gen_ai.operation.name': 'experiment'}
    if metadata is not None:
        extra_attributes['metadata'] = metadata
    with (
        logfire_span(
            'evaluate {name}',
            name=name,
            task_name=task_name,
            dataset_name=self.name,
            n_cases=len(self.cases),
            **extra_attributes,
        ) as eval_span,
        progress_bar or nullcontext(),
    ):
        task_id = progress_bar.add_task(f'Evaluating {task_name}', total=total_cases) if progress_bar else None

        async def _handle_case(case: Case[InputsT, OutputT, MetadataT], report_case_name: str):
            async with limiter:
                result = await _run_task_and_evaluators(
                    task, case, report_case_name, self.evaluators, retry_task, retry_evaluators
                )
                if progress_bar and task_id is not None:  # pragma: no branch
                    progress_bar.update(task_id, advance=1)
                return result

        if (context := eval_span.context) is None:  # pragma: no cover
            trace_id = None
            span_id = None
        else:
            trace_id = f'{context.trace_id:032x}'
            span_id = f'{context.span_id:016x}'
        cases_and_failures = await task_group_gather(
            [
                lambda case=case, i=i: _handle_case(case, case.name or f'Case {i}')
                for i, case in enumerate(self.cases, 1)
            ]
        )
        cases: list[ReportCase] = []
        failures: list[ReportCaseFailure] = []
        for item in cases_and_failures:
            if isinstance(item, ReportCase):
                cases.append(item)
            else:
                failures.append(item)
        report = EvaluationReport(
            name=name,
            cases=cases,
            failures=failures,
            experiment_metadata=metadata,
            span_id=span_id,
            trace_id=trace_id,
        )
        full_experiment_metadata: dict[str, Any] = {'n_cases': len(self.cases)}
        if metadata is not None:
            full_experiment_metadata['metadata'] = metadata
        if (averages := report.averages()) is not None:
            full_experiment_metadata['averages'] = averages
            if averages.assertions is not None:
                eval_span.set_attribute('assertion_pass_rate', averages.assertions)
        eval_span.set_attribute('logfire.experiment.metadata', full_experiment_metadata)
    return report

```

#### evaluate_sync

```python
evaluate_sync(
    task: (
        Callable[[InputsT], Awaitable[OutputT]]
        | Callable[[InputsT], OutputT]
    ),
    name: str | None = None,
    max_concurrency: int | None = None,
    progress: bool = True,
    retry_task: RetryConfig | None = None,
    retry_evaluators: RetryConfig | None = None,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None
) -> EvaluationReport[InputsT, OutputT, MetadataT]

```

Evaluates the test cases in the dataset using the given task.

This is a synchronous wrapper around evaluate provided for convenience.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `task` | `Callable[[InputsT], Awaitable[OutputT]] | Callable[[InputsT], OutputT]` | The task to evaluate. This should be a callable that takes the inputs of the case and returns the output. | *required* | | `name` | `str | None` | The name of the experiment being run, this is used to identify the experiment in the report. If omitted, the task_name will be used; if that is not specified, the name of the task function is used. | `None` | | `max_concurrency` | `int | None` | The maximum number of concurrent evaluations of the task to allow. If None, all cases will be evaluated concurrently. | `None` | | `progress` | `bool` | Whether to show a progress bar for the evaluation. Defaults to True. | `True` | | `retry_task` | `RetryConfig | None` | Optional retry configuration for the task execution. | `None` | | `retry_evaluators` | `RetryConfig | None` | Optional retry configuration for evaluator execution. | `None` | | `task_name` | `str | None` | Optional override to the name of the task being executed, otherwise the name of the task function will be used. | `None` | | `metadata` | `dict[str, Any] | None` | Optional dict of experiment metadata. | `None` |

Returns:

| Type | Description | | --- | --- | | `EvaluationReport[InputsT, OutputT, MetadataT]` | A report containing the results of the evaluation. |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
def evaluate_sync(
    self,
    task: Callable[[InputsT], Awaitable[OutputT]] | Callable[[InputsT], OutputT],
    name: str | None = None,
    max_concurrency: int | None = None,
    progress: bool = True,
    retry_task: RetryConfig | None = None,
    retry_evaluators: RetryConfig | None = None,
    *,
    task_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport[InputsT, OutputT, MetadataT]:
    """Evaluates the test cases in the dataset using the given task.

    This is a synchronous wrapper around [`evaluate`][pydantic_evals.Dataset.evaluate] provided for convenience.

    Args:
        task: The task to evaluate. This should be a callable that takes the inputs of the case
            and returns the output.
        name: The name of the experiment being run, this is used to identify the experiment in the report.
            If omitted, the task_name will be used; if that is not specified, the name of the task function is used.
        max_concurrency: The maximum number of concurrent evaluations of the task to allow.
            If None, all cases will be evaluated concurrently.
        progress: Whether to show a progress bar for the evaluation. Defaults to `True`.
        retry_task: Optional retry configuration for the task execution.
        retry_evaluators: Optional retry configuration for evaluator execution.
        task_name: Optional override to the name of the task being executed, otherwise the name of the task
            function will be used.
        metadata: Optional dict of experiment metadata.

    Returns:
        A report containing the results of the evaluation.
    """
    return get_event_loop().run_until_complete(
        self.evaluate(
            task,
            name=name,
            max_concurrency=max_concurrency,
            progress=progress,
            retry_task=retry_task,
            retry_evaluators=retry_evaluators,
            task_name=task_name,
            metadata=metadata,
        )
    )

```

#### add_case

```python
add_case(
    *,
    name: str | None = None,
    inputs: InputsT,
    metadata: MetadataT | None = None,
    expected_output: OutputT | None = None,
    evaluators: tuple[
        Evaluator[InputsT, OutputT, MetadataT], ...
    ] = ()
) -> None

```

Adds a case to the dataset.

This is a convenience method for creating a Case and adding it to the dataset.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | None` | Optional name for the case. If not provided, a generic name will be assigned. | `None` | | `inputs` | `InputsT` | The inputs to the task being evaluated. | *required* | | `metadata` | `MetadataT | None` | Optional metadata for the case, which can be used by evaluators. | `None` | | `expected_output` | `OutputT | None` | The expected output of the task, used for comparison in evaluators. | `None` | | `evaluators` | `tuple[Evaluator[InputsT, OutputT, MetadataT], ...]` | Tuple of evaluators specific to this case, in addition to dataset-level evaluators. | `()` |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
def add_case(
    self,
    *,
    name: str | None = None,
    inputs: InputsT,
    metadata: MetadataT | None = None,
    expected_output: OutputT | None = None,
    evaluators: tuple[Evaluator[InputsT, OutputT, MetadataT], ...] = (),
) -> None:
    """Adds a case to the dataset.

    This is a convenience method for creating a [`Case`][pydantic_evals.Case] and adding it to the dataset.

    Args:
        name: Optional name for the case. If not provided, a generic name will be assigned.
        inputs: The inputs to the task being evaluated.
        metadata: Optional metadata for the case, which can be used by evaluators.
        expected_output: The expected output of the task, used for comparison in evaluators.
        evaluators: Tuple of evaluators specific to this case, in addition to dataset-level evaluators.
    """
    if name in {case.name for case in self.cases}:
        raise ValueError(f'Duplicate case name: {name!r}')

    case = Case[InputsT, OutputT, MetadataT](
        name=name,
        inputs=inputs,
        metadata=metadata,
        expected_output=expected_output,
        evaluators=evaluators,
    )
    self.cases.append(case)

```

#### add_evaluator

```python
add_evaluator(
    evaluator: Evaluator[InputsT, OutputT, MetadataT],
    specific_case: str | None = None,
) -> None

```

Adds an evaluator to the dataset or a specific case.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `evaluator` | `Evaluator[InputsT, OutputT, MetadataT]` | The evaluator to add. | *required* | | `specific_case` | `str | None` | If provided, the evaluator will only be added to the case with this name. If None, the evaluator will be added to all cases in the dataset. | `None` |

Raises:

| Type | Description | | --- | --- | | `ValueError` | If specific_case is provided but no case with that name exists in the dataset. |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
def add_evaluator(
    self,
    evaluator: Evaluator[InputsT, OutputT, MetadataT],
    specific_case: str | None = None,
) -> None:
    """Adds an evaluator to the dataset or a specific case.

    Args:
        evaluator: The evaluator to add.
        specific_case: If provided, the evaluator will only be added to the case with this name.
            If None, the evaluator will be added to all cases in the dataset.

    Raises:
        ValueError: If `specific_case` is provided but no case with that name exists in the dataset.
    """
    if specific_case is None:
        self.evaluators.append(evaluator)
    else:
        # If this is too slow, we could try to add a case lookup dict.
        # Note that if we do that, we'd need to make the cases list private to prevent modification.
        added = False
        for case in self.cases:
            if case.name == specific_case:
                case.evaluators.append(evaluator)
                added = True
        if not added:
            raise ValueError(f'Case {specific_case!r} not found in the dataset')

```

#### from_file

```python
from_file(
    path: Path | str,
    fmt: Literal["yaml", "json"] | None = None,
    custom_evaluator_types: Sequence[
        type[Evaluator[InputsT, OutputT, MetadataT]]
    ] = (),
) -> Self

```

Load a dataset from a file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `path` | `Path | str` | Path to the file to load. | *required* | | `fmt` | `Literal['yaml', 'json'] | None` | Format of the file. If None, the format will be inferred from the file extension. Must be either 'yaml' or 'json'. | `None` | | `custom_evaluator_types` | `Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]]` | Custom evaluator classes to use when deserializing the dataset. These are additional evaluators beyond the default ones. | `()` |

Returns:

| Type | Description | | --- | --- | | `Self` | A new Dataset instance loaded from the file. |

Raises:

| Type | Description | | --- | --- | | `ValidationError` | If the file cannot be parsed as a valid dataset. | | `ValueError` | If the format cannot be inferred from the file extension. |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
@classmethod
def from_file(
    cls,
    path: Path | str,
    fmt: Literal['yaml', 'json'] | None = None,
    custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
) -> Self:
    """Load a dataset from a file.

    Args:
        path: Path to the file to load.
        fmt: Format of the file. If None, the format will be inferred from the file extension.
            Must be either 'yaml' or 'json'.
        custom_evaluator_types: Custom evaluator classes to use when deserializing the dataset.
            These are additional evaluators beyond the default ones.

    Returns:
        A new Dataset instance loaded from the file.

    Raises:
        ValidationError: If the file cannot be parsed as a valid dataset.
        ValueError: If the format cannot be inferred from the file extension.
    """
    path = Path(path)
    fmt = cls._infer_fmt(path, fmt)

    raw = Path(path).read_text()
    try:
        return cls.from_text(raw, fmt=fmt, custom_evaluator_types=custom_evaluator_types, default_name=path.stem)
    except ValidationError as e:  # pragma: no cover
        raise ValueError(f'{path} contains data that does not match the schema for {cls.__name__}:\n{e}.') from e

```

#### from_text

```python
from_text(
    contents: str,
    fmt: Literal["yaml", "json"] = "yaml",
    custom_evaluator_types: Sequence[
        type[Evaluator[InputsT, OutputT, MetadataT]]
    ] = (),
    *,
    default_name: str | None = None
) -> Self

```

Load a dataset from a string.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `contents` | `str` | The string content to parse. | *required* | | `fmt` | `Literal['yaml', 'json']` | Format of the content. Must be either 'yaml' or 'json'. | `'yaml'` | | `custom_evaluator_types` | `Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]]` | Custom evaluator classes to use when deserializing the dataset. These are additional evaluators beyond the default ones. | `()` | | `default_name` | `str | None` | Default name of the dataset, to be used if not specified in the serialized contents. | `None` |

Returns:

| Type | Description | | --- | --- | | `Self` | A new Dataset instance parsed from the string. |

Raises:

| Type | Description | | --- | --- | | `ValidationError` | If the content cannot be parsed as a valid dataset. |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
@classmethod
def from_text(
    cls,
    contents: str,
    fmt: Literal['yaml', 'json'] = 'yaml',
    custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    *,
    default_name: str | None = None,
) -> Self:
    """Load a dataset from a string.

    Args:
        contents: The string content to parse.
        fmt: Format of the content. Must be either 'yaml' or 'json'.
        custom_evaluator_types: Custom evaluator classes to use when deserializing the dataset.
            These are additional evaluators beyond the default ones.
        default_name: Default name of the dataset, to be used if not specified in the serialized contents.

    Returns:
        A new Dataset instance parsed from the string.

    Raises:
        ValidationError: If the content cannot be parsed as a valid dataset.
    """
    if fmt == 'yaml':
        loaded = yaml.safe_load(contents)
        return cls.from_dict(loaded, custom_evaluator_types, default_name=default_name)
    else:
        dataset_model_type = cls._serialization_type()
        dataset_model = dataset_model_type.model_validate_json(contents)
        return cls._from_dataset_model(dataset_model, custom_evaluator_types, default_name)

```

#### from_dict

```python
from_dict(
    data: dict[str, Any],
    custom_evaluator_types: Sequence[
        type[Evaluator[InputsT, OutputT, MetadataT]]
    ] = (),
    *,
    default_name: str | None = None
) -> Self

```

Load a dataset from a dictionary.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `data` | `dict[str, Any]` | Dictionary representation of the dataset. | *required* | | `custom_evaluator_types` | `Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]]` | Custom evaluator classes to use when deserializing the dataset. These are additional evaluators beyond the default ones. | `()` | | `default_name` | `str | None` | Default name of the dataset, to be used if not specified in the data. | `None` |

Returns:

| Type | Description | | --- | --- | | `Self` | A new Dataset instance created from the dictionary. |

Raises:

| Type | Description | | --- | --- | | `ValidationError` | If the dictionary cannot be converted to a valid dataset. |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
@classmethod
def from_dict(
    cls,
    data: dict[str, Any],
    custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    *,
    default_name: str | None = None,
) -> Self:
    """Load a dataset from a dictionary.

    Args:
        data: Dictionary representation of the dataset.
        custom_evaluator_types: Custom evaluator classes to use when deserializing the dataset.
            These are additional evaluators beyond the default ones.
        default_name: Default name of the dataset, to be used if not specified in the data.

    Returns:
        A new Dataset instance created from the dictionary.

    Raises:
        ValidationError: If the dictionary cannot be converted to a valid dataset.
    """
    dataset_model_type = cls._serialization_type()
    dataset_model = dataset_model_type.model_validate(data)
    return cls._from_dataset_model(dataset_model, custom_evaluator_types, default_name)

```

#### to_file

```python
to_file(
    path: Path | str,
    fmt: Literal["yaml", "json"] | None = None,
    schema_path: (
        Path | str | None
    ) = DEFAULT_SCHEMA_PATH_TEMPLATE,
    custom_evaluator_types: Sequence[
        type[Evaluator[InputsT, OutputT, MetadataT]]
    ] = (),
)

```

Save the dataset to a file.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `path` | `Path | str` | Path to save the dataset to. | *required* | | `fmt` | `Literal['yaml', 'json'] | None` | Format to use. If None, the format will be inferred from the file extension. Must be either 'yaml' or 'json'. | `None` | | `schema_path` | `Path | str | None` | Path to save the JSON schema to. If None, no schema will be saved. Can be a string template with {stem} which will be replaced with the dataset filename stem. | `DEFAULT_SCHEMA_PATH_TEMPLATE` | | `custom_evaluator_types` | `Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]]` | Custom evaluator classes to include in the schema. | `()` |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
def to_file(
    self,
    path: Path | str,
    fmt: Literal['yaml', 'json'] | None = None,
    schema_path: Path | str | None = DEFAULT_SCHEMA_PATH_TEMPLATE,
    custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
):
    """Save the dataset to a file.

    Args:
        path: Path to save the dataset to.
        fmt: Format to use. If None, the format will be inferred from the file extension.
            Must be either 'yaml' or 'json'.
        schema_path: Path to save the JSON schema to. If None, no schema will be saved.
            Can be a string template with {stem} which will be replaced with the dataset filename stem.
        custom_evaluator_types: Custom evaluator classes to include in the schema.
    """
    path = Path(path)
    fmt = self._infer_fmt(path, fmt)

    schema_ref: str | None = None
    if schema_path is not None:  # pragma: no branch
        if isinstance(schema_path, str):  # pragma: no branch
            schema_path = Path(schema_path.format(stem=path.stem))

        if not schema_path.is_absolute():
            schema_ref = str(schema_path)
            schema_path = path.parent / schema_path
        elif schema_path.is_relative_to(path):  # pragma: no cover
            schema_ref = str(_get_relative_path_reference(schema_path, path))
        else:  # pragma: no cover
            schema_ref = str(schema_path)
        self._save_schema(schema_path, custom_evaluator_types)

    context: dict[str, Any] = {'use_short_form': True}
    if fmt == 'yaml':
        dumped_data = self.model_dump(mode='json', by_alias=True, context=context)
        content = yaml.dump(dumped_data, sort_keys=False)
        if schema_ref:  # pragma: no branch
            yaml_language_server_line = f'{_YAML_SCHEMA_LINE_PREFIX}{schema_ref}'
            content = f'{yaml_language_server_line}\n{content}'
        path.write_text(content)
    else:
        context['$schema'] = schema_ref
        json_data = self.model_dump_json(indent=2, by_alias=True, context=context)
        path.write_text(json_data + '\n')

```

#### model_json_schema_with_evaluators

```python
model_json_schema_with_evaluators(
    custom_evaluator_types: Sequence[
        type[Evaluator[InputsT, OutputT, MetadataT]]
    ] = (),
) -> dict[str, Any]

```

Generate a JSON schema for this dataset type, including evaluator details.

This is useful for generating a schema that can be used to validate YAML-format dataset files.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `custom_evaluator_types` | `Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]]` | Custom evaluator classes to include in the schema. | `()` |

Returns:

| Type | Description | | --- | --- | | `dict[str, Any]` | A dictionary representing the JSON schema. |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
@classmethod
def model_json_schema_with_evaluators(
    cls,
    custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
) -> dict[str, Any]:
    """Generate a JSON schema for this dataset type, including evaluator details.

    This is useful for generating a schema that can be used to validate YAML-format dataset files.

    Args:
        custom_evaluator_types: Custom evaluator classes to include in the schema.

    Returns:
        A dictionary representing the JSON schema.
    """
    # Note: this function could maybe be simplified now that Evaluators are always dataclasses
    registry = _get_registry(custom_evaluator_types)

    evaluator_schema_types: list[Any] = []
    for name, evaluator_class in registry.items():
        type_hints = _typing_extra.get_function_type_hints(evaluator_class)
        type_hints.pop('return', None)
        required_type_hints: dict[str, Any] = {}

        for p in inspect.signature(evaluator_class).parameters.values():
            type_hints.setdefault(p.name, Any)
            if p.default is not p.empty:
                type_hints[p.name] = NotRequired[type_hints[p.name]]
            else:
                required_type_hints[p.name] = type_hints[p.name]

        def _make_typed_dict(cls_name_prefix: str, fields: dict[str, Any]) -> Any:
            td = TypedDict(f'{cls_name_prefix}_{name}', fields)  # pyright: ignore[reportArgumentType]
            config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
            # TODO: Replace with pydantic.with_config once pydantic 2.11 is the min supported version
            td.__pydantic_config__ = config  # pyright: ignore[reportAttributeAccessIssue]
            return td

        # Shortest form: just the call name
        if len(type_hints) == 0 or not required_type_hints:
            evaluator_schema_types.append(Literal[name])

        # Short form: can be called with only one parameter
        if len(type_hints) == 1:
            [type_hint_type] = type_hints.values()
            evaluator_schema_types.append(_make_typed_dict('short_evaluator', {name: type_hint_type}))
        elif len(required_type_hints) == 1:  # pragma: no branch
            [type_hint_type] = required_type_hints.values()
            evaluator_schema_types.append(_make_typed_dict('short_evaluator', {name: type_hint_type}))

        # Long form: multiple parameters, possibly required
        if len(type_hints) > 1:
            params_td = _make_typed_dict('evaluator_params', type_hints)
            evaluator_schema_types.append(_make_typed_dict('evaluator', {name: params_td}))

    in_type, out_type, meta_type = cls._params()

    # Note: we shadow the `Case` and `Dataset` class names here to generate a clean JSON schema
    class Case(BaseModel, extra='forbid'):  # pyright: ignore[reportUnusedClass]  # this _is_ used below, but pyright doesn't seem to notice..
        name: str | None = None
        inputs: in_type  # pyright: ignore[reportInvalidTypeForm]
        metadata: meta_type | None = None  # pyright: ignore[reportInvalidTypeForm,reportUnknownVariableType]
        expected_output: out_type | None = None  # pyright: ignore[reportInvalidTypeForm,reportUnknownVariableType]
        if evaluator_schema_types:  # pragma: no branch
            evaluators: list[Union[tuple(evaluator_schema_types)]] = []  # pyright: ignore  # noqa UP007

    class Dataset(BaseModel, extra='forbid'):
        name: str | None = None
        cases: list[Case]
        if evaluator_schema_types:  # pragma: no branch
            evaluators: list[Union[tuple(evaluator_schema_types)]] = []  # pyright: ignore  # noqa UP007

    json_schema = Dataset.model_json_schema()
    # See `_add_json_schema` below, since `$schema` is added to the JSON, it has to be supported in the JSON
    json_schema['properties']['$schema'] = {'type': 'string'}
    return json_schema

```

### set_eval_attribute

```python
set_eval_attribute(name: str, value: Any) -> None

```

Set an attribute on the current task run.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of the attribute. | *required* | | `value` | `Any` | The value of the attribute. | *required* |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
def set_eval_attribute(name: str, value: Any) -> None:
    """Set an attribute on the current task run.

    Args:
        name: The name of the attribute.
        value: The value of the attribute.
    """
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:  # pragma: no branch
        current_case.record_attribute(name, value)

```

### increment_eval_metric

```python
increment_eval_metric(
    name: str, amount: int | float
) -> None

```

Increment a metric on the current task run.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of the metric. | *required* | | `amount` | `int | float` | The amount to increment by. | *required* |

Source code in `pydantic_evals/pydantic_evals/dataset.py`

```python
def increment_eval_metric(name: str, amount: int | float) -> None:
    """Increment a metric on the current task run.

    Args:
        name: The name of the metric.
        amount: The amount to increment by.
    """
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:  # pragma: no branch
        current_case.increment_metric(name, amount)

```

# `pydantic_evals.evaluators`

### Contains

Bases: `Evaluator[object, object, object]`

Check if the output contains the expected output.

For strings, checks if expected_output is a substring of output. For lists/tuples, checks if expected_output is in output. For dicts, checks if all key-value pairs in expected_output are in output.

Note: case_sensitive only applies when both the value and output are strings.

Source code in `pydantic_evals/pydantic_evals/evaluators/common.py`

```python
@dataclass(repr=False)
class Contains(Evaluator[object, object, object]):
    """Check if the output contains the expected output.

    For strings, checks if expected_output is a substring of output.
    For lists/tuples, checks if expected_output is in output.
    For dicts, checks if all key-value pairs in expected_output are in output.

    Note: case_sensitive only applies when both the value and output are strings.
    """

    value: Any
    case_sensitive: bool = True
    as_strings: bool = False
    evaluation_name: str | None = field(default=None)

    def evaluate(
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> EvaluationReason:
        # Convert objects to strings if requested
        failure_reason: str | None = None
        as_strings = self.as_strings or (isinstance(self.value, str) and isinstance(ctx.output, str))
        if as_strings:
            output_str = str(ctx.output)
            expected_str = str(self.value)

            if not self.case_sensitive:
                output_str = output_str.lower()
                expected_str = expected_str.lower()

            failure_reason: str | None = None
            if expected_str not in output_str:
                output_trunc = _truncated_repr(output_str, max_length=100)
                expected_trunc = _truncated_repr(expected_str, max_length=100)
                failure_reason = f'Output string {output_trunc} does not contain expected string {expected_trunc}'
            return EvaluationReason(value=failure_reason is None, reason=failure_reason)

        try:
            # Handle different collection types
            if isinstance(ctx.output, dict):
                if isinstance(self.value, dict):
                    # Cast to Any to avoid type checking issues
                    output_dict = cast(dict[Any, Any], ctx.output)  # pyright: ignore[reportUnknownMemberType]
                    expected_dict = cast(dict[Any, Any], self.value)  # pyright: ignore[reportUnknownMemberType]
                    for k in expected_dict:
                        if k not in output_dict:
                            k_trunc = _truncated_repr(k, max_length=30)
                            failure_reason = f'Output dictionary does not contain expected key {k_trunc}'
                            break
                        elif output_dict[k] != expected_dict[k]:
                            k_trunc = _truncated_repr(k, max_length=30)
                            output_v_trunc = _truncated_repr(output_dict[k], max_length=100)
                            expected_v_trunc = _truncated_repr(expected_dict[k], max_length=100)
                            failure_reason = f'Output dictionary has different value for key {k_trunc}: {output_v_trunc} != {expected_v_trunc}'
                            break
                else:
                    if self.value not in ctx.output:  # pyright: ignore[reportUnknownMemberType]
                        output_trunc = _truncated_repr(ctx.output, max_length=200)  # pyright: ignore[reportUnknownMemberType]
                        failure_reason = f'Output {output_trunc} does not contain provided value as a key'
            elif self.value not in ctx.output:  # pyright: ignore[reportOperatorIssue]  # will be handled by except block
                output_trunc = _truncated_repr(ctx.output, max_length=200)
                failure_reason = f'Output {output_trunc} does not contain provided value'
        except (TypeError, ValueError) as e:
            failure_reason = f'Containment check failed: {e}'

        return EvaluationReason(value=failure_reason is None, reason=failure_reason)

```

### Equals

Bases: `Evaluator[object, object, object]`

Check if the output exactly equals the provided value.

Source code in `pydantic_evals/pydantic_evals/evaluators/common.py`

```python
@dataclass(repr=False)
class Equals(Evaluator[object, object, object]):
    """Check if the output exactly equals the provided value."""

    value: Any
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return ctx.output == self.value

```

### EqualsExpected

Bases: `Evaluator[object, object, object]`

Check if the output exactly equals the expected output.

Source code in `pydantic_evals/pydantic_evals/evaluators/common.py`

```python
@dataclass(repr=False)
class EqualsExpected(Evaluator[object, object, object]):
    """Check if the output exactly equals the expected output."""

    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool | dict[str, bool]:
        if ctx.expected_output is None:
            return {}  # Only compare if expected output is provided
        return ctx.output == ctx.expected_output

```

### HasMatchingSpan

Bases: `Evaluator[object, object, object]`

Check if the span tree contains a span that matches the specified query.

Source code in `pydantic_evals/pydantic_evals/evaluators/common.py`

```python
@dataclass(repr=False)
class HasMatchingSpan(Evaluator[object, object, object]):
    """Check if the span tree contains a span that matches the specified query."""

    query: SpanQuery
    evaluation_name: str | None = field(default=None)

    def evaluate(
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> bool:
        return ctx.span_tree.any(self.query)

```

### IsInstance

Bases: `Evaluator[object, object, object]`

Check if the output is an instance of a type with the given name.

Source code in `pydantic_evals/pydantic_evals/evaluators/common.py`

```python
@dataclass(repr=False)
class IsInstance(Evaluator[object, object, object]):
    """Check if the output is an instance of a type with the given name."""

    type_name: str
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        output = ctx.output
        for cls in type(output).__mro__:
            if cls.__name__ == self.type_name or cls.__qualname__ == self.type_name:
                return EvaluationReason(value=True)

        reason = f'output is of type {type(output).__name__}'
        if type(output).__qualname__ != type(output).__name__:
            reason += f' (qualname: {type(output).__qualname__})'
        return EvaluationReason(value=False, reason=reason)

```

### LLMJudge

Bases: `Evaluator[object, object, object]`

Judge whether the output of a language model meets the criteria of a provided rubric.

If you do not specify a model, it uses the default model for judging. This starts as 'openai:gpt-4o', but can be overridden by calling set_default_judge_model.

Source code in `pydantic_evals/pydantic_evals/evaluators/common.py`

```python
@dataclass(repr=False)
class LLMJudge(Evaluator[object, object, object]):
    """Judge whether the output of a language model meets the criteria of a provided rubric.

    If you do not specify a model, it uses the default model for judging. This starts as 'openai:gpt-4o', but can be
    overridden by calling [`set_default_judge_model`][pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model].
    """

    rubric: str
    model: models.Model | models.KnownModelName | None = None
    include_input: bool = False
    include_expected_output: bool = False
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = False
    assertion: OutputConfig | Literal[False] = field(default_factory=lambda: OutputConfig(include_reason=True))

    async def evaluate(
        self,
        ctx: EvaluatorContext[object, object, object],
    ) -> EvaluatorOutput:
        if self.include_input:
            if self.include_expected_output:
                from .llm_as_a_judge import judge_input_output_expected

                grading_output = await judge_input_output_expected(
                    ctx.inputs, ctx.output, ctx.expected_output, self.rubric, self.model, self.model_settings
                )
            else:
                from .llm_as_a_judge import judge_input_output

                grading_output = await judge_input_output(
                    ctx.inputs, ctx.output, self.rubric, self.model, self.model_settings
                )
        else:
            if self.include_expected_output:
                from .llm_as_a_judge import judge_output_expected

                grading_output = await judge_output_expected(
                    ctx.output, ctx.expected_output, self.rubric, self.model, self.model_settings
                )
            else:
                from .llm_as_a_judge import judge_output

                grading_output = await judge_output(ctx.output, self.rubric, self.model, self.model_settings)

        output: dict[str, EvaluationScalar | EvaluationReason] = {}
        include_both = self.score is not False and self.assertion is not False
        evaluation_name = self.get_default_evaluation_name()

        if self.score is not False:
            default_name = f'{evaluation_name}_score' if include_both else evaluation_name
            _update_combined_output(output, grading_output.score, grading_output.reason, self.score, default_name)

        if self.assertion is not False:
            default_name = f'{evaluation_name}_pass' if include_both else evaluation_name
            _update_combined_output(output, grading_output.pass_, grading_output.reason, self.assertion, default_name)

        return output

    def build_serialization_arguments(self):
        result = super().build_serialization_arguments()
        # always serialize the model as a string when present; use its name if it's a KnownModelName
        if (model := result.get('model')) and isinstance(model, models.Model):  # pragma: no branch
            result['model'] = f'{model.system}:{model.model_name}'

        # Note: this may lead to confusion if you try to serialize-then-deserialize with a custom model.
        # I expect that is rare enough to be worth not solving yet, but common enough that we probably will want to
        # solve it eventually. I'm imagining some kind of model registry, but don't want to work out the details yet.
        return result

```

### MaxDuration

Bases: `Evaluator[object, object, object]`

Check if the execution time is under the specified maximum.

Source code in `pydantic_evals/pydantic_evals/evaluators/common.py`

```python
@dataclass(repr=False)
class MaxDuration(Evaluator[object, object, object]):
    """Check if the execution time is under the specified maximum."""

    seconds: float | timedelta

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        duration = timedelta(seconds=ctx.duration)
        seconds = self.seconds
        if not isinstance(seconds, timedelta):
            seconds = timedelta(seconds=seconds)
        return duration <= seconds

```

### OutputConfig

Bases: `TypedDict`

Configuration for the score and assertion outputs of the LLMJudge evaluator.

Source code in `pydantic_evals/pydantic_evals/evaluators/common.py`

```python
class OutputConfig(TypedDict, total=False):
    """Configuration for the score and assertion outputs of the LLMJudge evaluator."""

    evaluation_name: str
    include_reason: bool

```

### EvaluatorContext

Bases: `Generic[InputsT, OutputT, MetadataT]`

Context for evaluating a task execution.

An instance of this class is the sole input to all Evaluators. It contains all the information needed to evaluate the task execution, including inputs, outputs, metadata, and telemetry data.

Evaluators use this context to access the task inputs, actual output, expected output, and other information when evaluating the result of the task execution.

Example:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ExactMatch(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        # Use the context to access task inputs, outputs, and expected outputs
        return ctx.output == ctx.expected_output

```

Source code in `pydantic_evals/pydantic_evals/evaluators/context.py`

````python
@dataclass(kw_only=True)
class EvaluatorContext(Generic[InputsT, OutputT, MetadataT]):
    """Context for evaluating a task execution.

    An instance of this class is the sole input to all Evaluators. It contains all the information
    needed to evaluate the task execution, including inputs, outputs, metadata, and telemetry data.

    Evaluators use this context to access the task inputs, actual output, expected output, and other
    information when evaluating the result of the task execution.

    Example:
    ```python
    from dataclasses import dataclass

    from pydantic_evals.evaluators import Evaluator, EvaluatorContext


    @dataclass
    class ExactMatch(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            # Use the context to access task inputs, outputs, and expected outputs
            return ctx.output == ctx.expected_output
    ```
    """

    name: str | None
    """The name of the case."""
    inputs: InputsT
    """The inputs provided to the task for this case."""
    metadata: MetadataT | None
    """Metadata associated with the case, if provided. May be None if no metadata was specified."""
    expected_output: OutputT | None
    """The expected output for the case, if provided. May be None if no expected output was specified."""

    output: OutputT
    """The actual output produced by the task for this case."""
    duration: float
    """The duration of the task run for this case."""
    _span_tree: SpanTree | SpanTreeRecordingError = field(repr=False)
    """The span tree for the task run for this case.

    This will be `None` if `logfire.configure` has not been called.
    """

    attributes: dict[str, Any]
    """Attributes associated with the task run for this case.

    These can be set by calling `pydantic_evals.dataset.set_eval_attribute` in any code executed
    during the evaluation task."""
    metrics: dict[str, int | float]
    """Metrics associated with the task run for this case.

    These can be set by calling `pydantic_evals.dataset.increment_eval_metric` in any code executed
    during the evaluation task."""

    @property
    def span_tree(self) -> SpanTree:
        """Get the `SpanTree` for this task execution.

        The span tree is a graph where each node corresponds to an OpenTelemetry span recorded during the task
        execution, including timing information and any custom spans created during execution.

        Returns:
            The span tree for the task execution.

        Raises:
            SpanTreeRecordingError: If spans were not captured during execution of the task, e.g. due to not having
                the necessary dependencies installed.
        """
        if isinstance(self._span_tree, SpanTreeRecordingError):
            # In this case, there was a reason we couldn't record the SpanTree. We raise that now
            raise self._span_tree
        return self._span_tree

````

#### name

```python
name: str | None

```

The name of the case.

#### inputs

```python
inputs: InputsT

```

The inputs provided to the task for this case.

#### metadata

```python
metadata: MetadataT | None

```

Metadata associated with the case, if provided. May be None if no metadata was specified.

#### expected_output

```python
expected_output: OutputT | None

```

The expected output for the case, if provided. May be None if no expected output was specified.

#### output

```python
output: OutputT

```

The actual output produced by the task for this case.

#### duration

```python
duration: float

```

The duration of the task run for this case.

#### attributes

```python
attributes: dict[str, Any]

```

Attributes associated with the task run for this case.

These can be set by calling `pydantic_evals.dataset.set_eval_attribute` in any code executed during the evaluation task.

#### metrics

```python
metrics: dict[str, int | float]

```

Metrics associated with the task run for this case.

These can be set by calling `pydantic_evals.dataset.increment_eval_metric` in any code executed during the evaluation task.

#### span_tree

```python
span_tree: SpanTree

```

Get the `SpanTree` for this task execution.

The span tree is a graph where each node corresponds to an OpenTelemetry span recorded during the task execution, including timing information and any custom spans created during execution.

Returns:

| Type | Description | | --- | --- | | `SpanTree` | The span tree for the task execution. |

Raises:

| Type | Description | | --- | --- | | `SpanTreeRecordingError` | If spans were not captured during execution of the task, e.g. due to not having the necessary dependencies installed. |

### EvaluationReason

The result of running an evaluator with an optional explanation.

Contains a scalar value and an optional "reason" explaining the value.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `value` | `EvaluationScalar` | The scalar result of the evaluation (boolean, integer, float, or string). | *required* | | `reason` | `str | None` | An optional explanation of the evaluation result. | `None` |

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
@dataclass
class EvaluationReason:
    """The result of running an evaluator with an optional explanation.

    Contains a scalar value and an optional "reason" explaining the value.

    Args:
        value: The scalar result of the evaluation (boolean, integer, float, or string).
        reason: An optional explanation of the evaluation result.
    """

    value: EvaluationScalar
    reason: str | None = None

```

### EvaluationResult

Bases: `Generic[EvaluationScalarT]`

The details of an individual evaluation result.

Contains the name, value, reason, and source evaluator for a single evaluation.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str` | The name of the evaluation. | *required* | | `value` | `EvaluationScalarT` | The scalar result of the evaluation. | *required* | | `reason` | `str | None` | An optional explanation of the evaluation result. | *required* | | `source` | `EvaluatorSpec` | The spec of the evaluator that produced this result. | *required* |

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
@dataclass
class EvaluationResult(Generic[EvaluationScalarT]):
    """The details of an individual evaluation result.

    Contains the name, value, reason, and source evaluator for a single evaluation.

    Args:
        name: The name of the evaluation.
        value: The scalar result of the evaluation.
        reason: An optional explanation of the evaluation result.
        source: The spec of the evaluator that produced this result.
    """

    name: str
    value: EvaluationScalarT
    reason: str | None
    source: EvaluatorSpec

    def downcast(self, *value_types: type[T]) -> EvaluationResult[T] | None:
        """Attempt to downcast this result to a more specific type.

        Args:
            *value_types: The types to check the value against.

        Returns:
            A downcast version of this result if the value is an instance of one of the given types,
            otherwise None.
        """
        # Check if value matches any of the target types, handling bool as a special case
        for value_type in value_types:
            if isinstance(self.value, value_type):
                # Only match bool with explicit bool type
                if isinstance(self.value, bool) and value_type is not bool:
                    continue
                return cast(EvaluationResult[T], self)
        return None

```

#### downcast

```python
downcast(
    *value_types: type[T],
) -> EvaluationResult[T] | None

```

Attempt to downcast this result to a more specific type.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `*value_types` | `type[T]` | The types to check the value against. | `()` |

Returns:

| Type | Description | | --- | --- | | `EvaluationResult[T] | None` | A downcast version of this result if the value is an instance of one of the given types, | | `EvaluationResult[T] | None` | otherwise None. |

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
def downcast(self, *value_types: type[T]) -> EvaluationResult[T] | None:
    """Attempt to downcast this result to a more specific type.

    Args:
        *value_types: The types to check the value against.

    Returns:
        A downcast version of this result if the value is an instance of one of the given types,
        otherwise None.
    """
    # Check if value matches any of the target types, handling bool as a special case
    for value_type in value_types:
        if isinstance(self.value, value_type):
            # Only match bool with explicit bool type
            if isinstance(self.value, bool) and value_type is not bool:
                continue
            return cast(EvaluationResult[T], self)
    return None

```

### Evaluator

Bases: `Generic[InputsT, OutputT, MetadataT]`

Base class for all evaluators.

Evaluators can assess the performance of a task in a variety of ways, as a function of the EvaluatorContext.

Subclasses must implement the `evaluate` method. Note it can be defined with either `def` or `async def`.

Example:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ExactMatch(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output == ctx.expected_output

```

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

````python
@dataclass(repr=False)
class Evaluator(Generic[InputsT, OutputT, MetadataT], metaclass=_StrictABCMeta):
    """Base class for all evaluators.

    Evaluators can assess the performance of a task in a variety of ways, as a function of the EvaluatorContext.

    Subclasses must implement the `evaluate` method. Note it can be defined with either `def` or `async def`.

    Example:
    ```python
    from dataclasses import dataclass

    from pydantic_evals.evaluators import Evaluator, EvaluatorContext


    @dataclass
    class ExactMatch(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return ctx.output == ctx.expected_output
    ```
    """

    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the 'name' of this Evaluator to use during serialization.

        Returns:
            The name of the Evaluator, which is typically the class name.
        """
        return cls.__name__

    @classmethod
    @deprecated('`name` has been renamed, use `get_serialization_name` instead.')
    def name(cls) -> str:
        """`name` has been renamed, use `get_serialization_name` instead."""
        return cls.get_serialization_name()

    def get_default_evaluation_name(self) -> str:
        """Return the default name to use in reports for the output of this evaluator.

        By default, if the evaluator has an attribute called `evaluation_name` of type string, that will be used.
        Otherwise, the serialization name of the evaluator (which is usually the class name) will be used.

        This can be overridden to get a more descriptive name in evaluation reports, e.g. using instance information.

        Note that evaluators that return a mapping of results will always use the keys of that mapping as the names
        of the associated evaluation results.
        """
        evaluation_name = getattr(self, 'evaluation_name', None)
        if isinstance(evaluation_name, str):
            # If the evaluator has an attribute `name` of type string, use that
            return evaluation_name

        return self.get_serialization_name()

    @abstractmethod
    def evaluate(
        self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]
    ) -> EvaluatorOutput | Awaitable[EvaluatorOutput]:  # pragma: no cover
        """Evaluate the task output in the given context.

        This is the main evaluation method that subclasses must implement. It can be either synchronous
        or asynchronous, returning either an EvaluatorOutput directly or an Awaitable[EvaluatorOutput].

        Args:
            ctx: The context containing the inputs, outputs, and metadata for evaluation.

        Returns:
            The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
            of evaluation names to either of those. Can be returned either synchronously or as an
            awaitable for asynchronous evaluation.
        """
        raise NotImplementedError('You must implement `evaluate`.')

    def evaluate_sync(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        """Run the evaluator synchronously, handling both sync and async implementations.

        This method ensures synchronous execution by running any async evaluate implementation
        to completion using run_until_complete.

        Args:
            ctx: The context containing the inputs, outputs, and metadata for evaluation.

        Returns:
            The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
            of evaluation names to either of those.
        """
        output = self.evaluate(ctx)
        if inspect.iscoroutine(output):  # pragma: no cover
            return get_event_loop().run_until_complete(output)
        else:
            return cast(EvaluatorOutput, output)

    async def evaluate_async(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        """Run the evaluator asynchronously, handling both sync and async implementations.

        This method ensures asynchronous execution by properly awaiting any async evaluate
        implementation. For synchronous implementations, it returns the result directly.

        Args:
            ctx: The context containing the inputs, outputs, and metadata for evaluation.

        Returns:
            The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
            of evaluation names to either of those.
        """
        # Note: If self.evaluate is synchronous, but you need to prevent this from blocking, override this method with:
        # return await anyio.to_thread.run_sync(self.evaluate, ctx)
        output = self.evaluate(ctx)
        if inspect.iscoroutine(output):
            return await output
        else:
            return cast(EvaluatorOutput, output)

    @model_serializer(mode='plain')
    def serialize(self, info: SerializationInfo) -> Any:
        """Serialize this Evaluator to a JSON-serializable form.

        Returns:
            A JSON-serializable representation of this evaluator as an EvaluatorSpec.
        """
        return to_jsonable_python(
            self.as_spec(),
            context=info.context,
            serialize_unknown=True,
        )

    def as_spec(self) -> EvaluatorSpec:
        raw_arguments = self.build_serialization_arguments()

        arguments: None | tuple[Any,] | dict[str, Any]
        if len(raw_arguments) == 0:
            arguments = None
        elif len(raw_arguments) == 1:
            arguments = (next(iter(raw_arguments.values())),)
        else:
            arguments = raw_arguments

        return EvaluatorSpec(name=self.get_serialization_name(), arguments=arguments)

    def build_serialization_arguments(self) -> dict[str, Any]:
        """Build the arguments for serialization.

        Evaluators are serialized for inclusion as the "source" in an `EvaluationResult`.
        If you want to modify how the evaluator is serialized for that or other purposes, you can override this method.

        Returns:
            A dictionary of arguments to be used during serialization.
        """
        raw_arguments: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # always exclude defaults:
            if field.default is not MISSING:
                if value == field.default:
                    continue
            if field.default_factory is not MISSING:
                if value == field.default_factory():  # pragma: no branch
                    continue
            raw_arguments[field.name] = value
        return raw_arguments

    __repr__ = _utils.dataclasses_no_defaults_repr

````

#### get_serialization_name

```python
get_serialization_name() -> str

```

Return the 'name' of this Evaluator to use during serialization.

Returns:

| Type | Description | | --- | --- | | `str` | The name of the Evaluator, which is typically the class name. |

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
@classmethod
def get_serialization_name(cls) -> str:
    """Return the 'name' of this Evaluator to use during serialization.

    Returns:
        The name of the Evaluator, which is typically the class name.
    """
    return cls.__name__

```

#### name

```python
name() -> str

```

Deprecated

`name` has been renamed, use `get_serialization_name` instead.

`name` has been renamed, use `get_serialization_name` instead.

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
@classmethod
@deprecated('`name` has been renamed, use `get_serialization_name` instead.')
def name(cls) -> str:
    """`name` has been renamed, use `get_serialization_name` instead."""
    return cls.get_serialization_name()

```

#### get_default_evaluation_name

```python
get_default_evaluation_name() -> str

```

Return the default name to use in reports for the output of this evaluator.

By default, if the evaluator has an attribute called `evaluation_name` of type string, that will be used. Otherwise, the serialization name of the evaluator (which is usually the class name) will be used.

This can be overridden to get a more descriptive name in evaluation reports, e.g. using instance information.

Note that evaluators that return a mapping of results will always use the keys of that mapping as the names of the associated evaluation results.

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
def get_default_evaluation_name(self) -> str:
    """Return the default name to use in reports for the output of this evaluator.

    By default, if the evaluator has an attribute called `evaluation_name` of type string, that will be used.
    Otherwise, the serialization name of the evaluator (which is usually the class name) will be used.

    This can be overridden to get a more descriptive name in evaluation reports, e.g. using instance information.

    Note that evaluators that return a mapping of results will always use the keys of that mapping as the names
    of the associated evaluation results.
    """
    evaluation_name = getattr(self, 'evaluation_name', None)
    if isinstance(evaluation_name, str):
        # If the evaluator has an attribute `name` of type string, use that
        return evaluation_name

    return self.get_serialization_name()

```

#### evaluate

```python
evaluate(
    ctx: EvaluatorContext[InputsT, OutputT, MetadataT],
) -> EvaluatorOutput | Awaitable[EvaluatorOutput]

```

Evaluate the task output in the given context.

This is the main evaluation method that subclasses must implement. It can be either synchronous or asynchronous, returning either an EvaluatorOutput directly or an Awaitable[EvaluatorOutput].

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `EvaluatorContext[InputsT, OutputT, MetadataT]` | The context containing the inputs, outputs, and metadata for evaluation. | *required* |

Returns:

| Type | Description | | --- | --- | | `EvaluatorOutput | Awaitable[EvaluatorOutput]` | The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping | | `EvaluatorOutput | Awaitable[EvaluatorOutput]` | of evaluation names to either of those. Can be returned either synchronously or as an | | `EvaluatorOutput | Awaitable[EvaluatorOutput]` | awaitable for asynchronous evaluation. |

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
@abstractmethod
def evaluate(
    self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]
) -> EvaluatorOutput | Awaitable[EvaluatorOutput]:  # pragma: no cover
    """Evaluate the task output in the given context.

    This is the main evaluation method that subclasses must implement. It can be either synchronous
    or asynchronous, returning either an EvaluatorOutput directly or an Awaitable[EvaluatorOutput].

    Args:
        ctx: The context containing the inputs, outputs, and metadata for evaluation.

    Returns:
        The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
        of evaluation names to either of those. Can be returned either synchronously or as an
        awaitable for asynchronous evaluation.
    """
    raise NotImplementedError('You must implement `evaluate`.')

```

#### evaluate_sync

```python
evaluate_sync(
    ctx: EvaluatorContext[InputsT, OutputT, MetadataT],
) -> EvaluatorOutput

```

Run the evaluator synchronously, handling both sync and async implementations.

This method ensures synchronous execution by running any async evaluate implementation to completion using run_until_complete.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `EvaluatorContext[InputsT, OutputT, MetadataT]` | The context containing the inputs, outputs, and metadata for evaluation. | *required* |

Returns:

| Type | Description | | --- | --- | | `EvaluatorOutput` | The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping | | `EvaluatorOutput` | of evaluation names to either of those. |

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
def evaluate_sync(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
    """Run the evaluator synchronously, handling both sync and async implementations.

    This method ensures synchronous execution by running any async evaluate implementation
    to completion using run_until_complete.

    Args:
        ctx: The context containing the inputs, outputs, and metadata for evaluation.

    Returns:
        The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
        of evaluation names to either of those.
    """
    output = self.evaluate(ctx)
    if inspect.iscoroutine(output):  # pragma: no cover
        return get_event_loop().run_until_complete(output)
    else:
        return cast(EvaluatorOutput, output)

```

#### evaluate_async

```python
evaluate_async(
    ctx: EvaluatorContext[InputsT, OutputT, MetadataT],
) -> EvaluatorOutput

```

Run the evaluator asynchronously, handling both sync and async implementations.

This method ensures asynchronous execution by properly awaiting any async evaluate implementation. For synchronous implementations, it returns the result directly.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `ctx` | `EvaluatorContext[InputsT, OutputT, MetadataT]` | The context containing the inputs, outputs, and metadata for evaluation. | *required* |

Returns:

| Type | Description | | --- | --- | | `EvaluatorOutput` | The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping | | `EvaluatorOutput` | of evaluation names to either of those. |

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
async def evaluate_async(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
    """Run the evaluator asynchronously, handling both sync and async implementations.

    This method ensures asynchronous execution by properly awaiting any async evaluate
    implementation. For synchronous implementations, it returns the result directly.

    Args:
        ctx: The context containing the inputs, outputs, and metadata for evaluation.

    Returns:
        The evaluation result, which can be a scalar value, an EvaluationReason, or a mapping
        of evaluation names to either of those.
    """
    # Note: If self.evaluate is synchronous, but you need to prevent this from blocking, override this method with:
    # return await anyio.to_thread.run_sync(self.evaluate, ctx)
    output = self.evaluate(ctx)
    if inspect.iscoroutine(output):
        return await output
    else:
        return cast(EvaluatorOutput, output)

```

#### serialize

```python
serialize(info: SerializationInfo) -> Any

```

Serialize this Evaluator to a JSON-serializable form.

Returns:

| Type | Description | | --- | --- | | `Any` | A JSON-serializable representation of this evaluator as an EvaluatorSpec. |

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
@model_serializer(mode='plain')
def serialize(self, info: SerializationInfo) -> Any:
    """Serialize this Evaluator to a JSON-serializable form.

    Returns:
        A JSON-serializable representation of this evaluator as an EvaluatorSpec.
    """
    return to_jsonable_python(
        self.as_spec(),
        context=info.context,
        serialize_unknown=True,
    )

```

#### build_serialization_arguments

```python
build_serialization_arguments() -> dict[str, Any]

```

Build the arguments for serialization.

Evaluators are serialized for inclusion as the "source" in an `EvaluationResult`. If you want to modify how the evaluator is serialized for that or other purposes, you can override this method.

Returns:

| Type | Description | | --- | --- | | `dict[str, Any]` | A dictionary of arguments to be used during serialization. |

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
def build_serialization_arguments(self) -> dict[str, Any]:
    """Build the arguments for serialization.

    Evaluators are serialized for inclusion as the "source" in an `EvaluationResult`.
    If you want to modify how the evaluator is serialized for that or other purposes, you can override this method.

    Returns:
        A dictionary of arguments to be used during serialization.
    """
    raw_arguments: dict[str, Any] = {}
    for field in fields(self):
        value = getattr(self, field.name)
        # always exclude defaults:
        if field.default is not MISSING:
            if value == field.default:
                continue
        if field.default_factory is not MISSING:
            if value == field.default_factory():  # pragma: no branch
                continue
        raw_arguments[field.name] = value
    return raw_arguments

```

### EvaluatorFailure

Represents a failure raised during the execution of an evaluator.

Source code in `pydantic_evals/pydantic_evals/evaluators/evaluator.py`

```python
@dataclass
class EvaluatorFailure:
    """Represents a failure raised during the execution of an evaluator."""

    name: str
    error_message: str
    error_stacktrace: str
    source: EvaluatorSpec

```

### EvaluatorOutput

```python
EvaluatorOutput = (
    EvaluationScalar
    | EvaluationReason
    | Mapping[str, EvaluationScalar | EvaluationReason]
)

```

Type for the output of an evaluator, which can be a scalar, an EvaluationReason, or a mapping of names to either.

### EvaluatorSpec

Bases: `BaseModel`

The specification of an evaluator to be run.

This class is used to represent evaluators in a serializable format, supporting various short forms for convenience when defining evaluators in YAML or JSON dataset files.

In particular, each of the following forms is supported for specifying an evaluator with name `MyEvaluator`: * `'MyEvaluator'` - Just the (string) name of the Evaluator subclass is used if its `__init__` takes no arguments * `{'MyEvaluator': first_arg}` - A single argument is passed as the first positional argument to `MyEvaluator.__init__` * `{'MyEvaluator': {k1: v1, k2: v2}}` - Multiple kwargs are passed to `MyEvaluator.__init__`

Source code in `pydantic_evals/pydantic_evals/evaluators/spec.py`

```python
class EvaluatorSpec(BaseModel):
    """The specification of an evaluator to be run.

    This class is used to represent evaluators in a serializable format, supporting various
    short forms for convenience when defining evaluators in YAML or JSON dataset files.

    In particular, each of the following forms is supported for specifying an evaluator with name `MyEvaluator`:
    * `'MyEvaluator'` - Just the (string) name of the Evaluator subclass is used if its `__init__` takes no arguments
    * `{'MyEvaluator': first_arg}` - A single argument is passed as the first positional argument to `MyEvaluator.__init__`
    * `{'MyEvaluator': {k1: v1, k2: v2}}` - Multiple kwargs are passed to `MyEvaluator.__init__`
    """

    name: str
    """The name of the evaluator class; should be the value returned by `EvaluatorClass.get_serialization_name()`"""

    arguments: None | tuple[Any] | dict[str, Any]
    """The arguments to pass to the evaluator's constructor.

    Can be None (no arguments), a tuple (a single positional argument), or a dict (keyword arguments).
    """

    @property
    def args(self) -> tuple[Any, ...]:
        """Get the positional arguments for the evaluator.

        Returns:
            A tuple of positional arguments if arguments is a tuple, otherwise an empty tuple.
        """
        if isinstance(self.arguments, tuple):
            return self.arguments
        return ()

    @property
    def kwargs(self) -> dict[str, Any]:
        """Get the keyword arguments for the evaluator.

        Returns:
            A dictionary of keyword arguments if arguments is a dict, otherwise an empty dict.
        """
        if isinstance(self.arguments, dict):
            return self.arguments
        return {}

    @model_validator(mode='wrap')
    @classmethod
    def deserialize(cls, value: Any, handler: ModelWrapValidatorHandler[EvaluatorSpec]) -> EvaluatorSpec:
        """Deserialize an EvaluatorSpec from various formats.

        This validator handles the various short forms of evaluator specifications,
        converting them to a consistent EvaluatorSpec instance.

        Args:
            value: The value to deserialize.
            handler: The validator handler.

        Returns:
            The deserialized EvaluatorSpec.

        Raises:
            ValidationError: If the value cannot be deserialized.
        """
        try:
            result = handler(value)
            return result
        except ValidationError as exc:
            try:
                deserialized = _SerializedEvaluatorSpec.model_validate(value)
            except ValidationError:
                raise exc  # raise the original error
            return deserialized.to_evaluator_spec()

    @model_serializer(mode='wrap')
    def serialize(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> Any:
        """Serialize using the appropriate short-form if possible.

        Returns:
            The serialized evaluator specification, using the shortest form possible:
            - Just the name if there are no arguments
            - {name: first_arg} if there's a single positional argument
            - {name: {kwargs}} if there are multiple (keyword) arguments
        """
        if isinstance(info.context, dict) and info.context.get('use_short_form'):  # pyright: ignore[reportUnknownMemberType]
            if self.arguments is None:
                return self.name
            elif isinstance(self.arguments, tuple):
                return {self.name: self.arguments[0]}
            else:
                return {self.name: self.arguments}
        else:
            return handler(self)

```

#### name

```python
name: str

```

The name of the evaluator class; should be the value returned by `EvaluatorClass.get_serialization_name()`

#### arguments

```python
arguments: None | tuple[Any] | dict[str, Any]

```

The arguments to pass to the evaluator's constructor.

Can be None (no arguments), a tuple (a single positional argument), or a dict (keyword arguments).

#### args

```python
args: tuple[Any, ...]

```

Get the positional arguments for the evaluator.

Returns:

| Type | Description | | --- | --- | | `tuple[Any, ...]` | A tuple of positional arguments if arguments is a tuple, otherwise an empty tuple. |

#### kwargs

```python
kwargs: dict[str, Any]

```

Get the keyword arguments for the evaluator.

Returns:

| Type | Description | | --- | --- | | `dict[str, Any]` | A dictionary of keyword arguments if arguments is a dict, otherwise an empty dict. |

#### deserialize

```python
deserialize(
    value: Any,
    handler: ModelWrapValidatorHandler[EvaluatorSpec],
) -> EvaluatorSpec

```

Deserialize an EvaluatorSpec from various formats.

This validator handles the various short forms of evaluator specifications, converting them to a consistent EvaluatorSpec instance.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `value` | `Any` | The value to deserialize. | *required* | | `handler` | `ModelWrapValidatorHandler[EvaluatorSpec]` | The validator handler. | *required* |

Returns:

| Type | Description | | --- | --- | | `EvaluatorSpec` | The deserialized EvaluatorSpec. |

Raises:

| Type | Description | | --- | --- | | `ValidationError` | If the value cannot be deserialized. |

Source code in `pydantic_evals/pydantic_evals/evaluators/spec.py`

```python
@model_validator(mode='wrap')
@classmethod
def deserialize(cls, value: Any, handler: ModelWrapValidatorHandler[EvaluatorSpec]) -> EvaluatorSpec:
    """Deserialize an EvaluatorSpec from various formats.

    This validator handles the various short forms of evaluator specifications,
    converting them to a consistent EvaluatorSpec instance.

    Args:
        value: The value to deserialize.
        handler: The validator handler.

    Returns:
        The deserialized EvaluatorSpec.

    Raises:
        ValidationError: If the value cannot be deserialized.
    """
    try:
        result = handler(value)
        return result
    except ValidationError as exc:
        try:
            deserialized = _SerializedEvaluatorSpec.model_validate(value)
        except ValidationError:
            raise exc  # raise the original error
        return deserialized.to_evaluator_spec()

```

#### serialize

```python
serialize(
    handler: SerializerFunctionWrapHandler,
    info: SerializationInfo,
) -> Any

```

Serialize using the appropriate short-form if possible.

Returns:

| Type | Description | | --- | --- | | `Any` | The serialized evaluator specification, using the shortest form possible: | | `Any` | Just the name if there are no arguments | | `Any` | {name: first_arg} if there's a single positional argument | | `Any` | {name: {kwargs}} if there are multiple (keyword) arguments |

Source code in `pydantic_evals/pydantic_evals/evaluators/spec.py`

```python
@model_serializer(mode='wrap')
def serialize(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> Any:
    """Serialize using the appropriate short-form if possible.

    Returns:
        The serialized evaluator specification, using the shortest form possible:
        - Just the name if there are no arguments
        - {name: first_arg} if there's a single positional argument
        - {name: {kwargs}} if there are multiple (keyword) arguments
    """
    if isinstance(info.context, dict) and info.context.get('use_short_form'):  # pyright: ignore[reportUnknownMemberType]
        if self.arguments is None:
            return self.name
        elif isinstance(self.arguments, tuple):
            return {self.name: self.arguments[0]}
        else:
            return {self.name: self.arguments}
    else:
        return handler(self)

```

### GradingOutput

Bases: `BaseModel`

The output of a grading operation.

Source code in `pydantic_evals/pydantic_evals/evaluators/llm_as_a_judge.py`

```python
class GradingOutput(BaseModel, populate_by_name=True):
    """The output of a grading operation."""

    reason: str
    pass_: bool = Field(validation_alias='pass', serialization_alias='pass')
    score: float

```

### judge_output

```python
judge_output(
    output: Any,
    rubric: str,
    model: Model | KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput

```

Judge the output of a model based on a rubric.

If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o', but this can be changed using the `set_default_judge_model` function.

Source code in `pydantic_evals/pydantic_evals/evaluators/llm_as_a_judge.py`

```python
async def judge_output(
    output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
    but this can be changed using the `set_default_judge_model` function.
    """
    user_prompt = _build_prompt(output=output, rubric=rubric)
    return (
        await _judge_output_agent.run(user_prompt, model=model or _default_model, model_settings=model_settings)
    ).output

```

### judge_input_output

```python
judge_input_output(
    inputs: Any,
    output: Any,
    rubric: str,
    model: Model | KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput

```

Judge the output of a model based on the inputs and a rubric.

If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o', but this can be changed using the `set_default_judge_model` function.

Source code in `pydantic_evals/pydantic_evals/evaluators/llm_as_a_judge.py`

```python
async def judge_input_output(
    inputs: Any,
    output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on the inputs and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
    but this can be changed using the `set_default_judge_model` function.
    """
    user_prompt = _build_prompt(inputs=inputs, output=output, rubric=rubric)

    return (
        await _judge_input_output_agent.run(user_prompt, model=model or _default_model, model_settings=model_settings)
    ).output

```

### judge_input_output_expected

```python
judge_input_output_expected(
    inputs: Any,
    output: Any,
    expected_output: Any,
    rubric: str,
    model: Model | KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput

```

Judge the output of a model based on the inputs and a rubric.

If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o', but this can be changed using the `set_default_judge_model` function.

Source code in `pydantic_evals/pydantic_evals/evaluators/llm_as_a_judge.py`

```python
async def judge_input_output_expected(
    inputs: Any,
    output: Any,
    expected_output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on the inputs and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
    but this can be changed using the `set_default_judge_model` function.
    """
    user_prompt = _build_prompt(inputs=inputs, output=output, rubric=rubric, expected_output=expected_output)

    return (
        await _judge_input_output_expected_agent.run(
            user_prompt, model=model or _default_model, model_settings=model_settings
        )
    ).output

```

### judge_output_expected

```python
judge_output_expected(
    output: Any,
    expected_output: Any,
    rubric: str,
    model: Model | KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput

```

Judge the output of a model based on the expected output, output, and a rubric.

If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o', but this can be changed using the `set_default_judge_model` function.

Source code in `pydantic_evals/pydantic_evals/evaluators/llm_as_a_judge.py`

```python
async def judge_output_expected(
    output: Any,
    expected_output: Any,
    rubric: str,
    model: models.Model | models.KnownModelName | None = None,
    model_settings: ModelSettings | None = None,
) -> GradingOutput:
    """Judge the output of a model based on the expected output, output, and a rubric.

    If the model is not specified, a default model is used. The default model starts as 'openai:gpt-4o',
    but this can be changed using the `set_default_judge_model` function.
    """
    user_prompt = _build_prompt(output=output, rubric=rubric, expected_output=expected_output)
    return (
        await _judge_output_expected_agent.run(
            user_prompt, model=model or _default_model, model_settings=model_settings
        )
    ).output

```

### set_default_judge_model

```python
set_default_judge_model(
    model: Model | KnownModelName,
) -> None

```

Set the default model used for judging.

This model is used if `None` is passed to the `model` argument of `judge_output` and `judge_input_output`.

Source code in `pydantic_evals/pydantic_evals/evaluators/llm_as_a_judge.py`

```python
def set_default_judge_model(model: models.Model | models.KnownModelName) -> None:
    """Set the default model used for judging.

    This model is used if `None` is passed to the `model` argument of `judge_output` and `judge_input_output`.
    """
    global _default_model
    _default_model = model

```

# `pydantic_evals.generation`

Utilities for generating example datasets for pydantic_evals.

This module provides functions for generating sample datasets for testing and examples, using LLMs to create realistic test data with proper structure.

### generate_dataset

```python
generate_dataset(
    *,
    dataset_type: type[
        Dataset[InputsT, OutputT, MetadataT]
    ],
    path: Path | str | None = None,
    custom_evaluator_types: Sequence[
        type[Evaluator[InputsT, OutputT, MetadataT]]
    ] = (),
    model: Model | KnownModelName = "openai:gpt-4o",
    n_examples: int = 3,
    extra_instructions: str | None = None
) -> Dataset[InputsT, OutputT, MetadataT]

```

Use an LLM to generate a dataset of test cases, each consisting of input, expected output, and metadata.

This function creates a properly structured dataset with the specified input, output, and metadata types. It uses an LLM to attempt to generate realistic test cases that conform to the types' schemas.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `path` | `Path | str | None` | Optional path to save the generated dataset. If provided, the dataset will be saved to this location. | `None` | | `dataset_type` | `type[Dataset[InputsT, OutputT, MetadataT]]` | The type of dataset to generate, with the desired input, output, and metadata types. | *required* | | `custom_evaluator_types` | `Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]]` | Optional sequence of custom evaluator classes to include in the schema. | `()` | | `model` | `Model | KnownModelName` | The Pydantic AI model to use for generation. Defaults to 'gpt-4o'. | `'openai:gpt-4o'` | | `n_examples` | `int` | Number of examples to generate. Defaults to 3. | `3` | | `extra_instructions` | `str | None` | Optional additional instructions to provide to the LLM. | `None` |

Returns:

| Type | Description | | --- | --- | | `Dataset[InputsT, OutputT, MetadataT]` | A properly structured Dataset object with generated test cases. |

Raises:

| Type | Description | | --- | --- | | `ValidationError` | If the LLM's response cannot be parsed as a valid dataset. |

Source code in `pydantic_evals/pydantic_evals/generation.py`

```python
async def generate_dataset(
    *,
    dataset_type: type[Dataset[InputsT, OutputT, MetadataT]],
    path: Path | str | None = None,
    custom_evaluator_types: Sequence[type[Evaluator[InputsT, OutputT, MetadataT]]] = (),
    model: models.Model | models.KnownModelName = 'openai:gpt-4o',
    n_examples: int = 3,
    extra_instructions: str | None = None,
) -> Dataset[InputsT, OutputT, MetadataT]:
    """Use an LLM to generate a dataset of test cases, each consisting of input, expected output, and metadata.

    This function creates a properly structured dataset with the specified input, output, and metadata types.
    It uses an LLM to attempt to generate realistic test cases that conform to the types' schemas.

    Args:
        path: Optional path to save the generated dataset. If provided, the dataset will be saved to this location.
        dataset_type: The type of dataset to generate, with the desired input, output, and metadata types.
        custom_evaluator_types: Optional sequence of custom evaluator classes to include in the schema.
        model: The Pydantic AI model to use for generation. Defaults to 'gpt-4o'.
        n_examples: Number of examples to generate. Defaults to 3.
        extra_instructions: Optional additional instructions to provide to the LLM.

    Returns:
        A properly structured Dataset object with generated test cases.

    Raises:
        ValidationError: If the LLM's response cannot be parsed as a valid dataset.
    """
    output_schema = dataset_type.model_json_schema_with_evaluators(custom_evaluator_types)

    # TODO: Use `output_type=StructuredDict(output_schema)` (and `from_dict` below) once https://github.com/pydantic/pydantic/issues/12145
    # is fixed and `StructuredDict` no longer needs to use `InlineDefsJsonSchemaTransformer`.
    agent = Agent(
        model,
        system_prompt=(
            f'Generate an object that is in compliance with this JSON schema:\n{output_schema}\n\n'
            f'Include {n_examples} example cases.'
            ' You must not include any characters in your response before the opening { of the JSON object, or after the closing }.'
        ),
        output_type=str,
        retries=1,
    )

    result = await agent.run(extra_instructions or 'Please generate the object.')
    output = strip_markdown_fences(result.output)
    try:
        result = dataset_type.from_text(output, fmt='json', custom_evaluator_types=custom_evaluator_types)
    except ValidationError as e:  # pragma: no cover
        print(f'Raw response from model:\n{result.output}')
        raise e
    if path is not None:
        result.to_file(path, custom_evaluator_types=custom_evaluator_types)  # pragma: no cover
    return result

```

