<!-- Source: _complete-doc.md -->

# Evals

# Pydantic Evals

**Pydantic Evals** is a powerful evaluation framework for systematically testing and evaluating AI systems, from simple LLM calls to complex multi-agent applications.

## Design Philosophy

Code-First Approach

Pydantic Evals follows a code-first philosophy where all evaluation components are defined in Python. This differs from platforms with web-based configuration. You write and run evals in code, and can write the results to disk or view them in your terminal or in [Pydantic Logfire](https://logfire.pydantic.dev/docs/guides/web-ui/evals/).

Evals are an Emerging Practice

Unlike unit tests, evals are an emerging art/science. Anyone who claims to know exactly how your evals should be defined can safely be ignored. We've designed Pydantic Evals to be flexible and useful without being too opinionated.

## Quick Navigation

**Getting Started:**

- [Installation](#installation)
- [Quick Start](quick-start/)
- [Core Concepts](core-concepts/)

**Evaluators:**

- [Evaluators Overview](evaluators/overview/) - Compare evaluator types and learn when to use each approach
- [Built-in Evaluators](evaluators/built-in/) - Complete reference for exact match, instance checks, and other ready-to-use evaluators
- [LLM as a Judge](evaluators/llm-judge/) - Use LLMs to evaluate subjective qualities, complex criteria, and natural language outputs
- [Custom Evaluators](evaluators/custom/) - Implement domain-specific scoring logic and custom evaluation metrics
- [Span-Based Evaluation](evaluators/span-based/) - Evaluate internal agent behavior (tool calls, execution flow) using OpenTelemetry traces. Essential for complex agents where correctness depends on *how* the answer was reached, not just the final output. Also ensures eval assertions align with production telemetry.

**How-To Guides:**

- [Logfire Integration](how-to/logfire-integration/) - Visualize results
- [Dataset Management](how-to/dataset-management/) - Save, load, generate
- [Concurrency & Performance](how-to/concurrency/) - Control parallel execution
- [Retry Strategies](how-to/retry-strategies/) - Handle transient failures
- [Metrics & Attributes](how-to/metrics-attributes/) - Track custom data

**Examples:**

- [Simple Validation](examples/simple-validation/) - Basic example

**Reference:**

- [API Documentation](../api/pydantic_evals/dataset/)

## Code-First Evaluation

Pydantic Evals follows a **code-first approach** where you define all evaluation components (datasets, experiments, tasks, cases and evaluators) in Python code, or as serialized data loaded by Python code. This differs from platforms with fully web-based configuration.

When you run an *Experiment* you'll see a progress indicator and can print the results wherever you run your python code (IDE, terminal, etc). You also get a report object back that you can serialize and store or send to a notebook or other application for further visualization and analysis.

If you are using [Pydantic Logfire](https://logfire.pydantic.dev/docs/guides/web-ui/evals/), your experiment results automatically appear in the Logfire web interface for visualization, comparison, and collaborative analysis. Logfire serves as a observability layer - you write and run evals in code, then view and analyze results in the web UI.

## Installation

To install the Pydantic Evals package, run:

```bash
pip install pydantic-evals

```

```bash
uv add pydantic-evals

```

`pydantic-evals` does not depend on `pydantic-ai`, but has an optional dependency on `logfire` if you'd like to use OpenTelemetry traces in your evals, or send evaluation results to [logfire](https://pydantic.dev/logfire).

```bash
pip install 'pydantic-evals[logfire]'

```

```bash
uv add 'pydantic-evals[logfire]'

```

## Pydantic Evals Data Model

Pydantic Evals is built around a simple data model:

### Data Model Diagram

```text
Dataset (1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ (Many) Case
â”‚                        â”‚
â”‚                        â”‚
â””â”€â”€â”€ (Many) Experiment â”€â”€â”´â”€â”€â”€ (Many) Case results
     â”‚
     â””â”€â”€â”€ (1) Task
     â”‚
     â””â”€â”€â”€ (Many) Evaluator

```

### Key Relationships

1. **Dataset â†’ Cases**: One Dataset contains many Cases
1. **Dataset â†’ Experiments**: One Dataset can be used across many Experiments over time
1. **Experiment â†’ Case results**: One Experiment generates results by executing each Case
1. **Experiment â†’ Task**: One Experiment evaluates one defined Task
1. **Experiment â†’ Evaluators**: One Experiment uses multiple Evaluators. Dataset-wide Evaluators are run against all Cases, and Case-specific Evaluators against their respective Cases

### Data Flow

1. **Dataset creation**: Define cases and evaluators in YAML/JSON, or directly in Python
1. **Experiment execution**: Run `dataset.evaluate_sync(task_function)`
1. **Cases run**: Each Case is executed against the Task
1. **Evaluation**: Evaluators score the Task outputs for each Case
1. **Results**: All Case results are collected into a summary report

A metaphor

A useful metaphor (although not perfect) is to think of evals like a **Unit Testing** framework:

- **Cases + Evaluators** are your individual unit tests - each one defines a specific scenario you want to test, complete with inputs and expected outcomes. Just like a unit test, a case asks: *"Given this input, does my system produce the right output?"*
- **Datasets** are like test suites - they are the scaffolding that holds your unit tests together. They group related cases and define shared evaluation criteria that should apply across all tests in the suite.
- **Experiments** are like running your entire test suite and getting a report. When you execute `dataset.evaluate_sync(my_ai_function)`, you're running all your cases against your AI system and collecting the results - just like running `pytest` and getting a summary of passes, failures, and performance metrics.

The key difference from traditional unit testing is that AI systems are probabilistic. If you're type checking you'll still get a simple pass/fail, but scores for text outputs are likely qualitative and/or categorical, and more open to interpretation.

For a deeper understanding, see [Core Concepts](core-concepts/).

## Datasets and Cases

In Pydantic Evals, everything begins with Datasets and Cases:

- **Dataset**: A collection of test Cases designed for the evaluation of a specific task or function
- **Case**: A single test scenario corresponding to Task inputs, with optional expected outputs, metadata, and case-specific evaluators

simple_eval_dataset.py

```python
from pydantic_evals import Case, Dataset

case1 = Case(
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)

dataset = Dataset(cases=[case1])

```

*(This example is complete, it can be run "as is")*

See [Dataset Management](how-to/dataset-management/) to learn about saving, loading, and generating datasets.

## Evaluators

Evaluators analyze and score the results of your Task when tested against a Case.

These can be deterministic, code-based checks (such as testing model output format with a regex, or checking for the appearance of PII or sensitive data), or they can assess non-deterministic model outputs for qualities like accuracy, precision/recall, hallucinations, or instruction-following.

While both kinds of testing are useful in LLM systems, classical code-based tests are cheaper and easier than tests which require either human or machine review of model outputs.

Pydantic Evals includes several [built-in evaluators](evaluators/built-in/) and allows you to define [custom evaluators](evaluators/custom/):

simple_eval_evaluator.py

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.common import IsInstance

from simple_eval_dataset import dataset

dataset.add_evaluator(IsInstance(type_name='str'))  # (1)!


@dataclass
class MyEvaluator(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:  # (2)!
        if ctx.output == ctx.expected_output:
            return 1.0
        elif (
            isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 0.8
        else:
            return 0.0


dataset.add_evaluator(MyEvaluator())

```

1. You can add built-in evaluators to a dataset using the add_evaluator method.
1. This custom evaluator returns a simple score based on whether the output matches the expected output.

*(This example is complete, it can be run "as is")*

Learn more:

- [Evaluators Overview](evaluators/overview/) - When to use different types
- [Built-in Evaluators](evaluators/built-in/) - Complete reference
- [LLM Judge](evaluators/llm-judge/) - Using LLMs as evaluators
- [Custom Evaluators](evaluators/custom/) - Write your own logic
- [Span-Based Evaluation](evaluators/span-based/) - Analyze execution traces

## Running Experiments

Performing evaluations involves running a task against all cases in a dataset, also known as running an "experiment".

Putting the above two examples together and using the more declarative `evaluators` kwarg to Dataset:

simple_eval_complete.py

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

case1 = Case(  # (1)!
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)


class MyEvaluator(Evaluator[str, str]):
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if ctx.output == ctx.expected_output:
            return 1.0
        elif (
            isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 0.8
        else:
            return 0.0


dataset = Dataset(
    cases=[case1],
    evaluators=[IsInstance(type_name='str'), MyEvaluator()],  # (2)!
)


async def guess_city(question: str) -> str:  # (3)!
    return 'Paris'


report = dataset.evaluate_sync(guess_city)  # (4)!
report.print(include_input=True, include_output=True, include_durations=False)  # (5)!
"""
                              Evaluation Summary: guess_city
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”³â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”³â”â”â”â”â”â”â”â”â”â”³â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”³â”â”â”â”â”â”â”â”â”â”â”â”â”“
â”ƒ Case ID     â”ƒ Inputs                         â”ƒ Outputs â”ƒ Scores            â”ƒ Assertions â”ƒ
â”¡â”â”â”â”â”â”â”â”â”â”â”â”â”â•‡â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â•‡â”â”â”â”â”â”â”â”â”â•‡â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â•‡â”â”â”â”â”â”â”â”â”â”â”â”â”©
â”‚ simple_case â”‚ What is the capital of France? â”‚ Paris   â”‚ MyEvaluator: 1.00 â”‚ âœ”          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚ Averages    â”‚                                â”‚         â”‚ MyEvaluator: 1.00 â”‚ 100.0% âœ”   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
"""

```

1. Create a test case as above
1. Create a Dataset with test cases and evaluators
1. Our function to evaluate.
1. Run the evaluation with evaluate_sync, which runs the function against all test cases in the dataset, and returns an EvaluationReport object.
1. Print the report with print, which shows the results of the evaluation. We have omitted duration here just to keep the printed output from changing from run to run.

*(This example is complete, it can be run "as is")*

See [Quick Start](quick-start/) for more examples and [Concurrency & Performance](how-to/concurrency/) to learn about controlling parallel execution.

## API Reference

For comprehensive coverage of all classes, methods, and configuration options, see the detailed [API Reference documentation](https://ai.pydantic.dev/api/pydantic_evals/dataset/).

## Next Steps

1. **Start with simple evaluations** using [Quick Start](quick-start/)
1. **Understand the data model** with [Core Concepts](core-concepts/)
1. **Explore built-in evaluators** in [Built-in Evaluators](evaluators/built-in/)
1. **Integrate with Logfire** for visualization: [Logfire Integration](how-to/logfire-integration/)
1. **Build comprehensive test suites** with [Dataset Management](how-to/dataset-management/)
1. **Implement custom evaluators** for domain-specific metrics: [Custom Evaluators](evaluators/custom/)
# Durable Execution

# Durable Execution with DBOS

[DBOS](https://www.dbos.dev/) is a lightweight [durable execution](https://docs.dbos.dev/architecture) library natively integrated with Pydantic AI.

## Durable Execution

DBOS workflows make your program **durable** by checkpointing its state in a database. If your program ever fails, when it restarts all your workflows will automatically resume from the last completed step.

- **Workflows** must be deterministic and generally cannot include I/O.
- **Steps** may perform I/O (network, disk, API calls). If a step fails, it restarts from the beginning.

Every workflow input and step output is durably stored in the system database. When workflow execution fails, whether from crashes, network issues, or server restarts, DBOS leverages these checkpoints to recover workflows from their last completed step.

DBOS **queues** provide durable, database-backed alternatives to systems like Celery or BullMQ, supporting features such as concurrency limits, rate limits, timeouts, and prioritization. See the [DBOS docs](https://docs.dbos.dev/architecture) for details.

The diagram below shows the overall architecture of an agentic application in DBOS. DBOS runs fully in-process as a library. Functions remain normal Python functions but are checkpointed into a database (Postgres or SQLite).

```text
                    Clients
            (HTTP, RPC, Kafka, etc.)
                        |
                        v
+------------------------------------------------------+
|               Application Servers                    |
|                                                      |
|   +----------------------------------------------+   |
|   |        Pydantic AI + DBOS Libraries          |   |
|   |                                              |   |
|   |  [ Workflows (Agent Run Loop) ]              |   |
|   |  [ Steps (Tool, MCP, Model) ]                |   |
|   |  [ Queues ]   [ Cron Jobs ]   [ Messaging ]  |   |
|   +----------------------------------------------+   |
|                                                      |
+------------------------------------------------------+
                        |
                        v
+------------------------------------------------------+
|                      Database                        |
|   (Stores workflow and step state, schedules tasks)  |
+------------------------------------------------------+

```

See the [DBOS documentation](https://docs.dbos.dev/architecture) for more information.

## Durable Agent

Any agent can be wrapped in a DBOSAgent to get durable execution. `DBOSAgent` automatically:,

- Wraps `Agent.run` and `Agent.run_sync` as DBOS workflows.
- Wraps [model requests](../../models/overview/) and [MCP communication](../../mcp/client/) as DBOS steps.

Custom tool functions and event stream handlers are **not automatically wrapped** by DBOS. If they involve non-deterministic behavior or perform I/O, you should explicitly decorate them with `@DBOS.step`.

The original agent, model, and MCP server can still be used as normal outside the DBOS workflow.

Here is a simple but complete example of wrapping an agent for durable execution. All it requires is to install Pydantic AI with the DBOS [open-source library](https://github.com/dbos-inc/dbos-transact-py):

```bash
pip install pydantic-ai[dbos]

```

```bash
uv add pydantic-ai[dbos]

```

Or if you're using the slim package, you can install it with the `dbos` optional group:

```bash
pip install pydantic-ai-slim[dbos]

```

```bash
uv add pydantic-ai-slim[dbos]

```

dbos_agent.py

```python
from dbos import DBOS, DBOSConfig

from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSAgent

dbos_config: DBOSConfig = {
    'name': 'pydantic_dbos_agent',
    'system_database_url': 'sqlite:///dbostest.sqlite',  # (3)!
}
DBOS(config=dbos_config)

agent = Agent(
    'gpt-5',
    instructions="You're an expert in geography.",
    name='geography',  # (4)!
)

dbos_agent = DBOSAgent(agent)  # (1)!

async def main():
    DBOS.launch()
    result = await dbos_agent.run('What is the capital of Mexico?')  # (2)!
    print(result.output)
    #> Mexico City (Ciudad de MÃ©xico, CDMX)

```

1. Workflows and `DBOSAgent` must be defined before `DBOS.launch()` so that recovery can correctly find all workflows.
1. DBOSAgent.run() works like Agent.run(), but runs as a DBOS workflow and executes model requests, decorated tool calls, and MCP communication as DBOS steps.
1. This example uses SQLite. Postgres is recommended for production.
1. The agent's `name` is used to uniquely identify its workflows.

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

Because DBOS workflows need to be defined before calling `DBOS.launch()` and the `DBOSAgent` instance automatically registers `run` and `run_sync` as workflows, it needs to be defined before calling `DBOS.launch()` as well.

For more information on how to use DBOS in Python applications, see their [Python SDK guide](https://docs.dbos.dev/python/programming-guide).

## DBOS Integration Considerations

When using DBOS with Pydantic AI agents, there are a few important considerations to ensure workflows and toolsets behave correctly.

### Agent and Toolset Requirements

Each agent instance must have a unique `name` so DBOS can correctly resume workflows after a failure or restart.

Tools and event stream handlers are not automatically wrapped by DBOS. You can decide how to integrate them:

- Decorate with `@DBOS.step` if the function involves non-determinism or I/O.
- Skip the decorator if durability isn't needed, so you avoid the extra DB checkpoint write.
- If the function needs to enqueue tasks or invoke other DBOS workflows, run it inside the agent's main workflow (not as a step).

Other than that, any agent and toolset will just work!

### Agent Run Context and Dependencies

DBOS checkpoints workflow inputs/outputs and step outputs into a database using [`pickle`](https://docs.python.org/3/library/pickle.html). This means you need to make sure [dependencies](../../dependencies/) object provided to DBOSAgent.run() or DBOSAgent.run_sync(), and tool outputs can be serialized using pickle. You may also want to keep the inputs and outputs small (under ~2 MB). PostgreSQL and SQLite support up to 1 GB per field, but large objects may impact performance.

### Streaming

Because DBOS cannot stream output directly to the workflow or step call site, Agent.run_stream() and Agent.run_stream_events() are not supported when running inside of a DBOS workflow.

Instead, you can implement streaming by setting an event_stream_handler on the `Agent` or `DBOSAgent` instance and using DBOSAgent.run(). The event stream handler function will receive the agent run context and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](../../agents/#streaming-all-events).

## Step Configuration

You can customize DBOS step behavior, such as retries, by passing StepConfig objects to the `DBOSAgent` constructor:

- `mcp_step_config`: The DBOS step config to use for MCP server communication. No retries if omitted.
- `model_step_config`: The DBOS step config to use for model request steps. No retries if omitted.

For custom tools, you can annotate them directly with [`@DBOS.step`](https://docs.dbos.dev/python/reference/decorators#step) or [`@DBOS.workflow`](https://docs.dbos.dev/python/reference/decorators#workflow) decorators as needed. These decorators have no effect outside DBOS workflows, so tools remain usable in non-DBOS agents.

## Step Retries

On top of the automatic retries for request failures that DBOS will perform, Pydantic AI and various provider API clients also have their own request retry logic. Enabling these at the same time may cause the request to be retried more often than expected, with improper `Retry-After` handling.

When using DBOS, it's recommended to not use [HTTP Request Retries](../../retries/) and to turn off your provider API client's own retry logic, for example by setting `max_retries=0` on a [custom `OpenAIProvider` API client](../../models/openai/#custom-openai-client).

You can customize DBOS's retry policy using [step configuration](#step-configuration).

## Observability with Logfire

DBOS can be configured to generate OpenTelemetry spans for each workflow and step execution, and Pydantic AI emits spans for each agent run, model request, and tool invocation. You can send these spans to [Pydantic Logfire](../../logfire/) to get a full, end-to-end view of what's happening in your application.

For more information about DBOS logging and tracing, please see the [DBOS docs](https://docs.dbos.dev/python/tutorials/logging-and-tracing) for details.

# Durable Execution

Pydantic AI allows you to build durable agents that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. Durable agents have full support for [streaming](../../agents/#streaming-all-events) and [MCP](../../mcp/client/), with the added benefit of fault tolerance.

Pydantic AI natively supports three durable execution solutions:

- [Temporal](../temporal/)
- [DBOS](../dbos/)
- [Prefect](../prefect/)

These integrations only use Pydantic AI's public interface, so they also serve as a reference for integrating with other durable systems.

# Durable Execution with Prefect

[Prefect](https://www.prefect.io/) is a workflow orchestration framework for building resilient data pipelines in Python, natively integrated with Pydantic AI.

## Durable Execution

Prefect 3.0 brings [transactional semantics](https://www.prefect.io/blog/transactional-ml-pipelines-with-prefect-3-0) to your Python workflows, allowing you to group tasks into atomic units and define failure modes. If any part of a transaction fails, the entire transaction can be rolled back to a clean state.

- **Flows** are the top-level entry points for your workflow. They can contain tasks and other flows.
- **Tasks** are individual units of work that can be retried, cached, and monitored independently.

Prefect 3.0's approach to transactional orchestration makes your workflows automatically **idempotent**: rerunnable without duplication or inconsistency across any environment. Every task is executed within a transaction that governs when and where the task's result record is persisted. If the task runs again under an identical context, it will not re-execute but instead load its previous result.

The diagram below shows the overall architecture of an agentic application with Prefect. Prefect uses client-side task orchestration by default, with optional server connectivity for advanced features like scheduling and monitoring.

```text
            +---------------------+
            |   Prefect Server    |      (Monitoring,
            |      or Cloud       |       scheduling, UI,
            +---------------------+       orchestration)
                     ^
                     |
        Flow state,  |   Schedule flows,
        metadata,    |   track execution
        logs         |
                     |
+------------------------------------------------------+
|               Application Process                    |
|   +----------------------------------------------+   |
|   |              Flow (Agent.run)                |   |
|   +----------------------------------------------+   |
|          |          |                |               |
|          v          v                v               |
|   +-----------+ +------------+ +-------------+       |
|   |   Task    | |    Task    | |    Task     |       |
|   |  (Tool)   | | (MCP Tool) | | (Model API) |       |
|   +-----------+ +------------+ +-------------+       |
|         |           |                |               |
|       Cache &     Cache &          Cache &           |
|       persist     persist          persist           |
|         to           to               to             |
|         v            v                v              |
|   +----------------------------------------------+   |
|   |     Result Storage (Local FS, S3, etc.)     |    |
|   +----------------------------------------------+   |
+------------------------------------------------------+
          |           |                |
          v           v                v
      [External APIs, services, databases, etc.]

```

See the [Prefect documentation](https://docs.prefect.io/) for more information.

## Durable Agent

Any agent can be wrapped in a PrefectAgent to get durable execution. `PrefectAgent` automatically:

- Wraps Agent.run and Agent.run_sync as Prefect flows.
- Wraps [model requests](../../models/overview/) as Prefect tasks.
- Wraps [tool calls](../../tools/) as Prefect tasks (configurable per-tool).
- Wraps [MCP communication](../../mcp/client/) as Prefect tasks.

Event stream handlers are **automatically wrapped** by Prefect when running inside a Prefect flow. Each event from the stream is processed in a separate Prefect task for durability. You can customize the task behavior using the `event_stream_handler_task_config` parameter when creating the `PrefectAgent`. Do **not** manually decorate event stream handlers with `@task`. For examples, see the [streaming docs](../../agents/#streaming-all-events)

The original agent, model, and MCP server can still be used as normal outside the Prefect flow.

Here is a simple but complete example of wrapping an agent for durable execution. All it requires is to install Pydantic AI with Prefect:

```bash
pip install pydantic-ai[prefect]

```

```bash
uv add pydantic-ai[prefect]

```

Or if you're using the slim package, you can install it with the `prefect` optional group:

```bash
pip install pydantic-ai-slim[prefect]

```

```bash
uv add pydantic-ai-slim[prefect]

```

prefect_agent.py

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent

agent = Agent(
    'gpt-5',
    instructions="You're an expert in geography.",
    name='geography',  # (1)!
)

prefect_agent = PrefectAgent(agent)  # (2)!

async def main():
    result = await prefect_agent.run('What is the capital of Mexico?')  # (3)!
    print(result.output)
    #> Mexico City (Ciudad de MÃ©xico, CDMX)

```

1. The agent's `name` is used to uniquely identify its flows and tasks.
1. Wrapping the agent with `PrefectAgent` enables durable execution for all agent runs.
1. PrefectAgent.run() works like Agent.run(), but runs as a Prefect flow and executes model requests, decorated tool calls, and MCP communication as Prefect tasks.

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

For more information on how to use Prefect in Python applications, see their [Python documentation](https://docs.prefect.io/v3/how-to-guides/workflows/write-and-run).

## Prefect Integration Considerations

When using Prefect with Pydantic AI agents, there are a few important considerations to ensure workflows behave correctly.

### Agent Requirements

Each agent instance must have a unique `name` so Prefect can correctly identify and track its flows and tasks.

### Tool Wrapping

Agent tools are automatically wrapped as Prefect tasks, which means they benefit from:

- **Retry logic**: Failed tool calls can be retried automatically
- **Caching**: Tool results are cached based on their inputs
- **Observability**: Tool execution is tracked in the Prefect UI

You can customize tool task behavior using `tool_task_config` (applies to all tools) or `tool_task_config_by_name` (per-tool configuration):

prefect_agent_config.py

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent, TaskConfig

agent = Agent('gpt-5', name='my_agent')

@agent.tool_plain
def fetch_data(url: str) -> str:
    # This tool will be wrapped as a Prefect task
    ...

prefect_agent = PrefectAgent(
    agent,
    tool_task_config=TaskConfig(retries=3),  # Default for all tools
    tool_task_config_by_name={
        'fetch_data': TaskConfig(timeout_seconds=10.0),  # Specific to fetch_data
        'simple_tool': None,  # Disable task wrapping for simple_tool
    },
)

```

Set a tool's config to `None` in `tool_task_config_by_name` to disable task wrapping for that specific tool.

### Streaming

When running inside a Prefect flow, Agent.run_stream() works but doesn't provide real-time streaming because Prefect tasks consume their entire execution before returning results. The method will execute fully and return the complete result at once.

For real-time streaming behavior inside Prefect flows, you can set an event_stream_handler on the `Agent` or `PrefectAgent` instance and use PrefectAgent.run().

**Note**: Event stream handlers behave differently when running inside a Prefect flow versus outside:

- **Outside a flow**: The handler receives events as they stream from the model
- **Inside a flow**: Each event is wrapped as a Prefect task for durability, which may affect timing but ensures reliability

The event stream handler function will receive the agent run context and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](../../agents/#streaming-all-events).

## Task Configuration

You can customize Prefect task behavior, such as retries and timeouts, by passing TaskConfig objects to the `PrefectAgent` constructor:

- `mcp_task_config`: Configuration for MCP server communication tasks
- `model_task_config`: Configuration for model request tasks
- `tool_task_config`: Default configuration for all tool calls
- `tool_task_config_by_name`: Per-tool task configuration (overrides `tool_task_config`)
- `event_stream_handler_task_config`: Configuration for event stream handler tasks (applies when running inside a Prefect flow)

Available `TaskConfig` options:

- `retries`: Maximum number of retries for the task (default: `0`)
- `retry_delay_seconds`: Delay between retries in seconds (can be a single value or list for exponential backoff, default: `1.0`)
- `timeout_seconds`: Maximum time in seconds for the task to complete
- `cache_policy`: Custom Prefect cache policy for the task
- `persist_result`: Whether to persist the task result
- `result_storage`: Prefect result storage for the task (e.g., `'s3-bucket/my-storage'` or a `WritableFileSystem` block)
- `log_prints`: Whether to log print statements from the task (default: `False`)

Example:

prefect_agent_config.py

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent, TaskConfig

agent = Agent(
    'gpt-5',
    instructions="You're an expert in geography.",
    name='geography',
)

prefect_agent = PrefectAgent(
    agent,
    model_task_config=TaskConfig(
        retries=3,
        retry_delay_seconds=[1.0, 2.0, 4.0],  # Exponential backoff
        timeout_seconds=30.0,
    ),
)

async def main():
    result = await prefect_agent.run('What is the capital of France?')
    print(result.output)
    #> Paris

```

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

### Retry Considerations

Pydantic AI and provider API clients have their own retry logic. When using Prefect, you may want to:

- Disable [HTTP Request Retries](../../retries/) in Pydantic AI
- Turn off your provider API client's retry logic (e.g., `max_retries=0` on a [custom OpenAI client](../../models/openai/#custom-openai-client))
- Rely on Prefect's task-level retry configuration for consistency

This prevents requests from being retried multiple times at different layers.

## Caching and Idempotency

Prefect 3.0 provides built-in caching and transactional semantics. Tasks with identical inputs will not re-execute if their results are already cached, making workflows naturally idempotent and resilient to failures.

- **Task inputs**: Messages, settings, parameters, tool arguments, and serializable dependencies

**Note**: For user dependencies to be included in cache keys, they must be serializable (e.g., Pydantic models or basic Python types). Non-serializable dependencies are automatically excluded from cache computation.

## Observability with Prefect and Logfire

Prefect provides a built-in UI for monitoring flow runs, task executions, and failures. You can:

- View real-time flow run status
- Debug failures with full stack traces
- Set up alerts and notifications

To access the Prefect UI, you can either:

1. Use [Prefect Cloud](https://www.prefect.io/cloud) (managed service)
1. Run a local [Prefect server](https://docs.prefect.io/v3/how-to-guides/self-hosted/server-cli) with `prefect server start`

You can also use [Pydantic Logfire](../../logfire/) for detailed observability. When using both Prefect and Logfire, you'll get complementary views:

- **Prefect**: Workflow-level orchestration, task status, and retry history
- **Logfire**: Fine-grained tracing of agent runs, model requests, and tool invocations

When using Logfire with Prefect, you can enable distributed tracing to see spans for your Prefect runs included with your agent runs, model requests, and tool invocations.

For more information about Prefect monitoring, see the [Prefect documentation](https://docs.prefect.io/).

## Deployments and Scheduling

To deploy and schedule a `PrefectAgent`, wrap it in a Prefect flow and use the flow's [`serve()`](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments#create-a-deployment-with-serve) or [`deploy()`](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python) methods:

[Learn about Gateway](../../gateway) serve_agent.py

```python
from prefect import flow

from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent


@flow
async def daily_report_flow(user_prompt: str):
    """Generate a daily report using the agent."""
    agent = Agent(  # (1)!
        'gateway/openai:gpt-5',
        name='daily_report_agent',
        instructions='Generate a daily summary report.',
    )

    prefect_agent = PrefectAgent(agent)

    result = await prefect_agent.run(user_prompt)
    return result.output



# Serve the flow with a daily schedule
if __name__ == 'gateway/__main__':
    daily_report_flow.serve(
        name='daily-report-deployment',
        cron='0 9 * * *',  # Run daily at 9am
        parameters={'user_prompt': "Generate today's report"},
        tags=['production', 'reports'],
    )

```

1. Each flow run executes in an isolated process, and all inputs and dependencies must be serializable. Because Agent instances cannot be serialized, instantiate the agent inside the flow rather than at the module level.

serve_agent.py

```python
from prefect import flow

from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent


@flow
async def daily_report_flow(user_prompt: str):
    """Generate a daily report using the agent."""
    agent = Agent(  # (1)!
        'openai:gpt-5',
        name='daily_report_agent',
        instructions='Generate a daily summary report.',
    )

    prefect_agent = PrefectAgent(agent)

    result = await prefect_agent.run(user_prompt)
    return result.output



# Serve the flow with a daily schedule
if __name__ == '__main__':
    daily_report_flow.serve(
        name='daily-report-deployment',
        cron='0 9 * * *',  # Run daily at 9am
        parameters={'user_prompt': "Generate today's report"},
        tags=['production', 'reports'],
    )

```

1. Each flow run executes in an isolated process, and all inputs and dependencies must be serializable. Because Agent instances cannot be serialized, instantiate the agent inside the flow rather than at the module level.

The `serve()` method accepts scheduling options:

- **`cron`**: Cron schedule string (e.g., `'0 9 * * *'` for daily at 9am)
- **`interval`**: Schedule interval in seconds or as a timedelta
- **`rrule`**: iCalendar RRule schedule string

For production deployments with Docker, Kubernetes, or other infrastructure, use the flow's [`deploy()`](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python) method. See the [Prefect deployment documentation](https://docs.prefect.io/v3/how-to-guides/deployments/create-deploymentsy) for more information.

# Durable Execution with Temporal

[Temporal](https://temporal.io) is a popular [durable execution](https://docs.temporal.io/evaluate/understanding-temporal#durable-execution) platform that's natively supported by Pydantic AI.

## Durable Execution

In Temporal's durable execution implementation, a program that crashes or encounters an exception while interacting with a model or API will retry until it can successfully complete.

Temporal relies primarily on a replay mechanism to recover from failures. As the program makes progress, Temporal saves key inputs and decisions, allowing a re-started program to pick up right where it left off.

The key to making this work is to separate the application's repeatable (deterministic) and non-repeatable (non-deterministic) parts:

1. Deterministic pieces, termed [**workflows**](https://docs.temporal.io/workflow-definition), execute the same way when re-run with the same inputs.
1. Non-deterministic pieces, termed [**activities**](https://docs.temporal.io/activities), can run arbitrary code, performing I/O and any other operations.

Workflow code can run for extended periods and, if interrupted, resume exactly where it left off. Critically, workflow code generally *cannot* include any kind of I/O, over the network, disk, etc. Activity code faces no restrictions on I/O or external interactions, but if an activity fails part-way through it is restarted from the beginning.

Note

If you are familiar with celery, it may be helpful to think of Temporal activities as similar to celery tasks, but where you wait for the task to complete and obtain its result before proceeding to the next step in the workflow. However, Temporal workflows and activities offer a great deal more flexibility and functionality than celery tasks.

See the [Temporal documentation](https://docs.temporal.io/evaluate/understanding-temporal#temporal-application-the-building-blocks) for more information

In the case of Pydantic AI agents, integration with Temporal means that [model requests](../../models/overview/), [tool calls](../../tools/) that may require I/O, and [MCP server communication](../../mcp/client/) all need to be offloaded to Temporal activities due to their I/O requirements, while the logic that coordinates them (i.e. the agent run) lives in the workflow. Code that handles a scheduled job or web request can then execute the workflow, which will in turn execute the activities as needed.

The diagram below shows the overall architecture of an agentic application in Temporal. The Temporal Server is responsible for tracking program execution and making sure the associated state is preserved reliably (i.e., stored to an internal database, and possibly replicated across cloud regions). Temporal Server manages data in encrypted form, so all data processing occurs on the Worker, which runs the workflow and activities.

```text
            +---------------------+
            |   Temporal Server   |      (Stores workflow state,
            +---------------------+       schedules activities,
                     ^                    persists progress)
                     |
        Save state,  |   Schedule Tasks,
        progress,    |   load state on resume
        timeouts     |
                     |
+------------------------------------------------------+
|                      Worker                          |
|   +----------------------------------------------+   |
|   |              Workflow Code                   |   |
|   |       (Agent Run Loop)                       |   |
|   +----------------------------------------------+   |
|          |          |                |               |
|          v          v                v               |
|   +-----------+ +------------+ +-------------+       |
|   | Activity  | | Activity   | |  Activity   |       |
|   | (Tool)    | | (MCP Tool) | | (Model API) |       |
|   +-----------+ +------------+ +-------------+       |
|         |           |                |               |
+------------------------------------------------------+
          |           |                |
          v           v                v
      [External APIs, services, databases, etc.]

```

See the [Temporal documentation](https://docs.temporal.io/evaluate/understanding-temporal#temporal-application-the-building-blocks) for more information.

## Durable Agent

Any agent can be wrapped in a TemporalAgent to get a durable agent that can be used inside a deterministic Temporal workflow, by automatically offloading all work that requires I/O (namely model requests, tool calls, and MCP server communication) to non-deterministic activities.

At the time of wrapping, the agent's [model](../../models/overview/) and [toolsets](../../toolsets/) (including function tools registered on the agent and MCP servers) are frozen, activities are dynamically created for each, and the original model and toolsets are wrapped to call on the worker to execute the corresponding activities instead of directly performing the actions inside the workflow. The original agent can still be used as normal outside the Temporal workflow, but any changes to its model or toolsets after wrapping will not be reflected in the durable agent.

Here is a simple but complete example of wrapping an agent for durable execution, creating a Temporal workflow with durable execution logic, connecting to a Temporal server, and running the workflow from non-durable code. All it requires is a Temporal server to be [running locally](https://github.com/temporalio/temporal#download-and-start-temporal-server-locally):

```sh
brew install temporal
temporal server start-dev

```

temporal_agent.py

```python
import uuid

from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import (
    AgentPlugin,
    PydanticAIPlugin,
    TemporalAgent,
)

agent = Agent(
    'gpt-5',
    instructions="You're an expert in geography.",
    name='geography',  # (10)!
)

temporal_agent = TemporalAgent(agent)  # (1)!


@workflow.defn
class GeographyWorkflow:  # (2)!
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent.run(prompt)  # (3)!
        return result.output


async def main():
    client = await Client.connect(  # (4)!
        'localhost:7233',  # (5)!
        plugins=[PydanticAIPlugin()],  # (6)!
    )

    async with Worker(  # (7)!
        client,
        task_queue='geography',
        workflows=[GeographyWorkflow],
        plugins=[AgentPlugin(temporal_agent)],  # (8)!
    ):
        output = await client.execute_workflow(  # (9)!
            GeographyWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=f'geography-{uuid.uuid4()}',
            task_queue='geography',
        )
        print(output)
        #> Mexico City (Ciudad de MÃ©xico, CDMX)

```

1. The original `Agent` cannot be used inside a deterministic Temporal workflow, but the `TemporalAgent` can.
1. As explained above, the workflow represents a deterministic piece of code that can use non-deterministic activities for operations that require I/O.
1. TemporalAgent.run() works just like Agent.run(), but it will automatically offload model requests, tool calls, and MCP server communication to Temporal activities.
1. We connect to the Temporal server which keeps track of workflow and activity execution.
1. This assumes the Temporal server is [running locally](https://github.com/temporalio/temporal#download-and-start-temporal-server-locally).
1. The PydanticAIPlugin tells Temporal to use Pydantic for serialization and deserialization, and to treat UserError exceptions as non-retryable.
1. We start the worker that will listen on the specified task queue and run workflows and activities. In a real world application, this might be run in a separate service.
1. The AgentPlugin registers the `TemporalAgent`'s activities with the worker.
1. We call on the server to execute the workflow on a worker that's listening on the specified task queue.
1. The agent's `name` is used to uniquely identify its activities.

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

In a real world application, the agent, workflow, and worker are typically defined separately from the code that calls for a workflow to be executed. Because Temporal workflows need to be defined at the top level of the file and the `TemporalAgent` instance is needed inside the workflow and when starting the worker (to register the activities), it needs to be defined at the top level of the file as well.

For more information on how to use Temporal in Python applications, see their [Python SDK guide](https://docs.temporal.io/develop/python).

## Temporal Integration Considerations

There are a few considerations specific to agents and toolsets when using Temporal for durable execution. These are important to understand to ensure that your agents and toolsets work correctly with Temporal's workflow and activity model.

### Agent Names and Toolset IDs

To ensure that Temporal knows what code to run when an activity fails or is interrupted and then restarted, even if your code is changed in between, each activity needs to have a name that's stable and unique.

When `TemporalAgent` dynamically creates activities for the wrapped agent's model requests and toolsets (specifically those that implement their own tool listing and calling, i.e. FunctionToolset and MCPServer), their names are derived from the agent's name and the toolsets' ids. These fields are normally optional, but are required to be set when using Temporal. They should not be changed once the durable agent has been deployed to production as this would break active workflows.

Other than that, any agent and toolset will just work!

### Instructions Functions, Output Functions, and History Processors

Pydantic AI runs non-async [instructions](../../agents/#instructions) and [system prompt](../../agents/#system-prompts) functions, [history processors](../../message-history/#processing-message-history), [output functions](../../output/#output-functions), and [output validators](../../output/#output-validator-functions) in threads, which are not supported inside Temporal workflows and require an activity. Ensure that these functions are async instead.

Synchronous tool functions are supported, as tools are automatically run in activities unless this is [explicitly disabled](#activity-configuration). Still, it's recommended to make tool functions async as well to improve performance.

### Agent Run Context and Dependencies

As workflows and activities run in separate processes, any values passed between them need to be serializable. As these payloads are stored in the workflow execution event history, Temporal limits their size to 2MB.

To account for these limitations, tool functions and the [event stream handler](#streaming) running inside activities receive a limited version of the agent's RunContext, and it's your responsibility to make sure that the [dependencies](../../dependencies/) object provided to TemporalAgent.run() can be serialized using Pydantic.

Specifically, only the `deps`, `run_id`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `retry`, `max_retries`, `run_step`, `usage`, and `partial_output` fields are available by default, and trying to access `model`, `prompt`, `messages`, or `tracer` will raise an error. If you need one or more of these attributes to be available inside activities, you can create a TemporalRunContext subclass with custom `serialize_run_context` and `deserialize_run_context` class methods and pass it to TemporalAgent as `run_context_type`.

### Streaming

Because Temporal activities cannot stream output directly to the activity call site, Agent.run_stream(), Agent.run_stream_events(), and Agent.iter() are not supported.

Instead, you can implement streaming by setting an event_stream_handler on the `Agent` or `TemporalAgent` instance and using TemporalAgent.run() inside the workflow. The event stream handler function will receive the agent run context and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](../../agents/#streaming-all-events).

As the streaming model request activity, workflow, and workflow execution call all take place in separate processes, passing data between them requires some care:

- To get data from the workflow call site or workflow to the event stream handler, you can use a [dependencies object](#agent-run-context-and-dependencies).
- To get data from the event stream handler to the workflow, workflow call site, or a frontend, you need to use an external system that the event stream handler can write to and the event consumer can read from, like a message queue. You can use the dependency object to make sure the same connection string or other unique ID is available in all the places that need it.

## Activity Configuration

Temporal activity configuration, like timeouts and retry policies, can be customized by passing [`temporalio.workflow.ActivityConfig`](https://python.temporal.io/temporalio.workflow.ActivityConfig.html) objects to the `TemporalAgent` constructor:

- `activity_config`: The base Temporal activity config to use for all activities. If no config is provided, a `start_to_close_timeout` of 60 seconds is used.

- `model_activity_config`: The Temporal activity config to use for model request activities. This is merged with the base activity config.

- `toolset_activity_config`: The Temporal activity config to use for get-tools and call-tool activities for specific toolsets identified by ID. This is merged with the base activity config.

- `tool_activity_config`: The Temporal activity config to use for specific tool call activities identified by toolset ID and tool name. This is merged with the base and toolset-specific activity configs.

  If a tool does not use I/O, you can specify `False` to disable using an activity. Note that the tool is required to be defined as an `async` function as non-async tools are run in threads which are non-deterministic and thus not supported outside of activities.

## Activity Retries

On top of the automatic retries for request failures that Temporal will perform, Pydantic AI and various provider API clients also have their own request retry logic. Enabling these at the same time may cause the request to be retried more often than expected, with improper `Retry-After` handling.

When using Temporal, it's recommended to not use [HTTP Request Retries](../../retries/) and to turn off your provider API client's own retry logic, for example by setting `max_retries=0` on a [custom `OpenAIProvider` API client](../../models/openai/#custom-openai-client).

You can customize Temporal's retry policy using [activity configuration](#activity-configuration).

## Observability with Logfire

Temporal generates telemetry events and metrics for each workflow and activity execution, and Pydantic AI generates events for each agent run, model request and tool call. These can be sent to [Pydantic Logfire](../../logfire/) to get a complete picture of what's happening in your application.

To use Logfire with Temporal, you need to pass a LogfirePlugin object to Temporal's `Client.connect()`:

logfire_plugin.py

```python
from temporalio.client import Client

from pydantic_ai.durable_exec.temporal import LogfirePlugin, PydanticAIPlugin


async def main():
    client = await Client.connect(
        'localhost:7233',
        plugins=[PydanticAIPlugin(), LogfirePlugin()],
    )

```

By default, the `LogfirePlugin` will instrument Temporal (including metrics) and Pydantic AI and send all data to Logfire. To customize Logfire configuration and instrumentation, you can pass a `logfire_setup` function to the `LogfirePlugin` constructor and return a custom `Logfire` instance (i.e. the result of `logfire.configure()`). To disable sending Temporal metrics to Logfire, you can pass `metrics=False` to the `LogfirePlugin` constructor.

## Known Issues

### Pandas

When `logfire.info` is used inside an activity and the `pandas` package is among your project's dependencies, you may encounter the following error which seems to be the result of an import race condition:

```text
AttributeError: partially initialized module 'pandas' has no attribute '_pandas_parser_CAPI' (most likely due to a circular import)

```

To fix this, you can use the [`temporalio.workflow.unsafe.imports_passed_through()`](https://python.temporal.io/temporalio.workflow.unsafe.html#imports_passed_through) context manager to proactively import the package and not have it be reloaded in the workflow sandbox:

temporal_activity.py

```python
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import pandas

```
# MCP

# Client

Pydantic AI can act as an [MCP client](https://modelcontextprotocol.io/quickstart/client), connecting to MCP servers to use their tools.

## Install

You need to either install [`pydantic-ai`](../../install/), or [`pydantic-ai-slim`](../../install/#slim-install) with the `mcp` optional group:

```bash
pip install "pydantic-ai-slim[mcp]"

```

```bash
uv add "pydantic-ai-slim[mcp]"

```

## Usage

Pydantic AI comes with three ways to connect to MCP servers:

- MCPServerStreamableHTTP which connects to an MCP server using the [Streamable HTTP](https://modelcontextprotocol.io/introduction#streamable-http) transport
- MCPServerSSE which connects to an MCP server using the [HTTP SSE](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse) transport
- MCPServerStdio which runs the server as a subprocess and connects to it using the [stdio](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) transport

Examples of all three are shown below.

Each MCP server instance is a [toolset](../../toolsets/) and can be registered with an Agent using the `toolsets` argument.

You can use the async with agent context manager to open and close connections to all registered servers (and in the case of stdio servers, start and stop the subprocesses) around the context where they'll be used in agent runs. You can also use async with server to manage the connection or subprocess of a specific server, for example if you'd like to use it with multiple agents. If you don't explicitly enter one of these context managers to set up the server, this will be done automatically when it's needed (e.g. to list the available tools or call a specific tool), but it's more efficient to do so around the entire context where you expect the servers to be used.

### Streamable HTTP Client

MCPServerStreamableHTTP connects over HTTP using the [Streamable HTTP](https://modelcontextprotocol.io/introduction#streamable-http) transport to a server.

Note

MCPServerStreamableHTTP requires an MCP server to be running and accepting HTTP connections before running the agent. Running the server is not managed by Pydantic AI.

Before creating the Streamable HTTP client, we need to run a server that supports the Streamable HTTP transport.

streamable_http_server.py

```python
from mcp.server.fastmcp import FastMCP

app = FastMCP()

@app.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == '__main__':
    app.run(transport='streamable-http')

```

Then we can create the client:

[Learn about Gateway](../../gateway) mcp_streamable_http_client.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')  # (1)!
agent = Agent('gateway/openai:gpt-5', toolsets=[server])  # (2)!

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.

```

1. Define the MCP server with the URL used to connect.
1. Create an agent with the MCP server attached.

mcp_streamable_http_client.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')  # (1)!
agent = Agent('openai:gpt-5', toolsets=[server])  # (2)!

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.

```

1. Define the MCP server with the URL used to connect.
1. Create an agent with the MCP server attached.

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

**What's happening here?**

- The model receives the prompt "What is 7 plus 5?"
- The model decides "Oh, I've got this `add` tool, that will be a good way to answer this question"
- The model returns a tool call
- Pydantic AI sends the tool call to the MCP server using the Streamable HTTP transport
- The model is called again with the return value of running the `add` tool (12)
- The model returns the final answer

You can visualise this clearly, and even see the tool call, by adding three lines of code to instrument the example with [logfire](https://logfire.pydantic.dev/docs):

mcp_sse_client_logfire.py

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

```

### SSE Client

MCPServerSSE connects over HTTP using the [HTTP + Server Sent Events transport](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse) to a server.

Note

The SSE transport in MCP is deprecated, you should use Streamable HTTP instead.

Before creating the SSE client, we need to run a server that supports the SSE transport.

sse_server.py

```python
from mcp.server.fastmcp import FastMCP

app = FastMCP()

@app.tool()
def add(a: int, b: int) -> int:
    return a + b

if __name__ == '__main__':
    app.run(transport='sse')

```

Then we can create the client:

[Learn about Gateway](../../gateway) mcp_sse_client.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE('http://localhost:3001/sse')  # (1)!
agent = Agent('gateway/openai:gpt-5', toolsets=[server])  # (2)!


async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.

```

1. Define the MCP server with the URL used to connect.
1. Create an agent with the MCP server attached.

mcp_sse_client.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE('http://localhost:3001/sse')  # (1)!
agent = Agent('openai:gpt-5', toolsets=[server])  # (2)!


async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.

```

1. Define the MCP server with the URL used to connect.
1. Create an agent with the MCP server attached.

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

### MCP "stdio" Server

MCP also offers [stdio transport](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) where the server is run as a subprocess and communicates with the client over `stdin` and `stdout`. In this case, you'd use the MCPServerStdio class.

In this example [mcp-run-python](https://github.com/pydantic/mcp-run-python) is used as the MCP server.

[Learn about Gateway](../../gateway) mcp_stdio_client.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(  # (1)!
    'uv', args=['run', 'mcp-run-python', 'stdio'], timeout=10
)
agent = Agent('gateway/openai:gpt-5', toolsets=[server])


async def main():
    result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.

```

1. See [MCP Run Python](https://github.com/pydantic/mcp-run-python) for more information.

mcp_stdio_client.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(  # (1)!
    'uv', args=['run', 'mcp-run-python', 'stdio'], timeout=10
)
agent = Agent('openai:gpt-5', toolsets=[server])


async def main():
    result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.

```

1. See [MCP Run Python](https://github.com/pydantic/mcp-run-python) for more information.

## Loading MCP Servers from Configuration

Instead of creating MCP server instances individually in code, you can load multiple servers from a JSON configuration file using load_mcp_servers().

This is particularly useful when you need to manage multiple MCP servers or want to configure servers externally without modifying code.

### Configuration Format

The configuration file should be a JSON file with an `mcpServers` object containing server definitions. Each server is identified by a unique key and contains the configuration for that server type:

mcp_config.json

```json
{
  "mcpServers": {
    "python-runner": {
      "command": "uv",
      "args": ["run", "mcp-run-python", "stdio"]
    },
    "weather-api": {
      "url": "http://localhost:3001/sse"
    },
    "calculator": {
      "url": "http://localhost:8000/mcp"
    }
  }
}

```

Note

The MCP server is only inferred to be an SSE server because of the `/sse` suffix. Any other server with the "url" field will be inferred to be a Streamable HTTP server.

We made this decision given that the SSE transport is deprecated.

### Environment Variables

The configuration file supports environment variable expansion using the `${VAR}` and `${VAR:-default}` syntax, [like Claude Code](https://code.claude.com/docs/en/mcp#environment-variable-expansion-in-mcp-json). This is useful for keeping sensitive information like API keys or host names out of your configuration files:

mcp_config_with_env.json

```json
{
  "mcpServers": {
    "python-runner": {
      "command": "${PYTHON_CMD:-python3}",
      "args": ["run", "${MCP_MODULE}", "stdio"],
      "env": {
        "API_KEY": "${MY_API_KEY}"
      }
    },
    "weather-api": {
      "url": "https://${SERVER_HOST:-localhost}:${SERVER_PORT:-8080}/sse"
    }
  }
}

```

When loading this configuration with load_mcp_servers():

- `${VAR}` references will be replaced with the corresponding environment variable values.
- `${VAR:-default}` references will use the environment variable value if set, otherwise the default value.

Warning

If a referenced environment variable using `${VAR}` syntax is not defined, a `ValueError` will be raised. Use the `${VAR:-default}` syntax to provide a fallback value.

### Usage

[Learn about Gateway](../../gateway) mcp_config_loader.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_servers

# Load all servers from configuration file
servers = load_mcp_servers('mcp_config.json')

# Create agent with all loaded servers
agent = Agent('gateway/openai:gpt-5', toolsets=servers)

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)

```

mcp_config_loader.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_servers

# Load all servers from configuration file
servers = load_mcp_servers('mcp_config.json')

# Create agent with all loaded servers
agent = Agent('openai:gpt-5', toolsets=servers)

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)

```

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

## Tool call customization

The MCP servers provide the ability to set a `process_tool_call` which allows the customization of tool call requests and their responses.

A common use case for this is to inject metadata to the requests which the server call needs:

mcp_process_tool_call.py

```python
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPServerStdio, ToolResult
from pydantic_ai.models.test import TestModel


async def process_tool_call(
    ctx: RunContext[int],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """A tool call processor that passes along the deps."""
    return await call_tool(name, tool_args, {'deps': ctx.deps})


server = MCPServerStdio('python', args=['mcp_server.py'], process_tool_call=process_tool_call)
agent = Agent(
    model=TestModel(call_tools=['echo_deps']),
    deps_type=int,
    toolsets=[server]
)


async def main():
    result = await agent.run('Echo with deps set to 42', deps=42)
    print(result.output)
    #> {"echo_deps":{"echo":"This is an echo message","deps":42}}

```

How to access the metadata is MCP server SDK specific. For example with the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), it is accessible via the [`ctx: Context`](https://github.com/modelcontextprotocol/python-sdk#context) argument that can be included on tool call handlers:

mcp_server.py

```python
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

mcp = FastMCP('Pydantic AI MCP Server')
log_level = 'unset'


@mcp.tool()
async def echo_deps(ctx: Context[ServerSession, None]) -> dict[str, Any]:
    """Echo the run context.

    Args:
        ctx: Context object containing request and session information.

    Returns:
        Dictionary with an echo message and the deps.
    """
    await ctx.info('This is an info message')

    deps: Any = getattr(ctx.request_context.meta, 'deps')
    return {'echo': 'This is an echo message', 'deps': deps}

if __name__ == '__main__':
    mcp.run()

```

## Using Tool Prefixes to Avoid Naming Conflicts

When connecting to multiple MCP servers that might provide tools with the same name, you can use the `tool_prefix` parameter to avoid naming conflicts. This parameter adds a prefix to all tool names from a specific server.

This allows you to use multiple servers that might have overlapping tool names without conflicts:

[Learn about Gateway](../../gateway) mcp_tool_prefix_http_client.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

# Create two servers with different prefixes
weather_server = MCPServerSSE(
    'http://localhost:3001/sse',
    tool_prefix='weather'  # Tools will be prefixed with 'weather_'
)

calculator_server = MCPServerSSE(
    'http://localhost:3002/sse',
    tool_prefix='calc'  # Tools will be prefixed with 'calc_'
)

# Both servers might have a tool named 'get_data', but they'll be exposed as:
# - 'weather_get_data'
# - 'calc_get_data'
agent = Agent('gateway/openai:gpt-5', toolsets=[weather_server, calculator_server])

```

mcp_tool_prefix_http_client.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

# Create two servers with different prefixes
weather_server = MCPServerSSE(
    'http://localhost:3001/sse',
    tool_prefix='weather'  # Tools will be prefixed with 'weather_'
)

calculator_server = MCPServerSSE(
    'http://localhost:3002/sse',
    tool_prefix='calc'  # Tools will be prefixed with 'calc_'
)

# Both servers might have a tool named 'get_data', but they'll be exposed as:
# - 'weather_get_data'
# - 'calc_get_data'
agent = Agent('openai:gpt-5', toolsets=[weather_server, calculator_server])

```

## Server Instructions

MCP servers can provide instructions during initialization that give context about how to best interact with the server's tools. These instructions are accessible via the instructions property after the server connection is established.

[Learn about Gateway](../../gateway) mcp_server_instructions.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')
agent = Agent('gateway/openai:gpt-5', toolsets=[server])

@agent.instructions
async def mcp_server_instructions():
    return server.instructions  # (1)!

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.

```

1. The server connection is guaranteed to be established by this point, so `server.instructions` is available.

mcp_server_instructions.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8000/mcp')
agent = Agent('openai:gpt-5', toolsets=[server])

@agent.instructions
async def mcp_server_instructions():
    return server.instructions  # (1)!

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.

```

1. The server connection is guaranteed to be established by this point, so `server.instructions` is available.

## Tool metadata

MCP tools can include metadata that provides additional information about the tool's characteristics, which can be useful when filtering tools. The `meta`, `annotations`, and `output_schema` fields can be found on the `metadata` dict on the ToolDefinition object that's passed to filter functions.

## Resources

MCP servers can provide [resources](https://modelcontextprotocol.io/docs/concepts/resources) - files, data, or content that can be accessed by the client. Resources in MCP are application-driven, with host applications determining how to incorporate context manually, based on their needs. This means they will *not* be exposed to the LLM automatically (unless a tool returns a `ResourceLink` or `EmbeddedResource`).

Pydantic AI provides methods to discover and read resources from MCP servers:

- list_resources() - List all available resources on the server
- list_resource_templates() - List resource templates with parameter placeholders
- read_resource(uri) - Read the contents of a specific resource by URI

Resources are automatically converted: text content is returned as `str`, and binary content is returned as BinaryContent.

Before consuming resources, we need to run a server that exposes some:

mcp_resource_server.py

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('Pydantic AI MCP Server')
log_level = 'unset'


@mcp.resource('resource://user_name.txt', mime_type='text/plain')
async def user_name_resource() -> str:
    return 'Alice'


if __name__ == '__main__':
    mcp.run()

```

Then we can create the client:

mcp_resources.py

```python
import asyncio

from pydantic_ai.mcp import MCPServerStdio


async def main():
    server = MCPServerStdio('python', args=['-m', 'mcp_resource_server'])

    async with server:
        # List all available resources
        resources = await server.list_resources()
        for resource in resources:
            print(f' - {resource.name}: {resource.uri} ({resource.mime_type})')
            #>  - user_name_resource: resource://user_name.txt (text/plain)

        # Read a text resource
        user_name = await server.read_resource('resource://user_name.txt')
        print(f'Text content: {user_name}')
        #> Text content: Alice


if __name__ == '__main__':
    asyncio.run(main())

```

*(This example is complete, it can be run "as is")*

## Custom TLS / SSL configuration

In some environments you need to tweak how HTTPS connections are established â€“ for example to trust an internal Certificate Authority, present a client certificate for **mTLS**, or (during local development only!) disable certificate verification altogether. All HTTP-based MCP client classes (MCPServerStreamableHTTP and MCPServerSSE) expose an `http_client` parameter that lets you pass your own pre-configured [`httpx.AsyncClient`](https://www.python-httpx.org/async/).

[Learn about Gateway](../../gateway) mcp_custom_tls_client.py

```python
import ssl

import httpx

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

# Trust an internal / self-signed CA
ssl_ctx = ssl.create_default_context(cafile='/etc/ssl/private/my_company_ca.pem')

# OPTIONAL: if the server requires **mutual TLS** load your client certificate
ssl_ctx.load_cert_chain(certfile='/etc/ssl/certs/client.crt', keyfile='/etc/ssl/private/client.key',)

http_client = httpx.AsyncClient(
    verify=ssl_ctx,
    timeout=httpx.Timeout(10.0),
)

server = MCPServerSSE(
    'http://localhost:3001/sse',
    http_client=http_client,  # (1)!
)
agent = Agent('gateway/openai:gpt-5', toolsets=[server])

async def main():
    result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.

```

1. When you supply `http_client`, Pydantic AI re-uses this client for every request. Anything supported by **httpx** (`verify`, `cert`, custom proxies, timeouts, etc.) therefore applies to all MCP traffic.

mcp_custom_tls_client.py

```python
import ssl

import httpx

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

# Trust an internal / self-signed CA
ssl_ctx = ssl.create_default_context(cafile='/etc/ssl/private/my_company_ca.pem')

# OPTIONAL: if the server requires **mutual TLS** load your client certificate
ssl_ctx.load_cert_chain(certfile='/etc/ssl/certs/client.crt', keyfile='/etc/ssl/private/client.key',)

http_client = httpx.AsyncClient(
    verify=ssl_ctx,
    timeout=httpx.Timeout(10.0),
)

server = MCPServerSSE(
    'http://localhost:3001/sse',
    http_client=http_client,  # (1)!
)
agent = Agent('openai:gpt-5', toolsets=[server])

async def main():
    result = await agent.run('How many days between 2000-01-01 and 2025-03-18?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.

```

1. When you supply `http_client`, Pydantic AI re-uses this client for every request. Anything supported by **httpx** (`verify`, `cert`, custom proxies, timeouts, etc.) therefore applies to all MCP traffic.

## MCP Sampling

What is MCP Sampling?

In MCP [sampling](https://modelcontextprotocol.io/docs/concepts/sampling) is a system by which an MCP server can make LLM calls via the MCP client - effectively proxying requests to an LLM via the client over whatever transport is being used.

Sampling is extremely useful when MCP servers need to use Gen AI but you don't want to provision them each with their own LLM credentials or when a public MCP server would like the connecting client to pay for LLM calls.

Confusingly it has nothing to do with the concept of "sampling" in observability, or frankly the concept of "sampling" in any other domain.

Sampling Diagram

Here's a mermaid diagram that may or may not make the data flow clearer:

```
sequenceDiagram
    participant LLM
    participant MCP_Client as MCP client
    participant MCP_Server as MCP server

    MCP_Client->>LLM: LLM call
    LLM->>MCP_Client: LLM tool call response

    MCP_Client->>MCP_Server: tool call
    MCP_Server->>MCP_Client: sampling "create message"

    MCP_Client->>LLM: LLM call
    LLM->>MCP_Client: LLM text response

    MCP_Client->>MCP_Server: sampling response
    MCP_Server->>MCP_Client: tool call response
```

Pydantic AI supports sampling as both a client and server. See the [server](../server/#mcp-sampling) documentation for details on how to use sampling within a server.

Sampling is automatically supported by Pydantic AI agents when they act as a client.

To be able to use sampling, an MCP server instance needs to have a sampling_model set. This can be done either directly on the server using the constructor keyword argument or the property, or by using agent.set_mcp_sampling_model() to set the agent's model or one specified as an argument as the sampling model on all MCP servers registered with that agent.

Let's say we have an MCP server that wants to use sampling (in this case to generate an SVG as per the tool arguments).

Sampling MCP Server generate_svg.py

````python
import re
from pathlib import Path

from mcp import SamplingMessage
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import TextContent

app = FastMCP()


@app.tool()
async def image_generator(ctx: Context, subject: str, style: str) -> str:
    prompt = f'{subject=} {style=}'
    # `ctx.session.create_message` is the sampling call
    result = await ctx.session.create_message(
        [SamplingMessage(role='user', content=TextContent(type='text', text=prompt))],
        max_tokens=1_024,
        system_prompt='Generate an SVG image as per the user input',
    )
    assert isinstance(result.content, TextContent)

    path = Path(f'{subject}_{style}.svg')
    # remove triple backticks if the svg was returned within markdown
    if m := re.search(r'^```\w*$(.+?)```$', result.content.text, re.S | re.M):
        path.write_text(m.group(1))
    else:
        path.write_text(result.content.text)
    return f'See {path}'


if __name__ == '__main__':
    # run the server via stdio
    app.run()

````

Using this server with an `Agent` will automatically allow sampling:

sampling_mcp_client.py

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio('python', args=['generate_svg.py'])
agent = Agent('openai:gpt-5', toolsets=[server])


async def main():
    agent.set_mcp_sampling_model()
    result = await agent.run('Create an image of a robot in a punk style.')
    print(result.output)
    #> Image file written to robot_punk.svg.

```

*(This example is complete, it can be run "as is")*

You can disallow sampling by setting allow_sampling=False when creating the server reference, e.g.:

sampling_disallowed.py

```python
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(
    'python',
    args=['generate_svg.py'],
    allow_sampling=False,
)

```

## Elicitation

In MCP, [elicitation](https://modelcontextprotocol.io/docs/concepts/elicitation) allows a server to request for [structured input](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation#supported-schema-types) from the client for missing or additional context during a session.

Elicitation let models essentially say "Hold on - I need to know X before i can continue" rather than requiring everything upfront or taking a shot in the dark.

### How Elicitation works

Elicitation introduces a new protocol message type called [`ElicitRequest`](https://modelcontextprotocol.io/specification/2025-06-18/schema#elicitrequest), which is sent from the server to the client when it needs additional information. The client can then respond with an [`ElicitResult`](https://modelcontextprotocol.io/specification/2025-06-18/schema#elicitresult) or an `ErrorData` message.

Here's a typical interaction:

- User makes a request to the MCP server (e.g. "Book a table at that Italian place")
- The server identifies that it needs more information (e.g. "Which Italian place?", "What date and time?")
- The server sends an `ElicitRequest` to the client asking for the missing information.
- The client receives the request, presents it to the user (e.g. via a terminal prompt, GUI dialog, or web interface).
- User provides the requested information, `decline` or `cancel` the request.
- The client sends an `ElicitResult` back to the server with the user's response.
- With the structured data, the server can continue processing the original request.

This allows for a more interactive and user-friendly experience, especially for multi-staged workflows. Instead of requiring all information upfront, the server can ask for it as needed, making the interaction feel more natural.

### Setting up Elicitation

To enable elicitation, provide an elicitation_callback function when creating your MCP server instance:

restaurant_server.py

```python
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP(name='Restaurant Booking')


class BookingDetails(BaseModel):
    """Schema for restaurant booking information."""

    restaurant: str = Field(description='Choose a restaurant')
    party_size: int = Field(description='Number of people', ge=1, le=8)
    date: str = Field(description='Reservation date (DD-MM-YYYY)')


@mcp.tool()
async def book_table(ctx: Context) -> str:
    """Book a restaurant table with user input."""
    # Ask user for booking details using Pydantic schema
    result = await ctx.elicit(message='Please provide your booking details:', schema=BookingDetails)

    if result.action == 'accept' and result.data:
        booking = result.data
        return f'âœ… Booked table for {booking.party_size} at {booking.restaurant} on {booking.date}'
    elif result.action == 'decline':
        return 'No problem! Maybe another time.'
    else:  # cancel
        return 'Booking cancelled.'


if __name__ == '__main__':
    mcp.run(transport='stdio')

```

This server demonstrates elicitation by requesting structured booking details from the client when the `book_table` tool is called. Here's how to create a client that handles these elicitation requests:

[Learn about Gateway](../../gateway) client_example.py

```python
import asyncio
from typing import Any

from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio


async def handle_elicitation(
    context: RequestContext[ClientSession, Any, Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Handle elicitation requests from MCP server."""
    print(f'\n{params.message}')

    if not params.requestedSchema:
        response = input('Response: ')
        return ElicitResult(action='accept', content={'response': response})

    # Collect data for each field
    properties = params.requestedSchema['properties']
    data = {}

    for field, info in properties.items():
        description = info.get('description', field)

        value = input(f'{description}: ')

        # Convert to proper type based on JSON schema
        if info.get('type') == 'integer':
            data[field] = int(value)
        else:
            data[field] = value

    # Confirm
    confirm = input('\nConfirm booking? (y/n/c): ').lower()

    if confirm == 'y':
        print('Booking details:', data)
        return ElicitResult(action='accept', content=data)
    elif confirm == 'n':
        return ElicitResult(action='decline')
    else:
        return ElicitResult(action='cancel')


# Set up MCP server connection
restaurant_server = MCPServerStdio(
    'python', args=['restaurant_server.py'], elicitation_callback=handle_elicitation
)

# Create agent
agent = Agent('gateway/openai:gpt-5', toolsets=[restaurant_server])


async def main():
    """Run the agent to book a restaurant table."""
    result = await agent.run('Book me a table')
    print(f'\nResult: {result.output}')


if __name__ == '__main__':
    asyncio.run(main())

```

client_example.py

```python
import asyncio
from typing import Any

from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio


async def handle_elicitation(
    context: RequestContext[ClientSession, Any, Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Handle elicitation requests from MCP server."""
    print(f'\n{params.message}')

    if not params.requestedSchema:
        response = input('Response: ')
        return ElicitResult(action='accept', content={'response': response})

    # Collect data for each field
    properties = params.requestedSchema['properties']
    data = {}

    for field, info in properties.items():
        description = info.get('description', field)

        value = input(f'{description}: ')

        # Convert to proper type based on JSON schema
        if info.get('type') == 'integer':
            data[field] = int(value)
        else:
            data[field] = value

    # Confirm
    confirm = input('\nConfirm booking? (y/n/c): ').lower()

    if confirm == 'y':
        print('Booking details:', data)
        return ElicitResult(action='accept', content=data)
    elif confirm == 'n':
        return ElicitResult(action='decline')
    else:
        return ElicitResult(action='cancel')


# Set up MCP server connection
restaurant_server = MCPServerStdio(
    'python', args=['restaurant_server.py'], elicitation_callback=handle_elicitation
)

# Create agent
agent = Agent('openai:gpt-5', toolsets=[restaurant_server])


async def main():
    """Run the agent to book a restaurant table."""
    result = await agent.run('Book me a table')
    print(f'\nResult: {result.output}')


if __name__ == '__main__':
    asyncio.run(main())

```

### Supported Schema Types

MCP elicitation supports string, number, boolean, and enum types with flat object structures only. These limitations ensure reliable cross-client compatibility. See [supported schema types](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation#supported-schema-types) for details.

### Security

MCP Elicitation requires careful handling - servers must not request sensitive information, and clients must implement user approval controls with clear explanations. See [security considerations](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation#security-considerations) for details.

# FastMCP Client

[FastMCP](https://gofastmcp.com/) is a higher-level MCP framework that bills itself as "The fast, Pythonic way to build MCP servers and clients." It supports additional capabilities on top of the MCP specification like [Tool Transformation](https://gofastmcp.com/patterns/tool-transformation), [OAuth](https://gofastmcp.com/clients/auth/oauth), and more.

As an alternative to Pydantic AI's standard [`MCPServer` MCP client](../client/) built on the [MCP SDK](https://github.com/modelcontextprotocol/python-sdk), you can use the FastMCPToolset [toolset](../../toolsets/) that leverages the [FastMCP Client](https://gofastmcp.com/clients/) to connect to local and remote MCP servers, whether or not they're built using [FastMCP Server](https://gofastmcp.com/servers/).

Note that it does not yet support integration elicitation or sampling, which are supported by the [standard `MCPServer` client](../client/).

## Install

To use the `FastMCPToolset`, you will need to install [`pydantic-ai-slim`](../../install/#slim-install) with the `fastmcp` optional group:

```bash
pip install "pydantic-ai-slim[fastmcp]"

```

```bash
uv add "pydantic-ai-slim[fastmcp]"

```

## Usage

A `FastMCPToolset` can then be created from:

- A FastMCP Server: `FastMCPToolset(fastmcp.FastMCP('my_server'))`
- A FastMCP Client: `FastMCPToolset(fastmcp.Client(...))`
- A FastMCP Transport: `FastMCPToolset(fastmcp.StdioTransport(command='uvx', args=['mcp-run-python', 'stdio']))`
- A Streamable HTTP URL: `FastMCPToolset('http://localhost:8000/mcp')`
- An HTTP SSE URL: `FastMCPToolset('http://localhost:8000/sse')`
- A Python Script: `FastMCPToolset('my_server.py')`
- A Node.js Script: `FastMCPToolset('my_server.js')`
- A JSON MCP Configuration: `FastMCPToolset({'mcpServers': {'my_server': {'command': 'uvx', 'args': ['mcp-run-python', 'stdio']}}})`

If you already have a [FastMCP Server](https://gofastmcp.com/servers) in the same codebase as your Pydantic AI agent, you can create a `FastMCPToolset` directly from it and save agent a network round trip:

[Learn about Gateway](../../gateway)

```python
from fastmcp import FastMCP

from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

fastmcp_server = FastMCP('my_server')
@fastmcp_server.tool()
async def add(a: int, b: int) -> int:
    return a + b

toolset = FastMCPToolset(fastmcp_server)

agent = Agent('gateway/openai:gpt-5', toolsets=[toolset])

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.

```

```python
from fastmcp import FastMCP

from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

fastmcp_server = FastMCP('my_server')
@fastmcp_server.tool()
async def add(a: int, b: int) -> int:
    return a + b

toolset = FastMCPToolset(fastmcp_server)

agent = Agent('openai:gpt-5', toolsets=[toolset])

async def main():
    result = await agent.run('What is 7 plus 5?')
    print(result.output)
    #> The answer is 12.

```

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

Connecting your agent to a Streamable HTTP MCP Server is as simple as:

[Learn about Gateway](../../gateway)

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

toolset = FastMCPToolset('http://localhost:8000/mcp')

agent = Agent('gateway/openai:gpt-5', toolsets=[toolset])

```

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

toolset = FastMCPToolset('http://localhost:8000/mcp')

agent = Agent('openai:gpt-5', toolsets=[toolset])

```

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

You can also create a `FastMCPToolset` from a JSON MCP Configuration:

[Learn about Gateway](../../gateway)

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

mcp_config = {
    'mcpServers': {
        'time_mcp_server': {
            'command': 'uvx',
            'args': ['mcp-run-python', 'stdio']
        }
    }
}

toolset = FastMCPToolset(mcp_config)

agent = Agent('gateway/openai:gpt-5', toolsets=[toolset])

```

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

mcp_config = {
    'mcpServers': {
        'time_mcp_server': {
            'command': 'uvx',
            'args': ['mcp-run-python', 'stdio']
        }
    }
}

toolset = FastMCPToolset(mcp_config)

agent = Agent('openai:gpt-5', toolsets=[toolset])

```

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

# Model Context Protocol (MCP)

Pydantic AI supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io) in multiple ways:

1. [Agents](../../agents/) can connect to MCP servers and use their tools using three different methods:
   1. Pydantic AI can act as an MCP client and connect directly to local and remote MCP servers. [Learn more](../client/) about MCPServer.
   1. Pydantic AI can use the [FastMCP Client](https://gofastmcp.com/clients/client/) to connect to local and remote MCP servers, whether or not they're built using [FastMCP Server](https://gofastmcp.com/servers). [Learn more](../fastmcp-client/) about FastMCPToolset.
   1. Some model providers can themselves connect to remote MCP servers using a "built-in tool". [Learn more](../../builtin-tools/#mcp-server-tool) about MCPServerTool.
1. Agents can be used within MCP servers. [Learn more](../server/)

## What is MCP?

The Model Context Protocol is a standardized protocol that allow AI applications (including programmatic agents like Pydantic AI, coding agents like [cursor](https://www.cursor.com/), and desktop applications like [Claude Desktop](https://claude.ai/download)) to connect to external tools and services using a common interface.

As with other protocols, the dream of MCP is that a wide range of applications can speak to each other without the need for specific integrations.

There is a great list of MCP servers at [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers).

Some examples of what this means:

- Pydantic AI could use a web search service implemented as an MCP server to implement a deep research agent
- Cursor could connect to the [Pydantic Logfire](https://github.com/pydantic/logfire-mcp) MCP server to search logs, traces and metrics to gain context while fixing a bug
- Pydantic AI, or any other MCP client could connect to our [Run Python](https://github.com/pydantic/mcp-run-python) MCP server to run arbitrary Python code in a sandboxed environment

# Server

Pydantic AI models can also be used within MCP Servers.

## MCP Server

Here's a simple example of a [Python MCP server](https://github.com/modelcontextprotocol/python-sdk) using Pydantic AI within a tool call:

[Learn about Gateway](../../gateway) mcp_server.py

```python
from mcp.server.fastmcp import FastMCP

from pydantic_ai import Agent

server = FastMCP('Pydantic AI Server')
server_agent = Agent(
    'gateway/anthropic:claude-haiku-4-5', system_prompt='always reply in rhyme'
)


@server.tool()
async def poet(theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {theme}')
    return r.output


if __name__ == '__main__':
    server.run()

```

mcp_server.py

```python
from mcp.server.fastmcp import FastMCP

from pydantic_ai import Agent

server = FastMCP('Pydantic AI Server')
server_agent = Agent(
    'anthropic:claude-haiku-4-5', system_prompt='always reply in rhyme'
)


@server.tool()
async def poet(theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {theme}')
    return r.output


if __name__ == '__main__':
    server.run()

```

## Simple client

This server can be queried with any MCP client. Here is an example using the Python SDK directly:

mcp_client.py

```python
import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def client():
    server_params = StdioServerParameters(
        command='python', args=['mcp_server.py'], env=os.environ
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool('poet', {'theme': 'socks'})
            print(result.content[0].text)
            """
            Oh, socks, those garments soft and sweet,
            That nestle softly 'round our feet,
            From cotton, wool, or blended thread,
            They keep our toes from feeling dread.
            """


if __name__ == '__main__':
    asyncio.run(client())

```

## MCP Sampling

What is MCP Sampling?

See the [MCP client docs](../client/#mcp-sampling) for details of what MCP sampling is, and how you can support it when using Pydantic AI as an MCP client.

When Pydantic AI agents are used within MCP servers, they can use sampling via MCPSamplingModel.

We can extend the above example to use sampling so instead of connecting directly to the LLM, the agent calls back through the MCP client to make LLM calls.

mcp_server_sampling.py

```python
from mcp.server.fastmcp import Context, FastMCP

from pydantic_ai import Agent
from pydantic_ai.models.mcp_sampling import MCPSamplingModel

server = FastMCP('Pydantic AI Server with sampling')
server_agent = Agent(system_prompt='always reply in rhyme')


@server.tool()
async def poet(ctx: Context, theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f'write a poem about {theme}', model=MCPSamplingModel(session=ctx.session))
    return r.output


if __name__ == '__main__':
    server.run()  # run the server over stdio

```

The [above](#simple-client) client does not support sampling, so if you tried to use it with this server you'd get an error.

The simplest way to support sampling in an MCP client is to [use](../client/#mcp-sampling) a Pydantic AI agent as the client, but if you wanted to support sampling with the vanilla MCP SDK, you could do so like this:

mcp_client_sampling.py

```python
import asyncio
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ErrorData,
    TextContent,
)


async def sampling_callback(
    context: RequestContext[ClientSession, Any], params: CreateMessageRequestParams
) -> CreateMessageResult | ErrorData:
    print('sampling system prompt:', params.systemPrompt)
    #> sampling system prompt: always reply in rhyme
    print('sampling messages:', params.messages)
    """
    sampling messages:
    [
        SamplingMessage(
            role='user',
            content=TextContent(
                type='text',
                text='write a poem about socks',
                annotations=None,
                meta=None,
            ),
        )
    ]
    """

    # TODO get the response content by calling an LLM...
    response_content = 'Socks for a fox.'

    return CreateMessageResult(
        role='assistant',
        content=TextContent(type='text', text=response_content),
        model='fictional-llm',
    )


async def client():
    server_params = StdioServerParameters(command='python', args=['mcp_server_sampling.py'])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=sampling_callback) as session:
            await session.initialize()
            result = await session.call_tool('poet', {'theme': 'socks'})
            print(result.content[0].text)
            #> Socks for a fox.


if __name__ == '__main__':
    asyncio.run(client())

```

*(This example is complete, it can be run "as is")*
# UI Event Streams

# Agent-User Interaction (AG-UI) Protocol

The [Agent-User Interaction (AG-UI) Protocol](https://docs.ag-ui.com/introduction) is an open standard introduced by the [CopilotKit](https://webflow.copilotkit.ai/blog/introducing-ag-ui-the-protocol-where-agents-meet-users) team that standardises how frontend applications communicate with AI agents, with support for streaming, frontend tools, shared state, and custom events.

Note

The AG-UI integration was originally built by the team at [Rocket Science](https://www.rocketscience.gg/) and contributed in collaboration with the Pydantic AI and CopilotKit teams. Thanks Rocket Science!

## Installation

The only dependencies are:

- [ag-ui-protocol](https://docs.ag-ui.com/introduction): to provide the AG-UI types and encoder.
- [starlette](https://www.starlette.io): to handle [ASGI](https://asgi.readthedocs.io/en/latest/) requests from a framework like FastAPI.

You can install Pydantic AI with the `ag-ui` extra to ensure you have all the required AG-UI dependencies:

```bash
pip install 'pydantic-ai-slim[ag-ui]'

```

```bash
uv add 'pydantic-ai-slim[ag-ui]'

```

To run the examples you'll also need:

- [uvicorn](https://www.uvicorn.org/) or another ASGI compatible server

```bash
pip install uvicorn

```

```bash
uv add uvicorn

```

## Usage

There are three ways to run a Pydantic AI agent based on AG-UI run input with streamed AG-UI events as output, from most to least flexible. If you're using a Starlette-based web framework like FastAPI, you'll typically want to use the second method.

1. The AGUIAdapter.run_stream() method, when called on an AGUIAdapter instantiated with an agent and an AG-UI [`RunAgentInput`](https://docs.ag-ui.com/sdk/python/core/types#runagentinput) object, will run the agent and return a stream of AG-UI events. It also takes optional Agent.iter() arguments including `deps`. Use this if you're using a web framework not based on Starlette (e.g. Django or Flask) or want to modify the input or output some way.
1. The AGUIAdapter.dispatch_request() class method takes an agent and a Starlette request (e.g. from FastAPI) coming from an AG-UI frontend, and returns a streaming Starlette response of AG-UI events that you can return directly from your endpoint. It also takes optional Agent.iter() arguments including `deps`, that you can vary for each request (e.g. based on the authenticated user). This is a convenience method that combines AGUIAdapter.from_request(), AGUIAdapter.run_stream(), and AGUIAdapter.streaming_response().
1. AGUIApp represents an ASGI application that handles every AG-UI request by running the agent. It also takes optional Agent.iter() arguments including `deps`, but these will be the same for each request, with the exception of the AG-UI state that's injected as described under [state management](#state-management). This ASGI app can be [mounted](https://fastapi.tiangolo.com/advanced/sub-applications/) at a given path in an existing FastAPI app.

### Handle run input and output directly

This example uses AGUIAdapter.run_stream() and performs its own request parsing and response generation. This can be modified to work with any web framework.

[Learn about Gateway](../../gateway) run_ag_ui.py

```python
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('gateway/openai:gpt-5', instructions='Be fun!')

app = FastAPI()


@app.post('/')
async def run_agent(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = AGUIAdapter.build_run_input(await request.body())  # (1)
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream() # (2)

    sse_event_stream = adapter.encode_stream(event_stream)
    return StreamingResponse(sse_event_stream, media_type=accept) # (3)

```

1. AGUIAdapter.build_run_input() takes the request body as bytes and returns an AG-UI [`RunAgentInput`](https://docs.ag-ui.com/sdk/python/core/types#runagentinput) object. You can also use the AGUIAdapter.from_request() class method to build an adapter directly from a request.
1. AGUIAdapter.run_stream() runs the agent and returns a stream of AG-UI events. It supports the same optional arguments as [`Agent.run_stream_events()`](../../agents/#running-agents), including `deps`. You can also use AGUIAdapter.run_stream_native() to run the agent and return a stream of Pydantic AI events instead, which can then be transformed into AG-UI events using AGUIAdapter.transform_stream().
1. AGUIAdapter.encode_stream() encodes the stream of AG-UI events as strings according to the accept header value. You can also use AGUIAdapter.streaming_response() to generate a streaming response directly from the AG-UI event stream returned by `run_stream()`.

run_ag_ui.py

```python
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('openai:gpt-5', instructions='Be fun!')

app = FastAPI()


@app.post('/')
async def run_agent(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = AGUIAdapter.build_run_input(await request.body())  # (1)
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream() # (2)

    sse_event_stream = adapter.encode_stream(event_stream)
    return StreamingResponse(sse_event_stream, media_type=accept) # (3)

```

1. AGUIAdapter.build_run_input() takes the request body as bytes and returns an AG-UI [`RunAgentInput`](https://docs.ag-ui.com/sdk/python/core/types#runagentinput) object. You can also use the AGUIAdapter.from_request() class method to build an adapter directly from a request.
1. AGUIAdapter.run_stream() runs the agent and returns a stream of AG-UI events. It supports the same optional arguments as [`Agent.run_stream_events()`](../../agents/#running-agents), including `deps`. You can also use AGUIAdapter.run_stream_native() to run the agent and return a stream of Pydantic AI events instead, which can then be transformed into AG-UI events using AGUIAdapter.transform_stream().
1. AGUIAdapter.encode_stream() encodes the stream of AG-UI events as strings according to the accept header value. You can also use AGUIAdapter.streaming_response() to generate a streaming response directly from the AG-UI event stream returned by `run_stream()`.

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn run_ag_ui:app

```

This will expose the agent as an AG-UI server, and your frontend can start sending requests to it.

### Handle a Starlette request

This example uses AGUIAdapter.dispatch_request() to directly handle a FastAPI request and return a response. Something analogous to this will work with any Starlette-based web framework.

[Learn about Gateway](../../gateway) handle_ag_ui_request.py

```python
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('gateway/openai:gpt-5', instructions='Be fun!')

app = FastAPI()

@app.post('/')
async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent) # (1)

```

1. This method essentially does the same as the previous example, but it's more convenient to use when you're already using a Starlette/FastAPI app.

handle_ag_ui_request.py

```python
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('openai:gpt-5', instructions='Be fun!')

app = FastAPI()

@app.post('/')
async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent) # (1)

```

1. This method essentially does the same as the previous example, but it's more convenient to use when you're already using a Starlette/FastAPI app.

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn handle_ag_ui_request:app

```

This will expose the agent as an AG-UI server, and your frontend can start sending requests to it.

### Stand-alone ASGI app

This example uses AGUIApp to turn the agent into a stand-alone ASGI application:

[Learn about Gateway](../../gateway) ag_ui_app.py

```python
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent('gateway/openai:gpt-5', instructions='Be fun!')
app = AGUIApp(agent)

```

ag_ui_app.py

```python
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent('openai:gpt-5', instructions='Be fun!')
app = AGUIApp(agent)

```

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn ag_ui_app:app

```

This will expose the agent as an AG-UI server, and your frontend can start sending requests to it.

## Design

The Pydantic AI AG-UI integration supports all features of the spec:

- [Events](https://docs.ag-ui.com/concepts/events)
- [Messages](https://docs.ag-ui.com/concepts/messages)
- [State Management](https://docs.ag-ui.com/concepts/state)
- [Tools](https://docs.ag-ui.com/concepts/tools)

The integration receives messages in the form of a [`RunAgentInput`](https://docs.ag-ui.com/sdk/python/core/types#runagentinput) object that describes the details of the requested agent run including message history, state, and available tools.

These are converted to Pydantic AI types and passed to the agent's run method. Events from the agent, including tool calls, are converted to AG-UI events and streamed back to the caller as Server-Sent Events (SSE).

A user request may require multiple round trips between client UI and Pydantic AI server, depending on the tools and events needed.

## Features

### State management

The integration provides full support for [AG-UI state management](https://docs.ag-ui.com/concepts/state), which enables real-time synchronization between agents and frontend applications.

In the example below we have document state which is shared between the UI and server using the StateDeps [dependencies type](../../dependencies/) that can be used to automatically validate state contained in [`RunAgentInput.state`](https://docs.ag-ui.com/sdk/js/core/types#runagentinput) using a Pydantic `BaseModel` specified as a generic parameter.

Custom dependencies type with AG-UI state

If you want to use your own dependencies type to hold AG-UI state as well as other things, it needs to implements the StateHandler protocol, meaning it needs to be a [dataclass](https://docs.python.org/3/library/dataclasses.html) with a non-optional `state` field. This lets Pydantic AI ensure that state is properly isolated between requests by building a new dependencies object each time.

If the `state` field's type is a Pydantic `BaseModel` subclass, the raw state dictionary on the request is automatically validated. If not, you can validate the raw value yourself in your dependencies dataclass's `__post_init__` method.

If AG-UI state is provided but your dependencies do not implement StateHandler, Pydantic AI will emit a warning and ignore the state. Use StateDeps or a custom StateHandler implementation to receive and validate the incoming state.

[Learn about Gateway](../../gateway) ag_ui_state.py

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui.app import AGUIApp


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'gateway/openai:gpt-5',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = AGUIApp(agent, deps=StateDeps(DocumentState()))

```

ag_ui_state.py

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui.app import AGUIApp


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'openai:gpt-5',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = AGUIApp(agent, deps=StateDeps(DocumentState()))

```

Since `app` is an ASGI application, it can be used with any ASGI server:

```bash
uvicorn ag_ui_state:app --host 0.0.0.0 --port 9000

```

### Tools

AG-UI frontend tools are seamlessly provided to the Pydantic AI agent, enabling rich user experiences with frontend user interfaces.

### Events

Pydantic AI tools can send [AG-UI events](https://docs.ag-ui.com/concepts/events) simply by returning a [`ToolReturn`](../../tools-advanced/#advanced-tool-returns) object with a [`BaseEvent`](https://docs.ag-ui.com/sdk/python/core/events#baseevent) (or a list of events) as `metadata`, which allows for custom events and state updates.

[Learn about Gateway](../../gateway) ag_ui_tool_events.py

```python
from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui.app import AGUIApp


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'gateway/openai:gpt-5',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = AGUIApp(agent, deps=StateDeps(DocumentState()))


@agent.tool
async def update_state(ctx: RunContext[StateDeps[DocumentState]]) -> ToolReturn:
    return ToolReturn(
        return_value='State updated',
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=ctx.deps.state,
            ),
        ],
    )


@agent.tool_plain
async def custom_events() -> ToolReturn:
    return ToolReturn(
        return_value='Count events sent',
        metadata=[
            CustomEvent(
                type=EventType.CUSTOM,
                name='count',
                value=1,
            ),
            CustomEvent(
                type=EventType.CUSTOM,
                name='count',
                value=2,
            ),
        ]
    )

```

ag_ui_tool_events.py

```python
from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui.app import AGUIApp


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'openai:gpt-5',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = AGUIApp(agent, deps=StateDeps(DocumentState()))


@agent.tool
async def update_state(ctx: RunContext[StateDeps[DocumentState]]) -> ToolReturn:
    return ToolReturn(
        return_value='State updated',
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=ctx.deps.state,
            ),
        ],
    )


@agent.tool_plain
async def custom_events() -> ToolReturn:
    return ToolReturn(
        return_value='Count events sent',
        metadata=[
            CustomEvent(
                type=EventType.CUSTOM,
                name='count',
                value=1,
            ),
            CustomEvent(
                type=EventType.CUSTOM,
                name='count',
                value=2,
            ),
        ]
    )

```

Since `app` is an ASGI application, it can be used with any ASGI server:

```bash
uvicorn ag_ui_tool_events:app --host 0.0.0.0 --port 9000

```

## Examples

For more examples of how to use AGUIApp see [`pydantic_ai_examples.ag_ui`](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples/ag_ui), which includes a server for use with the [AG-UI Dojo](https://docs.ag-ui.com/tutorials/debugging#the-ag-ui-dojo).

# UI Event Streams

If you're building a chat app or other interactive frontend for an AI agent, your backend will need to receive agent run input (like a chat message or complete [message history](../../message-history/)) from the frontend, and will need to stream the [agent's events](../../agents/#streaming-all-events) (like text, thinking, and tool calls) to the frontend so that the user knows what's happening in real time.

While your frontend could use Pydantic AI's ModelRequest and AgentStreamEvent directly, you'll typically want to use a UI event stream protocol that's natively supported by your frontend framework.

Pydantic AI natively supports two UI event stream protocols:

- [Agent-User Interaction (AG-UI) Protocol](../ag-ui/)
- [Vercel AI Data Stream Protocol](../vercel-ai/)

These integrations are implemented as subclasses of the abstract UIAdapter class, so they also serve as a reference for integrating with other UI event stream protocols.

## Usage

The protocol-specific UIAdapter subclass (i.e. AGUIAdapter or VercelAIAdapter) is responsible for transforming agent run input received from the frontend into arguments for [`Agent.run_stream_events()`](../../agents/#running-agents), running the agent, and then transforming Pydantic AI events into protocol-specific events. The event stream transformation is handled by a protocol-specific UIEventStream subclass, but you typically won't use this directly.

If you're using a Starlette-based web framework like FastAPI, you can use the UIAdapter.dispatch_request() class method from an endpoint function to directly handle a request and return a streaming response of protocol-specific events. This is demonstrated in the next section.

If you're using a web framework not based on Starlette (e.g. Django or Flask) or need fine-grained control over the input or output, you can create a `UIAdapter` instance and directly use its methods. This is demonstrated in "Advanced Usage" section below.

### Usage with Starlette/FastAPI

Besides the request, UIAdapter.dispatch_request() takes the agent, the same optional arguments as [`Agent.run_stream_events()`](../../agents/#running-agents), and an optional `on_complete` callback function that receives the completed AgentRunResult and can optionally yield additional protocol-specific events.

Note

These examples use the `VercelAIAdapter`, but the same patterns apply to all `UIAdapter` subclasses.

[Learn about Gateway](../../gateway) dispatch_request.py

```python
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('gateway/openai:gpt-5')

app = FastAPI()

@app.post('/chat')
async def chat(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent)

```

dispatch_request.py

```python
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('openai:gpt-5')

app = FastAPI()

@app.post('/chat')
async def chat(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent)

```

### Advanced Usage

If you're using a web framework not based on Starlette (e.g. Django or Flask) or need fine-grained control over the input or output, you can create a `UIAdapter` instance and directly use its methods, which can be chained to accomplish the same thing as the `UIAdapter.dispatch_request()` class method shown above:

1. The UIAdapter.build_run_input() class method takes the request body as bytes and returns a protocol-specific run input object, which you can then pass to the UIAdapter() constructor along with the agent.
   - You can also use the UIAdapter.from_request() class method to build an adapter directly from a Starlette/FastAPI request.
1. The UIAdapter.run_stream() method runs the agent and returns a stream of protocol-specific events. It supports the same optional arguments as [`Agent.run_stream_events()`](../../agents/#running-agents) and an optional `on_complete` callback function that receives the completed AgentRunResult and can optionally yield additional protocol-specific events.
   - You can also use UIAdapter.run_stream_native() to run the agent and return a stream of Pydantic AI events instead, which can then be transformed into protocol-specific events using UIAdapter.transform_stream().
1. The UIAdapter.encode_stream() method encodes the stream of protocol-specific events as SSE (HTTP Server-Sent Events) strings, which you can then return as a streaming response.
   - You can also use UIAdapter.streaming_response() to generate a Starlette/FastAPI streaming response directly from the protocol-specific event stream returned by `run_stream()`.

Note

This example uses FastAPI, but can be modified to work with any web framework.

[Learn about Gateway](../../gateway) run_stream.py

```python
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('gateway/openai:gpt-5')

app = FastAPI()


@app.post('/chat')
async def chat(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = VercelAIAdapter.build_run_input(await request.body())
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = VercelAIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream()

    sse_event_stream = adapter.encode_stream(event_stream)
    return StreamingResponse(sse_event_stream, media_type=accept)

```

run_stream.py

```python
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('openai:gpt-5')

app = FastAPI()


@app.post('/chat')
async def chat(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = VercelAIAdapter.build_run_input(await request.body())
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = VercelAIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream()

    sse_event_stream = adapter.encode_stream(event_stream)
    return StreamingResponse(sse_event_stream, media_type=accept)

```

# Vercel AI Data Stream Protocol

Pydantic AI natively supports the [Vercel AI Data Stream Protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol) to receive agent run input from, and stream events to, a [Vercel AI Elements](https://ai-sdk.dev/elements) frontend.

## Usage

The VercelAIAdapter class is responsible for transforming agent run input received from the frontend into arguments for [`Agent.run_stream_events()`](../../agents/#running-agents), running the agent, and then transforming Pydantic AI events into Vercel AI events. The event stream transformation is handled by the VercelAIEventStream class, but you typically won't use this directly.

If you're using a Starlette-based web framework like FastAPI, you can use the VercelAIAdapter.dispatch_request() class method from an endpoint function to directly handle a request and return a streaming response of Vercel AI events. This is demonstrated in the next section.

If you're using a web framework not based on Starlette (e.g. Django or Flask) or need fine-grained control over the input or output, you can create a `VercelAIAdapter` instance and directly use its methods. This is demonstrated in "Advanced Usage" section below.

### Usage with Starlette/FastAPI

Besides the request, VercelAIAdapter.dispatch_request() takes the agent, the same optional arguments as [`Agent.run_stream_events()`](../../agents/#running-agents), and an optional `on_complete` callback function that receives the completed AgentRunResult and can optionally yield additional Vercel AI events.

[Learn about Gateway](../../gateway) dispatch_request.py

```python
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('gateway/openai:gpt-5')

app = FastAPI()

@app.post('/chat')
async def chat(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent)

```

dispatch_request.py

```python
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('openai:gpt-5')

app = FastAPI()

@app.post('/chat')
async def chat(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent)

```

### Advanced Usage

If you're using a web framework not based on Starlette (e.g. Django or Flask) or need fine-grained control over the input or output, you can create a `VercelAIAdapter` instance and directly use its methods, which can be chained to accomplish the same thing as the `VercelAIAdapter.dispatch_request()` class method shown above:

1. The VercelAIAdapter.build_run_input() class method takes the request body as bytes and returns a Vercel AI RequestData run input object, which you can then pass to the VercelAIAdapter() constructor along with the agent.
   - You can also use the VercelAIAdapter.from_request() class method to build an adapter directly from a Starlette/FastAPI request.
1. The VercelAIAdapter.run_stream() method runs the agent and returns a stream of Vercel AI events. It supports the same optional arguments as [`Agent.run_stream_events()`](../../agents/#running-agents) and an optional `on_complete` callback function that receives the completed AgentRunResult and can optionally yield additional Vercel AI events.
   - You can also use VercelAIAdapter.run_stream_native() to run the agent and return a stream of Pydantic AI events instead, which can then be transformed into Vercel AI events using VercelAIAdapter.transform_stream().
1. The VercelAIAdapter.encode_stream() method encodes the stream of Vercel AI events as SSE (HTTP Server-Sent Events) strings, which you can then return as a streaming response.
   - You can also use VercelAIAdapter.streaming_response() to generate a Starlette/FastAPI streaming response directly from the Vercel AI event stream returned by `run_stream()`.

Note

This example uses FastAPI, but can be modified to work with any web framework.

[Learn about Gateway](../../gateway) run_stream.py

```python
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('gateway/openai:gpt-5')

app = FastAPI()


@app.post('/chat')
async def chat(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = VercelAIAdapter.build_run_input(await request.body())
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = VercelAIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream()

    sse_event_stream = adapter.encode_stream(event_stream)
    return StreamingResponse(sse_event_stream, media_type=accept)

```

run_stream.py

```python
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('openai:gpt-5')

app = FastAPI()


@app.post('/chat')
async def chat(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = VercelAIAdapter.build_run_input(await request.body())
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = VercelAIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream()

    sse_event_stream = adapter.encode_stream(event_stream)
    return StreamingResponse(sse_event_stream, media_type=accept)

```
