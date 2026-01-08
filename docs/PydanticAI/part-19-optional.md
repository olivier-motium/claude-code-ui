<!-- Source: _complete-doc.md -->

# Optional

# Upgrade Guide

In September 2025, Pydantic AI reached V1, which means we're committed to API stability: we will not introduce changes that break your code until V2. For more information, review our [Version Policy](../version-policy/).

## Breaking Changes

Here's a filtered list of the breaking changes for each version to help you upgrade Pydantic AI.

### v1.0.1 (2025-09-05)

The following breaking change was accidentally left out of v1.0.0:

- See [#2808](https://github.com/pydantic/pydantic-ai/pull/2808) - Remove `Python` evaluator from `pydantic_evals` for security reasons

### v1.0.0 (2025-09-04)

- See [#2725](https://github.com/pydantic/pydantic-ai/pull/2725) - Drop support for Python 3.9
- See [#2738](https://github.com/pydantic/pydantic-ai/pull/2738) - Make many dataclasses require keyword arguments
- See [#2715](https://github.com/pydantic/pydantic-ai/pull/2715) - Remove `cases` and `averages` attributes from `pydantic_evals` spans
- See [#2798](https://github.com/pydantic/pydantic-ai/pull/2798) - Change `ModelRequest.parts` and `ModelResponse.parts` types from `list` to `Sequence`
- See [#2726](https://github.com/pydantic/pydantic-ai/pull/2726) - Default `InstrumentationSettings` version to 2
- See [#2717](https://github.com/pydantic/pydantic-ai/pull/2717) - Remove errors when passing `AsyncRetrying` or `Retrying` object to `AsyncTenacityTransport` or `TenacityTransport` instead of `RetryConfig`

### v0.x.x

Before V1, minor versions were used to introduce breaking changes:

**v0.8.0 (2025-08-26)**

See [#2689](https://github.com/pydantic/pydantic-ai/pull/2689) - `AgentStreamEvent` was expanded to be a union of `ModelResponseStreamEvent` and `HandleResponseEvent`, simplifying the `event_stream_handler` function signature. Existing code accepting `AgentStreamEvent | HandleResponseEvent` will continue to work.

**v0.7.6 (2025-08-26)**

The following breaking change was inadvertently released in a patch version rather than a minor version:

See [#2670](https://github.com/pydantic/pydantic-ai/pull/2670) - `TenacityTransport` and `AsyncTenacityTransport` now require the use of `pydantic_ai.retries.RetryConfig` (which is just a `TypedDict` containing the kwargs to `tenacity.retry`) instead of `tenacity.Retrying` or `tenacity.AsyncRetrying`.

**v0.7.0 (2025-08-12)**

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.StreamedResponse` now yields a `FinalResultEvent` along with the existing `PartStartEvent` and `PartDeltaEvent`. If you're using `pydantic_ai.direct.model_request_stream` or `pydantic_ai.direct.model_request_stream_sync`, you may need to update your code to account for this.

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.Model.request_stream` now receives a `run_context` argument. If you've implemented a custom `Model` subclass, you will need to account for this.

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.StreamedResponse` now requires a `model_request_parameters` field and constructor argument. If you've implemented a custom `Model` subclass and implemented `request_stream`, you will need to account for this.

**v0.6.0 (2025-08-06)**

This release was meant to clean some old deprecated code, so we can get a step closer to V1.

See [#2440](https://github.com/pydantic/pydantic-ai/pull/2440) - The `next` method was removed from the `Graph` class. Use `async with graph.iter(...) as run: run.next()` instead.

See [#2441](https://github.com/pydantic/pydantic-ai/pull/2441) - The `result_type`, `result_tool_name` and `result_tool_description` arguments were removed from the `Agent` class. Use `output_type` instead.

See [#2441](https://github.com/pydantic/pydantic-ai/pull/2441) - The `result_retries` argument was also removed from the `Agent` class. Use `output_retries` instead.

See [#2443](https://github.com/pydantic/pydantic-ai/pull/2443) - The `data` property was removed from the `FinalResult` class. Use `output` instead.

See [#2445](https://github.com/pydantic/pydantic-ai/pull/2445) - The `get_data` and `validate_structured_result` methods were removed from the `StreamedRunResult` class. Use `get_output` and `validate_structured_output` instead.

See [#2446](https://github.com/pydantic/pydantic-ai/pull/2446) - The `format_as_xml` function was moved to the `pydantic_ai.format_as_xml` module. Import it via `from pydantic_ai import format_as_xml` instead.

See [#2451](https://github.com/pydantic/pydantic-ai/pull/2451) - Removed deprecated `Agent.result_validator` method, `Agent.last_run_messages` property, `AgentRunResult.data` property, and `result_tool_return_content` parameters from result classes.

**v0.5.0 (2025-08-04)**

See [#2388](https://github.com/pydantic/pydantic-ai/pull/2388) - The `source` field of an `EvaluationResult` is now of type `EvaluatorSpec` rather than the actual source `Evaluator` instance, to help with serialization/deserialization.

See [#2163](https://github.com/pydantic/pydantic-ai/pull/2163) - The `EvaluationReport.print` and `EvaluationReport.console_table` methods now require most arguments be passed by keyword.

**v0.4.0 (2025-07-08)**

See [#1799](https://github.com/pydantic/pydantic-ai/pull/1799) - Pydantic Evals `EvaluationReport` and `ReportCase` are now generic dataclasses instead of Pydantic models. If you were serializing them using `model_dump()`, you will now need to use the `EvaluationReportAdapter` and `ReportCaseAdapter` type adapters instead.

See [#1507](https://github.com/pydantic/pydantic-ai/pull/1507) - The `ToolDefinition` `description` argument is now optional and the order of positional arguments has changed from `name, description, parameters_json_schema, ...` to `name, parameters_json_schema, description, ...` to account for this.

**v0.3.0 (2025-06-18)**

See [#1142](https://github.com/pydantic/pydantic-ai/pull/1142) â€” Adds support for thinking parts.

We now convert the thinking blocks (`"<think>..."</think>"`) in provider specific text parts to Pydantic AI `ThinkingPart`s. Also, as part of this release, we made the choice to not send back the `ThinkingPart`s to the provider - the idea is to save costs on behalf of the user. In the future, we intend to add a setting to customize this behavior.

**v0.2.0 (2025-05-12)**

See [#1647](https://github.com/pydantic/pydantic-ai/pull/1647) â€” usage makes sense as part of `ModelResponse`, and could be really useful in "messages" (really a sequence of requests and response). In this PR:

- Adds `usage` to `ModelResponse` (field has a default factory of `Usage()` so it'll work to load data that doesn't have usage)
- changes the return type of `Model.request` to just `ModelResponse` instead of `tuple[ModelResponse, Usage]`

**v0.1.0 (2025-04-15)**

See [#1248](https://github.com/pydantic/pydantic-ai/pull/1248) â€” the attribute/parameter name `result` was renamed to `output` in many places. Hopefully all changes keep a deprecated attribute or parameter with the old name, so you should get many deprecation warnings.

See [#1484](https://github.com/pydantic/pydantic-ai/pull/1484) â€” `format_as_xml` was moved and made available to import from the package root, e.g. `from pydantic_ai import format_as_xml`.

## Full Changelog

For the full changelog, see [GitHub Releases](https://github.com/pydantic/pydantic-ai/releases).

# Command Line Interface (CLI)

**Pydantic AI** comes with a CLI, `clai` (pronounced "clay") which you can use to interact with various LLMs from the command line. It provides a convenient way to chat with language models and quickly get answers right in the terminal.

We originally developed this CLI for our own use, but found ourselves using it so frequently that we decided to share it as part of the Pydantic AI package.

We plan to continue adding new features, such as interaction with MCP servers, access to tools, and more.

## Usage

You'll need to set an environment variable depending on the provider you intend to use.

E.g. if you're using OpenAI, set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'

```

Then with [`uvx`](https://docs.astral.sh/uv/guides/tools/), run:

```bash
uvx clai

```

Or to install `clai` globally [with `uv`](https://docs.astral.sh/uv/guides/tools/#installing-tools), run:

```bash
uv tool install clai
...
clai

```

Or with `pip`, run:

```bash
pip install clai
...
clai

```

Either way, running `clai` will start an interactive session where you can chat with the AI model. Special commands available in interactive mode:

- `/exit`: Exit the session
- `/markdown`: Show the last response in markdown format
- `/multiline`: Toggle multiline input mode (use Ctrl+D to submit)
- `/cp`: Copy the last response to clipboard

### Help

To get help on the CLI, use the `--help` flag:

```bash
uvx clai --help

```

### Choose a model

You can specify which model to use with the `--model` flag:

```bash
uvx clai --model anthropic:claude-sonnet-4-0

```

(a full list of models available can be printed with `uvx clai --list-models`)

### Custom Agents

You can specify a custom agent using the `--agent` flag with a module path and variable name:

[Learn about Gateway](../gateway) custom_agent.py

```python
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5', instructions='You always respond in Italian.')

```

custom_agent.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='You always respond in Italian.')

```

Then run:

```bash
uvx clai --agent custom_agent:agent "What's the weather today?"

```

The format must be `module:variable` where:

- `module` is the importable Python module path
- `variable` is the name of the Agent instance in that module

Additionally, you can directly launch CLI mode from an `Agent` instance using `Agent.to_cli_sync()`:

[Learn about Gateway](../gateway) agent_to_cli_sync.py

```python
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5', instructions='You always respond in Italian.')
agent.to_cli_sync()

```

agent_to_cli_sync.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='You always respond in Italian.')
agent.to_cli_sync()

```

You can also use the async interface with `Agent.to_cli()`:

[Learn about Gateway](../gateway) agent_to_cli.py

```python
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5', instructions='You always respond in Italian.')

async def main():
    await agent.to_cli()

```

agent_to_cli.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='You always respond in Italian.')

async def main():
    await agent.to_cli()

```

*(You'll need to add `asyncio.run(main())` to run `main`)*

### Message History

Both `Agent.to_cli()` and `Agent.to_cli_sync()` support a `message_history` parameter, allowing you to continue an existing conversation or provide conversation context:

[Learn about Gateway](../gateway) agent_with_history.py

```python
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

agent = Agent('gateway/openai:gpt-5')

# Create some conversation history
message_history: list[ModelMessage] = [
    ModelRequest([UserPromptPart(content='What is 2+2?')]),
    ModelResponse([TextPart(content='2+2 equals 4.')])
]

# Start CLI with existing conversation context
agent.to_cli_sync(message_history=message_history)

```

agent_with_history.py

```python
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

agent = Agent('openai:gpt-5')

# Create some conversation history
message_history: list[ModelMessage] = [
    ModelRequest([UserPromptPart(content='What is 2+2?')]),
    ModelResponse([TextPart(content='2+2 equals 4.')])
]

# Start CLI with existing conversation context
agent.to_cli_sync(message_history=message_history)

```

The CLI will start with the provided conversation history, allowing the agent to refer back to previous exchanges and maintain context throughout the session.

We'd love you to contribute to Pydantic AI!

## Installation and Setup

Clone your fork and cd into the repo directory

```bash
git clone git@github.com:<your username>/pydantic-ai.git
cd pydantic-ai

```

Install `uv` (version 0.4.30 or later), `pre-commit` and `deno`:

- [`uv` install docs](https://docs.astral.sh/uv/getting-started/installation/)
- [`pre-commit` install docs](https://pre-commit.com/#install)
- [`deno` install docs](https://docs.deno.com/runtime/getting_started/installation/)

To install `pre-commit` you can run the following command:

```bash
uv tool install pre-commit

```

For `deno`, you can run the following, or check [their documentation](https://docs.deno.com/runtime/getting_started/installation/) for alternative installation methods:

```bash
curl -fsSL https://deno.land/install.sh | sh

```

Install `pydantic-ai`, all dependencies and pre-commit hooks

```bash
make install

```

## Running Tests etc.

We use `make` to manage most commands you'll need to run.

For details on available commands, run:

```bash
make help

```

To run code formatting, linting, static type checks, and tests with coverage report generation, run:

```bash
make

```

## Documentation Changes

To run the documentation page locally, run:

```bash
uv run mkdocs serve

```

## Rules for adding new models to Pydantic AI

To avoid an excessive workload for the maintainers of Pydantic AI, we can't accept all model contributions, so we're setting the following rules for when we'll accept new models and when we won't. This should hopefully reduce the chances of disappointment and wasted work.

- To add a new model with an extra dependency, that dependency needs > 500k monthly downloads from PyPI consistently over 3 months or more
- To add a new model which uses another models logic internally and has no extra dependencies, that model's GitHub org needs > 20k stars in total
- For any other model that's just a custom URL and API key, we're happy to add a one-paragraph description with a link and instructions on the URL to use
- For any other model that requires more logic, we recommend you release your own Python package `pydantic-ai-xxx`, which depends on [`pydantic-ai-slim`](../install/#slim-install) and implements a model that inherits from our Model ABC

If you're unsure about adding a model, please [create an issue](https://github.com/pydantic/pydantic-ai/issues).

# Pydantic Logfire Debugging and Monitoring

Applications that use LLMs have some challenges that are well known and understood: LLMs are **slow**, **unreliable** and **expensive**.

These applications also have some challenges that most developers have encountered much less often: LLMs are **fickle** and **non-deterministic**. Subtle changes in a prompt can completely change a model's performance, and there's no `EXPLAIN` query you can run to understand why.

Warning

From a software engineers point of view, you can think of LLMs as the worst database you've ever heard of, but worse.

If LLMs weren't so bloody useful, we'd never touch them.

To build successful applications with LLMs, we need new tools to understand both model performance, and the behavior of applications that rely on them.

LLM Observability tools that just let you understand how your model is performing are useless: making API calls to an LLM is easy, it's building that into an application that's hard.

## Pydantic Logfire

[Pydantic Logfire](https://pydantic.dev/logfire) is an observability platform developed by the team who created and maintain Pydantic Validation and Pydantic AI. Logfire aims to let you understand your entire application: Gen AI, classic predictive AI, HTTP traffic, database queries and everything else a modern application needs, all using OpenTelemetry.

Pydantic Logfire is a commercial product

Logfire is a commercially supported, hosted platform with an extremely generous and perpetual [free tier](https://pydantic.dev/pricing/). You can sign up and start using Logfire in a couple of minutes. Logfire can also be self-hosted on the enterprise tier.

Pydantic AI has built-in (but optional) support for Logfire. That means if the `logfire` package is installed and configured and agent instrumentation is enabled then detailed information about agent runs is sent to Logfire. Otherwise there's virtually no overhead and nothing is sent.

Here's an example showing details of running the [Weather Agent](../examples/weather-agent/) in Logfire:

A trace is generated for the agent run, and spans are emitted for each model request and tool call.

## Using Logfire

To use Logfire, you'll need a Logfire [account](https://logfire.pydantic.dev). The Logfire Python SDK is included with `pydantic-ai`:

```bash
pip install pydantic-ai

```

```bash
uv add pydantic-ai

```

Or if you're using the slim package, you can install it with the `logfire` optional group:

```bash
pip install "pydantic-ai-slim[logfire]"

```

```bash
uv add "pydantic-ai-slim[logfire]"

```

Then authenticate your local environment with Logfire:

```bash
 logfire auth

```

```bash
uv run logfire auth

```

And configure a project to send data to:

```bash
 logfire projects new

```

```bash
uv run logfire projects new

```

(Or use an existing project with `logfire projects use`)

This will write to a `.logfire` directory in the current working directory, which the Logfire SDK will use for configuration at run time.

With that, you can start using Logfire to instrument Pydantic AI code:

[Learn about Gateway](../gateway) instrument_pydantic_ai.py

```python
import logfire

from pydantic_ai import Agent

logfire.configure()  # (1)!
logfire.instrument_pydantic_ai()  # (2)!

agent = Agent('gateway/openai:gpt-5', instructions='Be concise, reply with one sentence.')
result = agent.run_sync('Where does "hello world" come from?')  # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""

```

1. logfire.configure() configures the SDK, by default it will find the write token from the `.logfire` directory, but you can also pass a token directly.
1. logfire.instrument_pydantic_ai() enables instrumentation of Pydantic AI.
1. Since we've enabled instrumentation, a trace will be generated for each run, with spans emitted for models calls and tool function execution

instrument_pydantic_ai.py

```python
import logfire

from pydantic_ai import Agent

logfire.configure()  # (1)!
logfire.instrument_pydantic_ai()  # (2)!

agent = Agent('openai:gpt-5', instructions='Be concise, reply with one sentence.')
result = agent.run_sync('Where does "hello world" come from?')  # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""

```

1. logfire.configure() configures the SDK, by default it will find the write token from the `.logfire` directory, but you can also pass a token directly.
1. logfire.instrument_pydantic_ai() enables instrumentation of Pydantic AI.
1. Since we've enabled instrumentation, a trace will be generated for each run, with spans emitted for models calls and tool function execution

*(This example is complete, it can be run "as is")*

Which will display in Logfire thus:

The [Logfire documentation](https://logfire.pydantic.dev/docs/) has more details on how to use Logfire, including how to instrument other libraries like [HTTPX](https://logfire.pydantic.dev/docs/integrations/http-clients/httpx/) and [FastAPI](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/).

Since Logfire is built on [OpenTelemetry](https://opentelemetry.io/), you can use the Logfire Python SDK to send data to any OpenTelemetry collector, see [below](#using-opentelemetry).

### Debugging

To demonstrate how Logfire can let you visualise the flow of a Pydantic AI run, here's the view you get from Logfire while running the [chat app examples](../examples/chat-app/):

### Monitoring Performance

We can also query data with SQL in Logfire to monitor the performance of an application. Here's a real world example of using Logfire to monitor Pydantic AI runs inside Logfire itself:

### Monitoring HTTP Requests

As per Hamel Husain's influential 2024 blog post ["Fuck You, Show Me The Prompt."](https://hamel.dev/blog/posts/prompt/) (bear with the capitalization, the point is valid), it's often useful to be able to view the raw HTTP requests and responses made to model providers.

To observe raw HTTP requests made to model providers, you can use Logfire's [HTTPX instrumentation](https://logfire.pydantic.dev/docs/integrations/http-clients/httpx/) since all provider SDKs (except for [Bedrock](../models/bedrock/)) use the [HTTPX](https://www.python-httpx.org/) library internally:

[Learn about Gateway](../gateway) with_logfire_instrument_httpx.py

```python
import logfire

from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)  # (1)!

agent = Agent('gateway/openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.

```

1. See the logfire.instrument_httpx docs more details, `capture_all=True` means both headers and body are captured for both the request and response.

with_logfire_instrument_httpx.py

```python
import logfire

from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)  # (1)!

agent = Agent('openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.

```

1. See the logfire.instrument_httpx docs more details, `capture_all=True` means both headers and body are captured for both the request and response.

## Using OpenTelemetry

Pydantic AI's instrumentation uses [OpenTelemetry](https://opentelemetry.io/) (OTel), which Logfire is based on.

This means you can debug and monitor Pydantic AI with any OpenTelemetry backend.

Pydantic AI follows the [OpenTelemetry Semantic Conventions for Generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/), so while we think you'll have the best experience using the Logfire platform , you should be able to use any OTel service with GenAI support.

### Logfire with an alternative OTel backend

You can use the Logfire SDK completely freely and send the data to any OpenTelemetry backend.

Here's an example of configuring the Logfire library to send data to the excellent [otel-tui](https://github.com/ymtdzzz/otel-tui) â€” an open source terminal based OTel backend and viewer (no association with Pydantic Validation).

Run `otel-tui` with docker (see [the otel-tui readme](https://github.com/ymtdzzz/otel-tui) for more instructions):

Terminal

```text
docker run --rm -it -p 4318:4318 --name otel-tui ymtdzzz/otel-tui:latest

```

then run,

[Learn about Gateway](../gateway) otel_tui.py

```python
import os

import logfire

from pydantic_ai import Agent

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318'  # (1)!
logfire.configure(send_to_logfire=False)  # (2)!
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

agent = Agent('gateway/openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> Paris

```

1. Set the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable to the URL of your OpenTelemetry backend. If you're using a backend that requires authentication, you may need to set [other environment variables](https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/). Of course, these can also be set outside the process, e.g. with `export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`.
1. We configure Logfire to disable sending data to the Logfire OTel backend itself. If you removed `send_to_logfire=False`, data would be sent to both Logfire and your OpenTelemetry backend.

otel_tui.py

```python
import os

import logfire

from pydantic_ai import Agent

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318'  # (1)!
logfire.configure(send_to_logfire=False)  # (2)!
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

agent = Agent('openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> Paris

```

1. Set the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable to the URL of your OpenTelemetry backend. If you're using a backend that requires authentication, you may need to set [other environment variables](https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/). Of course, these can also be set outside the process, e.g. with `export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`.
1. We configure Logfire to disable sending data to the Logfire OTel backend itself. If you removed `send_to_logfire=False`, data would be sent to both Logfire and your OpenTelemetry backend.

Running the above code will send tracing data to `otel-tui`, which will display like this:

Running the [weather agent](../examples/weather-agent/) example connected to `otel-tui` shows how it can be used to visualise a more complex trace:

For more information on using the Logfire SDK to send data to alternative backends, see [the Logfire documentation](https://logfire.pydantic.dev/docs/how-to-guides/alternative-backends/).

### OTel without Logfire

You can also emit OpenTelemetry data from Pydantic AI without using Logfire at all.

To do this, you'll need to install and configure the OpenTelemetry packages you need. To run the following examples, use

Terminal

```text
uv run \
  --with 'pydantic-ai-slim[openai]' \
  --with opentelemetry-sdk \
  --with opentelemetry-exporter-otlp \
  raw_otel.py

```

[Learn about Gateway](../gateway) raw_otel.py

```python
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider

from pydantic_ai import Agent

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318'
exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(exporter)
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(span_processor)

set_tracer_provider(tracer_provider)

Agent.instrument_all()
agent = Agent('gateway/openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> Paris

```

raw_otel.py

```python
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider

from pydantic_ai import Agent

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://localhost:4318'
exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(exporter)
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(span_processor)

set_tracer_provider(tracer_provider)

Agent.instrument_all()
agent = Agent('openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> Paris

```

### Alternative Observability backends

Because Pydantic AI uses OpenTelemetry for observability, you can easily configure it to send data to any OpenTelemetry-compatible backend, not just our observability platform [Pydantic Logfire](#pydantic-logfire).

The following providers have dedicated documentation on Pydantic AI:

- [Langfuse](https://langfuse.com/docs/integrations/pydantic-ai)
- [W&B Weave](https://weave-docs.wandb.ai/guides/integrations/pydantic_ai/)
- [Arize](https://arize.com/docs/ax/observe/tracing-integrations-auto/pydantic-ai)
- [Openlayer](https://www.openlayer.com/docs/integrations/pydantic-ai)
- [OpenLIT](https://docs.openlit.io/latest/integrations/pydantic)
- [LangWatch](https://docs.langwatch.ai/integration/python/integrations/pydantic-ai)
- [Patronus AI](https://docs.patronus.ai/docs/percival/pydantic)
- [Opik](https://www.comet.com/docs/opik/tracing/integrations/pydantic-ai)
- [mlflow](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/pydantic_ai)
- [Agenta](https://docs.agenta.ai/observability/integrations/pydanticai)
- [Confident AI](https://documentation.confident-ai.com/docs/llm-tracing/integrations/pydanticai)
- [LangWatch](https://docs.langwatch.ai/integration/python/integrations/pydantic-ai)
- [Braintrust](https://www.braintrust.dev/docs/integrations/sdk-integrations/pydantic-ai)

## Advanced usage

### Configuring data format

Pydantic AI follows the [OpenTelemetry Semantic Conventions for Generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Specifically, it follows version 1.37.0 of the conventions by default, with a few exceptions. Certain span and attribute names are not spec compliant by default for compatibility reasons, but can be made compliant by passing InstrumentationSettings(version=3) (the default is currently `version=2`). This will change the following:

- The span name `agent run` becomes `invoke_agent {gen_ai.agent.name}` (with the agent name filled in)
- The span name `running tool` becomes `execute_tool {gen_ai.tool.name}` (with the tool name filled in)
- The attribute name `tool_arguments` becomes `gen_ai.tool.call.arguments`
- The attribute name `tool_response` becomes `gen_ai.tool.call.result`

To use [OpenTelemetry semantic conventions version 1.36.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/README.md) or older, pass InstrumentationSettings(version=1). Moreover, those semantic conventions specify that messages should be captured as individual events (logs) that are children of the request span, whereas by default, Pydantic AI instead collects these events into a JSON array which is set as a single large attribute called `events` on the request span. To change this, use `event_mode='logs'`:

[Learn about Gateway](../gateway) instrumentation_settings_event_mode.py

```python
import logfire

from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai(version=1, event_mode='logs')
agent = Agent('gateway/openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.

```

instrumentation_settings_event_mode.py

```python
import logfire

from pydantic_ai import Agent

logfire.configure()
logfire.instrument_pydantic_ai(version=1, event_mode='logs')
agent = Agent('openai:gpt-5')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.

```

This won't look as good in the Logfire UI, and will also be removed from Pydantic AI in a future release, but may be useful for backwards compatibility.

Note that the OpenTelemetry Semantic Conventions are still experimental and are likely to change.

### Setting OpenTelemetry SDK providers

By default, the global `TracerProvider` and `EventLoggerProvider` are used. These are set automatically by `logfire.configure()`. They can also be set by the `set_tracer_provider` and `set_event_logger_provider` functions in the OpenTelemetry Python SDK. You can set custom providers with InstrumentationSettings.

[Learn about Gateway](../gateway) instrumentation_settings_providers.py

```python
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk.trace import TracerProvider

from pydantic_ai import Agent, InstrumentationSettings

instrumentation_settings = InstrumentationSettings(
    tracer_provider=TracerProvider(),
    event_logger_provider=EventLoggerProvider(),
)

agent = Agent('gateway/openai:gpt-5', instrument=instrumentation_settings)
# or to instrument all agents:
Agent.instrument_all(instrumentation_settings)

```

instrumentation_settings_providers.py

```python
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk.trace import TracerProvider

from pydantic_ai import Agent, InstrumentationSettings

instrumentation_settings = InstrumentationSettings(
    tracer_provider=TracerProvider(),
    event_logger_provider=EventLoggerProvider(),
)

agent = Agent('openai:gpt-5', instrument=instrumentation_settings)
# or to instrument all agents:
Agent.instrument_all(instrumentation_settings)

```

### Instrumenting a specific `Model`

instrumented_model_example.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel

settings = InstrumentationSettings()
model = InstrumentedModel('openai:gpt-5', settings)
agent = Agent(model)

```

### Excluding binary content

[Learn about Gateway](../gateway) excluding_binary_content.py

```python
from pydantic_ai import Agent, InstrumentationSettings

instrumentation_settings = InstrumentationSettings(include_binary_content=False)

agent = Agent('gateway/openai:gpt-5', instrument=instrumentation_settings)
# or to instrument all agents:
Agent.instrument_all(instrumentation_settings)

```

excluding_binary_content.py

```python
from pydantic_ai import Agent, InstrumentationSettings

instrumentation_settings = InstrumentationSettings(include_binary_content=False)

agent = Agent('openai:gpt-5', instrument=instrumentation_settings)
# or to instrument all agents:
Agent.instrument_all(instrumentation_settings)

```

### Excluding prompts and completions

For privacy and security reasons, you may want to monitor your agent's behavior and performance without exposing sensitive user data or proprietary prompts in your observability platform. Pydantic AI allows you to exclude the actual content from instrumentation events while preserving the structural information needed for debugging and monitoring.

When `include_content=False` is set, Pydantic AI will exclude sensitive content from OpenTelemetry events, including user prompts and model completions, tool call arguments and responses, and any other message content.

[Learn about Gateway](../gateway) excluding_sensitive_content.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

instrumentation_settings = InstrumentationSettings(include_content=False)

agent = Agent('gateway/openai:gpt-5', instrument=instrumentation_settings)
# or to instrument all agents:
Agent.instrument_all(instrumentation_settings)

```

excluding_sensitive_content.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

instrumentation_settings = InstrumentationSettings(include_content=False)

agent = Agent('openai:gpt-5', instrument=instrumentation_settings)
# or to instrument all agents:
Agent.instrument_all(instrumentation_settings)

```

This setting is particularly useful in production environments where compliance requirements or data sensitivity concerns make it necessary to limit what content is sent to your observability platform.

# Unit testing

Writing unit tests for Pydantic AI code is just like unit tests for any other Python code.

Because for the most part they're nothing new, we have pretty well established tools and patterns for writing and running these kinds of tests.

Unless you're really sure you know better, you'll probably want to follow roughly this strategy:

- Use [`pytest`](https://docs.pytest.org/en/stable/) as your test harness
- If you find yourself typing out long assertions, use [inline-snapshot](https://15r10nk.github.io/inline-snapshot/latest/)
- Similarly, [dirty-equals](https://dirty-equals.helpmanual.io/latest/) can be useful for comparing large data structures
- Use TestModel or FunctionModel in place of your actual model to avoid the usage, latency and variability of real LLM calls
- Use Agent.override to replace an agent's model, dependencies, or toolsets inside your application logic
- Set ALLOW_MODEL_REQUESTS=False globally to block any requests from being made to non-test models accidentally

### Unit testing with `TestModel`

The simplest and fastest way to exercise most of your application code is using TestModel, this will (by default) call all tools in the agent, then return either plain text or a structured response depending on the return type of the agent.

`TestModel` is not magic

The "clever" (but not too clever) part of `TestModel` is that it will attempt to generate valid structured data for [function tools](../tools/) and [output types](../output/#structured-output) based on the schema of the registered tools.

There's no ML or AI in `TestModel`, it's just plain old procedural Python code that tries to generate data that satisfies the JSON schema of a tool.

The resulting data won't look pretty or relevant, but it should pass Pydantic's validation in most cases. If you want something more sophisticated, use FunctionModel and write your own data generation logic.

Let's write unit tests for the following application code:

[Learn about Gateway](../gateway) weather_app.py

```python
import asyncio
from datetime import date

from pydantic_ai import Agent, RunContext

from fake_database import DatabaseConn  # (1)!
from weather_service import WeatherService  # (2)!

weather_agent = Agent(
    'gateway/openai:gpt-5',
    deps_type=WeatherService,
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
def weather_forecast(
    ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
    if forecast_date < date.today():  # (3)!
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)


async def run_weather_forecast(  # (4)!
    user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
    """Run weather forecast for a list of user prompts and save."""
    async with WeatherService() as weather_service:

        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.output)

        # run all prompts in parallel
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )

```

1. `DatabaseConn` is a class that holds a database connection
1. `WeatherService` has methods to get weather forecasts and historic data about the weather
1. We need to call a different endpoint depending on whether the date is in the past or the future, you'll see why this nuance is important below
1. This function is the code we want to test, together with the agent it uses

weather_app.py

```python
import asyncio
from datetime import date

from pydantic_ai import Agent, RunContext

from fake_database import DatabaseConn  # (1)!
from weather_service import WeatherService  # (2)!

weather_agent = Agent(
    'openai:gpt-5',
    deps_type=WeatherService,
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
def weather_forecast(
    ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
    if forecast_date < date.today():  # (3)!
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)


async def run_weather_forecast(  # (4)!
    user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
    """Run weather forecast for a list of user prompts and save."""
    async with WeatherService() as weather_service:

        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.output)

        # run all prompts in parallel
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )

```

1. `DatabaseConn` is a class that holds a database connection
1. `WeatherService` has methods to get weather forecasts and historic data about the weather
1. We need to call a different endpoint depending on whether the date is in the past or the future, you'll see why this nuance is important below
1. This function is the code we want to test, together with the agent it uses

Here we have a function that takes a list of `(user_prompt, user_id)` tuples, gets a weather forecast for each prompt, and stores the result in the database.

**We want to test this code without having to mock certain objects or modify our code so we can pass test objects in.**

Here's how we would write tests using TestModel:

test_weather_app.py

```python
from datetime import timezone
import pytest

from dirty_equals import IsNow, IsStr

from pydantic_ai import models, capture_run_messages, RequestUsage
from pydantic_ai.models.test import TestModel
from pydantic_ai import (
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    ModelRequest,
)

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio  # (1)!
models.ALLOW_MODEL_REQUESTS = False  # (2)!


async def test_forecast():
    conn = DatabaseConn()
    user_id = 1
    with capture_run_messages() as messages:
        with weather_agent.override(model=TestModel()):  # (3)!
            prompt = 'What will the weather be like in London on 2024-11-28?'
            await run_weather_forecast([(prompt, user_id)], conn)  # (4)!

    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}'  # (5)!

    assert messages == [  # (6)!
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content='Providing a weather forecast at the locations the user provides.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                UserPromptPart(
                    content='What will the weather be like in London on 2024-11-28?',
                    timestamp=IsNow(tz=timezone.utc),  # (7)!
                ),
            ],
            run_id=IsStr(),
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='weather_forecast',
                    args={
                        'location': 'a',
                        'forecast_date': '2024-01-01',  # (8)!
                    },
                    tool_call_id=IsStr(),
                )
            ],
            usage=RequestUsage(
                input_tokens=71,
                output_tokens=7,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='weather_forecast',
                    content='Sunny with a chance of rain',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            run_id=IsStr(),
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content='{"weather_forecast":"Sunny with a chance of rain"}',
                )
            ],
            usage=RequestUsage(
                input_tokens=77,
                output_tokens=16,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
    ]

```

1. We're using [anyio](https://anyio.readthedocs.io/en/stable/) to run async tests.
1. This is a safety measure to make sure we don't accidentally make real requests to the LLM while testing, see ALLOW_MODEL_REQUESTS for more details.
1. We're using Agent.override to replace the agent's model with TestModel, the nice thing about `override` is that we can replace the model inside agent without needing access to the agent `run*` methods call site.
1. Now we call the function we want to test inside the `override` context manager.
1. But default, `TestModel` will return a JSON string summarising the tools calls made, and what was returned. If you wanted to customise the response to something more closely aligned with the domain, you could add custom_output_text='Sunny' when defining `TestModel`.
1. So far we don't actually know which tools were called and with which values, we can use capture_run_messages to inspect messages from the most recent run and assert the exchange between the agent and the model occurred as expected.
1. The IsNow helper allows us to use declarative asserts even with data which will contain timestamps that change over time.
1. `TestModel` isn't doing anything clever to extract values from the prompt, so these values are hardcoded.

### Unit testing with `FunctionModel`

The above tests are a great start, but careful readers will notice that the `WeatherService.get_forecast` is never called since `TestModel` calls `weather_forecast` with a date in the past.

To fully exercise `weather_forecast`, we need to use FunctionModel to customise how the tools is called.

Here's an example of using `FunctionModel` to test the `weather_forecast` tool with custom inputs

test_weather_app2.py

```python
import re

import pytest

from pydantic_ai import models
from pydantic_ai import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


def call_weather_forecast(  # (1)!
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        # first call, call the weather forecast tool
        user_prompt = messages[0].parts[-1]
        m = re.search(r'\d{4}-\d{2}-\d{2}', user_prompt.content)
        assert m is not None
        args = {'location': 'London', 'forecast_date': m.group()}  # (2)!
        return ModelResponse(parts=[ToolCallPart('weather_forecast', args)])
    else:
        # second call, return the forecast
        msg = messages[-1].parts[0]
        assert msg.part_kind == 'tool-return'
        return ModelResponse(parts=[TextPart(f'The forecast is: {msg.content}')])


async def test_forecast_future():
    conn = DatabaseConn()
    user_id = 1
    with weather_agent.override(model=FunctionModel(call_weather_forecast)):  # (3)!
        prompt = 'What will the weather be like in London on 2032-01-01?'
        await run_weather_forecast([(prompt, user_id)], conn)

    forecast = await conn.get_forecast(user_id)
    assert forecast == 'The forecast is: Rainy with a chance of sun'

```

1. We define a function `call_weather_forecast` that will be called by `FunctionModel` in place of the LLM, this function has access to the list of ModelMessages that make up the run, and AgentInfo which contains information about the agent and the function tools and return tools.
1. Our function is slightly intelligent in that it tries to extract a date from the prompt, but just hard codes the location.
1. We use FunctionModel to replace the agent's model with our custom function.

### Overriding model via pytest fixtures

If you're writing lots of tests that all require model to be overridden, you can use [pytest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) to override the model with TestModel or FunctionModel in a reusable way.

Here's an example of a fixture that overrides the model with `TestModel`:

test_agent.py

```python
import pytest

from pydantic_ai.models.test import TestModel

from weather_app import weather_agent


@pytest.fixture
def override_weather_agent():
    with weather_agent.override(model=TestModel()):
        yield


async def test_forecast(override_weather_agent: None):
    ...
    # test code here

```

## Version Policy

We will not intentionally make breaking changes in minor releases of V1. V2 will be released in April 2026 at the earliest, 6 months after the release of V1 in September 2025.

Once we release V2, we'll continue to provide security fixes for V1 for another 6 months minimum, so you have time to upgrade your applications.

Functionality marked as deprecated will not be removed until V2.

Of course, some apparently safe changes and bug fixes will inevitably break some users' code â€” obligatory link to [xkcd](https://xkcd.com/1172/).

The following changes will **NOT** be considered breaking changes, and may occur in minor releases:

- Bug fixes that may result in existing code breaking, provided that such code was relying on undocumented features/constructs/assumptions.
- Adding new message parts, stream events, or optional fields on existing message (part) and event types. Always code defensively when consuming message parts or event streams, and use the ModelMessagesTypeAdapter to (de)serialize message histories.
- Changing OpenTelemetry span attributes. Because different [observability platforms](../logfire/#using-opentelemetry) support different versions of the [OpenTelemetry Semantic Conventions for Generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/), Pydantic AI lets you configure the [instrumentation version](../logfire/#configuring-data-format), but the default version may change in a minor release. Span attributes for [Pydantic Evals](../evals/) may also change as we iterate on Evals support in [Pydantic Logfire](https://logfire.pydantic.dev/docs/guides/web-ui/evals/).
- Changing how `__repr__` behaves, even of public classes.

In all cases we will aim to minimize churn and do so only when justified by the increase of quality of Pydantic AI for users.

## Beta Features

At Pydantic, we like to move quickly and innovate! To that end, minor releases may introduce beta features (indicated by a `beta` module) that are active works in progress. While in its beta phase, a feature's API and behaviors may not be stable, and it's very possible that changes made to the feature will not be backward-compatible. We aim to move beta features out of beta within a few months after initial release, once users have had a chance to provide feedback and test the feature in production.

## Support for Python versions

Pydantic will drop support for a Python version when the following conditions are met:

- The Python version has reached its [expected end of life](https://devguide.python.org/versions/).
- less than 5% of downloads of the most recent minor release are using that version.
# Examples

# Agent User Interaction (AG-UI)

Example of using Pydantic AI agents with the [AG-UI Dojo](https://github.com/ag-ui-protocol/ag-ui/tree/main/apps/dojo) example app.

See the [AG-UI docs](../../ui/ag-ui/) for more information about the AG-UI integration.

Demonstrates:

- [AG-UI](../../ui/ag-ui/)
- [Tools](../../tools/)

## Prerequisites

- An [OpenAI API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)

## Running the Example

With [dependencies installed and environment variables set](../setup/#usage) you will need two command line windows.

### Pydantic AI AG-UI backend

Setup your OpenAI API Key

```bash
export OPENAI_API_KEY=<your api key>

```

Start the Pydantic AI AG-UI example backend.

```bash
python -m pydantic_ai_examples.ag_ui

```

```bash
uv run -m pydantic_ai_examples.ag_ui

```

### AG-UI Dojo example frontend

Next run the AG-UI Dojo example frontend.

1. Clone the [AG-UI repository](https://github.com/ag-ui-protocol/ag-ui)

   ```shell
   git clone https://github.com/ag-ui-protocol/ag-ui.git

   ```

1. Change into to the `ag-ui/typescript-sdk` directory

   ```shell
   cd ag-ui/sdks/typescript

   ```

1. Run the Dojo app following the [official instructions](https://github.com/ag-ui-protocol/ag-ui/tree/main/apps/dojo#development-setup)

1. Visit <http://localhost:3000/pydantic-ai>

1. Select View `Pydantic AI` from the sidebar

## Feature Examples

### Agentic Chat

This demonstrates a basic agent interaction including Pydantic AI server side tools and AG-UI client side tools.

If you've [run the example](#running-the-example), you can view it at <http://localhost:3000/pydantic-ai/feature/agentic_chat>.

#### Agent Tools

- `time` - Pydantic AI tool to check the current time for a time zone
- `background` - AG-UI tool to set the background color of the client window

#### Agent Prompts

```text
What is the time in New York?

```

```text
Change the background to blue

```

A complex example which mixes both AG-UI and Pydantic AI tools:

```text
Perform the following steps, waiting for the response of each step before continuing:
1. Get the time
2. Set the background to red
3. Get the time
4. Report how long the background set took by diffing the two times

```

#### Agentic Chat - Code

[Learn about Gateway](../../gateway) [ag_ui/api/agentic_chat.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/agentic_chat.py)

```python
"""Agentic Chat feature."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent('gateway/openai:gpt-5-mini')


@agent.tool_plain
async def current_time(timezone: str = 'UTC') -> str:
    """Get the current time in ISO format.

    Args:
        timezone: The timezone to use.

    Returns:
        The current time in ISO format string.
    """
    tz: ZoneInfo = ZoneInfo(timezone)
    return datetime.now(tz=tz).isoformat()


app = AGUIApp(agent)

```

[ag_ui/api/agentic_chat.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/agentic_chat.py)

```python
"""Agentic Chat feature."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent('openai:gpt-5-mini')


@agent.tool_plain
async def current_time(timezone: str = 'UTC') -> str:
    """Get the current time in ISO format.

    Args:
        timezone: The timezone to use.

    Returns:
        The current time in ISO format string.
    """
    tz: ZoneInfo = ZoneInfo(timezone)
    return datetime.now(tz=tz).isoformat()


app = AGUIApp(agent)

```

### Agentic Generative UI

Demonstrates a long running task where the agent sends updates to the frontend to let the user know what's happening.

If you've [run the example](#running-the-example), you can view it at <http://localhost:3000/pydantic-ai/feature/agentic_generative_ui>.

#### Plan Prompts

```text
Create a plan for breakfast and execute it

```

#### Agentic Generative UI - Code

[Learn about Gateway](../../gateway) [ag_ui/api/agentic_generative_ui.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/agentic_generative_ui.py)

```python
"""Agentic Generative UI feature."""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Literal

from pydantic import BaseModel, Field

from ag_ui.core import EventType, StateDeltaEvent, StateSnapshotEvent
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

StepStatus = Literal['pending', 'completed']


class Step(BaseModel):
    """Represents a step in a plan."""

    description: str = Field(description='The description of the step')
    status: StepStatus = Field(
        default='pending',
        description='The status of the step (e.g., pending, completed)',
    )


class Plan(BaseModel):
    """Represents a plan with multiple steps."""

    steps: list[Step] = Field(default_factory=list, description='The steps in the plan')


class JSONPatchOp(BaseModel):
    """A class representing a JSON Patch operation (RFC 6902)."""

    op: Literal['add', 'remove', 'replace', 'move', 'copy', 'test'] = Field(
        description='The operation to perform: add, remove, replace, move, copy, or test',
    )
    path: str = Field(description='JSON Pointer (RFC 6901) to the target location')
    value: Any = Field(
        default=None,
        description='The value to apply (for add, replace operations)',
    )
    from_: str | None = Field(
        default=None,
        alias='from',
        description='Source path (for move, copy operations)',
    )


agent = Agent(
    'gateway/openai:gpt-5-mini',
    instructions=dedent(
        """
        When planning use tools only, without any other messages.
        IMPORTANT:
        - Use the `create_plan` tool to set the initial state of the steps
        - Use the `update_plan_step` tool to update the status of each step
        - Do NOT repeat the plan or summarise it in a message
        - Do NOT confirm the creation or updates in a message
        - Do NOT ask the user for additional information or next steps

        Only one plan can be active at a time, so do not call the `create_plan` tool
        again until all the steps in current plan are completed.
        """
    ),
)


@agent.tool_plain
async def create_plan(steps: list[str]) -> StateSnapshotEvent:
    """Create a plan with multiple steps.

    Args:
        steps: List of step descriptions to create the plan.

    Returns:
        StateSnapshotEvent containing the initial state of the steps.
    """
    plan: Plan = Plan(
        steps=[Step(description=step) for step in steps],
    )
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=plan.model_dump(),
    )


@agent.tool_plain
async def update_plan_step(
    index: int, description: str | None = None, status: StepStatus | None = None
) -> StateDeltaEvent:
    """Update the plan with new steps or changes.

    Args:
        index: The index of the step to update.
        description: The new description for the step.
        status: The new status for the step.

    Returns:
        StateDeltaEvent containing the changes made to the plan.
    """
    changes: list[JSONPatchOp] = []
    if description is not None:
        changes.append(
            JSONPatchOp(
                op='replace', path=f'/steps/{index}/description', value=description
            )
        )
    if status is not None:
        changes.append(
            JSONPatchOp(op='replace', path=f'/steps/{index}/status', value=status)
        )
    return StateDeltaEvent(
        type=EventType.STATE_DELTA,
        delta=changes,
    )


app = AGUIApp(agent)

```

[ag_ui/api/agentic_generative_ui.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/agentic_generative_ui.py)

```python
"""Agentic Generative UI feature."""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Literal

from pydantic import BaseModel, Field

from ag_ui.core import EventType, StateDeltaEvent, StateSnapshotEvent
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

StepStatus = Literal['pending', 'completed']


class Step(BaseModel):
    """Represents a step in a plan."""

    description: str = Field(description='The description of the step')
    status: StepStatus = Field(
        default='pending',
        description='The status of the step (e.g., pending, completed)',
    )


class Plan(BaseModel):
    """Represents a plan with multiple steps."""

    steps: list[Step] = Field(default_factory=list, description='The steps in the plan')


class JSONPatchOp(BaseModel):
    """A class representing a JSON Patch operation (RFC 6902)."""

    op: Literal['add', 'remove', 'replace', 'move', 'copy', 'test'] = Field(
        description='The operation to perform: add, remove, replace, move, copy, or test',
    )
    path: str = Field(description='JSON Pointer (RFC 6901) to the target location')
    value: Any = Field(
        default=None,
        description='The value to apply (for add, replace operations)',
    )
    from_: str | None = Field(
        default=None,
        alias='from',
        description='Source path (for move, copy operations)',
    )


agent = Agent(
    'openai:gpt-5-mini',
    instructions=dedent(
        """
        When planning use tools only, without any other messages.
        IMPORTANT:
        - Use the `create_plan` tool to set the initial state of the steps
        - Use the `update_plan_step` tool to update the status of each step
        - Do NOT repeat the plan or summarise it in a message
        - Do NOT confirm the creation or updates in a message
        - Do NOT ask the user for additional information or next steps

        Only one plan can be active at a time, so do not call the `create_plan` tool
        again until all the steps in current plan are completed.
        """
    ),
)


@agent.tool_plain
async def create_plan(steps: list[str]) -> StateSnapshotEvent:
    """Create a plan with multiple steps.

    Args:
        steps: List of step descriptions to create the plan.

    Returns:
        StateSnapshotEvent containing the initial state of the steps.
    """
    plan: Plan = Plan(
        steps=[Step(description=step) for step in steps],
    )
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=plan.model_dump(),
    )


@agent.tool_plain
async def update_plan_step(
    index: int, description: str | None = None, status: StepStatus | None = None
) -> StateDeltaEvent:
    """Update the plan with new steps or changes.

    Args:
        index: The index of the step to update.
        description: The new description for the step.
        status: The new status for the step.

    Returns:
        StateDeltaEvent containing the changes made to the plan.
    """
    changes: list[JSONPatchOp] = []
    if description is not None:
        changes.append(
            JSONPatchOp(
                op='replace', path=f'/steps/{index}/description', value=description
            )
        )
    if status is not None:
        changes.append(
            JSONPatchOp(op='replace', path=f'/steps/{index}/status', value=status)
        )
    return StateDeltaEvent(
        type=EventType.STATE_DELTA,
        delta=changes,
    )


app = AGUIApp(agent)

```

### Human in the Loop

Demonstrates simple human in the loop workflow where the agent comes up with a plan and the user can approve it using checkboxes.

#### Task Planning Tools

- `generate_task_steps` - AG-UI tool to generate and confirm steps

#### Task Planning Prompt

```text
Generate a list of steps for cleaning a car for me to review

```

#### Human in the Loop - Code

[Learn about Gateway](../../gateway) [ag_ui/api/human_in_the_loop.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/human_in_the_loop.py)

```python
"""Human in the Loop Feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from textwrap import dedent

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent(
    'gateway/openai:gpt-5-mini',
    instructions=dedent(
        """
        When planning tasks use tools only, without any other messages.
        IMPORTANT:
        - Use the `generate_task_steps` tool to display the suggested steps to the user
        - Never repeat the plan, or send a message detailing steps
        - If accepted, confirm the creation of the plan and the number of selected (enabled) steps only
        - If not accepted, ask the user for more information, DO NOT use the `generate_task_steps` tool again
        """
    ),
)

app = AGUIApp(agent)

```

[ag_ui/api/human_in_the_loop.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/human_in_the_loop.py)

```python
"""Human in the Loop Feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from textwrap import dedent

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent(
    'openai:gpt-5-mini',
    instructions=dedent(
        """
        When planning tasks use tools only, without any other messages.
        IMPORTANT:
        - Use the `generate_task_steps` tool to display the suggested steps to the user
        - Never repeat the plan, or send a message detailing steps
        - If accepted, confirm the creation of the plan and the number of selected (enabled) steps only
        - If not accepted, ask the user for more information, DO NOT use the `generate_task_steps` tool again
        """
    ),
)

app = AGUIApp(agent)

```

### Predictive State Updates

Demonstrates how to use the predictive state updates feature to update the state of the UI based on agent responses, including user interaction via user confirmation.

If you've [run the example](#running-the-example), you can view it at <http://localhost:3000/pydantic-ai/feature/predictive_state_updates>.

#### Story Tools

- `write_document` - AG-UI tool to write the document to a window
- `document_predict_state` - Pydantic AI tool that enables document state prediction for the `write_document` tool

This also shows how to use custom instructions based on shared state information.

#### Story Example

Starting document text

```markdown
Bruce was a good dog,

```

Agent prompt

```text
Help me complete my story about bruce the dog, is should be no longer than a sentence.

```

#### Predictive State Updates - Code

[Learn about Gateway](../../gateway) [ag_ui/api/predictive_state_updates.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/predictive_state_updates.py)

```python
"""Predictive State feature."""

from __future__ import annotations

from textwrap import dedent

from pydantic import BaseModel

from ag_ui.core import CustomEvent, EventType
from pydantic_ai import Agent, RunContext
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui.app import AGUIApp


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent('gateway/openai:gpt-5-mini', deps_type=StateDeps[DocumentState])


# Tools which return AG-UI events will be sent to the client as part of the
# event stream, single events and iterables of events are supported.
@agent.tool_plain
async def document_predict_state() -> list[CustomEvent]:
    """Enable document state prediction.

    Returns:
        CustomEvent containing the event to enable state prediction.
    """
    return [
        CustomEvent(
            type=EventType.CUSTOM,
            name='PredictState',
            value=[
                {
                    'state_key': 'document',
                    'tool': 'write_document',
                    'tool_argument': 'document',
                },
            ],
        ),
    ]


@agent.instructions()
async def story_instructions(ctx: RunContext[StateDeps[DocumentState]]) -> str:
    """Provide instructions for writing document if present.

    Args:
        ctx: The run context containing document state information.

    Returns:
        Instructions string for the document writing agent.
    """
    return dedent(
        f"""You are a helpful assistant for writing documents.

        Before you start writing, you MUST call the `document_predict_state`
        tool to enable state prediction.

        To present the document to the user for review, you MUST use the
        `write_document` tool.

        When you have written the document, DO NOT repeat it as a message.
        If accepted briefly summarize the changes you made, 2 sentences
        max, otherwise ask the user to clarify what they want to change.

        This is the current document:

        {ctx.deps.state.document}
        """
    )


app = AGUIApp(agent, deps=StateDeps(DocumentState()))

```

[ag_ui/api/predictive_state_updates.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/predictive_state_updates.py)

```python
"""Predictive State feature."""

from __future__ import annotations

from textwrap import dedent

from pydantic import BaseModel

from ag_ui.core import CustomEvent, EventType
from pydantic_ai import Agent, RunContext
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui.app import AGUIApp


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent('openai:gpt-5-mini', deps_type=StateDeps[DocumentState])


# Tools which return AG-UI events will be sent to the client as part of the
# event stream, single events and iterables of events are supported.
@agent.tool_plain
async def document_predict_state() -> list[CustomEvent]:
    """Enable document state prediction.

    Returns:
        CustomEvent containing the event to enable state prediction.
    """
    return [
        CustomEvent(
            type=EventType.CUSTOM,
            name='PredictState',
            value=[
                {
                    'state_key': 'document',
                    'tool': 'write_document',
                    'tool_argument': 'document',
                },
            ],
        ),
    ]


@agent.instructions()
async def story_instructions(ctx: RunContext[StateDeps[DocumentState]]) -> str:
    """Provide instructions for writing document if present.

    Args:
        ctx: The run context containing document state information.

    Returns:
        Instructions string for the document writing agent.
    """
    return dedent(
        f"""You are a helpful assistant for writing documents.

        Before you start writing, you MUST call the `document_predict_state`
        tool to enable state prediction.

        To present the document to the user for review, you MUST use the
        `write_document` tool.

        When you have written the document, DO NOT repeat it as a message.
        If accepted briefly summarize the changes you made, 2 sentences
        max, otherwise ask the user to clarify what they want to change.

        This is the current document:

        {ctx.deps.state.document}
        """
    )


app = AGUIApp(agent, deps=StateDeps(DocumentState()))

```

### Shared State

Demonstrates how to use the shared state between the UI and the agent.

State sent to the agent is detected by a function based instruction. This then validates the data using a custom pydantic model before using to create the instructions for the agent to follow and send to the client using a AG-UI tool.

If you've [run the example](#running-the-example), you can view it at <http://localhost:3000/pydantic-ai/feature/shared_state>.

#### Recipe Tools

- `display_recipe` - AG-UI tool to display the recipe in a graphical format

#### Recipe Example

1. Customise the basic settings of your recipe
1. Click `Improve with AI`

#### Shared State - Code

[Learn about Gateway](../../gateway) [ag_ui/api/shared_state.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/shared_state.py)

```python
"""Shared State feature."""

from __future__ import annotations

from enum import Enum
from textwrap import dedent

from pydantic import BaseModel, Field

from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai import Agent, RunContext
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui.app import AGUIApp


class SkillLevel(str, Enum):
    """The level of skill required for the recipe."""

    BEGINNER = 'Beginner'
    INTERMEDIATE = 'Intermediate'
    ADVANCED = 'Advanced'


class SpecialPreferences(str, Enum):
    """Special preferences for the recipe."""

    HIGH_PROTEIN = 'High Protein'
    LOW_CARB = 'Low Carb'
    SPICY = 'Spicy'
    BUDGET_FRIENDLY = 'Budget-Friendly'
    ONE_POT_MEAL = 'One-Pot Meal'
    VEGETARIAN = 'Vegetarian'
    VEGAN = 'Vegan'


class CookingTime(str, Enum):
    """The cooking time of the recipe."""

    FIVE_MIN = '5 min'
    FIFTEEN_MIN = '15 min'
    THIRTY_MIN = '30 min'
    FORTY_FIVE_MIN = '45 min'
    SIXTY_PLUS_MIN = '60+ min'


class Ingredient(BaseModel):
    """A class representing an ingredient in a recipe."""

    icon: str = Field(
        default='ingredient',
        description="The icon emoji (not emoji code like '\x1f35e', but the actual emoji like ðŸ¥•) of the ingredient",
    )
    name: str
    amount: str


class Recipe(BaseModel):
    """A class representing a recipe."""

    skill_level: SkillLevel = Field(
        default=SkillLevel.BEGINNER,
        description='The skill level required for the recipe',
    )
    special_preferences: list[SpecialPreferences] = Field(
        default_factory=list,
        description='Any special preferences for the recipe',
    )
    cooking_time: CookingTime = Field(
        default=CookingTime.FIVE_MIN, description='The cooking time of the recipe'
    )
    ingredients: list[Ingredient] = Field(
        default_factory=list,
        description='Ingredients for the recipe',
    )
    instructions: list[str] = Field(
        default_factory=list, description='Instructions for the recipe'
    )


class RecipeSnapshot(BaseModel):
    """A class representing the state of the recipe."""

    recipe: Recipe = Field(
        default_factory=Recipe, description='The current state of the recipe'
    )


agent = Agent('gateway/openai:gpt-5-mini', deps_type=StateDeps[RecipeSnapshot])


@agent.tool_plain
async def display_recipe(recipe: Recipe) -> StateSnapshotEvent:
    """Display the recipe to the user.

    Args:
        recipe: The recipe to display.

    Returns:
        StateSnapshotEvent containing the recipe snapshot.
    """
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot={'recipe': recipe},
    )


@agent.instructions
async def recipe_instructions(ctx: RunContext[StateDeps[RecipeSnapshot]]) -> str:
    """Instructions for the recipe generation agent.

    Args:
        ctx: The run context containing recipe state information.

    Returns:
        Instructions string for the recipe generation agent.
    """
    return dedent(
        f"""
        You are a helpful assistant for creating recipes.

        IMPORTANT:
        - Create a complete recipe using the existing ingredients
        - Append new ingredients to the existing ones
        - Use the `display_recipe` tool to present the recipe to the user
        - Do NOT repeat the recipe in the message, use the tool instead
        - Do NOT run the `display_recipe` tool multiple times in a row

        Once you have created the updated recipe and displayed it to the user,
        summarise the changes in one sentence, don't describe the recipe in
        detail or send it as a message to the user.

        The current state of the recipe is:

        {ctx.deps.state.recipe.model_dump_json(indent=2)}
        """,
    )


app = AGUIApp(agent, deps=StateDeps(RecipeSnapshot()))

```

[ag_ui/api/shared_state.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/shared_state.py)

```python
"""Shared State feature."""

from __future__ import annotations

from enum import Enum
from textwrap import dedent

from pydantic import BaseModel, Field

from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai import Agent, RunContext
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui.app import AGUIApp


class SkillLevel(str, Enum):
    """The level of skill required for the recipe."""

    BEGINNER = 'Beginner'
    INTERMEDIATE = 'Intermediate'
    ADVANCED = 'Advanced'


class SpecialPreferences(str, Enum):
    """Special preferences for the recipe."""

    HIGH_PROTEIN = 'High Protein'
    LOW_CARB = 'Low Carb'
    SPICY = 'Spicy'
    BUDGET_FRIENDLY = 'Budget-Friendly'
    ONE_POT_MEAL = 'One-Pot Meal'
    VEGETARIAN = 'Vegetarian'
    VEGAN = 'Vegan'


class CookingTime(str, Enum):
    """The cooking time of the recipe."""

    FIVE_MIN = '5 min'
    FIFTEEN_MIN = '15 min'
    THIRTY_MIN = '30 min'
    FORTY_FIVE_MIN = '45 min'
    SIXTY_PLUS_MIN = '60+ min'


class Ingredient(BaseModel):
    """A class representing an ingredient in a recipe."""

    icon: str = Field(
        default='ingredient',
        description="The icon emoji (not emoji code like '\x1f35e', but the actual emoji like ðŸ¥•) of the ingredient",
    )
    name: str
    amount: str


class Recipe(BaseModel):
    """A class representing a recipe."""

    skill_level: SkillLevel = Field(
        default=SkillLevel.BEGINNER,
        description='The skill level required for the recipe',
    )
    special_preferences: list[SpecialPreferences] = Field(
        default_factory=list,
        description='Any special preferences for the recipe',
    )
    cooking_time: CookingTime = Field(
        default=CookingTime.FIVE_MIN, description='The cooking time of the recipe'
    )
    ingredients: list[Ingredient] = Field(
        default_factory=list,
        description='Ingredients for the recipe',
    )
    instructions: list[str] = Field(
        default_factory=list, description='Instructions for the recipe'
    )


class RecipeSnapshot(BaseModel):
    """A class representing the state of the recipe."""

    recipe: Recipe = Field(
        default_factory=Recipe, description='The current state of the recipe'
    )


agent = Agent('openai:gpt-5-mini', deps_type=StateDeps[RecipeSnapshot])


@agent.tool_plain
async def display_recipe(recipe: Recipe) -> StateSnapshotEvent:
    """Display the recipe to the user.

    Args:
        recipe: The recipe to display.

    Returns:
        StateSnapshotEvent containing the recipe snapshot.
    """
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot={'recipe': recipe},
    )


@agent.instructions
async def recipe_instructions(ctx: RunContext[StateDeps[RecipeSnapshot]]) -> str:
    """Instructions for the recipe generation agent.

    Args:
        ctx: The run context containing recipe state information.

    Returns:
        Instructions string for the recipe generation agent.
    """
    return dedent(
        f"""
        You are a helpful assistant for creating recipes.

        IMPORTANT:
        - Create a complete recipe using the existing ingredients
        - Append new ingredients to the existing ones
        - Use the `display_recipe` tool to present the recipe to the user
        - Do NOT repeat the recipe in the message, use the tool instead
        - Do NOT run the `display_recipe` tool multiple times in a row

        Once you have created the updated recipe and displayed it to the user,
        summarise the changes in one sentence, don't describe the recipe in
        detail or send it as a message to the user.

        The current state of the recipe is:

        {ctx.deps.state.recipe.model_dump_json(indent=2)}
        """,
    )


app = AGUIApp(agent, deps=StateDeps(RecipeSnapshot()))

```

### Tool Based Generative UI

Demonstrates customised rendering for tool output with used confirmation.

If you've [run the example](#running-the-example), you can view it at <http://localhost:3000/pydantic-ai/feature/tool_based_generative_ui>.

#### Haiku Tools

- `generate_haiku` - AG-UI tool to display a haiku in English and Japanese

#### Haiku Prompt

```text
Generate a haiku about formula 1

```

#### Tool Based Generative UI - Code

[Learn about Gateway](../../gateway) [ag_ui/api/tool_based_generative_ui.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/tool_based_generative_ui.py)

```python
"""Tool Based Generative UI feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent('gateway/openai:gpt-5-mini')
app = AGUIApp(agent)

```

[ag_ui/api/tool_based_generative_ui.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/ag_ui/api/tool_based_generative_ui.py)

```python
"""Tool Based Generative UI feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

agent = Agent('openai:gpt-5-mini')
app = AGUIApp(agent)

```

Small but complete example of using Pydantic AI to build a support agent for a bank.

Demonstrates:

- [dynamic system prompt](../../agents/#system-prompts)
- [structured `output_type`](../../output/#structured-output)
- [tools](../../tools/)

## Running the Example

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.bank_support

```

```bash
uv run -m pydantic_ai_examples.bank_support

```

(or `PYDANTIC_AI_MODEL=gemini-3-pro-preview ...`)

## Example Code

[Learn about Gateway](../../gateway) [bank_support.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/bank_support.py)

```python
"""Small but complete example of using Pydantic AI to build a support agent for a bank.

Run with:

    uv run -m pydantic_ai_examples.bank_support
"""

from dataclasses import dataclass

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext


class DatabaseConn:
    """This is a fake database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g. PostgreSQL) to get information about customers.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return 'John'

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123:
            if include_pending:
                return 123.45
            else:
                return 100.00
        else:
            raise ValueError('Customer not found')


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    support_advice: str
    """Advice returned to the customer"""
    block_card: bool
    """Whether to block their card or not"""
    risk: int
    """Risk level of query"""


support_agent = Agent(
    'gateway/openai:gpt-5',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    instructions=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query. '
        "Reply using the customer's name."
    ),
)


@support_agent.instructions
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f'${balance:.2f}'


if __name__ == '__main__':
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = support_agent.run_sync('What is my balance?', deps=deps)
    print(result.output)
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = support_agent.run_sync('I just lost my card!', deps=deps)
    print(result.output)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """

```

[bank_support.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/bank_support.py)

```python
"""Small but complete example of using Pydantic AI to build a support agent for a bank.

Run with:

    uv run -m pydantic_ai_examples.bank_support
"""

from dataclasses import dataclass

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext


class DatabaseConn:
    """This is a fake database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g. PostgreSQL) to get information about customers.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return 'John'

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123:
            if include_pending:
                return 123.45
            else:
                return 100.00
        else:
            raise ValueError('Customer not found')


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    support_advice: str
    """Advice returned to the customer"""
    block_card: bool
    """Whether to block their card or not"""
    risk: int
    """Risk level of query"""


support_agent = Agent(
    'openai:gpt-5',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    instructions=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query. '
        "Reply using the customer's name."
    ),
)


@support_agent.instructions
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f'${balance:.2f}'


if __name__ == '__main__':
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = support_agent.run_sync('What is my balance?', deps=deps)
    print(result.output)
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = support_agent.run_sync('I just lost my card!', deps=deps)
    print(result.output)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """

```

# Chat App with FastAPI

Simple chat app example build with FastAPI.

Demonstrates:

- [reusing chat history](../../message-history/)
- [serializing messages](../../message-history/#accessing-messages-from-results)
- [streaming responses](../../output/#streamed-results)

This demonstrates storing chat history between requests and using it to give the model context for new responses.

Most of the complex logic here is between `chat_app.py` which streams the response to the browser, and `chat_app.ts` which renders messages in the browser.

## Running the Example

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.chat_app

```

```bash
uv run -m pydantic_ai_examples.chat_app

```

Then open the app at [localhost:8000](http://localhost:8000).

## Example Code

Python code that runs the chat app:

[Learn about Gateway](../../gateway) [chat_app.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/chat_app.py)

```python
"""Simple chat app example build with FastAPI.

Run with:

    uv run -m pydantic_ai_examples.chat_app
"""

from __future__ import annotations as _annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import fastapi
import logfire
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UnexpectedModelBehavior,
    UserPromptPart,
)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

agent = Agent('gateway/openai:gpt-5')
THIS_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {'db': db}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


@app.get('/')
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')


@app.get('/chat_app.ts')
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')


async def get_db(request: Request) -> Database:
    return request.state.db


@app.get('/chat/')
async def get_chat(database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages()
    return Response(
        b'\n'.join(json.dumps(to_chat_message(m)).encode('utf-8') for m in msgs),
        media_type='text/plain',
    )


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


@app.post('/chat/')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # stream the user prompt so that can be displayed straight away
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                }
            ).encode('utf-8')
            + b'\n'
        )
        # get the chat history so far to pass as context to the agent
        messages = await database.get_messages()
        # run the agent with the user prompt and the chat history
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream_output(debounce_by=0.01):
                # text here is a `str` and the frontend wants
                # JSON encoded ModelResponse, so we create one
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')


P = ParamSpec('P')
R = TypeVar('R')


@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
    ) -> AsyncIterator[Database]:
        with logfire.span('connect to DB'):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        cur.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute, 'SELECT message_list FROM messages order by id'
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )

```

[chat_app.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/chat_app.py)

```python
"""Simple chat app example build with FastAPI.

Run with:

    uv run -m pydantic_ai_examples.chat_app
"""

from __future__ import annotations as _annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import fastapi
import logfire
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from typing_extensions import LiteralString, ParamSpec, TypedDict

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UnexpectedModelBehavior,
    UserPromptPart,
)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

agent = Agent('openai:gpt-5')
THIS_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {'db': db}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


@app.get('/')
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')


@app.get('/chat_app.ts')
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')


async def get_db(request: Request) -> Database:
    return request.state.db


@app.get('/chat/')
async def get_chat(database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages()
    return Response(
        b'\n'.join(json.dumps(to_chat_message(m)).encode('utf-8') for m in msgs),
        media_type='text/plain',
    )


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


@app.post('/chat/')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # stream the user prompt so that can be displayed straight away
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                }
            ).encode('utf-8')
            + b'\n'
        )
        # get the chat history so far to pass as context to the agent
        messages = await database.get_messages()
        # run the agent with the user prompt and the chat history
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream_output(debounce_by=0.01):
                # text here is a `str` and the frontend wants
                # JSON encoded ModelResponse, so we create one
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')


P = ParamSpec('P')
R = TypeVar('R')


@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
    ) -> AsyncIterator[Database]:
        with logfire.span('connect to DB'):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        cur.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute, 'SELECT message_list FROM messages order by id'
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )

```

Simple HTML page to render the app:

[chat_app.html](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/chat_app.html)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chat App</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    main {
      max-width: 700px;
    }
    #conversation .user::before {
      content: 'You asked: ';
      font-weight: bold;
      display: block;
    }
    #conversation .model::before {
      content: 'AI Response: ';
      font-weight: bold;
      display: block;
    }
    #spinner {
      opacity: 0;
      transition: opacity 500ms ease-in;
      width: 30px;
      height: 30px;
      border: 3px solid #222;
      border-bottom-color: transparent;
      border-radius: 50%;
      animation: rotation 1s linear infinite;
    }
    @keyframes rotation {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    #spinner.active {
      opacity: 1;
    }
  </style>
</head>
<body>
  <main class="border rounded mx-auto my-5 p-4">
    <h1>Chat App</h1>
    <p>Ask me anything...</p>
    <div id="conversation" class="px-2"></div>
    <div class="d-flex justify-content-center mb-3">
      <div id="spinner"></div>
    </div>
    <form method="post">
      <input id="prompt-input" name="prompt" class="form-control"/>
      <div class="d-flex justify-content-end">
        <button class="btn btn-primary mt-2">Send</button>
      </div>
    </form>
    <div id="error" class="d-none text-danger">
      Error occurred, check the browser developer console for more information.
    </div>
  </main>
</body>
</html>
<script src="https://cdnjs.cloudflare.com/ajax/libs/typescript/5.6.3/typescript.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script type="module">
  // to let me write TypeScript, without adding the burden of npm we do a dirty, non-production-ready hack
  // and transpile the TypeScript code in the browser
  // this is (arguably) A neat demo trick, but not suitable for production!
  async function loadTs() {
    const response = await fetch('/chat_app.ts');
    const tsCode = await response.text();
    const jsCode = window.ts.transpile(tsCode, { target: "es2015" });
    let script = document.createElement('script');
    script.type = 'module';
    script.text = jsCode;
    document.body.appendChild(script);
  }

  loadTs().catch((e) => {
    console.error(e);
    document.getElementById('error').classList.remove('d-none');
    document.getElementById('spinner').classList.remove('active');
  });
</script>

```

TypeScript to handle rendering the messages, to keep this simple (and at the risk of offending frontend developers) the typescript code is passed to the browser as plain text and transpiled in the browser.

[chat_app.ts](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/chat_app.ts)

```ts
// BIG FAT WARNING: to avoid the complexity of npm, this typescript is compiled in the browser
// there's currently no static type checking

import { marked } from 'https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.0/lib/marked.esm.js'
const convElement = document.getElementById('conversation')

const promptInput = document.getElementById('prompt-input') as HTMLInputElement
const spinner = document.getElementById('spinner')

// stream the response and render messages as each chunk is received
// data is sent as newline-delimited JSON
async function onFetchResponse(response: Response): Promise<void> {
  let text = ''
  let decoder = new TextDecoder()
  if (response.ok) {
    const reader = response.body.getReader()
    while (true) {
      const {done, value} = await reader.read()
      if (done) {
        break
      }
      text += decoder.decode(value)
      addMessages(text)
      spinner.classList.remove('active')
    }
    addMessages(text)
    promptInput.disabled = false
    promptInput.focus()
  } else {
    const text = await response.text()
    console.error(`Unexpected response: ${response.status}`, {response, text})
    throw new Error(`Unexpected response: ${response.status}`)
  }
}

// The format of messages, this matches pydantic-ai both for brevity and understanding
// in production, you might not want to keep this format all the way to the frontend
interface Message {
  role: string
  content: string
  timestamp: string
}

// take raw response text and render messages into the `#conversation` element
// Message timestamp is assumed to be a unique identifier of a message, and is used to deduplicate
// hence you can send data about the same message multiple times, and it will be updated
// instead of creating a new message elements
function addMessages(responseText: string) {
  const lines = responseText.split('\n')
  const messages: Message[] = lines.filter(line => line.length > 1).map(j => JSON.parse(j))
  for (const message of messages) {
    // we use the timestamp as a crude element id
    const {timestamp, role, content} = message
    const id = `msg-${timestamp}`
    let msgDiv = document.getElementById(id)
    if (!msgDiv) {
      msgDiv = document.createElement('div')
      msgDiv.id = id
      msgDiv.title = `${role} at ${timestamp}`
      msgDiv.classList.add('border-top', 'pt-2', role)
      convElement.appendChild(msgDiv)
    }
    msgDiv.innerHTML = marked.parse(content)
  }
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
}

function onError(error: any) {
  console.error(error)
  document.getElementById('error').classList.remove('d-none')
  document.getElementById('spinner').classList.remove('active')
}

async function onSubmit(e: SubmitEvent): Promise<void> {
  e.preventDefault()
  spinner.classList.add('active')
  const body = new FormData(e.target as HTMLFormElement)

  promptInput.value = ''
  promptInput.disabled = true

  const response = await fetch('/chat/', {method: 'POST', body})
  await onFetchResponse(response)
}

// call onSubmit when the form is submitted (e.g. user clicks the send button or hits Enter)
document.querySelector('form').addEventListener('submit', (e) => onSubmit(e).catch(onError))

// load messages on page load
fetch('/chat/').then(onFetchResponse).catch(onError)

```

# Data Analyst

Sometimes in an agent workflow, the agent does not need to know the exact tool output, but still needs to process the tool output in some ways. This is especially common in data analytics: the agent needs to know that the result of a query tool is a `DataFrame` with certain named columns, but not necessarily the content of every single row.

With Pydantic AI, you can use a [dependencies object](../../dependencies/) to store the result from one tool and use it in another tool.

In this example, we'll build an agent that analyzes the [Rotten Tomatoes movie review dataset from Cornell](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes).

Demonstrates:

- [agent dependencies](../../dependencies/)

## Running the Example

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.data_analyst

```

```bash
uv run -m pydantic_ai_examples.data_analyst

```

Output (debug):

> Based on my analysis of the Cornell Movie Review dataset (rotten_tomatoes), there are **4,265 negative comments** in the training split. These are the reviews labeled as 'neg' (represented by 0 in the dataset).

## Example Code

[Learn about Gateway](../../gateway) [data_analyst.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/data_analyst.py)

```python
from dataclasses import dataclass, field

import datasets
import duckdb
import pandas as pd

from pydantic_ai import Agent, ModelRetry, RunContext


@dataclass
class AnalystAgentDeps:
    output: dict[str, pd.DataFrame] = field(default_factory=dict)

    def store(self, value: pd.DataFrame) -> str:
        """Store the output in deps and return the reference such as Out[1] to be used by the LLM."""
        ref = f'Out[{len(self.output) + 1}]'
        self.output[ref] = value
        return ref

    def get(self, ref: str) -> pd.DataFrame:
        if ref not in self.output:
            raise ModelRetry(
                f'Error: {ref} is not a valid variable reference. Check the previous messages and try again.'
            )
        return self.output[ref]


analyst_agent = Agent(
    'gateway/openai:gpt-5',
    deps_type=AnalystAgentDeps,
    instructions='You are a data analyst and your job is to analyze the data according to the user request.',
)


@analyst_agent.tool
def load_dataset(
    ctx: RunContext[AnalystAgentDeps],
    path: str,
    split: str = 'train',
) -> str:
    """Load the `split` of dataset `dataset_name` from huggingface.

    Args:
        ctx: Pydantic AI agent RunContext
        path: name of the dataset in the form of `<user_name>/<dataset_name>`
        split: load the split of the dataset (default: "train")
    """
    # begin load data from hf
    builder = datasets.load_dataset_builder(path)  # pyright: ignore[reportUnknownMemberType]
    splits: dict[str, datasets.SplitInfo] = builder.info.splits or {}  # pyright: ignore[reportUnknownMemberType]
    if split not in splits:
        raise ModelRetry(
            f'{split} is not valid for dataset {path}. Valid splits are {",".join(splits.keys())}'
        )

    builder.download_and_prepare()  # pyright: ignore[reportUnknownMemberType]
    dataset = builder.as_dataset(split=split)
    assert isinstance(dataset, datasets.Dataset)
    dataframe = dataset.to_pandas()
    assert isinstance(dataframe, pd.DataFrame)
    # end load data from hf

    # store the dataframe in the deps and get a ref like "Out[1]"
    ref = ctx.deps.store(dataframe)
    # construct a summary of the loaded dataset
    output = [
        f'Loaded the dataset as `{ref}`.',
        f'Description: {dataset.info.description}'
        if dataset.info.description
        else None,
        f'Features: {dataset.info.features!r}' if dataset.info.features else None,
    ]
    return '\n'.join(filter(None, output))


@analyst_agent.tool
def run_duckdb(ctx: RunContext[AnalystAgentDeps], dataset: str, sql: str) -> str:
    """Run DuckDB SQL query on the DataFrame.

    Note that the virtual table name used in DuckDB SQL must be `dataset`.

    Args:
        ctx: Pydantic AI agent RunContext
        dataset: reference string to the DataFrame
        sql: the query to be executed using DuckDB
    """
    data = ctx.deps.get(dataset)
    result = duckdb.query_df(df=data, virtual_table_name='dataset', sql_query=sql)
    # pass the result as ref (because DuckDB SQL can select many rows, creating another huge dataframe)
    ref = ctx.deps.store(result.df())  # pyright: ignore[reportUnknownMemberType]
    return f'Executed SQL, result is `{ref}`'


@analyst_agent.tool
def display(ctx: RunContext[AnalystAgentDeps], name: str) -> str:
    """Display at most 5 rows of the dataframe."""
    dataset = ctx.deps.get(name)
    return dataset.head().to_string()  # pyright: ignore[reportUnknownMemberType]


if __name__ == '__main__':
    deps = AnalystAgentDeps()
    result = analyst_agent.run_sync(
        user_prompt='Count how many negative comments are there in the dataset `cornell-movie-review-data/rotten_tomatoes`',
        deps=deps,
    )
    print(result.output)

```

[data_analyst.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/data_analyst.py)

```python
from dataclasses import dataclass, field

import datasets
import duckdb
import pandas as pd

from pydantic_ai import Agent, ModelRetry, RunContext


@dataclass
class AnalystAgentDeps:
    output: dict[str, pd.DataFrame] = field(default_factory=dict)

    def store(self, value: pd.DataFrame) -> str:
        """Store the output in deps and return the reference such as Out[1] to be used by the LLM."""
        ref = f'Out[{len(self.output) + 1}]'
        self.output[ref] = value
        return ref

    def get(self, ref: str) -> pd.DataFrame:
        if ref not in self.output:
            raise ModelRetry(
                f'Error: {ref} is not a valid variable reference. Check the previous messages and try again.'
            )
        return self.output[ref]


analyst_agent = Agent(
    'openai:gpt-5',
    deps_type=AnalystAgentDeps,
    instructions='You are a data analyst and your job is to analyze the data according to the user request.',
)


@analyst_agent.tool
def load_dataset(
    ctx: RunContext[AnalystAgentDeps],
    path: str,
    split: str = 'train',
) -> str:
    """Load the `split` of dataset `dataset_name` from huggingface.

    Args:
        ctx: Pydantic AI agent RunContext
        path: name of the dataset in the form of `<user_name>/<dataset_name>`
        split: load the split of the dataset (default: "train")
    """
    # begin load data from hf
    builder = datasets.load_dataset_builder(path)  # pyright: ignore[reportUnknownMemberType]
    splits: dict[str, datasets.SplitInfo] = builder.info.splits or {}  # pyright: ignore[reportUnknownMemberType]
    if split not in splits:
        raise ModelRetry(
            f'{split} is not valid for dataset {path}. Valid splits are {",".join(splits.keys())}'
        )

    builder.download_and_prepare()  # pyright: ignore[reportUnknownMemberType]
    dataset = builder.as_dataset(split=split)
    assert isinstance(dataset, datasets.Dataset)
    dataframe = dataset.to_pandas()
    assert isinstance(dataframe, pd.DataFrame)
    # end load data from hf

    # store the dataframe in the deps and get a ref like "Out[1]"
    ref = ctx.deps.store(dataframe)
    # construct a summary of the loaded dataset
    output = [
        f'Loaded the dataset as `{ref}`.',
        f'Description: {dataset.info.description}'
        if dataset.info.description
        else None,
        f'Features: {dataset.info.features!r}' if dataset.info.features else None,
    ]
    return '\n'.join(filter(None, output))


@analyst_agent.tool
def run_duckdb(ctx: RunContext[AnalystAgentDeps], dataset: str, sql: str) -> str:
    """Run DuckDB SQL query on the DataFrame.

    Note that the virtual table name used in DuckDB SQL must be `dataset`.

    Args:
        ctx: Pydantic AI agent RunContext
        dataset: reference string to the DataFrame
        sql: the query to be executed using DuckDB
    """
    data = ctx.deps.get(dataset)
    result = duckdb.query_df(df=data, virtual_table_name='dataset', sql_query=sql)
    # pass the result as ref (because DuckDB SQL can select many rows, creating another huge dataframe)
    ref = ctx.deps.store(result.df())  # pyright: ignore[reportUnknownMemberType]
    return f'Executed SQL, result is `{ref}`'


@analyst_agent.tool
def display(ctx: RunContext[AnalystAgentDeps], name: str) -> str:
    """Display at most 5 rows of the dataframe."""
    dataset = ctx.deps.get(name)
    return dataset.head().to_string()  # pyright: ignore[reportUnknownMemberType]


if __name__ == '__main__':
    deps = AnalystAgentDeps()
    result = analyst_agent.run_sync(
        user_prompt='Count how many negative comments are there in the dataset `cornell-movie-review-data/rotten_tomatoes`',
        deps=deps,
    )
    print(result.output)

```

## Appendix

### Choosing a Model

This example requires using a model that understands DuckDB SQL. You can check with `clai`:

```sh
> clai -m bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
clai - Pydantic AI CLI v0.0.1.dev920+41dd069 with bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
clai âž¤ do you understand duckdb sql?
# DuckDB SQL

Yes, I understand DuckDB SQL. DuckDB is an in-process analytical SQL database
that uses syntax similar to PostgreSQL. It specializes in analytical queries
and is designed for high-performance analysis of structured data.

Some key features of DuckDB SQL include:

 â€¢ OLAP (Online Analytical Processing) optimized
 â€¢ Columnar-vectorized query execution
 â€¢ Standard SQL support with PostgreSQL compatibility
 â€¢ Support for complex analytical queries
 â€¢ Efficient handling of CSV/Parquet/JSON files

I can help you with DuckDB SQL queries, schema design, optimization, or other
DuckDB-related questions.

```

Example of a multi-agent flow where one agent delegates work to another, then hands off control to a third agent.

Demonstrates:

- [agent delegation](../../multi-agent-applications/#agent-delegation)
- [programmatic agent hand-off](../../multi-agent-applications/#programmatic-agent-hand-off)
- [usage limits](../../agents/#usage-limits)

In this scenario, a group of agents work together to find the best flight for a user.

The control flow for this example can be summarised as follows:

```
graph TD
  START --> search_agent("search agent")
  search_agent --> extraction_agent("extraction agent")
  extraction_agent --> search_agent
  search_agent --> human_confirm("human confirm")
  human_confirm --> search_agent
  search_agent --> FAILED
  human_confirm --> find_seat_function("find seat function")
  find_seat_function --> human_seat_choice("human seat choice")
  human_seat_choice --> find_seat_agent("find seat agent")
  find_seat_agent --> find_seat_function
  find_seat_function --> buy_flights("buy flights")
  buy_flights --> SUCCESS
```

## Running the Example

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.flight_booking

```

```bash
uv run -m pydantic_ai_examples.flight_booking

```

## Example Code

[Learn about Gateway](../../gateway) [flight_booking.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/flight_booking.py)

```python
"""Example of a multi-agent flow where one agent delegates work to another.

In this scenario, a group of agents work together to find flights for a user.
"""

import datetime
from dataclasses import dataclass
from typing import Literal

import logfire
from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRetry,
    RunContext,
    RunUsage,
    UsageLimits,
)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


class FlightDetails(BaseModel):
    """Details of the most suitable flight."""

    flight_number: str
    price: int
    origin: str = Field(description='Three-letter airport code')
    destination: str = Field(description='Three-letter airport code')
    date: datetime.date


class NoFlightFound(BaseModel):
    """When no valid flight is found."""


@dataclass
class Deps:
    web_page_text: str
    req_origin: str
    req_destination: str
    req_date: datetime.date


# This agent is responsible for controlling the flow of the conversation.
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    'openai:gpt-5',
    output_type=FlightDetails | NoFlightFound,  # type: ignore
    retries=4,
    system_prompt=(
        'Your job is to find the cheapest flight for the user on the given date. '
    ),
)


# This agent is responsible for extracting flight details from web page text.
extraction_agent = Agent(
    'gateway/openai:gpt-5',
    output_type=list[FlightDetails],
    system_prompt='Extract all the flight details from the given text.',
)


@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """Get details of all flights."""
    # we pass the usage to the search agent so requests within this agent are counted
    result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
    logfire.info('found {flight_count} flights', flight_count=len(result.output))
    return result.output


@search_agent.output_validator
async def validate_output(
    ctx: RunContext[Deps], output: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
    """Procedural validation that the flight meets the constraints."""
    if isinstance(output, NoFlightFound):
        return output

    errors: list[str] = []
    if output.origin != ctx.deps.req_origin:
        errors.append(
            f'Flight should have origin {ctx.deps.req_origin}, not {output.origin}'
        )
    if output.destination != ctx.deps.req_destination:
        errors.append(
            f'Flight should have destination {ctx.deps.req_destination}, not {output.destination}'
        )
    if output.date != ctx.deps.req_date:
        errors.append(f'Flight should be on {ctx.deps.req_date}, not {output.date}')

    if errors:
        raise ModelRetry('\n'.join(errors))
    else:
        return output


class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']


class Failed(BaseModel):
    """Unable to extract a seat selection."""


# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[None, SeatPreference | Failed](
    'openai:gpt-5',
    output_type=SeatPreference | Failed,
    system_prompt=(
        "Extract the user's seat preference. "
        'Seats A and F are window seats. '
        'Row 1 is the front row and has extra leg room. '
        'Rows 14, and 20 also have extra leg room. '
    ),
)


# in reality this would be downloaded from a booking site,
# potentially using another agent to navigate the site
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025

3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025

4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025

6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025

7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025

8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""

# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)


async def main():
    deps = Deps(
        web_page_text=flights_web_page,
        req_origin='SFO',
        req_destination='ANC',
        req_date=datetime.date(2025, 1, 10),
    )
    message_history: list[ModelMessage] | None = None
    usage: RunUsage = RunUsage()
    # run the agent until a satisfactory flight is found
    while True:
        result = await search_agent.run(
            f'Find me a flight from {deps.req_origin} to {deps.req_destination} on {deps.req_date}',
            deps=deps,
            usage=usage,
            message_history=message_history,
            usage_limits=usage_limits,
        )
        if isinstance(result.output, NoFlightFound):
            print('No flight found')
            break
        else:
            flight = result.output
            print(f'Flight found: {flight}')
            answer = Prompt.ask(
                'Do you want to buy this flight, or keep searching? (buy/*search)',
                choices=['buy', 'search', ''],
                show_choices=False,
            )
            if answer == 'buy':
                seat = await find_seat(usage)
                await buy_tickets(flight, seat)
                break
            else:
                message_history = result.all_messages(
                    output_tool_return_content='Please suggest another flight'
                )


async def find_seat(usage: RunUsage) -> SeatPreference:
    message_history: list[ModelMessage] | None = None
    while True:
        answer = Prompt.ask('What seat would you like?')

        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.output, SeatPreference):
            return result.output
        else:
            print('Could not understand seat preference. Please try again.')
            message_history = result.all_messages()


async def buy_tickets(flight_details: FlightDetails, seat: SeatPreference):
    print(f'Purchasing flight {flight_details=!r} {seat=!r}...')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())

```

[flight_booking.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/flight_booking.py)

```python
"""Example of a multi-agent flow where one agent delegates work to another.

In this scenario, a group of agents work together to find flights for a user.
"""

import datetime
from dataclasses import dataclass
from typing import Literal

import logfire
from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRetry,
    RunContext,
    RunUsage,
    UsageLimits,
)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


class FlightDetails(BaseModel):
    """Details of the most suitable flight."""

    flight_number: str
    price: int
    origin: str = Field(description='Three-letter airport code')
    destination: str = Field(description='Three-letter airport code')
    date: datetime.date


class NoFlightFound(BaseModel):
    """When no valid flight is found."""


@dataclass
class Deps:
    web_page_text: str
    req_origin: str
    req_destination: str
    req_date: datetime.date


# This agent is responsible for controlling the flow of the conversation.
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    'openai:gpt-5',
    output_type=FlightDetails | NoFlightFound,  # type: ignore
    retries=4,
    system_prompt=(
        'Your job is to find the cheapest flight for the user on the given date. '
    ),
)


# This agent is responsible for extracting flight details from web page text.
extraction_agent = Agent(
    'openai:gpt-5',
    output_type=list[FlightDetails],
    system_prompt='Extract all the flight details from the given text.',
)


@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """Get details of all flights."""
    # we pass the usage to the search agent so requests within this agent are counted
    result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
    logfire.info('found {flight_count} flights', flight_count=len(result.output))
    return result.output


@search_agent.output_validator
async def validate_output(
    ctx: RunContext[Deps], output: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
    """Procedural validation that the flight meets the constraints."""
    if isinstance(output, NoFlightFound):
        return output

    errors: list[str] = []
    if output.origin != ctx.deps.req_origin:
        errors.append(
            f'Flight should have origin {ctx.deps.req_origin}, not {output.origin}'
        )
    if output.destination != ctx.deps.req_destination:
        errors.append(
            f'Flight should have destination {ctx.deps.req_destination}, not {output.destination}'
        )
    if output.date != ctx.deps.req_date:
        errors.append(f'Flight should be on {ctx.deps.req_date}, not {output.date}')

    if errors:
        raise ModelRetry('\n'.join(errors))
    else:
        return output


class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']


class Failed(BaseModel):
    """Unable to extract a seat selection."""


# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[None, SeatPreference | Failed](
    'openai:gpt-5',
    output_type=SeatPreference | Failed,
    system_prompt=(
        "Extract the user's seat preference. "
        'Seats A and F are window seats. '
        'Row 1 is the front row and has extra leg room. '
        'Rows 14, and 20 also have extra leg room. '
    ),
)


# in reality this would be downloaded from a booking site,
# potentially using another agent to navigate the site
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025

3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025

4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025

6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025

7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025

8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""

# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)


async def main():
    deps = Deps(
        web_page_text=flights_web_page,
        req_origin='SFO',
        req_destination='ANC',
        req_date=datetime.date(2025, 1, 10),
    )
    message_history: list[ModelMessage] | None = None
    usage: RunUsage = RunUsage()
    # run the agent until a satisfactory flight is found
    while True:
        result = await search_agent.run(
            f'Find me a flight from {deps.req_origin} to {deps.req_destination} on {deps.req_date}',
            deps=deps,
            usage=usage,
            message_history=message_history,
            usage_limits=usage_limits,
        )
        if isinstance(result.output, NoFlightFound):
            print('No flight found')
            break
        else:
            flight = result.output
            print(f'Flight found: {flight}')
            answer = Prompt.ask(
                'Do you want to buy this flight, or keep searching? (buy/*search)',
                choices=['buy', 'search', ''],
                show_choices=False,
            )
            if answer == 'buy':
                seat = await find_seat(usage)
                await buy_tickets(flight, seat)
                break
            else:
                message_history = result.all_messages(
                    output_tool_return_content='Please suggest another flight'
                )


async def find_seat(usage: RunUsage) -> SeatPreference:
    message_history: list[ModelMessage] | None = None
    while True:
        answer = Prompt.ask('What seat would you like?')

        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.output, SeatPreference):
            return result.output
        else:
            print('Could not understand seat preference. Please try again.')
            message_history = result.all_messages()


async def buy_tickets(flight_details: FlightDetails, seat: SeatPreference):
    print(f'Purchasing flight {flight_details=!r} {seat=!r}...')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())

```

# Pydantic Model

Simple example of using Pydantic AI to construct a Pydantic model from a text input.

Demonstrates:

- [structured `output_type`](../../output/#structured-output)

## Running the Example

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.pydantic_model

```

```bash
uv run -m pydantic_ai_examples.pydantic_model

```

This examples uses `openai:gpt-5` by default, but it works well with other models, e.g. you can run it with Gemini using:

```bash
PYDANTIC_AI_MODEL=gemini-2.5-pro python -m pydantic_ai_examples.pydantic_model

```

```bash
PYDANTIC_AI_MODEL=gemini-2.5-pro uv run -m pydantic_ai_examples.pydantic_model

```

(or `PYDANTIC_AI_MODEL=gemini-3-pro-preview ...`)

## Example Code

[pydantic_model.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/pydantic_model.py)

```python
"""Simple example of using Pydantic AI to construct a Pydantic model from a text input.

Run with:

    uv run -m pydantic_ai_examples.pydantic_model
"""

import os

import logfire
from pydantic import BaseModel

from pydantic_ai import Agent

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


class MyModel(BaseModel):
    city: str
    country: str


model = os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-5')
print(f'Using model: {model}')
agent = Agent(model, output_type=MyModel)

if __name__ == '__main__':
    result = agent.run_sync('The windy city in the US of A.')
    print(result.output)
    print(result.usage())

```

# Question Graph

Example of a graph for asking and evaluating questions.

Demonstrates:

- [`pydantic_graph`](../../graph/)

## Running the Example

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.question_graph

```

```bash
uv run -m pydantic_ai_examples.question_graph

```

## Example Code

[Learn about Gateway](../../gateway) [question_graph.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/question_graph.py)

```python
"""Example of a graph for asking and evaluating questions.

Run with:

    uv run -m pydantic_ai_examples.question_graph
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from pathlib import Path

import logfire
from groq import BaseModel

from pydantic_ai import Agent, ModelMessage, format_as_xml
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphRunContext,
)
from pydantic_graph.persistence.file import FileStatePersistence

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

ask_agent = Agent('gateway/openai:gpt-5', output_type=str)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    async def run(self, ctx: GraphRunContext[QuestionState]) -> Answer:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.all_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        answer = input(f'{self.question}: ')
        return Evaluate(answer)


class EvaluationOutput(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    'gateway/openai:gpt-5',
    output_type=EvaluationOutput,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> End[str] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.all_messages()
        if result.output.correct:
            return End(result.output.comment)
        else:
            return Reprimand(result.output.comment)


@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        print(f'Comment: {self.comment}')
        ctx.state.question = None
        return Ask()


question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState
)


async def run_as_continuous():
    state = QuestionState()
    node = Ask()
    end = await question_graph.run(node, state=state)
    print('END:', end.output)


async def run_as_cli(answer: str | None):
    persistence = FileStatePersistence(Path('question_graph.json'))
    persistence.set_graph_types(question_graph)

    if snapshot := await persistence.load_next():
        state = snapshot.state
        assert answer is not None, (
            'answer required, usage "uv run -m pydantic_ai_examples.question_graph cli <answer>"'
        )
        node = Evaluate(answer)
    else:
        state = QuestionState()
        node = Ask()
    # debug(state, node)

    async with question_graph.iter(node, state=state, persistence=persistence) as run:
        while True:
            node = await run.next()
            if isinstance(node, End):
                print('END:', node.data)
                history = await persistence.load_all()
                print('history:', '\n'.join(str(e.node) for e in history), sep='\n')
                print('Finished!')
                break
            elif isinstance(node, Answer):
                print(node.question)
                break
            # otherwise just continue


if __name__ == '__main__':
    import asyncio
    import sys

    try:
        sub_command = sys.argv[1]
        assert sub_command in ('continuous', 'cli', 'mermaid')
    except (IndexError, AssertionError):
        print(
            'Usage:\n'
            '  uv run -m pydantic_ai_examples.question_graph mermaid\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph continuous\n'
            'or:\n'
            '  uv run -m pydantic_ai_examples.question_graph cli [answer]',
            file=sys.stderr,
        )
        sys.exit(1)

    if sub_command == 'mermaid':
        print(question_graph.mermaid_code(start_node=Ask))
    elif sub_command == 'continuous':
        asyncio.run(run_as_continuous())
    else:
        a = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(run_as_cli(a))

```

[question_graph.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/question_graph.py)

```python
"""Example of a graph for asking and evaluating questions.

Run with:

    uv run -m pydantic_ai_examples.question_graph
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from pathlib import Path

import logfire
from groq import BaseModel

from pydantic_ai import Agent, ModelMessage, format_as_xml
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphRunContext,
)
from pydantic_graph.persistence.file import FileStatePersistence

