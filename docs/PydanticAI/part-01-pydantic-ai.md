<!-- Source: _complete-doc.md -->

# Pydantic AI

> GenAI Agent Framework, the Pydantic way

Pydantic AI is a Python agent framework designed to make it less painful to build production grade
applications with Generative AI.

# Introduction

# Pydantic AI

*GenAI Agent Framework, the Pydantic way*

Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.

FastAPI revolutionized web development by offering an innovative and ergonomic design, built on the foundation of [Pydantic Validation](https://docs.pydantic.dev) and modern Python features like type hints.

Yet despite virtually every Python agent framework and LLM library using Pydantic Validation, when we began to use LLMs in [Pydantic Logfire](https://pydantic.dev/logfire), we couldn't find anything that gave us the same feeling.

We built Pydantic AI with one simple aim: to bring that FastAPI feeling to GenAI app and agent development.

## Why use Pydantic AI

1. **Built by the Pydantic Team**: [Pydantic Validation](https://docs.pydantic.dev/latest/) is the validation layer of the OpenAI SDK, the Google ADK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more. *Why use the derivative when you can go straight to the source?*
1. **Model-agnostic**: Supports virtually every [model](models/overview/) and provider: OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, and Perplexity; Azure AI Foundry, Amazon Bedrock, Google Vertex AI, Ollama, LiteLLM, Groq, OpenRouter, Together AI, Fireworks AI, Cerebras, Hugging Face, GitHub, Heroku, Vercel, Nebius, OVHcloud, and Outlines. If your favorite model or provider is not listed, you can easily implement a [custom model](models/overview/#custom-models).
1. **Seamless Observability**: Tightly [integrates](logfire/) with [Pydantic Logfire](https://pydantic.dev/logfire), our general-purpose OpenTelemetry observability platform, for real-time debugging, evals-based performance monitoring, and behavior, tracing, and cost tracking. If you already have an observability platform that supports OTel, you can [use that too](logfire/#alternative-observability-backends).
1. **Fully Type-safe**: Designed to give your IDE or AI coding agent as much context as possible for auto-completion and [type checking](agents/#static-type-checking), moving entire classes of errors from runtime to write-time for a bit of that Rust "if it compiles, it works" feel.
1. **Powerful Evals**: Enables you to systematically test and [evaluate](evals/) the performance and accuracy of the agentic systems you build, and monitor the performance over time in Pydantic Logfire.
1. **MCP, A2A, and UI**: Integrates the [Model Context Protocol](mcp/overview/), [Agent2Agent](a2a/), and various [UI event stream](ui/overview/) standards to give your agent access to external tools and data, let it interoperate with other agents, and build interactive applications with streaming event-based communication.
1. **Human-in-the-Loop Tool Approval**: Easily lets you flag that certain tool calls [require approval](deferred-tools/#human-in-the-loop-tool-approval) before they can proceed, possibly depending on tool call arguments, conversation history, or user preferences.
1. **Durable Execution**: Enables you to build [durable agents](durable_execution/overview/) that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability.
1. **Streamed Outputs**: Provides the ability to [stream](output/#streamed-results) structured output continuously, with immediate validation, ensuring real time access to generated data.
1. **Graph Support**: Provides a powerful way to define [graphs](graph/) using type hints, for use in complex applications where standard control flow can degrade to spaghetti code.

Realistically though, no list is going to be as convincing as [giving it a try](#next-steps) and seeing how it makes you feel!

**Sign up for our newsletter, *The Pydantic Stack*, with updates & tutorials on Pydantic AI, Logfire, and Pydantic:**

Subscribe

## Hello World Example

Here's a minimal example of Pydantic AI:

[Learn about Gateway](../gateway) hello_world.py

```python
from pydantic_ai import Agent

agent = Agent(  # (1)!
    'gateway/anthropic:claude-sonnet-4-0',
    instructions='Be concise, reply with one sentence.',  # (2)!
)

result = agent.run_sync('Where does "hello world" come from?')  # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""

```

1. We configure the agent to use [Anthropic's Claude Sonnet 4.0](api/models/anthropic/) model, but you can also set the model when running the agent.
1. Register static [instructions](agents/#instructions) using a keyword argument to the agent.
1. [Run the agent](agents/#running-agents) synchronously, starting a conversation with the LLM.

hello_world.py

```python
from pydantic_ai import Agent

agent = Agent(  # (1)!
    'anthropic:claude-sonnet-4-0',
    instructions='Be concise, reply with one sentence.',  # (2)!
)

result = agent.run_sync('Where does "hello world" come from?')  # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""

```

1. We configure the agent to use [Anthropic's Claude Sonnet 4.0](api/models/anthropic/) model, but you can also set the model when running the agent.
1. Register static [instructions](agents/#instructions) using a keyword argument to the agent.
1. [Run the agent](agents/#running-agents) synchronously, starting a conversation with the LLM.

*(This example is complete, it can be run "as is", assuming you've [installed the `pydantic_ai` package](install/))*

The exchange will be very short: Pydantic AI will send the instructions and the user prompt to the LLM, and the model will return a text response.

Not very interesting yet, but we can easily add [tools](tools/), [dynamic instructions](agents/#instructions), and [structured outputs](output/) to build more powerful agents.

## Tools & Dependency Injection Example

Here is a concise example using Pydantic AI to build a support agent for a bank:

[Learn about Gateway](../gateway) bank_support.py

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:  # (3)!
    customer_id: int
    db: DatabaseConn  # (12)!


class SupportOutput(BaseModel):  # (13)!
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(  # (1)!
    'gateway/openai:gpt-5',  # (2)!
    deps_type=SupportDependencies,
    output_type=SupportOutput,  # (9)!
    instructions=(  # (4)!
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)


@support_agent.instructions  # (5)!
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool  # (6)!
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""  # (7)!
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )


...  # (11)!


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = await support_agent.run('What is my balance?', deps=deps)  # (8)!
    print(result.output)  # (10)!
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.output)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """

```

1. This [agent](agents/) will act as first-tier support in a bank. Agents are generic in the type of dependencies they accept and the type of output they return. In this case, the support agent has type `Agent[SupportDependencies, SupportOutput]`.
1. Here we configure the agent to use [OpenAI's GPT-5 model](api/models/openai/), you can also set the model when running the agent.
1. The `SupportDependencies` dataclass is used to pass data, connections, and logic into the model that will be needed when running [instructions](agents/#instructions) and [tool](tools/) functions. Pydantic AI's system of dependency injection provides a [type-safe](agents/#static-type-checking) way to customise the behavior of your agents, and can be especially useful when running [unit tests](testing/) and evals.
1. Static [instructions](agents/#instructions) can be registered with the instructions keyword argument to the agent.
1. Dynamic [instructions](agents/#instructions) can be registered with the @agent.instructions decorator, and can make use of dependency injection. Dependencies are carried via the RunContext argument, which is parameterized with the `deps_type` from above. If the type annotation here is wrong, static type checkers will catch it.
1. The [`@agent.tool`](tools/) decorator let you register functions which the LLM may call while responding to a user. Again, dependencies are carried via RunContext, any other arguments become the tool schema passed to the LLM. Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.
1. The docstring of a tool is also passed to the LLM as the description of the tool. Parameter descriptions are [extracted](tools/#function-tools-and-schema) from the docstring and added to the parameter schema sent to the LLM.
1. [Run the agent](agents/#running-agents) asynchronously, conducting a conversation with the LLM until a final response is reached. Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.
1. The response from the agent will be guaranteed to be a `SupportOutput`. If validation fails [reflection](agents/#reflection-and-self-correction), the agent is prompted to try again.
1. The output will be validated with Pydantic to guarantee it is a `SupportOutput`, since the agent is generic, it'll also be typed as a `SupportOutput` to aid with static type checking.
1. In a real use case, you'd add more tools and longer instructions to the agent to extend the context it's equipped with and support it can provide.
1. This is a simple sketch of a database connection, used to keep the example short and readable. In reality, you'd be connecting to an external database (e.g. PostgreSQL) to get information about customers.
1. This [Pydantic](https://docs.pydantic.dev) model is used to constrain the structured data returned by the agent. From this simple definition, Pydantic builds the JSON Schema that tells the LLM how to return the data, and performs validation to guarantee the data is correct at the end of the run.

bank_support.py

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:  # (3)!
    customer_id: int
    db: DatabaseConn  # (12)!


class SupportOutput(BaseModel):  # (13)!
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(  # (1)!
    'openai:gpt-5',  # (2)!
    deps_type=SupportDependencies,
    output_type=SupportOutput,  # (9)!
    instructions=(  # (4)!
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)


@support_agent.instructions  # (5)!
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool  # (6)!
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""  # (7)!
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )


...  # (11)!


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = await support_agent.run('What is my balance?', deps=deps)  # (8)!
    print(result.output)  # (10)!
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.output)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """

```

1. This [agent](agents/) will act as first-tier support in a bank. Agents are generic in the type of dependencies they accept and the type of output they return. In this case, the support agent has type `Agent[SupportDependencies, SupportOutput]`.
1. Here we configure the agent to use [OpenAI's GPT-5 model](api/models/openai/), you can also set the model when running the agent.
1. The `SupportDependencies` dataclass is used to pass data, connections, and logic into the model that will be needed when running [instructions](agents/#instructions) and [tool](tools/) functions. Pydantic AI's system of dependency injection provides a [type-safe](agents/#static-type-checking) way to customise the behavior of your agents, and can be especially useful when running [unit tests](testing/) and evals.
1. Static [instructions](agents/#instructions) can be registered with the instructions keyword argument to the agent.
1. Dynamic [instructions](agents/#instructions) can be registered with the @agent.instructions decorator, and can make use of dependency injection. Dependencies are carried via the RunContext argument, which is parameterized with the `deps_type` from above. If the type annotation here is wrong, static type checkers will catch it.
1. The [`@agent.tool`](tools/) decorator let you register functions which the LLM may call while responding to a user. Again, dependencies are carried via RunContext, any other arguments become the tool schema passed to the LLM. Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.
1. The docstring of a tool is also passed to the LLM as the description of the tool. Parameter descriptions are [extracted](tools/#function-tools-and-schema) from the docstring and added to the parameter schema sent to the LLM.
1. [Run the agent](agents/#running-agents) asynchronously, conducting a conversation with the LLM until a final response is reached. Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.
1. The response from the agent will be guaranteed to be a `SupportOutput`. If validation fails [reflection](agents/#reflection-and-self-correction), the agent is prompted to try again.
1. The output will be validated with Pydantic to guarantee it is a `SupportOutput`, since the agent is generic, it'll also be typed as a `SupportOutput` to aid with static type checking.
1. In a real use case, you'd add more tools and longer instructions to the agent to extend the context it's equipped with and support it can provide.
1. This is a simple sketch of a database connection, used to keep the example short and readable. In reality, you'd be connecting to an external database (e.g. PostgreSQL) to get information about customers.
1. This [Pydantic](https://docs.pydantic.dev) model is used to constrain the structured data returned by the agent. From this simple definition, Pydantic builds the JSON Schema that tells the LLM how to return the data, and performs validation to guarantee the data is correct at the end of the run.

Complete `bank_support.py` example

The code included here is incomplete for the sake of brevity (the definition of `DatabaseConn` is missing); you can find the complete `bank_support.py` example [here](examples/bank-support/).

## Instrumentation with Pydantic Logfire

Even a simple agent with just a handful of tools can result in a lot of back-and-forth with the LLM, making it nearly impossible to be confident of what's going on just from reading the code. To understand the flow of the above runs, we can watch the agent in action using Pydantic Logfire.

To do this, we need to [set up Logfire](logfire/#using-logfire), and add the following to our code:

[Learn about Gateway](../gateway) bank_support_with_logfire.py

```python
...
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn

import logfire
logfire.configure()  # (1)!
logfire.instrument_pydantic_ai()  # (2)!
logfire.instrument_asyncpg()  # (3)!

...

support_agent = Agent(
    'gateway/openai:gpt-5',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)

```

1. Configure the Logfire SDK, this will fail if project is not set up.
1. This will instrument all Pydantic AI agents used from here on out. If you want to instrument only a specific agent, you can pass the instrument=True keyword argument to the agent.
1. In our demo, `DatabaseConn` uses `asyncpg` to connect to a PostgreSQL database, so [`logfire.instrument_asyncpg()`](https://magicstack.github.io/asyncpg/current/) is used to log the database queries.

bank_support_with_logfire.py

```python
...
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn

import logfire
logfire.configure()  # (1)!
logfire.instrument_pydantic_ai()  # (2)!
logfire.instrument_asyncpg()  # (3)!

...

support_agent = Agent(
    'openai:gpt-5',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)

```

1. Configure the Logfire SDK, this will fail if project is not set up.
1. This will instrument all Pydantic AI agents used from here on out. If you want to instrument only a specific agent, you can pass the instrument=True keyword argument to the agent.
1. In our demo, `DatabaseConn` uses `asyncpg` to connect to a PostgreSQL database, so [`logfire.instrument_asyncpg()`](https://magicstack.github.io/asyncpg/current/) is used to log the database queries.

That's enough to get the following view of your agent in action:

See [Monitoring and Performance](logfire/) to learn more.

## `llms.txt`

The Pydantic AI documentation is available in the [llms.txt](https://llmstxt.org/) format. This format is defined in Markdown and suited for LLMs and AI coding assistants and agents.

Two formats are available:

- [`llms.txt`](https://ai.pydantic.dev/llms.txt): a file containing a brief description of the project, along with links to the different sections of the documentation. The structure of this file is described in details [here](https://llmstxt.org/#format).
- [`llms-full.txt`](https://ai.pydantic.dev/llms-full.txt): Similar to the `llms.txt` file, but every link content is included. Note that this file may be too large for some LLMs.

As of today, these files are not automatically leveraged by IDEs or coding agents, but they will use it if you provide a link or the full text.

## Next Steps

To try Pydantic AI for yourself, [install it](install/) and follow the instructions [in the examples](examples/setup/).

Read the [docs](agents/) to learn more about building applications with Pydantic AI.

Read the [API Reference](api/agent/) to understand Pydantic AI's interface.

Join [Slack](https://logfire.pydantic.dev/docs/join-slack/) or file an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues) if you have any questions.

# Getting Help

If you need help getting started with Pydantic AI or with advanced usage, the following sources may be useful.

## Slack

Join the `#pydantic-ai` channel in the [Pydantic Slack](https://logfire.pydantic.dev/docs/join-slack/) to ask questions, get help, and chat about Pydantic AI. There's also channels for Pydantic, Logfire, and FastUI.

If you're on a [Logfire](https://pydantic.dev/logfire) Pro plan, you can also get a dedicated private slack collab channel with us.

## GitHub Issues

The [Pydantic AI GitHub Issues](https://github.com/pydantic/pydantic-ai/issues) are a great place to ask questions and give us feedback.

# Installation

Pydantic AI is available on PyPI as [`pydantic-ai`](https://pypi.org/project/pydantic-ai/) so installation is as simple as:

```bash
pip install pydantic-ai

```

```bash
uv add pydantic-ai

```

(Requires Python 3.10+)

This installs the `pydantic_ai` package, core dependencies, and libraries required to use all the models included in Pydantic AI. If you want to install only those dependencies required to use a specific model, you can install the ["slim"](#slim-install) version of Pydantic AI.

## Use with Pydantic Logfire

Pydantic AI has an excellent (but completely optional) integration with [Pydantic Logfire](https://pydantic.dev/logfire) to help you view and understand agent runs.

Logfire comes included with `pydantic-ai` (but not the ["slim" version](#slim-install)), so you can typically start using it immediately by following the [Logfire setup docs](../logfire/#using-logfire).

## Running Examples

We distribute the [`pydantic_ai_examples`](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples) directory as a separate PyPI package ([`pydantic-ai-examples`](https://pypi.org/project/pydantic-ai-examples/)) to make examples extremely easy to customize and run.

To install examples, use the `examples` optional group:

```bash
pip install "pydantic-ai[examples]"

```

```bash
uv add "pydantic-ai[examples]"

```

To run the examples, follow instructions in the [examples docs](../examples/setup/).

## Slim Install

If you know which model you're going to use and want to avoid installing superfluous packages, you can use the [`pydantic-ai-slim`](https://pypi.org/project/pydantic-ai-slim/) package. For example, if you're using just OpenAIChatModel, you would run:

```bash
pip install "pydantic-ai-slim[openai]"

```

```bash
uv add "pydantic-ai-slim[openai]"

```

`pydantic-ai-slim` has the following optional groups:

- `logfire` â€” installs [Pydantic Logfire](../logfire/) dependency `logfire` [PyPI â†—](https://pypi.org/project/logfire)
- `evals` â€” installs [Pydantic Evals](../evals/) dependency `pydantic-evals` [PyPI â†—](https://pypi.org/project/pydantic-evals)
- `openai` â€” installs [OpenAI Model](../models/openai/) dependency `openai` [PyPI â†—](https://pypi.org/project/openai)
- `vertexai` â€” installs `GoogleVertexProvider` dependencies `google-auth` [PyPI â†—](https://pypi.org/project/google-auth) and `requests` [PyPI â†—](https://pypi.org/project/requests)
- `google` â€” installs [Google Model](../models/google/) dependency `google-genai` [PyPI â†—](https://pypi.org/project/google-genai)
- `anthropic` â€” installs [Anthropic Model](../models/anthropic/) dependency `anthropic` [PyPI â†—](https://pypi.org/project/anthropic)
- `groq` â€” installs [Groq Model](../models/groq/) dependency `groq` [PyPI â†—](https://pypi.org/project/groq)
- `mistral` â€” installs [Mistral Model](../models/mistral/) dependency `mistralai` [PyPI â†—](https://pypi.org/project/mistralai)
- `cohere` - installs [Cohere Model](../models/cohere/) dependency `cohere` [PyPI â†—](https://pypi.org/project/cohere)
- `bedrock` - installs [Bedrock Model](../models/bedrock/) dependency `boto3` [PyPI â†—](https://pypi.org/project/boto3)
- `huggingface` - installs [Hugging Face Model](../models/huggingface/) dependency `huggingface-hub[inference]` [PyPI â†—](https://pypi.org/project/huggingface-hub)
- `outlines-transformers` - installs [Outlines Model](../models/outlines/) dependency `outlines[transformers]` [PyPI â†—](https://pypi.org/project/outlines)
- `outlines-llamacpp` - installs [Outlines Model](../models/outlines/) dependency `outlines[llamacpp]` [PyPI â†—](https://pypi.org/project/outlines)
- `outlines-mlxlm` - installs [Outlines Model](../models/outlines/) dependency `outlines[mlxlm]` [PyPI â†—](https://pypi.org/project/outlines)
- `outlines-sglang` - installs [Outlines Model](../models/outlines/) dependency `outlines[sglang]` [PyPI â†—](https://pypi.org/project/outlines)
- `outlines-vllm-offline` - installs [Outlines Model](../models/outlines/) dependencies `outlines` [PyPI â†—](https://pypi.org/project/outlines) and `vllm` [PyPI â†—](https://pypi.org/project/vllm)
- `duckduckgo` - installs [DuckDuckGo Search Tool](../common-tools/#duckduckgo-search-tool) dependency `ddgs` [PyPI â†—](https://pypi.org/project/ddgs)
- `tavily` - installs [Tavily Search Tool](../common-tools/#tavily-search-tool) dependency `tavily-python` [PyPI â†—](https://pypi.org/project/tavily-python)
- `cli` - installs [CLI](../cli/) dependencies `rich` [PyPI â†—](https://pypi.org/project/rich), `prompt-toolkit` [PyPI â†—](https://pypi.org/project/prompt-toolkit), and `argcomplete` [PyPI â†—](https://pypi.org/project/argcomplete)
- `mcp` - installs [MCP](../mcp/client/) dependency `mcp` [PyPI â†—](https://pypi.org/project/mcp)
- `fastmcp` - installs [FastMCP](../mcp/fastmcp-client/) dependency `fastmcp` [PyPI â†—](https://pypi.org/project/fastmcp)
- `a2a` - installs [A2A](../a2a/) dependency `fasta2a` [PyPI â†—](https://pypi.org/project/fasta2a)
- `ui` - installs [UI Event Streams](../ui/overview/) dependency `starlette` [PyPI â†—](https://pypi.org/project/starlette)
- `ag-ui` - installs [AG-UI Event Stream Protocol](../ui/ag-ui/) dependencies `ag-ui-protocol` [PyPI â†—](https://pypi.org/project/ag-ui-protocol) and `starlette` [PyPI â†—](https://pypi.org/project/starlette)
- `dbos` - installs [DBOS Durable Execution](../durable_execution/dbos/) dependency `dbos` [PyPI â†—](https://pypi.org/project/dbos)
- `prefect` - installs [Prefect Durable Execution](../durable_execution/prefect/) dependency `prefect` [PyPI â†—](https://pypi.org/project/prefect)

You can also install dependencies for multiple models and use cases, for example:

```bash
pip install "pydantic-ai-slim[openai,google,logfire]"

```

```bash
uv add "pydantic-ai-slim[openai,google,logfire]"

```

# Troubleshooting

Below are suggestions on how to fix some common errors you might encounter while using Pydantic AI. If the issue you're experiencing is not listed below or addressed in the documentation, please feel free to ask in the [Pydantic Slack](../help/) or create an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).

## Jupyter Notebook Errors

### `RuntimeError: This event loop is already running`

This error is caused by conflicts between the event loops in Jupyter notebook and Pydantic AI's. One way to manage these conflicts is by using [`nest-asyncio`](https://pypi.org/project/nest-asyncio/). Namely, before you execute any agent runs, do the following:

```python
import nest_asyncio

nest_asyncio.apply()

```

Note: This fix also applies to Google Colab and [Marimo](https://github.com/marimo-team/marimo).

## API Key Configuration

### `UserError: API key must be provided or set in the [MODEL]_API_KEY environment variable`

If you're running into issues with setting the API key for your model, visit the [Models](../models/overview/) page to learn more about how to set an environment variable and/or pass in an `api_key` argument.

## Monitoring HTTPX Requests

You can use custom `httpx` clients in your models in order to access specific requests, responses, and headers at runtime.

It's particularly helpful to use `logfire`'s [HTTPX integration](../logfire/#monitoring-http-requests) to monitor the above.
# Concepts documentation

# Agent2Agent (A2A) Protocol

The [Agent2Agent (A2A) Protocol](https://google.github.io/A2A/) is an open standard introduced by Google that enables communication and interoperability between AI agents, regardless of the framework or vendor they are built on.

At Pydantic, we built the [FastA2A](#fasta2a) library to make it easier to implement the A2A protocol in Python.

We also built a convenience method that expose Pydantic AI agents as A2A servers - let's have a quick look at how to use it:

[Learn about Gateway](../gateway) agent_to_a2a.py

```python
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5', instructions='Be fun!')
app = agent.to_a2a()

```

agent_to_a2a.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='Be fun!')
app = agent.to_a2a()

```

*You can run the example with `uvicorn agent_to_a2a:app --host 0.0.0.0 --port 8000`*

This will expose the agent as an A2A server, and you can start sending requests to it.

See more about [exposing Pydantic AI agents as A2A servers](#pydantic-ai-agent-to-a2a-server).

## FastA2A

**FastA2A** is an agentic framework agnostic implementation of the A2A protocol in Python. The library is designed to be used with any agentic framework, and is **not exclusive to Pydantic AI**.

### Design

**FastA2A** is built on top of [Starlette](https://www.starlette.io), which means it's fully compatible with any ASGI server.

Given the nature of the A2A protocol, it's important to understand the design before using it, as a developer you'll need to provide some components:

- Storage: to save and load tasks, as well as store context for conversations
- Broker: to schedule tasks
- Worker: to execute tasks

Let's have a look at how those components fit together:

```
flowchart TB
    Server["HTTP Server"] <--> |Sends Requests/<br>Receives Results| TM

    subgraph CC[Core Components]
        direction RL
        TM["TaskManager<br>(coordinates)"] --> |Schedules Tasks| Broker
        TM <--> Storage
        Broker["Broker<br>(queues & schedules)"] <--> Storage["Storage<br>(persistence)"]
        Broker --> |Delegates Execution| Worker
    end

    Worker["Worker<br>(implementation)"]
```

FastA2A allows you to bring your own Storage, Broker and Worker.

#### Understanding Tasks and Context

In the A2A protocol:

- **Task**: Represents one complete execution of an agent. When a client sends a message to the agent, a new task is created. The agent runs until completion (or failure), and this entire execution is considered one task. The final output is stored as a task artifact.
- **Context**: Represents a conversation thread that can span multiple tasks. The A2A protocol uses a `context_id` to maintain conversation continuity:
- When a new message is sent without a `context_id`, the server generates a new one
- Subsequent messages can include the same `context_id` to continue the conversation
- All tasks sharing the same `context_id` have access to the complete message history

#### Storage Architecture

The Storage component serves two purposes:

1. **Task Storage**: Stores tasks in A2A protocol format, including their status, artifacts, and message history
1. **Context Storage**: Stores conversation context in a format optimized for the specific agent implementation

This design allows for agents to store rich internal state (e.g., tool calls, reasoning traces) as well as store task-specific A2A-formatted messages and artifacts.

For example, a Pydantic AI agent might store its complete internal message format (including tool calls and responses) in the context storage, while storing only the A2A-compliant messages in the task history.

### Installation

FastA2A is available on PyPI as [`fasta2a`](https://pypi.org/project/fasta2a/) so installation is as simple as:

```bash
pip install fasta2a

```

```bash
uv add fasta2a

```

The only dependencies are:

- [starlette](https://www.starlette.io): to expose the A2A server as an [ASGI application](https://asgi.readthedocs.io/en/latest/)
- [pydantic](https://pydantic.dev): to validate the request/response messages
- [opentelemetry-api](https://opentelemetry-python.readthedocs.io/en/latest): to provide tracing capabilities

You can install Pydantic AI with the `a2a` extra to include **FastA2A**:

```bash
pip install 'pydantic-ai-slim[a2a]'

```

```bash
uv add 'pydantic-ai-slim[a2a]'

```

### Pydantic AI Agent to A2A Server

To expose a Pydantic AI agent as an A2A server, you can use the `to_a2a` method:

[Learn about Gateway](../gateway) agent_to_a2a.py

```python
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5', instructions='Be fun!')
app = agent.to_a2a()

```

agent_to_a2a.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='Be fun!')
app = agent.to_a2a()

```

Since `app` is an ASGI application, it can be used with any ASGI server.

```bash
uvicorn agent_to_a2a:app --host 0.0.0.0 --port 8000

```

Since the goal of `to_a2a` is to be a convenience method, it accepts the same arguments as the FastA2A constructor.

When using `to_a2a()`, Pydantic AI automatically:

- Stores the complete conversation history (including tool calls and responses) in the context storage
- Ensures that subsequent messages with the same `context_id` have access to the full conversation history
- Persists agent results as A2A artifacts:
- String results become `TextPart` artifacts and also appear in the message history
- Structured data (Pydantic models, dataclasses, tuples, etc.) become `DataPart` artifacts with the data wrapped as `{"result": <your_data>}`
- Artifacts include metadata with type information and JSON schema when available

## Introduction

Agents are Pydantic AI's primary interface for interacting with LLMs.

In some use cases a single Agent will control an entire application or component, but multiple agents can also interact to embody more complex workflows.

The Agent class has full API documentation, but conceptually you can think of an agent as a container for:

| **Component** | **Description** | | --- | --- | | [Instructions](#instructions) | A set of instructions for the LLM written by the developer. | | [Function tool(s)](../tools/) and [toolsets](../toolsets/) | Functions that the LLM may call to get information while generating a response. | | [Structured output type](../output/) | The structured datatype the LLM must return at the end of a run, if specified. | | [Dependency type constraint](../dependencies/) | Dynamic instructions functions, tools, and output functions may all use dependencies when they're run. | | [LLM model](../api/models/base/) | Optional default LLM model associated with the agent. Can also be specified when running the agent. | | [Model Settings](#additional-configuration) | Optional default model settings to help fine tune requests. Can also be specified when running the agent. |

In typing terms, agents are generic in their dependency and output types, e.g., an agent which required dependencies of type `Foobar` and produced outputs of type `list[str]` would have type `Agent[Foobar, list[str]]`. In practice, you shouldn't need to care about this, it should just mean your IDE can tell you when you have the right type, and if you choose to use [static type checking](#static-type-checking) it should work well with Pydantic AI.

Here's a toy example of an agent that simulates a roulette wheel:

[Learn about Gateway](../gateway) roulette_wheel.py

```python
from pydantic_ai import Agent, RunContext

roulette_agent = Agent(  # (1)!
    'gateway/openai:gpt-5',
    deps_type=int,
    output_type=bool,
    system_prompt=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  # (2)!
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'


# Run the agent
success_number = 18  # (3)!
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.output)  # (4)!
#> True

result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.output)
#> False

```

1. Create an agent, which expects an integer dependency and produces a boolean output. This agent will have type `Agent[int, bool]`.
1. Define a tool that checks if the square is a winner. Here RunContext is parameterized with the dependency type `int`; if you got the dependency type wrong you'd get a typing error.
1. In reality, you might want to use a random number here e.g. `random.randint(0, 36)`.
1. `result.output` will be a boolean indicating if the square is a winner. Pydantic performs the output validation, and it'll be typed as a `bool` since its type is derived from the `output_type` generic parameter of the agent.

roulette_wheel.py

```python
from pydantic_ai import Agent, RunContext

roulette_agent = Agent(  # (1)!
    'openai:gpt-5',
    deps_type=int,
    output_type=bool,
    system_prompt=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  # (2)!
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'


# Run the agent
success_number = 18  # (3)!
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.output)  # (4)!
#> True

result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.output)
#> False

```

1. Create an agent, which expects an integer dependency and produces a boolean output. This agent will have type `Agent[int, bool]`.
1. Define a tool that checks if the square is a winner. Here RunContext is parameterized with the dependency type `int`; if you got the dependency type wrong you'd get a typing error.
1. In reality, you might want to use a random number here e.g. `random.randint(0, 36)`.
1. `result.output` will be a boolean indicating if the square is a winner. Pydantic performs the output validation, and it'll be typed as a `bool` since its type is derived from the `output_type` generic parameter of the agent.

Agents are designed for reuse, like FastAPI Apps

Agents are intended to be instantiated once (frequently as module globals) and reused throughout your application, similar to a small FastAPI app or an APIRouter.

## Running Agents

There are five ways to run an agent:

1. agent.run() â€” an async function which returns a RunResult containing a completed response.
1. agent.run_sync() â€” a plain, synchronous function which returns a RunResult containing a completed response (internally, this just calls `loop.run_until_complete(self.run())`).
1. agent.run_stream() â€” an async context manager which returns a StreamedRunResult, which contains methods to stream text and structured output as an async iterable. agent.run_stream_sync() is a synchronous variation that returns a StreamedRunResultSync with synchronous versions of the same methods.
1. agent.run_stream_events() â€” a function which returns an async iterable of AgentStreamEvents and a AgentRunResultEvent containing the final run result.
1. agent.iter() â€” a context manager which returns an AgentRun, an async iterable over the nodes of the agent's underlying Graph.

Here's a simple example demonstrating the first four:

[Learn about Gateway](../gateway) run_agent.py

```python
from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

agent = Agent('gateway/openai:gpt-5')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)
#> The capital of Italy is Rome.


async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)
    #> The capital of France is Paris.

    async with agent.run_stream('What is the capital of the UK?') as response:
        async for text in response.stream_text():
            print(text)
            #> The capital of
            #> The capital of the UK is
            #> The capital of the UK is London.

    events: list[AgentStreamEvent | AgentRunResultEvent] = []
    async for event in agent.run_stream_events('What is the capital of Mexico?'):
        events.append(event)
    print(events)
    """
    [
        PartStartEvent(index=0, part=TextPart(content='The capital of ')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='Mexico is Mexico ')),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='City.')),
        PartEndEvent(
            index=0, part=TextPart(content='The capital of Mexico is Mexico City.')
        ),
        AgentRunResultEvent(
            result=AgentRunResult(output='The capital of Mexico is Mexico City.')
        ),
    ]
    """

```

run_agent.py

```python
from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

agent = Agent('openai:gpt-5')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)
#> The capital of Italy is Rome.


async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)
    #> The capital of France is Paris.

    async with agent.run_stream('What is the capital of the UK?') as response:
        async for text in response.stream_text():
            print(text)
            #> The capital of
            #> The capital of the UK is
            #> The capital of the UK is London.

    events: list[AgentStreamEvent | AgentRunResultEvent] = []
    async for event in agent.run_stream_events('What is the capital of Mexico?'):
        events.append(event)
    print(events)
    """
    [
        PartStartEvent(index=0, part=TextPart(content='The capital of ')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='Mexico is Mexico ')),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='City.')),
        PartEndEvent(
            index=0, part=TextPart(content='The capital of Mexico is Mexico City.')
        ),
        AgentRunResultEvent(
            result=AgentRunResult(output='The capital of Mexico is Mexico City.')
        ),
    ]
    """

```

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

You can also pass messages from previous runs to continue a conversation or provide context, as described in [Messages and Chat History](../message-history/).

### Streaming Events and Final Output

As shown in the example above, run_stream() makes it easy to stream the agent's final output as it comes in. It also takes an optional `event_stream_handler` argument that you can use to gain insight into what is happening during the run before the final output is produced.

The example below shows how to stream events and text output. You can also [stream structured output](../output/#streaming-structured-output).

Note

As the `run_stream()` method will consider the first output matching the [output type](../output/#structured-output) to be the final output, it will stop running the agent graph and will not execute any tool calls made by the model after this "final" output.

If you want to always run the agent graph to completion and stream all events from the model's streaming response and the agent's execution of tools, use agent.run_stream_events() or agent.iter() instead, as described in the following sections.

[Learn about Gateway](../gateway) run_stream_event_stream_handler.py

```python
import asyncio
from collections.abc import AsyncIterable
from datetime import date

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)

weather_agent = Agent(
    'gateway/openai:gpt-5',
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext,
    location: str,
    forecast_date: date,
) -> str:
    return f'The forecast in {location} on {forecast_date} is 24Â°C and sunny.'


output_messages: list[str] = []

async def handle_event(event: AgentStreamEvent):
    if isinstance(event, PartStartEvent):
        output_messages.append(f'[Request] Starting part {event.index}: {event.part!r}')
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            output_messages.append(f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ThinkingPartDelta):
            output_messages.append(f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ToolCallPartDelta):
            output_messages.append(f'[Request] Part {event.index} args delta: {event.delta.args_delta}')
    elif isinstance(event, FunctionToolCallEvent):
        output_messages.append(
            f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
        )
    elif isinstance(event, FunctionToolResultEvent):
        output_messages.append(f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}')
    elif isinstance(event, FinalResultEvent):
        output_messages.append(f'[Result] The model starting producing a final result (tool_name={event.tool_name})')


async def event_stream_handler(
    ctx: RunContext,
    event_stream: AsyncIterable[AgentStreamEvent],
):
    async for event in event_stream:
        await handle_event(event)

async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    async with weather_agent.run_stream(user_prompt, event_stream_handler=event_stream_handler) as run:
        async for output in run.stream_text():
            output_messages.append(f'[Output] {output}')


if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
    """
    [
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24Â°C and sunny.",
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model starting producing a final result (tool_name=None)',
        '[Output] It will be ',
        '[Output] It will be warm and sunny ',
        '[Output] It will be warm and sunny in Paris on ',
        '[Output] It will be warm and sunny in Paris on Tuesday.',
    ]
    """

```

run_stream_event_stream_handler.py

```python
import asyncio
from collections.abc import AsyncIterable
from datetime import date

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)

weather_agent = Agent(
    'openai:gpt-5',
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext,
    location: str,
    forecast_date: date,
) -> str:
    return f'The forecast in {location} on {forecast_date} is 24Â°C and sunny.'


output_messages: list[str] = []

async def handle_event(event: AgentStreamEvent):
    if isinstance(event, PartStartEvent):
        output_messages.append(f'[Request] Starting part {event.index}: {event.part!r}')
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            output_messages.append(f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ThinkingPartDelta):
            output_messages.append(f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ToolCallPartDelta):
            output_messages.append(f'[Request] Part {event.index} args delta: {event.delta.args_delta}')
    elif isinstance(event, FunctionToolCallEvent):
        output_messages.append(
            f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
        )
    elif isinstance(event, FunctionToolResultEvent):
        output_messages.append(f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}')
    elif isinstance(event, FinalResultEvent):
        output_messages.append(f'[Result] The model starting producing a final result (tool_name={event.tool_name})')


async def event_stream_handler(
    ctx: RunContext,
    event_stream: AsyncIterable[AgentStreamEvent],
):
    async for event in event_stream:
        await handle_event(event)

async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    async with weather_agent.run_stream(user_prompt, event_stream_handler=event_stream_handler) as run:
        async for output in run.stream_text():
            output_messages.append(f'[Output] {output}')


if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
    """
    [
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24Â°C and sunny.",
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model starting producing a final result (tool_name=None)',
        '[Output] It will be ',
        '[Output] It will be warm and sunny ',
        '[Output] It will be warm and sunny in Paris on ',
        '[Output] It will be warm and sunny in Paris on Tuesday.',
    ]
    """

```

### Streaming All Events

Like `agent.run_stream()`, agent.run() takes an optional `event_stream_handler` argument that lets you stream all events from the model's streaming response and the agent's execution of tools. Unlike `run_stream()`, it always runs the agent graph to completion even if text was received ahead of tool calls that looked like it could've been the final result.

For convenience, a agent.run_stream_events() method is also available as a wrapper around `run(event_stream_handler=...)`, which returns an async iterable of AgentStreamEvents and a AgentRunResultEvent containing the final run result.

Note

As they return raw events as they come in, the `run_stream_events()` and `run(event_stream_handler=...)` methods require you to piece together the streamed text and structured output yourself from the `PartStartEvent` and subsequent `PartDeltaEvent`s.

To get the best of both worlds, at the expense of some additional complexity, you can use agent.iter() as described in the next section, which lets you [iterate over the agent graph](#iterating-over-an-agents-graph) and [stream both events and output](#streaming-all-events-and-output) at every step.

run_events.py

```python
import asyncio

from pydantic_ai import AgentRunResultEvent

from run_stream_event_stream_handler import handle_event, output_messages, weather_agent


async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    async for event in weather_agent.run_stream_events(user_prompt):
        if isinstance(event, AgentRunResultEvent):
            output_messages.append(f'[Final Output] {event.result.output}')
        else:
            await handle_event(event)

if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
    """
    [
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24Â°C and sunny.",
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model starting producing a final result (tool_name=None)',
        "[Request] Part 0 text delta: 'warm and sunny '",
        "[Request] Part 0 text delta: 'in Paris on '",
        "[Request] Part 0 text delta: 'Tuesday.'",
        '[Final Output] It will be warm and sunny in Paris on Tuesday.',
    ]
    """

```

*(This example is complete, it can be run "as is")*

### Iterating Over an Agent's Graph

Under the hood, each `Agent` in Pydantic AI uses **pydantic-graph** to manage its execution flow. **pydantic-graph** is a generic, type-centric library for building and running finite state machines in Python. It doesn't actually depend on Pydantic AI â€” you can use it standalone for workflows that have nothing to do with GenAI â€” but Pydantic AI makes use of it to orchestrate the handling of model requests and model responses in an agent's run.

In many scenarios, you don't need to worry about pydantic-graph at all; calling `agent.run(...)` simply traverses the underlying graph from start to finish. However, if you need deeper insight or control â€” for example to inject your own logic at specific stages â€” Pydantic AI exposes the lower-level iteration process via Agent.iter. This method returns an AgentRun, which you can async-iterate over, or manually drive node-by-node via the next method. Once the agent's graph returns an End, you have the final result along with a detailed history of all steps.

#### `async for` iteration

Here's an example of using `async for` with `iter` to record each node the agent executes:

[Learn about Gateway](../gateway) agent_iter_async_for.py

```python
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5')


async def main():
    nodes = []
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            nodes.append(node)
    print(nodes)
    """
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
                model_name='gpt-5',
                timestamp=datetime.datetime(...),
                run_id='...',
            )
        ),
        End(data=FinalResult(output='The capital of France is Paris.')),
    ]
    """
    print(agent_run.result.output)
    #> The capital of France is Paris.

```

agent_iter_async_for.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')


async def main():
    nodes = []
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            nodes.append(node)
    print(nodes)
    """
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
                model_name='gpt-5',
                timestamp=datetime.datetime(...),
                run_id='...',
            )
        ),
        End(data=FinalResult(output='The capital of France is Paris.')),
    ]
    """
    print(agent_run.result.output)
    #> The capital of France is Paris.

```

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

- The `AgentRun` is an async iterator that yields each node (`BaseNode` or `End`) in the flow.
- The run ends when an `End` node is returned.

#### Using `.next(...)` manually

You can also drive the iteration manually by passing the node you want to run next to the `AgentRun.next(...)` method. This allows you to inspect or modify the node before it executes or skip nodes based on your own logic, and to catch errors in `next()` more easily:

[Learn about Gateway](../gateway) agent_iter_next.py

```python
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('gateway/openai:gpt-5')


async def main():
    async with agent.iter('What is the capital of France?') as agent_run:
        node = agent_run.next_node  # (1)!

        all_nodes = [node]

        # Drive the iteration manually:
        while not isinstance(node, End):  # (2)!
            node = await agent_run.next(node)  # (3)!
            all_nodes.append(node)  # (4)!

        print(all_nodes)
        """
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
                    model_name='gpt-5',
                    timestamp=datetime.datetime(...),
                    run_id='...',
                )
            ),
            End(data=FinalResult(output='The capital of France is Paris.')),
        ]
        """

```

1. We start by grabbing the first node that will be run in the agent's graph.
1. The agent run is finished once an `End` node has been produced; instances of `End` cannot be passed to `next`.
1. When you call `await agent_run.next(node)`, it executes that node in the agent's graph, updates the run's history, and returns the *next* node to run.
1. You could also inspect or mutate the new `node` here as needed.

agent_iter_next.py

```python
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-5')


async def main():
    async with agent.iter('What is the capital of France?') as agent_run:
        node = agent_run.next_node  # (1)!

        all_nodes = [node]

        # Drive the iteration manually:
        while not isinstance(node, End):  # (2)!
            node = await agent_run.next(node)  # (3)!
            all_nodes.append(node)  # (4)!

        print(all_nodes)
        """
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
                    model_name='gpt-5',
                    timestamp=datetime.datetime(...),
                    run_id='...',
                )
            ),
            End(data=FinalResult(output='The capital of France is Paris.')),
        ]
        """

```

1. We start by grabbing the first node that will be run in the agent's graph.
1. The agent run is finished once an `End` node has been produced; instances of `End` cannot be passed to `next`.
1. When you call `await agent_run.next(node)`, it executes that node in the agent's graph, updates the run's history, and returns the *next* node to run.
1. You could also inspect or mutate the new `node` here as needed.

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

#### Accessing usage and final output

You can retrieve usage statistics (tokens, requests, etc.) at any time from the AgentRun object via `agent_run.usage()`. This method returns a RunUsage object containing the usage data.

Once the run finishes, `agent_run.result` becomes a AgentRunResult object containing the final output (and related metadata).

#### Streaming All Events and Output

Here is an example of streaming an agent run in combination with `async for` iteration:

streaming_iter.py

```python
import asyncio
from dataclasses import dataclass
from datetime import date

from pydantic_ai import (
    Agent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)


@dataclass
class WeatherService:
    async def get_forecast(self, location: str, forecast_date: date) -> str:
        # In real code: call weather API, DB queries, etc.
        return f'The forecast in {location} on {forecast_date} is 24Â°C and sunny.'

    async def get_historic_weather(self, location: str, forecast_date: date) -> str:
        # In real code: call a historical weather API or DB
        return f'The weather in {location} on {forecast_date} was 18Â°C and partly cloudy.'


weather_agent = Agent[WeatherService, str](
    'openai:gpt-5',
    deps_type=WeatherService,
    output_type=str,  # We'll produce a final answer as plain text
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext[WeatherService],
    location: str,
    forecast_date: date,
) -> str:
    if forecast_date >= date.today():
        return await ctx.deps.get_forecast(location, forecast_date)
    else:
        return await ctx.deps.get_historic_weather(location, forecast_date)


output_messages: list[str] = []


async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    # Begin a node-by-node, streaming iteration
    async with weather_agent.iter(user_prompt, deps=WeatherService()) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                # A user prompt node => The user has provided input
                output_messages.append(f'=== UserPromptNode: {node.user_prompt} ===')
            elif Agent.is_model_request_node(node):
                # A model request node => We can stream tokens from the model's request
                output_messages.append('=== ModelRequestNode: streaming partial request tokens ===')
                async with node.stream(run.ctx) as request_stream:
                    final_result_found = False
                    async for event in request_stream:
                        if isinstance(event, PartStartEvent):
                            output_messages.append(f'[Request] Starting part {event.index}: {event.part!r}')
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                output_messages.append(
                                    f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}'
                                )
                            elif isinstance(event.delta, ThinkingPartDelta):
                                output_messages.append(
                                    f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}'
                                )
                            elif isinstance(event.delta, ToolCallPartDelta):
                                output_messages.append(
                                    f'[Request] Part {event.index} args delta: {event.delta.args_delta}'
                                )
                        elif isinstance(event, FinalResultEvent):
                            output_messages.append(
                                f'[Result] The model started producing a final result (tool_name={event.tool_name})'
                            )
                            final_result_found = True
                            break

                    if final_result_found:
                        # Once the final result is found, we can call `AgentStream.stream_text()` to stream the text.
                        # A similar `AgentStream.stream_output()` method is available to stream structured output.
                        async for output in request_stream.stream_text():
                            output_messages.append(f'[Output] {output}')
            elif Agent.is_call_tools_node(node):
                # A handle-response node => The model returned some data, potentially calls a tool
                output_messages.append('=== CallToolsNode: streaming partial response & tool usage ===')
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            output_messages.append(
                                f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
                            )
                        elif isinstance(event, FunctionToolResultEvent):
                            output_messages.append(
                                f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}'
                            )
            elif Agent.is_end_node(node):
                # Once an End node is reached, the agent run is complete
                assert run.result is not None
                assert run.result.output == node.data.output
                output_messages.append(f'=== Final Agent Output: {run.result.output} ===')


if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
    """
    [
        '=== UserPromptNode: What will the weather be like in Paris on Tuesday? ===',
        '=== ModelRequestNode: streaming partial request tokens ===',
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        '=== CallToolsNode: streaming partial response & tool usage ===',
        '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24Â°C and sunny.",
        '=== ModelRequestNode: streaming partial request tokens ===',
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model started producing a final result (tool_name=None)',
        '[Output] It will be ',
        '[Output] It will be warm and sunny ',
        '[Output] It will be warm and sunny in Paris on ',
        '[Output] It will be warm and sunny in Paris on Tuesday.',
        '=== CallToolsNode: streaming partial response & tool usage ===',
        '=== Final Agent Output: It will be warm and sunny in Paris on Tuesday. ===',
    ]
    """

```

*(This example is complete, it can be run "as is")*

### Additional Configuration

#### Usage Limits

Pydantic AI offers a UsageLimits structure to help you limit your usage (tokens, requests, and tool calls) on model runs.

You can apply these settings by passing the `usage_limits` argument to the `run{_sync,_stream}` functions.

Consider the following example, where we limit the number of response tokens:

[Learn about Gateway](../gateway)

```python
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

agent = Agent('gateway/anthropic:claude-sonnet-4-5')

result_sync = agent.run_sync(
    'What is the capital of Italy? Answer with just the city.',
    usage_limits=UsageLimits(response_tokens_limit=10),
)
print(result_sync.output)
#> Rome
print(result_sync.usage())
#> RunUsage(input_tokens=62, output_tokens=1, requests=1)

try:
    result_sync = agent.run_sync(
        'What is the capital of Italy? Answer with a paragraph.',
        usage_limits=UsageLimits(response_tokens_limit=10),
    )
except UsageLimitExceeded as e:
    print(e)
    #> Exceeded the output_tokens_limit of 10 (output_tokens=32)

```

```python
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

agent = Agent('anthropic:claude-sonnet-4-5')

result_sync = agent.run_sync(
    'What is the capital of Italy? Answer with just the city.',
    usage_limits=UsageLimits(response_tokens_limit=10),
)
print(result_sync.output)
#> Rome
print(result_sync.usage())
#> RunUsage(input_tokens=62, output_tokens=1, requests=1)

try:
    result_sync = agent.run_sync(
        'What is the capital of Italy? Answer with a paragraph.',
        usage_limits=UsageLimits(response_tokens_limit=10),
    )
except UsageLimitExceeded as e:
    print(e)
    #> Exceeded the output_tokens_limit of 10 (output_tokens=32)

```

Restricting the number of requests can be useful in preventing infinite loops or excessive tool calling:

[Learn about Gateway](../gateway)

```python
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelRetry, UsageLimitExceeded, UsageLimits


class NeverOutputType(TypedDict):
    """
    Never ever coerce data to this type.
    """

    never_use_this: str


agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    retries=3,
    output_type=NeverOutputType,
    system_prompt='Any time you get a response, call the `infinite_retry_tool` to produce another response.',
)


@agent.tool_plain(retries=5)  # (1)!
def infinite_retry_tool() -> int:
    raise ModelRetry('Please try again.')


try:
    result_sync = agent.run_sync(
        'Begin infinite retry loop!', usage_limits=UsageLimits(request_limit=3)  # (2)!
    )
except UsageLimitExceeded as e:
    print(e)
    #> The next request would exceed the request_limit of 3

```

1. This tool has the ability to retry 5 times before erroring, simulating a tool that might get stuck in a loop.
1. This run will error after 3 requests, preventing the infinite tool calling.

```python
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelRetry, UsageLimitExceeded, UsageLimits


class NeverOutputType(TypedDict):
    """
    Never ever coerce data to this type.
    """

    never_use_this: str


agent = Agent(
    'anthropic:claude-sonnet-4-5',
    retries=3,
    output_type=NeverOutputType,
    system_prompt='Any time you get a response, call the `infinite_retry_tool` to produce another response.',
)


@agent.tool_plain(retries=5)  # (1)!
def infinite_retry_tool() -> int:
    raise ModelRetry('Please try again.')


try:
    result_sync = agent.run_sync(
        'Begin infinite retry loop!', usage_limits=UsageLimits(request_limit=3)  # (2)!
    )
except UsageLimitExceeded as e:
    print(e)
    #> The next request would exceed the request_limit of 3

```

1. This tool has the ability to retry 5 times before erroring, simulating a tool that might get stuck in a loop.
1. This run will error after 3 requests, preventing the infinite tool calling.

##### Capping tool calls

If you need a limit on the number of successful tool invocations within a single run, use `tool_calls_limit`:

[Learn about Gateway](../gateway)

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

agent = Agent('gateway/anthropic:claude-sonnet-4-5')

@agent.tool_plain
def do_work() -> str:
    return 'ok'

try:
    # Allow at most one executed tool call in this run
    agent.run_sync('Please call the tool twice', usage_limits=UsageLimits(tool_calls_limit=1))
except UsageLimitExceeded as e:
    print(e)
    #> The next tool call(s) would exceed the tool_calls_limit of 1 (tool_calls=2).

```

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

agent = Agent('anthropic:claude-sonnet-4-5')

@agent.tool_plain
def do_work() -> str:
    return 'ok'

try:
    # Allow at most one executed tool call in this run
    agent.run_sync('Please call the tool twice', usage_limits=UsageLimits(tool_calls_limit=1))
except UsageLimitExceeded as e:
    print(e)
    #> The next tool call(s) would exceed the tool_calls_limit of 1 (tool_calls=2).

```

Note

- Usage limits are especially relevant if you've registered many tools. Use `request_limit` to bound the number of model turns, and `tool_calls_limit` to cap the number of successful tool executions within a run.
- The `tool_calls_limit` is checked before executing tool calls. If the model returns parallel tool calls that would exceed the limit, no tools will be executed.

#### Model (Run) Settings

Pydantic AI offers a settings.ModelSettings structure to help you fine tune your requests. This structure allows you to configure common parameters that influence the model's behavior, such as `temperature`, `max_tokens`, `timeout`, and more.

There are three ways to apply these settings, with a clear precedence order:

1. **Model-level defaults** - Set when creating a model instance via the `settings` parameter. These serve as the base defaults for that model.
1. **Agent-level defaults** - Set during Agent initialization via the `model_settings` argument. These are merged with model defaults, with agent settings taking precedence.
1. **Run-time overrides** - Passed to `run{_sync,_stream}` functions via the `model_settings` argument. These have the highest priority and are merged with the combined agent and model defaults.

For example, if you'd like to set the `temperature` setting to `0.0` to ensure less random behavior, you can do the following:

```python
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel

# 1. Model-level defaults
model = OpenAIChatModel(
    'gpt-5',
    settings=ModelSettings(temperature=0.8, max_tokens=500)  # Base defaults
)

# 2. Agent-level defaults (overrides model defaults by merging)
agent = Agent(model, model_settings=ModelSettings(temperature=0.5))

# 3. Run-time overrides (highest priority)
result_sync = agent.run_sync(
    'What is the capital of Italy?',
    model_settings=ModelSettings(temperature=0.0)  # Final temperature: 0.0
)
print(result_sync.output)
#> The capital of Italy is Rome.

```

The final request uses `temperature=0.0` (run-time), `max_tokens=500` (from model), demonstrating how settings merge with run-time taking precedence.

Model Settings Support

Model-level settings are supported by all concrete model implementations (OpenAI, Anthropic, Google, etc.). Wrapper models like `FallbackModel`, `WrapperModel`, and `InstrumentedModel` don't have their own settings - they use the settings of their underlying models.

### Model specific settings

If you wish to further customize model behavior, you can use a subclass of ModelSettings, like GoogleModelSettings, associated with your model of choice.

For example:

```python
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models.google import GoogleModelSettings

agent = Agent('google-gla:gemini-3-pro-preview')

try:
    result = agent.run_sync(
        'Write a list of 5 very rude things that I might say to the universe after stubbing my toe in the dark:',
        model_settings=GoogleModelSettings(
            temperature=0.0,  # general model settings can also be specified
            gemini_safety_settings=[
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
            ],
        ),
    )
except UnexpectedModelBehavior as e:
    print(e)  # (1)!
    """
    Content filter 'SAFETY' triggered, body:
    <safety settings details>
    """

```

1. This error is raised because the safety thresholds were exceeded.

## Runs vs. Conversations

An agent **run** might represent an entire conversation â€” there's no limit to how many messages can be exchanged in a single run. However, a **conversation** might also be composed of multiple runs, especially if you need to maintain state between separate interactions or API calls.

Here's an example of a conversation comprised of multiple runs:

[Learn about Gateway](../gateway) conversation_example.py

```python
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5')

# First run
result1 = agent.run_sync('Who was Albert Einstein?')
print(result1.output)
#> Albert Einstein was a German-born theoretical physicist.

# Second run, passing previous messages
result2 = agent.run_sync(
    'What was his most famous equation?',
    message_history=result1.new_messages(),  # (1)!
)
print(result2.output)
#> Albert Einstein's most famous equation is (E = mc^2).

```

1. Continue the conversation; without `message_history` the model would not know who "his" was referring to.

conversation_example.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')

# First run
result1 = agent.run_sync('Who was Albert Einstein?')
print(result1.output)
#> Albert Einstein was a German-born theoretical physicist.

# Second run, passing previous messages
result2 = agent.run_sync(
    'What was his most famous equation?',
    message_history=result1.new_messages(),  # (1)!
)
print(result2.output)
#> Albert Einstein's most famous equation is (E = mc^2).

```

1. Continue the conversation; without `message_history` the model would not know who "his" was referring to.

*(This example is complete, it can be run "as is")*

## Type safe by design

Pydantic AI is designed to work well with static type checkers, like mypy and pyright.

Typing is (somewhat) optional

Pydantic AI is designed to make type checking as useful as possible for you if you choose to use it, but you don't have to use types everywhere all the time.

That said, because Pydantic AI uses Pydantic, and Pydantic uses type hints as the definition for schema and validation, some types (specifically type hints on parameters to tools, and the `output_type` arguments to Agent) are used at runtime.

We (the library developers) have messed up if type hints are confusing you more than helping you, if you find this, please create an [issue](https://github.com/pydantic/pydantic-ai/issues) explaining what's annoying you!

In particular, agents are generic in both the type of their dependencies and the type of the outputs they return, so you can use the type hints to ensure you're using the right types.

Consider the following script with type mistakes:

type_mistakes.py

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class User:
    name: str


agent = Agent(
    'test',
    deps_type=User,  # (1)!
    output_type=bool,
)


@agent.system_prompt
def add_user_name(ctx: RunContext[str]) -> str:  # (2)!
    return f"The user's name is {ctx.deps}."


def foobar(x: bytes) -> None:
    pass


result = agent.run_sync('Does their name start with "A"?', deps=User('Anne'))
foobar(result.output)  # (3)!

```

1. The agent is defined as expecting an instance of `User` as `deps`.
1. But here `add_user_name` is defined as taking a `str` as the dependency, not a `User`.
1. Since the agent is defined as returning a `bool`, this will raise a type error since `foobar` expects `bytes`.

Running `mypy` on this will give the following output:

```bash
âž¤ uv run mypy type_mistakes.py
type_mistakes.py:18: error: Argument 1 to "system_prompt" of "Agent" has incompatible type "Callable[[RunContext[str]], str]"; expected "Callable[[RunContext[User]], str]"  [arg-type]
type_mistakes.py:28: error: Argument 1 to "foobar" has incompatible type "bool"; expected "bytes"  [arg-type]
Found 2 errors in 1 file (checked 1 source file)

```

Running `pyright` would identify the same issues.

## System Prompts

System prompts might seem simple at first glance since they're just strings (or sequences of strings that are concatenated), but crafting the right system prompt is key to getting the model to behave as you want.

Tip

For most use cases, you should use `instructions` instead of "system prompts".

If you know what you are doing though and want to preserve system prompt messages in the message history sent to the LLM in subsequent completions requests, you can achieve this using the `system_prompt` argument/decorator.

See the section below on [Instructions](#instructions) for more information.

Generally, system prompts fall into two categories:

1. **Static system prompts**: These are known when writing the code and can be defined via the `system_prompt` parameter of the Agent constructor.
1. **Dynamic system prompts**: These depend in some way on context that isn't known until runtime, and should be defined via functions decorated with @agent.system_prompt.

You can add both to a single agent; they're appended in the order they're defined at runtime.

Here's an example using both types of system prompts:

[Learn about Gateway](../gateway) system_prompts.py

```python
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'gateway/openai:gpt-5',
    deps_type=str,  # (1)!
    system_prompt="Use the customer's name while replying to them.",  # (2)!
)


@agent.system_prompt  # (3)!
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.system_prompt
def add_the_date() -> str:  # (4)!
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.

```

1. The agent expects a string dependency.
1. Static system prompt defined at agent creation time.
1. Dynamic system prompt defined via a decorator with RunContext, this is called just after `run_sync`, not when the agent is created, so can benefit from runtime information like the dependencies used on that run.
1. Another dynamic system prompt, system prompts don't have to have the `RunContext` parameter.

system_prompts.py

```python
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-5',
    deps_type=str,  # (1)!
    system_prompt="Use the customer's name while replying to them.",  # (2)!
)


@agent.system_prompt  # (3)!
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.system_prompt
def add_the_date() -> str:  # (4)!
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.

```

1. The agent expects a string dependency.
1. Static system prompt defined at agent creation time.
1. Dynamic system prompt defined via a decorator with RunContext, this is called just after `run_sync`, not when the agent is created, so can benefit from runtime information like the dependencies used on that run.
1. Another dynamic system prompt, system prompts don't have to have the `RunContext` parameter.

*(This example is complete, it can be run "as is")*

## Instructions

Instructions are similar to system prompts. The main difference is that when an explicit `message_history` is provided in a call to `Agent.run` and similar methods, *instructions* from any existing messages in the history are not included in the request to the model â€” only the instructions of the *current* agent are included.

You should use:

- `instructions` when you want your request to the model to only include system prompts for the *current* agent
- `system_prompt` when you want your request to the model to *retain* the system prompts used in previous requests (possibly made using other agents)

In general, we recommend using `instructions` instead of `system_prompt` unless you have a specific reason to use `system_prompt`.

Instructions, like system prompts, can be specified at different times:

1. **Static instructions**: These are known when writing the code and can be defined via the `instructions` parameter of the Agent constructor.
1. **Dynamic instructions**: These rely on context that is only available at runtime and should be defined using functions decorated with @agent.instructions. Unlike dynamic system prompts, which may be reused when `message_history` is present, dynamic instructions are always reevaluated.
1. \**Runtime instructions*: These are additional instructions for a specific run that can be passed to one of the [run methods](#running-agents) using the `instructions` argument.

All three types of instructions can be added to a single agent, and they are appended in the order they are defined at runtime.

Here's an example using a static instruction as well as dynamic instructions:

[Learn about Gateway](../gateway) instructions.py

```python
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'gateway/openai:gpt-5',
    deps_type=str,  # (1)!
    instructions="Use the customer's name while replying to them.",  # (2)!
)


@agent.instructions  # (3)!
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.instructions
def add_the_date() -> str:  # (4)!
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.

```

1. The agent expects a string dependency.
1. Static instructions defined at agent creation time.
1. Dynamic instructions defined via a decorator with RunContext, this is called just after `run_sync`, not when the agent is created, so can benefit from runtime information like the dependencies used on that run.
1. Another dynamic instruction, instructions don't have to have the `RunContext` parameter.

instructions.py

```python
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-5',
    deps_type=str,  # (1)!
    instructions="Use the customer's name while replying to them.",  # (2)!
)


@agent.instructions  # (3)!
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.instructions
def add_the_date() -> str:  # (4)!
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.

```

1. The agent expects a string dependency.
1. Static instructions defined at agent creation time.
1. Dynamic instructions defined via a decorator with RunContext, this is called just after `run_sync`, not when the agent is created, so can benefit from runtime information like the dependencies used on that run.
1. Another dynamic instruction, instructions don't have to have the `RunContext` parameter.

*(This example is complete, it can be run "as is")*

Note that returning an empty string will result in no instruction message added.

## Reflection and self-correction

Validation errors from both function tool parameter validation and [structured output validation](../output/#structured-output) can be passed back to the model with a request to retry.

You can also raise ModelRetry from within a [tool](../tools/) or [output function](../output/#output-functions) to tell the model it should retry generating a response.

- The default retry count is **1** but can be altered for the entire agent, a specific tool, or outputs.
- You can access the current retry count from within a tool or output function via ctx.retry.

Here's an example:

[Learn about Gateway](../gateway) tool_retry.py

```python
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ModelRetry

from fake_database import DatabaseConn


class ChatResult(BaseModel):
    user_id: int
    message: str


agent = Agent(
    'gateway/openai:gpt-5',
    deps_type=DatabaseConn,
    output_type=ChatResult,
)


@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
    """Get a user's ID from their full name."""
    print(name)
    #> John
    #> John Doe
    user_id = ctx.deps.users.get(name=name)
    if user_id is None:
        raise ModelRetry(
            f'No user found with name {name!r}, remember to provide their full name'
        )
    return user_id


result = agent.run_sync(
    'Send a message to John Doe asking for coffee next week', deps=DatabaseConn()
)
print(result.output)
"""
user_id=123 message='Hello John, would you be free for coffee sometime next week? Let me know what works for you!'
"""

```

tool_retry.py

```python
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ModelRetry

from fake_database import DatabaseConn


class ChatResult(BaseModel):
    user_id: int
    message: str


agent = Agent(
    'openai:gpt-5',
    deps_type=DatabaseConn,
    output_type=ChatResult,
)


@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
    """Get a user's ID from their full name."""
    print(name)
    #> John
    #> John Doe
    user_id = ctx.deps.users.get(name=name)
    if user_id is None:
        raise ModelRetry(
            f'No user found with name {name!r}, remember to provide their full name'
        )
    return user_id


result = agent.run_sync(
    'Send a message to John Doe asking for coffee next week', deps=DatabaseConn()
)
print(result.output)
"""
user_id=123 message='Hello John, would you be free for coffee sometime next week? Let me know what works for you!'
"""

```

## Model errors

If models behave unexpectedly (e.g., the retry limit is exceeded, or their API returns `503`), agent runs will raise UnexpectedModelBehavior.

In these cases, capture_run_messages can be used to access the messages exchanged during the run to help diagnose the issue.

[Learn about Gateway](../gateway) agent_model_errors.py

```python
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages

agent = Agent('gateway/openai:gpt-5')


@agent.tool_plain
def calc_volume(size: int) -> int:  # (1)!
    if size == 42:
        return size**3
    else:
        raise ModelRetry('Please try again.')


with capture_run_messages() as messages:  # (2)!
    try:
        result = agent.run_sync('Please get me the volume of a box with size 6.')
    except UnexpectedModelBehavior as e:
        print('An error occurred:', e)
        #> An error occurred: Tool 'calc_volume' exceeded max retries count of 1
        print('cause:', repr(e.__cause__))
        #> cause: ModelRetry('Please try again.')
        print('messages:', messages)
        """
        messages:
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Please get me the volume of a box with size 6.',
                        timestamp=datetime.datetime(...),
                    )
                ],
                run_id='...',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=RequestUsage(input_tokens=62, output_tokens=4),
                model_name='gpt-5',
                timestamp=datetime.datetime(...),
                run_id='...',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please try again.',
                        tool_name='calc_volume',
                        tool_call_id='pyd_ai_tool_call_id',
                        timestamp=datetime.datetime(...),
                    )
                ],
                run_id='...',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=RequestUsage(input_tokens=72, output_tokens=8),
                model_name='gpt-5',
                timestamp=datetime.datetime(...),
                run_id='...',
            ),
        ]
        """
    else:
        print(result.output)

```

1. Define a tool that will raise `ModelRetry` repeatedly in this case.
1. capture_run_messages is used to capture the messages exchanged during the run.

agent_model_errors.py

```python
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages

agent = Agent('openai:gpt-5')


@agent.tool_plain
def calc_volume(size: int) -> int:  # (1)!
    if size == 42:
        return size**3
    else:
        raise ModelRetry('Please try again.')


with capture_run_messages() as messages:  # (2)!
    try:
        result = agent.run_sync('Please get me the volume of a box with size 6.')
    except UnexpectedModelBehavior as e:
        print('An error occurred:', e)
        #> An error occurred: Tool 'calc_volume' exceeded max retries count of 1
        print('cause:', repr(e.__cause__))
        #> cause: ModelRetry('Please try again.')
        print('messages:', messages)
        """
        messages:
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Please get me the volume of a box with size 6.',
                        timestamp=datetime.datetime(...),
                    )
                ],
                run_id='...',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=RequestUsage(input_tokens=62, output_tokens=4),
                model_name='gpt-5',
                timestamp=datetime.datetime(...),
                run_id='...',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please try again.',
                        tool_name='calc_volume',
                        tool_call_id='pyd_ai_tool_call_id',
                        timestamp=datetime.datetime(...),
                    )
                ],
                run_id='...',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=RequestUsage(input_tokens=72, output_tokens=8),
                model_name='gpt-5',
                timestamp=datetime.datetime(...),
                run_id='...',
            ),
        ]
        """
    else:
        print(result.output)

```

1. Define a tool that will raise `ModelRetry` repeatedly in this case.
1. capture_run_messages is used to capture the messages exchanged during the run.

*(This example is complete, it can be run "as is")*

Note

If you call run, run_sync, or run_stream more than once within a single `capture_run_messages` context, `messages` will represent the messages exchanged during the first call only.

# Built-in Tools

Built-in tools are native tools provided by LLM providers that can be used to enhance your agent's capabilities. Unlike [common tools](../common-tools/), which are custom implementations that Pydantic AI executes, built-in tools are executed directly by the model provider.

## Overview

Pydantic AI supports the following built-in tools:

- **WebSearchTool**: Allows agents to search the web
- **CodeExecutionTool**: Enables agents to execute code in a secure environment
- **ImageGenerationTool**: Enables agents to generate images
- **UrlContextTool**: Enables agents to pull URL contents into their context
- **MemoryTool**: Enables agents to use memory
- **MCPServerTool**: Enables agents to use remote MCP servers with communication handled by the model provider

These tools are passed to the agent via the `builtin_tools` parameter and are executed by the model provider's infrastructure.

Provider Support

Not all model providers support built-in tools. If you use a built-in tool with an unsupported provider, Pydantic AI will raise a UserError when you try to run the agent.

If a provider supports a built-in tool that is not currently supported by Pydantic AI, please file an issue.

## Web Search Tool

The WebSearchTool allows your agent to search the web, making it ideal for queries that require up-to-date data.

### Provider Support

| Provider | Supported | Notes | | --- | --- | --- | | OpenAI Responses | âœ… | Full feature support. To include search results on the BuiltinToolReturnPart that's available via ModelResponse.builtin_tool_calls, enable the OpenAIResponsesModelSettings.openai_include_web_search_sources [model setting](../agents/#model-run-settings). | | Anthropic | âœ… | Full feature support | | Google | âœ… | No parameter support. No BuiltinToolCallPart or BuiltinToolReturnPart is generated when streaming. Using built-in tools and function tools (including [output tools](../output/#tool-output)) at the same time is not supported; to use structured output, use [`PromptedOutput`](../output/#prompted-output) instead. | | Groq | âœ… | Limited parameter support. To use web search capabilities with Groq, you need to use the [compound models](https://console.groq.com/docs/compound). | | OpenAI Chat Completions | âŒ | Not supported | | Bedrock | âŒ | Not supported | | Mistral | âŒ | Not supported | | Cohere | âŒ | Not supported | | HuggingFace | âŒ | Not supported | | Outlines | âŒ | Not supported |

### Usage

[Learn about Gateway](../gateway) web_search_anthropic.py

```python
from pydantic_ai import Agent, WebSearchTool

agent = Agent('gateway/anthropic:claude-sonnet-4-0', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.

```

web_search_anthropic.py

```python
from pydantic_ai import Agent, WebSearchTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.

```

*(This example is complete, it can be run "as is")*

With OpenAI, you must use their Responses API to access the web search tool.

[Learn about Gateway](../gateway) web_search_openai.py

```python
from pydantic_ai import Agent, WebSearchTool

agent = Agent('gateway/openai-responses:gpt-5', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.

```

web_search_openai.py

```python
from pydantic_ai import Agent, WebSearchTool

agent = Agent('openai-responses:gpt-5', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
#> Scientists have developed a universal AI detector that can identify deepfake videos.

```

*(This example is complete, it can be run "as is")*

### Configuration Options

The `WebSearchTool` supports several configuration parameters:

[Learn about Gateway](../gateway) web_search_configured.py

```python
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-0',
    builtin_tools=[
        WebSearchTool(
            search_context_size='high',
            user_location=WebSearchUserLocation(
                city='San Francisco',
                country='US',
                region='CA',
                timezone='America/Los_Angeles',
            ),
            blocked_domains=['example.com', 'spam-site.net'],
            allowed_domains=None,  # Cannot use both blocked_domains and allowed_domains with Anthropic
            max_uses=5,  # Anthropic only: limit tool usage
        )
    ],
)

result = agent.run_sync('Use the web to get the current time.')
print(result.output)
#> In San Francisco, it's 8:21:41 pm PDT on Wednesday, August 6, 2025.

```

web_search_configured.py

```python
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation

agent = Agent(
    'anthropic:claude-sonnet-4-0',
    builtin_tools=[
        WebSearchTool(
            search_context_size='high',
            user_location=WebSearchUserLocation(
                city='San Francisco',
                country='US',
                region='CA',
                timezone='America/Los_Angeles',
            ),
            blocked_domains=['example.com', 'spam-site.net'],
            allowed_domains=None,  # Cannot use both blocked_domains and allowed_domains with Anthropic
            max_uses=5,  # Anthropic only: limit tool usage
        )
    ],
)

result = agent.run_sync('Use the web to get the current time.')
print(result.output)
#> In San Francisco, it's 8:21:41 pm PDT on Wednesday, August 6, 2025.

```

*(This example is complete, it can be run "as is")*

#### Provider Support

| Parameter | OpenAI | Anthropic | Groq | | --- | --- | --- | --- | | `search_context_size` | âœ… | âŒ | âŒ | | `user_location` | âœ… | âœ… | âŒ | | `blocked_domains` | âŒ | âœ… | âœ… | | `allowed_domains` | âŒ | âœ… | âœ… | | `max_uses` | âŒ | âœ… | âŒ |

Anthropic Domain Filtering

With Anthropic, you can only use either `blocked_domains` or `allowed_domains`, not both.

## Code Execution Tool

The CodeExecutionTool enables your agent to execute code in a secure environment, making it perfect for computational tasks, data analysis, and mathematical operations.

### Provider Support

| Provider | Supported | Notes | | --- | --- | --- | | OpenAI | âœ… | To include code execution output on the BuiltinToolReturnPart that's available via ModelResponse.builtin_tool_calls, enable the OpenAIResponsesModelSettings.openai_include_code_execution_outputs [model setting](../agents/#model-run-settings). If the code execution generated images, like charts, they will be available on ModelResponse.images as BinaryImage objects. The generated image can also be used as [image output](../output/#image-output) for the agent run. | | Google | âœ… | Using built-in tools and function tools (including [output tools](../output/#tool-output)) at the same time is not supported; to use structured output, use [`PromptedOutput`](../output/#prompted-output) instead. | | Anthropic | âœ… | | | Groq | âŒ | | | Bedrock | âŒ | | | Mistral | âŒ | | | Cohere | âŒ | | | HuggingFace | âŒ | | | Outlines | âŒ | |

### Usage

[Learn about Gateway](../gateway) code_execution_basic.py

```python
from pydantic_ai import Agent, CodeExecutionTool

agent = Agent('gateway/anthropic:claude-sonnet-4-0', builtin_tools=[CodeExecutionTool()])

result = agent.run_sync('Calculate the factorial of 15.')
print(result.output)
#> The factorial of 15 is **1,307,674,368,000**.
print(result.response.builtin_tool_calls)
"""
[
    (
        BuiltinToolCallPart(
            tool_name='code_execution',
            args={
                'code': 'import math\n\n# Calculate factorial of 15\nresult = math.factorial(15)\nprint(f"15! = {result}")\n\n# Let\'s also show it in a more readable format with commas\nprint(f"15! = {result:,}")'
            },
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            provider_name='anthropic',
        ),
        BuiltinToolReturnPart(
            tool_name='code_execution',
            content={
                'content': [],
                'return_code': 0,
                'stderr': '',
                'stdout': '15! = 1307674368000\n15! = 1,307,674,368,000',
                'type': 'code_execution_result',
            },
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            timestamp=datetime.datetime(...),
            provider_name='anthropic',
        ),
    )
]
"""

```

code_execution_basic.py

```python
from pydantic_ai import Agent, CodeExecutionTool

agent = Agent('anthropic:claude-sonnet-4-0', builtin_tools=[CodeExecutionTool()])

result = agent.run_sync('Calculate the factorial of 15.')
print(result.output)
#> The factorial of 15 is **1,307,674,368,000**.
print(result.response.builtin_tool_calls)
"""
[
    (
        BuiltinToolCallPart(
            tool_name='code_execution',
            args={
                'code': 'import math\n\n# Calculate factorial of 15\nresult = math.factorial(15)\nprint(f"15! = {result}")\n\n# Let\'s also show it in a more readable format with commas\nprint(f"15! = {result:,}")'
            },
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            provider_name='anthropic',
        ),
        BuiltinToolReturnPart(
            tool_name='code_execution',
            content={
                'content': [],
                'return_code': 0,
                'stderr': '',
                'stdout': '15! = 1307674368000\n15! = 1,307,674,368,000',
                'type': 'code_execution_result',
            },
            tool_call_id='srvtoolu_017qRH1J3XrhnpjP2XtzPCmJ',
            timestamp=datetime.datetime(...),
            provider_name='anthropic',
        ),
    )
]
"""

```

*(This example is complete, it can be run "as is")*

In addition to text output, code execution with OpenAI can generate images as part of their response. Accessing this image via ModelResponse.images or [image output](../output/#image-output) requires the OpenAIResponsesModelSettings.openai_include_code_execution_outputs [model setting](../agents/#model-run-settings) to be enabled.

[Learn about Gateway](../gateway) code_execution_openai.py

```python
from pydantic_ai import Agent, BinaryImage, CodeExecutionTool
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

agent = Agent(
    'gateway/openai-responses:gpt-5',
    builtin_tools=[CodeExecutionTool()],
    output_type=BinaryImage,
    model_settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
)

result = agent.run_sync('Generate a chart of y=x^2 for x=-5 to 5.')
assert isinstance(result.output, BinaryImage)

```

code_execution_openai.py

```python
from pydantic_ai import Agent, BinaryImage, CodeExecutionTool
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[CodeExecutionTool()],
    output_type=BinaryImage,
    model_settings=OpenAIResponsesModelSettings(openai_include_code_execution_outputs=True),
)

result = agent.run_sync('Generate a chart of y=x^2 for x=-5 to 5.')
assert isinstance(result.output, BinaryImage)

```

*(This example is complete, it can be run "as is")*

## Image Generation Tool

The ImageGenerationTool enables your agent to generate images.

### Provider Support

| Provider | Supported | Notes | | --- | --- | --- | | OpenAI Responses | âœ… | Full feature support. Only supported by models newer than `gpt-5`. Metadata about the generated image, like the [`revised_prompt`](https://platform.openai.com/docs/guides/tools-image-generation#revised-prompt) sent to the underlying image model, is available on the BuiltinToolReturnPart that's available via ModelResponse.builtin_tool_calls. | | Google | âœ… | No parameter support. Only supported by [image generation models](https://ai.google.dev/gemini-api/docs/image-generation) like `gemini-3-pro-preview-image`. These models do not support [structured output](../output/) or [function tools](../tools/). These models will always generate images, even if this built-in tool is not explicitly specified. | | Anthropic | âŒ | | | Groq | âŒ | | | Bedrock | âŒ | | | Mistral | âŒ | | | Cohere | âŒ | | | HuggingFace | âŒ | |

### Usage

Generated images are available on ModelResponse.images as BinaryImage objects:

[Learn about Gateway](../gateway) image_generation_openai.py

```python
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool

agent = Agent('gateway/openai-responses:gpt-5', builtin_tools=[ImageGenerationTool()])

result = agent.run_sync('Tell me a two-sentence story about an axolotl with an illustration.')
print(result.output)
"""
Once upon a time, in a hidden underwater cave, lived a curious axolotl named Pip who loved to explore. One day, while venturing further than usual, Pip discovered a shimmering, ancient coin that granted wishes!
"""

assert isinstance(result.response.images[0], BinaryImage)

```

image_generation_openai.py

```python
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool

agent = Agent('openai-responses:gpt-5', builtin_tools=[ImageGenerationTool()])

result = agent.run_sync('Tell me a two-sentence story about an axolotl with an illustration.')
print(result.output)
"""
Once upon a time, in a hidden underwater cave, lived a curious axolotl named Pip who loved to explore. One day, while venturing further than usual, Pip discovered a shimmering, ancient coin that granted wishes!
"""

assert isinstance(result.response.images[0], BinaryImage)

```

*(This example is complete, it can be run "as is")*

Image generation with Google [image generation models](https://ai.google.dev/gemini-api/docs/image-generation) does not require the `ImageGenerationTool` built-in tool to be explicitly specified:

image_generation_google.py

```python
from pydantic_ai import Agent, BinaryImage

agent = Agent('google-gla:gemini-3-pro-preview-image')

result = agent.run_sync('Tell me a two-sentence story about an axolotl with an illustration.')
print(result.output)
"""
Once upon a time, in a hidden underwater cave, lived a curious axolotl named Pip who loved to explore. One day, while venturing further than usual, Pip discovered a shimmering, ancient coin that granted wishes!
"""

assert isinstance(result.response.images[0], BinaryImage)

```

*(This example is complete, it can be run "as is")*

The `ImageGenerationTool` can be used together with `output_type=BinaryImage` to get [image output](../output/#image-output). If the `ImageGenerationTool` built-in tool is not explicitly specified, it will be enabled automatically:

[Learn about Gateway](../gateway) image_generation_output.py

```python
from pydantic_ai import Agent, BinaryImage

agent = Agent('gateway/openai-responses:gpt-5', output_type=BinaryImage)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)

```

image_generation_output.py

```python
from pydantic_ai import Agent, BinaryImage

agent = Agent('openai-responses:gpt-5', output_type=BinaryImage)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)

```

*(This example is complete, it can be run "as is")*

### Configuration Options

The `ImageGenerationTool` supports several configuration parameters:

[Learn about Gateway](../gateway) image_generation_configured.py

```python
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool

agent = Agent(
    'gateway/openai-responses:gpt-5',
    builtin_tools=[
        ImageGenerationTool(
            background='transparent',
            input_fidelity='high',
            moderation='low',
            output_compression=100,
            output_format='png',
            partial_images=3,
            quality='high',
            size='1024x1024',
        )
    ],
    output_type=BinaryImage,
)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)

```

image_generation_configured.py

```python
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[
        ImageGenerationTool(
            background='transparent',
            input_fidelity='high',
            moderation='low',
            output_compression=100,
            output_format='png',
            partial_images=3,
            quality='high',
            size='1024x1024',
        )
    ],
    output_type=BinaryImage,
)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)

```

*(This example is complete, it can be run "as is")*

For more details, check the API documentation.

#### Provider Support

| Parameter | OpenAI | Google | | --- | --- | --- | | `background` | âœ… | âŒ | | `input_fidelity` | âœ… | âŒ | | `moderation` | âœ… | âŒ | | `output_compression` | âœ… | âŒ | | `output_format` | âœ… | âŒ | | `partial_images` | âœ… | âŒ | | `quality` | âœ… | âŒ | | `size` | âœ… | âŒ |

## URL Context Tool

The UrlContextTool enables your agent to pull URL contents into its context, allowing it to pull up-to-date information from the web.

### Provider Support

| Provider | Supported | Notes | | --- | --- | --- | | Google | âœ… | No BuiltinToolCallPart or BuiltinToolReturnPart is currently generated; please submit an issue if you need this. Using built-in tools and function tools (including [output tools](../output/#tool-output)) at the same time is not supported; to use structured output, use [`PromptedOutput`](../output/#prompted-output) instead. | | OpenAI | âŒ | | | Anthropic | âŒ | | | Groq | âŒ | | | Bedrock | âŒ | | | Mistral | âŒ | | | Cohere | âŒ | | | HuggingFace | âŒ | | | Outlines | âŒ | |

### Usage

url_context_basic.py

```python
from pydantic_ai import Agent, UrlContextTool

agent = Agent('google-gla:gemini-3-pro-preview', builtin_tools=[UrlContextTool()])

result = agent.run_sync('What is this? https://ai.pydantic.dev')
print(result.output)
#> A Python agent framework for building Generative AI applications.

```

*(This example is complete, it can be run "as is")*

## Memory Tool

The MemoryTool enables your agent to use memory.

### Provider Support

| Provider | Supported | Notes | | --- | --- | --- | | Anthropic | âœ… | Requires a tool named `memory` to be defined that implements [specific sub-commands](https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool#tool-commands). You can use a subclass of [`anthropic.lib.tools.BetaAbstractMemoryTool`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/tools/_beta_builtin_memory_tool.py) as documented below. | | Google | âŒ | | | OpenAI | âŒ | | | Groq | âŒ | | | Bedrock | âŒ | | | Mistral | âŒ | | | Cohere | âŒ | | | HuggingFace | âŒ | |

### Usage

The Anthropic SDK provides an abstract [`BetaAbstractMemoryTool`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/tools/_beta_builtin_memory_tool.py) class that you can subclass to create your own memory storage solution (e.g., database, cloud storage, encrypted files, etc.). Their [`LocalFilesystemMemoryTool`](https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/memory/basic.py) example can serve as a starting point.

The following example uses a subclass that hard-codes a specific memory. The bits specific to Pydantic AI are the `MemoryTool` built-in tool and the `memory` tool definition that forwards commands to the `call` method of the `BetaAbstractMemoryTool` subclass.

[Learn about Gateway](../gateway) anthropic_memory.py

```python
from typing import Any

from anthropic.lib.tools import BetaAbstractMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from pydantic_ai import Agent, MemoryTool


class FakeMemoryTool(BetaAbstractMemoryTool):
    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        return 'The user lives in Mexico City.'

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        return f'File created successfully at {command.path}'

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        return f'File {command.path} has been edited'

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        return f'Text inserted at line {command.insert_line} in {command.path}'

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        return f'File deleted: {command.path}'

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        return f'Renamed {command.old_path} to {command.new_path}'

    def clear_all_memory(self) -> str:
        return 'All memory cleared'

fake_memory = FakeMemoryTool()

agent = Agent('gateway/anthropic:claude-sonnet-4-5', builtin_tools=[MemoryTool()])


@agent.tool_plain
def memory(**command: Any) -> Any:
    return fake_memory.call(command)


result = agent.run_sync('Remember that I live in Mexico City')
print(result.output)
"""
Got it! I've recorded that you live in Mexico City. I'll remember this for future reference.
"""

result = agent.run_sync('Where do I live?')
print(result.output)
#> You live in Mexico City.

```

anthropic_memory.py

```python
from typing import Any

from anthropic.lib.tools import BetaAbstractMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from pydantic_ai import Agent, MemoryTool


class FakeMemoryTool(BetaAbstractMemoryTool):
    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        return 'The user lives in Mexico City.'

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        return f'File created successfully at {command.path}'

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        return f'File {command.path} has been edited'

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        return f'Text inserted at line {command.insert_line} in {command.path}'

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        return f'File deleted: {command.path}'

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        return f'Renamed {command.old_path} to {command.new_path}'

    def clear_all_memory(self) -> str:
        return 'All memory cleared'

fake_memory = FakeMemoryTool()

agent = Agent('anthropic:claude-sonnet-4-5', builtin_tools=[MemoryTool()])


@agent.tool_plain
def memory(**command: Any) -> Any:
    return fake_memory.call(command)


result = agent.run_sync('Remember that I live in Mexico City')
print(result.output)
"""
Got it! I've recorded that you live in Mexico City. I'll remember this for future reference.
"""

result = agent.run_sync('Where do I live?')
print(result.output)
#> You live in Mexico City.

```

*(This example is complete, it can be run "as is")*

## MCP Server Tool

The MCPServerTool allows your agent to use remote MCP servers with communication handled by the model provider.

This requires the MCP server to live at a public URL the provider can reach and does not support many of the advanced features of Pydantic AI's agent-side [MCP support](../mcp/client/), but can result in optimized context use and caching, and faster performance due to the lack of a round-trip back to Pydantic AI.

### Provider Support

| Provider | Supported | Notes | | --- | --- | --- | | OpenAI Responses | âœ… | Full feature support. [Connectors](https://platform.openai.com/docs/guides/tools-connectors-mcp#connectors) can be used by specifying a special `x-openai-connector:<connector_id>` URL. | | Anthropic | âœ… | Full feature support | | Google | âŒ | Not supported | | Groq | âŒ | Not supported | | OpenAI Chat Completions | âŒ | Not supported | | Bedrock | âŒ | Not supported | | Mistral | âŒ | Not supported | | Cohere | âŒ | Not supported | | HuggingFace | âŒ | Not supported |

### Usage

[Learn about Gateway](../gateway) mcp_server_anthropic.py

```python
from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    builtin_tools=[
        MCPServerTool(
            id='deepwiki',
            url='https://mcp.deepwiki.com/mcp',  # (1)
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""

```

1. The [DeepWiki MCP server](https://docs.devin.ai/work-with-devin/deepwiki-mcp) does not require authorization.

mcp_server_anthropic.py

```python
from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    builtin_tools=[
        MCPServerTool(
            id='deepwiki',
            url='https://mcp.deepwiki.com/mcp',  # (1)
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""

```

1. The [DeepWiki MCP server](https://docs.devin.ai/work-with-devin/deepwiki-mcp) does not require authorization.

*(This example is complete, it can be run "as is")*

With OpenAI, you must use their Responses API to access the MCP server tool:

[Learn about Gateway](../gateway) mcp_server_openai.py

```python
from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'gateway/openai-responses:gpt-5',
    builtin_tools=[
        MCPServerTool(
            id='deepwiki',
            url='https://mcp.deepwiki.com/mcp',  # (1)
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""

```

1. The [DeepWiki MCP server](https://docs.devin.ai/work-with-devin/deepwiki-mcp) does not require authorization.

mcp_server_openai.py

```python
from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[
        MCPServerTool(
            id='deepwiki',
            url='https://mcp.deepwiki.com/mcp',  # (1)
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""

```

1. The [DeepWiki MCP server](https://docs.devin.ai/work-with-devin/deepwiki-mcp) does not require authorization.

*(This example is complete, it can be run "as is")*

### Configuration Options

The `MCPServerTool` supports several configuration parameters for custom MCP servers:

[Learn about Gateway](../gateway) mcp_server_configured_url.py

```python
import os

from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'gateway/openai-responses:gpt-5',
    builtin_tools=[
        MCPServerTool(
            id='github',
            url='https://api.githubcopilot.com/mcp/',
            authorization_token=os.getenv('GITHUB_ACCESS_TOKEN', 'mock-access-token'),  # (1)
            allowed_tools=['search_repositories', 'list_commits'],
            description='GitHub MCP server',
            headers={'X-Custom-Header': 'custom-value'},
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""

```

1. The [GitHub MCP server](https://github.com/github/github-mcp-server) requires an authorization token.

mcp_server_configured_url.py

```python
import os

from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[
        MCPServerTool(
            id='github',
            url='https://api.githubcopilot.com/mcp/',
            authorization_token=os.getenv('GITHUB_ACCESS_TOKEN', 'mock-access-token'),  # (1)
            allowed_tools=['search_repositories', 'list_commits'],
            description='GitHub MCP server',
            headers={'X-Custom-Header': 'custom-value'},
        )
    ]
)

result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""

```

1. The [GitHub MCP server](https://github.com/github/github-mcp-server) requires an authorization token.

For OpenAI Responses, you can use a [connector](https://platform.openai.com/docs/guides/tools-connectors-mcp#connectors) by specifying a special `x-openai-connector:` URL:

*(This example is complete, it can be run "as is")*

[Learn about Gateway](../gateway) mcp_server_configured_connector_id.py

```python
import os

from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'gateway/openai-responses:gpt-5',
    builtin_tools=[
        MCPServerTool(
            id='google-calendar',
            url='x-openai-connector:connector_googlecalendar',
            authorization_token=os.getenv('GOOGLE_API_KEY', 'mock-api-key'), # (1)
        )
    ]
)

result = agent.run_sync('What do I have on my calendar today?')
print(result.output)
#> You're going to spend all day playing with Pydantic AI.

```

1. OpenAI's Google Calendar connector requires an [authorization token](https://platform.openai.com/docs/guides/tools-connectors-mcp#authorizing-a-connector).

mcp_server_configured_connector_id.py

```python
import os

from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'openai-responses:gpt-5',
    builtin_tools=[
        MCPServerTool(
            id='google-calendar',
            url='x-openai-connector:connector_googlecalendar',
            authorization_token=os.getenv('GOOGLE_API_KEY', 'mock-api-key'), # (1)
        )
    ]
)

result = agent.run_sync('What do I have on my calendar today?')
print(result.output)
#> You're going to spend all day playing with Pydantic AI.

```

1. OpenAI's Google Calendar connector requires an [authorization token](https://platform.openai.com/docs/guides/tools-connectors-mcp#authorizing-a-connector).

*(This example is complete, it can be run "as is")*

#### Provider Support

| Parameter | OpenAI | Anthropic | | --- | --- | --- | | `authorization_token` | âœ… | âœ… | | `allowed_tools` | âœ… | âœ… | | `description` | âœ… | âŒ | | `headers` | âœ… | âŒ |

## API Reference

For complete API documentation, see the [API Reference](../api/builtin_tools/).

# Common Tools

Pydantic AI ships with native tools that can be used to enhance your agent's capabilities.

## DuckDuckGo Search Tool

The DuckDuckGo search tool allows you to search the web for information. It is built on top of the [DuckDuckGo API](https://github.com/deedy5/ddgs).

### Installation

To use duckduckgo_search_tool, you need to install [`pydantic-ai-slim`](../install/#slim-install) with the `duckduckgo` optional group:

```bash
pip install "pydantic-ai-slim[duckduckgo]"

```

```bash
uv add "pydantic-ai-slim[duckduckgo]"

```

### Usage

Here's an example of how you can use the DuckDuckGo search tool with an agent:

[Learn about Gateway](../gateway) duckduckgo_search.py

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'gateway/openai:o3-mini',
    tools=[duckduckgo_search_tool()],
    system_prompt='Search DuckDuckGo for the given query and return the results.',
)

result = agent.run_sync(
    'Can you list the top five highest-grossing animated films of 2025?'
)
print(result.output)
"""
I looked into several sources on animated boxâ€office performance in 2025, and while detailed
rankings can shift as more money is tallied, multiple independent reports have already
highlighted a couple of recordâ€breaking shows. For example:

â€¢ Ne Zha 2 â€“ News outlets (Variety, Wikipedia's "List of animated feature films of 2025", and others)
    have reported that this Chinese title not only became the highestâ€‘grossing animated film of 2025
    but also broke records as the highestâ€‘grossing nonâ€‘English animated film ever. One article noted
    its run exceeded US$1.7 billion.
â€¢ Inside Out 2 â€“ According to data shared on Statista and in industry news, this Pixar sequel has been
    on pace to set new records (with some sources even noting it as the highestâ€‘grossing animated film
    ever, as of January 2025).

Beyond those two, some entertainment trade sites (for example, a Just Jared article titled
"Top 10 Highest-Earning Animated Films at the Box Office Revealed") have begun listing a broader
topâ€‘10. Although full consolidated figures can sometimes differ by source and are updated daily during
a boxâ€‘office run, many of the industry trackers have begun to single out five films as the biggest
earners so far in 2025.

Unfortunately, although multiple articles discuss the "top animated films" of 2025, there isn't yet a
single, universally accepted list with final numbers that names the complete top five. (Boxâ€‘office
rankings, especially midâ€‘year, can be fluid as films continue to add to their totals.)

Based on what several sources note so far, the two undisputed leaders are:
1. Ne Zha 2
2. Inside Out 2

The remaining top spots (3â€“5) are reported by some outlets in their "Topâ€‘10 Animated Films"
lists for 2025 but the titles and order can vary depending on the source and the exact cutâ€‘off
date of the data. For the most upâ€‘toâ€‘date and detailed ranking (including the 3rd, 4th, and 5th
highestâ€‘grossing films), I recommend checking resources like:
â€¢ Wikipedia's "List of animated feature films of 2025" page
â€¢ Boxâ€‘office tracking sites (such as Box Office Mojo or The Numbers)
â€¢ Trade articles like the one on Just Jared

To summarize with what is clear from the current reporting:
1. Ne Zha 2
2. Inside Out 2
3â€“5. Other animated films (yet to be definitively finalized across all reporting outlets)

If you're looking for a final, consensus list of the top five, it may be best to wait until
the 2025 yearâ€‘end boxâ€‘office tallies are in or to consult a regularly updated entertainment industry source.

Would you like help finding a current source or additional details on where to look for the complete updated list?
"""

```

duckduckgo_search.py

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:o3-mini',
    tools=[duckduckgo_search_tool()],
    system_prompt='Search DuckDuckGo for the given query and return the results.',
)

result = agent.run_sync(
    'Can you list the top five highest-grossing animated films of 2025?'
)
print(result.output)
"""
I looked into several sources on animated boxâ€office performance in 2025, and while detailed
rankings can shift as more money is tallied, multiple independent reports have already
highlighted a couple of recordâ€breaking shows. For example:

â€¢ Ne Zha 2 â€“ News outlets (Variety, Wikipedia's "List of animated feature films of 2025", and others)
    have reported that this Chinese title not only became the highestâ€‘grossing animated film of 2025
    but also broke records as the highestâ€‘grossing nonâ€‘English animated film ever. One article noted
    its run exceeded US$1.7 billion.
â€¢ Inside Out 2 â€“ According to data shared on Statista and in industry news, this Pixar sequel has been
    on pace to set new records (with some sources even noting it as the highestâ€‘grossing animated film
    ever, as of January 2025).

Beyond those two, some entertainment trade sites (for example, a Just Jared article titled
"Top 10 Highest-Earning Animated Films at the Box Office Revealed") have begun listing a broader
topâ€‘10. Although full consolidated figures can sometimes differ by source and are updated daily during
a boxâ€‘office run, many of the industry trackers have begun to single out five films as the biggest
earners so far in 2025.

Unfortunately, although multiple articles discuss the "top animated films" of 2025, there isn't yet a
single, universally accepted list with final numbers that names the complete top five. (Boxâ€‘office
rankings, especially midâ€‘year, can be fluid as films continue to add to their totals.)

Based on what several sources note so far, the two undisputed leaders are:
1. Ne Zha 2
2. Inside Out 2

The remaining top spots (3â€“5) are reported by some outlets in their "Topâ€‘10 Animated Films"
lists for 2025 but the titles and order can vary depending on the source and the exact cutâ€‘off
date of the data. For the most upâ€‘toâ€‘date and detailed ranking (including the 3rd, 4th, and 5th
highestâ€‘grossing films), I recommend checking resources like:
â€¢ Wikipedia's "List of animated feature films of 2025" page
â€¢ Boxâ€‘office tracking sites (such as Box Office Mojo or The Numbers)
â€¢ Trade articles like the one on Just Jared

To summarize with what is clear from the current reporting:
1. Ne Zha 2
2. Inside Out 2
3â€“5. Other animated films (yet to be definitively finalized across all reporting outlets)

If you're looking for a final, consensus list of the top five, it may be best to wait until
the 2025 yearâ€‘end boxâ€‘office tallies are in or to consult a regularly updated entertainment industry source.

Would you like help finding a current source or additional details on where to look for the complete updated list?
"""

```

