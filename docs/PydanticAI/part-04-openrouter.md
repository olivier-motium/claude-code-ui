<!-- Source: _complete-doc.md -->

# OpenRouter

## Install

To use `OpenRouterModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openrouter` optional group:

```bash
pip install "pydantic-ai-slim[openrouter]"

```

```bash
uv add "pydantic-ai-slim[openrouter]"

```

## Configuration

To use [OpenRouter](https://openrouter.ai), first create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).

You can set the `OPENROUTER_API_KEY` environment variable and use OpenRouterProvider by name:

```python
from pydantic_ai import Agent

agent = Agent('openrouter:anthropic/claude-3.5-sonnet')
...

```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

model = OpenRouterModel(
    'anthropic/claude-3.5-sonnet',
    provider=OpenRouterProvider(api_key='your-openrouter-api-key'),
)
agent = Agent(model)
...

```

## App Attribution

OpenRouter has an [app attribution](https://openrouter.ai/docs/app-attribution) feature to track your application in their public ranking and analytics.

You can pass in an `app_url` and `app_title` when initializing the provider to enable app attribution.

```python
from pydantic_ai.providers.openrouter import OpenRouterProvider

provider=OpenRouterProvider(
    api_key='your-openrouter-api-key',
    app_url='https://your-app.com',
    app_title='Your App',
),
...

```

# Outlines

## Install

As Outlines is a library allowing you to run models from various different providers, it does not include the necessary dependencies for any provider by default. As a result, to use the OutlinesModel, you must install `pydantic-ai-slim` with an optional group composed of outlines, a dash, and the name of the specific model provider you would use through Outlines. For instance:

```bash
pip install "pydantic-ai-slim[outlines-transformers]"

```

```bash
uv add "pydantic-ai-slim[outlines-transformers]"

```

Or

```bash
pip install "pydantic-ai-slim[outlines-mlxlm]"

```

```bash
uv add "pydantic-ai-slim[outlines-mlxlm]"

```

There are 5 optional groups for the 5 model providers supported through Outlines:

- `outlines-transformers`
- `outlines-llamacpp`
- `outlines-mlxlm`
- `outlines-sglang`
- `outlines-vllm-offline`

## Model Initialization

As Outlines is not an inference provider, but instead a library allowing you to run both local and API-based models, instantiating the model is a bit different from the other models available on Pydantic AI.

To initialize the `OutlinesModel` through the `__init__` method, the first argument you must provide has to be an `outlines.Model` or an `outlines.AsyncModel` instance.

For instance:

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

from pydantic_ai.models.outlines import OutlinesModel

outlines_model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained('erwanf/gpt2-mini'),
    AutoTokenizer.from_pretrained('erwanf/gpt2-mini')
)
model = OutlinesModel(outlines_model)

```

As you already providing an Outlines model instance, there is no need to provide an `OutlinesProvider` yourself.

### Model Loading Methods

Alternatively, you can use some `OutlinesModel` class methods made to load a specific type of Outlines model directly. To do so, you must provide as arguments the same arguments you would have given to the associated Outlines model loading function (except in the case of SGLang).

There are methods for the 5 Outlines models that are officially supported in the integration into Pydantic AI:

- from_transformers
- from_llamacpp
- from_mlxlm
- from_sglang
- from_vllm_offline

#### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_transformers(
    AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct'),
    AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
)

```

#### LlamaCpp

```python
from llama_cpp import Llama

from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_llamacpp(
    Llama.from_pretrained(
        repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        filename='mistral-7b-instruct-v0.2.Q5_K_M.gguf',
    )
)

```

#### MLXLM

```python
from mlx_lm import load

from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_mlxlm(
    *load('mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit')
)

```

#### SGLang

```python
from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_sglang(
    'http://localhost:11434',
    'api_key',
    'meta-llama/Llama-3.1-8B'
)

```

#### vLLM Offline

```python
from vllm import LLM

from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_vllm_offline(
    LLM('microsoft/Phi-3-mini-4k-instruct')
)

```

## Running the model

Once you have initialized an `OutlinesModel`, you can use it with an Agent as with all other Pydantic AI models.

As Outlines is focused on structured output, this provider supports the `output_type` component through the NativeOutput format. There is not need to include information on the required output format in your prompt, instructions based on the `output_type` will be included automatically.

```python
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pydantic_ai import Agent
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.settings import ModelSettings


class Box(BaseModel):
    """Class representing a box"""
    width: int
    height: int
    depth: int
    units: str

model = OutlinesModel.from_transformers(
    AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct'),
    AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
)
agent = Agent(model, output_type=Box)

result = agent.run_sync(
    'Give me the dimensions of a box',
    model_settings=ModelSettings(extra_body={'max_new_tokens': 100})
)
print(result.output) # width=20 height=30 depth=40 units='cm'

```

Outlines does not support tools yet, but support for that feature will be added in the near future.

## Multimodal models

If the model you are running through Outlines and the provider selected supports it, you can include images in your prompts using ImageUrl or BinaryImage. In that case, the prompt you provide when running the agent should be a list containing a string and one or several images. See the [input documentation](../../input/) for details and examples on using assets in model inputs.

This feature is supported in Outlines for the `SGLang` and `Transformers` models. If you want to run a multimodal model through `transformers`, you must provide a processor instead of a tokenizer as the second argument when initializing the model with the `OutlinesModel.from_transformers` method.

```python
from datetime import date
from typing import Literal

import torch
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.messages import ImageUrl
from pydantic_ai.models.outlines import OutlinesModel

MODEL_NAME = 'Qwen/Qwen2-VL-7B-Instruct'

class Item(BaseModel):
    name: str
    quantity: int | None
    price_per_unit: float | None
    total_price: float | None

class ReceiptSummary(BaseModel):
    store_name: str
    store_address: str
    store_number: int | None
    items: list[Item]
    tax: float | None
    total: float | None
    date: date
    payment_method: Literal['cash', 'credit', 'debit', 'check', 'other']

tf_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    dtype=torch.bfloat16
)
tf_processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    device_map='auto'
)
model = OutlinesModel.from_transformers(tf_model, tf_processor)

agent = Agent(model, output_type=ReceiptSummary)

result = agent.run_sync(
    [
        'You are an expert at extracting information from receipts. Please extract the information from the receipt. Be as detailed as possible, do not miss any information',
        ImageUrl('https://raw.githubusercontent.com/dottxt-ai/outlines/refs/heads/main/docs/examples/images/trader-joes-receipt.jpg')
    ],
    model_settings=ModelSettings(extra_body={'max_new_tokens': 1000})
)
print(result.output)
# store_name="Trader Joe's"
# store_address='401 Bay Street, San Francisco, CA 94133'
# store_number=0
# items=[
#   Item(name='BANANA EACH', quantity=7, price_per_unit=0.23, total_price=1.61),
#   Item(name='BAREBELLS CHOCOLATE DOUG',quantity=1, price_per_unit=2.29, total_price=2.29),
#   Item(name='BAREBELLS CREAMY CRISP', quantity=1, price_per_unit=2.29, total_price=2.29),
#   Item(name='BAREBELLS CHOCOLATE DOUG', quantity=1, price_per_unit=2.29, total_price=2.29),
#   Item(name='BAREBELLS CARAMEL CASHEW', quantity=2, price_per_unit=2.29, total_price=4.58),
#   Item(name='BAREBELLS CREAMY CRISP', quantity=1, price_per_unit=2.29, total_price=2.29),
#   Item(name='T SPINDRIFT ORANGE MANGO 8', quantity=1, price_per_unit=7.49, total_price=7.49),
#   Item(name='T Bottle Deposit', quantity=8, price_per_unit=0.05, total_price=0.4),
#   Item(name='MILK ORGANIC GALLON WHOL', quantity=1, price_per_unit=6.79, total_price=6.79),
#   Item(name='CLASSIC GREEK SALAD', quantity=1, price_per_unit=3.49, total_price=3.49),
#   Item(name='COBB SALAD', quantity=1, price_per_unit=5.99, total_price=5.99),
#   Item(name='PEPPER BELL RED XL EACH', quantity=1, price_per_unit=1.29, total_price=1.29),
#   Item(name='BAG FEE.', quantity=1, price_per_unit=0.25, total_price=0.25),
#   Item(name='BAG FEE.', quantity=1, price_per_unit=0.25, total_price=0.25)]
# tax=7.89
# total=41.98
# date='2023-04-01'
# payment_method='credit'

```

# Model Providers

Pydantic AI is model-agnostic and has built-in support for multiple model providers:

- [OpenAI](../openai/)
- [Anthropic](../anthropic/)
- [Gemini](../google/) (via two different APIs: Generative Language API and VertexAI API)
- [Bedrock](../bedrock/)
- [Cohere](../cohere/)
- [Groq](../groq/)
- [Hugging Face](../huggingface/)
- [Mistral](../mistral/)
- [OpenRouter](../openrouter/)
- [Outlines](../outlines/)

## OpenAI-compatible Providers

In addition, many providers are compatible with the OpenAI API, and can be used with `OpenAIChatModel` in Pydantic AI:

- [Azure AI Foundry](../openai/#azure-ai-foundry)
- [Cerebras](../openai/#cerebras)
- [DeepSeek](../openai/#deepseek)
- [Fireworks AI](../openai/#fireworks-ai)
- [GitHub Models](../openai/#github-models)
- [Grok (xAI)](../openai/#grok-xai)
- [Heroku](../openai/#heroku-ai)
- [LiteLLM](../openai/#litellm)
- [Nebius AI Studio](../openai/#nebius-ai-studio)
- [Ollama](../openai/#ollama)
- [OVHcloud AI Endpoints](../openai/#ovhcloud-ai-endpoints)
- [Perplexity](../openai/#perplexity)
- [Together AI](../openai/#together-ai)
- [Vercel AI Gateway](../openai/#vercel-ai-gateway)

Pydantic AI also comes with [`TestModel`](../../api/models/test/) and [`FunctionModel`](../../api/models/function/) for testing and development.

To use each model provider, you need to configure your local environment and make sure you have the right packages installed. If you try to use the model without having done so, you'll be told what to install.

## Models and Providers

Pydantic AI uses a few key terms to describe how it interacts with different LLMs:

- **Model**: This refers to the Pydantic AI class used to make requests following a specific LLM API (generally by wrapping a vendor-provided SDK, like the `openai` python SDK). These classes implement a vendor-SDK-agnostic API, ensuring a single Pydantic AI agent is portable to different LLM vendors without any other code changes just by swapping out the Model it uses. Model classes are named roughly in the format `<VendorSdk>Model`, for example, we have `OpenAIChatModel`, `AnthropicModel`, `GoogleModel`, etc. When using a Model class, you specify the actual LLM model name (e.g., `gpt-5`, `claude-sonnet-4-5`, `gemini-3-pro-preview`) as a parameter.
- **Provider**: This refers to provider-specific classes which handle the authentication and connections to an LLM vendor. Passing a non-default *Provider* as a parameter to a Model is how you can ensure that your agent will make requests to a specific endpoint, or make use of a specific approach to authentication (e.g., you can use Azure auth with the `OpenAIChatModel` by way of the `AzureProvider`). In particular, this is how you can make use of an AI gateway, or an LLM vendor that offers API compatibility with the vendor SDK used by an existing Model (such as `OpenAIChatModel`).
- **Profile**: This refers to a description of how requests to a specific model or family of models need to be constructed to get the best results, independent of the model and provider classes used. For example, different models have different restrictions on the JSON schemas that can be used for tools, and the same schema transformer needs to be used for Gemini models whether you're using `GoogleModel` with model name `gemini-2.5-pro-preview`, or `OpenAIChatModel` with `OpenRouterProvider` and model name `google/gemini-2.5-pro-preview`.

When you instantiate an Agent with just a name formatted as `<provider>:<model>`, e.g. `openai:gpt-5` or `openrouter:google/gemini-2.5-pro-preview`, Pydantic AI will automatically select the appropriate model class, provider, and profile. If you want to use a different provider or profile, you can instantiate a model class directly and pass in `provider` and/or `profile` arguments.

## Custom Models

Note

If a model API is compatible with the OpenAI API, you do not need a custom model class and can provide your own [custom provider](../openai/#openai-compatible-models) instead.

To implement support for a model API that's not already supported, you will need to subclass the Model abstract base class. For streaming, you'll also need to implement the StreamedResponse abstract base class.

The best place to start is to review the source code for existing implementations, e.g. [`OpenAIChatModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).

For details on when we'll accept contributions adding new models to Pydantic AI, see the [contributing guidelines](../../contributing/#new-model-rules).

## Fallback Model

You can use FallbackModel to attempt multiple models in sequence until one successfully returns a result. Under the hood, Pydantic AI automatically switches from one model to the next if the current model returns a 4xx or 5xx status code.

Note

The provider SDKs on which Models are based (like OpenAI, Anthropic, etc.) often have built-in retry logic that can delay the `FallbackModel` from activating.

```text
When using `FallbackModel`, it's recommended to disable provider SDK retries to ensure immediate fallback, for example by setting `max_retries=0` on a [custom OpenAI client](openai.md#custom-openai-client).

```

In the following example, the agent first makes a request to the OpenAI model (which fails due to an invalid API key), and then falls back to the Anthropic model.

fallback_model.py

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

openai_model = OpenAIChatModel('gpt-5')
anthropic_model = AnthropicModel('claude-sonnet-4-5')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
response = agent.run_sync('What is the capital of France?')
print(response.data)
#> Paris

print(response.all_messages())
"""
[
    ModelRequest(
        parts=[
            UserPromptPart(
                content='What is the capital of France?',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            )
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[TextPart(content='Paris', part_kind='text')],
        model_name='claude-sonnet-4-5',
        timestamp=datetime.datetime(...),
        kind='response',
        provider_response_id=None,
    ),
]
"""

```

The `ModelResponse` message above indicates in the `model_name` field that the output was returned by the Anthropic model, which is the second model specified in the `FallbackModel`.

Note

Each model's options should be configured individually. For example, `base_url`, `api_key`, and custom clients should be set on each model itself, not on the `FallbackModel`.

### Per-Model Settings

You can configure different ModelSettings for each model in a fallback chain by passing the `settings` parameter when creating each model. This is particularly useful when different providers have different optimal configurations:

fallback_model_per_settings.py

```python
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

# Configure each model with provider-specific optimal settings
openai_model = OpenAIChatModel(
    'gpt-5',
    settings=ModelSettings(temperature=0.7, max_tokens=1000)  # Higher creativity for OpenAI
)
anthropic_model = AnthropicModel(
    'claude-sonnet-4-5',
    settings=ModelSettings(temperature=0.2, max_tokens=1000)  # Lower temperature for consistency
)

fallback_model = FallbackModel(openai_model, anthropic_model)
agent = Agent(fallback_model)

result = agent.run_sync('Write a creative story about space exploration')
print(result.output)
"""
In the year 2157, Captain Maya Chen piloted her spacecraft through the vast expanse of the Andromeda Galaxy. As she discovered a planet with crystalline mountains that sang in harmony with the cosmic winds, she realized that space exploration was not just about finding new worlds, but about finding new ways to understand the universe and our place within it.
"""

```

In this example, if the OpenAI model fails, the agent will automatically fall back to the Anthropic model with its own configured settings. The `FallbackModel` itself doesn't have settings - it uses the individual settings of whichever model successfully handles the request.

In this next example, we demonstrate the exception-handling capabilities of `FallbackModel`. If all models fail, a FallbackExceptionGroup is raised, which contains all the exceptions encountered during the `run` execution.

fallback_model_failure.py

```python
from pydantic_ai import Agent, ModelAPIError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

openai_model = OpenAIChatModel('gpt-5')
anthropic_model = AnthropicModel('claude-sonnet-4-5')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
try:
    response = agent.run_sync('What is the capital of France?')
except* ModelAPIError as exc_group:
    for exc in exc_group.exceptions:
        print(exc)

```

Since [`except*`](https://docs.python.org/3/reference/compound_stmts.html#except-star) is only supported in Python 3.11+, we use the [`exceptiongroup`](https://github.com/agronholm/exceptiongroup) backport package for earlier Python versions:

fallback_model_failure.py

```python
from exceptiongroup import catch

from pydantic_ai import Agent, ModelAPIError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel


def model_status_error_handler(exc_group: BaseExceptionGroup) -> None:
    for exc in exc_group.exceptions:
        print(exc)


openai_model = OpenAIChatModel('gpt-5')
anthropic_model = AnthropicModel('claude-sonnet-4-5')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
with catch({ModelAPIError: model_status_error_handler}):
    response = agent.run_sync('What is the capital of France?')

```

By default, the `FallbackModel` only moves on to the next model if the current model raises a ModelAPIError, which includes ModelHTTPError. You can customize this behavior by passing a custom `fallback_on` argument to the `FallbackModel` constructor.
# Graphs

# Graphs

Don't use a nail gun unless you need a nail gun

If Pydantic AI [agents](../agents/) are a hammer, and [multi-agent workflows](../multi-agent-applications/) are a sledgehammer, then graphs are a nail gun:

- sure, nail guns look cooler than hammers
- but nail guns take a lot more setup than hammers
- and nail guns don't make you a better builder, they make you a builder with a nail gun
- Lastly, (and at the risk of torturing this metaphor), if you're a fan of medieval tools like mallets and untyped Python, you probably won't like nail guns or our approach to graphs. (But then again, if you're not a fan of type hints in Python, you've probably already bounced off Pydantic AI to use one of the toy agent frameworks â€” good luck, and feel free to borrow my sledgehammer when you realize you need it)

In short, graphs are a powerful tool, but they're not the right tool for every job. Please consider other [multi-agent approaches](../multi-agent-applications/) before proceeding.

If you're not confident a graph-based approach is a good idea, it might be unnecessary.

Graphs and finite state machines (FSMs) are a powerful abstraction to model, execute, control and visualize complex workflows.

Alongside Pydantic AI, we've developed `pydantic-graph` â€” an async graph and state machine library for Python where nodes and edges are defined using type hints.

While this library is developed as part of Pydantic AI; it has no dependency on `pydantic-ai` and can be considered as a pure graph-based state machine library. You may find it useful whether or not you're using Pydantic AI or even building with GenAI.

`pydantic-graph` is designed for advanced users and makes heavy use of Python generics and type hints. It is not designed to be as beginner-friendly as Pydantic AI.

## Installation

`pydantic-graph` is a required dependency of `pydantic-ai`, and an optional dependency of `pydantic-ai-slim`, see [installation instructions](../install/#slim-install) for more information. You can also install it directly:

```bash
pip install pydantic-graph

```

```bash
uv add pydantic-graph

```

## Graph Types

`pydantic-graph` is made up of a few key components:

### GraphRunContext

GraphRunContext â€” The context for the graph run, similar to Pydantic AI's RunContext. This holds the state of the graph and dependencies and is passed to nodes when they're run.

`GraphRunContext` is generic in the state type of the graph it's used in, StateT.

### End

End â€” return value to indicate the graph run should end.

`End` is generic in the graph return type of the graph it's used in, RunEndT.

### Nodes

Subclasses of BaseNode define nodes for execution in the graph.

Nodes, which are generally dataclasses, generally consist of:

- fields containing any parameters required/optional when calling the node
- the business logic to execute the node, in the run method
- return annotations of the run method, which are read by `pydantic-graph` to determine the outgoing edges of the node

Nodes are generic in:

- **state**, which must have the same type as the state of graphs they're included in, StateT has a default of `None`, so if you're not using state you can omit this generic parameter, see [stateful graphs](#stateful-graphs) for more information
- **deps**, which must have the same type as the deps of the graph they're included in, DepsT has a default of `None`, so if you're not using deps you can omit this generic parameter, see [dependency injection](#dependency-injection) for more information
- **graph return type** â€” this only applies if the node returns End. RunEndT has a default of Never so this generic parameter can be omitted if the node doesn't return `End`, but must be included if it does.

Here's an example of a start or intermediate node in a graph â€” it can't end the run as it doesn't return End:

intermediate_node.py

```python
from dataclasses import dataclass

from pydantic_graph import BaseNode, GraphRunContext


@dataclass
class MyNode(BaseNode[MyState]):  # (1)!
    foo: int  # (2)!

    async def run(
        self,
        ctx: GraphRunContext[MyState],  # (3)!
    ) -> AnotherNode:  # (4)!
        ...
        return AnotherNode()

```

1. State in this example is `MyState` (not shown), hence `BaseNode` is parameterized with `MyState`. This node can't end the run, so the `RunEndT` generic parameter is omitted and defaults to `Never`.
1. `MyNode` is a dataclass and has a single field `foo`, an `int`.
1. The `run` method takes a `GraphRunContext` parameter, again parameterized with state `MyState`.
1. The return type of the `run` method is `AnotherNode` (not shown), this is used to determine the outgoing edges of the node.

We could extend `MyNode` to optionally end the run if `foo` is divisible by 5:

intermediate_or_end_node.py

```python
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext


@dataclass
class MyNode(BaseNode[MyState, None, int]):  # (1)!
    foo: int

    async def run(
        self,
        ctx: GraphRunContext[MyState],
    ) -> AnotherNode | End[int]:  # (2)!
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return AnotherNode()

```

1. We parameterize the node with the return type (`int` in this case) as well as state. Because generic parameters are positional-only, we have to include `None` as the second parameter representing deps.
1. The return type of the `run` method is now a union of `AnotherNode` and `End[int]`, this allows the node to end the run if `foo` is divisible by 5.

### Graph

Graph â€” this is the execution graph itself, made up of a set of [node classes](#nodes) (i.e., `BaseNode` subclasses).

`Graph` is generic in:

- **state** the state type of the graph, StateT
- **deps** the deps type of the graph, DepsT
- **graph return type** the return type of the graph run, RunEndT

Here's an example of a simple graph:

graph_example.py

```python
from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class DivisibleBy5(BaseNode[None, None, int]):  # (1)!
    foo: int

    async def run(
        self,
        ctx: GraphRunContext,
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)


@dataclass
class Increment(BaseNode):  # (2)!
    foo: int

    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        return DivisibleBy5(self.foo + 1)


fives_graph = Graph(nodes=[DivisibleBy5, Increment])  # (3)!
result = fives_graph.run_sync(DivisibleBy5(4))  # (4)!
print(result.output)
#> 5

```

1. The `DivisibleBy5` node is parameterized with `None` for the state param and `None` for the deps param as this graph doesn't use state or deps, and `int` as it can end the run.
1. The `Increment` node doesn't return `End`, so the `RunEndT` generic parameter is omitted, state can also be omitted as the graph doesn't use state.
1. The graph is created with a sequence of nodes.
1. The graph is run synchronously with run_sync. The initial node is `DivisibleBy5(4)`. Because the graph doesn't use external state or deps, we don't pass `state` or `deps`.

*(This example is complete, it can be run "as is")*

A [mermaid diagram](#mermaid-diagrams) for this graph can be generated with the following code:

graph_example_diagram.py

```python
from graph_example import DivisibleBy5, fives_graph

fives_graph.mermaid_code(start_node=DivisibleBy5)

```

```
---
title: fives_graph
---
stateDiagram-v2
  [*] --> DivisibleBy5
  DivisibleBy5 --> Increment
  DivisibleBy5 --> [*]
  Increment --> DivisibleBy5
```

In order to visualize a graph within a `jupyter-notebook`, `IPython.display` needs to be used:

jupyter_display_mermaid.py

```python
from graph_example import DivisibleBy5, fives_graph
from IPython.display import Image, display

display(Image(fives_graph.mermaid_image(start_node=DivisibleBy5)))

```

## Stateful Graphs

The "state" concept in `pydantic-graph` provides an optional way to access and mutate an object (often a `dataclass` or Pydantic model) as nodes run in a graph. If you think of Graphs as a production line, then your state is the engine being passed along the line and built up by each node as the graph is run.

`pydantic-graph` provides state persistence, with the state recorded after each node is run. (See [State Persistence](#state-persistence).)

Here's an example of a graph which represents a vending machine where the user may insert coins and select a product to purchase.

vending_machine.py

```python
from __future__ import annotations

from dataclasses import dataclass

from rich.prompt import Prompt

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class MachineState:  # (1)!
    user_balance: float = 0.0
    product: str | None = None


@dataclass
class InsertCoin(BaseNode[MachineState]):  # (3)!
    async def run(self, ctx: GraphRunContext[MachineState]) -> CoinsInserted:  # (16)!
        return CoinsInserted(float(Prompt.ask('Insert coins')))  # (4)!


@dataclass
class CoinsInserted(BaseNode[MachineState]):
    amount: float  # (5)!

    async def run(
        self, ctx: GraphRunContext[MachineState]
    ) -> SelectProduct | Purchase:  # (17)!
        ctx.state.user_balance += self.amount  # (6)!
        if ctx.state.product is not None:  # (7)!
            return Purchase(ctx.state.product)
        else:
            return SelectProduct()


@dataclass
class SelectProduct(BaseNode[MachineState]):
    async def run(self, ctx: GraphRunContext[MachineState]) -> Purchase:
        return Purchase(Prompt.ask('Select product'))


PRODUCT_PRICES = {  # (2)!
    'water': 1.25,
    'soda': 1.50,
    'crisps': 1.75,
    'chocolate': 2.00,
}


@dataclass
class Purchase(BaseNode[MachineState, None, None]):  # (18)!
    product: str

    async def run(
        self, ctx: GraphRunContext[MachineState]
    ) -> End | InsertCoin | SelectProduct:
        if price := PRODUCT_PRICES.get(self.product):  # (8)!
            ctx.state.product = self.product  # (9)!
            if ctx.state.user_balance >= price:  # (10)!
                ctx.state.user_balance -= price
                return End(None)
            else:
                diff = price - ctx.state.user_balance
                print(f'Not enough money for {self.product}, need {diff:0.2f} more')
                #> Not enough money for crisps, need 0.75 more
                return InsertCoin()  # (11)!
        else:
            print(f'No such product: {self.product}, try again')
            return SelectProduct()  # (12)!


vending_machine_graph = Graph(  # (13)!
    nodes=[InsertCoin, CoinsInserted, SelectProduct, Purchase]
)


async def main():
    state = MachineState()  # (14)!
    await vending_machine_graph.run(InsertCoin(), state=state)  # (15)!
    print(f'purchase successful item={state.product} change={state.user_balance:0.2f}')
    #> purchase successful item=crisps change=0.25

```

1. The state of the vending machine is defined as a dataclass with the user's balance and the product they've selected, if any.
1. A dictionary of products mapped to prices.
1. The `InsertCoin` node, BaseNode is parameterized with `MachineState` as that's the state used in this graph.
1. The `InsertCoin` node prompts the user to insert coins. We keep things simple by just entering a monetary amount as a float. Before you start thinking this is a toy too since it's using rich's Prompt.ask within nodes, see [below](#example-human-in-the-loop) for how control flow can be managed when nodes require external input.
1. The `CoinsInserted` node; again this is a dataclass with one field `amount`.
1. Update the user's balance with the amount inserted.
1. If the user has already selected a product, go to `Purchase`, otherwise go to `SelectProduct`.
1. In the `Purchase` node, look up the price of the product if the user entered a valid product.
1. If the user did enter a valid product, set the product in the state so we don't revisit `SelectProduct`.
1. If the balance is enough to purchase the product, adjust the balance to reflect the purchase and return End to end the graph. We're not using the run return type, so we call `End` with `None`.
1. If the balance is insufficient, go to `InsertCoin` to prompt the user to insert more coins.
1. If the product is invalid, go to `SelectProduct` to prompt the user to select a product again.
1. The graph is created by passing a list of nodes to Graph. Order of nodes is not important, but it can affect how [diagrams](#mermaid-diagrams) are displayed.
1. Initialize the state. This will be passed to the graph run and mutated as the graph runs.
1. Run the graph with the initial state. Since the graph can be run from any node, we must pass the start node â€” in this case, `InsertCoin`. Graph.run returns a GraphRunResult that provides the final data and a history of the run.
1. The return type of the node's run method is important as it is used to determine the outgoing edges of the node. This information in turn is used to render [mermaid diagrams](#mermaid-diagrams) and is enforced at runtime to detect misbehavior as soon as possible.
1. The return type of `CoinsInserted`'s run method is a union, meaning multiple outgoing edges are possible.
1. Unlike other nodes, `Purchase` can end the run, so the RunEndT generic parameter must be set. In this case it's `None` since the graph run return type is `None`.

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

A [mermaid diagram](#mermaid-diagrams) for this graph can be generated with the following code:

vending_machine_diagram.py

```python
from vending_machine import InsertCoin, vending_machine_graph

vending_machine_graph.mermaid_code(start_node=InsertCoin)

```

The diagram generated by the above code is:

```
---
title: vending_machine_graph
---
stateDiagram-v2
  [*] --> InsertCoin
  InsertCoin --> CoinsInserted
  CoinsInserted --> SelectProduct
  CoinsInserted --> Purchase
  SelectProduct --> Purchase
  Purchase --> InsertCoin
  Purchase --> SelectProduct
  Purchase --> [*]
```

See [below](#mermaid-diagrams) for more information on generating diagrams.

## GenAI Example

So far we haven't shown an example of a Graph that actually uses Pydantic AI or GenAI at all.

In this example, one agent generates a welcome email to a user and the other agent provides feedback on the email.

This graph has a very simple structure:

```
---
title: feedback_graph
---
stateDiagram-v2
  [*] --> WriteEmail
  WriteEmail --> Feedback
  Feedback --> WriteEmail
  Feedback --> [*]
```

genai_email_feedback.py

```python
from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, EmailStr

from pydantic_ai import Agent, ModelMessage, format_as_xml
from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]


@dataclass
class Email:
    subject: str
    body: str


@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)


email_writer_agent = Agent(
    'google-gla:gemini-2.5-pro',
    output_type=Email,
    system_prompt='Write a welcome email to our tech blog.',
)


@dataclass
class WriteEmail(BaseNode[State]):
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> Feedback:
        if self.email_feedback:
            prompt = (
                f'Rewrite the email for the user:\n'
                f'{format_as_xml(ctx.state.user)}\n'
                f'Feedback: {self.email_feedback}'
            )
        else:
            prompt = (
                f'Write a welcome email for the user:\n'
                f'{format_as_xml(ctx.state.user)}'
            )

        result = await email_writer_agent.run(
            prompt,
            message_history=ctx.state.write_agent_messages,
        )
        ctx.state.write_agent_messages += result.new_messages()
        return Feedback(result.output)


class EmailRequiresWrite(BaseModel):
    feedback: str


class EmailOk(BaseModel):
    pass


feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    'openai:gpt-5',
    output_type=EmailRequiresWrite | EmailOk,  # type: ignore
    system_prompt=(
        'Review the email and provide feedback, email must reference the users specific interests.'
    ),
)


@dataclass
class Feedback(BaseNode[State, None, Email]):
    email: Email

    async def run(
        self,
        ctx: GraphRunContext[State],
    ) -> WriteEmail | End[Email]:
        prompt = format_as_xml({'user': ctx.state.user, 'email': self.email})
        result = await feedback_agent.run(prompt)
        if isinstance(result.output, EmailRequiresWrite):
            return WriteEmail(email_feedback=result.output.feedback)
        else:
            return End(self.email)


async def main():
    user = User(
        name='John Doe',
        email='john.joe@example.com',
        interests=['Haskel', 'Lisp', 'Fortran'],
    )
    state = State(user)
    feedback_graph = Graph(nodes=(WriteEmail, Feedback))
    result = await feedback_graph.run(WriteEmail(), state=state)
    print(result.output)
    """
    Email(
        subject='Welcome to our tech blog!',
        body='Hello John, Welcome to our tech blog! ...',
    )
    """

```

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

## Iterating Over a Graph

### Using `Graph.iter` for `async for` iteration

Sometimes you want direct control or insight into each node as the graph executes. The easiest way to do that is with the Graph.iter method, which returns a **context manager** that yields a GraphRun object. The `GraphRun` is an async-iterable over the nodes of your graph, allowing you to record or modify them as they execute.

Here's an example:

count_down.py

```python
from __future__ import annotations as _annotations

from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End, GraphRunContext


@dataclass
class CountDownState:
    counter: int


@dataclass
class CountDown(BaseNode[CountDownState, None, int]):
    async def run(self, ctx: GraphRunContext[CountDownState]) -> CountDown | End[int]:
        if ctx.state.counter <= 0:
            return End(ctx.state.counter)
        ctx.state.counter -= 1
        return CountDown()


count_down_graph = Graph(nodes=[CountDown])


async def main():
    state = CountDownState(counter=3)
    async with count_down_graph.iter(CountDown(), state=state) as run:  # (1)!
        async for node in run:  # (2)!
            print('Node:', node)
            #> Node: CountDown()
            #> Node: CountDown()
            #> Node: CountDown()
            #> Node: CountDown()
            #> Node: End(data=0)
    print('Final output:', run.result.output)  # (3)!
    #> Final output: 0

```

1. `Graph.iter(...)` returns a GraphRun.
1. Here, we step through each node as it is executed.
1. Once the graph returns an End, the loop ends, and `run.result` becomes a GraphRunResult containing the final outcome (`0` here).

### Using `GraphRun.next(node)` manually

Alternatively, you can drive iteration manually with the GraphRun.next method, which allows you to pass in whichever node you want to run next. You can modify or selectively skip nodes this way.

Below is a contrived example that stops whenever the counter is at 2, ignoring any node runs beyond that:

count_down_next.py

```python
from pydantic_graph import End, FullStatePersistence
from count_down import CountDown, CountDownState, count_down_graph


async def main():
    state = CountDownState(counter=5)
    persistence = FullStatePersistence()  # (7)!
    async with count_down_graph.iter(
        CountDown(), state=state, persistence=persistence
    ) as run:
        node = run.next_node  # (1)!
        while not isinstance(node, End):  # (2)!
            print('Node:', node)
            #> Node: CountDown()
            #> Node: CountDown()
            #> Node: CountDown()
            #> Node: CountDown()
            if state.counter == 2:
                break  # (3)!
            node = await run.next(node)  # (4)!

        print(run.result)  # (5)!
        #> None

        for step in persistence.history:  # (6)!
            print('History Step:', step.state, step.state)
            #> History Step: CountDownState(counter=5) CountDownState(counter=5)
            #> History Step: CountDownState(counter=4) CountDownState(counter=4)
            #> History Step: CountDownState(counter=3) CountDownState(counter=3)
            #> History Step: CountDownState(counter=2) CountDownState(counter=2)

```

1. We start by grabbing the first node that will be run in the agent's graph.
1. The agent run is finished once an `End` node has been produced; instances of `End` cannot be passed to `next`.
1. If the user decides to stop early, we break out of the loop. The graph run won't have a real final result in that case (`run.result` remains `None`).
1. At each step, we call `await run.next(node)` to run it and get the next node (or an `End`).
1. Because we did not continue the run until it finished, the `result` is not set.
1. The run's history is still populated with the steps we executed so far.
1. Use FullStatePersistence so we can show the history of the run, see [State Persistence](#state-persistence) below for more information.

## State Persistence

One of the biggest benefits of finite state machine (FSM) graphs is how they simplify the handling of interrupted execution. This might happen for a variety of reasons:

- the state machine logic might fundamentally need to be paused â€” e.g. the returns workflow for an e-commerce order needs to wait for the item to be posted to the returns center or because execution of the next node needs input from a user so needs to wait for a new http request,
- the execution takes so long that the entire graph can't reliably be executed in a single continuous run â€” e.g. a deep research agent that might take hours to run,
- you want to run multiple graph nodes in parallel in different processes / hardware instances (note: parallel node execution is not yet supported in `pydantic-graph`, see [#704](https://github.com/pydantic/pydantic-ai/issues/704)).

Trying to make a conventional control flow (i.e., boolean logic and nested function calls) implementation compatible with these usage scenarios generally results in brittle and over-complicated spaghetti code, with the logic required to interrupt and resume execution dominating the implementation.

To allow graph runs to be interrupted and resumed, `pydantic-graph` provides state persistence â€” a system for snapshotting the state of a graph run before and after each node is run, allowing a graph run to be resumed from any point in the graph.

`pydantic-graph` includes three state persistence implementations:

- SimpleStatePersistence â€” Simple in memory state persistence that just hold the latest snapshot. If no state persistence implementation is provided when running a graph, this is used by default.
- FullStatePersistence â€” In memory state persistence that hold a list of snapshots.
- FileStatePersistence â€” File-based state persistence that saves snapshots to a JSON file.

In production applications, developers should implement their own state persistence by subclassing BaseStatePersistence abstract base class, which might persist runs in a relational database like PostgresQL.

At a high level the role of `StatePersistence` implementations is to store and retrieve NodeSnapshot and EndSnapshot objects.

graph.iter_from_persistence() may be used to run the graph based on the state stored in persistence.

We can run the `count_down_graph` from [above](#iterating-over-a-graph), using graph.iter_from_persistence() and FileStatePersistence.

As you can see in this code, `run_node` requires no external application state (apart from state persistence) to be run, meaning graphs can easily be executed by distributed execution and queueing systems.

count_down_from_persistence.py

```python
from pathlib import Path

from pydantic_graph import End
from pydantic_graph.persistence.file import FileStatePersistence

from count_down import CountDown, CountDownState, count_down_graph


async def main():
    run_id = 'run_abc123'
    persistence = FileStatePersistence(Path(f'count_down_{run_id}.json'))  # (1)!
    state = CountDownState(counter=5)
    await count_down_graph.initialize(  # (2)!
        CountDown(), state=state, persistence=persistence
    )

    done = False
    while not done:
        done = await run_node(run_id)


async def run_node(run_id: str) -> bool:  # (3)!
    persistence = FileStatePersistence(Path(f'count_down_{run_id}.json'))
    async with count_down_graph.iter_from_persistence(persistence) as run:  # (4)!
        node_or_end = await run.next()  # (5)!

    print('Node:', node_or_end)
    #> Node: CountDown()
    #> Node: CountDown()
    #> Node: CountDown()
    #> Node: CountDown()
    #> Node: CountDown()
    #> Node: End(data=0)
    return isinstance(node_or_end, End)  # (6)!

```

1. Create a FileStatePersistence to use to start the graph.
1. Call graph.initialize() to set the initial graph state in the persistence object.
1. `run_node` is a pure function that doesn't need access to any other process state to run the next node of the graph, except the ID of the run.
1. Call graph.iter_from_persistence() create a GraphRun object that will run the next node of the graph from the state stored in persistence. This will return either a node or an `End` object.
1. graph.run() will return either a node or an End object.
1. Check if the node is an End object, if it is, the graph run is complete.

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

### Example: Human in the loop.

As noted above, state persistence allows graphs to be interrupted and resumed. One use case of this is to allow user input to continue.

In this example, an AI asks the user a question, the user provides an answer, the AI evaluates the answer and ends if the user got it right or asks another question if they got it wrong.

Instead of running the entire graph in a single process invocation, we run the graph by running the process repeatedly, optionally providing an answer to the question as a command line argument.

`ai_q_and_a_graph.py` â€” `question_graph` definition

[Learn about Gateway](../gateway) ai_q_and_a_graph.py

```python
from __future__ import annotations as _annotations

from typing import Annotated
from pydantic_graph import Edge
from dataclasses import dataclass, field
from pydantic import BaseModel
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphRunContext,
)
from pydantic_ai import Agent, format_as_xml
from pydantic_ai import ModelMessage

ask_agent = Agent('gateway/openai:gpt-5', output_type=str, instrument=True)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPT-5."""
    docstring_notes = True
    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[Answer, Edge(label='Ask the question')]:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.new_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        answer = input(f'{self.question}: ')
        return Evaluate(answer)


class EvaluationResult(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    'gateway/openai:gpt-5',
    output_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.new_messages()
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

```

ai_q_and_a_graph.py

```python
from __future__ import annotations as _annotations

from typing import Annotated
from pydantic_graph import Edge
from dataclasses import dataclass, field
from pydantic import BaseModel
from pydantic_graph import (
    BaseNode,
    End,
    Graph,
    GraphRunContext,
)
from pydantic_ai import Agent, format_as_xml
from pydantic_ai import ModelMessage

ask_agent = Agent('openai:gpt-5', output_type=str, instrument=True)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPT-5."""
    docstring_notes = True
    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[Answer, Edge(label='Ask the question')]:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.new_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        answer = input(f'{self.question}: ')
        return Evaluate(answer)


class EvaluationResult(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    'openai:gpt-5',
    output_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.new_messages()
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

```

*(This example is complete, it can be run "as is")*

ai_q_and_a_run.py

```python
import sys
from pathlib import Path

from pydantic_graph import End
from pydantic_graph.persistence.file import FileStatePersistence
from pydantic_ai import ModelMessage  # noqa: F401

from ai_q_and_a_graph import Ask, question_graph, Evaluate, QuestionState, Answer


async def main():
    answer: str | None = sys.argv[1] if len(sys.argv) > 1 else None  # (1)!
    persistence = FileStatePersistence(Path('question_graph.json'))  # (2)!
    persistence.set_graph_types(question_graph)  # (3)!

    if snapshot := await persistence.load_next():  # (4)!
        state = snapshot.state
        assert answer is not None
        node = Evaluate(answer)
    else:
        state = QuestionState()
        node = Ask()  # (5)!

    async with question_graph.iter(node, state=state, persistence=persistence) as run:
        while True:
            node = await run.next()  # (6)!
            if isinstance(node, End):  # (7)!
                print('END:', node.data)
                history = await persistence.load_all()  # (8)!
                print([e.node for e in history])
                break
            elif isinstance(node, Answer):  # (9)!
                print(node.question)
                #> What is the capital of France?
                break
            # otherwise just continue

```

1. Get the user's answer from the command line, if provided. See [question graph example](../examples/question-graph/) for a complete example.
1. Create a state persistence instance the `'question_graph.json'` file may or may not already exist.
1. Since we're using the persistence interface outside a graph, we need to call set_graph_types to set the graph generic types `StateT` and `RunEndT` for the persistence instance. This is necessary to allow the persistence instance to know how to serialize and deserialize graph nodes.
1. If we're run the graph before, load_next will return a snapshot of the next node to run, here we use `state` from that snapshot, and create a new `Evaluate` node with the answer provided on the command line.
1. If the graph hasn't been run before, we create a new `QuestionState` and start with the `Ask` node.
1. Call GraphRun.next() to run the node. This will return either a node or an `End` object.
1. If the node is an `End` object, the graph run is complete. The `data` field of the `End` object contains the comment returned by the `evaluate_agent` about the correct answer.
1. To demonstrate the state persistence, we call load_all to get all the snapshots from the persistence instance. This will return a list of Snapshot objects.
1. If the node is an `Answer` object, we print the question and break out of the loop to end the process and wait for user input.

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

For a complete example of this graph, see the [question graph example](../examples/question-graph/).

## Dependency Injection

As with Pydantic AI, `pydantic-graph` supports dependency injection via a generic parameter on Graph and BaseNode, and the GraphRunContext.deps field.

As an example of dependency injection, let's modify the `DivisibleBy5` example [above](#graph) to use a ProcessPoolExecutor to run the compute load in a separate process (this is a contrived example, `ProcessPoolExecutor` wouldn't actually improve performance in this example):

deps_example.py

```python
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, FullStatePersistence, Graph, GraphRunContext


@dataclass
class GraphDeps:
    executor: ProcessPoolExecutor


@dataclass
class DivisibleBy5(BaseNode[None, GraphDeps, int]):
    foo: int

    async def run(
        self,
        ctx: GraphRunContext[None, GraphDeps],
    ) -> Increment | End[int]:
        if self.foo % 5 == 0:
            return End(self.foo)
        else:
            return Increment(self.foo)


@dataclass
class Increment(BaseNode[None, GraphDeps]):
    foo: int

    async def run(self, ctx: GraphRunContext[None, GraphDeps]) -> DivisibleBy5:
        loop = asyncio.get_running_loop()
        compute_result = await loop.run_in_executor(
            ctx.deps.executor,
            self.compute,
        )
        return DivisibleBy5(compute_result)

    def compute(self) -> int:
        return self.foo + 1


fives_graph = Graph(nodes=[DivisibleBy5, Increment])


async def main():
    with ProcessPoolExecutor() as executor:
        deps = GraphDeps(executor)
        result = await fives_graph.run(DivisibleBy5(3), deps=deps, persistence=FullStatePersistence())
    print(result.output)
    #> 5
    # the full history is quite verbose (see below), so we'll just print the summary
    print([item.node for item in result.persistence.history])
    """
    [
        DivisibleBy5(foo=3),
        Increment(foo=3),
        DivisibleBy5(foo=4),
        Increment(foo=4),
        DivisibleBy5(foo=5),
        End(data=5),
    ]
    """

```

*(This example is complete, it can be run "as is" â€” you'll need to add `asyncio.run(main())` to run `main`)*

## Mermaid Diagrams

Pydantic Graph can generate [mermaid](https://mermaid.js.org/) [`stateDiagram-v2`](https://mermaid.js.org/syntax/stateDiagram.html) diagrams for graphs, as shown above.

These diagrams can be generated with:

- Graph.mermaid_code to generate the mermaid code for a graph
- Graph.mermaid_image to generate an image of the graph using [mermaid.ink](https://mermaid.ink/)
- Graph.mermaid_save to generate an image of the graph using [mermaid.ink](https://mermaid.ink/) and save it to a file

Beyond the diagrams shown above, you can also customize mermaid diagrams with the following options:

- Edge allows you to apply a label to an edge
- BaseNode.docstring_notes and BaseNode.get_note allows you to add notes to nodes
- The highlighted_nodes parameter allows you to highlight specific node(s) in the diagram

Putting that together, we can edit the last [`ai_q_and_a_graph.py`](#example-human-in-the-loop) example to:

- add labels to some edges
- add a note to the `Ask` node
- highlight the `Answer` node
- save the diagram as a `PNG` image to file

[Learn about Gateway](../gateway) ai_q_and_a_graph_extra.py

```python
from typing import Annotated
from pydantic_graph import BaseNode, End, Graph, GraphRunContext, Edge
ask_agent = Agent('gateway/openai:gpt-5', output_type=str, instrument=True)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)

@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPT-5."""
    docstring_notes = True
    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[Answer, Edge(label='Ask the question')]:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.new_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        answer = input(f'{self.question}: ')
        return Evaluate(answer)


class EvaluationResult(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    'gateway/openai:gpt-5',
    output_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.new_messages()
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

```

ai_q_and_a_graph_extra.py

```python
from typing import Annotated
from pydantic_graph import BaseNode, End, Graph, GraphRunContext, Edge
ask_agent = Agent('openai:gpt-5', output_type=str, instrument=True)


@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)

@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPT-5."""
    docstring_notes = True
    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[Answer, Edge(label='Ask the question')]:
        result = await ask_agent.run(
            'Ask a simple question with a single correct answer.',
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.new_messages()
        ctx.state.question = result.output
        return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        answer = input(f'{self.question}: ')
        return Evaluate(answer)


class EvaluationResult(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    'openai:gpt-5',
    output_type=EvaluationResult,
    system_prompt='Given a question and answer, evaluate if the answer is correct.',
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.new_messages()
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

```

*(This example is not complete and cannot be run directly)*

This would generate an image that looks like this:

```
---
title: question_graph
---
stateDiagram-v2
  Ask --> Answer: Ask the question
  note right of Ask
    Judge the answer.
    Decide on next step.
  end note
  Answer --> Evaluate
  Evaluate --> Reprimand
  Evaluate --> [*]: success
  Reprimand --> Ask

classDef highlighted fill:#fdff32
class Answer highlighted
```

### Setting Direction of the State Diagram

You can specify the direction of the state diagram using one of the following values:

- `'TB'`: Top to bottom, the diagram flows vertically from top to bottom.
- `'LR'`: Left to right, the diagram flows horizontally from left to right.
- `'RL'`: Right to left, the diagram flows horizontally from right to left.
- `'BT'`: Bottom to top, the diagram flows vertically from bottom to top.

Here is an example of how to do this using 'Left to Right' (LR) instead of the default 'Top to Bottom' (TB):

vending_machine_diagram.py

```python
from vending_machine import InsertCoin, vending_machine_graph

vending_machine_graph.mermaid_code(start_node=InsertCoin, direction='LR')

```

```
---
title: vending_machine_graph
---
stateDiagram-v2
  direction LR
  [*] --> InsertCoin
  InsertCoin --> CoinsInserted
  CoinsInserted --> SelectProduct
  CoinsInserted --> Purchase
  SelectProduct --> Purchase
  Purchase --> InsertCoin
  Purchase --> SelectProduct
  Purchase --> [*]
```
# API Reference

# `pydantic_ai.ag_ui`

Provides an AG-UI protocol adapter for the Pydantic AI agent.

This package provides seamless integration between pydantic-ai agents and ag-ui for building interactive AI applications with streaming event-based communication.

### SSE_CONTENT_TYPE

```python
SSE_CONTENT_TYPE = 'text/event-stream'

```

Content type header value for Server-Sent Events (SSE).

### OnCompleteFunc

```python
OnCompleteFunc: TypeAlias = (
    Callable[[AgentRunResult[Any]], None]
    | Callable[[AgentRunResult[Any]], Awaitable[None]]
    | Callable[[AgentRunResult[Any]], AsyncIterator[EventT]]
)

```

Callback function type that receives the `AgentRunResult` of the completed run. Can be sync, async, or an async generator of protocol-specific events.

### StateDeps

Bases: `Generic[StateT]`

Dependency type that holds state.

This class is used to manage the state of an agent run. It allows setting the state of the agent run with a specific type of state model, which must be a subclass of `BaseModel`.

The state is set using the `state` setter by the `Adapter` when the run starts.

Implements the `StateHandler` protocol.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@dataclass
class StateDeps(Generic[StateT]):
    """Dependency type that holds state.

    This class is used to manage the state of an agent run. It allows setting
    the state of the agent run with a specific type of state model, which must
    be a subclass of `BaseModel`.

    The state is set using the `state` setter by the `Adapter` when the run starts.

    Implements the `StateHandler` protocol.
    """

    state: StateT

```

### StateHandler

Bases: `Protocol`

Protocol for state handlers in agent runs. Requires the class to be a dataclass with a `state` field.

Source code in `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`

```python
@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs. Requires the class to be a dataclass with a `state` field."""

    # Has to be a dataclass so we can use `replace` to update the state.
    # From https://github.com/python/typeshed/blob/9ab7fde0a0cd24ed7a72837fcb21093b811b80d8/stdlib/_typeshed/__init__.pyi#L352
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def state(self) -> Any:
        """Get the current state of the agent run."""
        ...

    @state.setter
    def state(self, state: Any) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Args:
            state: The run state.
        """
        ...

```

#### state

```python
state: Any

```

Get the current state of the agent run.

### AGUIApp

Bases: `Generic[AgentDepsT, OutputDataT]`, `Starlette`

ASGI application for running Pydantic AI agents with AG-UI protocol support.

Source code in `pydantic_ai_slim/pydantic_ai/ui/ag_ui/app.py`

```python
class AGUIApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """ASGI application for running Pydantic AI agents with AG-UI protocol support."""

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        # AGUIAdapter.dispatch_request parameters
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[Any] | None = None,
        # Starlette parameters
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[Self] | None = None,
    ) -> None:
        """An ASGI application that handles every request by running the agent and streaming the response.

        Note that the `deps` will be the same for each request, with the exception of the frontend state that's
        injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ui.StateHandler] protocol.
        To provide different `deps` for each request (e.g. based on the authenticated user),
        use [`AGUIAdapter.run_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.run_stream] or
        [`AGUIAdapter.dispatch_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.dispatch_request] instead.

        Args:
            agent: The agent to run.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request. A starlette application will always
                automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
                outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
                `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
                exception cases occurring in the routing or endpoints.
            exception_handlers: A mapping of either integer status codes, or exception class types onto
                callables which handle the exceptions. Exception handler callables should be of the form
                `handler(request, exc) -> response` and may be either standard functions, or async functions.
            on_startup: A list of callables to run on application startup. Startup handler callables do not
                take any arguments, and may be either standard functions, or async functions.
            on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
                not take any arguments, and may be either standard functions, or async functions.
            lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
                This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
                the other, not both.
        """
        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )

        async def run_agent(request: Request) -> Response:
            """Endpoint to run the agent with the provided input data."""
            # `dispatch_request` will store the frontend state from the request on `deps.state` (if it implements the `StateHandler` protocol),
            # so we need to copy the deps to avoid different requests mutating the same deps object.
            nonlocal deps
            if isinstance(deps, StateHandler):  # pragma: no branch
                deps = replace(deps)

            return await AGUIAdapter[AgentDepsT, OutputDataT].dispatch_request(
                request,
                agent=agent,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                on_complete=on_complete,
            )

        self.router.add_route('/', run_agent, methods=['POST'])

```

#### __init__

```python
__init__(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    on_complete: OnCompleteFunc[Any] | None = None,
    debug: bool = False,
    routes: Sequence[BaseRoute] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: (
        Mapping[Any, ExceptionHandler] | None
    ) = None,
    on_startup: Sequence[Callable[[], Any]] | None = None,
    on_shutdown: Sequence[Callable[[], Any]] | None = None,
    lifespan: Lifespan[Self] | None = None
) -> None

```

An ASGI application that handles every request by running the agent and streaming the response.

Note that the `deps` will be the same for each request, with the exception of the frontend state that's injected into the `state` field of a `deps` object that implements the StateHandler protocol. To provide different `deps` for each request (e.g. based on the authenticated user), use AGUIAdapter.run_stream() or AGUIAdapter.dispatch_request() instead.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `agent` | `AbstractAgent[AgentDepsT, OutputDataT]` | The agent to run. | *required* | | `output_type` | `OutputSpec[Any] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` | | `on_complete` | `OnCompleteFunc[Any] | None` | Optional callback function called when the agent run completes successfully. The callback receives the completed AgentRunResult and can access all_messages() and other result data. | `None` | | `debug` | `bool` | Boolean indicating if debug tracebacks should be returned on errors. | `False` | | `routes` | `Sequence[BaseRoute] | None` | A list of routes to serve incoming HTTP and WebSocket requests. | `None` | | `middleware` | `Sequence[Middleware] | None` | A list of middleware to run for every request. A starlette application will always automatically include two middleware classes. ServerErrorMiddleware is added as the very outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack. ExceptionMiddleware is added as the very innermost middleware, to deal with handled exception cases occurring in the routing or endpoints. | `None` | | `exception_handlers` | `Mapping[Any, ExceptionHandler] | None` | A mapping of either integer status codes, or exception class types onto callables which handle the exceptions. Exception handler callables should be of the form handler(request, exc) -> response and may be either standard functions, or async functions. | `None` | | `on_startup` | `Sequence[Callable[[], Any]] | None` | A list of callables to run on application startup. Startup handler callables do not take any arguments, and may be either standard functions, or async functions. | `None` | | `on_shutdown` | `Sequence[Callable[[], Any]] | None` | A list of callables to run on application shutdown. Shutdown handler callables do not take any arguments, and may be either standard functions, or async functions. | `None` | | `lifespan` | `Lifespan[Self] | None` | A lifespan context function, which can be used to perform startup and shutdown tasks. This is a newer style that replaces the on_startup and on_shutdown handlers. Use one or the other, not both. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/ui/ag_ui/app.py`

```python
def __init__(
    self,
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    # AGUIAdapter.dispatch_request parameters
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    on_complete: OnCompleteFunc[Any] | None = None,
    # Starlette parameters
    debug: bool = False,
    routes: Sequence[BaseRoute] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
    on_startup: Sequence[Callable[[], Any]] | None = None,
    on_shutdown: Sequence[Callable[[], Any]] | None = None,
    lifespan: Lifespan[Self] | None = None,
) -> None:
    """An ASGI application that handles every request by running the agent and streaming the response.

    Note that the `deps` will be the same for each request, with the exception of the frontend state that's
    injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ui.StateHandler] protocol.
    To provide different `deps` for each request (e.g. based on the authenticated user),
    use [`AGUIAdapter.run_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.run_stream] or
    [`AGUIAdapter.dispatch_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.dispatch_request] instead.

    Args:
        agent: The agent to run.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
            no output validators since output validators would expect an argument that matches the agent's
            output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

        debug: Boolean indicating if debug tracebacks should be returned on errors.
        routes: A list of routes to serve incoming HTTP and WebSocket requests.
        middleware: A list of middleware to run for every request. A starlette application will always
            automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
            outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
            `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
            exception cases occurring in the routing or endpoints.
        exception_handlers: A mapping of either integer status codes, or exception class types onto
            callables which handle the exceptions. Exception handler callables should be of the form
            `handler(request, exc) -> response` and may be either standard functions, or async functions.
        on_startup: A list of callables to run on application startup. Startup handler callables do not
            take any arguments, and may be either standard functions, or async functions.
        on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
            not take any arguments, and may be either standard functions, or async functions.
        lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
            This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
            the other, not both.
    """
    super().__init__(
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        lifespan=lifespan,
    )

    async def run_agent(request: Request) -> Response:
        """Endpoint to run the agent with the provided input data."""
        # `dispatch_request` will store the frontend state from the request on `deps.state` (if it implements the `StateHandler` protocol),
        # so we need to copy the deps to avoid different requests mutating the same deps object.
        nonlocal deps
        if isinstance(deps, StateHandler):  # pragma: no branch
            deps = replace(deps)

        return await AGUIAdapter[AgentDepsT, OutputDataT].dispatch_request(
            request,
            agent=agent,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        )

    self.router.add_route('/', run_agent, methods=['POST'])

```

### handle_ag_ui_request

```python
handle_ag_ui_request(
    agent: AbstractAgent[AgentDepsT, Any],
    request: Request,
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    on_complete: OnCompleteFunc[BaseEvent] | None = None
) -> Response

```

Handle an AG-UI request by running the agent and returning a streaming response.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `agent` | `AbstractAgent[AgentDepsT, Any]` | The agent to run. | *required* | | `request` | `Request` | The Starlette request (e.g. from FastAPI) containing the AG-UI run input. | *required* | | `output_type` | `OutputSpec[Any] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `on_complete` | `OnCompleteFunc[BaseEvent] | None` | Optional callback function called when the agent run completes successfully. The callback receives the completed AgentRunResult and can access all_messages() and other result data. | `None` |

Returns:

| Type | Description | | --- | --- | | `Response` | A streaming Starlette response with AG-UI protocol events. |

Source code in `pydantic_ai_slim/pydantic_ai/ag_ui.py`

```python
async def handle_ag_ui_request(
    agent: AbstractAgent[AgentDepsT, Any],
    request: Request,
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    on_complete: OnCompleteFunc[BaseEvent] | None = None,
) -> Response:
    """Handle an AG-UI request by running the agent and returning a streaming response.

    Args:
        agent: The agent to run.
        request: The Starlette request (e.g. from FastAPI) containing the AG-UI run input.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

    Returns:
        A streaming Starlette response with AG-UI protocol events.
    """
    return await AGUIAdapter[AgentDepsT].dispatch_request(
        request,
        agent=agent,
        deps=deps,
        output_type=output_type,
        message_history=message_history,
        deferred_tool_results=deferred_tool_results,
        model=model,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
        on_complete=on_complete,
    )

```

### run_ag_ui

```python
run_ag_ui(
    agent: AbstractAgent[AgentDepsT, Any],
    run_input: RunAgentInput,
    accept: str = SSE_CONTENT_TYPE,
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    on_complete: OnCompleteFunc[BaseEvent] | None = None
) -> AsyncIterator[str]

```

Run the agent with the AG-UI run input and stream AG-UI protocol events.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `agent` | `AbstractAgent[AgentDepsT, Any]` | The agent to run. | *required* | | `run_input` | `RunAgentInput` | The AG-UI run input containing thread_id, run_id, messages, etc. | *required* | | `accept` | `str` | The accept header value for the run. | `SSE_CONTENT_TYPE` | | `output_type` | `OutputSpec[Any] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `on_complete` | `OnCompleteFunc[BaseEvent] | None` | Optional callback function called when the agent run completes successfully. The callback receives the completed AgentRunResult and can access all_messages() and other result data. | `None` |

Yields:

| Type | Description | | --- | --- | | `AsyncIterator[str]` | Streaming event chunks encoded as strings according to the accept header value. |

Source code in `pydantic_ai_slim/pydantic_ai/ag_ui.py`

```python
def run_ag_ui(
    agent: AbstractAgent[AgentDepsT, Any],
    run_input: RunAgentInput,
    accept: str = SSE_CONTENT_TYPE,
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    on_complete: OnCompleteFunc[BaseEvent] | None = None,
) -> AsyncIterator[str]:
    """Run the agent with the AG-UI run input and stream AG-UI protocol events.

    Args:
        agent: The agent to run.
        run_input: The AG-UI run input containing thread_id, run_id, messages, etc.
        accept: The accept header value for the run.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

    Yields:
        Streaming event chunks encoded as strings according to the accept header value.
    """
    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
    return adapter.encode_stream(
        adapter.run_stream(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            on_complete=on_complete,
        ),
    )

```

# `pydantic_ai.agent`

### Agent

Bases: `AbstractAgent[AgentDepsT, OutputDataT]`

Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.

Agents are generic in the dependency type they take AgentDepsT and the output type they return, OutputDataT.

By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.

Minimal usage example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.

```

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

````python
@dataclasses.dataclass(init=False)
class Agent(AbstractAgent[AgentDepsT, OutputDataT]):
    """Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.

    Agents are generic in the dependency type they take [`AgentDepsT`][pydantic_ai.tools.AgentDepsT]
    and the output type they return, [`OutputDataT`][pydantic_ai.output.OutputDataT].

    By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.

    Minimal usage example:

    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')
    result = agent.run_sync('What is the capital of France?')
    print(result.output)
    #> The capital of France is Paris.
    ```
    """

    _model: models.Model | models.KnownModelName | str | None

    _name: str | None
    end_strategy: EndStrategy
    """Strategy for handling tool calls when a final result is found."""

    model_settings: ModelSettings | None
    """Optional model request settings to use for this agents's runs, by default.

    Note, if `model_settings` is provided by `run`, `run_sync`, or `run_stream`, those settings will
    be merged with this value, with the runtime argument taking priority.
    """

    _output_type: OutputSpec[OutputDataT]

    instrument: InstrumentationSettings | bool | None
    """Options to automatically instrument with OpenTelemetry."""

    _instrument_default: ClassVar[InstrumentationSettings | bool] = False

    _deps_type: type[AgentDepsT] = dataclasses.field(repr=False)
    _output_schema: _output.OutputSchema[OutputDataT] = dataclasses.field(repr=False)
    _output_validators: list[_output.OutputValidator[AgentDepsT, OutputDataT]] = dataclasses.field(repr=False)
    _instructions: list[str | _system_prompt.SystemPromptFunc[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompts: tuple[str, ...] = dataclasses.field(repr=False)
    _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(
        repr=False
    )
    _function_toolset: FunctionToolset[AgentDepsT] = dataclasses.field(repr=False)
    _output_toolset: OutputToolset[AgentDepsT] | None = dataclasses.field(repr=False)
    _user_toolsets: list[AbstractToolset[AgentDepsT]] = dataclasses.field(repr=False)
    _prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = dataclasses.field(repr=False)
    _prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = dataclasses.field(repr=False)
    _max_result_retries: int = dataclasses.field(repr=False)
    _max_tool_retries: int = dataclasses.field(repr=False)

    _event_stream_handler: EventStreamHandler[AgentDepsT] | None = dataclasses.field(repr=False)

    _enter_lock: Lock = dataclasses.field(repr=False)
    _entered_count: int = dataclasses.field(repr=False)
    _exit_stack: AsyncExitStack | None = dataclasses.field(repr=False)

    @overload
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: Instructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AbstractBuiltinTool] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> None: ...

    @overload
    @deprecated('`mcp_servers` is deprecated, use `toolsets` instead.')
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: Instructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AbstractBuiltinTool] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        mcp_servers: Sequence[MCPServer] = (),
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> None: ...

    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: Instructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AbstractBuiltinTool] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Any,
    ):
        """Create an agent.

        Args:
            model: The default model to use for this agent, if not provided,
                you must provide the model when calling it. We allow `str` here since the actual list of allowed models changes frequently.
            output_type: The type of the output data, used to validate the data returned by the model,
                defaults to `str`.
            instructions: Instructions to use for this agent, you can also register instructions via a function with
                [`instructions`][pydantic_ai.Agent.instructions] or pass additional, temporary, instructions when executing a run.
            system_prompt: Static system prompts to use for this agent, you can also register system
                prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
            deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
                parameterize the agent, and therefore get the best out of static type checking.
                If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
                or add a type hint `: Agent[None, <return type>]`.
            name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
                when the agent is first run.
            model_settings: Optional model request settings to use for this agent's runs, by default.
            retries: The default number of retries to allow for tool calls and output validation, before raising an error.
                For model request retries, see the [HTTP Request Retries](../retries.md) documentation.
            output_retries: The maximum number of retries to allow for output validation, defaults to `retries`.
            tools: Tools to register with the agent, you can also register tools via the decorators
                [`@agent.tool`][pydantic_ai.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain].
            builtin_tools: The builtin tools that the agent will use. This depends on the model, as some models may not
                support certain tools. If the model doesn't support the builtin tools, an error will be raised.
            prepare_tools: Custom function to prepare the tool definition of all tools for each step, except output tools.
                This is useful if you want to customize the definition of multiple tools or you want to register
                a subset of tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
            prepare_output_tools: Custom function to prepare the tool definition of all output tools for each step.
                This is useful if you want to customize the definition of multiple output tools or you want to register
                a subset of output tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
            toolsets: Toolsets to register with the agent, including MCP servers and functions which take a run context
                and return a toolset. See [`ToolsetFunc`][pydantic_ai.toolsets.ToolsetFunc] for more information.
            defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
                it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
                which checks for the necessary environment variables. Set this to `false`
                to defer the evaluation until the first run. Useful if you want to
                [override the model][pydantic_ai.Agent.override] for testing.
            end_strategy: Strategy for handling tool calls that are requested alongside a final result.
                See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
            instrument: Set to True to automatically instrument with OpenTelemetry,
                which will use Logfire if it's configured.
                Set to an instance of [`InstrumentationSettings`][pydantic_ai.agent.InstrumentationSettings] to customize.
                If this isn't set, then the last value set by
                [`Agent.instrument_all()`][pydantic_ai.Agent.instrument_all]
                will be used, which defaults to False.
                See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
            history_processors: Optional list of callables to process the message history before sending it to the model.
                Each processor takes a list of messages and returns a modified list of messages.
                Processors can be sync or async and are applied in sequence.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools.
        """
        if model is None or defer_model_check:
            self._model = model
        else:
            self._model = models.infer_model(model)

        self._name = name
        self.end_strategy = end_strategy
        self.model_settings = model_settings

        self._output_type = output_type
        self.instrument = instrument
        self._deps_type = deps_type

        if mcp_servers := _deprecated_kwargs.pop('mcp_servers', None):
            if toolsets is not None:  # pragma: no cover
                raise TypeError('`mcp_servers` and `toolsets` cannot be set at the same time.')
            warnings.warn('`mcp_servers` is deprecated, use `toolsets` instead', DeprecationWarning)
            toolsets = mcp_servers

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        self._output_schema = _output.OutputSchema[OutputDataT].build(output_type)
        self._output_validators = []

        self._instructions = self._normalize_instructions(instructions)

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._system_prompt_functions = []
        self._system_prompt_dynamic_functions = {}

        self._max_result_retries = output_retries if output_retries is not None else retries
        self._max_tool_retries = retries

        self._builtin_tools = builtin_tools

        self._prepare_tools = prepare_tools
        self._prepare_output_tools = prepare_output_tools

        self._output_toolset = self._output_schema.toolset
        if self._output_toolset:
            self._output_toolset.max_retries = self._max_result_retries

        self._function_toolset = _AgentFunctionToolset(
            tools, max_retries=self._max_tool_retries, output_schema=self._output_schema
        )
        self._dynamic_toolsets = [
            DynamicToolset[AgentDepsT](toolset_func=toolset)
            for toolset in toolsets or []
            if not isinstance(toolset, AbstractToolset)
        ]
        self._user_toolsets = [toolset for toolset in toolsets or [] if isinstance(toolset, AbstractToolset)]

        self.history_processors = history_processors or []

        self._event_stream_handler = event_stream_handler

        self._override_name: ContextVar[_utils.Option[str]] = ContextVar('_override_name', default=None)
        self._override_deps: ContextVar[_utils.Option[AgentDepsT]] = ContextVar('_override_deps', default=None)
        self._override_model: ContextVar[_utils.Option[models.Model]] = ContextVar('_override_model', default=None)
        self._override_toolsets: ContextVar[_utils.Option[Sequence[AbstractToolset[AgentDepsT]]]] = ContextVar(
            '_override_toolsets', default=None
        )
        self._override_tools: ContextVar[
            _utils.Option[Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]]]
        ] = ContextVar('_override_tools', default=None)
        self._override_instructions: ContextVar[
            _utils.Option[list[str | _system_prompt.SystemPromptFunc[AgentDepsT]]]
        ] = ContextVar('_override_instructions', default=None)

        self._enter_lock = Lock()
        self._entered_count = 0
        self._exit_stack = None

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the instrumentation options for all agents where `instrument` is not set."""
        Agent._instrument_default = instrument

    @property
    def model(self) -> models.Model | models.KnownModelName | str | None:
        """The default model configured for this agent."""
        return self._model

    @model.setter
    def model(self, value: models.Model | models.KnownModelName | str | None) -> None:
        """Set the default model configured for this agent.

        We allow `str` here since the actual list of allowed models changes frequently.
        """
        self._model = value

    @property
    def name(self) -> str | None:
        """The name of the agent, used for logging.

        If `None`, we try to infer the agent name from the call frame when the agent is first run.
        """
        name_ = self._override_name.get()
        return name_.value if name_ else self._name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the name of the agent, used for logging."""
        self._name = value

    @property
    def deps_type(self) -> type:
        """The type of dependencies used by the agent."""
        return self._deps_type

    @property
    def output_type(self) -> OutputSpec[OutputDataT]:
        """The type of data output by agent runs, used to validate the data returned by the model, defaults to `str`."""
        return self._output_type

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        """Optional handler for events from the model's streaming response and the agent's execution of tools."""
        return self._event_stream_handler

    def __repr__(self) -> str:
        return f'{type(self).__name__}(model={self.model!r}, name={self.name!r}, end_strategy={self.end_strategy!r}, model_settings={self.model_settings!r}, output_type={self.output_type!r}, instrument={self.instrument!r})'

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
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

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        model_used = self._get_model(model)
        del model

        deps = self._get_deps(deps)
        output_schema = self._prepare_output_schema(output_type)

        output_type_ = output_type or self.output_type

        # We consider it a user error if a user tries to restrict the result type while having an output validator that
        # may change the result type from the restricted type to something else. Therefore, we consider the following
        # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
        output_validators = self._output_validators

        output_toolset = self._output_toolset
        if output_schema != self._output_schema or output_validators:
            output_toolset = output_schema.toolset
            if output_toolset:
                output_toolset.max_retries = self._max_result_retries
                output_toolset.output_validators = output_validators
        toolset = self._get_toolset(output_toolset=output_toolset, additional_toolsets=toolsets)
        tool_manager = ToolManager[AgentDepsT](toolset)

        # Build the graph
        graph = _agent_graph.build_agent_graph(self.name, self._deps_type, output_type_)

        # Build the initial state
        usage = usage or _usage.RunUsage()
        state = _agent_graph.GraphAgentState(
            message_history=list(message_history) if message_history else [],
            usage=usage,
            retries=0,
            run_step=0,
        )

        # Merge model settings in order of precedence: run > agent > model
        merged_settings = merge_model_settings(model_used.settings, self.model_settings)
        model_settings = merge_model_settings(merged_settings, model_settings)
        usage_limits = usage_limits or _usage.UsageLimits()

        instructions_literal, instructions_functions = self._get_instructions(additional_instructions=instructions)

        async def get_instructions(run_context: RunContext[AgentDepsT]) -> str | None:
            parts = [
                instructions_literal,
                *[await func.run(run_context) for func in instructions_functions],
            ]

            parts = [p for p in parts if p]
            if not parts:
                return None
            return '\n\n'.join(parts).strip()

        if isinstance(model_used, InstrumentedModel):
            instrumentation_settings = model_used.instrumentation_settings
            tracer = model_used.instrumentation_settings.tracer
        else:
            instrumentation_settings = None
            tracer = NoOpTracer()

        graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, OutputDataT](
            user_deps=deps,
            prompt=user_prompt,
            new_message_index=len(message_history) if message_history else 0,
            model=model_used,
            model_settings=model_settings,
            usage_limits=usage_limits,
            max_result_retries=self._max_result_retries,
            end_strategy=self.end_strategy,
            output_schema=output_schema,
            output_validators=output_validators,
            history_processors=self.history_processors,
            builtin_tools=[*self._builtin_tools, *(builtin_tools or [])],
            tool_manager=tool_manager,
            tracer=tracer,
            get_instructions=get_instructions,
            instrumentation_settings=instrumentation_settings,
        )

        user_prompt_node = _agent_graph.UserPromptNode[AgentDepsT](
            user_prompt=user_prompt,
            deferred_tool_results=deferred_tool_results,
            instructions=instructions_literal,
            instructions_functions=instructions_functions,
            system_prompts=self._system_prompts,
            system_prompt_functions=self._system_prompt_functions,
            system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
        )

        agent_name = self.name or 'agent'
        instrumentation_names = InstrumentationNames.for_version(
            instrumentation_settings.version if instrumentation_settings else DEFAULT_INSTRUMENTATION_VERSION
        )

        run_span = tracer.start_span(
            instrumentation_names.get_agent_run_span_name(agent_name),
            attributes={
                'model_name': model_used.model_name if model_used else 'no-model',
                'agent_name': agent_name,
                'gen_ai.agent.name': agent_name,
                'logfire.msg': f'{agent_name} run',
            },
        )

        try:
            async with graph.iter(
                inputs=user_prompt_node,
                state=state,
                deps=graph_deps,
                span=use_span(run_span) if run_span.is_recording() else None,
                infer_name=False,
            ) as graph_run:
                async with toolset:
                    agent_run = AgentRun(graph_run)
                    yield agent_run
                    if (final_result := agent_run.result) is not None and run_span.is_recording():
                        if instrumentation_settings and instrumentation_settings.include_content:
                            run_span.set_attribute(
                                'final_result',
                                (
                                    final_result.output
                                    if isinstance(final_result.output, str)
                                    else json.dumps(InstrumentedModel.serialize_any(final_result.output))
                                ),
                            )
        finally:
            try:
                if instrumentation_settings and run_span.is_recording():
                    run_span.set_attributes(
                        self._run_span_end_attributes(
                            instrumentation_settings, usage, state.message_history, graph_deps.new_message_index
                        )
                    )
            finally:
                run_span.end()

    def _run_span_end_attributes(
        self,
        settings: InstrumentationSettings,
        usage: _usage.RunUsage,
        message_history: list[_messages.ModelMessage],
        new_message_index: int,
    ):
        if settings.version == 1:
            attrs = {
                'all_messages_events': json.dumps(
                    [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(message_history)]
                )
            }
        else:
            # Store the last instructions here for convenience
            last_instructions = InstrumentedModel._get_instructions(message_history)  # pyright: ignore[reportPrivateUsage]
            attrs: dict[str, Any] = {
                'pydantic_ai.all_messages': json.dumps(settings.messages_to_otel_messages(list(message_history))),
                **settings.system_instructions_attributes(last_instructions),
            }

            # If this agent run was provided with existing history, store an attribute indicating the point at which the
            # new messages begin.
            if new_message_index > 0:
                attrs['pydantic_ai.new_message_index'] = new_message_index

            # If the instructions for this agent run were not always the same, store an attribute that indicates that.
            # This can signal to an observability UI that different steps in the agent run had different instructions.
            # Note: We purposely only look at "new" messages because they are the only ones produced by this agent run.
            if any(
                (
                    isinstance(m, _messages.ModelRequest)
                    and m.instructions is not None
                    and m.instructions != last_instructions
                )
                for m in message_history[new_message_index:]
            ):
                attrs['pydantic_ai.variable_instructions'] = True

        return {
            **usage.opentelemetry_attributes(),
            **attrs,
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {
                        **{k: {'type': 'array'} if isinstance(v, str) else {} for k, v in attrs.items()},
                        'final_result': {'type': 'object'},
                    },
                }
            ),
        }

    @contextmanager
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            instructions: The instructions to use instead of the instructions registered with the agent.
        """
        if _utils.is_set(name):
            name_token = self._override_name.set(_utils.Some(name))
        else:
            name_token = None

        if _utils.is_set(deps):
            deps_token = self._override_deps.set(_utils.Some(deps))
        else:
            deps_token = None

        if _utils.is_set(model):
            model_token = self._override_model.set(_utils.Some(models.infer_model(model)))
        else:
            model_token = None

        if _utils.is_set(toolsets):
            toolsets_token = self._override_toolsets.set(_utils.Some(toolsets))
        else:
            toolsets_token = None

        if _utils.is_set(tools):
            tools_token = self._override_tools.set(_utils.Some(tools))
        else:
            tools_token = None

        if _utils.is_set(instructions):
            normalized_instructions = self._normalize_instructions(instructions)
            instructions_token = self._override_instructions.set(_utils.Some(normalized_instructions))
        else:
            instructions_token = None

        try:
            yield
        finally:
            if name_token is not None:
                self._override_name.reset(name_token)
            if deps_token is not None:
                self._override_deps.reset(deps_token)
            if model_token is not None:
                self._override_model.reset(model_token)
            if toolsets_token is not None:
                self._override_toolsets.reset(toolsets_token)
            if tools_token is not None:
                self._override_tools.reset(tools_token)
            if instructions_token is not None:
                self._override_instructions.reset(instructions_token)

    @overload
    def instructions(
        self, func: Callable[[RunContext[AgentDepsT]], str], /
    ) -> Callable[[RunContext[AgentDepsT]], str]: ...

    @overload
    def instructions(
        self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str]], /
    ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]: ...

    @overload
    def instructions(self, func: Callable[[], str], /) -> Callable[[], str]: ...

    @overload
    def instructions(self, func: Callable[[], Awaitable[str]], /) -> Callable[[], Awaitable[str]]: ...

    @overload
    def instructions(
        self, /
    ) -> Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]: ...

    def instructions(
        self,
        func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
        /,
    ) -> (
        Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
        | _system_prompt.SystemPromptFunc[AgentDepsT]
    ):
        """Decorator to register an instructions function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
        Can decorate a sync or async functions.

        The decorator can be used bare (`agent.instructions`).

        Overloads for every possible signature of `instructions` are included so the decorator doesn't obscure
        the type of the function.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=str)

        @agent.instructions
        def simple_instructions() -> str:
            return 'foobar'

        @agent.instructions
        async def async_instructions(ctx: RunContext[str]) -> str:
            return f'{ctx.deps} is the best'
        ```
        """
        if func is None:

            def decorator(
                func_: _system_prompt.SystemPromptFunc[AgentDepsT],
            ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
                self._instructions.append(func_)
                return func_

            return decorator
        else:
            self._instructions.append(func)
            return func

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], str], /
    ) -> Callable[[RunContext[AgentDepsT]], str]: ...

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str]], /
    ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]: ...

    @overload
    def system_prompt(self, func: Callable[[], str], /) -> Callable[[], str]: ...

    @overload
    def system_prompt(self, func: Callable[[], Awaitable[str]], /) -> Callable[[], Awaitable[str]]: ...

    @overload
    def system_prompt(
        self, /, *, dynamic: bool = False
    ) -> Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]: ...

    def system_prompt(
        self,
        func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
        /,
        *,
        dynamic: bool = False,
    ) -> (
        Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
        | _system_prompt.SystemPromptFunc[AgentDepsT]
    ):
        """Decorator to register a system prompt function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
        Can decorate a sync or async functions.

        The decorator can be used either bare (`agent.system_prompt`) or as a function call
        (`agent.system_prompt(...)`), see the examples below.

        Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Args:
            func: The function to decorate
            dynamic: If True, the system prompt will be reevaluated even when `messages_history` is provided,
                see [`SystemPromptPart.dynamic_ref`][pydantic_ai.messages.SystemPromptPart.dynamic_ref]

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=str)

        @agent.system_prompt
        def simple_system_prompt() -> str:
            return 'foobar'

        @agent.system_prompt(dynamic=True)
        async def async_system_prompt(ctx: RunContext[str]) -> str:
            return f'{ctx.deps} is the best'
        ```
        """
        if func is None:

            def decorator(
                func_: _system_prompt.SystemPromptFunc[AgentDepsT],
            ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
                runner = _system_prompt.SystemPromptRunner[AgentDepsT](func_, dynamic=dynamic)
                self._system_prompt_functions.append(runner)
                if dynamic:  # pragma: lax no cover
                    self._system_prompt_dynamic_functions[func_.__qualname__] = runner
                return func_

            return decorator
        else:
            assert not dynamic, "dynamic can't be True in this case"
            self._system_prompt_functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](func, dynamic=dynamic))
            return func

    @overload
    def output_validator(
        self, func: Callable[[RunContext[AgentDepsT], OutputDataT], OutputDataT], /
    ) -> Callable[[RunContext[AgentDepsT], OutputDataT], OutputDataT]: ...

    @overload
    def output_validator(
        self, func: Callable[[RunContext[AgentDepsT], OutputDataT], Awaitable[OutputDataT]], /
    ) -> Callable[[RunContext[AgentDepsT], OutputDataT], Awaitable[OutputDataT]]: ...

    @overload
    def output_validator(
        self, func: Callable[[OutputDataT], OutputDataT], /
    ) -> Callable[[OutputDataT], OutputDataT]: ...

    @overload
    def output_validator(
        self, func: Callable[[OutputDataT], Awaitable[OutputDataT]], /
    ) -> Callable[[OutputDataT], Awaitable[OutputDataT]]: ...

    def output_validator(
        self, func: _output.OutputValidatorFunc[AgentDepsT, OutputDataT], /
    ) -> _output.OutputValidatorFunc[AgentDepsT, OutputDataT]:
        """Decorator to register an output validator function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
        Can decorate a sync or async functions.

        Overloads for every possible signature of `output_validator` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Example:
        ```python
        from pydantic_ai import Agent, ModelRetry, RunContext

        agent = Agent('test', deps_type=str)

        @agent.output_validator
        def output_validator_simple(data: str) -> str:
            if 'wrong' in data:
                raise ModelRetry('wrong response')
            return data

        @agent.output_validator
        async def output_validator_deps(ctx: RunContext[str], data: str) -> str:
            if ctx.deps in data:
                raise ModelRetry('wrong response')
            return data

        result = agent.run_sync('foobar', deps='spam')
        print(result.output)
        #> success (no tool calls)
        ```
        """
        self._output_validators.append(_output.OutputValidator[AgentDepsT, Any](func))
        return func

    @overload
    def tool(self, func: ToolFuncContext[AgentDepsT, ToolParams], /) -> ToolFuncContext[AgentDepsT, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[ToolFuncContext[AgentDepsT, ToolParams]], ToolFuncContext[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=int)

        @agent.tool
        def foobar(ctx: RunContext[int], x: int) -> int:
            return ctx.deps + x

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str], y: float) -> float:
            return ctx.deps + y

        result = agent.run_sync('foobar', deps=1)
        print(result.output)
        #> {"foobar":1,"spam":1.0}
        ```

        Args:
            func: The tool function to register.
            name: The name of the tool, defaults to the function name.
            description: The description of the tool, defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
        """

        def tool_decorator(
            func_: ToolFuncContext[AgentDepsT, ToolParams],
        ) -> ToolFuncContext[AgentDepsT, ToolParams]:
            # noinspection PyTypeChecker
            self._function_toolset.add_function(
                func_,
                takes_ctx=True,
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

    @overload
    def tool_plain(self, func: ToolFuncPlain[ToolParams], /) -> ToolFuncPlain[ToolParams]: ...

    @overload
    def tool_plain(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]]: ...

    def tool_plain(
        self,
        func: ToolFuncPlain[ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Decorator to register a tool function which DOES NOT take `RunContext` as an argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test')

        @agent.tool
        def foobar(ctx: RunContext[int]) -> int:
            return 123

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str]) -> float:
            return 3.14

        result = agent.run_sync('foobar', deps=1)
        print(result.output)
        #> {"foobar":123,"spam":3.14}
        ```

        Args:
            func: The tool function to register.
            name: The name of the tool, defaults to the function name.
            description: The description of the tool, defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
        """

        def tool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
            # noinspection PyTypeChecker
            self._function_toolset.add_function(
                func_,
                takes_ctx=False,
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

    @overload
    def toolset(self, func: ToolsetFunc[AgentDepsT], /) -> ToolsetFunc[AgentDepsT]: ...

    @overload
    def toolset(
        self,
        /,
        *,
        per_run_step: bool = True,
    ) -> Callable[[ToolsetFunc[AgentDepsT]], ToolsetFunc[AgentDepsT]]: ...

    def toolset(
        self,
        func: ToolsetFunc[AgentDepsT] | None = None,
        /,
        *,
        per_run_step: bool = True,
    ) -> Any:
        """Decorator to register a toolset function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.

        Can decorate a sync or async functions.

        The decorator can be used bare (`agent.toolset`).

        Example:
        ```python
        from pydantic_ai import AbstractToolset, Agent, FunctionToolset, RunContext

        agent = Agent('test', deps_type=str)

        @agent.toolset
        async def simple_toolset(ctx: RunContext[str]) -> AbstractToolset[str]:
            return FunctionToolset()
        ```

        Args:
            func: The toolset function to register.
            per_run_step: Whether to re-evaluate the toolset for each run step. Defaults to True.
        """

        def toolset_decorator(func_: ToolsetFunc[AgentDepsT]) -> ToolsetFunc[AgentDepsT]:
            self._dynamic_toolsets.append(DynamicToolset(func_, per_run_step=per_run_step))
            return func_

        return toolset_decorator if func is None else toolset_decorator(func)

    def _get_model(self, model: models.Model | models.KnownModelName | str | None) -> models.Model:
        """Create a model configured for this agent.

        Args:
            model: model to use for this run, required if `model` was not set when creating the agent.

        Returns:
            The model used
        """
        model_: models.Model
        if some_model := self._override_model.get():
            # we don't want `override()` to cover up errors from the model not being defined, hence this check
            if model is None and self.model is None:
                raise exceptions.UserError(
                    '`model` must either be set on the agent or included when calling it. '
                    '(Even when `override(model=...)` is customizing the model that will actually be called)'
                )
            model_ = some_model.value
        elif model is not None:
            model_ = models.infer_model(model)
        elif self.model is not None:
            # noinspection PyTypeChecker
            model_ = self.model = models.infer_model(self.model)
        else:
            raise exceptions.UserError('`model` must either be set on the agent or included when calling it.')

        instrument = self.instrument
        if instrument is None:
            instrument = self._instrument_default

        return instrument_model(model_, instrument)

    def _get_deps(self: Agent[T, OutputDataT], deps: T) -> T:
        """Get deps for a run.

        If we've overridden deps via `_override_deps`, use that, otherwise use the deps passed to the call.

        We could do runtime type checking of deps against `self._deps_type`, but that's a slippery slope.
        """
        if some_deps := self._override_deps.get():
            return some_deps.value
        else:
            return deps

    def _normalize_instructions(
        self,
        instructions: Instructions[AgentDepsT],
    ) -> list[str | _system_prompt.SystemPromptFunc[AgentDepsT]]:
        if instructions is None:
            return []
        if isinstance(instructions, str) or callable(instructions):
            return [instructions]
        return list(instructions)

    def _get_instructions(
        self,
        additional_instructions: Instructions[AgentDepsT] = None,
    ) -> tuple[str | None, list[_system_prompt.SystemPromptRunner[AgentDepsT]]]:
        override_instructions = self._override_instructions.get()
        if override_instructions:
            instructions = override_instructions.value
        else:
            instructions = self._instructions.copy()
            if additional_instructions is not None:
                instructions.extend(self._normalize_instructions(additional_instructions))

        literal_parts: list[str] = []
        functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = []

        for instruction in instructions:
            if isinstance(instruction, str):
                literal_parts.append(instruction)
            else:
                functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](instruction))

        literal = '\n'.join(literal_parts).strip() or None
        return literal, functions

    def _get_toolset(
        self,
        output_toolset: AbstractToolset[AgentDepsT] | None | _utils.Unset = _utils.UNSET,
        additional_toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AbstractToolset[AgentDepsT]:
        """Get the complete toolset.

        Args:
            output_toolset: The output toolset to use instead of the one built at agent construction time.
            additional_toolsets: Additional toolsets to add, unless toolsets have been overridden.
        """
        toolsets = self.toolsets
        # Don't add additional toolsets if the toolsets have been overridden
        if additional_toolsets and self._override_toolsets.get() is None:
            toolsets = [*toolsets, *additional_toolsets]

        toolset = CombinedToolset(toolsets)

        # Copy the dynamic toolsets to ensure each run has its own instances
        def copy_dynamic_toolsets(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            if isinstance(toolset, DynamicToolset):
                return dataclasses.replace(toolset)
            else:
                return toolset

        toolset = toolset.visit_and_replace(copy_dynamic_toolsets)

        if self._prepare_tools:
            toolset = PreparedToolset(toolset, self._prepare_tools)

        output_toolset = output_toolset if _utils.is_set(output_toolset) else self._output_toolset
        if output_toolset is not None:
            if self._prepare_output_tools:
                output_toolset = PreparedToolset(output_toolset, self._prepare_output_tools)
            toolset = CombinedToolset([output_toolset, toolset])

        return toolset

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        """All toolsets registered on the agent, including a function toolset holding tools that were registered on the agent directly.

        Output tools are not included.
        """
        toolsets: list[AbstractToolset[AgentDepsT]] = []

        if some_tools := self._override_tools.get():
            function_toolset = _AgentFunctionToolset(
                some_tools.value, max_retries=self._max_tool_retries, output_schema=self._output_schema
            )
        else:
            function_toolset = self._function_toolset
        toolsets.append(function_toolset)

        if some_user_toolsets := self._override_toolsets.get():
            user_toolsets = some_user_toolsets.value
        else:
            user_toolsets = [*self._user_toolsets, *self._dynamic_toolsets]
        toolsets.extend(user_toolsets)

        return toolsets

    @overload
    def _prepare_output_schema(self, output_type: None) -> _output.OutputSchema[OutputDataT]: ...

    @overload
    def _prepare_output_schema(
        self, output_type: OutputSpec[RunOutputDataT]
    ) -> _output.OutputSchema[RunOutputDataT]: ...

    def _prepare_output_schema(self, output_type: OutputSpec[Any] | None) -> _output.OutputSchema[Any]:
        if output_type is not None:
            if self._output_validators:
                raise exceptions.UserError('Cannot set a custom run `output_type` when the agent has output validators')
            schema = _output.OutputSchema.build(output_type)
        else:
            schema = self._output_schema

        return schema

    async def __aenter__(self) -> Self:
        """Enter the agent context.

        This will start all [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] registered as `toolsets` so they are ready to be used.

        This is a no-op if the agent has already been entered.
        """
        async with self._enter_lock:
            if self._entered_count == 0:
                async with AsyncExitStack() as exit_stack:
                    toolset = self._get_toolset()
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

    def set_mcp_sampling_model(self, model: models.Model | models.KnownModelName | str | None = None) -> None:
        """Set the sampling model on all MCP servers registered with the agent.

        If no sampling model is provided, the agent's model will be used.
        """
        try:
            sampling_model = models.infer_model(model) if model else self._get_model(None)
        except exceptions.UserError as e:
            raise exceptions.UserError('No sampling model provided and no model set on the agent.') from e

        from ..mcp import MCPServer

        def _set_sampling_model(toolset: AbstractToolset[AgentDepsT]) -> None:
            if isinstance(toolset, MCPServer):
                toolset.sampling_model = sampling_model

        self._get_toolset().apply(_set_sampling_model)

    @asynccontextmanager
    @deprecated(
        '`run_mcp_servers` is deprecated, use `async with agent:` instead. If you need to set a sampling model on all MCP servers, use `agent.set_mcp_sampling_model()`.'
    )
    async def run_mcp_servers(
        self, model: models.Model | models.KnownModelName | str | None = None
    ) -> AsyncIterator[None]:
        """Run [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] so they can be used by the agent.

        Deprecated: use [`async with agent`][pydantic_ai.agent.Agent.__aenter__] instead.
        If you need to set a sampling model on all MCP servers, use [`agent.set_mcp_sampling_model()`][pydantic_ai.agent.Agent.set_mcp_sampling_model].

        Returns: a context manager to start and shutdown the servers.
        """
        try:
            self.set_mcp_sampling_model(model)
        except exceptions.UserError:
            if model is not None:
                raise

        async with self:
            yield

````

#### __init__

```python
__init__(
    model: Model | KnownModelName | str | None = None,
    *,
    output_type: OutputSpec[OutputDataT] = str,
    instructions: Instructions[AgentDepsT] = None,
    system_prompt: str | Sequence[str] = (),
    deps_type: type[AgentDepsT] = NoneType,
    name: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 1,
    output_retries: int | None = None,
    tools: Sequence[
        Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]
    ] = (),
    builtin_tools: Sequence[AbstractBuiltinTool] = (),
    prepare_tools: (
        ToolsPrepareFunc[AgentDepsT] | None
    ) = None,
    prepare_output_tools: (
        ToolsPrepareFunc[AgentDepsT] | None
    ) = None,
    toolsets: (
        Sequence[
            AbstractToolset[AgentDepsT]
            | ToolsetFunc[AgentDepsT]
        ]
        | None
    ) = None,
    defer_model_check: bool = False,
    end_strategy: EndStrategy = "early",
    instrument: (
        InstrumentationSettings | bool | None
    ) = None,
    history_processors: (
        Sequence[HistoryProcessor[AgentDepsT]] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> None

```

```python
__init__(
    model: Model | KnownModelName | str | None = None,
    *,
    output_type: OutputSpec[OutputDataT] = str,
    instructions: Instructions[AgentDepsT] = None,
    system_prompt: str | Sequence[str] = (),
    deps_type: type[AgentDepsT] = NoneType,
    name: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 1,
    output_retries: int | None = None,
    tools: Sequence[
        Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]
    ] = (),
    builtin_tools: Sequence[AbstractBuiltinTool] = (),
    prepare_tools: (
        ToolsPrepareFunc[AgentDepsT] | None
    ) = None,
    prepare_output_tools: (
        ToolsPrepareFunc[AgentDepsT] | None
    ) = None,
    mcp_servers: Sequence[MCPServer] = (),
    defer_model_check: bool = False,
    end_strategy: EndStrategy = "early",
    instrument: (
        InstrumentationSettings | bool | None
    ) = None,
    history_processors: (
        Sequence[HistoryProcessor[AgentDepsT]] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> None

```

```python
__init__(
    model: Model | KnownModelName | str | None = None,
    *,
    output_type: OutputSpec[OutputDataT] = str,
    instructions: Instructions[AgentDepsT] = None,
    system_prompt: str | Sequence[str] = (),
    deps_type: type[AgentDepsT] = NoneType,
    name: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 1,
    output_retries: int | None = None,
    tools: Sequence[
        Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]
    ] = (),
    builtin_tools: Sequence[AbstractBuiltinTool] = (),
    prepare_tools: (
        ToolsPrepareFunc[AgentDepsT] | None
    ) = None,
    prepare_output_tools: (
        ToolsPrepareFunc[AgentDepsT] | None
    ) = None,
    toolsets: (
        Sequence[
            AbstractToolset[AgentDepsT]
            | ToolsetFunc[AgentDepsT]
        ]
        | None
    ) = None,
    defer_model_check: bool = False,
    end_strategy: EndStrategy = "early",
    instrument: (
        InstrumentationSettings | bool | None
    ) = None,
    history_processors: (
        Sequence[HistoryProcessor[AgentDepsT]] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None,
    **_deprecated_kwargs: Any
)

```

Create an agent.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `model` | `Model | KnownModelName | str | None` | The default model to use for this agent, if not provided, you must provide the model when calling it. We allow str here since the actual list of allowed models changes frequently. | `None` | | `output_type` | `OutputSpec[OutputDataT]` | The type of the output data, used to validate the data returned by the model, defaults to str. | `str` | | `instructions` | `Instructions[AgentDepsT]` | Instructions to use for this agent, you can also register instructions via a function with instructions or pass additional, temporary, instructions when executing a run. | `None` | | `system_prompt` | `str | Sequence[str]` | Static system prompts to use for this agent, you can also register system prompts via a function with system_prompt. | `()` | | `deps_type` | `type[AgentDepsT]` | The type used for dependency injection, this parameter exists solely to allow you to fully parameterize the agent, and therefore get the best out of static type checking. If you're not using deps, but want type checking to pass, you can set deps=None to satisfy Pyright or add a type hint : Agent\[None, <return type>\]. | `NoneType` | | `name` | `str | None` | The name of the agent, used for logging. If None, we try to infer the agent name from the call frame when the agent is first run. | `None` | | `model_settings` | `ModelSettings | None` | Optional model request settings to use for this agent's runs, by default. | `None` | | `retries` | `int` | The default number of retries to allow for tool calls and output validation, before raising an error. For model request retries, see the HTTP Request Retries documentation. | `1` | | `output_retries` | `int | None` | The maximum number of retries to allow for output validation, defaults to retries. | `None` | | `tools` | `Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]]` | Tools to register with the agent, you can also register tools via the decorators @agent.tool and @agent.tool_plain. | `()` | | `builtin_tools` | `Sequence[AbstractBuiltinTool]` | The builtin tools that the agent will use. This depends on the model, as some models may not support certain tools. If the model doesn't support the builtin tools, an error will be raised. | `()` | | `prepare_tools` | `ToolsPrepareFunc[AgentDepsT] | None` | Custom function to prepare the tool definition of all tools for each step, except output tools. This is useful if you want to customize the definition of multiple tools or you want to register a subset of tools for a given step. See ToolsPrepareFunc | `None` | | `prepare_output_tools` | `ToolsPrepareFunc[AgentDepsT] | None` | Custom function to prepare the tool definition of all output tools for each step. This is useful if you want to customize the definition of multiple output tools or you want to register a subset of output tools for a given step. See ToolsPrepareFunc | `None` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT]] | None` | Toolsets to register with the agent, including MCP servers and functions which take a run context and return a toolset. See ToolsetFunc for more information. | `None` | | `defer_model_check` | `bool` | by default, if you provide a named model, it's evaluated to create a Model instance immediately, which checks for the necessary environment variables. Set this to false to defer the evaluation until the first run. Useful if you want to override the model for testing. | `False` | | `end_strategy` | `EndStrategy` | Strategy for handling tool calls that are requested alongside a final result. See EndStrategy for more information. | `'early'` | | `instrument` | `InstrumentationSettings | bool | None` | Set to True to automatically instrument with OpenTelemetry, which will use Logfire if it's configured. Set to an instance of InstrumentationSettings to customize. If this isn't set, then the last value set by Agent.instrument_all() will be used, which defaults to False. See the Debugging and Monitoring guide for more info. | `None` | | `history_processors` | `Sequence[HistoryProcessor[AgentDepsT]] | None` | Optional list of callables to process the message history before sending it to the model. Each processor takes a list of messages and returns a modified list of messages. Processors can be sync or async and are applied in sequence. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional handler for events from the model's streaming response and the agent's execution of tools. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

```python
def __init__(
    self,
    model: models.Model | models.KnownModelName | str | None = None,
    *,
    output_type: OutputSpec[OutputDataT] = str,
    instructions: Instructions[AgentDepsT] = None,
    system_prompt: str | Sequence[str] = (),
    deps_type: type[AgentDepsT] = NoneType,
    name: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 1,
    output_retries: int | None = None,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
    builtin_tools: Sequence[AbstractBuiltinTool] = (),
    prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
    prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
    toolsets: Sequence[AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT]] | None = None,
    defer_model_check: bool = False,
    end_strategy: EndStrategy = 'early',
    instrument: InstrumentationSettings | bool | None = None,
    history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    **_deprecated_kwargs: Any,
):
    """Create an agent.

    Args:
        model: The default model to use for this agent, if not provided,
            you must provide the model when calling it. We allow `str` here since the actual list of allowed models changes frequently.
        output_type: The type of the output data, used to validate the data returned by the model,
            defaults to `str`.
        instructions: Instructions to use for this agent, you can also register instructions via a function with
            [`instructions`][pydantic_ai.Agent.instructions] or pass additional, temporary, instructions when executing a run.
        system_prompt: Static system prompts to use for this agent, you can also register system
            prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
        deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
            parameterize the agent, and therefore get the best out of static type checking.
            If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
            or add a type hint `: Agent[None, <return type>]`.
        name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
            when the agent is first run.
        model_settings: Optional model request settings to use for this agent's runs, by default.
        retries: The default number of retries to allow for tool calls and output validation, before raising an error.
            For model request retries, see the [HTTP Request Retries](../retries.md) documentation.
        output_retries: The maximum number of retries to allow for output validation, defaults to `retries`.
        tools: Tools to register with the agent, you can also register tools via the decorators
            [`@agent.tool`][pydantic_ai.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain].
        builtin_tools: The builtin tools that the agent will use. This depends on the model, as some models may not
            support certain tools. If the model doesn't support the builtin tools, an error will be raised.
        prepare_tools: Custom function to prepare the tool definition of all tools for each step, except output tools.
            This is useful if you want to customize the definition of multiple tools or you want to register
            a subset of tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
        prepare_output_tools: Custom function to prepare the tool definition of all output tools for each step.
            This is useful if you want to customize the definition of multiple output tools or you want to register
            a subset of output tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
        toolsets: Toolsets to register with the agent, including MCP servers and functions which take a run context
            and return a toolset. See [`ToolsetFunc`][pydantic_ai.toolsets.ToolsetFunc] for more information.
        defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
            it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
            which checks for the necessary environment variables. Set this to `false`
            to defer the evaluation until the first run. Useful if you want to
            [override the model][pydantic_ai.Agent.override] for testing.
        end_strategy: Strategy for handling tool calls that are requested alongside a final result.
            See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
        instrument: Set to True to automatically instrument with OpenTelemetry,
            which will use Logfire if it's configured.
            Set to an instance of [`InstrumentationSettings`][pydantic_ai.agent.InstrumentationSettings] to customize.
            If this isn't set, then the last value set by
            [`Agent.instrument_all()`][pydantic_ai.Agent.instrument_all]
            will be used, which defaults to False.
            See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
        history_processors: Optional list of callables to process the message history before sending it to the model.
            Each processor takes a list of messages and returns a modified list of messages.
            Processors can be sync or async and are applied in sequence.
        event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools.
    """
    if model is None or defer_model_check:
        self._model = model
    else:
        self._model = models.infer_model(model)

    self._name = name
    self.end_strategy = end_strategy
    self.model_settings = model_settings

    self._output_type = output_type
    self.instrument = instrument
    self._deps_type = deps_type

    if mcp_servers := _deprecated_kwargs.pop('mcp_servers', None):
        if toolsets is not None:  # pragma: no cover
            raise TypeError('`mcp_servers` and `toolsets` cannot be set at the same time.')
        warnings.warn('`mcp_servers` is deprecated, use `toolsets` instead', DeprecationWarning)
        toolsets = mcp_servers

    _utils.validate_empty_kwargs(_deprecated_kwargs)

    self._output_schema = _output.OutputSchema[OutputDataT].build(output_type)
    self._output_validators = []

    self._instructions = self._normalize_instructions(instructions)

    self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
    self._system_prompt_functions = []
    self._system_prompt_dynamic_functions = {}

    self._max_result_retries = output_retries if output_retries is not None else retries
    self._max_tool_retries = retries

    self._builtin_tools = builtin_tools

    self._prepare_tools = prepare_tools
    self._prepare_output_tools = prepare_output_tools

    self._output_toolset = self._output_schema.toolset
    if self._output_toolset:
        self._output_toolset.max_retries = self._max_result_retries

    self._function_toolset = _AgentFunctionToolset(
        tools, max_retries=self._max_tool_retries, output_schema=self._output_schema
    )
    self._dynamic_toolsets = [
        DynamicToolset[AgentDepsT](toolset_func=toolset)
        for toolset in toolsets or []
        if not isinstance(toolset, AbstractToolset)
    ]
    self._user_toolsets = [toolset for toolset in toolsets or [] if isinstance(toolset, AbstractToolset)]

    self.history_processors = history_processors or []

    self._event_stream_handler = event_stream_handler

    self._override_name: ContextVar[_utils.Option[str]] = ContextVar('_override_name', default=None)
    self._override_deps: ContextVar[_utils.Option[AgentDepsT]] = ContextVar('_override_deps', default=None)
    self._override_model: ContextVar[_utils.Option[models.Model]] = ContextVar('_override_model', default=None)
    self._override_toolsets: ContextVar[_utils.Option[Sequence[AbstractToolset[AgentDepsT]]]] = ContextVar(
        '_override_toolsets', default=None
    )
    self._override_tools: ContextVar[
        _utils.Option[Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]]]
    ] = ContextVar('_override_tools', default=None)
    self._override_instructions: ContextVar[
        _utils.Option[list[str | _system_prompt.SystemPromptFunc[AgentDepsT]]]
    ] = ContextVar('_override_instructions', default=None)

    self._enter_lock = Lock()
    self._entered_count = 0
    self._exit_stack = None

```

#### end_strategy

```python
end_strategy: EndStrategy = end_strategy

```

Strategy for handling tool calls when a final result is found.

#### model_settings

```python
model_settings: ModelSettings | None = model_settings

```

Optional model request settings to use for this agents's runs, by default.

Note, if `model_settings` is provided by `run`, `run_sync`, or `run_stream`, those settings will be merged with this value, with the runtime argument taking priority.

#### instrument

```python
instrument: InstrumentationSettings | bool | None = (
    instrument
)

```

Options to automatically instrument with OpenTelemetry.

#### instrument_all

```python
instrument_all(
    instrument: InstrumentationSettings | bool = True,
) -> None

```

Set the instrumentation options for all agents where `instrument` is not set.

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

```python
@staticmethod
def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
    """Set the instrumentation options for all agents where `instrument` is not set."""
    Agent._instrument_default = instrument

```

#### model

```python
model: Model | KnownModelName | str | None

```

The default model configured for this agent.

#### name

```python
name: str | None

```

The name of the agent, used for logging.

If `None`, we try to infer the agent name from the call frame when the agent is first run.

#### deps_type

```python
deps_type: type

```

The type of dependencies used by the agent.

#### output_type

```python
output_type: OutputSpec[OutputDataT]

```

The type of data output by agent runs, used to validate the data returned by the model, defaults to `str`.

#### event_stream_handler

```python
event_stream_handler: EventStreamHandler[AgentDepsT] | None

```

Optional handler for events from the model's streaming response and the agent's execution of tools.

#### iter

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, OutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, RunOutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]

```

A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the stream of events coming from the execution of tools.

The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics, and the final result of the run once it has completed.

For more details, see the documentation of `AgentRun`.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    nodes = []
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

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[Any] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentRun[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

````python
@asynccontextmanager
async def iter(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
    """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

    This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
    `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
    executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
    stream of events coming from the execution of tools.

    The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
    and the final result of the run once it has completed.

    For more details, see the documentation of `AgentRun`.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
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

    Args:
        user_prompt: User input to start/continue the conversation.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())

    model_used = self._get_model(model)
    del model

    deps = self._get_deps(deps)
    output_schema = self._prepare_output_schema(output_type)

    output_type_ = output_type or self.output_type

    # We consider it a user error if a user tries to restrict the result type while having an output validator that
    # may change the result type from the restricted type to something else. Therefore, we consider the following
    # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
    output_validators = self._output_validators

    output_toolset = self._output_toolset
    if output_schema != self._output_schema or output_validators:
        output_toolset = output_schema.toolset
        if output_toolset:
            output_toolset.max_retries = self._max_result_retries
            output_toolset.output_validators = output_validators
    toolset = self._get_toolset(output_toolset=output_toolset, additional_toolsets=toolsets)
    tool_manager = ToolManager[AgentDepsT](toolset)

    # Build the graph
    graph = _agent_graph.build_agent_graph(self.name, self._deps_type, output_type_)

    # Build the initial state
    usage = usage or _usage.RunUsage()
    state = _agent_graph.GraphAgentState(
        message_history=list(message_history) if message_history else [],
        usage=usage,
        retries=0,
        run_step=0,
    )

    # Merge model settings in order of precedence: run > agent > model
    merged_settings = merge_model_settings(model_used.settings, self.model_settings)
    model_settings = merge_model_settings(merged_settings, model_settings)
    usage_limits = usage_limits or _usage.UsageLimits()

    instructions_literal, instructions_functions = self._get_instructions(additional_instructions=instructions)

    async def get_instructions(run_context: RunContext[AgentDepsT]) -> str | None:
        parts = [
            instructions_literal,
            *[await func.run(run_context) for func in instructions_functions],
        ]

        parts = [p for p in parts if p]
        if not parts:
            return None
        return '\n\n'.join(parts).strip()

    if isinstance(model_used, InstrumentedModel):
        instrumentation_settings = model_used.instrumentation_settings
        tracer = model_used.instrumentation_settings.tracer
    else:
        instrumentation_settings = None
        tracer = NoOpTracer()

    graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, OutputDataT](
        user_deps=deps,
        prompt=user_prompt,
        new_message_index=len(message_history) if message_history else 0,
        model=model_used,
        model_settings=model_settings,
        usage_limits=usage_limits,
        max_result_retries=self._max_result_retries,
        end_strategy=self.end_strategy,
        output_schema=output_schema,
        output_validators=output_validators,
        history_processors=self.history_processors,
        builtin_tools=[*self._builtin_tools, *(builtin_tools or [])],
        tool_manager=tool_manager,
        tracer=tracer,
        get_instructions=get_instructions,
        instrumentation_settings=instrumentation_settings,
    )

    user_prompt_node = _agent_graph.UserPromptNode[AgentDepsT](
        user_prompt=user_prompt,
        deferred_tool_results=deferred_tool_results,
        instructions=instructions_literal,
        instructions_functions=instructions_functions,
        system_prompts=self._system_prompts,
        system_prompt_functions=self._system_prompt_functions,
        system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
    )

    agent_name = self.name or 'agent'
    instrumentation_names = InstrumentationNames.for_version(
        instrumentation_settings.version if instrumentation_settings else DEFAULT_INSTRUMENTATION_VERSION
    )

    run_span = tracer.start_span(
        instrumentation_names.get_agent_run_span_name(agent_name),
        attributes={
            'model_name': model_used.model_name if model_used else 'no-model',
            'agent_name': agent_name,
            'gen_ai.agent.name': agent_name,
            'logfire.msg': f'{agent_name} run',
        },
    )

    try:
        async with graph.iter(
            inputs=user_prompt_node,
            state=state,
            deps=graph_deps,
            span=use_span(run_span) if run_span.is_recording() else None,
            infer_name=False,
        ) as graph_run:
            async with toolset:
                agent_run = AgentRun(graph_run)
                yield agent_run
                if (final_result := agent_run.result) is not None and run_span.is_recording():
                    if instrumentation_settings and instrumentation_settings.include_content:
                        run_span.set_attribute(
                            'final_result',
                            (
                                final_result.output
                                if isinstance(final_result.output, str)
                                else json.dumps(InstrumentedModel.serialize_any(final_result.output))
                            ),
                        )
    finally:
        try:
            if instrumentation_settings and run_span.is_recording():
                run_span.set_attributes(
                    self._run_span_end_attributes(
                        instrumentation_settings, usage, state.message_history, graph_deps.new_message_index
                    )
                )
        finally:
            run_span.end()

````

#### override

```python
override(
    *,
    name: str | Unset = UNSET,
    deps: AgentDepsT | Unset = UNSET,
    model: Model | KnownModelName | str | Unset = UNSET,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | Unset
    ) = UNSET,
    tools: (
        Sequence[
            Tool[AgentDepsT]
            | ToolFuncEither[AgentDepsT, ...]
        ]
        | Unset
    ) = UNSET,
    instructions: Instructions[AgentDepsT] | Unset = UNSET
) -> Iterator[None]

```

Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

This is particularly useful when testing. You can find an example of this [here](../../testing/#overriding-model-via-pytest-fixtures).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | Unset` | The name to use instead of the name passed to the agent constructor and agent run. | `UNSET` | | `deps` | `AgentDepsT | Unset` | The dependencies to use instead of the dependencies passed to the agent run. | `UNSET` | | `model` | `Model | KnownModelName | str | Unset` | The model to use instead of the model passed to the agent run. | `UNSET` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | Unset` | The toolsets to use instead of the toolsets passed to the agent constructor and agent run. | `UNSET` | | `tools` | `Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | Unset` | The tools to use instead of the tools registered with the agent. | `UNSET` | | `instructions` | `Instructions[AgentDepsT] | Unset` | The instructions to use instead of the instructions registered with the agent. | `UNSET` |

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

```python
@contextmanager
def override(
    self,
    *,
    name: str | _utils.Unset = _utils.UNSET,
    deps: AgentDepsT | _utils.Unset = _utils.UNSET,
    model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
    instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
) -> Iterator[None]:
    """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

    This is particularly useful when testing.
    You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

    Args:
        name: The name to use instead of the name passed to the agent constructor and agent run.
        deps: The dependencies to use instead of the dependencies passed to the agent run.
        model: The model to use instead of the model passed to the agent run.
        toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
        tools: The tools to use instead of the tools registered with the agent.
        instructions: The instructions to use instead of the instructions registered with the agent.
    """
    if _utils.is_set(name):
        name_token = self._override_name.set(_utils.Some(name))
    else:
        name_token = None

    if _utils.is_set(deps):
        deps_token = self._override_deps.set(_utils.Some(deps))
    else:
        deps_token = None

    if _utils.is_set(model):
        model_token = self._override_model.set(_utils.Some(models.infer_model(model)))
    else:
        model_token = None

    if _utils.is_set(toolsets):
        toolsets_token = self._override_toolsets.set(_utils.Some(toolsets))
    else:
        toolsets_token = None

    if _utils.is_set(tools):
        tools_token = self._override_tools.set(_utils.Some(tools))
    else:
        tools_token = None

    if _utils.is_set(instructions):
        normalized_instructions = self._normalize_instructions(instructions)
        instructions_token = self._override_instructions.set(_utils.Some(normalized_instructions))
    else:
        instructions_token = None

    try:
        yield
    finally:
        if name_token is not None:
            self._override_name.reset(name_token)
        if deps_token is not None:
            self._override_deps.reset(deps_token)
        if model_token is not None:
            self._override_model.reset(model_token)
        if toolsets_token is not None:
            self._override_toolsets.reset(toolsets_token)
        if tools_token is not None:
            self._override_tools.reset(tools_token)
        if instructions_token is not None:
            self._override_instructions.reset(instructions_token)

```

#### instructions

```python
instructions(
    func: Callable[[RunContext[AgentDepsT]], str],
) -> Callable[[RunContext[AgentDepsT]], str]

```

```python
instructions(
    func: Callable[
        [RunContext[AgentDepsT]], Awaitable[str]
    ],
) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]

```

```python
instructions(func: Callable[[], str]) -> Callable[[], str]

```

```python
instructions(
    func: Callable[[], Awaitable[str]],
) -> Callable[[], Awaitable[str]]

```

```python
instructions() -> Callable[
    [SystemPromptFunc[AgentDepsT]],
    SystemPromptFunc[AgentDepsT],
]

```

```python
instructions(
    func: SystemPromptFunc[AgentDepsT] | None = None,
) -> (
    Callable[
        [SystemPromptFunc[AgentDepsT]],
        SystemPromptFunc[AgentDepsT],
    ]
    | SystemPromptFunc[AgentDepsT]
)

```

Decorator to register an instructions function.

Optionally takes RunContext as its only argument. Can decorate a sync or async functions.

The decorator can be used bare (`agent.instructions`).

Overloads for every possible signature of `instructions` are included so the decorator doesn't obscure the type of the function.

Example:

```python
from pydantic_ai import Agent, RunContext

agent = Agent('test', deps_type=str)

@agent.instructions
def simple_instructions() -> str:
    return 'foobar'

@agent.instructions
async def async_instructions(ctx: RunContext[str]) -> str:
    return f'{ctx.deps} is the best'

```

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

````python
def instructions(
    self,
    func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
    /,
) -> (
    Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
    | _system_prompt.SystemPromptFunc[AgentDepsT]
):
    """Decorator to register an instructions function.

    Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
    Can decorate a sync or async functions.

    The decorator can be used bare (`agent.instructions`).

    Overloads for every possible signature of `instructions` are included so the decorator doesn't obscure
    the type of the function.

    Example:
    ```python
    from pydantic_ai import Agent, RunContext

    agent = Agent('test', deps_type=str)

    @agent.instructions
    def simple_instructions() -> str:
        return 'foobar'

    @agent.instructions
    async def async_instructions(ctx: RunContext[str]) -> str:
        return f'{ctx.deps} is the best'
    ```
    """
    if func is None:

        def decorator(
            func_: _system_prompt.SystemPromptFunc[AgentDepsT],
        ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
            self._instructions.append(func_)
            return func_

        return decorator
    else:
        self._instructions.append(func)
        return func

````

#### system_prompt

```python
system_prompt(
    func: Callable[[RunContext[AgentDepsT]], str],
) -> Callable[[RunContext[AgentDepsT]], str]

```

```python
system_prompt(
    func: Callable[
        [RunContext[AgentDepsT]], Awaitable[str]
    ],
) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]

```

```python
system_prompt(func: Callable[[], str]) -> Callable[[], str]

```

```python
system_prompt(
    func: Callable[[], Awaitable[str]],
) -> Callable[[], Awaitable[str]]

```

```python
system_prompt(*, dynamic: bool = False) -> Callable[
    [SystemPromptFunc[AgentDepsT]],
    SystemPromptFunc[AgentDepsT],
]

```

```python
system_prompt(
    func: SystemPromptFunc[AgentDepsT] | None = None,
    /,
    *,
    dynamic: bool = False,
) -> (
    Callable[
        [SystemPromptFunc[AgentDepsT]],
        SystemPromptFunc[AgentDepsT],
    ]
    | SystemPromptFunc[AgentDepsT]
)

```

Decorator to register a system prompt function.

Optionally takes RunContext as its only argument. Can decorate a sync or async functions.

The decorator can be used either bare (`agent.system_prompt`) or as a function call (`agent.system_prompt(...)`), see the examples below.

Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure the type of the function, see `tests/typed_agent.py` for tests.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `func` | `SystemPromptFunc[AgentDepsT] | None` | The function to decorate | `None` | | `dynamic` | `bool` | If True, the system prompt will be reevaluated even when messages_history is provided, see SystemPromptPart.dynamic_ref | `False` |

Example:

```python
from pydantic_ai import Agent, RunContext

agent = Agent('test', deps_type=str)

@agent.system_prompt
def simple_system_prompt() -> str:
    return 'foobar'

@agent.system_prompt(dynamic=True)
async def async_system_prompt(ctx: RunContext[str]) -> str:
    return f'{ctx.deps} is the best'

```

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

````python
def system_prompt(
    self,
    func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
    /,
    *,
    dynamic: bool = False,
) -> (
    Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
    | _system_prompt.SystemPromptFunc[AgentDepsT]
):
    """Decorator to register a system prompt function.

    Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
    Can decorate a sync or async functions.

    The decorator can be used either bare (`agent.system_prompt`) or as a function call
    (`agent.system_prompt(...)`), see the examples below.

    Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure
    the type of the function, see `tests/typed_agent.py` for tests.

    Args:
        func: The function to decorate
        dynamic: If True, the system prompt will be reevaluated even when `messages_history` is provided,
            see [`SystemPromptPart.dynamic_ref`][pydantic_ai.messages.SystemPromptPart.dynamic_ref]

    Example:
    ```python
    from pydantic_ai import Agent, RunContext

    agent = Agent('test', deps_type=str)

    @agent.system_prompt
    def simple_system_prompt() -> str:
        return 'foobar'

    @agent.system_prompt(dynamic=True)
    async def async_system_prompt(ctx: RunContext[str]) -> str:
        return f'{ctx.deps} is the best'
    ```
    """
    if func is None:

        def decorator(
            func_: _system_prompt.SystemPromptFunc[AgentDepsT],
        ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
            runner = _system_prompt.SystemPromptRunner[AgentDepsT](func_, dynamic=dynamic)
            self._system_prompt_functions.append(runner)
            if dynamic:  # pragma: lax no cover
                self._system_prompt_dynamic_functions[func_.__qualname__] = runner
            return func_

        return decorator
    else:
        assert not dynamic, "dynamic can't be True in this case"
        self._system_prompt_functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](func, dynamic=dynamic))
        return func

````

#### output_validator

```python
output_validator(
    func: Callable[
        [RunContext[AgentDepsT], OutputDataT], OutputDataT
    ],
) -> Callable[
    [RunContext[AgentDepsT], OutputDataT], OutputDataT
]

```

```python
output_validator(
    func: Callable[
        [RunContext[AgentDepsT], OutputDataT],
        Awaitable[OutputDataT],
    ],
) -> Callable[
    [RunContext[AgentDepsT], OutputDataT],
    Awaitable[OutputDataT],
]

```

```python
output_validator(
    func: Callable[[OutputDataT], OutputDataT],
) -> Callable[[OutputDataT], OutputDataT]

```

```python
output_validator(
    func: Callable[[OutputDataT], Awaitable[OutputDataT]],
) -> Callable[[OutputDataT], Awaitable[OutputDataT]]

```

```python
output_validator(
    func: OutputValidatorFunc[AgentDepsT, OutputDataT],
) -> OutputValidatorFunc[AgentDepsT, OutputDataT]

```

Decorator to register an output validator function.

Optionally takes RunContext as its first argument. Can decorate a sync or async functions.

Overloads for every possible signature of `output_validator` are included so the decorator doesn't obscure the type of the function, see `tests/typed_agent.py` for tests.

Example:

```python
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('test', deps_type=str)

@agent.output_validator
def output_validator_simple(data: str) -> str:
    if 'wrong' in data:
        raise ModelRetry('wrong response')
    return data

@agent.output_validator
async def output_validator_deps(ctx: RunContext[str], data: str) -> str:
    if ctx.deps in data:
        raise ModelRetry('wrong response')
    return data

result = agent.run_sync('foobar', deps='spam')
print(result.output)
#> success (no tool calls)

```

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

````python
def output_validator(
    self, func: _output.OutputValidatorFunc[AgentDepsT, OutputDataT], /
) -> _output.OutputValidatorFunc[AgentDepsT, OutputDataT]:
    """Decorator to register an output validator function.

    Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
    Can decorate a sync or async functions.

    Overloads for every possible signature of `output_validator` are included so the decorator doesn't obscure
    the type of the function, see `tests/typed_agent.py` for tests.

    Example:
    ```python
    from pydantic_ai import Agent, ModelRetry, RunContext

    agent = Agent('test', deps_type=str)

    @agent.output_validator
    def output_validator_simple(data: str) -> str:
        if 'wrong' in data:
            raise ModelRetry('wrong response')
        return data

    @agent.output_validator
    async def output_validator_deps(ctx: RunContext[str], data: str) -> str:
        if ctx.deps in data:
            raise ModelRetry('wrong response')
        return data

    result = agent.run_sync('foobar', deps='spam')
    print(result.output)
    #> success (no tool calls)
    ```
    """
    self._output_validators.append(_output.OutputValidator[AgentDepsT, Any](func))
    return func

````

#### tool

```python
tool(
    func: ToolFuncContext[AgentDepsT, ToolParams],
) -> ToolFuncContext[AgentDepsT, ToolParams]

```

```python
tool(
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = "auto",
    require_parameter_descriptions: bool = False,
    schema_generator: type[
        GenerateJsonSchema
    ] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None
) -> Callable[
    [ToolFuncContext[AgentDepsT, ToolParams]],
    ToolFuncContext[AgentDepsT, ToolParams],
]

```

```python
tool(
    func: (
        ToolFuncContext[AgentDepsT, ToolParams] | None
    ) = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = "auto",
    require_parameter_descriptions: bool = False,
    schema_generator: type[
        GenerateJsonSchema
    ] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Any

```

Decorator to register a tool function which takes RunContext as its first argument.

Can decorate a sync or async functions.

The docstring is inspected to extract both the tool description and description of each parameter, [learn more](../../tools/#function-tools-and-schema).

We can't add overloads for every possible signature of tool, since the return type is a recursive union so the signature of functions decorated with `@agent.tool` is obscured.

Example:

```python
from pydantic_ai import Agent, RunContext

agent = Agent('test', deps_type=int)

@agent.tool
def foobar(ctx: RunContext[int], x: int) -> int:
    return ctx.deps + x

@agent.tool(retries=2)
async def spam(ctx: RunContext[str], y: float) -> float:
    return ctx.deps + y

result = agent.run_sync('foobar', deps=1)
print(result.output)
#> {"foobar":1,"spam":1.0}

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `func` | `ToolFuncContext[AgentDepsT, ToolParams] | None` | The tool function to register. | `None` | | `name` | `str | None` | The name of the tool, defaults to the function name. | `None` | | `description` | `str | None` | The description of the tool, defaults to the function docstring. | `None` | | `retries` | `int | None` | The number of retries to allow for this tool, defaults to the agent's default retries, which defaults to 1. | `None` | | `prepare` | `ToolPrepareFunc[AgentDepsT] | None` | custom method to prepare the tool definition for each step, return None to omit this tool from a given step. This is useful if you want to customise a tool at call time, or omit it completely from a step. See ToolPrepareFunc. | `None` | | `docstring_format` | `DocstringFormat` | The format of the docstring, see DocstringFormat. Defaults to 'auto', such that the format is inferred from the structure of the docstring. | `'auto'` | | `require_parameter_descriptions` | `bool` | If True, raise an error if a parameter description is missing. Defaults to False. | `False` | | `schema_generator` | `type[GenerateJsonSchema]` | The JSON schema generator class to use for this tool. Defaults to GenerateToolJsonSchema. | `GenerateToolJsonSchema` | | `strict` | `bool | None` | Whether to enforce JSON schema compliance (only affects OpenAI). See ToolDefinition for more info. | `None` | | `sequential` | `bool` | Whether the function requires a sequential/serial execution environment. Defaults to False. | `False` | | `requires_approval` | `bool` | Whether this tool requires human-in-the-loop approval. Defaults to False. See the tools documentation for more info. | `False` | | `metadata` | `dict[str, Any] | None` | Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

````python
def tool(
    self,
    func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

    Can decorate a sync or async functions.

    The docstring is inspected to extract both the tool description and description of each parameter,
    [learn more](../tools.md#function-tools-and-schema).

    We can't add overloads for every possible signature of tool, since the return type is a recursive union
    so the signature of functions decorated with `@agent.tool` is obscured.

    Example:
    ```python
    from pydantic_ai import Agent, RunContext

    agent = Agent('test', deps_type=int)

    @agent.tool
    def foobar(ctx: RunContext[int], x: int) -> int:
        return ctx.deps + x

    @agent.tool(retries=2)
    async def spam(ctx: RunContext[str], y: float) -> float:
        return ctx.deps + y

    result = agent.run_sync('foobar', deps=1)
    print(result.output)
    #> {"foobar":1,"spam":1.0}
    ```

    Args:
        func: The tool function to register.
        name: The name of the tool, defaults to the function name.
        description: The description of the tool, defaults to the function docstring.
        retries: The number of retries to allow for this tool, defaults to the agent's default retries,
            which defaults to 1.
        prepare: custom method to prepare the tool definition for each step, return `None` to omit this
            tool from a given step. This is useful if you want to customise a tool at call time,
            or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
        docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
            Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
        require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
        schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
        strict: Whether to enforce JSON schema compliance (only affects OpenAI).
            See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
        sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
        requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
            See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
        metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
    """

    def tool_decorator(
        func_: ToolFuncContext[AgentDepsT, ToolParams],
    ) -> ToolFuncContext[AgentDepsT, ToolParams]:
        # noinspection PyTypeChecker
        self._function_toolset.add_function(
            func_,
            takes_ctx=True,
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

#### tool_plain

```python
tool_plain(
    func: ToolFuncPlain[ToolParams],
) -> ToolFuncPlain[ToolParams]

```

```python
tool_plain(
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = "auto",
    require_parameter_descriptions: bool = False,
    schema_generator: type[
        GenerateJsonSchema
    ] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None
) -> Callable[
    [ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]
]

```

```python
tool_plain(
    func: ToolFuncPlain[ToolParams] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = "auto",
    require_parameter_descriptions: bool = False,
    schema_generator: type[
        GenerateJsonSchema
    ] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Any

```

Decorator to register a tool function which DOES NOT take `RunContext` as an argument.

Can decorate a sync or async functions.

The docstring is inspected to extract both the tool description and description of each parameter, [learn more](../../tools/#function-tools-and-schema).

We can't add overloads for every possible signature of tool, since the return type is a recursive union so the signature of functions decorated with `@agent.tool` is obscured.

Example:

```python
from pydantic_ai import Agent, RunContext

agent = Agent('test')

@agent.tool
def foobar(ctx: RunContext[int]) -> int:
    return 123

@agent.tool(retries=2)
async def spam(ctx: RunContext[str]) -> float:
    return 3.14

result = agent.run_sync('foobar', deps=1)
print(result.output)
#> {"foobar":123,"spam":3.14}

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `func` | `ToolFuncPlain[ToolParams] | None` | The tool function to register. | `None` | | `name` | `str | None` | The name of the tool, defaults to the function name. | `None` | | `description` | `str | None` | The description of the tool, defaults to the function docstring. | `None` | | `retries` | `int | None` | The number of retries to allow for this tool, defaults to the agent's default retries, which defaults to 1. | `None` | | `prepare` | `ToolPrepareFunc[AgentDepsT] | None` | custom method to prepare the tool definition for each step, return None to omit this tool from a given step. This is useful if you want to customise a tool at call time, or omit it completely from a step. See ToolPrepareFunc. | `None` | | `docstring_format` | `DocstringFormat` | The format of the docstring, see DocstringFormat. Defaults to 'auto', such that the format is inferred from the structure of the docstring. | `'auto'` | | `require_parameter_descriptions` | `bool` | If True, raise an error if a parameter description is missing. Defaults to False. | `False` | | `schema_generator` | `type[GenerateJsonSchema]` | The JSON schema generator class to use for this tool. Defaults to GenerateToolJsonSchema. | `GenerateToolJsonSchema` | | `strict` | `bool | None` | Whether to enforce JSON schema compliance (only affects OpenAI). See ToolDefinition for more info. | `None` | | `sequential` | `bool` | Whether the function requires a sequential/serial execution environment. Defaults to False. | `False` | | `requires_approval` | `bool` | Whether this tool requires human-in-the-loop approval. Defaults to False. See the tools documentation for more info. | `False` | | `metadata` | `dict[str, Any] | None` | Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization. | `None` |

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

````python
def tool_plain(
    self,
    func: ToolFuncPlain[ToolParams] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorator to register a tool function which DOES NOT take `RunContext` as an argument.

    Can decorate a sync or async functions.

    The docstring is inspected to extract both the tool description and description of each parameter,
    [learn more](../tools.md#function-tools-and-schema).

    We can't add overloads for every possible signature of tool, since the return type is a recursive union
    so the signature of functions decorated with `@agent.tool` is obscured.

    Example:
    ```python
    from pydantic_ai import Agent, RunContext

    agent = Agent('test')

    @agent.tool
    def foobar(ctx: RunContext[int]) -> int:
        return 123

    @agent.tool(retries=2)
    async def spam(ctx: RunContext[str]) -> float:
        return 3.14

    result = agent.run_sync('foobar', deps=1)
    print(result.output)
    #> {"foobar":123,"spam":3.14}
    ```

    Args:
        func: The tool function to register.
        name: The name of the tool, defaults to the function name.
        description: The description of the tool, defaults to the function docstring.
        retries: The number of retries to allow for this tool, defaults to the agent's default retries,
            which defaults to 1.
        prepare: custom method to prepare the tool definition for each step, return `None` to omit this
            tool from a given step. This is useful if you want to customise a tool at call time,
            or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
        docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
            Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
        require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
        schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
        strict: Whether to enforce JSON schema compliance (only affects OpenAI).
            See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
        sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
        requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
            See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
        metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
    """

    def tool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
        # noinspection PyTypeChecker
        self._function_toolset.add_function(
            func_,
            takes_ctx=False,
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

#### toolset

```python
toolset(
    func: ToolsetFunc[AgentDepsT],
) -> ToolsetFunc[AgentDepsT]

```

```python
toolset(
    *, per_run_step: bool = True
) -> Callable[
    [ToolsetFunc[AgentDepsT]], ToolsetFunc[AgentDepsT]
]

```

```python
toolset(
    func: ToolsetFunc[AgentDepsT] | None = None,
    /,
    *,
    per_run_step: bool = True,
) -> Any

```

Decorator to register a toolset function which takes RunContext as its only argument.

Can decorate a sync or async functions.

The decorator can be used bare (`agent.toolset`).

Example:

```python
from pydantic_ai import AbstractToolset, Agent, FunctionToolset, RunContext

agent = Agent('test', deps_type=str)

@agent.toolset
async def simple_toolset(ctx: RunContext[str]) -> AbstractToolset[str]:
    return FunctionToolset()

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `func` | `ToolsetFunc[AgentDepsT] | None` | The toolset function to register. | `None` | | `per_run_step` | `bool` | Whether to re-evaluate the toolset for each run step. Defaults to True. | `True` |

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

````python
def toolset(
    self,
    func: ToolsetFunc[AgentDepsT] | None = None,
    /,
    *,
    per_run_step: bool = True,
) -> Any:
    """Decorator to register a toolset function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.

    Can decorate a sync or async functions.

    The decorator can be used bare (`agent.toolset`).

    Example:
    ```python
    from pydantic_ai import AbstractToolset, Agent, FunctionToolset, RunContext

    agent = Agent('test', deps_type=str)

    @agent.toolset
    async def simple_toolset(ctx: RunContext[str]) -> AbstractToolset[str]:
        return FunctionToolset()
    ```

    Args:
        func: The toolset function to register.
        per_run_step: Whether to re-evaluate the toolset for each run step. Defaults to True.
    """

    def toolset_decorator(func_: ToolsetFunc[AgentDepsT]) -> ToolsetFunc[AgentDepsT]:
        self._dynamic_toolsets.append(DynamicToolset(func_, per_run_step=per_run_step))
        return func_

    return toolset_decorator if func is None else toolset_decorator(func)

````

#### toolsets

```python
toolsets: Sequence[AbstractToolset[AgentDepsT]]

```

All toolsets registered on the agent, including a function toolset holding tools that were registered on the agent directly.

Output tools are not included.

#### __aenter__

```python
__aenter__() -> Self

```

Enter the agent context.

This will start all MCPServerStdios registered as `toolsets` so they are ready to be used.

This is a no-op if the agent has already been entered.

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

```python
async def __aenter__(self) -> Self:
    """Enter the agent context.

    This will start all [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] registered as `toolsets` so they are ready to be used.

    This is a no-op if the agent has already been entered.
    """
    async with self._enter_lock:
        if self._entered_count == 0:
            async with AsyncExitStack() as exit_stack:
                toolset = self._get_toolset()
                await exit_stack.enter_async_context(toolset)

                self._exit_stack = exit_stack.pop_all()
        self._entered_count += 1
    return self

```

#### set_mcp_sampling_model

```python
set_mcp_sampling_model(
    model: Model | KnownModelName | str | None = None,
) -> None

```

Set the sampling model on all MCP servers registered with the agent.

If no sampling model is provided, the agent's model will be used.

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

```python
def set_mcp_sampling_model(self, model: models.Model | models.KnownModelName | str | None = None) -> None:
    """Set the sampling model on all MCP servers registered with the agent.

    If no sampling model is provided, the agent's model will be used.
    """
    try:
        sampling_model = models.infer_model(model) if model else self._get_model(None)
    except exceptions.UserError as e:
        raise exceptions.UserError('No sampling model provided and no model set on the agent.') from e

    from ..mcp import MCPServer

    def _set_sampling_model(toolset: AbstractToolset[AgentDepsT]) -> None:
        if isinstance(toolset, MCPServer):
            toolset.sampling_model = sampling_model

    self._get_toolset().apply(_set_sampling_model)

```

#### run_mcp_servers

```python
run_mcp_servers(
    model: Model | KnownModelName | str | None = None,
) -> AsyncIterator[None]

```

Deprecated

`run_mcp_servers` is deprecated, use `async with agent:` instead. If you need to set a sampling model on all MCP servers, use `agent.set_mcp_sampling_model()`.

Run MCPServerStdios so they can be used by the agent.

Deprecated: use async with agent instead. If you need to set a sampling model on all MCP servers, use agent.set_mcp_sampling_model().

Returns: a context manager to start and shutdown the servers.

Source code in `pydantic_ai_slim/pydantic_ai/agent/__init__.py`

```python
@asynccontextmanager
@deprecated(
    '`run_mcp_servers` is deprecated, use `async with agent:` instead. If you need to set a sampling model on all MCP servers, use `agent.set_mcp_sampling_model()`.'
)
async def run_mcp_servers(
    self, model: models.Model | models.KnownModelName | str | None = None
) -> AsyncIterator[None]:
    """Run [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] so they can be used by the agent.

    Deprecated: use [`async with agent`][pydantic_ai.agent.Agent.__aenter__] instead.
    If you need to set a sampling model on all MCP servers, use [`agent.set_mcp_sampling_model()`][pydantic_ai.agent.Agent.set_mcp_sampling_model].

    Returns: a context manager to start and shutdown the servers.
    """
    try:
        self.set_mcp_sampling_model(model)
    except exceptions.UserError:
        if model is not None:
            raise

    async with self:
        yield

```

### AbstractAgent

Bases: `Generic[AgentDepsT, OutputDataT]`, `ABC`

Abstract superclass for Agent, WrapperAgent, and your own custom agent implementations.

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
class AbstractAgent(Generic[AgentDepsT, OutputDataT], ABC):
    """Abstract superclass for [`Agent`][pydantic_ai.agent.Agent], [`WrapperAgent`][pydantic_ai.agent.WrapperAgent], and your own custom agent implementations."""

    @property
    @abstractmethod
    def model(self) -> models.Model | models.KnownModelName | str | None:
        """The default model configured for this agent."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str | None:
        """The name of the agent, used for logging.

        If `None`, we try to infer the agent name from the call frame when the agent is first run.
        """
        raise NotImplementedError

    @name.setter
    @abstractmethod
    def name(self, value: str | None) -> None:
        """Set the name of the agent, used for logging."""
        raise NotImplementedError

    @property
    @abstractmethod
    def deps_type(self) -> type:
        """The type of dependencies used by the agent."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_type(self) -> OutputSpec[OutputDataT]:
        """The type of data output by agent runs, used to validate the data returned by the model, defaults to `str`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        """Optional handler for events from the model's streaming response and the agent's execution of tools."""
        raise NotImplementedError

    @property
    @abstractmethod
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        """All toolsets registered on the agent.

        Output tools are not included.
        """
        raise NotImplementedError

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then
        runs the graph to completion. The result of the run is returned.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            agent_run = await agent.run('What is the capital of France?')
            print(agent_run.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        event_stream_handler = event_stream_handler or self.event_stream_handler

        async with self.iter(
            user_prompt=user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        ) as agent_run:
            async for node in agent_run:
                if event_stream_handler is not None and (
                    self.is_model_request_node(node) or self.is_call_tools_node(node)
                ):
                    async with node.stream(agent_run.ctx) as stream:
                        await event_stream_handler(_agent_graph.build_run_context(agent_run.ctx), stream)

        assert agent_run.result is not None, 'The graph run did not finish properly'
        return agent_run.result

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[Any]:
        """Synchronously run the agent with a user prompt.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        result_sync = agent.run_sync('What is the capital of Italy?')
        print(result_sync.output)
        #> The capital of Italy is Rome.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        return _utils.get_event_loop().run_until_complete(
            self.run(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=False,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                event_stream_handler=event_stream_handler,
            )
        )

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_stream(  # noqa C901
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AsyncIterator[result.StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async streaming mode.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then
        runs the graph until the model produces output matching the `output_type`, for example text or structured data.
        At this point, a streaming run result object is yielded from which you can stream the output as it comes in,
        and -- once this output has completed streaming -- get the complete output, message history, and usage.

        As this method will consider the first output matching the `output_type` to be the final output,
        it will stop running the agent graph and will not execute any tool calls made by the model after this "final" output.
        If you want to always run the agent graph to completion and stream events and output at the same time,
        use [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] with an `event_stream_handler` or [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] instead.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.run_stream('What is the capital of the UK?') as response:
                print(await response.get_output())
                #> The capital of the UK is London.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
                It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.
                Note that it does _not_ receive any events after the final result is found.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        event_stream_handler = event_stream_handler or self.event_stream_handler

        yielded = False
        async with self.iter(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            instructions=instructions,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=False,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        ) as agent_run:
            first_node = agent_run.next_node  # start with the first node
            assert isinstance(first_node, _agent_graph.UserPromptNode)  # the first node should be a user prompt node
            node = first_node
            while True:
                graph_ctx = agent_run.ctx
                if self.is_model_request_node(node):
                    async with node.stream(graph_ctx) as stream:
                        final_result_event = None

                        async def stream_to_final(
                            stream: AgentStream,
                        ) -> AsyncIterator[_messages.ModelResponseStreamEvent]:
                            nonlocal final_result_event
                            async for event in stream:
                                yield event
                                if isinstance(event, _messages.FinalResultEvent):
                                    final_result_event = event
                                    break

                        if event_stream_handler is not None:
                            await event_stream_handler(
                                _agent_graph.build_run_context(graph_ctx), stream_to_final(stream)
                            )
                        else:
                            async for _ in stream_to_final(stream):
                                pass

                        if final_result_event is not None:
                            final_result = FinalResult(
                                None, final_result_event.tool_name, final_result_event.tool_call_id
                            )
                            if yielded:
                                raise exceptions.AgentRunError('Agent run produced final results')  # pragma: no cover
                            yielded = True

                            messages = graph_ctx.state.message_history.copy()

                            async def on_complete() -> None:
                                """Called when the stream has completed.

                                The model response will have been added to messages by now
                                by `StreamedRunResult._marked_completed`.
                                """
                                nonlocal final_result
                                final_result = FinalResult(
                                    await stream.get_output(), final_result.tool_name, final_result.tool_call_id
                                )

                                # When we get here, the `ModelRequestNode` has completed streaming after the final result was found.
                                # When running an agent with `agent.run`, we'd then move to `CallToolsNode` to execute the tool calls and
                                # find the final result.
                                # We also want to execute tool calls (in case `agent.end_strategy == 'exhaustive'`) here, but
                                # we don't want to use run the `CallToolsNode` logic to determine the final output, as it would be
                                # wasteful and could produce a different result (e.g. when text output is followed by tool calls).
                                # So we call `process_tool_calls` directly and then end the run with the found final result.

                                parts: list[_messages.ModelRequestPart] = []
                                async for _event in _agent_graph.process_tool_calls(
                                    tool_manager=graph_ctx.deps.tool_manager,
                                    tool_calls=stream.response.tool_calls,
                                    tool_call_results=None,
                                    final_result=final_result,
                                    ctx=graph_ctx,
                                    output_parts=parts,
                                ):
                                    pass

                                # For backwards compatibility, append a new ModelRequest using the tool returns and retries
                                if parts:
                                    messages.append(_messages.ModelRequest(parts, run_id=graph_ctx.state.run_id))

                                await agent_run.next(_agent_graph.SetFinalResult(final_result))

                            yield StreamedRunResult(
                                messages,
                                graph_ctx.deps.new_message_index,
                                stream,
                                on_complete,
                            )
                            break
                elif self.is_call_tools_node(node) and event_stream_handler is not None:
                    async with node.stream(agent_run.ctx) as stream:
                        await event_stream_handler(_agent_graph.build_run_context(agent_run.ctx), stream)

                next_node = await agent_run.next(node)
                if isinstance(next_node, End) and agent_run.result is not None:
                    # A final output could have been produced by the CallToolsNode rather than the ModelRequestNode,
                    # if a tool function raised CallDeferred or ApprovalRequired.
                    # In this case there's no response to stream, but we still let the user access the output etc as normal.
                    yield StreamedRunResult(
                        graph_ctx.state.message_history,
                        graph_ctx.deps.new_message_index,
                        run_result=agent_run.result,
                    )
                    yielded = True
                    break
                if not isinstance(next_node, _agent_graph.AgentNode):
                    raise exceptions.AgentRunError(  # pragma: no cover
                        'Should have produced a StreamedRunResult before getting here'
                    )
                node = cast(_agent_graph.AgentNode[Any, Any], next_node)

        if not yielded:
            raise exceptions.AgentRunError('Agent run finished without producing a final result')  # pragma: no cover

    @overload
    def run_stream_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> result.StreamedRunResultSync[AgentDepsT, OutputDataT]: ...

    @overload
    def run_stream_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> result.StreamedRunResultSync[AgentDepsT, RunOutputDataT]: ...

    def run_stream_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> result.StreamedRunResultSync[AgentDepsT, Any]:
        """Run the agent with a user prompt in sync streaming mode.

        This is a convenience method that wraps [`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then
        runs the graph until the model produces output matching the `output_type`, for example text or structured data.
        At this point, a streaming run result object is yielded from which you can stream the output as it comes in,
        and -- once this output has completed streaming -- get the complete output, message history, and usage.

        As this method will consider the first output matching the `output_type` to be the final output,
        it will stop running the agent graph and will not execute any tool calls made by the model after this "final" output.
        If you want to always run the agent graph to completion and stream events and output at the same time,
        use [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] with an `event_stream_handler` or [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] instead.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        def main():
            response = agent.run_stream_sync('What is the capital of the UK?')
            print(response.get_output())
            #> The capital of the UK is London.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
                It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.
                Note that it does _not_ receive any events after the final result is found.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        async def _consume_stream():
            async with self.run_stream(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                event_stream_handler=event_stream_handler,
            ) as stream_result:
                yield stream_result

        async_result = _utils.get_event_loop().run_until_complete(anext(_consume_stream()))
        return result.StreamedRunResultSync(async_result)

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[OutputDataT]]: ...

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]]: ...

    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
        """Run the agent with a user prompt in async mode and stream events from the run.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
        uses the `event_stream_handler` kwarg to get a stream of events from the run.

        Example:
        ```python
        from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

        agent = Agent('openai:gpt-4o')

        async def main():
            events: list[AgentStreamEvent | AgentRunResultEvent] = []
            async for event in agent.run_stream_events('What is the capital of France?'):
                events.append(event)
            print(events)
            '''
            [
                PartStartEvent(index=0, part=TextPart(content='The capital of ')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
                PartEndEvent(
                    index=0, part=TextPart(content='The capital of France is Paris. ')
                ),
                AgentRunResultEvent(
                    result=AgentRunResult(output='The capital of France is Paris. ')
                ),
            ]
            '''
        ```

        Arguments are the same as for [`self.run`][pydantic_ai.agent.AbstractAgent.run],
        except that `event_stream_handler` is now allowed.

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
            run result.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        # unfortunately this hack of returning a generator rather than defining it right here is
        # required to allow overloads of this method to work in python's typing system, or at least with pyright
        # or at least I couldn't make it work without
        return self._run_stream_events(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        )

    async def _run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
        send_stream, receive_stream = anyio.create_memory_object_stream[
            _messages.AgentStreamEvent | AgentRunResultEvent[Any]
        ]()

        async def event_stream_handler(
            _: RunContext[AgentDepsT], events: AsyncIterable[_messages.AgentStreamEvent]
        ) -> None:
            async for event in events:
                await send_stream.send(event)

        async def run_agent() -> AgentRunResult[Any]:
            async with send_stream:
                return await self.run(
                    user_prompt,
                    output_type=output_type,
                    message_history=message_history,
                    deferred_tool_results=deferred_tool_results,
                    model=model,
                    instructions=instructions,
                    deps=deps,
                    model_settings=model_settings,
                    usage_limits=usage_limits,
                    usage=usage,
                    infer_name=False,
                    toolsets=toolsets,
                    builtin_tools=builtin_tools,
                    event_stream_handler=event_stream_handler,
                )

        task = asyncio.create_task(run_agent())

        async with receive_stream:
            async for message in receive_stream:
                yield message

        result = await task
        yield AgentRunResultEvent(result)

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    @abstractmethod
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
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

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        raise NotImplementedError
        yield

    @contextmanager
    @abstractmethod
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            instructions: The instructions to use instead of the instructions registered with the agent.
        """
        raise NotImplementedError
        yield

    def _infer_name(self, function_frame: FrameType | None) -> None:
        """Infer the agent name from the call frame.

        RunUsage should be `self._infer_name(inspect.currentframe())`.
        """
        assert self.name is None, 'Name already set'
        if function_frame is not None:  # pragma: no branch
            if parent_frame := function_frame.f_back:  # pragma: no branch
                for name, item in parent_frame.f_locals.items():
                    if item is self:
                        self.name = name
                        return
                if parent_frame.f_locals != parent_frame.f_globals:  # pragma: no branch
                    # if we couldn't find the agent in locals and globals are a different dict, try globals
                    for name, item in parent_frame.f_globals.items():
                        if item is self:
                            self.name = name
                            return

    @staticmethod
    @contextmanager
    def sequential_tool_calls() -> Iterator[None]:
        """Run tool calls sequentially during the context."""
        with ToolManager.sequential_tool_calls():
            yield

    @staticmethod
    def is_model_request_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[_agent_graph.ModelRequestNode[T, S]]:
        """Check if the node is a `ModelRequestNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.ModelRequestNode)

    @staticmethod
    def is_call_tools_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[_agent_graph.CallToolsNode[T, S]]:
        """Check if the node is a `CallToolsNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.CallToolsNode)

    @staticmethod
    def is_user_prompt_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[_agent_graph.UserPromptNode[T, S]]:
        """Check if the node is a `UserPromptNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.UserPromptNode)

    @staticmethod
    def is_end_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[End[result.FinalResult[S]]]:
        """Check if the node is a `End`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, End)

    @abstractmethod
    async def __aenter__(self) -> AbstractAgent[AgentDepsT, OutputDataT]:
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(self, *args: Any) -> bool | None:
        raise NotImplementedError

    # TODO (v2): Remove in favor of using `AGUIApp` directly -- we don't have `to_temporal()` or `to_vercel_ai()` either.
    def to_ag_ui(
        self,
        *,
        # Agent.iter parameters
        output_type: OutputSpec[OutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        # Starlette
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[AGUIApp[AgentDepsT, OutputDataT]] | None = None,
    ) -> AGUIApp[AgentDepsT, OutputDataT]:
        """Returns an ASGI application that handles every AG-UI request by running the agent.

        Note that the `deps` will be the same for each request, with the exception of the AG-UI state that's
        injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ag_ui.StateHandler] protocol.
        To provide different `deps` for each request (e.g. based on the authenticated user),
        use [`pydantic_ai.ag_ui.run_ag_ui`][pydantic_ai.ag_ui.run_ag_ui] or
        [`pydantic_ai.ag_ui.handle_ag_ui_request`][pydantic_ai.ag_ui.handle_ag_ui_request] instead.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')
        app = agent.to_ag_ui()
        ```

        The `app` is an ASGI application that can be used with any ASGI server.

        To run the application, you can use the following command:

        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```

        See [AG-UI docs](../ui/ag-ui.md) for more information.

        Args:
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request. A starlette application will always
                automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
                outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
                `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
                exception cases occurring in the routing or endpoints.
            exception_handlers: A mapping of either integer status codes, or exception class types onto
                callables which handle the exceptions. Exception handler callables should be of the form
                `handler(request, exc) -> response` and may be either standard functions, or async functions.
            on_startup: A list of callables to run on application startup. Startup handler callables do not
                take any arguments, and may be either standard functions, or async functions.
            on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
                not take any arguments, and may be either standard functions, or async functions.
            lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
                This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
                the other, not both.

        Returns:
            An ASGI application for running Pydantic AI agents with AG-UI protocol support.
        """
        from pydantic_ai.ui.ag_ui.app import AGUIApp

        return AGUIApp(
            agent=self,
            # Agent.iter parameters
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            # Starlette
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )

    def to_a2a(
        self,
        *,
        storage: Storage | None = None,
        broker: Broker | None = None,
        # Agent card
        name: str | None = None,
        url: str = 'http://localhost:8000',
        version: str = '1.0.0',
        description: str | None = None,
        provider: AgentProvider | None = None,
        skills: list[Skill] | None = None,
        # Starlette
        debug: bool = False,
        routes: Sequence[Route] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: dict[Any, ExceptionHandler] | None = None,
        lifespan: Lifespan[FastA2A] | None = None,
    ) -> FastA2A:
        """Convert the agent to a FastA2A application.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')
        app = agent.to_a2a()
        ```

        The `app` is an ASGI application that can be used with any ASGI server.

        To run the application, you can use the following command:

        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```
        """
        from .._a2a import agent_to_a2a

        return agent_to_a2a(
            self,
            storage=storage,
            broker=broker,
            name=name,
            url=url,
            version=version,
            description=description,
            provider=provider,
            skills=skills,
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            lifespan=lifespan,
        )

    async def to_cli(
        self: Self,
        deps: AgentDepsT = None,
        prog_name: str = 'pydantic-ai',
        message_history: Sequence[_messages.ModelMessage] | None = None,
    ) -> None:
        """Run the agent in a CLI chat interface.

        Args:
            deps: The dependencies to pass to the agent.
            prog_name: The name of the program to use for the CLI. Defaults to 'pydantic-ai'.
            message_history: History of the conversation so far.

        Example:
        ```python {title="agent_to_cli.py" test="skip"}
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o', instructions='You always respond in Italian.')

        async def main():
            await agent.to_cli()
        ```
        """
        from rich.console import Console

        from pydantic_ai._cli import run_chat

        await run_chat(
            stream=True,
            agent=self,
            deps=deps,
            console=Console(),
            code_theme='monokai',
            prog_name=prog_name,
            message_history=message_history,
        )

    def to_cli_sync(
        self: Self,
        deps: AgentDepsT = None,
        prog_name: str = 'pydantic-ai',
        message_history: Sequence[_messages.ModelMessage] | None = None,
    ) -> None:
        """Run the agent in a CLI chat interface with the non-async interface.

        Args:
            deps: The dependencies to pass to the agent.
            prog_name: The name of the program to use for the CLI. Defaults to 'pydantic-ai'.
            message_history: History of the conversation so far.

        ```python {title="agent_to_cli_sync.py" test="skip"}
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o', instructions='You always respond in Italian.')
        agent.to_cli_sync()
        agent.to_cli_sync(prog_name='assistant')
        ```
        """
        return _utils.get_event_loop().run_until_complete(
            self.to_cli(deps=deps, prog_name=prog_name, message_history=message_history)
        )

````

#### model

```python
model: Model | KnownModelName | str | None

```

The default model configured for this agent.

#### name

```python
name: str | None

```

The name of the agent, used for logging.

If `None`, we try to infer the agent name from the call frame when the agent is first run.

#### deps_type

```python
deps_type: type

```

The type of dependencies used by the agent.

#### output_type

```python
output_type: OutputSpec[OutputDataT]

```

The type of data output by agent runs, used to validate the data returned by the model, defaults to `str`.

#### event_stream_handler

```python
event_stream_handler: EventStreamHandler[AgentDepsT] | None

```

Optional handler for events from the model's streaming response and the agent's execution of tools.

#### toolsets

```python
toolsets: Sequence[AbstractToolset[AgentDepsT]]

```

All toolsets registered on the agent.

Output tools are not included.

#### run

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[OutputDataT]

```

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[RunOutputDataT]

```

```python
run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[Any]

```

Run the agent with a user prompt in async mode.

This method builds an internal agent graph (using system prompts, tools and output schemas) and then runs the graph to completion. The result of the run is returned.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    agent_run = await agent.run('What is the capital of France?')
    print(agent_run.output)
    #> The capital of France is Paris.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AgentRunResult[Any]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
async def run(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
) -> AgentRunResult[Any]:
    """Run the agent with a user prompt in async mode.

    This method builds an internal agent graph (using system prompts, tools and output schemas) and then
    runs the graph to completion. The result of the run is returned.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        agent_run = await agent.run('What is the capital of France?')
        print(agent_run.output)
        #> The capital of France is Paris.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())

    event_stream_handler = event_stream_handler or self.event_stream_handler

    async with self.iter(
        user_prompt=user_prompt,
        output_type=output_type,
        message_history=message_history,
        deferred_tool_results=deferred_tool_results,
        model=model,
        instructions=instructions,
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        toolsets=toolsets,
        builtin_tools=builtin_tools,
    ) as agent_run:
        async for node in agent_run:
            if event_stream_handler is not None and (
                self.is_model_request_node(node) or self.is_call_tools_node(node)
            ):
                async with node.stream(agent_run.ctx) as stream:
                    await event_stream_handler(_agent_graph.build_run_context(agent_run.ctx), stream)

    assert agent_run.result is not None, 'The graph run did not finish properly'
    return agent_run.result

````

#### run_sync

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[OutputDataT]

```

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[RunOutputDataT]

```

```python
run_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AgentRunResult[Any]

```

Synchronously run the agent with a user prompt.

This is a convenience method that wraps self.run with `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)
#> The capital of Italy is Rome.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AgentRunResult[Any]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
def run_sync(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
) -> AgentRunResult[Any]:
    """Synchronously run the agent with a user prompt.

    This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
    You therefore can't use this method inside async code or if there's an active event loop.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    result_sync = agent.run_sync('What is the capital of Italy?')
    print(result_sync.output)
    #> The capital of Italy is Rome.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())

    return _utils.get_event_loop().run_until_complete(
        self.run(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=False,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            event_stream_handler=event_stream_handler,
        )
    )

````

#### run_stream

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AbstractAsyncContextManager[
    StreamedRunResult[AgentDepsT, OutputDataT]
]

```

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AbstractAsyncContextManager[
    StreamedRunResult[AgentDepsT, RunOutputDataT]
]

```

```python
run_stream(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]

```

Run the agent with a user prompt in async streaming mode.

This method builds an internal agent graph (using system prompts, tools and output schemas) and then runs the graph until the model produces output matching the `output_type`, for example text or structured data. At this point, a streaming run result object is yielded from which you can stream the output as it comes in, and -- once this output has completed streaming -- get the complete output, message history, and usage.

As this method will consider the first output matching the `output_type` to be the final output, it will stop running the agent graph and will not execute any tool calls made by the model after this "final" output. If you want to always run the agent graph to completion and stream events and output at the same time, use agent.run() with an `event_stream_handler` or agent.iter() instead.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.run_stream('What is the capital of the UK?') as response:
        print(await response.get_output())
        #> The capital of the UK is London.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager. Note that it does not receive any events after the final result is found. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[StreamedRunResult[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
@asynccontextmanager
async def run_stream(  # noqa C901
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
) -> AsyncIterator[result.StreamedRunResult[AgentDepsT, Any]]:
    """Run the agent with a user prompt in async streaming mode.

    This method builds an internal agent graph (using system prompts, tools and output schemas) and then
    runs the graph until the model produces output matching the `output_type`, for example text or structured data.
    At this point, a streaming run result object is yielded from which you can stream the output as it comes in,
    and -- once this output has completed streaming -- get the complete output, message history, and usage.

    As this method will consider the first output matching the `output_type` to be the final output,
    it will stop running the agent graph and will not execute any tool calls made by the model after this "final" output.
    If you want to always run the agent graph to completion and stream events and output at the same time,
    use [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] with an `event_stream_handler` or [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] instead.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        async with agent.run_stream('What is the capital of the UK?') as response:
            print(await response.get_output())
            #> The capital of the UK is London.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools for this run.
        event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
            It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.
            Note that it does _not_ receive any events after the final result is found.

    Returns:
        The result of the run.
    """
    if infer_name and self.name is None:
        # f_back because `asynccontextmanager` adds one frame
        if frame := inspect.currentframe():  # pragma: no branch
            self._infer_name(frame.f_back)

    event_stream_handler = event_stream_handler or self.event_stream_handler

    yielded = False
    async with self.iter(
        user_prompt,
        output_type=output_type,
        message_history=message_history,
        deferred_tool_results=deferred_tool_results,
        model=model,
        deps=deps,
        instructions=instructions,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=False,
        toolsets=toolsets,
        builtin_tools=builtin_tools,
    ) as agent_run:
        first_node = agent_run.next_node  # start with the first node
        assert isinstance(first_node, _agent_graph.UserPromptNode)  # the first node should be a user prompt node
        node = first_node
        while True:
            graph_ctx = agent_run.ctx
            if self.is_model_request_node(node):
                async with node.stream(graph_ctx) as stream:
                    final_result_event = None

                    async def stream_to_final(
                        stream: AgentStream,
                    ) -> AsyncIterator[_messages.ModelResponseStreamEvent]:
                        nonlocal final_result_event
                        async for event in stream:
                            yield event
                            if isinstance(event, _messages.FinalResultEvent):
                                final_result_event = event
                                break

                    if event_stream_handler is not None:
                        await event_stream_handler(
                            _agent_graph.build_run_context(graph_ctx), stream_to_final(stream)
                        )
                    else:
                        async for _ in stream_to_final(stream):
                            pass

                    if final_result_event is not None:
                        final_result = FinalResult(
                            None, final_result_event.tool_name, final_result_event.tool_call_id
                        )
                        if yielded:
                            raise exceptions.AgentRunError('Agent run produced final results')  # pragma: no cover
                        yielded = True

                        messages = graph_ctx.state.message_history.copy()

                        async def on_complete() -> None:
                            """Called when the stream has completed.

                            The model response will have been added to messages by now
                            by `StreamedRunResult._marked_completed`.
                            """
                            nonlocal final_result
                            final_result = FinalResult(
                                await stream.get_output(), final_result.tool_name, final_result.tool_call_id
                            )

                            # When we get here, the `ModelRequestNode` has completed streaming after the final result was found.
                            # When running an agent with `agent.run`, we'd then move to `CallToolsNode` to execute the tool calls and
                            # find the final result.
                            # We also want to execute tool calls (in case `agent.end_strategy == 'exhaustive'`) here, but
                            # we don't want to use run the `CallToolsNode` logic to determine the final output, as it would be
                            # wasteful and could produce a different result (e.g. when text output is followed by tool calls).
                            # So we call `process_tool_calls` directly and then end the run with the found final result.

                            parts: list[_messages.ModelRequestPart] = []
                            async for _event in _agent_graph.process_tool_calls(
                                tool_manager=graph_ctx.deps.tool_manager,
                                tool_calls=stream.response.tool_calls,
                                tool_call_results=None,
                                final_result=final_result,
                                ctx=graph_ctx,
                                output_parts=parts,
                            ):
                                pass

                            # For backwards compatibility, append a new ModelRequest using the tool returns and retries
                            if parts:
                                messages.append(_messages.ModelRequest(parts, run_id=graph_ctx.state.run_id))

                            await agent_run.next(_agent_graph.SetFinalResult(final_result))

                        yield StreamedRunResult(
                            messages,
                            graph_ctx.deps.new_message_index,
                            stream,
                            on_complete,
                        )
                        break
            elif self.is_call_tools_node(node) and event_stream_handler is not None:
                async with node.stream(agent_run.ctx) as stream:
                    await event_stream_handler(_agent_graph.build_run_context(agent_run.ctx), stream)

            next_node = await agent_run.next(node)
            if isinstance(next_node, End) and agent_run.result is not None:
                # A final output could have been produced by the CallToolsNode rather than the ModelRequestNode,
                # if a tool function raised CallDeferred or ApprovalRequired.
                # In this case there's no response to stream, but we still let the user access the output etc as normal.
                yield StreamedRunResult(
                    graph_ctx.state.message_history,
                    graph_ctx.deps.new_message_index,
                    run_result=agent_run.result,
                )
                yielded = True
                break
            if not isinstance(next_node, _agent_graph.AgentNode):
                raise exceptions.AgentRunError(  # pragma: no cover
                    'Should have produced a StreamedRunResult before getting here'
                )
            node = cast(_agent_graph.AgentNode[Any, Any], next_node)

    if not yielded:
        raise exceptions.AgentRunError('Agent run finished without producing a final result')  # pragma: no cover

````

#### run_stream_sync

```python
run_stream_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> StreamedRunResultSync[AgentDepsT, OutputDataT]

```

```python
run_stream_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> StreamedRunResultSync[AgentDepsT, RunOutputDataT]

```

```python
run_stream_sync(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None,
    event_stream_handler: (
        EventStreamHandler[AgentDepsT] | None
    ) = None
) -> StreamedRunResultSync[AgentDepsT, Any]

```

Run the agent with a user prompt in sync streaming mode.

This is a convenience method that wraps run_stream() with `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

This method builds an internal agent graph (using system prompts, tools and output schemas) and then runs the graph until the model produces output matching the `output_type`, for example text or structured data. At this point, a streaming run result object is yielded from which you can stream the output as it comes in, and -- once this output has completed streaming -- get the complete output, message history, and usage.

As this method will consider the first output matching the `output_type` to be the final output, it will stop running the agent graph and will not execute any tool calls made by the model after this "final" output. If you want to always run the agent graph to completion and stream events and output at the same time, use agent.run() with an `event_stream_handler` or agent.iter() instead.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

def main():
    response = agent.run_stream_sync('What is the capital of the UK?')
    print(response.get_output())
    #> The capital of the UK is London.

```

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` | | `event_stream_handler` | `EventStreamHandler[AgentDepsT] | None` | Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager. Note that it does not receive any events after the final result is found. | `None` |

Returns:

| Type | Description | | --- | --- | | `StreamedRunResultSync[AgentDepsT, Any]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
def run_stream_sync(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
) -> result.StreamedRunResultSync[AgentDepsT, Any]:
    """Run the agent with a user prompt in sync streaming mode.

    This is a convenience method that wraps [`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] with `loop.run_until_complete(...)`.
    You therefore can't use this method inside async code or if there's an active event loop.

    This method builds an internal agent graph (using system prompts, tools and output schemas) and then
    runs the graph until the model produces output matching the `output_type`, for example text or structured data.
    At this point, a streaming run result object is yielded from which you can stream the output as it comes in,
    and -- once this output has completed streaming -- get the complete output, message history, and usage.

    As this method will consider the first output matching the `output_type` to be the final output,
    it will stop running the agent graph and will not execute any tool calls made by the model after this "final" output.
    If you want to always run the agent graph to completion and stream events and output at the same time,
    use [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] with an `event_stream_handler` or [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] instead.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    def main():
        response = agent.run_stream_sync('What is the capital of the UK?')
        print(response.get_output())
        #> The capital of the UK is London.
    ```

    Args:
        user_prompt: User input to start/continue the conversation.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools for this run.
        event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
            It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.
            Note that it does _not_ receive any events after the final result is found.

    Returns:
        The result of the run.
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())

    async def _consume_stream():
        async with self.run_stream(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            event_stream_handler=event_stream_handler,
        ) as stream_result:
            yield stream_result

    async_result = _utils.get_event_loop().run_until_complete(anext(_consume_stream()))
    return result.StreamedRunResultSync(async_result)

````

#### run_stream_events

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[OutputDataT]
]

```

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]
]

```

```python
run_stream_events(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AsyncIterator[
    AgentStreamEvent | AgentRunResultEvent[Any]
]

```

Run the agent with a user prompt in async mode and stream events from the run.

This is a convenience method that wraps self.run and uses the `event_stream_handler` kwarg to get a stream of events from the run.

Example:

```python
from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

agent = Agent('openai:gpt-4o')

async def main():
    events: list[AgentStreamEvent | AgentRunResultEvent] = []
    async for event in agent.run_stream_events('What is the capital of France?'):
        events.append(event)
    print(events)
    '''
    [
        PartStartEvent(index=0, part=TextPart(content='The capital of ')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
        PartEndEvent(
            index=0, part=TextPart(content='The capital of France is Paris. ')
        ),
        AgentRunResultEvent(
            result=AgentRunResult(output='The capital of France is Paris. ')
        ),
    ]
    '''

```

Arguments are the same as for self.run, except that `event_stream_handler` is now allowed.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]` | An async iterable of stream events AgentStreamEvent and finally a AgentRunResultEvent with the final | | `AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]` | run result. |

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
def run_stream_events(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
    """Run the agent with a user prompt in async mode and stream events from the run.

    This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
    uses the `event_stream_handler` kwarg to get a stream of events from the run.

    Example:
    ```python
    from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

    agent = Agent('openai:gpt-4o')

    async def main():
        events: list[AgentStreamEvent | AgentRunResultEvent] = []
        async for event in agent.run_stream_events('What is the capital of France?'):
            events.append(event)
        print(events)
        '''
        [
            PartStartEvent(index=0, part=TextPart(content='The capital of ')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
            PartEndEvent(
                index=0, part=TextPart(content='The capital of France is Paris. ')
            ),
            AgentRunResultEvent(
                result=AgentRunResult(output='The capital of France is Paris. ')
            ),
        ]
        '''
    ```

    Arguments are the same as for [`self.run`][pydantic_ai.agent.AbstractAgent.run],
    except that `event_stream_handler` is now allowed.

    Args:
        user_prompt: User input to start/continue the conversation.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
        run result.
    """
    if infer_name and self.name is None:
        self._infer_name(inspect.currentframe())

    # unfortunately this hack of returning a generator rather than defining it right here is
    # required to allow overloads of this method to work in python's typing system, or at least with pyright
    # or at least I couldn't make it work without
    return self._run_stream_events(
        user_prompt,
        output_type=output_type,
        message_history=message_history,
        deferred_tool_results=deferred_tool_results,
        model=model,
        instructions=instructions,
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        toolsets=toolsets,
        builtin_tools=builtin_tools,
    )

````

#### iter

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, OutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, RunOutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]

```

A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the stream of events coming from the execution of tools.

The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics, and the final result of the run once it has completed.

For more details, see the documentation of `AgentRun`.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    nodes = []
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

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentRun[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
@asynccontextmanager
@abstractmethod
async def iter(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
    """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

    This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
    `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
    executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
    stream of events coming from the execution of tools.

    The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
    and the final result of the run once it has completed.

    For more details, see the documentation of `AgentRun`.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
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

    Args:
        user_prompt: User input to start/continue the conversation.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    raise NotImplementedError
    yield

````

#### override

```python
override(
    *,
    name: str | Unset = UNSET,
    deps: AgentDepsT | Unset = UNSET,
    model: Model | KnownModelName | str | Unset = UNSET,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | Unset
    ) = UNSET,
    tools: (
        Sequence[
            Tool[AgentDepsT]
            | ToolFuncEither[AgentDepsT, ...]
        ]
        | Unset
    ) = UNSET,
    instructions: Instructions[AgentDepsT] | Unset = UNSET
) -> Iterator[None]

```

Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

This is particularly useful when testing. You can find an example of this [here](../../testing/#overriding-model-via-pytest-fixtures).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | Unset` | The name to use instead of the name passed to the agent constructor and agent run. | `UNSET` | | `deps` | `AgentDepsT | Unset` | The dependencies to use instead of the dependencies passed to the agent run. | `UNSET` | | `model` | `Model | KnownModelName | str | Unset` | The model to use instead of the model passed to the agent run. | `UNSET` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | Unset` | The toolsets to use instead of the toolsets passed to the agent constructor and agent run. | `UNSET` | | `tools` | `Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | Unset` | The tools to use instead of the tools registered with the agent. | `UNSET` | | `instructions` | `Instructions[AgentDepsT] | Unset` | The instructions to use instead of the instructions registered with the agent. | `UNSET` |

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

```python
@contextmanager
@abstractmethod
def override(
    self,
    *,
    name: str | _utils.Unset = _utils.UNSET,
    deps: AgentDepsT | _utils.Unset = _utils.UNSET,
    model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
    instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
) -> Iterator[None]:
    """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

    This is particularly useful when testing.
    You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

    Args:
        name: The name to use instead of the name passed to the agent constructor and agent run.
        deps: The dependencies to use instead of the dependencies passed to the agent run.
        model: The model to use instead of the model passed to the agent run.
        toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
        tools: The tools to use instead of the tools registered with the agent.
        instructions: The instructions to use instead of the instructions registered with the agent.
    """
    raise NotImplementedError
    yield

```

#### sequential_tool_calls

```python
sequential_tool_calls() -> Iterator[None]

```

Run tool calls sequentially during the context.

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

```python
@staticmethod
@contextmanager
def sequential_tool_calls() -> Iterator[None]:
    """Run tool calls sequentially during the context."""
    with ToolManager.sequential_tool_calls():
        yield

```

#### is_model_request_node

```python
is_model_request_node(
    node: AgentNode[T, S] | End[FinalResult[S]],
) -> TypeIs[ModelRequestNode[T, S]]

```

Check if the node is a `ModelRequestNode`, narrowing the type if it is.

This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

```python
@staticmethod
def is_model_request_node(
    node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
) -> TypeIs[_agent_graph.ModelRequestNode[T, S]]:
    """Check if the node is a `ModelRequestNode`, narrowing the type if it is.

    This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
    """
    return isinstance(node, _agent_graph.ModelRequestNode)

```

#### is_call_tools_node

```python
is_call_tools_node(
    node: AgentNode[T, S] | End[FinalResult[S]],
) -> TypeIs[CallToolsNode[T, S]]

```

Check if the node is a `CallToolsNode`, narrowing the type if it is.

This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

```python
@staticmethod
def is_call_tools_node(
    node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
) -> TypeIs[_agent_graph.CallToolsNode[T, S]]:
    """Check if the node is a `CallToolsNode`, narrowing the type if it is.

    This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
    """
    return isinstance(node, _agent_graph.CallToolsNode)

```

#### is_user_prompt_node

```python
is_user_prompt_node(
    node: AgentNode[T, S] | End[FinalResult[S]],
) -> TypeIs[UserPromptNode[T, S]]

```

Check if the node is a `UserPromptNode`, narrowing the type if it is.

This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

```python
@staticmethod
def is_user_prompt_node(
    node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
) -> TypeIs[_agent_graph.UserPromptNode[T, S]]:
    """Check if the node is a `UserPromptNode`, narrowing the type if it is.

    This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
    """
    return isinstance(node, _agent_graph.UserPromptNode)

```

#### is_end_node

```python
is_end_node(
    node: AgentNode[T, S] | End[FinalResult[S]],
) -> TypeIs[End[FinalResult[S]]]

```

Check if the node is a `End`, narrowing the type if it is.

This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

```python
@staticmethod
def is_end_node(
    node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
) -> TypeIs[End[result.FinalResult[S]]]:
    """Check if the node is a `End`, narrowing the type if it is.

    This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
    """
    return isinstance(node, End)

```

#### to_ag_ui

```python
to_ag_ui(
    *,
    output_type: OutputSpec[OutputDataT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    debug: bool = False,
    routes: Sequence[BaseRoute] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: (
        Mapping[Any, ExceptionHandler] | None
    ) = None,
    on_startup: Sequence[Callable[[], Any]] | None = None,
    on_shutdown: Sequence[Callable[[], Any]] | None = None,
    lifespan: (
        Lifespan[AGUIApp[AgentDepsT, OutputDataT]] | None
    ) = None
) -> AGUIApp[AgentDepsT, OutputDataT]

```

Returns an ASGI application that handles every AG-UI request by running the agent.

Note that the `deps` will be the same for each request, with the exception of the AG-UI state that's injected into the `state` field of a `deps` object that implements the StateHandler protocol. To provide different `deps` for each request (e.g. based on the authenticated user), use pydantic_ai.ag_ui.run_ag_ui or pydantic_ai.ag_ui.handle_ag_ui_request instead.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
app = agent.to_ag_ui()

```

The `app` is an ASGI application that can be used with any ASGI server.

To run the application, you can use the following command:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000

```

See [AG-UI docs](../../ui/ag-ui/) for more information.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `output_type` | `OutputSpec[OutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `debug` | `bool` | Boolean indicating if debug tracebacks should be returned on errors. | `False` | | `routes` | `Sequence[BaseRoute] | None` | A list of routes to serve incoming HTTP and WebSocket requests. | `None` | | `middleware` | `Sequence[Middleware] | None` | A list of middleware to run for every request. A starlette application will always automatically include two middleware classes. ServerErrorMiddleware is added as the very outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack. ExceptionMiddleware is added as the very innermost middleware, to deal with handled exception cases occurring in the routing or endpoints. | `None` | | `exception_handlers` | `Mapping[Any, ExceptionHandler] | None` | A mapping of either integer status codes, or exception class types onto callables which handle the exceptions. Exception handler callables should be of the form handler(request, exc) -> response and may be either standard functions, or async functions. | `None` | | `on_startup` | `Sequence[Callable[[], Any]] | None` | A list of callables to run on application startup. Startup handler callables do not take any arguments, and may be either standard functions, or async functions. | `None` | | `on_shutdown` | `Sequence[Callable[[], Any]] | None` | A list of callables to run on application shutdown. Shutdown handler callables do not take any arguments, and may be either standard functions, or async functions. | `None` | | `lifespan` | `Lifespan[AGUIApp[AgentDepsT, OutputDataT]] | None` | A lifespan context function, which can be used to perform startup and shutdown tasks. This is a newer style that replaces the on_startup and on_shutdown handlers. Use one or the other, not both. | `None` |

Returns:

| Type | Description | | --- | --- | | `AGUIApp[AgentDepsT, OutputDataT]` | An ASGI application for running Pydantic AI agents with AG-UI protocol support. |

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
def to_ag_ui(
    self,
    *,
    # Agent.iter parameters
    output_type: OutputSpec[OutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    # Starlette
    debug: bool = False,
    routes: Sequence[BaseRoute] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
    on_startup: Sequence[Callable[[], Any]] | None = None,
    on_shutdown: Sequence[Callable[[], Any]] | None = None,
    lifespan: Lifespan[AGUIApp[AgentDepsT, OutputDataT]] | None = None,
) -> AGUIApp[AgentDepsT, OutputDataT]:
    """Returns an ASGI application that handles every AG-UI request by running the agent.

    Note that the `deps` will be the same for each request, with the exception of the AG-UI state that's
    injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ag_ui.StateHandler] protocol.
    To provide different `deps` for each request (e.g. based on the authenticated user),
    use [`pydantic_ai.ag_ui.run_ag_ui`][pydantic_ai.ag_ui.run_ag_ui] or
    [`pydantic_ai.ag_ui.handle_ag_ui_request`][pydantic_ai.ag_ui.handle_ag_ui_request] instead.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')
    app = agent.to_ag_ui()
    ```

    The `app` is an ASGI application that can be used with any ASGI server.

    To run the application, you can use the following command:

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

    See [AG-UI docs](../ui/ag-ui.md) for more information.

    Args:
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
            no output validators since output validators would expect an argument that matches the agent's
            output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.

        debug: Boolean indicating if debug tracebacks should be returned on errors.
        routes: A list of routes to serve incoming HTTP and WebSocket requests.
        middleware: A list of middleware to run for every request. A starlette application will always
            automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
            outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
            `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
            exception cases occurring in the routing or endpoints.
        exception_handlers: A mapping of either integer status codes, or exception class types onto
            callables which handle the exceptions. Exception handler callables should be of the form
            `handler(request, exc) -> response` and may be either standard functions, or async functions.
        on_startup: A list of callables to run on application startup. Startup handler callables do not
            take any arguments, and may be either standard functions, or async functions.
        on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
            not take any arguments, and may be either standard functions, or async functions.
        lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
            This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
            the other, not both.

    Returns:
        An ASGI application for running Pydantic AI agents with AG-UI protocol support.
    """
    from pydantic_ai.ui.ag_ui.app import AGUIApp

    return AGUIApp(
        agent=self,
        # Agent.iter parameters
        output_type=output_type,
        message_history=message_history,
        deferred_tool_results=deferred_tool_results,
        model=model,
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
        # Starlette
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        lifespan=lifespan,
    )

````

#### to_a2a

```python
to_a2a(
    *,
    storage: Storage | None = None,
    broker: Broker | None = None,
    name: str | None = None,
    url: str = "http://localhost:8000",
    version: str = "1.0.0",
    description: str | None = None,
    provider: AgentProvider | None = None,
    skills: list[Skill] | None = None,
    debug: bool = False,
    routes: Sequence[Route] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: (
        dict[Any, ExceptionHandler] | None
    ) = None,
    lifespan: Lifespan[FastA2A] | None = None
) -> FastA2A

```

Convert the agent to a FastA2A application.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
app = agent.to_a2a()

```

The `app` is an ASGI application that can be used with any ASGI server.

To run the application, you can use the following command:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000

```

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
def to_a2a(
    self,
    *,
    storage: Storage | None = None,
    broker: Broker | None = None,
    # Agent card
    name: str | None = None,
    url: str = 'http://localhost:8000',
    version: str = '1.0.0',
    description: str | None = None,
    provider: AgentProvider | None = None,
    skills: list[Skill] | None = None,
    # Starlette
    debug: bool = False,
    routes: Sequence[Route] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: dict[Any, ExceptionHandler] | None = None,
    lifespan: Lifespan[FastA2A] | None = None,
) -> FastA2A:
    """Convert the agent to a FastA2A application.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')
    app = agent.to_a2a()
    ```

    The `app` is an ASGI application that can be used with any ASGI server.

    To run the application, you can use the following command:

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```
    """
    from .._a2a import agent_to_a2a

    return agent_to_a2a(
        self,
        storage=storage,
        broker=broker,
        name=name,
        url=url,
        version=version,
        description=description,
        provider=provider,
        skills=skills,
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        lifespan=lifespan,
    )

````

#### to_cli

```python
to_cli(
    deps: AgentDepsT = None,
    prog_name: str = "pydantic-ai",
    message_history: Sequence[ModelMessage] | None = None,
) -> None

```

Run the agent in a CLI chat interface.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `deps` | `AgentDepsT` | The dependencies to pass to the agent. | `None` | | `prog_name` | `str` | The name of the program to use for the CLI. Defaults to 'pydantic-ai'. | `'pydantic-ai'` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` |

Example: agent_to_cli.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', instructions='You always respond in Italian.')

async def main():
    await agent.to_cli()

```

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
async def to_cli(
    self: Self,
    deps: AgentDepsT = None,
    prog_name: str = 'pydantic-ai',
    message_history: Sequence[_messages.ModelMessage] | None = None,
) -> None:
    """Run the agent in a CLI chat interface.

    Args:
        deps: The dependencies to pass to the agent.
        prog_name: The name of the program to use for the CLI. Defaults to 'pydantic-ai'.
        message_history: History of the conversation so far.

    Example:
    ```python {title="agent_to_cli.py" test="skip"}
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o', instructions='You always respond in Italian.')

    async def main():
        await agent.to_cli()
    ```
    """
    from rich.console import Console

    from pydantic_ai._cli import run_chat

    await run_chat(
        stream=True,
        agent=self,
        deps=deps,
        console=Console(),
        code_theme='monokai',
        prog_name=prog_name,
        message_history=message_history,
    )

````

#### to_cli_sync

```python
to_cli_sync(
    deps: AgentDepsT = None,
    prog_name: str = "pydantic-ai",
    message_history: Sequence[ModelMessage] | None = None,
) -> None

```

Run the agent in a CLI chat interface with the non-async interface.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `deps` | `AgentDepsT` | The dependencies to pass to the agent. | `None` | | `prog_name` | `str` | The name of the program to use for the CLI. Defaults to 'pydantic-ai'. | `'pydantic-ai'` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` |

agent_to_cli_sync.py

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', instructions='You always respond in Italian.')
agent.to_cli_sync()
agent.to_cli_sync(prog_name='assistant')

```

Source code in `pydantic_ai_slim/pydantic_ai/agent/abstract.py`

````python
def to_cli_sync(
    self: Self,
    deps: AgentDepsT = None,
    prog_name: str = 'pydantic-ai',
    message_history: Sequence[_messages.ModelMessage] | None = None,
) -> None:
    """Run the agent in a CLI chat interface with the non-async interface.

    Args:
        deps: The dependencies to pass to the agent.
        prog_name: The name of the program to use for the CLI. Defaults to 'pydantic-ai'.
        message_history: History of the conversation so far.

    ```python {title="agent_to_cli_sync.py" test="skip"}
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o', instructions='You always respond in Italian.')
    agent.to_cli_sync()
    agent.to_cli_sync(prog_name='assistant')
    ```
    """
    return _utils.get_event_loop().run_until_complete(
        self.to_cli(deps=deps, prog_name=prog_name, message_history=message_history)
    )

````

### WrapperAgent

Bases: `AbstractAgent[AgentDepsT, OutputDataT]`

Agent which wraps another agent.

Does nothing on its own, used as a base class.

Source code in `pydantic_ai_slim/pydantic_ai/agent/wrapper.py`

````python
class WrapperAgent(AbstractAgent[AgentDepsT, OutputDataT]):
    """Agent which wraps another agent.

    Does nothing on its own, used as a base class.
    """

    def __init__(self, wrapped: AbstractAgent[AgentDepsT, OutputDataT]):
        self.wrapped = wrapped

    @property
    def model(self) -> models.Model | models.KnownModelName | str | None:
        return self.wrapped.model

    @property
    def name(self) -> str | None:
        return self.wrapped.name

    @name.setter
    def name(self, value: str | None) -> None:
        self.wrapped.name = value

    @property
    def deps_type(self) -> type:
        return self.wrapped.deps_type

    @property
    def output_type(self) -> OutputSpec[OutputDataT]:
        return self.wrapped.output_type

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        return self.wrapped.event_stream_handler

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        return self.wrapped.toolsets

    async def __aenter__(self) -> AbstractAgent[AgentDepsT, OutputDataT]:
        return await self.wrapped.__aenter__()

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
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

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        async with self.wrapped.iter(
            user_prompt=user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        ) as run:
            yield run

    @contextmanager
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            instructions: The instructions to use instead of the instructions registered with the agent.
        """
        with self.wrapped.override(
            name=name,
            deps=deps,
            model=model,
            toolsets=toolsets,
            tools=tools,
            instructions=instructions,
        ):
            yield

````

#### iter

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, OutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT],
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AbstractAsyncContextManager[
    AgentRun[AgentDepsT, RunOutputDataT]
]

```

```python
iter(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: (
        DeferredToolResults | None
    ) = None,
    model: Model | KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | None
    ) = None,
    builtin_tools: (
        Sequence[AbstractBuiltinTool] | None
    ) = None
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]

```

A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the stream of events coming from the execution of tools.

The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics, and the final result of the run once it has completed.

For more details, see the documentation of `AgentRun`.

Example:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    nodes = []
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

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `user_prompt` | `str | Sequence[UserContent] | None` | User input to start/continue the conversation. | `None` | | `output_type` | `OutputSpec[RunOutputDataT] | None` | Custom output type to use for this run, output_type may only be used if the agent has no output validators since output validators would expect an argument that matches the agent's output type. | `None` | | `message_history` | `Sequence[ModelMessage] | None` | History of the conversation so far. | `None` | | `deferred_tool_results` | `DeferredToolResults | None` | Optional results for deferred tool calls in the message history. | `None` | | `model` | `Model | KnownModelName | str | None` | Optional model to use for this run, required if model was not set when creating the agent. | `None` | | `instructions` | `Instructions[AgentDepsT]` | Optional additional instructions to use for this run. | `None` | | `deps` | `AgentDepsT` | Optional dependencies to use for this run. | `None` | | `model_settings` | `ModelSettings | None` | Optional settings to use for this model's request. | `None` | | `usage_limits` | `UsageLimits | None` | Optional limits on model request count or token usage. | `None` | | `usage` | `RunUsage | None` | Optional usage to start with, useful for resuming a conversation or agents used in tools. | `None` | | `infer_name` | `bool` | Whether to try to infer the agent name from the call frame if it's not set. | `True` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | None` | Optional additional toolsets for this run. | `None` | | `builtin_tools` | `Sequence[AbstractBuiltinTool] | None` | Optional additional builtin tools for this run. | `None` |

Returns:

| Type | Description | | --- | --- | | `AsyncIterator[AgentRun[AgentDepsT, Any]]` | The result of the run. |

Source code in `pydantic_ai_slim/pydantic_ai/agent/wrapper.py`

````python
@asynccontextmanager
async def iter(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    instructions: Instructions[AgentDepsT] = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
    """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

    This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
    `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
    executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
    stream of events coming from the execution of tools.

    The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
    and the final result of the run once it has completed.

    For more details, see the documentation of `AgentRun`.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
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

    Args:
        user_prompt: User input to start/continue the conversation.
        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        instructions: Optional additional instructions to use for this run.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        builtin_tools: Optional additional builtin tools for this run.

    Returns:
        The result of the run.
    """
    async with self.wrapped.iter(
        user_prompt=user_prompt,
        output_type=output_type,
        message_history=message_history,
        deferred_tool_results=deferred_tool_results,
        model=model,
        instructions=instructions,
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
        builtin_tools=builtin_tools,
    ) as run:
        yield run

````

#### override

```python
override(
    *,
    name: str | Unset = UNSET,
    deps: AgentDepsT | Unset = UNSET,
    model: Model | KnownModelName | str | Unset = UNSET,
    toolsets: (
        Sequence[AbstractToolset[AgentDepsT]] | Unset
    ) = UNSET,
    tools: (
        Sequence[
            Tool[AgentDepsT]
            | ToolFuncEither[AgentDepsT, ...]
        ]
        | Unset
    ) = UNSET,
    instructions: Instructions[AgentDepsT] | Unset = UNSET
) -> Iterator[None]

```

Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

This is particularly useful when testing. You can find an example of this [here](../../testing/#overriding-model-via-pytest-fixtures).

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `name` | `str | Unset` | The name to use instead of the name passed to the agent constructor and agent run. | `UNSET` | | `deps` | `AgentDepsT | Unset` | The dependencies to use instead of the dependencies passed to the agent run. | `UNSET` | | `model` | `Model | KnownModelName | str | Unset` | The model to use instead of the model passed to the agent run. | `UNSET` | | `toolsets` | `Sequence[AbstractToolset[AgentDepsT]] | Unset` | The toolsets to use instead of the toolsets passed to the agent constructor and agent run. | `UNSET` | | `tools` | `Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | Unset` | The tools to use instead of the tools registered with the agent. | `UNSET` | | `instructions` | `Instructions[AgentDepsT] | Unset` | The instructions to use instead of the instructions registered with the agent. | `UNSET` |

Source code in `pydantic_ai_slim/pydantic_ai/agent/wrapper.py`

```python
@contextmanager
def override(
    self,
    *,
    name: str | _utils.Unset = _utils.UNSET,
    deps: AgentDepsT | _utils.Unset = _utils.UNSET,
    model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
    instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
) -> Iterator[None]:
    """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

    This is particularly useful when testing.
    You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

    Args:
        name: The name to use instead of the name passed to the agent constructor and agent run.
        deps: The dependencies to use instead of the dependencies passed to the agent run.
        model: The model to use instead of the model passed to the agent run.
        toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
        tools: The tools to use instead of the tools registered with the agent.
        instructions: The instructions to use instead of the instructions registered with the agent.
    """
    with self.wrapped.override(
        name=name,
        deps=deps,
        model=model,
        toolsets=toolsets,
        tools=tools,
        instructions=instructions,
    ):
        yield

```

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

### EndStrategy

```python
EndStrategy = Literal['early', 'exhaustive']

```

The strategy for handling multiple tool calls when a final result is found.

- `'early'`: Stop processing other tool calls once a final result is found
- `'exhaustive'`: Process all tool calls even after finding a final result

### RunOutputDataT

```python
RunOutputDataT = TypeVar('RunOutputDataT')

```

Type variable for the result data of a run where `output_type` was customized on the run call.

### capture_run_messages

```python
capture_run_messages() -> Iterator[list[ModelMessage]]

```

Context manager to access the messages used in a run, run_sync, or run_stream call.

Useful when a run may raise an exception, see [model errors](../../agents/#model-errors) for more information.

Examples:

```python
from pydantic_ai import Agent, capture_run_messages

agent = Agent('test')

with capture_run_messages() as messages:
    try:
        result = agent.run_sync('foobar')
    except Exception:
        print(messages)
        raise

```

Note

If you call `run`, `run_sync`, or `run_stream` more than once within a single `capture_run_messages` context, `messages` will represent the messages exchanged during the first call only.

Source code in `pydantic_ai_slim/pydantic_ai/_agent_graph.py`

````python
@contextmanager
def capture_run_messages() -> Iterator[list[_messages.ModelMessage]]:
    """Context manager to access the messages used in a [`run`][pydantic_ai.agent.AbstractAgent.run], [`run_sync`][pydantic_ai.agent.AbstractAgent.run_sync], or [`run_stream`][pydantic_ai.agent.AbstractAgent.run_stream] call.

    Useful when a run may raise an exception, see [model errors](../agents.md#model-errors) for more information.

    Examples:
    ```python
    from pydantic_ai import Agent, capture_run_messages

    agent = Agent('test')

    with capture_run_messages() as messages:
        try:
            result = agent.run_sync('foobar')
        except Exception:
            print(messages)
            raise
    ```

    !!! note
        If you call `run`, `run_sync`, or `run_stream` more than once within a single `capture_run_messages` context,
        `messages` will represent the messages exchanged during the first call only.
    """
    token = None
    messages: list[_messages.ModelMessage] = []

    # Try to reuse existing message context if available
    try:
        messages = _messages_ctx_var.get().messages
    except LookupError:
        # No existing context, create a new one
        token = _messages_ctx_var.set(_RunMessages(messages))

    try:
        yield messages
    finally:
        # Clean up context if we created it
        if token is not None:
            _messages_ctx_var.reset(token)

````

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

### EventStreamHandler

```python
EventStreamHandler: TypeAlias = Callable[
    [
        RunContext[AgentDepsT],
        AsyncIterable[AgentStreamEvent],
    ],
    Awaitable[None],
]

```

A function that receives agent RunContext and an async iterable of events from the model's streaming response and the agent's execution of tools.

