<!-- Source: _complete-doc.md -->

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

ask_agent = Agent('openai:gpt-5', output_type=str)


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
    'openai:gpt-5',
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

The mermaid diagram generated in this example looks like this:

```
---
title: question_graph
---
stateDiagram-v2
  [*] --> Ask
  Ask --> Answer: ask the question
  Answer --> Evaluate: answer the question
  Evaluate --> Congratulate
  Evaluate --> Castigate
  Congratulate --> [*]: success
  Castigate --> Ask: try again
```

# RAG

RAG search example. This demo allows you to ask question of the [logfire](https://pydantic.dev/logfire) documentation.

Demonstrates:

- [tools](../../tools/)
- [agent dependencies](../../dependencies/)
- RAG search

This is done by creating a database containing each section of the markdown documentation, then registering the search tool with the Pydantic AI agent.

Logic for extracting sections from markdown files and a JSON file with that data is available in [this gist](https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992).

[PostgreSQL with pgvector](https://github.com/pgvector/pgvector) is used as the search database, the easiest way to download and run pgvector is using Docker:

```bash
mkdir postgres-data
docker run --rm \
  -e POSTGRES_PASSWORD=postgres \
  -p 54320:5432 \
  -v `pwd`/postgres-data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17

```

As with the [SQL gen](../sql-gen/) example, we run postgres on port `54320` to avoid conflicts with any other postgres instances you may have running. We also mount the PostgreSQL `data` directory locally to persist the data if you need to stop and restart the container.

With that running and [dependencies installed and environment variables set](../setup/#usage), we can build the search database with (**WARNING**: this requires the `OPENAI_API_KEY` env variable and will calling the OpenAI embedding API around 300 times to generate embeddings for each section of the documentation):

```bash
python -m pydantic_ai_examples.rag build

```

```bash
uv run -m pydantic_ai_examples.rag build

```

(Note building the database doesn't use Pydantic AI right now, instead it uses the OpenAI SDK directly.)

You can then ask the agent a question with:

```bash
python -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"

```

```bash
uv run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"

```

## Example Code

[Learn about Gateway](../../gateway) [rag.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/rag.py)

```python
"""RAG example with pydantic-ai â€” using vector search to augment a chat agent.

Run pgvector with:

    mkdir postgres-data
    docker run --rm -e POSTGRES_PASSWORD=postgres \
        -p 54320:5432 \
        -v `pwd`/postgres-data:/var/lib/postgresql/data \
        pgvector/pgvector:pg17

Build the search DB with:

    uv run -m pydantic_ai_examples.rag build

Ask the agent a question with:

    uv run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
"""

from __future__ import annotations as _annotations

import asyncio
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import httpx
import logfire
import pydantic_core
from anyio import create_task_group
from openai import AsyncOpenAI
from pydantic import TypeAdapter
from typing_extensions import AsyncGenerator

from pydantic_ai import Agent, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool


agent = Agent('gateway/openai:gpt-5', deps_type=Deps)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    with logfire.span(
        'create embedding for {search_query=}', search_query=search_query
    ):
        embedding = await context.deps.openai.embeddings.create(
            input=search_query,
            model='text-embedding-3-small',
        )

    assert len(embedding.data) == 1, (
        f'Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}'
    )
    embedding = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
        embedding_json,
    )
    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for row in rows
    )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as pool:
        deps = Deps(openai=openai, pool=pool)
        answer = await agent.run(question, deps=deps)
    print(answer.output)


#######################################################
# The rest of this file is dedicated to preparing the #
# search database, and some utilities.                #
#######################################################

# JSON document from
# https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992
DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)


async def build_search_db():
    """Build the search database."""
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sections_ta.validate_json(response.content)

    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    async with database_connect(True) as pool:
        with logfire.span('create schema'):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with create_task_group() as tg:
            for section in sections:
                tg.start_soon(insert_doc_section, sem, openai, pool, section)


async def insert_doc_section(
    sem: asyncio.Semaphore,
    openai: AsyncOpenAI,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            logfire.info('Skipping {url=}', url=url)
            return

        with logfire.span('create embedding for {url=}', url=url):
            embedding = await openai.embeddings.create(
                input=section.embedding_content(),
                model='text-embedding-3-small',
            )
        assert len(embedding.data) == 1, (
            f'Expected 1 embedding, got {len(embedding.data)}, doc section: {section}'
        )
        embedding = embedding.data[0].embedding
        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
            url,
            section.title,
            section.content,
            embedding_json,
        )


@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r'\.md$', '', self.path)
        return (
            f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, "-")}'
        )

    def embedding_content(self) -> str:
        return '\n\n'.join((f'path: {self.path}', f'title: {self.title}', self.content))


sections_ta = TypeAdapter(list[DocsSection])


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn, database = (
        'postgresql://postgres:postgres@localhost:54320',
        'pydantic_ai_rag',
    )
    if create_db:
        with logfire.span('check and create DB'):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval(
                    'SELECT 1 FROM pg_database WHERE datname = $1', database
                )
                if not db_exists:
                    await conn.execute(f'CREATE DATABASE {database}')
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- text-embedding-3-small returns a vector of 1536 floats
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `Å¾lutÃ½` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == 'build':
        asyncio.run(build_search_db())
    elif action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'How do I configure logfire to work with FastAPI?'
        asyncio.run(run_agent(q))
    else:
        print(
            'uv run --extra examples -m pydantic_ai_examples.rag build|search',
            file=sys.stderr,
        )
        sys.exit(1)

```

[rag.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/rag.py)

```python
"""RAG example with pydantic-ai â€” using vector search to augment a chat agent.

Run pgvector with:

    mkdir postgres-data
    docker run --rm -e POSTGRES_PASSWORD=postgres \
        -p 54320:5432 \
        -v `pwd`/postgres-data:/var/lib/postgresql/data \
        pgvector/pgvector:pg17

Build the search DB with:

    uv run -m pydantic_ai_examples.rag build

Ask the agent a question with:

    uv run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
"""

from __future__ import annotations as _annotations

import asyncio
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import httpx
import logfire
import pydantic_core
from anyio import create_task_group
from openai import AsyncOpenAI
from pydantic import TypeAdapter
from typing_extensions import AsyncGenerator

from pydantic_ai import Agent, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    openai: AsyncOpenAI
    pool: asyncpg.Pool


agent = Agent('openai:gpt-5', deps_type=Deps)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    with logfire.span(
        'create embedding for {search_query=}', search_query=search_query
    ):
        embedding = await context.deps.openai.embeddings.create(
            input=search_query,
            model='text-embedding-3-small',
        )

    assert len(embedding.data) == 1, (
        f'Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}'
    )
    embedding = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
        embedding_json,
    )
    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for row in rows
    )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as pool:
        deps = Deps(openai=openai, pool=pool)
        answer = await agent.run(question, deps=deps)
    print(answer.output)


#######################################################
# The rest of this file is dedicated to preparing the #
# search database, and some utilities.                #
#######################################################

# JSON document from
# https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992
DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)


async def build_search_db():
    """Build the search database."""
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sections_ta.validate_json(response.content)

    openai = AsyncOpenAI()
    logfire.instrument_openai(openai)

    async with database_connect(True) as pool:
        with logfire.span('create schema'):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with create_task_group() as tg:
            for section in sections:
                tg.start_soon(insert_doc_section, sem, openai, pool, section)


async def insert_doc_section(
    sem: asyncio.Semaphore,
    openai: AsyncOpenAI,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            logfire.info('Skipping {url=}', url=url)
            return

        with logfire.span('create embedding for {url=}', url=url):
            embedding = await openai.embeddings.create(
                input=section.embedding_content(),
                model='text-embedding-3-small',
            )
        assert len(embedding.data) == 1, (
            f'Expected 1 embedding, got {len(embedding.data)}, doc section: {section}'
        )
        embedding = embedding.data[0].embedding
        embedding_json = pydantic_core.to_json(embedding).decode()
        await pool.execute(
            'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
            url,
            section.title,
            section.content,
            embedding_json,
        )


@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r'\.md$', '', self.path)
        return (
            f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, "-")}'
        )

    def embedding_content(self) -> str:
        return '\n\n'.join((f'path: {self.path}', f'title: {self.title}', self.content))


sections_ta = TypeAdapter(list[DocsSection])


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn, database = (
        'postgresql://postgres:postgres@localhost:54320',
        'pydantic_ai_rag',
    )
    if create_db:
        with logfire.span('check and create DB'):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval(
                    'SELECT 1 FROM pg_database WHERE datname = $1', database
                )
                if not db_exists:
                    await conn.execute(f'CREATE DATABASE {database}')
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- text-embedding-3-small returns a vector of 1536 floats
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `Å¾lutÃ½` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == 'build':
        asyncio.run(build_search_db())
    elif action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'How do I configure logfire to work with FastAPI?'
        asyncio.run(run_agent(q))
    else:
        print(
            'uv run --extra examples -m pydantic_ai_examples.rag build|search',
            file=sys.stderr,
        )
        sys.exit(1)

```

# Examples

Here we include some examples of how to use Pydantic AI and what it can do.

## Usage

These examples are distributed with `pydantic-ai` so you can run them either by cloning the [pydantic-ai repo](https://github.com/pydantic/pydantic-ai) or by simply installing `pydantic-ai` from PyPI with `pip` or `uv`.

### Installing required dependencies

Either way you'll need to install extra dependencies to run some examples, you just need to install the `examples` optional dependency group.

If you've installed `pydantic-ai` via pip/uv, you can install the extra dependencies with:

```bash
pip install "pydantic-ai[examples]"

```

```bash
uv add "pydantic-ai[examples]"

```

If you clone the repo, you should instead use `uv sync --extra examples` to install extra dependencies.

### Setting model environment variables

These examples will need you to set up authentication with one or more of the LLMs, see the [model configuration](../../models/overview/) docs for details on how to do this.

TL;DR: in most cases you'll need to set one of the following environment variables:

```bash
export OPENAI_API_KEY=your-api-key

```

```bash
export GEMINI_API_KEY=your-api-key

```

### Running Examples

To run the examples (this will work whether you installed `pydantic_ai`, or cloned the repo), run:

```bash
python -m pydantic_ai_examples.<example_module_name>

```

```bash
uv run -m pydantic_ai_examples.<example_module_name>

```

For examples, to run the very simple [`pydantic_model`](../pydantic-model/) example:

```bash
python -m pydantic_ai_examples.pydantic_model

```

```bash
uv run -m pydantic_ai_examples.pydantic_model

```

If you like one-liners and you're using uv, you can run a pydantic-ai example with zero setup:

```bash
OPENAI_API_KEY='your-api-key' \
  uv run --with "pydantic-ai[examples]" \
  -m pydantic_ai_examples.pydantic_model

```

______________________________________________________________________

You'll probably want to edit examples in addition to just running them. You can copy the examples to a new directory with:

```bash
python -m pydantic_ai_examples --copy-to examples/

```

```bash
uv run -m pydantic_ai_examples --copy-to examples/

```

# Slack Lead Qualifier with Modal

In this example, we're going to build an agentic app that:

- automatically researches each new member that joins a company's public Slack community to see how good of a fit they are for the company's commercial product,
- sends this analysis into a (private) Slack channel, and
- sends a daily summary of the top 5 leads from the previous 24 hours into a (different) Slack channel.

We'll be deploying the app on [Modal](https://modal.com), as it lets you use Python to define an app with web endpoints, scheduled functions, and background functions, and deploy them with a CLI, without needing to set up or manage any infrastructure. It's a great way to lower the barrier for people in your organization to start building and deploying AI agents to make their jobs easier.

We also add [Pydantic Logfire](https://pydantic.dev/logfire) to get observability into the app and agent as they're running in response to webhooks and the schedule

## Screenshots

This is what the analysis sent into Slack will look like:

This is what the corresponding trace in [Logfire](https://pydantic.dev/logfire) will look like:

All of these entries can be clicked on to get more details about what happened at that step, including the full conversation with the LLM and HTTP requests and responses.

## Prerequisites

If you just want to see the code without actually going through the effort of setting up the bits necessary to run it, feel free to [jump ahead](#the-code).

### Slack app

You need to have a Slack workspace and the necessary permissions to create apps.

2. Create a new Slack app using the instructions at <https://docs.slack.dev/quickstart>.

   1. In step 2, "Requesting scopes", request the following scopes:
      - [`users.read`](https://docs.slack.dev/reference/scopes/users.read)
      - [`users.read.email`](https://docs.slack.dev/reference/scopes/users.read.email)
      - [`users.profile.read`](https://docs.slack.dev/reference/scopes/users.profile.read)
   1. In step 3, "Installing and authorizing the app", note down the Access Token as we're going to need to store it as a Secret in Modal.
   1. You can skip steps 4 and 5. We're going to need to subscribe to the `team_join` event, but at this point you don't have a webhook URL yet.

1. Create the channels the app will post into, and add the Slack app to them:

   - `#new-slack-leads`
   - `#daily-slack-leads-summary`

   These names are hard-coded in the example. If you want to use different channels, you can clone the repo and change them in `examples/pydantic_examples/slack_lead_qualifier/functions.py`.

### Logfire Write Token

1. If you don't have a Logfire account yet, create one on <https://logfire-us.pydantic.dev/>.
1. Create a new project named, for example, `slack-lead-qualifier`.
1. Generate a new Write Token and note it down, as we're going to need to store it as a Secret in Modal.

### OpenAI API Key

1. If you don't have an OpenAI account yet, create one on <https://platform.openai.com/>.
1. Create a new API Key in Settings and note it down, as we're going to need to store it as a Secret in Modal.

### Modal account

1. If you don't have a Modal account yet, create one on <https://modal.com/signup>.
1. Create 3 Secrets of type "Custom" on <https://modal.com/secrets>:
   - Name: `slack`, key: `SLACK_API_KEY`, value: the Slack Access Token you generated earlier
   - Name: `logfire`, key: `LOGFIRE_TOKEN`, value: the Logfire Write Token you generated earlier
   - Name: `openai`, key: `OPENAI_API_KEY`, value: the OpenAI API Key you generated earlier

## Usage

1. Make sure you have the [dependencies installed](../setup/#usage).

1. Authenticate with Modal:

   ```bash
   python/uv-run -m modal setup

   ```

1. Run the example as an [ephemeral Modal app](https://modal.com/docs/guide/apps#ephemeral-apps), meaning it will only run until you quit it using Ctrl+C:

   ```bash
   python/uv-run -m modal serve -m pydantic_ai_examples.slack_lead_qualifier.modal

   ```

1. Note down the URL after `Created web function web_app =>`, this is your webhook endpoint URL.

1. Go back to <https://docs.slack.dev/quickstart> and follow step 4, "Configuring the app for event listening", to subscribe to the `team_join` event with the webhook endpoint URL you noted down as the Request URL.

Now when someone new (possibly you with a throwaway email) joins the Slack workspace, you'll see the webhook event being processed in the terminal where you ran `modal serve` and in the Logfire Live view, and after waiting a few seconds you should see the result appear in the `#new-slack-leads` Slack channel!

Faking a Slack signup

You can also fake a Slack signup event and try out the agent like this, with any name or email you please:

```bash
curl -X POST <webhook endpoint URL> \
-H "Content-Type: application/json" \
-d '{
    "type": "event_callback",
    "event": {
        "type": "team_join",
        "user": {
            "profile": {
                "email": "samuel@pydantic.dev",
                "first_name": "Samuel",
                "last_name": "Colvin",
                "display_name": "Samuel Colvin"
            }
        }
    }
}'

```

Deploying to production

If you'd like to deploy this app into your Modal workspace in a persistent fashion, you can use this command:

```bash
python/uv-run -m modal deploy -m pydantic_ai_examples.slack_lead_qualifier.modal

```

You'll likely want to [download the code](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples/slack_lead_qualifier) first, put it in a new repo, and then do [continuous deployment](https://modal.com/docs/guide/continuous-deployment#github-actions) using GitHub Actions.

Don't forget to update the Slack event request URL to the new persistent URL! You'll also want to modify the [instructions for the agent](#agent) to your own situation.

## The code

We're going to start with the basics, and then gradually build up into the full app.

### Models

#### `Profile`

First, we define a [Pydantic](https://docs.pydantic.dev) model that represents a Slack user profile. These are the fields we get from the [`team_join`](https://docs.slack.dev/reference/events/team_join) event that's sent to the webhook endpoint that we'll define in a bit.

[slack_lead_qualifier/models.py (L11-L15)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/models.py#L11-L15)

```python
...

class Profile(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    email: str

...

```

We also define a `Profile.as_prompt()` helper method that uses format_as_xml to turn the profile into a string that can be sent to the model.

[slack_lead_qualifier/models.py (L7-L19)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/models.py#L7-L19)

```python
...

from pydantic_ai import format_as_xml

...

class Profile(BaseModel):

...

    def as_prompt(self) -> str:
        return format_as_xml(self, root_tag='profile')

...

```

#### `Analysis`

The second model we'll need represents the result of the analysis that the agent will perform. We include docstrings to provide additional context to the model on what these fields should contain.

[slack_lead_qualifier/models.py (L23-L31)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/models.py#L23-L31)

```python
...

class Analysis(BaseModel):
    profile: Profile
    organization_name: str
    organization_domain: str
    job_title: str
    relevance: Annotated[int, Ge(1), Le(5)]
    """Estimated fit for Pydantic Logfire: 1 = low, 5 = high"""
    summary: str
    """One-sentence welcome note summarising who they are and how we might help"""

...

```

We also define a `Analysis.as_slack_blocks()` helper method that turns the analysis into some [Slack blocks](https://api.slack.com/reference/block-kit/blocks) that can be sent to the Slack API to post a new message.

[slack_lead_qualifier/models.py (L23-L46)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/models.py#L23-L46)

```python
...

class Analysis(BaseModel):

...

    def as_slack_blocks(self, include_relevance: bool = False) -> list[dict[str, Any]]:
        profile = self.profile
        relevance = f'({self.relevance}/5)' if include_relevance else ''
        return [
            {
                'type': 'markdown',
                'text': f'[{profile.display_name}](mailto:{profile.email}), {self.job_title} at [**{self.organization_name}**](https://{self.organization_domain}) {relevance}',
            },
            {
                'type': 'markdown',
                'text': self.summary,
            },
        ]

```

### Agent

Now it's time to get into Pydantic AI and define the agent that will do the actual analysis!

We specify the model we'll use (`openai:gpt-5`), provide [instructions](../../agents/#instructions), give the agent access to the [DuckDuckGo search tool](../../common-tools/#duckduckgo-search-tool), and tell it to output either an `Analysis` or `None` using the [Native Output](../../output/#native-output) structured output mode.

The real meat of the app is in the instructions that tell the agent how to evaluate each new Slack member. If you plan to use this app yourself, you'll of course want to modify them to your own situation.

[Learn about Gateway](../../gateway) [slack_lead_qualifier/agent.py (L7-L40)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/agent.py#L7-L40)

```python
...

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

...

agent = Agent(
    'gateway/openai:gpt-5',
    instructions=dedent(
        """
        When a new person joins our public Slack, please put together a brief snapshot so we can be most useful to them.

        **What to include**

        1. **Who they are:**  Any details about their professional role or projects (e.g. LinkedIn, GitHub, company bio).
        2. **Where they work:**  Name of the organisation and its domain.
        3. **How we can help:**  On a scale of 1â€“5, estimate how likely they are to benefit from **Pydantic Logfire**
           (our paid observability tool) based on factors such as company size, product maturity, or AI usage.
           *1 = probably not relevant, 5 = very strong fit.*

        **Our products (for context only)**
        â€¢ **Pydantic Validation** â€“ Python data-validation (open source)
        â€¢ **Pydantic AI** â€“ Python agent framework (open source)
        â€¢ **Pydantic Logfire** â€“ Observability for traces, logs & metrics with first-class AI support (commercial)

        **How to research**

        â€¢ Use the provided DuckDuckGo search tool to research the person and the organization they work for, based on the email domain or what you find on e.g. LinkedIn and GitHub.
        â€¢ If you can't find enough to form a reasonable view, return **None**.
        """
    ),
    tools=[duckduckgo_search_tool()],
    output_type=NativeOutput([Analysis, NoneType]),
)

...

```

[slack_lead_qualifier/agent.py (L7-L40)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/agent.py#L7-L40)

```python
...

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

...

agent = Agent(
    'openai:gpt-5',
    instructions=dedent(
        """
        When a new person joins our public Slack, please put together a brief snapshot so we can be most useful to them.

        **What to include**

        1. **Who they are:**  Any details about their professional role or projects (e.g. LinkedIn, GitHub, company bio).
        2. **Where they work:**  Name of the organisation and its domain.
        3. **How we can help:**  On a scale of 1â€“5, estimate how likely they are to benefit from **Pydantic Logfire**
           (our paid observability tool) based on factors such as company size, product maturity, or AI usage.
           *1 = probably not relevant, 5 = very strong fit.*

        **Our products (for context only)**
        â€¢ **Pydantic Validation** â€“ Python data-validation (open source)
        â€¢ **Pydantic AI** â€“ Python agent framework (open source)
        â€¢ **Pydantic Logfire** â€“ Observability for traces, logs & metrics with first-class AI support (commercial)

        **How to research**

        â€¢ Use the provided DuckDuckGo search tool to research the person and the organization they work for, based on the email domain or what you find on e.g. LinkedIn and GitHub.
        â€¢ If you can't find enough to form a reasonable view, return **None**.
        """
    ),
    tools=[duckduckgo_search_tool()],
    output_type=NativeOutput([Analysis, NoneType]),
)

...

```

#### `analyze_profile`

We also define a `analyze_profile` helper function that takes a `Profile`, runs the agent, and returns an `Analysis` (or `None`), and instrument it using [Logfire](../../logfire/).

[slack_lead_qualifier/agent.py (L44-L47)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/agent.py#L44-L47)

```python
...

@logfire.instrument('Analyze profile')
async def analyze_profile(profile: Profile) -> Analysis | None:
    result = await agent.run(profile.as_prompt())
    return result.output

```

### Analysis store

The next building block we'll need is a place to store all the analyses that have been done so that we can look them up when we send the daily summary.

Fortunately, Modal provides us with a convenient way to store some data that can be read back in a subsequent Modal run (webhook or scheduled): [`modal.Dict`](https://modal.com/docs/reference/modal.Dict).

We define some convenience methods to easily add, list, and clear analyses.

[slack_lead_qualifier/store.py (L4-L31)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/store.py#L4-L31)

```python
...

import modal

...

class AnalysisStore:
    @classmethod
    @logfire.instrument('Add analysis to store')
    async def add(cls, analysis: Analysis):
        await cls._get_store().put.aio(analysis.profile.email, analysis.model_dump())

    @classmethod
    @logfire.instrument('List analyses from store')
    async def list(cls) -> list[Analysis]:
        return [
            Analysis.model_validate(analysis)
            async for analysis in cls._get_store().values.aio()
        ]

    @classmethod
    @logfire.instrument('Clear analyses from store')
    async def clear(cls):
        await cls._get_store().clear.aio()

    @classmethod
    def _get_store(cls) -> modal.Dict:
        return modal.Dict.from_name('analyses', create_if_missing=True)  # type: ignore

```

Note

Note that `# type: ignore` on the last line -- unfortunately `modal` does not fully define its types, so we need this to stop our static type checker `pyright`, which we run over all Pydantic AI code including examples, from complaining.

### Send Slack message

Next, we'll need a way to actually send a Slack message, so we define a simple function that uses Slack's [`chat.postMessage`](https://api.slack.com/methods/chat.postMessage) API.

[slack_lead_qualifier/slack.py (L8-L30)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/slack.py#L8-L30)

```python
...

API_KEY = os.getenv('SLACK_API_KEY')
assert API_KEY, 'SLACK_API_KEY is not set'


@logfire.instrument('Send Slack message')
async def send_slack_message(channel: str, blocks: list[dict[str, Any]]):
    client = httpx.AsyncClient()
    response = await client.post(
        'https://slack.com/api/chat.postMessage',
        json={
            'channel': channel,
            'blocks': blocks,
        },
        headers={
            'Authorization': f'Bearer {API_KEY}',
        },
        timeout=5,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get('ok', False):
        error = result.get('error', 'Unknown error')
        raise Exception(f'Failed to send to Slack: {error}')

```

### Features

Now we can start putting these building blocks together to implement the actual features we want!

#### `process_slack_member`

This function takes a [`Profile`](#profile), [analyzes](#analyze_profile) it using the agent, adds it to the [`AnalysisStore`](#analysis-store), and [sends](#send-slack-message) the analysis into the `#new-slack-leads` channel.

[slack_lead_qualifier/functions.py (L4-L45)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/functions.py#L4-L45)

```python
...

from .agent import analyze_profile
from .models import Profile

from .slack import send_slack_message
from .store import AnalysisStore

...

NEW_LEAD_CHANNEL = '#new-slack-leads'

...

@logfire.instrument('Process Slack member')
async def process_slack_member(profile: Profile):
    analysis = await analyze_profile(profile)
    logfire.info('Analysis', analysis=analysis)

    if analysis is None:
        return

    await AnalysisStore().add(analysis)

    await send_slack_message(
        NEW_LEAD_CHANNEL,
        [
            {
                'type': 'header',
                'text': {
                    'type': 'plain_text',
                    'text': f'New Slack member with score {analysis.relevance}/5',
                },
            },
            {
                'type': 'divider',
            },
            *analysis.as_slack_blocks(),
        ],
    )

...

```

#### `send_daily_summary`

This function list all of the analyses in the [`AnalysisStore`](#analysis-store), takes the top 5 by relevance, [sends](#send-slack-message) them into the `#daily-slack-leads-summary` channel, and clears the `AnalysisStore` so that the next daily run won't process these analyses again.

[slack_lead_qualifier/functions.py (L8-L85)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/functions.py#L8-L85)

```python
...

from .slack import send_slack_message
from .store import AnalysisStore

...

DAILY_SUMMARY_CHANNEL = '#daily-slack-leads-summary'

...

@logfire.instrument('Send daily summary')
async def send_daily_summary():
    analyses = await AnalysisStore().list()
    logfire.info('Analyses', analyses=analyses)

    if len(analyses) == 0:
        return

    sorted_analyses = sorted(analyses, key=lambda x: x.relevance, reverse=True)
    top_analyses = sorted_analyses[:5]

    blocks = [
        {
            'type': 'header',
            'text': {
                'type': 'plain_text',
                'text': f'Top {len(top_analyses)} new Slack members from the last 24 hours',
            },
        },
    ]

    for analysis in top_analyses:
        blocks.extend(
            [
                {
                    'type': 'divider',
                },
                *analysis.as_slack_blocks(include_relevance=True),
            ]
        )

    await send_slack_message(
        DAILY_SUMMARY_CHANNEL,
        blocks,
    )

    await AnalysisStore().clear()

```

### Web app

As it stands, neither of these functions are actually being called from anywhere.

Let's implement a [FastAPI](https://fastapi.tiangolo.com/) endpoint to handle the `team_join` Slack webhook (also known as the [Slack Events API](https://docs.slack.dev/apis/events-api)) and call the [`process_slack_member`](#process_slack_member) function we just defined. We also instrument FastAPI using Logfire for good measure.

[slack_lead_qualifier/app.py (L20-L36)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/app.py#L20-L36)

```python
...

app = FastAPI()
logfire.instrument_fastapi(app, capture_headers=True)


@app.post('/')
async def process_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    if payload['type'] == 'url_verification':
        return {'challenge': payload['challenge']}
    elif (
        payload['type'] == 'event_callback' and payload['event']['type'] == 'team_join'
    ):
        profile = Profile.model_validate(payload['event']['user']['profile'])

        process_slack_member(profile)
        return {'status': 'OK'}

    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

```

#### `process_slack_member` with Modal

I was a little sneaky there -- we're not actually calling the [`process_slack_member`](#process_slack_member) function we defined in `functions.py` directly, as Slack requires webhooks to respond within 3 seconds, and we need a bit more time than that to talk to the LLM, do some web searches, and send the Slack message.

Instead, we're calling the following function defined alongside the app, which uses Modal's [`modal.Function.spawn`](https://modal.com/docs/reference/modal.Function#spawn) feature to run a function in the background. (If you're curious what the Modal side of this function looks like, you can [jump ahead](#backgrounded-process_slack_member).)

Because `modal.py` (which we'll see in the next section) imports `app.py`, we import from `modal.py` inside the function definition because doing so at the top level would have resulted in a circular import error.

We also pass along the current Logfire context to get [Distributed Tracing](https://logfire.pydantic.dev/docs/how-to-guides/distributed-tracing/), meaning that the background function execution will show up nested under the webhook request trace, so that we have everything related to that request in one place.

[slack_lead_qualifier/app.py (L11-L16)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/app.py#L11-L16)

```python
...

def process_slack_member(profile: Profile):
    from .modal import process_slack_member as _process_slack_member

    _process_slack_member.spawn(
        profile.model_dump(), logfire_ctx=get_context()
    )

...

```

### Modal app

Now let's see how easy Modal makes it to deploy all of this.

#### Set up Modal

The first thing we do is define the Modal app, by specifying the base image to use (Debian with Python 3.13), all the Python packages it needs, and all of the secrets defined in the Modal interface that need to be made available during runtime.

[slack_lead_qualifier/modal.py (L4-L21)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py#L4-L21)

```python
...

import modal

image = modal.Image.debian_slim(python_version='3.13').pip_install(
    'pydantic',
    'pydantic_ai_slim[openai,duckduckgo]',
    'logfire[httpx,fastapi]',
    'fastapi[standard]',
    'httpx',
)
app = modal.App(
    name='slack-lead-qualifier',
    image=image,
    secrets=[
        modal.Secret.from_name('logfire'),
        modal.Secret.from_name('openai'),
        modal.Secret.from_name('slack'),
    ],
)

...

```

#### Set up Logfire

Next, we define a function to set up Logfire instrumentation for Pydantic AI and HTTPX.

We cannot do this at the top level of the file, as the requested packages (like `logfire`) will only be available within functions running on Modal (like the ones we'll define next). This file, `modal.py`, runs on your local machine and only has access to the `modal` package.

[slack_lead_qualifier/modal.py (L25-L30)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py#L25-L30)

```python
...

def setup_logfire():
    import logfire

    logfire.configure(service_name=app.name)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)

...

```

#### Web app

To deploy a [web endpoint](https://modal.com/docs/guide/webhooks) on Modal, we simply define a function that returns an ASGI app (like FastAPI) and decorate it with `@app.function()` and `@modal.asgi_app()`.

This `web_app` function will be run on Modal, so inside the function we can call the `setup_logfire` function that requires the `logfire` package, and import `app.py` which uses the other requested packages.

By default, Modal spins up a container to handle a function call (like a web request) on-demand, meaning there's a little bit of startup time to each request. However, Slack requires webhooks to respond within 3 seconds, so we specify `min_containers=1` to keep the web endpoint running and ready to answer requests at all times. This is a bit annoying and wasteful, but fortunately [Modal's pricing](https://modal.com/pricing) is pretty reasonable, you get $30 free monthly compute, and they offer up to $50k in free credits for startup and academic researchers.

[slack_lead_qualifier/modal.py (L34-L41)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py#L34-L41)

```python
...

@app.function(min_containers=1)
@modal.asgi_app()  # type: ignore
def web_app():
    setup_logfire()

    from .app import app as _app

    return _app

...

```

Note

Note that `# type: ignore` on the `@modal.asgi_app()` line -- unfortunately `modal` does not fully define its types, so we need this to stop our static type checker `pyright`, which we run over all Pydantic AI code including examples, from complaining.

#### Scheduled `send_daily_summary`

To define a [scheduled function](https://modal.com/docs/guide/cron), we can use the `@app.function()` decorator with a `schedule` argument. This Modal function will call our imported [`send_daily_summary`](#send_daily_summary) function every day at 8 am UTC.

[slack_lead_qualifier/modal.py (L60-L66)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py#L60-L66)

```python
...

@app.function(schedule=modal.Cron('0 8 * * *'))  # Every day at 8am UTC
async def send_daily_summary():
    setup_logfire()

    from .functions import send_daily_summary as _send_daily_summary

    await _send_daily_summary()

```

#### Backgrounded `process_slack_member`

Finally, we define a Modal function that wraps our [`process_slack_member`](#process_slack_member) function, so that it can run in the background.

As you'll remember from when we [spawned this function from the web app](#process_slack_member-with-modal), we passed along the Logfire context to get [Distributed Tracing](https://logfire.pydantic.dev/docs/how-to-guides/distributed-tracing/), so we need to attach it here.

[slack_lead_qualifier/modal.py (L45-L56)](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py#L45-L56)

```python
...

@app.function()
async def process_slack_member(profile_raw: dict[str, Any], logfire_ctx: Any):
    setup_logfire()

    from logfire.propagate import attach_context

    from .functions import process_slack_member as _process_slack_member
    from .models import Profile

    with attach_context(logfire_ctx):
        profile = Profile.model_validate(profile_raw)
        await _process_slack_member(profile)

...

```

## Conclusion

And that's it! Now, assuming you've met the [prerequisites](#prerequisites), you can run or deploy the app using the commands under [usage](#usage).

# SQL Generation

Example demonstrating how to use Pydantic AI to generate SQL queries based on user input.

Demonstrates:

- [dynamic system prompt](../../agents/#system-prompts)
- [structured `output_type`](../../output/#structured-output)
- [output validation](../../output/#output-validator-functions)
- [agent dependencies](../../dependencies/)

## Running the Example

The resulting SQL is validated by running it as an `EXPLAIN` query on PostgreSQL. To run the example, you first need to run PostgreSQL, e.g. via Docker:

```bash
docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 postgres

```

*(we run postgres on port `54320` to avoid conflicts with any other postgres instances you may have running)*

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.sql_gen

```

```bash
uv run -m pydantic_ai_examples.sql_gen

```

or to use a custom prompt:

```bash
python -m pydantic_ai_examples.sql_gen "find me errors"

```

```bash
uv run -m pydantic_ai_examples.sql_gen "find me errors"

```

This model uses `gemini-3-pro-preview` by default since Gemini is good at single shot queries of this kind.

## Example Code

[sql_gen.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/sql_gen.py)

```python
"""Example demonstrating how to use Pydantic AI to generate SQL queries based on user input.

Run postgres with:

    mkdir postgres-data
    docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 postgres

Run with:

    uv run -m pydantic_ai_examples.sql_gen "show me logs from yesterday, with level 'error'"
"""

import asyncio
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, TypeAlias

import asyncpg
import logfire
from annotated_types import MinLen
from devtools import debug
from pydantic import BaseModel, Field

from pydantic_ai import Agent, ModelRetry, RunContext, format_as_xml

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()

DB_SCHEMA = """
CREATE TABLE records (
    created_at timestamptz,
    start_timestamp timestamptz,
    end_timestamp timestamptz,
    trace_id text,
    span_id text,
    parent_span_id text,
    level log_level,
    span_name text,
    message text,
    attributes_json_schema text,
    attributes jsonb,
    tags text[],
    is_exception boolean,
    otel_status_message text,
    service_name text
);
"""
SQL_EXAMPLES = [
    {
        'request': 'show me records where foobar is false',
        'response': "SELECT * FROM records WHERE attributes->>'foobar' = false",
    },
    {
        'request': 'show me records where attributes include the key "foobar"',
        'response': "SELECT * FROM records WHERE attributes ? 'foobar'",
    },
    {
        'request': 'show me records from yesterday',
        'response': "SELECT * FROM records WHERE start_timestamp::date > CURRENT_TIMESTAMP - INTERVAL '1 day'",
    },
    {
        'request': 'show me error records with the tag "foobar"',
        'response': "SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)",
    },
]


@dataclass
class Deps:
    conn: asyncpg.Connection


class Success(BaseModel):
    """Response when SQL could be successfully generated."""

    sql_query: Annotated[str, MinLen(1)]
    explanation: str = Field(
        '', description='Explanation of the SQL query, as markdown'
    )


class InvalidRequest(BaseModel):
    """Response the user input didn't include enough information to generate SQL."""

    error_message: str


Response: TypeAlias = Success | InvalidRequest
agent = Agent[Deps, Response](
    'google-gla:gemini-3-pro-preview',
    # Type ignore while we wait for PEP-0747, nonetheless unions will work fine everywhere else
    output_type=Response,  # type: ignore
    deps_type=Deps,
)


@agent.system_prompt
async def system_prompt() -> str:
    return f"""\
Given the following PostgreSQL table of records, your job is to
write a SQL query that suits the user's request.

Database schema:

{DB_SCHEMA}

today's date = {date.today()}

{format_as_xml(SQL_EXAMPLES)}
"""


@agent.output_validator
async def validate_output(ctx: RunContext[Deps], output: Response) -> Response:
    if isinstance(output, InvalidRequest):
        return output

    # gemini often adds extraneous backslashes to SQL
    output.sql_query = output.sql_query.replace('\\', '')
    if not output.sql_query.upper().startswith('SELECT'):
        raise ModelRetry('Please create a SELECT query')

    try:
        await ctx.deps.conn.execute(f'EXPLAIN {output.sql_query}')
    except asyncpg.exceptions.PostgresError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return output


async def main():
    if len(sys.argv) == 1:
        prompt = 'show me logs from yesterday, with level "error"'
    else:
        prompt = sys.argv[1]

    async with database_connect(
        'postgresql://postgres:postgres@localhost:54320', 'pydantic_ai_sql_gen'
    ) as conn:
        deps = Deps(conn)
        result = await agent.run(prompt, deps=deps)
    debug(result.output)


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(server_dsn: str, database: str) -> AsyncGenerator[Any, None]:
    with logfire.span('check and create DB'):
        conn = await asyncpg.connect(server_dsn)
        try:
            db_exists = await conn.fetchval(
                'SELECT 1 FROM pg_database WHERE datname = $1', database
            )
            if not db_exists:
                await conn.execute(f'CREATE DATABASE {database}')
        finally:
            await conn.close()

    conn = await asyncpg.connect(f'{server_dsn}/{database}')
    try:
        with logfire.span('create schema'):
            async with conn.transaction():
                if not db_exists:
                    await conn.execute(
                        "CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical')"
                    )
                    await conn.execute(DB_SCHEMA)
        yield conn
    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())

```

This example shows how to stream markdown from an agent, using the [`rich`](https://github.com/Textualize/rich) library to highlight the output in the terminal.

It'll run the example with both OpenAI and Google Gemini models if the required environment variables are set.

Demonstrates:

- [streaming text responses](../../output/#streaming-text)

## Running the Example

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.stream_markdown

```

```bash
uv run -m pydantic_ai_examples.stream_markdown

```

## Example Code

[stream_markdown.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/stream_markdown.py)

```python
"""This example shows how to stream markdown from an agent, using the `rich` library to display the markdown.

Run with:

    uv run -m pydantic_ai_examples.stream_markdown
"""

import asyncio
import os

import logfire
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

agent = Agent()

# models to try, and the appropriate env var
models: list[tuple[KnownModelName, str]] = [
    ('google-gla:gemini-3-pro-preview', 'GEMINI_API_KEY'),
    ('openai:gpt-5-mini', 'OPENAI_API_KEY'),
    ('groq:llama-3.3-70b-versatile', 'GROQ_API_KEY'),
]


async def main():
    prettier_code_blocks()
    console = Console()
    prompt = 'Show me a short example of using Pydantic.'
    console.log(f'Asking: {prompt}...', style='cyan')
    for model, env_var in models:
        if env_var in os.environ:
            console.log(f'Using model: {model}')
            with Live('', console=console, vertical_overflow='visible') as live:
                async with agent.run_stream(prompt, model=model) as result:
                    async for message in result.stream_output():
                        live.update(Markdown(message))
            console.log(result.usage())
        else:
            console.log(f'{model} requires {env_var} to be set.')


def prettier_code_blocks():
    """Make rich code blocks prettier and easier to copy.

    From https://github.com/samuelcolvin/aicli/blob/v0.8.0/samuelcolvin_aicli.py#L22
    """

    class SimpleCodeBlock(CodeBlock):
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            code = str(self.text).rstrip()
            yield Text(self.lexer_name, style='dim')
            yield Syntax(
                code,
                self.lexer_name,
                theme=self.theme,
                background_color='default',
                word_wrap=True,
            )
            yield Text(f'/{self.lexer_name}', style='dim')

    Markdown.elements['fence'] = SimpleCodeBlock


if __name__ == '__main__':
    asyncio.run(main())

```

Information about whales â€” an example of streamed structured response validation.

Demonstrates:

- [streaming structured output](../../output/#streaming-structured-output)

This script streams structured responses from GPT-4 about whales, validates the data and displays it as a dynamic table using [`rich`](https://github.com/Textualize/rich) as the data is received.

## Running the Example

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.stream_whales

```

```bash
uv run -m pydantic_ai_examples.stream_whales

```

Should give an output like this:

## Example Code

[Learn about Gateway](../../gateway) [stream_whales.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/stream_whales.py)

```python
"""Information about whales â€” an example of streamed structured response validation.

This script streams structured responses from GPT-4 about whales, validates the data
and displays it as a dynamic table using Rich as the data is received.

Run with:

    uv run -m pydantic_ai_examples.stream_whales
"""

from typing import Annotated

import logfire
from pydantic import Field
from rich.console import Console
from rich.live import Live
from rich.table import Table
from typing_extensions import NotRequired, TypedDict

from pydantic_ai import Agent

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


class Whale(TypedDict):
    name: str
    length: Annotated[
        float, Field(description='Average length of an adult whale in meters.')
    ]
    weight: NotRequired[
        Annotated[
            float,
            Field(description='Average weight of an adult whale in kilograms.', ge=50),
        ]
    ]
    ocean: NotRequired[str]
    description: NotRequired[Annotated[str, Field(description='Short Description')]]


agent = Agent('gateway/openai:gpt-4', output_type=list[Whale])


async def main():
    console = Console()
    with Live('\n' * 36, console=console) as live:
        console.print('Requesting data...', style='cyan')
        async with agent.run_stream(
            'Generate me details of 5 species of Whale.'
        ) as result:
            console.print('Response:', style='green')

            async for whales in result.stream_output(debounce_by=0.01):
                table = Table(
                    title='Species of Whale',
                    caption='Streaming Structured responses from GPT-4',
                    width=120,
                )
                table.add_column('ID', justify='right')
                table.add_column('Name')
                table.add_column('Avg. Length (m)', justify='right')
                table.add_column('Avg. Weight (kg)', justify='right')
                table.add_column('Ocean')
                table.add_column('Description', justify='right')

                for wid, whale in enumerate(whales, start=1):
                    table.add_row(
                        str(wid),
                        whale['name'],
                        f'{whale["length"]:0.0f}',
                        f'{w:0.0f}' if (w := whale.get('weight')) else 'â€¦',
                        whale.get('ocean') or 'â€¦',
                        whale.get('description') or 'â€¦',
                    )
                live.update(table)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())

```

[stream_whales.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/stream_whales.py)

```python
"""Information about whales â€” an example of streamed structured response validation.

This script streams structured responses from GPT-4 about whales, validates the data
and displays it as a dynamic table using Rich as the data is received.

Run with:

    uv run -m pydantic_ai_examples.stream_whales
"""

from typing import Annotated

import logfire
from pydantic import Field
from rich.console import Console
from rich.live import Live
from rich.table import Table
from typing_extensions import NotRequired, TypedDict

from pydantic_ai import Agent

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


class Whale(TypedDict):
    name: str
    length: Annotated[
        float, Field(description='Average length of an adult whale in meters.')
    ]
    weight: NotRequired[
        Annotated[
            float,
            Field(description='Average weight of an adult whale in kilograms.', ge=50),
        ]
    ]
    ocean: NotRequired[str]
    description: NotRequired[Annotated[str, Field(description='Short Description')]]


agent = Agent('openai:gpt-4', output_type=list[Whale])


async def main():
    console = Console()
    with Live('\n' * 36, console=console) as live:
        console.print('Requesting data...', style='cyan')
        async with agent.run_stream(
            'Generate me details of 5 species of Whale.'
        ) as result:
            console.print('Response:', style='green')

            async for whales in result.stream_output(debounce_by=0.01):
                table = Table(
                    title='Species of Whale',
                    caption='Streaming Structured responses from GPT-4',
                    width=120,
                )
                table.add_column('ID', justify='right')
                table.add_column('Name')
                table.add_column('Avg. Length (m)', justify='right')
                table.add_column('Avg. Weight (kg)', justify='right')
                table.add_column('Ocean')
                table.add_column('Description', justify='right')

                for wid, whale in enumerate(whales, start=1):
                    table.add_row(
                        str(wid),
                        whale['name'],
                        f'{whale["length"]:0.0f}',
                        f'{w:0.0f}' if (w := whale.get('weight')) else 'â€¦',
                        whale.get('ocean') or 'â€¦',
                        whale.get('description') or 'â€¦',
                    )
                live.update(table)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())

```

Example of Pydantic AI with multiple tools which the LLM needs to call in turn to answer a question.

Demonstrates:

- [tools](../../tools/)
- [agent dependencies](../../dependencies/)
- [streaming text responses](../../output/#streaming-text)
- Building a [Gradio](https://www.gradio.app/) UI for the agent

In this case the idea is a "weather" agent â€” the user can ask for the weather in multiple locations, the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use the `get_weather` tool to get the weather for those locations.

## Running the Example

To run this example properly, you might want to add two extra API keys **(Note if either key is missing, the code will fall back to dummy data, so they're not required)**:

- A weather API key from [tomorrow.io](https://www.tomorrow.io/weather-api/) set via `WEATHER_API_KEY`
- A geocoding API key from [geocode.maps.co](https://geocode.maps.co/) set via `GEO_API_KEY`

With [dependencies installed and environment variables set](../setup/#usage), run:

```bash
python -m pydantic_ai_examples.weather_agent

```

```bash
uv run -m pydantic_ai_examples.weather_agent

```

## Example Code

[Learn about Gateway](../../gateway) [weather_agent.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/weather_agent.py)

```python
"""Example of Pydantic AI with multiple tools which the LLM needs to call in turn to answer a question.

In this case the idea is a "weather" agent â€” the user can ask for the weather in multiple cities,
the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use
the `get_weather` tool to get the weather.

Run with:

    uv run -m pydantic_ai_examples.weather_agent
"""

from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import logfire
from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    client: AsyncClient


weather_agent = Agent(
    'gateway/openai:gpt-5-mini',
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    instructions='Be concise, reply with one sentence.',
    deps_type=Deps,
    retries=2,
)


class LatLng(BaseModel):
    lat: float
    lng: float


@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    # NOTE: the response here will be random, and is not related to the location description.
    r = await ctx.deps.client.get(
        'https://demo-endpoints.pydantic.workers.dev/latlng',
        params={'location': location_description},
    )
    r.raise_for_status()
    return LatLng.model_validate_json(r.content)


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    # NOTE: the responses here will be random, and are not related to the lat and lng.
    temp_response, descr_response = await asyncio.gather(
        ctx.deps.client.get(
            'https://demo-endpoints.pydantic.workers.dev/number',
            params={'min': 10, 'max': 30},
        ),
        ctx.deps.client.get(
            'https://demo-endpoints.pydantic.workers.dev/weather',
            params={'lat': lat, 'lng': lng},
        ),
    )
    temp_response.raise_for_status()
    descr_response.raise_for_status()
    return {
        'temperature': f'{temp_response.text} Â°C',
        'description': descr_response.text,
    }


async def main():
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        deps = Deps(client=client)
        result = await weather_agent.run(
            'What is the weather like in London and in Wiltshire?', deps=deps
        )
        print('Response:', result.output)


if __name__ == '__main__':
    asyncio.run(main())

```

[weather_agent.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/weather_agent.py)

```python
"""Example of Pydantic AI with multiple tools which the LLM needs to call in turn to answer a question.

In this case the idea is a "weather" agent â€” the user can ask for the weather in multiple cities,
the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use
the `get_weather` tool to get the weather.

Run with:

    uv run -m pydantic_ai_examples.weather_agent
"""

from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import logfire
from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    client: AsyncClient


weather_agent = Agent(
    'openai:gpt-5-mini',
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    instructions='Be concise, reply with one sentence.',
    deps_type=Deps,
    retries=2,
)


class LatLng(BaseModel):
    lat: float
    lng: float


@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    # NOTE: the response here will be random, and is not related to the location description.
    r = await ctx.deps.client.get(
        'https://demo-endpoints.pydantic.workers.dev/latlng',
        params={'location': location_description},
    )
    r.raise_for_status()
    return LatLng.model_validate_json(r.content)


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    # NOTE: the responses here will be random, and are not related to the lat and lng.
    temp_response, descr_response = await asyncio.gather(
        ctx.deps.client.get(
            'https://demo-endpoints.pydantic.workers.dev/number',
            params={'min': 10, 'max': 30},
        ),
        ctx.deps.client.get(
            'https://demo-endpoints.pydantic.workers.dev/weather',
            params={'lat': lat, 'lng': lng},
        ),
    )
    temp_response.raise_for_status()
    descr_response.raise_for_status()
    return {
        'temperature': f'{temp_response.text} Â°C',
        'description': descr_response.text,
    }


async def main():
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        deps = Deps(client=client)
        result = await weather_agent.run(
            'What is the weather like in London and in Wiltshire?', deps=deps
        )
        print('Response:', result.output)


if __name__ == '__main__':
    asyncio.run(main())

```

## Running the UI

You can build multi-turn chat applications for your agent with [Gradio](https://www.gradio.app/), a framework for building AI web applications entirely in python. Gradio comes with built-in chat components and agent support so the entire UI will be implemented in a single python file!

Here's what the UI looks like for the weather agent:

```bash
pip install gradio>=5.9.0
python/uv-run -m pydantic_ai_examples.weather_agent_gradio

```

## UI Code

[weather_agent_gradio.py](https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/weather_agent_gradio.py)

```python
from __future__ import annotations as _annotations

import json

from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import ToolCallPart, ToolReturnPart
from pydantic_ai_examples.weather_agent import Deps, weather_agent

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        'Please install gradio with `pip install gradio`. You must use python>=3.10.'
    ) from e

TOOL_TO_DISPLAY_NAME = {'get_lat_lng': 'Geocoding API', 'get_weather': 'Weather API'}

client = AsyncClient()
deps = Deps(client=client)


async def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
    chatbot.append({'role': 'user', 'content': prompt})
    yield gr.Textbox(interactive=False, value=''), chatbot, gr.skip()
    async with weather_agent.run_stream(
        prompt, deps=deps, message_history=past_messages
    ) as result:
        for message in result.new_messages():
            for call in message.parts:
                if isinstance(call, ToolCallPart):
                    call_args = call.args_as_json_str()
                    metadata = {
                        'title': f'ðŸ› ï¸ Using {TOOL_TO_DISPLAY_NAME[call.tool_name]}',
                    }
                    if call.tool_call_id is not None:
                        metadata['id'] = call.tool_call_id

                    gr_message = {
                        'role': 'assistant',
                        'content': 'Parameters: ' + call_args,
                        'metadata': metadata,
                    }
                    chatbot.append(gr_message)
                if isinstance(call, ToolReturnPart):
                    for gr_message in chatbot:
                        if (
                            gr_message.get('metadata', {}).get('id', '')
                            == call.tool_call_id
                        ):
                            if isinstance(call.content, BaseModel):
                                json_content = call.content.model_dump_json()
                            else:
                                json_content = json.dumps(call.content)
                            gr_message['content'] += f'\nOutput: {json_content}'
                yield gr.skip(), chatbot, gr.skip()
        chatbot.append({'role': 'assistant', 'content': ''})
        async for message in result.stream_text():
            chatbot[-1]['content'] = message
            yield gr.skip(), chatbot, gr.skip()
        past_messages = result.all_messages()

        yield gr.Textbox(interactive=True), gr.skip(), past_messages


async def handle_retry(chatbot, past_messages: list, retry_data: gr.RetryData):
    new_history = chatbot[: retry_data.index]
    previous_prompt = chatbot[retry_data.index]['content']
    past_messages = past_messages[: retry_data.index]
    async for update in stream_from_agent(previous_prompt, new_history, past_messages):
        yield update


def undo(chatbot, past_messages: list, undo_data: gr.UndoData):
    new_history = chatbot[: undo_data.index]
    past_messages = past_messages[: undo_data.index]
    return chatbot[undo_data.index]['content'], new_history, past_messages


def select_data(message: gr.SelectData) -> str:
    return message.value['text']


with gr.Blocks() as demo:
    gr.HTML(
        """
<div style="display: flex; justify-content: center; align-items: center; gap: 2rem; padding: 1rem; width: 100%">
    <img src="https://ai.pydantic.dev/img/logo-white.svg" style="max-width: 200px; height: auto">
    <div>
        <h1 style="margin: 0 0 1rem 0">Weather Assistant</h1>
        <h3 style="margin: 0 0 0.5rem 0">
            This assistant answer your weather questions.
        </h3>
    </div>
</div>
"""
    )
    past_messages = gr.State([])
    chatbot = gr.Chatbot(
        label='Packing Assistant',
        type='messages',
        avatar_images=(None, 'https://ai.pydantic.dev/img/logo-white.svg'),
        examples=[
            {'text': 'What is the weather like in Miami?'},
            {'text': 'What is the weather like in London?'},
        ],
    )
    with gr.Row():
        prompt = gr.Textbox(
            lines=1,
            show_label=False,
            placeholder='What is the weather like in New York City?',
        )
    generation = prompt.submit(
        stream_from_agent,
        inputs=[prompt, chatbot, past_messages],
        outputs=[prompt, chatbot, past_messages],
    )
    chatbot.example_select(select_data, None, [prompt])
    chatbot.retry(
        handle_retry, [chatbot, past_messages], [prompt, chatbot, past_messages]
    )
    chatbot.undo(undo, [chatbot, past_messages], [prompt, chatbot, past_messages])


if __name__ == '__main__':
    demo.launch()

```
