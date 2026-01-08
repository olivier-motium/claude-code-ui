# Distributed Systems

## Contents

- [Detect Service is Down](#detect-service-is-down)
- [Trace across Multiple Services](#trace-across-multiple-services)

---

## Detect Service is Down

For now, **Logfire** doesn't have a built-in way to detect if a service is down, in the sense that we don't ping
services via HTTP or any other protocol to check if they are up or down.

For now we don't have it, but...

If you would like to see this feature in **Logfire**, [let us know](https://logfire.pydantic.dev/docs/help/)!

It's useful for us to understand the use cases and requirements for this feature.

However, you can create alerts to notify you when a log message is not received for a certain amount of time.
This can be used to detect if a service is down.

Let's say you have a [FastAPI application](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/) that has a health check endpoint at `/health`.

```
import logfire
from fastapi import FastAPI

logfire.configure(service_name="backend")
app = FastAPI()
logfire.instrument_fastapi(app)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

You probably have this endpoint because you have a mechanism that restarts the service if it's down.
In this case, you can use **Logfire** to send you an alert if the health check endpoint is not called
for a certain amount of time.

## Create the Alert

Go to [your alerts tab](https://logfire.pydantic.dev/-/redirect/latest-project/alerts/) and click on "New Alert".
Then add the following query to the alert:

```
SELECT
    CASE
        WHEN COUNT(*) = 0 THEN 'backend is down'
        ELSE 'backend is up'
    END AS message
FROM
    records
WHERE
    service_name = 'backend' and span_name = 'GET /health';
```

This query will return `backend is down` if the `/health` endpoint on the `'backend'` service is not called.

On the "Alert Parameters", we want to be notified as soon as possible, so we should execute the query `"every minute"`,
include rows from `"the last minute"`, and notify us if `"the query's results change"`.

Then you need to set up a channel to send this notification, which can be a Slack channel or a webhook.
See more about it on the [alerts documentation](https://logfire.pydantic.dev/docs/guides/web-ui/alerts/).

---

## Trace across Multiple Services

**Logfire** builds on OpenTelemetry, which keeps track of _context_ to determine the parent trace/span of a new span/log and whether it should be included by sampling. _Context propagation_ is when this context is serialized and sent to another process, so that tracing can be distributed across services while allowing the full tree of spans to be cleanly reconstructed and viewed together.

## Manual Context Propagation

**Logfire** provides a thin wrapper around the OpenTelemetry context propagation API to make manual distributed tracing easier. You shouldn't usually need to do this yourself, but it demonstrates the concept nicely. Here's an example:

```
import logfire

logfire.configure()

with logfire.span('parent'):
    ctx = logfire.get_context()

print(ctx)

# Attach the context in another execution environment
with logfire.attach_context(ctx):
    logfire.info('child')  # This log will be a child of the parent span.
```

`ctx` will look something like this:

```
{'traceparent': '00-d1b9e555b056907ee20b0daebf62282c-7dcd821387246e1c-01'}
```

This contains 4 fields:

- `00` is a version number which you can ignore.
- `d1b9e555b056907ee20b0daebf62282c` is the `trace_id`.
- `7dcd821387246e1c` is the `span_id` of the parent span, i.e. the `parent_span_id` of the child log.
- `01` is the `trace_flags` field and indicates that the trace should be included by sampling.

See the [API reference](https://logfire.pydantic.dev/docs/reference/api/propagate/) for more details about these functions.

## Integrations

OpenTelemetry instrumentation libraries (which **Logfire** uses for its integrations) handle context propagation automatically, even across different programming languages. For example:

- Instrumented HTTP clients such as [`requests`](https://logfire.pydantic.dev/docs/integrations/http-clients/requests/) and [`httpx`](https://logfire.pydantic.dev/docs/integrations/http-clients/httpx/) will automatically set the `traceparent` header when making requests.
- Instrumented web servers such as [`flask`](https://logfire.pydantic.dev/docs/integrations/web-frameworks/flask/) and [`fastapi`](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/) will automatically extract the `traceparent` header and use it to set the context for server spans.
- The [`celery` integration](https://logfire.pydantic.dev/docs/integrations/event-streams/celery/) will automatically propagate the context to child tasks.

## Thread and Pool executors

**Logfire** automatically patches [`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor) and [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor) to propagate context to child threads and processes. This means that logs and spans created in child threads and processes will be correctly associated with the parent span. Here's an example to demonstrate:

```
import logfire
from concurrent.futures import ThreadPoolExecutor

logfire.configure()

@logfire.instrument("Doubling {x}")
def double(x: int):
    return x * 2

with logfire.span("Doubling everything") as span:
    executor = ThreadPoolExecutor()
    results = list(executor.map(double, range(3)))
    span.set_attribute("results", results)
```

## Unintentional Distributed Tracing

Because instrumented web servers automatically extract the `traceparent` header by default, your spans can accidentally pick up the wrong context from an externally instrumented client, or from your cloud provider such as Google Cloud Run. This can lead to:

- Spans missing their parent.
- Spans being mysteriously grouped together.
- Spans missing entirely because the original trace was excluded by sampling.

By default, **Logfire** warns you when trace context is extracted, e.g. when server instrumentation finds a `traceparent` header. You can deal with this by setting the [`distributed_tracing` argument of `logfire.configure()`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.configure(distributed_tracing)) or by setting the `LOGFIRE_DISTRIBUTED_TRACING` environment variable:

- Setting to `False` will prevent trace context from being extracted. This is recommended for web services exposed to the public internet. You can still attach/inject context to propagate to other services and create distributed traces with the web service as the root.
- Setting to `True` implies that the context propagation is intentional and will silence the warning.

The `distributed_tracing` configuration (including the warning by default) only applies when the raw OpenTelemetry API is used to extract context, as this is typically done by third-party libraries. By default, [`logfire.attach_context`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.attach_context) assumes that context propagation is intended by the application. If you are writing a library, use `attach_context(context, third_party=True)` to respect the `distributed_tracing` configuration.

---