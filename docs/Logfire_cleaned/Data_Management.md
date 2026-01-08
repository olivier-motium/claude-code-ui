# Data Management

## Contents

- [Sampling](#sampling)
- [Scrubbing sensitive data](#scrubbing-sensitive-data)
- [Suppress Spans and Metrics](#suppress-spans-and-metrics)

---

## Sampling

Sampling is the practice of discarding some traces or spans in order to reduce the amount of data that needs to be
stored and analyzed. Sampling is a trade-off between cost and completeness of data.

_Head sampling_ means the decision to sample is made at the beginning of a trace. This is simpler and more common.

_Tail sampling_ means the decision to sample is delayed, possibly until the end of a trace. This means there is more
information available to make the decision, but this adds complexity.

Sampling usually happens at the trace level, meaning entire traces are kept or discarded. This way the remaining traces
are generally complete.

## Random head sampling

Here's an example of randomly sampling 50% of traces:

```
import logfire

logfire.configure(sampling=logfire.SamplingOptions(head=0.5))

for x in range(10):
    with logfire.span(f'span {x}'):
        logfire.info(f'log {x}')
```

This outputs something like:

```
11:09:29.041 span 0
11:09:29.041   log 0
11:09:29.041 span 1
11:09:29.042   log 1
11:09:29.042 span 4
11:09:29.042   log 4
11:09:29.042 span 5
11:09:29.042   log 5
11:09:29.042 span 7
11:09:29.042   log 7
```

Note that 5 out of 10 traces are kept, and that the child log is kept if and only if the parent span is kept.

## Tail sampling by level and duration

Random head sampling often works well, but you may not want to lose any traces which indicate problems. In this case,
you can use tail sampling. Here's a simple example:

```
import time

import logfire

logfire.configure(sampling=logfire.SamplingOptions.level_or_duration())

for x in range(3):
    # None of these are logged
    with logfire.span('excluded span'):
        logfire.info(f'info {x}')

    # All of these are logged
    with logfire.span('included span'):
        logfire.error(f'error {x}')

for t in range(1, 10, 2):
    with logfire.span(f'span with duration {t}'):
        time.sleep(t)
```

This outputs something like:

```
11:37:45.484 included span
11:37:45.484   error 0
11:37:45.485 included span
11:37:45.485   error 1
11:37:45.485 included span
11:37:45.485   error 2
11:37:49.493 span with duration 5
11:37:54.499 span with duration 7
11:38:01.505 span with duration 9
```

[`logfire.SamplingOptions.level_or_duration()`](https://logfire.pydantic.dev/docs/reference/api/sampling/#logfire.sampling.SamplingOptions.level_or_duration) creates an instance
of [`logfire.SamplingOptions`](https://logfire.pydantic.dev/docs/reference/api/sampling/#logfire.sampling.SamplingOptions) with simple tail sampling. With no arguments,
it means that a trace will be included if and only if it has at least one span/log that:

1. has a log level greater than `info` (the default of any span), or
2. has a duration greater than 5 seconds.

This way you won't lose information about warnings/errors or long-running operations. You can customize what to keep
with the `level_threshold` and `duration_threshold` arguments.

## Combining head and tail sampling

You can combine head and tail sampling. For example:

```
import logfire

logfire.configure(sampling=logfire.SamplingOptions.level_or_duration(head=0.1))
```

This will only keep 10% of traces, even if they have a high log level or duration. Traces that don't meet the tail
sampling criteria will be discarded every time.

## Keeping a fraction of all traces

To keep some traces even if they don't meet the tail sampling criteria, you can use the `background_rate` argument. For
example, this script:

```
import logfire

logfire.configure(sampling=logfire.SamplingOptions.level_or_duration(background_rate=0.3))

for x in range(10):
    logfire.info(f'info {x}')
for x in range(5):
    logfire.error(f'error {x}')
```

will output something like:

```
12:24:40.293 info 2
12:24:40.293 info 3
12:24:40.293 info 7
12:24:40.294 error 0
12:24:40.294 error 1
12:24:40.294 error 2
12:24:40.294 error 3
12:24:40.295 error 4
```

i.e. about 30% of the info logs and 100% of the error logs are kept.

(Technical note: the trace ID is compared against the head and background rates to determine inclusion, so the
probabilities don't depend on the number of spans in the trace, and the rates give the probabilities directly without
needing any further calculations. For example, with a head sample rate of `0.6` and a background rate of `0.3`, the
chance of a non-notable trace being included is `0.3`, not `0.6 * 0.3`.)

## Caveats of tail sampling

### Memory usage

For tail sampling to work, all the spans in a trace must be kept in memory until either the trace is included by
sampling or the trace is completed and discarded. In the above example, the spans named `included span` don't have a
high enough level to be included, so they are kept in memory until the error logs cause the entire trace to be included.
This means that traces with a large number of spans can consume a lot of memory, whereas without tail sampling the spans
would be regularly exported and freed from memory without waiting for the rest of the trace.

In practice this is usually OK, because such large traces will usually exceed the duration threshold, at which point the
trace will be included and the spans will be exported and freed. This works because the duration is measured as the time
between the start of the trace and the start/end of the most recent span, so the tail sampler can know that a span will
exceed the duration threshold even before it's complete. For example, running this script:

```
import time

import logfire

logfire.configure(sampling=logfire.SamplingOptions.level_or_duration())

with logfire.span('span'):
    for x in range(1, 10):
        time.sleep(1)
        logfire.info(f'info {x}')
```

will do nothing for the first 5 seconds, before suddenly logging all this at once:

```
12:29:43.063 span
12:29:44.065   info 1
12:29:45.066   info 2
12:29:46.072   info 3
12:29:47.076   info 4
12:29:48.082   info 5
```

followed by additional logs once per second. This is despite the fact that at this stage the outer span hasn't completed
yet and the inner logs each have 0 duration.

However, memory usage can still be a problem in any of the following cases:

- The duration threshold is set to a high value
- Spans are produced extremely rapidly
- Spans contain large attributes

### Distributed tracing

Logfire's tail sampling is implemented in the SDK and only works for traces within one process. If you need tail
sampling with distributed tracing, consider deploying
the [Tail Sampling Processor in the OpenTelemetry Collector](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/processor/tailsamplingprocessor/README.md).

If a trace was started on another process and its context was propagated to the process using the Logfire SDK tail
sampling, the whole trace will be included.

If you start a trace with the Logfire SDK with tail sampling, and then propagate the context to another process, the
spans generated by the SDK may be discarded, while the spans generated by the other process may be included, leading to
an incomplete trace.

### Spans starting after root ended, e.g. background tasks

When the root span of a trace ends, if the trace doesn't meet the tail sampling criteria, all spans in the trace are
discarded. If you start a new span in that trace (i.e. as a descendant of the root span) after the root span has ended,
the new span will always be included anyway, and its parent will be missing. This is because the tail sampling mechanism
only keeps track of active traces to save memory. This is similar to the distributed tracing case above.

Here's an example with a FastAPI background task which starts after the root span corresponding to the request has
ended:

```
import uvicorn
from fastapi import BackgroundTasks, FastAPI

import logfire

app = FastAPI()

logfire.configure(
    sampling=logfire.SamplingOptions.level_or_duration(
        duration_threshold=0.1,
    ),
)
logfire.instrument_fastapi(app)

async def background_task():
    # This will be included even if the root span was excluded.
    logfire.info('background')

@app.get('/')
async def index(background_tasks: BackgroundTasks):
    # Uncomment to prevent request span from being sampled out.
    # await asyncio.sleep(0.2)

    background_tasks.add_task(background_task)
    return {}

uvicorn.run(app)
```

A workaround is to explicitly put the new spans in their own trace using [`attach_context`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.attach_context):

```
import logfire

async def background_task():
   # `attach_context({})` forgets existing context
   # so that spans within start a new trace.
   with logfire.attach_context({}):
      with logfire.span('new trace'):
         await asyncio.sleep(0.2)
         logfire.info('background')
```

## Custom head sampling

If you need more control than random sampling, you can pass an [OpenTelemetry\\
`Sampler`](https://opentelemetry-python.readthedocs.io/en/latest/sdk/trace.sampling.html). For example:

```
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    ParentBased,
    Sampler,
    TraceIdRatioBased,
)

import logfire

class MySampler(Sampler):
    def should_sample(
            self,
            parent_context,
            trace_id,
            name,
            *args,
            **kwargs,
    ):
        if name == 'exclude me':
            sampler = ALWAYS_OFF
        elif name == 'include me minimally':
            sampler = TraceIdRatioBased(0.01)  # 1% sampling
        elif name == 'include me partially':
            sampler = TraceIdRatioBased(0.5)   # 50% sampling
        else:
            sampler = ALWAYS_ON
        return sampler.should_sample(
            parent_context,
            trace_id,
            name,
            *args,
            **kwargs,
        )

    def get_description(self):
        return 'MySampler'

logfire.configure(
    sampling=logfire.SamplingOptions(
        head=ParentBased(
            MySampler(),
        )
    )
)

with logfire.span('keep me'):
    logfire.info('kept child')

for i in range(5):
    with logfire.span('include me partially'):
        logfire.info(f'partial sample {i}')

for i in range(270):
    with logfire.span('include me minimally'):
        logfire.info(f'minimal sample {i}')

with logfire.span('exclude me'):
    logfire.info('excluded child')
```

This will output something like:

```
10:37:30.897 keep me
10:37:30.898   kept child
10:37:30.899 include me partially
10:37:30.900   partial sample 0
10:37:30.901 include me partially
10:37:30.902   partial sample 3
10:37:30.905 include me minimally
10:37:30.906   minimal sample 47
10:37:30.910 include me minimally
10:37:30.911   minimal sample 183
```

The sampler applies different strategies based on span names:

- `exclude me`: Never sampled (0% using `ALWAYS_OFF`)
- `include me partially`: 50% sampling (roughly half appear)
- `include me minimally`: 1% sampling (roughly 1 in a 100 appears)
- `keep me` and all others: Always sampled (100% using `ALWAYS_ON`)

The sampler is wrapped in a `ParentBased` sampler, which ensures child spans follow their parent's sampling decision.
If you remove that and simply pass `head=MySampler()`, child spans might be included even when their parents are
excluded, resulting in incomplete traces.

You can also pass a `Sampler` to the `head` argument of `SamplingOptions.level_or_duration` to combine tail sampling
with custom head sampling.

## Custom tail sampling

If you want tail sampling with more control than `level_or_duration`, you can pass a function to [`tail`](https://logfire.pydantic.dev/docs/reference/api/sampling/#logfire.sampling.SamplingOptions.tail) which will accept an instance of [`TailSamplingSpanInfo`](https://logfire.pydantic.dev/docs/reference/api/sampling/#logfire.sampling.TailSamplingSpanInfo) and return a float between 0 and 1 representing the
probability that the trace should be included. For example:

```
import logfire

def get_tail_sample_rate(span_info):
    if span_info.duration >= 1:
        return 0.5

    if span_info.level > 'warn':
        return 0.3

    return 0.1

logfire.configure(
    sampling=logfire.SamplingOptions(
        head=0.5,
        tail=get_tail_sample_rate,
    ),
)
```

---

## Scrubbing sensitive data

The **Logfire** SDK scans for and redacts potentially sensitive data from logs and spans before exporting them.

## Disabling scrubbing

To disable scrubbing entirely, set [`scrubbing`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.configure(scrubbing)) to `False`:

```
import logfire

logfire.configure(scrubbing=False)
```

## Scrubbing more with custom patterns

By default, the SDK looks for some sensitive regular expressions. To add your own patterns, set [`extra_patterns`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.ScrubbingOptions.extra_patterns) to a list of regex strings:

```
import logfire

logfire.configure(scrubbing=logfire.ScrubbingOptions(extra_patterns=['my_pattern']))

logfire.info('Hello', data={
    'key_matching_my_pattern': 'This string will be redacted because its key matches',
    'other_key': 'This string will also be redacted because it matches MY_PATTERN case-insensitively',
    'password': 'This will be redacted because custom patterns are combined with the default patterns',
})
```

Here are the default scrubbing patterns:

```
['password', 'passwd', 'mysql_pwd', 'secret', 'auth(?!ors?\\b)', 'credential', 'private[._ -]?key', 'api[._ -]?key',\
 'session', 'cookie', 'social[._ -]?security', 'credit[._ -]?card', '(?:\\b|_)csrf(?:\\b|_)', '(?:\\b|_)xsrf(?:\\b|_)',\
 '(?:\\b|_)jwt(?:\\b|_)', '(?:\\b|_)ssn(?:\\b|_)']
```

## Scrubbing less with a callback

On the other hand, if the scrubbing is too aggressive, you can pass a function to [`callback`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.ScrubbingOptions.callback) to prevent certain data from being redacted.

The function will be called for each potential match found by the scrubber. If it returns `None`, the value is redacted. Otherwise, the returned value replaces the matched value. The function accepts a single argument of type [`logfire.ScrubMatch`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.ScrubMatch).

Here's an example:

```
import logfire

def scrubbing_callback(match: logfire.ScrubMatch):
    # `my_safe_value` often contains the string 'password' but it's not actually sensitive.
    if match.path == ('attributes', 'my_safe_value') and match.pattern_match.group(0) == 'password':
        # Return the original value to prevent redaction.
        return match.value

logfire.configure(scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback))
```

## Security tips

### Use message templates

The full span/log message is not scrubbed, only the fields within. For example, this:

```
logfire.info('User details: {user}', user=User(id=123, password='secret'))
```

...may log something like:

```
User details: [Scrubbed due to 'password']
```

...but this:

```
user = User(id=123, password='secret')
logfire.info('User details: ' + str(user))
```

will log:

```
User details: User(id=123, password='secret')
```

This is necessary so that safe messages such as 'Password is correct' are not redacted completely.

Using f-strings (e.g. `logfire.info(f'User details: {user}')`) _is_ safe if `inspect_arguments` is enabled (the default in Python 3.11+) and working correctly.
[See here](https://logfire.pydantic.dev/docs/guides/onboarding-checklist/add-manual-tracing/#f-strings) for more information.

In short, don't format the message yourself. This is also a good practice in general for [other reasons](https://logfire.pydantic.dev/docs/guides/onboarding-checklist/add-manual-tracing/#messages-and-span-names).

### Keep sensitive data out of URLs

The attribute `"http.url"` which is recorded by OpenTelemetry instrumentation libraries is considered safe so that URLs like `"http://example.com/users/123/authenticate"` are not redacted.

As a general rule, not just for Logfire, assume that URLs (including query parameters) will be logged, so sensitive data should be put in the request body or headers instead.

### Use parameterized database queries

The `"db.statement"` attribute which is recorded by OpenTelemetry database instrumentation libraries is considered safe so that SQL queries like `"SELECT secret_value FROM table WHERE ..."` are not redacted.

Use parameterized queries (e.g. prepared statements) so that sensitive data is not interpolated directly into the query string, even if
you use an interpolation method that's safe from SQL injection.

---

## Suppress Spans and Metrics

At **Logfire** we want to provide you with the best experience possible. We understand that sometimes you might want to
fine tune the data you're sending to **Logfire**. That's why we provide you with the ability to suppress spans and metrics.

We provide two ways to suppress the data you're sending to **Logfire**: [Suppress Scopes](https://logfire.pydantic.dev/docs/how-to-guides/suppress/#suppress-scopes) and
[Suppress Instrumentation](https://logfire.pydantic.dev/docs/how-to-guides/suppress/#suppress-instrumentation).

## Suppress Scopes

You can suppress spans and metrics from a specific OpenTelemetry scope.
This is useful when you want to suppress data from a specific package.

For example, if you have [BigQuery](https://logfire.pydantic.dev/docs/integrations/databases/bigquery/) installed, it automatically instruments itself with OpenTelemetry.
Which means that you need to opt-out of instrumentation if you don't want to send data to **Logfire** related to BigQuery.

You can do this by calling the [`suppress_scopes`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.Logfire.suppress_scopes) method.

```
import logfire

logfire.configure()
logfire.suppress_scopes("google.cloud.bigquery.opentelemetry_tracing")
```

In this case, we're suppressing the scope `google.cloud.bigquery.opentelemetry_tracing`.
All spans and metrics related to BigQuery will not be sent to **Logfire**.

## Suppress Instrumentation

Sometimes you might want to suppress spans from a specific part of your code, and not a whole package.

For example, assume you are using \[HTTPX\], but you don't want to suppress all the spans and metrics related to it.
You just want to suppress a small part of the code that you know will generate a lot of spans.

You can do this by using the [`suppress_instrumentation`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.suppress_instrumentation) context manager.

```
import httpx
import logfire

logfire.configure()

client = httpx.Client()
logfire.instrument_httpx(client)

# The span generated will be sent to Logfire.
client.get("https://httpbin.org/get")

# The span generated will NOT be sent to Logfire.
with logfire.suppress_instrumentation():
    client.get("https://httpbin.org/get")
```

In this case, the span generated inside the `with logfire.suppress_instrumentation():` block will not be sent to **Logfire**.

---