# Dashboard And Queries

## Contents

- [Link to Code Source](#link-to-code-source)
- [Export your Logfire Data](#export-your-logfire-data)
- [Writing SQL Queries for Dashboards](#writing-sql-queries-for-dashboards)

---

## Link to Code Source

We support linking to the source code on GitHub, GitLab, and any other VCS provider that uses the same URL format.

[![Link to GitHub](https://logfire.pydantic.dev/docs/images/guide/link-to-github.gif)](https://logfire.pydantic.dev/docs/images/guide/link-to-github.gif)

## Usage

Here's an example:

```
import logfire

logfire.configure(
    code_source=logfire.CodeSource(
        repository='https://github.com/pydantic/logfire',  The URL of the repository e.g. https://github.com/pydantic/logfire.

        revision='<hash of commit used on release>',  The specific branch, tag, or commit hash to link to e.g. main.

        root_path='path/within/repo',  The path from the root of the repository to the current working directory of the process. If your code is in a
   subdirectory of your repo, you can specify it here. Otherwise you can probably omit this.

    )
)
```

You can learn more in our [`logfire.CodeSource`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.CodeSource) API reference.

## Alternative Configuration

For other OpenTelemetry SDKs, you can configure these settings using resource attributes, e.g. by setting the
[`OTEL_RESOURCE_ATTRIBUTES`](https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/#general-sdk-configuration) environment variable:

```
OTEL_RESOURCE_ATTRIBUTES=vcs.repository.url.full=https://github.com/pydantic/platform
OTEL_RESOURCE_ATTRIBUTES=${OTEL_RESOURCE_ATTRIBUTES},vcs.repository.ref.revision=main
OTEL_RESOURCE_ATTRIBUTES=${OTEL_RESOURCE_ATTRIBUTES},vcs.root.path=path/within/repo
```

---

## Export your Logfire Data

**Logfire** provides a web API for programmatically running arbitrary SQL queries against the data in your **Logfire** projects.
This API can be used to retrieve data for export, analysis, or integration with other tools, allowing you to leverage
your data in a variety of ways.

The API is available at `https://logfire-api.pydantic.dev/v1/query` and requires a **read token** for authentication.
Read tokens can be generated from the Logfire web interface and provide secure access to your data.

The API can return data in various formats, including JSON, Apache Arrow, and CSV, to suit your needs.
See [here](https://logfire.pydantic.dev/docs/how-to-guides/query-api/#additional-configuration) for more details about the available response formats.

## How to Create a Read Token

If you've set up Logfire following the [getting started guide](https://logfire.pydantic.dev/docs/), you can generate read tokens either from
the Logfire web interface or via the CLI.

### Via Web Interface

To create a read token using the web interface:

1. Open the **Logfire** web interface at [logfire.pydantic.dev](https://logfire.pydantic.dev/).
2. Select your project from the **Projects** section on the left-hand side of the page.
3. Click on the ⚙️ **Settings** tab in the top right corner of the page.
4. Select the **Read tokens** tab from the left-hand menu.
5. Click on the **Create read token** button.

After creating the read token, you'll see a dialog with the token value.
**Copy this value and store it securely, it will not be shown again.**

### Via CLI

You can also create read tokens programmatically using the Logfire CLI:

```
logfire read-tokens --project <organization>/<project> create
```

This command will output the read token directly to stdout, making it convenient for use in scripts.

## Using the Read Clients

While you can [make direct HTTP requests](https://logfire.pydantic.dev/docs/how-to-guides/query-api/#making-direct-http-requests) to Logfire's querying API,
we provide Python clients to simplify the process of interacting with the API from Python.

Logfire provides both synchronous and asynchronous clients.
To use these clients, you can import them from the `query_client` namespace:

```
from logfire.query_client import AsyncLogfireQueryClient, LogfireQueryClient
```

Additional required dependencies

To use the query clients provided in `logfire.query_client`, you need to install `httpx`.

If you want to retrieve Arrow-format responses, you will also need to install `pyarrow`.

### Client Usage Examples

The `AsyncLogfireQueryClient` allows for asynchronous interaction with the Logfire API.
If blocking I/O is acceptable and you want to avoid the complexities of asynchronous programming,
you can use the plain `LogfireQueryClient`.

Here's an example of how to use these clients:

[Async](https://logfire.pydantic.dev/docs/how-to-guides/query-api/#__tabbed_1_1)[Sync](https://logfire.pydantic.dev/docs/how-to-guides/query-api/#__tabbed_1_2)

```
from io import StringIO

import polars as pl
from logfire.query_client import AsyncLogfireQueryClient

async def main():
    query = """
    SELECT start_timestamp
    FROM records
    LIMIT 1
    """

    async with AsyncLogfireQueryClient(read_token='<your_read_token>') as client:
        # Load data as JSON, in column-oriented format
        json_cols = await client.query_json(sql=query)
        print(json_cols)

        # Load data as JSON, in row-oriented format
        json_rows = await client.query_json_rows(sql=query)
        print(json_rows)

        # Retrieve data in arrow format, and load into a polars DataFrame
        # Note that JSON columns such as `attributes` will be returned as
        # JSON-serialized strings
        df_from_arrow = pl.from_arrow(await client.query_arrow(sql=query))
        print(df_from_arrow)

        # Retrieve data in CSV format, and load into a polars DataFrame
        # Note that JSON columns such as `attributes` will be returned as
        # JSON-serialized strings
        df_from_csv = pl.read_csv(StringIO(await client.query_csv(sql=query)))
        print(df_from_csv)

        # Get read token info
        read_token_info = await client.info()
        print(read_token_info)

if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
```

```
from io import StringIO

import polars as pl
from logfire.query_client import LogfireQueryClient

def main():
    query = """
    SELECT start_timestamp
    FROM records
    LIMIT 1
    """

    with LogfireQueryClient(read_token='<your_read_token>') as client:
        # Load data as JSON, in column-oriented format
        json_cols = client.query_json(sql=query)
        print(json_cols)

        # Load data as JSON, in row-oriented format
        json_rows = client.query_json_rows(sql=query)
        print(json_rows)

        # Retrieve data in arrow format, and load into a polars DataFrame
        # Note that JSON columns such as `attributes` will be returned as
        # JSON-serialized strings
        df_from_arrow = pl.from_arrow(client.query_arrow(sql=query))
        print(df_from_arrow)

        # Retrieve data in CSV format, and load into a polars DataFrame
        # Note that JSON columns such as `attributes` will be returned as
        # JSON-serialized strings
        df_from_csv = pl.read_csv(StringIO(client.query_csv(sql=query)))
        print(df_from_csv)

        # Get read token info
        read_token_info = client.info()
        print(read_token_info)

if __name__ == '__main__':
    main()
```

## Making Direct HTTP Requests

If you prefer not to use the provided clients, you can make direct HTTP requests to the Logfire API using any HTTP
client library, such as `requests` in Python. Below are the general steps and an example to guide you:

### General Steps to Make a Direct HTTP Request

1. **Set the Endpoint URL**: The base URL for the Logfire API is `https://logfire-us.pydantic.dev` for accounts in the US region, and `https://logfire-eu.pydantic.dev` for accounts in the EU region.

2. **Add Authentication**: Include the read token in your request headers to authenticate.
    The header key should be `Authorization` with the value `Bearer <your_read_token_here>`.

3. **Define the SQL Query**: Write the SQL query you want to execute.

4. **Send the Request**: Use an HTTP GET request to the `/v1/query` endpoint with the SQL query as a query parameter.

**Note:** You can provide additional query parameters to control the behavior of your requests.
You can also use the `Accept` header to specify the desired format for the response data (JSON, Arrow, or CSV).

### Example: Using Python `requests` Library

```
import requests

# Define the base URL and your read token
base_url = 'https://logfire-us.pydantic.dev'  # or 'https://logfire-eu.pydantic.dev' for EU accounts
read_token = '<your_read_token_here>'

# Set the headers for authentication
headers = {'Authorization': f'Bearer {read_token}'}

# Define your SQL query
query = """
SELECT start_timestamp
FROM records
LIMIT 1
"""

# Prepare the query parameters for the GET request
params = {
    'sql': query
}

# Send the GET request to the Logfire API
response = requests.get(f'{base_url}/v1/query', params=params, headers=headers)

# Check the response status
if response.status_code == 200:
    print("Query Successful!")
    print(response.json())
else:
    print(f"Failed to execute query. Status code: {response.status_code}")
    print(response.text)
```

### Additional Configuration

The Logfire API supports various response formats and query parameters to give you flexibility in how you retrieve your data:

- **Response Format**: Use the `Accept` header to specify the response format. Supported values include:
  - `application/json`: Returns the data in JSON format. By default, this will be column-oriented unless specified otherwise with the `json_rows` parameter.
  - `application/vnd.apache.arrow.stream`: Returns the data in Apache Arrow format, suitable for high-performance data processing.
  - `text/csv`: Returns the data in CSV format, which is easy to use with many data tools.
  - If no `Accept` header is provided, the default response format is JSON.
- **Query Parameters**:
  - **`sql`**: The SQL query to execute. This is the only required query parameter.
  - **`min_timestamp`**: An optional ISO-format timestamp to filter records with `start_timestamp` greater than this value for the `records` table or `recorded_timestamp` greater than this value for the `metrics` table. The same filtering can also be done manually within the query itself.
  - **`max_timestamp`**: Similar to `min_timestamp`, but serves as an upper bound for filtering `start_timestamp` in the `records` table or `recorded_timestamp` in the `metrics` table. The same filtering can also be done manually within the query itself.
  - **`limit`**: An optional parameter to limit the number of rows returned by the query. If not specified, **the default limit is 500**. The maximum allowed value is 10,000.
  - **`row_oriented`**: Only affects JSON responses. If set to `true`, the JSON response will be row-oriented; otherwise, it will be column-oriented.

All query parameters besides `sql` are optional and can be used in any combination to tailor the API response to your needs.

---

## Writing SQL Queries for Dashboards

This guide provides practical recipes and patterns for writing useful SQL queries in **Logfire**. We'll focus on querying the [`records`](https://logfire.pydantic.dev/docs/reference/sql/#records-columns) table, which contains your logs and spans. The goal is to help you create useful dashboards, but we recommend using the Explore view to learn and experiment.

For a complete list of available tables and columns, please see the [SQL Reference](https://logfire.pydantic.dev/docs/reference/sql/).

## Finding common cases

### Simple examples

Here are two quick useful examples to try out immediately in the Explore view.

To find the most common operations based on [`span_name`](https://logfire.pydantic.dev/docs/reference/sql/#span_name):

```
SELECT
    COUNT() AS count,
    span_name
FROM records
GROUP BY span_name
ORDER BY count DESC
```

Similarly, to find the most common [`exception_type`](https://logfire.pydantic.dev/docs/reference/sql/#exception_type) s:

```
SELECT
    COUNT() AS count,
    exception_type
FROM records
WHERE is_exception
GROUP BY exception_type
ORDER BY count DESC
```

Finally, to find the biggest traces, which may be a sign of an operation doing too many things:

```
SELECT
    COUNT() AS count,
    trace_id
FROM records
GROUP BY trace_id
ORDER BY count DESC
```

### The general pattern

The basic template is:

```
SELECT
    COUNT() AS count,
    <columns_to_group_by>
FROM records
WHERE <filter_conditions>
GROUP BY <columns_to_group_by>
ORDER BY count DESC
LIMIT 10
```

- The alias `AS count` allows us to refer to the count in the `ORDER BY` clause.
- `ORDER BY count DESC` sorts the results to show the most common groups first.
- `WHERE <filter_conditions>` is optional and depends on your specific use case.
- `LIMIT 10` isn't usually needed in the Explore view, but is helpful when creating charts.
- `<columns_to_group_by>` can be one or more columns and should be the same in the `SELECT` and `GROUP BY` clauses.

### Useful things to group by

- [`span_name`](https://logfire.pydantic.dev/docs/reference/sql/#span_name): this is nice and generic and shouldn't have too much variability, creating decently sized meaningful groups. It's especially good for HTTP server request spans, where it typically contains the HTTP method and route (the path template) without the specific parameters. To focus on such spans, trying filtering by the appropriate [`otel_scope_name`](https://logfire.pydantic.dev/docs/reference/sql/#otel_scope_name), e.g. `WHERE otel_scope_name = 'opentelemetry.instrumentation.fastapi'` if you're using [`logfire.instrument_fastapi()`](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/).
- [`exception_type`](https://logfire.pydantic.dev/docs/reference/sql/#exception_type)
- [`http_response_status_code`](https://logfire.pydantic.dev/docs/reference/sql/#http_response_status_code)
- [`attributes->>'...'`](https://logfire.pydantic.dev/docs/reference/sql/#attributes): this will depend on your specific data. Try taking a look at some relevant spans in the Live view first to see what attributes are available.
  - If you have a [custom attribute](https://logfire.pydantic.dev/docs/guides/onboarding-checklist/add-manual-tracing/#attributes) like `attributes->>'user_id'`, you can group by that to see which users are most active or have the most errors.
  - If you're using [`logfire.instrument_fastapi()`](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/), there's often useful values inside the `fastapi.arguments.values` attribute, e.g. `attributes->'fastapi.arguments.values'->>'user_id'`.
  - All web frameworks also typically capture several other potentially useful HTTP attributes. In particular if you [capture headers](https://logfire.pydantic.dev/docs/integrations/web-frameworks/#capturing-http-server-request-and-response-headers) (e.g. with `logfire.instrument_django(capture_headers=True)`) then you can try e.g. `attributes->>'http.request.header.host'`.
  - For database query spans, try `attributes->>'db.statement'`, `attributes->>'db.query.text'`, or `attributes->>'db.query.summary'`.
- [`message`](https://logfire.pydantic.dev/docs/reference/sql/#message) \- this is often very variable which may make it less useful depending on your data. But if you're using a SQL instrumentation with the Python **Logfire** SDK, then this contains a rough summary of the SQL query, helping to group similar queries together if `attributes->>'db.statement'` or `attributes->>'db.query.text'` is too granular. In this case you probably want a filter such as `otel_scope_name = 'opentelemetry.instrumentation.sqlalchemy'` to focus on SQL queries.
- [`service_name`](https://logfire.pydantic.dev/docs/reference/sql/#service_name) if you've configured multiple services. This can easily be combined with any other column.
- [`deployment_environment`](https://logfire.pydantic.dev/docs/reference/sql/#deployment_environment) if you've configured multiple [environments](https://logfire.pydantic.dev/docs/how-to-guides/environments/) (e.g. `production` and `staging`, `development`) and have more than one selected in the UI at the moment. Like `service_name`, this is good for combining with other columns.

### Useful `WHERE` clauses

- [`is_exception`](https://logfire.pydantic.dev/docs/reference/sql/#is_exception)
- [`level >= 'error'`](https://logfire.pydantic.dev/docs/reference/sql/#level) \- maybe combined with the above with either `AND` or `OR`, since you can have non-error exceptions and non-exception errors.
- [`level > 'info'`](https://logfire.pydantic.dev/docs/reference/sql/#level) to also find warnings and other notable records, not just errors.
- [`http_response_status_code >= 400`](https://logfire.pydantic.dev/docs/reference/sql/#http_response_status_code) or `500` to find HTTP requests that didn't go well.
- [`service_name = '...'`](https://logfire.pydantic.dev/docs/reference/sql/#service_name) can help a lot to make queries faster.
- [`otel_scope_name = '...'`](https://logfire.pydantic.dev/docs/reference/sql/#otel_scope_name) to focus on a specific (instrumentation) library, e.g. `opentelemetry.instrumentation.fastapi`, `logfire.openai`, or `pydantic-ai`.
- [`duration > 2`](https://logfire.pydantic.dev/docs/reference/sql/#duration) to find slow spans. Replace `2` with your desired number of seconds.
- [`parent_span_id IS NULL`](https://logfire.pydantic.dev/docs/reference/sql/#parent_span_id) to find top-level spans, i.e. the root of a trace.

### Using in Dashboards

For starters:

1. Create a panel in a [new or existing custom (not standard) dashboard](https://logfire.pydantic.dev/docs/guides/web-ui/dashboards/#creating-custom-dashboards).
2. Click the **Type** dropdown and select **Table**.
3. Paste your query into the SQL editor.
4. Give your panel a name.
5. Click **Apply** to save it.

Tables are easy and flexible. They can handle any number of `GROUP BY` columns, and don't really need a `LIMIT` to be practical. But they're not very pretty. Try making a bar chart instead:

1. Click the pencil icon to edit your panel.
2. Change the **Type** to **Bar Chart**.
3. Add `LIMIT 10` or so at the end of your query if there are too many bars.
4. If you have multiple `GROUP BY` columns, you will need to convert them to one expression by concatenating strings. Replace each `,` in the `GROUP BY` clause with `|| ' - ' ||` to create a single string with dashes between the values, e.g. replace `service_name, span_name` with `service_name || ' -  ' || span_name`. Do this in both the `SELECT` and `GROUP BY` clauses. You can optionally add an `AS` alias to the concatenated string in the `SELECT` clause, like `service_name || ' - ' || span_name AS service_and_span`, and then use that alias in the `GROUP BY` clause instead of repeating the concatenation.
5. If your grouping column happens to be a number, add `::text` to it to convert it to a string so that it's recognized as a category by the chart.
6. Bar charts generally put the first rows in the query result at the bottom. If you want the most common items at the top, wrap the whole query in `SELECT * FROM (<original query>) ORDER BY count ASC` to flip the order. Now the inner `ORDER BY count DESC` ensures that we have the most common items (before the limit is applied) and the outer `ORDER BY count ASC` is for appearance.

For a complete example, you can replace this:

```
SELECT
    COUNT() AS count,
    service_name,
    span_name
FROM records
GROUP BY service_name, span_name
ORDER BY count DESC
```

with:

```
SELECT * FROM (
    SELECT
        COUNT() AS count,
        service_name || ' - ' || span_name AS service_and_span
    FROM records
    GROUP BY service_and_span
    ORDER BY count DESC
    LIMIT 10
) ORDER BY count ASC
```

Finally, check the **Settings** tab in the panel editor to tweak the appearance of your chart.

## Aggregating Numerical Data

Instead of just counting rows, you can perform calculations on numerical expressions. [`duration`](https://logfire.pydantic.dev/docs/reference/sql/#duration) is a particularly useful column for performance analysis. Tweaking our first example:

```
SELECT
    SUM(duration) AS total_duration_seconds,
    span_name
FROM records
WHERE duration IS NOT NULL -- Ignore logs, which have no duration
GROUP BY span_name
ORDER BY total_duration_seconds DESC
```

This will show you which operations take the most time overall, whether it's because they run frequently or take a long time each time.

Alternatively you could use `AVG(duration)` (for the mean) or `MEDIAN(duration)` to find which operations are slow on average, or `MAX(duration)` to find the worst case scenarios. See the [Datafusion documentation](https://datafusion.apache.org/user-guide/sql/aggregate_functions.html) for the full list of available aggregation functions.

Other numerical values can typically be found inside `attributes` and will depend on your data. LLM spans often have `attributes->'gen_ai.usage.input_tokens'` and `attributes->'gen_ai.usage.output_tokens'` which you can use to monitor costs.

Warning

`SUM(attributes->'...')` and other numerical aggregations will typically return an error because the database doesn't know the type of the JSON value inside `attributes`, so use `SUM((attributes->'...')::numeric)` to convert it to a number.

### Percentiles

A slightly more advanced aggregation is to calculate percentiles. For example, the 95th percentile means the value below which 95% of the data falls. This is typically referred to as P95, and tells you a more 'typical' worst-case scenario while ignoring the extreme outliers found by using `MAX()`. P90 and P99 are also commonly used.

To calculate this requires a bit of extra syntax:

```
SELECT
    approx_percentile_cont(0.95) WITHIN GROUP (ORDER BY duration) as P95,
    span_name
FROM records
WHERE duration IS NOT NULL
GROUP BY span_name
ORDER BY P95 DESC
```

This query calculates the 95th percentile of the `duration` for each `span_name`. The general pattern is `approx_percentile_cont(<percentile>) WITHIN GROUP (ORDER BY <column>)` where `<percentile>` is a number between 0 and 1.

## Time series

Create a new panel in a dashboard, and by default it will have the type **Time Series Chart** with a query like this:

```
SELECT
    time_bucket($resolution, start_timestamp) AS x,
    count() as count
FROM records
GROUP BY x
```

Here the `time_bucket($resolution, start_timestamp)` is essential. [`$resolution` is a special variable that exists in all dashboards](https://logfire.pydantic.dev/docs/guides/web-ui/dashboards/#resolution-variable) and adjusts automatically based on the time range. You can adjust it while viewing the dashboard using the dropdown in the top left corner. It doesn't exist in the Explore view, so you have to use a concrete interval like `time_bucket('1 hour', start_timestamp)` there. Tick **Show rendered query** in the panel editor to fill in the resolution and other variables so that you can copy the query to the Explore view.

Warning

If you're querying `metrics`, use `recorded_timestamp` instead of `start_timestamp`.

You can give the time bucket expression any name (`x` in the example above), but it must be the same in both the `SELECT` and `GROUP BY` clauses. The chart will detect that the type is a timestamp and use it as the x-axis.

You can replace the `count()` with any aggregation(s) you want. For example, you can show multiple levels of percentiles:

```
SELECT
    time_bucket($resolution, start_timestamp) AS x,
    approx_percentile_cont(0.80) WITHIN GROUP (ORDER BY duration) AS p80,
    approx_percentile_cont(0.90) WITHIN GROUP (ORDER BY duration) AS p90,
    approx_percentile_cont(0.95) WITHIN GROUP (ORDER BY duration) AS p95
FROM records
GROUP BY x
```

### Grouping by Dimension

You can make time series charts with multiple lines (series) for the same numerical metric by grouping by a dimension. This means adding a column to the `SELECT` and `GROUP BY` clauses and then selecting it from the 'Dimension' dropdown next to the SQL editor.

#### Low cardinality dimensions

For a simple example, paste this into the SQL editor:

```
SELECT
    time_bucket($resolution, start_timestamp) AS x,
    log(count()) as log_count,
    level_name(level) as level
FROM records
GROUP BY x, level
```

Then set the **Dimension** dropdown to `level` and the **Metrics** dropdown to `log_count`. This will create a time series chart with multiple lines, one for each log level (e.g. 'info', 'warning', 'error'). Here's what the configuration and result would look like:

[![Time Series Chart with Multiple Lines](https://logfire.pydantic.dev/docs/images/guide/dashboard-queries-level-dimension.png)](https://logfire.pydantic.dev/docs/images/guide/dashboard-queries-level-dimension.png)

Note

You can only set one dimension, but you can set multiple metrics.

Here we use `log(count())` instead of just `count()` because you probably have way more 'info' records than 'error' records, making it hard to notice any spikes in the number of errors. This compresses the y-axis into a logarithmic scale to make it more visually useful, but the numbers are harder to interpret. The `level_name(level)` function converts the numeric [`level`](https://logfire.pydantic.dev/docs/reference/sql/#level) value to a human-readable string.

You can replace `level` with [other useful things to group by](https://logfire.pydantic.dev/docs/how-to-guides/write-dashboard-queries/#useful-things-to-group-by), but they need to be very low cardinality (i.e. not too many unique values) for this simple query to work well in a time series chart. Good examples are:

- `service_name`
- `deployment_environment`
- `exception_type`
- `http_response_status_code`
- `attributes->>'http.method'` (or `attributes->>'http.request.method'` for newer OpenTelemetry instrumentations)

#### Multiple dimensions

Because time series charts can only have one dimension, if you want to group by multiple columns, you need to concatenate them into a single string, e.g. instead of:

```
SELECT
    time_bucket($resolution, start_timestamp) AS x,
    log(count()) as log_count,
    service_name,
    level_name(level) as level
FROM records
GROUP BY x, service_name, level
```

You would do:

```
SELECT
    time_bucket($resolution, start_timestamp) AS x,
    log(count()) as log_count,
    service_name || ' - ' || level_name(level) AS service_and_level
FROM records
GROUP BY x, service_and_level
```

Then set the 'Dimension' dropdown to `service_and_level`. This will create a time series chart with a line for each combination of `service_name` and `level`. Of course this increases the cardinality quickly, making the next section more relevant.

#### High cardinality dimensions

If you try grouping by something with more than a few unique values, you'll end up with a cluttered chart with too many lines. For example, this will look like a mess unless your data is very simple:

```
SELECT
    time_bucket($resolution, start_timestamp) AS x,
    count() as count,
    span_name
FROM records
GROUP BY x, span_name
```

The quick and dirty solution is to add these lines at the end:

```
ORDER BY count DESC
LIMIT 200
```

This will give you a point for the 200 most common _combinations of `x` and `span_name`_. This will often work reasonably well, but the limit will need to be tuned based on the data, and the number of points at each time bucket will vary. Here's the better version:

```
WITH original AS (
    SELECT
        time_bucket($resolution, start_timestamp) AS x,
        count() as count,
        span_name
    FROM records
    GROUP BY x, span_name
),
ranked AS (
    SELECT
        x,
        count,
        span_name,
        ROW_NUMBER() OVER (PARTITION BY x ORDER BY count DESC) AS row_num
    FROM original
)
SELECT
    x,
    count,
    span_name
FROM ranked
WHERE row_num <= 5
ORDER BY x
```

This selects the top 5 `span_name`s for each time bucket, and will usually work perfectly. It may look intimidating, but constructing a query like this can be done very mechanically. Start with a basic query in this form:

```
SELECT
    time_bucket($resolution, start_timestamp) AS x,
    <aggregation_expression> AS metric,
    <dimension_expression> as dimension
FROM records
GROUP BY x, dimension
```

Fill in `<aggregation_expression>` with your desired aggregation (e.g. `count()`, `SUM(duration)`, etc.) and `<dimension_expression>` with the column(s) you want to group by (e.g. `span_name`). Set the 'Dimension' dropdown to `dimension` and the 'Metrics' dropdown to `metric`. That should give you a working (but probably cluttered) time series chart. Then simply paste it into `<original_query>` in the following template:

```
WITH original AS (
    <original_query>
),
ranked AS (
    SELECT
        x,
        metric,
        dimension,
        ROW_NUMBER() OVER (PARTITION BY x ORDER BY metric DESC) AS row_num
    FROM original
)
SELECT
    x,
    metric,
    dimension
FROM ranked
WHERE row_num <= 5
ORDER BY x
```

## Linking to the Live view

While aggregating data with `GROUP BY` is powerful for seeing trends, sometimes you need to investigate specific events, like a single slow operation or a costly API call. In these cases, it's good to include the [`trace_id`](https://logfire.pydantic.dev/docs/reference/sql/#trace_id) column in your `SELECT` clause. Tables in dashboards, the explore view, or alert run results with this column will render the `trace_id` values as clickable links to the Live View.

For example, to find the 10 slowest spans in your system, you can create a 'Table' panel with this query:

```
SELECT
    trace_id,
    duration,
    message
FROM records
ORDER BY duration DESC
LIMIT 10
```

The table alone won't tell you much, but you can click on the `trace_id` of any row to investigate the full context further.

You can also select the [`span_id`](https://logfire.pydantic.dev/docs/reference/sql/#span_id) column to get a link directly to a specific span within the trace viewer. However, this only works if the `trace_id` column is also included in your `SELECT` statement.

Other columns that may be useful to include in such queries:

- [`message`](https://logfire.pydantic.dev/docs/reference/sql/#message) is a human readable description of the span, as seen in the Live view list of records.
- [`start_timestamp`](https://logfire.pydantic.dev/docs/reference/sql/#start_timestamp) and [`end_timestamp`](https://logfire.pydantic.dev/docs/reference/sql/#end_timestamp)
- [`attributes`](https://logfire.pydantic.dev/docs/reference/sql/#attributes)
- [`service_name`](https://logfire.pydantic.dev/docs/reference/sql/#service_name)
- [`otel_scope_name`](https://logfire.pydantic.dev/docs/reference/sql/#otel_scope_name)
- [`deployment_environment`](https://logfire.pydantic.dev/docs/reference/sql/#deployment_environment)
- [`otel_resource_attributes`](https://logfire.pydantic.dev/docs/reference/sql/#otel_resource_attributes)
- [`exception_type`](https://logfire.pydantic.dev/docs/reference/sql/#exception_type) and [`exception_message`](https://logfire.pydantic.dev/docs/reference/sql/#exception_message)
- [`http_response_status_code`](https://logfire.pydantic.dev/docs/reference/sql/#http_response_status_code)
- [`level_name(level)`](https://logfire.pydantic.dev/docs/reference/sql/#level)

## Creating Histograms

Histograms are useful for visualizing the distribution of numerical data, such as the duration of spans. They provide richer information than simple summary statistics like averages. Currently the UI has no built in way to display histograms, but it's possible with SQL. Just copy this template and fill in the `source_data` CTE with your actual data:

```
WITH
source_data AS (
    -- Replace this with your actual source data query.
    -- It must return a single column named `amount` with numeric values.
    select duration as amount from records
),
histogram_config AS (
    -- Tweak this number if you want.
    SELECT 40 AS num_buckets
),

-- The rest of the query is fully automatic, leave it as is.
raw_params AS (
    SELECT MIN(amount)::numeric AS min_a, MAX(amount)::numeric AS max_a, num_buckets
    FROM source_data, histogram_config GROUP BY num_buckets),
params_with_shift AS (
    SELECT *, CASE WHEN min_a <= 0 THEN 1 - min_a ELSE 0 END AS shift
    FROM raw_params),
params AS (
    SELECT *, CASE WHEN min_a = max_a THEN 1.000000001 ELSE exp(ln((max_a + shift) / (min_a + shift)) / num_buckets::double) END AS b
    FROM params_with_shift),
actual_counts AS (
    SELECT floor(log(b, (amount + shift) / (min_a + shift)))::int AS ind, COUNT() AS count
    FROM source_data, params GROUP BY ind),
all_buckets AS (
    SELECT UNNEST(generate_series(0, num_buckets - 1)) as ind
    FROM params),
midpoints AS (
    SELECT ind, (min_a + shift) * power(b, ind + 0.5) - shift as mid
    FROM all_buckets, params)
SELECT round(mid, 3)::text as approx_amount, COALESCE(count, 0) as count
FROM midpoints m LEFT JOIN actual_counts c ON m.ind = c.ind ORDER BY mid;
```

Then set the chart type to **Bar Chart**. Each bar represents a 'bucket' that actual values are placed into. The x-axis will show the approximate amount that the values in the bucket can be rounded to, and the y-axis will show the count of rows in each bucket. This is an exponential histogram, meaning that the buckets are wider for larger values, which is useful for data that has a long tail distribution.

How does this work?

Here's the query again with detailed comments:

```
-- =============================================================================
-- STEP 1: DEFINE YOUR DATA SOURCE
-- =============================================================================
WITH source_data AS (
    -- PASTE YOUR QUERY HERE
    -- This query must select one numerical column named 'amount'.
    -- Example: Test case including negative, zero, and positive values.
    select duration as amount from records
),

-- =============================================================================
-- STEP 2: CONFIGURE HISTOGRAM
-- =============================================================================
histogram_config AS (
    -- This sets the desired number of bars in the final chart.
    SELECT 40 AS num_buckets
),

-- =============================================================================
-- The rest of the query is fully automatic.
-- =============================================================================

-- This CTE performs a single pass over the source data to find its true range.
raw_params AS (
    SELECT
        MIN(amount)::numeric AS min_a,
        MAX(amount)::numeric AS max_a,
        num_buckets
    FROM source_data, histogram_config
    GROUP BY num_buckets
),

-- This CTE calculates a 'shift' value. Logarithms are only defined for positive
-- numbers, so if the data contains 0 or negative values, we must temporarily
-- shift the entire dataset into the positive domain (where the new minimum is 1).
params_with_shift AS (
    SELECT
        *,
        CASE WHEN min_a <= 0 THEN 1 - min_a ELSE 0 END AS shift
    FROM raw_params
),

-- This CTE calculates the final exponential 'base' for the histogram scaling.
-- The base determines how quickly the bucket sizes grow. It is calculated such
-- that 'num_buckets' steps will perfectly cover the shifted data range.
params AS (
    SELECT
        *,
        -- If min = max, we use a base slightly > 1. This prevents log(1) errors
        -- and allows the binning logic to work without a special case.
        CASE
            WHEN min_a = max_a THEN 1.000000001
            ELSE exp(ln((max_a + shift) / (min_a + shift)) / num_buckets::double)
        END AS base
    FROM params_with_shift
),

-- This CTE takes every record from the source data and assigns it to a bucket
-- index (0, 1, 2, ...). This is the core binning logic.
actual_counts AS (
    SELECT
        -- The formula log_base(value/start) calculates "how many exponential steps
        -- of size 'base' are needed to get from the start of the range to the
        -- current value". We use the shifted values for this calculation.
        floor(log(base, (amount + shift) / (min_a + shift)))::int AS bucket_index,
        COUNT() AS count
    FROM source_data, params
    GROUP BY bucket_index
),

-- This CTE generates a perfect, gap-free template of all possible bucket indices,
-- ensuring the final chart has exactly 'num_buckets' bars.
all_buckets AS (
    SELECT UNNEST(generate_series(0, num_buckets - 1)) as bucket_index
    FROM params
),

-- This CTE calculates the representative midpoint for every possible bucket.
-- This logic is separated from the final join to prevent query planner bugs.
midpoints AS (
    SELECT
        bucket_index,
        -- We calculate the geometric midpoint of each bucket in the shifted space
        -- using 'power(base, index + 0.5)' and then shift it back to the
        -- original data's scale. This provides a more representative label for
        -- an exponential scale than a simple arithmetic mean.
        (min_a + shift) * power(base, bucket_index + 0.5) - shift as bucket_midpoint
    FROM all_buckets
    CROSS JOIN params
)

-- This final step assembles the chart data. It joins the calculated midpoints
-- with the actual counts and formats the output.
SELECT
    round(m.bucket_midpoint, 3)::text as bucket_midpoint,
    -- If a bucket has no items, its count will be NULL after the join.
    -- COALESCE turns these NULLs into 0 for the chart.
    COALESCE(c.count, 0) as count
FROM midpoints m
LEFT JOIN actual_counts c ON m.bucket_index = c.bucket_index
ORDER BY m.bucket_midpoint;
```

---