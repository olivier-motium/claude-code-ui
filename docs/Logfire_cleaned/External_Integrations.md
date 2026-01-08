# External Integrations

## Contents

- [Alternative backends](#alternative-backends)
- [Alternative clients](#alternative-clients)
- [Collecting Metrics from Cloud Providers](#collecting-metrics-from-cloud-providers)
- [Logfire MCP Server](#logfire-mcp-server)
- [Setup Slack Alerts](#setup-slack-alerts)

---

## Alternative backends

**Logfire** uses the OpenTelemetry standard. This means that you can configure the SDK to export to any backend that supports OpenTelemetry.

The easiest way is to set the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable to a URL that points to your backend.
This will be used as a base, and the SDK will append `/v1/traces` and `/v1/metrics` to the URL to send traces and metrics, respectively.

Alternatively, you can use the `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` and `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT`
environment variables to specify the URLs for traces, metrics and logs separately.
These URLs should include the full path, including `/v1/traces` and `/v1/metrics`.

Note

The data will be encoded using **Protobuf** (not JSON) and sent over **HTTP** (not gRPC).

Make sure that your backend supports this! ![🤓](https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/svg/1f913.svg)

## Example with Jaeger

Run this minimal command to start a [Jaeger](https://www.jaegertracing.io/) container:

```
docker run --rm \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Then run this code:

```
import os

import logfire

# Jaeger only supports traces, not metrics, so only set the traces endpoint
# to avoid errors about failing to export metrics.
# Use port 4318 for HTTP, not 4317 for gRPC.
traces_endpoint = 'http://localhost:4318/v1/traces'
os.environ['OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'] = traces_endpoint

logfire.configure(
    # Setting a service name is good practice in general, but especially
    # important for Jaeger, otherwise spans will be labeled as 'unknown_service'
    service_name='my_logfire_service',

    # Sending to Logfire is on by default regardless of the OTEL env vars.
    # Keep this line here if you don't want to send to both Jaeger and Logfire.
    send_to_logfire=False,
)

with logfire.span('This is a span'):
    logfire.info('Logfire logs are also actually just spans!')
```

Finally open [http://localhost:16686/search?service=my\_logfire\_service](http://localhost:16686/search?service=my_logfire_service) to see the traces in the Jaeger UI.

[![Jaeger traces view](https://logfire.pydantic.dev/docs/images/guide/jaeger-traces-view.png)](https://logfire.pydantic.dev/docs/images/guide/jaeger-traces-view.png)

You can click on a specific trace to get a more detailed view:
[![Jager trace details](https://logfire.pydantic.dev/docs/images/guide/jaeger-trace-details.png)](https://logfire.pydantic.dev/docs/images/guide/jaeger-trace-details.png)

And this is how a more "complex" trace would look like:
[![Jager complete trace](https://logfire.pydantic.dev/docs/images/guide/jaeger-complete-trace-view.png)](https://logfire.pydantic.dev/docs/images/guide/jaeger-complete-trace-view.png)

## Other environment variables

If `OTEL_TRACES_EXPORTER` and/or `OTEL_METRICS_EXPORTER` are set to any non-empty value other than `otlp`, then **Logfire** will ignore the corresponding `OTEL_EXPORTER_OTLP_*` variables. This is because **Logfire** doesn't support other exporters, so we assume that the environment variables are intended to be used by something else. Normally you don't need to worry about this, and you don't need to set these variables at all unless you want to prevent **Logfire** from setting up these exporters.

See the [OpenTelemetry documentation](https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html) for information about the other headers you can set, such as `OTEL_EXPORTER_OTLP_HEADERS`.

---

## Alternative clients

**Logfire** uses the OpenTelemetry standard. This means that you can configure standard OpenTelemetry SDKs
in many languages to export to the **Logfire** backend, including those outside our
[first-class supported languages](https://logfire.pydantic.dev/docs/languages/). Depending on your SDK, you may need to set only
these [environment variables](https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/):

- `OTEL_EXPORTER_OTLP_ENDPOINT=https://logfire-us.pydantic.dev` for both traces and metrics, or:
  - `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://logfire-us.pydantic.dev/v1/traces` for just traces
  - `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://logfire-us.pydantic.dev/v1/metrics` for just metrics
  - `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=https://logfire-us.pydantic.dev/v1/logs` for just logs
- `OTEL_EXPORTER_OTLP_HEADERS='Authorization=your-write-token'` \- see [Create Write Tokens](https://logfire.pydantic.dev/docs/how-to-guides/create-write-tokens/)
to obtain a write token and replace `your-write-token` with it.
- `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` to export in Protobuf format over HTTP (not gRPC).
The **Logfire** backend supports both Protobuf and JSON, but only over HTTP for now. Some SDKs (such as Python) already use this value as the default so setting this isn't required, but other SDKs use `grpc` as the default.

Note

This page shows `https://logfire-us.pydantic.dev` as the base URL which is for the US [region](https://logfire.pydantic.dev/docs/reference/data-regions/).
If you are using the EU region, use `https://logfire-eu.pydantic.dev` instead.

## Example with Python

First, run these commands:

```
pip install opentelemetry-exporter-otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=https://logfire-us.pydantic.dev
export OTEL_EXPORTER_OTLP_HEADERS='Authorization=your-write-token'
```

Then run this script with `python`:

```
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(exporter)
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(span_processor)
tracer = tracer_provider.get_tracer('my_tracer')

tracer.start_span('Hello World').end()
```

Then navigate to the Live view for your project in your browser. You should see a trace with a single span named `Hello World`.

To configure the exporter without environment variables:

```
exporter = OTLPSpanExporter(
    endpoint='https://logfire-us.pydantic.dev/v1/traces',
    headers={'Authorization': 'your-write-token'},
)
```

## Example with NodeJS

> See also our [JS/TS SDK](https://github.com/pydantic/logfire-js) which supports many JS environments, including NodeJS, web browsers, and Cloudflare Workers.

Create a `main.js` file containing the following:

main.js

```
import {NodeSDK} from "@opentelemetry/sdk-node";
import {OTLPTraceExporter} from "@opentelemetry/exporter-trace-otlp-proto";
import {BatchSpanProcessor} from "@opentelemetry/sdk-trace-node";
import {trace} from "@opentelemetry/api";
import {Resource} from "@opentelemetry/resources";
import {ATTR_SERVICE_NAME} from "@opentelemetry/semantic-conventions";

const traceExporter = new OTLPTraceExporter();
const spanProcessor = new BatchSpanProcessor(traceExporter);
const resource = new Resource({[ATTR_SERVICE_NAME]: "my_service"});
const sdk = new NodeSDK({spanProcessor, resource});
sdk.start();

const tracer = trace.getTracer("my_tracer");
tracer.startSpan("Hello World").end();

sdk.shutdown().catch(console.error);
```

Then run these commands:

```
export OTEL_EXPORTER_OTLP_ENDPOINT=https://logfire-us.pydantic.dev
export OTEL_EXPORTER_OTLP_HEADERS='Authorization=your-write-token'

npm init es6 -y # creates package.json with type module
npm install @opentelemetry/sdk-node
node main.js
```

## Example with Rust

> See also our [Rust SDK](https://github.com/pydantic/logfire-rust) which provides a more streamlined developer experience for Rust applications.

First, set up a new Cargo project:

```
cargo new --bin otel-example && cd otel-example
export OTEL_EXPORTER_OTLP_ENDPOINT=https://logfire-us.pydantic.dev
export OTEL_EXPORTER_OTLP_HEADERS='Authorization=your-write-token'
```

Update the `Cargo.toml` and `main.rs` files with the following contents:

Cargo.toml

```
[package]
name = "otel-example"
version = "0.1.0"
edition = "2021"

[dependencies]
opentelemetry = { version = "*", default-features = false, features = ["trace"] }
# Note: `reqwest-rustls` feature is necessary else you'll have a cryptic failure to export;
# see https://github.com/open-telemetry/opentelemetry-rust/issues/2169
opentelemetry-otlp = { version = "*", default-features = false, features = ["trace", "http-proto", "reqwest-blocking-client", "reqwest-rustls"] }
```

src/main.rs

```
use opentelemetry::{
    global::ObjectSafeSpan,
    trace::{Tracer, TracerProvider},
};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let otlp_exporter = opentelemetry_otlp::new_exporter()
        .http()
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
        // If you don't want to export environment variables, you can also configure
        // programmatically like so:
        //
        // (You'll need to add `use opentelemetry_otlp::WithExportConfig;` to the top of the
        // file to access the `.with_endpoint` method.)
        //
        // .with_endpoint("https://logfire-us.pydantic.dev/v1/traces")
        // .with_headers({
        //     let mut headers = std::collections::HashMap::new();
        //     headers.insert(
        //         "Authorization".into(),
        //         "your-write-token".into(),
        //     );
        //     headers
        // })
        ;

    let tracer_provider = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(otlp_exporter)
        .install_simple()?;
    let tracer = tracer_provider.tracer("my_tracer");

    tracer.span_builder("Hello World").start(&tracer).end();

    Ok(())
}
```

Finally, use `cargo run` to execute.

## Example with Go

Create a file `main.go` containing the following:

```
package main

import (
    "context"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
    "go.opentelemetry.io/otel/sdk/trace"
)

func main() {
    ctx := context.Background()
    traceExporter, _ := otlptracehttp.New(ctx)
    batchSpanProcessor := trace.NewBatchSpanProcessor(traceExporter)
    tracerProvider := trace.NewTracerProvider(trace.WithSpanProcessor(batchSpanProcessor))
    tracer := tracerProvider.Tracer("my_tracer")

    ctx, span := tracer.Start(ctx, "Hello World")
    span.End()

    tracerProvider.Shutdown(ctx)
}
```

Then run these commands:

```
export OTEL_EXPORTER_OTLP_ENDPOINT=https://logfire-us.pydantic.dev
export OTEL_EXPORTER_OTLP_HEADERS='Authorization=your-write-token'

# Optional, but otherwise you will see the service name set to `unknown_service:otel_example`
export OTEL_RESOURCE_ATTRIBUTES="service.name=my_service"

go mod init otel_example
go mod tidy
go run .
```

---

## Collecting Metrics from Cloud Providers

Cloud metrics provide valuable insights into the performance, health, and usage of your cloud infrastructure. By collecting metrics from your cloud provider and centralizing them in Logfire, you can create a single pane of glass for monitoring your entire infrastructure stack.

Key benefits of collecting cloud metrics include:

- **Single pane of glass visibility**: Correlate metrics across different cloud providers and services
- **Centralized alerting**: Set up consistent alerting rules across your entire infrastructure
- **Cost optimization**: Identify resource usage patterns and optimize spending
- **Performance monitoring**: Track application performance alongside infrastructure metrics

## 1\. Why Use the OpenTelemetry Collector?

Rather than you giving us access to your cloud provider directly we recommend using the [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/) to collect metrics from your cloud provider. The OpenTelemetry Collector is a vendor-agnostic service that can collect, process, and export telemetry data (metrics, logs, traces) from various sources.
The advantages of this approach include:

- **Security**: You maintain control over your cloud credentials and don't need to share them with external services
- **Data governance**: Filter sensitive or unnecessary metrics before they leave your environment
- **Cost control**: Reduce data transfer costs by filtering and sampling metrics locally
- **Flexibility**: Transform, enrich, or aggregate metrics before sending them to Logfire
- **Vendor lock in**: Send the same metrics to multiple monitoring systems if needed

For general information about setting up and configuring the OpenTelemetry Collector, see our [OpenTelemetry Collector guide](https://logfire.pydantic.dev/docs/how-to-guides/otel-collector/otel-collector-overview/).

One important consideration before you embark on this guide is what your overall data flow is going to be.
For example, you don't want to export your application metrics to Logfire and Google Cloud Monitoring and _also_ export your Google Cloud Monitoring metrics to Logfire, you'll end up with duplicate metrics!

We recommend you export all application metrics to Logfire directly and then use the OpenTelemetry Collector to collect metrics from your cloud provider that are _not_ already being exported to Logfire.

## 2\. Collecting Metrics from Google Cloud Platform (GCP)

The [Google Cloud Monitoring receiver](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/receiver/googlecloudmonitoringreceiver) allows you to collect metrics from Google Cloud Monitoring (formerly Stackdriver) and forward them to Logfire.

### 2.1 Prerequisites

1. A GCP project with the Cloud Monitoring API enabled
2. Service account credentials with appropriate IAM permissions (see IAM Setup below)
3. OpenTelemetry Collector with the `googlecloudmonitoring` receiver

### 2.2 Enabling the Cloud Monitoring API

To enable the Cloud Monitoring API for your GCP project follow the steps listed in [the official documentation](https://cloud.google.com/monitoring/api/enable-api).

### 2.3 IAM Setup

To collect metrics from Google Cloud Monitoring, you need to create a service account with the appropriate permissions:

#### 2.3.1 Required Permissions

The service account needs the following specific roles:

- `roles/monitoring.viewer`: grants read-only access to Monitoring in the Google Cloud console and the Cloud Monitoring API.

See the [official documentation](https://cloud.google.com/monitoring/access-control) for a complete list of permissions required for the Monitoring API.

#### 2.3.2 Creating a Service Account

To create a service account you can use the Google Cloud CLI or the GCP Console. Here are the steps using the CLI:

```
gcloud iam service-accounts create logfire-metrics-collector \
    --display-name="Logfire Metrics Collector" \
    --description="Service account for collecting metrics to send to Logfire"
```

Grant the service account the necessary permissions:

```
export PROJECT_ID="your-gcp-project-id"
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:logfire-metrics-collector@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/monitoring.viewer"
```

### 2.4 Configuration

Create a collector configuration file with the Google Cloud Monitoring receiver:

gcp-metrics-collector.yaml

```
receivers:
  googlecloudmonitoring:
    # Your GCP project ID
    project_id: "${env:PROJECT_ID}"
    # Collection interval
    collection_interval: 60s
    # Example of metric names to collect
    # See https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/receiver/googlecloudmonitoringreceiver#configuration
    metrics_list:
      - metric_name: "cloudsql.googleapis.com/database/cpu/utilization"
      - metric_name: "kubernetes.io/container/memory/limit_utilization"
      # This will collect the CPU usage for the container we are deploying the collector itself in!
      - metric_name: "run.googleapis.com/container/cpu/usage"

exporters:
  debug:
  otlphttp:
    # Configure the US / EU endpoint for Logfire.
    # - US: https://logfire-us.pydantic.dev
    # - EU: https://logfire-eu.pydantic.dev
    endpoint: "https://logfire-us.pydantic.dev"
    headers:
      Authorization: "Bearer ${env:LOGFIRE_TOKEN}"

extensions:
  health_check:
    # The PORT env var is set by CloudRun
    endpoint: "0.0.0.0:${env:PORT:-13133}"

service:
  pipelines:
    metrics:
      receivers: [googlecloudmonitoring]
      exporters: [otlphttp, debug]
  extensions: [health_check]
```

### 2.5 Authentication

Authentication to Google Cloud via [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials).
If you are running on Kubernetes you will have to set up [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) to allow the OpenTelemetry Collector to access Google Cloud resources.
If you are running on Cloud Run or other GCP services, the default service account will be used automatically.
You can either give the default service account the necessary permissions (in which case you can skip creating the service account above) or create a new service account and configure the workload running the OpenTelemetry Collector to use this service account.
The latter is advisable from a security perspective, as it allows you to limit the permissions of the service account to only what is necessary for the OpenTelemetry Collector.

Authentication to Logfire must happen via a write token.
It is recommended that you store the write token as a secret (e.g. in Kubernetes secrets) and reference it in the collector configuration file as an environment variable to avoid hardcoding sensitive information in the configuration file.

### 2.6 Example deployment using Cloud Run

This section shows how to deploy the OpenTelemetry Collector to Google Cloud Run using the service account created in section 2.3.

#### 2.6.1 Create a Dockerfile

First, create a Dockerfile that uses the official OpenTelemetry Collector contrib image and copies your configuration:

Dockerfile

```
# Update the base image to the latest version as needed
# It's good practice to use a specific version tag for stability
FROM otel/opentelemetry-collector-contrib:0.128.0

# Copy the collector configuration created previously to the default location
COPY gcp-metrics-collector.yaml /etc/otelcol-contrib/config.yaml
```

#### 2.6.2 Create a secret with your Logfire token

To securely store your Logfire write token, create a secret in Google Secret Manager.

First [enable the Secrets Manager API](https://cloud.google.com/secret-manager/docs/configuring-secret-manager) for your project.
Using the Google Cloud CLI:

```
# Enable the Secret Manager API
gcloud services enable secretmanager.googleapis.com
```

Then, create a secret and grant the service account access to it:

```
# Set your project ID
export PROJECT_ID="your-gcp-project-id"
export LOGFIRE_TOKEN="your-logfire-write-token"
# Create the secret
echo -n "$LOGFIRE_TOKEN" | gcloud secrets create logfire-token --data-file=-
# Grant the service account access to the secret
gcloud secrets add-iam-policy-binding logfire-token \
  --member="serviceAccount:logfire-metrics-collector@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

#### 2.6.2 Build and push the container image

Build and push your container image to Google Container Registry or Artifact Registry.

With the following project structure:

```
.
├── Dockerfile
└── gcp-metrics-collector.yaml
```

You can run:

```
# Set your project ID
export PROJECT_ID="your-gcp-project-id"

# Set the port for health checks
# Set no CPU throttling so that the collector runs even though it is not receiving external HTTP requests
# Do not allow any external HTTP traffic to the service
# Set the service to use the service account created earlier
# Inject the project ID as an environment variable
# Set the minimum number of instances to 1 so that the collector is always running
# Inject the Logfire token secret as an environment variable

gcloud run deploy otel-collector-gcp-metrics \
--source . \
--project $PROJECT_ID \
--port 13133 \
--no-allow-unauthenticated \
--service-account logfire-metrics-collector@$PROJECT_ID.iam.gserviceaccount.com \
--set-env-vars PROJECT_ID=$PROJECT_ID \
--no-cpu-throttling \
--min-instances 1 \
--update-secrets=LOGFIRE_TOKEN=logfire-token:latest
```

Once the deployment is complete you should be able to run the following query in Logfire to verify metrics are being received:

```
SELECT metric_name, count(*) AS metric_count
FROM metrics
WHERE metric_name IN ('cloudsql.googleapis.com/database/cpu/utilization', 'kubernetes.io/container/memory/limit_utilization')
GROUP BY metric_name;
```

#### 2.6.4 Configuring scaling

Depending on the amount of metrics data points you are collecting you may need to do more advanced configuration of the OpenTelemetry Collector to handle the load.
For example, you may want to configure the `batch` processor to batch metrics before sending them to Logfire, or use the `memory_limiter` processor to limit memory usage.
You also may need to tweak the resources allocated to the Cloud Run service to ensure it can handle the load.

## 3\. Collecting Metrics from Amazon Web Services (AWS)

The [AWS CloudWatch metrics receiver](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/receiver/awscloudwatchmetricsreceiver) allows you to collect metrics from Amazon CloudWatch and forward them to Logfire.

### 3.1 Prerequisites

1. An AWS account with CloudWatch metrics enabled
2. IAM credentials with appropriate permissions (see IAM Setup below)
3. OpenTelemetry Collector with the `awscloudwatchmetrics` receiver

### 3.2 IAM Setup

To collect metrics from AWS CloudWatch, you need to configure IAM credentials with the appropriate permissions:

#### 3.2.1 Required Permissions

The IAM role or user needs the following CloudWatch permissions:

- `cloudwatch:GetMetricData`: Retrieve metric data points
- `cloudwatch:GetMetricStatistics`: Get aggregated metric statistics
- `cloudwatch:ListMetrics`: List available metrics

For ECS-specific metrics, you may also need EC2 permissions:

- `ec2:DescribeTags`: Get resource tags
- `ec2:DescribeInstances`: Get instance information
- `ec2:DescribeRegions`: List available regions

### 3.3 Configuration

Create a collector configuration file with the AWS ECS container metrics receiver:

aws-metrics-collector.yaml

```
receivers:
  # Collect ECS container metrics directly from the ECS task metadata endpoint
  awsecscontainermetrics:
    # Collection interval
    collection_interval: 60s

exporters:
  debug:
  otlphttp:
    # Configure the US / EU endpoint for Logfire.
    # - US: https://logfire-us.pydantic.dev
    # - EU: https://logfire-eu.pydantic.dev
    endpoint: "https://logfire-us.pydantic.dev"
    headers:
      Authorization: "Bearer ${env:LOGFIRE_TOKEN}"

extensions:
  health_check:
    endpoint: "0.0.0.0:13133"

service:
  pipelines:
    metrics:
      receivers: [awsecscontainermetrics]
      exporters: [otlphttp, debug]
  extensions: [health_check]
```

This configuration collects metrics directly from the ECS task metadata endpoint including:

- Container CPU utilization
- Container memory utilization
- Container network I/O metrics
- Container storage I/O metrics

**For CloudWatch Metrics**: If you need to collect metrics from other AWS services (RDS, ALB, etc.), use the [AWS Distro for OpenTelemetry (ADOT)](https://aws-otel.github.io/docs/introduction) collector image `public.ecr.aws/aws-observability/aws-otel-collector` which includes the `awscloudwatchmetrics` receiver.

### 3.4 Authentication

The AWS ECS container metrics receiver collects metrics directly from the ECS task metadata endpoint, so **no AWS credentials or IAM permissions are required** for the metrics collection itself.

However, you still need:

1. **ECS Task Execution Role permissions** for:
    \- Pulling container images from ECR
    \- Writing logs to CloudWatch Logs
    \- Reading secrets from AWS Secrets Manager

2. **Logfire authentication** via a write token. Store the write token as a secret in AWS Secrets Manager and reference it as an environment variable.

### 3.5 Example deployment using Amazon ECS

This section shows how to deploy the OpenTelemetry Collector to Amazon ECS using an IAM role for tasks.

#### 3.5.1 Create ECS Task Role

Since the ECS container metrics receiver doesn't require AWS API access, we only need a basic ECS task execution role:

```
# Create the ECS task trust policy file
cat > ecs-task-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [\
    {\
      "Effect": "Allow",\
      "Principal": {\
        "Service": "ecs-tasks.amazonaws.com"\
      },\
      "Action": "sts:AssumeRole"\
    }\
  ]
}
EOF

# Create the Secrets Manager policy for accessing the Logfire token
cat > secretsmanager-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [\
        {\
            "Effect": "Allow",\
            "Action": [\
                "secretsmanager:GetSecretValue"\
            ],\
            "Resource": "arn:aws:secretsmanager:*:*:secret:logfire-token*"\
        }\
    ]
}
EOF

# Get your AWS account ID
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create the IAM role
aws iam create-role \
    --role-name LogfireMetricsCollectorRole \
    --assume-role-policy-document file://ecs-task-trust-policy.json \
    --description "ECS task role for Logfire metrics collector"

# Attach ECS task execution permissions
aws iam attach-role-policy \
    --role-name LogfireMetricsCollectorRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Grant access to read the Logfire token secret
aws iam put-role-policy \
    --role-name LogfireMetricsCollectorRole \
    --policy-name LogfireSecretsAccess \
    --policy-document file://secretsmanager-policy.json
```

#### 3.5.2 Store Logfire token in AWS Secrets Manager

Store your Logfire write token as an ECS secret using AWS Secrets Manager. This follows the [ECS best practices for specifying sensitive data](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/specifying-sensitive-data.html):

```
# Create the secret for Logfire token
aws secretsmanager create-secret \
    --name logfire-token \
    --description "Logfire write token for metrics collection" \
    --secret-string "your-logfire-write-token-here"
```

#### 3.5.3 Create a Dockerfile

Create a Dockerfile that uses the official OpenTelemetry Collector contrib image:

Dockerfile

```
# Update the base image to the latest version as needed
FROM otel/opentelemetry-collector-contrib:0.128.0

# Copy the collector configuration to the default location
COPY aws-metrics-collector.yaml /etc/otelcol-contrib/config.yaml
```

#### 3.5.4 Create an ECS Task Definition

Create an ECS task definition that uses the IAM role. First, get your AWS account ID and create the task definition:

```
# Get your AWS account ID (reuse from earlier or get it again)
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION="us-east-1"  # Change to your preferred region

# Create the task definition using the account ID
cat > task-definition.json << EOF
{
  "family": "logfire-metrics-collector",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/LogfireMetricsCollectorRole",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/LogfireMetricsCollectorRole",
  "containerDefinitions": [\
    {\
      "name": "otel-collector",\
      "image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/logfire-metrics-collector:latest",\
      "essential": true,\
      "portMappings": [\
        {\
          "containerPort": 13133,\
          "protocol": "tcp"\
        }\
      ],\
      "environment": [\
        {\
          "name": "AWS_REGION",\
          "value": "${AWS_REGION}"\
        }\
      ],\
      "secrets": [\
        {\
          "name": "LOGFIRE_TOKEN",\
          "valueFrom": "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:logfire-token-XXXXXX"\
        }\
      ],\
      "logConfiguration": {\
        "logDriver": "awslogs",\
        "options": {\
          "awslogs-group": "/ecs/logfire-metrics-collector",\
          "awslogs-region": "${AWS_REGION}",\
          "awslogs-stream-prefix": "ecs"\
        }\
      }\
    }\
  ]
}
EOF
```

**Important**: Replace `XXXXXX` in the secret ARN with the actual suffix generated by AWS Secrets Manager. You can get the complete ARN by running:

```
aws secretsmanager describe-secret --secret-id logfire-token --query 'ARN' --output text
```

**Note**: This task definition uses a single IAM role (`LogfireMetricsCollectorRole`) for both execution and task permissions, which simplifies the setup while providing all necessary permissions:

- **ECS Task Execution**: Pull images from ECR, access secrets, write logs
- **CloudWatch Access**: Collect metrics from AWS CloudWatch APIs
- **Secrets Access**: Retrieve Logfire token from AWS Secrets Manager

The next step will show you how to create the ECR repository and build the image.

#### 3.5.5 Create ECR Repository and Build Container Image

Create an ECR repository and build your container image with the OpenTelemetry Collector configuration:

```
# Create an ECR repository
aws ecr create-repository \
    --repository-name logfire-metrics-collector \
    --region ${AWS_REGION}

# Get the ECR repository URI
export ECR_REPOSITORY_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/logfire-metrics-collector"

# Create the project structure
mkdir -p logfire-otel-collector
cd logfire-otel-collector

# Create the OpenTelemetry Collector configuration (same as section 3.3)
cat > aws-metrics-collector.yaml << 'EOF'
receivers:
  awsecscontainermetrics:
    collection_interval: 60s

exporters:
  debug:
  otlphttp:
    endpoint: "https://logfire-us.pydantic.dev"
    headers:
      Authorization: "Bearer ${env:LOGFIRE_TOKEN}"

extensions:
  health_check:
    endpoint: "0.0.0.0:13133"

service:
  pipelines:
    metrics:
      receivers: [awsecscontainermetrics]
      exporters: [otlphttp, debug]
  extensions: [health_check]
EOF

# Create the Dockerfile
cat > Dockerfile << 'EOF'
FROM otel/opentelemetry-collector-contrib:0.128.0
COPY aws-metrics-collector.yaml /etc/otelcol-contrib/config.yaml
EOF

# Build and push the container image
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY_URI}
docker build --platform linux/amd64 -t logfire-metrics-collector .
docker tag logfire-metrics-collector:latest ${ECR_REPOSITORY_URI}:latest
docker push ${ECR_REPOSITORY_URI}:latest
```

#### 3.5.6 Deploy to ECS

**Prerequisites**: Before deploying, ensure you have the following AWS infrastructure:

- **ECS Cluster**: [Creating an Amazon ECS cluster](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/create-cluster.html)
- **VPC and Subnets**: Use existing VPC/subnets or [create new ones](https://docs.aws.amazon.com/vpc/latest/userguide/create-vpc.html)

For this example, we'll use the default VPC and public subnets for simplicity:

```
# Get your default VPC and subnets
export VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
export SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0:2].SubnetId' --output text | tr '\t' ',')

# Create ECS cluster
aws ecs create-cluster --cluster-name logfire-metrics-cluster

# Register the task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create an ECS service
aws ecs create-service \
    --cluster logfire-metrics-cluster \
    --service-name logfire-metrics-collector \
    --task-definition logfire-metrics-collector:1 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS}],assignPublicIp=ENABLED}"
```

**Important Security Considerations:**

⚠️ **This example uses public subnets for simplicity**. For production deployments, you should:

- **Use private subnets** with NAT Gateway or VPC endpoints for outbound internet access
- **Disable public IP assignment** (`assignPublicIp=DISABLED`)
- **Create restrictive security groups** that only allow outbound HTTPS (port 443)
- **Use VPC endpoints** for AWS services (ECR, Secrets Manager, CloudWatch) to avoid internet routing

For production networking setup, see:

- [VPC and subnets](https://docs.aws.amazon.com/vpc/latest/userguide/create-vpc.html)
- [NAT Gateway](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html)
- [VPC endpoints](https://docs.aws.amazon.com/vpc/latest/privatelink/vpc-endpoints.html)

Once the deployment is complete, you should be able to run the following query in Logfire to verify metrics are being received:

```
-- For ECS container metrics
SELECT metric_name, count(*) AS metric_count
FROM metrics
WHERE metric_name IN ('ecs.task.memory.utilized', 'ecs.task.cpu.utilized', 'ecs.task.network.rate.rx', 'ecs.task.network.rate.tx')
GROUP BY metric_name;
```

#### 3.5.7 Cost Considerations

**ECS Container Metrics**: The ECS container metrics receiver collects data directly from the ECS task metadata endpoint at **no additional cost**. This is much more cost-effective than using CloudWatch APIs.

**CloudWatch Metrics**: If you use the AWS Distro for OpenTelemetry (ADOT) to collect CloudWatch metrics, be aware that the `GetMetricData` API is **not included in the AWS free tier**. Monitor your CloudWatch API usage and costs, especially when collecting metrics at high frequencies or from many resources.

---

## Logfire MCP Server

An [MCP (Model Context Protocol) server](https://modelcontextprotocol.io/introduction) that provides
access to OpenTelemetry traces and metrics through Logfire. This server enables LLMs to query your
application's telemetry data, analyze distributed traces, and perform custom queries using
**Logfire**'s OpenTelemetry-native API.

Cursor with Logfire MCP - YouTube

[Photo image of The FastAPI Expert](https://www.youtube.com/channel/UC91TdNbobUqT3d2CHcTkx8A?embeds_referring_euri=https%3A%2F%2Flogfire.pydantic.dev%2F)

The FastAPI Expert

2.25K subscribers

[Cursor with Logfire MCP](https://www.youtube.com/watch?v=z56NOvrtG74)

The FastAPI Expert

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=z56NOvrtG74&embeds_referring_euri=https%3A%2F%2Flogfire.pydantic.dev%2F)

0:00

0:00 / 4:04

•Live

•

You can check the [Logfire MCP server](https://github.com/pydantic/logfire-mcp) repository
for more information.

## Installation

The MCP server is a CLI tool that you can run from the command line.

You'll need a read token to use the MCP server. See
[Create Read Token](https://logfire.pydantic.dev/docs/how-to-guides/query-api/#how-to-create-a-read-token) for more information.

You can then start the MCP server with the following command:

```
LOGFIRE_READ_TOKEN=<your-token> uvx logfire-mcp@latest
```

Note

The `uvx` command will download the PyPI package [`logfire-mcp`](https://pypi.org/project/logfire-mcp/),
and run the `logfire-mcp` command.

### Configuration

The way to configure the MCP server depends on the software you're using.

Note

If you are in the EU region, you need to set the `LOGFIRE_BASE_URL` environment variable to `https://api-eu.pydantic.dev`. You can also use the `--base-url` flag to set the base URL.

#### Cursor

[Cursor](https://www.cursor.com/) is a popular IDE that supports MCP servers. You can configure
it by creating a `.cursor/mcp.json` file in your project root:

```
{
  "mcpServers": {
    "logfire": {
      "command": "uvx",
      "args": ["logfire-mcp", "--read-token=YOUR-TOKEN"],
    }
  }
}
```

Note

You need to pass the token via the `--read-token` flag, because Cursor doesn't
support the `env` field in the MCP configuration.

For more detailed information, you can check the
[Cursor documentation](https://docs.cursor.com/context/model-context-protocol).

#### Claude Desktop

[Claude Desktop](https://claude.ai/download) is a desktop application for the popular
LLM Claude.

You can configure it to use the MCP server by adding the following configuration to the
`~/claude_desktop_config.json` file:

```
{
  "mcpServers": {
    "logfire": {
      "command": "uvx",
      "args": [\
        "logfire-mcp",\
      ],
      "env": {
        "LOGFIRE_READ_TOKEN": "your_token"
      }
    }
  }
}
```

Check out the [MCP quickstart](https://modelcontextprotocol.io/quickstart/user)
for more information.

#### Claude Code

[Claude Code](https://claude.ai/code) is a coding tool that is used via CLI.

You can run the following command to add the Logfire MCP server to your Claude Code:

```
claude mcp add logfire -e LOGFIRE_READ_TOKEN="your-token" -- uvx logfire-mcp@latest
```

#### Cline

[Cline](https://docs.cline.bot/) is a popular chatbot platform that supports MCP servers.

You can configure it to use the MCP server by adding the following configuration to the
`cline_mcp_settings.json` file:

```
{
  "mcpServers": {
    "logfire": {
      "command": "uvx",
      "args": [\
        "logfire-mcp",\
      ],
      "env": {
        "LOGFIRE_READ_TOKEN": "your_token"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### Tools

There are four tools available in the MCP server:

1. `find_exceptions(age: int)` \- Get exception counts from traces grouped by file.

Required arguments:
   - `age`: Number of minutes to look back (e.g., 30 for last 30 minutes, max 7 days)
2. `find_exceptions_in_file(filepath: str, age: int)` \- Get detailed trace information about exceptions in a specific file.

Required arguments:
   - `filepath`: Path to the file to analyze
   - `age`: Number of minutes to look back (max 7 days)
3. `arbitrary_query(query: str, age: int)` \- Run custom SQL queries on your OpenTelemetry traces and metrics.

Required arguments:
   - `query`: SQL query to execute
   - `age`: Number of minutes to look back (max 7 days)
4. `get_logfire_records_schema()` \- Get the OpenTelemetry schema to help with custom queries.

---

## Setup Slack Alerts

**Logfire** allows you to send alerts via **Slack** based upon the configured alert criteria.

## Creating a Slack Incoming Webhook

**Logfire** uses **Slack's** Incoming Webhooks feature to send alerts.

The [Incoming Webhooks Slack docs](https://api.slack.com/messaging/webhooks) has all the details on setting up and using incoming webhooks.

For brevity, here's a list of steps you will need to perform:

1. In your Slack Workspace, create or identify a channel where you want to send Logfire alerts.
2. Create a new Slack app (or use an existing one) by navigating to [https://api.slack.com/apps/new](https://api.slack.com/apps/new). Give this a meaningful name such as "Logfire Alerts" or similar so you can identify it later.
3. In the [Apps Management Dashboard](https://api.slack.com/apps), Underneath the **Features** heading on the side bar, select **Incoming Webhooks**
4. Click on the **Add New Webhook** button. This will guide you to a page where you select the channel you want to send alerts to.
5. Click the **Allow** button. You will be redirected back to the **Incoming Webhooks** page, and in the list, you will see your new Webhook URL. This will be a URL that looks similar to something like this:

```
https://hooks.slack.com/services/...
```

6. Copy that somewhere, and save it for the next step

## Creating an Alert

There are a few ways to create an alert. You can:

- Follow our [Detect Service is Down](https://logfire.pydantic.dev/docs/how-to-guides/detect-service-is-down/) guide
- Have a look at the [alerts documentation](https://logfire.pydantic.dev/docs/guides/web-ui/alerts/).

### Define alert

We'll create an alert that will let us know if any HTTP request takes longer than a second to execute.

- Login to **Logfire** and [navigate to your project](https://logfire-us.pydantic.dev/-/redirect/latest-project)
- Click on **Alerts** in the top navigation bar
- Select the **New Alert** button in the top right
- Let's give this Alert a name of **Slow Requests**
- For the query, we'll group results by the http path and duration. We want to include the **max** duration in a given time frame. We also want to filter out any traces that aren't http requests, and order by the max duration, so we can see which routes are the slowest. This query looks like:

```
SELECT
      max(duration),
      attributes->>'http.route'
FROM
      records
WHERE
      duration > 1
      AND attributes->>'http.route' IS NOT NULL
GROUP BY
      attributes->>'http.route'
ORDER BY
    max(duration) desc
```

- Click **Preview query results** and make sure you get some results back. If your service is lightning fast, firstly congratulations! Secondly try adjust the duration cutoff to something smaller, like `duration > 0.1` (i.e, any requests taking longer than 100ms).

[![](https://logfire.pydantic.dev/docs/images/guide/browser-alerts-create-alert.png)](https://logfire.pydantic.dev/docs/images/guide/browser-alerts-create-alert.png)

- You can adjust when alerts are sent based upon the alert parameters. With this style of alert, we just want to know if anything within the last 5 minutes has been slow. So we can use the following options:

  - **Execute the query**: every 5 minutes
  - **Include rows from**: the last 5 minutes
  - **Notify me when**: the query has any results

[![](https://logfire.pydantic.dev/docs/images/guide/browser-alerts-parameters.png)](https://logfire.pydantic.dev/docs/images/guide/browser-alerts-parameters.png)

### Send Alert to a Slack Channel

Our alert is almost done, let's send it to a slack channel.

For this, you will need the [Webhook URL](https://logfire.pydantic.dev/docs/how-to-guides/setup-slack-alerts/#creating-a-slack-incoming-webhook) you created & copied from the Slack [Apps Management Dashboard](https://api.slack.com/apps).

Let's set up a channel, then test that alerts can be sent with the URL:

- Select **New channel** to open the New Channel dialog
- Put in a name such as **Logfire Alerts**. This does need to be the name of your Slack channel
- Select **Slack** as the format
- Paste in your Webhook URL from the Slack \[Apps Management Dashboard\] (https://api.slack.com/apps)
- Click on **Send a test alert** and check that you can see the alert in Slack.
- Click **Create Channel** to create the channel and close the dialog
- Click the checkbox next to your new channel to select it

[![](https://logfire.pydantic.dev/docs/images/guide/browser-alerts-create-channel.png)](https://logfire.pydantic.dev/docs/images/guide/browser-alerts-create-channel.png)

Once your Slack channel is connected, click **Create alert** to save all your changes. Your alert is now live!

You will now receive notifications within your slack channel when the alert is triggered!

---