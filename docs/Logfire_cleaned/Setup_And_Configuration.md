# Setup And Configuration

## Contents

- [How to Convert a Personal Account to an Organization](#how-to-convert-a-personal-account-to-an-organization)
- [Create Write Tokens](#create-write-tokens)
- [Combine Multiple Configurations](#combine-multiple-configurations)
- [Use different environments](#use-different-environments)

---

## How to Convert a Personal Account to an Organization

Logfire allows you to convert your personal account into an organization, making it easier to collaborate with a team and manage projects at scale. This is also handy for users upgrading to Pro, who might want to move projects under a corporate organization name.

This guide walks you through the conversion process step by step.

* * *

## 1\. Open Account Settings

Navigate to your account settings. On the page, you'll see an option to **Convert to organization**.

[![Account settings with Convert to org option](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-settings.png)](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-settings.png)

* * *

## 2\. Start the Conversion

Click **Convert to org**. A modal will appear, outlining the main points of the conversion:

- All existing **projects, members, alerts, dashboards, and settings** will be moved to the new organization.
- **Write tokens** will continue to work; you do not need to change any ingest URLs.
- You'll define your new organization's **handle** and **display name**.
- You can optionally edit the username and display name for your new personal account.

[![Convert to org modal with main points](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-modal-main-points.png)](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-modal-main-points.png)

Click **Acknowledge & continue** to proceed.

* * *

## 3\. Set Up Your Organization

In the next modal, you can:

- Upload an **organization avatar**.
- Specify the **organization handle** (used in URLs).
- Set the **organization display name**.

On the right, you'll see a summary of the migration:

- All your projects and members will be moved to the new organization.
- The project URLs will change from:
`https://logfire-eu.pydantic.dev/your-username/project-name`
to
`https://logfire-eu.pydantic.dev/your-org-handle/project-name`.

[![Set up new organization modal](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-setup-org.png)](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-setup-org.png)

* * *

## 4\. Confirm New Personal Account

After setting up the organization, you'll be prompted to create a new (empty) personal account with the same name as before. You can confirm and complete the conversion, or go back if you wish to make changes.

[![Confirm new personal account modal](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-new-personal.png)](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-new-personal.png)

* * *

## 5\. Complete the Conversion

Click **Confirm & convert**. The conversion process will complete, and you'll be redirected to your new organization's projects page.

[![Organization projects page after conversion](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-org-projects.png)](https://logfire.pydantic.dev/docs/images/guide/convert-to-org-org-projects.png)

* * *

## Summary

- All your data, projects, and settings are preserved during the migration.
- Only the URL changes to reflect the new organization handle.
- Your new personal account will be empty, ready for individual use if needed.

* * *

**See also:** [Organization Structure Reference](https://logfire.pydantic.dev/docs/guides/web-ui/organizations-and-projects/)

---

## Create Write Tokens

To send data to **Logfire**, you need to create a write token.
A write token is a unique identifier that allows you to send data to a specific **Logfire** project.
If you set up Logfire according to the [getting started guide](https://logfire.pydantic.dev/docs/), you already have a write token locally tied to the project you created.
But if you want to configure other computers to write to that project, for example in a deployed application, you need to create a new write token.

You can create a write token by following these steps:

1. Open the **Logfire** web interface at [logfire.pydantic.dev](https://logfire.pydantic.dev/).
2. Select your project from the **Projects** section on the left hand side of the page.
3. Click on the ⚙️ **Settings** tab in the top right corner of the page.
4. Select the **Write tokens** tab from the left hand menu.
5. Click on the **New write token** button.

After creating the write token, you'll see a dialog with the token value.
**Copy this value and store it securely, it will not be shown again**.

Now you can use this write token to send data to your **Logfire** project from any computer or application.

We recommend you inject your write token via environment variables in your deployed application.
Set the token as the value for the environment variable `LOGFIRE_TOKEN` and logfire will automatically use it to send data to your project.

## Setting `send_to_logfire='if-token-present'`

You may want to not send data to logfire during local development, but still have the option to send it in production without changing your code.
To do this we provide the parameter `send_to_logfire='if-token-present'` in the `logfire.configure()` function.
If you set it to `'if-token-present'`, logfire will only send data to logfire if a write token is present in the environment variable `LOGFIRE_TOKEN` or there is a token saved locally.
If you run tests in CI no data will be sent.

You can also set the environment variable `LOGFIRE_SEND_TO_LOGFIRE` to configure this option.
For example, you can set it to `LOGFIRE_SEND_TO_LOGFIRE=true` in your deployed application and `LOGFIRE_SEND_TO_LOGFIRE=false` in your tests setup.

---

## Combine Multiple Configurations

Sometimes you need different Logfire configurations for different parts of your application. You can do this with [`logfire.configure(local=True, ...)`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.configure(local)).

For example, here's how to disable console logging for database operations while keeping it enabled for other parts:

```
import logfire

# Global configuration is the default and should generally only be done once:
logfire.configure()

# Locally configured instance without console logging
no_console_logfire = logfire.configure(local=True, console=False)

# Simple demonstration:
logfire.info('This uses the global config and will appear in the console')
no_console_logfire.info('This uses the local config and will NOT appear in the console')

# Calling functions on the `logfire` module will use the global configuration
# This will send spans about HTTP requests to both Logfire and the console
logfire.instrument_httpx()

# This will send spans about DB queries to Logfire but not to the console
no_console_logfire.instrument_psycopg()
```

---

## Use different environments

As developers, we find ourselves working on different environments for a project: local,
production, sometimes staging, and depending on your company deployment strategy... You can have even more! 😅

With **Logfire** you can distinguish which environment you are sending data to.
You just need to set the `environment` parameter in [`logfire.configure()`](https://logfire.pydantic.dev/docs/reference/api/logfire/#logfire.configure(environment)).

main.py

```
import logfire

logfire.configure(environment='local')  Usually you would retrieve the environment information from an environment variable.
```

Under the hood, this sets the OpenTelemetry resource attribute [`deployment.environment.name`](https://opentelemetry.io/docs/specs/semconv/resource/deployment-environment/).
Note that you can also set this via the `LOGFIRE_ENVIRONMENT` environment variable.

#### Setting environments in other languages

If you are using languages other than Python, you can set the environment like this:
`OTEL_RESOURCE_ATTRIBUTES="deployment.environment.name=prod"`

* * *

Once set, you will see your environment in the Logfire UI `all envs` dropdown,
which is present on the [Live View](https://logfire.pydantic.dev/docs/guides/web-ui/live/), [Dashboards](https://logfire.pydantic.dev/docs/guides/web-ui/dashboards/)
and [Explore](https://logfire.pydantic.dev/docs/guides/web-ui/explore/) pages:

[![Environments](https://logfire.pydantic.dev/docs/images/guide/environments.png)](https://logfire.pydantic.dev/docs/images/guide/environments.png)

Info

When using an environment for the first time, it may take a **few minutes** for the environment to appear in the UI.

Note that by default there are system generated environments:

- `all envs`: Searches will include everything, including spans that had no environment set.
- `not specified`: Searches will _only_ include spans that had no environment set.

So `not specified` is a subset of `all envs`.

Any environments you create via the SDK will appear below the system generated environments.
When you select an environment, all subsequent queries (e.g. on live view, dashboards or explore)
will filter by that environment.

## Can I create an environment in the UI?

No, you cannot create or delete set environments via the UI, instead use the SDK.

## How do I delete an environment?

Once an environment has been configured and received by logfire, technically it’s available for
the length of the data retention period while that environment exists in the data.
You can however add new ones, and change the configuration of which data is assigned to which
environment name.

## Should I use environments or projects?

Environments are more lightweight than projects. Projects give you the ability to assign specific
user groups and permissions levels (see this [organization and projects](https://logfire.pydantic.dev/docs/guides/web-ui/organizations-and-projects/) documentation
for details). So if you need to allow different team members to view dev vs. prod traces, then projects would be a better fit.

---