## Terraform Repository Documentation Audit (Markdown-Only Audit, Code Cross-Referenced)

Conduct an exhaustive audit of **ALL Markdown documentation files** in this **Terraform / Infrastructure-as-Code repository**. **Leave no stone unturned.** Ultrathink.

### Scope: Read Every Doc, Verify Against the Real IaC + Pipelines

* Scan **EVERY `*.md` file** in the repository (root-level, nested module READMEs, `/docs`, `/examples`, `/scripts`, `.github`, etc.).
* **Read each document in full** (no skimming or sampling).
* Cross-reference docs against the *actual repository contents*, including (as applicable):

  * Terraform: `*.tf`, `*.tf.json`, `*.tfvars`, `*.auto.tfvars`, `*.tftpl`
  * Tooling/wrappers: `terragrunt.hcl`, `*.hcl`, Atlantis configs, etc.
  * CI/CD & GitHub: `.github/workflows/*.yml`, composite actions, scripts used by workflows
  * Repo automation & quality gates: `pre-commit-config.yaml`, `Makefile`, `Taskfile.yml`, lint configs
  * **SQL scripts**: `*.sql` (and any migration folders), plus shell/PowerShell/Python helpers that apply them
  * Secrets/config: `*.yaml`, `*.yml`, `*.json` used for environment config, policy, or pipeline configuration
* The audit output is about **Markdown documentation quality and completeness**, but you must **validate claims by inspecting the code and pipeline configs**. Do not assume; verify.

---

## Your Priorities (in order)

### 1) Structure & Organization Audit (Exhaustive)

Map the complete documentation landscape and assess whether it matches Terraform-repo expectations:

**Inventory & entry points**

* Catalog **every `*.md` file** with:

  * full path
  * line count
  * apparent purpose (module docs, environment runbook, contributor guide, etc.)
* Verify a clear entry point exists (typically `README.md` at repo root):

  * Does it explain what infrastructure this repo manages (cloud/provider, accounts/subscriptions/projects, high-level scope)?
  * Does it clearly explain how this repo is meant to be used (module registry vs live environments vs mono-repo)?
  * Does it point to “next steps” for different audiences (operators, contributors, module consumers)?

**Organization quality**

* Is there a logical structure for Terraform documentation, for example:

  * `/docs/getting-started/` (toolchain, auth, bootstrap)
  * `/docs/environments/` (per-environment apply/runbooks)
  * `/docs/modules/` (how to consume, standards)
  * `/docs/cicd/` (plans/applies, approvals, drift detection)
  * `/docs/operations/` (state recovery, import, troubleshooting)
  * `/docs/security/` (secrets, IAM/OIDC, policies)
  * `/docs/sql/` (what SQL exists, when/how to run safely)
  * `/docs/adr/` (architecture decision records)
* Are documents at the right granularity?

  * Avoid one mega “everything” doc.
  * Avoid fragmentation into dozens of tiny orphan docs.
  * Module-level docs should live with modules; environment runbooks should live with environments (or be clearly linked).

**Navigation**

* Is there an index or TOC that provides navigation across **ALL docs**?

  * Root `README.md` and/or `/docs/index.md` should function as the documentation hub.
* Are related topics linked, or do docs exist as isolated islands?
* Check every internal link:

  * Flag broken links
  * Flag links that point to renamed/moved content
  * Flag relative links that don’t work from GitHub’s renderer context (e.g., wrong path depth)

**Terraform-doc conventions**

* If the repo uses `terraform-docs` (or similar), verify:

  * module READMEs consistently include Inputs/Outputs/Requirements/Providers/Resources sections
  * the docs appear generated/updated (or are stale)
  * the generation mechanism is documented (make target, pre-commit hook, CI step)

---

### 2) Gap Analysis (IaC Reality vs Docs) — Comprehensive Coverage

Systematically scan the repository contents and verify documentation exists for **each major IaC concept** and **each operational workflow**.

For each major area, explicitly state either:

* **“Documentation exists at: X”**, or
* **“NO DOCUMENTATION FOUND.”**

#### A. Repository-level essentials (must be documented)

Verify docs exist for:

* **Purpose & scope**: what infrastructure, what boundaries (what’s *in* scope vs *out* of scope)
* **Cloud/provider targets**: AWS/Azure/GCP/Kubernetes/GitHub/etc. and account/subscription/project layout
* **Toolchain requirements**:

  * Terraform version constraints (and how enforced)
  * provider versions / lockfile policy (`.terraform.lock.hcl`)
  * wrappers (Terragrunt), if present
  * required CLIs (awscli/az/gcloud/kubectl/vault/sops/etc.), if applicable
* **Authentication model**:

  * local dev auth method(s)
  * CI auth method(s) (e.g., OIDC to cloud provider)
  * role assumptions / identities / permissions boundaries
* **Remote state**:

  * backend type and configuration strategy
  * state isolation model (per-env, per-stack, per-module)
  * locking behavior (DynamoDB, etc.) and failure recovery guidance
* **Environment strategy**:

  * directory-per-environment vs workspaces vs terragrunt stacks
  * promotion model (dev → stage → prod) and how changes flow
* **Secrets & sensitive values**:

  * what must never be committed
  * how secrets are stored, injected, rotated
  * redaction practices and logging considerations

#### B. Terraform code structure (must be discoverable)

Identify the actual structure (don’t assume naming):

* Are there module directories (e.g., `modules/`, `terraform/modules/`)?
* Are there live stacks/environments (e.g., `envs/`, `live/`, `stacks/`, `accounts/`, `regions/`)?
* Are there examples (e.g., `examples/`)?
* Are there policy checks or guardrails (OPA/Sentinel/custom scripts)?

Verify documentation exists for:

* **Each top-level stack/environment** (what it creates, how to run it safely)
* **Each reusable module** (purpose, interface, examples)
* **Shared conventions** used repeatedly:

  * naming conventions
  * tagging/labeling standards
  * resource ownership boundaries
  * module versioning approach (git tags, registry publishing, monorepo refs)
  * provider aliasing patterns (multi-account, multi-region)
* **Non-obvious patterns**:

  * `for_each` patterns requiring stable keys
  * lifecycle ignores and why
  * data sources vs resources rationale
  * dependency ordering assumptions (`depends_on`, implicit dependencies)
  * import strategy for existing infra

#### C. CI/CD and GitHub workflows (must be documented)

Inspect `.github/workflows/*` and related scripts; verify docs exist for:

* PR checks: `fmt`, `validate`, `tflint`, security scans (tfsec/checkov/terrascan), unit/integration tests (if any)
* Plan/apply workflow:

  * how plans are generated, where they are stored/displayed (PR comment, artifact, etc.)
  * what gates apply (approvals, environment protection rules, CODEOWNERS)
  * who can apply, and under what conditions
  * concurrency/locking and how collisions are prevented
* Promotion model across environments
* Drift detection:

  * whether scheduled plans exist
  * how drift is reported and remediated
* Runner/permissions model:

  * GitHub OIDC setup (if used)
  * secrets usage policy and exposure risk controls
* Release/versioning workflow for modules (if applicable)

#### D. SQL scripts and “Terraform can’t do this” operations (must be documented)

Since the repo contains SQL, verify docs exist for:

* **Why SQL exists here** (what Terraform can’t/shouldn’t do)
* Inventory of SQL scripts (purpose, target system)
* Execution method:

  * manual vs automated
  * how credentials/connectivity are obtained (and secured)
  * required client tooling
* Safety characteristics:

  * idempotency expectations
  * ordering dependencies
  * rollback strategy / reversibility
  * environment restrictions (e.g., “never run on prod without approval”)
* Relationship to Terraform:

  * what Terraform provisions that SQL depends on
  * how sequencing is ensured (if at all)
  * whether SQL changes are treated as migrations (and how tracked)

---

### 3) Redundancy, Contradictions, Staleness (Full Inventory)

Identify ALL documentation debt across Terraform docs:

* Duplicated guidance (e.g., “how to run terraform” repeated in multiple places inconsistently)
* Contradictory instructions (quote conflicts), especially around:

  * backend/state configuration
  * workspace vs directory flows
  * CI apply permissions and approvals
  * secrets handling
* Orphaned docs not linked from anywhere (especially old runbooks)
* Stale sections referencing:

  * removed modules/environments
  * deprecated commands (`terraform destroy` guidance, legacy `-var-file` paths, old workflow filenames)
  * renamed directories (e.g., `live/` → `envs/`)
* Docs that are too long and should be split
* Placeholder/empty docs that were never filled in

---

### 4) Filter Noise (Right Abstraction Level for Terraform)

Do **not** replicate low-level code line-by-line.

However, Terraform documentation **must** cover critical *interfaces and workflows*, including:

* module inputs/outputs (at least conceptually; and if the repo standard is `terraform-docs`, ensure it’s present and accurate)
* environment apply/run steps with correct ordering (`init`, backend config, workspace selection if used, plan/apply, etc.)
* operational invariants and risk areas (state, drift, credentials, destructive changes, imports)

---

### 5) Content Enrichment Requirements (For Every Gap)

For EVERY documentation gap you identify, draft Markdown content that includes:

* **The “Why”**: business intent, security/compliance rationale, architectural reasoning
* **The “How”**:

  * high-level flow of how infra is provisioned/updated
  * how state is managed and isolated
  * how changes move from PR → plan → apply → promotion
* **The “Gotchas”**:

  * state/locking pitfalls
  * drift scenarios
  * provider auth edge cases
  * ordering/dependency traps
  * destructive operations safeguards
  * SQL safety pitfalls (idempotency, ordering, credential exposure)

Also include **exact commands** where appropriate (examples):

* local dev: init/plan/apply steps (and workspace/env selection if used)
* CI: what triggers runs, what artifacts exist, how to re-run safely
* SQL: how to run scripts safely and how to validate results

---

### 6) Audience & Discoverability (Complete Assessment)

Assess whether docs serve ALL intended readers for a Terraform repo:

* **New contributor**:

  * can they set up the toolchain, auth, and run a plan in a sandbox?
  * can they understand repo layout (modules vs environments vs scripts)?
* **Module consumer** (if modules are intended to be reused):

  * can they find module docs and usage examples quickly?
  * is versioning/pinning guidance provided?
* **Operator / SRE / Platform engineer**:

  * can they deploy safely?
  * can they troubleshoot a failed apply?
  * do runbooks exist for state issues, imports, drift, provider outages?
* **Security/compliance reviewer**:

  * is least privilege, secrets handling, and audit trail documented?
  * are policy checks explained (and enforced)?

Check for missing doc types common in Terraform repos:

* Quickstart / onboarding
* Environment runbooks (per env)
* Module catalog / module standards
* CI/CD pipeline documentation
* Troubleshooting / runbooks (state lock, import, drift, apply failure)
* ADRs (architecture decisions)
* Security guide (secrets, identity, approvals)
* Changelog / release notes (especially for modules)
* Contributing guide + PR standards (including formatting/linting expectations)

---

## Output Format

### Part A — Complete Documentation Inventory

```
Files Found: <total count>
<full tree view of ALL .md files in repo with line counts>

Broken Internal Links:
- <source file> → <broken link target> (line X)
- ...

Orphaned Files (not linked from anywhere):
- <file path>
- ...

Terraform/IaC Landscape Summary (derived from code scan):
- Environments/Stacks found: <count> (list directories)
- Modules found: <count> (list directories)
- CI workflows found: <count> (list workflow files)
- SQL scripts found: <count> (list directories/patterns)
```

### Part B — Structural Recommendations

```
Current Structure:
<tree view of existing /docs and root-level .md files>

Proposed Structure:
  /docs
  ├── index.md                      <- Central navigation hub
  ├── getting-started/
  │   ├── prerequisites.md
  │   ├── authentication.md
  │   ├── repo-layout.md
  │   └── quickstart-plan.md
  ├── architecture/
  │   ├── overview.md
  │   ├── state-and-backends.md
  │   ├── environment-strategy.md
  │   └── module-design-standards.md
  ├── environments/
  │   ├── <env-or-stack-1>.md
  │   └── <env-or-stack-2>.md
  ├── modules/
  │   ├── index.md                  <- Module catalog & usage standards
  │   └── <module-name>.md (optional if module README is canonical)
  ├── cicd/
  │   ├── workflows.md
  │   ├── plan-apply-gates.md
  │   └── drift-detection.md
  ├── operations/
  │   ├── troubleshooting.md
  │   ├── state-lock-recovery.md
  │   ├── imports-and-moves.md
  │   └── disaster-recovery.md
  ├── security/
  │   ├── secrets-handling.md
  │   ├── identity-and-oidc.md
  │   └── compliance-and-scanning.md
  ├── sql/
  │   ├── overview.md
  │   ├── execution-runbook.md
  │   └── scripts-inventory.md
  └── adr/
      ├── 0001-<title>.md
      └── ...

Reorganization Actions:
- <MOVE: old/path.md → new/path.md — reason>
- <MERGE: file1.md + file2.md → combined.md — reason>
- <SPLIT: monolith.md → section1.md, section2.md — reason>
- <DELETE: orphaned-doc.md — reason>
- <CREATE: missing-doc.md — purpose and what it should cover>
```

### Part C — Content Findings (Every Issue)

Report EVERY finding (not just the most important):

```
Finding #<N>
Target File: <path to existing .md file> or <suggestion for new .md file>
Category: <Gap | Outdated | Redundant | Contradictory | Misplaced | Missing Links | Incomplete>
Severity: <Critical | Warning | Advisory>

The Discrepancy:
<Detailed explanation of what the Terraform code/CI workflows/SQL scripts do vs. what the docs say or omit.
Include specific file paths like modules/<name>/*.tf, envs/<env>/main.tf, .github/workflows/<workflow>.yml, scripts/*.sh, sql/*.sql, etc.>

Suggested Update (Markdown):

## <Section Title>
<Proposed documentation text that resolves the discrepancy, including Why/How/Gotchas and correct commands where relevant>
```

### Part D — Proposed Index

Generate `/docs/index.md` (or update `README.md`) as the documentation entry point containing:

* Brief repo description (what infra it manages and what it doesn’t)
* Links to ALL doc sections with one-line descriptions
* “Start here” guidance for:

  * new contributors (setup → auth → plan)
  * module consumers (find module → usage → version pinning)
  * operators (apply flow → approvals → troubleshooting)
  * security/compliance (secrets → identity → scanning)
* Quick reference:

  * most common commands
  * links to CI workflows docs
  * links to environment runbooks
  * links to SQL runbook (if SQL exists)

### Part E — Summary Statistics

```
Total .md files: X
Modules discovered: X
Environments/stacks discovered: X
CI workflows discovered: X
SQL scripts discovered: X

Total gaps found: X
Total outdated sections: X
Total redundancies: X
Total contradictions: X
Total broken links: X

Documentation coverage estimate:
- Module coverage: X% (modules with adequate docs / total modules)
- Environment/runbook coverage: X%
- CI/CD documentation coverage: X%
- SQL operational documentation coverage: X%

Priority fixes (ranked by impact):
1. ...
2. ...
3. ...
```

### Completeness Check

Before finalizing the report, verify:

* [ ] Every top-level directory was checked for corresponding docs
* [ ] Every Terraform environment/stack directory was checked for documentation
* [ ] Every Terraform module was checked for module-level docs and usage examples
* [ ] Every CI workflow in `.github/workflows` was checked for documentation
* [ ] Every SQL script directory/pattern was checked for documentation
* [ ] Every `*.md` file was read and cataloged
* [ ] Every internal link was validated
* [ ] Every finding includes a concrete suggested fix (with Markdown text), not just problem description








## Docs we want (generic Terraform repo standard)

1. **`README.md` (root) — entry point**

* What this repo manages, who it’s for, and the **safe “how to change infra” path** (PR → plan → apply).
* Repo map (where stacks/environments, modules, CI, and scripts live).
* Links to the docs below.

2. **`docs/architecture.md` — 1‑page architecture**

* High-level topology/components, boundaries, and the “why”.
* State/backends strategy (high level), environment model, key gotchas.

3. **`docs/cicd.md` — CI/CD behavior**

* What each workflow does, triggers, approvals/gates, where plan/apply output appears, auth method.

4. **Runbook per deployable stack/environment — placed next to the stack**

* Each deployable directory has **one** `README.md` (or `RUNBOOK.md`) describing exactly how to plan/apply safely and troubleshoot.

5. **Module README per module**

* Each module directory has a `README.md` with purpose, usage example, gotchas/security notes, and a `terraform-docs` generated block.

6. **If SQL exists: `db/README.md` (or `docs/sql.md`) — SQL runbook + inventory**

* Why SQL exists, how to run safely, and an inventory table of scripts (purpose, env, idempotency, verify).

---

# Templates (examples)

## 1) `README.md` (root)

````md
# <Repo Name>

Terraform infrastructure repo for <what it manages>.

## TL;DR (safe path)
1) Open PR → CI runs plan
2) Review plan + approvals
3) Apply via CI (preferred)

## Repo map
- Stacks/environments: `<path>`
- Platform/shared stacks: `<path>`
- Modules: `<path>`
- CI workflows: `.github/workflows/`
- SQL/scripts (if any): `<path>`

## Quickstart (local, if allowed)
**Prereqs:** Terraform <version>, <cloud CLI>, <auth method>
```bash
cd <stack-path>
terraform init
terraform plan -out tfplan
terraform apply tfplan
````

## Docs

* Architecture: `docs/architecture.md`
* CI/CD: `docs/cicd.md`
* Stack runbooks: `<stack directories>/README.md`
* Module docs: `<module directories>/README.md`
* SQL runbook (if applicable): `db/README.md`

````

---

## 2) `docs/architecture.md`
```md
# Architecture

## Overview
What this repo provisions and why (3–6 bullets).

## Building blocks
- Network: <high-level>
- Compute: <high-level>
- Data: <high-level>
- Observability: <high-level>
- Security: <high-level>

## Environment model
How dev/stage/prod (or accounts/subscriptions/projects) are separated.

## State/backends (high level)
Backend type, isolation model (per env/stack), locking expectations.

## Key gotchas
- <replacement risks>
- <ordering dependencies>
- <state/lock pitfalls>
- <permissions/auth pitfalls>
````

---

## 3) Stack/Environment runbook (put next to each deployable stack)

`<stack-dir>/README.md` or `<stack-dir>/RUNBOOK.md`

````md
# Runbook: <Stack Name> (<env if relevant>)

## TL;DR
- Plan: CI workflow `<name>`
- Apply: CI workflow `<name>` (requires: <approvals/gates>)
- Local apply: <allowed/not allowed>

## What this stack manages
- <1–5 bullets>

## Prereqs / auth
- Local auth: <how>
- CI auth: <how>
- Required tools: Terraform <version>, <other CLIs>

## How to plan/apply (CI)
- Trigger: <PR/push/manual>
- Where plan output appears: <PR comment/logs/artifact>
- Apply rules: <who/when>

## How to run locally (only if allowed)
```bash
cd <this directory>
terraform init
terraform plan -out tfplan
terraform apply tfplan
````

## Post-apply steps (SQL/scripts if applicable)

* Runbook: `db/README.md`
* Required steps: <list>

## Troubleshooting / rollback

* State lock: <what to do>
* Common failures: <1–3 bullets>
* Rollback meaning here: <brief>

````

---

## 4) Module README (for every module dir)
`<module-dir>/README.md`
```md
# <module-name>

## TL;DR
Creates <what>. Use when <scenario>.

## Purpose
- <why this module exists / what it standardizes>

## Usage
```hcl
module "<name>" {
  source = "<path|registry|git-ref>"
  # required inputs...
}
````

## Gotchas

* <1–5 bullets>

## Security notes

* Permissions: <what it needs>
* Secrets: <how sourced; never plaintext>

## Inputs/Outputs (generated)

<!-- BEGIN_TF_DOCS -->

<!-- END_TF_DOCS -->

````

---

## 5) `docs/cicd.md`
```md
# CI/CD

| Workflow | Purpose | Triggers | Apply? | Approvals/Gates |
|---|---|---|---|---|
| `<workflow.yml>` | <plan/lint/apply/drift> | <PR/push/schedule/manual> | <Y/N> | <rules> |

## Plan/apply behavior
- Plan location: <PR comment/logs/artifact>
- Apply constraints: <branch/env protections/manual only/etc.>

## CI authentication
- Method: <OIDC / service principal / role assumption>
- Notes: <any important limitations>
````

---

## 6) SQL runbook + inventory (only if SQL exists)

`db/README.md` (or `docs/sql.md`)

````md
# SQL Runbook

## Why SQL exists here
SQL covers changes Terraform can’t/shouldn’t do (e.g., DB roles, grants, schema ops, audit config).

## How to run safely
- Confirm target env before execution.
- Use approved credential source.
- Prod requires: <approval rule>.

## Scripts inventory
| Script | Env | Purpose | Idempotent? | When to run | Verify |
|---|---|---|---|---|---|
| `<path>.sql` | <dev/prod> | <what> | <Y/N> | <timing> | <query/expected> |

## Required SQL header (copy into each script)
```sql
/*
Purpose:
Target (env/system):
Idempotent (Y/N):
Dependencies:
Rollback:
Verification:
*/
````

```
```

