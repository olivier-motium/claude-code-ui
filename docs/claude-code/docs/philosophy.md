## 1. Core Philosophy

- **Clarity Over Cleverness**: Write explicit, boring, obvious code. Optimize for human review and AI modification, not keystrokes saved.
- **Locality Over Abstraction**: Prefer self-contained modules over deep inheritance or distant shared code. Duplication is acceptable when it preserves locality and independence.
- **Compose Small Units**: Build features from small, single-purpose modules with clear interfaces. Each module should be safely rewritable in isolation.
- **Stateless by Default**: Keep functions pure where possible; pass state explicitly. Side effects (DB, HTTP, storage) live at the edges behind clear boundaries.
- **Fail Fast & Loud**: Surface errors to central handlers; no silent catches. Log enough context (request ID, user, operation) for fast triage.
- **Tests as Specification**: Tests define correct behavior. Code is disposable; tests and interfaces are the source of truth.

## 2. Tooling

- Python 3.11+, `uv` for all environment/package management (never `pip`).
- Formatting: Black (88 cols) + Ruff. Run on save; no style debates.
- Type checking: strict mode; CI must pass.

## 3. Code Style

- **Type hints everywhere**: Prefer `list[str]`, `dict[str, T]` over `List`, `Dict`. Avoid `Any`; if unavoidable, use `typing.cast` with a justifying comment.
- **Naming**: Python files `snake_case`, TS files `kebab-case`. Classes/enums `PascalCase`, constants `UPPER_SNAKE_CASE`.
- **Imports**: Absolute only. Group: stdlib → third-party → local.
- **Data contracts**: Pydantic models for request/response validation and API boundaries. No business logic in models.