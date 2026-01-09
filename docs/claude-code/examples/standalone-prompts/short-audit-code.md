**Codebase Architecture Audit**

Scan every source file. No sampling.

**Focus Areas:**

**1. Size Violations**
- List ALL files over 500 lines (with line counts)
- List ALL functions over 50 lines
- For each: is the size justified or should it be split?

**2. God Modules**
- Files with 5+ unrelated responsibilities
- Utility files that became junk drawers
- Modules with 10+ exports (unfocused)
- "Catch-all" shared folders containing domain-specific code

**3. Separation of Concerns**

*Frontend:*
- Components mixing business logic + data fetching + presentation
- Components calling APIs directly (no data layer)
- Business logic in components instead of hooks/services

*Backend:*
- Route handlers containing business logic
- Services directly accessing DB (no repository layer)
- Domain objects knowing about HTTP/serialization

*Cross-cutting:*
- Cross-domain imports (list source → target)
- Circular dependencies (list all cycles)

**4. DRY Violations**
- Copy-pasted logic (2+ instances) — include locations and line counts
- Near-duplicate functions varying by 1-2 lines
- Repeated patterns that should be abstracted

**Output Format:**

**Census:**
```
Files > 500 lines: [list all with counts]
Functions > 50 lines: [list all]
Circular dependencies: [list cycles]
Cross-domain imports: [list all]
```

**Findings:**
```
#N | File:line | Category | Severity
Issue: [what's wrong]
Fix: [concrete refactor approach]
```

**Prioritized Roadmap:**
```
Critical: [blocks development]
High: [compounding debt]
Medium: [velocity improvement]
```

**Summary:**
- Total findings by category
- Top 5 worst files
- Top 5 highest-impact refactors