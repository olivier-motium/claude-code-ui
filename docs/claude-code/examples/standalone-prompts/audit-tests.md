Conduct an exhaustive audit of the ENTIRE test suite to ensure tests are functional, current, comprehensive, and trustworthy. ultrathink

**Scope: Leave No Stone Unturned.**
- Run EVERY test in the repository — do not sample or spot-check.
- Map EVERY production file to its corresponding test file(s) — verify coverage exists.
- Read EVERY test file to assess quality, not just pass/fail status.
- Trace EVERY mock/stub to verify it matches current implementation interfaces.
- The goal is a test suite that actually catches regressions, not one that just produces green checkmarks.

---

## PHASE 1: HEALTH CHECK (Do Tests Run?)

**1.1 Execute Full Test Suite:**
Run all tests and capture:
```
pytest --tb=short -v          # Python
npm test -- --verbose         # React/TS (Jest)
npm run test -- --run         # React/TS (Vitest)
```

Record for every test:
- Pass / Fail / Skip / Error
- Execution time (flag tests > 5 seconds)
- Any deprecation warnings

**1.2 Identify All Broken Tests:**
For each failing test, diagnose the root cause:

*Import/Module Errors:*
- Missing imports (file moved/renamed/deleted)
- Circular import issues
- Missing dependencies (package not installed in test env)

*Signature Mismatches:*
- Function parameters changed but test not updated
- Return type changed but assertions not updated
- Class constructor changed but test instantiation not updated

*Environment Issues:*
- Missing environment variables
- Database/fixture setup failures
- Port conflicts or external service dependencies

*Flaky Tests:*
- Tests that pass/fail inconsistently
- Race conditions in async tests
- Time-dependent tests (dates, timeouts)
- Order-dependent tests (pass in isolation, fail in suite or vice versa)

**1.3 Identify All Skipped Tests:**
For every `@pytest.mark.skip`, `@pytest.mark.xfail`, `it.skip()`, `describe.skip()`, `test.todo()`:
- Document the skip reason (if provided)
- Assess if skip is still valid or if underlying issue was fixed
- Flag skips with no reason or stale reasons

---

## PHASE 2: GAP ANALYSIS (Code vs. Tests)

**2.1 Complete Coverage Mapping:**
Create an exhaustive map of production code to tests:

```
Production File                    | Test File(s)              | Coverage Status
-----------------------------------|---------------------------|------------------
src/services/auth.py               | tests/test_auth.py        | Partial (60%)
src/services/billing.py            | <NONE>                    | MISSING
src/components/UserProfile.tsx     | __tests__/UserProfile.test.tsx | Full
src/utils/validators.ts            | <NONE>                    | MISSING
```

For EVERY production file, explicitly confirm: "Test exists at X" or "NO TEST FILE FOUND."

**2.2 Python-Specific Coverage Gaps:**
Scan for untested:

*Public API Surfaces:*
- Functions/classes in `__init__.py` exports
- Functions without leading underscore (public by convention)
- CLI entry points
- API route handlers (FastAPI, Flask, Django views)

*Business Logic:*
- Functions with `if/elif/else` branches (test each branch)
- Functions with `try/except` blocks (test both paths)
- Functions with loops that could be empty, single, or multiple iterations
- State machines or workflow logic
- Calculation/transformation functions

*Data Layer:*
- ORM models: creation, relationships, constraints, cascades
- Database queries: filters, joins, aggregations
- Migrations: up and down paths
- Serializers/deserializers: all fields, edge cases, validation errors

*Integration Points:*
- External API calls (should be mocked but tested)
- File I/O operations
- Queue/message handlers
- Scheduled tasks/cron jobs
- Webhooks

**2.3 React/TypeScript-Specific Coverage Gaps:**
Scan for untested:

*Components:*
- All exported components (check every `.tsx` file with JSX)
- Component render states: loading, error, empty, populated
- Conditional rendering branches
- User interactions: clicks, inputs, form submissions
- Props variations: required, optional, edge cases
- Error boundaries

*Hooks:*
- All custom hooks (`use*.ts` files)
- Hook state transitions
- Hook side effects and cleanup
- Hook error states

*State Management:*
- Redux/Zustand/Recoil: reducers, actions, selectors
- Context providers: value changes, consumer updates
- Async state: loading, success, error transitions

*Utilities:*
- All exported functions from `/utils`, `/helpers`, `/lib`
- Type guards and validators
- Formatters and transformers
- API client functions

*Routes/Pages:*
- All route components
- Route guards/protected routes
- Dynamic route parameters
- 404/error pages

**2.4 Critical Path Identification:**
For every untested file, assess:
- Is this code reachable in production? (Not dead code?)
- What breaks if this code breaks? (Blast radius)
- How often does this code change? (Change frequency = risk)
- Does this code handle money, auth, or PII? (Sensitivity)

---

## PHASE 3: STALENESS DETECTION (Tests vs. Reality)

**3.1 Behavioral Drift:**
Identify tests that pass but test outdated behavior:

*Assertion Staleness:*
- Assertions on fields/properties that no longer exist
- Expected values that no longer match actual behavior
- Status codes or error messages that changed
- Assertions so loose they always pass (`expect(result).toBeTruthy()`)

*Dead Test Code:*
- Test files for deleted production files
- Test cases for removed features
- Describe blocks with no `it()` cases inside
- Tests that are commented out

**3.2 Mock Integrity Audit:**
For EVERY mock/stub/fake in the test suite:

*Python (unittest.mock, pytest-mock):*
```python
# Check: Does this mock signature match the real function?
@patch('module.function')
def test_something(mock_func):
    mock_func.return_value = {...}  # Does return shape match reality?
```

Verify:
- Mocked function/class still exists at that import path
- Mock return values match current return types
- Mock side effects match current behavior
- `spec=` or `autospec=` used to enforce interface matching

*React/TypeScript (Jest, Vitest):*
```typescript
// Check: Does this mock match the real module?
jest.mock('../services/api', () => ({
    fetchUser: jest.fn().mockResolvedValue({ name: 'Test' })
}));
```

Verify:
- Mocked module path is still valid
- Mocked function names still exist in real module
- Mock return shapes match current TypeScript types
- All mocked functions in module are accounted for

**3.3 Snapshot Staleness:**
For ALL snapshot tests (`.snap` files, inline snapshots):

- When was snapshot last updated? (Check git history)
- Was it bulk-updated with `-u` without review?
- Does snapshot contain dynamic values (dates, IDs) that make it flaky?
- Is the snapshot so large it's unreviable? (> 100 lines)
- Does the snapshot test meaningful structure or just noise?

Flag snapshots that should be replaced with explicit assertions.

**3.4 Over-Mocking Detection:**
Identify tests that mock so much they test nothing:

```python
# SLOP: Everything is mocked, test proves nothing
@patch('module.database')
@patch('module.cache')
@patch('module.logger')
@patch('module.external_api')
def test_process(mock_api, mock_logger, mock_cache, mock_db):
    result = process()
    assert result == mock_db.get.return_value  # Circular!
```

Flag tests where:
- The assertion just checks that a mock was called (not the outcome)
- The expected value comes from another mock
- More than 50% of the function's dependencies are mocked
- The test would pass if the implementation were `return None`

**3.5 Test-Implementation Coupling:**
Identify tests that break on valid refactors:

- Tests asserting on private/internal method calls
- Tests checking exact call counts on mocks (brittle)
- Tests verifying implementation order rather than outcomes
- Tests that import and check internal variables

---

## PHASE 4: TEST QUALITY ASSESSMENT

**4.1 Test Structure Quality:**

*Naming:*
- Do test names describe the scenario and expected behavior?
- `test_something` vs `test_user_creation_with_invalid_email_returns_400`
- Are test names consistent across the codebase?

*Arrange-Act-Assert:*
- Is each test focused on one behavior?
- Is setup clearly separated from action and assertion?
- Are tests independent (no shared mutable state)?

*Test Organization:*
- Logical grouping (by feature, by function, by scenario)?
- Consistent file naming (`test_*.py`, `*.test.ts`, `*.spec.ts`)?
- Test file location matches production file location?

**4.2 Assertion Quality:**

*Weak Assertions (flag all):*
```python
assert result  # Just checks truthiness
assert result is not None  # Doesn't check value
assert len(result) > 0  # Doesn't check contents
```

```typescript
expect(result).toBeDefined();     // Weak
expect(result).toBeTruthy();      // Weak
expect(component).toBeInTheDocument();  // Doesn't check content
```

*Missing Assertions:*
- Tests with no assertions at all
- Tests that only check happy path, not error cases
- Tests that check function was called but not with what arguments

*Assertion Specificity:*
- Are error messages/codes checked, not just "it threw"?
- Are specific fields checked, not just "object exists"?
- Are boundary conditions tested (0, 1, many, max)?

**4.3 Async Test Quality:**

*Python:*
```python
# SLOP: Not actually awaiting async code
def test_async_function():
    result = async_function()  # Returns coroutine, not result!
    assert result  # Always passes!

# CORRECT:
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

*TypeScript:*
```typescript
// SLOP: Not waiting for async operations
it('loads data', () => {
    render(<Component />);
    expect(screen.getByText('Data')).toBeInTheDocument();  // Race condition!
});

// CORRECT:
it('loads data', async () => {
    render(<Component />);
    expect(await screen.findByText('Data')).toBeInTheDocument();
});
```

Flag all async tests that may have race conditions.

**4.4 React Testing Library Patterns:**

*Anti-patterns (flag all):*
```typescript
// Testing implementation details
expect(component.state.isOpen).toBe(true);

// Using getByTestId when better queries exist
getByTestId('submit-button')  // Should be getByRole('button', { name: /submit/i })

// Not using userEvent for interactions
fireEvent.click(button);  // Should be userEvent.click(button)

// Querying by class or element type
container.querySelector('.submit-btn');
container.querySelector('button');
```

*Missing patterns:*
- No `waitFor()` or `findBy*` for async state changes
- No accessibility queries (`getByRole`, `getByLabelText`)
- No testing of keyboard navigation where relevant

**4.5 Pytest/Jest Pattern Quality:**

*Fixture/Setup Issues:*
- Overly complex fixtures that are hard to understand
- Fixtures with hidden side effects
- Missing teardown/cleanup
- Shared fixtures that create test interdependence

*Parameterization Opportunities:*
- Duplicate tests that vary only by input (should use `@pytest.mark.parametrize` or `it.each`)
- Copy-pasted test bodies with minor variations

---

## PHASE 5: COVERAGE METRICS

**5.1 Run Coverage Tools:**
```
pytest --cov=src --cov-report=html --cov-report=term-missing
npm test -- --coverage
```

**5.2 Analyze Coverage Report:**
For each module with < 80% coverage:
- What specific lines/branches are untested?
- Are the untested lines error handlers? Edge cases? Dead code?
- Is low coverage justified (generated code, trivial code)?

**5.3 Coverage Quality vs. Quantity:**
High coverage doesn't mean good tests. Flag:
- Files with 100% coverage but weak assertions
- Tests that execute code but don't verify behavior
- Coverage achieved through overly broad integration tests

---

## OUTPUT FORMAT

**Part A — Test Execution Summary:**
```
================================================================================
TEST SUITE EXECUTION RESULTS
================================================================================
Total Test Files: X
Total Test Cases: X

Results:
- Passing: X (X%)
- Failing: X (X%)
- Skipped: X (X%)
- Errored: X (X%) (couldn't run due to import/setup issues)

Execution Time: X minutes
Slowest Tests (> 5s):
1. <test name> — Xs
2. ...

Flaky Tests Detected:
1. <test name> — <symptoms>
2. ...
```

**Part B — Failing Tests (Detailed):**
```
================================================================================
FAILING TEST #<N>
================================================================================
Test: <full test path and name>
File: <test file path>:<line number>
Error Type: <Import | Signature | Assertion | Environment | Timeout>

Error Output:
    <captured error message/traceback, max 20 lines>

Root Cause Analysis:
    <diagnosis of why this test is failing>

Production Code Affected:
    <path to the production code this test covers>

Fix Required:
    <specific steps to fix — update test | update production code | delete test>

Priority: <Critical | High | Medium | Low>
    <rationale for priority>
```

**Part C — Coverage Gaps (Exhaustive):**
```
================================================================================
COVERAGE GAP REPORT
================================================================================

UNTESTED FILES (No corresponding test file exists):
--------------------------------------------------------------------------------
#   | Production File                  | Risk Level | Recommended Action
----|----------------------------------|------------|-------------------------
1   | src/services/billing.py          | CRITICAL   | Create test_billing.py with X test cases
2   | src/utils/crypto.ts              | HIGH       | Create crypto.test.ts with X test cases
...

PARTIALLY TESTED FILES (Test exists but gaps remain):
--------------------------------------------------------------------------------
File: src/services/auth.py
Test: tests/test_auth.py
Coverage: 45%

Untested Functions/Methods:
- `refresh_token()` (lines 45-67) — handles token refresh logic
- `revoke_session()` (lines 89-102) — handles session cleanup
- `validate_permissions()` (lines 120-145) — permission checking

Untested Branches:
- Line 34: `else` branch when user is disabled
- Line 78: `except TokenExpired` handler
- Line 95: `if not session` early return

Recommended Test Cases:
1. test_refresh_token_with_valid_token — happy path
2. test_refresh_token_with_expired_token — should raise/return error
3. test_refresh_token_with_revoked_token — should raise/return error
4. test_revoke_session_clears_all_tokens — verify cleanup
...

---
<repeat for each partially tested file>
```

**Part D — Stale Tests (Detailed):**
```
================================================================================
STALE TEST #<N>
================================================================================
Test: <full test path and name>
File: <test file path>:<line number>
Staleness Type: <Behavioral Drift | Dead Test | Mock Mismatch | Over-Mocked | Snapshot Rot>

Evidence:
    <specific evidence this test is stale>

Current Test Code:
    <relevant code snippet>

Production Code Reality:
    <what the production code actually does now>

Recommended Action: <Update | Delete | Rewrite>
    <specific steps>
```

**Part E — Mock Audit:**
```
================================================================================
MOCK INTEGRITY REPORT
================================================================================

Total Mocks/Stubs Found: X
- Valid (interface matches): X
- Stale (interface mismatch): X
- Missing autospec: X
- Over-mocked tests: X

STALE MOCKS:
--------------------------------------------------------------------------------
#   | Test File:Line        | Mocked Path                | Issue
----|----------------------|----------------------------|----------------------
1   | test_auth.py:45      | services.user.get_user     | Return type changed
2   | UserCard.test.tsx:23 | ../api/fetchUser           | Function renamed to getUser
...

OVER-MOCKED TESTS:
--------------------------------------------------------------------------------
Test: <test name>
Mocks: <count> dependencies mocked
Issue: Test only verifies mocks interact, doesn't test real behavior
Recommendation: Reduce mocking, test integration or use fakes
```

**Part F — Test Quality Issues:**
```
================================================================================
TEST QUALITY FINDINGS
================================================================================

WEAK ASSERTIONS:
--------------------------------------------------------------------------------
#   | Test File:Line    | Assertion                        | Recommendation
----|-------------------|----------------------------------|------------------
1   | test_user.py:56   | assert result                    | Assert specific fields
2   | App.test.tsx:34   | expect(x).toBeTruthy()           | Check specific value
...

ASYNC ISSUES:
--------------------------------------------------------------------------------
#   | Test File:Line    | Issue                            | Fix
----|-------------------|----------------------------------|------------------
1   | test_api.py:89    | Missing await                    | Add await to async call
2   | Modal.test.tsx:45 | Missing waitFor                  | Wrap assertion in waitFor
...

REACT TESTING ANTI-PATTERNS:
--------------------------------------------------------------------------------
#   | Test File:Line    | Anti-Pattern                     | Better Approach
----|-------------------|----------------------------------|------------------
1   | Form.test.tsx:23  | getByTestId('submit')            | getByRole('button', {name: /submit/i})
2   | Input.test.tsx:45 | fireEvent.change()               | userEvent.type()
...

PARAMETERIZATION OPPORTUNITIES:
--------------------------------------------------------------------------------
File: test_validators.py
Lines: 45-120
Issue: 8 nearly identical tests varying only by input
Recommendation: Convert to @pytest.mark.parametrize with X test cases
```

**Part G — Prioritized Remediation Roadmap:**
```
================================================================================
REMEDIATION ROADMAP
================================================================================

PHASE 1 — Critical (Fix immediately, blocks CI/deployment):
--------------------------------------------------------------------------------
1. [ ] Fix failing test: test_auth.py::test_login — Import error
2. [ ] Fix failing test: api.test.ts::handles errors — Assertion mismatch
3. [ ] Add tests for: src/services/payment.py — Handles money, zero coverage

PHASE 2 — High (Fix this sprint, significant risk):
--------------------------------------------------------------------------------
1. [ ] Add tests for: src/services/auth.py::refresh_token — Auth bypass risk
2. [ ] Fix stale mock: test_billing.py — Mock returns wrong shape
3. [ ] Add error handling tests for: src/api/routes.py — No error path coverage

PHASE 3 — Medium (Fix this month, improves confidence):
--------------------------------------------------------------------------------
1. [ ] Add tests for: src/utils/validators.ts — 15 untested functions
2. [ ] Fix snapshot rot: components/__snapshots__ — 23 stale snapshots
3. [ ] Reduce over-mocking in: test_integration.py — Tests prove nothing

PHASE 4 — Low (Backlog, nice to have):
--------------------------------------------------------------------------------
1. [ ] Add parameterization to: test_formatters.py — Reduce duplication
2. [ ] Improve assertions in: UserProfile.test.tsx — Too weak
3. [ ] Remove dead tests for deleted feature: test_legacy_export.py
```

**Part H — Summary Statistics:**
```
================================================================================
FINAL SUMMARY
================================================================================

Test Health Score: <Healthy | Needs Work | Critical Issues> 

Test Execution:
- Total: X | Pass: X | Fail: X | Skip: X | Error: X

Coverage:
- Line Coverage: X%
- Branch Coverage: X%
- Files with 0% Coverage: X
- Files with < 50% Coverage: X

Quality:
- Stale Tests: X
- Over-Mocked Tests: X
- Weak Assertions: X
- Async Issues: X

Risk Assessment:
- Critical untested code paths: X
- High-risk untested code paths: X

Top 5 Highest-Risk Untested Areas:
1. <file/function> — <why it's high risk>
2. ...

Estimated Effort to Reach Healthy State:
- Fix failing tests: ~X hours
- Add missing critical tests: ~X hours
- Fix stale tests: ~X hours
- Total: ~X hours
```

---

**Completeness Checklist:**
Before finalizing, verify:
- [ ] Every test file was executed
- [ ] Every production file was mapped to test files (or marked as untested)
- [ ] Every failing test has root cause analysis
- [ ] Every coverage gap includes specific recommended test cases
- [ ] Every mock was traced to its production counterpart
- [ ] Every stale test has specific evidence of staleness
- [ ] Findings are prioritized by risk, not just quantity
- [ ] Remediation roadmap is actionable with clear ownership