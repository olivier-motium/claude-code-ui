Conduct an exhaustive SECURITY AUDIT of the ENTIRE codebase, testing for vulnerabilities, misconfigurations, and security anti-patterns. ultrathink

**Stack Context:**
- Backend: FastAPI (Python)
- Frontend: Next.js / Vite (React/TypeScript)
- Database: PostgreSQL
- Infrastructure: Local dev + Azure (dev/prod environments)
- Credentials: Available in .env files for active security testing
- Dev Pattern: .env files for local development (this is expected and acceptable)

**Scope: Full Adversarial Analysis.**
- Scan EVERY source file — do not sample or spot-check.
- Trace ALL data flows from user input to storage/output (taint analysis).
- Examine EVERY API endpoint, authentication check, and authorization gate.
- Test with actual credentials from .env against local/dev environments.
- Think like an attacker: what can be exploited, not just what looks wrong.

**Your Priorities (in order):**

**1. Authentication & Session Security (Complete Audit):**
Examine every authentication mechanism:

*Authentication Implementation:*
- Weak password policies (no complexity requirements, no length minimums).
- Missing or weak password hashing (anything other than bcrypt/argon2/scrypt with proper cost factors).
- Timing attacks in authentication (non-constant-time comparison of secrets).
- Missing account lockout after failed attempts.
- Insecure "remember me" implementations.
- Password reset flows: predictable tokens, no expiration, token reuse, email enumeration.
- Missing MFA where it should exist (admin accounts, sensitive operations).

*Session Management:*
- Session tokens in URLs (logged, cached, referer leakage).
- Predictable or weak session token generation.
- Missing session expiration or overly long session lifetimes.
- Sessions not invalidated on logout, password change, or privilege change.
- Session fixation vulnerabilities.
- Missing secure/httponly/samesite flags on session cookies.
- JWT issues: weak signing algorithms (none, HS256 with weak secret), missing expiration, sensitive data in payload, tokens not invalidated on logout.

*OAuth/SSO (if applicable):*
- Missing state parameter (CSRF in OAuth flow).
- Open redirect in callback URLs.
- Token leakage through referer headers.
- Insufficient scope validation.

**2. Authorization & Access Control (Every Endpoint):**
Map and test every authorization decision:

*Endpoint Authorization:*
- List ALL API endpoints and their required permissions.
- Identify endpoints missing authentication entirely.
- Identify endpoints missing authorization checks (authenticated but any user can access).
- IDOR (Insecure Direct Object Reference): can user A access user B's resources by changing IDs?
- Horizontal privilege escalation: access to peer users' data.
- Vertical privilege escalation: regular user accessing admin functions.
- Missing function-level access control (admin endpoints accessible to regular users).

*Authorization Logic:*
- Authorization checks in wrong layer (frontend only, not backend).
- Race conditions in authorization (TOCTOU — time-of-check vs time-of-use).
- Inconsistent authorization across similar endpoints.
- Mass assignment: can users set fields they shouldn't (is_admin, role, owner_id)?
- GraphQL/nested queries bypassing authorization on related objects.
- Batch endpoints not checking authorization on each item.

*Multi-tenancy (if applicable):*
- Tenant isolation: can tenant A access tenant B's data?
- Missing tenant context in ALL database queries.
- Tenant ID from user input rather than session.
- Shared resources leaking cross-tenant data.

**3. Injection Vulnerabilities (Every Input Vector):**
Trace every path from user input to dangerous sinks:

*SQL Injection:*
- Raw SQL with string concatenation or f-strings (catalog EVERY instance).
- ORM queries with raw() or text() containing user input.
- Dynamic table/column names from user input.
- ORDER BY, LIMIT, OFFSET with unsanitized input.
- LIKE clauses with unescaped wildcards.
- JSON/JSONB queries with user-controlled paths.
- Stored procedures called with unsanitized parameters.

*Command Injection:*
- subprocess, os.system, exec with user input.
- Shell=True with any external input.
- Template strings in shell commands.
- Indirect command injection through filenames, environment variables.

*Code Injection:*
- eval(), exec() with any user-influenced input.
- Dynamic imports based on user input.
- pickle/yaml.load with untrusted data (deserialization attacks).
- Template injection (Jinja2, etc.) with user-controlled templates.

*NoSQL/Other Injection:*
- MongoDB query injection through object properties.
- LDAP injection.
- XML External Entity (XXE) injection.
- XPath injection.
- Header injection (CRLF injection in HTTP headers).

*Path Traversal:*
- File operations with user-controlled paths (../../../etc/passwd).
- Archive extraction (zip slip vulnerability).
- User-controlled paths in static file serving.
- Log file paths based on user input.

**4. Frontend Security (XSS, CSRF, Client-Side Issues):**
Examine every output point and client-side security control:

*Cross-Site Scripting (XSS):*
- Reflected XSS: user input echoed in responses without encoding.
- Stored XSS: user content saved and displayed to other users.
- DOM-based XSS: client-side code using location, document.referrer, postMessage unsafely.
- dangerouslySetInnerHTML usage (catalog EVERY instance with data source).
- innerHTML, outerHTML assignments.
- href/src attributes with user-controlled URLs (javascript: protocol).
- SVG uploads or rendering with embedded scripts.
- CSS injection (expression(), url() with user data).
- Missing Content-Security-Policy headers or overly permissive CSP.

*Cross-Site Request Forgery (CSRF):*
- State-changing operations via GET requests.
- Missing CSRF tokens on forms and AJAX requests.
- CSRF tokens not validated server-side.
- CSRF tokens predictable or reusable.
- SameSite cookie attribute not set or set to None.
- CORS misconfigurations allowing credentialed cross-origin requests.

*Client-Side Security:*
- Sensitive data in localStorage/sessionStorage (tokens, PII).
- Secrets in client-side JavaScript bundles (API keys, credentials) — THIS IS A REAL ISSUE.
- postMessage without origin validation.
- Sensitive data in URL parameters (logged, cached, referer leakage).
- Client-side validation without server-side validation.
- Open redirects (user-controlled redirect destinations).
- Clickjacking (missing X-Frame-Options, frame-ancestors CSP).

**5. API Security (Every Endpoint Deep Dive):**
Audit every API endpoint for security issues:

*Input Validation:*
- Missing input validation (no schema, no type checking).
- Incomplete validation (some fields checked, others not).
- Validation bypass through type juggling, encoding, case variation.
- Missing length limits on strings (DoS through large payloads).
- Missing array length limits (DoS through large arrays).
- Missing depth limits on nested objects.
- File upload validation: type, size, filename, content validation.
- Content-Type validation (does the server trust client headers?).

*Rate Limiting & DoS Protection:*
- Missing rate limiting on authentication endpoints (brute force).
- Missing rate limiting on expensive operations.
- Missing rate limiting on public endpoints.
- Rate limits bypassable (different headers, API keys, IP rotation).
- Resource exhaustion: endpoints that can trigger expensive operations.
- Regex DoS (ReDoS) in input validation patterns.
- Missing request size limits.
- Unbounded queries without pagination (return millions of rows).

*API Design Issues:*
- Verbose error messages leaking implementation details.
- Stack traces exposed to users.
- API versioning allowing access to deprecated vulnerable versions.
- HTTP methods not restricted (PUT/DELETE on read-only resources).
- Missing HTTPS enforcement (HTTP allowed in production).
- Sensitive data in query parameters (logged in server logs, browser history).

**6. Secrets & Configuration Security:**
Audit secrets handling with practical focus:

*Actual Security Risks (flag these):*
- .env or .env.local committed to git repository.
- Secrets in git history (even if removed from current HEAD).
- Hardcoded credentials in source code (not reading from env).
- Secrets in client-side bundles (exposed to browser).
- Secrets in Docker images or build artifacts.
- Secrets logged or included in error messages.
- Default/fallback credentials in code (if env var missing, use "admin123").
- .env.example containing real secret values instead of placeholders.
- Production secrets accessible in dev environment or vice versa.

*Non-Issues (do NOT flag):*
- Using .env files for local development configuration.
- Database passwords in .env for local PostgreSQL.
- API keys in .env for local testing.

*Configuration Issues:*
- DEBUG=True or equivalent in production config.
- Verbose logging in production exposing sensitive data.
- CORS allowing all origins in production.
- Insecure default configurations that aren't overridden in prod.

*Azure-Specific (for production):*
- Secrets hardcoded instead of using Azure Key Vault.
- Managed Identity not used where available.
- Storage account keys instead of SAS tokens or managed identity.
- Overly permissive RBAC assignments.
- Connection strings in app settings instead of Key Vault references.

**7. Database Security (PostgreSQL Focus):**
Examine all database interactions:

*Query Security:*
- All SQL injection vectors (verify every query).
- Sensitive data not encrypted at column level where warranted.
- Excessive data exposure in queries (selecting more than needed).
- Missing row-level security where multi-tenancy exists.

*Connection Security:*
- SSL/TLS not enforced for database connections in production.
- Database user with excessive permissions (superuser for app).
- Same credentials used across all environments.

*Data Handling:*
- Sensitive data types wrong (credit cards as plain varchar).
- PII stored without considering data retention policies.
- Soft deletes exposing "deleted" data through queries.
- Missing audit trails for sensitive operations.

**8. Cryptography Audit:**
Examine all cryptographic operations:

*Weak Cryptography:*
- MD5 or SHA1 for security purposes (password hashing, integrity).
- ECB mode for encryption.
- Static/predictable IVs or nonces.
- Weak random number generation (random module instead of secrets).
- Hardcoded encryption keys in source code.
- Short key lengths (< 2048 RSA, < 256 AES).

*Implementation Issues:*
- Custom cryptography implementations.
- Missing integrity protection (encryption without authentication).
- Sensitive comparison not constant-time (timing attacks).
- Insufficient entropy in token/ID generation.

**9. Dependency & Supply Chain Security:**
Audit all dependencies:

*Vulnerable Dependencies:*
- Run `pip audit` and `npm audit` (or equivalent).
- Check both direct and transitive dependencies.
- Identify dependencies with known CVEs and severity.
- Dependencies with no recent updates (abandoned/unmaintained).

*Dependency Configuration:*
- Lock files missing or not committed.
- Version ranges allowing unexpected updates.
- Development dependencies included in production builds.

**10. Logging & Error Handling:**
Examine security visibility and information leakage:

*Sensitive Data Exposure:*
- Passwords logged (even partially).
- Tokens or session IDs logged.
- PII logged unnecessarily.
- Full request bodies logged including sensitive fields.
- Stack traces returned to users in production.

*Security Event Logging:*
- Failed authentication attempts not logged.
- Authorization failures not logged.
- Admin/sensitive actions not logged.
- Missing correlation IDs for tracing.

**11. Security Headers & Transport Security:**
Check HTTP security configurations:

*Required Headers (for production):*
- Content-Security-Policy.
- X-Content-Type-Options: nosniff.
- X-Frame-Options or frame-ancestors CSP.
- Strict-Transport-Security (HSTS).
- Referrer-Policy.

*CORS Configuration:*
- Wildcard origins with credentials.
- Reflecting Origin header without validation.
- Overly permissive in production.

---

**Output Format:**

**Part A — Attack Surface Inventory:**
```
API Endpoints: X total
- Public (no auth): X endpoints (list all)
- Authenticated: X endpoints  
- Admin only: X endpoints
- Missing authorization: X endpoints (CRITICAL — list all)

User Input Vectors:
- Forms/JSON bodies: X
- URL/query parameters: X
- File uploads: X locations
- Headers processed: X

Data Stores:
- PostgreSQL tables with sensitive data: (list)
- Caches: (list with contents)
- File storage: (list with access controls)
```

**Part B — Vulnerability Findings:**

```
Finding #<N>
Category: <Auth | AuthZ | Injection | XSS | CSRF | Crypto | Config | Secrets | etc.>
CWE ID: <CWE-XXX>
Target: <file:line, endpoint, or component>
Severity: <Critical | High | Medium | Low | Info>
Exploitability: <Trivial | Easy | Moderate | Difficult>

The Vulnerability:
<Technical explanation>

Attack Scenario:
<How an attacker exploits this>

Evidence:
<Code snippet or request/response, max 15 lines>

Proof of Concept (if tested locally):
<Commands/requests that demonstrate the issue>

Impact:
<What can attacker achieve? Who is affected?>

Remediation:
<Specific fix with code example>

Related Findings: <#s that share root cause>
```

**Part C — Secrets Audit:**
```
ACTUAL ISSUES FOUND:

Secrets Committed to Git:
- <file> — <secret type> — <commit if in history>

Secrets in Source Code:
- <file:line> — <description>

Secrets in Client Bundles:
- <file> — <what's exposed>

Secrets in Logs/Errors:
- <location> — <what leaks>

STATUS: .env files for local dev — OK (expected pattern)
```

**Part D — Injection Point Analysis:**
```
SQL Queries: X total
- Safe (parameterized): X
- Unsafe (string building): X (list all with file:line)

Command Execution: X locations
- User input reaches shell: X (CRITICAL — list all)

File Operations: X locations  
- Path traversal risk: X (list all)

Deserialization: X locations
- Untrusted data: X (list all)
```

**Part E — Authorization Matrix:**
```
| Endpoint | Method | Auth | Role | IDOR Tested | Result |
|----------|--------|------|------|-------------|--------|
| /api/users/{id} | GET | Yes | Any | Yes | VULN |
| ... | ... | ... | ... | ... | ... |

Gaps Found:
1. <endpoint> — <issue>
```

**Part F — Dependency Scan:**
```
Critical CVEs:
- <package>@<version> — <CVE> — <fixed in>

High CVEs:
- ...

Outdated (potential risk):
- <package> — current: X, latest: Y
```

**Part G — Prioritized Remediation:**
```
IMMEDIATE (blocks deploy):
1. Finding #X: <summary> — Effort: <hours>

THIS WEEK (high risk):
1. ...

THIS MONTH (moderate risk):  
1. ...

BACKLOG (hardening):
1. ...
```

**Part H — Summary:**
```
Total Findings: X
- Critical: X
- High: X
- Medium: X
- Low: X

By Category:
- Access Control: X
- Injection: X
- XSS/CSRF: X
- Secrets: X
- Config: X
- Dependencies: X

Top 5 Riskiest Files:
1. <file> — X findings

Overall Security Posture: <Poor | Fair | Good | Excellent>
```

**Testing Notes:**
- Use .env credentials to actively test auth/authz against local/dev
- Create test users at different privilege levels
- Test IDOR by swapping IDs between user contexts
- Document test data created for cleanup
- DO NOT test exploits against production