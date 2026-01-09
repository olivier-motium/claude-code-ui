Conduct an exhaustive audit of ALL documentation related to AI/LLM usage and AI-driven business logic in this repository. ultrathink

**Context:**
This codebase uses LLMs (Large Language Models) as part of its core functionality. Documentation of AI systems is uniquely critical because:
- Prompts ARE business logic — undocumented prompts are undocumented features
- AI behavior is non-deterministic — documentation must capture intent, not just implementation
- LLM integrations are expensive — undocumented token usage becomes invisible cost
- AI systems require different mental models — developers need to understand probabilistic reasoning
- Prompt engineering decisions are easily lost — the "why" behind prompt choices evaporates without docs

**Scope: Leave No Stone Unturned.**
- Scan EVERY file containing prompts, LLM calls, or AI-related logic
- Cross-reference EVERY AI feature against existing documentation
- Trace EVERY prompt from user intent → prompt template → LLM call → response handling
- Identify EVERY place where AI behavior affects business outcomes
- Check for documentation of model selection, parameters, costs, and limitations

---

## PHASE 1: AI ASSET DISCOVERY

**1.1 Locate All AI/LLM Code:**

Find and catalog every file containing:
```bash
# Prompt templates and strings
grep -r "system.*prompt\|user.*prompt\|assistant.*prompt" --include="*.ts" --include="*.py" --include="*.tsx"
grep -r "You are\|As an AI\|Your task is\|Please " --include="*.ts" --include="*.py"

# LLM API calls
grep -r "openai\|anthropic\|claude\|gpt-4\|gpt-3.5\|llama\|gemini\|bedrock" --include="*.ts" --include="*.py"
grep -r "createCompletion\|createChatCompletion\|messages\.create\|generate\|complete" --include="*.ts" --include="*.py"

# AI-related configuration
grep -r "temperature\|max_tokens\|top_p\|model.*=" --include="*.ts" --include="*.py" --include="*.json" --include="*.yaml"

# Prompt files
find . -name "*prompt*" -o -name "*template*" -o -name "*.prompt" -o -name "*.txt" | grep -i prompt
```

**1.2 Catalog AI Assets:**

Create a complete inventory:

| Asset Type | Location | Purpose | Documented? |
|------------|----------|---------|-------------|
| System Prompt | /prompts/analyst.txt | Financial analysis | NO |
| Prompt Template | /lib/prompts.ts:45 | Email generation | Partial |
| LLM Service | /services/ai.ts | OpenAI wrapper | YES |
| AI Feature | /features/chat/ | Customer support bot | NO |
| Model Config | /config/models.yaml | Model selection | NO |
| Prompt Chain | /workflows/research.ts | Multi-step research | NO |

---

## PHASE 2: PROMPT DOCUMENTATION AUDIT

**2.1 Prompt Inventory (Every Single Prompt):**

For EVERY prompt or prompt template in the codebase:

```
Prompt ID: <unique identifier or file:line>
Location: <file path and line number>
Type: <System | User | Few-shot | Chain-of-thought | Tool-use | Other>
Purpose: <What business function does this serve?>
Documented: <Yes | Partial | No>

Current Documentation (if any):
<quote existing docs>

Documentation Gaps:
- [ ] Purpose/intent not explained
- [ ] Input variables not documented
- [ ] Expected output format not specified
- [ ] Edge cases not covered
- [ ] Failure modes not documented
- [ ] Model requirements not specified
- [ ] Token/cost implications not noted
- [ ] Version history not tracked
```

**2.2 Prompt Documentation Requirements:**

Every prompt MUST have documentation covering:

*Intent & Context:*
- What business problem does this prompt solve?
- Who is the end user of this AI feature?
- What decision or output does this prompt support?
- Why was this approach chosen over alternatives?

*Input Specification:*
- What variables/placeholders exist in the prompt?
- What are the expected formats and constraints for each input?
- What happens if inputs are missing, malformed, or adversarial?
- Are there length limits on inputs (token budgets)?

*Output Specification:*
- What format should the LLM response be in? (JSON, markdown, plain text, etc.)
- How is the response parsed/validated?
- What constitutes a "successful" response?
- How are malformed or unexpected responses handled?

*Model Requirements:*
- Which model(s) is this prompt designed for?
- What parameters are required (temperature, max_tokens, etc.)?
- Will this prompt work with other models, or is it model-specific?
- What's the expected token usage (input + output)?

*Behavioral Notes:*
- Known edge cases or limitations
- Prompt injection considerations
- Sensitive content handling
- Consistency/determinism expectations

**2.3 Prompt Template Analysis:**

For prompt templates with variables, verify documentation of:

```typescript
// Example: This prompt has 3 variables — are they ALL documented?
const analysisPrompt = `
You are a financial analyst. Analyze the following data:

Company: {{company_name}}
Financials: {{financial_data}}
Time Period: {{time_period}}

Provide your analysis in the following format:
...
`;
```

| Variable | Type | Required | Constraints | Documented |
|----------|------|----------|-------------|------------|
| company_name | string | Yes | Max 100 chars | NO |
| financial_data | JSON | Yes | Schema at X | NO |
| time_period | string | Yes | Format: YYYY-QN | NO |

---

## PHASE 3: AI BUSINESS LOGIC DOCUMENTATION

**3.1 AI Feature Documentation:**

For EVERY user-facing AI feature, verify documentation exists for:

*Feature Overview:*
- What does this AI feature do from the user's perspective?
- What is the user journey / interaction flow?
- What inputs does the user provide?
- What outputs does the user receive?
- What are the feature's limitations (documented for users)?

*Business Logic:*
- What business rules govern this AI feature?
- What validation happens before/after LLM calls?
- What fallback behavior exists if AI fails?
- What human oversight or review is required?
- What audit logging captures AI decisions?

*AI Pipeline:*
- What is the end-to-end data flow?
- How is user input preprocessed before prompting?
- How is LLM output postprocessed before showing to user?
- Are there multiple LLM calls (chains/agents)?
- What caching or optimization exists?

**3.2 AI Decision Documentation:**

For AI systems that make or influence decisions:

*Decision Documentation:*
- What decisions does the AI make or recommend?
- What's the impact of these decisions (financial, operational, user-facing)?
- What confidence thresholds or guardrails exist?
- When does AI defer to human judgment?
- How can users contest or override AI decisions?

*Explainability:*
- Can the AI explain its reasoning?
- Is there documentation of how explanations are generated?
- What information is logged for audit purposes?

**3.3 AI Workflow Documentation:**

For multi-step AI workflows (agents, chains, RAG, etc.):

```
Workflow: <name>
Location: <file path>
Documentation Status: <Complete | Partial | Missing>

Steps:
1. <Step name> → <What happens> → <Documented?>
2. <Step name> → <What happens> → <Documented?>
...

Missing Documentation:
- Step 3 decision logic not explained
- Retry/error handling not documented
- Token budget across steps not specified
```

---

## PHASE 4: MODEL & CONFIGURATION DOCUMENTATION

**4.1 Model Selection Documentation:**

For EVERY model used in the system:

| Model | Used For | Why This Model | Config Location | Documented |
|-------|----------|----------------|-----------------|------------|
| gpt-4o | Complex analysis | Accuracy critical | /config/ai.yaml | NO |
| gpt-4o-mini | Simple tasks | Cost optimization | /config/ai.yaml | NO |
| claude-3-5-sonnet | Long context | 200k context window | /config/ai.yaml | NO |
| text-embedding-3-small | Embeddings | Cost/performance | /config/ai.yaml | NO |

Document for each model:
- Why was this model chosen?
- What are its strengths/weaknesses for this use case?
- What's the fallback if this model is unavailable?
- What's the cost implication?
- When should it be upgraded/changed?

**4.2 Parameter Documentation:**

For EVERY AI configuration:

```yaml
# Are these parameters documented? WHY these values?
ai:
  default_model: "gpt-4o"        # Documented: NO — why not gpt-4o-mini?
  temperature: 0.7                # Documented: NO — why 0.7?
  max_tokens: 4096               # Documented: NO — how was this chosen?
  timeout_seconds: 30            # Documented: NO — what happens on timeout?
  retry_attempts: 3              # Documented: NO — what's retry strategy?
```

**4.3 Cost & Rate Limit Documentation:**

Verify documentation exists for:

- [ ] Token budgets per feature/endpoint
- [ ] Cost estimates per operation
- [ ] Rate limits and how they're handled
- [ ] Cost monitoring and alerting
- [ ] Budget caps and what happens when exceeded
- [ ] Cost optimization strategies implemented

---

## PHASE 5: AI SAFETY & GUARDRAILS DOCUMENTATION

**5.1 Prompt Injection Protection:**

For EVERY user-input-to-prompt path:

| Input Point | Reaches Prompt | Sanitization | Documented |
|-------------|----------------|--------------|------------|
| Chat input | Yes | Basic escape | NO |
| Document upload | Yes (RAG) | None | NO |
| API parameter | Yes | Type validation | NO |

Document for each:
- What injection attacks are possible?
- What mitigations are in place?
- What's the blast radius if injection succeeds?

**5.2 Output Safety Documentation:**

Verify documentation of:

- [ ] Content filtering on LLM outputs
- [ ] PII detection/redaction
- [ ] Hallucination mitigation strategies
- [ ] Citation/source verification (for RAG)
- [ ] Output validation before user display
- [ ] Harmful content detection

**5.3 AI Failure Mode Documentation:**

For EVERY AI feature, document:

| Failure Mode | Detection | User Experience | Documented |
|--------------|-----------|-----------------|------------|
| LLM timeout | Timeout error | "Try again" message | NO |
| Malformed response | JSON parse fail | Fallback response | NO |
| Rate limited | 429 error | Queue + retry | NO |
| Content filtered | Refusal detected | Apology message | NO |
| Hallucination | Confidence low | Human review flag | NO |

---

## PHASE 6: RAG & CONTEXT DOCUMENTATION (if applicable)

**6.1 RAG Pipeline Documentation:**

If using Retrieval-Augmented Generation:

*Data Sources:*
- [ ] What data sources feed the RAG system?
- [ ] How is data ingested and updated?
- [ ] What's the freshness/staleness of data?
- [ ] How are data quality issues handled?

*Embedding & Retrieval:*
- [ ] What embedding model is used? Why?
- [ ] What vector database is used?
- [ ] What chunking strategy is used? Why those chunk sizes?
- [ ] What retrieval strategy (similarity, hybrid, rerank)?
- [ ] How many chunks are retrieved? Why that number?

*Context Assembly:*
- [ ] How is retrieved context formatted for the LLM?
- [ ] What's the token budget for context vs. response?
- [ ] How are conflicting sources handled?
- [ ] How is source attribution maintained?

**6.2 Knowledge Base Documentation:**

| Knowledge Source | Update Frequency | Owner | Documented |
|------------------|------------------|-------|------------|
| Product docs | Weekly | Docs team | NO |
| Support tickets | Daily | Support | NO |
| Internal wiki | Ad-hoc | Engineering | NO |

---

## PHASE 7: AI TESTING & EVALUATION DOCUMENTATION

**7.1 AI Test Documentation:**

Verify documentation of:

- [ ] How are prompts tested before deployment?
- [ ] What evaluation metrics are used?
- [ ] What's the test dataset / golden set?
- [ ] How are regressions detected?
- [ ] What's the prompt review/approval process?

**7.2 Evaluation Criteria Documentation:**

For EACH AI feature:

| Feature | Accuracy Metric | Target | Measured | Documented |
|---------|-----------------|--------|----------|------------|
| Summarization | ROUGE score | >0.7 | Unknown | NO |
| Classification | Precision/Recall | >95% | Yes | NO |
| Generation | Human review | 4/5 stars | Yes | NO |

---

## PHASE 8: AI OPERATIONAL DOCUMENTATION

**8.1 Monitoring Documentation:**

Verify documentation of:

- [ ] What AI metrics are monitored?
- [ ] What alerts exist for AI failures?
- [ ] How is AI quality tracked over time?
- [ ] How are token costs monitored?
- [ ] What dashboards exist?

**8.2 Incident Response Documentation:**

- [ ] What's the runbook for AI service outage?
- [ ] How are prompt-related bugs triaged?
- [ ] How are AI quality degradations detected and addressed?
- [ ] What's the rollback strategy for prompt changes?

---

## OUTPUT FORMAT

**Part A — AI Asset Inventory:**
```
================================================================================
AI/LLM ASSET DISCOVERY
================================================================================

Prompts Found: X
- System prompts: X
- User prompt templates: X
- Few-shot examples: X
- Chain-of-thought prompts: X

LLM Integration Points: X
- Direct API calls: X
- Wrapper services: X
- Agent/chain orchestration: X

AI Features: X (list each with documentation status)

Models Referenced: X (list each with documentation status)

Files Containing AI Logic:
<tree view of all files with AI/LLM code>
```

**Part B — Prompt Documentation Gaps:**
```
================================================================================
PROMPT DOCUMENTATION AUDIT
================================================================================

Total Prompts: X
- Fully documented: X
- Partially documented: X
- Undocumented: X

UNDOCUMENTED PROMPTS (Critical):

Prompt #1
Location: /services/ai/prompts/analyst.ts:23-45
Purpose: Financial analysis generation
Business Impact: Customer-facing reports
Documentation Status: NONE

Missing Documentation:
- [ ] Intent and business context
- [ ] Input variable specifications
- [ ] Output format requirements
- [ ] Model requirements
- [ ] Failure handling

Suggested Documentation:

```markdown
## Financial Analyst Prompt

### Purpose
Generates financial analysis reports for quarterly earnings data. Used in the 
"Earnings Summary" feature visible to Pro tier customers.

### Business Context
This prompt powers the automated earnings analysis that previously required 
a human analyst (2-3 hours per report). Accuracy is critical as customers 
make investment decisions based on this output.

### Input Variables

| Variable | Type | Required | Description | Constraints |
|----------|------|----------|-------------|-------------|
| company_name | string | Yes | Company being analyzed | Max 100 chars |
| financial_data | object | Yes | Quarterly financials | Schema: /schemas/financials.json |
| comparison_period | string | No | Prior period for comparison | Format: YYYY-QN |

### Output Format
JSON object conforming to `/schemas/analysis-output.json`:
- `summary`: 2-3 paragraph executive summary
- `metrics`: Key financial metrics with YoY change
- `risks`: Identified risk factors
- `outlook`: Forward-looking statements

### Model Requirements
- Model: gpt-4o (requires complex reasoning)
- Temperature: 0.3 (consistency important)
- Max tokens: 2000
- Expected cost: ~$0.08 per analysis

### Failure Handling
- Malformed JSON: Retry once with explicit JSON instruction
- Refusal: Log and fall back to template-based summary
- Timeout: Return partial results with disclaimer

### Known Limitations
- Struggles with non-US GAAP financials
- May hallucinate competitor comparisons if not provided
- Quarterly data only; annual reports require different prompt
```

---
<repeat for each undocumented/partially documented prompt>
```

**Part C — AI Business Logic Gaps:**
```
================================================================================
AI BUSINESS LOGIC DOCUMENTATION
================================================================================

AI Features Audit:

Feature: Customer Support Chatbot
Location: /features/support-chat/
Documentation: MISSING

Required Documentation:

```markdown
## Customer Support Chatbot

### Feature Overview
AI-powered first-line customer support that handles common queries before 
escalating to human agents.

### User Journey
1. Customer opens chat widget
2. Bot greets and asks for issue category
3. Customer describes issue in natural language
4. Bot attempts to resolve using knowledge base
5. If unresolved after 3 turns OR customer requests, escalate to human

### Business Rules
- Always identify as AI ("I'm an AI assistant...")
- Never make promises about refunds/credits (escalate)
- Never share internal policies or pricing logic
- Escalate any mention of: legal, lawsuit, regulatory, fraud
- Maximum 5 AI turns before mandatory escalation offer

### AI Pipeline
1. User input → Content filter (block harmful)
2. Input → Intent classification (gpt-4o-mini, ~50 tokens)
3. Intent → RAG retrieval (top 5 docs from support KB)
4. Context + Input → Response generation (gpt-4o, ~500 tokens)
5. Response → Output filter (PII redaction, brand voice check)
6. Response → User

### Fallback Behavior
- LLM timeout (>10s): "I'm having trouble thinking. Let me connect you with a human."
- 3 consecutive low-confidence responses: Auto-escalate
- After business hours: Collect info, promise follow-up

### Metrics & Monitoring
- Resolution rate (target: 40%)
- Escalation rate (target: <30%)
- CSAT after AI resolution (target: 4.0+)
- Avg tokens per conversation (budget: 2000)
```

---
<repeat for each AI feature>
```

**Part D — Model & Configuration Gaps:**
```
================================================================================
MODEL & CONFIGURATION DOCUMENTATION
================================================================================

Models Used (Undocumented):

Model: gpt-4o
Config Location: /config/ai.yaml:12
Used By: Complex analysis, support chat
Documentation: MISSING

Suggested Documentation:

```markdown
## Model: gpt-4o

### When to Use
- Complex reasoning tasks requiring multi-step logic
- Customer-facing outputs where quality is critical
- Tasks requiring current knowledge (training data more recent)

### When NOT to Use
- Simple classification (use gpt-4o-mini)
- High-volume, low-stakes tasks (cost prohibitive)
- When latency is critical (<1s requirement)

### Configuration
```yaml
model: gpt-4o
temperature: 0.7  # Balance creativity/consistency
max_tokens: 4096  # Sufficient for most outputs
timeout: 30s      # Generous for complex reasoning
```

### Cost
- Input: $2.50 / 1M tokens
- Output: $10.00 / 1M tokens
- Typical request: ~$0.02-0.05

### Fallback
If gpt-4o unavailable, fall back to gpt-4o-mini with warning log.
Expect quality degradation on complex tasks.
```

---

Parameter Documentation Gaps:

| Parameter | Location | Value | Why This Value? | Documented |
|-----------|----------|-------|-----------------|------------|
| temperature | ai.yaml:15 | 0.7 | Unknown | NO |
| max_tokens | ai.yaml:16 | 4096 | Unknown | NO |
| chunk_size | rag.yaml:8 | 512 | Unknown | NO |
| top_k | rag.yaml:12 | 5 | Unknown | NO |
```

**Part E — Safety & Guardrails Gaps:**
```
================================================================================
AI SAFETY DOCUMENTATION
================================================================================

Prompt Injection Audit:

| Input Path | Risk Level | Mitigation | Documented |
|------------|------------|------------|------------|
| Chat input | HIGH | Basic escaping | NO |
| Doc upload | HIGH | None visible | NO |
| Search query | MEDIUM | Length limit | NO |

Missing Safety Documentation:
1. No prompt injection mitigation docs
2. No output filtering docs
3. No content policy docs
4. No incident response runbook for AI issues

Suggested Documentation:

```markdown
## AI Safety: Prompt Injection

### Risk Overview
User inputs are incorporated into prompts. Malicious users may attempt to:
- Override system instructions
- Extract system prompts
- Generate harmful content
- Access other users' data

### Mitigations Implemented

1. **Input Sanitization** (`/lib/ai/sanitize.ts`)
   - Strip common injection patterns
   - Truncate inputs to token budget
   - Encode special characters

2. **Prompt Structure**
   - System prompt emphasizes instruction hierarchy
   - User input wrapped in XML tags
   - Post-input reminder of constraints

3. **Output Validation** (`/lib/ai/validate.ts`)
   - Check for system prompt leakage
   - Check for disallowed content patterns
   - Validate expected output format

### Testing
Injection test suite: `/tests/ai/injection.test.ts`
Run before any prompt changes: `npm run test:ai:security`

### Incident Response
If injection is detected in production:
1. Block the specific input pattern immediately
2. Review logs for similar attempts
3. Page on-call if data exposure suspected
4. Post-mortem within 48 hours
```
```

**Part F — RAG Documentation Gaps (if applicable):**
```
================================================================================
RAG SYSTEM DOCUMENTATION
================================================================================

RAG Pipeline: UNDOCUMENTED

Suggested Documentation:

```markdown
## RAG System: Customer Support Knowledge Base

### Architecture Overview
```
User Query → Embedding → Vector Search → Rerank → Context Assembly → LLM
```

### Data Sources

| Source | Type | Update Frequency | Owner |
|--------|------|------------------|-------|
| Help Center | HTML scrape | Daily 2am UTC | Docs team |
| Product FAQ | Manual upload | Ad-hoc | Product |
| Past Tickets | API sync | Hourly | Support |

### Embedding Configuration
- Model: text-embedding-3-small
- Dimensions: 1536
- Why: Best cost/performance for short support docs

### Chunking Strategy
- Chunk size: 512 tokens
- Overlap: 50 tokens
- Why: Support articles are short; small chunks = precise retrieval

### Retrieval Configuration
- Initial retrieval: Top 20 by cosine similarity
- Reranking: Cohere rerank-english-v2.0, return top 5
- Why: Reranking significantly improves relevance for support queries

### Context Assembly
- Format: Numbered list with source attribution
- Token budget: 3000 tokens for context
- Conflict handling: Most recent source wins, flag for human review

### Known Limitations
- Multi-language queries have degraded performance
- Very recent changes (<24h) not reflected
- PDF attachments not indexed (text extraction TODO)
```
```

**Part G — Prioritized Remediation:**
```
================================================================================
REMEDIATION ROADMAP
================================================================================

CRITICAL (document immediately — risk of incidents or major confusion):
1. [ ] Document main customer-facing AI feature (support chat)
2. [ ] Document prompt injection mitigations
3. [ ] Document AI failure modes and fallbacks
4. [ ] Document model selection rationale

HIGH (document this sprint — affects development velocity):
1. [ ] Create prompt documentation template
2. [ ] Document all production prompts (X prompts)
3. [ ] Document RAG pipeline configuration
4. [ ] Document token budgets and cost expectations

MEDIUM (document this month — improves maintainability):
1. [ ] Document AI testing and evaluation approach
2. [ ] Document prompt change/review process
3. [ ] Document AI monitoring and alerting
4. [ ] Create AI architecture decision records (ADRs)

LOW (document when possible — nice to have):
1. [ ] Document historical prompt iterations and learnings
2. [ ] Document failed approaches and why
3. [ ] Create prompt engineering style guide
```

**Part H — Summary Statistics:**
```
================================================================================
FINAL SUMMARY
================================================================================

AI Documentation Health: POOR

Assets Discovered:
- Prompts: X total (Y undocumented)
- AI Features: X total (Y undocumented)
- Models used: X (Y undocumented)
- LLM call sites: X

Documentation Coverage:
- Prompts with full docs: X%
- AI features with full docs: X%
- Model choices documented: X%
- Safety measures documented: X%
- RAG pipeline documented: X%

Critical Gaps:
1. No documentation of why prompts are structured as they are
2. No documentation of expected AI behavior for edge cases
3. No documentation of AI failure modes and fallbacks
4. No documentation of token budgets or cost implications
5. No prompt injection / safety documentation

Risk Assessment:
- New developer can understand AI system: NO
- Prompt changes can be made safely: NO
- AI costs are predictable: NO
- AI failures are handled gracefully: UNKNOWN
- Security review would pass: UNLIKELY

Estimated Effort to Reach Acceptable State:
- Critical gaps: ~8 hours
- High priority: ~16 hours
- Full documentation: ~40 hours
```

---

**Completeness Checklist:**
- [ ] Every file with prompts was identified and cataloged
- [ ] Every prompt has documentation requirements assessed
- [ ] Every AI feature has business logic documentation assessed
- [ ] Every model has selection rationale assessed
- [ ] Every AI configuration parameter has documentation assessed
- [ ] Safety and guardrails documentation assessed
- [ ] RAG system (if present) documentation assessed
- [ ] AI testing and evaluation documentation assessed
- [ ] Every gap includes concrete suggested documentation
- [ ] Remediation is prioritized by risk and impact