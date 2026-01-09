# Chrome: Form Testing Pattern

Testing form validation, submission, and error handling using Chrome integration.

## Workflow

```
1. tabs_context_mcp → Get/create tab
2. navigate → Go to form page
3. read_page → Discover form fields
4. form_input → Fill fields with test data
5. computer (click) → Submit form
6. read_page → Verify success/error states
7. read_console_messages → Check for JS errors
```

## Example: Login Form Validation

**User prompt:**
```
Test the login form at localhost:3000/login:
1. Try empty submission - check error messages
2. Try invalid email format - check validation
3. Try valid credentials - verify redirect
```

**Claude's approach:**

### Step 1: Get Tab Context
```
Tool: mcp__claude-in-chrome__tabs_context_mcp
Params: { createIfEmpty: true }
```

### Step 2: Create Fresh Tab
```
Tool: mcp__claude-in-chrome__tabs_create_mcp
```

### Step 3: Navigate
```
Tool: mcp__claude-in-chrome__navigate
Params: { url: "http://localhost:3000/login", tabId: <tab_id> }
```

### Step 4: Read Form Structure
```
Tool: mcp__claude-in-chrome__read_page
Params: { tabId: <tab_id>, filter: "interactive" }
```

Returns accessibility tree showing form fields with ref IDs.

### Step 5: Test Empty Submission
```
Tool: mcp__claude-in-chrome__find
Params: { query: "submit button", tabId: <tab_id> }

Tool: mcp__claude-in-chrome__computer
Params: { action: "left_click", ref: "<submit_ref>", tabId: <tab_id> }
```

### Step 6: Check Error Messages
```
Tool: mcp__claude-in-chrome__read_page
Params: { tabId: <tab_id> }
```

Look for error text in accessibility tree.

### Step 7: Fill Invalid Email
```
Tool: mcp__claude-in-chrome__form_input
Params: { ref: "<email_ref>", value: "notanemail", tabId: <tab_id> }

Tool: mcp__claude-in-chrome__form_input
Params: { ref: "<password_ref>", value: "test123", tabId: <tab_id> }
```

### Step 8: Submit and Verify
```
Tool: mcp__claude-in-chrome__computer
Params: { action: "left_click", ref: "<submit_ref>", tabId: <tab_id> }

Tool: mcp__claude-in-chrome__read_page
Params: { tabId: <tab_id> }
```

## Key Patterns

### Finding Form Fields
Use `filter: "interactive"` to get only actionable elements:
```
Tool: mcp__claude-in-chrome__read_page
Params: { tabId: <tab_id>, filter: "interactive" }
```

### Natural Language Element Search
```
Tool: mcp__claude-in-chrome__find
Params: { query: "email input field", tabId: <tab_id> }
```

### Checking Validation Messages
After triggering validation, read the page and look for:
- Error text near form fields
- Toast notifications
- Alert banners

### Screenshot for Visual Verification
```
Tool: mcp__claude-in-chrome__computer
Params: { action: "screenshot", tabId: <tab_id> }
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Form field not found | Use `read_page` first to discover actual element refs |
| Submit doesn't work | Check if button is disabled, look for JS errors in console |
| Validation not triggering | May need to blur field first - click elsewhere then back |
| Redirect not happening | Check `read_console_messages` for errors |
