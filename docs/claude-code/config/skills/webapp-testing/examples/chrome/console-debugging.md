# Chrome: Console Debugging Pattern

Using Chrome integration to debug JavaScript errors, warnings, and application logs.

## Workflow

```
1. tabs_context_mcp → Get/create tab
2. navigate → Go to target page
3. (perform actions that might cause errors)
4. read_console_messages → Capture logs with pattern filter
5. Fix code based on errors
6. Repeat
```

## Example: Debug Page Load Errors

**User prompt:**
```
Open localhost:3000/dashboard and check for any console errors
when the page loads.
```

**Claude's approach:**

### Step 1: Navigate to Page
```
Tool: mcp__claude-in-chrome__navigate
Params: { url: "http://localhost:3000/dashboard", tabId: <tab_id> }
```

### Step 2: Read Console Messages
```
Tool: mcp__claude-in-chrome__read_console_messages
Params: {
  tabId: <tab_id>,
  onlyErrors: true  # Filter for errors only
}
```

### Step 3: Or Filter by Pattern
```
Tool: mcp__claude-in-chrome__read_console_messages
Params: {
  tabId: <tab_id>,
  pattern: "Error|Warning|Failed"
}
```

## Pattern Filtering

The `pattern` parameter accepts regex-compatible patterns:

| Pattern | Captures |
|---------|----------|
| `Error` | Messages containing "Error" |
| `Error\|Warning` | Errors or warnings |
| `\[MyApp\]` | App-specific logs prefixed with [MyApp] |
| `API\|fetch\|request` | Network-related logs |
| `undefined\|null` | Null reference issues |

## Example: Debug API Failures

**User prompt:**
```
The dashboard isn't loading data. Check the console for API errors.
```

```
Tool: mcp__claude-in-chrome__read_console_messages
Params: {
  tabId: <tab_id>,
  pattern: "API|fetch|Failed|error|401|403|404|500"
}
```

## Example: Monitor Specific Component

**User prompt:**
```
Watch for errors from the DataTable component.
```

```
Tool: mcp__claude-in-chrome__read_console_messages
Params: {
  tabId: <tab_id>,
  pattern: "DataTable|table|row|column"
}
```

## Combining with Network Requests

For API debugging, also check network requests:

```
Tool: mcp__claude-in-chrome__read_network_requests
Params: {
  tabId: <tab_id>,
  urlPattern: "/api/"
}
```

This shows:
- Request URLs
- Status codes
- Response times
- Failed requests

## Debug Workflow

```
1. Navigate to problematic page
2. read_console_messages with onlyErrors: true
3. If errors found, analyze and fix
4. If no errors, check read_network_requests
5. If API issues, check backend logs
6. Clear console and retest: read_console_messages with clear: true
```

## Best Practices

1. **Always filter** - Console can be very verbose
2. **Check after actions** - Read console after form submits, button clicks
3. **Use clear: true** - Avoid duplicate messages on subsequent reads
4. **Combine with network** - Console errors often correlate with failed requests
5. **Look for stack traces** - Error messages often include file:line info
