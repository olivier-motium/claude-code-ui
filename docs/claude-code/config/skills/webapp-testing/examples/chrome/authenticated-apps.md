# Chrome: Authenticated Apps Pattern

Testing and automating apps that require login, using Chrome's existing session state.

## Key Advantage

Chrome integration uses your actual browser session:
- Already logged into Google? Claude can access Google Docs, Sheets, Gmail
- Already logged into Notion? Claude can edit pages directly
- No API keys or OAuth flows needed

## Workflow

```
1. Ensure you're logged into the target app in Chrome
2. tabs_context_mcp → Get tab context
3. tabs_create_mcp → Create fresh tab (shares login state)
4. navigate → Go to authenticated page
5. read_page → Verify logged-in state
6. (perform actions)
```

## Example: Google Sheets

**User prompt:**
```
Add a row to my Google Sheet with today's metrics.
```

**Prerequisites:** User is logged into Google in Chrome.

### Step 1: Navigate to Sheet
```
Tool: mcp__claude-in-chrome__navigate
Params: {
  url: "https://docs.google.com/spreadsheets/d/abc123/edit",
  tabId: <tab_id>
}
```

### Step 2: Find Last Row
```
Tool: mcp__claude-in-chrome__read_page
Params: { tabId: <tab_id>, filter: "interactive" }
```

### Step 3: Click Cell and Type
```
Tool: mcp__claude-in-chrome__computer
Params: {
  action: "left_click",
  coordinate: [x, y],  # Cell coordinates
  tabId: <tab_id>
}

Tool: mcp__claude-in-chrome__computer
Params: {
  action: "type",
  text: "2025-01-04",
  tabId: <tab_id>
}
```

## Example: Notion

**User prompt:**
```
Create a new page in my Notion workspace titled "Meeting Notes".
```

### Navigate and Create
```
Tool: mcp__claude-in-chrome__navigate
Params: { url: "https://notion.so", tabId: <tab_id> }

Tool: mcp__claude-in-chrome__find
Params: { query: "new page button", tabId: <tab_id> }

Tool: mcp__claude-in-chrome__computer
Params: { action: "left_click", ref: "<button_ref>", tabId: <tab_id> }

Tool: mcp__claude-in-chrome__computer
Params: { action: "type", text: "Meeting Notes", tabId: <tab_id> }
```

## Example: Gmail

**User prompt:**
```
Check my unread emails and summarize the subjects.
```

```
Tool: mcp__claude-in-chrome__navigate
Params: { url: "https://mail.google.com", tabId: <tab_id> }

Tool: mcp__claude-in-chrome__get_page_text
Params: { tabId: <tab_id> }
```

## Handling Login Prompts

If Claude navigates to a page requiring login:

1. Claude will pause and ask you to log in
2. Log in manually in the browser
3. Tell Claude to continue

**Example interaction:**
```
Claude: "I see a login page. Please log in and tell me when ready."
User: "Done, continue"
Claude: (continues with automation)
```

## Security Considerations

- Claude can only access sites you're already logged into
- Claude sees what you'd see in that browser session
- Sensitive actions (purchases, account changes) require explicit permission
- Use Chrome extension permissions to limit site access

## Supported Apps

Any web app you're logged into works:

| Category | Examples |
|----------|----------|
| Google | Docs, Sheets, Gmail, Calendar, Drive |
| Productivity | Notion, Airtable, Coda, Monday |
| Communication | Slack (web), Discord (web), Teams |
| Development | GitHub, GitLab, Jira, Linear |
| CRM | Salesforce, HubSpot, Pipedrive |

## Best Practices

1. **Verify login state** - Use `read_page` to confirm you're on the authenticated view
2. **Fresh tabs** - Create new tabs to avoid interfering with your work
3. **Check for 2FA** - Some sites may prompt for re-authentication
4. **Be explicit** - Tell Claude exactly what actions to take
5. **Review before submit** - For important actions, ask Claude to pause for confirmation
