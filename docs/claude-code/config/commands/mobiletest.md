---
description: Run Maestro E2E tests and diagnose failures with fix suggestions
argument-hint: [android|ios|both] [suite-path]
allowed-tools: Bash(./scripts/mobiletest.sh:*), Bash(maestro:*), Bash(cat:*), Bash(ls:*), Bash(find:*), Bash(tail:*)
---

# /mobiletest

Use ultrathink to thoroughly analyze test results and propose targeted fixes.

## Prerequisites

- Maestro CLI installed (`brew install maestro`)
- Simulator/emulator running or device connected
- Test flows in `maestro/e2e/` directory
- Project script at `scripts/mobiletest.sh`

## Run Tests

Test output:
!`./scripts/mobiletest.sh $ARGUMENTS 2>&1 || true`

## Artifacts

- JUnit report: @artifacts/mobiletest/report.xml
- Maestro log: @artifacts/mobiletest/maestro.log

## Your Task

1. **Parse Results**: Summarize pass/fail by flow and platform
2. **Diagnose Failures**: For each failure:
   - Identify the exact step and selector that failed
   - Analyze log context for root cause
   - Check if it's a timing issue, missing element, or app bug
3. **Propose Fixes**: Suggest specific changes:
   - App code fixes (component, navigation, state)
   - Maestro flow improvements (better selectors, waits, testIDs)
4. **Iterate**: If you propose code changes, implement them and rerun:
   ```bash
   ./scripts/mobiletest.sh $ARGUMENTS
   ```
   Continue until tests pass or you need user input.

## Output Format

```
## Test Summary
- Total: X flows
- Passed: Y
- Failed: Z

## Failures

### Flow: <flow-name>
**Failed Step**: <step description>
**Selector**: <id/text/accessibilityLabel used>
**Error**: <Maestro error message>
**Root Cause**: <analysis>
**Fix**:
- [ ] App change: <specific code change>
- [ ] Flow change: <Maestro yaml change>
```

---

## Project Setup

Your project needs these files (not part of toolkit):

### `scripts/mobiletest.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/mobiletest.sh [android|ios|both] [suite-path]
PLATFORM="${1:-both}"
SUITE="${2:-maestro/e2e}"

boot_simulator() {
    local platform="$1"
    case "$platform" in
        ios)
            echo "Booting iOS Simulator..."
            xcrun simctl boot "iPhone 15 Pro" 2>/dev/null || true
            xcrun simctl bootstatus "iPhone 15 Pro" -b
            ;;
        android)
            echo "Booting Android Emulator..."
            if ! adb devices | grep -q "emulator"; then
                emulator -avd "Pixel_7_API_34" -no-window &
                adb wait-for-device
                sleep 10
            fi
            ;;
        both)
            boot_simulator ios
            boot_simulator android
            ;;
    esac
}

run_tests() {
    local platform="$1"

    echo "Running tests on $platform..."
    # Maestro 2.x auto-detects running simulator/emulator
    maestro test "$SUITE" \
        --format junit \
        --output "artifacts/mobiletest/report_${platform}.xml" \
        2>&1 | tee -a artifacts/mobiletest/maestro.log || true
}

rm -rf artifacts/mobiletest
mkdir -p artifacts/mobiletest

boot_simulator "$PLATFORM"

if [ "$PLATFORM" = "both" ]; then
    run_tests ios
    run_tests android
else
    run_tests "$PLATFORM"
fi

echo ""
echo "Test artifacts written to artifacts/mobiletest/"
```

### Example Flow (`maestro/e2e/login.yaml`)

```yaml
appId: com.yourapp
---
- launchApp
- tapOn:
    id: "email-input"
- inputText: "test@example.com"
- tapOn:
    id: "password-input"
- inputText: "password123"
- tapOn:
    id: "login-button"
- assertVisible:
    id: "dashboard-screen"
```

---

## Maestro MCP (Advanced)

If your project has `.mcp.json` with Maestro server configured, you can use live inspection tools:

| Tool | Purpose |
|------|---------|
| `maestro_take_screenshot` | Capture current screen state |
| `maestro_view_hierarchy` | Get accessibility tree |
| `maestro_tap` | Tap element by text/id |
| `maestro_input_text` | Enter text in focused field |
| `maestro_launch_app` | Start the app |
| `maestro_run_flow` | Execute a Maestro flow file |

Use MCP for investigating specific failures found in batch runs.
