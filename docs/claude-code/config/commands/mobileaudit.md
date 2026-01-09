---
description: Capture screenshots via Maestro and perform vision-based UI/design audit
argument-hint: [android|ios|both] [suite-path]
allowed-tools: Bash(./scripts/mobileaudit.sh:*), Bash(maestro:*), Bash(cat:*), Bash(ls:*), Bash(find:*)
---

# /mobileaudit

Use ultrathink to thoroughly analyze screenshots for UI/design issues.

## Prerequisites

- Maestro CLI installed (`brew install maestro`)
- Simulator/emulator running or device connected
- Audit flows in `maestro/audit/` with `takeScreenshot` commands
- Project script at `scripts/mobileaudit.sh`

## Capture Screenshots

Audit output:
!`./scripts/mobileaudit.sh $ARGUMENTS 2>&1 || true`

## Artifacts

- Screenshot manifest: @artifacts/mobileaudit/screenshots.txt
- Maestro log: @artifacts/mobileaudit/maestro.log

## Your Task (Visual Audit)

For each screenshot in the manifest:

1. **Open and analyze** the image using vision capabilities
2. **Identify UI/design issues** that logs cannot show:
   - Spacing/alignment inconsistencies
   - Truncation, overflow, clipped content
   - Safe area violations (notch, home indicator)
   - Contrast/readability problems
   - Platform inconsistencies (iOS vs Android)
   - Wrong component states (loading, disabled, error)
   - Missing or broken images/icons
3. **Cross-reference** with design system if `docs/ui-audit-rules.md` exists

## Iteration Contract

**First run**: Report only (no code changes)
- Generate prioritized findings
- Wait for user to select which issues to fix

**After user selection**:
- Implement minimal code changes
- Rerun `./scripts/mobileaudit.sh $ARGUMENTS`
- Verify improvements in new screenshots

## Output Format

```
## Audit Summary
- Screenshots analyzed: X
- Issues found: Y (Z critical, W major, V minor)

## Findings

### Finding #1
**Screenshot**: <filename>
**Location**: <describe UI location>
**Severity**: Critical | Major | Minor
**Issue**: <description of visual problem>
**Root Cause**: <hypothesis - style, layout, token, etc.>
**Fix**:
```tsx
// File: <path>
// Change:
<code snippet>
```

### Finding #2
...

## Recommendations
- Top 3 issues to fix first
- Patterns to add to linter/design system
```

## Design System Reference

If your project has `docs/ui-audit-rules.md`, include:
@docs/ui-audit-rules.md

This ensures audit findings align with your established design standards.

---

## Project Setup

Your project needs these files (not part of toolkit):

### `scripts/mobileaudit.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/mobileaudit.sh [android|ios|both] [suite-path]
PLATFORM="${1:-both}"
SUITE="${2:-maestro/audit}"

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

run_audit() {
    local platform="$1"

    echo "Capturing screenshots on $platform..."
    # Maestro 2.x auto-detects running simulator/emulator
    maestro test "$SUITE" \
        --format junit \
        --output "artifacts/mobileaudit/report_${platform}.xml" \
        2>&1 | tee -a artifacts/mobileaudit/maestro.log || true
}

rm -rf artifacts/mobileaudit
mkdir -p artifacts/mobileaudit

boot_simulator "$PLATFORM"

if [ "$PLATFORM" = "both" ]; then
    run_audit ios
    run_audit android
else
    run_audit "$PLATFORM"
fi

find artifacts/mobileaudit -name "*.png" -type f | sort > artifacts/mobileaudit/screenshots.txt

echo ""
echo "Audit artifacts written to artifacts/mobileaudit/"
echo "Screenshots captured: $(wc -l < artifacts/mobileaudit/screenshots.txt)"
```

### Example Flow (`maestro/audit/onboarding.yaml`)

```yaml
appId: com.yourapp
---
- launchApp
- takeScreenshot: onboarding_welcome
- tapOn: "Get Started"
- takeScreenshot: onboarding_step1
- tapOn: "Continue"
- takeScreenshot: onboarding_step2_light
# Switch to dark mode
- tapOn:
    id: "settings-button"
- tapOn: "Dark Mode"
- back
- takeScreenshot: onboarding_step2_dark
```

---

## Maestro MCP (Advanced)

If your project has `.mcp.json` with Maestro server configured, you can use live inspection tools for deeper investigation:

| Tool | Purpose |
|------|---------|
| `maestro_take_screenshot` | Capture current screen state |
| `maestro_view_hierarchy` | Get accessibility tree |
| `maestro_tap` | Tap element by text/id |
| `maestro_input_text` | Enter text in focused field |
| `maestro_launch_app` | Start the app |
| `maestro_run_flow` | Execute a Maestro flow file |

### Hybrid Workflow

1. Run `/mobileaudit` (batch) to capture baseline screenshots
2. If you identify a suspicious area, use MCP to:
   - Navigate to that screen
   - Take additional screenshots at different states
   - Inspect view hierarchy for element details
3. Propose fixes based on combined evidence
