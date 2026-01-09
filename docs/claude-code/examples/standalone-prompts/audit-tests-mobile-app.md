Comprehensive audit of the RevOnc Expo React Native test suite to catch bugs before TestFlight/Play Store builds.

---

## Context & Goals

- **Platform:** Expo SDK 52 (React Native) — iOS + Android
- **Build System:** EAS Build → TestFlight (iOS) / Play Store Internal Testing (Android)
- **Navigation:** Expo Router (file-based routing)
- **Storage:** expo-sqlite/kv-store (NOT AsyncStorage or expo-secure-store)
- **E2E Framework:** Maestro (NOT Detox)
- **Goal:** Catch bugs before pushing builds to testers
- **Philosophy:** Green CI = safe to run `eas build` and distribute

---

## PHASE 1: TEST EXECUTION & HEALTH CHECK

### 1.1 Execute Full Test Suite

```bash
# Unit & Component Tests
npm run test:ci

# With coverage report
npm run test:coverage

# Maestro E2E Tests
npm run e2e

# Or run specific Maestro flow
npx maestro test .maestro/flows/auth/login.yaml
```

### 1.2 Verify Jest Configuration

Check `jest.config.js` has correct setup:

```javascript
{
  preset: 'jest-expo',           // Must be jest-expo
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['./jest.setup.js'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',  // Path alias support
    '\\.svg$': '<rootDir>/test-utils/__mocks__/svgMock.js',
  },
}
```

### 1.3 Record Test Results

For every test, record:
- Pass / Fail / Skip status
- Execution time (flag tests > 3 seconds)
- Any `jest-expo` or SDK deprecation warnings
- Mock-related errors

### 1.4 Identify Broken Tests

**Common Failure Categories:**

| Category | Symptom | Fix |
|----------|---------|-----|
| Missing mock | `Cannot find module 'expo-video'` | Add mock to jest.setup.js |
| Stale mock | Mock returns wrong shape | Update mock to match SDK 52 API |
| Async issue | Test times out | Add `await waitFor()` |
| Import error | Module not found | Check path aliases in jest.config.js |

---

## PHASE 2: EXPO MODULE COVERAGE

### 2.1 Modules Actually Used in RevOnc

| Module | Imports | Mock Status | Test Coverage |
|--------|---------|-------------|---------------|
| `expo-router` | 32 | ✅ Mocked | Navigation tested |
| `expo-video` | 3 | ❌ Not mocked | Needs mock for component tests |
| `expo-sqlite/kv-store` | 3 | ✅ Mocked | Storage tested |
| `expo-haptics` | 2 | ✅ Mocked | Feedback tested |
| `expo-av` | 2 | ❌ Not mocked | Audio config untested |
| `expo-symbols` | 1 | ❌ Not mocked | UI only |
| `expo-splash-screen` | 1 | ❌ Not mocked | Handled by Expo |
| `expo-notifications` | 1 | ✅ Mocked | Push token tested |
| `expo-linear-gradient` | 1 | ❌ Not mocked | UI only |
| `expo-keep-awake` | 1 | ❌ Not mocked | Exercise screen |
| `expo-font` | 1 | ❌ Not mocked | Handled by Expo |
| `expo-device` | 1 | ✅ Mocked | Device detection |
| `expo-constants` | 1 | ✅ Mocked | Config access |
| `expo-blur` | 1 | ❌ Not mocked | UI only |

### 2.2 Required Mocks for Full Coverage

Add these to `jest.setup.js` if writing tests for components that use them:

```javascript
// expo-video mock
jest.mock('expo-video', () => ({
  useVideoPlayer: jest.fn(() => ({
    play: jest.fn(),
    pause: jest.fn(),
    muted: false,
    loop: false,
  })),
  VideoView: jest.fn(({ children }) => children),
}));

// expo-av mock (for Audio configuration)
jest.mock('expo-av', () => ({
  Audio: {
    setAudioModeAsync: jest.fn().mockResolvedValue(undefined),
  },
  InterruptionModeIOS: { MixWithOthers: 1 },
  InterruptionModeAndroid: { DuckOthers: 1 },
}));

// expo-keep-awake mock
jest.mock('expo-keep-awake', () => ({
  activateKeepAwakeAsync: jest.fn().mockResolvedValue(undefined),
  deactivateKeepAwake: jest.fn(),
}));

// expo-linear-gradient mock
jest.mock('expo-linear-gradient', () => ({
  LinearGradient: jest.fn(({ children }) => children),
}));

// expo-blur mock
jest.mock('expo-blur', () => ({
  BlurView: jest.fn(({ children }) => children),
}));
```

---

## PHASE 3: EXPO ROUTER COVERAGE

### 3.1 Route Structure

```
app/
├── _layout.tsx                    # Root layout (providers)
├── index.tsx                      # Auth decision router
│
├── (auth)/                        # RequireGuest guard
│   ├── _layout.tsx
│   ├── index.tsx                  # Register
│   └── login.tsx                  # Login
│
├── (onboarding)/                  # RequireOnboarding guard
│   ├── _layout.tsx
│   ├── index.tsx                  # Dream input
│   ├── dreamConfirmation.tsx
│   ├── disclaimer.tsx
│   ├── rejection.tsx
│   ├── onboardingConfirmation.tsx
│   └── questions/
│       ├── index.tsx
│       └── [id].tsx               # Dynamic question
│
├── (app)/                         # RequireAuth + RequireProgramReady guards
│   ├── _layout.tsx                # Tab navigator
│   ├── index.tsx                  # Stepping stones (Tab 1)
│   ├── community.tsx              # Facebook community (Tab 2)
│   ├── profile.tsx                # User stats (Tab 3)
│   ├── settings.tsx
│   └── weeklySummary.tsx
│
├── (exercise)/
│   ├── _layout.tsx
│   ├── [id].tsx                   # Exercise execution
│   └── addCustomExercise.tsx
│
└── (steppingStone)/
    ├── _layout.tsx
    └── details/[id].tsx           # Stone details
```

### 3.2 Route Test Coverage Checklist

| Route | Renders | Navigation | Params | Guard |
|-------|---------|------------|--------|-------|
| `(auth)/login` | ✅ E2E | ✅ E2E | N/A | RequireGuest |
| `(auth)/index` | ✅ E2E | ✅ E2E | N/A | RequireGuest |
| `(onboarding)/questions/[id]` | ✅ E2E | ✅ E2E | ✅ E2E | RequireOnboarding |
| `(app)/index` | ❌ | ❌ | N/A | RequireAuth |
| `(app)/profile` | ❌ | ❌ | N/A | RequireAuth |
| `(exercise)/[id]` | ❌ | ❌ | ❌ | Via app nav |
| `(steppingStone)/details/[id]` | ❌ | ❌ | ❌ | Via app nav |

### 3.3 Expo Router Mock (already in jest.setup.js)

```javascript
jest.mock('expo-router', () => ({
  useRouter: jest.fn(() => ({
    push: jest.fn(),
    replace: jest.fn(),
    back: jest.fn(),
    navigate: jest.fn(),
    canGoBack: jest.fn(() => true),
  })),
  useLocalSearchParams: jest.fn(() => ({})),
  useGlobalSearchParams: jest.fn(() => ({})),
  useSegments: jest.fn(() => []),
  usePathname: jest.fn(() => '/'),
  Link: jest.fn(({ children }) => children),
  Redirect: jest.fn(() => null),
  Stack: jest.fn(({ children }) => children),
  Tabs: jest.fn(({ children }) => children),
}));
```

---

## PHASE 4: CRITICAL PATH COVERAGE

### 4.1 Authentication Flow

| Step | File | Unit Test | E2E Test |
|------|------|-----------|----------|
| Login form | `(auth)/login.tsx` | ❌ | ✅ |
| Register form | `(auth)/index.tsx` | ❌ | ✅ |
| Auth context | `util/contexts/Auth.context.tsx` | ❌ | Via E2E |
| Session query | `api/AuthApi.ts` | ❌ | Via E2E |

### 4.2 Onboarding Flow

| Step | File | Unit Test | E2E Test |
|------|------|-----------|----------|
| Dream input | `(onboarding)/index.tsx` | ❌ | ✅ |
| Questions | `(onboarding)/questions/[id].tsx` | ❌ | ✅ |
| Score calculation | `services/OnboardingDomainService.ts` | ✅ | Via E2E |
| Eligibility check | `services/OnboardingDomainService.ts` | ✅ | Via E2E |
| Program init | `services/ProgramInitializationService.ts` | ✅ | Via E2E |

### 4.3 Exercise Flow (PRIORITY - Most bugs here)

| Step | File | Unit Test | E2E Test |
|------|------|-----------|----------|
| Exercise screen | `(exercise)/[id].tsx` | ❌ | ❌ |
| Timer component | `components/molecules/Timer.tsx` | ❌ | ❌ |
| Background timer | `util/hooks/useBackgroundTimer.ts` | ❌ | ❌ |
| Rep exercise | `components/templates/RepExercise.tsx` | ✅ | ❌ |
| Timer exercise | `components/templates/TimerExercise.tsx` | ✅ | ❌ |
| Exercise completion | `services/ExerciseCompletionService.ts` | ✅ | ❌ |
| XP award | `services/XpService.ts` | ✅ | ❌ |

### 4.4 Program/Droompad Flow

| Step | File | Unit Test | E2E Test |
|------|------|-----------|----------|
| Fetch program | `api/DroompadApi.ts` | ✅ | Via E2E |
| Normalize program | `services/RehabDayService.ts` | ✅ | Via E2E |
| Day completion | `services/RehabDayService.ts` | ✅ | ❌ |

---

## PHASE 5: TEST FILE AUDIT

### 5.1 Existing Test Files (10 total)

| File | Purpose | Tests | Status |
|------|---------|-------|--------|
| `api/__tests__/UserApi.test.ts` | XP, delete account | 7 | ✅ |
| `api/__tests__/OnboardingApi.test.ts` | Score, eligibility | ~10 | ✅ |
| `api/__tests__/DroompadApi.test.ts` | Program fetch | ~5 | ✅ |
| `api/__tests__/initializeProgram.test.ts` | Program init | ~5 | ✅ |
| `components/templates/__tests__/RepExercise.test.tsx` | Rep exercise | 3 | ✅ |
| `components/templates/__tests__/TimerExercise.test.tsx` | Timer exercise | ~3 | ✅ |
| `util/__tests__/Progress.context.test.tsx` | Progress context | 4 | ✅ |
| `util/__tests__/guards.test.tsx` | Route guards | ~4 | ✅ |
| `util/__tests__/exercises.functions.test.ts` | Exercise utils | ~3 | ✅ |
| `services/posthog.test.ts` | Analytics | 5 | ✅ |

### 5.2 Missing Test Files (HIGH PRIORITY)

| File | Why Important | Priority |
|------|---------------|----------|
| `util/hooks/useBackgroundTimer.test.ts` | Critical bug fix area | 🔴 HIGH |
| `components/molecules/Timer.test.tsx` | Timer sync bugs | 🔴 HIGH |
| `services/StorageService.test.ts` | Data persistence | 🟡 MEDIUM |
| `services/AuthService.test.ts` | Login/logout | 🟡 MEDIUM |
| `components/organisms/ExerciseHeader.test.tsx` | Video display | 🟢 LOW |

---

## PHASE 6: E2E COVERAGE (MAESTRO)

### 6.1 Existing Flows

| Flow | File | Steps | Status |
|------|------|-------|--------|
| Login | `.maestro/flows/auth/login.yaml` | 48 | ✅ |
| Onboarding | `.maestro/flows/onboarding/complete-onboarding.yaml` | 142 | ✅ |

### 6.2 Missing E2E Flows

| Flow | Priority | Estimated Steps |
|------|----------|-----------------|
| Exercise execution | 🔴 HIGH | ~30 |
| Day completion | 🔴 HIGH | ~20 |
| Profile/stats view | 🟡 MEDIUM | ~15 |
| Settings/logout | 🟡 MEDIUM | ~10 |
| Weekly summary | 🟢 LOW | ~10 |

### 6.3 Maestro Flow Template

```yaml
# .maestro/flows/exercise/complete-exercise.yaml
appId: be.revonc.mobileapp
---
- launchApp:
    clearState: false  # Keep logged-in state

- assertVisible: "Stepping Stones"  # Verify on home

- tapOn:
    text: ".*current.*"  # Tap current day stone

- assertVisible: "Exercise"

- tapOn: "Start"  # Start timer

- waitForAnimationToEnd

- tapOn: "Next"  # Complete exercise

- assertVisible: "RPE"  # Feedback dialog

- tapOn: "7"  # Select RPE score

- tapOn: "Submit"

- assertVisible: "Completed"
```

---

## PHASE 7: MOCK INTEGRITY CHECK

### 7.1 Verify Mocks Match SDK 52

For each mock in `jest.setup.js`, verify:

| Mock | Current API | SDK 52 API | Match |
|------|-------------|------------|-------|
| `expo-router` | useRouter, Link, Stack | Same | ✅ |
| `expo-notifications` | getExpoPushTokenAsync | Same | ✅ |
| `expo-haptics` | impactAsync, notificationAsync | Same | ✅ |
| `expo-sqlite/kv-store` | Storage.getItem/setItem | Same | ✅ |
| `expo-constants` | expoConfig.extra | Same | ✅ |
| `expo-device` | isDevice | Same | ✅ |

### 7.2 Check Mock Return Types

```javascript
// Verify mock returns match actual SDK types

// BAD - wrong return shape
mockGetExpoPushTokenAsync.mockResolvedValue('token');

// GOOD - correct return shape
mockGetExpoPushTokenAsync.mockResolvedValue({
  data: 'ExponentPushToken[xxx]',
  type: 'expo',
});
```

---

## PHASE 8: CI/CD VERIFICATION

### 8.1 Workflows

| Workflow | Trigger | Tests | Status |
|----------|---------|-------|--------|
| `pr-checks.yml` | PR to main/develop | Lint, typecheck, unit | ✅ |
| `unit-tests.yml` | Push to main/develop | Full suite + coverage | ✅ |
| `e2e-ios.yml` | Push to main | Maestro on iOS sim | ✅ (non-blocking) |
| `e2e-android.yml` | Push to main | Maestro on Android emu | ✅ (non-blocking) |

### 8.2 Coverage Thresholds

Current thresholds in `jest.config.js`:

```javascript
coverageThreshold: {
  global: {
    branches: 15,    // Target: 50%
    functions: 13,   // Target: 50%
    lines: 20,       // Target: 60%
    statements: 20,  // Target: 60%
  },
}
```

---

## OUTPUT FORMAT

### Part A — Test Execution Summary

```
================================================================================
REVONC TEST SUITE RESULTS
================================================================================
Expo SDK: 52.0.0
jest-expo: 52.0.6

Unit Tests:
- Total: X tests in 10 files
- Passing: X (X%)
- Failing: X (X%)
- Skipped: X

E2E Tests (Maestro):
- Login flow: ✅ / ❌
- Onboarding flow: ✅ / ❌

Coverage:
- Branches: X% (threshold: 15%)
- Functions: X% (threshold: 13%)
- Lines: X% (threshold: 20%)
- Statements: X% (threshold: 20%)
```

### Part B — Critical Gaps

```
================================================================================
HIGH-RISK UNTESTED CODE
================================================================================

#1 Timer/Background Timer (util/hooks/useBackgroundTimer.ts)
   Risk: Timer bugs slip through to testers
   Missing:
   - [ ] Timer accuracy test
   - [ ] Background/foreground transition test
   - [ ] Completion callback test

#2 Exercise Screen (app/(exercise)/[id].tsx)
   Risk: Exercise flow bugs
   Missing:
   - [ ] Screen render test
   - [ ] Exercise completion flow
   - [ ] E2E test for full flow
```

### Part C — Remediation Checklist

```
================================================================================
FIX BEFORE NEXT EAS BUILD
================================================================================

BLOCK BUILD:
- [ ] Fix any failing unit tests
- [ ] Fix any failing E2E tests

THIS SPRINT:
- [ ] Add useBackgroundTimer.test.ts
- [ ] Add Timer.test.tsx
- [ ] Add Maestro exercise flow

BACKLOG:
- [ ] Add missing expo module mocks
- [ ] Increase coverage thresholds
- [ ] Add remaining E2E flows
```

---

## Quick Reference Commands

```bash
# Run all tests
npm run test:ci

# Run with coverage
npm run test:coverage

# Run specific test file
npm run test:run -- services/posthog.test.ts

# Run tests matching pattern
npm run test:run -- --testNamePattern="background"

# Run Maestro E2E
npm run e2e

# Run specific Maestro flow
npx maestro test .maestro/flows/auth/login.yaml

# Record new Maestro flow
npm run e2e:record
```

---

## Estimated Effort

| Task | Time |
|------|------|
| Run existing tests, fix failures | 30 min |
| Add useBackgroundTimer tests | 1-2 hours |
| Add Timer component tests | 1 hour |
| Add Maestro exercise flow | 1-2 hours |
| Add missing mocks | 30 min |
| **Total** | **4-6 hours** |
