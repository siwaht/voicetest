---
name: smoke-test
description: >-
  Use this skill when the user asks to run a "smoke test", "smoketest", or
  asks you to "do the smoke test". Performs a trivial randomized pass/fail
  check to confirm the agent is wired up and responding correctly.
---

# Smoke Test

This is a minimal connectivity/sanity check skill. It does not call any tools
or external systems. When triggered, randomly pick pass or fail (roughly 50/50)
and respond with exactly one of the following, and nothing else:

- `smoke test successful, hurray`
- `smoke test fail, disappointment`

## Instructions

1. Pick one of the two outcomes at random (e.g. flip a mental coin, or use any
   available randomness such as the current time's parity).
2. Reply with only the corresponding message below, verbatim, with no extra
   commentary, punctuation, or explanation:
   - `smoke test successful, hurray`
   - `smoke test fail, disappointment`
