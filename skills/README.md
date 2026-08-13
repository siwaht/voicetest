# skills

DeepAgents "skills" are markdown instruction files the agent can load at runtime. This directory is mounted into the agent's virtual filesystem at `/skills/`, with write access denied so the agent can read but never modify its own skill definitions.

- **`smoke-test/SKILL.md`** — Triggers when a user asks for a "smoke test". Makes no tool calls or external requests; the agent randomly replies with one of two fixed pass/fail messages, purely to confirm it's alive and responding correctly.
