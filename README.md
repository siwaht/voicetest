# voicetest

A real-time, speech-to-speech AI voice agent built with [LiveKit Agents](https://docs.livekit.io/agents/) and LangChain's [DeepAgents](https://github.com/langchain-ai/deepagents) (LangGraph). You talk to it out loud; it reasons with a planning-capable "deep agent" that can call tools and delegate to a retrieval subagent, then speaks its answer back.

Created by **Asif Shah**.

## Scope

This is a reference implementation of a real-time voice agent. It demonstrates the full speech-to-speech pipeline and the deep-agent architecture behind it in a form that runs standalone, with a small sample corpus and a mocked tool so the whole thing is inspectable end to end.

Production voice deployments built on this architecture are covered by confidentiality agreements and are not published here.

## What the agent actually does

- **Real-time voice conversation.** Audio in, audio out, over a LiveKit room. Speech is transcribed, reasoned about, and the reply is synthesized back to speech.
- **A "deep agent" brain, not just a chat model.** The core LLM is wrapped with `deepagents.create_deep_agent`, which gives it planning/todo tools, a virtual filesystem, persistent memory, and the ability to delegate to subagents — on top of whatever custom tools it's given.
- **Document-grounded Q&A (RAG).** A dedicated `retriever agent` subagent answers questions about a loaded document by searching an in-memory vector store (built in `raga.py` from `files/office.txt`) and only responds using passages it actually retrieved, instead of guessing.
- **Tool use.** `get_weather` (in `tools/get_weather.py`) is the only custom tool on the main agent; it returns a canned, hardcoded response and exists to show how a tool plugs in. The retriever subagent carries its own tool, `search_document`, defined in `raga.py`. Both sit on top of the planning, filesystem and skill tools DeepAgents supplies automatically.
- **A deliberate connectivity check, not a test suite.** Asking the agent to "run a smoke test" triggers `skills/smoke-test/SKILL.md`. By design it calls no tools and makes no external requests: the skill instructs the model to pick pass or fail at random and reply with one of two fixed strings. Its only purpose is to confirm the loop is wired up — that the skill was discovered and loaded from the virtual filesystem, and that a reply makes it back through TTS. Hearing "smoke test successful, hurray" proves the plumbing, and nothing about the agent's answers.
- **Memory.** Conversation state is checkpointed per session (LangGraph `InMemorySaver`), and a separate long-term memory store is routed to a virtual `/memories/AGENTS.md` file.
- **Virtual filesystem backend.** `files/` and `skills/` are mounted into the agent's own virtual filesystem (via `CompositeBackend`/`FilesystemBackend`), with write access explicitly denied on `/skills/**` so the agent can read but not modify its own skill definitions.

## Architecture

```
Caller audio
   │  (LiveKit room)
   ▼
Silero VAD  ──►  Deepgram STT (speech → text)
   │
   ▼
DeepAgents graph (LangGraph)
   │  model: Cloudflare Workers AI, @cf/zai-org/glm-5.2 (OpenAI-compatible endpoint)
   │  tools: get_weather
   │  skills: /skills/smoke-test
   │  subagent: "retriever agent" (same model) → search_document tool → in-memory vector store (files/office.txt)
   │  memory: InMemorySaver (per-session) + InMemoryStore (/memories/AGENTS.md)
   ▼
AIMessageChunk stream (text → text)
   │
   ▼
ElevenLabs TTS (text → speech)  ──►  Caller audio
```

`agent.py` is the main entrypoint. It builds the DeepAgents graph (`create_graph`), wires it into a LiveKit `AgentSession` as the LLM (via `livekit.plugins.langchain.LLMAdapter`), and streams only assistant-authored chunks back through TTS so tool calls and intermediate reasoning are never spoken aloud.

`raga.py` builds the retriever used for RAG: it loads `files/office.txt`, splits it with `RecursiveCharacterTextSplitter` (chunk size 500, overlap 100), embeds the chunks with `OpenAIEmbeddings`, and indexes them in an `InMemoryVectorStore`. It also defines the `search_document` tool that wraps that retriever, and builds `sub_agent_retriever` around it. `agent.py` then wraps that as the `retriever agent` `CompiledSubAgent`, which is how the deep agent delegates document questions to it.

`testagent.py` is a second, simpler LiveKit entrypoint (`gpt-4.1-mini`, Inworld TTS) kept as a lighter experimental/reference agent alongside the main one.

`demo.ipynb` is the notebook where this pipeline (retriever → subagent → deep agent → backend) was originally prototyped, cell by cell, before being productionized into `agent.py`.

## From reference to production

The gaps below are deliberate: they keep the repository runnable without standing up any external infrastructure.

| Reference implementation | Production requires |
| --- | --- |
| `InMemoryVectorStore`, rebuilt and re-embedded on every process start | Persistent vector database, indexed once |
| Single-file corpus (`files/office.txt`) with fixed 500/100 chunking | Ingestion pipeline handling many documents, updates and re-indexing |
| `InMemorySaver` checkpointer | Durable checkpoint store shared across workers |
| `InMemoryStore` for `/memories/AGENTS.md` | Persistent memory backend that survives restarts |
| Hardcoded `thread_id: "1"` | Thread IDs scoped per caller and per session |
| Turn history kept in both the LiveKit `ChatContext` and the graph checkpointer | One source of truth for conversation history |
| `get_weather` returns a canned string | Real integrations with auth, timeouts and error handling |
| Corpus and skills read from local `./files/` and `./skills/` | Shared storage every worker can reach |
| Smoke-test skill picks pass/fail at random | Health checks that actually probe dependencies |
| One local session at a time (`agent.py console`) | Concurrent sessions, worker pooling, telephony ingress |
| Secrets in a local `.env` | Managed secret storage |
| Module-level `logging` only | Per-session tracing and structured logs |

## Why these components

- **Silero VAD** — decides when the caller has started and stopped speaking, so the graph is invoked on whole utterances instead of mid-sentence fragments.
- **Deepgram (`nova-3`, multilingual) for STT** — streaming transcription, so text reaches the agent while the caller is still talking.
- **ElevenLabs (`eleven_turbo_v2_5`) for TTS** — streaming synthesis fed by partial `AIMessageChunk`s, so playback can begin before the reply is fully generated.
- **LiveKit for transport** — carries audio both ways and owns the room, WebRTC and telephony ingress, so the agent process only ever sees an audio stream; its `LLMAdapter` is also the seam that lets a LangGraph graph stand in where a chat model normally goes.
- **DeepAgents for the reasoning layer** — supplies planning/todo tools, a virtual filesystem, skills, memory and subagent delegation, so multi-step work resolves inside one graph invocation rather than being orchestrated turn by turn from the voice loop.

## Project structure

```
voicetest/
├── agent.py               # Main LiveKit voice agent entrypoint
├── testagent.py            # Secondary/experimental LiveKit agent entrypoint
├── raga.py                  # Builds the document retriever used for RAG
├── main.py                  # Intentionally empty — not part of the runtime; the entrypoints are agent.py and testagent.py
├── demo.ipynb                # Notebook prototype of the agent pipeline
├── tools/
│   └── get_weather.py        # Custom tool: canned weather response (mock)
├── skills/
│   └── smoke-test/SKILL.md    # Connectivity sanity-check skill (no tools, no external calls)
├── files/
│   ├── office.txt              # Sample knowledge-base document (RAG corpus)
│   └── story.txt               # Sample narrative document
├── pyproject.toml / uv.lock / requirements.txt / .python-version
├── .env.example                # Template for required environment variables
└── .env                        # Local secrets/config — not committed
```

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- API keys/credentials for:
  - **Cloudflare Workers AI** — `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` (hosts the LLM used by the deep agent)
  - **OpenAI** — `OPENAI_API_KEY` (used for embeddings in the document retriever)
  - **Deepgram** — `DEEPGRAM_API_KEY` (speech-to-text)
  - **ElevenLabs** — `ELEVENLABS_API_KEY` (text-to-speech)
  - **LiveKit** — `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (real-time transport/room infrastructure)

## Setup

```bash
uv sync
```

or, with plain pip:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the keys listed above.

## Running the agent

LiveKit Agents ships a small CLI on top of the entrypoint file:

```bash
# Talk to the agent directly in your terminal, no LiveKit room needed
uv run python agent.py console

# Dev mode: connect a frontend or telephony number to a LiveKit room
uv run python agent.py dev

# Production
uv run python agent.py start
```

The experimental agent runs the same way: `uv run python testagent.py console`.

## How a conversation flows

1. LiveKit captures the caller's audio; Silero VAD detects when someone is speaking.
2. Deepgram transcribes the speech to text.
3. The transcript becomes a `HumanMessage` fed into the DeepAgents graph.
4. The deep agent plans, optionally calls `get_weather`, or delegates to the `retriever agent` subagent, which calls `search_document` to pull grounded passages from `files/office.txt` before answering.
5. The reply streams back as `AIMessageChunk`s, which ElevenLabs converts to speech for the caller.

## Author

Created by **Asif Shah**.
