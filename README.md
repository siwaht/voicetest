# voicetest

A real-time, speech-to-speech AI voice agent built with [LiveKit Agents](https://docs.livekit.io/agents/) and LangChain's [DeepAgents](https://github.com/langchain-ai/deepagents) (LangGraph). You talk to it out loud; it reasons with a planning-capable "deep agent" that can call tools and delegate to a retrieval subagent, then speaks its answer back.

Created by **Asif Shah**.

## What the agent actually does

- **Real-time voice conversation.** Audio in, audio out, over a LiveKit room. Speech is transcribed, reasoned about, and the reply is synthesized back to speech.
- **A "deep agent" brain, not just a chat model.** The core LLM is wrapped with `deepagents.create_deep_agent`, which gives it planning/todo tools, a virtual filesystem, persistent memory, and the ability to delegate to subagents — on top of whatever custom tools it's given.
- **Document-grounded Q&A (RAG).** A dedicated `retriever agent` subagent answers questions about a loaded document by searching an in-memory vector store (built in `raga.py` from `files/office.txt`) and only responds using passages it actually retrieved, instead of guessing.
- **Tool use.** A `get_weather` tool is wired in (currently a mocked/canned response) to demonstrate how tools plug into the agent.
- **A "smoke test" skill.** Asking the agent to "run a smoke test" triggers `skills/smoke-test/SKILL.md`, a trivial connectivity check that replies with a fixed pass/fail message and makes no external calls — useful for confirming the agent is wired up correctly.
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
   │  model: Cloudflare Workers AI (GLM-5.2, OpenAI-compatible endpoint)
   │  tools: get_weather
   │  skills: /skills/smoke-test
   │  subagent: "retriever agent" → search_document tool → in-memory vector store (files/office.txt)
   │  memory: InMemorySaver (per-session) + InMemoryStore (/memories/AGENTS.md)
   ▼
AIMessageChunk stream (text → text)
   │
   ▼
ElevenLabs TTS (text → speech)  ──►  Caller audio
```

`agent.py` is the main entrypoint. It builds the DeepAgents graph (`create_graph`), wires it into a LiveKit `AgentSession` as the LLM (via `livekit.plugins.langchain.LLMAdapter`), and streams only assistant-authored chunks back through TTS so tool calls and intermediate reasoning are never spoken aloud.

`raga.py` builds the retriever used for RAG: it loads `files/office.txt`, splits it with `RecursiveCharacterTextSplitter` (chunk size 500, overlap 100), embeds the chunks with `OpenAIEmbeddings`, and indexes them in an `InMemoryVectorStore`. `tools/search_document.py` wraps that retriever as a LangChain tool, which the `retriever agent` subagent (defined in `agent.py`) uses to ground its answers.

`testagent.py` is a second, simpler LiveKit entrypoint (`gpt-4.1-mini`, Inworld TTS) kept as a lighter experimental/reference agent alongside the main one.

`demo.ipynb` is the notebook where this pipeline (retriever → subagent → deep agent → backend) was originally prototyped, cell by cell, before being productionized into `agent.py`.

## Project structure

```
voicetest/
├── agent.py               # Main LiveKit voice agent entrypoint
├── testagent.py            # Secondary/experimental LiveKit agent entrypoint
├── raga.py                  # Builds the document retriever used for RAG
├── main.py                  # Unused placeholder (currently empty)
├── demo.ipynb                # Notebook prototype of the agent pipeline
├── tools/
│   ├── get_weather.py        # Example/mock tool
│   └── search_document.py    # RAG tool used by the retriever subagent
├── skills/
│   └── smoke-test/SKILL.md    # Connectivity sanity-check skill
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
