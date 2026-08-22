import logging
import os

from dotenv import load_dotenv
from deepagents import create_deep_agent, FilesystemPermission, CompiledSubAgent
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from deepagents.backends import StateBackend, StoreBackend, FilesystemBackend, CompositeBackend
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from tools.get_weather import get_weather
from raga import sub_agent_retriever
from llm_config import model


from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    inference,
    llm,
)
from livekit.plugins import langchain, silero

logger = logging.getLogger("deepagents-agent")

load_dotenv()

# model = init_chat_model('openai:gpt-4')

model = model


retriever_agent = CompiledSubAgent(
    name="retriever agent",
    description=(
        "Answers questions about the loaded office document, The Meridian "
        "Office Chronicles. It covers Meridian Analytics: company facts and "
        "policies such as the hybrid schedule and stipends, staff names and "
        "roles, the office layout with its conference rooms and equipment, "
        "and the postmortem of a PulseGrid outage. Use this for any question "
        "about the company, its people, the office, or that incident. It "
        "searches the document and answers only from passages it retrieves."
    ),
    runnable=sub_agent_retriever
)
########################################################################


memory = InMemorySaver()
store = InMemoryStore()
MEMORY_FILE = '/memories/AGENTS.md'

backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/memories/":StoreBackend(namespace= lambda _rt: ("use_one",),store=store),
        "/skills/": FilesystemBackend(root_dir='./skills/',virtual_mode=True),
        "/memories/": FilesystemBackend(root_dir='./memories/',virtual_mode=True)
    }
)


def create_graph():
    # NOTE: a checkpointer IS used here, so every graph call must supply a
    # `thread_id` in the RunnableConfig (see `config` in entrypoint below).
    # Be aware LiveKit also replays the full ChatContext each turn, so history
    # is tracked in two places; revisit if replies start repeating context.
    return create_deep_agent(
        # model="openai:gpt-4.1-mini",
        model=model,
        system_prompt="You are a friendly assistant.",
        tools=[get_weather],
        checkpointer=memory,
        store=store,
        memory=[MEMORY_FILE],
        backend=backend,
        skills=["/skills/"],
        subagents=[retriever_agent],
        permissions=[
            FilesystemPermission(
                operations=['write'],
                paths=['/skills/**'],
                mode='deny'
            )
        ]

    )





























def _pending_input(chat_ctx):
    """Build LangGraph input from only the turns the graph hasn't seen yet.

    There are two independent stores of conversation history here. The graph is
    compiled with a checkpointer and invoked with a per-room `thread_id`, so
    LangGraph already persists every earlier turn. LiveKit separately replays its
    entire ChatContext into `llm_node` on every single turn.

    Passing that full replay to the graph (as the previous
    `_chat_ctx_to_state` did) appends a fresh copy of the whole conversation to
    the checkpointed state each turn. History then grows quadratically and the
    model is shown every exchange multiple times, which on a phone call means
    rising latency, rising token spend, and an agent that repeats itself.

    So this walks backwards from the newest message and stops at the last
    assistant turn, keeping only what arrived after it. The checkpointer stays
    the single source of truth for anything older.

    Assistant messages are deliberately never forwarded: the graph writes its own
    AI turns into the checkpoint as it produces them.
    """
    pending = []
    for msg in reversed(list(chat_ctx.messages())):
        # Everything at or before the last assistant turn is already checkpointed.
        if msg.role == "assistant":
            break
        content = msg.raw_text_content
        if not content:
            continue
        if msg.role == "user":
            pending.append(HumanMessage(content=content, id=msg.id))
        elif msg.role in ("system", "developer"):
            # Per-turn instructions, e.g. from session.generate_reply(instructions=...).
            # The agent's standing persona lives in the graph's own system_prompt.
            #
            # CAUTION: the deep agent treats an inbound SystemMessage as content to
            # respond to, not as a directive, so anything routed through here can be
            # spoken aloud verbatim or paraphrased. That is exactly what happened to
            # the old greeting. Prefer session.say() for fixed lines, and only use
            # generate_reply(instructions=...) if you have confirmed the wording is
            # safe to be overheard by the caller.
            pending.append(SystemMessage(content=content, id=msg.id))
    pending.reverse()
    return {"messages": pending}


class DeepAgentVoice(Agent):
    def __init__(self, graph, config=None) -> None:
        super().__init__(instructions="")
        self._graph = graph
        self._config = config

    async def llm_node(self, chat_ctx, tools, model_settings=None):
        state = _pending_input(chat_ctx)

        async for item in self._graph.astream(
            state, self._config, stream_mode="messages"
        ):
            # stream_mode="messages" yields (chunk, metadata)
            chunk = item[0] if isinstance(item, tuple) else item
            if not isinstance(chunk, AIMessageChunk):
                continue
            text = chunk.text
            if not text:
                continue
            logger.debug("yielding AI chunk: %r", text)
            yield llm.ChatChunk(
                id=chunk.id or "",
                delta=llm.ChoiceDelta(role="assistant", content=text),
            )


# num_idle_processes caps the pool of pre-forked job processes. The framework's
# production default is 16 (dev is 0, which is why this never bites locally).
# Every idle process imports this module, so it pays for a Silero VAD load *and*
# a full raga.py vector-store build -- 16 of them means 16x the memory and 16
# rounds of OpenAI embedding calls on every deploy. One is enough for a small
# instance; raise it once the box has cores to spare.
_server_options: dict[str, object] = {"num_idle_processes": 1}

# The agent dials out to LiveKit over a WebSocket and never serves inbound
# traffic, so it belongs in a Render *background worker*, which needs no port.
# If it's run as a Render *web* service instead, Render assigns a port via $PORT
# and fails the deploy when nothing listens on it. AgentServer's health check
# endpoint is the only HTTP surface the agent has, so bind it to $PORT whenever
# the platform provides one. With $PORT unset the framework defaults apply
# (8081 in production mode, a random free port under `dev`).
_port = os.getenv("PORT")
if _port:
    _server_options.update(host="0.0.0.0", port=int(_port))

server = AgentServer(**_server_options)  # type: ignore[arg-type]


def prewarm(proc: JobProcess):
    # Load the VAD model once per process to cut connection latency.
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


# agent_name switches this worker to explicit dispatch: it stops auto-joining
# every room in the project and is instead requested by name -- by the SIP
# dispatch rule for phone calls, or via room config in a client token. Telephony
# needs the name, since a dispatch rule targets agents by `agentName`.
@server.rtc_session(agent_name="phone-agent")
async def entrypoint(ctx: JobContext):
    # Build the graph once and share it between the Agent and the session's
    # LLMAdapter, so both operate on the same compiled graph / checkpointer.
    graph = create_graph()
    # One checkpointer thread per room rather than a single global "1". The SIP
    # dispatch rule puts every caller in their own room, so keying on the room
    # name scopes conversation memory to one call instead of leaking it across
    # every caller that ever dials in.
    config = {"configurable": {"thread_id": ctx.room.name}}

    # Instructions and tool calls live in the deep agent graph, so the
    # Agent's own instructions can be left empty. The custom llm_node above
    # streams from the graph and filters to AIMessageChunk only, so tool
    # results and intermediate messages are never spoken.
    agent = DeepAgentVoice(graph=graph, config=config)

    session = AgentSession(
        # Plain Flux (English) instead of flux-general-multi. LiveKit Inference
        # serves `deepgram/flux-general` from its Mumbai deployment, which is
        # co-located with this project's India West region, so the STT round trip
        # stays in-region. flux-general-multi is not in the co-located set.
        stt=inference.STT("deepgram/flux-general", language="en"),

        # The LLM stays on Cloudflare Workers AI via llm_config.model, wrapped in
        # the deep agent graph. Only STT and TTS go through LiveKit Inference,
        # which is why no Deepgram or Inworld API key is needed.
        llm=langchain.LLMAdapter(graph=graph, config=config),

        # Every elevenlabs/* model in LiveKit Inference is deprecated and retires
        # 2026-08-31, which is why this is no longer eleven_flash_v2_5. Inworld is
        # what testagent.py already uses.
        tts=inference.TTS("inworld/inworld-tts-2", voice="Ashley"),

        vad=ctx.proc.userdata["vad"],
        expressive=True

    )

    await session.start(agent=agent, room=ctx.room)

    # A fixed greeting, not generate_reply(instructions=...). Both reasons were
    # observed in a real session transcript in LiveKit's agent insights:
    #   1. Feeding the instructions through the deep agent made it paraphrase them
    #      out loud. It greeted with "really amiable and welcoming Hey, how are
    #      you doing today?", speaking its own directive back to the caller.
    #   2. That path measured 6.9s time-to-first-token. On a phone call that is
    #      long enough that the caller assumes the line is dead.
    # say() bypasses the LLM, so the first words arrive as fast as TTS can render
    # them. add_to_chat_ctx defaults to True, so this still lands in the chat
    # context as the assistant turn that _pending_input treats as its boundary.
    await session.say("Hi, thanks for calling. How can I help you today?")


if __name__ == "__main__":
    agents.cli.run_app(server)
