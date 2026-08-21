import logging
import os

from dotenv import load_dotenv
from deepagents import create_deep_agent, FilesystemPermission, CompiledSubAgent
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
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





























def _chat_ctx_to_state(chat_ctx):
    """Convert LiveKit's ChatContext into LangGraph input.

    Mirrors LangGraphStream._chat_ctx_to_state in livekit-plugins-langchain.
    Done inline so we don't instantiate an LLMAdapter/LLMStream here: building
    an LLMStream immediately spawns a task that runs the graph, which would
    execute the graph a second time and race the checkpointer on thread_id.
    """
    messages = []
    for msg in chat_ctx.messages():
        content = msg.raw_text_content
        if not content:
            continue
        if msg.role == "assistant":
            messages.append(AIMessage(content=content, id=msg.id))
        elif msg.role == "user":
            messages.append(HumanMessage(content=content, id=msg.id))
        elif msg.role in ("system", "developer"):
            messages.append(SystemMessage(content=content, id=msg.id))
    return {"messages": messages}


class DeepAgentVoice(Agent):
    def __init__(self, graph, config=None) -> None:
        super().__init__(instructions="")
        self._graph = graph
        self._config = config

    async def llm_node(self, chat_ctx, tools, model_settings=None):
        state = _chat_ctx_to_state(chat_ctx)

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


# The agent dials out to LiveKit over a WebSocket and never serves inbound
# traffic, but Render runs this as a *web* service: it assigns a port via $PORT
# and fails the deploy if nothing listens on it. AgentServer's built-in health
# check endpoint is the only HTTP surface the agent has, so bind it to $PORT
# when the platform provides one. Locally, where $PORT is unset, keep the
# framework defaults (8081 in production mode, a random free port under `dev`).
_port = os.getenv("PORT")
server = AgentServer(host="0.0.0.0", port=int(_port)) if _port else AgentServer()


def prewarm(proc: JobProcess):
    # Load the VAD model once per process to cut connection latency.
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Build the graph once and share it between the Agent and the session's
    # LLMAdapter, so both operate on the same compiled graph / checkpointer.
    graph = create_graph()
    config = {"configurable": {"thread_id": "1"}}

    # Instructions and tool calls live in the deep agent graph, so the
    # Agent's own instructions can be left empty. The custom llm_node above
    # streams from the graph and filters to AIMessageChunk only, so tool
    # results and intermediate messages are never spoken.
    agent = DeepAgentVoice(graph=graph, config=config)

    session = AgentSession(
        stt=inference.STT("deepgram/flux-general-multi"),

        llm=langchain.LLMAdapter(graph=graph, config=config),

        tts=inference.TTS("elevenlabs/eleven_flash_v2_5"),

        vad=ctx.proc.userdata["vad"],
        expressive=True

    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="Ask the user how they're doing.")


if __name__ == "__main__":
    agents.cli.run_app(server)
