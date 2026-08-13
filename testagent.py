from dotenv import load_dotenv

from deepagents import create_deep_agent
from langchain_core.tools import tool

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    TurnHandlingOptions,
    inference,
)
from livekit.plugins import langchain

load_dotenv(".env.local")
load_dotenv(".env")

# 1. tools for the deep agent (it also gets todo + filesystem tools for free)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's 22 degrees and sunny in {city}."

# 2. the deep agent -> a CompiledStateGraph
def build_graph():
    return create_deep_agent(
        model="openai:gpt-4.1-mini",
        tools=[get_weather],
        system_prompt=(
            "You help users over voice. Use your planning tools for multi-step "
            "work, then answer in one or two short spoken sentences."
        ),
    )

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Speak in short plain sentences. No markdown, lists, or emojis."
        )

server = AgentServer()

@server.rtc_session(agent_name="deep-agent")
async def deep_agent_session(ctx: agents.JobContext):
    session = AgentSession(
        # 3. this single argument is the entire integration
        llm=langchain.LLMAdapter(graph=build_graph()),
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        tts=inference.TTS(model="inworld/inworld-tts-2", voice="Ashley"),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )

    await session.start(agent=Assistant(), room=ctx.room)
    await session.generate_reply(instructions="Greet the user briefly.")

if __name__ == "__main__":
    agents.cli.run_app(server)
