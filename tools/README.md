# tools

Custom tools exposed to the deep agent, on top of the planning/filesystem tools that DeepAgents provides automatically.

- **`get_weather.py`** — `get_weather(city: str)`. Returns a canned, hardcoded weather string for the given city. It's a placeholder for wiring up a real weather API.
- **`search_document.py`** — `search_document(query: str)`. Queries the in-memory vector store built in `raga.py` and returns the most relevant passage(s) from the loaded document (`files/office.txt`). Used by the `retriever agent` subagent defined in `agent.py` to ground its answers instead of guessing.
