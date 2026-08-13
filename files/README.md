# files

Sample documents used as the knowledge base for the agent's retrieval-augmented (RAG) question answering. This directory is mounted into the agent's virtual filesystem at `/files/`.

- **`office.txt`** — A fictional profile of "Meridian Analytics," a fictional software company, covering its people, org structure, and product. This is the document `raga.py` indexes and that the `search_document` tool searches over.
- **`story.txt`** — A short fictional narrative about three coworkers having lunch. Extra sample text for retrieval experiments.
