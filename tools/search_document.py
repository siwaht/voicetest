from langchain.tools import tool
from raga import retriever

@tool
def search_document(query:str)-> str:
    """return relavant passage from the loaded document."""
    found = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in found)