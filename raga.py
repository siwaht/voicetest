from pathlib import Path
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain.tools import tool
from llm_config import model
from dotenv import load_dotenv

load_dotenv()

model =model

source_pth = Path("./files/office.txt")

docs = [Document(page_content=source_pth.read_text(encoding='utf-8'), metadata={'source': str(source_pth)})]

from langchain_text_splitters import RecursiveCharacterTextSplitter

splits = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)

chunks = splits.split_documents(docs)

from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

from langchain_core.vectorstores import InMemoryVectorStore

vectorstore = InMemoryVectorStore.from_documents(chunks, embedding)

retriever = vectorstore.as_retriever()

@tool
def search_document(query: str) -> str:
    """Search the loaded office document and return the passages that match.

    The document is "The Meridian Office Chronicles", covering Meridian
    Analytics: company facts and policies, staff names and their roles, the
    office layout with its rooms and equipment, and the postmortem of a
    PulseGrid outage.

    Pass a natural-language query. Returns the matching passages verbatim,
    separated by blank lines.
    """
    found = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in found)

sub_agent_retriever = create_agent(
    model=model,
    tools=[search_document],
    system_prompt=(
        "You answer questions about the loaded document. "
        "Always use search_document to ground your answers."
    ),
)
