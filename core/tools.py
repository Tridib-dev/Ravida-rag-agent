"""
core/tools.py
-------------
Factory for the LangChain `search_query` tool used by the RAG agent.

Why a factory?
──────────────
The tool needs a reference to the *current* Chroma vector store, which is
created at runtime after the PDF is uploaded.  A factory function lets us
bind the live store into the tool's closure cleanly, without global state.
"""

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_chroma import Chroma

from core.config import SIMILARITY_TOP_K


class SearchQueryInput(BaseModel):
    """Input schema for the search_query tool."""

    query: str = Field(description="Specific search query about the medical report")


def make_search_tool(vector_store: Chroma):
    """
    Return a LangChain tool that performs similarity search against *vector_store*.

    Parameters
    ----------
    vector_store : An already-populated Chroma collection.

    Returns
    -------
    A `@tool`-decorated function ready to be passed to `create_agent`.
    """

    @tool(args_schema=SearchQueryInput)
    def search_query(query: str) -> str:
        """Search the medical record vector database to retrieve specific patient data,
        clinical notes, and laboratory results. Use this tool to find factual evidence
        before providing a medical summary or answering specific health-related questions.

        Args:
            query: A specific search query about patient information, test results,
                   diagnoses, doctors, or treatments. Be specific and use keywords.

        Examples:
            - "patient name"
            - "blood test results glucose"
            - "diagnosis hypertension"

        Returns the relevant excerpts from the medical report.
        """
        if vector_store is None:
            return "ERROR: No vector store available."
        if vector_store._collection.count() == 0:
            return "ERROR: The vector store is empty."

        results = vector_store.similarity_search(query=query, k=SIMILARITY_TOP_K)
        context = "\n\n".join(doc.page_content for doc in results)
        return context

    return search_query
