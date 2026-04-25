"""
utils/question_generator.py
---------------------------
Generates sample questions from a document's opening chunks using Gemini.

Keeping this in `utils/` signals that it is a supporting utility, not part
of the core RAG pipeline.  It can be swapped out or disabled independently.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from core.config import (
    EXAMPLE_QUESTION_COUNT,
    EXAMPLE_CHUNKS_FOR_QUESTIONS,
    QUESTION_GENERATION_PROMPT,
    get_question_llm,
)


class _ExampleQuestions(BaseModel):
    """Structured output schema for the question-generation chain."""

    questions: list[str] = Field(description="List of sample questions")


def generate_example_questions(chunks: list[Document]) -> list[str]:
    """
    Generate diverse example questions from the first few document chunks.

    Parameters
    ----------
    chunks : The full list of document chunks produced by the ingestion pipeline.

    Returns
    -------
    A list of question strings.
    """

    sample_chunks = chunks[:EXAMPLE_CHUNKS_FOR_QUESTIONS]
    context = "\n\n".join(c.page_content for c in sample_chunks)

    llm = get_question_llm().with_structured_output(_ExampleQuestions)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                QUESTION_GENERATION_PROMPT.format(count=EXAMPLE_QUESTION_COUNT),
            ),
            ("human", "Document excerpt:\n{context}"),
        ]
    )

    chain = prompt | llm
    result: _ExampleQuestions = chain.invoke({"context": context})
    return result.questions
