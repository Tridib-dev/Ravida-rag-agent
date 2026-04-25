"""
core/config.py
--------------
Centralised configuration and model initialisation.
All environment variables and tuneable constants live here.
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
CHROMA_COLLECTION: str = "reports"
SIMILARITY_TOP_K: int = 5

AGENT_MODEL: str = "gemini-2.5-flash"
AGENT_TEMPERATURE: float = 0.0

QUESTION_MODEL: str = "gemini-2.5-flash"
QUESTION_TEMPERATURE: float = 0.2
EXAMPLE_QUESTION_COUNT: int = 25
EXAMPLE_CHUNKS_FOR_QUESTIONS: int = 3

AGENT_SYSTEM_PROMPT: str = """You are an expert Medical Report Analyst. Your goal is to provide
accurate, evidence-based answers using the medical records provided.

Operational Protocol:
  Search First : For any question regarding patient names, dates, vitals, or diagnoses,
                 always use the search_query tool first.
  Analyse      : Review the retrieved context. If the information is missing from the
                 records, state clearly that the information is not available in the
                 provided documents.
  Summarise    : Provide a concise answer based ONLY on the retrieved data.
                 Do not hallucinate details.
Format: Keep your tone professional and clinical."""

QUESTION_GENERATION_PROMPT: str = """You are a medical report analyst. Generate {count} diverse
questions about this medical report.

Include a mix of:
  - 3 simple questions (patient demographics, basic facts)
  - 10 clinical questions (diagnosis, test results, findings)
  - 12 technical/analytical questions (medical terminology, detailed analysis)

Make questions specific to the document content. Avoid generic questions."""


def get_agent_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=AGENT_MODEL, temperature=AGENT_TEMPERATURE)


def get_question_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=QUESTION_MODEL, temperature=QUESTION_TEMPERATURE
    )
