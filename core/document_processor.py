"""
core/document_processor.py
---------------------------
Handles the full PDF → chunks → vector-store pipeline.

Responsibilities
────────────────
• Accept an in-memory uploaded file (Streamlit UploadedFile).
• Write it to a temp file so PyPDFLoader can read it.
• Split the pages into chunks via RecursiveCharacterTextSplitter.
• Ingest chunks into the provided Chroma vector store.
• Return the chunks so callers can do further work (e.g. question generation).
"""

import os
import tempfile
from typing import IO

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Splitter (session-cached) ─────────────────────────────────────────────────


def get_splitter() -> RecursiveCharacterTextSplitter:
    """Return a session-cached text splitter."""
    if "doc_splitter" not in st.session_state:
        st.session_state.doc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    return st.session_state.doc_splitter


# ── Core pipeline ─────────────────────────────────────────────────────────────


def load_pdf(uploaded_file: IO[bytes]) -> list[Document]:
    """
    Write the uploaded file to a temporary path, load it with PyPDFLoader,
    then delete the temp file.

    Returns a list of LangChain Document objects (one per PDF page).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split a list of page-level Documents into smaller chunks."""
    splitter = get_splitter()
    return splitter.split_documents(documents)


def ingest(
    uploaded_file: IO[bytes],
    vector_store: Chroma,
) -> list[Document]:
    """
    Full pipeline: load → split → embed → store.

    Parameters
    ----------
    uploaded_file : The raw bytes from st.file_uploader.
    vector_store  : The Chroma store to write chunks into.

    Returns
    -------
    The list of chunks (useful for downstream tasks like question generation).
    """
    documents = load_pdf(uploaded_file)
    chunks = split_documents(documents)
    vector_store.add_documents(documents=chunks)
    return chunks
