"""
core/vector_store.py
--------------------
Manages the Chroma vector store and the FastEmbed embedding model.

Responsibilities
────────────────
• Load (and cache) the embedding model.
• Create / reset the Chroma collection.
• Expose helper methods used by the document-processing pipeline.
"""

import streamlit as st
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

from core.config import EMBEDDING_MODEL, CHROMA_COLLECTION


# ── Embedding model ───────────────────────────────────────────────────────────

def get_embedder() -> FastEmbedEmbeddings:
    """
    Load the FastEmbed embedding model once per session and cache it in
    st.session_state so Streamlit reruns don't reload it from disk.
    """
    if "embedder" not in st.session_state:
        with st.spinner("Loading embedding model…"):
            st.session_state.embedder = FastEmbedEmbeddings(
                model_name=EMBEDDING_MODEL
            )
    return st.session_state.embedder


# ── Vector store ──────────────────────────────────────────────────────────────

def create_vector_store(embedder: FastEmbedEmbeddings) -> Chroma:
    """Create a fresh (in-memory) Chroma collection."""
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedder,
    )


def reset_vector_store(embedder: FastEmbedEmbeddings) -> Chroma:
    """
    Tear down the existing Chroma collection (if any) and return a new one.
    Call this whenever a new PDF is uploaded so stale vectors are purged.
    """
    existing: Chroma | None = st.session_state.get("vector_store")
    if existing is not None:
        existing.delete_collection()

    new_store = create_vector_store(embedder)
    st.session_state.vector_store = new_store
    return new_store


def get_vector_store() -> Chroma | None:
    """Return the current vector store, or None if not yet initialised."""
    return st.session_state.get("vector_store")
