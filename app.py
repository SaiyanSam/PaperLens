"""PaperLens Streamlit application."""

from __future__ import annotations

import streamlit as st

from src.ingestion.chunker import chunk_documents
from src.ingestion.pdf_parser import (
    PDFExtractionError,
    parse_pdf_bytes,
    save_parsed_document,
)
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.vector_store import FAISSVectorStore


@st.cache_resource
def load_embedding_model() -> EmbeddingModel:
    """Load the embedding model once and reuse it across reruns."""

    return EmbeddingModel()


def initialize_session_state() -> None:
    """Initialize Streamlit session-state variables."""

    defaults = {
        "parsed_documents": [],
        "retriever": None,
        "chunk_count": 0,
        "search_results": [],
        "last_query": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_semantic_retriever(
    parsed_documents: list[dict],
) -> tuple[SemanticRetriever, int]:
    """
    Chunk all parsed documents, embed them, and build a FAISS retriever.

    Returns:
        A semantic retriever and the number of indexed chunks.
    """

    documents = [
        item["document"]
        for item in parsed_documents
    ]

    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError(
            "No searchable text chunks could be created "
            "from the uploaded documents."
        )

    embedding_model = load_embedding_model()

    embeddings = embedding_model.encode_documents(
        [chunk.text for chunk in chunks]
    )

    vector_store = FAISSVectorStore(
        dimension=embedding_model.dimension
    )

    vector_store.add(
        embeddings=embeddings,
        chunks=chunks,
    )

    retriever = SemanticRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    return retriever, len(chunks)


st.set_page_config(
    page_title="PaperLens",
    page_icon="📄",
    layout="wide",
)

initialize_session_state()

st.title("PaperLens")
st.caption(
    "Agentic, Citation-Grounded Analysis of Scientific Literature"
)

st.info(
    "Upload research papers, build a semantic search index, "
    "and retrieve page-grounded evidence."
)

uploaded_files = st.file_uploader(
    "Upload one or more research-paper PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button(
    "Process documents",
    type="primary",
):
    parsed_documents = []
    errors = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        status_text.write(
            f"Extracting `{uploaded_file.name}`..."
        )

        try:
            document = parse_pdf_bytes(
                pdf_bytes=uploaded_file.getvalue(),
                document_name=uploaded_file.name,
            )

            output_path = save_parsed_document(document)

            parsed_documents.append(
                {
                    "document": document,
                    "output_path": output_path,
                }
            )

        except (ValueError, PDFExtractionError) as exc:
            errors.append(
                {
                    "document_name": uploaded_file.name,
                    "error": str(exc),
                }
            )

        progress_bar.progress(index / total_files)

    if parsed_documents:
        try:
            status_text.write(
                "Chunking documents and building semantic index..."
            )

            retriever, chunk_count = build_semantic_retriever(
                parsed_documents
            )

            st.session_state.parsed_documents = (
                parsed_documents
            )
            st.session_state.retriever = retriever
            st.session_state.chunk_count = chunk_count
            st.session_state.search_results = []
            st.session_state.last_query = ""

            st.success(
                f"Processed {len(parsed_documents)} document(s) "
                f"and indexed {chunk_count} searchable chunks."
            )

        except Exception as exc:
            st.session_state.parsed_documents = (
                parsed_documents
            )
            st.session_state.retriever = None
            st.session_state.chunk_count = 0

            st.error(
                "Documents were extracted, but the semantic "
                f"index could not be built: {exc}"
            )

    for error in errors:
        st.error(
            f"Failed to process `{error['document_name']}`: "
            f"{error['error']}"
        )

    progress_bar.empty()
    status_text.empty()


if st.session_state.parsed_documents:
    st.header("Processed Documents")

for item in st.session_state.parsed_documents:
    document = item["document"]
    output_path = item["output_path"]

    st.divider()
    st.subheader(document.document_name)

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Pages",
        document.num_pages,
    )
    metric_columns[1].metric(
        "Words",
        f"{document.total_words:,}",
    )
    metric_columns[2].metric(
        "Characters",
        f"{document.total_characters:,}",
    )
    metric_columns[3].metric(
        "Empty pages",
        len(document.empty_pages),
    )

    st.write(
        f"**Document ID:** `{document.document_id}`"
    )
    st.write(
        f"**Saved JSON:** `{output_path}`"
    )

    if document.empty_pages:
        st.warning(
            "No extractable text was found on pages: "
            + ", ".join(
                map(str, document.empty_pages)
            )
        )

    with st.expander(
        f"Preview extracted pages — {document.document_name}"
    ):
        for page in document.pages[:3]:
            st.markdown(
                f"#### Page {page.page_number}"
            )

            if page.is_empty:
                st.caption(
                    "No extractable text found."
                )
            else:
                preview = page.text[:2000]
                st.text(preview)

                if len(page.text) > len(preview):
                    st.caption(
                        "Preview truncated."
                    )

    json_content = document.model_dump_json(
        indent=2
    )

    st.download_button(
        label=(
            f"Download JSON — "
            f"{document.document_name}"
        ),
        data=json_content,
        file_name=(
            f"{document.document_id}.json"
        ),
        mime="application/json",
        key=(
            f"download_{document.document_id}"
        ),
    )


st.divider()
st.header("Semantic Search")

if st.session_state.retriever is None:
    st.info(
        "Upload and process one or more PDFs "
        "before searching."
    )

else:
    st.caption(
        f"Searching across "
        f"{st.session_state.chunk_count} "
        f"chunks from "
        f"{len(st.session_state.parsed_documents)} "
        f"document(s)."
    )

    search_query = st.text_input(
        "Ask a question about the uploaded papers",
        value=st.session_state.last_query,
        placeholder=(
            "For example: "
            "How do the robots coordinate their plans?"
        ),
    )

    top_k = st.slider(
        "Number of retrieved passages",
        min_value=1,
        max_value=10,
        value=5,
    )

    search_clicked = st.button(
        "Search Papers",
        type="primary",
        disabled=not bool(search_query.strip()),
    )

    if search_clicked:
        try:
            results = (
                st.session_state.retriever.retrieve(
                    query=search_query,
                    top_k=top_k,
                )
            )

            st.session_state.search_results = results
            st.session_state.last_query = search_query

        except ValueError as exc:
            st.error(str(exc))

    if st.session_state.search_results:
        st.subheader("Retrieved Evidence")

        for result in st.session_state.search_results:
            chunk = result.chunk

            if chunk.page_start == chunk.page_end:
                page_label = (
                    f"Page {chunk.page_start}"
                )
            else:
                page_label = (
                    f"Pages {chunk.page_start}"
                    f"–{chunk.page_end}"
                )

            expander_title = (
                f"Rank {result.rank} · "
                f"{chunk.document_name} · "
                f"{page_label} · "
                f"Score "
                f"{result.similarity_score:.4f}"
            )

            with st.expander(
                expander_title,
                expanded=result.rank == 1,
            ):
                st.write(chunk.text)

                st.caption(
                    f"Chunk ID: {chunk.chunk_id} · "
                    f"Chunk index: {chunk.chunk_index} · "
                    f"Words: {chunk.word_count}"
                )
