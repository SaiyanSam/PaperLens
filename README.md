# PaperLens

> **Agentic Research Paper Analysis Assistant**

PaperLens is an **agentic AI assistant** for scientific literature that enables citation-grounded question answering, paper summarization, cross-paper comparison, and structured information extraction from research papers.

Unlike a traditional Retrieval-Augmented Generation (RAG) chatbot, PaperLens dynamically selects specialized tools based on the user's request. The system combines semantic retrieval, tool orchestration, structured validation, and LLM-based reasoning to provide reliable, evidence-backed responses with page-level citations.

## Planned Features

- 📄 Multi-PDF document ingestion
- 🔍 Semantic search over scientific papers
- 💬 Citation-grounded question answering
- 📝 Structured paper summarization
- ⚖️ Cross-paper comparison
- 📊 Experimental result extraction and visualization
- 🤖 Agent-based tool routing using LangChain
- ✅ Pydantic-based tool and output validation

---

# Development Roadmap (Week 1)

| Tasl | Tasks | Deliverable | Progress |
|-----|-------|-------------|----------|
| Task 1 | Repository setup, project structure, PDF ingestion, page-wise text extraction, Pydantic document validation, JSON export, Streamlit upload interface | Validated document ingestion pipeline | ✅ Done |
| Task 2 | Text cleaning refinement, chunking pipeline, Sentence Transformer embeddings, FAISS vector store, semantic retrieval | Retrieve relevant document chunks from uploaded papers | 🔄 Active |
| Task 3 | LCEL-based QA chain, grounded prompting, citation formatting, answer generation | End-to-end citation-grounded question answering | ⏳ Planned |
| Task 4 | LangChain tool registry, `search_papers` tool, Pydantic tool schemas, validated tool execution | Agent can execute validated search tool | ⏳ Planned |
| Task 5 | `summarize_paper` tool, section-aware summarization, structured summaries | Agent generates structured paper summaries | ⏳ Planned |
| Task 6 | `compare_papers` tool, independent retrieval for multiple papers, comparison chain, improved Streamlit interface | Cross-paper comparison with citations | ⏳ Planned |
| Task 7 | Project polish, README improvements, architecture diagram, screenshots, demo GIF, final testing | Placement-ready GitHub repository | ⏳ Planned |

---

## Technology Stack

| Component | Technology |
|----------|------------|
| Language | Python 3.11 |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Data Validation | Pydantic |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| LLM Framework | LangChain + LCEL |
| Agent Framework | LangChain Tool Calling |
| Testing | PyTest |

---

## Project Status

Current milestone:

```text
PDF Upload
    ↓
Page-wise Text Extraction
    ↓
Document Validation
    ↓
JSON Export
```

Next milestone:

```text
PDF
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS
    ↓
Semantic Retrieval
```

---

## Long-Term Vision

```text
Research Papers
        ↓
Document Processing
        ↓
Vector Database
        ↓
Agent Tool Selection
        ↓
Search / Summary / Comparison / Extraction
        ↓
Validated Outputs
        ↓
Citation-Grounded Response
```

---

## License

This project is being developed for educational, research, and portfolio purposes.
