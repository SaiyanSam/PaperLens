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


This project is being developed for educational, research, and portfolio purposes.
