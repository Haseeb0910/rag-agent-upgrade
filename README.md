# Agentic RAG Chatbot

An agentic upgrade to a traditional RAG chatbot — built with **LangGraph**, **Groq**, and **FAISS**. Instead of always following a fixed retrieve-then-answer pipeline, this agent reasons about each question and decides for itself whether to search an uploaded document, run a calculation, or answer directly from its own knowledge.

## Why this exists

The original version of this chatbot ([https://github.com/Haseeb0910/Rag-Based-ChatBot]) used a standard RAG pipeline: every question triggered a document search, regardless of whether the question actually needed one. This upgrade rebuilds that system as a **graph of decisions** instead of a fixed chain — the LLM itself chooses the right tool for each question, remembers conversation context across turns, and its reasoning can be traced step-by-step.

## What it does

- **Upload a PDF** and ask questions about its content
- **Multi-tool reasoning** — the agent picks between:
  - `search_doc` — semantic search over the uploaded document (FAISS + sentence-transformers)
  - `calculate` — basic math evaluation
  - Or answers directly from its own knowledge when no tool is needed
- **Conversation memory** — follow-up questions like "summarize that" correctly resolve to earlier context, per session
- **Full observability** — every decision, tool call, and result is traced in LangSmith

## Architecture

```
                    ┌─────────────┐
   User question →  │  call_model │ ←──────────┐
                    └──────┬──────┘             │
                           │                     │
                  Tool call needed?              │
                     ┌─────┴─────┐               │
                    Yes          No               │
                     │            │               │
              ┌──────▼─────┐      ▼            (loop back
              │  call_tool │     END          with tool result)
              └──────┬─────┘                     │
                     │                            │
                     └────────────────────────────┘
```

The graph has two nodes — `call_model` (the LLM, bound to available tools) and `call_tool` (executes whichever tool the LLM requested). A conditional edge checks if the LLM's response includes a tool call; if so, it routes to `call_tool` and loops back, otherwise it ends and returns the answer. A `recursion_limit` prevents runaway loops if the agent can't find a satisfying answer.

## Tech stack

| Component | Tool |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| PDF parsing | `pypdf` |
| UI | Gradio |
| Observability | LangSmith |
| Memory | LangGraph `MemorySaver` checkpointer |

## What changed from v1

The [original RAG chatbot](#) used a single function that always ran retrieval before every LLM call, and rebuilt the FAISS index from scratch on every question. This version:

- Builds the index **once** per uploaded document, not per question
- Lets the LLM **decide** whether retrieval is even necessary
- Adds a **second tool** (calculator) to demonstrate multi-tool routing, not just single-purpose RAG
- Adds **persistent memory** across conversation turns
- Adds **tracing** so the agent's reasoning is inspectable, not a black box

## Running locally

```bash
git clone https://github.com/Haseeb0910/rag-agent-upgrade.git
cd rag-agent-upgrade
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=rag-agent-upgrade
```

Run:
```bash
python app.py
```

Upload a PDF in the UI, then ask questions — try one about the document's content, a math question, and a general knowledge question to see the routing in action.

## Engineering notes / lessons learned

A few real issues surfaced while building this, worth documenting since they're common pitfalls with agentic systems, not just bugs specific to this project:

- **Tool-calling reliability varies by model.** `llama-3.3-70b-versatile` occasionally produced malformed tool-call syntax on Groq; switching to `llama-3.1-8b-instant` resolved it. Worth testing tool-calling reliability before committing to a model.
- **Agents can loop indefinitely without a cap.** Early versions of the tool-calling loop had no limit, and the agent would repeatedly re-query a thin document hoping for a better answer. Fixed with `recursion_limit` plus a system prompt instructing the model to use each tool at most once per question.
- **LLMs will invent tools that don't exist** rather than admit a question is outside their available capabilities (e.g., calling a nonexistent `brave_search` tool for a general knowledge question). The fix was an explicit system prompt boundary: "do not invent tools that don't exist," plus clear guidance on when *not* to use any tool at all.

## Future improvements

- Multi-document support (currently one PDF per session)
- Streaming responses in the UI
- Additional tools (e.g., web search, code execution)
- Persistent (non-in-memory) conversation storage
