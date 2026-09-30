# Crime Data Warehouse → Crime Intelligence Assistant

This project keeps the existing SQLite star-schema warehouse, notebook analytics, rule-based natural-language SQL, and spatial forecasting models, and adds a small AI assistant that is easy to inspect in an interview.

## Architecture

```text
User → Streamlit chat → deterministic query router
                         ├─ SQL → existing SQLite warehouse / safe query templates
                         ├─ RAG → aggregate analytical documents → embeddings → FAISS
                         ├─ FORECAST → existing ExtendedCrimeBot / forecasting pipeline
                         └─ HYBRID → SQL + RAG evidence → grounded Gemini answer (optional)
```

The app always exposes the route selected and the actual SQL rows and/or retrieved RAG documents used as evidence.

## RAG pipeline

`rag/document_builder.py` creates readable evidence from aggregates already supported by the warehouse: city/year trends (including year-on-year change), city/type and city/domain summaries, city case status, and global crime-type/domain summaries. It never embeds individual `Crime_Fact` rows.

`rag/embeddings.py` uses the lightweight `all-MiniLM-L6-v2` sentence-transformer model when available. For offline or failed model downloads it uses a deterministic 384-feature hashing fallback so the demonstrator and tests remain operational. `rag/retriever.py` writes a FAISS inner-product index and matching `documents.json` metadata, and returns text, metadata, and similarity score for every retrieval.

Build or rebuild the index:

```bash
./.venv/bin/python -m rag.build_index --db crime_warehouse.db
```

The build is offline-safe by default; add `--download-model` once to download MiniLM when network access is available.

## LLM pipeline

The optional Gemini client in `llm/client.py` uses the official `google-genai` SDK. It reads `GEMINI_API_KEY` from the environment (or `.env`) and passes only the user question plus collected evidence to a grounding prompt. It explicitly instructs Gemini not to invent statistics and to report insufficient data. No key is hardcoded. If the key is missing or the API fails, the assistant returns the retrieved evidence directly instead of failing.

```bash
cp .env.example .env
# add GEMINI_API_KEY to .env; .env is ignored by Git
```

## Query routing

`src/router.py` is intentionally deterministic and testable:

- **SQL**: exact counts, rankings, totals, rates.
- **RAG**: explanatory/pattern questions.
- **FORECAST**: predictions, future risk, or next-week questions, reusing `ExtendedCrimeBot`.
- **HYBRID**: explanation/comparison requests that also require measured data, such as year-on-year change.

The existing notebook and `src/chatbot.py` are preserved. `src/database.py` adds safe read-only templates for the assistant rather than accepting arbitrary LLM SQL.

## Setup and run

```bash
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m rag.build_index --db crime_warehouse.db
./.venv/bin/streamlit run app.py
```

Examples:

- `What city had the most crimes in 2023?`
- `Explain the major crime patterns in Pune.`
- `What does the forecasting model predict for Mumbai next week?`
- `Why did crime increase in Pune compared with the previous year?`
- `What happened in the zzzxqv jurisdiction?` (shows insufficient-data handling)

## Tests

```bash
./.venv/bin/python -m pytest -q
```

The suite covers aggregate-document generation, FAISS build/persistence/retrieval, empty retrieval, routing, read-only database access, and a basic end-to-end SQL/RAG/unanswerable flow.

## Limits

Forecast requests in the Streamlit assistant require a loaded `ExtendedCrimeBot` forecasting pipeline/model; the original `run_pipeline.py` remains the existing way to train it. The assistant reports this clearly rather than fabricating a forecast. The warehouse records observed aggregates, so the assistant describes increases but does not claim a real-world cause without supporting data.
