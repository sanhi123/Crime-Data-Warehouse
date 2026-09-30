from pathlib import Path
from rag.document_builder import CrimeDocumentBuilder
from rag.embeddings import EmbeddingModel
from rag.retriever import FaissRetriever
from src.assistant import CrimeIntelligenceAssistant, INSUFFICIENT
from src.database import CrimeWarehouse
from src.router import Route, route_question
from llm.client import GroundedLLM

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "crime_warehouse.db"

def offline_embedder():
    # Deliberately prevents any model download during test runs.
    return EmbeddingModel(model_name="not-a-real-local-model", allow_download=False)

def test_document_generation_is_aggregated_not_raw_rows():
    docs = CrimeDocumentBuilder(DB).build()
    assert len(docs) > 100
    assert any(d.metadata["kind"] == "city_year_trend" for d in docs)
    assert all("report_number" not in d.text.lower() for d in docs)

def test_faiss_persistence_and_retrieval(tmp_path):
    retriever = FaissRetriever(tmp_path, offline_embedder())
    retriever.build(CrimeDocumentBuilder(DB).build())
    found = retriever.search("Explain crime patterns in Pune", top_k=3)
    assert found
    assert all(item.text and item.metadata for item in found)
    reloaded = FaissRetriever(tmp_path, offline_embedder())
    assert reloaded.search("Pune violent crime", top_k=2)

def test_trend_question_prioritizes_city_year_trends(tmp_path):
    retriever = FaissRetriever(tmp_path, offline_embedder())
    retriever.build(CrimeDocumentBuilder(DB).build())
    found = retriever.search("Why did crime increase in Pune compared with the previous year?")
    assert found
    assert all(item.metadata["kind"] == "city_year_trend" for item in found)
    assert all(item.metadata["city"] == "Pune" for item in found)

def test_empty_retrieval_is_graceful(tmp_path):
    retriever = FaissRetriever(tmp_path, offline_embedder())
    retriever.build(CrimeDocumentBuilder(DB).build())
    assert retriever.search("zzzxqv non-existent martian jurisdiction", min_score=0.12) == []

def test_router_representative_questions():
    assert route_question("What city had the most crimes in 2023?") == Route.SQL
    assert route_question("Explain the major crime patterns in Pune.") == Route.RAG
    assert route_question("What does the forecasting model predict for Mumbai next week?") == Route.FORECAST
    assert route_question("Why did crime increase in Pune compared with the previous year?") == Route.HYBRID

def test_database_access_reuses_warehouse():
    result = CrimeWarehouse(DB).answer("What city had the most crimes in 2023?")
    assert result and result.rows and "2023" in result.answer
    assert result.sql.startswith("SELECT")

def test_city_ranking_question_variations():
    warehouse = CrimeWarehouse(DB)
    questions = [
        "What city had the highest number of crimes in 2023?",
        "Which city had the most crimes in 2023?",
        "What city had the highest crime count?",
    ]
    for question in questions:
        result = warehouse.answer(question)
        assert result is not None and result.rows
        assert result.rows[0]["city"]

def test_end_to_end_sql_rag_and_unanswerable(tmp_path):
    retriever = FaissRetriever(tmp_path, offline_embedder())
    retriever.build(CrimeDocumentBuilder(DB).build())
    assistant = CrimeIntelligenceAssistant(DB, retriever=retriever)
    sql = assistant.ask("What city had the most crimes in 2023?")
    rag = assistant.ask("Explain the major crime patterns in Pune.")
    unknown = assistant.ask("zzzxqv non-existent martian jurisdiction")
    assert sql.route == "SQL" and sql.evidence
    assert rag.route == "RAG" and rag.evidence
    assert unknown.answer == INSUFFICIENT

def test_end_to_end_forecast_uses_existing_bot(tmp_path):
    retriever = FaissRetriever(tmp_path, offline_embedder())
    retriever.build(CrimeDocumentBuilder(DB).build())
    response = CrimeIntelligenceAssistant(DB, retriever=retriever).ask("Predict risk for Mumbai next week")
    assert response.route == "FORECAST"
    assert "Forecast" in response.answer

def test_llm_reads_gemini_key_only(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert GroundedLLM().api_key == "test-key"
