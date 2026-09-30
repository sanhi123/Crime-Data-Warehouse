"""Orchestrates router, existing SQL/forecast paths, RAG evidence, and grounded generation."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from llm.client import GroundedLLM
from rag.retriever import FaissRetriever, RetrievedDocument
from src.database import CrimeWarehouse
from src.router import Route, route_question

INSUFFICIENT = "The available data is insufficient to answer that question."

@dataclass
class AssistantResponse:
    answer: str
    route: str
    evidence: list[dict]
    sql: str | None = None

class CrimeIntelligenceAssistant:
    def __init__(self, db_path: str | Path = "crime_warehouse.db", index_dir: str | Path = "rag_index", retriever=None, llm=None, forecast_bot=None):
        self.warehouse = CrimeWarehouse(db_path)
        self.retriever = retriever or FaissRetriever(index_dir)
        self.llm = llm or GroundedLLM()
        # Preserve the repository's forecasting entry point.  Loading its tensor is
        # deferred inside ExtendedCrimeBot until a forecast question arrives.
        if forecast_bot is None:
            try:
                from src.chatbot import ExtendedCrimeBot
                from src.data_pipeline import CrimeDataPipeline
                forecast_bot = ExtendedCrimeBot(pipeline=CrimeDataPipeline(str(db_path)))
            except Exception:
                forecast_bot = None
        self.forecast_bot = forecast_bot

    def _retrieval(self, question: str) -> list[RetrievedDocument]:
        try: return self.retriever.search(question)
        except FileNotFoundError: return []

    def _ground(self, question: str, evidence: list[dict], default: str) -> str:
        context = "\n".join(item.get("text", str(item)) for item in evidence)
        return self.llm.generate(question, context) or default

    def ask(self, question: str) -> AssistantResponse:
        route = route_question(question)
        sql_result = self.warehouse.answer(question) if route in (Route.SQL, Route.HYBRID) else None
        retrieved = self._retrieval(question) if route in (Route.RAG, Route.HYBRID) else []
        evidence = ([{"source": "SQL warehouse", "text": str(row), "metadata": row} for row in (sql_result.rows if sql_result else [])]
                    + [{"source": "RAG analytical document", "text": doc.text, "metadata": doc.metadata, "score": doc.score} for doc in retrieved])
        if route == Route.FORECAST:
            if not self.forecast_bot: return AssistantResponse("Forecasting is available through the existing model pipeline; load a trained model or use its built-in heuristic fallback before requesting a forecast.", route.value, [])
            try: return AssistantResponse(self.forecast_bot.predict_risk_intent(question), route.value, [])
            except Exception as exc: return AssistantResponse(f"Forecasting could not run: {exc}", route.value, [])
        if route == Route.SQL:
            return AssistantResponse(sql_result.answer if sql_result else INSUFFICIENT, route.value, evidence, sql_result.sql if sql_result else None)
        if route == Route.RAG:
            if not retrieved: return AssistantResponse(INSUFFICIENT, route.value, [])
            return AssistantResponse(self._ground(question, evidence, "\n".join(f"- {d.text}" for d in retrieved)), route.value, evidence)
        # Hybrid deliberately includes both evidence sources; no causal conclusion beyond observed change.
        if not evidence: return AssistantResponse(INSUFFICIENT, route.value, [])
        default = (sql_result.answer + "\n\nRelevant warehouse context:\n" if sql_result else "Relevant warehouse context:\n") + "\n".join(f"- {d.text}" for d in retrieved)
        return AssistantResponse(self._ground(question, evidence, default), route.value, evidence, sql_result.sql if sql_result else None)
