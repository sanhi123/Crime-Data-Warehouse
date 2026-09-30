"""Deterministic, explainable routing for the four assistant modes."""
from enum import Enum

class Route(str, Enum): SQL = "SQL"; RAG = "RAG"; FORECAST = "FORECAST"; HYBRID = "HYBRID"

def route_question(question: str) -> Route:
    q = question.lower()
    forecast = any(x in q for x in ("forecast", "predict", "next week", "future risk", "risk for"))
    explanatory = any(x in q for x in ("why", "explain", "pattern", "insight", "compare", "compared", "increase", "decrease"))
    structured = any(x in q for x in ("how many", "most", "highest", "total", "count", "rate", "top", "which city", "what city"))
    if forecast and (explanatory or structured): return Route.HYBRID
    if forecast: return Route.FORECAST
    if explanatory and (structured or any(x in q for x in ("increase", "decrease", "compared", "previous year"))): return Route.HYBRID
    if explanatory: return Route.RAG
    return Route.SQL if structured else Route.RAG
