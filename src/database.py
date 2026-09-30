"""Safe, read-only access to the existing SQLite warehouse and existing SQL intent rules."""
from __future__ import annotations
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SqlResult:
    answer: str
    sql: str
    rows: list[dict]

class CrimeWarehouse:
    CITY_RANKING_PHRASES = (
        "most crimes",
        "most crime",
        "highest crime",
        "highest number of crimes",
        "most number of crimes",
        "maximum crimes",
        "city with the most crimes",
        "city with the highest crime count",
        "highest crime count",
    )

    def __init__(self, db_path: str | Path = "crime_warehouse.db"):
        self.db_path = str(db_path)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        # Assistant-generated SQL is not accepted: this layer runs only owned templates.
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    @classmethod
    def _is_city_ranking_question(cls, question: str) -> bool:
        """Recognize clear natural-language variations of the city crime ranking intent."""
        return any(phrase in question for phrase in cls.CITY_RANKING_PHRASES)

    def answer(self, question: str) -> SqlResult | None:
        q = question.lower()
        year = re.search(r"\b(20\d{2})\b", q)
        if self._is_city_ranking_question(q) and year:
            sql = """SELECT l.city, SUM(f.cases) AS total_cases FROM Crime_Fact f
                     JOIN Location_Dim l USING(location_dim_id) JOIN Date_Dim d USING(date_dim_id)
                     WHERE d.year=? GROUP BY l.city ORDER BY total_cases DESC LIMIT 5"""
            rows = self.query(sql, (int(year.group()),))
            if rows: return SqlResult(f"{rows[0]['city']} had the most recorded crimes in {year.group()}: {rows[0]['total_cases']:,} cases.", sql, rows)
        if self._is_city_ranking_question(q):
            sql = """SELECT l.city, SUM(f.cases) AS total_cases FROM Crime_Fact f
                     JOIN Location_Dim l USING(location_dim_id) GROUP BY l.city ORDER BY total_cases DESC LIMIT 5"""
            rows = self.query(sql)
            return SqlResult(f"{rows[0]['city']} has the highest recorded crime volume: {rows[0]['total_cases']:,} cases.", sql, rows)
        if "closure" in q or "closed cases" in q:
            sql = """SELECT s.case_closed, SUM(f.cases) AS cases FROM Crime_Fact f
                     JOIN Status_Dim s USING(status_dim_id) GROUP BY s.case_closed"""
            rows = self.query(sql)
            total, closed = sum(r['cases'] for r in rows), next((r['cases'] for r in rows if r['case_closed'] == 'Yes'), 0)
            return SqlResult(f"The recorded case-closure rate is {closed / total * 100:.1f}% ({closed:,} of {total:,} cases).", sql, rows)
        if "top crime type" in q or "most common crime" in q or "common crime type" in q:
            sql = """SELECT c.crime_type, SUM(f.cases) AS total_cases FROM Crime_Fact f
                     JOIN Crime_Dim c USING(crime_dim_id) GROUP BY c.crime_type ORDER BY total_cases DESC LIMIT 5"""
            rows = self.query(sql)
            return SqlResult(f"The most common recorded crime type is {rows[0]['crime_type']} ({rows[0]['total_cases']:,} cases).", sql, rows)
        return None
