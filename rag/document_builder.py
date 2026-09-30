"""Build compact analytical documents from the warehouse, never raw fact rows."""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AnalyticalDocument:
    text: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


class CrimeDocumentBuilder:
    """Turns supported star-schema aggregations into readable evidence documents."""

    def __init__(self, db_path: str | Path = "crime_warehouse.db"):
        self.db_path = str(db_path)

    def build(self) -> list[AnalyticalDocument]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return self._build(conn)

    def _build(self, conn: sqlite3.Connection) -> list[AnalyticalDocument]:
        docs: list[AnalyticalDocument] = []
        overview = conn.execute("""
            SELECT SUM(f.cases) AS cases, COUNT(DISTINCT l.city) AS cities,
                   MIN(d.year) AS first_year, MAX(d.year) AS last_year
            FROM Crime_Fact f JOIN Location_Dim l USING(location_dim_id)
            JOIN Date_Dim d USING(date_dim_id)
        """).fetchone()
        docs.append(AnalyticalDocument(
            "This India crime warehouse contains "
            f"{int(overview['cases']):,} recorded cases across {int(overview['cities'])} cities "
            f"from {overview['first_year']} through {overview['last_year']}.",
            {"kind": "india_overview", "cases": int(overview["cases"]),
             "cities": int(overview["cities"]), "first_year": overview["first_year"], "last_year": overview["last_year"]},
        ))
        yearly = conn.execute("""
            SELECT l.city, d.year, SUM(f.cases) AS cases
            FROM Crime_Fact f JOIN Location_Dim l USING(location_dim_id)
            JOIN Date_Dim d USING(date_dim_id)
            GROUP BY l.city, d.year ORDER BY l.city, d.year
        """).fetchall()
        previous: dict[str, int] = {}
        for row in yearly:
            city, year, cases = row["city"], row["year"], int(row["cases"])
            prior = previous.get(city)
            change = "No prior-year comparison is available."
            if prior is not None:
                pct = (cases - prior) / prior * 100 if prior else 0
                direction = "increased" if cases >= prior else "decreased"
                change = f"This {direction} by {abs(pct):.1f}% from {prior:,} cases in {year - 1}."
            docs.append(AnalyticalDocument(
                f"{city} recorded {cases:,} crime cases in {year}. {change}",
                {"kind": "city_year_trend", "city": city, "year": year, "cases": cases},
            ))
            previous[city] = cases

        queries = [
            ("city_crime_type", """SELECT l.city AS city, c.crime_type AS label, SUM(f.cases) AS cases
                FROM Crime_Fact f JOIN Location_Dim l USING(location_dim_id)
                JOIN Crime_Dim c USING(crime_dim_id) GROUP BY l.city, c.crime_type""",
             "In {city}, {label} accounts for {cases:,} recorded cases."),
            ("city_domain", """SELECT l.city AS city, c.crime_domain AS label, SUM(f.cases) AS cases
                FROM Crime_Fact f JOIN Location_Dim l USING(location_dim_id)
                JOIN Crime_Dim c USING(crime_dim_id) GROUP BY l.city, c.crime_domain""",
             "In {city}, the {label} domain has {cases:,} recorded cases."),
            ("city_case_closure", """SELECT l.city AS city, s.case_closed AS label, SUM(f.cases) AS cases
                FROM Crime_Fact f JOIN Location_Dim l USING(location_dim_id)
                JOIN Status_Dim s USING(status_dim_id) GROUP BY l.city, s.case_closed""",
             "In {city}, {cases:,} cases have case_closed={label}."),
            ("crime_domain_summary", """SELECT c.crime_domain AS label, SUM(f.cases) AS cases
                FROM Crime_Fact f JOIN Crime_Dim c USING(crime_dim_id) GROUP BY c.crime_domain""",
             "Across the warehouse, {label} has {cases:,} recorded cases."),
            ("crime_type_summary", """SELECT c.crime_type AS label, SUM(f.cases) AS cases
                FROM Crime_Fact f JOIN Crime_Dim c USING(crime_dim_id) GROUP BY c.crime_type""",
             "Across the warehouse, {label} has {cases:,} recorded cases."),
        ]
        for kind, sql, template in queries:
            for row in conn.execute(sql):
                data = dict(row)
                data["cases"] = int(data["cases"])
                docs.append(AnalyticalDocument(template.format(**data), {"kind": kind, **data}))
        return docs
