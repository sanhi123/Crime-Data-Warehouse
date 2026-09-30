import re
import sqlite3
import numpy as np
import torch
import pandas as pd

class ExtendedCrimeBot:
    """
    Enhanced Crime Q&A Bot featuring:
    1. SQL Data Warehouse Analytics
    2. Deep Learning Spatial Risk Forecasting ("predict risk for [area] next week")
    """
    def __init__(self, db_conn=None, pipeline=None, trained_model=None, db_path='crime_warehouse.db'):
        self.db_path = db_path if isinstance(db_conn, str) else getattr(db_conn, 'db_path', db_path)
        if isinstance(db_conn, str):
            self.conn = sqlite3.connect(db_conn)
        elif db_conn is not None:
            self.conn = db_conn
        else:
            self.conn = sqlite3.connect(self.db_path)
        self.pipeline = pipeline
        self.model = trained_model

    def set_model(self, model):
        self.model = model

    def _q(self, sql, params=()):
        try:
            return pd.read_sql_query(sql, self.conn, params=params)
        except Exception:
            self.conn = sqlite3.connect(self.db_path)
            return pd.read_sql_query(sql, self.conn, params=params)

    def extract_city(self, question):
        q = question.lower()
        for city in self.pipeline.city_to_grid.keys():
            if city in q:
                return city.capitalize()
        return None

    def predict_risk_intent(self, question):
        # Populate city/grid mappings before resolving the requested location.  This
        # also keeps standalone use of the existing bot working outside run_pipeline.
        if self.pipeline.aggregated_tensor is None:
            self.pipeline.load_and_preprocess()
        city = self.extract_city(question)
        if not city:
            # Pick a default prominent city if not specified
            city = "Mumbai"

        city_lower = city.lower()
        if city_lower not in self.pipeline.city_to_grid:
            return f"⚠️ Location '{city}' was not found in the spatial grid dictionary."

        r, c = self.pipeline.city_to_grid[city_lower]

        # Get recent 8-week tensor window
        recent_seq = self.pipeline.aggregated_tensor[-8:] # (8, 1, H, W)
        hist_avg = self.pipeline.aggregated_tensor[:-8, 0, r, c].mean()

        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor(recent_seq[np.newaxis, :], dtype=torch.float32)
                # If model expects CPU
                pred = self.model(input_tensor) # (1, 1, H, W)
                pred_count = float(pred[0, 0, r, c].item())
        else:
            # Fallback heuristic using recent trend
            pred_count = float(recent_seq[:, 0, r, c].mean() * 1.05)

        # Classify risk level relative to historical average
        if pred_count > hist_avg * 1.25:
            risk_level = "🚨 HIGH RISK"
            color_emoji = "🔴"
        elif pred_count > hist_avg * 0.9:
            risk_level = "⚠️ MEDIUM RISK"
            color_emoji = "🟡"
        else:
            risk_level = "✅ LOW RISK"
            color_emoji = "🟢"

        diff_pct = ((pred_count - hist_avg) / (hist_avg + 1e-5)) * 100
        trend_str = f"+{diff_pct:.1f}% vs baseline" if diff_pct >= 0 else f"{diff_pct:.1f}% vs baseline"

        response = (
            f"\n🔮 **Spatiotemporal Crime Risk Forecast for Next Week**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏙️ **Location:** {city} (Grid Cell [{r}, {c}])\n"
            f"🎯 **Predicted Risk Level:** {risk_level} {color_emoji}\n"
            f"📊 **Expected Crime Count:** {pred_count:.2f} incidents\n"
            f"📈 **Historical Comparison:** {hist_avg:.2f} avg ({trend_str})\n"
            f"💡 **Recommendation:** {'Increase police patrols in target grid.' if pred_count > hist_avg else 'Maintain baseline monitoring.'}\n"
        )
        return response

    def answer(self, question):
        q = question.lower().strip()

        # Check for prediction intent keywords
        if any(kw in q for kw in ['predict', 'forecast', 'next week', 'future risk', 'risk for']):
            return self.predict_risk_intent(question)

        # Standard SQL intents
        if 'most crime' in q or 'highest crime' in q:
            r = self._q('''
                SELECT ld.city, SUM(cf.cases) AS total
                FROM Crime_Fact cf
                JOIN Location_Dim ld ON cf.location_dim_id=ld.location_dim_id
                GROUP BY ld.city ORDER BY total DESC LIMIT 3
            ''')
            rows = [f"  {i+1}. {row['city']} ({int(row['total']):,} cases)" for i, (_, row) in enumerate(r.iterrows())]
            return "Cities with highest historical crime volume:\n" + "\n".join(rows)

        if 'crime type' in q or 'common crime' in q:
            r = self._q('''
                SELECT cd.crime_type, SUM(cf.cases) AS total
                FROM Crime_Fact cf
                JOIN Crime_Dim cd ON cf.crime_dim_id=cd.crime_dim_id
                GROUP BY cd.crime_type ORDER BY total DESC LIMIT 5
            ''')
            rows = [f"  {i+1}. {row['crime_type']} ({int(row['total']):,} cases)" for i, (_, row) in enumerate(r.iterrows())]
            return "Top 5 most frequent crime types:\n" + "\n".join(rows)

        city = self.extract_city(question)
        if city and ('how many' in q or 'total' in q or 'count' in q):
            r = self._q('''
                SELECT SUM(cf.cases) AS total
                FROM Crime_Fact cf
                JOIN Location_Dim ld ON cf.location_dim_id=ld.location_dim_id
                WHERE LOWER(ld.city) = ?
            ''', [city.lower()])
            total = int(r['total'].iloc[0] or 0)
            return f"Total historical crimes recorded in {city}: {total:,} incidents."

        return (
            f"Bot Response to '{question}':\n"
            f"You can ask me: 'Predict risk for Mumbai next week', 'Which city has the most crimes?', or 'What is the top crime type?'"
        )
