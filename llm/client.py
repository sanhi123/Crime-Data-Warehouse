"""Optional Gemini client. Missing keys and API errors leave a safe local answer path."""
from __future__ import annotations
import os
from dotenv import load_dotenv
from .prompts import SYSTEM_PROMPT, grounded_prompt

class GroundedLLM:
    def __init__(self, model: str | None = None):
        load_dotenv()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.getenv("GEMINI_API_KEY")

    def generate(self, question: str, evidence: str) -> str | None:
        if not self.api_key or not evidence.strip(): return None
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=grounded_prompt(question, evidence),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=500,
                ),
            )
            return (response.text or "").strip() or None
        except Exception:
            return None
