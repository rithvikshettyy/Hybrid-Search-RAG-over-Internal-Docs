"""Thin Gemini wrapper shared by answer + verify."""

from google import genai
from google.genai import types

from config import GEMINI_MODEL, gemini_api_key


def complete(system: str, user: str, max_tokens: int = 1024) -> str:
    client = genai.Client(api_key=gemini_api_key())
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            # Thinking tokens count toward the output cap: keep thinking low, leave headroom.
            max_output_tokens=max_tokens + 1024,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    return resp.text or ""
