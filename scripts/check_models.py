#!/usr/bin/env python3
"""Check which Gemini models your key / Vertex project can call (1 tiny request each)."""
import sys

from google import genai
from google.genai import types

wanted = sys.argv[1:] or ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
client = genai.Client()
try:
    names = sorted(m.name.split("/")[-1] for m in client.models.list()
                   if "flash" in m.name or "pro" in m.name)
    print("Models visible to this account:", ", ".join(names))
except Exception as exc:  # listing is not essential
    print("Could not list models:", exc)
ok = []
for model in wanted:
    try:
        r = client.models.generate_content(model=model, contents="Reply with OK.",
                                           config=types.GenerateContentConfig(temperature=0, max_output_tokens=20))
        print(f"  OK      {model}: {(r.text or '').strip()[:20]!r}")
        ok.append(model)
    except Exception as exc:
        print(f"  FAILED  {model}: {str(exc)[:160]}")
sys.exit(0 if len(ok) == len(wanted) else 1)
