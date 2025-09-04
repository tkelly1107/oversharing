# predictors.py
# Self-contained predictors for the Oversharing project.
# Exports:
#   predict_rules(text) -> {"labels": [...], "spans": [...], "explanations": {...}}
#   predict_prompt_only(text) -> same schema
#   predict_hybrid(text) -> same schema

from typing import Dict, List
import os, json, time, random, re
from openai import OpenAI
from detector import Detector  # uses your saved detector.py

# -----------------------
# Initialize components
# -----------------------
# Detector for rules/NER baseline and hybrid anchors
det = Detector(enable_spacy=True)

# OpenAI client; rely on OPENAI_API_KEY in environment
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()
OPENAI_MODEL = os.environ.get("OVERSHARE_OPENAI_MODEL", "gpt-4o-mini")  # override via env if needed

# -----------------------
# Prompt scaffold
# -----------------------
FEWSHOTS = """
Task: Classify the social post for privacy risks and return ONLY valid JSON with fields:
{"labels": [<one or more categories>],
 "spans_text": [{"label":"<category>","text":"<exact substring>"}],
 "explanations": {"<category>":"<≤25 words>"}}
Categories: ["Location&Time","Contact&IDs","Gov&Financial IDs","Health&Sensitive","Credentials&Security","Workplace/Academic","Minors","Metadata/Device","None"].
Rules:
- Multi-label allowed. Use "None" only if no other labels apply.
- spans_text MUST quote exact substrings from the post (no paraphrase).
- Keep JSON minimal: only labels, spans_text, explanations.

Example 1
Post: "Call me at 555-214-7821 — I am at Blue Finch Café until noon."
JSON:
{"labels":["Contact&IDs","Location&Time"],
 "spans_text":[
   {"label":"Contact&IDs","text":"555-214-7821"},
   {"label":"Location&Time","text":"Blue Finch Café"},
   {"label":"Location&Time","text":"noon"}],
 "explanations":{
   "Contact&IDs":"Shares a direct phone number.",
   "Location&Time":"Reveals time/place enabling tracking."}
}

Example 2
Post: "Utility wants my SSN: 123-45-6789"
JSON:
{"labels":["Gov&Financial IDs"],
 "spans_text":[{"label":"Gov&Financial IDs","text":"123-45-6789"}],
 "explanations":{"Gov&Financial IDs":"Exposes a government identifier."}
}

Example 3
Post: "Daycare pickup code for my toddler is 1140—text me"
JSON:
{"labels":["Credentials&Security","Minors"],
 "spans_text":[
   {"label":"Credentials&Security","text":"1140"},
   {"label":"Minors","text":"my toddler"}],
 "explanations":{
   "Credentials&Security":"Shares an access code.",
   "Minors":"Mentions a dependent child."}
}

Example 4
Post: "Clinic follow-up after my anxiety diagnosis went well."
JSON:
{"labels":["Health&Sensitive"],
 "spans_text":[{"label":"Health&Sensitive","text":"anxiety diagnosis"}],
 "explanations":{"Health&Sensitive":"Reveals a health condition."}
}

Example 5
Post: "Internal beta launch next week; NDA still active, don’t share."
JSON:
{"labels":["Workplace/Academic","Location&Time"],
 "spans_text":[
   {"label":"Workplace/Academic","text":"Internal beta"},
   {"label":"Location&Time","text":"next week"},
   {"label":"Workplace/Academic","text":"NDA"}],
 "explanations":{
   "Workplace/Academic":"Mentions non-public work information.",
   "Location&Time":"Mentions timing that could enable tracking."}
}

Example 6
Post: "Boarding pass code AB12-CD34; QR in pic."
JSON:
{"labels":["Metadata/Device"],
 "spans_text":[
   {"label":"Metadata/Device","text":"AB12-CD34"},
   {"label":"Metadata/Device","text":"QR"}],
 "explanations":{"Metadata/Device":"Codes can expose itineraries or accounts."}
}
""".strip()

def _escape_quotes(s: str) -> str:
    return s.replace('"', '\\"')

def _build_prompt_only(text: str) -> list:
    return [
        {"role": "system", "content": "You are a careful privacy-risk classifier. Output ONLY valid JSON."},
        {"role": "user", "content": FEWSHOTS + f'\n\nPost: "{_escape_quotes(text)}"\nJSON:\n'}
    ]

def _build_prompt_hybrid(text: str, cands_json: str) -> list:
    note = (
        "\n\nYou are given candidate spans as hints; fix them if wrong and add any that are missing.\n"
        'Candidates JSON (list of {"label","text"}): ' + cands_json +
        f'\n\nPost: "{_escape_quotes(text)}"\nJSON:\n'
    )
    return [
        {"role": "system", "content": "You are a careful privacy-risk classifier. Output ONLY valid JSON."},
        {"role": "user", "content": FEWSHOTS + note}
    ]

# -----------------------
# Robust OpenAI call (retry + backoff + cache)
# -----------------------
_CALL_CACHE: Dict[str, Dict] = {}

def _cache_key(messages: list, model: str, max_tokens: int) -> str:
    sys = next((m["content"] for m in messages if m["role"] == "system"), "")
    usr = next((m["content"] for m in messages if m["role"] == "user"), "")
    return f"{model}|{max_tokens}|{hash(sys)}|{hash(usr)}"

def _generate_openai(messages: list, max_tokens: int = 220, model: str = OPENAI_MODEL) -> Dict:
    key = _cache_key(messages, model, max_tokens)
    if key in _CALL_CACHE:
        return _CALL_CACHE[key]

    base_delay = 0.25
    max_retries = 6
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or "{}"
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                parsed = {}
            out = {
                "labels": parsed.get("labels", []),
                "spans_text": parsed.get("spans_text", []),
                "explanations": parsed.get("explanations", {}) or {},
            }
            _CALL_CACHE[key] = out
            return out
        except Exception as e:
            msg = str(e).lower()
            retryable = ("rate limit" in msg) or ("429" in msg) or ("timeout" in msg) or ("502" in msg) or ("503" in msg) or ("504" in msg)
            if attempt == max_retries or not retryable:
                return {"labels": [], "spans_text": [], "explanations": {}}
            sleep_s = (base_delay * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random())
            time.sleep(sleep_s)

# -----------------------
# Span offset mapper
# -----------------------
def _attach_offsets(text: str, spans_text: List[Dict]) -> List[Dict]:
    cursor = 0
    results = []
    for s in spans_text or []:
        frag = s.get("text", "")
        if not frag:
            continue
        start = text.find(frag, cursor)
        if start == -1:
            start = text.lower().find(frag.lower(), cursor)
        if start == -1:
            start = text.find(frag)
            if start == -1:
                start = text.lower().find(frag.lower())
        if start == -1:
            continue
        end = start + len(frag)
        results.append({"start": start, "end": end, "text": text[start:end], "label": s.get("label","")})
        cursor = end
    return results

# -----------------------
# Public classification APIs
# -----------------------
def predict_rules(text: str) -> Dict:
    out = det.analyze(text)
    return {"labels": out.get("labels", []), "spans": out.get("spans", []), "explanations": {}}

def predict_prompt_only(text: str) -> Dict:
    msgs = _build_prompt_only(text)
    out = _generate_openai(msgs)
    labels = out.get("labels") or []
    spans_text = out.get("spans_text") or []
    if any(l for l in labels if l != "None") and "None" in labels:
        labels = [l for l in labels if l != "None"]
    spans = _attach_offsets(text, spans_text)
    return {"labels": labels, "spans": spans, "explanations": out.get("explanations", {}) or {}}

def predict_hybrid(text: str) -> Dict:
    # Derive a small set of candidate spans from rules/NER to keep prompts compact
    base = det.analyze(text)
    seen = set()
    hints = []
    for s in base.get("spans", []):
        k = (s["label"], s["text"])
        if k in seen:
            continue
        seen.add(k)
        hints.append({"label": s["label"], "text": s["text"]})
        if len(hints) >= 6:
            break
    msgs = _build_prompt_hybrid(text, json.dumps(hints, ensure_ascii=False))
    out = _generate_openai(msgs)
    labels = out.get("labels") or []
    spans_text = out.get("spans_text") or []
    if any(l for l in labels if l != "None") and "None" in labels:
        labels = [l for l in labels if l != "None"]
    spans = _attach_offsets(text, spans_text)
    return {"labels": labels, "spans": spans, "explanations": out.get("explanations", {}) or {}}

# Optional: explicit export list
__all__ = ["predict_rules", "predict_prompt_only", "predict_hybrid"]
