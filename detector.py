# detector.py
from typing import List, Dict, Any, Tuple
import re

# Optional spaCy NER
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None
    _SPACY_AVAILABLE = False

# Public label set
LABELS = [
    "Location&Time",
    "Contact&IDs",
    "Gov&Financial IDs",
    "Health&Sensitive",
    "Credentials&Security",
    "Workplace/Academic",
    "Minors",
    "Metadata/Device",
]

# -------- Contact & IDs --------
PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)

# street-like addresses (conservative)
ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9.\-']+\s)+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Way)\b",
    re.I,
)
ADDRESS_HINTS = [
    " apt ", " suite ", " unit ",
    " street", " st.", " st ", " avenue", " ave", " blvd", " road", " rd", " lane", " ln", " drive", " dr", " way"
]

# -------- Gov/Financial IDs --------
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Credit-like digit runs; precision gate uses nearby keywords
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
POLICY = re.compile(r"\b(?:DL|ID|POLICY|POL|INS|LIC)[ -]?[A-Z0-9]{2,}[- ]?[A-Z0-9]{2,}\b", re.I)
CARD_KEYWORDS = ("card", "visa", "mastercard", "amex", "credit", "debit", "number", "cvv")

# -------- Credentials & Security --------
OTP6 = re.compile(r"\b\d{6}\b")
PIN4 = re.compile(r"\b\d{4}\b")
STRIPE = re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]+\b")
SLACK = re.compile(r"\bxox[aboprs]-[A-Za-z0-9-]+\b")
SSH = re.compile(r"\bAAAAB3Nza[^\s]{10,}\b")
TOKEN_WORDS = ["token", "password", "passcode", "pin", "otp", "secret", "webhook", "api key", "login", "passphrase"]
# simple password-like "key: value" pattern
PASS_ASSIGN = re.compile(r"\b(pass|password|passcode)\b\s*[:=]\s*(\S+)", re.I)

# -------- Metadata / Device --------
QR = re.compile(r"\bqr\b", re.I)
BARCODE = re.compile(r"\bbarcode\b", re.I)
PLATE = re.compile(r"\blicense plate\b", re.I)
VIN = re.compile(r"\b[0-9A-HJ-NPR-Z]{17}\b")  # VIN excludes I/O/Q
TRACKING = re.compile(r"\b1Z[0-9A-Z ]{12,}\b")
RESCODE = re.compile(r"\b[A-Z0-9]{2,4}-[A-Z0-9]{2,4}\b")
SERIAL = re.compile(r"\bSN[- ]?[A-Z0-9-]{4,}\b", re.I)

# -------- Location & Time --------
TIME = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b|\b(?:\d{1,2})(?::\d{2})?\s?(?:am|pm)\b", re.I)
TIME_WORDS = [
    "am","pm","noon","midnight","tonight","tomorrow","today",
    "friday","saturday","sunday","monday","tuesday","wednesday","thursday",
    "next week","this weekend","tonite","tmrw"
]

# -------- Health & Sensitive --------
HEALTH_WORDS = [
    "ocd","ptsd","insulin","covid","ssri","strep","methadone","migraine","sober",
    "therapy","therapist","diagnosed","depression","anxiety","hiv","diabetes","cancer","aura","clinic"
]

# -------- Workplace / Academic --------
WORK_WORDS = [
    "nda","private repo","confidential","client","launch date","beta","irb","pilot",
    "pay bands","internal","anonymized transcripts","dataset","grades","midterm","study",
    "partner hospital","api key","staging","do not share","under embargo","non-public"
]

# -------- Minors --------
MINOR_CUES = [
    "my son","my daughter","my kid","my toddler","my niece","my nephew","kids’","kids'",
    "grade","homeroom","elementary","middle","high school","preschool","bus stop","carpool","room "
]


def find_spans(text: str, needle: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    start = 0
    low = text.lower()
    nlow = needle.lower()
    while True:
        i = low.find(nlow, start)
        if i == -1:
            break
        spans.append((i, i + len(needle), text[i:i + len(needle)]))
        start = i + len(needle)
    return spans


class Detector:
    def __init__(self, enable_spacy: bool = False):
        self.enable_spacy = enable_spacy and _SPACY_AVAILABLE
        self.nlp = None
        if self.enable_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.enable_spacy = False
                self.nlp = None

    def _add(self, out: Dict[str, Any], label: str, start: int, end: int, text: str):
        out.setdefault("labels", set()).add(label)
        out.setdefault("spans", {}).setdefault(label, []).append(
            {"start": start, "end": end, "text": text, "label": label}
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"labels": set(), "spans": {}, "explanations": {}}
        tlow = text.lower()

        # Contact & IDs
        for m in PHONE.finditer(text):
            self._add(result, "Contact&IDs", m.start(), m.end(), m.group())
        for m in EMAIL.finditer(text):
            self._add(result, "Contact&IDs", m.start(), m.end(), m.group())
        for m in ADDRESS.finditer(text):
            self._add(result, "Contact&IDs", m.start(), m.end(), m.group())
        for hint in ADDRESS_HINTS:
            for s, e, _ in find_spans(tlow, hint):
                # require at least 10 chars before the suffix to avoid “yesterday”
                if s >= 10:
                    self._add(result, "Contact&IDs", s, e, text[s:e])

        # Gov/Financial IDs
        for m in SSN.finditer(text):
            self._add(result, "Gov&Financial IDs", m.start(), m.end(), m.group())
        for m in CARD.finditer(text):
            window = text[max(0, m.start() - 16):m.end() + 16].lower()
            if any(k in window for k in CARD_KEYWORDS):
                # rough Luhn-like length filter to cut random digit runs
                digits = re.sub(r"\D", "", m.group())
                if 13 <= len(digits) <= 19:
                    self._add(result, "Gov&Financial IDs", m.start(), m.end(), m.group())
        for m in POLICY.finditer(text):
            self._add(result, "Gov&Financial IDs", m.start(), m.end(), m.group())

        # Credentials & Security
        for m in STRIPE.finditer(text):
            self._add(result, "Credentials&Security", m.start(), m.end(), m.group())
        for m in SLACK.finditer(text):
            self._add(result, "Credentials&Security", m.start(), m.end(), m.group())
        for m in SSH.finditer(text):
            self._add(result, "Credentials&Security", m.start(), m.end(), m.group())
        for m in PASS_ASSIGN.finditer(text):
            self._add(result, "Credentials&Security", m.start(), m.end(), m.group())
        if any(w in tlow for w in TOKEN_WORDS):
            for m in PIN4.finditer(text):
                self._add(result, "Credentials&Security", m.start(), m.end(), m.group())
            for m in OTP6.finditer(text):
                self._add(result, "Credentials&Security", m.start(), m.end(), m.group())

        # Metadata / Device
        for rx in (QR, BARCODE, PLATE, VIN, TRACKING, RESCODE, SERIAL):
            for m in rx.finditer(text):
                self._add(result, "Metadata/Device", m.start(), m.end(), m.group())

        # Location & Time
        for m in TIME.finditer(text):
            self._add(result, "Location&Time", m.start(), m.end(), m.group())
        for w in TIME_WORDS:
            for s, e, _ in find_spans(tlow, w):
                self._add(result, "Location&Time", s, e, text[s:e])

        # Health & Sensitive
        for w in HEALTH_WORDS:
            for s, e, _ in find_spans(tlow, w):
                self._add(result, "Health&Sensitive", s, e, text[s:e])

        # Workplace / Academic
        for w in WORK_WORDS:
            for s, e, _ in find_spans(tlow, w):
                self._add(result, "Workplace/Academic", s, e, text[s:e])

        # Minors
        for w in MINOR_CUES:
            for s, e, _ in find_spans(tlow, w):
                self._add(result, "Minors", s, e, text[s:e])

        # spaCy boosts (optional)
        if self.enable_spacy and self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"GPE", "LOC", "FAC"}:
                    self._add(result, "Location&Time", ent.start_char, ent.end_char, ent.text)
                if ent.label_ in {"DATE", "TIME"}:
                    self._add(result, "Location&Time", ent.start_char, ent.end_char, ent.text)
                if ent.label_ == "ORG":
                    if any(k in ent.text.lower() for k in ["university", "clinic", "hospital", "school", "corp", "inc", "labs"]):
                        self._add(result, "Workplace/Academic", ent.start_char, ent.end_char, ent.text)

        # finalize
        result["labels"] = sorted(list(result["labels"]))
        flat_spans: List[Dict[str, Any]] = []
        for label, spans in result["spans"].items():
            flat_spans.extend(spans)
        result["spans"] = flat_spans
        return result


def candidates_from_text(det: "Detector", text: str) -> List[Dict[str, Any]]:
    """
    Convenience helper for the hybrid FLAN pass.
    Returns [{"label": ..., "text": ...}, ...] extracted by the rules/NER detector.
    """
    out = det.analyze(text)
    return [{"label": s["label"], "text": s["text"]} for s in out.get("spans", [])]


# Optional local smoke test
if __name__ == "__main__":
    d = Detector(enable_spacy=True)
    print(d.analyze("Utility wants my SSN: 123-45-6789. Door PIN is 4421, WiFi password BlueHouse!."))
