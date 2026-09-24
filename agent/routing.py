"""Question routing helpers shared by the supervisor and tests."""

from __future__ import annotations

import re

CONCEPT_HINTS = (
    "what is", "what's", "what are", "what does", "explain",
    "how does", "how do", "how will", "how can", "how to",
    "definition", "meaning", "in material", "textbook",
    "theory", "principle", "difference between", "sample prep",
    "prepare a", "why is", "why does",
)

PREDICT_HINTS = (
    "brittle", "ductile", "fracture", "will it fail", "axle",
    "crack", "classify", "predict", "risk",
)


def extract_hv(text: str) -> float | None:
    m = re.search(r"(?:hv10|hv)\s*[:=]\s*(\d{2,4}(?:\.\d+)?)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d{3,4}(?:\.\d+)?)\s*(?:hv10|hv)\b", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"hardness\s*[:=]\s*(\d{2,4}(?:\.\d+)?)", text, re.I)
    return float(m.group(1)) if m else None


def extract_profile(text: str) -> int | None:
    m = re.search(r"(?:profile|session|cycle)\s*(?:id|#)?\s*[:=]?\s*(\d{1,3})", text, re.I)
    return int(m.group(1)) if m else None


def is_conceptual(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in CONCEPT_HINTS)


def wants_prediction(text: str) -> bool:
    t = text.lower()
    if is_conceptual(t):
        return False
    return any(k in t for k in PREDICT_HINTS)


def route_specialists(
    question: str,
    *,
    image_path: str | None = None,
    motor_profile_id: int | None = None,
) -> list[str]:
    """Supervisor: choose which specialist agents should run."""
    agents = ["standards", "knowledge"]
    hv = extract_hv(question)
    profile = motor_profile_id if motor_profile_id is not None else extract_profile(question)
    if hv is not None or wants_prediction(question):
        agents.append("fracture")
    if image_path:
        agents.append("vision")
    if profile is not None or "motor" in question.lower() or "winding" in question.lower():
        agents.append("twin")
    if "fracture" in agents or "vision" in agents:
        agents.append("critic")
    return agents
