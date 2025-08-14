# app/prompt.py
import re
from typing import List

DIGIT_RE  = re.compile(r"\b([0-3])\b")              # matches 0..3
LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)  # matches A..D

def build_prompt_numeric(question: str, choices: List[str]) -> str:
    a, b, c, d = choices  # assume exactly 4 choices
    return (
        "You are a medical expert. Choose the best option.\n"
        "Return only one digit: 0, 1, 2, or 3. No words, no punctuation.\n\n"
        f"Question: {question}\n"
        "Options:\n"
        f"0) {a}\n1) {b}\n2) {c}\n3) {d}\n\n"
        "Answer:"
    )

def parse_index(text: str):
    """Extract the first valid index (0–3) from model text."""
    if not text:
        return None
    m = DIGIT_RE.search(text.strip())   # ← use DIGIT_RE here
    return int(m.group(1)) if m else None

def parse_index_or_letter(text: str):
    """Return 0..3 if we find a digit or a letter A..D (mapped to 0..3)."""
    if not text:
        return None
    s = text.strip()
    m = DIGIT_RE.search(s)
    if m:
        return int(m.group(1))
    m = LETTER_RE.search(s)
    if m:
        return "ABCD".index(m.group(1).upper())
    return None
