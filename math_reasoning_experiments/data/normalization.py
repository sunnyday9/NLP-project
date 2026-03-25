from __future__ import annotations

import re
from fractions import Fraction
from typing import Tuple

from .schema import DatasetName, NormalizedAnswer


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:/[1-9]\d*)?")


def _strip_latex(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\\[a-zA-Z]+", " ", cleaned)  # remove LaTeX commands
    cleaned = cleaned.replace("$", " ")
    cleaned = cleaned.replace(r"\left", " ").replace(r"\right", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _parse_number_token(token: str) -> Tuple[str, str]:
    token = token.strip()
    if "/" in token:
        try:
            frac = Fraction(token)
            return str(frac), "float"
        except Exception:
            return token, "expression"
    try:
        if "." in token:
            return str(float(token)), "float"
        return str(int(token)), "integer"
    except Exception:
        return token, "expression"


def normalize_gold_answer(raw: str, dataset: DatasetName) -> NormalizedAnswer:
    """
    Normalize gold answer from dataset.
    """
    text = raw.strip()

    # AIME: usually 0-999 integer
    if dataset == "aime2024":
        digits = re.findall(r"\d+", text)
        value = digits[-1] if digits else text
        try:
            normalized = str(int(value))
            return NormalizedAnswer(raw=text, normalized=normalized, type="integer")
        except Exception:
            return NormalizedAnswer(raw=text, normalized=value.strip(), type="string")

    # First try boxed in LaTeX (MATH-500 style)
    boxed = _BOXED_RE.findall(text)
    if boxed:
        inner = boxed[-1].strip()
        token, kind = _parse_number_token(inner) if _NUMBER_RE.fullmatch(inner) else (inner, "expression")
        return NormalizedAnswer(raw=text, normalized=token, type=kind)

    # GSM8K / general: take last numeric-looking token
    cleaned = _strip_latex(text)
    matches = list(_NUMBER_RE.finditer(cleaned))
    if matches:
        token = matches[-1].group(0)
        norm, kind = _parse_number_token(token)
        return NormalizedAnswer(raw=text, normalized=norm, type=kind)

    # fallback: normalized plain string
    norm = cleaned.strip()
    return NormalizedAnswer(raw=text, normalized=norm, type="string")


def normalize_prediction(raw_output: str, dataset: DatasetName) -> NormalizedAnswer:
    """
    Normalize model prediction for comparison with gold answer.
    """
    text = raw_output.strip()

    # If model already outputs explicit answer snippet, often numeric at end.
    cleaned = _strip_latex(text)
    matches = list(_NUMBER_RE.finditer(cleaned))
    if matches:
        token = matches[-1].group(0)
        norm, kind = _parse_number_token(token)
        return NormalizedAnswer(raw=text, normalized=norm, type=kind)

    # Fallback to full string
    return NormalizedAnswer(raw=text, normalized=cleaned, type="string")

