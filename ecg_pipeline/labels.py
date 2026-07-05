"""AAMI EC57 5-superclass beat labeling.

Symbol->superclass mapping confirmed against the canonical WFDB
`ecgcodes.h` annotation reference (see docs/data-pipeline-architecture.md
for the cross-check that caught and corrected an initial wrong source).
"""

from __future__ import annotations

AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")

SYMBOL_TO_AAMI: dict[str, str] = {
    # N -- Normal, LBBB, RBBB, atrial escape, nodal escape
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # S -- Supraventricular ectopic
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # V -- Ventricular ectopic
    "V": "V",
    "E": "V",
    # F -- Fusion of ventricular and normal
    "F": "F",
    # Q -- Paced, fusion of paced and normal, unclassifiable
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


def to_aami_class(symbol: str) -> str | None:
    """Map a MIT-BIH beat annotation symbol to its AAMI EC57 superclass.

    Returns None for symbols that aren't classified beat annotations at all
    (e.g. rhythm-change markers like "+" or noise markers like "~") -- callers
    must exclude these from the dataset rather than mapping them to a default
    class.
    """
    return SYMBOL_TO_AAMI.get(symbol)
