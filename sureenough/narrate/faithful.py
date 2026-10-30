import re
from functools import lru_cache

COMMON_WORD_CONCEPTS = frozenset({"can", "top", "watch", "match", "spring", "case", "bar", "file", "set", "mold"})


@lru_cache(maxsize=None)
def _term(term: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)


def faithful(text: str, allowed: set[str], vocabulary: set[str]) -> bool:
    if re.search(r"\bprobabl", text, re.IGNORECASE):
        return False
    remaining = text
    for term in sorted(allowed, key=len, reverse=True):
        remaining = _term(term).sub(" ", remaining)
    forbidden = vocabulary - set(allowed) - COMMON_WORD_CONCEPTS
    return not any(_term(term).search(remaining) for term in forbidden)
