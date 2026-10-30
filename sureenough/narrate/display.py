MAX_CATEGORY_COVERAGE = 0.25
MASS_SUFFIXES = ("clothing", "wear", "ware", "equipment", "food", "jewelry", "lighting", "decor", "furniture")
PHRASES = {"part of car": "part of a car"}


def with_article(noun: str) -> str:
    return f"{'an' if noun[0].lower() in 'aeiou' else 'a'} {noun}"


def category_phrase(category: str) -> str:
    if category in PHRASES:
        return PHRASES[category]
    return category if category.endswith(MASS_SUFFIXES) else with_article(category)


def shared_category(ids: set[str], concepts) -> str | None:
    gallery = set(concepts.test_things_ids)
    limit = MAX_CATEGORY_COVERAGE * len(gallery)
    candidates = [
        (len(members & gallery), name)
        for name, members in concepts.categories53.items()
        if ids <= members and len(members & gallery) <= limit
    ]
    return min(candidates)[1] if candidates else None


def _claim(kind: str, text: str, members: list[int], category: str | None = None) -> dict:
    return {"kind": kind, "text": text, "category": category, "members": members}


def display_claim(members: list[int], concepts) -> dict:
    members = list(members)
    size = len(members)
    if size == len(concepts.test_things_ids):
        return _claim("full", f"Not sure yet: it could be any of the {size}.", members)
    if size == 1:
        return _claim("single", f"It's {with_article(concepts.test_names[members[0]])}.", members)
    category = shared_category({concepts.test_things_ids[i] for i in members}, concepts)
    if category:
        return _claim("category", f"It's {category_phrase(category)}.", members, category)
    return _claim("list", f"It's one of these {size}.", members)
