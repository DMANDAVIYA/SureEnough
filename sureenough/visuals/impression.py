from sureenough.narrate.display import category_phrase, with_article

STYLE = "Simple, centred, on a neutral grey background."
MAX_NAMES = 8


def impression_prompt(claim: dict, member_names: list[str]) -> str:
    if claim["kind"] == "single":
        subject = with_article(member_names[0])
    elif claim["kind"] == "category":
        subject = category_phrase(claim["category"])
    elif claim["kind"] == "full":
        subject = "an everyday object that cannot be identified yet"
    else:
        subject = "one of: " + ", ".join(member_names[:MAX_NAMES])
    return f"An artist's impression of {subject}. {STYLE}"
