from sureenough.meta_api import output_text, spark_payload
from sureenough.narrate.faithful import faithful

SYSTEM = (
    "Rewrite the claim about which object a person saw as one short, plain English sentence. "
    "Use only the object or category names you are given and add no other objects or categories. "
    "Never hedge with words like probably, likely or maybe, and give no percentages. "
    "Do not mention brains, EEG or models."
)
EST_COST = 0.002


def narrate(client, claim: dict, member_names: list[str], vocabulary: set[str]) -> dict:
    payload = spark_payload(SYSTEM, f"Claim: {claim['text']} Names: {', '.join(member_names)}.")
    body = client.responses(payload, est_cost=EST_COST)
    text = output_text(body).strip()
    allowed = set(member_names) | ({claim["category"]} if claim["category"] else set())
    ok = bool(text) and faithful(text, allowed, vocabulary)
    return {"text": text if ok else claim["text"], "faithful": ok, "request": payload, "response": body}
