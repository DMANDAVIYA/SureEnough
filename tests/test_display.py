from types import SimpleNamespace

from sureenough.narrate.display import display_claim
from sureenough.narrate.faithful import faithful

IDS = [f"c{i}" for i in range(8)]
NAMES = ["wolf", "husky", "antelope", "sofa", "apple", "bread", "rice", "mystery"]


def _concepts(categories):
    return SimpleNamespace(test_things_ids=IDS, test_names=NAMES, categories53=categories)


BASE = {
    "animal": {"c0", "c1", "c2"},
    "mammal": {"c0", "c1"},
    "pet": {"c0", "c1"},
    "furniture": {"c3"},
    "food": {"c4", "c5", "c6"},
}


def test_singleton_names_the_object_with_article():
    assert display_claim([0], _concepts(BASE)) == {"kind": "single", "text": "It's a wolf.", "category": None, "members": [0]}
    assert display_claim([2], _concepts(BASE))["text"] == "It's an antelope."


def test_uncategorised_member_falls_back_to_list():
    claim = display_claim([0, 7], _concepts(BASE))
    assert claim["kind"] == "list" and claim["text"] == "It's one of these 2."


def test_category_covering_more_than_a_quarter_is_not_used():
    claim = display_claim([4, 5], _concepts(BASE))
    assert claim["kind"] == "list"


def test_smallest_qualifying_category_wins_then_alphabetical():
    cats = {**BASE, "animal": {"c0", "c1"}}
    claim = display_claim([0, 1], _concepts(cats))
    assert claim["kind"] == "category" and claim["category"] == "animal"
    assert claim["text"] == "It's an animal."


def test_mass_noun_category_has_no_article():
    cats = {**BASE, "furniture": {"c3", "c7"}}
    assert display_claim([3, 7], _concepts(cats))["text"] == "It's furniture."


def test_full_gallery_is_not_sure_yet():
    claim = display_claim(list(range(8)), _concepts(BASE))
    assert claim["kind"] == "full" and claim["text"] == "Not sure yet: it could be any of the 8."


VOCAB = {"wolf", "husky", "dog", "hot dog", "animal", "can"}


def test_faithful_accepts_only_claimed_terms():
    assert faithful("It's one of these: wolf, husky.", {"wolf", "husky"}, VOCAB)
    assert not faithful("It's a wolf, husky or dog.", {"wolf", "husky"}, VOCAB)


def test_faithful_rejects_probably():
    assert not faithful("It's probably a wolf.", {"wolf"}, VOCAB)


def test_faithful_ignores_terms_inside_allowed_phrases():
    assert faithful("It's a hot dog.", {"hot dog"}, VOCAB)


def test_faithful_tolerates_common_word_concepts():
    assert faithful("It can only be a wolf.", {"wolf"}, VOCAB)
