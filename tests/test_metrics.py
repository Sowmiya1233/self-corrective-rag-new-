from src.metrics import exact_match, f1_score, normalize


def test_normalize():
    assert normalize("The Eiffel Tower!") == "eiffel tower"


def test_em_f1():
    assert exact_match("The Beatles", "beatles") == 1.0
    assert f1_score("Paris France", "Paris") == 2 * 0.5 * 1 / 1.5
    assert f1_score("London", "Paris") == 0.0
