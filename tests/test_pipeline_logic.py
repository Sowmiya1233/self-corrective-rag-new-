from src.pipeline import adaptive_depth, adaptive_threshold
from src.verifier import SUPPORTED, UNSUPPORTED, faithfulness


def test_adaptive_threshold_monotonic():
    assert adaptive_threshold(0.8) < adaptive_threshold(0.5) < adaptive_threshold(0.2)


def test_adaptive_depth():
    assert adaptive_depth(0.8) < adaptive_depth(0.5) < adaptive_depth(0.2)


def test_faithfulness():
    r = [{"verdict": SUPPORTED}, {"verdict": UNSUPPORTED}]
    assert faithfulness(r) == 0.5
    assert faithfulness([]) == 0.0
