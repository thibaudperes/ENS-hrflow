import pytest

from hrflow_recsys.metrics import action_accuracy, final_score, mrr_at_k

RECS = list(range(1, 11))


def test_target_first():
    assert mrr_at_k([RECS], [1]) == 1.0


def test_target_third():
    assert mrr_at_k([RECS], [3]) == pytest.approx(1 / 3)


def test_target_missing():
    assert mrr_at_k([RECS + [11]], [11]) == 0.0


def test_mean_and_final_score():
    mrr = mrr_at_k([RECS, RECS, RECS], [1, 3, 99])
    acc = action_accuracy(["view", "apply"], ["view", "view"])
    assert mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert acc == 0.5
    assert final_score(mrr, acc) == pytest.approx(0.7 * mrr + 0.15)
