import pandas as pd

from hrflow_recsys.models.markov import Markov


def test_counts_transitions_and_excludes_seen_jobs():
    x = pd.DataFrame({"session_id": [0, 1], "job_ids": [[1, 2], [2, 3]]})
    y = pd.DataFrame({"session_id": [0, 1], "job_id": [3, 4]})
    model = Markov().fit(x, y)
    assert model.transitions[2][3] == 2
    assert model.transitions[3][4] == 1
    assert model.votes([1, 2]) == {3: 2.0}
    assert 3 not in model.votes([2, 3])
