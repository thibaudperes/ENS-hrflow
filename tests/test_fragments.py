import pandas as pd

from hrflow_recsys.fragments import build_chains, with_next_first


def test_build_chains_links_only_matching_next_session():
    x = pd.DataFrame({"session_id": [0, 1, 2], "job_ids": [[10, 11], [20, 21], [30, 31]]})
    y = pd.DataFrame({"session_id": [0, 1, 2], "job_id": [20, 99, 40]})
    chains = build_chains(x, y)
    assert chains[0] == chains[1] != chains[2]
    assert chains.nunique() == 2


def test_with_next_first_puts_next_job_first_only_when_next_exists():
    x = pd.DataFrame(
        {
            "session_id": [0, 1],
            "job_ids": [[1, 2], [7, 8]],
            "actions": [["view", "view"], ["apply", "view"]],
        }
    )
    recs = [list(range(100, 110)), list(range(100, 110))]
    result = with_next_first(recs, x, x)
    assert result[0] == [7] + list(range(100, 109))
    assert result[1] == recs[1]
