import pandas as pd

from hrflow_recsys.models.ranker import Ranker


def test_ranker_ranks_the_known_next_job_first():
    words = ["data", "sales", "python", "logistics", "finance", "design"]
    jobs = pd.Series({j: f"{words[j % 6]} {words[(j + 1) % 6]} offer" for j in range(1, 301)})
    # Every session goes down the list two ids at a time: the next job is last - 2.
    starts = range(100, 280)
    x = pd.DataFrame(
        {
            "session_id": [i * 10 for i in range(len(starts))],
            "job_ids": [[s, s - 2, s - 4] for s in starts],
        }
    )
    y = pd.DataFrame({"session_id": x["session_id"], "job_id": [s - 6 for s in starts]})
    model = Ranker().fit(x, y, jobs)
    recs = model.recommend_batch([[201, 199, 197]])[0]
    assert len(recs) == 10
    assert not {201, 199, 197} & set(recs)
    assert recs[0] == 195
