import numpy as np
import pandas as pd

from hrflow_recsys.drift import drift_table


def test_detects_shift_and_not_noise():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"same": rng.normal(0, 1, 2000), "shifted": rng.normal(0, 1, 2000)})
    current = pd.DataFrame({"same": rng.normal(0, 1, 500), "shifted": rng.normal(1, 1, 500)})
    table = drift_table(reference, current).set_index("feature")
    assert not table.loc["same", "drift"]
    assert table.loc["shifted", "drift"]
