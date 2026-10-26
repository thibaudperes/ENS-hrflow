from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["last_apply", "final_streak", "apply_share", "switches", "length"]


def session_features(actions: Sequence[str]) -> list[float]:
    applies = np.asarray(actions) == "apply"
    changes = np.flatnonzero(applies[1:] != applies[:-1])
    final_streak = len(applies) - (changes[-1] + 1) if len(changes) else len(applies)
    return [applies[-1], final_streak, applies.mean(), len(changes), len(applies)]


def action_features(x: pd.DataFrame) -> pd.DataFrame:
    rows = [session_features(actions) for actions in x["actions"]]
    return pd.DataFrame(rows, columns=FEATURES, index=x.index, dtype=float)


class ActionModel:
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> "ActionModel":
        sessions = x.merge(y, on="session_id", validate="one_to_one")
        features = action_features(sessions)
        self.scaler = StandardScaler().fit(features)
        self.model = LogisticRegression().fit(
            self.scaler.transform(features), sessions["action"] == "apply"
        )
        return self

    def predict(self, x: pd.DataFrame) -> list[str]:
        is_apply = self.model.predict(self.scaler.transform(action_features(x)))
        return np.where(is_apply, "apply", "view").tolist()
