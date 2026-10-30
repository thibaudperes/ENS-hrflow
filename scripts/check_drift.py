from hrflow_recsys.data import load_data
from hrflow_recsys.drift import drift_table, session_profile


def main() -> None:
    x_train, y_train, x_test, _ = load_data()
    known_jobs = set(x_train["job_ids"].explode()) | set(y_train["job_id"])
    reference = session_profile(x_train, known_jobs)
    current = session_profile(x_test, known_jobs)
    table = drift_table(reference, current)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    drifted = table.loc[table["drift"], "feature"].tolist()
    print(f"\nDrift on {len(drifted)}/{len(table)} features: {drifted or 'none'}")


if __name__ == "__main__":
    main()
