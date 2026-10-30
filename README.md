# HrFlow.ai session-based job recommendation

[![CI](https://github.com/thibaudperes/ens-challenge-hrflow-recsys/actions/workflows/ci.yml/badge.svg)](https://github.com/thibaudperes/ens-challenge-hrflow-recsys/actions/workflows/ci.yml)

Predict the next job offers a candidate will open, and whether they will view or apply,
from their browsing session (ENS Challenge Data × HrFlow.ai).

**[Live demo](https://hrflow-api-243544005364.europe-west1.run.app/)**: a real test session
and the 10 offers the model recommends next, with their titles.
The first call can take ~15 s (the service scales to zero).

- **Model:** a hybrid of Markov transitions, offer-id proximity and TF-IDF text similarity.
  MRR@10 0.094 on a validation grouped by session chain, against 0.005 for popularity.
- **Rigour:** 60.5 % of train sessions are cut pieces of one browsing. The leak is measured,
  kept out of the model, and the validation is grouped by chain to avoid it.
- **Production:** FastAPI service, Docker image, GitHub Actions CI on every pull request,
  continuous deployment to Google Cloud Run, experiments tracked with MLflow, input drift check.

## Architecture

```mermaid
flowchart LR
    data[Challenge data<br/>local only] --> train[train_final.py]
    data --> eval[run_eval.py]
    eval --> mlflow[MLflow<br/>runs and metrics]
    train --> bucket[(Cloud Storage<br/>private bucket<br/>hybrid.joblib, demo.json)]
    repo[GitHub repo] --> ci[GitHub Actions<br/>ruff, pytest, docker build]
    repo -->|push to main| build[Cloud Build<br/>Docker image]
    build --> run[Cloud Run<br/>FastAPI]
    bucket -->|mounted read-only on /models| run
    run --> users[Demo page and<br/>POST /recommend]
```

## Results

Validation: 20 % of train sessions, grouped by chain of session fragments (see below).

| recommender | MRR@10 | MRR@10 cold start (n=259) |
|---|---:|---:|
| Popularity | 0.0052 | 0.0000 |
| IdNeighbours | 0.0216 | 0.0000 |
| Markov | 0.0794 | 0.0000 |
| TextModel | 0.0263 | 0.0178 |
| Hybrid | 0.0899 | 0.0259 |
| Hybrid + online | 0.0937 | 0.0247 |

| action rule | accuracy |
|---|---:|
| always_view | 0.5886 |
| last_action | 0.8574 |
| ActionModel | 0.8558 |

Final score (0.7 x MRR@10 + 0.3 x accuracy), grouped by chain: **0.3228** with Hybrid + online
and last_action.

Separately, exploiting the session fragment leak gives a final score of **0.6916**
(notebook 02, random split: MRR@10 0.6236, accuracy 0.8501).

## Problem

Each session is a sequence of job offers a candidate opened, each with an action (view or apply).
Predict the next 10 job offers opened, ranked, and the next action (view or apply).
Score = 0.7 x MRR@10 + 0.3 x accuracy. Data: 15,882 train and 1,819 test sessions, 21,917 offers.

Challenge page: https://challengedata.ens.fr/challenges/164

## Approach

**Baselines.** Popularity recommends the most clicked jobs. IdNeighbours recommends the
jobs whose id is closest to the last job, then fills with popularity. Jobs already seen in
the session are always excluded: only 0.6 % of targets are in the session history (all train
sessions, notebook 01).

**Markov.** Transitions are counted between consecutive jobs, plus last job -> target.
The last 3 jobs of the session vote for their successors with weights 1, 0.5 and 0.25.
Missing slots are filled with IdNeighbours.

**Text.** TF-IDF on the job texts. A session profile is the mean of its job vectors, with
weights growing towards the end. Jobs are ranked by cosine similarity, in batches of 1,000
sessions. It is the only model that reaches cold start targets.

**Hybrid.** Candidates are the top 50 Markov votes, the jobs within 50 ids of the last job
and the top 50 text matches. Each has three scores (Markov vote, id proximity, text
similarity), divided by their max in the session. Weights (1, 0.5, 0.5) were picked from
a grid of 30 on half A of the validation (MRR@10 0.0917) and checked on half B (0.0880),
both halves grouped by chain.

**Action.** Repeating the last action of the session is right 85.7 % of the time (grouped by
chain). A logistic regression on 5 session features (last action, final streak, apply share,
switches, length) does not beat it, so last_action is kept.

**Online update.** Before predicting, Markov adds the transitions inside the sessions to
predict, never their targets. This adds 0.0038 of MRR@10 (grouped by chain).

## Session fragments

Many sessions are consecutive pieces of one browsing. Measured on all train sessions: in
60.5 %, the target is the first job of session s+1, and the action then matches 99.99 % of the time.
Controls: the same match with s-1 holds for 0.03 % of sessions, with s+2 for 0.01 %.
The 15,882 train sessions form 6,267 chains of up to 14 sessions.

A random split puts pieces of the same browsing on both sides. MRR@10 goes from 0.0890
(random split) to 0.0794 (grouped by chain) for Markov, and from 0.0990 to 0.0899 for Hybrid.
The grouped split is the reference validation.

Putting the first job of s+1 at rank 1, and keeping last_action, raises the final score to
0.6916 (random split). On the test, links cannot be checked without y. A proxy (first job of
s+1 within 100 ids of the last job of s) holds for 11.4 % of test sessions against 12.0 % of
all train sessions, which gives an estimated 56 % of real links in the test, against 60.5 %
in train.

This is an artefact of how the data was cut, not a property of candidate behaviour. It is
kept out of the model: it lives in `fragments.with_next_first` and only produces
`submission_leak.csv`. All results in the first tables are without it.

## What didn't work

- **A model for the next action.** I tried a logistic regression on a few session
  features (last action, length of the final streak, share of applies, number of
  switches). It never beat the one-line rule "repeat the last action": 85.6 % vs
  85.7 % accuracy. On the first validation split it disagreed with the rule on only
  21 sessions out of 3,176, and got most of those wrong. I kept the rule.

- **Normalising the Markov votes.** Dividing each job's votes by its number of
  outgoing transitions looked cleaner, but MRR dropped from 0.089 to 0.085 (random
  split). It gives a job seen once with a single successor as much weight as a job
  seen fifty times, and those rare transitions are mostly noise.

- **Text on its own.** TF-IDF similarity alone only reaches 0.026 MRR. Browsing seems
  to follow the order jobs are listed in more than their content (see the id-gap
  analysis in notebook 01). The text is still useful for targets the other models
  have never seen, which is why it stays as one of the three signals in the hybrid.

- **Online updates for cold start.** Feeding the validation sequences back into the
  transition counts helps overall (+0.004 MRR), but not on unseen targets
  (0.026 -> 0.025). I expected it to help there and it didn't.

## Production

**Service.** A FastAPI app serves the Hybrid recommender and the last_action rule.

| endpoint | what it does |
|---|---|
| `GET /` | demo page: a random test session and its 10 recommended offers |
| `POST /recommend` | session in, 10 unseen job ids (best first) and the next action out |
| `GET /demo/random` | the data behind the demo page, with offer titles and summaries |
| `GET /health` | liveness check |

```bash
curl -X POST https://hrflow-api-243544005364.europe-west1.run.app/recommend \
  -H "Content-Type: application/json" \
  -d '{"job_ids": [305, 299, 300], "actions": ["view", "view", "apply"]}'
```

Unknown job ids and mismatched lengths are rejected with a 422. `tests/test_api.py` checks the
API with a small model fitted on synthetic data, so the CI runs without the challenge files.

**Deployment.** `scripts/train_final.py` fits the Hybrid on all train sessions and writes
`models/hybrid.joblib` and `models/demo.json` (offer snippets and 50 test sessions). Both files
are built from the challenge data and are not committed: they live in a private Cloud Storage
bucket, mounted read-only on `/models` by Cloud Run. The Docker image holds only the code and
its locked dependencies. Every push to `main` triggers Cloud Build, which builds the image and
deploys a new revision.

**Quality gates.** `main` is protected: changes go through pull requests, and the `tests` job
(ruff, ruff format, pytest on dependencies locked with uv) must pass before merging. A second
job builds the Docker image.

**Experiment tracking.** Each `run_eval.py` launch is logged to MLflow (local `mlflow.db`): one
parent run with the split, one nested run per recommender (MRR@10, cold start MRR@10, time,
hybrid weights) and per action rule (accuracy), and the final score.

**Monitoring.** `scripts/check_drift.py` compares incoming sessions with the training sessions.
Each session is reduced to 7 features (the 5 action features, the median job id, the share of
jobs never seen in training), and each feature is compared with a two-sample Kolmogorov-Smirnov
test (drift if p < 0.01). Train vs test of the challenge: no drift on the 6 session features
(e.g. mean length 7.54 vs 7.61). 6.7 % of the jobs opened in test sessions were never seen in
training: Markov cannot score them and only the text model reaches them. The unseen share is
always 0 in the reference by construction, so in production it should be watched with a plain
threshold rather than the KS test. Evidently was not used: its current release does not import
with pandas 3.1.

## Reproduce

Put the challenge files in `data/raw/` (`x_train.csv`, `y_train.csv`, `x_test.csv`,
`job_listings.json`). They are not provided: download them after registering for the challenge.

```bash
uv sync
uv run python scripts/run_eval.py         # validation tables, logged to MLflow
uv run mlflow ui --port 5001              # browse logged runs at http://127.0.0.1:5001
uv run python scripts/tune_hybrid.py      # hybrid weight grid, halves A and B
uv run python scripts/make_submission.py  # submissions/submission_model.csv and submission_leak.csv
uv run python scripts/check_drift.py      # train vs test drift table
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_session_fragments.ipynb
```

Run the API locally, directly or with Docker (the model files stay outside the image):

```bash
uv run python scripts/train_final.py              # models/hybrid.joblib and models/demo.json
uv run fastapi dev src/hrflow_recsys/serve.py     # http://127.0.0.1:8000
docker build -t hrflow-api .
docker run --rm -p 8080:8080 -v "$(pwd)/models:/models" hrflow-api   # http://127.0.0.1:8080
```

## Project layout

```
notebooks/
  01_eda.ipynb                 data exploration
  02_session_fragments.ipynb   session chains and the leak
scripts/                       run_eval, tune_hybrid, make_submission, train_final, check_drift
src/hrflow_recsys/
  data.py  metrics.py  split.py  evaluate.py
  action.py                    action prediction
  fragments.py                 session chains
  models/                      baselines, markov, text, hybrid
  api.py  serve.py             FastAPI app and its entry point
  demo_page.py                 HTML of the demo page
  drift.py                     session profile and KS drift test
tests/
Dockerfile  .github/workflows/ci.yml
```

## Limits and next steps

- No sequence model such as GRU4Rec or SASRec: transitions are first order, over the last 3 jobs.
- Hybrid weights come from a simple grid of 30 combinations; a learned ranker could replace it.
- Action accuracy is stuck around 85 %: session features add nothing beyond the last action.
- The drift check runs by hand on the challenge test set; the API does not log incoming sessions yet.
