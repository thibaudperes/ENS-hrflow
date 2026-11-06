"""Draw the README figures from results/*.json into reports/figures/.

No figure contains a session or an offer from the challenge: only rates, scores and a schematic.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from hrflow_recsys.metrics import final_score  # noqa: E402
from hrflow_recsys.viz import PALETTE, save_fig, set_conclusion, set_style  # noqa: E402

RESULTS = Path("results")
MODELS = [
    "Popularity",
    "IdNeighbours",
    "Markov",
    "TextModel",
    "Hybrid",
    "Hybrid + online",
    "Ranker",
    "Ranker + online",
]
SUBMITTED = "Hybrid + online"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def model_color(name: str) -> str:
    if name == SUBMITTED:
        return PALETTE["final"]
    return PALETTE["alternative"] if name.startswith("Ranker") else PALETTE["neutral"]


def ordinal(rank: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")


def label_bars(ax, values, fmt: str) -> None:
    for y, value in enumerate(values):
        ax.annotate(
            format(value, fmt), (value, y), xytext=(5, 0), textcoords="offset points", va="center"
        )


# a. MRR@10 by model --------------------------------------------------------------------------


def mrr_by_model() -> None:
    validation = load("validation")
    mrr, cold = validation["mrr_at_10"], validation["mrr_at_10_cold"]
    colors = [model_color(name) for name in MODELS]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, scores, title in [
        (ax, mrr, f"All validation sessions (n={validation['n_val']:,})"),
        (bx, cold, f"Cold start: target never seen in training (n={validation['n_cold']})"),
    ]:
        values = [scores[name] for name in MODELS]
        axis.barh(range(len(MODELS)), values, color=colors, height=0.65)
        label_bars(axis, values, ".3f")
        axis.set_xlim(0, max(values) * 1.25)
        axis.grid(axis="y", visible=False)
        axis.set_xlabel("MRR@10, validation grouped by chain")
        axis.set_title(title)
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.invert_yaxis()
    gain = mrr[SUBMITTED] / mrr["Popularity"]
    set_conclusion(
        fig,
        f"The submitted hybrid reaches MRR@10 {mrr[SUBMITTED]:.3f}, {gain:.0f}x popularity,\n"
        f"and the ranker nearly doubles it on unseen offers ({cold[SUBMITTED]:.3f} to "
        f"{cold['Ranker + online']:.3f})",
    )
    save_fig(fig, "a_mrr_by_model")


# b. Session fragment leak --------------------------------------------------------------------


def draw_session(ax, x0: float, y: float, n_jobs: int, label: str, target_color: str) -> tuple:
    """Draw one session as a row of job boxes and a dashed target box; return the box centres."""
    ax.text(x0 - 0.3, y, label, ha="right", va="center", color=PALETTE["ink_secondary"])
    for i in range(n_jobs):
        ax.add_patch(Rectangle((x0 + i, y - 0.3), 0.8, 0.6, color=PALETTE["neutral_light"], lw=0))
    target_x = x0 + n_jobs + 0.4
    ax.add_patch(
        Rectangle(
            (target_x, y - 0.3), 0.8, 0.6, fill=False, ec=target_color, lw=1.6, ls=(0, (3, 2))
        )
    )
    ax.text(target_x + 0.4, y, "?", ha="center", va="center", color=target_color, weight="bold")
    return (x0 + 0.4, y), (target_x + 0.4, y)


def fragment_leak() -> None:
    fragments = load("fragments")
    rates = fragments["target_is_first_job_of"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=[1.35, 1])

    sessions = [("session s", 0.0, 5), ("session s+1", 1.2, 4), ("session s+2", 2.4, 6)]
    centres = [
        draw_session(ax, x0, -row, n, label, PALETTE["leak"])
        for row, (label, x0, n) in enumerate(sessions)
    ]
    for (_, target), (first_next, _) in zip(centres[:-1], centres[1:], strict=True):
        ax.add_patch(
            FancyArrowPatch(
                (target[0], target[1] - 0.32),
                (first_next[0], first_next[1] + 0.32),
                connectionstyle="arc3,rad=-0.25",
                arrowstyle="-|>",
                mutation_scale=12,
                color=PALETTE["leak"],
                lw=1.4,
            )
        )
    ax.text(
        0.0,
        -3.0,
        "grey: jobs opened in the session    dashed: target to predict\n"
        "arrow: the target is the first job of the next session",
        color=PALETTE["ink_secondary"],
        fontsize=9,
        va="top",
    )
    ax.set_xlim(-2.2, 10.0)
    ax.set_ylim(-3.8, 0.6)
    ax.axis("off")
    ax.set_title("Schematic: one browsing cut into consecutive sessions")

    labels = ["s−1", "s+1", "s+2"]
    values = [100 * rates[key] for key in ["s-1", "s+1", "s+2"]]
    colors = [PALETTE["neutral"], PALETTE["leak"], PALETTE["neutral"]]
    bx.bar(labels, values, color=colors, width=0.6)
    for x, value in enumerate(values):
        bx.annotate(
            f"{value:.2f} %" if value < 1 else f"{value:.1f} %",
            (x, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
        )
    bx.set_ylim(0, 75)
    bx.grid(axis="x", visible=False)
    bx.set_ylabel("Share of train targets (%)")
    bx.set_title(f"Target = first job of session ... (n={fragments['n_sessions']:,})")
    set_conclusion(
        fig,
        f"{100 * rates['s+1']:.1f} % of targets are the first job of the next session: "
        "sessions are cut pieces of one browsing",
    )
    save_fig(fig, "b_session_fragment_leak")


# c. Random vs grouped validation ------------------------------------------------------------


def random_vs_grouped() -> None:
    comparison = load("split_comparison")
    random, grouped = comparison["random"], comparison["grouped by chain"]
    models = list(random)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    width = 0.36
    for offset, (name, scores, role) in zip(
        [-width / 2, width / 2],
        [("Random split by session", random, "leak"), ("Grouped by chain", grouped, "final")],
        strict=True,
    ):
        ys = [y + offset for y in range(len(models))]
        values = [scores[model] for model in models]
        ax.barh(ys, values, height=width, color=PALETTE[role], label=name)
        for y, value in zip(ys, values, strict=True):
            ax.annotate(
                f"{value:.4f}", (value, y), xytext=(5, 0), textcoords="offset points", va="center"
            )
    ax.set_yticks(range(len(models)), models)
    ax.invert_yaxis()
    ax.set_xlim(0, max(random.values()) * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MRR@10 on 20 % of train sessions")
    ax.legend(loc="center right")
    ax.set_title("Same models, same data, two ways to hold out 20 % of sessions")
    overstated = random["Hybrid"] / grouped["Hybrid"] - 1
    set_conclusion(
        fig,
        f"A random split overstates the hybrid by {100 * overstated:.0f} %: "
        "fragments of one browsing land on both sides",
    )
    save_fig(fig, "c_random_vs_grouped")


# d. Offer id gaps ----------------------------------------------------------------------------


def id_gaps() -> None:
    gaps = load("id_gaps")
    counts = {int(gap): n for gap, n in gaps["signed_gap_counts"].items()}
    share = gaps["share_within"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=[1.3, 1])
    colors = [PALETTE["final"] if gap == -1 else PALETTE["neutral"] for gap in counts]
    ax.bar(list(counts), list(counts.values()), color=colors, width=0.75)
    ax.annotate(
        "previous id in the list",
        (-1, counts[-1]),
        xytext=(8, -4),
        textcoords="offset points",
        va="top",
        color=PALETTE["final"],
    )
    ax.set_xticks(range(-10, 11, 2))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Target id minus last job id")
    ax.set_ylabel("Train sessions")
    ax.set_title(f"Signed gap, |gap| <= 10 (n={gaps['n_sessions']:,} sessions)")

    thresholds = list(share)
    curves = [("target", "Next offer", "final"), ("random", "Random offer", "neutral")]
    for key, name, role in curves:
        values = [100 * share[d][key] for d in thresholds]
        bx.plot(range(len(thresholds)), values, "o-", color=PALETTE[role], label=name, ms=5)
    bx.annotate(
        f"{100 * share['100']['target']:.1f} %",
        (len(thresholds) - 1, 100 * share["100"]["target"]),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
    )
    bx.set_xticks(range(len(thresholds)), thresholds)
    bx.set_xlabel("Within this many ids of the last job")
    bx.set_ylabel("Share of sessions (%)")
    bx.legend(loc="upper left")
    bx.set_title("Cumulative share")
    set_conclusion(
        fig,
        f"Candidates browse the offer list in order: {100 * share['100']['target']:.1f} % of next "
        f"offers sit within 100 ids of the last one,\nagainst {100 * share['100']['random']:.1f} % "
        "for a random offer",
    )
    save_fig(fig, "d_id_gaps")


# e. Validation vs public score ---------------------------------------------------------------


def validation_vs_public() -> None:
    board, validation = load("leaderboard"), load("validation")
    model = board["submissions"]["model"]
    recomputed = final_score(
        validation["mrr_at_10"][SUBMITTED], validation["accuracy"]["last_action"]
    )
    assert round(recomputed, 4) == model["validation"], recomputed
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color=PALETTE["axis"], lw=1.2, ls=(0, (4, 3)))
    ax.annotate(
        "validation = public",
        (0.55, 0.55),
        rotation=45,
        color=PALETTE["muted"],
        ha="right",
        va="bottom",
        rotation_mode="anchor",
        xytext=(0, 4),
        textcoords="offset points",
    )
    ax.axhline(board["benchmark"], color=PALETTE["ink_secondary"], ls=(0, (2, 2)), lw=1)
    ax.annotate(
        f"official benchmark {board['benchmark']:.3f}",
        (0.15, board["benchmark"]),
        xytext=(2, -4),
        textcoords="offset points",
        va="top",
        color=PALETTE["ink_secondary"],
    )
    gaps = []
    points = [("model", "final", "Without the leak"), ("leak", "leak", "With the leak")]
    for key, role, name in points:
        sub = board["submissions"][key]
        gaps.append(abs(sub["public"] - sub["validation"]))
        ax.plot(sub["validation"], sub["public"], "o", ms=11, color=PALETTE[role], mec="white")
        ax.annotate(
            f"{name}\nvalidation {sub['validation']:.4f}\npublic {sub['public']:.4f}",
            (sub["validation"], sub["public"]),
            xytext=(12, -6),
            textcoords="offset points",
            va="top",
            color=PALETTE[role],
        )
    ax.set_xlim(0.15, 0.8)
    ax.set_ylim(0.15, 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Validation score (0.7 x MRR@10 + 0.3 x accuracy)")
    ax.set_ylabel("Public leaderboard score")
    ax.set_title("Grouped validation without the leak, random split with it")
    assert max(gaps) < 0.005, gaps
    set_conclusion(
        fig,
        f"Validation predicted both public scores within 0.005\n(largest gap {max(gaps):.4f})",
    )
    save_fig(fig, "e_validation_vs_public")


# f. Leaderboard ------------------------------------------------------------------------------


def leaderboard() -> None:
    board = load("leaderboard")
    n = board["n_participants"]
    model, leak = board["submissions"]["model"], board["submissions"]["leak"]
    rows = [
        ("Official benchmark", board["benchmark"], "neutral", ""),
        ("My model, without the leak", model["public"], "final", f"{ordinal(model['rank'])}/{n}"),
        ("Same model + fragment leak", leak["public"], "leak", f"{ordinal(leak['rank'])}/{n}"),
    ]
    fig, ax = plt.subplots(figsize=(9, 2.9))
    for y, (_, score, role, rank) in enumerate(rows):
        ax.barh(y, score, color=PALETTE[role], height=0.6)
        text = f"{score:.4f}" + (f"   {rank}" if rank else "")
        ax.annotate(
            text,
            (score, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontweight="bold" if rank else "normal",
        )
    ax.set_yticks(range(len(rows)), [name for name, _, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.85)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Public leaderboard score (0.7 x MRR@10 + 0.3 x accuracy)")
    ax.set_title(f"ENS Challenge Data x HrFlow.ai, {n} participants, {board['date']}")
    set_conclusion(
        fig,
        f"{ordinal(model['rank'])} of {n} without the leak, "
        f"{ordinal(leak['rank'])} of {n} with it, "
        f"{model['public'] / board['benchmark']:.1f}x the benchmark",
    )
    save_fig(fig, "f_leaderboard")


def main() -> None:
    set_style()
    mrr_by_model()
    fragment_leak()
    random_vs_grouped()
    id_gaps()
    validation_vs_public()
    leaderboard()
    plt.close("all")


if __name__ == "__main__":
    main()
