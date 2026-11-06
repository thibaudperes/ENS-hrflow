import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from hrflow_recsys.viz import (  # noqa: E402
    MIN_DELTA_E,
    PALETTE,
    RESULT_ROLES,
    palette_separation,
    save_fig,
    set_conclusion,
    set_style,
)


def test_result_colours_stay_distinct_for_colour_blind_readers():
    separation = palette_separation([PALETTE[role] for role in RESULT_ROLES])
    assert set(separation) == {"normal", "protanopia", "deuteranopia", "tritanopia"}
    assert min(separation.values()) >= MIN_DELTA_E, separation


def test_identical_colours_have_no_separation():
    assert palette_separation(["#12b886", "#12b886"])["normal"] < 1e-6


def test_save_fig_writes_png(tmp_path):
    set_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    set_conclusion(fig, "A line goes up")
    path = save_fig(fig, "line", fig_dir=tmp_path)
    plt.close(fig)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
