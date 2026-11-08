"""Chart/table specs, series building and evidence picks (``_notes_blocks``)."""

from __future__ import annotations

import pytest

from mlsweep import _notes_blocks as nb


def _spec(raw: str, kind: str = "chart") -> dict:
    return nb.normalize_spec(raw, kind)[0]


def _run(run_id: str, **combo: object) -> nb.RunRef:
    return nb.RunRef("exp", run_id, "done", combo)


def _rows(**per_run: list[tuple[int, float]]) -> dict[tuple[str, str], list[dict]]:
    return {("exp", run): [{"step": step, "loss": loss, "acc": 1.0 - loss}
                           for step, loss in points]
            for run, points in per_run.items()}


def test_parse_spec_line_format_and_json() -> None:
    spec = nb.parse_spec("experiments: a, b\nmetric: val_loss\nsmooth: 0.9\nlog_y: true")
    assert spec["experiments"] == "a, b"
    assert spec["metric"] == "val_loss" and spec["smooth"] == 0.9 and spec["log_y"] is True
    assert nb.parse_spec('{"metric": "loss", "experiments": ["a"]}')["metric"] == "loss"


def test_normalize_defaults_and_caption_id() -> None:
    spec, warnings = nb.normalize_spec(
        "experiments: [a]\nmetric: loss\ncaption: Seed spread {#fig-seeds}", "chart")
    assert warnings == []
    assert spec["band"] == "none" and spec["points"] == 400
    assert spec["caption"] == "Seed spread" and spec["id"] == "fig-seeds"


def test_normalize_band_defaults_with_split() -> None:
    spec, _ = nb.normalize_spec("experiments: [a]\nmetric: loss\nsplit_by: lr", "chart")
    assert spec["band"] == "minmax"


@pytest.mark.parametrize("raw,kind,fragment", [
    ("", "chart", "empty"),
    ("metric: loss", "chart", "experiments"),
    ("experiments: a", "chart", "metric"),
    ("experiments: a\nmetric: loss\nband: wide", "chart", "band"),
    ("experiments: a\nmetric: loss\nsmooth: 3", "chart", "smooth"),
    ("experiments: a\nmetric: loss\npoints: 1", "chart", "points"),
    ("experiments: a\nagg: average", "table", "agg"),
    ("experiments: a\nmetric: loss\nbogus: 1", "chart", "unknown key"),
])
def test_normalize_rejects_or_warns(raw: str, kind: str, fragment: str) -> None:
    if fragment == "unknown key":
        _, warnings = nb.normalize_spec(raw, kind)
        assert any(fragment in w for w in warnings)
    else:
        with pytest.raises(nb.SpecError, match=fragment):
            nb.normalize_spec(raw, kind)


def test_chart_one_series_per_run() -> None:
    spec = _spec("experiments: exp\nmetric: loss")
    out = nb.assemble_chart(spec, [_run("r0"), _run("r1")],
                            _rows(r0=[(1, 1.0), (2, 0.5)], r1=[(1, 2.0), (2, 1.0)]))
    assert [s["name"] for s in out["series"]] == ["r0", "r1"]
    assert out["title"] == "loss · exp"
    assert out["series"][0]["y"] == [1.0, 0.5]
    assert out["series"][0]["final"] == 0.5


def test_chart_groups_and_band() -> None:
    spec = _spec("experiments: exp\nmetric: loss\nsplit_by: lr\nband: minmax")
    runs = [_run("r0", lr=0.001), _run("r1", lr=0.001), _run("r2", lr=0.01)]
    out = nb.assemble_chart(spec, runs, _rows(
        r0=[(1, 1.0), (2, 0.8)],
        r1=[(1, 1.2), (2, 1.0)],
        r2=[(1, 2.0), (2, 1.5)],
    ))
    assert out["title"] == "loss by lr · exp"
    groups = {s["name"]: s for s in out["series"]}
    assert set(groups) == {"lr=0.001", "lr=0.01"}
    assert groups["lr=0.001"]["runs"] == 2
    assert groups["lr=0.001"]["y"] == [1.1, 0.9]
    assert groups["lr=0.001"]["lo"] == [1.0, 0.8]
    assert groups["lr=0.001"]["hi"] == [1.2, 1.0]
    assert "lo" not in groups["lr=0.01"]  # one run has no spread


def test_chart_smoothing_and_symbols() -> None:
    spec = _spec("experiments: exp\nmetric: loss\nsmooth: 0.5\nx_max: 2")
    out = nb.assemble_chart(spec, [_run("r0")], _rows(
        r0=[(1, 1.0), (2, 0.0), (3, 0.0)]))
    assert out["series"][0]["x"] == [1.0, 2.0]
    assert out["series"][0]["y"] == pytest.approx([1.0, 0.5])


def test_chart_downsample_keeps_ends() -> None:
    spec = _spec("experiments: exp\nmetric: loss\npoints: 5")
    points = [(i, float(i)) for i in range(1, 101)]
    out = nb.assemble_chart(spec, [_run("r0")], _rows(r0=points))
    s = out["series"][0]
    assert len(s["x"]) == 5 and s["x"][0] == 1.0 and s["x"][-1] == 100.0


def test_chart_reports_missing_metric() -> None:
    spec = _spec("experiments: exp\nmetric: nope")
    out = nb.assemble_chart(spec, [_run("r0")], _rows(r0=[(1, 1.0)]))
    assert out["series"] == []
    assert any("nope" in w for w in out["warnings"])


def test_table_rows_columns_and_sort() -> None:
    spec = _spec("experiments: exp\nmetrics: [loss]\nsort: loss\norder: desc", "table")
    runs = [_run("r0", lr=0.001), _run("r1", lr=0.01)]
    out = nb.assemble_table(spec, runs, _rows(r0=[(1, 2.0), (2, 1.0)],
                                             r1=[(1, 0.5), (2, 0.25)]))
    assert out["title"] == "loss · exp · 2 runs"
    assert [c["key"] for c in out["columns"]] == ["run", "lr", "loss"]
    assert [r["run"] for r in out["rows"]] == ["r0", "r1"]
    assert out["rows"][0]["metrics"]["loss"]["final"] == 1.0
    assert out["rows"][0]["metrics"]["loss"]["min"] == 1.0
    assert out["rows"][0]["metrics"]["loss"]["max"] == 2.0


def test_table_infers_metrics() -> None:
    spec = _spec("experiments: exp", "table")
    out = nb.assemble_table(spec, [_run("r0")], _rows(r0=[(1, 1.0)]))
    assert {c["key"] for c in out["columns"] if c["kind"] == "metric"} == {"loss", "acc"}


def test_evidence_ref_parsing() -> None:
    assert nb.parse_evidence_ref("mlsweep:exp/run/loss@500") == ("exp", "run", "loss", "500")
    assert nb.parse_evidence_ref("exp/run/loss")[3] == "final"
    assert nb.parse_evidence_ref("exp/run/train/loss@5") == ("exp", "run", "train/loss", "5")
    for bad in ("exp/run", "exp/run/loss@", "exp//loss", ""):
        with pytest.raises(nb.SpecError):
            nb.parse_evidence_ref(bad)


def test_pick_evidence_exact_nearest_and_aggregates() -> None:
    rows = [{"step": 1, "loss": 2.0}, {"step": 5, "loss": 1.0}, {"step": 9, "loss": 0.5}]
    assert nb.pick_evidence(rows, "loss", "5")["value"] == 1.0
    near = nb.pick_evidence(rows, "loss", "6")
    assert near["value"] == 1.0 and near["step"] == 5 and near["exact"] is False
    assert nb.pick_evidence(rows, "loss", "final")["value"] == 0.5
    assert nb.pick_evidence(rows, "loss", "min")["value"] == 0.5
    assert nb.pick_evidence(rows, "loss", "max")["value"] == 2.0
    assert nb.pick_evidence(rows, "loss", "mean")["value"] == pytest.approx(3.5 / 3)
    assert nb.pick_evidence(rows, "loss", "50%")["value"] == 1.0
    assert nb.pick_evidence(rows, "nope", "1")["ok"] is False


def test_lttb_keeps_endpoints() -> None:
    xs = [float(i) for i in range(200)]
    ys = [float(i % 17) for i in range(200)]
    idx = nb.lttb_indices(xs, ys, 20)
    assert len(idx) == 20 and idx[0] == 0 and idx[-1] == 199
    assert nb.lttb_indices(xs, ys, 500) == list(range(200))


def test_chart_group_order_is_natural() -> None:
    spec = _spec("experiments: exp\nmetric: loss\nsplit_by: depth")
    runs = [_run("r25", depth="25m"), _run("r5", depth="5m"), _run("r10", depth="10m")]
    out = nb.assemble_chart(spec, runs, _rows(r25=[(1, 3.0)], r5=[(1, 5.0)], r10=[(1, 4.0)]))
    assert [s["name"] for s in out["series"]] == ["depth=5m", "depth=10m", "depth=25m"]


def test_table_label_strips_a_shared_experiment_prefix() -> None:
    spec = _spec("experiments: exp\nmetrics: [loss]", "table")
    ref = nb.RunRef("exp", "exp_run_a", "done", {"lr": 0.001})
    out = nb.assemble_table(spec, [ref], {("exp", "exp_run_a"): [{"step": 1, "loss": 1.0}]})
    assert out["rows"][0]["run"] == "exp_run_a"
    assert out["rows"][0]["label"] == "run_a"


def test_titles_do_not_repeat_the_legend_names() -> None:
    grouped = _spec("experiments: exp\nmetric: loss\nsplit_by: _experiment")
    out = nb.assemble_chart(grouped, [_run("r0"), _run("r1")],
                            _rows(r0=[(1, 1.0)], r1=[(1, 2.0)]))
    assert out["title"] == "loss by experiment"
    assert [s["name"] for s in out["series"]] == ["exp"]

    named = _spec("runs: [exp/run_a, exp/run_b]\nmetric: loss")
    out = nb.assemble_chart(named, [_run("run_a"), _run("run_b")],
                            _rows(run_a=[(1, 1.0)], run_b=[(1, 2.0)]))
    assert out["title"] == "loss · exp"
    assert [s["name"] for s in out["series"]] == ["run_a", "run_b"]


def test_title_omits_experiment_names_the_legend_already_shows() -> None:
    spec = _spec("experiments: [a, b]\nmetric: loss")
    runs = [nb.RunRef("a", "r0", "done", {}), nb.RunRef("b", "r1", "done", {})]
    out = nb.assemble_chart(spec, runs, {
        ("a", "r0"): [{"step": 1, "loss": 1.0}],
        ("b", "r1"): [{"step": 1, "loss": 2.0}],
    })
    assert out["title"] == "loss"
    assert [s["name"] for s in out["series"]] == ["a/r0", "b/r1"]


def test_table_drops_a_run_column_that_repeats_a_parameter() -> None:
    spec = _spec("experiments: exp\nmetrics: [loss]\ncolumns: [arch]", "table")
    runs = [nb.RunRef("exp", "exp_gru", "done", {"arch": "gru"}),
            nb.RunRef("exp", "exp_ngram", "done", {"arch": "ngram"})]
    rows = {("exp", "exp_gru"): [{"step": 1, "loss": 1.0}],
            ("exp", "exp_ngram"): [{"step": 1, "loss": 2.0}]}
    out = nb.assemble_table(spec, runs, rows)
    assert out["run_link_column"] == "arch"
    assert [c["key"] for c in out["columns"]] == ["arch", "loss"]
    assert [r["run"] for r in out["rows"]] == ["exp_gru", "exp_ngram"]
    assert out["rows"][0]["label"] == "gru"


def test_experiment_and_run_names_are_shortened() -> None:
    exp = "expname_20260101_0000_abcd"
    ref = nb.RunRef(exp, "expname_runA", "done", {}, experiment_name="expname")
    points = {(exp, "expname_runA"): [{"step": 1, "loss": 1.0}]}
    chart = nb.assemble_chart(_spec(f"experiments: [{exp}]\nmetric: loss"), [ref], points)
    assert chart["title"] == "loss · expname"
    assert chart["series"][0]["name"] == "runA"
    assert chart["series"][0]["full_name"] == f"{exp}/expname_runA"

    table = nb.assemble_table(_spec(f"experiments: [{exp}]\nmetrics: [loss]", "table"),
                              [ref], points)
    assert table["title"] == "loss · expname · 1 run"
    assert table["rows"][0]["label"] == "runA"
    assert table["rows"][0]["experiment_name"] == "expname"
    assert table["rows"][0]["run"] == "expname_runA"


def test_two_experiments_with_one_name_get_a_suffix() -> None:
    a = nb.RunRef("diag_x_20260101_0000_aaaa", "diag_x_a", "done", {}, experiment_name="diag_x")
    b = nb.RunRef("diag_x_20260101_0001_bbbb", "diag_x_b", "done", {}, experiment_name="diag_x")
    assert nb._experiment_names([a, b]) == {
        "diag_x_20260101_0000_aaaa": "diag_x#aaaa",
        "diag_x_20260101_0001_bbbb": "diag_x#bbbb",
    }
