from sureenough.analysis.figures import write_figures


def _summary(naive, ours, stop):
    return {
        "naive_pop_risk": naive,
        "pop_risk": ours,
        "mean_stop": stop,
        "mean_set_size": 4.0,
        "fixed_points": [(30.0, 1.0), (12.0, 4.0), (5.0, 16.0), (4.0, 20.0)],
        "per_image": {"stop": [1.0, 5.0, 20.0, 7.0], "size": [1.0, 3.0, 200.0, 2.0], "loss": [0.0, 0.1, 0.0, 0.2]},
    }


def test_write_figures_creates_three_pngs(tmp_path):
    results = {"per_subject": {"1": _summary(0.31, 0.12, 9.0), "2": _summary(0.27, 0.15, 11.0)}}
    paths = write_figures(results, alpha=0.2, out_dir=tmp_path)
    assert sorted(p.name for p in paths) == ["fixed_k_hull.png", "promised_vs_delivered.png", "stop_histogram.png"]
    assert all(p.stat().st_size > 1000 for p in paths)
