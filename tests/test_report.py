import json

from llm.evaluation.report import build_report, speed_section


def test_report_marks_missing_results_and_computes_speedup(tmp_path):
    rows = [
        {
            "variant": "baseline",
            "steps_per_second": 0.3,
            "useful_tokens_per_second": 1000.0,
            "useful_fraction": 0.25,
            "peak_device_gb": 3.0,
            "swap_gb": 1.0,
        },
        {
            "variant": "fast",
            "steps_per_second": 0.36,
            "useful_tokens_per_second": 1200.0,
            "useful_fraction": 0.25,
            "peak_device_gb": 2.8,
            "swap_gb": 1.0,
        },
    ]
    assert "+20.0%" in speed_section(rows)
    (tmp_path / "profile_training.json").write_text(json.dumps(rows))
    report = build_report(tmp_path, "T")
    assert "+20.0%" in report and report.count("_not measured_") == 4
