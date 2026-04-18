import pandas as pd

import benchiq


def test_import_exposes_version() -> None:
    assert benchiq.__version__ == "0.1.0a0"


def test_import_exposes_public_validate_entrypoint() -> None:
    bundle = {
        "responses_long": pd.DataFrame(
            {
                "model_id": [pd.NA],
                "benchmark_id": ["b1"],
                "item_id": ["i1"],
                "score": [1],
            },
        ),
        "items": pd.DataFrame({"benchmark_id": ["b1"], "item_id": ["i1"]}),
        "models": pd.DataFrame({"model_id": ["m1"]}),
    }

    report = benchiq.validate(bundle, benchiq.BenchIQConfig())

    assert isinstance(report, benchiq.ValidationReport)
    assert not report.ok
    assert report.errors[0].code == "null_required_key_values"


def test_import_exposes_calibration_and_deployment_helpers() -> None:
    workflows = benchiq.public_workflows()

    assert callable(benchiq.calibrate)
    assert callable(benchiq.predict)
    assert benchiq.deploy is benchiq.predict
    assert workflows["calibrate"].startswith("fit the reusable calibration stack")
    assert workflows["predict"].startswith("score new reduced responses")
