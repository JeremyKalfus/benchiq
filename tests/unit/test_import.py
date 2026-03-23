import benchiq


def test_import_exposes_version() -> None:
    assert benchiq.__version__ == "0.1.0a0"
