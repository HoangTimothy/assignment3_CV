import pytest

from cvdoc.cli import main


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["doctor"],
        ["data"],
        ["data", "prepare-funsd"],
        ["data", "prepare-smartdoc"],
        ["data", "make-splits"],
        ["data", "inspect"],
        ["infer"],
        ["benchmark"],
        ["benchmark", "geometry"],
        ["benchmark", "ocr"],
        ["benchmark", "all"],
        ["report"],
        ["report", "build"],
    ],
)
def test_cli_help(command):
    with pytest.raises(SystemExit) as exc:
        main(command + ["--help"])
    assert exc.value.code == 0


def test_terms_not_silently_accepted(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: {tmp_path.as_posix()}/data")
    assert main(["data", "prepare-funsd", "--config", str(config)]) == 2
    assert not (tmp_path / "data/raw/funsd.zip").exists()


def test_missing_data_actionable(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: {tmp_path.as_posix()}/data")
    assert main(["benchmark", "geometry", "--config", str(config)]) == 2
