from pathlib import Path

from vns import cli


def test_main_validates_config_successfully(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("database:\n  path: custom.vnsdb\n", encoding="utf-8")

    exit_code = cli.main(["--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Configuration validation completed successfully." in captured.out
    assert "custom.vnsdb" in captured.out


def test_main_reports_invalid_config_yaml(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("database: [oops\n", encoding="utf-8")

    exit_code = cli.main(["--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Failed to parse configuration YAML" in captured.err


def test_main_reports_missing_database_file(tmp_path: Path, capsys) -> None:
    missing_database = tmp_path / "missing.vnsdb"

    exit_code = cli.main(["database", "inspect", str(missing_database)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Database file not found" in captured.err
