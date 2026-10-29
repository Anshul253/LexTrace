import pytest
import yaml
from typer.testing import CliRunner

from clauselens import cli as cli_mod
from clauselens.config import ConfigError, apply_patch, config_from_dict, load_config, save_config

from .conftest import small_config

runner = CliRunner()


def test_baseline_config_loads():
    cfg = load_config("baseline")
    assert cfg.retrieval.method == "bm25" and cfg.chunking.size == 500 and cfg.needs_calibration()


def test_unknown_keys_are_errors():
    with pytest.raises(ConfigError, match="unknown"):
        config_from_dict({"retrieval": {"methd": "bm25"}})
    with pytest.raises(ConfigError):
        config_from_dict({"retrieval": {"method": "magic"}})


def test_patch_and_extends(tmp_path):
    cfg = small_config()
    new = apply_patch(cfg, {"retrieval": {"method": "tfidf"}}, name="x")
    assert new.retrieval.method == "tfidf" and cfg.retrieval.method == "bm25" and new.name == "x"
    assert new.fingerprint() != cfg.fingerprint()
    base = tmp_path / "base.yaml"
    save_config(cfg, base)
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"extends": "base.yaml", "name": "child", "retrieval": {"candidates": 7}}))
    c = load_config(str(child))
    assert c.name == "child" and c.retrieval.candidates == 7 and c.chunking.size == 200


@pytest.fixture
def patched(monkeypatch, corpus, tmp_path):
    import clauselens.workspace as ws

    monkeypatch.setattr(ws, "corpus", lambda: corpus)
    path = tmp_path / "fx.yaml"
    save_config(small_config(), path)
    return str(path)


def test_cli_query_answers(patched):
    r = runner.invoke(cli_mod.app, ["query", "What is the governing law?", "--config", patched, "-k", "2"])
    assert r.exit_code == 0, r.output
    assert "#1" in r.output and "[" in r.output


def test_cli_query_json(patched):
    import json

    r = runner.invoke(cli_mod.app, ["query", "royalty", "--config", patched, "--json"])
    assert r.exit_code == 0
    assert json.loads(r.output)["passages"][0]["doc_id"].startswith("fx/")


def test_cli_query_invalid_and_abstain(patched):
    assert runner.invoke(cli_mod.app, ["query", "   ", "--config", patched]).exit_code == 2
    assert runner.invoke(cli_mod.app, ["query", "zzzz qqqq", "--config", patched]).exit_code == 10


def test_cli_test_split_requires_confirmation(patched):
    r = runner.invoke(cli_mod.app, ["evaluate", "--split", "test", "--config", patched])
    assert r.exit_code == 2 and "confirm-final" in r.output


def test_cli_unknown_config():
    assert runner.invoke(cli_mod.app, ["query", "x", "--config", "does-not-exist"]).exit_code == 3


def test_cli_analyze_refuses_test(patched):
    assert runner.invoke(cli_mod.app, ["analyze", "--split", "test", "--config", patched]).exit_code == 2
