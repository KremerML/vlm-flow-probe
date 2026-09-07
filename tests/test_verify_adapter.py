"""CPU checks for vfp-verify-adapter.

The command's point is to fail loudly on a real model, so what is testable
without one is the wiring: probe construction from a config, group selection,
the table, and the exit code. The stub stands in for the adapter; it records
knockout installs rather than registering hooks, so the knockout and end-to-end
groups are *expected* to fail against it -- which is exactly the signal the
command exists to give for an adapter that does not really install anything.
"""

import json

import pytest
import yaml

from tests.probes import stub_probe
from vlmflowprobe.cli import verify_adapter


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "stub.yaml"
    path.write_text(yaml.safe_dump({
        "model": {"adapter": "stub", "name": "stub/model"},
        "dataset": {"format": "clevr_lite", "data_dir": str(tmp_path / "no-such-data")},
    }))
    return str(path)


@pytest.fixture
def patched_adapter(monkeypatch):
    from vlmflowprobe.utils import runtime

    adapter = stub_probe().adapter
    monkeypatch.setattr(runtime, "load_adapter", lambda config: adapter)
    return adapter


def run(argv, monkeypatch):
    monkeypatch.setattr("sys.argv", ["vfp-verify-adapter", *argv])
    return verify_adapter.main()


def test_structural_groups_pass_against_the_stub(config_path, patched_adapter, monkeypatch, capsys):
    code = run(["--config", config_path, "--question", "what color is it",
                "--groups", "modules,geometry,execution"], monkeypatch)
    out = capsys.readouterr().out
    assert code == 0, out
    assert "FAIL" not in out
    for name in ("unknown_site_raises", "image_span_is_post_expansion",
                 "n_image_tokens_never_silently_zero", "forward_appends_extra_input_ids"):
        assert name in out


def test_missing_dataset_falls_back_to_the_question_flag(config_path, patched_adapter,
                                                         monkeypatch, capsys):
    # No --question, so the probe goes looking for the configured dataset first.
    run(["--config", config_path, "--groups", "modules"], monkeypatch)
    captured = capsys.readouterr()
    assert "no dataset sample" in captured.err


def test_an_adapter_that_installs_no_hooks_fails(config_path, patched_adapter,
                                                 monkeypatch, capsys):
    code = run(["--config", config_path, "--question", "q", "--groups", "knockout"], monkeypatch)
    out = capsys.readouterr().out
    assert code == 1
    assert "registered no hooks" in out


def test_json_report_is_written(config_path, patched_adapter, monkeypatch, tmp_path, capsys):
    report = tmp_path / "nested" / "report.json"
    run(["--config", config_path, "--question", "q", "--groups", "modules",
         "--json", str(report)], monkeypatch)
    capsys.readouterr()
    payload = json.loads(report.read_text())
    assert payload["passed"] is True
    assert payload["adapter"] == "stub"
    assert {c["name"] for c in payload["checks"]} == {
        "residual_site_is_the_decoder_layer", "unknown_site_raises", "out_of_range_layer_raises"
    }
    assert "git_sha" in payload["provenance"]
