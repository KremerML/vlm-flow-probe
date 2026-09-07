"""The isolation phase: ablation with Image->Question severed at every layer."""

from types import SimpleNamespace

from vlmflowprobe.cli.run_multilayer import build_conditions
from vlmflowprobe.contracts import classify_condition_id
from vlmflowprobe.core.config import Config


def _experiment(n_layers=6):
    exp = SimpleNamespace(adapter=SimpleNamespace(n_layers=n_layers))
    exp.features_for = lambda layers, k: {int(l): list(range(int(k))) for l in layers}
    return exp


def test_isolation_conditions_block_every_layer():
    config = Config()
    config.data["multilayer"]["layers"] = [2, 3]
    conditions = build_conditions(_experiment(), config, ["isolation"])
    ids = [c.condition_id for c in conditions]
    assert ids == ["full_knockout_L0-5", "isolated_ablate_L2", "isolated_ablate_L3", "isolated_joint_L2-3"]
    for c in conditions:
        assert c.knockout_layers == tuple(range(6))
        assert classify_condition_id(c.condition_id) == "isolation"
    assert conditions[0].features == {} and conditions[0].kind == "knockout"
    assert conditions[-1].kind == "combined" and sorted(conditions[-1].features) == [2, 3]
