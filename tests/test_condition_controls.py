"""Condition.control_kind replaces the old runner's string-prefix dispatch.

The archive's ``controls_for`` decided joint-control counts by matching
``condition_id`` prefixes: ``joint_``/``A0_`` got the full ``n_random_sets``,
``nested_L``/``nonnested_L``/``budget_spread`` got 1, everything else 0 — and
non-SAE kinds always 0. These tests pin that policy onto the explicit field so
the Phase-4 condition builders can be checked against it.
"""

import unittest

from vlmflowprobe.ablation.multilayer_experiments import (
    Condition,
    KIND_COMBINED,
    KIND_KNOCKOUT,
    KIND_NONE,
    KIND_PASSTHROUGH,
    KIND_SAE,
)

N_FULL = 15


class TestConditionControls(unittest.TestCase):
    def test_full_control_families(self):
        for cid in ("A0_regression_L11", "joint_L10-14"):
            c = Condition(cid, KIND_SAE, features={11: [1]}, control_kind="full")
            self.assertEqual(c.n_controls(N_FULL), N_FULL, cid)

    def test_single_control_families(self):
        for cid in ("nested_L14", "nonnested_L10,12,14", "budget_spread40x5"):
            c = Condition(cid, KIND_SAE, features={14: [1]}, control_kind="single")
            self.assertEqual(c.n_controls(N_FULL), 1, cid)

    def test_no_control_families(self):
        c = Condition("loo_drop13", KIND_SAE, features={10: [1]}, control_kind="none")
        self.assertEqual(c.n_controls(N_FULL), 0)

    def test_non_sae_kinds_never_get_controls(self):
        for kind in (KIND_KNOCKOUT, KIND_PASSTHROUGH, KIND_NONE):
            c = Condition("span_knockout_L10-14", kind, control_kind="full")
            self.assertEqual(c.n_controls(N_FULL), 0, kind)

    def test_combined_kind_respects_policy(self):
        c = Condition("downstream_combined_L11", KIND_COMBINED,
                      features={11: [1]}, control_kind="single")
        self.assertEqual(c.n_controls(N_FULL), 1)

    def test_default_is_none(self):
        self.assertEqual(Condition("x", KIND_SAE, features={0: [1]}).n_controls(N_FULL), 0)


class TestRunnerControlKindAssignment(unittest.TestCase):
    """assign_control_kinds must reproduce the archive's controls_for policy."""

    def test_family_mapping(self):
        from vlmflowprobe.cli.run_multilayer import assign_control_kinds

        cases = {
            "A0_regression_L11": "full",
            "joint_L10-14": "full",
            "nested_L13,14": "single",
            "nonnested_L10,12,14": "single",
            "budget_spread40x5": "single",
            "loo_drop13": "none",
            "budget_concentrated_L11_k800": "none",
            "downstream_combined_L11": "none",
            "sensitivity_joint_L10,11,12,14": "none",
        }
        conditions = [Condition(cid, KIND_SAE, features={11: [1]}) for cid in cases]
        stamped = assign_control_kinds(conditions)
        for condition in stamped:
            self.assertEqual(condition.control_kind, cases[condition.condition_id],
                             condition.condition_id)


class TestConditionIdGrammar(unittest.TestCase):
    def test_known_ids_classify(self):
        from vlmflowprobe.contracts import classify_condition_id

        for cid, family in [
            ("A0_regression_L11", "a0_regression"),
            ("gate_none", "gate_none"),
            ("gate_passthrough_L10-14", "gate_passthrough"),
            ("gate_passthrough_delta_L10-14", "gate_passthrough"),
            ("joint_L10-14", "joint"),
            ("span_knockout_L11-14", "span_knockout"),
            ("knockout_L14", "single_knockout"),
            ("nested_L12,13,14", "nested"),
            ("nested_knockout_L14", "nested"),
            ("nonnested_L10,12,14", "non_nested"),
            ("loo_drop13", "leave_one_out"),
            ("budget_spread40x5", "budget_spread"),
            ("budget_concentrated_L11_k800", "budget_concentrated"),
            ("downstream_ablate_L14", "downstream"),
            ("downstream_knockout_L12-31", "downstream"),
            ("sensitivity_joint_L10,11,12,14", "sensitivity"),
            ("sensitivity_passthrough_tail_L15", "sensitivity"),
        ]:
            self.assertEqual(classify_condition_id(cid), family, cid)

    def test_unknown_id_raises(self):
        from vlmflowprobe.contracts import classify_condition_id

        with self.assertRaises(ValueError):
            classify_condition_id("mystery_condition")


if __name__ == "__main__":
    unittest.main()
