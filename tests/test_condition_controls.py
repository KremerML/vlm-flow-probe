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


if __name__ == "__main__":
    unittest.main()
