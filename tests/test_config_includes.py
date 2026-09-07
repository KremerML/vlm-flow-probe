"""Config composition: include fragments, frozen guard, CLI overrides."""

import os
import tempfile
import unittest

from vlmflowprobe.core.config import DEFAULT_CONFIG, apply_overrides, load_config


def write(path, text):
    with open(path, "w") as handle:
        handle.write(text)


class TestIncludes(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def path(self, name):
        return os.path.join(self.root, name)

    def test_merge_order_defaults_fragments_own_overrides(self):
        write(self.path("frag_a.yaml"), "model:\n  target_layer: 5\nsae:\n  n_features: 111\n")
        write(self.path("frag_b.yaml"), "model:\n  target_layer: 7\n")
        write(
            self.path("exp.yaml"),
            "include: [frag_a.yaml, frag_b.yaml]\nsae:\n  l1_coeff: 0.5\n",
        )
        cfg = load_config(self.path("exp.yaml"), overrides=["sae.n_features=222"])
        # fragment b overrode fragment a
        self.assertEqual(cfg.get("model")["target_layer"], 7)
        # own keys merge over fragments; untouched fragment keys survive... unless overridden
        self.assertEqual(cfg.get("sae")["l1_coeff"], 0.5)
        self.assertEqual(cfg.get("sae")["n_features"], 222)
        # defaults still present underneath
        self.assertEqual(cfg.get("model")["adapter"], DEFAULT_CONFIG["model"]["adapter"])

    def test_fragment_may_not_include(self):
        write(self.path("frag.yaml"), "include: [other.yaml]\n")
        write(self.path("exp.yaml"), "include: [frag.yaml]\n")
        with self.assertRaises(ValueError):
            load_config(self.path("exp.yaml"))

    def test_missing_fragment_raises(self):
        write(self.path("exp.yaml"), "include: [nope.yaml]\n")
        with self.assertRaises(FileNotFoundError):
            load_config(self.path("exp.yaml"))

    def test_frozen_configs_must_be_self_contained(self):
        write(self.path("frozen.yaml"), "frozen: true\ninclude: [frag.yaml]\n")
        with self.assertRaises(ValueError):
            load_config(self.path("frozen.yaml"))
        # frozen without include is fine, and the flag is retained: it is what
        # exempts a reproduction record from the model-identity rule, which its
        # archive-era experiment name cannot satisfy.
        write(self.path("frozen_ok.yaml"), "frozen: true\nmodel:\n  target_layer: 3\n")
        cfg = load_config(self.path("frozen_ok.yaml"))
        self.assertEqual(cfg.get("model")["target_layer"], 3)
        self.assertIs(cfg.get("frozen"), True)

    def test_a_live_config_is_marked_not_frozen(self):
        write(self.path("live.yaml"), "model:\n  target_layer: 3\n")
        self.assertIs(load_config(self.path("live.yaml")).get("frozen"), False)


class TestOverrides(unittest.TestCase):
    def test_yaml_typed_values(self):
        cfg = {"a": {"b": 1}}
        apply_overrides(cfg, ["a.b=2", "a.c=true", "a.d=[1,2]", "a.e=null"])
        self.assertEqual(cfg["a"], {"b": 2, "c": True, "d": [1, 2], "e": None})

    def test_malformed_override_raises(self):
        with self.assertRaises(ValueError):
            apply_overrides({}, ["no_equals_sign"])


class TestHonestDefaults(unittest.TestCase):
    def test_random_control_defaults_are_strict_matched(self):
        # The archive's permissive defaults silently produced uniform controls;
        # the clean repo inverts them.
        rc = DEFAULT_CONFIG["random_control"]
        self.assertEqual(rc["matched_metric"], "activation_mean")
        self.assertTrue(rc["strict_matching"])

    def test_model_section_has_adapter_and_no_derivables(self):
        model = DEFAULT_CONFIG["model"]
        self.assertIn("adapter", model)
        for dead_key in ("d_model", "conv_mode", "model_base"):
            self.assertNotIn(dead_key, model)


if __name__ == "__main__":
    unittest.main()
