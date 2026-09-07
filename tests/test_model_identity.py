"""Experiment names must carry the model, so two models cannot collide.

Nothing in an experiment directory records which model produced it -- not the
SAE checkpoint, not the feature catalog, not the ablation results. Two models
run under the same ``experiment.name`` would interleave artifacts silently, and
the first sign of it would be a catalog that indexes another model's features.
The name is what prevents that, so it is enforced rather than documented.
"""

import unittest
from types import SimpleNamespace

from vlmflowprobe.core.config import load_config
from vlmflowprobe.utils.config_utils import model_tag, validate_model_identity


class TestModelTag(unittest.TestCase):
    def test_explicit_tag_wins(self):
        self.assertEqual(model_tag({"model": {"tag": "gemma3", "adapter": "hf-gemma3"}}), "gemma3")

    def test_derived_from_adapter_and_checkpoint_when_unset(self):
        tag = model_tag({"model": {"adapter": "hf-llava", "name": "llava-hf/llava-1.5-7b-hf"}})
        self.assertEqual(tag, "hf-llava-llava-1-5-7b-hf")

    def test_derivation_survives_a_config_with_nothing_set(self):
        self.assertEqual(model_tag({}), "model")


class TestValidation(unittest.TestCase):
    def config(self, name, **model):
        return {
            "model": {"tag": "llava15", "adapter": "hf-llava", **model},
            "experiment": {"name": name},
        }

    def test_name_carrying_the_tag_passes(self):
        self.assertEqual(
            validate_model_identity(self.config("llava15_sae_layer11")), "llava15"
        )

    def test_name_without_the_tag_is_refused_with_a_suggestion(self):
        with self.assertRaises(ValueError) as caught:
            validate_model_identity(self.config("sae_layer11"))
        message = str(caught.exception)
        self.assertIn("llava15", message)
        self.assertIn("llava15_sae_layer11", message)

    def test_frozen_reproduction_records_are_exempt(self):
        config = self.config("sae_clevr_lite_layer11_attn_out_question")
        config["frozen"] = True
        validate_model_identity(config)

    def test_opting_out_is_possible_but_explicit(self):
        config = self.config("sae_layer11")
        config["experiment"]["require_model_tag"] = False
        validate_model_identity(config)


class TestShippedConfigs(unittest.TestCase):
    """Every config in the repo has to satisfy the rule it enforces."""

    def test_experiment_configs_carry_the_model(self):
        import glob

        import os

        paths = sorted(glob.glob("configs/experiments/**/*.yaml", recursive=True))
        self.assertTrue(paths, "no experiment configs found")
        seen_tags = set()
        for path in paths:
            with self.subTest(path=path):
                config = load_config(path)
                # The directory under configs/experiments/ is the tag: that is the
                # layout docs/adding-a-model.md prescribes.
                expected = os.path.basename(os.path.dirname(path))
                self.assertEqual(validate_model_identity(config), expected)
                self.assertIn(expected, config.get("experiment")["name"])
                seen_tags.add(expected)
        self.assertIn("llava15", seen_tags)
        self.assertIn("gemma3_4b", seen_tags)

    def test_frozen_configs_still_reproduce_the_archived_names(self):
        import glob

        for path in sorted(glob.glob("configs/frozen/*.yaml")):
            with self.subTest(path=path):
                config = load_config(path)
                self.assertIs(config.get("frozen"), True)
                validate_model_identity(config)  # exempt, must not raise


class TestSetupExperiment(unittest.TestCase):
    def test_experiment_name_override_is_checked_too(self):
        from vlmflowprobe.utils.runtime import setup_experiment

        config = load_config("configs/experiments/llava15/sae_layer11_attn_out_question.yaml")
        args = SimpleNamespace(experiment_name="scratch_run", experiment_dir=None)
        with self.assertRaises(ValueError):
            setup_experiment(args, config)


if __name__ == "__main__":
    unittest.main()
