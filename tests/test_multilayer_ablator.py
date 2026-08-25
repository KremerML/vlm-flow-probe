import unittest

import torch
from torch import nn

from vlmflowprobe.ablation.feature_ablator import FeatureAblator
from vlmflowprobe.ablation.multilayer_ablator import MultiLayerFeatureAblator
from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder
from tests.stubs import DatasetStub, adapt


class MultiLayerModel(nn.Module):
    def __init__(self, n_layers=5, d_model=4, vocab=16):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.vocab = vocab

    def forward(self, input_ids=None, images=None, image_sizes=None, use_cache=False):
        seq_len = input_ids.shape[1]
        return torch.zeros(1, seq_len, self.vocab)

    def generate(self, **kwargs):
        scores = torch.zeros(1, self.vocab)
        scores[0, 1] = 1.0
        return {"sequences": torch.tensor([[1]], dtype=torch.long), "scores": [scores]}


def make_saes(layers, d_model=4, n_features=8, seed=0):
    torch.manual_seed(seed)
    return {layer: SparseAutoencoder(d_model=d_model, n_features=n_features) for layer in layers}


def hook_counts(model):
    return [len(layer._forward_hooks) for layer in model.layers]


class TestMultiLayerHookRegistration(unittest.TestCase):
    def test_registers_one_hook_per_named_layer_and_removes_all(self):
        model = MultiLayerModel(n_layers=5)
        ablator = MultiLayerFeatureAblator(
            adapt(model), make_saes([0, 1, 2, 3, 4]), activation_site="residual"
        )

        handles = ablator._register_sae_hooks(
            {1: [0, 1], 3: [2], 4: []},
            positions=[0, 1],
            mode="replace",
            delta_scale=1.0,
            operation="zero",
            operation_scale=1.0,
            diagnostics_buffer=[],
        )
        self.assertEqual(len(handles), 3)
        self.assertEqual(hook_counts(model), [0, 1, 0, 1, 1])

        for handle in handles:
            handle.remove()
        self.assertEqual(hook_counts(model), [0, 0, 0, 0, 0])

    def test_absent_layer_is_not_hooked_but_empty_layer_is(self):
        # In replace mode a pass-through hook still swaps in the SAE reconstruction, so
        # "not in the mapping" and "mapped to []" are different interventions.
        model = MultiLayerModel(n_layers=3)
        ablator = MultiLayerFeatureAblator(
            adapt(model), make_saes([0, 1, 2]), activation_site="residual"
        )

        handles = ablator._register_sae_hooks(
            {0: []},
            positions=None,
            mode="replace",
            delta_scale=1.0,
            operation="zero",
            operation_scale=1.0,
            diagnostics_buffer=[],
        )
        self.assertEqual(hook_counts(model), [1, 0, 0])
        for handle in handles:
            handle.remove()

    def test_unknown_layer_raises(self):
        model = MultiLayerModel(n_layers=3)
        ablator = MultiLayerFeatureAblator(
            adapt(model), make_saes([0, 1]), activation_site="residual"
        )
        with self.assertRaises(KeyError):
            ablator._register_sae_hooks(
                {2: [0]},
                positions=None,
                mode="replace",
                delta_scale=1.0,
                operation="zero",
                operation_scale=1.0,
                diagnostics_buffer=[],
            )

    def test_flat_feature_list_is_rejected(self):
        model = MultiLayerModel(n_layers=3)
        ablator = MultiLayerFeatureAblator(
            adapt(model), make_saes([0, 1]), activation_site="residual"
        )
        with self.assertRaises(TypeError):
            ablator._register_sae_hooks(
                [0, 1, 2],
                positions=None,
                mode="replace",
                delta_scale=1.0,
                operation="zero",
                operation_scale=1.0,
                diagnostics_buffer=[],
            )

    def test_hooks_are_removed_after_a_full_experiment(self):
        model = MultiLayerModel(n_layers=5)
        dataset = DatasetStub(num_samples=2)
        ablator = MultiLayerFeatureAblator(
            adapt(model), make_saes([1, 2, 3]), activation_site="residual"
        )

        ablator.batch_ablation_experiment(
            dataset,
            feature_indices={1: [0], 2: [1], 3: [2]},
            position_type="all",
            mode="replace",
            max_samples=2,
            score_options=False,
        )
        self.assertEqual(hook_counts(model), [0, 0, 0, 0, 0])


class TestEncodePositionsOnly(unittest.TestCase):
    """The optimisation must not change the numbers, only the memory and the FLOPs.

    The encoder is row-wise, so encoding a subset of positions yields the same rows and the
    results are bitwise identical, not merely close. Verified separately at production
    dimensions (d_model 4096, 8192 features, 620-token sequence, 25 positions) in both
    modes. A failure here means the slicing path has genuinely diverged.
    """

    def _compare(self, mode):
        torch.manual_seed(7)
        model = MultiLayerModel(n_layers=1)
        sae = SparseAutoencoder(d_model=4, n_features=8)
        ablator = FeatureAblator(adapt(model), sae, layer_idx=0, activation_site="residual")

        acts = torch.randn(1, 6, 4)
        positions = [1, 3, 4]
        kwargs = dict(positions=positions, mode=mode)

        full = ablator.create_ablation_hook([0, 2], encode_positions_only=False, **kwargs)
        sliced = ablator.create_ablation_hook([0, 2], encode_positions_only=True, **kwargs)

        out_full = full(None, None, acts)
        out_sliced = sliced(None, None, acts)

        self.assertTrue(torch.equal(out_full, out_sliced))
        # Guard against a vacuous pass: the intervention must actually change something.
        self.assertFalse(torch.equal(out_full, acts))
        # ...and only at the requested positions.
        untouched = [i for i in range(acts.shape[1]) if i not in positions]
        self.assertTrue(torch.equal(out_sliced[:, untouched, :], acts[:, untouched, :]))

    def test_replace_mode_matches_full_encoding(self):
        self._compare("replace")

    def test_delta_mode_matches_full_encoding(self):
        self._compare("residual")

    def test_no_positions_falls_back_to_full_encoding(self):
        torch.manual_seed(7)
        model = MultiLayerModel(n_layers=1)
        sae = SparseAutoencoder(d_model=4, n_features=8)
        ablator = FeatureAblator(adapt(model), sae, layer_idx=0, activation_site="residual")

        acts = torch.randn(1, 5, 4)
        full = ablator.create_ablation_hook([1], positions=None, mode="replace")
        sliced = ablator.create_ablation_hook(
            [1], positions=None, mode="replace", encode_positions_only=True
        )
        self.assertTrue(torch.equal(full(None, None, acts), sliced(None, None, acts)))


class TestPerLayerDiagnostics(unittest.TestCase):
    def test_diagnostics_are_tagged_and_broken_down_by_layer(self):
        model = MultiLayerModel(n_layers=4)
        ablator = MultiLayerFeatureAblator(
            adapt(model), make_saes([1, 2]), activation_site="residual"
        )
        acts = torch.randn(1, 4, 4)
        buffer = []

        handles = ablator._register_sae_hooks(
            {1: [0], 2: [1]},
            positions=[0, 1],
            mode="replace",
            delta_scale=1.0,
            operation="zero",
            operation_scale=1.0,
            diagnostics_buffer=buffer,
        )
        for layer in (1, 2):
            model.layers[layer](acts)
        for handle in handles:
            handle.remove()

        self.assertEqual([entry["layer"] for entry in buffer], [1, 2])

        summary = ablator._summarize_diagnostics(buffer)
        self.assertEqual(summary["perturb_calls"], 2)
        self.assertEqual(summary["perturb_layers"], [1, 2])
        self.assertEqual(sorted(summary["perturb_by_layer"]), ["1", "2"])
        self.assertAlmostEqual(
            summary["perturb_total_relative_norm"],
            sum(v["relative_norm"] for v in summary["perturb_by_layer"].values()),
        )

    def test_base_fields_survive_for_untagged_diagnostics(self):
        model = MultiLayerModel(n_layers=2)
        ablator = MultiLayerFeatureAblator(
            adapt(model), make_saes([0]), activation_site="residual"
        )
        summary = ablator._summarize_diagnostics([])
        self.assertEqual(summary["perturb_calls"], 0)
        self.assertIsNone(summary["perturb_mean_delta_norm"])
        self.assertEqual(summary["perturb_by_layer"], {})
        self.assertIsNone(summary["perturb_total_relative_norm"])


if __name__ == "__main__":
    unittest.main()
