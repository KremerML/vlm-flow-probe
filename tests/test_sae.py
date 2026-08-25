import unittest
import torch

from vlmflowprobe.core.sparse_autoencoder import SparseAutoencoder


class TestSparseAutoencoder(unittest.TestCase):
    def test_encode_decode_shapes(self):
        sae = SparseAutoencoder(d_model=8, n_features=16, l1_coeff=1e-3)
        x = torch.randn(2, 3, 8)
        recon, feats = sae(x)
        self.assertEqual(recon.shape, x.shape)
        self.assertEqual(feats.shape, (6, 16))

    def test_loss_and_grad(self):
        sae = SparseAutoencoder(d_model=4, n_features=8, l1_coeff=1e-3)
        x = torch.randn(5, 4, requires_grad=True)
        total, recon_loss, l1_loss = sae.get_loss(x)
        self.assertGreater(total.item(), 0.0)
        total.backward()
        self.assertIsNotNone(sae.encoder.weight.grad)
        self.assertIsNotNone(sae.decoder.weight.grad)


if __name__ == "__main__":
    unittest.main()
