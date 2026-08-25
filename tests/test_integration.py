import os
import tempfile
import unittest

from vlmflowprobe.core.config import load_config
from vlmflowprobe.features.feature_catalog import FeatureCatalog


class TestIntegration(unittest.TestCase):
    def test_load_default_config(self):
        config = load_config("nonexistent.yaml")
        self.assertIn("model", config.data)
        self.assertIn("sae", config.data)

    def test_feature_catalog_roundtrip(self):
        catalog = FeatureCatalog()
        catalog.add_feature(1, {"name": "feature_1", "type": "color"})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "catalog.json")
            catalog.export_to_json(path)
            new_catalog = FeatureCatalog()
            new_catalog.load_from_json(path)
            self.assertIn(1, new_catalog.features)


if __name__ == "__main__":
    unittest.main()
