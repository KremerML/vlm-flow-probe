import unittest

from vlmflowprobe.utils.config_utils import (
    resolve_primary_task_type,
    resolve_task_types,
    resolve_training_position_type,
)


class TestConfigUtils(unittest.TestCase):
    def test_resolve_task_types_from_string(self):
        self.assertEqual(resolve_task_types("ChooseAttr"), ["ChooseAttr"])

    def test_resolve_task_types_from_list(self):
        self.assertEqual(resolve_task_types(["ChooseAttr", "ChooseRel"]), ["ChooseAttr", "ChooseRel"])

    def test_resolve_task_types_filters_empty(self):
        self.assertEqual(resolve_task_types(["", "  ", "ChooseRel"]), ["ChooseRel"])

    def test_resolve_task_types_default_fallback(self):
        self.assertEqual(resolve_task_types(None), ["ChooseAttr"])
        self.assertEqual(resolve_task_types([]), ["ChooseAttr"])
        self.assertEqual(resolve_task_types(""), ["ChooseAttr"])

    def test_resolve_primary_task_type(self):
        self.assertEqual(resolve_primary_task_type(["ChooseRel", "ChooseAttr"]), "ChooseRel")
        self.assertEqual(resolve_primary_task_type("ChooseAttr"), "ChooseAttr")

    def test_resolve_training_position_type_precedence(self):
        training_cfg = {"position_type": "question"}
        feature_cfg = {"position_type": "attribute"}
        self.assertEqual(
            resolve_training_position_type("last", training_cfg, feature_cfg),
            "last",
        )
        self.assertEqual(
            resolve_training_position_type(None, training_cfg, feature_cfg),
            "question",
        )
        self.assertEqual(
            resolve_training_position_type(None, {}, feature_cfg),
            "attribute",
        )
        self.assertEqual(
            resolve_training_position_type(None, {}, {}),
            "question",
        )


if __name__ == "__main__":
    unittest.main()
