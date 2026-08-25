import unittest

from vlmflowprobe.knockout.block_config import (
    build_block_config,
    build_block_config_for_layers,
)

PAIRS = [(5, 1), (6, 2)]


class TestBuildBlockConfigForLayers(unittest.TestCase):
    def test_explicit_layer_set_is_used_verbatim(self):
        config = build_block_config_for_layers([10, 11, 12, 13, 14], PAIRS)
        self.assertEqual(sorted(config), [10, 11, 12, 13, 14])
        for pairs in config.values():
            self.assertEqual(pairs, PAIRS)

    def test_non_contiguous_and_unsorted_layers(self):
        # The nested spans always grow downward from 14, so separating "how many layers"
        # from "which layers" needs sets like {10, 12, 14} that no window can express.
        config = build_block_config_for_layers([14, 10, 12], PAIRS)
        self.assertEqual(sorted(config), [10, 12, 14])

    def test_duplicates_collapse(self):
        self.assertEqual(sorted(build_block_config_for_layers([3, 3, 4], PAIRS)), [3, 4])

    def test_downstream_range_is_expressible(self):
        config = build_block_config_for_layers(range(12, 32), PAIRS)
        self.assertEqual(min(config), 12)
        self.assertEqual(max(config), 31)
        self.assertEqual(len(config), 20)

    def test_pairs_are_copied_not_shared(self):
        config = build_block_config_for_layers([1, 2], PAIRS)
        config[1].append((9, 9))
        self.assertEqual(config[2], PAIRS)
        self.assertEqual(PAIRS, [(5, 1), (6, 2)])

    def test_single_layer_matches_the_window_form(self):
        self.assertEqual(
            build_block_config_for_layers([7], PAIRS),
            build_block_config(7, num_layers=32, window=1, src_tgt_pairs=PAIRS),
        )


if __name__ == "__main__":
    unittest.main()
