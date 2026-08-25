"""Pin the position-resolver policy divergences.

The archive carried three copies of this logic whose semantic differences
produced different published artifacts. Each divergence is pinned here with
the artifact that depends on it, so a future "cleanup" cannot silently change
either behavior.
"""

import unittest

from tests.stubs import adapt, CountingModel
from vlmflowprobe.positions import ABLATION_POLICY, COLLECTION_POLICY, resolve_positions


def make_adapter(**kwargs):
    return adapt(CountingModel(d_model=4), **kwargs)


class TestAllPositions(unittest.TestCase):
    def test_ablation_all_is_none_sentinel(self):
        # The ablator treats None as "no position restriction" (hook touches every
        # token). Published ablation runs with position_type=all depend on it.
        adapter = make_adapter()
        batch = adapter.build_inputs("q")
        self.assertIsNone(resolve_positions("all", batch, adapter, policy=ABLATION_POLICY))

    def test_collection_all_is_full_range(self):
        # The collector stores an explicit row per position; SAE training data
        # with position_type=all was built from the full index list.
        adapter = make_adapter()
        batch = adapter.build_inputs("q")
        self.assertEqual(
            resolve_positions("all", batch, adapter, policy=COLLECTION_POLICY),
            [0, 1, 2],
        )


class TestEmptyQuestion(unittest.TestCase):
    def test_ablation_empty_question_short_circuits_every_kind(self):
        # feature_ablator returned [] before branching — even "last" and "image",
        # which do not need the question span. Published ablation numbers were
        # computed under this guard.
        adapter = make_adapter(question_span=(), image_span=range(0, 2))
        batch = adapter.build_inputs("q")
        for kind in ("question", "last", "image", "attribute"):
            self.assertEqual(
                resolve_positions(kind, batch, adapter, policy=ABLATION_POLICY), [], kind
            )

    def test_collection_empty_question_still_resolves_last_and_image(self):
        # The collector had no early guard: "last"/"image" resolve regardless.
        adapter = make_adapter(question_span=(), image_span=range(0, 2))
        batch = adapter.build_inputs("q")
        self.assertEqual(
            resolve_positions("last", batch, adapter, policy=COLLECTION_POLICY), [2]
        )
        self.assertEqual(
            resolve_positions("image", batch, adapter, policy=COLLECTION_POLICY), [0, 1]
        )


class TestAttributeFallback(unittest.TestCase):
    def test_ablation_attribute_falls_back_to_question_span(self):
        # feature_ablator: no attribute tokens -> intervene on the whole question.
        adapter = make_adapter(question_span=(1, 2))
        batch = adapter.build_inputs("q")
        self.assertEqual(
            resolve_positions(
                "attribute", batch, adapter, policy=ABLATION_POLICY,
                line={"attribute_tokens": []},
            ),
            [1, 2],
        )

    def test_collection_attribute_falls_back_to_empty(self):
        # The collector skipped such samples instead (GQA attribute training data).
        adapter = make_adapter(question_span=(1, 2))
        batch = adapter.build_inputs("q")
        self.assertEqual(
            resolve_positions(
                "attribute", batch, adapter, policy=COLLECTION_POLICY,
                line={"attribute_tokens": []},
            ),
            [],
        )

    def test_attribute_offsets_are_question_relative_and_clamped(self):
        adapter = make_adapter(question_span=(1, 2))
        batch = adapter.build_inputs("q")
        line = {"attribute_tokens": [{"positions": [0, 1, 5]}]}
        # offsets 0,1 land on question positions 1,2; offset 5 exceeds the span.
        for policy in (ABLATION_POLICY, COLLECTION_POLICY):
            self.assertEqual(
                resolve_positions("attribute", batch, adapter, policy=policy, line=line),
                [1, 2],
            )


class TestSharedBranches(unittest.TestCase):
    def test_question_last_image_and_default(self):
        adapter = make_adapter(question_span=(1,), image_span=range(0, 1))
        batch = adapter.build_inputs("q")
        for policy in (ABLATION_POLICY, COLLECTION_POLICY):
            self.assertEqual(resolve_positions("question", batch, adapter, policy=policy), [1])
            self.assertEqual(resolve_positions("last", batch, adapter, policy=policy), [2])
            self.assertEqual(resolve_positions("image", batch, adapter, policy=policy), [0])
            # unknown kinds fall back to the question span
            self.assertEqual(resolve_positions("bogus", batch, adapter, policy=policy), [1])


if __name__ == "__main__":
    unittest.main()
