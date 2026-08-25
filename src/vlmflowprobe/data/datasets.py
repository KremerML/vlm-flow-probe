"""Dataset adapter for CLEVR-Lite, compatible with the AttributeVQADataset interface.

Loads from {data_dir}/{split}_questions.json and exposes the same
`questions`, `dataset_dict`, and `create_dataloader()` interface used by
all downstream pipeline scripts.

Key design choices:
- `true option` is set to the correct answer (e.g. "blue")
- `false option` is a deterministically sampled distractor from the closed
  color/shape vocabulary — allows `forced_choice_margin` metric to work
  without any changes to knockout_runner or feature_ablator.
- `img_id` is the bare filename (e.g. "00000000.png") so CustomDataset can
  construct the full path as os.path.join(image_folder, img_id).
- task_name "CLEVRLite" hits the else-branch in CustomDataset (no bboxes
  needed), setting mask_tensor=None.
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from vlmflowprobe.utils import token_utils

_COLORS = ["red", "blue", "green", "yellow", "purple", "black"]
_SHAPES = ["square", "circle", "triangle"]


def _sample_distractor(item: Dict[str, Any], rng: random.Random) -> str:
    """Return a plausible wrong answer from the closed vocabulary."""
    ans = item["answer"].lower().strip()
    qtype = item.get("question_type", "")
    if qtype.startswith("query_shape"):
        pool = [s for s in _SHAPES if s != ans]
    else:
        # query_color_unambiguous and query_color_negation both have color answers
        pool = [c for c in _COLORS if c != ans]
    return rng.choice(pool) if pool else ans


class CLEVRLiteVQADataset:
    """CLEVR-Lite dataset adapter with the same interface as AttributeVQADataset."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer,
        image_processor,
        model_config,
        conv_mode: str = "vicuna_v1",
        filter_held_out: Optional[bool] = None,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode
        self.filter_held_out = filter_held_out

        questions_path = os.path.join(data_dir, f"{split}_questions.json")
        with open(questions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if filter_held_out is not None:
            raw = [item for item in raw if item["is_held_out_combo"] == filter_held_out]

        rng = random.Random(seed)

        self.questions: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw):
            q_id = f"{item['scene_id']}_{idx}"
            img_id = Path(item["image_path"]).name  # "00000000.png"
            entry = {
                "q_id": q_id,
                "question": item["question"],
                "answer": item["answer"].lower().strip(),
                "img_id": img_id,
                "true option": item["answer"].lower().strip(),
                "false option": _sample_distractor(item, rng),
                "question_type": item["question_type"],
                "is_held_out": item["is_held_out_combo"],
                "scene_objects": item["scene_objects"],
            }
            self.questions.append(entry)

        self.dataset_dict: Dict[str, Dict[str, Any]] = {
            entry["q_id"]: entry for entry in self.questions
        }

        self._extract_attribute_info()

    @property
    def image_folder(self) -> str:
        return os.path.join(self.data_dir, self.split, "images")

    def _extract_attribute_info(self) -> None:
        """Populate attribute_tokens on each question entry."""
        for entry in self.questions:
            question = entry["question"]
            attrs = token_utils.extract_attribute_words(question)
            attr_entries = []
            seen: set = set()
            for word, category, _ in attrs:
                if word in seen:
                    continue
                positions = token_utils.get_token_positions(
                    question, [word], self.tokenizer
                )
                if not positions:
                    continue
                attr_entries.append(
                    {"word": word, "category": category, "positions": positions}
                )
                seen.add(word)
            entry["attribute_tokens"] = attr_entries

    def create_dataloader(self, batch_size: int = 1, num_workers: int = 0):
        # Rewired in Phase 2: batches are built by the model adapter
        # (vlmflowprobe.data.loading), not by a model-specific loader here.
        raise NotImplementedError(
            "create_dataloader moves to vlmflowprobe.data.loading (adapter-driven); "
            "pending Phase 2 of the port"
        )
