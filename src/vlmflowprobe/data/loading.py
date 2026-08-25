"""Model-agnostic sample iteration.

Replaces the archive's per-model DataLoader: the dataset supplies question
records (and optionally images), the adapter builds the batches. Iteration
order is exactly ``dataset.questions`` order — the sample caches and the
``question_ids_sha1`` pairing guard all depend on that.
"""

from typing import Iterator, Optional, Tuple

from tqdm import tqdm

from vlmflowprobe.adapters.base import ModelAdapter, ModelBatch


def iter_batches(
    dataset,
    adapter: ModelAdapter,
    max_samples: Optional[int] = None,
    show_progress: bool = False,
    progress_desc: str = "Samples",
) -> Iterator[Tuple[ModelBatch, dict]]:
    """Yield ``(ModelBatch, question line)`` in ``dataset.questions`` order.

    The dataset contract: ``.questions`` (list of dicts with ``q_id``),
    ``.dataset_dict`` (``q_id -> detail`` with at least ``question``), and
    optionally ``.load_image(line)`` returning a PIL image or path.
    """
    lines = dataset.questions
    total = len(lines)
    if max_samples is not None:
        total = min(total, max_samples)

    iterator = enumerate(lines)
    if show_progress:
        iterator = tqdm(iterator, total=total, desc=progress_desc)

    for idx, line in iterator:
        if max_samples is not None and idx >= max_samples:
            break
        detail = dataset.dataset_dict[line["q_id"]]
        image = dataset.load_image(line) if hasattr(dataset, "load_image") else None
        batch = adapter.build_inputs(detail.get("question", ""), image)
        yield batch, line
