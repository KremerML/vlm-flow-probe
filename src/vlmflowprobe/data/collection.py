"""Collects hidden activations from target model layers."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

# Legacy LLaVA-repo image placeholder sentinel; dies with the Phase-2 adapter rewiring.
IMAGE_TOKEN_INDEX = -200

from vlmflowprobe.hooks import HookManager, create_activation_capture_hook, get_target_module
from vlmflowprobe.knockout.knockout_utils import estimate_image_token_count
from vlmflowprobe.utils import token_utils


class ActivationCollector:
    """Collects activations from a specific layer in a model."""

    def __init__(self, model: torch.nn.Module, layer_idx: int, activation_site: str = "residual"):
        self.model = model
        self.layer_idx = layer_idx
        self.activation_site = activation_site
        self.hook_manager = HookManager(model)
        self.storage: Dict[str, torch.Tensor] = {}

    def register_hooks(self) -> None:
        layer = get_target_module(self.model, self.layer_idx, self.activation_site)
        self.hook_manager.register_forward_hook(
            layer, create_activation_capture_hook(self.storage, "acts")
        )

    def collect_from_dataset(
        self,
        dataset,
        position_type: str = "question",
        tokenizer=None,
        max_samples: Optional[int] = None,
        device: Optional[str] = None,
        show_progress: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5000,
    ):
        if hasattr(dataset, "create_dataloader"):
            data_loader = dataset.create_dataloader()
            questions = dataset.questions
            dataset_dict = dataset.dataset_dict
        else:
            data_loader = dataset
            questions = getattr(dataset, "questions", None)
            dataset_dict = getattr(dataset, "dataset_dict", {})
            if questions is None:
                raise ValueError("dataset must provide questions list")

        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        start_idx, offset, metadata, chunk_files = 0, 0, [], []
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            start_idx, offset, metadata, chunk_files = _load_checkpoint(
                checkpoint_dir
            )
            if start_idx > 0:
                n_rows = sum(
                    m["count"] for m in metadata
                )
                print(
                    f"[ActivationCollector] Resuming from sample {start_idx}, "
                    f"{len(metadata)} questions, {n_rows} activation rows, "
                    f"{len(chunk_files)} chunks on disk"
                )

        self.register_hooks()
        pending: List[torch.Tensor] = []
        samples_since_save = 0
        idx = start_idx
        try:
            iterator = zip(data_loader, questions)
            total = len(questions) if questions is not None else None
            if max_samples is not None and total is not None:
                total = min(total, max_samples)
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=total, desc="Collecting activations",
                                initial=start_idx)

            for idx, (batch, line) in enumerate(iterator):
                if max_samples is not None and idx >= max_samples:
                    break
                if idx < start_idx:
                    continue

                input_ids, image_tensor, image_sizes, prompts, _ = batch
                input_ids = input_ids.to(device)
                image_tensor = [img.to(device) for img in image_tensor]

                image_token_count = estimate_image_token_count(
                    self.model, input_ids, image_tensor, image_sizes
                )

                self.storage.pop("acts", None)
                with torch.no_grad():
                    _ = self.model(
                        input_ids=input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        use_cache=False,
                    )

                acts = self.storage.get("acts")
                if acts is None:
                    continue
                acts = self._normalize_activation_tensor(acts)
                if acts is None:
                    continue

                question_text = dataset_dict[line["q_id"]]["question"]
                positions = self._select_positions(
                    position_type,
                    input_ids,
                    image_token_count,
                    question_text,
                    tokenizer,
                    line,
                )
                if not positions:
                    continue

                pending.append(acts[positions].cpu().to(torch.float16))
                metadata.append(
                    {
                        "question_id": line["q_id"],
                        "question": question_text,
                        "answer": dataset_dict[line["q_id"]].get("answer", ""),
                        "positions": positions,
                        "attribute_tokens": line.get("attribute_tokens", []),
                        "start_idx": offset,
                        "count": len(positions),
                    }
                )
                offset += len(positions)
                samples_since_save += 1

                if (checkpoint_dir
                        and samples_since_save >= checkpoint_interval
                        and pending):
                    chunk_path = _flush_chunk(
                        checkpoint_dir, len(chunk_files), pending
                    )
                    chunk_files.append(chunk_path)
                    pending.clear()
                    _save_manifest(
                        checkpoint_dir, idx + 1, offset, metadata, chunk_files
                    )
                    samples_since_save = 0

        finally:
            self.hook_manager.remove_hooks()
            if checkpoint_dir and pending:
                chunk_path = _flush_chunk(
                    checkpoint_dir, len(chunk_files), pending
                )
                chunk_files.append(chunk_path)
                pending.clear()
                _save_manifest(
                    checkpoint_dir, idx + 1, offset, metadata, chunk_files
                )

        if checkpoint_dir:
            return metadata, chunk_files

        if not pending:
            return torch.empty(0), metadata
        total_rows = sum(p.shape[0] for p in pending)
        d_model = pending[0].shape[-1]
        combined = torch.empty(total_rows, d_model, dtype=torch.float16)
        row = 0
        for p in pending:
            n = p.shape[0]
            combined[row : row + n] = p
            row += n
        del pending
        return combined, metadata

    @staticmethod
    def _normalize_activation_tensor(acts):
        if acts is None:
            return None
        if isinstance(acts, (tuple, list)):
            acts = acts[0]
        if acts is None:
            return None
        while acts.ndim > 3 and acts.shape[0] == 1:
            acts = acts[0]
        if acts.ndim != 3:
            return None
        return acts[0]

    def _select_positions(
        self,
        position_type: str,
        input_ids: torch.Tensor,
        image_token_count: int,
        question_text: str,
        tokenizer,
        line: Dict[str, Any],
    ) -> List[int]:
        if position_type == "all":
            expanded_len = input_ids.shape[-1] - 1 + image_token_count
            return list(range(expanded_len))

        question_range = token_utils.get_question_token_range(
            input_ids[0],
            image_token_count,
            question_text=question_text,
            tokenizer=tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
        )

        if position_type == "last":
            ntoks = input_ids.shape[-1] + image_token_count - 1
            return [max(0, ntoks - 1)]

        if position_type == "question":
            return question_range

        if position_type == "attribute":
            attr_positions = []
            for attr in line.get("attribute_tokens", []):
                attr_positions.extend(attr.get("positions", []))
            if not question_range:
                return []
            start = question_range[0]
            end = question_range[-1]
            positions = [start + pos for pos in attr_positions]
            positions = [pos for pos in positions if start <= pos <= end]
            return sorted(set(positions))

        if position_type == "image":
            ids_list = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
            try:
                img_placeholder_idx = ids_list.index(IMAGE_TOKEN_INDEX)
            except ValueError:
                return []
            return list(range(img_placeholder_idx, img_placeholder_idx + image_token_count))

        return question_range


def reassemble_chunks(
    chunk_files: List[str],
    cleanup: bool = True,
) -> torch.Tensor:
    """Load chunk files one at a time into a pre-allocated tensor.

    Call this AFTER freeing the LLaVA model to avoid OOM.
    """
    if not chunk_files:
        return torch.empty(0)

    total_rows = 0
    d_model = None
    for cf in chunk_files:
        info = torch.load(cf, weights_only=True, map_location="meta")
        total_rows += info.shape[0]
        if d_model is None:
            d_model = info.shape[-1]
        del info

    combined = torch.empty(total_rows, d_model, dtype=torch.float16)
    row = 0
    for cf in chunk_files:
        c = torch.load(cf, weights_only=True, map_location="cpu")
        n = c.shape[0]
        combined[row : row + n] = c
        row += n
        del c

    if cleanup:
        checkpoint_dir = os.path.dirname(chunk_files[0])
        _cleanup_checkpoint(checkpoint_dir, chunk_files)

    return combined


def _flush_chunk(
    checkpoint_dir: str, chunk_idx: int, pending: List[torch.Tensor]
) -> str:
    total_rows = sum(t.shape[0] for t in pending)
    d_model = pending[0].shape[-1]
    chunk = torch.empty(total_rows, d_model, dtype=pending[0].dtype)
    row = 0
    for t in pending:
        n = t.shape[0]
        chunk[row : row + n] = t
        row += n
    path = os.path.join(checkpoint_dir, f"chunk_{chunk_idx:04d}.pt")
    torch.save(chunk, path)
    return path


def _save_manifest(
    checkpoint_dir: str,
    next_sample_idx: int,
    offset: int,
    metadata: List[Dict[str, Any]],
    chunk_files: List[str],
) -> None:
    manifest = {
        "next_sample_idx": next_sample_idx,
        "offset": offset,
        "n_metadata": len(metadata),
        "chunk_files": [os.path.basename(f) for f in chunk_files],
    }
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    meta_path = os.path.join(checkpoint_dir, "metadata.json")
    tmp_manifest = manifest_path + ".tmp"
    tmp_meta = meta_path + ".tmp"
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f)
    with open(tmp_meta, "w") as f:
        json.dump(metadata, f)
    os.replace(tmp_manifest, manifest_path)
    os.replace(tmp_meta, meta_path)


def _load_checkpoint(
    checkpoint_dir: str,
) -> Tuple[int, int, List[Dict[str, Any]], List[str]]:
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return 0, 0, [], []
    with open(manifest_path) as f:
        manifest = json.load(f)
    meta_path = os.path.join(checkpoint_dir, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)
    chunk_files = [
        os.path.join(checkpoint_dir, name)
        for name in manifest["chunk_files"]
    ]
    for cf in chunk_files:
        if not os.path.exists(cf):
            print(f"[ActivationCollector] Chunk missing: {cf}, starting fresh")
            return 0, 0, [], []
    return (
        manifest["next_sample_idx"],
        manifest["offset"],
        metadata,
        chunk_files,
    )


def _cleanup_checkpoint(
    checkpoint_dir: str, chunk_files: List[str]
) -> None:
    for cf in chunk_files:
        try:
            os.remove(cf)
        except OSError:
            pass
    for name in ("manifest.json", "metadata.json"):
        try:
            os.remove(os.path.join(checkpoint_dir, name))
        except OSError:
            pass
    try:
        os.rmdir(checkpoint_dir)
    except OSError:
        pass
