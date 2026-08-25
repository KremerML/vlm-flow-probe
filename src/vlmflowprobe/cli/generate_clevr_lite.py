"""Generate a CLEVR-Lite dataset for VLM circuit analysis.

Creates synthetic 2D scenes (squares, circles, triangles in 6 colors) with
unambiguous attribute queries suitable for attention knockout and SAE experiments.

Usage:
    python sae_experiments/tools/generate_clevr_lite.py \
        --num_train 50000 --num_val 2000 --output_dir datasets/clevr_lite

At 50K train scenes (~3.7 questions/scene × ~6 question tokens) this produces
~1.1M question-token activation vectors for SAE training.
"""

import argparse
import sys
from pathlib import Path


from vlmflowprobe.data.clevr_lite.generator import CLEVRLiteGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CLEVR-Lite dataset")
    parser.add_argument("--output_dir", type=str, default="datasets/clevr_lite",
                        help="Root directory to write images and JSON files")
    parser.add_argument("--num_train", type=int, default=50000,
                        help="Number of training scenes to generate")
    parser.add_argument("--num_val", type=int, default=2000,
                        help="Number of validation scenes to generate")
    parser.add_argument("--held_out_ratio", type=float, default=0.4,
                        help="Fraction of color×shape combinations to hold out")
    parser.add_argument("--seed", type=int, default=32,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    generator = CLEVRLiteGenerator(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        held_out_ratio=args.held_out_ratio,
        seed=args.seed,
    )
    generator.generate_dataset()


if __name__ == "__main__":
    main()
