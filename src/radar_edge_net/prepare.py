"""Command-line entry point for creating leakage-safe processed splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from radar_edge_net.data import load_preprocessed_directory, split_by_sequence, write_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create sequence-grouped train and validation HDF5 files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Directory containing *_preprocessed.h5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Destination for train_data.h5 and val_data.h5.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = load_preprocessed_directory(args.input_dir)
    train, validation = split_by_sequence(
        dataset,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    write_dataset(train, args.output_dir / "train_data.h5")
    write_dataset(validation, args.output_dir / "val_data.h5")
    print(
        f"Wrote {len(train.labels)} train and {len(validation.labels)} validation samples "
        f"from {len(set(dataset.sequence_ids))} sequences to {args.output_dir}"
    )


if __name__ == "__main__":
    main()

