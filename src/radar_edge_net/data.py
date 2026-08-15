"""Validated HDF5 loading and leakage-safe dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupShuffleSplit

CLOUDS_KEY = "fused_point_clouds"
LABELS_KEY = "labels"
SEQUENCE_IDS_KEY = "sequence_ids"
SCHEMA_VERSION = "1.0"
EXPECTED_VIEWS = 2
EXPECTED_POINTS = 1024
EXPECTED_FEATURES = 4
SUPPORTED_LABELS = frozenset({5, 7})


@dataclass(frozen=True)
class DatasetBundle:
    """In-memory samples with the source sequence retained for safe splitting."""

    clouds: NDArray[np.float32]
    labels: NDArray[np.int32]
    sequence_ids: NDArray[np.str_]

    def subset(self, indices: NDArray[np.integer]) -> DatasetBundle:
        return DatasetBundle(
            clouds=self.clouds[indices],
            labels=self.labels[indices],
            sequence_ids=self.sequence_ids[indices],
        )


def _validate_arrays(clouds: np.ndarray, labels: np.ndarray, source: Path) -> None:
    expected_tail = (EXPECTED_VIEWS, EXPECTED_POINTS, EXPECTED_FEATURES)
    if clouds.ndim != 4 or clouds.shape[1:] != expected_tail:
        raise ValueError(
            f"{source}: expected {CLOUDS_KEY} shape (n, {EXPECTED_VIEWS}, "
            f"{EXPECTED_POINTS}, {EXPECTED_FEATURES}), got {clouds.shape}"
        )
    if labels.ndim != 1 or len(labels) != len(clouds):
        raise ValueError(
            f"{source}: labels must have shape ({len(clouds)},), got {labels.shape}"
        )
    unexpected = set(np.unique(labels).tolist()) - SUPPORTED_LABELS
    if unexpected:
        raise ValueError(f"{source}: unsupported labels {sorted(unexpected)}")
    if np.isinf(clouds).any():
        raise ValueError(f"{source}: point clouds contain infinite values")


def load_preprocessed_directory(directory: str | Path) -> DatasetBundle:
    """Load per-sequence files produced by preprocessing.

    NaNs are allowed because they represent padded points. Every source filename is
    retained as a group id so downstream splitting cannot mix a sequence across
    train and validation sets.
    """

    directory = Path(directory)
    paths = sorted(directory.glob("*_preprocessed.h5"))
    if not paths:
        raise FileNotFoundError(f"No *_preprocessed.h5 files found in {directory}")

    cloud_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sequence_parts: list[np.ndarray] = []

    for path in paths:
        with h5py.File(path, "r") as handle:
            missing = {CLOUDS_KEY, LABELS_KEY} - set(handle.keys())
            if missing:
                raise ValueError(f"{path}: missing HDF5 datasets {sorted(missing)}")
            clouds = handle[CLOUDS_KEY][:].astype(np.float32, copy=False)
            labels = handle[LABELS_KEY][:].astype(np.int32, copy=False)

        _validate_arrays(clouds, labels, path)
        sequence_id = path.name.removesuffix("_preprocessed.h5")
        cloud_parts.append(clouds)
        label_parts.append(labels)
        sequence_parts.append(np.repeat(np.asarray(sequence_id, dtype=np.str_), len(labels)))

    return DatasetBundle(
        clouds=np.concatenate(cloud_parts),
        labels=np.concatenate(label_parts),
        sequence_ids=np.concatenate(sequence_parts),
    )


def split_by_sequence(
    dataset: DatasetBundle,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DatasetBundle, DatasetBundle]:
    """Split complete sequences while approximately preserving class balance."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(np.unique(dataset.sequence_ids)) < 2:
        raise ValueError("At least two source sequences are required for a grouped split")

    splitter = GroupShuffleSplit(
        n_splits=64,
        test_size=validation_fraction,
        random_state=seed,
    )
    classes = np.unique(dataset.labels)
    overall = _class_proportions(dataset.labels, classes)
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for train_indices, validation_indices in splitter.split(
        dataset.clouds, dataset.labels, groups=dataset.sequence_ids
    ):
        train_labels = dataset.labels[train_indices]
        validation_labels = dataset.labels[validation_indices]
        if not set(classes).issubset(train_labels) or not set(classes).issubset(validation_labels):
            continue
        score = float(
            np.abs(_class_proportions(train_labels, classes) - overall).sum()
            + np.abs(_class_proportions(validation_labels, classes) - overall).sum()
        )
        if best is None or score < best[0]:
            best = (score, train_indices, validation_indices)

    if best is None:
        raise ValueError("Could not create train and validation sets containing every class")

    _, train_indices, validation_indices = best
    return dataset.subset(train_indices), dataset.subset(validation_indices)


def _class_proportions(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.array([(labels == label).mean() for label in classes])


def write_dataset(dataset: DatasetBundle, output_path: str | Path) -> None:
    """Write one processed split using the canonical schema."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.create_dataset(CLOUDS_KEY, data=dataset.clouds, compression="gzip")
        handle.create_dataset(LABELS_KEY, data=dataset.labels, compression="gzip")
        string_type = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            SEQUENCE_IDS_KEY,
            data=dataset.sequence_ids.astype(object),
            dtype=string_type,
            compression="gzip",
        )
