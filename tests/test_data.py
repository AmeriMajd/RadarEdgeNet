from pathlib import Path

import h5py
import numpy as np
import pytest

from radar_edge_net.data import (
    CLOUDS_KEY,
    LABELS_KEY,
    SCHEMA_VERSION,
    SEQUENCE_IDS_KEY,
    DatasetBundle,
    load_preprocessed_directory,
    split_by_sequence,
    write_dataset,
)


def _write_sequence(path: Path, labels: list[int]) -> None:
    clouds = np.zeros((len(labels), 2, 1024, 4), dtype=np.float32)
    clouds[:, :, 10:, :] = np.nan
    with h5py.File(path, "w") as handle:
        handle.create_dataset(CLOUDS_KEY, data=clouds)
        handle.create_dataset(LABELS_KEY, data=np.asarray(labels, dtype=np.int32))


def test_load_and_split_keep_sequences_disjoint(tmp_path: Path) -> None:
    for index in range(10):
        _write_sequence(tmp_path / f"sequence_{index}_preprocessed.h5", [5, 7, 5, 7])

    dataset = load_preprocessed_directory(tmp_path)
    train, validation = split_by_sequence(dataset, seed=7)

    assert dataset.clouds.shape == (40, 2, 1024, 4)
    assert set(train.sequence_ids).isdisjoint(validation.sequence_ids)
    assert set(train.labels) == {5, 7}
    assert set(validation.labels) == {5, 7}


def test_loader_rejects_old_or_incomplete_schema(tmp_path: Path) -> None:
    path = tmp_path / "sequence_1_preprocessed.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("point_clouds", data=np.zeros((1, 1024, 4)))
        handle.create_dataset(LABELS_KEY, data=np.array([5]))

    with pytest.raises(ValueError, match="missing HDF5 datasets"):
        load_preprocessed_directory(tmp_path)


def test_writer_records_schema_and_provenance(tmp_path: Path) -> None:
    dataset = DatasetBundle(
        clouds=np.zeros((2, 2, 1024, 4), dtype=np.float32),
        labels=np.array([5, 7], dtype=np.int32),
        sequence_ids=np.array(["sequence_1", "sequence_2"]),
    )
    output = tmp_path / "train.h5"

    write_dataset(dataset, output)

    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema_version"] == SCHEMA_VERSION
        assert set(handle.keys()) == {CLOUDS_KEY, LABELS_KEY, SEQUENCE_IDS_KEY}

