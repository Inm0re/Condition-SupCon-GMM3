#!/usr/bin/env python3
"""Prepare the fixed-speed HUST bearing data for one-class TSAD.

The source .xls files are tab-separated text files with 22 header lines.
Only the X/Y/Z vibration channels are used. Normal recordings are split in
half before windowing; the first half forms the normal-only training set, and
the second half forms the normal portion of the test set. All fixed-speed
fault recordings are retained in full and placed only in the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


SAMPLE_RATE_HZ = 25_600
SOURCE_POINTS = 262_144
HALF_POINTS = SOURCE_POINTS // 2
CHANNELS = ("X", "Y", "Z")
WINDOW_LENGTH = 2048
WINDOW_STRIDE = 256
FIXED_SPEEDS_HZ = (20, 25, 30, 35, 40, 60, 65, 70, 75, 80)
STATE_TO_ID = {
    "H": 0,
    "0.5X_I": 1,
    "I": 2,
    "0.5X_O": 3,
    "O": 4,
    "0.5X_B": 5,
    "B": 6,
    "0.5X_C": 7,
    "C": 8,
}
STATE_DESCRIPTIONS = {
    "H": "healthy",
    "0.5X_I": "medium inner-race fault",
    "I": "severe inner-race fault",
    "0.5X_O": "medium outer-race fault",
    "O": "severe outer-race fault",
    "0.5X_B": "medium ball fault",
    "B": "severe ball fault",
    "0.5X_C": "medium combination fault",
    "C": "severe combination fault",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_fixed_file(path: Path) -> tuple[int, str] | None:
    stem = path.stem.upper()
    if "_VS_" in stem:
        return None
    for speed in FIXED_SPEEDS_HZ:
        suffix = f"_{speed}HZ"
        if stem.endswith(suffix):
            state = stem[: -len(suffix)]
            if state not in STATE_TO_ID:
                raise ValueError(f"Unknown state in {path.name}: {state}")
            return speed, state
    raise ValueError(f"Cannot parse fixed-speed filename: {path.name}")


def discover_files(source: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(source.glob("*.xls"), key=lambda p: p.name.upper()):
        parsed = parse_fixed_file(path)
        if parsed is None:
            continue
        speed, state = parsed
        records.append(
            {
                "path": path,
                "speed_hz": speed,
                "state": state,
                "class_id": STATE_TO_ID[state],
            }
        )
    records.sort(key=lambda r: (int(r["speed_hz"]), int(r["class_id"])))
    expected = len(FIXED_SPEEDS_HZ) * len(STATE_TO_ID)
    if len(records) != expected:
        raise ValueError(f"Expected {expected} fixed-speed files, found {len(records)}")
    coverage = Counter((int(r["speed_hz"]), str(r["state"])) for r in records)
    missing = [
        (speed, state)
        for speed in FIXED_SPEEDS_HZ
        for state in STATE_TO_ID
        if coverage[(speed, state)] != 1
    ]
    if missing:
        raise ValueError(f"Missing or duplicate speed/state combinations: {missing}")
    for file_id, record in enumerate(records):
        record["file_id"] = file_id
    return records


def load_signal(path: Path) -> np.ndarray:
    signal = np.loadtxt(
        path,
        delimiter="\t",
        skiprows=22,
        usecols=(2, 3, 4),
        dtype=np.float32,
    )
    if signal.shape != (SOURCE_POINTS, len(CHANNELS)):
        raise ValueError(f"Unexpected shape for {path.name}: {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite value found in {path.name}")
    return signal


def window_count(length: int) -> int:
    if length < WINDOW_LENGTH:
        return 0
    return (length - WINDOW_LENGTH) // WINDOW_STRIDE + 1


def make_windows(signal: np.ndarray) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(
        signal, window_shape=WINDOW_LENGTH, axis=0
    )[::WINDOW_STRIDE]
    windows = windows.transpose(0, 2, 1)
    expected = window_count(signal.shape[0])
    if windows.shape != (expected, WINDOW_LENGTH, len(CHANNELS)):
        raise AssertionError(f"Unexpected window shape: {windows.shape}")
    return windows


def calculate_scaler(normal_records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(CHANNELS), dtype=np.float64)
    total_sq = np.zeros(len(CHANNELS), dtype=np.float64)
    count = 0
    for record in normal_records:
        train_half = load_signal(Path(record["path"]))[:HALF_POINTS].astype(np.float64)
        total += train_half.sum(axis=0)
        total_sq += np.square(train_half).sum(axis=0)
        count += train_half.shape[0]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 1e-12):
        raise ValueError(f"Degenerate training standard deviation: {std}")
    return mean, std


def open_npy(path: Path, dtype: np.dtype | str, shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def write_readme(path: Path, config: dict[str, object]) -> None:
    text = f"""# HUST fixed-speed one-class TSAD dataset

This artifact was generated from the fixed-speed portion of the HUST bearing
dataset. The variable-speed `VS_0_40_0Hz` files are intentionally excluded.

## Protocol

- Effective input channels: X, Y, Z (3 vibration channels).
- Nominal sampling rate: {SAMPLE_RATE_HZ} Hz.
- Source length per recording: {SOURCE_POINTS} points.
- Training: first {HALF_POINTS} points of each of the 10 healthy recordings.
- Test normal portion: second {HALF_POINTS} points of each healthy recording.
- Test anomaly portion: all points from all 80 fixed-speed fault recordings.
- Window length: {WINDOW_LENGTH}; stride: {WINDOW_STRIDE}.
- Files are windowed independently. No window crosses a source-file boundary.
- Channel-wise z-score statistics are fitted only on normal training points.

## Layout

`sequences/` contains continuous standardized arrays and point-level labels.
`windows_L2048_H256/` contains ready-to-train windows and window-level labels.
`segments.csv` maps continuous output ranges back to the source recordings.
`file_index.json` maps numeric file IDs to source filenames and conditions.

## Shapes

- `sequences/train.npy`: `{tuple(config['sequence_shapes']['train'])}`
- `sequences/test.npy`: `{tuple(config['sequence_shapes']['test'])}`
- `windows_L2048_H256/train.npy`: `{tuple(config['window_shapes']['train'])}`
- `windows_L2048_H256/test.npy`: `{tuple(config['window_shapes']['test'])}`

Labels use `0` for healthy and `1` for anomalous. Detailed fault IDs are stored
separately and defined in `dataset_config.json`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    temp_output = output.with_name(output.name + ".tmp")
    if output.exists() or temp_output.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {temp_output}")

    records = discover_files(source)
    normal_records = [r for r in records if r["state"] == "H"]
    fault_records = [r for r in records if r["state"] != "H"]
    if len(normal_records) != 10 or len(fault_records) != 80:
        raise AssertionError((len(normal_records), len(fault_records)))

    train_points = len(normal_records) * HALF_POINTS
    test_normal_points = len(normal_records) * HALF_POINTS
    test_fault_points = len(fault_records) * SOURCE_POINTS
    test_points = test_normal_points + test_fault_points
    train_windows = len(normal_records) * window_count(HALF_POINTS)
    test_normal_windows = len(normal_records) * window_count(HALF_POINTS)
    test_fault_windows = len(fault_records) * window_count(SOURCE_POINTS)
    test_windows = test_normal_windows + test_fault_windows

    mean, std = calculate_scaler(normal_records)
    temp_output.mkdir(parents=True)
    sequence_dir = temp_output / "sequences"
    window_dir = temp_output / "windows_L2048_H256"
    sequence_dir.mkdir()
    window_dir.mkdir()

    np.savez(
        temp_output / "scaler.npz",
        mean=mean.astype(np.float64),
        std=std.astype(np.float64),
        channels=np.asarray(CHANNELS),
    )

    seq_train = open_npy(sequence_dir / "train.npy", np.float32, (train_points, 3))
    seq_train_label = open_npy(sequence_dir / "train_label.npy", np.uint8, (train_points,))
    seq_train_speed = open_npy(sequence_dir / "train_speed_hz.npy", np.uint8, (train_points,))
    seq_test = open_npy(sequence_dir / "test.npy", np.float32, (test_points, 3))
    seq_test_label = open_npy(sequence_dir / "test_label.npy", np.uint8, (test_points,))
    seq_test_fault = open_npy(sequence_dir / "test_fault_type.npy", np.uint8, (test_points,))
    seq_test_speed = open_npy(sequence_dir / "test_speed_hz.npy", np.uint8, (test_points,))

    win_train = open_npy(window_dir / "train.npy", np.float32, (train_windows, WINDOW_LENGTH, 3))
    win_train_label = open_npy(window_dir / "train_label.npy", np.uint8, (train_windows,))
    win_test = open_npy(window_dir / "test.npy", np.float32, (test_windows, WINDOW_LENGTH, 3))
    win_test_label = open_npy(window_dir / "test_label.npy", np.uint8, (test_windows,))
    win_test_fault = open_npy(window_dir / "test_fault_type.npy", np.uint8, (test_windows,))

    train_meta = {
        "file_id": np.empty(train_windows, dtype=np.int16),
        "speed_hz": np.empty(train_windows, dtype=np.uint8),
        "source_start": np.empty(train_windows, dtype=np.int32),
    }
    test_meta = {
        "file_id": np.empty(test_windows, dtype=np.int16),
        "speed_hz": np.empty(test_windows, dtype=np.uint8),
        "source_start": np.empty(test_windows, dtype=np.int32),
    }

    segments: list[dict[str, object]] = []
    train_point_cursor = 0
    test_point_cursor = 0
    train_window_cursor = 0
    test_window_cursor = 0

    # Healthy recordings: first half -> train, second half -> normal test.
    for record in normal_records:
        raw = load_signal(Path(record["path"]))
        scaled = ((raw.astype(np.float64) - mean) / std).astype(np.float32)
        speed = int(record["speed_hz"])
        file_id = int(record["file_id"])

        train_part = scaled[:HALF_POINTS]
        normal_test_part = scaled[HALF_POINTS:]
        train_end = train_point_cursor + HALF_POINTS
        test_end = test_point_cursor + HALF_POINTS
        seq_train[train_point_cursor:train_end] = train_part
        seq_train_label[train_point_cursor:train_end] = 0
        seq_train_speed[train_point_cursor:train_end] = speed
        seq_test[test_point_cursor:test_end] = normal_test_part
        seq_test_label[test_point_cursor:test_end] = 0
        seq_test_fault[test_point_cursor:test_end] = 0
        seq_test_speed[test_point_cursor:test_end] = speed

        train_w = make_windows(train_part)
        test_w = make_windows(normal_test_part)
        train_w_end = train_window_cursor + train_w.shape[0]
        test_w_end = test_window_cursor + test_w.shape[0]
        win_train[train_window_cursor:train_w_end] = train_w
        win_train_label[train_window_cursor:train_w_end] = 0
        win_test[test_window_cursor:test_w_end] = test_w
        win_test_label[test_window_cursor:test_w_end] = 0
        win_test_fault[test_window_cursor:test_w_end] = 0
        starts = np.arange(0, HALF_POINTS - WINDOW_LENGTH + 1, WINDOW_STRIDE, dtype=np.int32)
        train_meta["file_id"][train_window_cursor:train_w_end] = file_id
        train_meta["speed_hz"][train_window_cursor:train_w_end] = speed
        train_meta["source_start"][train_window_cursor:train_w_end] = starts
        test_meta["file_id"][test_window_cursor:test_w_end] = file_id
        test_meta["speed_hz"][test_window_cursor:test_w_end] = speed
        test_meta["source_start"][test_window_cursor:test_w_end] = starts + HALF_POINTS

        segments.extend(
            [
                {
                    "split": "train",
                    "file_id": file_id,
                    "source_file": Path(record["path"]).name,
                    "state": "H",
                    "fault_type_id": 0,
                    "speed_hz": speed,
                    "output_start": train_point_cursor,
                    "output_end_exclusive": train_end,
                    "source_start": 0,
                    "source_end_exclusive": HALF_POINTS,
                    "binary_label": 0,
                },
                {
                    "split": "test",
                    "file_id": file_id,
                    "source_file": Path(record["path"]).name,
                    "state": "H",
                    "fault_type_id": 0,
                    "speed_hz": speed,
                    "output_start": test_point_cursor,
                    "output_end_exclusive": test_end,
                    "source_start": HALF_POINTS,
                    "source_end_exclusive": SOURCE_POINTS,
                    "binary_label": 0,
                },
            ]
        )
        train_point_cursor = train_end
        test_point_cursor = test_end
        train_window_cursor = train_w_end
        test_window_cursor = test_w_end

    # Every point from every fixed-speed fault recording is retained in test.
    for record in fault_records:
        raw = load_signal(Path(record["path"]))
        scaled = ((raw.astype(np.float64) - mean) / std).astype(np.float32)
        speed = int(record["speed_hz"])
        file_id = int(record["file_id"])
        class_id = int(record["class_id"])
        state = str(record["state"])

        test_end = test_point_cursor + SOURCE_POINTS
        seq_test[test_point_cursor:test_end] = scaled
        seq_test_label[test_point_cursor:test_end] = 1
        seq_test_fault[test_point_cursor:test_end] = class_id
        seq_test_speed[test_point_cursor:test_end] = speed

        test_w = make_windows(scaled)
        test_w_end = test_window_cursor + test_w.shape[0]
        win_test[test_window_cursor:test_w_end] = test_w
        win_test_label[test_window_cursor:test_w_end] = 1
        win_test_fault[test_window_cursor:test_w_end] = class_id
        starts = np.arange(0, SOURCE_POINTS - WINDOW_LENGTH + 1, WINDOW_STRIDE, dtype=np.int32)
        test_meta["file_id"][test_window_cursor:test_w_end] = file_id
        test_meta["speed_hz"][test_window_cursor:test_w_end] = speed
        test_meta["source_start"][test_window_cursor:test_w_end] = starts

        segments.append(
            {
                "split": "test",
                "file_id": file_id,
                "source_file": Path(record["path"]).name,
                "state": state,
                "fault_type_id": class_id,
                "speed_hz": speed,
                "output_start": test_point_cursor,
                "output_end_exclusive": test_end,
                "source_start": 0,
                "source_end_exclusive": SOURCE_POINTS,
                "binary_label": 1,
            }
        )
        test_point_cursor = test_end
        test_window_cursor = test_w_end

    if (train_point_cursor, test_point_cursor) != (train_points, test_points):
        raise AssertionError((train_point_cursor, test_point_cursor))
    if (train_window_cursor, test_window_cursor) != (train_windows, test_windows):
        raise AssertionError((train_window_cursor, test_window_cursor))

    for array in (
        seq_train,
        seq_train_label,
        seq_train_speed,
        seq_test,
        seq_test_label,
        seq_test_fault,
        seq_test_speed,
        win_train,
        win_train_label,
        win_test,
        win_test_label,
        win_test_fault,
    ):
        array.flush()
    del seq_train, seq_train_label, seq_train_speed
    del seq_test, seq_test_label, seq_test_fault, seq_test_speed
    del win_train, win_train_label, win_test, win_test_label, win_test_fault

    np.savez(window_dir / "train_metadata.npz", **train_meta)
    np.savez(window_dir / "test_metadata.npz", **test_meta)

    with (temp_output / "segments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(segments[0]))
        writer.writeheader()
        writer.writerows(segments)

    file_index = [
        {
            "file_id": int(record["file_id"]),
            "source_file": Path(record["path"]).name,
            "speed_hz": int(record["speed_hz"]),
            "state": str(record["state"]),
            "fault_type_id": int(record["class_id"]),
            "description": STATE_DESCRIPTIONS[str(record["state"])],
        }
        for record in records
    ]
    (temp_output / "file_index.json").write_text(
        json.dumps(file_index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    config: dict[str, object] = {
        "dataset": "HUST bearing dataset",
        "protocol": "fixed-speed one-class TSAD; no validation split",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channels": list(CHANNELS),
        "ignored_source_columns": ["Time", "Speed/Tacho1"],
        "fixed_speeds_hz": list(FIXED_SPEEDS_HZ),
        "excluded_condition": "VS_0_40_0Hz",
        "source_points_per_file": SOURCE_POINTS,
        "healthy_split_point": HALF_POINTS,
        "window_length": WINDOW_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "normal_files": len(normal_records),
        "fault_files": len(fault_records),
        "state_to_id": STATE_TO_ID,
        "state_descriptions": STATE_DESCRIPTIONS,
        "scaling": "global channel-wise z-score fitted on normal training points only",
        "sequence_shapes": {
            "train": [train_points, 3],
            "test": [test_points, 3],
        },
        "window_shapes": {
            "train": [train_windows, WINDOW_LENGTH, 3],
            "test": [test_windows, WINDOW_LENGTH, 3],
        },
        "test_point_counts": {
            "normal": test_normal_points,
            "anomaly": test_fault_points,
        },
        "test_window_counts": {
            "normal": test_normal_windows,
            "anomaly": test_fault_windows,
        },
        "test_anomaly_ratio_points": test_fault_points / test_points,
        "test_anomaly_ratio_windows": test_fault_windows / test_windows,
        "boundary_policy": "window each source segment independently; no cross-file windows",
    }
    (temp_output / "dataset_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(temp_output / "README.md", config)

    # Compact verification after reopening the final arrays.
    train_check = np.load(sequence_dir / "train.npy", mmap_mode="r")
    test_check = np.load(sequence_dir / "test.npy", mmap_mode="r")
    test_label_check = np.load(sequence_dir / "test_label.npy", mmap_mode="r")
    train_windows_check = np.load(window_dir / "train.npy", mmap_mode="r")
    test_windows_check = np.load(window_dir / "test.npy", mmap_mode="r")
    test_window_labels_check = np.load(window_dir / "test_label.npy", mmap_mode="r")
    train_scaled_mean = np.asarray(train_check, dtype=np.float64).mean(axis=0)
    train_scaled_std = np.asarray(train_check, dtype=np.float64).std(axis=0)
    qa = {
        "sequence_train_shape": list(train_check.shape),
        "sequence_test_shape": list(test_check.shape),
        "window_train_shape": list(train_windows_check.shape),
        "window_test_shape": list(test_windows_check.shape),
        "test_point_label_counts": {
            "normal": int(np.count_nonzero(test_label_check == 0)),
            "anomaly": int(np.count_nonzero(test_label_check == 1)),
        },
        "test_window_label_counts": {
            "normal": int(np.count_nonzero(test_window_labels_check == 0)),
            "anomaly": int(np.count_nonzero(test_window_labels_check == 1)),
        },
        "standardized_train_mean": train_scaled_mean.tolist(),
        "standardized_train_std": train_scaled_std.tolist(),
        "all_sample_checks_finite": bool(
            np.isfinite(train_check[::4096]).all()
            and np.isfinite(test_check[::4096]).all()
            and np.isfinite(train_windows_check[::511]).all()
            and np.isfinite(test_windows_check[::1023]).all()
        ),
    }
    (temp_output / "qa_report.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )

    temp_output.rename(output)
    print(json.dumps({"output": str(output), "config": config, "qa": qa}, indent=2))


if __name__ == "__main__":
    main()
