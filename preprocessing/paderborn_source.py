#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.io import loadmat


RAW = Path(os.environ["CSGMM_RAW"])
OUT = Path(os.environ["CSGMM_OUTPUT"])
TMP = OUT.parent / (OUT.name + ".extract-tmp")
LENGTH = 2048
HOP = 2048
EXPECTED_FILES_PER_BEARING = 80
# A full-library audit found a minimum parseable high-rate length of 249,940,
# so 122 complete windows are common to every usable source file.
EXPECTED_WINDOWS_PER_FILE = 122
# The official KA08 archive contains one structurally unreadable MATLAB file:
# N15_M01_F10_KA08_2.mat.  Exclude the corresponding condition/repetition from
# every bearing, rather than silently dropping only one anomalous-bearing file.
UNIFORM_EXCLUDED_FILES = {(2, 2)}
USED_FILES_PER_BEARING = EXPECTED_FILES_PER_BEARING - len(UNIFORM_EXCLUDED_FILES)

TRAIN = ["K001", "K002", "K003", "K004"]
VAL = ["K005"]
TEST_NORMAL = ["K006"]

BEARING_INFO = {
    "K001": ("healthy", "normal"), "K002": ("healthy", "normal"),
    "K003": ("healthy", "normal"), "K004": ("healthy", "normal"),
    "K005": ("healthy", "normal"), "K006": ("healthy", "normal"),
    "KA01": ("artificial", "outer"), "KA03": ("artificial", "outer"),
    "KA05": ("artificial", "outer"), "KA06": ("artificial", "outer"),
    "KA07": ("artificial", "outer"), "KA08": ("artificial", "outer"),
    "KA09": ("artificial", "outer"),
    "KI01": ("artificial", "inner"), "KI03": ("artificial", "inner"),
    "KI05": ("artificial", "inner"), "KI07": ("artificial", "inner"),
    "KI08": ("artificial", "inner"),
    "KA04": ("real", "outer"), "KA15": ("real", "outer"),
    "KA16": ("real", "outer"), "KA22": ("real", "outer"),
    "KA30": ("real", "outer"), "KB27": ("real", "outer"),
    "KB23": ("real", "inner"), "KB24": ("real", "inner"),
    "KI04": ("real", "inner"), "KI14": ("real", "inner"),
    "KI16": ("real", "inner"), "KI17": ("real", "inner"),
    "KI18": ("real", "inner"), "KI21": ("real", "inner"),
}

CONDITION_MAP = {
    "N15_M07_F10": 0,
    "N09_M07_F10": 1,
    "N15_M01_F10": 2,
    "N15_M07_F04": 3,
}
CONDITION_INFO = {
    0: {"name": "N15_M07_F10", "speed_rpm": 1500, "torque_nm": 0.7, "radial_force_n": 1000},
    1: {"name": "N09_M07_F10", "speed_rpm": 900, "torque_nm": 0.7, "radial_force_n": 1000},
    2: {"name": "N15_M01_F10", "speed_rpm": 1500, "torque_nm": 0.1, "radial_force_n": 1000},
    3: {"name": "N15_M07_F04", "speed_rpm": 1500, "torque_nm": 0.7, "radial_force_n": 400},
}

FAULT_TYPE = {
    ("healthy", "normal"): 0,
    ("artificial", "outer"): 1,
    ("artificial", "inner"): 2,
    ("real", "outer"): 3,
    ("real", "inner"): 4,
}


def extract_archive(code: str) -> Path:
    archive = RAW / f"{code}.rar"
    target = TMP / code
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(["unrar", "x", "-o+", "-inul", str(archive), str(target) + "/"], check=True)
    mats = sorted(target.rglob("*.mat"))
    if len(mats) != EXPECTED_FILES_PER_BEARING:
        raise RuntimeError(f"{code}: expected 80 MAT files, found {len(mats)}")
    return target


def load_high_rate_channels(path: Path) -> np.ndarray:
    obj = loadmat(path, simplify_cells=True, variable_names=[path.stem])[path.stem]
    channels = {str(item["Name"]): np.asarray(item["Data"], dtype=np.float64) for item in obj["Y"]}
    names = ["phase_current_1", "phase_current_2", "vibration_1"]
    missing = [name for name in names if name not in channels]
    if missing:
        raise RuntimeError(f"{path}: missing channels {missing}")
    sizes = [channels[name].size for name in names]
    if len(set(sizes)) != 1:
        raise RuntimeError(f"{path}: channel length mismatch {sizes}")
    return np.column_stack([channels[name] for name in names])


def parse_file(path: Path) -> tuple[int, int]:
    stem = path.stem
    condition_key = "_".join(stem.split("_")[:3])
    if condition_key not in CONDITION_MAP:
        raise RuntimeError(f"unknown condition in {stem}")
    match = re.search(r"_(\d+)$", stem)
    if not match:
        raise RuntimeError(f"cannot parse repetition from {stem}")
    return CONDITION_MAP[condition_key], int(match.group(1))


def included_mats(target: Path) -> list[Path]:
    mats = []
    for path in sorted(target.rglob("*.mat")):
        if parse_file(path) not in UNIFORM_EXCLUDED_FILES:
            mats.append(path)
    if len(mats) != USED_FILES_PER_BEARING:
        raise RuntimeError(f"{target}: expected {USED_FILES_PER_BEARING} included MAT files, found {len(mats)}")
    return mats


def compute_train_stats() -> tuple[np.ndarray, np.ndarray, int]:
    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    for code in TRAIN:
        target = extract_archive(code)
        for mat in included_mats(target):
            x = load_high_rate_channels(mat)
            usable = EXPECTED_WINDOWS_PER_FILE * LENGTH
            if x.shape[0] < usable:
                raise RuntimeError(f"{mat}: requires {usable} samples, found {x.shape[0]}")
            x = x[:usable]
            count += x.shape[0]
            total += x.sum(axis=0)
            total_sq += np.square(x).sum(axis=0)
        shutil.rmtree(target)
    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return mean, np.sqrt(var), count


def split_for(code: str) -> str:
    if code in TRAIN:
        return "train"
    if code in VAL:
        return "val"
    return "test"


def main() -> None:
    started = time.time()
    if OUT.exists():
        raise FileExistsError(f"output already exists: {OUT}")
    archives = sorted(p.stem for p in RAW.glob("*.rar"))
    if set(archives) != set(BEARING_INFO):
        raise RuntimeError({"missing_info": sorted(set(archives) - set(BEARING_INFO)), "missing_archive": sorted(set(BEARING_INFO) - set(archives))})
    OUT.mkdir(parents=True)
    TMP.mkdir(parents=True, exist_ok=True)

    mean, std, stat_count = compute_train_stats()
    (OUT / "normalization.json").write_text(json.dumps({
        "source": "training bearings only",
        "train_bearings": TRAIN,
        "channel_order": ["phase_current_1", "phase_current_2", "vibration_1"],
        "mean": mean.tolist(), "std": std.tolist(), "sample_count": stat_count,
    }, indent=2), encoding="utf-8")

    split_codes = {"train": TRAIN, "val": VAL, "test": TEST_NORMAL + [c for c in archives if c not in TRAIN + VAL + TEST_NORMAL]}
    rows_per_bearing = USED_FILES_PER_BEARING * EXPECTED_WINDOWS_PER_FILE
    arrays = {}
    offsets = {k: 0 for k in split_codes}
    metadata_names = ["label", "fault_type", "condition", "bearing", "file_id", "repetition", "window_in_file"]
    for split, codes in split_codes.items():
        n = len(codes) * rows_per_bearing
        arrays[(split, "x")] = open_memmap(OUT / f"{split}_x.npy", mode="w+", dtype=np.float16, shape=(n, LENGTH, 3))
        for name in metadata_names:
            dtype = np.int16 if name in {"condition", "bearing", "repetition", "window_in_file", "fault_type", "label"} else np.int32
            arrays[(split, name)] = open_memmap(OUT / f"{split}_{name}.npy", mode="w+", dtype=dtype, shape=(n,))

    bearing_codes = TRAIN + VAL + TEST_NORMAL + [c for c in archives if c not in TRAIN + VAL + TEST_NORMAL]
    bearing_to_id = {c: i for i, c in enumerate(bearing_codes)}
    file_manifest = []
    distribution = []
    file_id = 0

    for code in bearing_codes:
        split = split_for(code)
        origin, location = BEARING_INFO[code]
        label = int(origin != "healthy")
        ftype = FAULT_TYPE[(origin, location)]
        target = extract_archive(code)
        condition_counts = {i: 0 for i in CONDITION_MAP.values()}
        for mat in included_mats(target):
            condition, repetition = parse_file(mat)
            raw = load_high_rate_channels(mat)
            raw_sample_count = int(raw.shape[0])
            nwin = EXPECTED_WINDOWS_PER_FILE
            usable = nwin * LENGTH
            if raw_sample_count < usable:
                raise RuntimeError(f"{mat}: requires {usable} samples, found {raw_sample_count}")
            raw = (raw[:usable] - mean) / std
            windows = raw.reshape(nwin, LENGTH, 3).astype(np.float16)
            start = offsets[split]
            end = start + nwin
            arrays[(split, "x")][start:end] = windows
            values = {
                "label": label, "fault_type": ftype, "condition": condition,
                "bearing": bearing_to_id[code], "file_id": file_id,
                "repetition": repetition,
            }
            for name, value in values.items():
                arrays[(split, name)][start:end] = value
            arrays[(split, "window_in_file")][start:end] = np.arange(nwin, dtype=np.int16)
            file_manifest.append({
                "file_id": file_id, "split": split, "bearing_code": code,
                "bearing_id": bearing_to_id[code], "origin": origin, "fault_location": location,
                "label": label, "fault_type": ftype, "condition_id": condition,
                "condition_name": CONDITION_INFO[condition]["name"], "repetition": repetition,
                "source_mat": mat.name, "raw_samples": raw_sample_count, "window_count": nwin,
            })
            offsets[split] = end
            condition_counts[condition] += nwin
            file_id += 1
        shutil.rmtree(target)
        for condition, count in condition_counts.items():
            distribution.append({
                "split": split, "bearing_code": code, "bearing_id": bearing_to_id[code],
                "origin": origin, "fault_location": location, "label": label, "fault_type": ftype,
                "condition_id": condition, **CONDITION_INFO[condition], "window_count": count,
            })

    for split, codes in split_codes.items():
        expected = len(codes) * rows_per_bearing
        if offsets[split] != expected:
            raise RuntimeError(f"{split}: wrote {offsets[split]}, expected {expected}")
    for arr in arrays.values():
        arr.flush()

    with (OUT / "file_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(file_manifest[0]))
        writer.writeheader(); writer.writerows(file_manifest)
    with (OUT / "distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(distribution[0]))
        writer.writeheader(); writer.writerows(distribution)

    protocol = {
        "dataset": "Paderborn University KAt Bearing Dataset",
        "version": "bearingwise_v1_20260718",
        "window_length": LENGTH, "window_hop": HOP, "overlap_fraction": 0.0,
        "windows_per_source_file": EXPECTED_WINDOWS_PER_FILE,
        "tail_policy": "keep the first 122 complete windows; discard the remaining raw tail without padding",
        "source_file_policy": "uniformly exclude condition N15_M01_F10 repetition 2 from every bearing because the corresponding KA08 MATLAB structure is unreadable",
        "source_files_used_per_bearing": USED_FILES_PER_BEARING,
        "uniform_excluded_condition_repetition": [[2, 2]],
        "channels": 3, "channel_order": ["phase_current_1", "phase_current_2", "vibration_1"],
        "train_bearings": TRAIN, "validation_bearings": VAL,
        "test_normal_bearings": TEST_NORMAL,
        "test_anomaly_bearings": [c for c in bearing_codes if BEARING_INFO[c][0] != "healthy"],
        "positive_class": "damaged bearing windows (label=1)",
        "negative_class": "healthy bearing windows (label=0)",
        "fault_type_map": {"0": "healthy", "1": "artificial_outer", "2": "artificial_inner", "3": "real_outer", "4": "real_inner"},
        "condition_map": CONDITION_INFO,
        "normalization": "global channel-wise z-score fitted on K001-K004 only",
        "split_unit": "physical bearing before window extraction",
        "rows": {split: offsets[split] for split in offsets},
        "bearing_codes": bearing_codes,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    audit = {"completed": True, "elapsed_seconds": time.time() - started, "checks": {}}
    checks = audit["checks"]
    checks["archive_count"] = len(archives)
    checks["file_count"] = len(file_manifest)
    checks["train_shape"] = list(np.load(OUT / "train_x.npy", mmap_mode="r").shape)
    checks["val_shape"] = list(np.load(OUT / "val_x.npy", mmap_mode="r").shape)
    checks["test_shape"] = list(np.load(OUT / "test_x.npy", mmap_mode="r").shape)
    checks["test_normal"] = int((np.load(OUT / "test_label.npy", mmap_mode="r") == 0).sum())
    checks["test_anomaly"] = int((np.load(OUT / "test_label.npy", mmap_mode="r") == 1).sum())
    checks["bearing_disjoint"] = not (set(TRAIN) & set(VAL + TEST_NORMAL)) and not (set(VAL) & set(TEST_NORMAL))
    checks["no_overlap"] = HOP >= LENGTH
    sample = np.asarray(np.load(OUT / "train_x.npy", mmap_mode="r")[:2048], dtype=np.float32)
    checks["finite_sample"] = bool(np.isfinite(sample).all())
    checks["sample_channel_mean"] = sample.mean(axis=(0, 1)).tolist()
    checks["sample_channel_std"] = sample.std(axis=(0, 1)).tolist()
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if TMP.exists():
        shutil.rmtree(TMP)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
