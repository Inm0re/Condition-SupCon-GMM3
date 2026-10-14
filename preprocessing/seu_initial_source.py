from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

RAW = Path(os.environ["CSGMM_RAW"])
OUT = Path(os.environ["CSGMM_OUTPUT"])
L = 2048
H = 2048
USECOLS = (0, 1, 2, 3, 5, 6, 7)
CHANNELS = [
    "motor_vibration",
    "planetary_gearbox_x",
    "planetary_gearbox_y",
    "planetary_gearbox_z",
    "parallel_gearbox_x",
    "parallel_gearbox_y",
    "parallel_gearbox_z",
]
FILES = [
    ("health_20_0.csv", 0, 0, "20Hz_0V"),
    ("health_30_2.csv", 0, 1, "30Hz_2V"),
    ("ball_20_0.csv", 1, 0, "20Hz_0V"),
    ("ball_30_2.csv", 1, 1, "30Hz_2V"),
    ("comb_20_0.csv", 2, 0, "20Hz_0V"),
    ("comb_30_2.csv", 2, 1, "30Hz_2V"),
    ("inner_20_0.csv", 3, 0, "20Hz_0V"),
    ("inner_30_2.csv", 3, 1, "30Hz_2V"),
    ("outer_20_0.csv", 4, 0, "20Hz_0V"),
    ("outer_30_2.csv", 4, 1, "30Hz_2V"),
]
FAULT_NAMES = {0: "healthy", 1: "ball", 2: "compound", 3: "inner_race", 4: "outer_race"}

def load_signal(path: Path) -> np.ndarray:
    with path.open("r", errors="replace") as f:
        for _ in range(16):
            next(f)
        first_numeric = next(f)
    delimiter = "," if first_numeric.count(",") > first_numeric.count(chr(9)) else chr(9)
    x = np.loadtxt(path, delimiter=delimiter, skiprows=16, usecols=USECOLS, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(USECOLS):
        raise RuntimeError(f"unexpected shape {x.shape} in {path}")
    if not np.isfinite(x).all():
        raise RuntimeError(f"non-finite values in {path}")
    return x

def windows(x: np.ndarray) -> np.ndarray:
    n = x.shape[0] // L
    return x[: n * L].reshape(n, L, x.shape[1])

def normalize_inplace(mm: np.memmap, mean: np.ndarray, std: np.ndarray, chunk: int = 64) -> None:
    for i in range(0, len(mm), chunk):
        j = min(i + chunk, len(mm))
        mm[i:j] = (mm[i:j] - mean) / std
    mm.flush()

def save_meta(prefix: str, rows: list[dict]) -> None:
    np.save(OUT / f"{prefix}_label.npy", np.asarray([r["label"] for r in rows], dtype=np.int8))
    np.save(OUT / f"{prefix}_fault_type.npy", np.asarray([r["fault_type"] for r in rows], dtype=np.int8))
    np.save(OUT / f"{prefix}_condition.npy", np.asarray([r["condition"] for r in rows], dtype=np.int8))
    np.save(OUT / f"{prefix}_file_id.npy", np.asarray([r["file_id"] for r in rows], dtype=np.int16))
    np.save(OUT / f"{prefix}_window_in_file.npy", np.asarray([r["window_in_file"] for r in rows], dtype=np.int32))
    np.save(OUT / f"{prefix}_start.npy", np.asarray([r["start"] for r in rows], dtype=np.int64))

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)

    manifest = []
    file_info = []
    train_count = 0
    test_count = 0
    loaded_cache: dict[str, np.ndarray] = {}

    for file_id, (name, fault, condition, condition_name) in enumerate(FILES):
        x = load_signal(RAW / name)
        nwin = x.shape[0] // L
        if nwin != 511:
            raise RuntimeError(f"{name}: expected 511 windows, got {nwin}")
        file_info.append((file_id, name, fault, condition, condition_name, x.shape[0], nwin))
        if fault == 0:
            train_count += nwin // 2
            test_count += nwin - (nwin // 2) - 1
            loaded_cache[name] = x
        else:
            test_count += nwin
        manifest.append({
            "file_id": file_id,
            "file": name,
            "fault_type": fault,
            "fault_name": FAULT_NAMES[fault],
            "condition": condition,
            "condition_name": condition_name,
            "raw_rows": int(x.shape[0]),
            "complete_windows": int(nwin),
        })

    train_x = open_memmap(OUT / "train_x.npy", mode="w+", dtype=np.float32,
                          shape=(train_count, L, len(CHANNELS)))
    train_meta: list[dict] = []
    pos = 0
    split_audit = []
    for file_id, name, fault, condition, condition_name, raw_rows, nwin in file_info:
        if fault != 0:
            continue
        w = windows(loaded_cache.pop(name))
        half = nwin // 2
        take = w[:half]
        train_x[pos:pos + half] = take
        for k in range(half):
            train_meta.append({
                "label": 0, "fault_type": 0, "condition": condition,
                "file_id": file_id, "window_in_file": k, "start": k * H,
            })
        split_audit.append({
            "file": name,
            "total_windows": nwin,
            "train_windows": half,
            "discarded_guard_window": half,
            "test_normal_windows": nwin - half - 1,
            "last_train_end_exclusive": half * L,
            "first_test_start": (half + 1) * H,
            "guard_samples": L,
        })
        pos += half
    train_x.flush()
    if pos != train_count:
        raise RuntimeError((pos, train_count))

    mean = np.asarray(train_x.mean(axis=(0, 1), dtype=np.float64))
    sq_mean = np.asarray(np.square(train_x, dtype=np.float64).mean(axis=(0, 1), dtype=np.float64))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 1e-12))
    raw_train_corr = np.corrcoef(np.asarray(train_x).reshape(-1, len(CHANNELS)), rowvar=False)
    normalize_inplace(train_x, mean, std)

    test_x = open_memmap(OUT / "test_x.npy", mode="w+", dtype=np.float32,
                         shape=(test_count, L, len(CHANNELS)))
    test_meta: list[dict] = []
    pos = 0
    for file_id, name, fault, condition, condition_name, raw_rows, nwin in file_info:
        x = load_signal(RAW / name)
        w = windows(x)
        if fault == 0:
            idxs = range((nwin // 2) + 1, nwin)
        else:
            idxs = range(nwin)
        for k in idxs:
            test_x[pos] = (w[k] - mean) / std
            test_meta.append({
                "label": int(fault != 0), "fault_type": fault, "condition": condition,
                "file_id": file_id, "window_in_file": k, "start": k * H,
            })
            pos += 1
    test_x.flush()
    if pos != test_count:
        raise RuntimeError((pos, test_count))

    save_meta("train", train_meta)
    save_meta("test", test_meta)

    for alias, target in [("train.npy", "train_x.npy"), ("test.npy", "test_x.npy")]:
        os.symlink(target, OUT / alias)

    with (OUT / "file_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    norm = {
        "source": "training-normal windows only",
        "channel_order": CHANNELS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sample_count": int(train_count * L),
    }
    (OUT / "normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")

    test_labels = np.load(OUT / "test_label.npy")
    test_fault = np.load(OUT / "test_fault_type.npy")
    train_post = np.load(OUT / "train_x.npy", mmap_mode="r")
    test_post = np.load(OUT / "test_x.npy", mmap_mode="r")
    post_mean = np.asarray(train_post.mean(axis=(0, 1), dtype=np.float64))
    post_std = np.asarray(train_post.std(axis=(0, 1), dtype=np.float64))
    off_diag = raw_train_corr[~np.eye(len(CHANNELS), dtype=bool)]

    protocol = {
        "dataset": "Southeast University Drivetrain Dynamics Simulator bearing subset",
        "version": "fixed_main_v1_20260719",
        "source_repository": "https://github.com/cathysiyu/Mechanical-datasets",
        "task": "one-class fixed-fault anomaly detection; not run-to-failure",
        "window_length": L,
        "window_hop": H,
        "overlap_fraction": 0.0,
        "tail_policy": "discard incomplete tail; no padding",
        "channels": len(CHANNELS),
        "channel_order": CHANNELS,
        "excluded_channel": "Channel5 motor torque",
        "normal_split": "within each healthy condition: first 255 windows train, middle window discarded as guard, final 255 windows normal test",
        "test_anomaly": "all complete windows from all 8 faulty files",
        "normalization": "global channel-wise z-score fitted on train-normal windows only",
        "condition_map": {"0": "20Hz_0V", "1": "30Hz_2V"},
        "fault_type_map": {str(k): v for k, v in FAULT_NAMES.items()},
        "counts": {
            "train_normal": train_count,
            "test_normal": int((test_labels == 0).sum()),
            "test_anomaly": int((test_labels == 1).sum()),
            "test_total": test_count,
        },
        "split_audit": split_audit,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    audit = {
        "completed": True,
        "train_shape": list(train_post.shape),
        "test_shape": list(test_post.shape),
        "train_finite": bool(np.isfinite(train_post).all()),
        "test_finite": bool(np.isfinite(test_post).all()),
        "train_labels_all_normal": bool((np.load(OUT / "train_label.npy") == 0).all()),
        "test_counts_by_fault": {
            FAULT_NAMES[int(k)]: int((test_fault == k).sum()) for k in np.unique(test_fault)
        },
        "post_normalization_train_mean": post_mean.tolist(),
        "post_normalization_train_std": post_std.tolist(),
        "raw_train_channel_correlation": raw_train_corr.tolist(),
        "max_abs_offdiagonal_channel_correlation": float(np.max(np.abs(off_diag))),
        "distinct_real_vibration_channels": bool(np.max(np.abs(off_diag)) < 0.999999),
        "normal_train_test_sample_overlap": False,
        "normal_guard_samples": L,
    }
    (OUT / "DATASET_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT / "COMPLETE").write_text("dataset build completed\n", encoding="utf-8")
    print(json.dumps({"protocol": protocol, "audit": audit}, indent=2))

if __name__ == "__main__":
    main()
