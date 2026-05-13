# Code from emg_lstm_v8 formatted into typical python file for data extraction
from __future__ import annotations

import re
import random
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessingConfig:
    """Parameters that control windowing and trimming."""

    # Windowing
    window_size: int = 600
    step_size: int = 600

    # Windows sampled per phase
    rest_windows_per_phase: int = 1
    gesture_windows_per_phase: int = 2

    # Dataset timing
    sample_rate_hz: int = 500
    phase_seconds: int = 3

    # Seconds trimmed from each phase boundary to avoid transition artefacts
    boundary_trim_seconds: float = 0.5

    # Phase-offset search to align the repeating 3-second gesture pattern
    phase_offset_search_step: int = 25
    phase_offset_search_max_abs_seconds: float = 1.0

    # Columns to discard (non-EMG sensors, trigger lines, etc.)
    drop_columns: frozenset[str] = field(default_factory=lambda: frozenset({
        "GyroX", "GyroY", "GyroZ",
        "AccX", "AccY", "AccZ",
        "PPG1", "PPG2",
        "rawPPG1", "rawPPG2", "rawPPG3",
        "Hr", "Hrv",
        "Battery",
        "Trigger", "PhysicalTrigger", "AutoTrigger",
    }))

    # Gesture labels to exclude entirely
    excluded_gestures: frozenset[str] = field(default_factory=lambda: frozenset({"2F"}))

    sort_by_time: bool = True


def _load_emg_file(path: Path) -> pd.DataFrame:
    """Read a tab-separated EMG CSV into a DataFrame."""
    return pd.read_csv(path, sep="\t", engine="python")


def _normalize_gesture_label(raw_label: str) -> str:
    """Normalise file-derived gesture labels (e.g. 'rest_3s' -> 'Rest')."""
    label = str(raw_label).strip()
    return "Rest" if label.lower().startswith("rest") else label


def _parse_subject_and_gesture(path: Path) -> tuple[str, str]:
    subject_id = path.parent.name
    parts = path.stem.split("_")

    if len(parts) < 2:
        raise ValueError(f"Cannot parse gesture from filename: {path.name}")

    gesture_parts = parts[1:]
    if len(gesture_parts) >= 2 and gesture_parts[-1].lower() == "3s":
        gesture_parts = gesture_parts[:-1]

    if not gesture_parts:
        raise ValueError(f"Cannot parse gesture from filename: {path.name}")

    gesture_label = _normalize_gesture_label("_".join(gesture_parts))
    return subject_id, gesture_label


def _infer_filtered_channel_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted list of columns whose names start with 'filteredchannel'."""
    cols = [c for c in df.columns if c.lower().startswith("filteredchannel")]

    def _sort_key(name: str) -> float:
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else float("inf")

    cols = sorted(cols, key=_sort_key)
    if not cols:
        raise ValueError(
            "No filtered EMG columns found. "
            "Expected columns like FilteredChannel1, FilteredChannel2, …"
        )
    return cols


def _extract_windows(
    signal: np.ndarray,
    phase_samples: int,
    boundary_trim: int,
    window_size: int,
    windows_per_phase: int,
    rng: np.random.Generator,
    phase_offset: int = 0,
) -> list[np.ndarray]:
    n_samples = signal.shape[0]
    n_phases = (n_samples - phase_offset) // phase_samples
    windows = []

    for p in range(n_phases):
        phase_start = phase_offset + p * phase_samples + boundary_trim
        phase_end = min(phase_offset + (p + 1) * phase_samples - boundary_trim, n_samples)

        usable = phase_end - phase_start - window_size
        if usable < 0:
            continue  # phase too short after trimming

        starts = rng.choice(
            np.arange(0, usable + 1),
            size=min(windows_per_phase, usable + 1),
            replace=False,
        )
        for s in starts:
            win_start = phase_start + s
            windows.append(signal[win_start: win_start + window_size])

    return windows


# ---------------------------------------------------------------------------
# Core function 1 — subject-level split
# ---------------------------------------------------------------------------

def split_files_by_subject(
    root_dir: str | Path,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int | None = None,
    cfg: PreprocessingConfig | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if cfg is None:
        cfg = PreprocessingConfig()

    if random_state is None:
        random_state = random.randint(0, 2**31 - 1)

    root_dir = Path(root_dir)
    all_files = sorted(root_dir.rglob("*.csv"))

    # Filter excluded gestures and build subject → [files] map
    subject_to_files: dict[str, list[Path]] = {}
    for f in all_files:
        try:
            subject_id, gesture_label = _parse_subject_and_gesture(f)
        except ValueError:
            continue
        if gesture_label in cfg.excluded_gestures:
            continue
        subject_to_files.setdefault(subject_id, []).append(f)

    unique_subjects = np.array(sorted(subject_to_files.keys()))

    if len(unique_subjects) < 3:
        raise ValueError(
            f"Only {len(unique_subjects)} subject(s) found; "
            "need at least 3 to create train/val/test splits."
        )

    train_subj, temp_subj = train_test_split(
        unique_subjects,
        test_size=1.0 - train_size,
        random_state=random_state,
        shuffle=True,
    )
    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
        shuffle=True,
    )

    train_set, val_set, test_set = set(train_subj), set(val_subj), set(test_subj)

    train_files = [f for s in train_set for f in subject_to_files[s]]
    val_files   = [f for s in val_set   for f in subject_to_files[s]]
    test_files  = [f for s in test_set  for f in subject_to_files[s]]

    print(f"Subjects  — train: {len(train_set):3d} | val: {len(val_set):3d} | test: {len(test_set):3d}")
    print(f"Files     — train: {len(train_files):3d} | val: {len(val_files):3d} | test: {len(test_files):3d}")

    return sorted(train_files), sorted(val_files), sorted(test_files)


# ---------------------------------------------------------------------------
# Core function 2 — windowed preprocessing
# ---------------------------------------------------------------------------

def preprocess_files(
    file_list: list[str | Path],
    cfg: PreprocessingConfig | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg is None:
        cfg = PreprocessingConfig()

    if random_state is None:
        random_state = random.randint(0, 2**31 - 1)

    rng = np.random.default_rng(random_state)

    phase_samples    = cfg.sample_rate_hz * cfg.phase_seconds
    boundary_trim    = int(cfg.boundary_trim_seconds * cfg.sample_rate_hz)
    max_offset_samps = int(cfg.phase_offset_search_max_abs_seconds * cfg.sample_rate_hz)

    all_X: list[np.ndarray] = []
    all_y: list[str]        = []
    all_s: list[str]        = []

    for path in file_list:
        path = Path(path)
        try:
            subject_id, gesture_label = _parse_subject_and_gesture(path)
        except ValueError as e:
            print(f"[WARN] Skipping {path.name}: {e}")
            continue

        # ── Load & clean ────────────────────────────────────────────────────
        df = _load_emg_file(path)

        cols_to_drop = [c for c in cfg.drop_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        channel_cols = _infer_filtered_channel_cols(df)

        keep_cols = channel_cols.copy()
        if "Timestamp" in df.columns:
            df = df.rename(columns={"Timestamp": "time"})
            keep_cols = ["time"] + keep_cols
        else:
            df["time"] = np.arange(len(df), dtype=np.int64)
            keep_cols = ["time"] + keep_cols

        df = df[keep_cols].copy()

        if cfg.sort_by_time:
            df = df.sort_values("time", kind="mergesort").reset_index(drop=True)

        signal = df[channel_cols].to_numpy(dtype=np.float32)

        search_rng = np.random.default_rng(random_state)
        best_offset = 0
        best_var    = -1.0
        for offset in range(0, max_offset_samps, cfg.phase_offset_search_step):
            test_wins = _extract_windows(
                signal, phase_samples, boundary_trim,
                cfg.window_size, 1, search_rng, phase_offset=offset,
            )
            if test_wins:
                v = float(np.var(np.stack(test_wins)))
                if v > best_var:
                    best_var    = v
                    best_offset = offset

        # ── Window extraction ────────────────────────────────────────────────
        windows_per_phase = (
            cfg.rest_windows_per_phase
            if gesture_label == "Rest"
            else cfg.gesture_windows_per_phase
        )

        windows = _extract_windows(
            signal, phase_samples, boundary_trim,
            cfg.window_size, windows_per_phase, rng, phase_offset=best_offset,
        )

        for w in windows:
            all_X.append(w)
            all_y.append(gesture_label)
            all_s.append(subject_id)

    if not all_X:
        raise RuntimeError("No windows were extracted. Check file paths and config.")

    X        = np.stack(all_X, axis=0).astype(np.float32)   # (N, T, C)
    y        = np.array(all_y,  dtype=object)
    subjects = np.array(all_s,  dtype=object)

    print(f"Extracted {X.shape[0]} windows | shape: {X.shape}")
    print(f"Gesture distribution:\n{pd.Series(y).value_counts().sort_index().to_string()}")

    return X, y, subjects


def rebalance_training_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rest_multiplier: float = 2.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample Rest to *rest_multiplier* × median(non-rest class count).
    All other classes are kept in full.
    """
    if random_state is None:
        random_state = random.randint(0, 2**31 - 1)

    rng = np.random.default_rng(random_state)
    classes = sorted(np.unique(y_train).tolist())
    class_to_idx = {c: np.where(y_train == c)[0] for c in classes}

    non_rest = [c for c in classes if c != "Rest"]
    if not non_rest:
        return X_train, y_train

    median_count = int(np.median([len(class_to_idx[c]) for c in non_rest]))
    median_count = max(median_count, 1)

    selected: list[np.ndarray] = []
    for cls in classes:
        idxs = class_to_idx[cls]
        if cls == "Rest":
            target = min(int(round(rest_multiplier * median_count)), len(idxs))
            chosen = rng.choice(idxs, size=target, replace=False)
        else:
            chosen = idxs
        selected.append(chosen)

    idx = np.concatenate(selected)
    rng.shuffle(idx)
    return X_train[idx], y_train[idx]


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    root_dir: str | Path,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    rest_multiplier: float = 2.0,
    random_state: int = 42,
    cfg: PreprocessingConfig | None = None,
) -> dict[str, np.ndarray]:
    if cfg is None:
        cfg = PreprocessingConfig()

    # 1. Subject-level file split
    train_files, val_files, test_files = split_files_by_subject(
        root_dir=root_dir,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        cfg=cfg,
    )

    # 2. Preprocess each split into windowed arrays
    X_train, y_train, s_train = preprocess_files(train_files, cfg=cfg, random_state=random_state)
    X_val,   y_val,   s_val   = preprocess_files(val_files,   cfg=cfg, random_state=random_state)
    X_test,  y_test,  s_test  = preprocess_files(test_files,  cfg=cfg, random_state=random_state)

    # 3. Rebalance training split
    X_train, y_train = rebalance_training_split(
        X_train, y_train,
        rest_multiplier=rest_multiplier,
        random_state=random_state,
    )

    print("\n── Final shapes ──────────────────────────────")
    print(f"Train : X={X_train.shape}  y={y_train.shape}")
    print(f"Val   : X={X_val.shape}  y={y_val.shape}")
    print(f"Test  : X={X_test.shape}  y={y_test.shape}")

    return {
        "X_train": X_train, "y_train": y_train, "s_train": s_train,
        "X_val":   X_val,   "y_val":   y_val,   "s_val":   s_val,
        "X_test":  X_test,  "y_test":  y_test,  "s_test":  s_test,
        "train_files": train_files,
        "val_files":   val_files,
        "test_files":  test_files,
    }


if __name__ == "__main__":
    dataset = build_dataset(root_dir="./subject_data")
    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val,   y_val   = dataset["X_val"],   dataset["y_val"]
    X_test,  y_test  = dataset["X_test"],  dataset["y_test"]