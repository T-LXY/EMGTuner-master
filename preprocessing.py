from __future__ import annotations

import os
import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PreprocessingConfig:
    """
    Stores configuration parameters for preprocessing and window creation.
    """

    # Windowing parameters
    window_size: int = 150
    step_size: int = 25

    # Number of windows to extract from each cleaned phase
    rest_windows_per_phase: int = 2
    gesture_windows_per_phase: int = 3

    # Dataset timing
    sample_rate_hz: int = 500
    phase_seconds: int = 3

    # Trim this many seconds from the start and end of each phase to avoid transition artifacts
    boundary_trim_seconds: float = 0.4

    # Search a small positive offset at the beginning of each file to better align the repeating 3s gesture pattern
    phase_offset_search_step: int = 25        
    phase_offset_search_max_abs_seconds: float = 1.0

    # Keep rows ordered by time
    sort_by_time: bool = True


cfg = PreprocessingConfig()

# Calculate derived parameters
phase_samples = cfg.sample_rate_hz * cfg.phase_seconds
boundary_trim_samples = int(cfg.boundary_trim_seconds * cfg.sample_rate_hz)

# Define the root directory containing the CSV files
root_dir = Path("/Users/tonyliu/Documents/EMGTuner-master/subject_data")

# Recursively find all CSV files in the root directory and its subdirectories
DATA_FILES = sorted([file for file in root_dir.rglob("*.csv")])

# Define columns and gestures to exclude from the dataset
DROP_COLUMNS = {
    "GyroX", "GyroY", "GyroZ",
    "AccX", "AccY", "AccZ",
    "PPG1", "PPG2",
    "rawPPG1", "rawPPG2", "rawPPG3",
    "Hr", "Hrv",
    "Battery",
    "Trigger", "PhysicalTrigger", "AutoTrigger"
}
EXCLUDED_GESTURES = {"2F", "3F"}

# Print out the configuration and dataset information for verification
print("Number of files found:", len(DATA_FILES))
if len(DATA_FILES) > 0:
    print("Example file:", DATA_FILES[0])
print("Samples per 3-second phase:", phase_samples)
print("Boundary trim per side:", boundary_trim_samples, "samples")

def load_emg_file(path: str | Path) -> pd.DataFrame:
    """
    Load one EMG CSV file from the dataset.
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t", engine="python")
    return df


def normalize_gesture_label(raw_label: str) -> str:
    """
    Normalize file-derived gesture labels.
    """
    label = str(raw_label).strip()
    if label.lower().startswith("rest"):
        return "Rest"
    return label


def parse_subject_and_gesture(path: str | Path) -> tuple[str, str]:
    """
    Parse subject ID and gesture label from the file path.
    """
    path = Path(path)

    subject_id = path.parent.name
    stem = path.stem

    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Could not parse gesture from filename: {path.name}")

    gesture_parts = parts[1:]

    if len(gesture_parts) >= 2 and gesture_parts[-1].lower() == "3s":
        gesture_parts = gesture_parts[:-1]

    if len(gesture_parts) == 0:
        raise ValueError(f"Could not parse gesture from filename: {path.name}")

    gesture_label = "_".join(gesture_parts)
    gesture_label = normalize_gesture_label(gesture_label)

    return subject_id, gesture_label


def infer_filtered_channel_cols(df: pd.DataFrame) -> list[str]:
    """
    Dynamically find filtered EMG columns
    """
    filtered_cols = [col for col in df.columns if col.lower().startswith("filteredchannel")]

    def sort_key(name: str):
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else float("inf")

    filtered_cols = sorted(filtered_cols, key=sort_key)

    if len(filtered_cols) == 0:
        raise ValueError(
            "No filtered EMG columns found. Expected columns like FilteredChannel1, FilteredChannel2, ..."
        )

    return filtered_cols


all_dfs = []
schemas = []

for file in DATA_FILES:
    df = load_emg_file(file)
    subject_id, file_gesture_label = parse_subject_and_gesture(file)

    if file_gesture_label in EXCLUDED_GESTURES:
        continue

    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    if len(cols_to_drop) > 0:
        df = df.drop(columns=cols_to_drop)

    channel_cols = infer_filtered_channel_cols(df)

    keep_cols = channel_cols.copy()
    if "Timestamp" in df.columns:
        keep_cols = ["Timestamp"] + keep_cols

    df = df[keep_cols].copy()

    if "Timestamp" in df.columns:
        df = df.rename(columns={"Timestamp": "time"})
    else:
        df["time"] = np.arange(len(df), dtype=np.int64)

    df = df[["time"] + channel_cols].copy()

    df["source_file"] = str(file)
    df["subject_id"] = subject_id
    df["file_gesture_label"] = file_gesture_label
    df["source_index"] = np.arange(len(df), dtype=np.int64)

    if cfg.sort_by_time and "time" in df.columns:
        df = df.sort_values("time", kind="mergesort").reset_index(drop=True)

    all_dfs.append(df)

    schemas.append({
        "file": str(file),
        "subject_id": subject_id,
        "file_gesture_label": file_gesture_label,
        "num_rows": len(df),
        "n_channels": len(channel_cols),
        "channels": channel_cols,
        "time_dtype": str(df["time"].dtype),
    })

raw_df = pd.concat(all_dfs, ignore_index=True)
schemas_df = pd.DataFrame(schemas)

channel_cols = infer_filtered_channel_cols(raw_df)

print("raw_df shape:", raw_df.shape)
print("Detected filtered EMG channels:", channel_cols)
print("Unique file-level labels:", sorted(raw_df["file_gesture_label"].unique().tolist()))
display(schemas_df.head())
display(raw_df.head())

filtered_df = raw_df.copy()

print("Using pre-filtered EMG channels from the files.")
print("Using known device sampling rate:", cfg.sample_rate_hz, "Hz")
print("Samples per 3-second phase:", phase_samples)
print("Boundary trim per side:", boundary_trim_samples, "samples")

def get_phase_label(file_gesture_label: str, phase_index: int) -> str:
    """
    Determine the label of a 3-second phase inside a file.
    """
    file_gesture_label = normalize_gesture_label(file_gesture_label)

    if file_gesture_label == "Rest":
        return "Rest"

    return "Rest" if phase_index % 2 == 0 else file_gesture_label


def compute_emg_energy(segment: np.ndarray) -> float:
    """
    Compute average RMS energy for an EMG segment/window.
    Higher values usually indicate stronger muscle activation.
    """
    if len(segment) == 0:
        return 0.0

    rms_per_channel = np.sqrt(np.mean(np.square(segment), axis=0))
    return float(np.mean(rms_per_channel))


def compute_phase_energy(segment: np.ndarray) -> float:
    return compute_emg_energy(segment)


def find_best_phase_offset(
    df: pd.DataFrame,
    channel_cols: list[str],
    phase_samples: int,
    cfg: PreprocessingConfig
) -> int:
    """
    Search a small positive offset to better align the repeating
    3s rest / 3s gesture structure.
    """
    file_gesture_label = normalize_gesture_label(str(df["file_gesture_label"].iloc[0]))
    if file_gesture_label == "Rest":
        return 0

    data = df[channel_cols].to_numpy(dtype=np.float32)
    n = len(data)

    max_offset = min(
        int(cfg.phase_offset_search_max_abs_seconds * cfg.sample_rate_hz),
        max(0, phase_samples - 1)
    )

    candidate_offsets = list(range(0, max_offset + 1, cfg.phase_offset_search_step))
    if len(candidate_offsets) == 0:
        return 0

    best_offset = 0
    best_score = -np.inf

    for offset in candidate_offsets:
        num_complete_phases = (n - offset) // phase_samples
        if num_complete_phases < 2:
            continue

        energies = []
        for phase_idx in range(num_complete_phases):
            start = offset + phase_idx * phase_samples
            end = start + phase_samples
            segment = data[start:end]
            energies.append(compute_phase_energy(segment))

        rest_energies = np.array(energies[0::2], dtype=np.float32)
        gesture_energies = np.array(energies[1::2], dtype=np.float32)

        if len(rest_energies) == 0 or len(gesture_energies) == 0:
            continue

        score = float(np.mean(gesture_energies) - np.mean(rest_energies))

        if score > best_score:
            best_score = score
            best_offset = offset

    return int(best_offset)


def split_file_into_phases(
    df: pd.DataFrame,
    channel_cols: list[str],
    phase_samples: int,
    cfg: PreprocessingConfig
) -> List[pd.DataFrame]:
    """
    Split one file into consecutive 3-second phases after first estimating a small alignment offset.
    """
    phases = []

    file_gesture_label = normalize_gesture_label(str(df["file_gesture_label"].iloc[0]))
    subject_id = str(df["subject_id"].iloc[0])
    source_file = str(df["source_file"].iloc[0])

    best_offset = find_best_phase_offset(df, channel_cols, phase_samples, cfg)

    n = len(df)
    num_complete_phases = (n - best_offset) // phase_samples

    for phase_idx in range(num_complete_phases):
        start = best_offset + phase_idx * phase_samples
        end = start + phase_samples

        phase = df.iloc[start:end].copy().reset_index(drop=True)
        phase["label"] = get_phase_label(file_gesture_label, phase_idx)
        phase["phase_index"] = phase_idx
        phase["subject_id"] = subject_id
        phase["source_file"] = source_file
        phase["file_gesture_label"] = file_gesture_label
        phase["best_offset"] = best_offset

        phases.append(phase)

    return phases


def make_windows_from_phase(
    phase_df: pd.DataFrame,
    channel_cols: list[str],
    cfg: PreprocessingConfig,
    rest_energy_threshold: float | None = None,
    gesture_energy_multiplier: float = 1.10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create windows from the stable middle part of a phase.

    Important real-time choice:
    - Do NOT only take the highest-energy gesture windows.
    - Instead, take evenly spaced windows across the cleaned phase.
    - Then remove only gesture windows that look too much like Rest.

    This teaches the model realistic gesture variation instead of only perfect,
    maximum-contraction examples.
    """
    trim = int(cfg.boundary_trim_seconds * cfg.sample_rate_hz)

    if len(phase_df) <= 2 * trim:
        return (
            np.empty((0, cfg.window_size, len(channel_cols)), dtype=np.float32),
            np.empty((0,), dtype=object)
        )

    core_df = phase_df.iloc[trim: len(phase_df) - trim].reset_index(drop=True)
    data = core_df[channel_cols].to_numpy(dtype=np.float32)
    label = str(core_df["label"].iloc[0])

    T, C = data.shape

    if T < cfg.window_size:
        return (
            np.empty((0, cfg.window_size, C), dtype=np.float32),
            np.empty((0,), dtype=object)
        )

    max_start = T - cfg.window_size

    if label == "Rest":
        num_windows = cfg.rest_windows_per_phase
    else:
        num_windows = cfg.gesture_windows_per_phase

    if num_windows <= 1:
        candidate_starts = [(T - cfg.window_size) // 2]
    else:
        candidate_starts = np.linspace(
            0,
            max_start,
            num=num_windows,
            dtype=int
        ).tolist()

    starts = []

    for s in candidate_starts:
        window = data[s:s + cfg.window_size]
        energy = compute_emg_energy(window)

        if label != "Rest" and rest_energy_threshold is not None:
            if energy < gesture_energy_multiplier * rest_energy_threshold:
                continue

        starts.append(s)

    if len(starts) == 0:
        return (
            np.empty((0, cfg.window_size, C), dtype=np.float32),
            np.empty((0,), dtype=object)
        )

    X = np.stack([data[s:s + cfg.window_size] for s in starts], axis=0).astype(np.float32)
    y = np.array([label] * len(starts), dtype=object)

    return X, y


all_phases = []

for source_file, sub in filtered_df.groupby("source_file", sort=False):
    sub = sub.reset_index(drop=True)
    phases = split_file_into_phases(sub, channel_cols, phase_samples, cfg)
    all_phases.extend(phases)

print("Number of 3-second phases found:", len(all_phases))

rest_window_energies = []

for phase_df in all_phases:
    label = str(phase_df["label"].iloc[0])

    if label != "Rest":
        continue

    trim = int(cfg.boundary_trim_seconds * cfg.sample_rate_hz)

    if len(phase_df) <= 2 * trim:
        continue

    core_df = phase_df.iloc[trim: len(phase_df) - trim].reset_index(drop=True)
    data = core_df[channel_cols].to_numpy(dtype=np.float32)

    if len(data) < cfg.window_size:
        continue

    start = (len(data) - cfg.window_size) // 2
    window = data[start:start + cfg.window_size]

    rest_window_energies.append(compute_emg_energy(window))

rest_energy_threshold = np.percentile(rest_window_energies, 95)

print("Rest windows used to estimate threshold:", len(rest_window_energies))
print("95th percentile Rest energy threshold:", rest_energy_threshold)

X_list = []
y_list = []
file_list = []
subject_list = []

dropped_phase_count = 0
kept_window_count = 0

for phase_df in all_phases:
    expected_label = str(phase_df["label"].iloc[0])

    X_phase, y_phase = make_windows_from_phase(
        phase_df,
        channel_cols,
        cfg,
        rest_energy_threshold=rest_energy_threshold,
        gesture_energy_multiplier=1.10
    )

    if len(y_phase) == 0:
        if expected_label != "Rest":
            dropped_phase_count += 1
        continue

    kept_window_count += len(y_phase)

    X_list.append(X_phase)
    y_list.append(y_phase)
    file_list.extend([phase_df["source_file"].iloc[0]] * len(y_phase))
    subject_list.extend([phase_df["subject_id"].iloc[0]] * len(y_phase))

print("Gesture phases dropped because they looked too much like Rest:", dropped_phase_count)
print("Total windows kept:", kept_window_count)

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
files = np.array(file_list)
subjects = np.array(subject_list)

print(f"Final dataset shape: X={X.shape}, y={y.shape}")
print("Window label distribution:")
print(pd.Series(y).value_counts().sort_index())
print("Number of unique subjects:", len(np.unique(subjects)))
