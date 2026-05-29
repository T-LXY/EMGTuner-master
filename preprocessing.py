from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

def remove_window_dc_offset(X: np.ndarray) -> np.ndarray:
    """
    Remove the per-window, per-channel baseline.

    X has shape:
    (num_windows, window_size, num_channels)
    """
    window_mean = np.mean(X, axis=1, keepdims=True)
    return (X - window_mean).astype(np.float32)


def add_delta_features(X: np.ndarray) -> np.ndarray:
    """
    Add first-difference features.
    """
    delta = np.diff(X, axis=1, prepend=X[:, :1, :])
    return np.concatenate([X, delta], axis=2).astype(np.float32)


def prepare_model_features(X: np.ndarray) -> np.ndarray:
    """
    Apply all feature engineering steps before standardization.
    """
    X_centered = remove_window_dc_offset(X)
    X_features = add_delta_features(X_centered)
    return X_features.astype(np.float32)


def fit_channel_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a channel-wise standardizer using only the training set.
    """
    flat = X.reshape(-1, X.shape[-1])

    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0) + 1e-8

    return mean.astype(np.float32), std.astype(np.float32)


def apply_channel_standardization(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Apply training-set standardization to train/val/test data.
    """
    return ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def preprocess_mindrove_data(
    root_dir: str | Path = "mindrove_data",
    exclude_file_names: Optional[list[str] | set[str] | tuple[str, ...]] = None
) -> dict:
    """
    Run the full preprocessing pipeline before train/validation/test splitting.
    """

    @dataclass
    class PreprocessingConfig:
        """
        Stores preprocessing and windowing settings for the new Mindrove CSV files.
        """

        # 200 samples at 500 Hz = 0.40 seconds of EMG history per prediction.
        window_size: int = 200

        # 25 samples at 500 Hz = one training window every 0.05 seconds.
        step_size: int = 25

        # Mindrove armband sampling rate.
        sample_rate_hz: int = 500

        # Ignore labeled segments that are too short to be useful.
        min_segment_seconds: float = 0.75

        # Optional safety filter: remove non-Rest windows whose centered RMS energy is too close to Rest.
        gesture_energy_multiplier: float = 1.10
        use_gesture_energy_filter: bool = False


    cfg = PreprocessingConfig()

    min_segment_samples = int(cfg.min_segment_seconds * cfg.sample_rate_hz)

    root_dir = Path(root_dir)

    DATA_FILES_ALL = sorted(root_dir.rglob("*.csv"))

    exclude_file_names = set(exclude_file_names or [])

    DATA_FILES = [
        file
        for file in DATA_FILES_ALL
        if file.name not in exclude_file_names
    ]

    excluded_data_files = [
        file
        for file in DATA_FILES_ALL
        if file.name in exclude_file_names
    ]

    # Label mapping for the new Mindrove row-level labels in the last CSV column.
    SPEC_LABEL_MAP = {
        "Extend": 0,
        "Fist": 1,
        "Flex": 2,
        "Pro": 3,
        "Radial": 4,
        "Rest": 5,
        "Sup": 6,
        "Ulnar": 7,
    }
    SPEC_INDEX_TO_LABEL = {v: k for k, v in SPEC_LABEL_MAP.items()}

    print("Number of CSV files found before exclusion:", len(DATA_FILES_ALL))
    print("Number of CSV files excluded:", len(excluded_data_files))
    print("Number of CSV files used:", len(DATA_FILES))

    if len(excluded_data_files) > 0:
        print("\nExcluded files:")
        for file in excluded_data_files:
            print(file)

    if len(DATA_FILES) > 0:
        print("\nExample file used:", DATA_FILES[0])

    print("\nSample rate:", cfg.sample_rate_hz, "Hz")
    print("Window size:", cfg.window_size, "samples =", cfg.window_size / cfg.sample_rate_hz, "seconds")
    print("Step size:", cfg.step_size, "samples =", cfg.step_size / cfg.sample_rate_hz, "seconds")


    def load_mindrove_file(path: str | Path) -> pd.DataFrame:
        """
        Load one Mindrove CSV file.
        """
        path = Path(path)

        df = pd.read_csv(
            path,
            header=None,
            sep=r"\s+",
            engine="python",
            na_values=["nan", "NaN", "NULL", "None", ""]
        )

        df = df.dropna(how="all").reset_index(drop=True)

        if df.shape[1] < 2:
            raise ValueError(f"{path} must contain at least one channel column and one label column.")

        num_channels = df.shape[1] - 1
        channel_cols = [f"Channel{i + 1}" for i in range(num_channels)]
        label_col = "label_value"

        df.columns = channel_cols + [label_col]

        for col in channel_cols + [label_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        bad_channel_rows = df[channel_cols].isna().any(axis=1).sum()
        if bad_channel_rows > 0:
            raise ValueError(
                f"{path} has {bad_channel_rows} rows with missing/non-numeric channel values. "
                "The label column may be NaN, but channel columns should not be NaN."
            )

        return df


    def label_value_to_name(value) -> object:
        """
        Convert numeric labels such as 0.0, 1.0, ..., 7.0 into class names.
        """
        if pd.isna(value):
            return np.nan

        value_float = float(value)
        label_id = int(value_float) if value_float.is_integer() else value_float

        return SPEC_INDEX_TO_LABEL.get(label_id, f"Label_{label_id}")


    def infer_subject_id(path: str | Path, root_dir: str | Path) -> str:
        """
        Infer subject ID if files are stored in subject subfolders.
        """
        path = Path(path)
        root_dir = Path(root_dir)

        try:
            rel = path.relative_to(root_dir)
            if len(rel.parts) > 1:
                return rel.parts[0]
        except ValueError:
            pass

        return "unknown_subject"


    def infer_channel_cols(df: pd.DataFrame) -> list[str]:
        """
        Dynamically find channel columns.
        """
        channel_cols = [col for col in df.columns if re.fullmatch(r"Channel\d+", str(col))]

        def sort_key(name: str):
            m = re.search(r"(\d+)$", name)
            return int(m.group(1)) if m else float("inf")

        return sorted(channel_cols, key=sort_key)


    all_dfs = []
    schemas = []

    for file in DATA_FILES:
        df = load_mindrove_file(file)
        file_channel_cols = infer_channel_cols(df)

        df["label"] = df["label_value"].apply(label_value_to_name)
        df["source_file"] = str(file)
        df["source_index"] = np.arange(len(df), dtype=np.int64)
        df["time_seconds"] = df["source_index"] / cfg.sample_rate_hz
        df["subject_id"] = infer_subject_id(file, root_dir)

        df = df[
            ["source_file", "subject_id", "source_index", "time_seconds"]
            + file_channel_cols
            + ["label_value", "label"]
        ].copy()

        all_dfs.append(df)

        schemas.append({
            "file": str(file),
            "subject_id": df["subject_id"].iloc[0],
            "num_rows": len(df),
            "n_channels": len(file_channel_cols),
            "channels": file_channel_cols,
            "labeled_rows": int(df["label"].notna().sum()),
            "pause_rows": int(df["label"].isna().sum()),
        })

    if len(all_dfs) == 0:
        raise FileNotFoundError(
            f"No CSV files found under {root_dir.resolve()} after exclusions. "
            "Set root_dir to your mindrove_data folder or remove file names from exclude_file_names."
        )

    raw_df = pd.concat(all_dfs, ignore_index=True)
    schemas_df = pd.DataFrame(schemas)

    channel_cols = infer_channel_cols(raw_df)

    if len(channel_cols) == 0:
        raise ValueError("No channel columns were detected.")

    channel_counts = schemas_df["n_channels"].unique()
    if len(channel_counts) != 1:
        raise ValueError(
            "Not all files have the same number of channels. "
            f"Detected channel counts: {sorted(channel_counts.tolist())}"
        )

    print("\nraw_df shape:", raw_df.shape)
    print("Detected EMG channels:", channel_cols)
    print("Detected number of channels:", len(channel_cols))

    print("\nLabel distribution including NaN pauses:")
    print(raw_df["label"].value_counts(dropna=False).sort_index())


    filtered_df = raw_df.copy()

    print("Using new Mindrove row-level labels from the last CSV column.")
    print("NaN labels are pause/unmarked rows and will be ignored during window creation.")
    print("Using known Mindrove sampling rate:", cfg.sample_rate_hz, "Hz")
    print("Window size:", cfg.window_size, "samples =", cfg.window_size / cfg.sample_rate_hz, "seconds")
    print("Step size:", cfg.step_size, "samples =", cfg.step_size / cfg.sample_rate_hz, "seconds")

    print("\nRows:")
    print("Total rows:", len(filtered_df))
    print("Labeled rows:", int(filtered_df["label"].notna().sum()))
    print("Pause/unmarked rows:", int(filtered_df["label"].isna().sum()))


    def report_label_column(df: pd.DataFrame):
        """
        Verify the new row-level label column.
        """
        labeled = df[df["label_value"].notna()].copy()
        pause = df[df["label_value"].isna()].copy()

        print("Total rows:", len(df))
        print("Labeled rows used for possible training windows:", len(labeled))
        print("Pause/unmarked rows ignored:", len(pause))

        seen_values = sorted(labeled["label_value"].unique().tolist())
        print("Numeric labels present:", seen_values)

        print("\nDecoded label distribution:")
        print(labeled["label"].value_counts().sort_index())

        known_ids = set(SPEC_INDEX_TO_LABEL.keys())
        seen_ids = set()

        for value in seen_values:
            value_float = float(value)
            seen_ids.add(int(value_float) if value_float.is_integer() else value_float)

        unknown_ids = sorted(seen_ids - known_ids, key=lambda x: str(x))

        if len(unknown_ids) > 0:
            print("\nWarning: these label IDs were not in SPEC_LABEL_MAP and were named automatically:")
            print(unknown_ids)
        else:
            print("\nAll non-NaN label IDs were found in SPEC_LABEL_MAP.")


    report_label_column(filtered_df)


    def compute_emg_energy(segment: np.ndarray) -> float:
        """
        Compute centered RMS energy for an EMG window.
        """
        if len(segment) == 0:
            return 0.0

        segment = segment.astype(np.float32)
        centered = segment - np.mean(segment, axis=0, keepdims=True)
        rms_per_channel = np.sqrt(np.mean(np.square(centered), axis=0))
        return float(np.mean(rms_per_channel))


    def add_labeled_segment_ids(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign a segment ID to each contiguous non-NaN labeled region.
        """
        df = df.copy()

        segment_ids = []
        current_segment_id = -1
        previous_label = None

        for label in df["label"].tolist():
            if pd.isna(label):
                segment_ids.append(np.nan)
                previous_label = None
                continue

            if previous_label is None or label != previous_label:
                current_segment_id += 1

            segment_ids.append(current_segment_id)
            previous_label = label

        df["segment_id"] = segment_ids
        return df


    def extract_labeled_segments(
        df: pd.DataFrame,
        channel_cols: list[str],
        cfg: PreprocessingConfig
    ) -> list[pd.DataFrame]:
        """
        Extract contiguous labeled gesture/rest segments from one file.
        Pause rows with NaN labels are ignored.
        """
        df = add_labeled_segment_ids(df)

        segments = []

        labeled_df = df[df["segment_id"].notna()].copy()
        if len(labeled_df) == 0:
            return segments

        for segment_id, segment_df in labeled_df.groupby("segment_id", sort=True):
            segment_df = segment_df.reset_index(drop=True)

            if len(segment_df) < min_segment_samples:
                continue

            segment_df["segment_id"] = int(segment_id)
            segment_df["segment_label"] = str(segment_df["label"].iloc[0])

            segments.append(segment_df)

        return segments


    def window_start_indices(length: int, window_size: int, step_size: int) -> list[int]:
        """
        Return sliding-window start indices for one segment.
        """
        if length < window_size:
            return []

        return list(range(0, length - window_size + 1, step_size))


    def make_windows_from_segment(
        segment_df: pd.DataFrame,
        channel_cols: list[str],
        cfg: PreprocessingConfig,
        rest_energy_threshold: float | None = None,
        gesture_energy_multiplier: float = 1.10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from one labeled segment.

        Each output window has shape:
        (window_size, num_channels)
        """
        data = segment_df[channel_cols].to_numpy(dtype=np.float32)
        label = str(segment_df["label"].iloc[0])

        T, C = data.shape

        starts = window_start_indices(T, cfg.window_size, cfg.step_size)

        if len(starts) == 0:
            return (
                np.empty((0, cfg.window_size, C), dtype=np.float32),
                np.empty((0,), dtype=object)
            )

        kept_starts = []

        for s in starts:
            window = data[s:s + cfg.window_size]

            # Optional filter for mislabeled gesture windows that look too much like Rest.
            # This does not apply to the actual Rest class.
            if label != "Rest" and rest_energy_threshold is not None:
                energy = compute_emg_energy(window)
                if energy < gesture_energy_multiplier * rest_energy_threshold:
                    continue

            kept_starts.append(s)

        if len(kept_starts) == 0:
            return (
                np.empty((0, cfg.window_size, C), dtype=np.float32),
                np.empty((0,), dtype=object)
            )

        X = np.stack(
            [data[s:s + cfg.window_size] for s in kept_starts],
            axis=0
        ).astype(np.float32)

        y = np.array([label] * len(kept_starts), dtype=object)

        return X, y


    all_segments = []

    for source_file, sub in filtered_df.groupby("source_file", sort=False):
        sub = sub.reset_index(drop=True)
        file_segments = extract_labeled_segments(sub, channel_cols, cfg)
        all_segments.extend(file_segments)

    print("Number of labeled segments found:", len(all_segments))

    segment_summary = pd.DataFrame([
        {
            "source_file": segment_df["source_file"].iloc[0],
            "segment_id": int(segment_df["segment_id"].iloc[0]),
            "label": segment_df["segment_label"].iloc[0],
            "samples": len(segment_df),
            "seconds": len(segment_df) / cfg.sample_rate_hz,
        }
        for segment_df in all_segments
    ])

    # Optional Rest-energy filter.
    # If this filter is enabled, subtle gesture windows can be removed before training.
    # That can make the training data too different from validation/test data and can make the model worse at recognizing weak gestures.
    rest_energy_threshold = None

    if cfg.use_gesture_energy_filter:
        rest_window_energies = []

        for segment_df in all_segments:
            label = str(segment_df["segment_label"].iloc[0])

            if label != "Rest":
                continue

            data = segment_df[channel_cols].to_numpy(dtype=np.float32)
            starts = window_start_indices(len(data), cfg.window_size, cfg.step_size)

            for s in starts:
                window = data[s:s + cfg.window_size]
                rest_window_energies.append(compute_emg_energy(window))

        if len(rest_window_energies) > 0:
            rest_energy_threshold = float(np.percentile(rest_window_energies, 95))
        else:
            rest_energy_threshold = None

        print("\nRest windows used to estimate threshold:", len(rest_window_energies))
        print("95th percentile Rest energy threshold:", rest_energy_threshold)
    else:
        print("\nGesture energy filter disabled.")
        print("No gesture windows will be removed for looking too similar to Rest.")


    X_list = []
    y_list = []
    file_list = []
    subject_list = []

    dropped_segment_count = 0
    kept_window_count = 0

    for segment_df in all_segments:
        expected_label = str(segment_df["segment_label"].iloc[0])

        X_segment, y_segment = make_windows_from_segment(
            segment_df,
            channel_cols,
            cfg,
            rest_energy_threshold=rest_energy_threshold,
            gesture_energy_multiplier=cfg.gesture_energy_multiplier
        )

        if len(y_segment) == 0:
            if expected_label != "Rest":
                dropped_segment_count += 1
            continue

        kept_window_count += len(y_segment)

        X_list.append(X_segment)
        y_list.append(y_segment)
        file_list.extend([segment_df["source_file"].iloc[0]] * len(y_segment))
        subject_list.extend([segment_df["subject_id"].iloc[0]] * len(y_segment))

    if len(X_list) == 0:
        raise RuntimeError(
            "No training windows were created. Try reducing window_size, "
            "reducing step_size, lowering min_segment_seconds, or disabling the Rest-energy filter."
        )

    print("\nNon-Rest segments dropped because all windows looked too much like Rest:", dropped_segment_count)
    print("Total windows kept:", kept_window_count)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    # Feature engineer X ig
    X = prepare_model_features(X)
    mean_std = fit_channel_standardizer(X)
    X = apply_channel_standardization(X, *mean_std)

    files = np.array(file_list)
    subjects = np.array(subject_list)

    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    print("Window label distribution:")
    print(pd.Series(y).value_counts().sort_index())
    print("Number of unique files:", len(np.unique(files)))
    print("Number of unique subjects:", len(np.unique(subjects)))

    return {
        # Main variables used immediately after preprocessing.
        "X": X,
        "y": y,
        "files": files,
        "subjects": subjects,

        # DataFrames and summaries.
        "raw_df": raw_df,
        "filtered_df": filtered_df,
        "schemas_df": schemas_df,
        "segment_summary": segment_summary,

        # Configuration and file information.
        "cfg": cfg,
        "PreprocessingConfig": PreprocessingConfig,
        "root_dir": root_dir,
        "DATA_FILES": DATA_FILES,
        "DATA_FILES_ALL": DATA_FILES_ALL,
        "excluded_data_files": excluded_data_files,
        "min_segment_samples": min_segment_samples,

        # Label and channel metadata needed later.
        "SPEC_LABEL_MAP": SPEC_LABEL_MAP,
        "SPEC_INDEX_TO_LABEL": SPEC_INDEX_TO_LABEL,
        "channel_cols": channel_cols,

        # Segment/window metadata.
        "all_segments": all_segments,
        "rest_energy_threshold": rest_energy_threshold,
        "dropped_segment_count": dropped_segment_count,
        "kept_window_count": kept_window_count,

        # Intermediate lists, returned for debugging/reproducibility.
        "X_list": X_list,
        "y_list": y_list,
        "file_list": file_list,
        "subject_list": subject_list,

        # Helper functions, returned so later code still has access to them.
        "load_mindrove_file": load_mindrove_file,
        "label_value_to_name": label_value_to_name,
        "infer_subject_id": infer_subject_id,
        "infer_channel_cols": infer_channel_cols,
        "report_label_column": report_label_column,
        "compute_emg_energy": compute_emg_energy,
        "add_labeled_segment_ids": add_labeled_segment_ids,
        "extract_labeled_segments": extract_labeled_segments,
        "window_start_indices": window_start_indices,
        "make_windows_from_segment": make_windows_from_segment,
    }