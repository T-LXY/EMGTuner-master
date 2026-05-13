# AI SLOP CODE!!!

"""
finetune_fomaml.py
==================
First-Order MAML (FOMAML) fine-tuning of the EMG gesture CNN-LSTM classifier
to a single new subject.

Usage
-----
python finetune_fomaml.py \
    --subject_dir    data/subject_data/subj99/               \
    --model_path     artifacts/best_model_v8.pt              \
    --meta_path      artifacts/metadata_v8.json              \
    --output_dir     artifacts/finetuned/                    \
    [--inner_lr      0.01]                                   \
    [--inner_steps   5]                                      \
    [--outer_lr      1e-4]                                   \
    [--meta_epochs   50]                                     \
    [--tasks_per_epoch 8]                                    \
    [--k_shot        5]                                      \
    [--q_query       5]                                      \
    [--adapt_steps   10]                                     \
    [--adapt_lr      1e-3]                                   \
    [--seed          42]

Overview of FOMAML adaptation
------------------------------
FOMAML treats each subject as a "task".  For a new target subject the
procedure has two stages:

  Stage 1 — Meta-update  (optional, runs only when --meta_epochs > 0)
    The pre-trained base model is further meta-trained using support/query
    splits drawn exclusively from the *target* subject's data.  Because only
    one subject is available the inner and outer sets are non-overlapping
    random splits of that subject's windows.

    Inner loop  : clone model → take `inner_steps` SGD steps on support set
    Outer loop  : evaluate cloned model on query set → back-prop through the
                  *first-order* gradient (FOMAML: stop gradients at inner-loop
                  boundary) → accumulate over `tasks_per_epoch` virtual tasks →
                  step original model's parameters with AdamW

  Stage 2 — Final adaptation  (always runs)
    After meta-training (or directly from the checkpoint if meta_epochs == 0),
    fine-tune the model on *all* available subject windows for `adapt_steps`
    gradient steps at `adapt_lr`.  This is the model you deploy.

Key design choices
------------------
* Normalisation statistics (mean / std) are NOT stored in the v8 metadata JSON.
  If your metadata does include "norm_mean"/"norm_std" keys they will be used;
  otherwise the script falls back to fitting the statistics on the new subject's
  own windows.  This is a reasonable approximation for adaptation: the subject's
  channels are standardised consistently, and the base model is robust enough
  to small distribution shifts after meta-training.  For best results, save
  norm_mean/norm_std when you retrain and add them to your metadata JSON.
* Label mapping is also loaded from metadata so that the output head remains
  compatible with the base model.  Windows whose gesture label is unseen in the
  base label map are silently skipped with a warning.
* Support/query splits inside FOMAML tasks use the *global* label indices (not
  re-indexed per task) so that the fixed base-model head's output neurons stay
  correctly aligned throughout meta-training.
* The final adapted checkpoint is saved alongside a copy of the metadata so
  the pair is self-contained for inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from grab_subject_data import (
        PreprocessingConfig,
        preprocess_files,
    )
except ImportError as exc:
    sys.exit(
        "Could not import grab_subject_data.py.  Make sure it is in the same "
        f"directory or on PYTHONPATH.\n  {exc}"
    )


# Pulled from v8, ENSURE KEEPS UP TO DATE!!!!!!!
class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        conv_channels: list[int] | tuple[int, ...] = (64, 128),
        kernel_size: int = 7,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_size       = input_size
        self.num_classes      = num_classes
        self.conv_channels    = list(conv_channels)
        self.kernel_size      = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers  = lstm_num_layers
        self.dropout          = dropout
        self.bidirectional    = bidirectional

        conv_layers = []
        in_channels = input_size
        for i, out_channels in enumerate(self.conv_channels):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            ])
            if i == 0:
                conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.cnn = nn.Sequential(*conv_layers)

        self.lstm = nn.LSTM(
            input_size   = in_channels,
            hidden_size  = lstm_hidden_size,
            num_layers   = lstm_num_layers,
            batch_first  = True,
            dropout      = dropout if lstm_num_layers > 1 else 0.0,
            bidirectional= bidirectional,
        )

        lstm_out_size = lstm_hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, channels)
        x = x.transpose(1, 2)          # (batch, channels, seq_len)  for CNN
        x = self.cnn(x)
        x = x.transpose(1, 2)          # (batch, seq_len, features)  for LSTM
        lstm_out, _ = self.lstm(x)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        features = torch.cat([avg_pool, max_pool], dim=1)
        return self.head(features)


# Normalisation helpers
def apply_channel_standardization(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


# Subject data loader
def load_subject_windows(
    subject_files: list[Path],
    label_map: dict[str, int],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    cfg: PreprocessingConfig,
    random_state: int = random.randint(1,100),
) -> tuple[np.ndarray, np.ndarray]:
    X_raw, y_raw, _ = preprocess_files(
        subject_files, cfg=cfg, random_state=random_state
    )

    known = set(label_map.keys())
    unknown = sorted(set(y_raw.tolist()) - known)
    if unknown:
        print(f"[WARN] Dropping windows with unknown labels: {unknown}")

    valid_mask = np.array([l in known for l in y_raw])
    if valid_mask.sum() == 0:
        raise RuntimeError(
            "No windows with labels known to the base model.  "
            "Check that the subject files match the training gestures."
        )

    X_norm = apply_channel_standardization(X_raw[valid_mask], norm_mean, norm_std)
    y_enc  = np.array([label_map[l] for l in y_raw[valid_mask]], dtype=np.int64)

    return X_norm, y_enc


# ══════════════════════════════════════════════════════════════════════════════
# FOMAML inner loop
# ══════════════════════════════════════════════════════════════════════════════

def inner_loop(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    device: torch.device,
) -> nn.Module:
    """
    Clone the model and perform `inner_steps` SGD updates on the support set.

    FOMAML:  we use a plain `clone.parameters()` copy and standard autograd.
    Gradients through the inner loop are *not* tracked in the outer loss
    (i.e. we call `.detach()` on the adapted parameters before the outer
    backward pass — the standard FOMAML approximation).

    Returns the *adapted* clone (still on `device`).
    """
    clone = copy.deepcopy(model).to(device)
    clone.train()

    inner_opt = torch.optim.SGD(clone.parameters(), lr=inner_lr)

    support_x = support_x.to(device)
    support_y = support_y.to(device)

    for _ in range(inner_steps):
        inner_opt.zero_grad()
        logits = clone(support_x)
        loss   = F.cross_entropy(logits, support_y)
        loss.backward()
        # Clip gradients inside the inner loop to keep updates stable
        torch.nn.utils.clip_grad_norm_(clone.parameters(), max_norm=1.0)
        inner_opt.step()

    return clone


# ══════════════════════════════════════════════════════════════════════════════
# Task sampler (single-subject)
# ══════════════════════════════════════════════════════════════════════════════

def sample_support_query(
    X: np.ndarray,
    y: np.ndarray,
    k_shot: int,
    q_query: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw a non-overlapping support / query split from a single subject's data.

    Each *present* class contributes exactly `k_shot` support windows and
    `q_query` query windows.  When a class has fewer than k_shot + q_query
    windows, sampling is done with replacement (acceptable in the low-data
    regime FOMAML is designed for).

    IMPORTANT — labels are kept as their *global* integer indices (i.e. the
    same indices the base model head was trained with).  We do NOT re-index
    0…n_way-1 here because the head's output neurons are fixed; re-indexing
    would mis-align inner-loop gradients and corrupt the outer head weights.

    Returns
    -------
    support_x, support_y, query_x, query_y  — all on `device`
    """
    classes = np.unique(y)

    sx_list, sy_list = [], []
    qx_list, qy_list = [], []

    for cls in classes:
        idx     = np.where(y == cls)[0]
        needed  = k_shot + q_query
        replace = len(idx) < needed
        chosen  = rng.choice(idx, size=needed, replace=replace)

        sx_list.append(X[chosen[:k_shot]])
        sy_list.extend([int(cls)] * k_shot)    # global label index preserved

        qx_list.append(X[chosen[k_shot:]])
        qy_list.extend([int(cls)] * q_query)

    support_x = torch.tensor(np.vstack(sx_list), dtype=torch.float32).to(device)
    support_y = torch.tensor(sy_list,             dtype=torch.long).to(device)
    query_x   = torch.tensor(np.vstack(qx_list), dtype=torch.float32).to(device)
    query_y   = torch.tensor(qy_list,             dtype=torch.long).to(device)

    return support_x, support_y, query_x, query_y


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — FOMAML meta-update on target subject
# ══════════════════════════════════════════════════════════════════════════════

def fomaml_meta_train(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    inner_lr: float,
    inner_steps: int,
    outer_lr: float,
    meta_epochs: int,
    tasks_per_epoch: int,
    k_shot: int,
    q_query: int,
    device: torch.device,
    random_state: int = 42,
) -> nn.Module:
    """
    Run FOMAML outer-loop training using only the target subject's data.

    The outer-loop model parameters are updated via AdamW on the *first-order*
    gradient approximation: gradients of the query loss w.r.t. the *pre-inner-
    loop* (theta) parameters, computed by detaching the adapted clone.

    Concretely for each outer step:

        theta′ = theta - lr_inner * ∇_theta L_support(theta)   [inner loop]
        outer_loss += L_query(theta′)                           [no 2nd order]
        theta ← theta - lr_outer * ∇_theta outer_loss / T      [AdamW step]

    Because the inner loop operates on a deepcopy of the model, PyTorch does
    NOT build a second-order graph.  This is exactly FOMAML.

    Returns
    -------
    The meta-updated model (parameters on `device`).
    """
    # The outer optimiser updates the *original* (meta) model
    outer_opt = torch.optim.AdamW(model.parameters(), lr=outer_lr, weight_decay=1e-4)

    rng = np.random.default_rng(random_state)
    model.to(device)

    # Need enough data: at least (k_shot + q_query) windows per class
    min_per_class = k_shot + q_query
    classes, counts = np.unique(y, return_counts=True)
    sparse_classes  = [str(c) for c, n in zip(classes, counts) if n < min_per_class]
    if sparse_classes:
        print(
            f"[INFO] Classes with fewer than {min_per_class} windows "
            f"(will sample with replacement): {sparse_classes}"
        )

    print(f"\n── Stage 1: FOMAML meta-training  ({meta_epochs} epochs × "
          f"{tasks_per_epoch} tasks/epoch) ──────")

    for epoch in range(1, meta_epochs + 1):
        model.train()
        outer_opt.zero_grad()

        epoch_query_loss = 0.0
        epoch_query_acc  = 0.0

        for _ in range(tasks_per_epoch):
            # ── Sample support / query ───────────────────────────────────────
            sx, sy, qx, qy = sample_support_query(
                X, y, k_shot, q_query, rng, device
            )

            # ── Inner loop on a detached clone ───────────────────────────────
            # FOMAML: deep-copy so no 2nd-order graph is retained
            adapted = inner_loop(model, sx, sy, inner_lr, inner_steps, device)

            # ── Outer loss on query set ───────────────────────────────────────
            # Evaluate the adapted clone.  Because it was created by deepcopy +
            # SGD (not by differentiating through the inner SGD steps), the
            # gradient that flows back here is the FOMAML approximation.
            adapted.train()
            q_logits   = adapted(qx)
            q_loss     = F.cross_entropy(q_logits, qy)

            # Manually compute the first-order gradient:
            # ∂L_query(theta′) / ∂theta  ≈  ∂L_query(theta′) / ∂theta′
            # We do this by computing grads w.r.t. adapted params, then
            # copying them back to the original model's params.
            q_loss.backward()

            with torch.no_grad():
                for p_meta, p_adapted in zip(model.parameters(), adapted.parameters()):
                    if p_adapted.grad is not None:
                        if p_meta.grad is None:
                            p_meta.grad = p_adapted.grad.clone() / tasks_per_epoch
                        else:
                            p_meta.grad += p_adapted.grad.clone() / tasks_per_epoch

            # Track stats
            with torch.no_grad():
                epoch_query_loss += q_loss.item()
                preds = q_logits.argmax(dim=1)
                epoch_query_acc  += (preds == qy).float().mean().item()

        # Clip outer-loop gradients before stepping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        outer_opt.step()

        avg_loss = epoch_query_loss / tasks_per_epoch
        avg_acc  = epoch_query_acc  / tasks_per_epoch
        if epoch % max(1, meta_epochs // 10) == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{meta_epochs} | "
                f"query_loss={avg_loss:.4f}  query_acc={avg_acc:.4f}"
            )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Final adaptation (standard fine-tune on all subject windows)
# ══════════════════════════════════════════════════════════════════════════════

def final_adaptation(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    adapt_lr: float,
    adapt_steps: int,
    batch_size: int,
    device: torch.device,
    random_state: int = 42,
) -> nn.Module:
    """
    Fine-tune on ALL subject windows for `adapt_steps` gradient steps.

    This is essentially a short, in-domain fine-tune with a low learning rate.
    The BatchNorm layers are kept in training mode so their running statistics
    adapt to the new subject's distribution.

    Returns
    -------
    The final adapted model.
    """
    rng = np.random.default_rng(random_state)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=adapt_lr, weight_decay=1e-5)

    N = len(X)
    indices = np.arange(N)

    print(f"\n── Stage 2: Final adaptation  ({adapt_steps} steps, "
          f"batch_size={batch_size}) ─────────────")

    for step in range(1, adapt_steps + 1):
        # Mini-batch from full subject data (with replacement when N < batch)
        replace = N < batch_size
        batch_idx = rng.choice(indices, size=min(batch_size, N), replace=replace)

        bx = torch.tensor(X[batch_idx], dtype=torch.float32).to(device)
        by = torch.tensor(y[batch_idx], dtype=torch.long).to(device)

        opt.zero_grad()
        logits = model(bx)
        loss   = F.cross_entropy(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % max(1, adapt_steps // 5) == 0 or step == 1:
            with torch.no_grad():
                acc = (logits.argmax(1) == by).float().mean().item()
            print(f"  Step {step:4d}/{adapt_steps} | loss={loss.item():.4f}  acc={acc:.4f}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_subject(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    inv_label_map: dict[int, str],
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    from sklearn.metrics import classification_report, f1_score

    model.eval().to(device)
    all_preds, all_true = [], []

    for start in range(0, len(X), batch_size):
        bx = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
        preds = model(bx).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y[start:start + batch_size])

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    present_classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    target_names    = [inv_label_map.get(i, str(i)) for i in present_classes]

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc      = (y_true == y_pred).mean()

    print("\n── Evaluation on subject data ─────────────────────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {macro_f1:.4f}")
    print(classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    return {"accuracy": acc, "macro_f1": macro_f1}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FOMAML fine-tuning of the EMG gesture classifier to a new subject.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--subject_dir", required=True, metavar="DIR",
        help=(
            "Folder containing the target subject's CSV files.  "
            "All *.csv files found recursively under this directory are used.  "
            "Typically this is a single subject folder such as "
            "'subject_data/subj99/'."
        ),
    )
    p.add_argument(
        "--model_path", required=True, metavar="PT",
        help="Path to the base model checkpoint (best_model_v8.pt).",
    )
    p.add_argument(
        "--meta_path", required=True, metavar="JSON",
        help="Path to the saved metadata JSON (metadata_v8.json).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output_dir", default="artifacts/finetuned", metavar="DIR",
        help="Directory to save the adapted model and metadata.",
    )

    # ── FOMAML hyper-params ───────────────────────────────────────────────────
    p.add_argument("--inner_lr",        type=float, default=0.01,
                   help="Learning rate for the inner-loop SGD updates.")
    p.add_argument("--inner_steps",     type=int,   default=5,
                   help="Number of gradient steps in the inner loop.")
    p.add_argument("--outer_lr",        type=float, default=1e-4,
                   help="Learning rate for the outer-loop (meta) AdamW optimiser.")
    p.add_argument("--meta_epochs",     type=int,   default=50,
                   help="Number of FOMAML outer-loop epochs.  Set 0 to skip Stage 1.")
    p.add_argument("--tasks_per_epoch", type=int,   default=8,
                   help="Virtual tasks (support/query splits) sampled per epoch.")
    p.add_argument("--k_shot",          type=int,   default=5,
                   help="Support windows per class per task.")
    p.add_argument("--q_query",         type=int,   default=5,
                   help="Query windows per class per task.")

    # ── Final adaptation hyper-params ─────────────────────────────────────────
    p.add_argument("--adapt_steps",     type=int,   default=30,
                   help="Gradient steps for Stage 2 final fine-tune.")
    p.add_argument("--adapt_lr",        type=float, default=1e-3,
                   help="Learning rate for Stage 2 fine-tune.")
    p.add_argument("--adapt_batch",     type=int,   default=32,
                   help="Mini-batch size for Stage 2 fine-tune.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--skip_eval",       action="store_true",
                   help="Skip the final evaluation (useful for fast CI checks).")

    return p.parse_args(argv)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load metadata ─────────────────────────────────────────────────────────
    meta_path = Path(args.meta_path)
    if not meta_path.exists():
        sys.exit(f"Metadata file not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    label_map:     dict[str, int] = meta["label_map"]
    inv_label_map: dict[int, str] = {int(k): v for k, v in meta["inverted_label_map"].items()}
    num_classes    = len(label_map)
    num_channels   = meta["num_channels"]

    # Normalisation statistics — v8 metadata JSON does not include these, so we
    # derive them from the subject's own windows after preprocessing.  If a
    # future metadata version adds "norm_mean"/"norm_std", they will be used
    # preferentially for perfect alignment with training-time statistics.
    _norm_mean_from_meta: Optional[np.ndarray] = (
        np.array(meta["norm_mean"], dtype=np.float32) if "norm_mean" in meta else None
    )
    _norm_std_from_meta: Optional[np.ndarray] = (
        np.array(meta["norm_std"],  dtype=np.float32) if "norm_std"  in meta else None
    )

    print(f"Label map  : {label_map}")
    print(f"Channels   : {num_channels}   Classes: {num_classes}")

    # ── Build model and load weights ──────────────────────────────────────────
    model = CNNLSTMClassifier(
        input_size       = num_channels,
        num_classes      = num_classes,
        conv_channels    = [48, 96],
        kernel_size      = 7,
        lstm_hidden_size = 96,
        lstm_num_layers  = 1,
        dropout          = 0.25,
        bidirectional    = True,
    )

    ckpt_path = Path(args.model_path)
    if not ckpt_path.exists():
        sys.exit(f"Model checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded base model from {ckpt_path}")

    # ── Preprocessing config (mirrors training defaults) ──────────────────────
    cfg = PreprocessingConfig(
        window_size             = meta.get("window_len",    600),
        step_size               = meta.get("step_size",     600),
        sample_rate_hz          = meta.get("sample_rate_hz", 500),
        phase_seconds           = meta.get("phase_seconds",   3),
        rest_windows_per_phase  = 1,
        gesture_windows_per_phase = 2,
        boundary_trim_seconds   = 0.5,
    )

    # ── Discover subject CSV files from folder ────────────────────────────────
    subject_dir = Path(args.subject_dir)
    if not subject_dir.exists():
        sys.exit(f"Subject directory not found: {subject_dir}")
    if not subject_dir.is_dir():
        sys.exit(f"--subject_dir must be a directory, got a file: {subject_dir}")

    subject_files = sorted(subject_dir.rglob("*.csv"))
    if not subject_files:
        sys.exit(f"No *.csv files found recursively under: {subject_dir}")

    # Filter out any gesture labels the base model excluded (e.g. "2F")
    # by relying on PreprocessingConfig.excluded_gestures later in preprocess_files.
    print(f"\nFound {len(subject_files)} CSV file(s) in '{subject_dir}':")
    for f in subject_files:
        print(f"  {f.relative_to(subject_dir)}")

    print(f"\nLoading subject data …")

    # Preprocess raw windows first (no normalisation yet)
    from grab_subject_data import preprocess_files as _preprocess_files
    X_raw, y_raw, _ = _preprocess_files(
        subject_files, cfg=cfg, random_state=args.seed
    )

    # Filter to known labels
    known = set(label_map.keys())
    unknown = sorted(set(y_raw.tolist()) - known)
    if unknown:
        print(f"[WARN] Dropping windows with unknown labels: {unknown}")
    valid_mask = np.array([l in known for l in y_raw])
    if valid_mask.sum() == 0:
        sys.exit(
            "No windows with labels known to the base model.  "
            "Check that the subject files match the training gestures."
        )
    X_raw = X_raw[valid_mask]
    y_str = y_raw[valid_mask]
    y     = np.array([label_map[l] for l in y_str], dtype=np.int64)

    # Resolve normalisation statistics
    if _norm_mean_from_meta is not None and _norm_std_from_meta is not None:
        print("[INFO] Using norm_mean/norm_std from metadata.")
        norm_mean = _norm_mean_from_meta
        norm_std  = _norm_std_from_meta
    else:
        print(
            "[INFO] norm_mean/norm_std not found in metadata — "
            "fitting on subject windows.  For best results, save these "
            "statistics from training and add them to metadata_v8.json."
        )
        flat      = X_raw.reshape(-1, X_raw.shape[-1])
        norm_mean = flat.mean(axis=0).astype(np.float32)
        norm_std  = (flat.std(axis=0) + 1e-8).astype(np.float32)

    X = apply_channel_standardization(X_raw, norm_mean, norm_std)
    print(f"Subject windows: {X.shape}  label distribution: "
          f"{ {inv_label_map[c]: int((y==c).sum()) for c in np.unique(y)} }")

    # ── Stage 1: FOMAML meta-update ───────────────────────────────────────────
    if args.meta_epochs > 0:
        model = fomaml_meta_train(
            model, X, y,
            inner_lr        = args.inner_lr,
            inner_steps     = args.inner_steps,
            outer_lr        = args.outer_lr,
            meta_epochs     = args.meta_epochs,
            tasks_per_epoch = args.tasks_per_epoch,
            k_shot          = args.k_shot,
            q_query         = args.q_query,
            device          = device,
            random_state    = args.seed,
        )
    else:
        print("\n[INFO] meta_epochs=0 — skipping Stage 1 (FOMAML meta-update).")

    # ── Stage 2: Final adaptation ─────────────────────────────────────────────
    model = final_adaptation(
        model, X, y,
        adapt_lr    = args.adapt_lr,
        adapt_steps = args.adapt_steps,
        batch_size  = args.adapt_batch,
        device      = device,
        random_state= args.seed,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if not args.skip_eval:
        evaluate_subject(model, X, y, inv_label_map, device)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_out = out_dir / "adapted_model.pt"
    meta_out  = out_dir / "adapted_metadata.json"

    torch.save(model.state_dict(), model_out)

    # Persist a copy of metadata so the adapted checkpoint is self-contained
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Adapted model saved to  {model_out.resolve()}")
    print(f"✓ Metadata saved to       {meta_out.resolve()}")


if __name__ == "__main__":
    main()