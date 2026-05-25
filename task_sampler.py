import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import random


# ─────────────────────────────────────────────
#  Task Definition  (unchanged)
# ─────────────────────────────────────────────
class Task:
    """
    One FOMAML task = N gesture classes × K support windows + Q query windows.

    Attributes:
        support_X : (N * K_shot,  timesteps, channels)
        support_y : (N * K_shot,)
        query_X   : (N * Q_query, timesteps, channels)
        query_y   : (N * Q_query,)
    """
    def __init__(
        self,
        task_id:   int,
        support_X: torch.Tensor,
        support_y: torch.Tensor,
        query_X:   torch.Tensor,
        query_y:   torch.Tensor,
    ):
        self.task_id   = task_id
        self.support_X = support_X
        self.support_y = support_y
        self.query_X   = query_X
        self.query_y   = query_y

    def support_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        ds = TensorDataset(self.support_X, self.support_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def query_loader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        ds = TensorDataset(self.query_X, self.query_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def __repr__(self):
        n_classes = self.support_y.unique().numel()
        return (
            f"Task(id={self.task_id} | "
            f"N={n_classes} | "
            f"support={len(self.support_y)} | "
            f"query={len(self.query_y)})"
        )


# ─────────────────────────────────────────────
#  Task Sampler
# ─────────────────────────────────────────────
class FOMAMLTaskSampler:
    """
    N-way K-shot episodic sampler for FOMAML.

    Tasks are subject-agnostic — windows are pooled across ALL subjects
    and sampled purely by gesture class. Each epoch exhausts the pool
    without replacement, then resets.

    Episode construction per task:
        • N gesture classes are selected (all classes if n_way == None)
        • K windows sampled per class  → support set  (inner loop)
        • Q windows sampled per class  → query set    (outer / meta-gradient)
        • Total windows per task: N * (K + Q)

    Args
    ----
    df            : general pool  — grab_data(..., ignore=tune_subject)
    df_sub        : tune pool     — grab_data(tune_folder)
    signal_col    : column holding the pre-windowed (timesteps, channels) array
    label_col     : gesture label column
    n_way         : number of gesture classes per task (None = use all)
    k_shot        : support windows per class
    q_query       : query windows per class
    seed          : reproducibility
    """

    def __init__(
        self,
        df,
        df_sub,
        signal_col: str         = "signal",
        label_col:  str         = "label",
        n_way:      Optional[int] = None,
        k_shot:     int         = 5,
        q_query:    int         = 5,
        seed:       Optional[int] = 42,
    ):
        self.signal_col = signal_col
        self.label_col  = label_col
        self.n_way      = n_way
        self.k_shot     = k_shot
        self.q_query    = q_query

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Pool all windows across subjects, indexed by class
        self._meta_train_pool = self._build_class_pool(df,     pool="meta_train")
        self._meta_tune_pool  = self._build_class_pool(df_sub, pool="meta_tune")

        # Per-epoch availability tracker (without-replacement)
        self._train_available: dict[int, list[int]] = {}
        self._tune_available:  dict[int, list[int]] = {}
        self._reset_availability("train")
        self._reset_availability("tune")

        # Auto-resolve n_way
        self._n_way_train = n_way or len(self._meta_train_pool)
        self._n_way_tune  = n_way or len(self._meta_tune_pool)

        self._task_counter = 0

        print(f"[TaskSampler] meta-train | classes: {len(self._meta_train_pool)} "
              f"| total windows: {sum(len(v) for v in self._meta_train_pool.values())}")
        print(f"[TaskSampler] meta-tune  | classes: {len(self._meta_tune_pool)} "
              f"| total windows: {sum(len(v) for v in self._meta_tune_pool.values())}")
        print(f"[TaskSampler] episode    | {self._n_way_train}-way {k_shot}-shot "
              f"+ {q_query}-query per class")

    # ── internals ─────────────────────────────────────────────────────────────

    def _build_class_pool(
        self, df, pool: str
    ) -> dict[int, np.ndarray]:
        """
        Returns { label -> X_windows (N_windows, timesteps, channels) }
        pooled across ALL subjects in df.
        """
        class_pool = {}
        for label, group in df.groupby(self.label_col):
            X = np.stack(group[self.signal_col].values).astype(np.float32)
            n_required = self.k_shot + self.q_query
            if len(X) < n_required:
                print(f"  [skip] {pool}/class={label} — "
                      f"only {len(X)} windows, need {n_required}")
                continue
            class_pool[int(label)] = X
        return class_pool

    def _reset_availability(self, split: str):
        """Shuffle and restore all window indices for a new epoch."""
        pool = self._meta_train_pool if split == "train" else self._meta_tune_pool
        avail = self._train_available if split == "train" else self._tune_available
        for label, X in pool.items():
            idx = list(range(len(X)))
            random.shuffle(idx)
            avail[label] = idx

    def _sample_task(
        self,
        pool:      dict[int, np.ndarray],
        available: dict[int, list[int]],
        n_way:     int,
        split:     str,
    ) -> Task:
        """
        Draw one N-way K-shot task.
        Pops indices from `available` (without replacement).
        Resets exhausted classes automatically.
        """
        # Refresh any class that can no longer fill K+Q windows
        n_required = self.k_shot + self.q_query
        for label in list(available.keys()):
            if len(available[label]) < n_required:
                idx = list(range(len(pool[label])))
                random.shuffle(idx)
                available[label] = idx

        # Sample N classes for this episode
        classes = random.sample(list(pool.keys()), k=n_way)

        sup_X_list, sup_y_list = [], []
        qry_X_list, qry_y_list = [], []

        for cls in classes:
            # Pop K + Q indices without replacement
            drawn   = [available[cls].pop() for _ in range(n_required)]
            sup_idx = drawn[:self.k_shot]
            qry_idx = drawn[self.k_shot:]

            sup_X_list.append(pool[cls][sup_idx])
            sup_y_list.extend([cls] * self.k_shot)
            qry_X_list.append(pool[cls][qry_idx])
            qry_y_list.extend([cls] * self.q_query)

        self._task_counter += 1
        return Task(
            task_id   = self._task_counter,
            support_X = torch.from_numpy(np.concatenate(sup_X_list, axis=0)),
            support_y = torch.tensor(sup_y_list, dtype=torch.long),
            query_X   = torch.from_numpy(np.concatenate(qry_X_list, axis=0)),
            query_y   = torch.tensor(qry_y_list, dtype=torch.long),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def sample_meta_train_batch(self, n_tasks: int) -> list[Task]:
        """
        Sample n_tasks episodes from the general (subject-agnostic) pool.
        Indices are consumed without replacement; exhausted classes auto-reset.
        Call reset_epoch('train') at the top of each meta-epoch to reshuffle.
        """
        return [
            self._sample_task(
                self._meta_train_pool,
                self._train_available,
                self._n_way_train,
                split="train",
            )
            for _ in range(n_tasks)
        ]

    def sample_meta_tune_batch(self, n_tasks: int = 1) -> list[Task]:
        """
        Sample n_tasks episodes from the fine-tune subject pool.
        """
        return [
            self._sample_task(
                self._meta_tune_pool,
                self._tune_available,
                self._n_way_tune,
                split="tune",
            )
            for _ in range(n_tasks)
        ]

    def reset_epoch(self, split: str = "train"):
        """
        Call at the start of each epoch to reshuffle and restore
        the without-replacement availability pool.

        Args:
            split : 'train', 'tune', or 'both'
        """
        if split in ("train", "both"):
            self._reset_availability("train")
        if split in ("tune", "both"):
            self._reset_availability("tune")

    @property
    def tasks_per_epoch(self) -> dict[str, int]:
        """
        Maximum number of tasks drawable per epoch before any class resets,
        based on the scarcest class in each pool.
        """
        def _min_tasks(pool, n_way):
            min_windows = min(len(v) for v in pool.values())
            return (min_windows // (self.k_shot + self.q_query)) * n_way

        return {
            "meta_train": _min_tasks(self._meta_train_pool, self._n_way_train),
            "meta_tune":  _min_tasks(self._meta_tune_pool,  self._n_way_tune),
        }
    
