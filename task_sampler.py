import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import random


# ─────────────────────────────────────────────
#  Task Definition
# ─────────────────────────────────────────────
class Task:
    """
    One FOMAML task = 
        N gesture classes 
        K support windows 
        Q query windows
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
    N-way K-shot episodic sampler for FOMAML

    pre-windowed X/y arrays
    pooled by gesture class
    EXHAUSTS task, does not reuse

    n_way = gesture classes per task (None = all classes)
    """

    def __init__(
        self,
        X_train:   np.ndarray,
        y_train:   np.ndarray,
        X_tune:    np.ndarray,
        y_tune:    np.ndarray,
        cfg,
        n_way:     Optional[int] = None,
    ):
        self.k_shot  = cfg.k_shot
        self.q_query = cfg.q_query
        self.n_way   = n_way

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.classes_    = np.unique(np.concatenate([y_train, y_tune]))
        self._label2idx  = {c: i for i, c in enumerate(self.classes_)}
        self.n_classes_  = len(self.classes_)

        y_train_enc = np.array([self._label2idx[c] for c in y_train], dtype=np.int64)
        y_tune_enc  = np.array([self._label2idx[c] for c in y_tune],  dtype=np.int64)

        self._meta_train_pool = self._build_class_pool(X_train, y_train_enc, pool="meta_train")
        self._meta_tune_pool  = self._build_class_pool(X_tune,  y_tune_enc,  pool="meta_tune")

        self._train_available: dict[int, list[int]] = {}
        self._tune_available:  dict[int, list[int]] = {}
        self._reset_availability("train")
        self._reset_availability("tune")

        self._n_way_train  = n_way or len(self._meta_train_pool)
        self._n_way_tune   = n_way or len(self._meta_tune_pool)
        self._task_counter = 0

        print(f"[TaskSampler] classes      : {list(self.classes_)}")
        print(f"[TaskSampler] meta-train   | classes: {len(self._meta_train_pool)} "
              f"| total windows: {sum(len(v) for v in self._meta_train_pool.values())}")
        print(f"[TaskSampler] meta-tune    | classes: {len(self._meta_tune_pool)} "
              f"| total windows: {sum(len(v) for v in self._meta_tune_pool.values())}")
        print(f"[TaskSampler] episode      | {self._n_way_train}-way {self.k_shot}-shot "
              f"+ {self.q_query}-query per class")


    def _build_class_pool(
        self, X: np.ndarray, y: np.ndarray, pool: str
    ) -> dict[int, np.ndarray]:
        """{ encoded_label -> (N_windows, timesteps, channels) }"""
        class_pool = {}
        n_required = self.k_shot + self.q_query

        for label in np.unique(y):
            idx = np.where(y == label)[0]
            X_cls = X[idx].astype(np.float32)

            if len(X_cls) < n_required:
                print(f"  [skip] {pool}/class={self.classes_[label]} — "
                      f"only {len(X_cls)} windows, need {n_required}")
                continue

            class_pool[int(label)] = X_cls

        return class_pool


    def _reset_availability(self, split: str):
        pool  = self._meta_train_pool if split == "train" else self._meta_tune_pool
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
    ) -> Task:
        n_required = self.k_shot + self.q_query

        # Refresh any class that can no longer fill k_shot + q_query
        for label in list(available.keys()):
            if len(available[label]) < n_required:
                idx = list(range(len(pool[label])))
                random.shuffle(idx)
                available[label] = idx

        classes = random.sample(list(pool.keys()), k=n_way)

        sup_X_list, sup_y_list = [], []
        qry_X_list, qry_y_list = [], []

        for cls in classes:
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


    def sample_meta_train_batch(self, n_tasks: int) -> list[Task]:
        return [
            self._sample_task(self._meta_train_pool, self._train_available, self._n_way_train)
            for _ in range(n_tasks)
        ]

    def sample_meta_tune_batch(self, n_tasks: int = 1) -> list[Task]:
        return [
            self._sample_task(self._meta_tune_pool, self._tune_available, self._n_way_tune)
            for _ in range(n_tasks)
        ]

    def reset_epoch(self, split: str = "both"):
        """Call at the start of each epoch to reshuffle the without-replacement pool."""
        if split in ("train", "both"):
            self._reset_availability("train")
        if split in ("tune", "both"):
            self._reset_availability("tune")

    def decode_labels(self, encoded: torch.Tensor) -> np.ndarray:
        """Convert integer-encoded labels back to original gesture strings."""
        return self.classes_[encoded.cpu().numpy()]

    @property
    def tasks_per_epoch(self) -> dict[str, int]:
        """Max tasks drawable before the scarcest class exhausts."""
        def _min_tasks(pool, n_way):
            min_windows = min(len(v) for v in pool.values())
            return (min_windows // (self.k_shot + self.q_query)) * n_way
        return {
            "meta_train": _min_tasks(self._meta_train_pool, self._n_way_train),
            "meta_tune":  _min_tasks(self._meta_tune_pool,  self._n_way_tune),
        }
    