import learn2learn as l2l
import torch
import torch.nn as nn
from task_sampler import Task

# ─────────────────────────────────────────────
#  Wrap model with MAML (FOMAML via first_order=True)
# ─────────────────────────────────────────────
def build_maml_model(model: nn.Module, cfg) -> l2l.algorithms.MAML:
    """
    Wraps CNNLSTMClassifier with learn2learn's MAML.
    first_order=True  -->   FOMAML (no second-order graph, much cheaper)
    lr                -->   inner loop learning rate (cfg.inner_lr)
    """
    return l2l.algorithms.MAML(
        model,
        lr           = cfg.inner_lr,
        first_order  = True,           # FOMAML
        allow_nograd = True,           # safe for BatchNorm buffers
    )


# ─────────────────────────────────────────────
#  Inner loop  (support set)
# ─────────────────────────────────────────────
def inner_loop(
    maml:    l2l.algorithms.MAML,
    task:    Task,
    cfg,
    device:  torch.device,
    loss_fn: nn.Module,
) -> l2l.algorithms.MAML:
    """
    Clone the meta-model and adapt it on one task's support set.

    maml.clone() creates a task-local copy whose gradients flow back
    to the original model parameters automatically — no manual grad
    copying needed.

    Returns the adapted learner (used directly for query loss).
    """
    learner   = maml.clone()           # task-local copy, grad-linked to maml
    learner.train()

    support_X = task.support_X.to(device)
    support_y = task.support_y.to(device)

    for _ in range(cfg.inner_steps):
        loss = loss_fn(learner(support_X), support_y)
        learner.adapt(loss)            # FOMAML grad step — no graph unrolling

    return learner


# ─────────────────────────────────────────────
#  Full FOMAML training loop
# ─────────────────────────────────────────────
def train_fomaml(model, sampler, cfg, device):
    loss_fn  = nn.CrossEntropyLoss()
    maml     = build_maml_model(model, cfg)
    meta_opt = torch.optim.Adam(maml.parameters(), lr=cfg.outer_lr)

    maml.to(device)

    for epoch in range(cfg.meta_epochs):
        sampler.reset_epoch("both")
        maml.train()

        epoch_losses = []

        for __ in range(cfg.tasks_per_epoch):
            task_batch   = sampler.sample_meta_train_batch(n_tasks=cfg.tasks_per_epoch)
            meta_opt.zero_grad()

            batch_query_loss = torch.tensor(0.0, device=device)

            for task in task_batch:
                # inner loop: adapt clone on support set
                learner = inner_loop(maml, task, cfg, device, loss_fn)

                # outer loop: query loss on adapted learner
                learner.eval()
                query_X = task.query_X.to(device)
                query_y = task.query_y.to(device)

                query_loss        = loss_fn(learner(query_X), query_y)
                batch_query_loss  = batch_query_loss + query_loss
                epoch_losses.append(query_loss.item())

            # Average loss across tasks, then single meta-update
            (batch_query_loss / len(task_batch)).backward()
            meta_opt.step()

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[Epoch {epoch+1:>3}/{cfg.meta_epochs}] avg query loss: {mean_loss:.4f}")

    return maml
