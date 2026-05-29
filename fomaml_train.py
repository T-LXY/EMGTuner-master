import learn2learn as l2l
import torch
import torch.nn as nn
from task_sampler import Task

#  Wrap model with MAML (FOMAML via first_order=True)
def build_maml_model(model: nn.Module, cfg) -> l2l.algorithms.MAML:
    """
    Wraps CNNLSTMClassifier with learn2learn's MAML
    first_order=True    : FOMAML (no second-order graph, much cheaper)
    lr                  : inner loop learning rate (cfg.inner_lr)
    """
    return l2l.algorithms.MAML(
        model,
        lr           = cfg.inner_lr,
        first_order  = True,           # FOMAML
        allow_nograd = True,           # safe for BatchNorm buffers
    )


def inner_loop(
    maml:    l2l.algorithms.MAML,
    task:    Task,
    cfg,
    device:  torch.device,
    loss_fn: nn.Module,
) -> l2l.algorithms.MAML:
    learner   = maml.clone()           # task-local copy, grad-linked to maml
    learner.train()

    support_X = task.support_X.to(device)
    support_y = task.support_y.to(device)

    for _ in range(cfg.inner_steps):
        loss = loss_fn(learner(support_X), support_y)
        learner.adapt(loss)            # FOMAML grad step — no graph unrolling

    return learner


#  Full FOMAML training loop
def train_fomaml(model, sampler, cfg, device):
    loss_fn  = nn.CrossEntropyLoss()
    maml     = build_maml_model(model, cfg)
    # apply l2 regularization to hopefully curve overfitting (even under fine tuning lol)
    meta_opt = torch.optim.Adam(maml.parameters(), lr=cfg.outer_lr, weight_decay=1e-3)
    maml.to(device)

    for epoch in range(cfg.meta_epochs):
        sampler.reset_epoch("both")
        maml.train()
        meta_opt.zero_grad()

        batch_query_loss = torch.tensor(0.0, device=device)
        epoch_losses = []

        # sample separate batches for inner (meta_train) and outer (meta_tune)
        train_tasks = sampler.sample_meta_train_batch(n_tasks=cfg.tasks_per_epoch)
        tune_tasks  = sampler.sample_meta_tune_batch(n_tasks=cfg.tasks_per_epoch)

        for train_task, tune_task in zip(train_tasks, tune_tasks):
            # inner loop: adapt on general training data
            learner = inner_loop(maml, train_task, cfg, device, loss_fn)
            learner.eval()

            # outer loop: evaluate on fine-tune subject data
            query_X = tune_task.query_X.to(device)
            query_y = tune_task.query_y.to(device)

            query_loss       = loss_fn(learner(query_X), query_y)
            batch_query_loss = batch_query_loss + query_loss
            epoch_losses.append(query_loss.item())

        (batch_query_loss / len(train_tasks)).backward()
        meta_opt.step()
        
        # eval on held-out data
        eval_tasks = sampler.sample_meta_tune_batch(n_tasks=10)
        eval_losses = []

        for task in eval_tasks:
            learner = inner_loop(maml, task, cfg, device, loss_fn)
            learner.eval()
            with torch.no_grad():
                q_loss = loss_fn(
                    learner(task.query_X.to(device)),
                    task.query_y.to(device)
                ).item()
            eval_losses.append(q_loss)

        eval_loss = sum(eval_losses) / len(eval_losses)
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[Epoch {epoch+1:>3}/{cfg.meta_epochs}] avg query loss: {mean_loss:.4f} | avg eval loss: {eval_loss:.4f}")
        
    return maml
