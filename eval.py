import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
    
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device) -> dict:
    """
    Evaluate the model and return loss, accuracy, and macro-F1.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == y_batch).sum().item()
        total_count += batch_size

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "macro_f1": macro_f1
    }


@torch.no_grad()
def evaluate_full(model: nn.Module, dataloader: DataLoader, device):
    """
    Evaluate the model on a full dataloader and return encoded true/pred labels.
    """
    model.eval()

    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(y_batch.numpy())
    
    return np.concatenate(all_labels), np.concatenate(all_preds)


def eval_cm(y_true,y_pred, target_names):
    num_classes = len(target_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize = "true")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, target_names, rotation=45, ha="right")
    plt.yticks(tick_marks, target_names)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
