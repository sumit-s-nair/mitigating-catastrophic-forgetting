"""
Joint Trainer — Upper Bound for HGNN Continual Learning
=========================================================
Trains on T1 ∪ T2 simultaneously for the same total epoch budget.
This represents the performance ceiling: no forgetting since all
tasks are trained jointly.

Results serve as the reference for:
  - Recovery Ratio = mitigated_T1_acc / joint_T1_acc
  - Plasticity normalization = mitigated_T2_acc / joint_T2_acc
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

warnings.filterwarnings("ignore", category=FutureWarning)


def run_joint_training(model: nn.Module, data: Data, class_weights: torch.Tensor,
                       config_module, seed: int) -> dict:
    """Train on all 6 subreddits simultaneously (upper bound).

    Args:
        model: HGNN backbone (fresh, untrained)
        data: full hypergraph Data object
        class_weights: [C] class weight tensor
        config_module: config module
        seed: random seed

    Returns:
        dict with accuracies and training history
    """
    torch.manual_seed(seed)
    device = torch.device(config_module.DEVICE)
    model = model.to(device)
    data = data.to(device)
    cw = class_weights.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_module.LEARNING_RATE,
        weight_decay=config_module.WEIGHT_DECAY,
    )

    # Joint training: use ALL labeled training nodes
    all_labels = torch.tensor(config_module.ALL_LABELS, device=device)
    joint_mask = data.train_mask & torch.isin(data.y, all_labels)
    t1_labels = torch.tensor(config_module.T1_LABELS, device=device)
    t2_labels = torch.tensor(config_module.T2_LABELS, device=device)

    epochs = config_module.EPOCHS_PER_TASK * 2  # match total sequential budget (T1 + T2)
    history = {"loss": [], "t1_acc": [], "t2_acc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.hyperedge_index,
                       getattr(data, 'hyperedge_weight', None))

        loss = F.cross_entropy(logits[joint_mask], data.y[joint_mask], weight=cw)
        loss.backward()
        optimizer.step()

        # Evaluate per task
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.hyperedge_index,
                        getattr(data, 'hyperedge_weight', None))

            t1_mask = data.test_mask & torch.isin(data.y, t1_labels)
            t1_acc = 0.0
            if t1_mask.sum() > 0:
                t1_acc = (out[t1_mask].argmax(-1) == data.y[t1_mask]).float().mean().item()

            t2_mask = data.test_mask & torch.isin(data.y, t2_labels)
            t2_acc = 0.0
            if t2_mask.sum() > 0:
                t2_acc = (out[t2_mask].argmax(-1) == data.y[t2_mask]).float().mean().item()

        history["loss"].append(loss.item())
        history["t1_acc"].append(t1_acc)
        history["t2_acc"].append(t2_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  Loss: {loss.item():.4f}  "
                  f"T1: {t1_acc:.4f}  T2: {t2_acc:.4f}")

    # Final per-class evaluation
    model.eval()
    per_class = {}
    with torch.no_grad():
        out = model(data.x, data.hyperedge_index,
                    getattr(data, 'hyperedge_weight', None))
        for label_id in range(config_module.NUM_CLASSES):
            name = config_module.LABEL_NAMES.get(label_id, f"class_{label_id}")
            t = torch.tensor([label_id], device=device)
            mask = data.test_mask & torch.isin(data.y, t)
            if mask.sum() == 0:
                per_class[name] = 0.0
            else:
                per_class[name] = (out[mask].argmax(-1) == data.y[mask]).float().mean().item()

    return {
        "history": history,
        "t1_acc": history["t1_acc"][-1],
        "t2_acc": history["t2_acc"][-1],
        "per_class": per_class,
    }
