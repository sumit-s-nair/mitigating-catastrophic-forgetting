import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import SAGEConv

# 1. Load Dataset (Only features/edges, not full computation graph)
dataset = Reddit(root='./data/reddit')
data = dataset[0]

# 2. Define Tasks (Grouping Reddit's 41 classes into 3 tasks)
# Task A: classes 0-13, Task B: 14-27, Task C: 28-40
tasks = [
    list(range(0, 14)),   # Task A
    list(range(14, 28)),  # Task B
    list(range(28, 41))   # Task C
]

# Simple GraphSAGE Model
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_features, 256)
        self.conv2 = SAGEConv(256, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# RandomNodeLoader partitions the graph into subgraphs — no torch-sparse/pyg-lib needed
# num_parts=40 gives ~5800 nodes per batch for Reddit's 232K nodes
train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True)
test_loader  = RandomNodeLoader(data, num_parts=40, shuffle=False)

# 3. Sequential Training (A -> B -> C)
for i, task_classes in enumerate(tasks):
    print(f"\n--- Training Task {chr(65+i)} (Classes {task_classes[0]}-{task_classes[-1]}) ---")
    task_tensor = torch.tensor(task_classes)

    model.train()
    for epoch in range(1, 4):  # Few epochs to see forgetting faster
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            local_task = task_tensor.to(device)
            mask = batch.train_mask & torch.isin(batch.y, local_task)
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[mask], batch.y[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch}, Loss: {total_loss / max(n_batches, 1):.4f}")

    # --- Evaluation for Forgetting ---
    model.eval()
    with torch.no_grad():
        for prev_idx in range(i + 1):
            prev_task_classes = tasks[prev_idx]
            prev_tensor = torch.tensor(prev_task_classes)
            correct, total = 0, 0
            for batch in test_loader:
                batch = batch.to(device)
                local_prev = prev_tensor.to(device)
                mask = batch.test_mask & torch.isin(batch.y, local_prev)
                if mask.sum() == 0:
                    continue
                out = model(batch.x, batch.edge_index)
                pred = out[mask].argmax(dim=-1)
                correct += (pred == batch.y[mask]).sum().item()
                total += mask.sum().item()
            acc = correct / max(total, 1)
            print(f"Accuracy on Task {chr(65+prev_idx)}: {acc:.4f}")