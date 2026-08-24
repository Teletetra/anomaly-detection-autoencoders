from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def incremental_update(
    model: nn.Module,
    loader: DataLoader,
    output_path: str | Path,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    device: str | None = None,
) -> None:
    """Fine-tune a trained model on a bounded batch of verified-normal windows."""
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for x in loader:
            x = x.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), x)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        print(f"incremental_epoch={epoch:02d} loss={total / max(count, 1):.6f}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "incremental_epochs": epochs}, path)
