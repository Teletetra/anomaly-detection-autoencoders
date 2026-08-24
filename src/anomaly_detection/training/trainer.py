from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)


class AutoencoderTrainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip_norm: float = 1.0,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()
        self.gradient_clip_norm = gradient_clip_norm

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total_loss = 0.0
        total_items = 0
        for batch in loader:
            x = batch.to(self.device)
            with torch.set_grad_enabled(training):
                x_hat = self.model(x)
                loss = self.loss_fn(x_hat, x)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            total_loss += loss.item() * len(x)
            total_items += len(x)
        return total_loss / max(total_items, 1)

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        patience: int = 8,
        checkpoint_path: str | Path = "artifacts/checkpoints/best.pt",
    ) -> TrainingHistory:
        history = TrainingHistory()
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best = float("inf")
        stale = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(validation_loader, training=False)
            history.train_loss.append(train_loss)
            history.validation_loss.append(val_loss)
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best:
                best = val_loss
                stale = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "validation_loss": val_loss,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
                if stale >= patience:
                    print("early stopping")
                    break
        return history
