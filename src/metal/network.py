"""The multilayer perceptron trained on EMPD.

One network per property, trained from scratch on each partition and each
initialization, and scored in R^2 on the partition's test rows
(sec:predictors).

The loop keeps the order in which the random-number streams are consumed:
seed, scale, build loaders, build the network, train, score. The EMPD values
the manuscript prints were produced in that order, and a
different order draws different initial weights and batches from the same seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# The locked configuration (sec:predictors). CHOSEN before any partition here
# was drawn, and not tuned on them: two hidden layers with batch normalisation,
# ReLU and dropout, AdamW, a learning rate halved when the loss over the
# early-stopping rows plateaus, the gradient norm clipped, and early stopping on
# that loss.
MLP_CONFIG = {
    "hidden_dim": 128,
    "task_hidden_dim": 64,
    "dropout": 0.3,
    "epochs": 200,
    "patience": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "lr_factor": 0.5,
    "lr_patience": 10,
    "max_grad_norm": 1.0,
}


class Network(nn.Sequential):
    """Two hidden layers, each Linear -> BatchNorm -> ReLU -> Dropout, then a
    linear output. Layers are built in this order, which fixes which initial
    weights each draws from the seeded stream."""

    def __init__(self, n_features: int, config: dict = MLP_CONFIG):
        layers, width = [], n_features
        for hidden in (config["hidden_dim"], config["task_hidden_dim"]):
            layers += [nn.Linear(width, hidden), nn.BatchNorm1d(hidden),
                       nn.ReLU(), nn.Dropout(config["dropout"])]
            width = hidden
        layers.append(nn.Linear(width, 1))
        super().__init__(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(-1)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    """The device a dataset's values were produced on, or a loud failure.

    The same seeds on another device follow a different trajectory in every
    cell, so a run that fell back from CUDA to the CPU would return values that
    look like the committed ones and are not.
    """
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "These values were produced on CUDA, which is not available here. "
            "The CPU gives a different trajectory per cell, not a rounded copy.")
    return torch.device(name)


def scaled_task(parts, feature_cols, target_col, mask_col, target_scaler):
    """Features and one property's target, scaled on the training part.

    Features are scaled by median and interquartile range over every training
    row, whichever property it carries. The target is scaled over the training
    rows that carry it. Both scalers are fitted on the training part alone and
    applied unchanged to the early-stopping and test parts. Returns one
    (features, target) pair per part, restricted to the rows carrying the
    property.
    """
    features = RobustScaler()
    target = target_scaler()
    out = []
    for i, df in enumerate(parts):
        X = df[feature_cols].values.astype(np.float32)
        X = (features.fit_transform(X) if i == 0 else features.transform(X))
        mask = df[mask_col].values.astype(bool)
        y = np.nan_to_num(df[target_col].values.copy().astype(np.float32), nan=0.0)
        y_task = y[mask].reshape(-1, 1)
        y[mask] = (target.fit_transform(y_task) if i == 0
                   else target.transform(y_task)).flatten()
        out.append((X.astype(np.float32)[mask], y[mask]))
    return out


def _loader(X, y, batch_size, train):
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      drop_last=train and len(dataset) > batch_size)


def _train(model, train_loader, stop_loader, config, device):
    """Early stopping on the loss over the early-stopping rows, best weights
    reloaded."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config["lr_patience"],
        factor=config["lr_factor"])
    loss_fn = nn.MSELoss()
    best_stop_loss, best_state, waited = float("inf"), None, 0
    for _ in range(config["epochs"]):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(features.to(device)), targets.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),
                                     max_norm=config["max_grad_norm"])
            optimizer.step()

        model.eval()
        stop_loss = 0.0
        with torch.no_grad():
            for features, targets in stop_loader:
                stop_loss += loss_fn(model(features.to(device)),
                                     targets.to(device)).item()
        stop_loss /= max(1, len(stop_loader))
        scheduler.step(stop_loss)

        if stop_loss < best_stop_loss:
            best_stop_loss, waited = stop_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config["patience"]:
                break
    model.load_state_dict(best_state)
    return model


def fit_and_score(parts, feature_cols, target_col, mask_col, target_scaler,
                  init_seed, device, config=MLP_CONFIG) -> float:
    """Train one network on one partition from one initialization.

    ``parts`` is the (training, early-stopping, test) triple of a partition.
    Returns the R^2 over the test rows carrying the property, computed
    against their own mean as scikit-learn computes it. The target scaler is
    affine, so the score equals the score on the unscaled target.
    """
    set_seed(init_seed)
    (X_tr, y_tr), (X_stop, y_stop), (X_te, y_te) = scaled_task(
        parts, feature_cols, target_col, mask_col, target_scaler)
    batch = config["batch_size"]
    train_loader = _loader(X_tr, y_tr, batch, train=True)
    stop_loader = _loader(X_stop, y_stop, batch, train=False)
    test_loader = _loader(X_te, y_te, batch, train=False)
    model = Network(len(feature_cols), config).to(device)
    model = _train(model, train_loader, stop_loader, config, device)

    model.eval()
    predicted, observed = [], []
    with torch.no_grad():
        for features, targets in test_loader:
            predicted.append(model(features.to(device)).cpu().numpy())
            observed.append(targets.numpy())
    return float(r2_score(np.concatenate(observed), np.concatenate(predicted)))
