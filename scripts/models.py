"""PyTorch models and training loops shared by the training and explanation scripts."""
import numpy as np
import torch
import torch.nn as nn

from common import SEED


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(256, 64), dropout=0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(self, n_in, latent=16, hidden=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_in), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def _batches(n, batch_size, rng):
    idx = rng.permutation(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def train_mlp(X, y, n_classes, epochs=80, lr=1e-3, weight_decay=1e-4, batch_size=64,
              input_mask_max=0.0, seed=SEED, class_weight=None):
    """Train an MLP classifier. input_mask_max > 0 randomly zeroes a fraction of
    input CpGs per batch (coverage augmentation, see the sparse-coverage experiment)."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    y = torch.as_tensor(np.asarray(y, dtype=np.int64))
    model = MLP(X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = None if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    model.train()
    for _ in range(epochs):
        for b in _batches(len(X), batch_size, rng):
            xb = X[b]
            if input_mask_max > 0:
                frac = rng.uniform(0, input_mask_max)
                keep = torch.as_tensor(rng.random(xb.shape[1]) >= frac, dtype=torch.float32)
                xb = xb * keep
            opt.zero_grad()
            loss_fn(model(xb), y[b]).backward()
            opt.step()
    model.eval()
    return model


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(np.asarray(X, dtype=np.float32)))
        return torch.softmax(logits, dim=1).numpy()


def train_autoencoder(X, latent=16, epochs=150, lr=1e-3, batch_size=64, seed=SEED):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    model = AutoEncoder(X.shape[1], latent=latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for b in _batches(len(X), batch_size, rng):
            opt.zero_grad()
            recon, _ = model(X[b])
            loss_fn(recon, X[b]).backward()
            opt.step()
    model.eval()
    return model


def encode(model, X):
    model.eval()
    with torch.no_grad():
        return model.encoder(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy()
