#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import random, gc

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Budgets
# ------------- ----------------

# -----------------------------
# Parameters
# -----------------------------
L = 28
K = L * L
Ntest = 10**2
N_time = 1
epsi = 0
Tsteps = N_time * K + 1
S = Tsteps - 1
q = 2  

n_classes = 10
cond_dim  = n_classes

# -----------------------------
# Data
# -----------------------------
with np.load("mnist.npz") as data:
    X_train_all, y_train_all = data["x_train"], data["y_train"]
    X_test_all,  y_test_all  = data["x_test"],  data["y_test"]

def to_onehot(y, n_classes=10):
    oh = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return oh

# binarize to {-1,+1}
X_train_bin01 = (X_train_all/255.0 > 0.5).astype(np.float32)
X_test_bin01  = (X_test_all /255.0 > 0.5).astype(np.float32)
Xflat_train_pm1 = (X_train_bin01.reshape(-1, K)*2.0 - 1.0).astype(np.float32)
Xflat_test_pm1  = (X_test_bin01.reshape(-1, K) *2.0 - 1.0).astype(np.float32)
Y_train_oh = to_onehot(y_train_all, n_classes)
Y_test_oh  = to_onehot(y_test_all,  n_classes)

Xflat_train_pm1 = torch.tensor(Xflat_train_pm1, device=device)
Xflat_test_pm1  = torch.tensor(Xflat_test_pm1,  device=device)
Y_train_oh      = torch.tensor(Y_train_oh,      device=device)

def make_train_val_split(X, Y, val_frac=0.1, seed=0):
    N = X.size(0)
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed)
    perm = torch.randperm(N, generator=gen, device=X.device)
    n_val = int(N * val_frac)
    trn_idx = perm[n_val:]
    val_idx = perm[:n_val]
    return X[trn_idx], Y[trn_idx], X[val_idx], Y[val_idx]

X_tra, Y_tr, X_val, Y_val = make_train_val_split(Xflat_train_pm1, Y_train_oh, 0.1, 0)

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Model 
# -----------------------------
class CommonNeuralNet(nn.Module):
    """
    Input: neighbors (K-1) + u_onehot (K) + step (1) + y_onehot (10)
    Output: 2 logits for {-1, +1}
    """
    def __init__(self, input_dim, hidden_dim, site_dim, step_dim, cond_dim, depth=3):
        super().__init__()
        self.depth = int(depth)  # 1..5
        self.fc1 = nn.Linear(input_dim + site_dim + step_dim + cond_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.act = nn.SiLU()
        for m in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.out]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x_neighbors, u_onehot, step_input, y_onehot):
        x = torch.cat([x_neighbors, u_onehot, step_input, y_onehot], dim=1)
        x = self.act(self.ln1(self.fc1(x)))
        if self.depth >= 2: x = self.act(self.ln2(self.fc2(x)))
        if self.depth >= 3: x = self.act(self.ln3(self.fc3(x)))
        if self.depth >= 4: x = self.act(self.ln4(self.fc4(x)))
        if self.depth >= 5: x = self.act(self.ln5(self.fc5(x)))
        return self.out(x)

def neurise_loss_batch(x_neighbors, u_onehot, step_input, sigma_u, y_onehot, mlp, q=2):
    logits = mlp(x_neighbors, u_onehot, step_input, y_onehot)  # (B,2)
    phi_um = torch.where(sigma_u == -1, 1 - 1/q, -1/q)
    phi_up = torch.where(sigma_u ==  1, 1 - 1/q, -1/q)
    dot = phi_um * logits[:, 0] + phi_up * logits[:, 1]
    return torch.exp(-dot).mean()

# -----------------------------
# On-the-fly forward-state sampler
# -----------------------------
def build_on_the_fly_batch(x0_pm1):
    """
    x0_pm1: (B, K) in {-1,+1}
    """
    B = x0_pm1.size(0)
    dev = x0_pm1.device
    dt  = x0_pm1.dtype

    s = torch.randint(low=0, high=S, size=(B,), device=dev)   # (B,)
    u = torch.remainder(s, K)                                 # (B,)

    # Keep step feature consistent with reverse: s/(Tsteps-1)
    step_input = (s.to(torch.float32)).div(float(Tsteps - 1)).unsqueeze(1)  # (B,1)

    idx = torch.arange(K, device=dev, dtype=torch.long).unsqueeze(0)        # (1,K)
    visits = torch.div(s.unsqueeze(1) - 1 - idx, K, rounding_mode='floor') + 1
    visits = torch.clamp(visits, min=0, max=N_time)                          # (B,K)

    keep_prob = torch.pow(torch.as_tensor(epsi, device=dev, dtype=torch.float32),
                          visits.to(torch.float32)).clamp_(0.0, 1.0)
    keep_mask = torch.bernoulli(keep_prob).to(torch.bool)

    rand_bits = (2 * torch.randint(0, 2, (B, K), device=dev, dtype=torch.int8) - 1).to(dt)
    state = torch.where(keep_mask, x0_pm1, rand_bits)                         # (B,K)

    sigma_u = state.gather(1, u.unsqueeze(1)).squeeze(1)                      # (B,)
    mask_u = torch.ones(B, K, dtype=torch.bool, device=dev)
    mask_u[torch.arange(B, device=dev), u] = False
    x_neighbors = state[mask_u].view(B, K - 1)                                # (B,K-1)

    u_onehot = torch.zeros(B, K, device=dev, dtype=torch.float32)
    u_onehot[torch.arange(B, device=dev), u] = 1.0
    return x_neighbors, u_onehot, step_input, sigma_u

def save_grid(images_hw, path, n=25, rows=5, cols=5):
    """
    images_hw: (N, L, L) in [0,1] or {0,1}
    Saves a rows x cols grid to `path`.
    """
    n = min(n, images_hw.shape[0], rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(images_hw[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid to {path}")

# -----------------------------
# Energy for reverse dynamics 
# -----------------------------
@torch.no_grad()
def compute_energy(sample, mlp, u, nsteps, y_onehot, q=2):
    """
    sample: (N_local, K) in {-1,+1} (float32)
    """
    N_local = sample.shape[0]
    sigma_u = sample[:, u]  # (N_local,)
    sigma_neighbors = torch.cat([sample[:, :u], sample[:, u+1:]], dim=1)  # (N_local, K-1)

    step_frac  = float(nsteps) / float(Tsteps - 1)
    step_input = torch.full((N_local, 1), step_frac, device=sample.device, dtype=sample.dtype)

    u_onehot = torch.zeros(N_local, K, device=sample.device, dtype=sample.dtype)
    u_onehot[:, u] = 1.0

    logits = mlp(sigma_neighbors, u_onehot, step_input, y_onehot)  # (N_local, 2)
    phi_um = torch.where(sigma_u == -1, 1 - 1/q, -1/q)
    phi_up = torch.where(sigma_u ==  1, 1 - 1/q, -1/q)
    energy = phi_um * logits[:, 0] + phi_up * logits[:, 1]
    return energy

# -----------------------------
# Validation loss
# -----------------------------
@torch.no_grad()
def eval_epoch_loss(model, X, Y, batch_size, q):
    model.eval()
    N = X.size(0)
    running, seen = 0.0, 0
    for i in range(0, N, batch_size):
        x0_b = X[i:i+batch_size]
        y0_b = Y[i:i+batch_size]
        xb, uoh, t, sig = build_on_the_fly_batch(x0_b)
        loss = neurise_loss_batch(xb, uoh, t, sig, y0_b, model, q=q)
        running += loss.item() * x0_b.size(0)
        seen    += x0_b.size(0)
    return running / max(1, seen)





# -----------------------------
# Train Model
# -----------------------------

X_tr_final, Y_tr_final, X_va_final, Y_va_final = make_train_val_split(
    Xflat_train_pm1, Y_train_oh,val_frac=0.1 )

lr           = 5e-4
weight_decay = 1e-6
HD           = 192
depth        = 3
batch_size   = 256



num_epochs   = 5000
min_delta    = 1e-6   
patience     = 20  

model = CommonNeuralNet(input_dim=K-1, hidden_dim=HD, site_dim=K, step_dim=1,
                        cond_dim=cond_dim, depth=depth).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

best_val, best_epoch = float("inf"), -1
best_state = None

Xtr = X_tr_final
Ytr = Y_tr_final



Ntr = Xtr.size(0)
t0 = time.time()
for epoch in range(num_epochs):
    model.train()
    perm = torch.randperm(Ntr, device=device)
    running, seen = 0.0, 0
    for i in range(0, Ntr, batch_size):
       for cyc in range(5):   # Reusing data
        idx_b = perm[i:i+batch_size]
        x0_b  = Xtr[idx_b]
        y0_b  = Ytr[idx_b]
        xb, uoh, t, sig = build_on_the_fly_batch(x0_b)
        loss = neurise_loss_batch(xb, uoh, t, sig, y0_b, model, q=q)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running += loss.item() * x0_b.size(0)
        seen    += x0_b.size(0)

    train_loss = running / max(1, seen)
    val_loss   = eval_epoch_loss(model, X_val, Y_val, batch_size, q)
    if val_loss < best_val - min_delta:
        best_val, best_epoch = val_loss, epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if ((epoch + 1) % 25 == 0 or epoch == num_epochs - 1):
        print(f"Epoch {epoch+1:4d} | train {train_loss:.6f} | val {val_loss:.6f}")

#%%

# -----------------------------
# Reverse dynamics (conditional sampling)
# -----------------------------
with torch.no_grad():
    for digit in range(0, 10):
        # float32 noise in {-1,+1}
        samples_neu = (2 * torch.randint(0, 2, (Ntest, K), device=device, dtype=torch.int8) - 1).to(torch.float32)

        # class condition
        y_testp_oh = torch.zeros(1, n_classes, device=device, dtype=torch.float32)
        y_testp_oh[0, digit] = 1.0
        y_test_oh = y_testp_oh.repeat(Ntest, 1)

        start = time.time()
        model.eval()
        for nsteps in reversed(range(0, Tsteps - 1)):
            j = nsteps % K
            temp_comb1 = samples_neu.clone()
            temp_comb1[:, j] = -temp_comb1[:, j]  # flip j

            energy_flipped = compute_energy(temp_comb1,  model, j, nsteps, y_test_oh, q=q)
            energy_current = compute_energy(samples_neu, model, j, nsteps, y_test_oh, q=q)

            pn0 = torch.exp(energy_flipped)
            pn1 = torch.exp(energy_current)
            safe_den = torch.clamp(pn0, min=1e-6)
            M_temp = (1 - epsi) / ((1 - epsi) + (1 + epsi) * pn1 / safe_den)
            M_temp = torch.clamp(M_temp, 0.0, 1.0)

            probs = torch.stack([M_temp, 1 - M_temp], dim=1)  # [flip, stay]
            picks = torch.distributions.Categorical(probs=probs).sample()
            flip_mask = (picks == 0)
            samples_neu[flip_mask, j] = -samples_neu[flip_mask, j]
        t_rev = time.time() - start
        print(f"  Reverse time: {t_rev:.2f}s")

        images = (samples_neu / 2.0 + 0.5).detach().cpu().numpy().reshape(-1, L, L)
        
        save_grid(images, path=f"samples_shared_digit{digit}_e{epsi}_HD_condMLPopt.png",
                  n=25, rows=5, cols=5)

        filename = f"Gendata_e{epsi}_N{Ntest}_T{Tsteps}_L{L}_HD_condMLPopt_digit{digit}.npz"
        np.savez_compressed(filename, samples_gen=samples_neu.detach().cpu().numpy())
        print(f"Saved grids & data for digit {digit}.")
