#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gc

# -----------------------------
# Device
# -----------------------------
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Problem/data
# -----------------------------
L = 5  # L * L lattice
K = L * L
samplesload = np.load(f"Edwards-Anderson_model_L{L}_samples.npy")   # (K, N_total) in {0,1}
ismod       = np.load(f"Edwards-Anderson_model_L{L}.npy")           # pmf over 2**K

Ntrain = 10**3   # Number of training samples used
Ntest  = 10**5   # Number of generated samples for evaluation
N_time = 1       # Number of noising cycles
epsi   = 0.2     # Keep probability Pi(phi)

# For histogram generation
original_array = np.array([2**(K - i - 1) for i in range(K)], dtype=np.int64)
twopow2 = np.tile(original_array, (Ntest, 1))

# -----------------------------
# Time constants
# -----------------------------
Tsteps = N_time * K + 1      # total steps incl. t=0
S = Tsteps - 1               # last forward step index
Nconf = 2**K
batch_size = Ntrain

# -----------------------------
# Model
# -----------------------------
class CommonNeuralNet(nn.Module):
    """
    Input: neighbors (K-1) + u_onehot (K) + step (1)
    Output: 2 logits corresponding to a = -1 and a = +1
    """
    def __init__(self, input_dim, hidden_dim, site_dim, step_dim, depth=1):
        super().__init__()
        layers = []
        in_dim = input_dim + site_dim + step_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, x_neighbors, u_onehot, step_input):
        x_aug = torch.cat([x_neighbors, u_onehot, step_input], dim=1)
        h = self.backbone(x_aug)
        return self.out(h)

def step_feature(t):
    # t in {0, ..., S-1}
    denom = max(1, S - 1)
    return t.to(torch.float32).div(float(denom)).unsqueeze(1)

def ggm_loss_batch(mlp, x_neighbors, sigma_obs, z_equals_obs, u_onehot, step_input):
    """
    Learns:
        f_tilde_theta(x_{-u}, onehot(u), t)_a
        ~= P(Z_t = a | X_{t+1,-u} = x_{-u}, X_{t+1,u} = a)

    sigma_obs    : observed token a = X_{t+1,u} in {-1, +1}
    z_equals_obs : binary label 1{Z_t = sigma_obs}

    In this binary keep-or-replace setting:
        1{Z_t = sigma_obs} == 1{Z_t != phi}
    because whenever Z_t != phi, the observed token is exactly Z_t.
    """
    logits = mlp(x_neighbors, u_onehot, step_input)   # (B, 2)
    obs_idx = (sigma_obs == 1).long()                 # -1 -> 0, +1 -> 1
    chosen_logits = logits.gather(1, obs_idx.unsqueeze(1)).squeeze(1)
    return F.binary_cross_entropy_with_logits(chosen_logits, z_equals_obs)

def build_on_the_fly_batch(x0_pm1):
    """
    x0_pm1: (K, B) in {-1,+1} = original samples X0

    Produces a training tuple consistent with:
        f_tilde_theta(x_{-i_t}, onehot(i_t), t)_a
        ~= P(Z_t = a | X_{t+1,-i_t} = x_{-i_t}, X_{t+1,i_t} = a)

    Forward noise:
        P(Z_t = phi) = epsi
        P(Z_t = -1)  = P(Z_t = +1) = (1 - epsi)/2
    """
    B = x0_pm1.size(1)
    dev = x0_pm1.device
    dt = x0_pm1.dtype

    # Sample forward time t and active site u = i_t
    t = torch.randint(low=0, high=S, size=(B,), device=dev)   # t in {0, ..., S-1}
    u = torch.remainder(t, K)
    step_input = step_feature(t)

    # Build X_t directly from X_0
    idx = torch.arange(K, device=dev, dtype=torch.long).unsqueeze(0)  # (1, K)
    visits = torch.div(t.unsqueeze(1) - 1 - idx, K, rounding_mode='floor') + 1
    visits = torch.clamp(visits, min=0, max=N_time)

    keep_prob_prev = torch.pow(
        torch.as_tensor(float(epsi), device=dev, dtype=torch.float32),
        visits.to(torch.float32)
    ).clamp_(0.0, 1.0)

    keep_mask_prev = torch.bernoulli(keep_prob_prev).to(torch.bool)
    rand_bits_prev = (2 * torch.randint(0, 2, (B, K), device=dev, dtype=torch.int8) - 1).to(dt)
    state_t = torch.where(keep_mask_prev, x0_pm1.T, rand_bits_prev)   # X_t, shape (B, K)

    # One more forward update at site u to get X_{t+1,u}
    sigma_t = state_t.gather(1, u.unsqueeze(1)).squeeze(1)            # X_{t,u}

    keep_now = torch.bernoulli(
        torch.full((B,), float(epsi), device=dev, dtype=torch.float32)
    ).to(torch.bool)

    z_new = (2 * torch.randint(0, 2, (B,), device=dev, dtype=torch.int8) - 1).to(dt)

    # Observed token a = X_{t+1,u}
    sigma_obs = torch.where(keep_now, sigma_t, z_new)

    # Label for the observed head: 1 iff Z_t = sigma_obs
    z_equals_obs = (~keep_now).to(torch.float32)

    # Context is X_{t+1,-u} = X_{t,-u}
    mask_u = torch.ones(B, K, dtype=torch.bool, device=dev)
    mask_u[torch.arange(B, device=dev), u] = False
    x_neighbors = state_t[mask_u].view(B, K - 1)

    u_onehot = torch.zeros(B, K, device=dev, dtype=torch.float32)
    u_onehot[torch.arange(B, device=dev), u] = 1.0

    return x_neighbors, u_onehot, step_input, sigma_obs, z_equals_obs

@torch.no_grad()
def ggm_reverse_probs(sample, mlp, u, nsteps):
    """
    sample: (N, K), interpreted as X_hat_{t+1}
    Returns probabilities for X_hat_{t,u} in order [-1, +1].

    Uses:
        P_hat(X_{t,u}=a | X_{t+1,-u})
        = Pi(a)/Pi(phi) * (1 / y_hat_a - 1)
    """
    x_neighbors = torch.cat([sample[:, :u], sample[:, u+1:]], dim=1)

    u_onehot = torch.zeros((sample.shape[0], sample.shape[1]),
                           device=sample.device, dtype=sample.dtype)
    u_onehot[:, u] = 1.0

    tvec = torch.full((sample.shape[0],), nsteps, device=sample.device, dtype=torch.long)
    step_input = step_feature(tvec).to(sample.dtype)

    logits = mlp(x_neighbors, u_onehot, step_input)
    y = torch.sigmoid(logits).clamp_(1e-6, 1.0 - 1e-6)   # columns correspond to [-1, +1]

    pi_phi = float(epsi)
    pi_tok = 0.5 * (1.0 - float(epsi))

    probs = (pi_tok / pi_phi) * (1.0 / y - 1.0)
    probs = torch.clamp(probs, min=1e-12)
    probs = probs / probs.sum(dim=1, keepdim=True)

    return probs

# -----------------------------
# Hyperparams
# -----------------------------
hidden_dim = 120
input_dim = K - 1
site_dim = K
step_dim = 1

num_epochs = 2000
patience = 10
min_delta = 1e-5
lr = 0.000434213
weight_decay = 0.0

# -----------------------------
# Load training data
# -----------------------------
base = samplesload[:, :Ntrain].astype(np.float32)                    # (K, Ntrain) in {0,1}
Xflat_pm1 = torch.tensor((base * 2.0 - 1.0), device=device)          # (K, Ntrain) in {-1,+1}

# -----------------------------
# Train with on-the-fly mini-batches
# -----------------------------
mlp = CommonNeuralNet(input_dim, hidden_dim, site_dim, step_dim, depth=1).to(device)
opt = optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

best = float('inf')
counter = 0
start = time.time()
mlp.train()

for epoch in range(num_epochs):
    perm = torch.randperm(Ntrain, device=device)

    running = 0.0
    seen = 0
    for cyc in range(0, L * L):
        for i in range(0, Ntrain, batch_size):
            idx = perm[i:i + batch_size]
            x0_b = Xflat_pm1[:, idx]  # (K, B)

            x_neighbors, u_onehot, step_input, sigma_obs, z_equals_obs = build_on_the_fly_batch(x0_b)

            loss = ggm_loss_batch(
                mlp,
                x_neighbors,
                sigma_obs,
                z_equals_obs,
                u_onehot,
                step_input
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 5.0)
            opt.step()

            bs = x_neighbors.size(0)
            running += loss.item() * bs
            seen += bs

    epoch_loss = running / max(1, seen)
    print(f"Epoch {epoch+1:4d} | loss = {epoch_loss:.6f}")

    if best - epoch_loss > min_delta:
        best = epoch_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best loss: {best:.6f}")
            break

t_train = time.time() - start
print(f"Training time: {t_train:.2f}s, best loss: {best:.6f}")

# -----------------------------
# Reverse dynamics
# -----------------------------
start = time.time()
samples_neu = (2 * torch.randint(0, 2, (Ntest, K), device=device, dtype=torch.float32) - 1)

mlp.eval()
with torch.no_grad():
    for nsteps in reversed(range(0, Tsteps - 1)):
        j = nsteps % K

        probs = ggm_reverse_probs(samples_neu, mlp, j, nsteps)  # columns: [-1, +1]
        p_plus = probs[:, 1]

        new_is_plus = torch.rand(Ntest, device=device) < p_plus
        samples_neu[:, j] = torch.where(
            new_is_plus,
            torch.ones_like(samples_neu[:, j]),
            -torch.ones_like(samples_neu[:, j]),
        )

t_rev = time.time() - start
print(f"Reverse time: {t_rev:.2f}s")

# -----------------------------
# Metrics
# -----------------------------
samples_neu_np = samples_neu.detach().cpu().numpy().astype(np.int8)

samplabel_rev_neu = np.sum(
    ((samples_neu_np.astype(np.float32) * 0.5 + 0.5) * twopow2),
    axis=1
)
cvrev_neu = np.bincount(samplabel_rev_neu.astype(np.int64), minlength=Nconf) / Ntest
tv = np.sum(np.abs(ismod - cvrev_neu))

print(f"TV: {tv:.8f}")
