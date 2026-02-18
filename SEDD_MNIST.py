#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import random, gc

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Parameters
# -----------------------------
L = 28
K = 784
N_time = 1
Tsteps = N_time * K + 1
Tf = 1.0
mask_token = 2.0



# Reverse sampler settings
Tsteps_rev = 160
dt = Tf / Tsteps_rev
Ntest = 100

n_classes  = 10
cond_dim   = n_classes

def to_onehot(y, n_classes=10):
    oh = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return oh

# -----------------------------
# Load MNIST (local npz) once
# -----------------------------
with np.load("mnist.npz") as data:
    X_train_all, y_train_all = data["x_train"], data["y_train"]
    X_test_all,  y_test_all  = data["x_test"],  data["y_test"]

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
Y_test_oh       = torch.tensor(Y_test_oh,       device=device)

def make_train_val_split(X, Y, val_frac=0.1, seed=0):
    N = X.size(0)
    gen = torch.Generator(device=X.device)
    gen.manual_seed(seed)
    perm = torch.randperm(N, generator=gen, device=X.device)
    n_val = int(N * val_frac)
    trn_idx = perm[n_val:]
    val_idx = perm[:n_val]
    return X[trn_idx], Y[trn_idx], X[val_idx], Y[val_idx]

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

X_tra, Y_tr, X_val, Y_val = make_train_val_split(Xflat_train_pm1, Y_train_oh, 0.1, 0)

# -----------------------------
# Noise Schedule (absorbing)
# -----------------------------
def log_linear_noise_schedule(t, epsilon=1e-3):
    return -torch.log(1 - (1 - epsilon) * t)

def log_linear_noise_derivative(t, epsilon=1e-3):
    return (1 - epsilon) / (1 - (1 - epsilon) * t)

# -----------------------------
# Model
# -----------------------------
class CommonNeuralNet(nn.Module):
    """
    Input: [x_t (K), class onehot (C), t (1)]
    Output: positive rates for 3 tokens per site -> shape (3K)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, depth=3):
        super().__init__()
        self.depth = depth
        self.fc1 = nn.Linear(input_dim , hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

        for m in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.out]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        if self.depth >= 2: x = self.act(self.ln2(self.fc2(x)))
        if self.depth >= 3: x = self.act(self.ln3(self.fc3(x)))
        if self.depth >= 4: x = self.act(self.ln4(self.fc4(x)))
        if self.depth >= 5: x = self.act(self.ln5(self.fc5(x)))
        return torch.exp(self.out(x))

# -----------------------------
# Loss (absorbing-mask variant)
# -----------------------------
def noise_on_fly(x_0, mask_token):
    N_batch = x_0.shape[1]
    K_batch = x_0.shape[0]
    t_sam = torch.rand(N_batch, dtype=torch.float32, device=device)
    sigma_t = log_linear_noise_schedule(t_sam)
    exp_neg_sigma = torch.exp(-sigma_t)

    probtoss = exp_neg_sigma.unsqueeze(0).repeat(K_batch, 1)
    toss = torch.rand((K_batch, N_batch), device=device)
    keep = (toss < probtoss).float()
    x_t = keep * x_0 + (1 - keep) * mask_token
    return x_t, t_sam

def compute_dwdse_loss_absorbing(mlp, x_0, x_class, mask_token=2.0, eps=1e-9):
    x_t, t_sam = noise_on_fly(x_0, mask_token)
    d, N = x_t.shape
    sigma_t = log_linear_noise_schedule(t_sam)
    r_t = torch.exp(-sigma_t) / (1 - torch.exp(-sigma_t) + eps)  # (N,)

    is_mask = (x_t == mask_token)                     # (d,N)
    x_t_input = x_t.float().T                         # (N,d)
    time_input = t_sam.unsqueeze(1)                   # (N,1)
    mlp_input = torch.cat([x_t_input, x_class, time_input], dim=1)  # (N, d+C+1)

    s_theta = mlp(mlp_input).view(N, 3, d).permute(1, 2, 0)  # (3,d,N)
    x0_long = torch.clamp(x_0.long(), min=0, max=1)          # (d,N) from {-1,+1} -> {0,1}
    idx = x0_long.unsqueeze(0)                                # (1,d,N)
    s_theta_target = torch.gather(s_theta, dim=0, index=idx).squeeze(0)  # (d,N)

    s_sum_01 = torch.sum(s_theta[0:2], dim=0)                # (d,N)

    mask_sel = is_mask
    r_t_selected = r_t.expand(d, N)[mask_sel]
    s_target_sel = s_theta_target[mask_sel]
    s_sum_01_sel = s_sum_01[mask_sel]

    entropy_term = -r_t_selected * torch.log(s_target_sel + eps)
    loss = 0.5 * (s_sum_01_sel + entropy_term).mean()
    return loss

def eval_epoch_loss(model, x0_src, xcl_src, batch_size, epsilon):
    model.eval()
    with torch.no_grad():
        N = xcl_src.shape[0]
        total = 0.0; seen = 0
        idxs = torch.arange(N, device=device)
        for i in range(0, N, batch_size):
            sel = idxs[i:i+batch_size]
            x0b = x0_src[:, sel]
            x_cb = xcl_src[sel]
            l = compute_dwdse_loss_absorbing(model, x0b, x_cb, mask_token=mask_token, eps=epsilon)
            total += l.item() * x_cb.shape[0]
            seen  += x_cb.shape[0]
    model.train()
    return total / max(1, seen)

# -----------------------------
# Train MLP (used by Hyperopt and final train)
# -----------------------------
def train_one(Xtr, Ytr, Xval, Yval, return_model=False):
    lr           = 0.0001770876
    weight_decay =  1e-6
    HD           = 480
    depth        = 5
    batch_size   = 64
    seed         = 0
    epsilon      =  1
  

    num_epochs = 10000

    # Build local training/val views
    x0_tr  = Xtr.T.contiguous()     # (K, Ntr)
    xcl_tr = Ytr.contiguous()       # (Ntr, C)
    x0_va  = Xval.T.contiguous()    # (K, Nval)
    xcl_va = Yval.contiguous()      # (Nval, C)

    model = CommonNeuralNet(K + n_classes + 1, HD, 3 * K, depth=depth).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_epoch = float("inf"), -1
    best_state = None
    start_time = time.time()

    Ntrain_act = Xtr.size(0)
    indices = torch.arange(Ntrain_act, device=device)

    for epoch in range(num_epochs):
        perm = indices[torch.randperm(Ntrain_act, device=device)]
        running = 0.0
        seen = 0

        for i in range(0, Ntrain_act, batch_size):
            sel = perm[i:i + batch_size]
            x0b = x0_tr[:, sel]
            x_cb = xcl_tr[sel]

            optimizer.zero_grad(set_to_none=True)
            loss = compute_dwdse_loss_absorbing(model, x0b, x_cb, mask_token=mask_token, eps=epsilon)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running += loss.item() * x_cb.shape[0]
            seen    += x_cb.shape[0]

        epoch_loss = running / max(1, seen)
        val_loss   = eval_epoch_loss(model, x0_va, xcl_va, batch_size, epsilon)

        if val_loss < best_val - 1e-6:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ((epoch + 1) % 25 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch+1:4d} | train {epoch_loss:.6f} | val {val_loss:.6f}")

    t_total = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    out = {"best_val": best_val, "best_epoch": best_epoch, "time": t_total, "epochs_run": epoch+1}
    if return_model:
        return model, out
    else:
        del model; torch.cuda.empty_cache(); gc.collect()
        return out


#%%
# -----------------------------
# Retrain final model
# -----------------------------

X_tr_final, Y_tr_final, X_va_final, Y_va_final = make_train_val_split(
    Xflat_train_pm1, Y_train_oh, val_frac=0.2)

model, final_out = train_one(X_tr_final, Y_tr_final, X_va_final, Y_va_final,
                             return_model=True)

print(f"Final best val: {final_out['best_val']:.6f} at epoch {final_out['best_epoch']+1}")

# %%
# -----------------------------
# Reverse dynamics (absorbing decode from mask)
# -----------------------------
def save_grid(images_hw, path, n=25, rows=5, cols=5):
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

print("  Reverse sampling from mask...")

with torch.no_grad():
    model.eval()
    for target_digit in range(n_classes):
        samples_neu = torch.full((Ntest, K), mask_token, device=device)  # start all masked
        # one-hot class condition
        samples_class = torch.zeros((Ntest, n_classes), device=device)
        samples_class[:, target_digit] = 1.0

        start = time.time()
        for nsteps in reversed(range(Tsteps_rev)):
            t = nsteps * dt
            t_tensor = torch.full((Ntest,), t, device=device)
            sigma_t = log_linear_noise_schedule(t_tensor)
            d_sigma_dt = log_linear_noise_derivative(t_tensor)

            time_input = t_tensor.unsqueeze(1)  # (Ntest,1)
            mlp_input = torch.cat([samples_neu, samples_class, time_input], dim=1)  # (Ntest, K+C+1)
            rates = model(mlp_input).view(Ntest, 3, K)  # (Ntest, 3, K)

            probs_01 = d_sigma_dt.view(-1, 1, 1) * dt * rates[:, :2, :]  # (N,2,K)
            sum_probs = probs_01.sum(dim=1, keepdim=True)                # (N,1,K)
            prob_stay = 1.0 - sum_probs
            probs_full = torch.cat([probs_01, prob_stay], dim=1)         # (N,3,K)
            probs_full = torch.clamp(probs_full, min=0.0, max=1.0)

            probs_for_sampling = probs_full.permute(0, 2, 1).contiguous()  # (N,K,3)
            dist = torch.distributions.Categorical(probs=probs_for_sampling.view(-1, 3))
            new_tokens = dist.sample().view(Ntest, K)                       # (N,K)

            is_mask = (samples_neu == mask_token)
            samples_neu = torch.where(is_mask, new_tokens.float(), samples_neu)

        t_rev = time.time() - start
        print(f"  Reverse time (digit {target_digit}): {t_rev:.2f}s")

        imgs = samples_neu.clone()
        imgs[imgs == mask_token] = 0.0
        imgs = imgs.view(Ntest, L, L).detach().cpu().numpy()
        save_grid(imgs, path=f"samples_digit{target_digit}_absorbing_MLP.png", n=25, rows=5, cols=5)

        out_npz = f"Gendata_digit{target_digit}_N{Ntest}_T{Tsteps_rev}_L{L}_HD{best['HD']}_absorbingMLP.npz"
        np.savez_compressed(out_npz, samples_gen=samples_neu.detach().cpu().numpy())
        print(f"Saved samples for digit {target_digit}.\n")
