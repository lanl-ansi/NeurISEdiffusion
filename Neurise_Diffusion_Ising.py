#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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
L = 5 # L * L Lattice
K = L * L
samplesload = np.load(f"Edwards-Anderson_model_L{L}_samples.npy")   # (K, N_total) in {0,1}
ismod       = np.load(f"Edwards-Anderson_model_L{L}.npy")           # pmf over 2**K

Ntrain = 10**4 # Number of training samples used
Ntest    = 10**5 # Number of training samples used
N_time  = 4 # Number of noising cycles
epsi = 0.2 # Noising parameter

# For histogram generation
original_array = np.array([2**(K - i - 1) for i in range(K)], dtype=np.int64)
twopow2 = np.tile(original_array, (Ntest, 1))


    
class CommonNeuralNet(nn.Module):
    """
    Input: neighbors (K-1) + u_onehot (K) + step (1)
    Output: 2 
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

    def forward(self, x, u_onehot, step_input):
        x_aug = torch.cat([x, u_onehot, step_input], dim=1)
        h = self.backbone(x_aug)
        return self.out(h)    


def neurise_loss_batch(mlp, x_neighbors, sigma_u, u_onehot, step_input):
    logits = mlp(x_neighbors, u_onehot, step_input)  # (B,2)
    phi_um = torch.where(sigma_u == -1, 0.5, -0.5)
    phi_up = torch.where(sigma_u ==  1, 0.5, -0.5)
    return torch.exp(-phi_um * logits[:, 0] - phi_up * logits[:, 1]).mean()


def build_on_the_fly_batch(x0_pm1):
    """
    x0_pm1: (B, K) in {-1,+1} = original image(s)
    Snapshot at random step s ∈ {0,...,S-1} (state after steps 0..s-1).
    Site updated each step is j = t % K; current site at time s is u = s % K.
    On each visit: keep with prob epsi; else refresh to fresh ±1.
    After c visits: P[state == original] = 0.5 + 0.5 * (epsi**c).
    """
    B = x0_pm1.size(1)
    dev = x0_pm1.device
    dt  = x0_pm1.dtype

    # sample snapshot time and current site (about to be updated)
    s = torch.randint(low=0, high=S, size=(B,), device=dev)           # (B,)
    u = torch.remainder(s, K)                                         # (B,)

    # step feature matches your convention
    step_input = (s.to(torch.float32) + 1.0).div(float(Tsteps - 1)).unsqueeze(1)

    # visits up to and INCLUDING step s-1
    idx = torch.arange(K, device=dev, dtype=torch.long).unsqueeze(0)  # (1,K)
    visits = torch.div(s.unsqueeze(1) - 1 - idx, K, rounding_mode='floor') + 1
    visits = torch.clamp(visits, min=0, max=N_time)                   # (B,K)

    # keep prob = epsi ** visits  (0**0 -> 1 is correct for unvisited)
    keep_prob = torch.pow(torch.as_tensor(epsi, device=dev, dtype=torch.float32),
                          visits.to(torch.float32)).clamp_(0.0, 1.0)
    keep_mask = torch.bernoulli(keep_prob).to(torch.bool)             # (B,K)

    # fresh ±1 for refreshed sites
    rand_bits = (2 * torch.randint(0, 2, (B, K), device=dev, dtype=torch.int8) - 1).to(dt)

    # assemble snapshot state
    state = torch.where(keep_mask, x0_pm1.T, rand_bits)               # (B,K)

    # build per-example inputs
    sigma_u = state.gather(1, u.unsqueeze(1)).squeeze(1)              # (B,)
    mask_u = torch.ones(B, K, dtype=torch.bool, device=dev)
    mask_u[torch.arange(B, device=dev), u] = False
    x_neighbors = state[mask_u].view(B, K - 1)                        # (B,K-1)

    u_onehot = torch.zeros(B, K, device=dev, dtype=torch.float32)
    u_onehot[torch.arange(B, device=dev), u] = 1.0

    return x_neighbors, u_onehot, step_input, sigma_u

# Energy compuation for reverse
@torch.no_grad()
def compute_energy(sample, mlp, u, nsteps):
    """
    sample: (N, K) float32 in {-1,+1}
    """
    sigma_u = sample[:, u]
    sigma_neighbors = torch.cat([sample[:, :u], sample[:, u+1:]], dim=1)
    u_onehot = torch.zeros((sample.shape[0], sample.shape[1]), device=sample.device, dtype=sample.dtype)
    u_onehot[:, u] = 1.0
    step_input = torch.full((sample.shape[0], 1),
                            float(nsteps) / float(Tsteps - 1),
                            device=sample.device, dtype=sample.dtype)
    logits = mlp(sigma_neighbors, u_onehot, step_input)
    phi_um = torch.where(sigma_u == -1, 0.5, -0.5)
    phi_up = torch.where(sigma_u ==  1, 0.5, -0.5)
    return phi_um * logits[:, 0] + phi_up * logits[:, 1]

# -----------------------------
# Hyperparams
# -----------------------------
hidden_dim = 100
input_dim  = K - 1
site_dim   = K
step_dim   = 1


num_epochs = 2000
patience   = 10
min_delta  = 1e-5
lr         = 0.00031213783834545955
weight_decay = 0.0

# -----------------------------
# Outer loops
# -----------------------------
sample_gen = np.zeros((Ntest,L*L,10,7),dtype = np.uint8)
sample_genTV = np.zeros((10,7), dtype = np.float32)
time_fwd = np.zeros((10,7), dtype = np.float32)
time_train = np.zeros((10,7), dtype = np.float32)
time_rev = np.zeros((10,7), dtype = np.float32)




     
 
Tsteps = N_time * K + 1      # total steps incl. t=0
S = Tsteps - 1               # last step index
Nconf = 2**K
batch_size = Ntrain        # mini-batch size for on-the-fly sampling
  # -----------------------------
  # Forward trajectories (unchanged logic; timing kept)
  # -----------------------------
start = time.time()
base = samplesload[:, :Ntrain].astype(np.float32)                    # (K, Ntrain) in {0,1}

# -----------------------------
# Training set (original states only; on-the-fly noising provides forward states)
# Keep on device as (K, Ntrain) float32 in {-1,+1}
# -----------------------------
Xflat_pm1 = torch.tensor((base * 2.0 - 1.0), device=device)          # (K, Ntrain)

# -----------------------------
# Train with on-the-fly mini-batches
# -----------------------------
mlp = CommonNeuralNet(input_dim, hidden_dim, site_dim, step_dim, depth=1).to(device)
opt = optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

best = float('inf'); counter = 0

mlp.train()

for epoch in range(num_epochs):
      
      perm = torch.randperm(Ntrain, device=device)

      running = 0.0; seen = 0
      for cyc in range(0,L*L): # Reuses data
          for i in range(0, Ntrain, batch_size):
              idx = perm[i:i+batch_size]
              
              x0_b = Xflat_pm1[:, idx]                                     # (K,B) in {-1,+1}
  
            
              x_neighbors, u_onehot, step_input, sigma_u = build_on_the_fly_batch(x0_b)
  
              loss = neurise_loss_batch(mlp, x_neighbors, sigma_u, u_onehot, step_input)
  
              opt.zero_grad(set_to_none=True)
              loss.backward()
              torch.nn.utils.clip_grad_norm_(mlp.parameters(), 5.0)
              opt.step()
  
              bs = x_neighbors.size(0)
              running += loss.item() * bs
              seen    += bs

      epoch_loss = running / max(1, seen)
      if best - epoch_loss > min_delta:
          best = epoch_loss
          counter = 0
      else:
          counter += 1
          if counter >= patience:
              print(f"  Early stopping at epoch {epoch+1}. Best loss: {best:.6f}")
              break

  
  

  # -----------------------------
  # Reverse dynamics (uses nsteps/(Tsteps-1) 
  # -----------------------------
  
samples_neu = (2 * torch.randint(0, 2, (Ntest, K), device=device, dtype=torch.float32) - 1)

mlp.eval()
with torch.no_grad():
      for nsteps in reversed(range(0, Tsteps - 1)):
          j = nsteps % K

          temp_comb1 = samples_neu.clone()
          temp_comb1[:, j] = -temp_comb1[:, j]  # flip j

          energy_flipped = compute_energy(temp_comb1,  mlp, j, nsteps)  # (Ntest,)
          energy_current = compute_energy(samples_neu, mlp, j, nsteps)  # (Ntest,)

          pn0 = torch.exp(energy_flipped)
          pn1 = torch.exp(energy_current)
    
          M_temp = (1 - epsi) / ((1 - epsi) + (1 + epsi) * pn1 / torch.clamp(pn0, min=1e-12))
          M_temp = torch.clamp(M_temp, 0.0, 1.0)

          flip_mask = torch.rand(Ntest, device=device) < M_temp
          samples_neu[flip_mask, j] = -samples_neu[flip_mask, j]

          

  # -----------------------------
  # Metrics 
  # -----------------------------
tv = np.sum(np.abs(ismod - cvrev_neu))  # Computed TV
samplabel_rev_neu = np.sum(((samples_neu.detach().cpu().numpy() * 0.5 + 0.5) * twopow2), axis=1)
cvrev_neu = np.bincount(samplabel_rev_neu.astype(int), minlength=2**K) / Ntest # Histogram info
