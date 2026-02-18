import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Noise Schedule ---
def log_linear_noise_schedule(t, epsilon=1e-3):
    # σ(t) = -log(1 - (1-ε) t),  t in [0,1]
    return -torch.log(1 - (1 - epsilon) * t)

def log_linear_noise_derivative(t, epsilon=1e-3):
    # dσ/dt = (1-ε) / (1 - (1-ε) t)
    denom = 1 - (1 - epsilon) * t
    denom = torch.clamp(denom, min=1e-6)  # stabilize
    return (1 - epsilon) / denom

# --- Parameters ---
L = 5
Tf = 1.0
Tsteps = 160
dt = Tf / Tsteps
mask_token = 2
hidden_dim = 344
Ntest = 10**5
depth = 2
epsilon = 1  # kept for schedule

Ntrain = 10**2

original_array = np.array([2**(L*L-i-1) for i in range(L*L)])

# --- Data ---
samplesload = np.load(f"Edwards-Anderson_model_L{L}_samples.npy")  # shape: (L*L, Ntrain_total)
ismod = np.load(f"Edwards-Anderson_model_L{L}.npy")                # shape: (2**(L*L),)

# 
def tokens_to_onehot(x_tokens: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """
    x_tokens: (N, d) integer tensor with values in {0,1,2} where 2 = mask.
    Returns: (N, d*num_classes) float one-hot flattened as [site0(3), site1(3), ...].
    """

    x_tokens = x_tokens.clamp(0, num_classes - 1)
    oh = F.one_hot(x_tokens, num_classes=num_classes).float()
    return oh.view(x_tokens.size(0), -1)

# --- Model(s) ---
class MLPVariableDepth(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Positive rates
        return F.softplus(self.net(x)) + 1e-6

# --- Noise & Loss ---
@torch.no_grad()
def noise_onfly(x0_batch, t_min=1e-3):
    """
    x0_batch: Float or int tensor of shape (d, N_batch) with entries in {0,1} (no masks)
    Returns x_t (masked with 'mask_token') and sampled times t
    """
    d, N_batch = x0_batch.shape
 
    t = (t_min + (Tf - t_min) * torch.rand(N_batch, device=device)).clamp(0.0, Tf)
    sigma_t = log_linear_noise_schedule(t)        # (N_batch,)
    exptotnoise = torch.exp(-sigma_t)             # (N_batch,), equals 1 - (1-ε) t
    exptotnoise = torch.clamp(exptotnoise, max=1 - 1e-6)

    prob_keep = exptotnoise.repeat(d, 1)          # (d, N_batch)
    toss = torch.rand((d, N_batch), device=device)
    keep = (toss < prob_keep).float()

    x0f = x0_batch.float()
    x_t = keep * x0f + (1 - keep) * float(mask_token)
    return x_t, t

def compute_dwdse_loss_absorbing(mlp, x0_batch, mask_token):
    """
    x0_batch: (d, N_batch), entries 0/1
    """
    
    tries = 0
    while True:
        x_t, t = noise_onfly(x0_batch)
        is_mask = (x_t == float(mask_token))
        if is_mask.any() or tries >= 3:
            break
        tries += 1

    d, N = x_t.shape
    sigma_t = log_linear_noise_schedule(t)
    expt = torch.exp(-sigma_t).clamp(max=1 - 1e-6)
    r_t = expt / (1 - expt + 1e-6)       # stabilize

    # === One-hot inputs ===
    # x_t: (d, N) -> (N, d) longs -> one-hot -> (N, 3d)
    x_t_tokens = x_t.T.long()
    x_t_oh = tokens_to_onehot(x_t_tokens, num_classes=3)  # (N, 3d)
    time_input = t.unsqueeze(1)                           # (N, 1)
    mlp_input = torch.cat([x_t_oh, time_input], dim=1)    # (N, 3d+1)

    # Predict per-site, 3-class rates
    s_theta = mlp(mlp_input).view(N, 3, d).permute(1, 2, 0)  # (3, d, N)

    # Target “channel” is the true token at x0 (0 or 1). 
    x0_long = x0_batch.long()
    s_theta_target = torch.gather(s_theta, dim=0, index=x0_long.unsqueeze(0)).squeeze(0)  # (d, N)

    # Only masked sites contribute to loss
    if is_mask.any():
        s1 = torch.sum(s_theta[0:2], dim=0)[is_mask]  # total non-mask rate
        r_t_selected = r_t.expand(d, N)[is_mask]
        entropy_term = -r_t_selected * torch.log(s_theta_target[is_mask] + 1e-9)
        loss = 0.5 * (s1 + entropy_term).mean()
        loss = (loss * sigma_t.view(1, N).expand(d, N)[is_mask]).mean()
    else:
        loss = (s_theta.sum() * 0.0) + torch.tensor(1e-8, device=device)

    return loss

# --- Containers for results ---
sample_gen = np.zeros((Ntest,L*L,10,7),dtype = np.uint8)
sample_gen_masked = np.zeros((Ntest,L*L,10,7),dtype = np.uint8)
sample_genTV = np.zeros((10,7), dtype = np.float32)
time_fwd = np.zeros((10,7), dtype = np.float32)
time_train = np.zeros((10,7), dtype = np.float32)
time_rev = np.zeros((10,7), dtype = np.float32)

slen = L * L
Nconf = 2 ** (L * L)

# Precompute weights for labeling configs
twopow2_np = original_array.astype(np.float32)                # (d,)
twopow2 = torch.tensor(twopow2_np, device=device).unsqueeze(0)  # (1, d)



# Load training slice: (d, Ntrain)
samples = samplesload[:, :Ntrain]
x_0 = torch.tensor(samples, device=device).float()

# Model/opt  === input_dim changes to 3*L*L + 1 ===
model = MLPVariableDepth(3*L*L + 1, hidden_dim, 3 * L * L, depth).to(device)
optimizer = optim.Adam(model.parameters(), lr=3.1213783834545955e-4)

patience, best_loss, epochs_no_improve, max_epochs = 20, float('inf'), 0, 5000

start_train = time.time()
batch_size = Ntrain  # practical batching
for epoch in range(max_epochs):
    model.train()
    perm = torch.randperm(Ntrain, device=device)
    current_loss = 0.0
    for cyc in range(0,L*L):
        for start_idx in range(0, Ntrain, batch_size):
            idx = perm[start_idx:start_idx + batch_size]
            loss = compute_dwdse_loss_absorbing(model, x_0[:, idx], mask_token)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

    if current_loss < best_loss - 1e-5:
        best_loss = current_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        break

t_train = time.time() - start_train
#%%
# --- Reverse Process ---
model.eval()
samples_neu = torch.full((Ntest, L * L), float(mask_token), device=device)

start_rev = time.time()
with torch.no_grad():
    for nsteps in reversed(range(Tsteps)):
        t = nsteps * dt
        t_tensor = torch.full((Ntest,), t, device=device)
        sigma_t = log_linear_noise_schedule(t_tensor)
        d_sigma_dt = log_linear_noise_derivative(t_tensor)

        # === One-hot inputs for reverse ===
        x_tokens = samples_neu.long()                      # (Ntest, d)
        x_oh = tokens_to_onehot(x_tokens, num_classes=3)   # (Ntest, 3d)
        time_input = t_tensor.unsqueeze(1)                  # (Ntest, 1)
        mlp_input = torch.cat([x_oh, time_input], dim=1)    # (Ntest, 3d+1)

        score = model(mlp_input).view(Ntest, 3, L * L)      # positive rates

        # Proposed probabilities for 0/1 transitions from mask
        rates_01 = (d_sigma_dt.view(-1, 1, 1) * dt) * score[:, :2, :]  # (N, 2, d)
        rates_01 = torch.clamp(rates_01, min=0.0)

        sum_rates = rates_01.sum(dim=1, keepdim=True)       # (N,1,d)
        prob_stay = 1.0 - sum_rates
        prob_stay = torch.clamp(prob_stay, min=0.0)

        probs_full = torch.cat([rates_01, prob_stay], dim=1)  # (N,3,d)

        # Ensure valid distribution per site: if all-zero, set stay=1; else normalize
        sums = probs_full.sum(dim=1, keepdim=True)            # (N,1,d)
        zero_mask = (sums <= 0)
        probs_full = torch.where(
            zero_mask,
            torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1),
            probs_full / (sums + (~zero_mask) * 0.0 + 1e-12)
        )

        # Sample new tokens only where still masked
        probs_for_sampling = probs_full.permute(0, 2, 1)      # (N, d, 3)
        dist = torch.distributions.Categorical(probs=probs_for_sampling)
        new_tokens = dist.sample()                            # (N, d) in {0,1,2}
        samples_neu = torch.where(samples_neu == float(mask_token),
                                  new_tokens.float(),
                                  samples_neu)


# --- Metrics / Plots ---
sam_np = samples_neu.cpu().numpy().copy().astype(np.int64)
sam_np1 = samples_neu.cpu().numpy().copy().astype(np.int64)
sam_np[sam_np == mask_token] = 1  # treat remaining masks as 1

weights = original_array.astype(np.int64)      # [2^(d-1), ..., 1]
labels = sam_np.dot(weights)                   # shape (Ntest,), int64

Nconf = 1 << (L*L)
cvrev_neu = np.bincount(labels, minlength=Nconf) / Ntest # Histogram
tv = np.sum(np.abs(ismod - cvrev_neu)) # TV from actual distribution
plt.clf()
plt.plot(np.arange(Nconf), cvrev_neu, 'o', label='Model')
plt.plot(np.arange(Nconf), ismod, label='True')
plt.xlim(0, Nconf + 1)
plt.legend()
plt.draw()
plt.pause(0.01)
