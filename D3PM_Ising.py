import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gc

# -----------------------------
# Setup
# -----------------------------
L = 5
Nclass = 2
Nconf  = 2 ** (L * L)
Ntrain = 100
Ntest = 10**5
n_epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = f"Edwards-Anderson_model_L{L}_samples.npy"
samplesload = np.load(filename)



    



samplesNP = np.load(f"Edwards-Anderson_model_L{L}_samples.npy")[:, :Ntrain]  # (L*L, Ntrain)
ismod = np.load(f"Edwards-Anderson_model_L{L}.npy")
x0 = torch.from_numpy(samplesNP.T.astype(np.int64)).to(device)               # (Ntrain, L*L)
P = x0.shape[1]  
# -----------------------------
# Model Definitions
# -----------------------------
hidden_dim = 128
class SimpleX0Model(nn.Module):
 def __init__(self, N=2, P=None):
    super().__init__()
    assert P is not None, "Pass P = x0.shape[1]"
    input_dim = P + 1                    # raw P spins + 1 time scalar
    self.net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, P * N)
    )
    self.N = N
    self.P = P

 def forward(self, x_t, t, cond=None):
    """
    x_t: (B, P) integer spins in [0..N-1]
    t:   (B,)   integer timestep
    """
    B = x_t.shape[0]

    # Normalize raw integers (kept simple & scale-stable)
    z = x_t.float()
    if self.N == 2:
        # Map {0,1} -> {-1,+1}
        z = 2.0 * z - 1.0
    else:
        # Scale to [0,1]
        z = z / (self.N - 1)

    # Normalize time
    t_feat = (t.float().view(-1, 1) / 1000.0)  # shape (B,1)

    # Concatenate per-sample vector: (B, P+1)
    inp = torch.cat([z, t_feat], dim=1)

    # MLP -> (B, P*N) -> reshape to (B, P, N)
    out = self.net(inp).view(B, self.P, self.N)
    return out

# -----------------------------
# D3PM with MNIST-style logic (cumulative Q, posterior, Gumbel, hybrid loss)
# -----------------------------
class D3PM(nn.Module):
     def __init__(self, x0_model, n_T=1000, num_classes=2, hybrid_loss_coeff=1e-3, forward_type="uniform"):
         super().__init__()
         self.x0_model = x0_model
         self.n_T = int(n_T)
         self.num_classes = int(num_classes)
         self.hybrid_loss_coeff = float(hybrid_loss_coeff)
         self.eps = 1e-6

         # Cosine schedule → beta_t
         steps = torch.arange(self.n_T + 1, dtype=torch.float64) / self.n_T
         alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
         beta_t = torch.minimum(
             1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
         ).float()  # (T,)

         # One-step transition matrices Q_t
         q_one_step = []
         for beta in beta_t:
             if forward_type != "uniform":
                 raise NotImplementedError("Only uniform forward is implemented.")
             K = self.num_classes
             mat = torch.ones(K, K, dtype=torch.float32) * (beta / K)
             mat.fill_diagonal_(1 - (K - 1) * beta / K)
             q_one_step.append(mat)
         q_one_step = torch.stack(q_one_step, dim=0)               # (T, K, K)
         q_one_step_T = q_one_step.transpose(1, 2).contiguous()    # (T, K, K)

         # Cumulative Q_{0→t} = Q1 @ Q2 @ ... @ Qt
         cum = q_one_step[0]
         q_cum = [cum]
         for idx in range(1, self.n_T):
             cum = cum @ q_one_step[idx]
             q_cum.append(cum)
         q_cum = torch.stack(q_cum, dim=0)                         # (T, K, K)

         self.register_buffer("q_one_step_T", q_one_step_T)
         self.register_buffer("q_cum", q_cum)

         self.last_info = {}

     # Helper: gather a[t-1, x, :]
     def _at(self, a, t, x):
         """
         a: (T, K, K)   transition matrices
         t: (B,)        timestep (1..T)
         x: (B, P)      integer labels in [0..K-1]
     
         returns: (B, P, K)
         """
         B, P = x.shape
         K = a.shape[-1]
     
         # pick per-batch matrix a[t-1]: (B, K, K)
         a_t = a[t - 1]
     
         # expand to (B, P, K, K)
         a_t_exp = a_t.unsqueeze(1).expand(B, P, K, K)
     
         # indices for rows we want (the x along the "source" K dim): (B, P, 1, K)
         idx = x.unsqueeze(-1).unsqueeze(-1).expand(B, P, 1, K)
     
         # gather along the row-dimension (dim=2), then drop that dim -> (B, P, K)
         out = torch.gather(a_t_exp, dim=2, index=idx).squeeze(2)
         return out

     # Forward diffusion: x_t ~ Cat( Q_{0→t} @ one_hot(x0) ), sampled via Gumbel-Max
     def q_sample(self, x0, t, noise):
         logits = torch.log(self._at(self.q_cum, t, x0) + self.eps)     # (B, P, K)
         g = -torch.log(-torch.log(torch.clamp(noise, self.eps, 1.0)))
         return torch.argmax(logits + g, dim=-1)                        # (B, P)

     # Posterior logits for x_{t-1} | x_t, (x0 or its logits)
     def q_posterior_logits(self, x0_logits, x_t, t):
         B, P = x_t.shape
         K = self.num_classes
     
         # x0 as logits
         if x0_logits.dtype in (torch.int64, torch.int32):
             x0_log = torch.log(nn.functional.one_hot(x0_logits, num_classes=K).float() + self.eps)
         else:
             x0_log = x0_logits
     
         # fact1: from x_t via one-step transpose Q_t^T  → (B, P, K)
         fact1 = torch.log(self._at(self.q_one_step_T, t, x_t) + self.eps)
     
         # fact2: from x0 via cumulative Q_{0→t-1}      → (B, P, K)
         probs_x0 = torch.softmax(x0_log, dim=-1)         # (B, P, K)
         t_minus_1 = torch.clamp(t - 1, min=1)
         q_prev = self.q_cum[t_minus_1 - 1].to(dtype=probs_x0.dtype)  # (B, K, K)
         fact2 = torch.matmul(probs_x0, q_prev)            # (B, P, K)
         fact2 = torch.log(fact2.clamp_min(self.eps))
     
         # Special case t==1 → use x0 logits directly (L0 term)
         out = torch.where(t.view(B, 1, 1) == 1, x0_log, fact1 + fact2)
         return out                                             # (B, P, K)

     # VB term between two posterior logits
     def vb(self, dist1_logits, dist2_logits):
         d1 = dist1_logits.flatten(0, -2)  # (B*P, K)
         d2 = dist2_logits.flatten(0, -2)
         p  = torch.softmax(d1 + self.eps, dim=-1)
         return (p * (torch.log_softmax(d1 + self.eps, -1) - torch.log_softmax(d2 + self.eps, -1))).sum(-1).mean()

     # Wrap model call
     def model_predict(self, x_t, t, cond=None):
         return self.x0_model(x_t, t, cond)  # (B, P, K)

     # Training step — returns a scalar loss
     def forward(self, x0, cond=None):
         B, P = x0.shape
         t = torch.randint(1, self.n_T, (B,), device=x0.device)
         noise = torch.rand(B, P, self.num_classes, device=x0.device)
         x_t = self.q_sample(x0, t, noise)

         pred_x0_logits = self.model_predict(x_t, t, cond)              # (B, P, K)

         true_post = self.q_posterior_logits(x0,            x_t, t)
         pred_post = self.q_posterior_logits(pred_x0_logits, x_t, t)
         vb_loss = self.vb(true_post, pred_post)

         ce_loss = nn.CrossEntropyLoss()(pred_x0_logits.view(-1, self.num_classes),
                                         x0.view(-1))
         loss = ce_loss + self.hybrid_loss_coeff * vb_loss
         self.last_info = {"vb_loss": float(vb_loss.detach()), "ce_loss": float(ce_loss.detach())}
         return loss

     # One reverse step with Gumbel (greedy at t==1)
     def p_sample(self, x_t, t, cond, noise):
         pred_x0_logits  = self.model_predict(x_t, t, cond)
         pred_post_logits = self.q_posterior_logits(pred_x0_logits, x_t, t)
         not_first = (t != 1).float().view(-1, 1, 1)
         g = -torch.log(-torch.log(torch.clamp(noise, self.eps, 1.0)))
         return torch.argmax(pred_post_logits + not_first * g, dim=-1)

     # Full reverse chain
     def sample(self, x_init, cond=None):
         x = x_init
         for tt in reversed(range(1, self.n_T)):
             t = torch.full((x.shape[0],), tt, device=x.device, dtype=torch.long)
             noise = torch.rand(x.shape[0], x.shape[1], self.num_classes, device=x.device)
             x = self.p_sample(x, t, cond, noise)
         return x

# -----------------------------
# Initialize and Train
# -----------------------------

x0_model = SimpleX0Model(N=Nclass, P=P).to(device)
d3pm = D3PM(x0_model, n_T=1000, num_classes=Nclass, hybrid_loss_coeff=1e-1).to(device)

opt = torch.optim.Adam(d3pm.parameters(), lr=0.009352434594439707, weight_decay = 0.00023806712410001036 )
batch_size = Ntrain
best_loss, patience, patience_counter, min_delta = float('inf'), 100, 0, 1e-6
start = time.time()

for epoch in range(1, n_epochs + 1):
 current_loss = 0 
 for cyc in range(0,1):
   d3pm.train()
   opt.zero_grad()
   loss = d3pm(x0)                # scalar hybrid loss (CE + λ·VB)
   loss.backward()
   opt.step()
   current_loss += loss.item()

# if epoch % 100 == 0:
#     info = d3pm.last_info
#     print(f"Epoch {epoch}: loss={loss.item():.6f} (CE={info.get('ce_loss',float('nan')):.6f}, VB={info.get('vb_loss',float('nan')):.6f})")


 if best_loss - current_loss > min_delta:
    best_loss = current_loss
    patience_counter = 0
 else:
    patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
        break

#%%
# -----------------------------
# Generate Samples and Evaluate
# -----------------------------
with torch.no_grad():
 d3pm.eval()
 x_init  = torch.randint(0, 1, (Ntest, P), device=device)  # start from integers
 samples = d3pm.sample(x_init).cpu().numpy()                    # (Ntest, P)

 t_rev = time.time() - start
    # Estimate empirical distribution over configurations (for small P)
    
 binary_weights = np.array([2**(P - i - 1) for i in range(P)])
 labels = np.sum(samples * binary_weights, axis=1)
 counts = np.bincount(labels, minlength=Nconf) / Ntest
    
 plt.figure(figsize=(10, 4))
 plt.plot(counts, 'o', label='Generated by D3PM (hybrid, no one-hot)')
 plt.plot(ismod, label='True Ising Distribution')
 plt.xlabel("Configuration Index")
 plt.ylabel("Probability")
 plt.title(f"D3PM Sampling vs True Distribution (L={L}, P={P})")
 plt.legend()
 plt.tight_layout()
 plt.show()

 tv = np.sum(np.abs(counts - ismod))
    # print(f"Total Variation Distance: {tv:.6f}")
    
  
