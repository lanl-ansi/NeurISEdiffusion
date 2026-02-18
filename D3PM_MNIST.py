import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L = 28
P = L * L                 
N_states = 2            
cond_dim = 10            
batch_size = 128
num_epochs = 1
patience = 200
min_delta = 1e-6
lr = 0.0005257224903737729
weight_decay = 1.4294881998914792e-06

# -----------------------------
# Data
# -----------------------------
def to_onehot(y, n_classes=10):
    oh = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return oh

with np.load("mnist.npz") as data:
    X_train_all, y_train_all = data["x_train"], data["y_train"]
    X_test_all,  y_test_all  = data["x_test"],  data["y_test"]

# binarize to {0,1} and flatten to (N, P)
Xtr_np = (X_train_all/255.0 > 0.5).astype(np.int64).reshape(-1, P)
Xte_np = (X_test_all /255.0 > 0.5).astype(np.int64).reshape(-1, P)
Ytr_oh = to_onehot(y_train_all, cond_dim)
Yte_oh = to_onehot(y_test_all,  cond_dim)

Xtr = torch.from_numpy(Xtr_np).to(device)            # (N,P) long in {0,1}
Ytr = torch.from_numpy(Ytr_oh).to(device)            # (N,10) float32
Xte = torch.from_numpy(Xte_np).to(device)
Yte = torch.from_numpy(Yte_oh).to(device)

ds = TensorDataset(Xtr.long(), Ytr.float())
n_val = max(1, int(0.1 * len(ds)))
trn_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
                              generator=torch.Generator().manual_seed(0))
trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# -----------------------------
# Model
# -----------------------------
class ConditionalX0Model(nn.Module):
    """
    Input:  x_t (B,P) ints in {0,1}, t (B,), cond (B,cond_dim)
    Output: logits (B,P,N_states)
    """
    def __init__(self, N=N_states, P=P, cond_dim=cond_dim, hidden=512, depth=3, ln_eps=1e-5):
        super().__init__()
        assert P is not None
        self.N, self.P = N, P
        in_dim = P + 1 + cond_dim  # x_t + time + class one-hot

        layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden, eps=ln_eps), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden, eps=ln_eps), nn.SiLU()]
        layers += [nn.Linear(hidden, P * N)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x_t, t, cond=None):
        if cond is None:
            raise ValueError("cond (one-hot class) must be provided")
        z = x_t.float()
        if self.N == 2: z = 2.0*z - 1.0
        else:           z = z / (self.N - 1)

        t_feat = (t.float().view(-1, 1) / 1000.0)
        inp = torch.cat([z, t_feat, cond.float()], dim=1)   # (B,P+1+cond_dim)
        return self.net(inp).view(-1, self.P, self.N)       # (B,P,N_states)

# -----------------------------
# D3PM 
# -----------------------------
class D3PM(nn.Module):
    def __init__(self, x0_model, n_T=1000, num_classes=N_states, hybrid_loss_coeff=1e-3, forward_type="uniform"):
        super().__init__()
        self.x0_model = x0_model
        self.n_T = int(n_T)
        self.num_classes = int(num_classes)
        self.hybrid_loss_coeff = float(hybrid_loss_coeff)
        self.eps = 1e-6

        steps = torch.arange(self.n_T + 1, dtype=torch.float64) / self.n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        ).float()  # (T,)

        q_one_step = []
        for beta in beta_t:
            if forward_type != "uniform":
                raise NotImplementedError("Only uniform forward is implemented.")
            C = self.num_classes
            mat = torch.ones(C, C, dtype=torch.float32) * (beta / C)
            mat.fill_diagonal_(1 - (C - 1) * beta / C)
            q_one_step.append(mat)
        q_one_step = torch.stack(q_one_step, dim=0)               # (T, C, C)
        q_one_step_T = q_one_step.transpose(1, 2).contiguous()    # (T, C, C)

        # cumulative Q
        cum = q_one_step[0]
        q_cum = [cum]
        for idx in range(1, self.n_T):
            cum = cum @ q_one_step[idx]
            q_cum.append(cum)
        q_cum = torch.stack(q_cum, dim=0)                         # (T, C, C)

        self.register_buffer("q_one_step_T", q_one_step_T)
        self.register_buffer("q_cum", q_cum)
        self.last_info = {}

    def _at(self, a, t, x):
        B, P = x.shape
        C = a.shape[-1]
        a_t = a[t - 1]                            # (B? no) (T->indexing gives (C,C)), but uses broadcasting below
        a_t_exp = a_t.unsqueeze(1).expand(B, P, C, C)
        idx = x.unsqueeze(-1).unsqueeze(-1).expand(B, P, 1, C)
        out = torch.gather(a_t_exp, dim=2, index=idx).squeeze(2)  # (B,P,C)
        return out

    def q_sample(self, x0, t, noise):
        logits = torch.log(self._at(self.q_cum, t, x0) + self.eps)   # (B,P,C)
        g = -torch.log(-torch.log(torch.clamp(noise, self.eps, 1.0)))
        return torch.argmax(logits + g, dim=-1)                      # (B,P)

    def q_posterior_logits(self, x0_logits, x_t, t):
        B, P = x_t.shape
        C = self.num_classes
        if x0_logits.dtype in (torch.int64, torch.int32):
            x0_log = torch.log(nn.functional.one_hot(x0_logits, num_classes=C).float() + self.eps)
        else:
            x0_log = x0_logits
        fact1 = torch.log(self._at(self.q_one_step_T, t, x_t) + self.eps)  # (B,P,C)

        probs_x0 = torch.softmax(x0_log, dim=-1)                            # (B,P,C)
        t_minus_1 = torch.clamp(t - 1, min=1)
        q_prev = self.q_cum[t_minus_1 - 1].to(dtype=probs_x0.dtype)         # (B?,C,C) via indexing+broadcast
        fact2 = torch.matmul(probs_x0, q_prev)                               # (B,P,C)
        fact2 = torch.log(fact2.clamp_min(self.eps))

        out = torch.where(t.view(B, 1, 1) == 1, x0_log, fact1 + fact2)
        return out  # (B,P,C)

    def vb(self, d1_logits, d2_logits):
        d1 = d1_logits.flatten(0, -2)  # (B*P,C)
        d2 = d2_logits.flatten(0, -2)
        p  = torch.softmax(d1 + self.eps, dim=-1)
        return (p * (torch.log_softmax(d1 + self.eps, -1) - torch.log_softmax(d2 + self.eps, -1))).sum(-1).mean()

    def model_predict(self, x_t, t, cond=None):
        return self.x0_model(x_t, t, cond)  # (B,P,C)

    def forward(self, x0, cond=None):
        B, P = x0.shape
        t = torch.randint(1, self.n_T, (B,), device=x0.device)
        noise = torch.rand(B, P, self.num_classes, device=x0.device)
        x_t = self.q_sample(x0, t, noise)

        pred_x0_logits = self.model_predict(x_t, t, cond)
        true_post = self.q_posterior_logits(x0,            x_t, t)
        pred_post = self.q_posterior_logits(pred_x0_logits, x_t, t)
        vb_loss = self.vb(true_post, pred_post)

        ce_loss = nn.CrossEntropyLoss()(pred_x0_logits.view(-1, self.num_classes),
                                        x0.view(-1))
        loss = ce_loss + self.hybrid_loss_coeff * vb_loss
        self.last_info = {"vb_loss": float(vb_loss.detach()), "ce_loss": float(ce_loss.detach())}
        return loss

    def p_sample(self, x_t, t, cond, noise):
        pred_x0_logits   = self.model_predict(x_t, t, cond)
        pred_post_logits = self.q_posterior_logits(pred_x0_logits, x_t, t)
        not_first = (t != 1).float().view(-1, 1, 1)
        g = -torch.log(-torch.log(torch.clamp(noise, self.eps, 1.0)))
        return torch.argmax(pred_post_logits + not_first * g, dim=-1)

    def sample(self, x_init, cond=None):
        x = x_init
        for tt in reversed(range(1, self.n_T)):
            t = torch.full((x.shape[0],), tt, device=x.device, dtype=torch.long)
            noise = torch.rand(x.shape[0], x.shape[1], self.num_classes, device=x.device)
            x = self.p_sample(x, t, cond, noise)
        return x

# -----------------------------
# Init & Train
# -----------------------------
x0_model = ConditionalX0Model(N=N_states, P=P, cond_dim=cond_dim, hidden=512, depth=4).to(device)
d3pm = D3PM(x0_model, n_T=1000, num_classes=N_states, hybrid_loss_coeff=1e-3).to(device)
opt = torch.optim.Adam(d3pm.parameters(), lr=5e-3)

best, counter = float('inf'), 0
for epoch in range(1, num_epochs + 1):
    d3pm.train()
    for xb, yb in trn_loader:
        opt.zero_grad()
        loss = d3pm(xb, cond=yb)   # <-- pass cond
        loss.backward()
        opt.step()
  #  print(loss.item())
     
    d3pm.eval()
    with torch.no_grad():
        vloss, vcnt = 0.0, 0
        for xb, yb in val_loader:
            vloss += d3pm(xb, cond=yb).item()
            vcnt += 1
        vloss /= max(1, vcnt)

    if vloss < best - min_delta:
        best, counter = vloss, 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best:.6f}")
            break

    if epoch % 20 == 0:
        info = d3pm.last_info
        print(f"Epoch {epoch}: train_loss={loss.item():.6f} "
              f"(CE={info.get('ce_loss', float('nan')):.6f}, VB={info.get('vb_loss', float('nan')):.6f}) "
              f"val_loss={vloss:.6f}")

# -----------------------------
# Sampling (conditioned)
# -----------------------------
#%%
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

with torch.no_grad():
    d3pm.eval()
    for gen_digit in range(0, 10):
        
        Ntest = 10**4
        x_init = torch.randint(0, N_states, (Ntest, P), device=device, dtype=torch.long)
        
        cond = torch.zeros(Ntest, cond_dim, device=device); 
        cond[:, gen_digit] = 1.0
        samples = d3pm.sample(x_init, cond=cond)  # (Ntest,P) ints in {0,1}
    
        imgs = samples.view(-1, 28, 28).float().cpu().numpy()
        
        out_npz = f"GendataD3PM_digit{gen_digit}_MNIST.npz"
        np.savez_compressed(out_npz, samples_gen=samples.detach().cpu().numpy())
        # quick viz of a few samples
        fig, axes = plt.subplots(5, 5, figsize=(8, 4))
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(imgs[i], cmap='gray'); ax.axis('off')
        plt.tight_layout(); plt.show()
        
        save_grid(
           imgs,
           path=f"Mnist_d3pm_digit_{gen_digit}.png",
           rows=2,
           cols=5
       )
    
    
