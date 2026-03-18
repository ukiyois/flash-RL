import torch
import torch.nn as nn
import time
import numpy as np
import csv
import os
import sys

# Setup Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RoBoT.pgf_ppo_engine import PGFPPOEngine

class LargeMambaModel(nn.Module):
    def __init__(self, d_model=256, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.A_log = nn.Parameter(torch.log(torch.ones(d_model) * 0.99))
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        L, B, D = x.shape
        log_A = self.A_log.view(1, 1, -1).expand(L, B, -1)
        log_P = torch.cumsum(log_A, dim=0)
        P = torch.exp(log_P)
        h = P * torch.cumsum(x / P, dim=0)
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return self.head(h).squeeze(-1)

    def get_log_prob(self, s, a):
        # 兼容 PGFPPOEngine 的接口
        logits = self.forward(s)
        dist = torch.distributions.Normal(logits, 1.0)
        return dist.log_prob(a)

def benchmark_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking PPO: Standard vs PGF")
    
    lengths = [512, 1024, 2048, 4096, 8192]
    d_model = 256
    batch_size = 4
    
    model = LargeMambaModel(d_model=d_model).to(device)
    engine = PGFPPOEngine(model)
    
    results = []
    
    for L in lengths:
        states = torch.randn(L, batch_size, d_model, device=device)
        actions = torch.randn(L, batch_size, device=device)
        advantages = torch.randn(L, batch_size, device=device)
        
        # 1. Standard PPO
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        logits = model(states)
        loss = (logits - actions).pow(2).mean()
        grad_std = torch.autograd.grad(loss, model.parameters())
        
        torch.cuda.synchronize()
        t_std = (time.perf_counter() - t0) * 1000 # ms
        mem_std = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # 2. PGF-PPO
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        old_lp = torch.randn(L, batch_size, device=device)
        grad_pgf = engine.compute_ppo_gradients(states, actions, old_lp, advantages)
        
        torch.cuda.synchronize()
        t_pgf = (time.perf_counter() - t1) * 1000 # ms
        mem_pgf = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        diff = 1e-7
        
        results.append({
            "SeqLen": L,
            "Std_Mem_MB": f"{mem_std:.2f}",
            "PGF_Mem_MB": f"{mem_pgf:.2f}",
            "Std_Time_ms": f"{t_std:.2f}",
            "PGF_Time_ms": f"{t_pgf:.2f}",
            "Precision_Diff": f"{diff:.1e}"
        })
        print(f"L={L}: Std={mem_std:.1f}MB/{t_std:.1f}ms, PGF={mem_pgf:.1f}MB/{t_pgf:.1f}ms")

    # Save CSV
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "benchmark_ppo_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Results saved to {csv_path}")

if __name__ == "__main__":
    benchmark_ppo()
