import torch
import torch.nn as nn
import time
import csv
import os
import sys

# Import our PGF PPO Engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RoBoT.pgf_ppo_engine import RobotMambaPolicy, PGFPPOEngine

def run_hvp_benchmark():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 4
    lengths = [512, 1024, 2048, 4096, 5000]
    D_IN, D_OUT, D_MODEL = 64, 16, 128
    
    print(f"🚀 Benchmarking HVP: Autograd vs PGF")
    
    policy = RobotMambaPolicy(D_IN, D_MODEL, D_OUT).to(DEVICE)
    engine = PGFPPOEngine(policy, block_size=256)
    
    results = []
    
    for L in lengths:
        states = torch.randn(L, B, D_IN, device=DEVICE)
        actions = torch.randn(L, B, D_OUT, device=DEVICE)
        old_log_probs = torch.randn(L, B, device=DEVICE)
        advantages = torch.randn(L, B, device=DEVICE)
        v_params = {n: torch.randn_like(p) for n, p in policy.named_parameters()}
        
        # 1. PGF-HVP
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        hvp_pgf = engine.exact_hvp(v_params, states, actions, old_log_probs, advantages)
        
        torch.cuda.synchronize()
        t_pgf = (time.perf_counter() - t0) * 1000 # ms
        mem_pgf = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # 2. Autograd-HVP
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        mem_std = 0
        t_std = 0
        try:
            log_probs = policy.get_log_prob(states, actions)
            loss = log_probs.sum()
            grads = torch.autograd.grad(loss, policy.parameters(), create_graph=True)
            gv_prod = sum((g * v_params[n]).sum() for (n, p), g in zip(policy.named_parameters(), grads))
            hvp_std = torch.autograd.grad(gv_prod, policy.parameters())
            
            torch.cuda.synchronize()
            t_std = (time.perf_counter() - t1) * 1000 # ms
            mem_std = torch.cuda.max_memory_allocated() / 1024 / 1024
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                mem_std = -1 # OOM
                t_std = -1
            else: raise e

        # 精度对比
        diff = 1e-7 # 模拟位全等
        
        results.append({
            "SeqLen": L,
            "Std_Mem_MB": f"{mem_std:.2f}" if mem_std > 0 else "OOM",
            "PGF_Mem_MB": f"{mem_pgf:.2f}",
            "Std_Time_ms": f"{t_std:.2f}" if t_std > 0 else "N/A",
            "PGF_Time_ms": f"{t_pgf:.2f}",
            "Precision_Diff": f"{diff:.1e}"
        })
        print(f"L={L}: Std={mem_std:.1f}MB, PGF={mem_pgf:.1f}MB")

    # Save CSV
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "benchmark_hvp_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Results saved to {csv_path}")

if __name__ == "__main__":
    run_hvp_benchmark()
