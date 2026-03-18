import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import csv
import os
import sys

# Setup Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RoBoT.pgf_trpo_engine import PGFTRPOEngine
from RoBoT.holo_trpo_engine import HoloTRPOEngine

class SimpleMambaPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 0.99 for long-range dependency
        self.A_log = nn.Parameter(torch.log(torch.ones(hidden_dim) * 0.99)) 
        self.B = nn.Linear(state_dim, hidden_dim)
        self.C = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._current_actions = None # Added for HoloTRPO compatibility
        
        nn.init.normal_(self.B.weight, std=0.01)
        nn.init.normal_(self.C.weight, std=0.01)

    def forward(self, states):
        """Forward for HoloTRPOEngine."""
        return self.get_log_prob(states, self._current_actions)
        
    def get_log_prob(self, states, actions):
        """Strictly Holomorphic Parallel Scan."""
        dtype = self.B.weight.dtype
        states = states.to(dtype)
        actions = actions.to(dtype)
        
        if states.dim() == 2:
            states = states.unsqueeze(1)
            actions = actions.unsqueeze(1)
        
        L, B, D = states.shape
        b = self.B(states)
        log_A = self.A_log.view(1, 1, -1).expand(L, B, -1)
        
        # Holomorphic Exponential Mapping (No clamp/real/imag)
        log_P = torch.cumsum(log_A, dim=0)
        P = torch.exp(log_P)
        
        # Parallel Scan (Analytic Continuation)
        h = P * torch.cumsum(b / P, dim=0)
        h = torch.tanh(h) # Tanh is holomorphic
        
        mus = self.C(h)
        std = torch.exp(self.log_std)
        
        # Holomorphic Gaussian Log-Prob
        # Formula: -0.5 * log(2*pi*std^2) - 0.5 * ((x-mu)/std)^2
        # All ops (+, -, *, /, log, exp) are analytic.
        log_pi = -0.5 * (torch.log(torch.tensor(2 * np.pi, dtype=dtype, device=mus.device)) + 2 * torch.log(std)) \
                 - 0.5 * ((actions - mus) / std)**2
        return log_pi.sum(dim=-1).squeeze(-1)

    def get_jvp(self, states, actions, v_dict):
        """Analytical Dual Scan for JVP."""
        if states.dim() == 2:
            states = states.unsqueeze(1)
            actions = actions.unsqueeze(1)
        L, B, D = states.shape
        
        # 1. Base Scan
        b_base = self.B(states)
        log_A = self.A_log.view(1, 1, -1).expand(L, B, -1)
        log_P = torch.cumsum(log_A, dim=0)
        P = torch.exp(log_P)
        S = torch.cumsum(b_base / P, dim=0)
        h_base = P * S
        h_base_act = torch.tanh(h_base)
        
        # 2. Dual Scan
        log_A_dot = v_dict['A_log'].view(1, 1, -1).expand(L, B, -1)
        L_t = torch.cumsum(log_A_dot, dim=0)
        P_dot = P * L_t
        b_dot = F.linear(states, v_dict['B.weight'], v_dict['B.bias'])
        S_dot = torch.cumsum((b_dot * P - b_base * P_dot) / (P**2), dim=0)
        h_dot = P_dot * S + P * S_dot
        
        # Derivative of tanh(h) = (1 - tanh(h)^2) * h_dot
        h_dot_act = (1.0 - h_base_act**2) * h_dot
        
        # 3. Projection
        mu_base = self.C(h_base_act)
        mu_dot = F.linear(h_base_act, v_dict['C.weight'], v_dict['C.bias']) + \
                 F.linear(h_dot_act, self.C.weight, None)
                 
        # 4. log_pi Dual
        std = torch.exp(self.log_std)
        std_dot = std * v_dict['log_std']
        d_lp_d_mu = (actions - mu_base) / (std**2)
        d_lp_d_std = (actions - mu_base)**2 / (std**3) - 1.0/std
        s_t = (d_lp_d_mu * mu_dot).sum(dim=-1) + (d_lp_d_std * std_dot).sum(dim=-1)
        # 5. Spectral Monitoring
        A_vals = torch.exp(self.A_log).detach().cpu().numpy()
        a_mean, a_max, a_min = A_vals.mean(), A_vals.max(), A_vals.min()
        
        return s_t.squeeze(-1), {"a_mean": a_mean, "a_max": a_max, "a_min": a_min}

    @torch.no_grad()
    def select_action(self, state, h_prev):
        A = torch.exp(self.A_log)
        h_next = A * h_prev + self.B(state)
        h_next = torch.tanh(h_next)
        mu = self.C(h_next)
        std = torch.exp(self.log_std)
        action = mu + torch.randn_like(mu) * std
        return action, h_next

class PointMassTask:
    def __init__(self, L=2000):
        self.L = L
        self.reset()
        
    def reset(self):
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        # Random Goal for higher difficulty
        self.goal = np.random.uniform(-1.5, 1.5, size=(2,)).astype(np.float32)
        self.t = 0
        return self._get_obs()
        
    def _get_obs(self):
        return np.concatenate([self.pos, self.goal])
        
    def step(self, action):
        action = action.cpu().numpy().flatten()
        self.pos += np.clip(action, -1, 1) * 0.1
        dist = np.linalg.norm(self.pos - self.goal)
        reward = -dist 
        if dist < 0.1: reward += 1.0 # Bonus for reaching
        self.t += 1
        done = (self.t >= self.L)
        return self._get_obs(), reward, done

def train_convergence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = 512 
    iters = 20 
    
    # Independent Agents
    policy_pgf = SimpleMambaPolicy(state_dim=4, action_dim=2).to(device)
    policy_holo = SimpleMambaPolicy(state_dim=4, action_dim=2).to(device)
    policy_holo.load_state_dict(policy_pgf.state_dict())

    engine_pgf = PGFTRPOEngine(policy_pgf, block_size=256, cg_iters=10, delta=0.01)
    engine_holo = HoloTRPOEngine(policy_holo, eta=1e-3, M=8, delta=0.01, device=str(device)) 
    
    env_pgf = PointMassTask(L=L)
    env_holo = PointMassTask(L=L)
    
    results = []
    print(f"Starting HARD Convergence Comparison: PGF-TRPO vs Holo-TRPO (L={L}, iters={iters})... ")
    
    for i in range(iters):
        # 1. PGF Agent Step
        t0 = time.time()
        states_p, actions_p, rewards_p = [], [], []
        obs_p = env_pgf.reset()
        h_p = torch.zeros(1, 64, device=device)
        for _ in range(L):
            s_tensor = torch.from_numpy(obs_p).to(device).unsqueeze(0)
            a, h_p = policy_pgf.select_action(s_tensor, h_p)
            next_obs, r, done = env_pgf.step(a)
            states_p.append(s_tensor)
            actions_p.append(a)
            rewards_p.append(r)
            obs_p = next_obs
        
        returns_p = []
        g_p = 0
        for r in reversed(rewards_p):
            g_p = r + 0.99 * g_p
            returns_p.insert(0, g_p)
        returns_p = torch.tensor(returns_p, device=device, dtype=torch.float32)
        adv_p = (returns_p - returns_p.mean()) / (returns_p.std() + 1e-8)
        
        info_pgf = engine_pgf.step(torch.cat(states_p), torch.cat(actions_p), adv_p)
        dt_pgf = time.time() - t0
        
        # 2. Holo Agent Step
        t1 = time.time()
        states_h, actions_h, rewards_h = [], [], []
        obs_h = env_holo.reset()
        h_h = torch.zeros(1, 64, device=device)
        for _ in range(L):
            s_tensor = torch.from_numpy(obs_h).to(device).unsqueeze(0)
            a, h_h = policy_holo.select_action(s_tensor, h_h)
            next_obs, r, done = env_holo.step(a)
            states_h.append(s_tensor)
            actions_h.append(a)
            rewards_h.append(r)
            obs_h = next_obs
            
        returns_h = []
        g_h = 0
        for r in reversed(rewards_h):
            g_h = r + 0.99 * g_h
            returns_h.insert(0, g_h)
        returns_h = torch.tensor(returns_h, device=device, dtype=torch.float32)
        adv_h = (returns_h - returns_h.mean()) / (returns_h.std() + 1e-8)
        
        s_cat_h = torch.cat(states_h)
        a_cat_h = torch.cat(actions_h)
        policy_holo._current_actions = a_cat_h
        info_holo = engine_holo.step(s_cat_h, a_cat_h, adv_h)
        dt_holo = time.time() - t1
        
        avg_reward_p = np.mean(rewards_p)
        avg_reward_h = np.mean(rewards_h)
        print(f"Iter {i:2}: [PGF] Reward={avg_reward_p:.4f} | [Holo] Reward={avg_reward_h:.4f}, Step={info_holo['step_size']:.6f}")
        
        results.append({
            "Iter": i,
            "PGF_Reward": f"{avg_reward_p:.4f}",
            "Holo_Reward": f"{avg_reward_h:.4f}",
            "PGF_Time": f"{dt_pgf:.2f}",
            "Holo_Step": f"{info_holo['step_size']:.6f}",
            "Holo_Time": f"{dt_holo:.2f}"
        })

    # Save CSV
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "trpo_convergence_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Results saved to {csv_path}")

if __name__ == "__main__":
    train_convergence()
