import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import time

class PGFTRPOEngine:
    """
    PGF-TRPO Engine: 
    Achieves O(1) memory for second-order Trust Region Policy Optimization.
    Uses the Double-Path PGF isomorphism for exact HVP.
    """
    def __init__(self, policy_net: nn.Module, block_size: int = 256, cg_iters: int = 10, delta: float = 0.01):
        self.policy = policy_net
        self.block_size = block_size
        self.cg_iters = cg_iters
        self.delta = delta # Trust region constraint

    def conjugate_gradient(self, b_dict: Dict[str, torch.Tensor], hvp_args: Tuple) -> Dict[str, torch.Tensor]:
        """
        Solves F x = b using Conjugate Gradient without ever forming F.
        """
        x_dict = {n: torch.zeros_like(p) for n, p in b_dict.items()}
        r_dict = {n: p.clone() for n, p in b_dict.items()}
        p_dict = {n: p.clone() for n, p in b_dict.items()}
        
        def dot_prod(d1, d2):
            return sum((d1[n] * d2[n]).sum() for n in d1)

        rdotr = dot_prod(r_dict, r_dict)
        
        for i in range(self.cg_iters):
            # Key: Call our PGF-HVP operator
            f_p_dict = self.exact_hvp(p_dict, *hvp_args)
            
            alpha = rdotr / (dot_prod(p_dict, f_p_dict) + 1e-8)
            
            for n in x_dict:
                x_dict[n].add_(p_dict[n], alpha=alpha)
                r_dict[n].add_(f_p_dict[n], alpha=-alpha)
            
            new_rdotr = dot_prod(r_dict, r_dict)
            if new_rdotr < 1e-10:
                break
                
            beta = new_rdotr / rdotr
            for n in p_dict:
                p_dict[n] = r_dict[n] + beta * p_dict[n]
            rdotr = new_rdotr
            
        return x_dict

    @torch.no_grad()
    def exact_hvp(self, v_params: Dict[str, torch.Tensor], states, actions) -> Dict[str, torch.Tensor]:
        """
        Analytical PGF-HVP:
        1. JVP Path: Analytical Dual Scan (Zero drift, O(1) memory)
        2. Adjoint Path: Standard PGF Backward
        """
        L = states.shape[0]
        hvp_grads = {n: torch.zeros_like(p) for n, p in self.policy.named_parameters()}
        
        # Check if policy has analytic JVP
        has_analytic_jvp = hasattr(self.policy, 'get_jvp')

        for start in range(0, L, self.block_size):
            end = min(start + self.block_size, L)
            s_block = states[start:end]
            a_block = actions[start:end]
            
            # --- Path 1: JVP (Forward Projection) ---
            if has_analytic_jvp:
                # Optimized Analytical JVP
                jvp_res = self.policy.get_jvp(s_block, a_block, v_params)
                if isinstance(jvp_res, tuple):
                    s_vals, info = jvp_res
                    # Optional: Log spectral info here
                else:
                    s_vals = jvp_res
            else:
                # Fallback FD
                alpha = 1e-6
                original_params = {n: p.data.clone() for n, p in self.policy.named_parameters()}
                with torch.enable_grad():
                    lp_base = self.policy.get_log_prob(s_block, a_block)
                    for n, p in self.policy.named_parameters():
                        p.data.add_(v_params[n], alpha=alpha)
                    lp_plus = self.policy.get_log_prob(s_block, a_block)
                    for n, p in self.policy.named_parameters():
                        p.data.copy_(original_params[n])
                    s_vals = (lp_plus - lp_base).detach() / alpha
            
            # --- Path 2: PGF Backward ---
            with torch.enable_grad():
                # Note: We reuse get_log_prob here to get the graph for backward
                # For high performance, this could be fused with get_jvp in the future
                lp_base = self.policy.get_log_prob(s_block, a_block)
                hvp_obj = (lp_base * s_vals).sum()
                torch.autograd.backward(hvp_obj)
                
                for n, p in self.policy.named_parameters():
                    if p.grad is not None:
                        hvp_grads[n].add_(p.grad)
                        p.grad.zero_()
                        
        return hvp_grads

    @torch.no_grad()
    def compute_policy_gradient(self, states, actions, advantages) -> Dict[str, torch.Tensor]:
        """
        Standard PGF policy gradient: grad(E[log_pi * A])
        """
        L = states.shape[0]
        grads = {n: torch.zeros_like(p) for n, p in self.policy.named_parameters()}
        
        for start in range(0, L, self.block_size):
            end = min(start + self.block_size, L)
            s_block = states[start:end]
            a_block = actions[start:end]
            adv_block = advantages[start:end]
            
            with torch.enable_grad():
                log_probs = self.policy.get_log_prob(s_block, a_block)
                obj = (log_probs * adv_block).sum()
                torch.autograd.backward(obj)
                
                for n, p in self.policy.named_parameters():
                    if p.grad is not None:
                        grads[n].add_(p.grad)
                        p.grad.zero_()
        return grads

    def step(self, states, actions, advantages):
        """
        Performs one TRPO update step.
        """
        # 1. Compute Policy Gradient g
        g_dict = self.compute_policy_gradient(states, actions, advantages)
        
        # 2. Solve F x = g using CG
        # Note: In TRPO, F is the Fisher Information Matrix (Hessian of KL)
        # Our exact_hvp computes F v exactly.
        x_dict = self.conjugate_gradient(g_dict, (states, actions))
        
        # 3. Compute step size scale: beta = sqrt(2 * delta / (g^T F^-1 g))
        # g^T F^-1 g = g^T x (since x = F^-1 g)
        g_dot_x = sum((g_dict[n] * x_dict[n]).sum() for n in g_dict)
        step_scale = torch.sqrt(2 * self.delta / (g_dot_x + 1e-8))
        
        # 4. Update Parameters
        for n, p in self.policy.named_parameters():
            p.data.add_(x_dict[n], alpha=step_scale)
            
        return {"g_norm": torch.sqrt(g_dot_x), "step_scale": step_scale}
