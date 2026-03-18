import torch
import numpy as np
from typing import Optional, Tuple
from RoBoT.pgf_trpo_engine import PGFTRPOEngine

class IsaacGymPGFWrapper:
    """
    High-level wrapper to connect PGF-TRPO Engine with Isaac Gym.
    Optimized for GPU-to-GPU data transfer.
    """
    def __init__(
        self, 
        env, # Isaac Gym Env (e.g. VecTask)
        policy_net: torch.nn.Module,
        block_size: int = 256,
        device: str = "cuda:0"
    ):
        self.env = env
        self.policy = policy_net
        self.engine = PGFTRPOEngine(policy_net, block_size=block_size)
        self.device = device
        
        self.obs = self.env.reset()
        self.num_envs = self.env.num_envs
        self.h_state = torch.zeros(self.num_envs, policy_net.hidden_dim, device=device)

    def collect_trajectories(self, L: int):
        """
        Collects L steps of data across all parallel Isaac Gym environments.
        Returns tensors ready for PGF update.
        """
        states, actions, rewards, log_probs = [], [], [], []
        
        # Reset hidden state for new trajectory batch
        self.h_state.zero_()
        
        for t in range(L):
            # 1. Policy step (Inference mode)
            # In Isaac Gym, obs is already a GPU tensor (B, D)
            with torch.no_grad():
                # Note: select_action must support (B, D) input
                action, next_h = self.policy.select_action(self.obs, self.h_state)
            
            # 2. Env step
            next_obs, reward, done, info = self.env.step(action)
            
            # 3. Store data (Still on GPU)
            states.append(self.obs)
            actions.append(action)
            rewards.append(reward)
            
            self.obs = next_obs
            self.h_state = next_h
            
            # Handle terminations if necessary (simple reset for now)
            # In PGF, we usually prefer continuous long trajectories.
            
        # Reshape to (L, B, D)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        
        return states, actions, rewards

    def train_step(self, L: int):
        """One full TRPO update cycle."""
        # 1. GPU-accelerated collection
        states, actions, rewards = self.collect_trajectories(L)
        
        # 2. Advantage estimation (GAE or simple returns)
        # Note: Needs to handle (L, B) shapes
        advantages = self.estimate_advantages(rewards)
        
        # 3. PGF-TRPO Engine Update
        # Our engine currently expects (L, B, D) or flattened.
        # We can update the engine to handle B dimension explicitly.
        info = self.engine.step(states, actions, advantages)
        
        return info

    def estimate_advantages(self, rewards: torch.Tensor, gamma: float = 0.99):
        """Simple discounted return estimation on GPU."""
        L, B = rewards.shape
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(B, device=self.device)
        
        for t in reversed(range(L)):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
            
        # Normalize across batch and time
        adv = (returns - returns.mean()) / (returns.std() + 1e-8)
        return adv

# Example usage (Pseudocode):
# env = gym.make("Ant", num_envs=4096, ...)
# policy = RobotMambaPolicy(env.num_obs, 128, env.num_acts)
# trainer = IsaacGymPGFWrapper(env, policy)
# for i in range(100):
#     stats = trainer.train_step(L=1000)
#     print(f"Iter {i}: {stats}")
