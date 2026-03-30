# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(__file__))

from config import M, K, MODELS_DIR
from env.edge_cloud_env import EdgeCloudEnv, N_ACTIONS_PER_DEVICE
from marl.policies import MAPPOAgent

OBS_DIM    = K * M + 2 * M + 3 * K + K   # 170 + K = 190
CKPT_PATH  = os.path.join(MODELS_DIR, "shared_agent.pt")
ACTION_LABELS = ["Local"] + [f"Edge {m+1}" for m in range(M)] + ["Cloud"]

# -----------------------------------------------------------------------------
# Load checkpoint
# -----------------------------------------------------------------------------
print(f"Loading checkpoint: {CKPT_PATH}")
if not os.path.exists(CKPT_PATH):
    print("ERROR: checkpoint not found. Run training first.")
    sys.exit(1)

device = torch.device("cpu")
agent  = MAPPOAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS_PER_DEVICE, device=device)
ckpt   = torch.load(CKPT_PATH, map_location=device)
agent.actor.load_state_dict(ckpt["actor"])
agent.critic.load_state_dict(ckpt["critic"])
agent.actor.eval()
agent.critic.eval()
print(f"Checkpoint loaded (trained to step {ckpt.get('step', 'unknown')})")

# -----------------------------------------------------------------------------
# Get one observation (device 0)
# -----------------------------------------------------------------------------
env = EdgeCloudEnv(seed=42)
obs, _ = env.reset(seed=42)
# Augment with device-0 one-hot to match training obs_dim=190
device_obs = np.concatenate([obs, np.eye(K, dtype=np.float32)[0]])
obs_tensor = torch.tensor(device_obs, dtype=torch.float32)

# -----------------------------------------------------------------------------
# Action probabilities
# -----------------------------------------------------------------------------
with torch.no_grad():
    logits = agent.actor(obs_tensor)
    probs  = torch.softmax(logits, dim=-1).numpy()
    value  = agent.critic(obs_tensor).item()

print(f"\nObservation shape : {obs.shape}")
print(f"Critic value V(s) : {value:.4f}")
print(f"\n{'Action':<10} {'Probability':>12} {'Bar'}")
print("-" * 40)
for i, (label, p) in enumerate(zip(ACTION_LABELS, probs)):
    bar = "#" * int(p * 40)
    print(f"{label:<10} {p:>11.4f}  {bar}")

top_action = int(np.argmax(probs))
print(f"\nMost probable action: {ACTION_LABELS[top_action]} (p={probs[top_action]:.4f})")

# -----------------------------------------------------------------------------
# Sample 20 actions and show distribution
# -----------------------------------------------------------------------------
dist    = torch.distributions.Categorical(logits=logits)
samples = [int(dist.sample().item()) for _ in range(20)]

counts = {label: 0 for label in ACTION_LABELS}
for s in samples:
    counts[ACTION_LABELS[s]] += 1

print("\n" + "-" * 40)
print(f"20 sampled actions distribution:")
print(f"{'Destination':<10} {'Count':>6}  {'Samples'}")
print("-" * 40)
for label, count in counts.items():
    bar = "*" * count
    print(f"{label:<10} {count:>6}  {bar}")

print(f"\nRaw samples: {[ACTION_LABELS[s] for s in samples]}")
