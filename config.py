# =============================================================================
# RAG-Enhanced LLM-Coordinated MARL for 6G Edge-Cloud Task Offloading
# Configuration & Hyperparameters
# =============================================================================

# -----------------------------------------------------------------------------
# Network Topology
# -----------------------------------------------------------------------------
M = 5                        # Number of edge nodes
K = 20                       # Number of IoT devices

# -----------------------------------------------------------------------------
# Wireless Channel Parameters
# -----------------------------------------------------------------------------
BANDWIDTH_MIN_HZ   = 50e6    # Minimum channel bandwidth (Hz)  [50 MHz]
BANDWIDTH_MAX_HZ   = 200e6   # Maximum channel bandwidth (Hz)  [200 MHz]
TX_POWER_MIN_W     = 0.1     # Minimum transmit power (W)
TX_POWER_MAX_W     = 2.0     # Maximum transmit power (W)
NOISE_DENSITY_DBM  = -174    # Thermal noise power spectral density (dBm/Hz)

# -----------------------------------------------------------------------------
# Task Characteristics
# -----------------------------------------------------------------------------
TASK_DATA_MIN_MB   = 0.5     # Minimum task input data size (MB)
TASK_DATA_MAX_MB   = 5.0     # Maximum task input data size (MB)
CPU_CYCLES_MIN     = 100e6   # Minimum required CPU cycles (cycles)  [100 Mcycles]
CPU_CYCLES_MAX     = 1000e6  # Maximum required CPU cycles (cycles)  [1000 Mcycles]
DEADLINE_MIN_S     = 0.5     # Minimum task deadline (seconds)
DEADLINE_MAX_S     = 3.0     # Maximum task deadline (seconds)

# -----------------------------------------------------------------------------
# Task Arrival Process
# -----------------------------------------------------------------------------
POISSON_LAMBDA     = 5       # Average task arrival rate (tasks/slot), Poisson process

# -----------------------------------------------------------------------------
# Edge Server Compute Resources
# -----------------------------------------------------------------------------
SERVER_CPU_MIN_HZ  = 2e9     # Minimum server CPU frequency (Hz)  [2 GHz]
SERVER_CPU_MAX_HZ  = 10e9    # Maximum server CPU frequency (Hz)  [10 GHz]
QUEUE_CAPACITY     = 20      # Maximum tasks per edge server queue

# -----------------------------------------------------------------------------
# Energy Model
# -----------------------------------------------------------------------------
KAPPA              = 1e-27   # Effective switched capacitance coefficient (J/cycle^3)

# -----------------------------------------------------------------------------
# MARL (Multi-Agent Reinforcement Learning)
# -----------------------------------------------------------------------------
GAMMA              = 0.99    # Discount factor
LR                 = 3e-4    # Learning rate (Adam optimizer)
BATCH_SIZE         = 256     # Mini-batch size for policy updates
HIDDEN_DIM         = 256     # Hidden layer dimension for actor/critic networks
N_COORDINATION     = 10      # LLM coordination interval (steps between LLM queries)

# -----------------------------------------------------------------------------
# Reward Function Weights  (w1 + w2 + w3 == 1.0)
# -----------------------------------------------------------------------------
W1                 = 0.5     # Weight for latency penalty
W2                 = 0.3     # Weight for energy consumption penalty
W3                 = 0.2     # Weight for deadline violation penalty

# -----------------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation)
# -----------------------------------------------------------------------------
RAG_TOP_K          = 5       # Number of similar past experiences to retrieve
EMBEDDING_DIM      = 384     # Sentence-embedding dimension (e.g. all-MiniLM-L6-v2)

# -----------------------------------------------------------------------------
# LLM COORDINATOR SETTINGS
# -----------------------------------------------------------------------------
USE_GROQ           = True        # True → Groq API; False → OpenAI API

# Groq
GROQ_API_KEY = ""  # Set via environment variable
GROQ_MODEL         = "llama-3.1-8b-instant"
GROQ_BASE_URL      = "https://api.groq.com/openai/v1"

# OpenAI
LLM_MODEL_OPENAI   = "gpt-4o-mini"
OPENAI_API_KEY     = ""          # Fill in before running final paper experiments

# Active model (resolved at import time)
ACTIVE_LLM_MODEL   = GROQ_MODEL if USE_GROQ else LLM_MODEL_OPENAI

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
TOTAL_STEPS        = 500_000  # Total environment interaction steps

# -----------------------------------------------------------------------------
# Filesystem Paths
# -----------------------------------------------------------------------------
RESULTS_DIR        = "results/"
MODELS_DIR         = "models/"
LOGS_DIR           = "logs/"
