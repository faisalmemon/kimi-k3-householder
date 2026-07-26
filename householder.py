import torch
import time

# Setup dimensions matching real LLM attention layers
d = 4096      # Hidden state dimension (e.g., Kimi / Llama)
C = 16        # Number of sequential updates (Chunk size)
batch = 8     # Batch size

# Create synthetic sequence of Householder vectors v_i and scalars beta_i
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V = torch.randn(C, d, device=device)  # 16 Householder vectors
V = V / torch.norm(V, dim=-1, keepdim=True) # Normalize
beta = torch.rand(C, device=device)

# The input state matrix S (batch, d)
S = torch.randn(batch, d, device=device)

# -------------------------------------------------------------
# METHOD A: Sequential Loop (Level 2 Operations)
# -------------------------------------------------------------
def run_sequential(S, V, beta):
    S_curr = S.clone()
    for j in range(C):
        v_j = V[j] # (d,)
        b_j = beta[j]
        # S_next = S_curr * (I - beta * v * v^T)
        # S_curr @ v_j is a vector dot product per batch item
        proj = torch.matmul(S_curr, v_j) # (batch,)
        S_curr = S_curr - b_j * torch.outer(proj, v_j)
    return S_curr

# -------------------------------------------------------------
# METHOD B: WY Representation (Level 3 Operations)
# -------------------------------------------------------------
def build_WY(V, beta):
    d_dim, C_dim = V.shape[1], V.shape[0]
    W = torch.zeros(d_dim, C_dim, device=V.device)
    Y = torch.zeros(d_dim, C_dim, device=V.device)
    
    # Construct W and Y according to Algorithm 5.1.2
    W[:, 0] = beta[0] * V[0]
    Y[:, 0] = V[0]
    
    for j in range(1, C_dim):
        v_j = V[j]
        b_j = beta[j]
        
        # z = beta_j * (I - W * Y^T) * v_j
        # Compute (Y^T @ v_j) first!
        Yt_v = torch.matmul(Y[:, :j].T, v_j) # (j,)
        WYt_v = torch.matmul(W[:, :j], Yt_v)  # (d,)
        z = b_j * (v_j - WYt_v)
        
        W[:, j] = z
        Y[:, j] = v_j
        
    return W, Y

def run_wy(S, W, Y):
    # Compute: S_out = S - S @ W @ Y^T
    # Group parenthesization for GEMM: (S @ W) @ Y^T
    SW = torch.matmul(S, W)       # (batch, C) - Level 3 GEMM
    SWYt = torch.matmul(SW, Y.T)  # (batch, d) - Level 3 GEMM
    return S - SWYt

# -------------------------------------------------------------
# BENCHMARK EXECUTION
# -------------------------------------------------------------
# Warmup CUDA
W, Y = build_WY(V, beta)
_ = run_sequential(S, V, beta)
_ = run_wy(S, W, Y)
if torch.cuda.is_available(): torch.cuda.synchronize()

# Time Sequential
start = time.perf_counter()
for _ in range(100):
    res_seq = run_sequential(S, V, beta)
if torch.cuda.is_available(): torch.cuda.synchronize()
time_seq = (time.perf_counter() - start) / 100

# Time WY Apply Step
start = time.perf_counter()
for _ in range(100):
    res_wy = run_wy(S, W, Y)
if torch.cuda.is_available(): torch.cuda.synchronize()
time_wy = (time.perf_counter() - start) / 100

# Verify correctness (Output matrices should match!)
diff = torch.max(torch.abs(res_seq - res_wy)).item()

print(f"Device: {device}")
print(f"Numerical Difference between methods: {diff:.2e}")
print(f"Sequential Execution Time: {time_seq * 1000:.3f} ms")
print(f"WY Execution Time:         {time_wy * 1000:.3f} ms")
print(f"Speedup Factor:            {time_seq / time_wy:.2f}x")
