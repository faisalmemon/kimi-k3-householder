import argparse
import csv
import itertools
import math
import time

import torch


def run_sequential(S, V, beta):
    C = V.shape[0]
    S_curr = S.clone()
    for j in range(C):
        v_j = V[j]
        b_j = beta[j]
        proj = torch.matmul(S_curr, v_j)
        S_curr = S_curr - b_j * torch.outer(proj, v_j)
    return S_curr


def build_WY(V, beta):
    d_dim, C_dim = V.shape[1], V.shape[0]
    W = torch.zeros(d_dim, C_dim, device=V.device)
    Y = torch.zeros(d_dim, C_dim, device=V.device)

    W[:, 0] = beta[0] * V[0]
    Y[:, 0] = V[0]

    for j in range(1, C_dim):
        v_j = V[j]
        b_j = beta[j]

        Yt_v = torch.matmul(Y[:, :j].T, v_j)
        WYt_v = torch.matmul(W[:, :j], Yt_v)
        z = b_j * (v_j - WYt_v)

        W[:, j] = z
        Y[:, j] = v_j

    return W, Y


def run_wy(S, W, Y):
    SW = torch.matmul(S, W)
    SWYt = torch.matmul(SW, Y.T)
    return S - SWYt


def parse_int_list(value):
    values = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value")
    return values


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_avg(fn, repeats, device):
    out = fn()
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    sync_if_needed(device)
    elapsed = (time.perf_counter() - start) / repeats
    return elapsed, out


def benchmark_case(d, C, batch, args, device):
    case_seed = args.seed + d * 1_000_003 + C * 9_173 + batch * 101
    torch.manual_seed(case_seed)

    V = torch.randn(C, d, device=device)
    V = V / torch.norm(V, dim=-1, keepdim=True)
    beta = torch.rand(C, device=device)
    S = torch.randn(batch, d, device=device)

    # Warm up kernels and memory allocations.
    W_warm, Y_warm = build_WY(V, beta)
    for _ in range(args.warmup_iters):
        _ = run_sequential(S, V, beta)
        _ = run_wy(S, W_warm, Y_warm)
    sync_if_needed(device)

    wy_build_s, WY = time_avg(lambda: build_WY(V, beta), args.build_iters, device)
    W, Y = WY

    seq_apply_s, res_seq = time_avg(lambda: run_sequential(S, V, beta), args.iters, device)
    wy_apply_s, res_wy = time_avg(lambda: run_wy(S, W, Y), args.iters, device)

    max_abs_diff = torch.max(torch.abs(res_seq - res_wy)).item()

    seq_apply_ms = seq_apply_s * 1_000.0
    wy_apply_ms = wy_apply_s * 1_000.0
    upfront_build_ms = wy_build_s * 1_000.0
    amortized_upfront_ms_per_apply = upfront_build_ms / args.iters
    wy_effective_apply_ms = wy_apply_ms + amortized_upfront_ms_per_apply

    speedup_apply_only = seq_apply_ms / wy_apply_ms if wy_apply_ms > 0 else math.inf
    speedup_including_upfront = (
        seq_apply_ms / wy_effective_apply_ms if wy_effective_apply_ms > 0 else math.inf
    )

    if seq_apply_ms > wy_apply_ms:
        break_even_applies = upfront_build_ms / (seq_apply_ms - wy_apply_ms)
    else:
        break_even_applies = math.inf

    return {
        "device": str(device),
        "d": d,
        "C": C,
        "batch": batch,
        "iters": args.iters,
        "warmup_iters": args.warmup_iters,
        "build_iters": args.build_iters,
        "max_abs_diff": max_abs_diff,
        "seq_apply_ms": seq_apply_ms,
        "wy_apply_ms": wy_apply_ms,
        "upfront_build_ms": upfront_build_ms,
        "amortized_upfront_ms_per_apply": amortized_upfront_ms_per_apply,
        "wy_effective_apply_ms": wy_effective_apply_ms,
        "speedup_apply_only": speedup_apply_only,
        "speedup_including_upfront": speedup_including_upfront,
        "break_even_applies": break_even_applies,
    }


def format_float(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.8g}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Householder benchmark metrics for multiple d, C, batch settings "
            "and save results to CSV."
        )
    )
    parser.add_argument("--d-values", type=parse_int_list, default=[1024, 2048, 4096])
    parser.add_argument("--C-values", type=parse_int_list, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--batch-values", type=parse_int_list, default=[4, 8, 16])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--build-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="householder_metrics.csv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fieldnames = [
        "device",
        "d",
        "C",
        "batch",
        "iters",
        "warmup_iters",
        "build_iters",
        "max_abs_diff",
        "seq_apply_ms",
        "wy_apply_ms",
        "upfront_build_ms",
        "amortized_upfront_ms_per_apply",
        "wy_effective_apply_ms",
        "speedup_apply_only",
        "speedup_including_upfront",
        "break_even_applies",
    ]

    rows = []
    grid = itertools.product(args.d_values, args.C_values, args.batch_values)
    for d, C, batch in grid:
        row = benchmark_case(d, C, batch, args, device)
        rows.append(row)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_float(v) for k, v in row.items()})

    print(f"Wrote {len(rows)} rows to {args.output}")
    for row in rows:
        print(
            "d={d}, C={C}, batch={batch}, diff={diff:.2e}, "
            "seq={seq:.4f} ms, wy={wy:.4f} ms, upfront={upfront:.4f} ms, "
            "eff_wy={eff:.4f} ms, speedup={speedup:.2f}x".format(
                d=row["d"],
                C=row["C"],
                batch=row["batch"],
                diff=row["max_abs_diff"],
                seq=row["seq_apply_ms"],
                wy=row["wy_apply_ms"],
                upfront=row["upfront_build_ms"],
                eff=row["wy_effective_apply_ms"],
                speedup=row["speedup_including_upfront"],
            )
        )


if __name__ == "__main__":
    main()