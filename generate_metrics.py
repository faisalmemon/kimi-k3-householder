import argparse
import csv
import itertools
import math
import os
import statistics
import time

import torch


def run_sequential(S, V, beta):
    C = V.shape[0]
    S_curr = S.clone()
    for j in range(C):
        v_j = V[j]
        b_j = beta[j]
        proj = torch.matmul(S_curr, v_j.to(S_curr.dtype))
        S_curr = S_curr - b_j.to(S_curr.dtype) * torch.outer(proj, v_j.to(S_curr.dtype))
    return S_curr


def build_WY(V, beta):
    d_dim, C_dim = V.shape[1], V.shape[0]
    W = torch.zeros(d_dim, C_dim, device=V.device, dtype=V.dtype)
    Y = torch.zeros(d_dim, C_dim, device=V.device, dtype=V.dtype)

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
    if isinstance(value, list):
        return value
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


def mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def collect_environment(device):
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "cpu"
    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.version.cuda is not None else "none",
    }


def benchmark_case(d, C, batch, args, device):
    seq_apply_ms_list = []
    wy_build_ms_list = []
    wy_apply_ms_list = []
    max_abs_diff_list = []
    max_rel_diff_list = []

    for trial in range(args.trials):
        case_seed = args.seed + d * 1_000_003 + C * 9_173 + batch * 101 + trial * 97
        torch.manual_seed(case_seed)

        V = torch.randn(C, d, device=device, dtype=args.torch_dtype)
        V = V / torch.norm(V, dim=-1, keepdim=True)
        beta = torch.rand(C, device=device, dtype=args.torch_dtype)
        S = torch.randn(batch, d, device=device, dtype=args.torch_dtype)

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

        abs_diff = torch.abs(res_seq - res_wy)
        rel_diff = abs_diff / (torch.abs(res_seq) + 1e-8)

        seq_apply_ms_list.append(seq_apply_s * 1_000.0)
        wy_build_ms_list.append(wy_build_s * 1_000.0)
        wy_apply_ms_list.append(wy_apply_s * 1_000.0)
        max_abs_diff_list.append(torch.max(abs_diff).item())
        max_rel_diff_list.append(torch.max(rel_diff).item())

    seq_apply_ms_mean, seq_apply_ms_std = mean_std(seq_apply_ms_list)
    wy_build_ms_mean, wy_build_ms_std = mean_std(wy_build_ms_list)
    wy_apply_ms_mean, wy_apply_ms_std = mean_std(wy_apply_ms_list)
    max_abs_diff_mean, max_abs_diff_std = mean_std(max_abs_diff_list)
    max_rel_diff_mean, max_rel_diff_std = mean_std(max_rel_diff_list)

    wy_total_ms_mean = wy_build_ms_mean + wy_apply_ms_mean
    wy_total_ms_std = math.sqrt((wy_build_ms_std**2) + (wy_apply_ms_std**2))

    amortized_upfront_ms_per_apply = wy_build_ms_mean / args.iters
    wy_effective_apply_ms = wy_apply_ms_mean + amortized_upfront_ms_per_apply

    speedup_apply_only = seq_apply_ms_mean / wy_apply_ms_mean if wy_apply_ms_mean > 0 else math.inf
    speedup_total_once = seq_apply_ms_mean / wy_total_ms_mean if wy_total_ms_mean > 0 else math.inf
    speedup_including_upfront = (
        seq_apply_ms_mean / wy_effective_apply_ms if wy_effective_apply_ms > 0 else math.inf
    )

    if seq_apply_ms_mean > wy_apply_ms_mean:
        break_even_applies = wy_build_ms_mean / (seq_apply_ms_mean - wy_apply_ms_mean)
    else:
        break_even_applies = math.inf

    env = collect_environment(device)

    return {
        "device": env["device"],
        "gpu_name": env["gpu_name"],
        "torch_version": env["torch_version"],
        "cuda_version": env["cuda_version"],
        "dtype": str(args.torch_dtype),
        "d": d,
        "C": C,
        "batch": batch,
        "trials": args.trials,
        "iters": args.iters,
        "warmup_iters": args.warmup_iters,
        "build_iters": args.build_iters,
        "max_abs_diff_mean": max_abs_diff_mean,
        "max_abs_diff_std": max_abs_diff_std,
        "max_rel_diff_mean": max_rel_diff_mean,
        "max_rel_diff_std": max_rel_diff_std,
        "seq_apply_ms_mean": seq_apply_ms_mean,
        "seq_apply_ms_std": seq_apply_ms_std,
        "wy_build_ms_mean": wy_build_ms_mean,
        "wy_build_ms_std": wy_build_ms_std,
        "wy_apply_ms_mean": wy_apply_ms_mean,
        "wy_apply_ms_std": wy_apply_ms_std,
        "wy_total_ms_mean": wy_total_ms_mean,
        "wy_total_ms_std": wy_total_ms_std,
        "amortized_upfront_ms_per_apply": amortized_upfront_ms_per_apply,
        "wy_effective_apply_ms": wy_effective_apply_ms,
        "speedup_apply_only": speedup_apply_only,
        "speedup_total_once": speedup_total_once,
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
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--build-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="householder_metrics.csv")
    parser.add_argument("--sweep", choices=["none", "C", "d", "batch", "all"], default="none")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    args.torch_dtype = dtype_map[args.dtype]

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_grid_and_write(d_values, C_values, batch_values, output_path):
        rows = []
        grid = itertools.product(d_values, C_values, batch_values)
        for d, C, batch in grid:
            row = benchmark_case(d, C, batch, args, device)
            rows.append(row)

        fieldnames = list(rows[0].keys())
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: format_float(v) for k, v in row.items()})

        print(f"Wrote {len(rows)} rows to {output_path}")
        for row in rows:
            print(
                "d={d}, C={C}, batch={batch}, "
                "seq={seq_m:.4f}+-{seq_s:.4f} ms, "
                "wy_build={build_m:.4f}+-{build_s:.4f} ms, "
                "wy_apply={wy_m:.4f}+-{wy_s:.4f} ms, "
                "speedup_apply={sp_a:.2f}x, speedup_total_once={sp_t:.2f}x, "
                "rel_err={rel:.2e}".format(
                    d=row["d"],
                    C=row["C"],
                    batch=row["batch"],
                    seq_m=row["seq_apply_ms_mean"],
                    seq_s=row["seq_apply_ms_std"],
                    build_m=row["wy_build_ms_mean"],
                    build_s=row["wy_build_ms_std"],
                    wy_m=row["wy_apply_ms_mean"],
                    wy_s=row["wy_apply_ms_std"],
                    sp_a=row["speedup_apply_only"],
                    sp_t=row["speedup_total_once"],
                    rel=row["max_rel_diff_mean"],
                )
            )

    env = collect_environment(device)
    print("Environment:")
    print(f"  device: {env['device']}")
    print(f"  gpu_name: {env['gpu_name']}")
    print(f"  torch_version: {env['torch_version']}")
    print(f"  cuda_available: {env['cuda_available']}")
    print(f"  cuda_version: {env['cuda_version']}")
    print(f"  dtype: {args.dtype}")

    if args.sweep == "none":
        run_grid_and_write(args.d_values, args.C_values, args.batch_values, args.output)
        return

    os.makedirs(args.results_dir, exist_ok=True)

    if args.sweep in ("C", "all"):
        run_grid_and_write(
            d_values=[4096],
            C_values=[4, 8, 16, 32, 64],
            batch_values=[8],
            output_path=os.path.join(args.results_dir, "sweep_C.csv"),
        )

    if args.sweep in ("d", "all"):
        run_grid_and_write(
            d_values=[512, 1024, 2048, 4096],
            C_values=[16],
            batch_values=[8],
            output_path=os.path.join(args.results_dir, "sweep_d.csv"),
        )

    if args.sweep in ("batch", "all"):
        run_grid_and_write(
            d_values=[4096],
            C_values=[16],
            batch_values=[1, 4, 8, 16, 32],
            output_path=os.path.join(args.results_dir, "sweep_batch.csv"),
        )


if __name__ == "__main__":
    main()