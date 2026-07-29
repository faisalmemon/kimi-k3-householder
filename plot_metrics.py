import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def to_number(text):
    text = text.strip()
    if text == "inf":
        return float("inf")
    try:
        return int(text)
    except ValueError:
        return float(text)


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for k, v in raw.items():
                if k in {"device", "gpu_name", "torch_version", "cuda_version"}:
                    row[k] = v
                    continue
                try:
                    row[k] = to_number(v)
                except ValueError:
                    row[k] = v

            # Backward/forward compatibility: normalize to common plotting keys.
            if "seq_apply_ms" not in row and "seq_apply_ms_mean" in row:
                row["seq_apply_ms"] = row["seq_apply_ms_mean"]
            if "wy_apply_ms" not in row and "wy_apply_ms_mean" in row:
                row["wy_apply_ms"] = row["wy_apply_ms_mean"]
            if "upfront_build_ms" not in row and "wy_build_ms_mean" in row:
                row["upfront_build_ms"] = row["wy_build_ms_mean"]
            if "speedup_including_upfront" not in row and "speedup_total_once" in row:
                # In research sweeps this is the realistic one-pass speedup.
                row["speedup_including_upfront"] = row["speedup_total_once"]
            if "wy_effective_apply_ms" not in row and "amortized_upfront_ms_per_apply" in row:
                row["wy_effective_apply_ms"] = (
                    float(row["wy_apply_ms"]) + float(row["amortized_upfront_ms_per_apply"])
                )
            rows.append(row)
    return rows


def group_by(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out


def save_speedup_logx_figure(rows, out_path):
    batches = sorted({int(r["batch"]) for r in rows})
    dims = sorted({int(r["d"]) for r in rows})
    c_values = sorted({int(r["C"]) for r in rows})

    fig, axes = plt.subplots(1, len(batches), figsize=(4.8 * len(batches), 4.6), sharey=True)
    if len(batches) == 1:
        axes = [axes]

    for idx, (ax, batch) in enumerate(zip(axes, batches)):
        batch_rows = [r for r in rows if int(r["batch"]) == batch]
        for d in dims:
            d_rows = sorted([r for r in batch_rows if int(r["d"]) == d], key=lambda x: x["C"])
            x = [int(r["C"]) for r in d_rows]
            y = [float(r["speedup_including_upfront"]) for r in d_rows]
            ax.plot(x, y, marker="o", linewidth=2, label=f"d={d}")

        ax.set_xscale("log", base=2)
        ax.set_xticks(c_values)
        ax.set_xticklabels([str(c) for c in c_values])
        ax.set_title(f"batch={batch}", fontsize=10, pad=8)
        ax.set_xlabel("C (log2 scale)")
        ax.grid(True, linestyle="--", alpha=0.35, which="both")
        if idx == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=9)

    axes[0].set_ylabel("Speedup vs sequential (including WY upfront)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_latency_figure(rows, out_path):
    dims = sorted({int(r["d"]) for r in rows})
    batches = sorted({int(r["batch"]) for r in rows})
    c_values = sorted({int(r["C"]) for r in rows})

    fig, axes = plt.subplots(1, len(dims), figsize=(4.8 * len(dims), 4.6), sharey=True)
    if len(dims) == 1:
        axes = [axes]

    target_batch = batches[len(batches) // 2]

    for idx, (ax, d) in enumerate(zip(axes, dims)):
        d_rows = [r for r in rows if int(r["d"]) == d and int(r["batch"]) == target_batch]
        d_rows = sorted(d_rows, key=lambda x: x["C"])

        x = [int(r["C"]) for r in d_rows]
        seq = [float(r["seq_apply_ms"]) for r in d_rows]
        wy_apply = [float(r["wy_apply_ms"]) for r in d_rows]
        wy_eff = [float(r["wy_effective_apply_ms"]) for r in d_rows]

        ax.plot(x, seq, marker="o", linewidth=2, label="Sequential apply")
        ax.plot(x, wy_apply, marker="s", linewidth=2, label="WY apply only")
        ax.plot(x, wy_eff, marker="^", linewidth=2, label="WY effective (amortized)")

        ax.set_title(f"d={d}, batch={target_batch}", fontsize=10, pad=8)
        ax.set_xlabel("C (number of updates)")
        ax.set_xticks(c_values)
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=9)

    axes[0].set_ylabel("Latency (ms)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_break_even_figure(rows, out_path):
    by_batch = group_by(rows, "batch")
    c_values = sorted({int(r["C"]) for r in rows})

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for batch in sorted(by_batch):
        batch_rows = by_batch[batch]
        # Average break-even applies across d for each C.
        by_c = group_by(batch_rows, "C")
        xs = []
        ys = []
        for C in sorted(by_c):
            vals = [float(r["break_even_applies"]) for r in by_c[C]]
            finite_vals = [v for v in vals if v != float("inf")]
            if not finite_vals:
                continue
            xs.append(int(C))
            ys.append(sum(finite_vals) / len(finite_vals))

        ax.plot(xs, ys, marker="o", linewidth=2, label=f"batch={int(batch)}")

    ax.set_title("Break-even Number of Applies")
    ax.set_xlabel("C (number of updates)")
    ax.set_ylabel("Applies needed to amortize WY build")
    ax.set_xticks(c_values)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_latex_snippet(out_dir):
    snippet = """% Add this in your LaTeX report preamble:
% \\usepackage{graphicx}

\\begin{figure}[t]
    \\centering
    \\includegraphics[width=\\linewidth]{fig_speedup_vs_C_logx.pdf}
    \\caption{Same speedup plot with a log2 x-axis for clearer comparison across small and large C.}
    \\label{fig:wy-speedup-logx}
\\end{figure}

\\begin{figure}[t]
  \\centering
  \\includegraphics[width=\\linewidth]{fig_latency_vs_C.pdf}
  \\caption{Sequential vs WY apply latency, including effective WY latency with amortized upfront cost.}
  \\label{fig:wy-latency}
\\end{figure}

\\begin{figure}[t]
  \\centering
  \\includegraphics[width=0.8\\linewidth]{fig_break_even.pdf}
  \\caption{Break-even number of applies needed to amortize WY build cost.}
  \\label{fig:wy-break-even}
\\end{figure}
"""
    with open(os.path.join(out_dir, "latex_figures_snippet.tex"), "w", encoding="utf-8") as f:
        f.write(snippet)


def main():
    parser = argparse.ArgumentParser(description="Create PDF plots from householder metrics CSV.")
    parser.add_argument("--input", default="householder_metrics.csv", help="Path to metrics CSV")
    parser.add_argument("--outdir", default="figures", help="Directory for generated plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.input)
    if not rows:
        raise RuntimeError("No rows found in CSV")

    c_values = sorted({int(r["C"]) for r in rows})
    if len(c_values) < 2:
        print(
            "Skipping plot generation: input is not a C-sweep (only one C value). "
            "This script now only emits focused C-sweep figures."
        )
        return

    save_speedup_logx_figure(rows, os.path.join(args.outdir, "fig_speedup_vs_C_logx.pdf"))
    save_latency_figure(rows, os.path.join(args.outdir, "fig_latency_vs_C.pdf"))
    save_break_even_figure(rows, os.path.join(args.outdir, "fig_break_even.pdf"))
    write_latex_snippet(args.outdir)

    print(f"Generated figures in {args.outdir}:")
    print("- fig_speedup_vs_C_logx.pdf")
    print("- fig_latency_vs_C.pdf")
    print("- fig_break_even.pdf")
    print("- latex_figures_snippet.tex")


if __name__ == "__main__":
    main()