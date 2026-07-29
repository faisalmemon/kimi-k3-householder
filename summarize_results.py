import argparse
import csv
import math
import os


def parse_value(text):
    text = text.strip()
    if text == "inf":
        return float("inf")
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k: parse_value(v) for k, v in raw.items()}
            rows.append(row)
    return rows


def fmt_ms(mean, std):
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def fmt_float(v, digits=2):
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:.{digits}f}"


def write_latex_table(path, caption, label, headers, body_rows):
    lines = []
    colspec = "l" + "r" * (len(headers) - 1)
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{colspec}}}")
    lines.append("    \\hline")
    lines.append("    " + " & ".join(headers) + " \\\\")
    lines.append("    \\hline")
    for r in body_rows:
        lines.append("    " + " & ".join(r) + " \\\\")
    lines.append("    \\hline")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_table_c(rows, out_dir):
    rows = sorted(rows, key=lambda r: int(r["C"]))
    headers = [
        "C",
        "Seq ms",
        "WY build ms",
        "WY apply ms",
        "Speedup apply",
        "Speedup total",
    ]
    body = []
    for r in rows:
        body.append(
            [
                str(int(r["C"])),
                fmt_ms(float(r["seq_apply_ms_mean"]), float(r["seq_apply_ms_std"])),
                fmt_ms(float(r["wy_build_ms_mean"]), float(r["wy_build_ms_std"])),
                fmt_ms(float(r["wy_apply_ms_mean"]), float(r["wy_apply_ms_std"])),
                f"{fmt_float(float(r['speedup_apply_only']), 2)}x",
                f"{fmt_float(float(r['speedup_total_once']), 2)}x",
            ]
        )
    write_latex_table(
        os.path.join(out_dir, "table_sweep_C.tex"),
        "Chunk-size sweep at d=4096, batch=8.",
        "tab:sweep-c",
        headers,
        body,
    )


def make_table_d(rows, out_dir):
    rows = sorted(rows, key=lambda r: int(r["d"]))
    headers = ["d", "Seq ms", "WY apply ms", "Speedup apply", "Rel err"]
    body = []
    for r in rows:
        body.append(
            [
                str(int(r["d"])),
                fmt_ms(float(r["seq_apply_ms_mean"]), float(r["seq_apply_ms_std"])),
                fmt_ms(float(r["wy_apply_ms_mean"]), float(r["wy_apply_ms_std"])),
                f"{fmt_float(float(r['speedup_apply_only']), 2)}x",
                f"{float(r['max_rel_diff_mean']):.2e}",
            ]
        )
    write_latex_table(
        os.path.join(out_dir, "table_sweep_d.tex"),
        "Hidden-dimension sweep at C=16, batch=8.",
        "tab:sweep-d",
        headers,
        body,
    )


def make_table_batch(rows, out_dir):
    rows = sorted(rows, key=lambda r: int(r["batch"]))
    headers = ["batch", "Seq ms", "WY apply ms", "Speedup apply", "Break-even"]
    body = []
    for r in rows:
        body.append(
            [
                str(int(r["batch"])),
                fmt_ms(float(r["seq_apply_ms_mean"]), float(r["seq_apply_ms_std"])),
                fmt_ms(float(r["wy_apply_ms_mean"]), float(r["wy_apply_ms_std"])),
                f"{fmt_float(float(r['speedup_apply_only']), 2)}x",
                fmt_float(float(r["break_even_applies"]), 2),
            ]
        )
    write_latex_table(
        os.path.join(out_dir, "table_sweep_batch.tex"),
        "Batch-size sweep at d=4096, C=16.",
        "tab:sweep-batch",
        headers,
        body,
    )


def make_key_findings(sweep_c, sweep_d, sweep_b, out_path):
    best_apply = max(sweep_c, key=lambda r: float(r["speedup_apply_only"]))
    best_total = max(sweep_c, key=lambda r: float(r["speedup_total_once"]))
    min_break_even = min(sweep_b, key=lambda r: float(r["break_even_applies"]))
    max_rel = max(
        [float(r["max_rel_diff_mean"]) for r in sweep_c + sweep_d + sweep_b]
    )

    text = []
    text.append("Key findings (GPU):")
    text.append(
        "- In the C sweep (d=4096, batch=8), apply-only speedup increases with C and peaks at "
        f"{float(best_apply['speedup_apply_only']):.2f}x (C={int(best_apply['C'])})."
    )
    text.append(
        "- Including one-time WY build cost, the best single-pass total speedup in the C sweep is "
        f"{float(best_total['speedup_total_once']):.2f}x (C={int(best_total['C'])})."
    )
    text.append(
        "- In the batch sweep (d=4096, C=16), the smallest break-even point is "
        f"{float(min_break_even['break_even_applies']):.2f} applies (batch={int(min_break_even['batch'])})."
    )
    text.append(
        "- Numerical agreement is strong in absolute terms; relative error can become large "
        "when reference values are near zero."
    )
    text.append(f"- Maximum observed mean relative error across sweeps: {max_rel:.2e}.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create report-ready LaTeX tables and a key-findings block from sweep CSVs."
    )
    parser.add_argument("--results-dir", default="results", help="Directory with sweep CSV files")
    parser.add_argument("--outdir", default="report_artifacts", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sweep_c_path = os.path.join(args.results_dir, "sweep_C.csv")
    sweep_d_path = os.path.join(args.results_dir, "sweep_d.csv")
    sweep_b_path = os.path.join(args.results_dir, "sweep_batch.csv")

    sweep_c = load_csv(sweep_c_path)
    sweep_d = load_csv(sweep_d_path)
    sweep_b = load_csv(sweep_b_path)

    if not sweep_c or not sweep_d or not sweep_b:
        raise RuntimeError("One or more sweep CSV files are empty")

    make_table_c(sweep_c, args.outdir)
    make_table_d(sweep_d, args.outdir)
    make_table_batch(sweep_b, args.outdir)
    make_key_findings(sweep_c, sweep_d, sweep_b, os.path.join(args.outdir, "key_findings.txt"))

    print(f"Wrote report artifacts to {args.outdir}:")
    print("- table_sweep_C.tex")
    print("- table_sweep_d.tex")
    print("- table_sweep_batch.tex")
    print("- key_findings.txt")


if __name__ == "__main__":
    main()