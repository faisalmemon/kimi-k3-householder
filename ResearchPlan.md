# Research Plan: WY Representation as a Kernel-Level Speedup in Kimi Delta Attention

## Research Question

Can the WY representation of a sequence of Householder (rank-1) updates be shown,
through controlled benchmarking, to deliver meaningful speedup over sequential
application — and does this speedup vary systematically with dimension, chunk size,
and batch size?

---

## Background and Motivation

Kimi K3 (2.8T parameter model, released 2026-07-27) is documented as using Kimi Delta
Attention (KDA), a hybrid linear attention mechanism. The Kimi Linear technical report
(arXiv:2510.26692) explains that KDA packs sequences of rank-1 state updates into a
compact WY-style representation, replacing bandwidth-limited sequential vector operations
with GEMM-heavy matrix-matrix operations. This is one of two main efficiency gains
claimed by the paper; the other (reduced KV-cache via hybrid linear/full-attention
layering) is out of scope for this project.

This mini research project reproduces and measures the kernel-level speedup in isolation,
using a clean PyTorch benchmark in a reproducible Docker environment.

---

## Reproducibility

The full benchmark runs inside a Docker container (see `Dockerfile` in this repo).
A reviewer can reproduce all results with a single command:

```bash
bash run_script.sh
```

Fixed random seed (`torch.manual_seed(42)`) is set throughout. Hardware and software
versions should be recorded in the report (GPU model, CUDA version, PyTorch version).

---

## Artifacts to Produce

1. `householder.py` — core benchmark (already written, to be extended with sweeps)
2. `results/` — CSV or JSON files with raw timing data from sweeps
3. `plots/` — figures generated from results
4. `ResearchReport.md` — the final 1-2 page written report

---

## Week 1: Stabilise the Benchmark Harness

### Goals
- Confirm the existing benchmark is numerically correct and timing is reliable.
- Add mean and standard deviation across repeated trials (not just one averaged loop).
- Separate and report WY **build** cost vs WY **apply** cost, so readers understand
  amortisation behaviour.
- Add relative error metric alongside max absolute difference.

### Checklist
- [ ] Replace single-loop timing with multiple independent runs; report mean ± std.
- [ ] Time `build_WY` separately from `run_wy` and print both.
- [ ] Add relative error: `torch.max(torch.abs(res_seq - res_wy) / (torch.abs(res_seq) + 1e-8))`.
- [ ] Run on both CPU and GPU; confirm results are consistent.
- [ ] Record environment: GPU model, CUDA version, PyTorch version — print at script start.

---

## Week 2: Run Controlled Sweeps

### Goals
- Show when and how much WY wins across a range of realistic parameters.
- Produce data for at least one clear figure and one table.

### Sweep 1 — Chunk size C (fixed d=4096, batch=8)
Values: C ∈ {4, 8, 16, 32, 64}

### Sweep 2 — Hidden dimension d (fixed C=16, batch=8)
Values: d ∈ {512, 1024, 2048, 4096}

### Sweep 3 — Batch size (fixed d=4096, C=16)
Values: batch ∈ {1, 4, 8, 16, 32}

### For each sweep, record
- Sequential time (mean ± std)
- WY build time (mean ± std)
- WY apply time (mean ± std)
- WY total time = build + apply (mean ± std)
- Speedup = sequential / WY-apply (apply-only, upper bound)
- Speedup = sequential / WY-total (realistic, amortised once)
- Max absolute error and relative error

### Checklist
- [ ] Write a sweep driver script or extend `householder.py` with a `--sweep` flag.
- [ ] Save results to `results/sweep_C.csv`, `results/sweep_d.csv`, `results/sweep_batch.csv`.
- [ ] Generate one plot per sweep: speedup on y-axis, parameter on x-axis.
- [ ] Check reproducibility: run twice and confirm numbers are stable.

---

## Week 3: Write the Report

### File: `ResearchReport.md`

Structure (target 1-2 pages):

#### 1. Problem
One paragraph. What is the kernel-level WY speedup claim, and why does it matter
for models like Kimi K3?

#### 2. Method
- Two methods described (Sequential Loop vs WY Representation).
- Key design choices: fixed seed, Docker container for reproducibility, timing
  protocol (warmup, synchronise, mean ± std).
- Hardware and software versions.

#### 3. Results
- One table: speedup vs chunk size C (apply-only and amortised).
- One figure: speedup vs hidden dimension d.
- Correctness: numerical error is negligible (report max relative error).

#### 4. Limitations and Threats to Validity
- Synthetic data only; real attention inputs may differ.
- Only the kernel-level effect is measured; system-level KV-cache reduction is not.
- WY build cost is amortised over one forward pass; real gains depend on reuse pattern.
- Single GPU model; results may not generalise to other hardware.

#### 5. Next Experiments
- Benchmark with half precision (fp16/bf16) inputs.
- Compare against a fused Triton kernel for the same operation.
- Extend to simulate the intra-chunk vs inter-chunk split from the KDA paper.

---

## CV Bullet Points (draft — fill in measured numbers)

> Reproduced and benchmarked the WY representation speedup underlying Kimi Delta
> Attention (Kimi K3 / Kimi Linear), demonstrating up to **Xx** kernel-level
> acceleration over sequential Householder application across hidden dimensions
> 512–4096. Produced reproducible experiments (Docker), controlled parameter sweeps,
> and a short technical report documenting scope and validity limits.

---

## Key References

- Kimi Linear technical report: arXiv:2510.26692
- Kimi K3 documentation: https://platform.kimi.ai/docs
- WY representation original paper: Bischof & Van Loan, SIAM J. Sci. Stat. Comput. (1987)
- BLAS level definitions: Lawson et al., ACM TOMS (1979)
