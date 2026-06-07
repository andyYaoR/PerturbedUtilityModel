"""
Figures for the debiased-FY simulation study (paper ``sec:simulation``).

Reads the JSON written by ``estim_run.py --out`` (structure
``{"network": ..., "results": {"1": {...}, ...}}``) and emits one PDF per claim
into ``--outdir`` (default ``figures/``).  Only the claims present in the JSON are
plotted, so partial runs still produce their figures.

Run:
    python dev/estim_run.py --claim all --out results/synth.json
    python dev/estim_plots.py --results results/synth.json --outdir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_GAMMA_LABELS = [r"$\gamma_3$", r"$\gamma_4$", r"$\gamma_5$", r"$\gamma_6$", r"$\gamma_7$"]


def _save(fig, path: Path) -> None:
    """Save and close a figure, reporting the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")


def fig_claim1(r: dict, path: Path) -> None:
    """Debiased vs naive |gamma-score| vs demand D (log y): debiased ~0, naive ~1/D."""
    Ds = sorted(int(d) for d in r)
    deb = [np.mean(np.abs(r[str(d)]["debiased_gamma_score_mean"])) for d in Ds]
    nav = [np.mean(np.abs(r[str(d)]["naive_gamma_score_mean"])) for d in Ds]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.loglog(Ds, nav, "o-", label="naive plug-in")
    ax.loglog(Ds, deb, "s-", label="debiased (U-stat)")
    ref = nav[0] * Ds[0] / np.array(Ds, float)
    ax.loglog(Ds, ref, "k:", alpha=0.6, label=r"$\propto 1/D$")
    ax.set_xlabel("demand $D$")
    ax.set_ylabel(r"mean $|\gamma\text{-score}|$ at $\theta_0$")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_claim2(r: dict, path: Path) -> None:
    """Coordinate slices of Q through theta_0 (convex bowls minimized at offset 0)."""
    sl = r["slices"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for j, s in sl.items():
        o = np.array(s["offsets"])
        q = np.array(s["Q"], float)
        q[~np.isfinite(q)] = np.nan
        ax.plot(o, q, ".-", lw=1, label=f"coord {j}")
    ax.axvline(0.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel(r"offset from $\theta_0$ in one coordinate")
    ax.set_ylabel(r"$Q_B(\theta)$")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, path)


def fig_claim3(r: dict, path: Path) -> None:
    """RMSE vs B on log-log with the fitted overall slope and the a/B+b/B^2 curve."""
    rmse = r["rmse"]
    Bs = sorted(int(b) for b in rmse)
    y = np.array([rmse[str(b)] for b in Bs])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.loglog(Bs, y, "o", label="RMSE")
    Bg = np.array(Bs, float)
    if "mse_fit" in r:
        a, b = r["mse_fit"]["a"], r["mse_fit"]["b"]
        ax.loglog(Bg, np.sqrt(a / Bg + b / Bg**2), "-", label=r"$\sqrt{a/B+b/B^2}$")
    ax.loglog(Bg, y[0] * np.sqrt(Bg[0] / Bg), "k:", alpha=0.6, label=r"slope $-1/2$")
    ax.set_xlabel("number of OD pairs $B$")
    ax.set_ylabel(r"RMSE$(\hat\theta)$")
    ax.set_title(f"overall slope {r.get('slope', float('nan')):.2f}", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_claim4(r: dict, path: Path) -> None:
    """Empirical SD vs mean sandwich SE per parameter, with coverage annotated."""
    sd = np.array(r["emp_sd"], float)
    se = np.array(r["mean_se"], float)
    cov = np.array(r["coverage"], float)
    x = np.arange(len(sd))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    w = 0.38
    ax.bar(x - w / 2, sd, w, label="empirical SD")
    ax.bar(x + w / 2, se, w, label="mean sandwich SE")
    for xi, c in zip(x, cov):
        ax.text(xi, max(sd[xi], se[xi]) * 1.02, f"{c:.0%}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\theta_{{{i}}}$" for i in range(len(sd))])
    ax.set_ylabel("standard error")
    ax.set_title("bars: SD vs SE;  labels: 95% CI coverage", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_claim5(r: dict, path: Path) -> None:
    """Heatmap of per-gamma_l median |error| over the (B, D) grid (one panel per l)."""
    Bs, Ds = r["Bs"], r["Ds"]
    grid = r.get("mae_per_gamma") or r["rmse_per_gamma"]
    ng = len(next(iter(grid.values())))
    fig, axes = plt.subplots(1, ng, figsize=(2.6 * ng, 3.0), squeeze=False)
    for li in range(ng):
        M = np.array([[grid[f"B{B}_D{D}"][li] for D in Ds] for B in Bs], float)
        ax = axes[0][li]
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(Ds)))
        ax.set_xticklabels(Ds)
        ax.set_yticks(range(len(Bs)))
        ax.set_yticklabels(Bs)
        ax.set_xlabel("$D$")
        if li == 0:
            ax.set_ylabel("$B$")
        ax.set_title(_GAMMA_LABELS[li] + r" median$|$err$|$", fontsize=9)
        for bi in range(len(Bs)):
            for di in range(len(Ds)):
                ax.text(di, bi, f"{M[bi, di]:.2f}", ha="center", va="center",
                        color="w", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def fig_claim6(r: dict, path: Path) -> None:
    """gamma_4 estimate and RMSE under R_+ vs Bernstein projection (true gamma_4=-0.3)."""
    projs = [p for p in ("nonneg", "bernstein") if p in r]
    means = [r[p]["gamma4_mean"] for p in projs]
    rmses = [r[p]["gamma4_rmse"] for p in projs]
    x = np.arange(len(projs))
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.bar(x, means, 0.5, label=r"$\hat\gamma_4$ mean")
    ax.axhline(-0.3, color="k", ls="--", label=r"true $\gamma_4=-0.3$")
    for xi, rm in zip(x, rmses):
        ax.text(xi, means[xi], f"RMSE {rm:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\mathbb{R}_+$" if p == "nonneg" else "Bernstein" for p in projs])
    ax.set_ylabel(r"$\hat\gamma_4$")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def fig_claim7(r: dict, path: Path) -> None:
    """Cold vs lambda-warm inner IPM iterations across the theta-sweep."""
    cold = r["cold_iters"]
    warm = r["warm_iters"]
    x = np.arange(len(cold))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(x, cold, "o-", label="cold start")
    ax.plot(x, warm, "s--", label=r"$\lambda$-warm start")
    ax.set_xlabel("sweep step")
    ax.set_ylabel("inner IPM iterations")
    ax.set_ylim(bottom=0)
    ax.set_title(f"oracle max|x-x*| = {r.get('oracle_maxdiff', float('nan')):.0e}", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


_PLOTS = {
    "1": fig_claim1,
    "2": fig_claim2,
    "3": fig_claim3,
    "4": fig_claim4,
    "5": fig_claim5,
    "6": fig_claim6,
    "7": fig_claim7,
}


def main() -> None:
    """Render every claim figure present in the results JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="JSON written by estim_run.py --out")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--tag", default=None, help="filename suffix (e.g. network name)")
    args = ap.parse_args()

    blob = json.loads(Path(args.results).read_text())
    results = blob.get("results", blob)
    tag = args.tag or blob.get("network") or "synth"
    outdir = Path(args.outdir)
    for c, res in results.items():
        if c in _PLOTS:
            try:
                _PLOTS[c](res, outdir / f"claim{c}_{tag}.pdf")
            except Exception as exc:  # pragma: no cover - a partial/odd result dict
                print(f"[skip] claim {c}: {exc}")


if __name__ == "__main__":
    main()
