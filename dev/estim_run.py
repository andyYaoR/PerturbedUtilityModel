"""
Driver for the debiased-FY estimation Monte Carlo study.

Runs one or all claim experiments (see :mod:`estim_claims`).  ``--quick`` uses
small networks and few replications for a fast smoke run; the default sizes are
the study sizes.

Run:
    python dev/estim_run.py --claim 1 --quick
    python dev/estim_run.py --claim all
    python dev/estim_run.py --claim 7 --network ChicagoSketch
    python dev/estim_run.py --claim 1 --network SiouxFalls --out results/claim1_sf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from estim_claims import ALL


def main() -> None:
    """Parse arguments and run the requested claim experiment(s)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", default="all", choices=["all", *ALL.keys()])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--network",
        default=None,
        help="Network: a TNTP name (SiouxFalls, ChicagoSketch, ChicagoRegional) or "
        "'synth-<n>'.  Omit for each claim's built-in synthetic graph.",
    )
    ap.add_argument("--out", default=None, help="Write the JSON summary to this path.")
    ap.add_argument(
        "--reps", type=int, default=None, help="Cap replications per cell (cost control)."
    )
    args = ap.parse_args()

    import estim_claims

    estim_claims.MAX_REPS = args.reps

    claims = ALL.keys() if args.claim == "all" else [args.claim]
    results = {}
    for c in claims:
        print("=" * 78)
        net = args.network or "synthetic (built-in)"
        print(f"CLAIM {c} [{net}]: {ALL[c].__doc__.strip().splitlines()[0]}")
        print("=" * 78)
        results[c] = ALL[c](args.quick, network=args.network)
        print()
    blob = json.dumps({"network": args.network, "results": results}, indent=2, default=float)
    print("SUMMARY (json):")
    print(blob)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(blob)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
