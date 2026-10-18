#!/usr/bin/env python3
"""
Submission validator for the Fusion Equilibrium Challenge.

Checks that a submission .npz has the right STRUCTURE before you upload it to Codabench — it does
NOT score (the scorer is held by the organizers and runs on the platform). Catching a malformed
file here saves you a wasted submission slot.

A submission predicts the flux map plus the two scalars that are not derivable from it — q95 and
betaN (grouped per shot; see README → "3. What you predict"). Nothing else is submitted:
the LCFS, R_axis/Z_axis, elongation/triangularity, volume and li are all DERIVED from your
submitted flux map by the scorer (with the same functionals it applies to the ground truth), so a
scalar can only be earned by a flux map that implies it. For a config this streams the
test inputs from Hugging Face (no ground truth needed) -- the public test from the main dataset,
or, for the Final phase's `*_private_test` configs, the private test from its own dataset repo --
and verifies, for every shot in stream order, with T = number of `efit_times`:
  - `shot_XXXX_psirz`  present, shape `(T, H, W)` in the machine's native grid (both dense 65×65:
    DIII-D 65×65, MAST 65×65), floating dtype, and (DIII-D only) no NaN/Inf since its GT is finite,
  - `shot_XXXX_q95` and `shot_XXXX_betaN` present, shape `(T,)`, floating.

Usage:
    python validate_submission.py submission/diii_d_public_test.npz --config diii_d_public_test
    python validate_submission.py submission/mast_public_test.npz  --config mast_public_test
    python validate_submission.py submission/diii_d_public_test.npz --config diii_d_public_test --max-shots 5
    # Final phase (private-test inputs, streamed from the Final repo):
    python validate_submission.py submission/diii_d_private_test.npz --config diii_d_private_test
    python validate_submission.py submission/mast_private_test.npz  --config mast_private_test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
# The Final phase's private-test inputs live in their OWN dataset repo (inputs only, same
# columns as the public test); the main repo above is unchanged. See repo_for().
FINAL_REPO_ID = "ganatma/fusion-equilibrium-challenge-final"
# Native flux grid per machine (rows = Z, cols = R).
GRID = {"DIII-D": (65, 65), "MAST": (65, 65)}
# The only two submitted scalars (metric v2): q95 needs F(psi), betaN needs p(psi) — neither is
# in psi(R,Z). Every other scalar is derived from your submitted flux map by the scorer.
SCALARS = ["q95", "betaN"]
# config -> (split, machine). One test config = one machine.
CONFIG_INFO = {
    "diii_d_public_test": ("public_test", "DIII-D"),
    "mast_public_test": ("public_test", "MAST"),
    # Final phase -- streamed from FINAL_REPO_ID, not REPO_ID.
    "diii_d_private_test": ("private_test", "DIII-D"),
    "mast_private_test": ("private_test", "MAST"),
}
FINAL_CONFIGS = ("diii_d_private_test", "mast_private_test")


def repo_for(config: str) -> str:
    """Hugging Face dataset repo a config streams from: the Final configs -> FINAL_REPO_ID,
    the public-test configs -> REPO_ID."""
    return FINAL_REPO_ID if config in FINAL_CONFIGS else REPO_ID


def _quantization_warning(arr, key: str):
    """Catch fixed-decimal rounding, which silently guts the Consistency term.

    `float16` is safe and recommended -- it keeps RELATIVE precision, so the error scales with
    |psi|. Fixed-decimal rounding (`np.round(psi, 3)`) does not: it leaves an absolute step that
    is invisible in R2_psi but large compared to the spread of the derived shape scalars. Measured
    on real ground truth, a PERFECT prediction rounded to 3 decimals still scores R2_psi 0.99997
    while MAST Consistency collapses to 0.652 (DIII-D 0.979). Participants reach for rounding to
    shrink the upload, so warn loudly rather than let it show up as an unexplained leaderboard gap.

    Detection: if every finite value is an exact multiple of some step much coarser than the local
    float16 resolution, the array has been decimal-rounded.
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < 64:
        return None
    scale = float(np.max(np.abs(a)))
    if scale <= 0:
        return None
    for decimals in range(1, 6):
        step = 10.0 ** (-decimals)
        if step < scale * 1e-4:          # finer than float16 would give anyway -> not a concern
            break
        if np.allclose(a / step, np.round(a / step), atol=1e-6, rtol=0):
            return (f"{key}: values look ROUNDED to {decimals} decimal(s) (step {step:g}). "
                    "Cast to float16 to shrink the file, never np.round -- rounding to 3 decimals "
                    "costs ~35% of the MAST Consistency term while leaving R2_psi at 0.99997.")
    return None


def validate(npz_path: Path, config: str, max_shots: int) -> int:
    split, machine = CONFIG_INFO[config]
    H, W = GRID[machine]

    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found", file=sys.stderr)
        return 2

    from datasets import load_dataset

    sub = dict(np.load(npz_path, allow_pickle=False))
    print(f"Validating {npz_path.name} against {config} (expects {machine} grid {(H, W)})")
    print(f"  {len(sub)} arrays in the .npz; streaming reference inputs from the Hub …")

    ds = load_dataset(repo_for(config), config, split=split, streaming=True)
    errors: list[str] = []
    warnings: list[str] = []
    n = 0
    capped = False
    for i, row in enumerate(ds):
        if max_shots and i >= max_shots:
            capped = True
            print(f"  (stopped at --max-shots {max_shots}; full validation needs --max-shots 0)")
            break
        n += 1
        prefix = f"shot_{i:04d}"
        T = len(np.asarray(row["efit_times"]))

        # flux map: (T, H, W)
        k = f"{prefix}_psirz"
        if k not in sub:
            errors.append(f"{k}: MISSING (expected ({T}, {H}, {W}))")
        else:
            arr = sub[k]
            if arr.shape != (T, H, W):
                errors.append(f"{k}: shape {arr.shape}, expected ({T}, {H}, {W})")
            if not np.issubdtype(arr.dtype, np.floating):
                errors.append(f"{k}: dtype {arr.dtype}, expected float")
            if machine == "DIII-D" and not np.isfinite(arr).all():
                errors.append(f"{k}: contains NaN/Inf (DIII-D is fully finite; those pixels score as error)")
            warn = _quantization_warning(arr, k)
            if warn:
                warnings.append(warn)

        # the two submitted scalars: one (T,) array each, named to prevent column mix-ups
        for name in SCALARS:
            k = f"{prefix}_{name}"
            if k not in sub:
                errors.append(f"{k}: MISSING (expected ({T},))")
                continue
            arr = sub[k]
            if arr.shape != (T,):
                errors.append(f"{k}: shape {arr.shape}, expected ({T},)")
            if not np.issubdtype(arr.dtype, np.floating):
                errors.append(f"{k}: dtype {arr.dtype}, expected float")

    if not capped:
        expected_keys = {f"shot_{i:04d}_{s}" for i in range(n) for s in ("psirz", *SCALARS)}
        extra = [k for k in sub if k not in expected_keys]
        if extra:
            errors.append(
                f"unexpected keys: {extra[:5]}{' …' if len(extra) > 5 else ''} — the v2 contract "
                "is psirz + q95 + betaN only (li/R_axis/Z_axis are derived from your flux "
                "map by the scorer, not submitted)")

    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s) — these do NOT fail validation but will cost score:")
        for w in warnings[:10]:
            print(f"    - {w}")
        if len(warnings) > 10:
            print(f"    … and {len(warnings) - 10} more")

    if errors:
        print(f"\n✗ {len(errors)} problem(s):")
        for e in errors[:40]:
            print(f"    - {e}")
        if len(errors) > 40:
            print(f"    … and {len(errors) - 40} more")
        return 1
    print(f"\n✓ OK — {n} shots, all shapes/keys/dtype valid for {config}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a submission .npz's shape (no scoring)")
    ap.add_argument("npz", type=Path, help="submission .npz to validate")
    ap.add_argument("--config", required=True, choices=list(CONFIG_INFO),
                    help="which test config this .npz targets (*_private_test = Final phase)")
    ap.add_argument("--max-shots", type=int, default=0, help="cap shots checked (0 = all)")
    args = ap.parse_args()
    return validate(args.npz, args.config, args.max_shots)


if __name__ == "__main__":
    raise SystemExit(main())
