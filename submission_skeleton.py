#!/usr/bin/env python3
"""
Submission skeleton — a runnable, format-correct example of a challenge submission.

This is the executable version of README → "3. What you predict". Run it and read the
printed shapes to see exactly what a submission looks like — no Git LFS / sample data needed
(it streams the public test inputs from Hugging Face; with --final, the Final phase's
private-test inputs instead).

What you submit (per shot, at each `efit_times` timestamp), grouped per shot in one .npz per config:
    shot_0000_psirz    (T, H, W)   flux map   DIII-D 65x65 / MAST 65x65 (both dense, no NaN region)
    shot_0000_q95      (T,)        edge safety factor
    shot_0000_betaN    (T,)        normalized beta

That is the WHOLE contract (metric v2): the flux map plus the only two scalars a flux map cannot
contain (q95 needs the toroidal field function F(psi), betaN needs the pressure profile p(psi)).
Everything else — the LCFS boundary, magnetic axis (R_axis/Z_axis), elongation, triangularity,
volume and internal inductance (li) — is DERIVED from your submitted
flux map by the scorer, with the same published functionals it applies to the ground-truth flux.
A scalar can only be earned by a psi that implies it: there is no separate scalar head to tune.

The leaderboard score is the composite
    S = 0.55*R2_psi + 0.15*R2_{q95,betaN} + 0.10*(1 - D_LCFS) + 0.20*Consistency
where Consistency is the mean agreement of the seven psi-derived scalars, f(psi_pred) vs
f(psi_gt). A perfect flux map scores D_LCFS = 0 and Consistency = 1 by construction.

To make a real submission, replace `your_model_predict()` with your trained model. The placeholder
here emits zeros of the correct shape (a valid-but-useless submission) so you can confirm the
plumbing before plugging in a model.

This one script does the whole submission: build -> validate -> push to Hugging Face -> write the
zip you upload to Codabench.

    # quick format check, 5 shots, build only
    uv run python submission_skeleton.py --max-shots 5

    # the real thing: every shot, pushed, with submission_pointer.zip ready to upload
    uv run python submission_skeleton.py --max-shots 0 \
        --repo your-username/fusion-eq-predictions --read-token hf_...

    # Final phase: the same, on the private-test inputs -> diii_d_private_test.npz +
    # mast_private_test.npz (streamed from ganatma/fusion-equilibrium-challenge-final)
    uv run python submission_skeleton.py --final --max-shots 0 \
        --repo your-username/fusion-eq-predictions --read-token hf_...

Run it once without --read-token to create the repo; Hugging Face cannot scope a token to a repo
that does not exist yet. See README -> "5. Build and submit".
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from datasets import load_dataset

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
TEST_CONFIGS = [("diii_d_public_test", "public_test"), ("mast_public_test", "public_test")]
# Final phase (--final): the private-test inputs live in their OWN dataset repo (inputs only, same
# columns and same .npz contract as the public test). Written as {config}.npz like the above.
FINAL_REPO_ID = "ganatma/fusion-equilibrium-challenge-final"
FINAL_CONFIGS = [("diii_d_private_test", "private_test"), ("mast_private_test", "private_test")]
# Native flux grid per machine (rows=Z, cols=R). Both machines are a dense, fully finite 65x65
# grid — MAST's upstream EFIT 65x129 grid (65 real R columns interleaved with 64 empty ones) is
# collapsed to a dense 65x65 in the corrected dataset, so there is no central NaN region.
GRID = {"DIII-D": (65, 65), "MAST": (65, 65)}
# The only two submitted scalars (metric v2), each under its own per-shot key (named, not
# positional, to make column mix-ups impossible). Everything else is derived from your flux map.
SCALARS = ["q95", "betaN"]


def your_model_predict(row: dict, source: str) -> dict:
    """REPLACE ME. Return predictions for this shot, aligned to row['efit_times'], as a dict:
        {"psirz":  (T, H, W) flux map,
         "q95":    (T,),
         "betaN":  (T,)}

    Inputs in `row` (NO magnetic-diagnostic array — that's the point of the challenge):
      - `magnetics_*` COIL CURRENTS = commanded actuators (F-coils, ECOILA/bcoil, MAST P-coils,
        Solenoid, TF, …), not measurements of the plasma's field.
      - `thomson_*` = kinetic profiles (electron temperature & density).
      - `efit_times` (+ MAST: `efit_grid_R/Z`).
    The targets are withheld in test configs. `magnetics_dsep` is EFIT-DERIVED (computed from
    the target equilibrium) and is neither an input nor a scored target. Also available as
    inputs: `coil_*` (PF-coil positions/turns, joined to the current columns by
    `coil_input_column`) and `thomson_chord_R/Z` (chord positions). Focus on making psi(R,Z)
    right; the geometry terms (boundary, axis, shape, li) all follow from it."""
    T = len(np.asarray(row["efit_times"]))
    H, W = GRID[source]
    out = {"psirz": np.zeros((T, H, W), dtype=np.float32)}      # placeholder baseline
    for name in SCALARS:
        out[name] = np.zeros(T, dtype=np.float32)               # placeholder scalars
    return out


def build_submission(config: str, split: str, out_dir: Path, max_shots: int) -> Path:
    repo = FINAL_REPO_ID if config in dict(FINAL_CONFIGS) else REPO_ID
    ds = load_dataset(repo, config, split=split, streaming=True)
    preds: dict[str, np.ndarray] = {}
    n = 0
    for i, row in enumerate(ds):
        if max_shots and i >= max_shots:
            break
        source = row.get("source", "DIII-D")
        T = len(np.asarray(row["efit_times"]))
        H, W = GRID[source]
        out = your_model_predict(row, source)

        assert out["psirz"].shape == (T, H, W), f"{config} shot {i}: psirz {out['psirz'].shape} != {(T, H, W)}"
        # float16 keeps relative precision everywhere and costs ~0.1% of score; the scorer
        # upcasts on read. Do NOT instead round to a fixed number of decimals -- np.round(psi, 3)
        # leaves R2_psi at 0.99997 while destroying ~35% of the MAST Consistency term.
        preds[f"shot_{i:04d}_psirz"] = out["psirz"].astype(np.float16)
        for name in SCALARS:
            arr = np.asarray(out[name])
            assert arr.shape == (T,), f"{config} shot {i}: {name} {arr.shape} != ({T},)"
            preds[f"shot_{i:04d}_{name}"] = arr.astype(np.float32)

        n = i + 1
        if n % 25 == 0:
            print(f"  {config}: {n} shots")

    out_path = out_dir / f"{config}.npz"
    np.savez_compressed(out_path, **preds)
    psh = preds.get("shot_0000_psirz", np.empty(0)).shape
    print(f"  {config}: {n} shots -> {out_path.name}  (e.g. shot_0000_psirz {psh}, "
          f"shot_0000_q95 {preds.get('shot_0000_q95', np.empty(0)).shape})")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a submission, validate it, push it to Hugging Face, and write the "
                    "pointer zip you upload to Codabench.")
    ap.add_argument("--max-shots", type=int, default=5, help="cap shots per config (0 = all)")
    ap.add_argument("--final", action="store_true",
                    help="Final phase: build diii_d_private_test.npz + mast_private_test.npz from "
                         f"the private-test inputs ({FINAL_REPO_ID}) instead of the public test")
    ap.add_argument("--out", type=Path, default=Path("submission"))
    ap.add_argument("--repo", help="Hugging Face dataset repo to push to, e.g. you/fusion-eq-preds. "
                                   "Given this, the script also pushes and writes the pointer zip.")
    ap.add_argument("--read-token", default=os.environ.get("HF_READ_TOKEN"),
                    help="fine-grained READ token scoped to --repo (or set HF_READ_TOKEN)")
    ap.add_argument("--zip", dest="zip_out", type=Path, default=Path("submission_pointer.zip"),
                    help="pointer zip to upload to Codabench (default: submission_pointer.zip)")
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the structure check (not recommended -- it is what catches a "
                         "malformed .npz before you spend a submission slot)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = FINAL_CONFIGS if args.final else TEST_CONFIGS

    print("Building submission (placeholder zeros — swap in your_model_predict):")
    written = []
    for config, split in configs:
        written.append(build_submission(config, split, args.out, args.max_shots).name)

    # No manifest is written here: the scorer locates predictions by FILENAME, and on the
    # pointer route push_and_write_pointer() writes the real {repo_id, revision, token} one.
    total_mb = sum((args.out / w).stat().st_size for w in written) / 1e6
    print(f"\nWrote {', '.join(written)} to {args.out.resolve()}  ({total_mb:.0f} MB)")

    if not args.skip_validate:
        print("\nValidating structure...")
        from validate_submission import validate
        for config, _ in configs:
            if validate(args.out / f"{config}.npz", config, args.max_shots):
                print(f"\nValidation FAILED for {config}. Fix your_model_predict and rebuild; "
                      "nothing was pushed.", file=sys.stderr)
                return 1

    if args.repo:
        print()
        from push_predictions import push_and_write_pointer
        return push_and_write_pointer(args.repo, args.read_token, args.out, args.zip_out)

    final = " --final" if args.final else ""
    print(f"\nNext: re-run with{final} --repo <you>/fusion-eq-predictions to push and build the zip you\n"
          "      upload to Codabench. See README -> '5. Build and submit'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
