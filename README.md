# ⚛️ The Fusion Equilibrium Challenge: A Hacker's Guide

Matthew Waller, Craig Michoski, Tapan Ganatma Nakkina, Brian Sammuli, William Boyes, Mitchell Clark, Sterling Smith, Raffi Nazikian

1. Sophelio  
2. General Atomics
3. UT Austin (IFS)

## What you are building

Reconstruct a tokamak's **2-D poloidal magnetic flux map** ψ(R,Z) from coil currents and Thomson
scattering alone — **no magnetic sensors** — on DIII-D, and zero-shot on MAST.

You submit ψ plus the only two scalars a flux map cannot contain, `q95` and `betaN`. The plasma
boundary (LCFS), magnetic axis, shape scalars and `li` are all **derived from your ψ by the
scorer**, using the same functionals it applies to the true ψ. Getting ψ right is what earns them —
there is no separate scalar head to tune.

One submission enters both challenges:

- **Challenge 1 — intra-machine.** Highest composite score on DIII-D.
- **Challenge 2 — cross-machine.** Highest `G_ratio = S_MAST / S_DIII-D`, among entries with a
  DIII-D composite `S ≥ 0.85`. Needs predictions for **both** machines; MAST has **no training
  data at all**, by design.

The six steps below are the whole workflow, in the order you will do them. Everything after them is
reference material — physics background, the complete signal dictionary, machine differences — to
consult when a column name confuses you.

---

## 1. Set up

```bash
git clone https://github.com/Sophelio/fusion-equilibrium-challenge-starter
cd fusion-equilibrium-challenge-starter
```

Pick **uv** (recommended), conda, or pip:

```bash
uv sync                            # core deps → .venv/
uv sync --group pytorch            # add PyTorch for the neural-net baselines
```

<details>
<summary>conda / pip alternatives</summary>

```bash
# conda / mamba  (core + PyTorch)
mamba env create -f environment.yml
mamba activate fusion-equilibrium-starter

# plain pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-pytorch.txt      # optional
```
</details>

**Log in to Hugging Face** with a **write** token from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). You need this in step 5
to push your predictions; it stays on your machine and is never submitted.

```bash
uv run huggingface-cli login
```

**Get the demo shots.** The six files in `parquet_data/` are tracked with **Git LFS**. Without it
you get ~130-byte pointer files instead of data:

```bash
git lfs install && git lfs pull
```

The training and test data on Hugging Face does **not** need LFS — only these local demo shots do.

---

## 2. Get the data

Everything lives in one Hugging Face dataset:
[`Sophelio/fusion-equilibrium-challenge`](https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge).

| Config | Shots | What it is |
|---|---|---|
| `diii_d_train` | 7,041 | DIII-D, inputs **and** targets |
| `diii_d_public_test` | 874 | DIII-D, **inputs only** — targets withheld |
| `mast_public_test` | 1,206 | MAST, **inputs only** — targets withheld |

```python
from datasets import load_dataset
train = load_dataset("Sophelio/fusion-equilibrium-challenge", "diii_d_train",
                     split="train", streaming=True)
row = next(iter(train))
```

There is deliberately **no `mast_train`** — Challenge 2 is zero-shot transfer. One row is one shot;
every time series and 2-D map is a nested array inside that row.

```bash
uv run python example_usage.py          # inspect the dataset
```

**Inputs** are coil currents (`magnetics_*` — commanded actuators, not field measurements), Thomson
scattering profiles (`thomson_*`), machine geometry (`coil_*`, `thomson_chord_*`), and `efit_times`.
**No magnetic-diagnostic array** — that is the point of the challenge. See
*Complete Signal Dictionary* in the reference below.

### Data errata (v1.1.0)

Two participant-reported issues in the released data, both with drop-in fixes in
[`data_fixes.py`](data_fixes.py). They affect **input columns only** — the targets, `efit_times`,
and all scoring are unaffected, and everyone has the same data. There will be no v1.1.1 release:
v1.1.0 plus these fixes is the final data for the competition, including the Final-phase test
inputs, so nothing you build on them is throwaway.

**1 · DIII-D `magnetics_plasma_current_times` is wrong on ~69% of shots**
([#6](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/6)). Every DIII-D
file shipped with the *same* Ip time axis. It is genuinely correct for ~31% of shots, but on the
rest (4,930 of 7,041 `diii_d_train` shots; 586 of 874 `diii_d_public_test`) the Ip trace actually
starts at `magnetics_time[0]` — about 3 s earlier — so interpolating Ip onto `efit_times` with
the shipped axis reads the pre-plasma noise floor (~2 kA) instead of the real current (~1 MA).
The Ip **values** are correct on every shot; only the timestamps need fixing. MAST is unaffected.

```python
from data_fixes import fix_d3d_ip_times

ip_times = fix_d3d_ip_times(row)   # per-shot corrected axis; works on train and test alike
ip_at_efit = np.interp(row["efit_times"], ip_times, row["magnetics_plasma_current"])
```

The helper decides per shot (breakdown must land near t = 0, the DIII-D time convention) and has
been validated to make the correct, unambiguous call on all 9,113 DIII-D source shots. Do **not**
blanket-shift every shot — ~31% ship with the correct axis, and shifting those would corrupt them.
The baseline in `experiments.py` already applies the fix.

**2 · MAST `thomson_core_R` has 130 entries but `thomson_core_Te`/`ne` have 131 channels**
([#5](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/5)). Channel 0 has
no radius calibration and is NaN in every shot; alignment is `R[j] ↔ Te[t][j+1]`. Drop data
channel 0:

```python
from data_fixes import align_mast_thomson_core

R, Te, ne = align_mast_thomson_core(row)   # (130,), (T, 130), (T, 130) — aligned
```

---

## 3. What you predict

Per shot, one prediction per `efit_times` timestamp:

| Key | Shape | What |
|---|---|---|
| `shot_XXXX_psirz` | `(T, 65, 65)` | flux map ψ(R,Z) — both machines, dense, no NaN region |
| `shot_XXXX_q95` | `(T,)` | edge safety factor |
| `shot_XXXX_betaN` | `(T,)` | normalized beta |

`T` = number of `efit_times` for that shot. Shots are numbered in **test-stream order**, 0-indexed.
That is the entire contract — nothing else is submitted.

- **Align to `efit_times`.** Resample your *inputs* onto those timestamps; never interpolate the
  target grid itself.
- **Preserve shot order.** Emit predictions in the order the test split streams rows.
- **Do not round, truncate, decimate, or drop frames.** `np.round(psi, 3)` looks harmless — it
  leaves R²ψ at 0.99997 — while destroying ~35% of the MAST Consistency term. The skeleton writes
  the right dtype for you; just don't post-process ψ.

**How you are scored:**

```
S = 0.55·R²ψ  +  0.15·R²{q95,βN}  +  0.10·(1 − D_LCFS)  +  0.20·Consistency
```

| Term | What it measures |
|---|---|
| `R²ψ` | Global R² of the flux map over all points × timesteps × shots |
| `R²{q95,βN}` | Pooled R² of your two submitted scalars |
| `D_LCFS` | Hausdorff distance between the LCFS contours extracted from your ψ and the true ψ |
| `Consistency` | Mean agreement of seven ψ-derived scalars: `R_axis, Z_axis, κ, δ_top, δ_bot, V, li` |

A perfect flux map scores `D_LCFS = 0` and `Consistency = 1` by construction — the same code runs
on both sides. **The flux sign is normalized for you**, so you do not need to guess DIII-D's or
MAST's convention. Full detail in *How scoring works* below and in `MODELING_GUIDE.md`.

---

## 4. Train a baseline

```bash
uv run python experiments.py --quick                    # fast sanity run
uv run python experiments.py --n-shots 50 --epochs 50   # something real
```

PyTorch baselines auto-detect CUDA → MPS → CPU; override with `--device`. To train from a
downloaded copy instead of streaming, use `--source local --local-data-dir /path/to/hf_dataset`.

`MODELING_GUIDE.md` is the ML walkthrough — start there for feature engineering, the PCA-on-ψ
baseline, and normalization advice (input scales span ~10⁴ A coils to ~10⁶ A plasma current).

### Score yourself locally

You can compute `R²ψ` yourself easily enough. `D_LCFS` and `Consistency` — **30% of the composite**
— need contour extraction and the seven ψ-derived functionals, and they are exactly where a
plausible-looking flux map and a good one come apart. `local_score.py` gives you the whole metric
on held-out training shots, so you are not spending submission slots to find out:

```bash
uv run python local_score.py --n-shots 20 --skip 7000     # shots your model did not train on
```

```
==============================================================
  COMPOSITE S = 0.8143      (20 held-out DIII-D shots)
==============================================================
            R2_psi    0.9312   x 0.55  =  0.5122
    R2_{q95,betaN}    0.7458   x 0.15  =  0.1119
        1 - D_LCFS    0.8821   x 0.10  =  0.0882
       Consistency    0.5103   x 0.20  =  0.1021
```

plus a per-scalar `R²` breakdown for the Consistency term, so you can see *which* derived quantity
your ψ is getting wrong.

`fusion_scoring/` holds the competition scorer's own modules, copied unmodified — the functionals
are the same code Codabench runs, not a reimplementation. What differs is the data: this scores
training shots you hold out, the leaderboard scores withheld test shots. **Treat it as a faithful
proxy for model selection, not a leaderboard prediction.**

Verify the harness itself any time:

```bash
uv run python local_score.py --mode perfect --n-shots 2    # S = 1.0 exactly
uv run python local_score.py --mode zeros   --n-shots 2    # S = 0.0
```

By default it calls `your_model_predict` from `submission_skeleton.py` — the same function that
builds your real submission. ⚠️ Training rows **do** contain the `efit_*` ground truth: if your
model reads those, the score is meaningless. Use only the inputs listed in that function's
docstring. Use `--pred my_preds.npz` to score a file you built another way.

Keep your own work in **`my_experiments/`** — it is gitignored, so it survives `git pull` and never
collides with the starter kit. Starter modules import from there:

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments import load_shot_from_hf_row, interpolate_magnetics_to_efit
```

---

## 5. Build and submit

Open `submission_skeleton.py` and replace `your_model_predict()` with your model. Then three steps
— the first two happen **once**, the third every time you submit.

Your predictions go to a **private** Hugging Face repo, and what you upload to Codabench is a small
file pointing at it. Nobody else can see your repo, and scoring starts in seconds instead of after
an hour of transfer.

### Step 1 · Create your predictions repo

```bash
uv run python submission_skeleton.py --max-shots 5 --repo your-username/fusion-eq-predictions
```

Use **your own username** as the namespace unless you have write access to an organization. This
builds 5 shots as a quick format check and creates the (private) repo.

✅ It ends by printing the repo URL and the exact recipe for step 2.

### Step 2 · Create a read token

At [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token** →
**Fine-grained** → under *Repository permissions* pick the repo you just made → tick **only**
"Read access to contents of selected repos".

Steps 1 and 2 are separate because Hugging Face cannot scope a token to a repo that does not exist
yet.

✅ You have a second token starting `hf_`. Keep it — you will paste it into every submission.

### Step 3 · Build the real submission

```bash
uv run python submission_skeleton.py --max-shots 0 \
    --repo your-username/fusion-eq-predictions \
    --read-token hf_...
```

`--max-shots 0` means **every shot**. The default is 5 — a submission built without this flag scores
almost everything as missing.

This streams the public test fold once to build, again to validate, and then uploads, so it takes a
while and is not stuck.

✅ It ends with `==> Upload this to Codabench: .../submission_pointer.zip`. **That file is your
entire submission** — a few hundred bytes. On to §6.

---

**The two tokens** trip people up, so to be explicit:

| | what it does | where it goes |
|---|---|---|
| **write** token | uploads your `.npz` to Hugging Face | `huggingface-cli login` back in §1; never leaves your machine |
| **fine-grained read** token | lets the scorer read your private repo | inside the zip; submitted |

The script **refuses** a write token or a classic read token in the `--read-token` slot — that is
not a bug. Both grant far more than the scorer needs, and that token travels with your submission.
Revoke it when the competition ends.

**Submitting predictions you did not produce is plagiarism and disqualifies the whole team.**
Organizers re-score leading entries from source before prizes, and your submission pins a commit
SHA so that what was scored cannot be changed afterwards.

<details>
<summary>Re-pushing without rebuilding, or skipping Hugging Face entirely</summary>

A full build streams the whole test fold, so you will not want to redo it just to re-push:

```bash
uv run python submission_skeleton.py --max-shots 0            # build only
uv run python validate_submission.py submission/diii_d_public_test.npz --config diii_d_public_test
uv run python push_predictions.py --repo you/fusion-eq-predictions --read-token hf_...
```

`--skip-validate` drops the validation pass, but that pass is what catches a malformed `.npz`
before it costs you a submission slot.

**Direct upload**, if you would rather not use Hugging Face at all: put the two `.npz` at the
**root** of a zip and upload that instead. It scores identically, but expect about an hour of
transfer per submission and it counts against your 15 GB Codabench quota.

```bash
uv run python -c "import zipfile,pathlib; \
z=zipfile.ZipFile('submission.zip','w'); \
[z.write(p, p.name) for p in pathlib.Path('submission').glob('*.npz')]; z.close()"
```
</details>

---

## 6. Upload to Codabench

1. Register at **[codabench.org/competitions/17739](https://www.codabench.org/competitions/17739/)**.
2. **Submit** tab → upload `submission_pointer.zip`.
3. Your score appears on the leaderboard in a few minutes. Open **Detailed Results** for the
   per-term breakdown, the per-scalar Consistency `R²`, and your derivation-failure rate.

The leaderboard has one column per challenge — sort by **Ch1: DIII-D S** or **Ch2: G_ratio**. One
submission enters both: Challenge 1 needs only the DIII-D file, Challenge 2 needs both machines, so
a DIII-D-only entry shows `G_ratio = 0`. The Challenge-2 eligibility gate is enforced in that
column too: an entry with DIII-D `S < 0.85` also shows `G_ratio = 0`.

| | when | how many |
|---|---|---|
| **Development** | now → **October 18, 2026** | 5 submissions/day, 100 total |
| **Final** | **October 19–26, 2026** | 3 total — blind, leaderboard hidden until close |

By submitting you agree to the competition rules and the dataset terms. Starter-kit code is
MIT-licensed (`LICENSE`).

---
---

# Reference

Background, the full signal dictionary, and machine-specific detail. You do not need to read this
top to bottom — come here when a column name or a convention is unclear.

## Welcome to the Machine

You are about to work with data from two large nuclear fusion research devices:

- **DIII-D** - General Atomics tokamak (San Diego, USA)
- **MAST** - Mega Ampere Spherical Tokamak (Culham, UK)

Your goal is to solve a control theory problem that is critical for the future of clean energy: **Predicting the shape of the plasma.**

### The "Jelly Donut" Analogy (Fusion 101)

Imagine you have a donut made of super-hot, invisible jelly (the plasma). This jelly is 100 million degrees, so you can't touch it. Instead, you hold it in place using powerful, invisible magnetic fields (the "magnetic bottle").

Usually, we use magnetic sensors to "feel" where the jelly is. **But in this challenge, you are blind.** The magnetic sensors are broken or unavailable.

**Your Mission:** You must infer the exact shape of the jelly in the donut using only:

1. **The Knobs:** How much current you are sending to the electromagnets.
2. **The Thermometer:** Lasers that measure how hot and dense the jelly is at specific points.



### What "blind" really means — and why it matters

Conventional EFIT reconstructs the 2D equilibrium from a dedicated suite of **magnetic
diagnostics**: an array of magnetic field probes and flux loops mounted around the vessel
that "feel" the plasma's own field. **This challenge withholds that diagnostic array.** Your
inputs are only:

- **Actuators** — the *commanded* coil currents (`magnetics_F`*, `ECOILA`/`bcoil`, MAST P-coils,
`Solenoid`, `TF`, …). These are knobs you *drive*, not measurements of the plasma's field.
- **Kinetic profiles** — Thomson scattering electron temperature & density (`thomson_`*).

The motivation is concrete and physical: if a model can reconstruct the equilibrium **without a
magnetic-diagnostic suite**, a tokamak could be built and operated more cheaply without that
instrumentation — or keep reconstructing when those sensors degrade or fail. This is the *proper
zero-shot* goal: equilibrium reconstruction from actuators + kinetic profiles alone. The
**MAST leg — Challenge 2** pushes it one step further: can the learned physics reconstruct a
machine the model has **never seen** (different size, shape, and coil set)?

> ⚠️ **`magnetics_dsep` is not an input.** It is **EFIT-derived** — computed *from the target
> equilibrium itself* — so it encodes x-point/divertor geometry straight from the label, and is
> **withheld on the test splits** alongside `efit_psirz` and the scalar labels. It is not a scored
> target either (it was dropped from the metric: DIII-D's `dsep` is a separatrix↔limiter clearance
> while MAST's is divertor balance — different physical quantities that one functional cannot score
> coherently). Never read `dsep` as a model input. (`magnetics_plasma_current` (Ip) *is* an allowed
> input: a single legitimate global magnetic scalar, not label-derived.)

The data isn't *purely* zero-shot — Ip is a real global measurement you may use — but that's fine:
the setup is a strong starting point for the two things that matter most here: **cross-machine
robustness** (models that learn physics, not one machine's wiring) and **synthetic diagnostics**
(deriving machine-agnostic, physics-meaningful inputs from actuators + kinetic profiles). Treat
those as the real targets.

---


## How scoring works (detail)

**No scalar is masked by boundary type.** Earlier versions restricted the shape scalars to diverted
frames; the dataset is now diverted-only, so that mask is gone. Six of the seven ψ-derived scalars
are well-posed on **100%** of frames and `li` on 99.99% (DIII-D) / 99.5% (MAST) — a frame leaves a
scalar's average only where the *ground-truth* derivation itself fails. If your ψ fails to yield a
scalar the truth does have, the frame is not skipped: it is mean-substituted and earns ~0.

**The flux sign is normalized for you.** DIII-D and MAST store ψ with opposite sign conventions
(see *Cross-machine convention notes*), so a model that transfers correctly still lands
sign-inverted. The scorer determines the global sign of your submitted flux map per machine, scores
you under it, and reports which it used (`psi_sign`: `+1` as submitted, `−1` normalized). It is one
bit per machine over the whole fold, and only the sign — not the amplitude — is normalized.

**Cross-machine (Challenge 2):** `G_ratio = S_MAST / S_DIII-D`, among entries with a composite
`S ≥ 0.85` on DIII-D (raised 2026-08-18 from `R²ψ > 0.6` — see
[issue #7](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/7); the scorer
enforces the gate, reporting `G_ratio = 0` below it). The two machines are scored separately. The scorer runs on Codabench against held-out
ground truth; it is not part of this starter kit.

**`validate_submission.py`** checks structure only — no ground truth, no score. It confirms the
per-shot keys, shot order and count, per-shot `T`, native grid, and dtypes against the streamed
public-test inputs. `submission_skeleton.py` runs it for you; call it directly with `--max-shots 0`
for a full check of an existing file.

## The target: `efit/` (the ground truth)

From **EFIT** (equilibrium fitting), the reconstruction code whose output you are learning to
reproduce without magnetics.

| Key | Shape | Description |
|-----|-------|-------------|
| `efit_psirz` | (T, 65, 65) both machines | Poloidal flux map — a 2-D image at each timestep, like a topographic map whose contours show the magnetic cage. *Withheld on test.* |
| `efit_times` | (T,) | Timestamps (ms) for the target images. Align all inputs to these. |
| `efit_grid_R` / `efit_grid_Z` | (65,) | Physical R/Z (m) labelling the flux-map columns/rows. Kept on every split. |

**EFIT scalar labels** (one value per `efit_times` step; in `train`, withheld on test). You *submit*
only `q95` and `betaN`; the rest are training supervision — at scoring time the axis, shape and `li`
are derived from your submitted flux map:

| Key | Shape | Description |
|-----|-------|-------------|
| `efit_q95` | (T,) | Safety factor at the 95% flux surface — **submitted scalar** |
| `efit_beta_n` | (T,) | Normalised beta β_N — **submitted scalar** |
| `efit_li` | (T,) | Internal inductance ℓi (supervision; derived from your ψ at scoring) |
| `efit_r_axis` / `efit_z_axis` | (T,) | Magnetic-axis R/Z (m) (supervision; derived from your ψ at scoring) |
| `magnetics_dsep` | (T,) | **Context only — not scored.** EFIT-derived despite the `magnetics_` prefix. On DIII-D it is the a-file `DSEP` separatrix↔limiter clearance, `>0` on every shipped frame and free of NaN. On MAST it is a *different* quantity — divertor balance δR_sep, which straddles zero on ordinary diverted plasmas, so **its sign is not a limited/diverted flag**. |
| *(bonus, train only)* `efit_lcfs_n`, `efit_lcfs_r`, `efit_lcfs_z` | (T,) / (T, N) | Last-closed-flux-surface boundary contour + valid-point count. Provided as context. |

> **Where you will actually see these.** Every column above is EFIT-derived, so all are withheld on
> `diii_d_public_test` and `mast_public_test`. Since there is no `mast_train` either, **MAST's
> labels appear in no released config at all** — the only MAST shots carrying `efit_psirz`, the
> scalars or `magnetics_dsep` are the three demo files in `parquet_data/`. Build your pipeline so it
> does not expect them on MAST.

**The flux map is dense on both machines.** The corrected MAST `_psirz` is a dense 65×65 grid — the
upstream EFIT stored it on a doubled 65×129 R grid (65 real columns interleaved with 64 empty ones),
which the dataset collapses to the 65 real columns. Like DIII-D, MAST has **no central-column NaN
region** to skip. R²ψ is computed only over finite ground-truth pixels, so any occasional
non-finite frame is handled for you.

Reconstructions often include electrical currents in major conductors such as the vacuum vessel,
which for simplicity are omitted here.

## Data organization

Record IDs indicate the source: `DIII-D_182494`, `MAST_25607`. Signal names are prefixed the same
way: `DIII-D: F1A`, `MAST: P2L`.

Training and evaluation data live on Hugging Face. The `parquet_data/` folder here holds six **demo
shots only** (3 DIII-D + 3 MAST) for local inspection and the dFL visualizer.

## 🔌 DIII-D: The Actuators (Magnets)

DIII-D uses a set of shaping coils (F-coils) and main field coils to control the plasma.

### Shaping Coils (F-coils)

These 18 coils act like invisible hands that mold the plasma:


| Signal                              | Description         | Range     |
| ----------------------------------- | ------------------- | --------- |
| `DIII-D: F1A` through `DIII-D: F9A` | Upper shaping coils | ~±600 kA·turn in-window |
| `DIII-D: F1B` through `DIII-D: F9B` | Lower shaping coils | ~±600 kA·turn in-window |




### Main Coils


| Signal           | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| `DIII-D: ECOILA` | Ohmic heating coil - central solenoid that drives plasma current    |
| `DIII-D: bcoil`  | Toroidal field coil - main stability field going "around the track" |


---



### Additional Quantities

| Signal | Description |
|--------|-------------|
| `DIII-D: ip` | Integrated electrical current carried in the plasma bulk (an **input**) |
| `DIII-D: dsep` | EFIT a-file `DSEP`: minimum separatrix↔limiter clearance (m); `>0` diverted, `<0` limited. **EFIT-derived**, so present in `train` and withheld on test. Context only — not scored. |

---



## 🔌 MAST: The Actuators (Magnets)

MAST is a spherical tokamak with a different coil configuration than conventional tokamaks.

### Poloidal Field Coils (P-coils)

MAST has 10 poloidal field coils (P2-P6, Lower and Upper):


| Signal                          | Description                |
| ------------------------------- | -------------------------- |
| `MAST: P2L` through `MAST: P6L` | Lower poloidal field coils |
| `MAST: P2U` through `MAST: P6U` | Upper poloidal field coils |


**Note:** MAST has no P1, P7, P8, or P9 coils (different machine geometry).

### Main Coils


| Signal           | Description                             |
| ---------------- | --------------------------------------- |
| `MAST: Solenoid` | Central solenoid (equivalent to ECOILA) |
| `MAST: TF`       | Toroidal field coil                     |
| `MAST: Ip`       | Plasma current measurement              |
| `MAST: EFPS`     | Error field protection system coil      |


---



## 🌡️ Thomson Scattering (Both Machines)

Both DIII-D and MAST use **Thomson Scattering** - lasers that bounce off electrons to measure:

1. **Temperature ($T_e$):** How hot are the electrons? (eV)
2. **Density ($n_e$):** How crowded are the electrons? (m⁻³)

Each system ships **one** spatial coordinate array (not an R/Z pair); which axis it
represents differs by machine, as noted below.

### Vertical "Core" System (Looking Down) — `thomson_core_*`

Vertical view, named core for purely historical reasons.


| Parquet column       | Description                                                                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `thomson_core_Te`    | Electron temperature (eV), one profile per timestep                                                                                                              |
| `thomson_core_ne`    | Electron density (m⁻³)                                                                                                                                           |
| `thomson_core_R`     | Channel radial position(s) (m). **DIII-D:** constant ≈ 1.94 (vertical chord — channels vary in Z, which is not provided). **MAST:** per-channel R (≈ 0.25–1.5 m). ⚠️ On MAST this has 130 entries vs 131 Te/ne channels — see [Data errata](#data-errata-v110) |
| `thomson_core_times` | Timestamps (ms)                                                                                                                                                  |




### Horizontal "Tangential" / Edge System (Looking Sideways) — `thomson_edge_*`

Horizontal view, named tangential for purely historical reasons.


| Parquet column         | Description                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `thomson_edge_Te`      | Electron temperature (eV)                                                                 |
| `thomson_edge_ne`      | Electron density (m⁻³)                                                                    |
| `thomson_edge_spatial` | Channel positions (m). **DIII-D:** Z (≈ −0.05 m near midplane). **MAST:** R (≈ 1.3–1.5 m) |
| `thomson_edge_times`   | Timestamps (ms)                                                                           |


There is no `tan_z`/`core_z` column: each machine provides a single spatial axis per
system (DIII-D edge = Z, MAST edge = R; core radius in `thomson_core_R`).

---

## 🆕 Machine geometry (both machines, every split)

A coil current means little until you know where the coil is, and a Thomson profile localises
nothing until you know where the chord is. Both now ship on every row — they are inputs, so
nothing is withheld:

| Column | Shape | Description |
|--------|-------|-------------|
| `coil_name` | (C,) str | Coil (or conductor element) identifier |
| `coil_input_column` | (C,) str | **Join key** — the current column this geometry belongs to, e.g. `magnetics_F1A` |
| `coil_R`, `coil_Z` | (C,) | Conductor-rectangle centre (m) |
| `coil_width`, `coil_height` | (C,) | Radial / vertical extent (m) |
| `coil_angle1`, `coil_angle2` | (C,) | Parallelogram skew angles (degrees) |
| `thomson_chord_name` | (N,) str | `TS_core_*`, `TS_tangential_*`, `TS_divertor_*` |
| `thomson_chord_R`, `thomson_chord_Z` | (N,) | Chord position (m) |

**The two machines describe their coils at different granularity — that is not a bug.** DIII-D
ships **C = 19 lumped rectangles** (18 F-coils + `ECOILA`), the representation EFIT's own
`mhdin.dat` uses. MAST ships **C = 812 individual conductor elements** from the FAIR level-2
`pf_active` IDS — one row per conductor turn, the solenoid alone being 656 of them, so a MAST
coil's turn count is how many rows share its `coil_input_column` (P2 = 20, P3 = 8, P4 = 23,
P5 = 23, P6 = 4). **Neither machine ships a `coil_turns` column**: the turn counts are folded
into the current values (see the units note).
Join either back to the currents with `coil_input_column`; several MAST elements share one column
(P2 inner/outer are parallel-fed; all 656 solenoid elements share `magnetics_sol_current`).

**Six DIII-D F-coils are parallelograms.** `coil_angle1` / `coil_angle2` are EFIT's `AF` / `AF2`
shear angles: `F5A`/`F5B` ±45°, `F6A`/`F6B` ±92.4°, `F7A`/`F7B` ±108.06°; everything else 0.0.
They only matter if you integrate over the conductor cross-section rather than treating each coil
as a filament. MAST's are structurally 0.0 — IMAS rectangles carry no skew.

> `0.0` means **no skew (plain axis-aligned rectangle)**, not "sides lie flat". EFIT branches on
> `angle1 == 0 and angle2 == 0` to emit a rectangle, and normalises `90 → 0` on write-out, so 90°
> and 0° denote the same unskewed coil — we ship the canonical `0`. MAST's zeros mean exactly what
> the 13 DIII-D zeros mean, so one code path covers both machines.

Every DIII-D F-coil rectangle (R, Z, width, height) matches EFIT's own `mhdin.dat` machine file
exactly — and that file's turn counts (58 or 55) are already folded into `magnetics_F*`, which
therefore carries **total ampere-turns per rectangle**, the quantity a Green's-function
calculation wants. No turn count ships as a column, and none is needed.
`ECOILA` is the exception: EFIT models that group as 48 single-turn elements over the same
envelope and `ECOILB` is a second co-located group not shipped here, so its turn convention is
ambiguous. `magnetics_ECOILA` is in **kA**, not kA·turn — don't apply an ampere-turn multiplier.

*Not covered:* `magnetics_bcoil` (DIII-D TF) and MAST's `magnetics_tf_current` /
`magnetics_efps_current` have no poloidal-plane rectangle — 19 of 21 DIII-D current columns and
11 of 14 MAST ones are covered.

**Chord positions close a real gap.** DIII-D's core Thomson is a *vertical* laser, so
`thomson_core_R` is a constant ≈ 1.94 m and the informative coordinate — Z — was never shipped;
the tangential system is the mirror image; and the **divertor** subsystem had no shipped
counterpart at all. `thomson_chord_R`/`_Z` carry all three subsystems with both coordinates.

> ⚠️ **DIII-D chord positions are per shot and they genuinely vary** — 22 distinct channel-name
> layouts in train (19 in public test) over 6 distinct subsystem
> layouts, channel counts from 59 to 138, because the divertor system was reconfigured between
> campaigns. Do not cache one shot's chord array and reuse it.

MAST's Thomson is a *horizontal midplane* laser: per-channel R (same values as
`thomson_core_R` / `thomson_edge_spatial`) with `thomson_chord_Z = 0`. That zero is not a
placeholder — FAIR level-2 exposes only `thomson_scattering.channel[:].position.r`.

A worked use: build each coil's vacuum Green's function from the shipped rectangle
(`coil_R`, `coil_Z`, `coil_width`, `coil_height`) and you get the coil-driven part of ψ
analytically rather than learning it. Fitting ψ outside the plasma envelope to those Green's
functions, driven by the shipped ampere-turn currents, reaches R² ≈ 0.94.

---

## ⚡ Complete Signal Dictionary

Each Parquet file holds **one shot per row**. All time-series, profiles, and
flux maps are stored as nested arrays within that row. Every time array is
in **milliseconds** (both machines).

Diagnostic groups:

- `efit/*` — magnetic equilibrium reconstruction (the *target*).
- `magnetics/*` — coil currents (the *actuators*) and one EFIT-derived
scalar (`dsep`, the x-point gap; see "Equilibrium-Derived Quantities").
- `thomson/*` — electron temperature & density profiles (the *sensors*).



### DIII-D columns

| Display name | Parquet column | Shape | Notes |
|---|---|---|---|
| — | `source` | scalar string | `"DIII-D"` |
| **EFIT (targets)** | | | |
| EFIT times | `efit_times` | `(T,)` float64 | ms; T ranges **1–445** across the dataset (median 241); 42 shots have T < 10 and 129 have T < 50 — do not assume a long record |
| EFIT psirz | `efit_psirz` | `(T,)` of `(65, 65)` | Poloidal flux maps (V·s/rad). *Primary target; withheld on test.* |
| EFIT grid R/Z | `efit_grid_R`, `efit_grid_Z` | `(65,)` float64 | Physical R/Z (m) of the flux grid. Kept on every split. |
| EFIT scalars | `efit_beta_n`, `efit_li`, `efit_q95`, `efit_r_axis`, `efit_z_axis` | `(T,)` float64 each | Scored scalar targets (β_N, ℓi, q95, axis R/Z). *Withheld on test.* |
| EFIT boundary | `efit_lcfs_n`, `efit_lcfs_r`, `efit_lcfs_z` | `(T,)` / `(T, N)` | LCFS contour + valid-point count. Bonus context in `train`. *Withheld on test.* |
| **Magnetics time bases** | | | |
| — | `magnetics_time` | `(M,)` float32 | ms; shared by every DIII-D magnetics signal. **M varies by shot**: 70.0% of train shots (67.0% of public test) are `(480256,)` at ~20 kHz, the rest mostly `(49152,)` or `(50176,)` at 2 kHz. Six distinct lengths occur in all. Both rates span the full ~24 s record — do not hard-code the length. |
| — | `magnetics_plasma_current_times` | `(30719,)` float32 | ms; Ip is on its own ADC at a different sample rate. **⚠️ Wrong on ~69% of shots as shipped — use `data_fixes.fix_d3d_ip_times`; see [Data errata](#data-errata-v110)** |
| — | `magnetics_dsep_times` | `(T,)` float32 | ms; identical to `efit_times` since dsep is EFIT-derived |
| **Main coils** | | | |
| `DIII-D: ECOILA` | `magnetics_ECOILA` | `(M,)` float64 | Ohmic / central solenoid — **kA** (not kA·turn; turn convention unresolved). Uses `magnetics_time`. |
| `DIII-D: bcoil` | `magnetics_bcoil` | `(M,)` float64 | Toroidal field — **kA** (toroidal coil: no PF turn count). Uses `magnetics_time`. |
| `DIII-D: Ip` | `magnetics_plasma_current` | `(30719,)` float32 | Plasma current — **kA** (matches MAST). Uses `magnetics_plasma_current_times` (**⚠️ see [Data errata](#data-errata-v110)**). |
| **Shaping coils (18)** | | | |
| `DIII-D: F1A`–`F9B` | `magnetics_F{1-9}{A,B}` | `(M,)` float64 each | Upper (A) / lower (B) shaping coils — **kA·turn** (58 or 55 turns folded in). All use `magnetics_time`. |
| **EFIT-derived (target)** | | | |
| `DIII-D: dsep` | `magnetics_dsep` | `(T,)` float32 | EFIT a-file `DSEP`: separatrix↔limiter clearance (m); `>0` diverted, `<0` limited. EFIT-derived, withheld on test, not scored. Uses `magnetics_dsep_times`. |
| **Thomson core** (vertical chord, ~R = 1.94 m, looks down) | | | |
| — | `thomson_core_times` | `(~1300–1900,)` float64 | ms |
| — | `thomson_core_Te` | `(~T_c,)` of `(C_c,)` | Electron temperature (eV) per profile; `C_c` varies by shot |
| — | `thomson_core_ne` | `(~T_c,)` of `(C_c,)` | Electron density (m⁻³) per profile; `C_c` varies by shot |
| — | `thomson_core_R` | `(C_c,)` float64 | **varies by shot: 40, 42, 43, 44 or 54 channels** (44 on 6,327 of 7,041 train shots). Radial positions (m) — constant ≈ 1.94 since this is a vertical chord |
| **Thomson edge** (horizontal tangential view, ~Z ≈ −0.05 m) | | | |
| — | `thomson_edge_times` | `(~200–500,)` float64 | ms |
| — | `thomson_edge_Te` | `(~T_e,)` of `(C_e,)` | Electron temperature (eV); `C_e` varies by shot |
| — | `thomson_edge_ne` | `(~T_e,)` of `(C_e,)` | Electron density (m⁻³); `C_e` varies by shot |
| — | `thomson_edge_spatial` | `(C_e,)` float64 | **varies by shot: 10 channels (6,385 shots) or 14 (656)**. Z positions (m) of the tangential channels |

### MAST columns

MAST is **zero-shot: a test split only**, so its EFIT targets (`efit_psirz` and the
scalars) are not distributed. The three MAST demo shots in `parquet_data/` do include a
`efit_psirz` (clean 65×65) purely for dFL visualization.

| Display name | Parquet column | Shape | Notes |
|---|---|---|---|
| — | `source` | scalar string | `"MAST"` |
| **EFIT (targets — withheld on the MAST test split)** | | | |
| EFIT times | `efit_times` | `(T,)` float64 | ms; T ranges **5–98** across the dataset (median 58); 397 of 1,206 shots have T < 50 |
| EFIT psirz | `efit_psirz` | `(T,)` of `(65, 65)` | Clean 65×65 flux map (no NaNs). Upstream MAST stores psirz on a doubled 129-column R grid — 65 real columns interleaved with 64 empty ones — which we drop to recover the dense grid. |
| EFIT grid R | `efit_grid_R` | `(65,)` float64 | Physical R (m) for the flux grid (≈ 0.06–2.0 m) |
| EFIT grid Z | `efit_grid_Z` | `(65,)` float64 | Physical Z (m) for the flux grid (≈ −2.0–2.0 m) |
| **Magnetics (shared time base)** | | | |
| — | `magnetics_time` | `(30000,)` or `(15482,)` float64 | ms; shared by every MAST magnetics signal. **Two populations** — see note below |
| `MAST: Ip` | `magnetics_plasma_current` | same as `magnetics_time` | Plasma current — **kA** |
| `MAST: TF` | `magnetics_tf_current` | same as `magnetics_time` | Toroidal field coil feed — **kA** |
| `MAST: Solenoid` | `magnetics_sol_current` | same as `magnetics_time` | Central solenoid feed — **kA** |
| `MAST: EFPS` | `magnetics_efps_current` | same as `magnetics_time` | Error field protection system — **kA** |
| **Poloidal field coils (10)** | | | |
| `MAST: P{2-6}{L,U}` | `magnetics_p{2-6}{l,u}_current` | same as `magnetics_time` | P2–P6 lower/upper (no P1/P7/P8/P9) — **kA·turn**, not kA. See the units note |

> **⚠️ MAST magnetics come in two time-base populations, and one of them has gaps.**
>
> - **1,092 of 1,206 test shots** (the later campaigns): `magnetics_time` is `(30000,)`,
>   a uniform 0.2 ms / 5 kHz grid spanning −2,000 → +3,999.8 ms, every column finite.
> - **114 shots** (the early campaign): `magnetics_time` is `(15482,)` spanning −2,500 → +5,499
>   ms and is the **union of two acquisition grids** — the poloidal set (P-coils, Ip, solenoid,
>   EFPS) at 0.2 ms / 5 kHz over −150 → +1,349.8 ms (7,500 samples), and the toroidal field coil
>   at 1.0 ms / 1 kHz over the full record (8,000 samples). Each column is therefore **null on
>   the samples belonging to the other grid**: ~52% of rows for the poloidal set, ~48% for TF.
>
> **No data is missing** — the nulls are holes in a union axis, not dropouts, and both native
> grids fully cover the plasma window on every one of the 114 shots. This is the layout
> FAIR-MAST's own `amc` group ships for those shots.
>
> The gaps are plain `NaN` (no parquet nulls anywhere in this release), so mask per column
> instead of assuming a dense array:
>
> ```python
> t   = np.asarray(row["magnetics_time"], dtype=float)
> tf  = np.asarray(row["magnetics_tf_current"], dtype=float)
> ok  = np.isfinite(tf)
> tf_t, tf_v = t[ok], tf[ok]          # native 1 kHz TF trace, gap-free
> ```
| **EFIT-derived** | | | |
| `MAST: dsep` | `magnetics_dsep` (+ `_times`) | `(T,)` float32 | **δR_sep — divertor *balance*, NOT the same quantity as DIII-D's `dsep`.** From `esm/dr_sep_out`: the radial gap between the upper and lower separatrices at the outboard midplane, so it straddles zero on ordinary diverted plasmas and **its sign is not a limited/diverted flag**. Not scored, and — like every EFIT-derived MAST column — present only in the `parquet_data/` demo shots, never in a released config. |
| **Thomson core** | | | |
| — | `thomson_core_times` | `(~50–112,)` float64 | ms |
| — | `thomson_core_Te` | `(T_c,)` of `(131,)` | Electron temperature (eV) |
| — | `thomson_core_ne` | `(T_c,)` of `(131,)` | Electron density (m⁻³) |
| — | `thomson_core_R` | `(130,)` float64 | Radial positions (m) of each core channel. **⚠️ 130 coordinates vs 131 Te/ne channels — data channel 0 has no coordinate and is NaN everywhere; use `data_fixes.align_mast_thomson_core`; see [Data errata](#data-errata-v110)** |
| **Thomson edge** | | | |
| — | `thomson_edge_times` | `(~50–112,)` float64 | ms |
| — | `thomson_edge_Te` | `(~T_e,)` of `(~16,)` | Electron temperature (eV) |
| — | `thomson_edge_ne` | `(~T_e,)` of `(~16,)` | Electron density (m⁻³) |
| — | `thomson_edge_spatial` | `(~16,)` float64 | Spatial positions (m) of edge channels |

### Cross-machine convention notes

- **⚠️ The two machines store `efit_psirz` with OPPOSITE SIGN CONVENTIONS.** On **DIII-D** the
  magnetic axis is the **minimum** of ψ; on **MAST** it is the **maximum**. Measured on the
  shipped corpus: DIII-D 99.98% of 1,559,340 frames with zero counter-examples, MAST 100%. This
  is a provenance difference between two EFIT implementations (DIII-D's EFIT vs
  EFIT++/FAIR-MAST), **not physics** — both machines run positive plasma current here.
  - A DIII-D-trained model applied zero-shot to MAST predicts a correct equilibrium with the
    wrong overall sign. A naive R² then reports a large *negative* number and looks like total
    failure. **Check `R²(−ψ_pred)` before concluding your transfer failed.**
  - Anything assuming "axis = maximum" (O-point search, contouring, ψ_N normalization) needs the
    per-machine sign or must detect it.
  - **The official scorer is sign-invariant**: it determines the global sign of your submitted
    flux map per machine, scores you under it, and reports which sign it used. You are not being
    tested on guessing a storage convention. Amplitude is *not* normalized.
- **⚠️ Current units, harmonized across both machines in v1.1.0.** Shaping-coil currents are
  already in **ampere-turns**, so the two machines are directly comparable and **there is no
  `coil_turns` column — you do not need one, and you must not apply a turn multiplier**:

  | Columns | Units as shipped |
  | :--- | :--- |
  | DIII-D `magnetics_F1A`–`F9B` | **kA·turn** (each coil's 58 or 55 turns already folded in) |
  | MAST `magnetics_p{2-6}{l,u}_current` | **kA·turn** (natively) |
  | `magnetics_plasma_current` (both machines) | **kA** |
  | DIII-D `magnetics_ECOILA`, `magnetics_bcoil` | **kA** |
  | MAST `magnetics_sol_current`, `_tf_current`, `_efps_current` | **kA** |

  So `F4A` and MAST `p4u` are the same kind of quantity — that is what a Green's-function
  calculation wants. The four kA channels stay in plain kA because no trustworthy turn count
  exists for them: `ECOILA`'s turn convention is genuinely ambiguous (EFIT models the group as 48
  single-turn elements, and `ECOILB` is a co-located group not shipped here), and `bcoil`/`tf` are
  toroidal field coils. Normalizing per machine (recommended) is still the simplest way to handle
  the remaining scale spread.
- **Time units are ms everywhere**, including MAST (`magnetics_time`, `efit_times`, `magnetics_dsep_times`). MAST upstream stores some signals in seconds; the conversion is applied at parquet build time so participants don't have to think about it.
- **Magnetics time base is shared per machine**: both DIII-D and MAST expose one `magnetics_time` array used by every coil signal at the primary sampling rate. On DIII-D, `magnetics_plasma_current` (Ip) sits on its own ADC at a different rate and therefore has its own `magnetics_plasma_current_times` companion — **⚠️ which shipped wrong on ~69% of shots; use `data_fixes.fix_d3d_ip_times` (see [Data errata](#data-errata-v110))**. On MAST, 114 early-campaign shots use a two-grid union base with per-column nulls — see the MAST magnetics note above.
- `dsep` **is on the EFIT time base**: `magnetics_dsep_times` is identical to `efit_times` on every shot for both machines. It's grouped under `magnetics_`* only for column-naming consistency; physically it's an EFIT-derived geometric quantity, not a magnetic measurement.
- `magnetics_time` **spans cover the full DAQ window** (pre-shot baseline through post-shot ringdown), so they extend well beyond the plasma's actual lifetime. The plasma window is bounded by `efit_times`.

---



## 🗂️ Repository Layout

```
fusion-equilibrium-challenge-starter/
├── parquet_data/                  # 6 demo shots (3 DIII-D + 3 MAST) for dFL / offline peek
│   ├── d3d_shot_203702.parquet
│   ├── d3d_shot_203703.parquet
│   ├── d3d_shot_203704.parquet
│   ├── mast_shot_28348.parquet
│   ├── mast_shot_28350.parquet
│   └── mast_shot_28351.parquet
├── fusion_data_provider.py        # dFL data provider (reads parquet_data/)
├── data_fixes.py                  # Drop-in fixes for the v1.1.0 data errata (Ip time axis, MAST Thomson alignment)
├── MODELING_GUIDE.md              # ML walkthrough
├── example_usage.py               # Load the Hugging Face dataset
├── experiments.py
├── experiments_torch.py           # Baseline models (train from Hugging Face)
├── submission_skeleton.py         # Produce a format-correct submission .npz
├── validate_submission.py         # Shape-check a submission before uploading (no scoring)
├── pyproject.toml                 # uv / pip dependency source of truth
├── environment.yml                # conda / mamba alternative
├── requirements.txt               # core deps (plain pip)
├── requirements-pytorch.txt       # optional PyTorch baselines
├── uv.lock                        # pinned deps (uv)
├── my_experiments/                # YOUR custom work (gitignored — create as needed)
├── LICENSE                        # MIT (starter-kit code; dataset has its own terms)
└── README.md                      # This file
```

Each demo parquet file is **one row per shot** with nested array columns (`efit_psirz`, coil currents, Thomson profiles, etc.). The full challenge dataset uses the same schema on [Hugging Face](https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge).

---



## 🔬 Key Differences Between Machines

| Feature | DIII-D | MAST |
|---------|--------|------|
| **Location** | San Diego, USA | Culham, UK |
| **Type** | Conventional tokamak | Spherical tokamak |
| **Flux grid shape** | 65×65 | 65×65 (dense; upstream 65×129 empty columns dropped) |
| **R coordinates** | Normalized 0-1 | Physical: 0.12 - 2.0 m |
| **Z coordinates** | Normalized 0-1 | Physical: -2.0 to 2.0 m |
| **Shaping coils** | 18 (F1A-F9B) | 10 (P2L-P6U) |
| **Thomson data orientation** | (spatial, time) | (time, spatial) |
| **Tangential axis** | Z (vertical) | R (radial) |

---



## 🔍 Understanding the Flux Data



### Geometry Differences

**DIII-D (Conventional Tokamak):**

- Large central solenoid with substantial magnetic core
- Plasma forms a "D" shape around the center
- Flux data covers the full computational domain
- Contours form a pattern concentric to the plasma axis that will appear more "shaped" at the boundary, becoming elliptic then circular close to the axis.

**MAST (Spherical Tokamak):**

- Very narrow central column (the "cored apple" design)
- Plasma wraps tightly around a thin central post
- More compact geometry but with unique measurement challenges
- Contours wrap around the hollow center, forming an asymmetric kidney-bean shape

### MAST's 65×65 grid (and why the raw grid was 65×129)

The corrected dataset ships MAST `efit_psirz` as a clean **65×65** grid with **no NaNs**,
matching DIII-D's dimensions. This is not a physical hole — it's a grid artifact:
MAST's upstream EFIT stores `psirz` on a doubled 129-column R grid, where **65 real R
columns are interleaved with 64 empty ones**. We drop the empty columns to recover the
dense grid the data is actually defined on (MAST R ∈ [0.06, 2.0] m, Z ∈ [−2.0, 2.0] m).

The dFL flux grapher in `fusion_data_provider.py` additionally filters any remaining
all-NaN columns defensively, so it renders both the corrected (65×65) and any legacy
(65×129) shots correctly.

### Flux Pattern Interpretation

The `psirz` flux map is like a topographical map:

- **Contour lines** = magnetic flux surfaces where plasma particles travel
- **Innermost contours** = plasma core (hottest, densest region)
- **Outermost contours** = plasma edge (separatrix boundary)
- **Color gradient** = flux magnitude (V·s/rad)



### Checking out the data in the dFL (Data Fusion Labeler)

The dFL can help you visualize any data (fusion or any other kind of dataset), and label the data for downstream ML/AI tasks.
You can download the dFL here:

Mac (Apple Silicon): [https://github.com/Sophelio/dFL/releases/latest/download/Labeler-mac-arm64.dmg](https://github.com/Sophelio/dFL/releases/latest/download/Labeler-mac-arm64.dmg)  
Windows: [https://github.com/Sophelio/dFL/releases/latest/download/Labeler-windows.exe](https://github.com/Sophelio/dFL/releases/latest/download/Labeler-windows.exe)  
Linux: [https://github.com/Sophelio/dFL/releases/latest/download/Labeler-linux.AppImage](https://github.com/Sophelio/dFL/releases/latest/download/Labeler-linux.AppImage)

Once you open the dFL, select a "custom script" and point it at `fusion_data_provider.py`. It will load the demo shots from `parquet_data/` in this repository.

# Disclaimer

Work supported by the U.S. Department of Energy, Office of Science, Office of Fusion Energy Sciences, using the DIII-D National Fusion Facility, a DOE Office of Science user facility, under Award No. DE-FC02-04ER54698, along with Office of Fusion Energy Sciences Awards No. DE-SC0024426, DE-SC0024499, DE-SC0024409, and DE-SC0024571.

Disclaimer: This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.