## Branches
- `main` — clean, reproducible showcase of the project (working code, verified scripts, finalized experiments).  
  ⚠️ **Notebooks are not allowed in `main`.**
- `dev` — integration branch for ongoing work; all features and experiments are merged here after quick review.
- `feat/<short-topic>` — feature/refactor branch (e.g., `feat/new-augment`).
- `exp/<initials>/<hypothesis>` — individual experiment (e.g., `exp/ap/cb-focal-loss`).

---

## Commit Rules
- Small, atomic commits.
- Format: `type(scope): short message`  
  `type ∈ {feat, fix, exp, docs, refactor, chore}`  

Examples:
- `exp(cb): tune depth=8`  
- `feat(eval): add det curve`  
- `fix(data): drop leaks`

---

## Pull Requests
- To `dev`: quick peer review required (1 approve).  
- To `main`: only from `dev`, after checkpoint (see below).  
- Merge strategy:
  - To `dev`: **squash** for features, **merge** (no squash) for `exp/*` branches (to keep experiment history).  
  - To `main`: **squash** with a clear summary.

---

## Experiment Logs
- Root folder `experiments/`.
- For each `exp/*` branch → create `experiments/<yyyy-mm-dd>__<author>__<short>/` containing:
  - `params.yaml` — hyperparameters.
  - `metrics.json` — key metrics (EER, AUC, F1, etc.).
  - `notes.md` — short notes/observations.
  - (optional) artifacts: configs, selected plots.
- Root-level `EXPERIMENTS.md` — table with: date, branch, description, best metric, link to folder/PR.

> If MLflow/DVC is used: keep `mlruns/` out of Git, commit only exported params/metrics and references.

---

## Checkpoints (Tags)
- Tag best results on `main`:  
  `ckpt-<yyyy.mm.dd>-<short>` (e.g., `ckpt-2025.09.08-cb-focal-0.945auc`).
- Tag description should include PR link and model artifact hash.

---

## Repository Structure
data/ # only manifests/symlinks (no raw data in Git)
data_analysis/ # notebooks with outputs allowed (EDA, reports, exploration)
experiments/ # experiment logs and configs
models/ # manifest files (hashes/paths), not raw weights
notebooks/ # working notebooks (must be output-stripped); not allowed in main
src/ # modular code
scripts/ # reproducible CLI training/eval scripts
EXPERIMENTS.md
README.md
requirements.txt | pyproject.toml

---

## Data & Artifacts
- **Git LFS** only for small, necessary binaries (≤ 50 MB).
- Datasets and model weights → **external storage** (S3, GCS, MinIO).  
  In `models/` and `data/` → manifest files (`.txt`/`.json`) with paths and hashes.
- Training scripts must support:
  - `--seed`  
  - `--data-manifest`  
  - `--params params.yaml`  
  - `--out experiments/...`

---

## Notebooks
- In `dev`: working notebooks allowed (clear outputs preferred).  
- In `main`: **strictly forbidden**.  
- In `data_analysis/`: notebooks with outputs are allowed (EDA, visualizations, reports).  
- Each notebook should have a small `README` explaining how to run it and what it shows.

---

## Code Style & Checks
- `pre-commit`: `black`, `ruff`, `nbstripout`.
- Basic `pytest` tests for critical functions (metrics, preprocessing).
- Dependencies pinned (`requirements.txt` or `poetry.lock`).

---

## Experiment Lifecycle
1. `git checkout -b exp/ap/cb-focal-loss`
2. Modify code/configs, run `scripts/train.py ...`
3. Save results in `experiments/.../params.yaml`, `metrics.json`, plots.
4. Open PR → `dev` with `[EXP]` prefix and short summary (metrics vs baseline).
5. After review → merge (no squash), update `EXPERIMENTS.md`.
6. If result is a new best → merge `dev` → `main` and tag with `ckpt-*`.

---

## Roles & Rhythm
- One maintainer responsible for `main` cleanliness.
- Weekly 15-minute “results review” and `EXPERIMENTS.md` update.

---

## PR Checklist (description template)
- [ ] Reproducible run command included
- [ ] `params.yaml` and `metrics.json` updated
- [ ] Baseline comparison included (table/plot)
- [ ] No raw data or model weights in Git
- [ ] Pre-commit checks passed