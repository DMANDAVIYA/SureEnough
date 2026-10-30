# SureEnough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build SureEnough, a calibrated "stop when sure" EEG-to-image decoder on THINGS-EEG2, plus a hosted demo, a ≤2:30 video and a submission package, by Nov 4, 2026, 5:00 PM ET.

**Architecture:**
- A Python package (`sureenough/`) turns the raw downloads into a few things:
  - float32 arrays and Perception Encoder targets;
  - per-subject EEG encoders;
  - cached replay streams of probabilities;
  - per-image, per-λ loss tables.
- Every statistic (LTT calibration, baselines, hypotheses) runs on those caches.
- Meta API scripts make the SAM 3.1 cutouts, Muse Image impressions and Muse Spark narration from CC0 images and our own text only.
- An export step writes a small JSON/WebP bundle. A static Vite + React + TypeScript app reads it, re-runs LTT in the browser, and deploys to GitHub Pages.

**Tech Stack:**
- Python 3.14 in `.venv`: torch 2.15 CPU, open_clip 3.3, timm 1.0.30, numpy 2.5, scipy, scikit-learn, pandas, pillow, requests, python-dotenv, pytest. matplotlib is still to be added.
- Node 24: Vite, React 18, TypeScript, vitest, @playwright/test.
- ffmpeg 8.1.1.

**Spec:** `docs/superpowers/specs/2026-10-27-sureenough-design.md` (v2.2, approved Oct 30). Section numbers below (§) refer to it.

## Global Constraints

- **Installs** go only into `.venv`, `web/node_modules`, or Playwright browsers under the project (`PLAYWRIGHT_BROWSERS_PATH=0`). Ask Dhruvil before anything system-wide.
- **Caches:** set `HF_HOME=D:/A5/metapr-data/hf-cache` before any weight download.
- **Data root:** `D:/A5/metapr-data`, overridable with the `SUREENOUGH_DATA` env var.
  - Raw downloads (untrusted) are in `raw/`. Converted data goes in `derived/`.
  - Repo caches go in `outputs/` (gitignored).
- **Untrusted files** (`raw/**`) are loaded only by the safe loader, in a `python -I` process. Never `allow_pickle=True`. ATM `.pt` files use `torch.load(..., weights_only=True)`.
- **THINGS stimuli** (`osf-y63gw/*_images.zip`) are never sent to any API, committed, exported to `web/`, or shown in the video. They are used locally only for PE targets and DINOv2 scoring.
- **Meta API:**
  - Base `https://api.meta.ai/v1`. Key from `.env` `META_API_KEY`.
  - Model IDs: `muse-spark-1.3`, `sam-3.1`, `muse-image-1.0` only, never `-contributor`. Send `store: false` on Responses calls.
  - The tracked-spend hard cap is **$60**.
  - Prices: Spark $1.25/M in, $4.25/M out; SAM $0.0025/image; Muse Image $0.01/image.
- **Pre-registered primary config:** K = 20, α = 0.2, δ = 0.1, stop size m = 3, R = 10 streams per image.
  - λ grid: `lam[j] = 1 - j/500` for j = 0..499 (λ = 1.0 down to 0.002), tested in that order.
  - Split: 100 calibration / 100 evaluation, 1,000 random splits.
- **Seeds (fixed before any analysis):**
  - stream seed = `np.random.default_rng([20261030, subject, image, r])`;
  - split seed s = `np.random.default_rng([7, subject, s])`;
  - validation concepts = `np.random.default_rng(0).choice(1654, 200, replace=False)`;
  - pinned demo split = split s = 0.
- **Badge copy (verbatim):** "Promise: for this person, the true object is outside SureEnough's final answer at most 2 in 10 times (checked with 90% confidence)."
- **Header label (verbatim):** "Replayed public lab EEG (THINGS-EEG2). Not live. Each repeat is a 100 ms flash in a rapid sequence."
- **Footer (verbatim):** "EEG: THINGS-EEG2 (Gifford et al. 2022), CC-BY 4.0 · Object photos: THINGSplus, CC0."
- **Copy rules:**
  - Say "repeats", never seconds or "a session". Say "lab EEG (17 channels)".
  - Never write "probably", or "90% sure this answer is right".
  - Write "efficient" only if H1 holds.
- **Web export:** at most 150 MB in `web/public/data/`. No PE features, no stimuli.
- **Cuts already applied (spec §11):**
  - stretch baselines (COINS / time-uniform);
  - H3/H4;
  - "You pick the picture" mode;
  - WSR p-values. They are exploratory per §6.6 and dropped for schedule. HB is primary.
  - Exploratory m ∈ {1, 5} and the selective point/abstain policy.
- **Deferred unless Nov 2 runs ahead:** the H5a text level (Spark blind captions; last on the spec's cut order).
- **Git (Dhruvil's rule):**
  - Every change goes issue → its own branch → PR with `Closes #N` → merge. Nothing is committed straight to `main`.
  - Commit and PR titles start with `feat:`, `fix:`, `refactor:`, `ui:` or `chore:`.
  - No AI attribution anywhere: no co-author trailers, no tool signatures. Dhruvil is the sole author.
  - Scan the staged diff for secrets before every push.
- **Code style (Dhruvil's rule):** lean, well-structured small modules. **No comments or docstrings** in code or tests.
- **Repo:** `https://github.com/DMANDAVIYA/SureEnough` (public). Pages base path is `/SureEnough/`.
- **Tests:** `.venv/Scripts/python.exe -m pytest` from the repo root. Web tests: `npm --prefix web test` and `npm --prefix web run e2e`.

## Review Focus

1. **λ = 1 and float sums.** Cumulative softmax probabilities in fp16/fp32 may never reach exactly 1.0. λ = 1 must mean exactly "the full 200 gallery, never stop early, loss 0", and must not stop early. (Test in Task 4.)
2. **Ties after quantisation.** Probabilities stored as fp16 (Python) or uint16 (web) can tie. Ranking must break ties by lower gallery index in both Python and TypeScript, so live recalibration matches Python bit for bit on the exported fixture. (Tests in Tasks 4 and 18.)
3. **API failures mid-batch.** A 429/5xx or a crash halfway through 200 SAM calls must not double-spend or lose work. Scripts resume by skipping finished outputs. The ledger is written *before* each call, and a call whose projected spend exceeds $60 raises before any network I/O. (Test in Task 11.)
4. **Display-rule edge cases.**
   - A set containing an uncategorised concept, or one whose only shared category is "food" (26.5% > 25%), falls back to "It's one of these N".
   - Category ties resolve to the smaller category, then alphabetically.
   - (Test in Task 14.)
5. **Phone width and the Pages sub-path.** At 390 px wide the panels stack, with no horizontal scroll. Data loads from `/SureEnough/` with HashRouter deep links like `#/results`. (Test in Task 20.)

---

## Part A: Data and features (critical path, Oct 30 evening; PE extraction runs overnight)

### Task 1: Package skeleton, safe loader, EEG conversion

**Files:**
- Create: `pyproject.toml`, `sureenough/__init__.py`, `sureenough/paths.py`, `sureenough/data/__init__.py`, `sureenough/data/safe_npy.py`, `sureenough/data/convert.py`, `scripts/01_convert.py`
- Test: `tests/test_safe_npy.py`, `tests/test_convert.py`

**Interfaces:**
- Produces:
  - `paths.DATA_ROOT, RAW, DERIVED, OUTPUTS: pathlib.Path`; `paths.subject_ids() -> list[int]` returning 1..10.
  - `safe_npy.load_pickled_npy(fileobj) -> dict`. Restricted unpickle, allow-list per spec §6.1, accepting `numpy.core.*` and `numpy._core.*` spellings.
  - `convert.convert_subject(zip_path: Path, out_dir: Path) -> dict` writes:
    - `out_dir/train.npy` float32 (16540, 4, 17, 100);
    - `out_dir/test.npy` float32 (200, 80, 17, 100);
    - `out_dir/meta.json` `{ch_names, times}`;
    - and returns the shapes.
  - `convert.convert_image_metadata(npy_path: Path, out_json: Path)` writes the six metadata keys as lists.
- Output layout: `DERIVED/eeg/sub-XX/{train,test}.npy`, `DERIVED/image_meta.json`.

- [ ] **Step 1: Write `pyproject.toml`** (setuptools, package `sureenough`, `requires-python = ">=3.14"`, pytest `testpaths = ["tests"]`). Run `.venv/Scripts/python.exe -m pip install -e . matplotlib`. Expected: installs only into `.venv`.
- [ ] **Step 2: Write the failing tests.**
  - `test_roundtrip_pickled_dict`: build `{"preprocessed_eeg_data": np.zeros((2,3),np.float64), "ch_names": ["O1"], "times": np.arange(3.)}`, `np.save(buf, d, allow_pickle=True)` *in the test only* (we create it), then `load_pickled_npy(buf)` equals it.
  - `test_rejects_malicious_pickle`: an `.npy` whose pickle payload references `os.system` raises `pickle.UnpicklingError` and does not execute.
  - `test_convert_subject_shapes` (skip if `RAW/osf-anp5v/sub-01.zip` is missing): convert into `tmp_path`, then assert the shapes above, dtype float32, 17 channels, `times[0] == -0.2` (±1e-6), 100 samples.
- [ ] **Step 3: Run** `.venv/Scripts/python.exe -m pytest tests/test_safe_npy.py tests/test_convert.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 4: Implement.**
  - `load_pickled_npy`: `np.lib.format.read_magic` and `read_array_header_1_0/2_0`, assert `dtype == object`, then feed the rest of the stream to the allow-list `Unpickler`.
  - `convert_subject`: read each `.npy` straight from the zip via `zipfile.ZipFile.open` (no extraction to disk), cast to float32, `np.save`, and `del` the float64 array before the next file. Peak RAM ≤ about 2.5 GB.
- [ ] **Step 5: Run the tests.** Expected: PASS.
- [ ] **Step 6: Convert all data.** Run `.venv/Scripts/python.exe -I scripts/01_convert.py` (loops over subjects one by one, then converts image metadata). Expected: 10 subject folders, `image_meta.json` with 16540 train and 200 test entries.
- [ ] **Step 7: Commit** `feat: safe loader and EEG conversion`.

### Task 2: Concept table and validation split

**Files:**
- Create: `sureenough/data/concepts.py`
- Test: `tests/test_concepts.py`

**Interfaces:**
- Consumes: `DERIVED/image_meta.json`; the `RAW/osf-jum2f/*.tsv` files; the `images_THINGSplus-CC0.zip` listing.
- Produces: `load_concepts() -> Concepts`, a frozen dataclass with:
  - `test_names: list[str]` (200, gallery order = test image order);
  - `test_things_ids: list[str]`;
  - `train_concept_of_image: np.ndarray[int]` (16540 → 0..1653);
  - `val_concepts: np.ndarray[int]` (200 sorted; the seed is in Global Constraints);
  - `cc0_member: dict[str, str]` (THINGS id → member path inside the CC0 zip);
  - `categories53: dict[str, set[str]]` (category → THINGS ids);
  - `all_names: list[str]` (1854);
  - `all_things_ids: list[str]` (1854).

- [ ] **Step 1: Write the failing tests** (skip if raw data is absent):
  - `len(test_names) == 200`;
  - 1654 distinct training concepts;
  - `len(all_names) == 1854`;
  - every test id has a CC0 member;
  - exactly **194** test concepts have at least one category;
  - category `"food"` covers **53** test concepts;
  - `val_concepts` is disjoint from nothing in test (concept-disjoint by construction), has 200 entries, and is deterministic across two calls.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement `load_concepts()`**, joining on THINGS unique IDs (`*_concepts_THINGS` ↔ `concepts-metadata_things.tsv` ↔ `category53_long-format.tsv` ↔ CC0 file stems).
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: concept table and validation split`.

### Task 3: Perception Encoder features (start the overnight run)

**Files:**
- Create: `sureenough/features/__init__.py`, `sureenough/features/pe.py`, `scripts/02_pe_features.py`
- Test: `tests/test_pe.py`

**Interfaces:**
- Produces:
  - `load_pe() -> (model, preprocess)` via `open_clip.create_model_and_transforms('hf-hub:timm/PE-Core-B-16')`, eval mode, `torch.set_num_threads(10)`.
  - `pe_image_features(model, preprocess, images: list[PIL.Image]) -> (inter: np.float16 (B, 12, 2, 768), final: np.float16 (B, 1024))`. Here `inter[:, l, 0]` is the CLS token and `inter[:, l, 1]` the mean of the patch tokens of block l, from `model.visual.trunk.forward_intermediates`. `final` is `encode_image`, not normalised.
  - `pe_text_features(model, names: list[str]) -> np.float16 (N, 1024)` with the template `"a photo of a {name}"`.
- Files written:
  - `DERIVED/features/pe_{train,test}_inter.npy` as memmaps (16540 or 200, 12, 2, 768);
  - `pe_{train,test}_final.npy`;
  - `pe_text_final.npy` (1854, 1024).
- The script is resumable: it writes `pe_progress.json` every 256 images.

- [ ] **Step 1: Write the failing tests** (they download weights once into `HF_HOME`):
  - The preprocess transform's resize is 224 with normalize mean = std = (0.5, 0.5, 0.5).
  - Two synthetic 300×200 RGB images give shapes (2, 12, 2, 768) and (2, 1024), all finite.
  - The same input twice gives identical output.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement `pe.py` and `scripts/02_pe_features.py`.**
  - The script reads images straight from `RAW/osf-y63gw/{training,test}_images.zip` in `image_meta.json` order, with batch 32.
  - `--benchmark 64` prints images per second and exits.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Benchmark.** Run `.venv/Scripts/python.exe -I scripts/02_pe_features.py --benchmark 64` and record the images per second in the commit message. Then launch the full run **in the background** (≈16.7k images).
- [ ] **Step 6: Commit** `feat: PE feature extraction (N img/s)`.

---

## Part B: Statistics core (test-first, synthetic data; runs while Task 3 extracts)

### Task 4: Set family and stopped-policy evaluator

**Files:**
- Create: `sureenough/calibrate/__init__.py`, `sureenough/calibrate/sets.py`
- Test: `tests/test_sets.py`

**Interfaces:**
- Produces:
  - `LAM_GRID: np.ndarray` (500,) per Global Constraints.
  - `rank_desc(p: np.ndarray (..., G)) -> np.ndarray[int]`: stable descending order, ties broken by lower index.
  - `mass_above(p) -> np.ndarray (..., G)`: for each item, the summed probability of the items ranked strictly above it.
  - `top_m_mass(p, m=3) -> np.ndarray (...)`.
  - `policy_tables(probs: np.ndarray (n_img, R, T, G), K: int, m: int = 3, lam=LAM_GRID) -> dict` of arrays:
    - `loss_count` uint8 (n_img, L): the number of the R streams whose final set misses the true item;
    - `stop_t` float32 (n_img, L): the mean over streams of the stop step (1-based);
    - `set_size` float32 (n_img, L): the mean final set size.
  - The true item of image i is gallery index i.

**Set family (exact):**
- `C_t(λ) = {j : a_j,t < λ}`, where `a` is `mass_above` of the step-t probabilities. This equals "the smallest top set with cumulative probability ≥ λ".
- Special case λ = 1.0 (index 0): `C = all G`, never stop early, loss 0.
- For λ < 1: stop at the first t ≤ K with `top_m_mass(p_t, m) ≥ λ`; otherwise t = K.
- The loss is `a_true,t* ≥ λ`.

- [ ] **Step 1: Write the failing tests.**
  - Hand case G = 4, `p = [0.5, 0.3, 0.15, 0.05]`, true = 0:
    - `C(0.5) = {0}` (`a = [0, .5, .8, .95]`);
    - `C(0.6) = {0, 1}`;
    - `C(0.81) = {0, 1, 2}`.
  - Ties: `p = [0.25]*4` gives `rank_desc == [0, 1, 2, 3]`.
  - **λ = 1 (Review Focus 1):** with probs whose fp16 cumsum is 0.9995, index 0 gives `loss_count == 0`, `stop_t == K` and `set_size == G` for every image.
  - Stopping: a stream whose top-3 mass crosses 0.9 at t = 5 yields `stop_t == 5` for λ = 0.9 and `stop_t == K` for λ = 0.999.
  - Monotonicity: on random Dirichlet streams, mean loss is non-decreasing as λ decreases (along the grid), and mean `stop_t` is non-increasing.
  - Vectorisation check: `policy_tables` equals a naive Python-loop reference on a random (5, 2, 6, 7) array.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement `sets.py`** (vectorised over λ with `searchsorted` on the per-step top-m mass; first-crossing via `argmax` on a boolean (T, L) mask).
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: set family and stopped-policy tables`.

### Task 5: Hoeffding–Bentkus p-values and fixed-sequence LTT

**Files:**
- Create: `sureenough/calibrate/pvalues.py`, `sureenough/calibrate/ltt.py`
- Test: `tests/test_ltt.py`

**Interfaces:**
- Produces:
  - `hb_pvalue(risk_hat: float | np.ndarray, n: int, alpha: float) -> same shape`.
  - `fixed_sequence(pvals: np.ndarray (L,), delta: float) -> int`: the index of the last rejection before the first p > δ, or 0 if index 0 is not rejected (λ = 1 is the fallback).
  - `calibrate(losses: np.ndarray (n, L) in [0, 1], image_ids: np.ndarray (n,), alpha, delta) -> int`. It asserts `len(np.unique(image_ids)) == n` and returns the deployed λ index.

**Formula (exact; from LTT, Angelopoulos et al. 2110.01052):**
```
h1(a, b) = a*log(a/b) + (1-a)*log((1-a)/(1-b))      # with 0*log0 = 0
p_HB     = min( exp(-n * h1(min(r, alpha), alpha)),
                e * BinomCDF(ceil(n*r); n, alpha) )
```
(r = the mean loss over the n images.)

- [ ] **Step 1: Write the failing tests.**
  - `hb_pvalue(0.0, 100, 0.2) ≈ 0.8**100` (rel 1e-9).
  - `hb_pvalue(0.2, 100, 0.2) == 1.0` after the `min` with 1 (clip p to ≤ 1).
  - The p-value is monotone non-decreasing in r.
  - `fixed_sequence([.01, .02, .5, .01], .1) == 1`; `fixed_sequence([.5, ...]) == 0`.
  - `calibrate` raises `AssertionError` when `image_ids` contain duplicates.
  - **Validity (the only asserted statistical property, spec §9.4):**
    - Build a synthetic population of 200 images with losses that increase as λ decreases (risk curve crossing α = 0.2 near the middle of the grid).
    - Draw 1,000 without-replacement 100-image calibration sets, calibrate, and compute the population risk at the deployed λ.
    - The violation rate's Clopper–Pearson 95% upper bound must be ≤ 0.12.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement** with `scipy.stats.binom.cdf`.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: HB p-values and fixed-sequence LTT`.

### Task 6: Baselines (fixed-k, naive repeated look, Bonferroni, uncalibrated)

**Files:**
- Create: `sureenough/calibrate/baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `sets.mass_above`, `sets.top_m_mass`, `ltt.calibrate`.
- Produces:
  - `fixed_k_tables(probs, k: int) -> dict(loss_count, set_size)`: per-λ losses of `C_k(λ)` at the fixed step k (shape (n, L)). Calibrated later by `ltt.calibrate`; repeats used = k.
  - `conformal_scores(probs, K) -> dict(a_true (n, R, K), top3 (n, R, K))`: `a_true` = mass above the true item, `top3` = top-3 mass.
  - `naive_policy(scores, cal_idx, eval_idx, alpha, K, m=3) -> dict(loss (n_eval,), stop_t (n_eval,), set_size (n_eval,), pop_loss (200,))`.
    - Per stream r and step t: `q[t, r]` = the ⌈(n_cal+1)(1−α)⌉-th smallest `a_true[cal, r, t]` (or +∞ if that index exceeds n_cal).
    - The set at (t, r) is `{j : a_j ≤ q}`.
    - Stop at the first t with `top3 > q[t, r]` (equivalently |C| ≤ 3); otherwise use K.
    - Loss = `a_true > q` at the stop.
    - Average over streams. `pop_loss` scores all 200 images with the same q.
  - `bonferroni_policy(...)`: the same with α/K.
  - `uncalibrated_policy(probs, K, thresh=0.8) -> dict(loss, stop_t)`: stop when max p ≥ 0.8, output the top-1.
  - `NaiveScores` arrays for each subject are also exported for the web (Task 16).

- [ ] **Step 1: Write the failing tests.**
  - On synthetic exchangeable data, `naive_policy` with the stop rule disabled (stop forced at fixed t) has mean coverage over 500 random splits ≥ 1 − α − 0.02. This is the fixed-t guarantee.
  - With stopping enabled, record and *print* the miss rate. Don't assert it (spec §9.4).
  - Bonferroni q ≥ naive q elementwise.
  - `fixed_k_tables(probs, k)` at λ index 0 has loss 0.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: baselines`.

### Task 7: Splits runner and hypothesis tests (H1, H2a, H2b, Holm)

**Files:**
- Create: `sureenough/analysis/__init__.py`, `sureenough/analysis/splits.py`, `sureenough/analysis/hypotheses.py`
- Test: `tests/test_hypotheses.py`

**Interfaces:**
- Consumes: Tasks 4–6.
- Produces:
  - `split(subject: int, s: int, n=200) -> (cal_idx, eval_idx)`, using the seed per Global Constraints.
  - `run_splits(tables_K20, fixed_tables: dict[int, dict], scores, subject, n_splits=1000, alpha=.2, delta=.1) -> dict`. Losses passed to `ltt.calibrate` are `loss_count / R`, in [0, 1]. Per split, it records:
    - for SureEnough: the deployed λ index, eval risk, population risk (200), eval mean `stop_t`, eval mean `set_size`;
    - for each fixed k: λ_k, eval risk, eval mean set size;
    - for naive and Bonferroni: eval risk, population risk, mean stop.
  - `h2a_check(losses_200: (200, L), alpha, delta, B=1000, rng) -> dict(violations, cp_upper, mean_risk, conservativeness)`. It draws calibration sets of 100 **with replacement**. Population risk is the exact mean over all 200.
  - `hull_repeats(points: list[(set_size, repeats)], at_size: float) -> (repeats: float, clamped: bool)`. It takes the lower convex hull in the (set_size, repeats) plane and interpolates linearly. Outside the hull's set-size range, it clamps to the nearest endpoint and sets `clamped = True`.
  - `h1_test(ratios: np.ndarray (10,)) -> p`: `scipy.stats.wilcoxon(ratios - 0.7, alternative='less').pvalue`.
  - `h2b_test(naive_pop_risk: (10,), alpha) -> (count, p)`: count = #subjects with mean naive population risk > α; p = `binom.sf(count-1, 10, .5)`.
  - `holm(pvals: dict[str, float], level=.05) -> dict[str, bool]`.

- [ ] **Step 1: Write the failing tests.**
  - `split` gives disjoint 100/100 halves and is deterministic.
  - On a synthetic population whose true risk crosses α, `h2a_check(...)['cp_upper'] ≤ 0.12`.
  - `hull_repeats([(10,1),(5,4),(2,20)], 5) == (4, False)`.
  - `hull_repeats` at size 1 gives (20, True).
  - `holm({'a':.01,'b':.04,'c':.03})` gives a True, b False, c False.
  - `h2b_test(np.full(10,.3), .2) == (10, 0.5**10)`.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: splits runner and hypothesis tests`.

---

## Part C: Decoding (Oct 31 morning; needs Task 3 features)

### Task 8: EEG encoders, layer sweep, Gate 1

**Files:**
- Create: `sureenough/models/__init__.py`, `sureenough/models/encoder.py`, `scripts/03_train.py`
- Test: `tests/test_encoder.py`

**Interfaces:**
- Consumes: `DERIVED/eeg/sub-XX/*.npy`, `DERIVED/features/pe_*`, `Concepts.val_concepts`.
- Produces:
  - `Standardizer.fit(train (N, 4, 17, 100)) / .transform(x (..., 17, 100))`: per-channel mean and std from the training split.
  - `fit_ridge(X (N, 1700), Y (N, D), alpha_grid) -> RidgeModel`, with alpha chosen on validation.
  - `MLPEncoder(in_dim=1700, hidden=2048, out_dim=D, p_drop=0.3)`: Linear, LayerNorm, GELU, Dropout, Linear, then L2-normalise.
  - `train_mlp(X_rep (N, 4, 1700), Y (N, D), epochs=40, batch=512, lr=1e-3, wd=0.01, seed=0) -> MLPEncoder`.
    - Loss: symmetric InfoNCE with a learnable logit scale (init log(1/0.07)).
    - Augmentation: each sample averages a random 1–4 of its 4 repeats.
  - `encode(model, x_avg (M, 17, 100)) -> np.float32 (M, D)`, L2-normalised.
  - `val_top1(model, subject, target) -> float`:
    - 200-way top-1 at k = 4, averaged over the 10 galleries g = 0..9;
    - gallery g = image g of each validation concept;
    - cosine to the targets of those 200 images.
  - Targets: `target_vectors(name) -> (train (16540, D), test (200, D))` for names in `{"final"} ∪ {f"L{l}_{cls|mean}" for l in 6..11}`.
  - Output: `outputs/models/sub-XX.pt` (state dict + standardizer + target name) and `outputs/gate1.json`.

- [ ] **Step 1: Write the failing tests.**
  - On toy data (200 classes, 64-d Gaussian prototypes plus noise, 4 repeats), `train_mlp` for 15 epochs reaches val top-1 > 0.9.
  - `encode` output rows have unit norm.
  - The augmentation averages between 1 and 4 repeats. Check by seeding and inspecting the sampled counts.
  - Training never touches validation concepts: assert in `scripts/03_train.py` that the indices of the training images exclude every image of `val_concepts`.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Run the layer sweep.** `scripts/03_train.py --sweep --subjects 1 2 3` does ridge over all 13 targets, then MLP on the top 2. It writes `outputs/sweep.json`. Pick the target with the best mean validation top-1. Ties go to `final`, because it needs no naming map.
- [ ] **Step 6: Train all 10 subjects** on the chosen target with `scripts/03_train.py --all`. **Gate 1:** `outputs/gate1.json` must show val top-1 ≥ 0.15 for ≥ 8/10 subjects.
  - If it fails, follow the spec §6.3 fallbacks in order: the CLIP-H target (Task 9b), ridge, temperature/layer changes, then the ATM recipe. Record each attempt in `gate1.json`.
- [ ] **Step 7: Commit** `feat: EEG encoders; Gate 1 = X/10 subjects >= 15%`.

### Task 9: β fit, config freeze, replay streams

**Files:**
- Create: `sureenough/streams/__init__.py`, `sureenough/streams/replay.py`, `sureenough/config.py`, `configs/primary.json`, `configs/primary.sha256`, `scripts/04_streams.py`
- Test: `tests/test_streams.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: Task 8 models.
- Produces:
  - `zsoftmax(sims (..., G), beta) -> probs`: z-score over the last axis, then softmax(β·z).
  - `fit_beta(val_sims_by_k: list[(sims (M, 200), true (M,))]) -> float`, using the validation split, k ∈ {1, 2, 3, 4} and the 10 galleries, maximising mean log p_true by 1-D bounded search over [0.1, 50].
  - `stream_orders(subject, image, R=10, n_rep=80) -> np.ndarray (R, 80)`, from seeds per Global Constraints.
  - `make_streams(subject) -> probs float16 (200, 10, 80, 200)`, saved to `outputs/streams/sub-XX.npy`. At step t it averages the first t repeats of the order, encodes, takes cosine to the 200 test targets, then applies `zsoftmax`.
  - `config.load_frozen() -> dict`. It raises if `sha256(configs/primary.json) != configs/primary.sha256`.
  - `configs/primary.json` holds: K = 20, alpha = 0.2, delta = 0.1, m = 3, R = 10, the λ grid spec, β per subject, the target name, seeds and the selection rule.
- Order matters:
  1. Commit `primary.json` + `.sha256` **before** running Task 10 on any test stream.
  2. `scripts/04_streams.py` refuses to run unless `load_frozen()` succeeds.

- [ ] **Step 1: Write the failing tests.**
  - Each row of `stream_orders` is a permutation of 0..79 and identical across calls.
  - `zsoftmax` rows sum to 1 (±1e-5), and are invariant to adding a constant to the sims.
  - `fit_beta` on synthetic logits generated with β = 5 recovers 5 ± 0.5.
  - `load_frozen` raises after a one-byte edit to a temp copy.
  - `test_ordering_never_sees_calibration`: `LAM_GRID` and the split seeds are module constants that take no data arguments. Assert via `inspect.signature` that `split()` has no array parameters.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement**, fit β per subject, write and **commit** `configs/primary.json` and `configs/primary.sha256` with message `chore: freeze primary config before any test calibration`.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Generate streams** for all 10 subjects (≈64 MB each), then `policy_tables` (Task 4) for K = 20 and K = 80 into `outputs/tables/sub-XX_K{20,80}.npz`, plus `conformal_scores` and fixed-k tables.
- [ ] **Step 6: Commit** `feat: replay streams and policy tables`.

### Task 9b: CLIP-H comparison targets (never cut, one line in results)

**Files:**
- Create: `sureenough/features/clip_h.py`
- Test: `tests/test_clip_h.py`

**Interfaces:**
- Produces: `load_clip_h() -> (train (16540, 1024), test (200, 1024))`.
  - It downloads `datasets/LidongYang/EEG_Image_decode` `ViT-H-14_features_{train,test}.pt` into `RAW/hf-atm/` with `huggingface_hub.hf_hub_download`.
  - It loads with `torch.load(weights_only=True)` in a `python -I` subprocess.
  - It verifies the row order by correlating the CLIP-H test rows against PE `final` test rows: the rank-1 match along the diagonal must be ≥ 0.9.

- [ ] **Step 1: Write the failing test** (shapes, and a diagonal-match fraction ≥ 0.9 if the data is present).
- [ ] **Step 2: Run it** and verify it fails.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it** and verify it passes.
- [ ] **Step 5: Run** `scripts/03_train.py --all --target clip_h`, compute "repeats needed for 50% top-1: PE vs CLIP-H" per subject into `outputs/pe_vs_cliph.json`.
- [ ] **Step 6: Commit.**

### Task 10: Calibrate, test hypotheses, figures

**Files:**
- Create: `scripts/05_calibrate.py`, `sureenough/analysis/figures.py`
- Test: `tests/test_calibrate_script.py`

**Interfaces:**
- Consumes: Tasks 4–7 and 9 outputs.
- Produces `outputs/results.json` with:
  - per-subject arrays: eval risk, population risk, conservativeness, mean stop, mean set size;
  - outcome mix (share single / category / list / full set at the pinned split);
  - fixed-k points, the H1 ratio and its clamp flag;
  - naive and Bonferroni risks;
  - H2a dicts;
  - p-values for H1, H2b and H5a (H5a filled by Task 15) and the Holm decisions;
  - `deployed_lam_idx` for the pinned split s = 0.
- Figures in `outputs/figures/*.png`:
  - promised vs delivered (naive vs SureEnough, per subject);
  - the fixed-k hull with the SureEnough point;
  - the stop-time histogram.

- [ ] **Step 1: Write the failing test.** Running the script on a 2-subject synthetic `outputs/` fixture (generated in the test via Tasks 4–6) writes `results.json` containing the keys above, and `len(per_subject) == 2`.
- [ ] **Step 2: Run it** and verify it fails.
- [ ] **Step 3: Implement** with matplotlib.
- [ ] **Step 4: Run it** and verify it passes.
- [ ] **Step 4b: Add uncertainty and seed range (spec §6.8 replication rules).**
  - Every headline mean in `results.json` gets a 95% hierarchical bootstrap CI: 2,000 resamples, first over subjects, then over images within each subject.
  - `scripts/03_train.py --all --seeds 1 2` trains two extra encoder seeds, and `--seed-range` reports the min–max of the headline numbers across seeds 0–2. Seed 0 is the pre-registered one and is never re-chosen.
- [ ] **Step 5: Run for real on 10 subjects. Gate 2:** inspect the figures, then write a short plain-English verdict for each hypothesis in `outputs/results_summary.md`, reporting negative results as they are.
- [ ] **Step 6: Commit** `feat: calibration results` (the code only; `outputs/` stays gitignored).

---

## Part D: Meta API visuals and words (Oct 31–Nov 2; API work can start Oct 30)

### Task 11: Meta API client with ledger, cap and smoke test

**Files:**
- Create: `sureenough/meta_api.py`, `scripts/00_smoke_api.py`
- Test: `tests/test_meta_api.py`, `tests/fixtures/api/*.json` (recorded responses)

**Interfaces:**
- Produces:
  - `MetaClient(key: str | None = None, ledger_path=OUTPUTS/'api_ledger.jsonl', cap_usd=60.0)`. The key is read from `.env` and never logged; `repr` redacts it.
  - `.responses(payload: dict, est_cost: float) -> dict`: POSTs `/v1/responses` with `stream: false` and `store: false` forced on.
  - `.image_edit(payload: dict, est_cost: float) -> dict`: POSTs `/v1/images/edits` as raw JSON.
  - `.spent() -> float`.
  - Every call:
    1. raises `BudgetExceeded` if `spent() + est_cost > cap` (no network I/O);
    2. appends a `pending` ledger line;
    3. sends with retry (429/5xx, exponential backoff, 5 tries);
    4. appends a `done` line with the usage.
  - Cost helpers: `cost_spark(in_tok, out_tok)`, `COST_SAM = 0.0025`, `COST_IMAGE = 0.01`.
  - `sam_payload(noun: str, image_data_url: str) -> dict`, `spark_payload(system: str, user: str) -> dict`, `image_edit_payload(prompt, image_data_urls, n=3) -> dict`. The last one sets `tool_enablement` with all three switches false, `reasoning_strength="low"`, `output_format="webp"` and `size="1024x1024"`.

- [ ] **Step 1: Write the failing tests** (no network: monkeypatch `requests.Session.post`).
  - **Review Focus 3:** with `spent() = 59.995`, an `.image_edit(est_cost=0.03)` raises `BudgetExceeded` and the patched `post` was never called.
  - After a simulated 429, 429, 200, the ledger has one `pending` and one `done` line, and `spent()` increases once.
  - `repr(client)` does not contain the key.
  - Every payload builder: `model` is in the allowed set and does not end with `-contributor`, and Responses payloads have `store is False`.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to verify they pass.**
- [ ] **Step 5: Smoke test (live, about $0.04).** `scripts/00_smoke_api.py` sends:
  - one Spark call ("Say OK");
  - one SAM call on one CC0 image with its noun;
  - one Muse Image edit with n = 1 and the search switches off, referencing one CC0 image.

  Save the raw responses (key-free) to `tests/fixtures/api/`. Report whether Muse Image succeeded. If it returns a geo or permission error, set `MUSE_IMAGE_ENABLED=false` in `outputs/flags.json` and continue.
- [ ] **Step 6: Commit** `feat: Meta API client with spend cap; smoke test`.

### Task 12: SAM 3.1 cutouts and composites

**Files:**
- Create: `sureenough/visuals/__init__.py`, `sureenough/visuals/sam.py`, `sureenough/visuals/composite.py`, `scripts/06_sam_cutouts.py`
- Test: `tests/test_sam.py`, `tests/test_composite.py`

**Interfaces:**
- Produces:
  - `parse_sam_line(text: str) -> list[dict(box, mask_rle, c)]`, parsing the `<0f>k<|box;...|><|mask;...|>` record format.
  - `decode_mask(record, w, h) -> np.bool_ (h, w)`, implemented exactly per `https://dev.meta.ai/docs/sam/reading-segmentation` (lossless `~` and one-bit `!`).
  - `cutout(img: PIL.Image, mask) -> PIL.Image` (RGBA, cropped to the box with 8% padding).
  - The script, for each of the 200 test concepts:
    - downscales the CC0 image to ≤ 768 px;
    - sends it as a base64 data URL with the noun;
    - keeps the highest-`c` mask;
    - writes `outputs/visuals/cutouts/{things_id}.webp`.
  - It falls back to the plain tile when there is no mask, or the mask area is < 2%, and logs the count. Resumable: it skips existing files.
  - `composite(members: list[str], weights: list[float], size=512) -> PIL.Image`: cutouts centred on a neutral grey (#808080) background, alpha-blended with weights normalised to sum to 1.

- [ ] **Step 1: Write the failing tests.**
  - Parsing a recorded SAM response from `tests/fixtures/api/` gives ≥ 1 record with a box inside (w, h).
  - The decoded mask area is > 0 and matches the box bounds.
  - `composite([a], [1.0])` equals the single cutout on grey.
  - Composite weights summing to 2 are normalised.
  - The output is 512×512 RGB.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to verify they pass.**
- [ ] **Step 5: Run the 200-concept job** (≈ $0.50) in the background.
- [ ] **Step 6: Commit.**

### Task 13: Muse Image impressions (only if the smoke test enabled it)

**Files:**
- Create: `sureenough/visuals/impression.py`, `scripts/07_impressions.py`
- Test: `tests/test_impression.py`

**Interfaces:**
- Produces: `impression_prompt(claim: dict) -> str`, a fixed template naming only the claim's concepts or category, e.g. "An artist's impression of one of: wolf, husky, coyote. Simple, centred, neutral background."
- The script covers demo subjects only (the top-2 validation top-1 subjects from `gate1.json`), the evaluation half of the pinned split, and the Hedged, Top-1, Answer key and Blind guess conditions. Each call:
  - uses n = 3 and the CC0 tiles of up to 4 top members as references;
  - writes `outputs/visuals/impressions/sub-XX/img-YYY_{cond}_{0..2}.webp`;
  - logs `usage`.
- Budget: about 2 subjects × ≤ 100 images × 4 conditions × 3 = 2,400 images ($24). Run Hedged and Top-1 first; Answer key and Blind guess follow only if spend < $35.

- [ ] **Step 1: Write the failing test.** The prompt for a set claim contains only the member names. The prompt for a category claim contains only the category name. Neither contains "probably".
- [ ] **Step 2: Run it** and verify it fails.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it** and verify it passes.
- [ ] **Step 5: Run.**
- [ ] **Step 6: Commit.**

### Task 14: Display rules, faithful narration, open-vocabulary naming

**Files:**
- Create: `sureenough/narrate/__init__.py`, `sureenough/narrate/display.py`, `sureenough/narrate/faithful.py`, `sureenough/narrate/spark.py`, `sureenough/narrate/naming.py`, `scripts/08_narrate.py`
- Test: `tests/test_display.py`, `tests/test_faithful.py`

**Interfaces:**
- Produces:
  - `display_claim(members: list[int], concepts: Concepts) -> dict(kind: 'single'|'category'|'list'|'full', text: str, category: str|None)`:
    - size 1 → `"It's a {name}."`;
    - all members share a category covering ≤ 25% of the 200 test concepts → `"It's {a/an} {category}."` (smallest such category, then alphabetical);
    - size 200 → kind `full`, `"Not sure yet: it could be any of the 200."`;
    - otherwise → `"It's one of these {N}."`
  - `faithful(text: str, allowed: set[str], vocabulary: set[str]) -> bool`: False if the text mentions any vocabulary term (1854 names plus 53 categories, word-boundary, case-insensitive) outside `allowed`, or contains "probably".
  - `narrate(client, claim) -> dict(text, faithful: bool, request, response)`: Spark rewrites the claim in one plain sentence. If not faithful, the text falls back to `claim['text']`.
  - `pe_guess(eeg_vec_final (1024,), text_feats (1854, 1024)) -> str`: argmax cosine. If the target is not `final`, a ridge map from the target layer to final, fitted on training images, comes first.

- [ ] **Step 1: Write the failing tests (Review Focus 4).**
  - A set containing an uncategorised concept gives kind `list`.
  - A set whose only shared category is `food` gives `list`.
  - Two qualifying categories pick the smaller one, then alphabetical.
  - A singleton gives `single`.
  - `faithful("It's one of: wolf, husky.", {"wolf","husky"}, vocab)` is True.
  - Adding "dog" makes it False.
  - "probably a wolf" is False.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to verify they pass.**
- [ ] **Step 5: Narrate** demo-subject examples (pinned split, eval half) into `outputs/narration/sub-XX.json` with the request and response kept for the provenance drawer. Cost is about $0.20.
- [ ] **Step 6: Commit.**

### Task 15: H5a scoring with DINOv2 (local only)

**Files:**
- Create: `sureenough/analysis/h5a.py`, `scripts/09_h5a.py`
- Test: `tests/test_h5a.py`

**Interfaces:**
- Produces:
  - `load_dino()`: `timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, img_size=224, num_classes=0)`, with the same 224 resize for every image.
  - `dino_embed(images) -> (B, 1024)`, L2-normalised.
  - `h5a(hedged_sims: (n,), top1_sims: (n,)) -> dict(p, median_diff, cvar20_hedged, cvar20_top1)`. Here p is `wilcoxon(hedged - top1, alternative='greater')` and CVaR20 is the mean of the lowest 20% of sims.
- Images used: the demo subjects' eval-half images where the stream-0 top-1 at the stop step is wrong. Composites from Task 12, impressions (mean of 3) when enabled. Each is compared to the **true stimulus** read locally from `test_images.zip`.

- [ ] **Step 1: Write the failing test.** `h5a` on synthetic sims where hedged = top1 + 0.05 gives p < 0.01 and the CVaR fields are present.
- [ ] **Step 2: Run it** and verify it fails.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it** and verify it passes.
- [ ] **Step 5: Run.** Write the H5a p-value into `results.json` and re-run Holm (`scripts/05_calibrate.py --holm-only`). Also record H5b descriptively, with no p-value: the mean sims for Answer key and Blind guess next to Hedged and Top-1, labelled "exploratory".
- [ ] **Step 6: Commit.**

---

## Part E: Export and web demo (skeleton Oct 31; full Nov 1–2)

### Task 16: Web data export

**Files:**
- Create: `sureenough/export/__init__.py`, `sureenough/export/web_data.py`, `scripts/10_export.py`, `web/src/data/schema.ts` (types mirrored by hand)
- Test: `tests/test_export.py`

**Interfaces (the contract the web app relies on):**
- `web/public/data/manifest.json`:
  ```
  {alpha, delta, K, m, R, lam: number[500],
   subjects: number[], demo_subjects: number[], muse_image: boolean,
   pinned: {subject: {cal: number[100], eval: number[100], lam_idx: number}},
   concepts: [{i, name, things_id, tile, cutout, category|null}],
   results: <subset of results.json for charts>}
  ```
- `tables/sub-XX.json`:
  - `loss_count`: number[200][500] (0..10);
  - `stop_x10`: number[200][500] (mean stop × 10, rounded);
  - `size_x10`: number[200][500];
  - `naive`: `{a_true_u16: number[200][10][20], top3_u16: number[200][10][20]}` (value × 65535, rounded; Review Focus 2).
- `examples/sub-XX/YYY.json`, eval half of the pinned split only. Per step t = 1..20 of stream 0:
  - `top`: [[idx, p_u16] × 30];
  - `set_size` at the pinned λ;
  - `stopped_at`;
  - `eeg`: number[50] (the occipital channel mean of the running average, downsampled 2×, scaled to int8);
  - plus `claim`, `narration`, `provenance`, `pe_guess`, and the impression/composite paths.
- `img/tiles/{things_id}.webp` (CC0, 256 px), `img/cutouts/…`, `img/composites/sub-XX/…`, `img/impressions/…`.
- `fixtures/ltt_fixture.json`: a 1-subject loss table plus 5 split seeds with the Python-deployed λ indices and eval risks, for Task 18.

- [ ] **Step 1: Write the failing tests.**
  - The export of a synthetic 2-subject fixture validates against the keys above.
  - Total size ≤ 150 MB.
  - **Forbidden content:** no file under `web/public/data` is byte-identical to any member of `test_images.zip` or `training_images.zip` (hash check, skipped if raw is absent), and no JSON key contains `feature` or `embedding`.
  - Example files exist only for eval-half indices.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to verify they pass.**
- [ ] **Step 5: Run the real export.**
- [ ] **Step 6: Commit** the code and `web/public/data` (CC0 and derived data only).

### Task 17: Web skeleton and Pages deploy

**Files:**
- Create: `web/` (Vite React TS template), `web/vite.config.ts` (`base: '/SureEnough/'`), `web/src/main.tsx` (HashRouter), `web/src/lib/data.ts`, `web/src/App.tsx`, `web/src/components/{Header,Footer}.tsx`, `.github/workflows/pages.yml`
- Test: `web/src/lib/data.test.ts` (vitest)

**Interfaces:**
- Produces: `dataUrl(path: string): string` = `import.meta.env.BASE_URL + 'data/' + path`; `loadManifest(): Promise<Manifest>`; `loadTable(subject): Promise<Table>`; `loadExample(subject, i): Promise<Example>`.
- Routes: `#/` (demo), `#/results`, `#/how`.
- `pages.yml`: Node 24, `npm ci`, `npm run build`, `actions/configure-pages@v6`, `actions/upload-pages-artifact@v5` (path `web/dist`), `actions/deploy-pages@v5`.

- [ ] **Step 1: Write the failing test.** `dataUrl('manifest.json')` ends with `/SureEnough/data/manifest.json` when `BASE_URL = '/SureEnough/'`.
- [ ] **Step 2: Scaffold and implement** the header (hook plus the verbatim label) and footer (verbatim credit).
- [ ] **Step 3: Verify.** `npm --prefix web test` and `npm --prefix web run build` pass.
- [ ] **Step 4: Commit.** Deploy once the repo is public (needs Dhruvil's go-ahead; see the execution notes).

### Task 18: TypeScript LTT port and naive baseline (live recalibration)

**Files:**
- Create: `web/src/lib/ltt.ts`, `web/src/lib/naive.ts`, `web/src/lib/rng.ts`
- Test: `web/src/lib/ltt.test.ts`

**Interfaces:**
- Produces:
  - `hbPvalue(r, n, alpha): number` (binomial CDF in log space);
  - `fixedSequence(p: number[], delta): number`;
  - `recalibrate(table, rng) -> {lamIdx, evalRisk, naiveEvalRisk, meanStop, outcomeMix}`;
  - `mulberry32(seed)`.
- The browser's random split doesn't need to match Python's RNG. Equality with Python is checked on the fixture's explicit index lists.

- [ ] **Step 1: Write the failing tests.**
  - `hbPvalue(0, 100, .2)` ≈ `0.8**100`.
  - For each of the 5 fixture splits (explicit cal/eval indices), `lamIdx` and `evalRisk` equal Python's exactly.
  - The naive eval risk equals Python's within 1e-12.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to verify they pass.**
- [ ] **Step 5: Commit.**

### Task 19: Replay view, badge, counters, race, provenance

**Files:**
- Create: `web/src/views/Demo.tsx`, `web/src/components/{BrainSignal,TileGrid,Composite,Narration,Badge,Reveal,OutcomeCounters,Race,Provenance,Recalibrate}.tsx`, `web/src/lib/player.ts`
- Test: `web/src/lib/player.test.ts`

**Interfaces:**
- Produces: `Player(example, lamIdx)` with `.step(t)` → `{setMembers: number[], stopped: boolean, repeat: number}`, rendered with requestAnimationFrame at about 1 repeat per 600 ms. All animation goes through `player.ts`, so the video can drive it with `page.clock`.
- Behaviour per spec §7.1:
  - the "7 of 20" counter and the tiles dimming (200 → … → 1);
  - the composite crossfade, with the impression panel only if `manifest.muse_image`;
  - the narration line;
  - the PE guess labelled "not covered by the promise";
  - the verbatim badge plus "Stopped after N repeats";
  - the reveal captioned "A different photo of the same thing. The real stimulus can't be shown (licence).";
  - modes: "Pick a mystery" and "Curated examples" (must include one wrong and one unsure);
  - Recalibrate: the "New random split" button shows delivered vs promised next to naive in under 1 s;
  - OutcomeCounters, always visible;
  - Race vs `results.best_fixed_k` (hidden if H1 failed and parity isn't shown);
  - Provenance drawer showing the Spark and SAM request/response JSON.

- [ ] **Step 1: Write the failing tests.**
  - `Player` stops exactly at `example.stopped_at` for the pinned λ.
  - The set size before the stop equals the exported `set_size[t]`.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to verify they pass.**
- [ ] **Step 5: Commit.**

### Task 20: Results tab, How it works, Meta stack, e2e smoke test

**Files:**
- Create: `web/src/views/{Results,How}.tsx`, `web/src/components/{PromiseChart,MetaStack}.tsx`, `web/e2e/smoke.spec.ts`, `web/playwright.config.ts` (webServer `vite preview --base /SureEnough/`)
- Test: `web/e2e/smoke.spec.ts`

**Interfaces:**
- Results tab:
  - first, a plain SVG chart of promised vs delivered (naive vs SureEnough per subject), with a strictness selector of 1, 2 or 3 in 10 (α ∈ {.1, .2, .3}, recalibrated live);
  - then the figures from Task 10 as captions with numbers from the manifest;
  - plus the PE vs CLIP-H line.
- How it works covers LTT, α/δ and HB.
- MetaStack: Perception Encoder → SureEnough calibration → SAM 3.1 composites → Muse Spark words (→ Muse Image).

- [ ] **Step 1: Write the failing e2e test (Review Focus 5).**
  - Load `/SureEnough/#/`, press "Pick a mystery", and wait until the badge appears.
  - Click "New random split" and get a delivered-error number within 1 s.
  - Navigate to `#/results` directly and see the chart rendered.
  - At a 390×844 viewport, `document.documentElement.scrollWidth <= 390`.
- [ ] **Step 2: Install Playwright locally.** `PLAYWRIGHT_BROWSERS_PATH=0 npx --prefix web playwright install chromium` (inside the project). Then run the test and verify it fails.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm --prefix web run e2e` and verify it passes.
- [ ] **Step 5: Commit**, then push to deploy.

---

## Part F: Video and submission (Nov 2–4)

### Task 21: Deterministic demo video (≤ 2:30)

**Files:**
- Create: `video/script.md` (shot list and captions), `video/capture.spec.ts` (Playwright, `page.clock.install()`, 1920×1080 at 30 fps, screenshots to `outputs/video/frames/`), `video/build.sh` (ffmpeg H.264 `-crf 18 -pix_fmt yuv420p`)
- Test: the build script asserts duration ≤ 150 s via `ffprobe`.

Shot list (per spec §7.3):
1. The grid collapsing plus the promised-vs-delivered bar (0:00–0:12), before any explanation.
2. The hook.
3. One clean stop.
4. One unsure case.
5. One wrong answer that hedged well (H5a).
6. Live recalibration.
7. The Meta stack.
8. The honesty notes (lab EEG, repeats, not live).
9. Links.

Captions only, following the copy rules.

- [ ] **Step 1: Write `video/script.md`** and get Dhruvil's OK on v0 (shown to one non-expert).
- [ ] **Step 2: Capture and build.** Expected: `outputs/video/sureenough.mp4` ≤ 150 s.
- [ ] **Step 3: Commit** the script, the capture code and the build script (not the frames).

### Task 22: README, submission text, novelty re-check

**Files:**
- Create: `README.md`, `docs/submission.md`, `scripts/11_headline.py`
- Test: `tests/test_headline.py`. `scripts/11_headline.py` recomputes the headline numbers from `outputs/tables` in under 60 s and matches `results.json`.

- [ ] **Step 1: Write the failing test** for `11_headline.py` (it runs on the synthetic fixture in under 60 s and the numbers match).
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Write the README**: hook, demo and video links, honesty notes, reproduction steps, the Meta models used, credits and licences.
- [ ] **Step 4: Re-check novelty (Nov 3).** Search arXiv and GitHub for "adaptive stopping EEG decoding conformal / learn then test". Adjust any "first" claim.
- [ ] **Step 5: Write `docs/submission.md`** (track, what was built during the event, Meta models, links, Dhruvil's role) and hand it to Dhruvil to paste.
- [ ] **Step 6: Commit.**

---

## Execution notes

- **Order and parallelism:**
  1. Tasks 1–3 first. Task 3's full run starts tonight in the background.
  2. Tasks 4–7 and 11 run while it extracts.
  3. Tasks 8 → 9 → 10 are the critical path on Oct 31.
  4. 12/14 run in the background once 11 is done.
  5. 16–20 start with fixture data on Oct 31 and switch to real data when Task 10 lands.
- **Gates:** Gate 1 is in Task 8 Step 6. The config freeze is in Task 9 Step 3 and must be committed before Task 9 Step 5. Gate 2 is in Task 10 Step 5.
- The repo exists and is public (created by Dhruvil on Oct 30).
