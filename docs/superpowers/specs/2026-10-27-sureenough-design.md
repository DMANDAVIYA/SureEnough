# SureEnough: Design (v2)

**Date:** 2026-10-27
**For:** Dhruvil Mandaviya (Meta Global AI Developer Hackathon, Indie developer track)
**Author:** Dhruvil Mandaviya
**Status:** **v2.2: approved by Dhruvil on Oct 30, 2026** ("pull whatever you wanna pull... just ask before installing anything inside the system"). He delegated the remaining §14 choices to the builder; §14 records each call.
**Deadline:** Nov 4, 2026, 9:00 PM ET. We aim to submit by 5:00 PM ET.
**Name:** SureEnough (delegated). It was renamed from the working name "SureSight" because that name belongs to a Welch Allyn vision screener.

> **Notation.** "Dhruvil decided" marks choices Dhruvil made directly. "Builder's call" marks
> technical choices made under his delegation.
> "(review)" marks a change made in response to the design review (builder, statistics,
> feasibility and judge reviewers, Oct 27).

---

## 1. Intent and why

**The plain hook (review):** *Brain decoders always guess. SureEnough knows when it has seen enough,
says "not sure" when it isn't, and keeps its promise about how often it's wrong.*

**Background.**
- Dhruvil is a student accepted into the remote Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026, $1M prize pool).
- He wanted an "EEG + OS" project, has no EEG hardware, and delegated choosing the idea and building it.
- **Dhruvil decided** the direction: decode what a person was looking at from recorded EEG, and show it with Meta's models.
- He then asked for a genuine research contribution on top ("enhance the research and add something of ours").
- After a 21-agent research sweep, he approved **calibrated adaptive stopping ("SureEnough") plus the redraw comparison** ("go bud").

**Why it matters.**
- Every brain-computer interface (communication aids, attention-based selection, image decoding) has to decide *how long to listen before acting*, and acting wrongly is costly.
- EEG-to-image decoding on public data is our testbed. Published results report accuracy after averaging up to 80 repeated viewings of each test image.
- Some work studies fewer repeats at a *fixed* number. We found no published stopping rule that decides image by image when to stop *and* guarantees its error rate. This must be re-checked before any "first" claim (§6.10).

**What success looks like.**
1. **(Protected core.)** A hosted, clickable demo plus a 2–3 minute video. Together they make an instantly understandable "watch it become sure, or honestly unsure" moment.
2. Reproducible findings on THINGS-EEG2. At least the guarantee check (H2) and the efficiency comparison (H1) must be solid enough to become a paper.
3. Meta models doing *visible*, essential work (§8).

## 2. What we're building, in one paragraph

A Python research pipeline and a static web demo.

**The pipeline:**
1. A small EEG encoder maps THINGS-EEG2 brain responses into the embedding space of Meta's Perception Encoder.
2. For each held-out image, its 80 recorded repeats are replayed in random order, one at a time.
3. After each repeat, SureEnough forms a **candidate set** and decides whether to stop. A single-dial stopping policy is calibrated with Learn-then-Test, so that the final answer misses the true object on at most α of new images for that person, with 1−δ confidence.
4. We compare it against fixed-repeat rules and against the naive "check after every repeat" approach, which breaks its promise.
5. Meta SAM 3.1 cuts each candidate object out of a public-domain photo. The demo blends these cutouts, weighted by likelihood, into a composite that sharpens as the set shrinks.
6. Meta Muse Spark turns the calibrated claim into words and blind-captions the visuals.
7. Meta Muse Image adds an "artist's impression" **only where it is legally available** (it is not available in India; §4).

**The demo** replays a recorded brain, recalibrates the promise live in the browser, and shows promised vs. delivered error.

## 3. Goals, non-goals, anti-goals

**Goals**
- G1. A calibrated, adaptive stopping policy for zero-shot EEG-to-image retrieval with a finite-sample guarantee (core).
- G2. Confirmatory findings H1, H2b and H5a, plus the H2a guarantee check (§6.8), each reported honestly, including negative results.
- G3. A hosted demo and video (the protected core of the hackathon submission).
- G4. A paper. *Pending Dhruvil (§14):* the review recommends drafting it **after** Nov 4, from the frozen results.
- G5. Visible, essential use of Meta models.

**Non-goals**
- Live EEG from a headset (no hardware). A BrainFlow/LSL adapter is future work.
- Pixel-level reconstruction. Our visuals are concept-level by construction.
- New statistical theory. LTT and conformal prediction are prior work; we contribute their first application here, plus the findings.
- Leaderboard SOTA, cross-subject SOTA, other datasets, and the consumer-montage study (a possible second paper).
- LOSO encoders (decoders are per subject; §6.6).

**Anti-goals** (failure even if it technically works)
- **Overclaiming.** "Mind reading", implying live decoding, hiding repetition averaging, or a badge that reads as per-answer confidence (see the copy rules in §6.10).
- **Cherry-picked demos.** Every showcase sits next to the aggregate outcome counts; the demo includes a failure and an "unsure" case; demo images come only from the evaluation half of the pinned split.
- **Licence or terms violations.** THINGS stimulus photos leave the laptop or are made public; Muse Image is used from a restricted country; anyone circumvents geo-controls.
- **Hallucinated citations or numbers.**
- **A demo that depends on the build machine or holds an API key in the browser.**
- **Tuning on the test split** (§6.3, §6.6).

## 4. Constraints

| Constraint | Detail |
|---|---|
| Deadline | Nov 4, 9:00 PM ET (internal target: 5:00 PM ET). |
| Hardware | i5-1335U (2P + 8E cores, no AVX-512/AMX), 16 GB RAM, Intel Iris Xe (no CUDA/XPU). Data on internal D: (`D:\A5\metapr-data`). E: is an external USB drive and is not used. |
| Meta Model API | `https://api.meta.ai/v1`, Bearer key (keys start `LLM_`, team-scoped). No embeddings endpoint, no free tier listed, credits unverified. |
| Meta Model API: models and prices | muse-spark-1.3: $1.25 in / $4.25 out per 1M tokens, 3,000 RPM. sam-3.1: $2.50 per 1,000 images, 3,000 RPM. muse-image-1.0: $0.01 per image, 150 RPM, no seed. |
| Geography | Meta's Geographic Use Policy (updated Oct 12, 2026) §2.2 lists **Muse Image: India**: it may not be used from India, and §2.1 forbids circumventing geo-controls. Muse Spark, Muse Wonder and SAM 3.1 have no model-specific restriction. Where Dhruvil calls from decides whether Muse Image is used (§14). |
| Builder | Dhruvil directs the build and supplies keys, accounts, rules, approvals and verification. |
| Licences | THINGS-EEG2 EEG: CC-BY 4.0. THINGS stimulus photos: research use only (no reproduction, distribution or modification). THINGSplus images: CC0, one per concept, 1,854 in total. THINGS metadata: CC BY, with credit. |
| Authorship | arXiv, NeurIPS, ICLR, ICML, Nature and Elsevier forbid AI authors. Dhruvil is the sole author. Dhruvil verifies every number and citation. |
| Hackathon rules | The Official Rules are behind the Cerebral Valley login and **still unread** (§14).<br>Public facts (Meta event page, checked Oct 30):<br>• Participants "use the Muse Spark series, Muse Image and SAM 3.1 through Meta Model API".<br>• Meta promotes Muse Code as an agentic coding agent, so AI-written code is expected.<br>• Indie track: judged on creativity, technical ambition and whether it works. No official weights.<br>Unverified secondary sources claim a 2–3 minute video, a public repo, and weights of impact 30 / creative use of Meta AI 25 / demo and video 25 / technical depth 20. |

## 5. Decided by Dhruvil
- The Mind's Eye direction, then the SureEnough contribution plus the redraw comparison.
- Technical choices are delegated to the builder.
- No hardware; public recordings.
- Indie developer track.
- A paper if results are good. Venue policy makes Dhruvil the sole author.

## 6. Research design

### 6.1 Data (builder's call)
- **EEG:** THINGS-EEG2 official 17-channel preprocessed release (OSF `anp5v`), downloaded and verified in `D:\A5\metapr-data\raw\osf-anp5v\`.
  - Each `sub-XX.zip` holds `preprocessed_eeg_training.npy` (16540, 4, 17, 100) and `preprocessed_eeg_test.npy` (200, 80, 17, 100).
  - Each file is a dict with keys `preprocessed_eeg_data`, `ch_names` and `times` (100 samples, −0.20 to 0.79 s, 100 Hz), stored as float64.
  - Channels: O1, Oz, O2, PO7, PO3, POz, PO4, PO8, P7, P5, P3, P1, Pz, P2, P4, P6, P8.
- **Images:** OSF `y63gw`, research-only. `image_metadata.npy` has the keys `train_img_concepts`, `train_img_files`, `train_img_concepts_THINGS`, `test_img_concepts`, `test_img_files` and `test_img_concepts_THINGS`.
- **CC0 proxies:** `images_THINGSplus-CC0.zip`, one per concept, named by THINGS unique ID. Map to them through `*_concepts_THINGS` and `concepts-metadata_things.tsv`.
- **Categories:** use the 53 higher-level categories (`category53_*.tsv`). They are **flat and overlapping, not a tree**: 406 of 1,854 concepts have no category, and 664 have two or more. 195 of the 200 test concepts have at least one category (corrected Oct 30 from 194; both category files agree).
- **Typicality:** available only for categorised concepts.
- **Safe loading (review):**
  - Every `.npy` here, including `image_metadata.npy`, is a pickled dict.
  - Read it with `np.lib.format.read_magic` plus the header readers, then unpickle with an allow-list `Unpickler`. The allow-list is `numpy(._)core.multiarray._reconstruct` and `.scalar`, `numpy.ndarray`, `numpy.dtype`, and builtins `dict`, `list` and `str`.
  - Run it in an isolated `python -I` process, and convert once to plain float32 `.npy` plus JSON. Never use `allow_pickle=True`.
- **Validation split:** hold out 200 of the 1,654 training concepts (all 10 images each). It is concept-disjoint, and it is **the only data used for model selection**.
- **Preprocessing note (for the paper):** the released MVNN whitening used each session's test and training trials together. This is unsupervised and symmetric across test images, so exchangeability is unaffected; disclose it.
- **Sessions (review):** each image's 80 test repeats come from 4 recording sessions (20 each) and were shuffled with a fixed seed. Random-order streams therefore **pool repeats from 4 sessions recorded on different days**. Never describe K repeats as "a session" or as "N seconds". Recovering session labels needs the raw event files; that is optional and post-hackathon.

### 6.2 Image target space (builder's call)
- **Loading:** Perception Encoder Core B/16 via `open_clip.create_model_and_transforms('hf-hub:timm/PE-Core-B-16')` (Apache-2.0, ungated, image and text towers).
- **Features (one CPU pass, after benchmarking 64 images):** for every image, for each of the 12 blocks of `model.visual.trunk.forward_intermediates`, keep the CLS token and the mean of the patch tokens, both 768-d. Also keep the final `encode_image` output (1,024-d, shared with the text tower). Store as fp16, about 0.6 GB in total.
- **Preprocessing:** 224×224, mean = std = 0.5.
- **Target choice:** a validation-only sweep over {final 1,024-d} ∪ {selected intermediate layers}.
- **Naming:** open-vocabulary naming (§7) uses the final 1,024-d space. If an intermediate layer wins, fit a ridge map from that layer to the final space for naming only.
- **Not cut (review):** the comparison with OpenCLIP ViT-H/14 features precomputed by ATM (`datasets/LidongYang/EEG_Image_decode`: `ViT-H-14_features_{train,test}.pt`, about 76 MB). Load with `torch.load(..., weights_only=True)` in an isolated process and verify the row order. Report "repeats needed for X%: PE vs CLIP-H" in one line, whichever wins.

### 6.3 EEG encoder (builder's call)
- **Model:** an MLP (flatten 17×100 → about 2,048 → target dimension) with LayerNorm, GELU and dropout. Trained per subject with symmetric InfoNCE and a learnable temperature.
- **Augmentation:** average a random 1–4 of the 4 training repeats.
- **Baseline:** closed-form ridge regression.
- **Gate 1 (review, validation only):** on Oct 28, 200-way validation top-1 at k = 4 repeats must be **≥ 15%** (chance is 0.5%) for at least 8 of 10 subjects. The target is ≥ 30%.
  - If it fails, try these fallbacks in order: CLIP-H target, ridge, temperature or layer changes, then the ATM recipe.
  - One look at test k = 80 top-1 is allowed for sanity only, and it is disclosed in the paper.

### 6.4 Repeat streams and scores
- **Streams:** for each subject and test image, draw **R = 10** random orders of the 80 repeats from seeds fixed in code before any analysis. Never redraw or drop streams after seeing outcomes.
- **Steps:** at step t, average the first t repeats, encode, and compute cosine similarity to all 200 test-image targets.
- **Score (review):** z-score the 200 similarities at each t, then apply softmax(β·z), with β fit on the validation split (k ≤ 4). Freeze it.
- **Cache:** per-step outputs (top-k indices and probabilities) and the per-image, per-λ losses and stopping times. Every analysis runs on these caches.
- **Budgets:** **K = 20** (primary) and K = 80 (secondary).

### 6.5 Policies (review: one dial, one loss)

**Primary policy (pre-registered).** This is the one in the demo badge.
- **Set:** at step t, C_t(λ) is the smallest set of top candidates whose cumulative probability is ≥ λ.
- **Stop:** at the first t ≤ K where |C_t(λ)| ≤ 3. Otherwise output C_K(λ) at the budget.
- **Loss:** 1 if the true image is not in the final set.
- **Dial:** λ ∈ (0, 1]. λ = 1 means "the full gallery, never stop early", which has risk 0 and is the default when nothing is certified.
- **Settings:** **K = 20, α = 0.2, δ = 0.1.**

**Display rules.** Every displayed claim is *implied* by the set claim, so it inherits the same guarantee.

| Final set | Shown as |
|---|---|
| Size 1 | "It's a dog." |
| Every member in one 53-category that covers ≤ 25% of the 200 test concepts | "It's an animal." Use the smallest such category, with a deterministic tie-break. ("Food" covers 26.5% and is excluded.) |
| Anything else | "It's one of these N." |

The per-output-type empirical accuracy is reported with confidence intervals as a descriptive result, not a guarantee.

**Exploratory variants** (labelled exploratory; never in the badge):
- K = 80.
- α ∈ {0.1, 0.3}.
- Stop size m ∈ {1, 5}.
- A selective point/abstain policy controlling P(wrong | answered) (LTT §3.2).
- The Ringel/Jazbec accuracy-gap risk.

### 6.6 Calibration
- **Method:** LTT fixed-sequence testing over a λ grid that runs **from λ = 1 down**, fixed a priori. Test H₀: risk(λ) > α with Hoeffding–Bentkus p-values. Deploy the **last λ rejected before the first non-rejection**, i.e. the boldest certified setting.
- **Losses:** each image's loss is **averaged over its R streams** (in [0, 1]), and n = the number of distinct images. The code asserts n equals the count of unique image IDs.
- **WSR betting p-values:** reported as exploratory. If they pass the synthetic validity suite by Oct 28, they may be promoted to primary before any test-split calibration, and the doc is amended.
- **Freeze (review):** grids, schedules, the λ order, β, the selection rule and the primary config are frozen in a hashed config committed *before* the first calibration on any subject's test streams. A unit test checks that the ordering and grid code never receives calibration-half arrays.
- **Regime 1 (primary, within-subject):** each subject's 200 test images, split 100 calibration / 100 evaluation over 1,000 random splits.
  - The guarantee is stated in **finite-population form**: *for a uniformly random image of this 200-image gallery, for this person and these recordings, replayed in random order, with probability ≥ 1−δ over the calibration draw*.
  - Generalising to new images or galleries is an explicit extra assumption.
- **Regime 2 (exploratory, H4):** "calibration transfer across subjects". Each subject keeps their own encoder; only thresholds come from the other 9.
  - Pooled certificates average each image's loss over the 9 subjects (n = 200).
  - A subject counts as a violation only if a one-sided 95% binomial lower bound exceeds α.
  - Personal recalibration runs LTT on personal images alone (n ∈ {25, 50, 100}). It is valid by construction and reported as an efficiency question. n = 10 is shown only to illustrate that it certifies nothing at α ≤ 0.2.
  - Do this only if it's done by Oct 31; otherwise it's post-hackathon.

### 6.7 Baselines
1. **Fixed-k rules (k ≤ K):** for each k ∈ {1, 2, 4, 8, 16, 20}, a set C_k(λ_k) LTT-calibrated at that fixed k. The fair comparator is the lower convex hull of the fixed-k points.
2. **Naive repeated look (review):** for each t = 1..K, a split-conformal threshold q̂_t calibrated at that t on the calibration half, using the same score. Each fixed-t set has ≥ 1−α coverage. Stop at the first t with |C_t| ≤ 3, otherwise output C_K. This is the peeking problem.
3. **Bonferroni repeated look:** each q̂_t at level α/K. It is valid, and it shows the cost of the obvious fix.
4. **Uncalibrated threshold:** stop when max probability ≥ 0.8 (matching α = 0.2).
5. **Stretch:** COINS (2609.07112) or time-uniform conformal (2602.06297). Label guarantee types: LTT gives (α, δ) with high probability; those give expectation guarantees.

### 6.8 Hypotheses (pre-registered before any test-split result)

**Confirmatory family (Holm at 0.05 across these three only):**

| ID | Statistic and test |
|---|---|
| **H1** (efficiency) | Primary config. Per subject, mean evaluation-half repeats of SureEnough vs. the fixed-k convex hull, interpolated at SureEnough's mean final set size. Hypothesis: SureEnough uses ≤ 0.7× the hull's repeats. One-sided Wilcoxon signed-rank across the 10 subjects. *Pre-registered as possibly negative.* |
| **H2b** (naive under-covers) | Per subject, the naive baseline's exact population miss rate (over all 200 images and streams) exceeds α. One-sided sign test across subjects; expected in ≥ 8/10. |
| **H5a** (hedging helps when wrong) | On test images where top-1 is wrong (demo subjects): is the **set visual** closer to the true stimulus than the **top-1 visual**? Measured as DINOv2-L cosine, averaged over samples per image. Paired Wilcoxon across images. Tail metric: CVaR at 20%. |

**H2a (the guarantee check, must pass; not a rejection test).**
- Per subject, treat the 200 images (with their stream-averaged losses) as a known population.
- Draw B = 1,000 calibration sets of 100 *with replacement*, run the whole pipeline, and compute the exact population risk of the deployed λ.
- The violation count is Binomial(B, ≤ δ). Report the Clopper–Pearson 95% upper bound, which should be ≤ δ + 0.02.
- Also report the real without-replacement 100/100 splits scored by the 200-image risk.
- Descriptively, report evaluation-half risk next to Binomial(100, α) and an oracle λ.
- Report **conservativeness (α − mean risk)** as a headline number.

**Exploratory (confidence intervals, no p-values claimed):**
- **H3:** difficulty spread and hubness.
  - "Stable correct" means correct at t and at every later t ≤ 80, averaged over streams. It is right-censored, so use Kaplan–Meier quantiles.
  - Hubness is the k-occurrence of each test target among the top-k retrievals of **validation** EEG embeddings against the test gallery.
  - Per-subject Spearman with a permutation test, combined across subjects with Fisher z.
  - Typicality applies to the categorised subset only, with n stated.
- **H4:** calibration transfer (§6.6 regime 2).
- **H5b:** the gap between brain-driven visuals and the "answer key" (the cost of brain uncertainty), and brain vs. "blind guess".
- All exploratory α, K and m variants, and PE vs. CLIP-H.

**Replication rules.** Subjects are the replication unit; splits are never independent replications. Bootstrap over images within a subject and over subjects hierarchically. Use 3–5 seeds for the main configuration; report the range and never pick a seed.

### 6.9 Visuals and the redraw comparison (review: Muse Image no longer required)

**Primary visual (works everywhere): the SAM 3.1 composite.**
- For each of the 200 test concepts, call Meta SAM 3.1 once on its **CC0** photo with the concept noun as the prompt. That's about $0.50.
- Keep the object cutout.
- A candidate set is rendered as a likelihood-weighted blend of its members' cutouts on a neutral background, so it sharpens as the set shrinks.
- This is deterministic, cheap, has no copying or search confound, and sends no THINGS stimuli anywhere.

**Optional "artist's impression": Muse Image.** Only if Dhruvil calls from a country where it is permitted (§14).
- Raw JSON POST to `/v1/images/edits`, not the SDK's multipart `images.edit`.
- `tool_enablement: {enable_image_search: false, enable_web_search: false, enable_shell: false}`.
- Fixed `reasoning_strength`, `temperature` and `size`; `n = 3` per call.
- Log `revised_prompt` and `usage` for every image.
- CC0 references downscaled to about 768 px and uploaded once via the Files API with an expiry.
- Verify on day 1 that the search switch takes effect.

**Conditions** (renamed in the review, so the UI and paper use these names):

| Condition | Source |
|---|---|
| **Hedged** | The stopped set |
| **Top-1** | The top-1 candidate |
| **Answer key** | The true concept (an upper bound, not a "guess") |
| **Blind guess** | The set from a different image's stream (shuffled EEG) |

Each condition is rendered as a composite. If Muse Image is available, also as an impression: n = 3, averaged within each image.

**Text level:** Muse Spark writes a hedged description from each condition's claim, and blind-captions each visual. Score the semantic distance between the true concept and the concept named in the caption, using THINGS embedding or WordNet distance.

**Scoring (local only):**
- DINOv2-L (`facebook/dinov2-large`, 224 px, the same resize for every image) cosine to the true stimulus.
- Concept-2AFC.
- The stimulus photo is used only locally.

**Honest framing:** the brain contributes only concept-level information. The headline is H5a (graceful failure), and the answer-key gap is "the cost of brain uncertainty". If H5a is null, it moves to the results tab and the video drops it.

### 6.10 Reporting and copy rules (review)
- Every number states k, within- or cross-subject, α, δ, and "repeats from 4 sessions".
- **Badge text:** "Promise: for this person, the true object is outside SureEnough's final answer at most 2 in 10 times (checked with 90% confidence)." The tooltip says the same. Never "90% sure this answer is right".
- **Narration** restates the calibrated claim exactly ("one of: wolf, husky, coyote" / "an animal"). Never "probably". An automatic check verifies that Muse Spark's text names only concepts or categories inside the claim.
- Say "lab EEG (17 channels)". Consumer headsets are future work.
- Say "repeats", never seconds or "a session".
- Use "efficient" only if H1 holds.
- Cite SOTA numbers only with a "preprint" label, or not at all. SureEnough is decoder-agnostic.
- Make "first" claims only after a fresh arXiv and GitHub re-check on Nov 3. Cite LTT, CRC, Ringel, COINS, time-uniform conformal and Jazbec as the foundations.
- Research sweep notes stay local and unpublished (they contain third-party and truncated-search claims).

## 7. Demo app

### 7.1 Experience
- **Header:** the plain hook (§1). A permanent label reads: *"Replayed public lab EEG (THINGS-EEG2). Not live. Each repeat is a 100 ms flash in a rapid sequence."* The footer gives credit: *"EEG: THINGS-EEG2 (Gifford et al. 2022), CC-BY 4.0 · Object photos: THINGSplus, CC0."*
- **Main view** (styling left to the builder; panels stack on phones; the grid scrolls):
  - **Brain signal:** the averaged waveform visibly gets cleaner as repeats accumulate, which shows why more repeats help. Repeat counter "7 of 20".
  - **What it could be:** 200 CC0 tiles dim as the set shrinks (200 → 31 → 3 → 1).
  - **What it thinks they saw:** the SAM 3.1 composite sharpening, plus the Muse Image impression where available.
  - Muse Spark's claim-faithful sentence.
  - "Perception Encoder's open-vocabulary guess: *wolf*", over all 1,854 THINGS nouns and labelled "not covered by the promise".
- **Badge:** the exact copy from §6.10, with "Stopped after 7 repeats".
- **Reveal:** the true concept name and its CC0 tile, captioned *"A different photo of the same thing. The real stimulus can't be shown (licence)."*
- **Modes:**
  - **Pick a mystery:** a random subject and evaluation-half image.
  - **You pick the picture:** the judge chooses a tile and watches that image's recording converge, or fail.
  - **Curated examples:** must include a wrong answer and an "unsure" case.
  - Replays are available for all 10 subjects. Muse visuals cover 1–2 subjects, and the page says so.
- **Live recalibration (review):** a **"New random split"** button re-runs HB and fixed-sequence LTT in TypeScript on the cached per-image losses. It reports the error delivered on the other 100 images next to the naive method, in under a second.
- **Outcome counters:** always visible. They show the share of answers that are a single object, a category, a list, and the full set.
- **Race:** SureEnough vs. **the best calibrated fixed-k rule at the same α** (answers given, wrong rate, repeats used). If H1 is negative, show parity honestly or drop the race.
- **Provenance drawer:** the actual Muse Spark and SAM 3.1 requests and responses for the example shown.
- **Results tab:** leads with one plain chart: promised vs. delivered error, naive vs. SureEnough, per subject. It has a strictness selector (1, 2 or 3 in 10), followed by the other figures with plain captions.
- **"How it works" drawer:** LTT, α/δ and HB, for experts.
- **Meta stack diagram:** Perception Encoder → SureEnough calibration → SAM 3.1 composites → Muse Spark words (→ Muse Image).

### 7.2 Architecture (builder's call)
- **Precompute in Python, export to `web/public/data/`.**
  - Size budget: ≤ 150 MB.
  - Per-example JSON with quantised per-step outputs: set members, top-k probabilities, stop step.
  - Per-subject per-image, per-λ loss tables for live recalibration.
  - Downsampled EEG traces.
  - CC0 tiles and cutouts as WebP, plus composites and impressions.
  - **Never** PE features of THINGS stimuli, or the stimuli themselves.
- **Front end:** Vite + React + TypeScript with HashRouter. All data URLs are built from `import.meta.env.BASE_URL`, and Vite `base: '/<repo>/'`.
- **Hosting:** GitHub Pages via an Actions workflow (configure-pages v6, upload-pages-artifact v5, deploy-pages v5). This needs a **public** repo (§14). Deploy a skeleton early (Oct 29). Drop the "works offline" claim.

### 7.3 Demo video (builder's call; format per the Official Rules)
- **Production:** scripted Playwright with `page.clock`, deterministic 1920×1080 frame capture, and H.264 MP4 via ffmpeg (8.1.1, installed). No low-quality WebM recording. All animations are JS/rAF-driven.
- **Content:** opens on the grid collapsing plus the promised-vs-delivered bar, *before* any explanation, then covers the failure and unsure cases and the Meta stack.
- **Optional:** a 15–20 s face or voice intro from Dhruvil (§14).
- **Schedule:** v0 by Sat Nov 1, shown to one non-expert; copy frozen Nov 2.

## 8. Meta model usage

| Model | Role | Visible where |
|---|---|---|
| **Perception Encoder Core B/16** (open weights) | The space the EEG is decoded into; text tower for open-vocabulary naming | Decoding results; the "PE's guess" line; PE vs. CLIP-H result |
| **SAM 3.1** (API) | Cuts each candidate object out of its CC0 photo for the composites | The composite panel; provenance drawer |
| **Muse Spark 1.3** (API) | Turns the calibrated claim into faithful words (with an automatic faithfulness check); blind-captions visuals for scoring | The narration line; provenance drawer; H5 text scores |
| **DINOv2-L** (open weights) | Held-out scoring space for visuals | H5a |
| **Muse Image 1.0** (API, *only where permitted*) | Artist's impression from the claim | Impression panel |
| Muse Code, Muse Wonder | Not used | — |

API hygiene: standard model IDs only, never `-contributor` (it grants training rights). Send `store: false` on Responses calls. Delete uploaded files after use. The key is read explicitly from `.env` (`META_API_KEY`).

## 9. Engineering

### 9.1 Environment (builder's call)
- **Rule from Dhruvil (Oct 30):** ask before installing anything system-wide. Everything installs inside the project.
  - Python goes in `.venv` (stdlib `venv` with the installed Python 3.14; no uv, because uv would be a system install).
  - npm packages go in `web/node_modules`.
  - Playwright browsers go under the project (`PLAYWRIGHT_BROWSERS_PATH=0`).
  - Hugging Face weights are cached at `D:\A5\metapr-data\hf-cache` (`HF_HOME`), not in the user profile.
- Hardware (checked Oct 30): i5-1335U (10 cores / 12 threads), 15.7 GB RAM (about 4 GB free at idle), no GPU, 44 GB free on D:. Memory-map large arrays and convert one subject at a time.
- Dependencies: torch and torchvision (CPU wheels), timm, open_clip_torch, numpy, scipy, scikit-learn, pandas, matplotlib, pillow, requests, python-dotenv, pytest. DINOv2 comes through timm (`vit_large_patch14_dinov2.lvd142m`), so transformers isn't needed.
- Web: Node 24 and npm (already installed).
- Paper: LaTeX in `paper/`, compiled with tectonic or on Overleaf.

### 9.2 Repo layout
```
sureenough/   data/ features/ models/ streams/ calibrate/ analysis/ visuals/ narrate/ export/
scripts/     numbered CLI steps
tests/       pytest; statistics validated on synthetic data
web/         Vite + React demo
paper/       LaTeX + generated figures
docs/        specs and plans
outputs/     (gitignored) caches, scores, figures, visuals
```

### 9.3 Data handling, safety and licences
- Untrusted downloads stay in `D:\A5\metapr-data\raw\<source>\`. Load them with the safe loader (§6.1).
- **THINGS stimuli never leave the laptop.** They are never sent to an API, committed, shown on the site or in the video, or exported as features. At most a few small figures appear in the paper, under research fair use, with Dhruvil's sign-off.
- Public material uses CC0 images only. EEG is shown with CC-BY credit.
- `.env` is gitignored. Keys are never logged, committed or sent to the front end.
- While free credits are in use, Meta may use API content for product improvement. Only CC0 images and our own text are sent.

### 9.4 Testing approach (builder's call)
- **Statistics, test-first:** HB and WSR p-values, fixed-sequence LTT, set construction, the stopped-policy evaluator, and the H2a resampling check. Validate on **synthetic streams with known ground truth**, including without-replacement finite-population splits.
- The tests assert only that **LTT is valid**: the CP upper bound on violations is ≤ δ + 0.02. Naive under-coverage is *observed*, not asserted.
- Assertions: n = the number of unique images; the ordering code never sees calibration arrays.
- **Loaders:** shape, dtype and key tests.
- **API clients:** recorded-response tests, with no live calls in the suite.
- **Web:** a Playwright smoke test of load, replay and recalibration. The TypeScript LTT port is cross-checked against Python on fixed inputs.

## 10. Deliverables
1. **Public GitHub repo.** README with the hook, honesty notes, reproduction steps (including a command that recomputes the headline numbers from cached scores in under a minute), the Meta models, credits and licences, and the licence notes.
2. **Hosted demo** on GitHub Pages.
3. **Demo video.**
4. **Submission text:** built during the event, the Meta models used, links, and Dhruvil's role (direction, decisions, verification).
5. **Paper draft:** after Nov 4, from the frozen results (delegated).

## 11. Timeline (critical path first)

**Compressed again on Oct 30 (builder's call):** approval landed at 4:20 PM ET on Oct 30. Same gates, same never-cut list. Cut items 1, 2 and 4 are **cut up front**, and each comes back only if a day finishes early.

| Day | Work |
|---|---|
| **Thu Oct 30 (from 4:30 PM)** | Repo public, with the approved-spec commit as the pre-registration timestamp. Venv. Safe data conversion. Statistics module test-first on synthetic data. **API smoke test:** 1 Spark call, 1 SAM 3.1 call, 1 Muse Image edit (checking the search switch). PE feature extraction runs **overnight**. |
| **Fri Oct 31** | Validation split. Encoder plus ridge on 2–3 subjects. Layer sweep. β fit. **Gate 1 (validation, by early afternoon).** Freeze the hashed config. Then all 10 subjects, cached streams, and the fixed-k, naive, Bonferroni and primary policies. **Outcome mix.** SAM 3.1 cutouts and Muse Image impressions run in the background. Web skeleton deployed to Pages. |
| **Sat Nov 1** | LTT over 1,000 splits plus the H2a resampling check. **Gate 2: H1/H2 figures.** Results JSON. Main replay view on real data. Live recalibration. |
| **Sun Nov 2** | Composites, impressions, and Muse Spark narration with the faithfulness check. H5a study. Curated examples. Race, provenance, mobile layout. **Video v0 (evening)**, shown to one non-expert. |
| **Mon Nov 3** | Fix what confused the viewer. **Freeze numbers and copy (noon).** Dhruvil verifies every number on the site and in the video. Final video. README. Submission text drafted. |
| **Tue Nov 4** | Buffer. **Submit by 5:00 PM ET.** |

**Cut order if behind schedule (items 1, 2 and 4 are already cut as of Oct 30):**
1. Stretch baselines.
2. H3/H4.
3. Muse Image impressions.
4. The "You pick" mode.
5. The H5a text level.

**Never cut:** H2a/H2b, the primary policy, the replay view, live recalibration, the honesty copy, the PE vs. CLIP-H line, and the video.

## 12. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Muse Image is blocked for Dhruvil's account (it bills in INR, so its country may be India), even though he is in the US | Already designed around: SAM 3.1 composites plus Muse Spark. Optionally, Dhruvil asks the organisers on Discord or Meta support. Never use a VPN, and never change the account country to anything untrue. |
| The API key is delayed | Data and statistics work needs no API until the SAM cutouts on Oct 30. Credits are confirmed ($150 free). |
| The outcome mix is mostly "full set" at α = 0.2, K = 20 | It's still an honest demo ("don't know" next to the uncalibrated rule's confident errors). Show the K = 80 exploratory view, labelled. No re-tuning of the primary config. |
| Gate 1 fails | Pre-committed fallback sequence (§6.3). |
| H1 is negative | Pre-registered. H2 carries the story ("keeps its promise; tells you which images are hard"). |
| The rules forbid AI-written code or require a tool | Read on day 1. Adjust the disclosure, add the required tool, or reconsider. |
| SAM 3.1 masks are poor on some CC0 photos | Fall back to the plain tile for those concepts and log the count. |
| Name collision | Resolved: the project was renamed from "SureSight" (a Welch Allyn vision screener) to SureEnough. |
| The Official Rules contain a surprise (video length, required links, a disclosure format) | Dhruvil pastes the rules from his Cerebral Valley dashboard when he can, and the submission is adjusted on Nov 3. A 2–3 minute video plus a public repo and a hosted demo already covers the likely requirements. |

## 13. Left to the builder
- Hyperparameters, the λ grid resolution, candidate PE layers and pooling, and exact composite rendering.
- The selection of curated examples (subject to the inclusion rules).
- UI styling, chart library and file formats.
- The paper template and venue, and figure designs.
- Module names.

## 14. Items only Dhruvil could resolve (resolved Oct 30 unless marked)
1. ~~**Country:** which country he'll call the Meta API from. This decides Muse Image.~~ **Answered Oct 27:** Dhruvil is physically in the **USA** (US Eastern time). The policy restricts *accessing Muse Image from India*, so using it from the US is within the policy text. His Meta Model API dashboard bills in **₹ (INR)**, probably because the account's country is India; the policy does not say whether access is enforced by location or by account country. **Test:** Dhruvil generates one image in the dashboard's **Image** playground from the US, with no VPN. If it works, the Muse Image layer is on (§6.9). If it is blocked, we accept the block and stay on SAM composites. The public demo shows only pre-made images and never calls Muse Image live, so it doesn't make the model available to viewers in India.
2. **Official Rules:** *partly resolved Oct 30.* The Chrome extension wasn't connected, and the rules sit behind the login. From the public Meta event page: the required models are Muse Spark, Muse Image and SAM 3.1 via the API, and Meta promotes an agentic coding agent, so AI-written code is in the spirit of the event. **Still to do:** Dhruvil pastes the rules text when convenient. The video is planned at **2:30 or less** to fit the likely 2–3 minute limit.
   - **Builder's call:** with Muse Image named as a core model, the impression layer is promoted from "optional" to **planned**, subject to the first real call succeeding.
3. **API key and budget:** **Credits answered Oct 29:** the account has **$150 of free credits in use**. Planned spend is about $15, or about $40 with Muse Image, and it all comes out of those credits. **Builder's call:** the API client keeps a running cost tally and stops itself at $60, so it never gets near the card. Dhruvil leaves "free credits" switched on. **Key verified Oct 30:** `GET /v1/models` returned 200 and listed muse-image-1.0, sam-3.1 and muse-spark-1.3.
   - **Data-use note:** while free credits are in use, Meta says content may be used for product improvement. That is fine, because only CC0 images and our own text are ever sent (§9.3). THINGS stimuli never are.
4. **Repo (delegated):** the name is **SureEnough**. It keeps the "sure" and says what the system does: it stops once it has seen enough. No product by that name was found. The repo is `DMANDAVIYA/SureEnough` and is **public now**. The approved-spec commit is the pre-registration timestamp, and Pages needs a public repo anyway.
5. **Paper timing (delegated):** draft after Nov 4, from the frozen results.
6. **Intro and Muse Code (delegated):** there's no face or voice intro unless Dhruvil offers one later; the video uses captions plus music-free narration text. Muse Code is not used.

## 15. Key references
LTT 2110.01052 · CRC 2208.02814 · Ringel et al. 2402.00857 · COINS 2609.07112 · time-uniform conformal
2602.06297 · Jazbec et al. 2405.20915 · THINGS-EEG2 (Gifford et al. 2022) · THINGSplus (Stoinski et al.,
PMC10991023) · ATM 2403.07721 · Perceptogram 2404.01250 · NEAR 2608.19128 · SATTC 2603.20738 · Shallow
Alignment 2601.21948 · Benchetrit, Banville & King 2310.19812 · Perception Encoder 2504.13181 · SAM 3
2511.16719 · Meta Geographic Use Policy (dev.meta.ai/legal/geographic-use-policy).
