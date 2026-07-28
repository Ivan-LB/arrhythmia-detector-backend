# Portfolio & CV Brief — Arrhythmia Detector

**Purpose.** This file is the canonical, fact-checked source for describing this project on Iván's CV (`~/Projects/CV/ivanCV.yaml`) and portfolio (`atelier-belli-portfolio`, bilingual en/es). It is written to be consumed by another agent.

**Rule for whoever uses this file:** every number below was read out of the actual artifacts in this repo, not recalled. If you need a figure that isn't here, measure it — do not estimate one. This project's whole differentiator is methodological honesty, so overselling it in the copy would undercut the thing that makes it worth showing.

Last verified: 2026-07-27, against tag `v2.0` (`daf2ade`).

---

## 1. What the project is

An ECG heartbeat classifier. It takes a WFDB-format ECG recording, finds each heartbeat, extracts engineered features from a window around it (FFT/PSD, wavelet, statistical), and classifies that beat into one of the 5 AAMI EC57 superclasses (N/S/V/F/Q) with a confidence score. A FastAPI service exposes this; a Next.js web app consumes it and lets you inspect the whole record and drill into any flagged beat.

**It is a from-scratch rebuild of Iván's 2023 degree project.** That framing matters — see §3.

## 2. The angle (lead with this)

Most ML portfolio projects report a high accuracy number. **This one's headline result is that the accuracy went down, on purpose, and that's the achievement.**

The original degree project reported ~98–99% accuracy. The rebuild reports **63.22%**. The drop is the finding: the original number came from a random beat-level train/test split, which puts beats from the *same patient* on both sides. A model can score extremely well on that by learning patient identity rather than pathology, and the result doesn't transfer to a new patient — which is the only thing that matters clinically.

The rebuild switched to the **inter-patient DS1/DS2 protocol** (de Chazal et al., 2004), where the test set is entirely held-out patients. 63.22% is what the same modelling approach is actually worth on someone it has never seen.

Two more decisions in the same vein, both worth telling:

- **SMOTE was tried and rejected.** The rarest class (Q) had **2 real training examples**. SMOTE oversampled that to match the majority class — roughly 38,000 synthetic rows interpolated along the single line segment between two real points, an ~18,900× amplification. That is fabricating data to hit a number. It was removed in favour of capped class weights.
- **The first replacement was also rejected.** scikit-learn's `class_weight="balanced"` assigned Q a weight of ~4185 vs ~20 for the next-rarest class, and training genuinely destabilised (`val_loss` ≈ 16 against a ≈1.6 random-guess baseline for 5 classes). The shipped fix caps the largest weight at the second-highest naturally-occurring one, derived from each run's own distribution rather than hardcoded.

**The one-sentence version:** *Rebuilt a degree project's ECG classifier after finding its ~98% accuracy came from patient-identity leakage; the honest inter-patient number is 63.22%, and shipping that instead was the point.*

## 3. Verified facts

Everything here is checkable in the repo. Source noted where non-obvious.

### Result
| Metric | Value | Source |
|---|---|---|
| Overall accuracy (DS2, held-out patients) | **63.22%** | `models/beat-classifier-*/metrics.json` |
| DS2 rows evaluated | 49,687 | same |
| DS1 training rows | 50,995 (41,848 after the internal validation split) | `training_config.json` |
| Train / validation records | 18 / 4 patients | same |
| Epochs | 12 run of 90 requested (early stopping) | same |
| Seed | 42 (deterministic) | same |

Per-class on DS2 — **report these together, never just the good ones**:

| Class | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|
| N (normal) | 64.8% | 73.2% | 95.1% | 20.4% |
| S (supraventricular ectopic) | 24.5% | 93.7% | 13.1% | 97.0% |
| V (ventricular ectopic) | 64.6% | 84.0% | 21.8% | 97.2% |
| F (fusion) | 61.3% | 87.1% | 3.6% | 99.7% |
| Q (unknown) | 0.0% | 100.0% | 0.0% | 100.0% |

Q is 0% because **only 2 Q examples exist in the training partition**. That is a dataset property, not a bug, and it's disclosed rather than hidden.

Minority-class precision is weak (lots of false alarms) — the direct, expected cost of weighting classes hard enough to get any recall at all on rare ones. **This is a first correct baseline, not a tuned model.** Say so.

### Scale & engineering
| | Backend | Frontend |
|---|---|---|
| Repo | [arrhythmia-detector-backend](https://github.com/Ivan-LB/arrhythmia-detector-backend) (public) | [arrhythmia-detector-web](https://github.com/Ivan-LB/arrhythmia-detector-web) (public) |
| Language | Python 3.11+ | TypeScript |
| Source LOC | 1,713 | 1,467 |
| Test LOC | 1,852 | 451 |
| Tests | **168**, **94% coverage** | **57** |
| CI | GitHub Actions on push/PR | — |

Backend stack: FastAPI, TensorFlow/Keras, scikit-learn, NumPy, SciPy, PyWavelets, `wfdb`, pandas — all exact-pinned.
Frontend stack: Next.js 16 (App Router), React 19, Tailwind CSS 4, TanStack Query, Vitest + React Testing Library, Playwright.

Note the test-to-source ratio: on the backend there is **more test code than source code**. That's a legitimate point to make about the engineering standard, alongside TDD (test-first, confirm red, implement, confirm green) on every phase and an independent code-review pass before every PR.

### Dataset
MIT-BIH Arrhythmia Database (PhysioNet `mitdb`), 44 records under the DS1/DS2 split. During the rebuild it emerged that **19 of 20 required 100-series records had no `.atr` annotation files committed at all**, and one was missing its header — re-fetched clean from PhysioNet. Good detail: it's the kind of data-integrity check that only happens if you actually verify your inputs.

### Timeline
- **v1.0** — original degree project, tagged and frozen: **2023-12-09**
- **v2.0** — rebuild, tagged on `main`: work spanning **2026-07-05 → 2026-07-25**
- **v2.1** — in progress, scoped but not implemented: a proposed beat-*detection* model (Pan-Tompkins baseline vs. a learned detector, compared head-to-head) plus beat-resolution UI. **Do not describe v2.1 as done.**

### Architecture (one line)
`ecg_pipeline/` is a single shared package for preprocessing and feature extraction, imported by *both* the offline training pipeline and the live API — fixing the original project, where that logic was copy-pasted between the training script and the desktop UI and had already drifted.

## 4. Guardrails — do not write these

Getting these wrong would be worse than not featuring the project.

| ❌ Don't | ✅ Instead |
|---|---|
| Quote ~98–99% accuracy as a result | That's the **discredited leaky number**. Only ever mention it as the thing that was wrong. |
| "Only 63%" / apologise for it | 63.22% is the *honest inter-patient* figure. Frame it as the correct measurement, with the leaky number as the contrast. |
| Any diagnostic or clinical claim | It's a research/portfolio project. Not a medical device, not clinically validated, not for clinical use. |
| "Detects arrhythmias" / "diagnoses arrhythmia" | **Per-beat AAMI classification only.** Rhythm classification is explicitly out of scope — it's a different task needing sequence context. |
| "Real-time" | It isn't. Inference is per-uploaded-record, ~2.3s cold then cached. |
| Hide the Q=0% or the weak PPVs | Disclose with the reason. The disclosure *is* the selling point. |
| Imply the ML is novel research | The modelling approach is deliberately conventional. **The methodology and engineering rigour are the contribution**, not a new architecture. |
| Call v2.1 shipped | It's scoped only. |

Also: the ~98–99% figure for v1 is what the original project reported and is recorded in `docs/progress.md`; it was not independently re-measured during the rebuild. Treat it as approximate and attribute it as "the original pipeline reported", not as a benchmark you measured.

## 5. Ready-to-use copy

### CV bullets

The current entry in `ivanCV.v2.yaml` is solid but buries the lede — it mentions the leakage fix in the second half of bullet 2. Suggested replacement, leading with the judgment call:

```yaml
- name: "[Arrhythmia Detector](https://github.com/Ivan-LB/arrhythmia-detector-backend): ECG heartbeat classifier (ML)"
  date: "2023, rebuilt 2026"
  highlights:
    - "Rebuilt a degree project's ECG classifier after tracing its ~98% reported accuracy to patient-identity leakage from a random beat-level split; re-evaluated under the inter-patient DS1/DS2 protocol (de Chazal 2004) for an honest 63.22% held-out baseline, and shipped the lower, correct number with full per-class Se/Sp/PPV/NPV."
    - "Caught and removed a SMOTE step that was synthesising ~38,000 rows for a class with 2 real training examples, replacing it with data-derived capped class weights after scikit-learn's balanced weighting destabilised training."
    - "Windowed feature-engineering pipeline (FFT/PSD, wavelet, statistical) into a Keras classifier over the 5 AAMI EC57 superclasses, served by FastAPI, with a shared package so training and inference cannot drift; 168 tests at 94% coverage, TDD throughout, CI on every push."
    - "Next.js 16 / TypeScript frontend to upload WFDB records and inspect the full trace with per-beat classification and confidence; 57 tests."
```

Shorter, if space is tight (2 bullets):

```yaml
    - "Rebuilt a degree project's ECG heartbeat classifier after tracing its ~98% accuracy to patient-identity leakage; re-evaluated under the inter-patient DS1/DS2 protocol for an honest 63.22% held-out baseline, and removed a SMOTE step that was fabricating ~38,000 rows from 2 real examples."
    - "FFT/wavelet feature pipeline into a Keras model over the 5 AAMI EC57 classes, served by FastAPI with a Next.js frontend for per-beat inspection; 168 backend tests at 94% coverage, TDD and independent code review on every phase."
```

### Portfolio — schema-matched replacement

The portfolio already has an `arrhythmia` case in `messages/{en,es}.json` under `home.cases`, and `arrhythmia` is already in `CASE_KEYS` in `app/[locale]/page.tsx`. **The existing copy is accurate but frames the project as "a web front end for a classifier"** — it sells the UI and omits the methodology story, which is the strongest thing here. Suggested upgrade, same keys:

**en:**
```json
{
  "titleIt": "read the ECG, beat by beat.",
  "descRich": "An ECG heartbeat classifier, rebuilt from my degree project after I traced its ~98% accuracy to patient-identity leakage. Re-evaluated on fully held-out patients, the honest number is <it>63.22%</it> — and shipping that instead was the point. Engineered features (FFT/PSD, wavelet, statistical) feed a model over the 5 AAMI classes, behind a <it>FastAPI</it> service with 168 tests at 94% coverage. The Next.js front end renders a whole 30-minute record at once and drills into any flagged beat.",
  "metaPlatform": "Web · Responsive",
  "metaStatus": "Working demo",
  "actionPrimary": "View the ML backend",
  "tag": "An honest 63% beats a leaky 98%."
}
```

**es:**
```json
{
  "titleIt": "lee el ECG, latido a latido.",
  "descRich": "Un clasificador de latidos en ECG, reconstruido desde mi proyecto de titulación tras rastrear su ~98% de exactitud a una fuga de identidad del paciente. Reevaluado con pacientes completamente fuera de entrenamiento, el número honesto es <it>63.22%</it> — y publicar ese fue justamente el punto. Features de ingeniería (FFT/PSD, wavelet, estadísticos) alimentan un modelo sobre las 5 clases AAMI, detrás de un servicio <it>FastAPI</it> con 168 pruebas y 94% de cobertura. El front en Next.js renderiza un registro completo de 30 minutos de una vez y permite entrar a cualquier latido marcado.",
  "metaPlatform": "Web · Responsiva",
  "metaStatus": "Demo funcional",
  "actionPrimary": "Ver el backend ML",
  "tag": "Un 63% honesto vale más que un 98% con fuga."
}
```

**Also fix in `app/[locale]/page.tsx`** — current values and what's wrong with them:

| Field | Currently | Suggested | Why |
|---|---|---|---|
| `meta` stack row | `"Next.js · TypeScript · FastAPI · TanStack Query"` | `"TensorFlow · FastAPI · Next.js · TypeScript"` | The current line is **entirely frontend** — an ML case study that never names its ML stack. TensorFlow/Keras and scikit-learn do the actual work. |
| `meta` year row | `"2026"` | `"2023, rebuilt 2026"` | The two-date form *is* the story; a single 2026 hides that this was a rebuild. |
| `stack` (index list) | `["Next.js", "FastAPI", "ML"]` | `["TensorFlow", "FastAPI", "Next.js"]` | `"ML"` as a stack chip is vague; naming the framework is stronger and truer. |
| `mshow` | `"Web · ML · FastAPI →"` | `"ML · FastAPI · Next.js →"` | Leads with the discipline rather than the surface. |
| `kicker` | `"Web · ML · 2026"` | `"ML · Web · 2023 → 2026"` | Same reason as the year row. |

`num: "05"`, `preview: "arrhythmia"`, and the primary action (→ the backend repo) are all fine as-is.

### One-liners (various lengths)

- **~10 words:** ECG heartbeat classifier rebuilt with a leakage-free inter-patient evaluation protocol.
- **~25 words:** An ECG heartbeat classifier over the 5 AAMI classes. Rebuilt from a degree project whose ~98% accuracy turned out to be patient-identity leakage; the honest held-out number is 63.22%.
- **Tweet-length:** My degree project scored ~98% on ECG beat classification. It was leaking patient identity through a random split. Rebuilt it with the inter-patient protocol: 63.22%. Shipped the lower number, because it's the real one.

## 6. Assets

Three real screenshots, captured via Playwright against a live backend and MIT-BIH record 230 — no mockups:

- `docs/screenshots/01-upload.png` — upload screen (in the **web** repo)
- `docs/screenshots/02-trace-overview.png` — full 30-min trace + classification breakdown *(best single hero image)*
- `docs/screenshots/03-beat-detail.png` — class filter + 60s context window + beat detail panel

Web repo local path: `~/Projects/Python/arrhythmia-detector-web` (**note: moved out of `~/Projects/React/`** — the map in the global `CLAUDE.md` is stale on this).

There is no public deployment. The demo runs locally against a local backend, so **"Working demo" is accurate but "Live" is not** — don't link a URL that doesn't exist.

## 7. Housekeeping — already done

Both GitHub repo descriptions were wrong and have been fixed (2026-07-27). Recorded here so nobody "fixes" them back:

- **backend** previously read *"System for the detection of Cardiac Arrhythmias using a Sequential ANN model"* — described the pre-rebuild project and implied rhythm detection, which is explicitly out of scope. Now: `ECG heartbeat classifier (AAMI EC57) with a leakage-free inter-patient evaluation protocol — FastAPI + TensorFlow, 168 tests`
- **web** had no description at all. Now: `Next.js frontend for the Arrhythmia Detector API — upload a WFDB ECG record and inspect per-beat AAMI classification`

Still open, if the portfolio agent wants it: neither repo has GitHub topics set (`machine-learning`, `ecg`, `fastapi`, `healthcare`, `signal-processing` would all be honest).

## 8. Deeper reading, if the copy needs more

- `README.md` — setup, architecture diagram, real API request/response examples
- `docs/progress.md` — the dated reasoning trail; **the SMOTE story and the leakage story are both there in full**, and it's the best source if you want a longer narrative
- `docs/neural-network-architecture.md` — model, training, results
- `docs/data-pipeline-architecture.md` — dataset, DS1/DS2 split, AAMI mapping
- `docs/plan.md` — roadmap, including v2.1 scope
