# Neural Network Architecture

Status: **design spec for the rebuild** — describes the target model, contrasted against the current `ModelCreation/ANNModel.py` it replaces. Not yet implemented.

## 1. Task definition

Single-window, single-label classification: given the 9 engineered features from one beat-centered window (see [data-pipeline-architecture.md](data-pipeline-architecture.md) §5), predict one of the 5 AAMI EC57 superclasses (N/S/V/F/Q). One model, not two — the current code's separate rhythm-classification model is deferred (see that doc's §11).

## 2. Input

9 features, `StandardScaler`-normalized (fit on DS1 only): `RPeakCount`, `SpectralEnergy`, `TotalPSD`, `WaveletEnergy`, `ShannonEntropy`, `SignalSTD`, `Skewness`, `Kurtosis`, `Variance`.

## 3. Architecture

Kept close to the current shape — a 3-hidden-layer Dense network with batch norm, dropout, and L1/L2 regularization — since it's a reasonable, already-tuned starting point for a 9-feature tabular input, not a raw-signal problem that would call for a CNN/RNN:

```
Input(9,)
  → Dense(64, activation='relu', kernel_regularizer=l1(0.0005))
  → BatchNormalization()
  → Dropout(0.4)
  → Dense(128, activation='relu', kernel_regularizer=l2(0.0005))
  → BatchNormalization()
  → Dropout(0.3)
  → Dense(64, activation='relu', kernel_regularizer=l2(0.0001))
  → Dense(5, activation='softmax')   # <- 5, matching the AAMI classes exactly
```

**The output layer is the one required change.** The current code's `Dense(y_train_beat.shape[1], ...)` derives its width from `to_categorical()` on the old non-contiguous label encoding (`{N:0, L:1, R:2, A:5, V:9, F:11}`), which produces **12** output columns — most of them permanently empty — while `class_names_beat` lists only 4 names in `ANNModel.py` and 6 in `ANNModel_Prueba.py`. None of the three numbers agree with each other. Under the AAMI mapping the label space is exactly 5 classes, contiguous (0–4), so this mismatch structurally cannot recur.

These layer sizes/dropout rates/regularization strengths are carried over as a starting point, not re-validated from scratch — flag if you'd rather start a hyperparameter sweep from a blank slate instead of the current values.

## 4. Training regime

| Setting | Value | Note |
|---|---|---|
| Loss | `categorical_crossentropy` | unchanged |
| Optimizer | `SGD(learning_rate=0.05, momentum=0.9)` | unchanged — `ANNModel_Prueba.py`'s alternative optimizers (Adamax/RMSprop, left commented out) are dropped along with that whole duplicate script |
| Epochs | 90 | unchanged |
| Batch size | 80 | unchanged |
| Validation split | 0.15, carved out of DS1 **by RecordID (patient), not randomly by beat** | changed — see rationale below |
| Callbacks | `EarlyStopping(monitor='val_loss', patience=10)`, `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)` | unchanged, both reasonable |

The validation carve-out uses `GroupShuffleSplit` grouped by `RecordID`, not a plain random `validation_split=0.15`. A random beat-level validation split would let the same patient's beats appear in both the training-proper and validation partitions — reintroducing the exact patient-identity leakage problem this whole rebuild exists to fix, just one level down at the validation step instead of the train/test step.

## 5. Class imbalance handling

**Changed from the original plan during implementation — not SMOTE.** The original plan (and the original project) used SMOTE to fully balance classes. Running it for real surfaced why that doesn't work at this data's actual scale: the Q class had only **2 real examples** in the training partition (of ~41,848 rows) — SMOTE oversampled that to match the ~37,782-row N class, an **18,891x amplification**, meaning nearly 38,000 "Q" training rows were mathematically interpolated on a single line segment between 2 real points. That's fabricating data to reach a number, not a legitimate imbalance-handling technique (and directly at odds with this project's standing rule against inventing data).

Plain sklearn `"balanced"` class_weight has the same problem in a different shape: with only 2 real Q examples, it computes a weight of ~4185 (vs ~20 for F, the next-rarest class) — confirmed empirically to blow up training (`val_loss` ~16 instead of the ~1.6 random-guessing baseline for 5 classes), since a single Q example in a batch dominates the gradient.

**Final approach**: `class_weight` passed to `model.fit()`, computed via `sklearn.utils.class_weight.compute_class_weight("balanced", ...)`, with the largest weight capped at the second-highest naturally-occurring weight in that run's own distribution — not a memorized constant, so it self-adjusts if the class distribution ever shifts. No oversampling, no fabricated rows at all. See `training/train.py`'s `compute_class_weights()` for the implementation and `docs/progress.md`'s Phase 2 entry for the full empirical trail (SMOTE → uncapped balanced weight → capped weight) that led here.

Reproducibility: `keras.utils.set_random_seed(seed)` covers Python/NumPy/TensorFlow in one call, set once at the top of `train()`.

## 6. Reproducibility

- `keras.utils.set_random_seed(seed)` — one call, covers Python/NumPy/TensorFlow together — set once, explicitly, at the top of `train()`. Absent anywhere in the original codebase.
- The train/test partition is the fixed DS1/DS2 list (see data-pipeline doc §3), not a `random_state`-seeded shuffle — one less source of run-to-run variance by construction. The internal train/validation carve-out within DS1 *is* seeded (`GroupShuffleSplit(random_state=seed)`) since it has to choose which records go where.

## 7. Evaluation protocol — implemented, with real results

Evaluated once, on DS2, after training is finalized on DS1 — never used for tuning decisions.

Reports **per-class Sensitivity, Positive Predictivity, Specificity, and Negative Predictivity**, not just overall accuracy — accuracy alone is misleading here because Normal beats dominate MIT-BIH so heavily that a model could score high accuracy while missing most S/V/F/Q beats entirely, which are the clinically important ones. `training/evaluate.py`'s `calculate_class_metrics()` is a direct, type-hinted port of `ANNModel.py`'s original `calculate_metrics()` function — that part of the original code was already correct and worth keeping as-is.

**Real DS2 result (2026-07-05, model `beat-classifier-20260705-dc9f983`): 63.22% overall accuracy.**

| Class | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|
| N | 64.8% | 73.2% | 95.1% | 20.4% |
| S | 24.5% | 93.7% | 13.1% | 97.0% |
| V | 64.6% | 84.0% | 21.8% | 97.2% |
| F | 61.3% | 87.1% | 3.6% | 99.7% |
| Q | 0.0% | 100.0% | 0.0% | 99.99% |

Honestly far below the original (leaky) pipeline's ~98-99% — that's the point: this reflects true inter-patient generalization, not memorized patient identity. Q's 0% is expected, not a bug: only 2 real training examples exist for it (see §5). Minority-class PPV is weak (S/V/F all under ~22%) — a real precision/recall tradeoff from the class weighting needed to get any recall at all on classes this rare, not a defect. This is a first correct baseline, not a tuned final model — see `docs/progress.md`'s Phase 2 entry for the full context and the comparison against published inter-patient benchmarks researched in Phase 0.

## 8. Model artifact versioning — implemented

Replaces the current unversioned, untraceable naming (`modelo_ecg_beat.h5` vs `modelo_ecg_beatV2.h5`, with no committed script that produced the `V2` files actually loaded by the UI). Implemented scheme:

```
models/
  beat-classifier-{YYYYMMDD}-{short-git-sha}/
    model.keras         # modern Keras 3 native format, not the legacy .h5
    scaler.pkl
    metrics.json          # DS2 confusion matrix + per-class Se/Sp/PPV/NPV, for traceability
    training_config.json  # hyperparameters, feature list, split record IDs, per-class train/val counts, class weights used
```

The API (see [system-design.md](system-design.md) §4) always loads a specific, named version — never "whatever happens to be in the `Models/` folder" — so it's always possible to answer "which training run produced the model currently being served."
