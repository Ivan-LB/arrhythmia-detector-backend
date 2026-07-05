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
| Validation split | 0.15 (carved out of DS1) | unchanged — DS2 is never touched until final evaluation |
| Callbacks | `EarlyStopping(monitor='val_loss', patience=10)`, `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)` | unchanged, both reasonable |

## 5. Class imbalance handling

SMOTE applied to the DS1 training partition **after** scaling (matching the correct half of the current code — `ANNModel.py` gets this right; `ANNModel_Prueba.py` has a live bug here, fitting a single `scaler` object twice on two incompatible feature sets and then resampling the *unscaled* arrays instead of the scaled ones. That whole script is retired, not fixed in place, since it's a near-duplicate of `ANNModel.py` with drifted, partially-broken logic).

`SMOTE(random_state=<fixed>)` — currently unseeded, so the synthetic minority oversampling itself is non-deterministic across runs even when everything else is held constant.

## 6. Reproducibility

- `numpy.random.seed(...)` and `tf.random.set_seed(...)` (or `keras.utils.set_random_seed(...)`) set once, explicitly, at the top of the training script — absent anywhere in the current codebase.
- The train/test partition is now the fixed DS1/DS2 list (see data-pipeline doc §3), not a `random_state`-seeded shuffle — one less source of run-to-run variance by construction.

## 7. Evaluation protocol

Evaluated once, on DS2, after training is finalized on DS1 — never used for tuning decisions.

Report **per-class Sensitivity, Positive Predictivity, and Specificity**, not just overall accuracy — accuracy alone is misleading here because Normal beats dominate MIT-BIH so heavily that a model could score high accuracy while missing most S/V/F beats entirely, which are the clinically important ones. This is standard practice throughout the inter-patient MIT-BIH literature.

Worth keeping as-is: `ANNModel.py` already has a `calculate_metrics()`/`get_all_metrics()` pair that computes exactly Sensitivity/Specificity/PPV/NPV per class from a confusion matrix — that part of the current code is correct and reusable, just needs to be pointed at the DS2-based confusion matrix with the corrected 5-class labels instead of the current buggy split.

Expect S-class sensitivity to be the weakest number — it's the rarest class by a wide margin (see data-pipeline doc §7) and is reported as the hardest class across the published literature, not a sign of a bug.

## 8. Model artifact versioning

Replaces the current unversioned, untraceable naming (`modelo_ecg_beat.h5` vs `modelo_ecg_beatV2.h5`, with no committed script that produced the `V2` files actually loaded by the UI). Proposed scheme:

```
models/
  beat-classifier-{YYYYMMDD}-{short-git-sha}/
    model.h5
    scaler.pkl
    metrics.json        # DS2 confusion matrix + per-class Se/P+/Sp, for traceability
    training_config.json  # hyperparameters, feature list, split identifiers (DS1/DS2)
```

The API (see [system-design.md](system-design.md) §4) always loads a specific, named version — never "whatever happens to be in the `Models/` folder" — so it's always possible to answer "which training run produced the model currently being served."
