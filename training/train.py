"""Train the AAMI EC57 beat classifier on DS1, with an internal group-aware
validation split -- DS2 is never touched here, only in evaluate.py.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from ecg_pipeline import features as features_module
from ecg_pipeline.labels import AAMI_CLASSES
from training.class_encoding import class_series_to_indices, encode_labels

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = features_module.FEATURE_NAMES
DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.15


def set_seeds(seed: int) -> None:
    keras.utils.set_random_seed(seed)


def split_train_validation(
    df: pd.DataFrame, validation_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware split within DS1, grouped by RecordID.

    Not a random beat-level split: that would let the same patient's beats
    land in both partitions, reintroducing the exact leakage problem this
    rebuild exists to fix, just one level down at the validation step.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["RecordID"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def warn_on_missing_classes(full_df: pd.DataFrame, train_df: pd.DataFrame) -> None:
    """Log a warning if the group-aware split happened to leave the training
    partition with zero examples of a class present in the full dataset.

    This is a real, seed-dependent risk, not hypothetical: on a real run,
    only 2 of DS1's 8 total Q-class rows ended up in the training partition
    (the other 6, across 2 records, landed in validation) purely because of
    which patient-groups the split happened to draw. A different seed could
    just as easily zero Q out of training entirely -- compute_class_weights
    would silently never assign it a weight, and the model would never see
    a single training example of it, with nothing surfacing that fact.
    """
    full_classes = set(full_df["AAMIClass"].unique())
    train_classes = set(train_df["AAMIClass"].unique())
    missing = full_classes - train_classes
    if missing:
        logger.warning(
            "Class(es) %s present in the full dataset but ABSENT from the training "
            "partition after the group-aware split -- the model will never see a "
            "training example of them. Consider a different seed or a split "
            "strategy aware of rare-class record placement.",
            sorted(missing),
        )


def compute_class_weights(y_indices: np.ndarray) -> dict[int, float]:
    """Inverse-frequency class weights for model.fit(class_weight=...),
    with the largest weight capped at the second-largest naturally-occurring
    weight in this data.

    Deliberately NOT SMOTE. An earlier version used SMOTE to fully balance
    classes, but the Q class had as few as 2 real examples in the training
    partition on a real run -- balancing that to match ~38,000 N-class rows
    means ~18,900x oversampling, i.e. nearly 38,000 "Q" training rows
    mathematically interpolated from a single line segment between 2 real
    points. That's fabricating data to reach a number, not a legitimate
    imbalance-handling technique.

    Switching to plain sklearn "balanced" class_weight has the same root
    problem in a different shape: with only 2 real Q examples out of ~41,848
    rows, "balanced" computes a weight of ~4185 for Q (vs ~20 for F, the next
    rarest class) -- confirmed empirically to blow up the loss (val_loss
    around 16, versus ~1.6 for random guessing on 5 classes) because a
    single Q example in a batch dominates the gradient.

    The cap is deliberately computed from this run's own weight
    distribution (the second-highest raw weight), not a memorized constant:
    an earlier version hardcoded 25.0 "because that's just above F's ~20x
    weight" -- reasonable for that one dataset snapshot, but a magic number
    with no way to notice if a future dataset's second-rarest class shifts
    and the constant stops meaning anything. Deriving it at call time makes
    it self-adjusting instead of silently stale.
    """
    present_classes = np.unique(y_indices)
    raw_weights = compute_class_weight("balanced", classes=present_classes, y=y_indices)
    if len(raw_weights) > 2:
        # With 3+ classes, the second-highest weight is a meaningful
        # reference point for "the rest of the distribution." With exactly
        # 2 classes there's no such reference -- capping the larger at the
        # smaller would collapse them to equal weights regardless of how
        # different their frequencies actually are, which is wrong.
        cap = np.sort(raw_weights)[-2]
        capped_weights = np.minimum(raw_weights, cap)
    else:
        capped_weights = raw_weights
    return dict(zip(present_classes.tolist(), capped_weights.tolist()))


def prepare_training_data(
    df: pd.DataFrame, feature_columns: list[str], seed: int
) -> tuple[np.ndarray, np.ndarray, StandardScaler, dict[int, float]]:
    """Fit the scaler on this data only, encode labels, and compute class
    weights -- no oversampling. Callers must pass only the training
    partition -- never validation or DS2 -- since the scaler fit happens
    here.
    """
    X = df[feature_columns].to_numpy()
    y_indices = class_series_to_indices(df["AAMIClass"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_encoded = keras.utils.to_categorical(y_indices, num_classes=len(AAMI_CLASSES))
    class_weights = compute_class_weights(y_indices)

    return X_scaled, y_encoded, scaler, class_weights


def build_model(input_dim: int, num_classes: int) -> keras.Model:
    """Same layer shape as the original project's model (Dense 64/128/64
    with BatchNorm + Dropout + L1/L2 regularization) -- kept close to the
    original approach, just with a correctly-sized 5-class AAMI output
    instead of the previous non-contiguous 12-column encoding.
    """
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1(0.0005))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as error:
        logger.debug("git sha lookup failed, falling back to 'unknown': %s", error)
        return "unknown"


def _version_name() -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"beat-classifier-{date}-{_git_short_sha()}"


def train(
    ds1_csv_path: Path,
    output_dir: Path,
    epochs: int = 90,
    batch_size: int = 80,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Train on DS1 (with an internal group-aware validation carve-out) and
    save a versioned model + scaler + training config. DS2 is untouched here.
    """
    set_seeds(seed)

    df = pd.read_csv(ds1_csv_path)
    train_df, val_df = split_train_validation(df, validation_fraction, seed)
    warn_on_missing_classes(df, train_df)

    feature_columns = list(FEATURE_COLUMNS)
    X_train, y_train, scaler, class_weights = prepare_training_data(train_df, feature_columns, seed)
    X_val = scaler.transform(val_df[feature_columns].to_numpy())
    y_val = encode_labels(val_df["AAMIClass"])

    model = build_model(input_dim=X_train.shape[1], num_classes=len(AAMI_CLASSES))

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    version_dir = output_dir / _version_name()
    version_dir.mkdir(parents=True, exist_ok=True)

    model.save(version_dir / "model.keras")
    joblib.dump(scaler, version_dir / "scaler.pkl")

    training_config = {
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_run": len(history.history["loss"]),
        "batch_size": batch_size,
        "validation_fraction": validation_fraction,
        "feature_columns": feature_columns,
        "aami_classes": list(AAMI_CLASSES),
        "train_records": sorted(train_df["RecordID"].unique().tolist()),
        "validation_records": sorted(val_df["RecordID"].unique().tolist()),
        "train_rows": len(train_df),
        "train_class_counts": train_df["AAMIClass"].value_counts().to_dict(),
        "validation_class_counts": val_df["AAMIClass"].value_counts().to_dict(),
        "class_weights": {AAMI_CLASSES[idx]: weight for idx, weight in class_weights.items()},
    }
    (version_dir / "training_config.json").write_text(json.dumps(training_config, indent=2))

    logger.info("Saved model artifacts to %s", version_dir)
    return version_dir


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ds1_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Data/dataset_ds1.csv")
    models_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models")
    train(ds1_csv, models_dir)
