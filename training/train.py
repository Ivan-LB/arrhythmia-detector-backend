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
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from ecg_pipeline import features as features_module
from ecg_pipeline.labels import AAMI_CLASSES

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = features_module.FEATURE_NAMES
CLASS_TO_INDEX: dict[str, int] = {cls: idx for idx, cls in enumerate(AAMI_CLASSES)}
DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_SMOTE_K_NEIGHBORS = 5


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


def encode_labels(class_series: pd.Series) -> np.ndarray:
    """One-hot encode AAMI class labels in a fixed column order (AAMI_CLASSES),
    not whatever order happens to appear in a given subset -- so train and
    eval always agree on which column is which class.
    """
    indices = class_series.map(CLASS_TO_INDEX)
    if indices.isna().any():
        unknown = sorted(class_series[indices.isna()].unique())
        raise ValueError(f"Unknown AAMI class(es) in data: {unknown}")
    return keras.utils.to_categorical(indices, num_classes=len(AAMI_CLASSES))


def _smote_k_neighbors(y_indices: np.ndarray) -> int:
    """SMOTE's k_neighbors must be smaller than the smallest present class's
    count. The Q class has as few as 7-8 examples in all of DS1, and shrinks
    further once the validation split removes a few records -- the default
    k_neighbors=5 can and does become invalid in practice, not just in theory.
    """
    class_counts = np.bincount(y_indices)
    smallest_class_count = int(class_counts[class_counts > 0].min())
    return max(1, min(DEFAULT_SMOTE_K_NEIGHBORS, smallest_class_count - 1))


def prepare_training_data(
    df: pd.DataFrame, feature_columns: list[str], seed: int
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit the scaler on this data only, then SMOTE-balance it.

    Callers must pass only the training partition -- never validation or
    DS2 -- since both the scaler fit and SMOTE happen here.
    """
    X = df[feature_columns].to_numpy()
    y_indices = df["AAMIClass"].map(CLASS_TO_INDEX).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=seed, k_neighbors=_smote_k_neighbors(y_indices))
    X_resampled, y_resampled_indices = smote.fit_resample(X_scaled, y_indices)
    y_resampled = keras.utils.to_categorical(y_resampled_indices, num_classes=len(AAMI_CLASSES))

    return X_resampled, y_resampled, scaler


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
    except Exception:
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

    feature_columns = list(FEATURE_COLUMNS)
    X_train, y_train, scaler = prepare_training_data(train_df, feature_columns, seed)
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
        "train_rows_before_smote": len(train_df),
        "train_rows_after_smote": int(X_train.shape[0]),
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
