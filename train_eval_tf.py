#!/usr/bin/env python3
"""
Chest-X-Ray Pneumonia vs Normal Classifier
==========================================

This script trains a CNN (DenseNet-121 by default) to separate
NORMAL vs PNEUMONIA images.

Directory layout  (≈ 2 GB dataset)

    chest_xray/
        train/
            NORMAL/       *.jpeg | *.png
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
        val/              # optional — if missing we auto-split 15 %
            NORMAL/
            PNEUMONIA/

Typical usage
-------------

Train (with mixed precision if available)  
    python train_eval_tf.py train

Evaluate on held-out test set  
    python train_eval_tf.py test

Single-image inference (FastAPI or CLI)  
    python train_eval_tf.py predict --image some_xray.jpg
"""

from __future__ import annotations
import os, sys, logging, random
from pathlib import Path

import numpy as np
import tensorflow as tf
import click
from sklearn.metrics import roc_curve
from tensorflow.keras import layers as L, models as M, optimizers as O

# ───────────────────────────── Config & Logging ──────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
RND_SEED = 13            # single source of determinism

# reproducible training
random.seed(RND_SEED)
np.random.seed(RND_SEED)
tf.keras.utils.set_random_seed(RND_SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ────────────────────────── Utility: mixed precision ─────────────────────────
def maybe_enable_mixed_precision(enable: bool):
    if not enable:
        return
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("✅  Mixed precision enabled")
    except ValueError:
        log.warning("⚠️   Mixed precision unavailable on this device")

# ───────────────────────────── Dataset pipeline ──────────────────────────────
def _load_directory(directory: Path, batch_size: int, shuffle: bool):
    """Load a directory as a tf.data.Dataset and keep its `class_names` attr."""
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        class_names=['NORMAL', 'PNEUMONIA'],
        label_mode="binary",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=shuffle,
        seed=RND_SEED,
    )
    class_names = ds.class_names             # capture before .prefetch
    ds = ds.prefetch(AUTOTUNE)
    ds.class_names = class_names             # re‑attach attribute
    return ds


def make_datasets(base_dir: Path, batch_size: int):
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    val_dir = base_dir / "val"

    assert train_dir.exists() and test_dir.exists(), (
        "Expecting train/ and test/ folders under the dataset root."
    )

    if val_dir.exists():
        train_ds = _load_directory(train_dir, batch_size, shuffle=True)
        val_ds = _load_directory(val_dir, batch_size, shuffle=False)
    else:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            validation_split=0.15,
            subset="training",
            seed=RND_SEED,
            label_mode="binary",
            class_names=['NORMAL', 'PNEUMONIA'],
            batch_size=batch_size,
            image_size=IMG_SIZE,
            shuffle=True,
        ).prefetch(AUTOTUNE)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            validation_split=0.15,
            subset="validation",
            seed=RND_SEED,
            label_mode="binary",
            class_names=['NORMAL', 'PNEUMONIA'],
            batch_size=batch_size,
            image_size=IMG_SIZE,
            shuffle=False,
        ).prefetch(AUTOTUNE)

    test_ds = _load_directory(test_dir, batch_size, shuffle=False)
    log.info("Class mapping: %s", train_ds.class_names)
    return train_ds, val_ds, test_ds


# ─────────────────────────────── Model builder ───────────────────────────────
def build_model(backbone_name: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = L.Rescaling(1.0 / 255)(inputs)
    x = L.RandomFlip("horizontal")(x)
    x = L.RandomRotation(0.15)(x)
    x = L.RandomZoom(0.20)(x)

    if backbone_name == "densenet121":
        base = tf.keras.applications.DenseNet121(
            weights="imagenet", include_top=False, input_tensor=x
        )
    elif backbone_name == "densenet169":
        base = tf.keras.applications.DenseNet169(
            weights="imagenet", include_top=False, input_tensor=x
        )
    elif backbone_name == "efficientnetb3":
        base = tf.keras.applications.EfficientNetB3(
            weights="imagenet", include_top=False, input_tensor=x
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # fine-tune last N layers
    for layer in base.layers[:-50]:
        layer.trainable = False

    x = L.GlobalAveragePooling2D()(base.output)
    x = L.BatchNormalization()(x)
    x = L.Dense(
        256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = L.Dropout(0.30)(x)
    outputs = L.Dense(1, activation="sigmoid", dtype="float32")(x)
    return M.Model(inputs, outputs)


# ────────────────────────────── Training loop ────────────────────────────────
def train(data_dir: str, epochs: int, batch_size: int, backbone: str, mixed_prec: bool):
    maybe_enable_mixed_precision(mixed_prec)
    base = Path(data_dir)
    train_ds, val_ds, test_ds = make_datasets(base, batch_size)

    # ----- class-balanced weights (0=NORMAL, 1=PNEUMONIA) -----
    y_train = np.concatenate([y.numpy() for _, y in train_ds]).ravel()
    counts = np.bincount(y_train.astype(int), minlength=2)
    class_weight = {0: len(y_train) / (2.0 * counts[0]),
                    1: len(y_train) / (2.0 * counts[1])}
    log.info("Class weights → %s", class_weight)

    model = build_model(backbone)
    model.compile(
        optimizer=O.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.2, patience=1, verbose=1
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cbs,
        class_weight=class_weight,
    )
   
    # ── sanity‑check on validation set ──────────────────────────────
    from sklearn.metrics import confusion_matrix, classification_report
    y_val_true, y_val_pred = [], []
    for xb, yb in val_ds:
        probs = model.predict(xb, verbose=0).ravel()
        y_val_true.extend(yb.numpy().astype(int))
        y_val_pred.extend((probs >= 0.5).astype(int))   # fixed 0.5 threshold
    cm_val = confusion_matrix(y_val_true, y_val_pred)
    log.info(f"VAL confusion @0.5 [[TN FP]\n [FN TP]]\n{cm_val}")
    log.info(classification_report(
        y_val_true, y_val_pred,
        target_names=['NORMAL', 'PNEUMONIA'],
        digits=3))

    # ----- threshold calibration on the validation set -----
    y_val, p_val = [], []
    for xb, yb in val_ds:
        p_val.extend(model.predict(xb, verbose=0).ravel())
        y_val.extend(yb.numpy())
    fpr, tpr, thr = roc_curve(y_val, p_val)
    best_thr = float(thr[(tpr - fpr).argmax()])
    best_thr = max(1e-3, min(best_thr, 0.999))
    np.save("best_threshold.npy", best_thr)
    log.info("Best threshold saved → %.3f", best_thr)

    log.info("Evaluation on test set:")
    model.evaluate(test_ds, verbose=2)

    model.save("chest_xray_best.keras")
    log.info("Model saved ➜ chest_xray_best.keras")


# ────────────────────────────── Evaluation loop ───────────────────────────────
def evaluate(
    data_dir: str,
    model_path: str,
    batch_size: int,
    threshold: float | None = None,
):
    """Evaluate MODEL on data_dir/test and print metrics + confusion matrix."""
    test_dir = Path(data_dir) / "test"
    assert test_dir.exists(), "Expecting a test/ folder."
    test_ds = _load_directory(test_dir, batch_size, shuffle=False)

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    results = model.evaluate(test_ds, verbose=2, return_dict=True)
    log.info("Test metrics →  " + ", ".join(f"{k}: {v:.4f}" for k, v in results.items()))

    if threshold is None:
        threshold = float(np.load("best_threshold.npy"))
        log.info("Loaded best_threshold.npy → %.3f", threshold)

    y_true, y_pred = [], []
    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0).ravel()
        y_true.extend(yb.numpy().astype(int))
        y_pred.extend((probs >= threshold).astype(int))
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()
    log.info(
        f"Confusion matrix @ thr={threshold:.2f}\n[[TN FP]\n [FN TP]] =\n{cm}"
    )


# ───────────────────────────── Inference helper ──────────────────────────────
def load_and_predict(model_path: str, image_path: str) -> float:
    model = tf.keras.models.load_model(model_path, compile=False)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)[None, ...]
    arr = tf.cast(arr, tf.float32)  # Rescaling layer handles /255
    return float(model.predict(arr, verbose=0)[0][0])  # prob of PNEUMONIA


# ──────────────────────────────── CLI section ────────────────────────────────
@click.group()
def cli():
    """Chest-Xray training & inference utility."""
    pass


@cli.command("train")
@click.option("--data-dir", default="chest_xray", help="Dataset root")
@click.option("--epochs", default=25, help="Max epochs")
@click.option("--batch-size", default=32, help="Batch size")
@click.option(
    "--backbone",
    type=click.Choice(["densenet121", "densenet169", "efficientnetb3"]),
    default="densenet121",
)
@click.option("--no-mp", is_flag=True, help="Disable mixed precision")
def cli_train(data_dir, epochs, batch_size, backbone, no_mp):
    train(data_dir, epochs, batch_size, backbone, mixed_prec=not no_mp)


@cli.command("predict")
@click.option("--model", default="chest_xray_best.keras", help="Saved .keras model")
@click.option("--image", required=True, help="Image file to classify")
@click.option("--threshold", default=None, type=float, help="Decision threshold")
def cli_predict(model, image, threshold):
    prob = load_and_predict(model, image)
    if threshold is None:
        threshold = float(np.load("best_threshold.npy"))
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    print(f"{label}  (prob={prob:.3f}, thr={threshold:.2f})")


@cli.command("test")
@click.option("--data-dir", default="chest_xray", help="Dataset root with test/")
@click.option("--model", default="chest_xray_best.keras", help="Saved .keras model")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--threshold", default=None, type=float, help="Decision threshold")
def cli_test(data_dir, model, batch_size, threshold):
    evaluate(data_dir, model, batch_size, threshold)


if __name__ == "__main__":
    cli()