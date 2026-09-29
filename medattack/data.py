"""Dataset loading, binarisation and the stratified 80/10/10 split."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from . import config
from .config import DatasetConfig


def load_image(path: Path) -> np.ndarray:
    """Read one image as RGB and resize it to 224x224 (uint8, 0-255)."""
    import tensorflow as tf  # imported lazily so the module loads without TensorFlow

    image = tf.keras.preprocessing.image.load_img(
        str(path), color_mode="rgb", target_size=config.IMAGE_SIZE
    )
    return np.array(image)


def build_arrays(ds: DatasetConfig):
    """Walk the raw class folders and return (images in [0, 1], binary labels).

    The class-to-label mapping lives in ``DatasetConfig.class_folders``. For
    Brain MRI the four original classes (glioma, meningioma, pituitary,
    no tumour) are collapsed into Tumour / Normal; for CT Kidney only the
    Normal and Tumor folders are read.
    """
    data, labels = [], []
    for folder, label in ds.class_folders.items():
        files = sorted((ds.raw_dir / folder).glob("*.*"))
        if not files:
            raise FileNotFoundError(f"No images found in {ds.raw_dir / folder}")
        print(f"  {folder:<22} -> label {label}  ({len(files)} images)")
        for path in files:
            data.append(load_image(path))
            labels.append(label)

    data = np.array(data) / 255.0
    labels = np.array(labels)
    data, labels = shuffle(data, labels, random_state=config.SEED)
    return data, labels


def prepare(ds: DatasetConfig):
    """Build the arrays once and cache them as .npy files."""
    ds.out_dir.mkdir(parents=True, exist_ok=True)
    data, labels = build_arrays(ds)
    np.save(ds.data_file, data)
    np.save(ds.labels_file, labels)
    print(f"Saved {data.shape} images and {labels.shape} labels to {ds.out_dir}")
    return data, labels


def load_splits(ds: DatasetConfig):
    """Return (x_train, y_train, x_val, y_val, x_test, y_test).

    80 % train, then the remaining 20 % is halved into validation and test.
    Both splits are stratified by class with seed 42, exactly as in the
    notebooks, so every script sees the same held-out test set.
    """
    if not ds.data_file.exists():
        raise SystemExit(
            f"{ds.data_file} not found - run `python -m scripts.prepare_data "
            f"--dataset {ds.key}` first."
        )
    data = np.load(ds.data_file)
    labels = np.load(ds.labels_file)

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=config.SEED, stratify=labels
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=config.SEED, stratify=y_test
    )
    return x_train, y_train, x_val, y_val, x_test, y_test
