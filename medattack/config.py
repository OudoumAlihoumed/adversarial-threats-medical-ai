"""Experiment configuration shared by every script.

All values mirror the settings used in the original Google Colab notebooks,
so re-running the pipeline reproduces the experimental protocol of the paper.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Everything is resolved relative to MEDATTACK_ROOT (default: the repository
# root). On Google Colab, point it at your Drive folder, e.g.
#   export MEDATTACK_ROOT=/content/drive/MyDrive/TEZ
ROOT = Path(os.environ.get("MEDATTACK_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# --------------------------------------------------------------------------- #
# Global experiment settings
# --------------------------------------------------------------------------- #
SEED = 42
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Tumour"]   # label 0, label 1
POSITIVE_CLASS = 1                   # metrics are computed with "Tumour" as positive

# --------------------------------------------------------------------------- #
# Training (two-phase transfer learning, identical for all 20 backbones)
# --------------------------------------------------------------------------- #
TRAIN_BATCH_SIZE = 32        # Keras default: the notebooks called model.fit() without batch_size
MAX_EPOCHS = 999             # upper bound, early stopping decides the real length
EARLY_STOP_PATIENCE = 20     # EarlyStopping(monitor="val_accuracy", patience=20)
LEARNING_RATE = 1e-4         # phase 1 - frozen backbone
FINE_TUNE_LEARNING_RATE = 1e-5   # phase 2 - all layers trainable
GAUSSIAN_NOISE_STD = 0.01    # training-time augmentation layer
L1_REG = 1e-7                # elastic-net penalty on Conv2D kernels while fine-tuning
L2_REG = 1e-6

# --------------------------------------------------------------------------- #
# Attacks  (epsilon and step size are in [0, 1] normalised intensity units)
# --------------------------------------------------------------------------- #
EPSILONS = (0.02, 0.05, 0.10)

# PGD: (epsilon, step size alpha, iterations)
PGD_SETTINGS = (
    (0.02, 0.010, 5),
    (0.05, 0.025, 5),
    (0.10, 0.050, 5),
)


@dataclass(frozen=True)
class DatasetConfig:
    """Everything that differs between the two imaging modalities."""

    key: str
    display_name: str
    raw_dir: Path
    # sub-folder (relative to raw_dir) -> binary label
    class_folders: dict = field(default_factory=dict)
    pgd_batch_size: int = 16         # PGD runs in mini-batches because of GPU memory
    example_index: int = 9           # test image used for the qualitative figures

    @property
    def out_dir(self) -> Path:
        return OUTPUT_DIR / self.key

    @property
    def data_file(self) -> Path:
        return self.out_dir / "data.npy"

    @property
    def labels_file(self) -> Path:
        return self.out_dir / "labels.npy"

    @property
    def models_dir(self) -> Path:
        return self.out_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.out_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.out_dir / "figures"


DATASETS = {
    # Brain Tumour MRI (Kaggle: masoudnickparvar/brain-tumor-mri-dataset)
    # The four original classes are collapsed into Normal (0) / Tumour (1).
    "brain_mri": DatasetConfig(
        key="brain_mri",
        display_name="Brain Tumour MRI",
        raw_dir=DATA_DIR / "brain_mri",
        class_folders={
            "Testing/notumor": 0,
            "Training/notumor": 0,
            "Testing/glioma": 1,
            "Training/glioma": 1,
            "Testing/meningioma": 1,
            "Training/meningioma": 1,
            "Testing/pituitary": 1,
            "Training/pituitary": 1,
        },
        pgd_batch_size=13,
        example_index=9,
    ),
    # CT Kidney (Kaggle: nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
    # Only the Normal and Tumor folders are used; Cyst and Stone are discarded.
    "kidney_ct": DatasetConfig(
        key="kidney_ct",
        display_name="CT Kidney",
        raw_dir=DATA_DIR / "kidney_ct",
        class_folders={
            "Normal": 0,
            "Tumor": 1,
        },
        pgd_batch_size=23,
        example_index=20,
    ),
}


def get_dataset(key: str) -> DatasetConfig:
    try:
        return DATASETS[key]
    except KeyError:
        raise SystemExit(f"Unknown dataset '{key}'. Choose one of: {', '.join(DATASETS)}")
