"""The 20 ImageNet backbones and the shared two-phase transfer-learning recipe."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config

apps = tf.keras.applications

# name -> (constructor, matching preprocess_input)
# Every backbone is loaded with include_top=False and ImageNet weights, and
# receives its own official preprocessing function inside the model graph.
BACKBONES = {
    "Xception":          (apps.Xception,          apps.xception.preprocess_input),
    "VGG16":             (apps.VGG16,             apps.vgg16.preprocess_input),
    "VGG19":             (apps.VGG19,             apps.vgg19.preprocess_input),
    "DenseNet121":       (apps.DenseNet121,       apps.densenet.preprocess_input),
    "DenseNet169":       (apps.DenseNet169,       apps.densenet.preprocess_input),
    "DenseNet201":       (apps.DenseNet201,       apps.densenet.preprocess_input),
    "EfficientNetV2B0":  (apps.EfficientNetV2B0,  apps.efficientnet_v2.preprocess_input),
    "EfficientNetV2B1":  (apps.EfficientNetV2B1,  apps.efficientnet_v2.preprocess_input),
    "EfficientNetV2B2":  (apps.EfficientNetV2B2,  apps.efficientnet_v2.preprocess_input),
    "EfficientNetV2B3":  (apps.EfficientNetV2B3,  apps.efficientnet_v2.preprocess_input),
    "EfficientNetV2S":   (apps.EfficientNetV2S,   apps.efficientnet_v2.preprocess_input),
    "InceptionResNetV2": (apps.InceptionResNetV2, apps.inception_resnet_v2.preprocess_input),
    "InceptionV3":       (apps.InceptionV3,       apps.inception_v3.preprocess_input),
    "MobileNet":         (apps.MobileNet,         apps.mobilenet.preprocess_input),
    "MobileNetV2":       (apps.MobileNetV2,       apps.mobilenet_v2.preprocess_input),
    "MobileNetV3Small":  (apps.MobileNetV3Small,  apps.mobilenet_v3.preprocess_input),
    "NASNetMobile":      (apps.NASNetMobile,      apps.nasnet.preprocess_input),
    "ResNet50V2":        (apps.ResNet50V2,        apps.resnet_v2.preprocess_input),
    "ResNet101V2":       (apps.ResNet101V2,       apps.resnet_v2.preprocess_input),
    "ResNet152V2":       (apps.ResNet152V2,       apps.resnet_v2.preprocess_input),
}


def resolve_models(names):
    """Validate a list of backbone names (``None`` or ``["all"]`` means all 20)."""
    if not names or names == ["all"]:
        return list(BACKBONES)
    unknown = [n for n in names if n not in BACKBONES]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Available: {list(BACKBONES)}")
    return names


def build_classifier(name: str):
    """Pretrained backbone + Gaussian noise + preprocessing + Flatten + softmax(2).

    Returns ``(model, base_model)``; the base model is needed to unfreeze it
    for fine-tuning.
    """
    constructor, preprocess_input = BACKBONES[name]
    base_model = constructor(include_top=False, weights="imagenet",
                             input_shape=config.INPUT_SHAPE)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=config.INPUT_SHAPE)
    # Mild augmentation: only active while training, so attacks later run
    # against a deterministic model.
    x = tf.keras.layers.GaussianNoise(config.GAUSSIAN_NOISE_STD)(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=name), base_model


def train(name: str, ds, x_train, y_train, x_val, y_val):
    """Two-phase transfer learning (``model_build_train`` in the notebooks).

    Phase 1 - feature extraction: backbone frozen, Adam(1e-4).
    Phase 2 - fine-tuning: every layer unfrozen, elastic-net (L1 1e-7,
              L2 1e-6) on all Conv2D kernels, Adam(1e-5).
    Both phases use early stopping on validation accuracy (patience 20).
    """
    tf.keras.backend.clear_session()
    gc.collect()

    history_dir = ds.models_dir / name / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    model, base_model = build_classifier(name)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def callbacks():
        return [tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=config.EARLY_STOP_PATIENCE, verbose=1)]

    # ---- Phase 1: train the new head on top of the frozen backbone -------- #
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss=loss, metrics=["accuracy"])
    print(f"[{name}] Phase 1 - training the classifier head (backbone frozen)")
    t0 = time.time()
    history = model.fit(x_train, y_train, epochs=config.MAX_EPOCHS,
                        batch_size=config.TRAIN_BATCH_SIZE,
                        validation_data=(x_val, y_val), callbacks=callbacks())
    phase1_time = time.time() - t0
    np.save(history_dir / "initial_history.npy", history.history)

    # ---- Phase 2: unfreeze everything and fine-tune with elastic-net ------ #
    base_model.trainable = True
    print(f"[{name}] Phase 2 - fine-tuning all {len(base_model.layers)} backbone layers")
    elastic_net = tf.keras.regularizers.L1L2(l1=config.L1_REG, l2=config.L2_REG)
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            base_model.add_loss(lambda layer=layer: elastic_net(layer.kernel))

    model.compile(optimizer=tf.keras.optimizers.Adam(config.FINE_TUNE_LEARNING_RATE),
                  loss=loss, metrics=["accuracy"])
    initial_epochs = history.epoch[-1] + 1
    t0 = time.time()
    ft_history = model.fit(x_train, y_train,
                           epochs=initial_epochs + config.MAX_EPOCHS,
                           initial_epoch=initial_epochs,
                           batch_size=config.TRAIN_BATCH_SIZE,
                           validation_data=(x_val, y_val), callbacks=callbacks())
    phase2_time = time.time() - t0
    np.save(history_dir / "ft_history.npy", ft_history.history)

    model.save(ds.models_dir / f"{name}.h5")

    summary = {
        "model": name,
        "backbone_parameters": int(base_model.count_params()),
        "backbone_layers": len(base_model.layers),
        "total_parameters": int(model.count_params()),
        "phase1_epochs": len(history.epoch),
        "phase1_seconds": round(phase1_time, 2),
        "phase2_last_epoch": ft_history.epoch[-1] + 1,
        "phase2_seconds": round(phase2_time, 2),
        "total_seconds": round(phase1_time + phase2_time, 2),
    }
    (history_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{name}] done in {summary['total_seconds']} s")
    return model, summary


def load(name: str, ds):
    """Load a fine-tuned model saved by :func:`train`."""
    path = ds.models_dir / f"{name}.h5"
    if not path.exists():
        raise SystemExit(f"{path} not found - train it first with scripts.train")
    return tf.keras.models.load_model(path)


def plot_training_history(name: str, ds, save_path: Path):
    """Training vs. validation accuracy with the start of fine-tuning marked."""
    import matplotlib.pyplot as plt

    history_dir = ds.models_dir / name / "history"
    history = np.load(history_dir / "initial_history.npy", allow_pickle=True).item()
    ft = np.load(history_dir / "ft_history.npy", allow_pickle=True).item()
    initial_epochs = len(history["accuracy"])
    acc = history["accuracy"] + ft["accuracy"]
    val_acc = history["val_accuracy"] + ft["val_accuracy"]

    plt.figure(figsize=(10, 5))
    plt.plot(acc, label="Training accuracy")
    plt.plot(val_acc, label="Validation accuracy")
    plt.axvline(initial_epochs, color="grey", linestyle="--", label="Start of fine-tuning")
    plt.ylim([min(plt.ylim()), 1])
    plt.title(f"{name} - training curve")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
