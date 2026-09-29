"""Figures produced for every model: confusion matrices, ROC curves and
before/after attack examples."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics as skm

from . import config


def confusion_matrix(y_true, y_pred, title, save_path: Path):
    cm = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(pd.DataFrame(cm), annot=True, fmt="g", cmap="GnBu", cbar=False,
                     xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def roc_curves(y_true, predictions_by_eps: dict, title, save_path: Path):
    """One ROC curve per perturbation budget (eps 0.00 = clean images).

    As in the paper, curves are computed from hard predicted labels.
    """
    plt.figure(figsize=(5, 5))
    for eps, y_pred in predictions_by_eps.items():
        fpr, tpr, _ = skm.roc_curve(y_true, y_pred)
        try:
            auc = skm.roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float("nan")
        plt.plot(fpr, tpr, label=f"ε: {eps:.2f} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def attack_example(model, clean, adversarial, true_label, attack, eps, save_path: Path,
                   pattern=None):
    """Three panels: clean image, perturbation pattern, adversarial image.

    ``clean`` and ``adversarial`` are single images (H, W, 3) in [0, 1].
    If ``pattern`` is not given, the min-max scaled difference is shown.
    """
    probs = model.predict(np.stack([clean, adversarial]), verbose=0)
    before, after = probs[0], probs[1]
    if pattern is None:
        diff = adversarial - clean
        pattern = (diff - diff.min()) / (diff.max() - diff.min() + 1e-12)

    names = config.CLASS_NAMES
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(clean)
    ax[0].set_title(f"Before {attack}\nTrue: {names[true_label]}\n"
                    f"Pred: {names[before.argmax()]}  p={before.max():.3f}")
    ax[1].imshow(pattern, cmap="gray")
    ax[1].set_title(f"Perturbation pattern\nε: {eps}")
    ax[2].imshow(adversarial)
    ax[2].set_title(f"After {attack}\nTrue: {names[true_label]}\n"
                    f"Pred: {names[after.argmax()]}  p={after.max():.3f}")
    for a in ax:
        a.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
