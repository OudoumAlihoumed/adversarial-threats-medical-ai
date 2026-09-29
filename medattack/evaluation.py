"""Classification metrics computed from the binary confusion matrix."""

import numpy as np

from . import config


def predict_labels(model, images, batch_size=32):
    """Hard predictions (argmax of the softmax output)."""
    return np.argmax(model.predict(images, batch_size=batch_size, verbose=0), axis=1)


def _safe_div(a, b):
    return a / b if b else float("nan")


def binary_metrics(y_true, y_pred, positive=config.POSITIVE_CLASS):
    """Accuracy, precision, recall, F1, specificity and FPR (``evalModel``).

    "Tumour" (label 1) is the positive class. Because both datasets are
    class-imbalanced, recall and specificity are always reported next to
    accuracy: a model can keep a decent accuracy while collapsing onto the
    majority class. Undefined ratios (e.g. precision with no positive
    predictions) are returned as NaN, shown as "-" in the paper's tables.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == positive) & (y_true == positive)))
    tn = int(np.sum((y_pred != positive) & (y_true != positive)))
    fp = int(np.sum((y_pred == positive) & (y_true != positive)))
    fn = int(np.sum((y_pred != positive) & (y_true == positive)))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if tp else float("nan")
    return {
        "accuracy": 100 * (tp + tn) / (tp + tn + fp + fn),
        "precision": 100 * precision,
        "recall": 100 * recall,
        "f1": 100 * f1,
        "specificity": 100 * _safe_div(tn, tn + fp),
        "fpr": 100 * _safe_div(fp, fp + tn),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def format_metrics(m):
    def f(v):
        return "  -  " if np.isnan(v) else f"{v:6.2f}"
    return (f"acc {f(m['accuracy'])} | prec {f(m['precision'])} | rec {f(m['recall'])} | "
            f"F1 {f(m['f1'])} | spec {f(m['specificity'])} | "
            f"TP {m['tp']} TN {m['tn']} FP {m['fp']} FN {m['fn']}")
