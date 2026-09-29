"""White-box L-infinity attacks: FGSM and PGD.

Both attacks follow the implementation used to produce the paper's results.
Images are float arrays in [0, 1]; the models output softmax probabilities,
so the loss is sparse categorical cross-entropy on probabilities.
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _loss_gradient(model, images, labels):
    """Gradient of the classification loss with respect to the input pixels."""
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(images)
        probs = model(images, training=False)
        # Summed per-sample loss: each image's gradient is the same as if it
        # were attacked on its own (the notebooks used batches of one).
        loss = tf.reduce_sum(
            tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
    return tape.gradient(loss, images)


# --------------------------------------------------------------------------- #
# FGSM - Fast Gradient Sign Method (Goodfellow et al., 2015)
# --------------------------------------------------------------------------- #
def fgsm_pattern(model, images, labels):
    """Perturbation pattern used by FGSM.

    As in the original notebooks, the sign of the gradient is clipped to
    [0, 1] before it is scaled by epsilon (``generate_adverarial_pattern`` +
    ``tf.clip_by_value(adv_pattern, 0., 1.)``).
    """
    signed_grad = tf.sign(_loss_gradient(model, images, labels))
    return tf.clip_by_value(signed_grad, 0.0, 1.0)


def fgsm(model, images, labels, eps):
    """One-step FGSM: x* = clip_[0,1](x + eps * pattern)."""
    adv = tf.convert_to_tensor(images, dtype=tf.float32) + eps * fgsm_pattern(model, images, labels)
    return tf.clip_by_value(adv, 0.0, 1.0).numpy()


def fgsm_dataset(model, x, y, eps, batch_size=32):
    """Apply FGSM to a whole test set (``generate_perturbed_images``)."""
    out = []
    for start in tqdm(range(0, len(x), batch_size), desc=f"FGSM eps={eps}"):
        out.append(fgsm(model, x[start:start + batch_size], y[start:start + batch_size], eps))
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# PGD - Projected Gradient Descent (Madry et al., 2018)
# --------------------------------------------------------------------------- #
def pgd(model, x_batch, y_batch, epsilon, alpha, num_iter, clip_min=0.0, clip_max=1.0):
    """Iterative L-infinity attack (``pgd_attack_batch`` in the notebooks).

    Starting from the clean image, take ``num_iter`` signed-gradient steps of
    size ``alpha``; after every step clip to the valid pixel range and project
    back onto the epsilon-ball around the original image.
    """
    x_batch = np.asarray(x_batch, dtype=np.float32)
    adv = np.copy(x_batch)
    for _ in range(num_iter):
        grad = _loss_gradient(model, adv, y_batch)
        adv = adv + alpha * tf.sign(grad).numpy()
        adv = np.clip(adv, clip_min, clip_max)                       # valid image
        adv = np.clip(adv, x_batch - epsilon, x_batch + epsilon)     # projection
    return adv


def pgd_dataset(model, x, y, epsilon, alpha, num_iter, batch_size):
    """Run PGD over the test set in mini-batches (13 for MRI, 23 for CT).

    Batching only limits GPU memory use; it does not change the perturbations.
    """
    out = []
    for start in tqdm(range(0, len(x), batch_size), desc=f"PGD eps={epsilon}"):
        out.append(pgd(model, x[start:start + batch_size], y[start:start + batch_size],
                       epsilon, alpha, num_iter))
    return np.vstack(out)
