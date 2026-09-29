"""Full-reference image-quality metrics between clean and adversarial images.

These measure how *visible* an attack is: SSIM, MS-SSIM, UQI and VIFP are 1
for identical images, MSE and RMSE are 0, and a higher PSNR means less
distortion. Computed with the `sewar` package on 8-bit images.
"""

import numpy as np
import pandas as pd
from sewar.full_ref import ergas, mse, msssim, psnr, rase, rmse, sam, scc, ssim, uqi, vifp

METRICS = {
    "SSIM": lambda a, b: ssim(a, b)[0],        # sewar returns (ssim, cs)
    "MS-SSIM": lambda a, b: np.real(msssim(a, b)),
    "UQI": uqi,
    "VIFP": vifp,
    "PSNR": psnr,
    "MSE": mse,
    "RMSE": rmse,
    "ERGAS": ergas,
    "SCC": scc,
    "RASE": rase,
    "SAM": sam,
}


def to_uint8(image):
    """[0, 1] float image -> 0-255 uint8 (same conversion as the notebooks)."""
    return (np.asarray(image) * 255).astype(np.uint8)


def compare_images(clean, adversarial):
    """All metrics for one clean/adversarial pair."""
    a, b = to_uint8(clean), to_uint8(adversarial)
    return {name: float(fn(a, b)) for name, fn in METRICS.items()}


def compare_datasets(clean_set, adversarial_set):
    """Per-image metrics for two aligned image sets, as a DataFrame."""
    assert clean_set.shape == adversarial_set.shape, "Datasets must have the same shape"
    rows = [compare_images(c, a) for c, a in zip(clean_set, adversarial_set)]
    return pd.DataFrame(rows)


def summarise(per_image: pd.DataFrame):
    """Mean of every metric over the whole test set (inf/NaN values ignored)."""
    finite = per_image.replace([np.inf, -np.inf], np.nan)
    return finite.mean(skipna=True)
