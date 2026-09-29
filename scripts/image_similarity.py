"""Step 4 - how visible are the attacks? Image-quality metrics of adversarial
examples against their clean originals.

    python -m scripts.image_similarity --dataset brain_mri
    python -m scripts.image_similarity --dataset kidney_ct --reference Xception

Generates FGSM and PGD adversarial test sets with a reference backbone
(Xception in the paper) and scores every image with SSIM, MS-SSIM, UQI, VIFP,
PSNR, MSE, RMSE, ERGAS, SCC, RASE and SAM. Writes per-image scores and a
summary table (<outputs>/<dataset>/results/image_similarity.csv).
"""

import argparse

import numpy as np
import pandas as pd

from medattack import attacks, config, data, models, similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    parser.add_argument("--reference", default="Xception",
                        help="backbone used to craft the adversarial images")
    args = parser.parse_args()

    ds = config.get_dataset(args.dataset)
    *_, x_test, y_test = data.load_splits(ds)
    model = models.load(args.reference, ds)
    adv_dir = ds.out_dir / "adversarial"
    adv_dir.mkdir(parents=True, exist_ok=True)
    ds.results_dir.mkdir(parents=True, exist_ok=True)

    conditions = [("FGSM", eps, None, None) for eps in config.EPSILONS]
    conditions += [("PGD", eps, alpha, n) for eps, alpha, n in config.PGD_SETTINGS]

    summary = {}
    for attack, eps, alpha, n_iter in conditions:
        if attack == "FGSM":
            x_adv = attacks.fgsm_dataset(model, x_test, y_test, eps)
        else:
            x_adv = attacks.pgd_dataset(model, x_test, y_test, eps, alpha, n_iter,
                                        ds.pgd_batch_size)
        np.save(adv_dir / f"adv_{eps}_{attack}.npy", x_adv)

        per_image = similarity.compare_datasets(x_test, x_adv)
        per_image.to_csv(ds.results_dir / f"similarity_{attack}_{eps}_per_image.csv",
                         index=False)
        summary[f"{attack} {eps}"] = similarity.summarise(per_image)
        print(f"{attack} ε={eps}: SSIM {summary[f'{attack} {eps}']['SSIM']:.3f}  "
              f"PSNR {summary[f'{attack} {eps}']['PSNR']:.2f} dB")

    table = pd.DataFrame(summary)
    table.to_csv(ds.results_dir / "image_similarity.csv")
    print(f"\n{ds.display_name} - image quality of adversarial examples ({args.reference})")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
