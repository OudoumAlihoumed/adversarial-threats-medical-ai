"""Step 3 - attack the trained models with FGSM or PGD and measure the damage.

    python -m scripts.attack --dataset brain_mri --attack fgsm
    python -m scripts.attack --dataset kidney_ct --attack pgd --models InceptionV3

For every model: clean metrics + metrics at eps = 0.02, 0.05, 0.10
(accuracy, precision, recall, F1, specificity), a confusion matrix per
condition, one ROC figure and a before/after example image per budget.
Results are written to <outputs>/<dataset>/results/<attack>_results.csv.
"""

import argparse

import pandas as pd

from medattack import attacks, config, data, evaluation, models, plots


def run_attack(attack, model, x, y, ds, eps, alpha=None, n_iter=None):
    if attack == "fgsm":
        return attacks.fgsm_dataset(model, x, y, eps)
    return attacks.pgd_dataset(model, x, y, eps, alpha, n_iter, ds.pgd_batch_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    parser.add_argument("--attack", required=True, choices=["fgsm", "pgd"])
    parser.add_argument("--models", nargs="*", default=None,
                        help="backbone names (default: all 20)")
    args = parser.parse_args()

    ds = config.get_dataset(args.dataset)
    *_, x_test, y_test = data.load_splits(ds)
    label = args.attack.upper()
    fig_dir = ds.figures_dir / args.attack
    fig_dir.mkdir(parents=True, exist_ok=True)
    ds.results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = ds.results_dir / f"{args.attack}_results.csv"

    if args.attack == "fgsm":
        settings = [(eps, None, None) for eps in config.EPSILONS]
    else:
        settings = list(config.PGD_SETTINGS)

    rows = []
    for name in models.resolve_models(args.models):
        model = models.load(name, ds)
        predictions = {0.0: evaluation.predict_labels(model, x_test)}
        clean = evaluation.binary_metrics(y_test, predictions[0.0])
        rows.append({"model": name, "attack": "none", "epsilon": 0.0, **clean})
        print(f"\n[{name}] clean       {evaluation.format_metrics(clean)}")
        plots.confusion_matrix(y_test, predictions[0.0], f"{name} - clean",
                               fig_dir / f"{name}_cm_clean.png")

        for eps, alpha, n_iter in settings:
            x_adv = run_attack(args.attack, model, x_test, y_test, ds, eps, alpha, n_iter)
            predictions[eps] = evaluation.predict_labels(model, x_adv)
            m = evaluation.binary_metrics(y_test, predictions[eps])
            rows.append({"model": name, "attack": label, "epsilon": eps, **m})
            print(f"[{name}] {label} ε={eps:<5}{evaluation.format_metrics(m)}")

            plots.confusion_matrix(y_test, predictions[eps], f"{name} - {label} ε: {eps}",
                                   fig_dir / f"{name}_cm_eps{eps}.png")
            i = ds.example_index
            pattern = None
            if args.attack == "fgsm":
                pattern = attacks.fgsm_pattern(model, x_test[i:i + 1], y_test[i:i + 1])[0].numpy()
            plots.attack_example(model, x_test[i], x_adv[i], int(y_test[i]), label, eps,
                                 fig_dir / f"{name}_example_eps{eps}.png", pattern=pattern)

        plots.roc_curves(y_test, predictions, f"{name} - ROC curve ({label})",
                         fig_dir / f"{name}_roc.png")
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()
