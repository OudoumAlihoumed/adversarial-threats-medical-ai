"""Step 2 - fine-tune the ImageNet backbones on one dataset.

    python -m scripts.train --dataset brain_mri                   # all 20 models
    python -m scripts.train --dataset kidney_ct --models Xception VGG16

Saves <outputs>/<dataset>/models/<Model>.h5, the training histories, a
training curve per model and a clean-test-set summary (training_summary.csv).
"""

import argparse

import pandas as pd

from medattack import config, data, evaluation, models


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    parser.add_argument("--models", nargs="*", default=None,
                        help="backbone names (default: all 20)")
    args = parser.parse_args()

    ds = config.get_dataset(args.dataset)
    x_train, y_train, x_val, y_val, x_test, y_test = data.load_splits(ds)
    ds.results_dir.mkdir(parents=True, exist_ok=True)
    ds.figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in models.resolve_models(args.models):
        model, summary = models.train(name, ds, x_train, y_train, x_val, y_val)
        models.plot_training_history(name, ds, ds.figures_dir / f"{name}_training.png")

        clean = evaluation.binary_metrics(y_test, evaluation.predict_labels(model, x_test))
        print(f"[{name}] clean test set: {evaluation.format_metrics(clean)}")
        rows.append({**summary, **{f"clean_{k}": v for k, v in clean.items()}})

        # Rewrite after every model so a Colab disconnect never loses results.
        pd.DataFrame(rows).to_csv(ds.results_dir / "training_summary.csv", index=False)


if __name__ == "__main__":
    main()
