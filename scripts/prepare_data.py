"""Step 1 - turn the raw Kaggle folders into cached NumPy arrays.

    python -m scripts.prepare_data --dataset brain_mri
    python -m scripts.prepare_data --dataset kidney_ct
"""

import argparse

from medattack import config, data


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    args = parser.parse_args()

    ds = config.get_dataset(args.dataset)
    print(f"Preparing {ds.display_name} from {ds.raw_dir}")
    data.prepare(ds)

    _, y_train, _, y_val, _, y_test = data.load_splits(ds)
    for split, y in (("train", y_train), ("validation", y_val), ("test", y_test)):
        print(f"  {split:<10} {len(y):>5} images  (normal {int((y == 0).sum())}, "
              f"tumour {int((y == 1).sum())})")


if __name__ == "__main__":
    main()
