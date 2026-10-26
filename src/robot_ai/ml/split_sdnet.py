import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# フォルダ名 → ラベル（1 = 亀裂あり）
CLASS_DIRS = {"CW": 1, "UW": 0}


def list_wall_images(root: Path) -> pd.DataFrame:
    """
    List the wall images of SDNET2018 with their labels and source photo IDs.

    :param root: Dataset root that contains the ``W`` folder.
    :return: DataFrame with columns ``path``, ``label`` and ``photo_id``,
        sorted by ``path``.
    """
    rows = []
    for class_dir, label in CLASS_DIRS.items():
        for image in (root / "W" / class_dir).glob("*.jpg"):
            rows.append(
                {
                    # 置き場所が変わっても使えるように root からの相対パスで持つ
                    "path": image.relative_to(root).as_posix(),
                    "label": label,
                    # ファイル名の「-」より前が元写真の番号
                    "photo_id": image.stem.split("-")[0],
                }
            )
    # glob の順番は環境で変わるので、並べ替えて分割を再現できるようにする
    return pd.DataFrame(rows).sort_values("path", ignore_index=True)


def split_by_photo(images: pd.DataFrame, seed: int, test_folds: int = 7, val_folds: int = 6):
    """
    Assign train, val and test splits so that no source photo is shared.

    :param images: DataFrame returned by :func:`list_wall_images`.
    :param seed: Random seed for the split.
    :param test_folds: Number of folds; one of them becomes the test set.
    :param val_folds: Number of folds for the rest; one of them becomes the val set.
    :return: Copy of ``images`` with a ``split`` column.
    """
    images = images.copy()
    images["split"] = "train"

    # 同じ写真の隣り合うマスは似ているので、写真ごとに分けてリークを防ぐ
    outer = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(images, images["label"], images["photo_id"]))
    images.loc[test_idx, "split"] = "test"

    rest = images.loc[rest_idx]
    inner = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed)
    _, val_pos = next(inner.split(rest, rest["label"], rest["photo_id"]))
    images.loc[rest.index[val_pos], "split"] = "val"
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the SDNET2018 wall images.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/SDNET2018"))
    parser.add_argument("--out", type=Path, default=Path("data/splits/sdnet_wall.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = split_by_photo(list_wall_images(args.root), args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images.to_csv(args.out, index=False)

    summary = images.groupby("split").agg(
        images=("path", "size"),
        cracks=("label", "sum"),
        photos=("photo_id", "nunique"),
    )
    summary["crack_ratio"] = (summary["cracks"] / summary["images"]).round(3)
    print(summary)
    print(f"Saved {len(images)} rows to {args.out}")


if __name__ == "__main__":
    main()
