import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from robot_ai.ml.evaluate_crack import save_confusion_matrix
from robot_ai.ml.train_crack import compute_metrics

# 判定の結果ごとの色（見逃しと誤検出を目立たせる）
OUTCOME_COLORS = {"TP": "#d62728", "TN": "#2ca02c", "FP": "#ff7f0e", "FN": "#7f7f7f"}


def match_cells(captures: pd.DataFrame, ground_truth: pd.DataFrame, decimals: int = 2):
    """
    Join the robot's judgement of every cell with the ground-truth map.

    :param captures: Rows of captures.csv with ``goal_u``, ``goal_v``, ``crack_prob``, ``crack``.
    :param ground_truth: Rows of ground_truth.csv with ``cell``, ``u``, ``v``, ``label``.
    :param decimals: Decimals to round the cell centers to before matching.
    :return: Ground-truth rows with ``crack_prob``, ``pred`` and ``outcome`` added.
    :raises ValueError: If a cell has no capture.
    """
    # 浮動小数点の座標をそのまま比べると一致しないことがあるので丸める
    judged = pd.DataFrame(
        {
            "u_key": captures["goal_u"].round(decimals),
            "v_key": captures["goal_v"].round(decimals),
            "crack_prob": captures["crack_prob"],
            "pred": captures["crack"],
        }
    )
    cells = ground_truth.assign(
        u_key=ground_truth["u"].round(decimals), v_key=ground_truth["v"].round(decimals)
    ).merge(judged, on=["u_key", "v_key"], how="left", validate="one_to_one")

    missing = cells.loc[cells["pred"].isna(), "cell"].tolist()
    if missing:
        raise ValueError(f"cells without a capture: {missing}")

    cells["pred"] = cells["pred"].astype(int)
    cells["outcome"] = [
        {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(label, pred)]
        for label, pred in zip(cells["label"], cells["pred"], strict=True)
    ]
    return cells.drop(columns=["u_key", "v_key"])


def save_result_map(cells: pd.DataFrame, cell_size: float, path: Path) -> None:
    """
    Save a map of the wall that shows the outcome of every cell.

    :param cells: Output of :func:`match_cells`.
    :param cell_size: Distance between cell centers in meters.
    :param path: Output file path.
    """
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot()
    half = cell_size / 2
    for row in cells.itertuples():
        ax.add_patch(
            Rectangle(
                (row.u - half, row.v - half),
                cell_size,
                cell_size,
                facecolor=OUTCOME_COLORS[row.outcome],
                edgecolor="white",
            )
        )
        ax.text(row.u, row.v, f"{row.outcome}\n{row.crack_prob:.2f}", ha="center", va="center")

    ax.set_xlim(cells["u"].min() - half, cells["u"].max() + half)
    ax.set_ylim(cells["v"].min() - half, cells["v"].max() + half)
    ax.set_aspect("equal")
    ax.set_xlabel("u [m] (right)")
    ax.set_ylabel("v [m] (up)")
    ax.set_title("Crack inspection vs. ground truth")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the inspection result with the wall map.")
    parser.add_argument("--captures", type=Path, default=Path("data/inspection/captures.csv"))
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("ros2_ws/src/wall_inspection_gazebo/models/wall_tiles/ground_truth.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/wall_map"))
    parser.add_argument("--cell-size", type=float, default=0.5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cells = match_cells(pd.read_csv(args.captures), pd.read_csv(args.ground_truth))

    metrics = compute_metrics(cells["label"].tolist(), cells["pred"].tolist())
    metrics["num_cells"] = len(cells)
    metrics["missed_cells"] = cells.loc[cells["outcome"] == "FN", "cell"].tolist()
    metrics["false_alarm_cells"] = cells.loc[cells["outcome"] == "FP", "cell"].tolist()

    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cells.to_csv(args.out / "cells.csv", index=False)
    save_confusion_matrix(metrics["confusion_matrix"], args.out / "confusion_matrix.png")
    save_result_map(cells, args.cell_size, args.out / "result_map.png")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
