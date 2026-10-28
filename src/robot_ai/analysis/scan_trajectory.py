import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def read_ground_truth(bag_path: Path) -> pd.DataFrame:
    """
    Read the ground-truth poses from a rosbag and convert them into wall coordinates.

    :param bag_path: Directory of the rosbag.
    :return: DataFrame with columns ``t`` (simulation time [s]), ``u`` and ``v`` [m].
    """
    # ROS 2 を source したときだけ import できるので、テストする関数に影響しないよう中で読み込む
    import rosbag2_py
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path)),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/ground_truth"]))

    rows = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, Odometry)
        stamp = msg.header.stamp
        position = msg.pose.pose.position
        # 壁座標は wall_inspection_control の wall_geometry と同じ定義（右が +u、上が +v）
        rows.append((stamp.sec + stamp.nanosec * 1e-9, -position.y, position.z))
    return pd.DataFrame(rows, columns=["t", "u", "v"])


def trim_before_first_goal(trajectory: pd.DataFrame, goal, tolerance: float = 0.05):
    """
    Drop the samples recorded before the robot first reaches the first goal.

    :param trajectory: DataFrame with columns ``u`` and ``v``.
    :param goal: First goal as ``(u, v)`` [m].
    :param tolerance: Distance at which the goal counts as reached [m].
    :return: DataFrame that starts at the first goal.
    """
    distance = np.hypot(trajectory["u"] - goal[0], trajectory["v"] - goal[1])
    # 最初の目標へ向かう斜めの移動は走査ではないので除く
    first = int(np.argmax(distance.to_numpy() < tolerance))
    return trajectory.iloc[first:].reset_index(drop=True)


def row_mask(trajectory, row_v, u_min, u_max, band=0.1, margin=0.1) -> pd.Series:
    """
    Select the samples that belong to one horizontal scan row.

    Samples near both ends of the row are skipped so that the vertical
    moves between rows are not counted.

    :param trajectory: DataFrame with columns ``u`` and ``v``.
    :param row_v: Height of the row [m].
    :param u_min: Left end of the row [m].
    :param u_max: Right end of the row [m].
    :param band: Maximum distance from the row for a sample to be used [m].
    :param margin: Distance from both ends of the row to skip [m].
    :return: Boolean Series that is True for the samples in the row.
    """
    return (
        (trajectory["u"] > u_min + margin)
        & (trajectory["u"] < u_max - margin)
        & ((trajectory["v"] - row_v).abs() < band)
    )


def row_deviation(trajectory, row_vs, u_min, u_max) -> list[dict]:
    """
    Measure how far the robot drifts vertically from each horizontal scan row.

    :param trajectory: DataFrame with columns ``u`` and ``v``.
    :param row_vs: Height of each scan row [m].
    :param u_min: Left end of the rows [m].
    :param u_max: Right end of the rows [m].
    :return: One dict per row with ``row_v``, ``num_samples``, ``mean`` and ``max_abs``.
    """
    results = []
    for row_v in row_vs:
        inside = row_mask(trajectory, row_v, u_min, u_max)
        dv = trajectory["v"][inside] - row_v
        result = {
            "row_v": row_v,
            "num_samples": int(inside.sum()),
            # 負の値は重力でずり下がっていることを表す
            "mean": float(dv.mean()),
            "max_abs": float(dv.abs().max()),
        }
        results.append(result)
    return results


def save_trajectory_plot(trajectory, goals, row_vs, u_min, u_max, path: Path) -> None:
    """
    Save the path on the wall and the vertical drift along each row as a PNG image.

    :param trajectory: DataFrame with columns ``u`` and ``v``.
    :param goals: Array of goal positions with shape (N, 2).
    :param row_vs: Height of each scan row [m].
    :param u_min: Left end of the rows [m].
    :param u_max: Right end of the rows [m].
    :param path: Output file path.
    """
    fig = Figure(figsize=(10, 4))
    ax_path, ax_drift = fig.subplots(1, 2)

    ax_path.plot(trajectory["u"], trajectory["v"], lw=1, label="path")
    ax_path.scatter(goals[:, 0], goals[:, 1], s=15, c="tab:red", zorder=3, label="goals")
    ax_path.set_aspect("equal")
    ax_path.set_xlabel("u [m] (right)")
    ax_path.set_ylabel("v [m] (up)")
    ax_path.set_title("Path on the wall")
    ax_path.legend(loc="upper right", fontsize=8)

    for row_v in row_vs:
        inside = row_mask(trajectory, row_v, u_min, u_max)
        dv_cm = 100 * (trajectory["v"][inside] - row_v)
        ax_drift.plot(trajectory["u"][inside], dv_cm, ".", ms=1, label=f"v={row_v:.1f}")
    ax_drift.axhline(0, color="gray", lw=0.5)
    ax_drift.set_xlabel("u [m] (right)")
    ax_drift.set_ylabel("v - row [cm]")
    ax_drift.set_title("Vertical drift along each row")
    ax_drift.legend(fontsize=8, markerscale=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the crawler path of a wall scan.")
    parser.add_argument("bag", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/scan_trajectory"))
    parser.add_argument("--u-min", type=float, default=-1.5)
    parser.add_argument("--u-max", type=float, default=1.5)
    parser.add_argument("--v-min", type=float, default=0.5)
    parser.add_argument("--v-max", type=float, default=2.5)
    parser.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # make_zigzag_goals と同じ格子（順番はグラフに関係ないので並べ替えない）
    us = np.arange(args.u_min, args.u_max + args.step / 2, args.step)
    row_vs = np.arange(args.v_min, args.v_max + args.step / 2, args.step).tolist()
    goals = np.array([(u, v) for v in row_vs for u in us])
    trajectory = trim_before_first_goal(read_ground_truth(args.bag), goals[0])

    deviations = row_deviation(trajectory, row_vs, args.u_min, args.u_max)
    summary = {
        "duration_s": float(trajectory["t"].iloc[-1] - trajectory["t"].iloc[0]),
        "num_samples": len(trajectory),
        "rows": deviations,
    }
    (args.out / "row_deviation.json").write_text(json.dumps(summary, indent=2))
    save_trajectory_plot(
        trajectory, goals, row_vs, args.u_min, args.u_max, args.out / "trajectory.png"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
