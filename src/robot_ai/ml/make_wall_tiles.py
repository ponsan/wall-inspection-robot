import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_CONFIG = """<?xml version="1.0"?>
<model>
  <name>wall_tiles</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>
    Concrete wall images from SDNET2018 (test split), placed on the wall surface.
    SDNET2018: Maguire, Dorafshan and Thomas (2018), Utah State University,
    https://doi.org/10.15142/T3TD19, licensed under CC BY 4.0.
  </description>
</model>
"""

VISUAL = """      <visual name="tile_{cell:02d}">
        <pose>-0.001 {y:.3f} {z:.3f} 0 0 0</pose>
        <cast_shadows>false</cast_shadows>
        <geometry>
          <plane><normal>-1 0 0</normal><size>{size} {size}</size></plane>
        </geometry>
        <material>
          <diffuse>1 1 1 1</diffuse>
          <pbr>
            <metal>
              <albedo_map>model://wall_tiles/materials/textures/{texture}</albedo_map>
              <roughness>1</roughness>
              <metalness>0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>"""


def choose_tiles(images, cells, num_crack, seed) -> pd.DataFrame:
    """
    Choose test images for the wall cells and shuffle their positions.

    :param images: DataFrame with columns ``path``, ``label`` and ``split``.
    :param cells: Cell centers as a list of ``(u, v)`` [m].
    :param num_crack: Number of cells that get a crack image.
    :param seed: Random seed.
    :return: DataFrame with columns ``cell``, ``u``, ``v``, ``label`` and ``source``.
    """
    # 学習に使っていない test の画像だけを貼る（リーク防止）
    test = images[images["split"] == "test"]
    crack = test[test["label"] == 1].sample(num_crack, random_state=seed)
    sound = test[test["label"] == 0].sample(len(cells) - num_crack, random_state=seed)
    order = np.random.default_rng(seed).permutation(len(cells))
    chosen = pd.concat([crack, sound]).iloc[order].reset_index(drop=True)

    return pd.DataFrame(
        {
            "cell": range(len(cells)),
            "u": [u for u, _ in cells],
            "v": [v for _, v in cells],
            "label": chosen["label"],
            "source": chosen["path"],
        }
    )


def make_model_sdf(tiles, size) -> str:
    """
    Build an SDF model with one textured plane per cell.

    :param tiles: DataFrame from :func:`choose_tiles` with a ``texture`` column.
    :param size: Length of one side of a tile [m].
    :return: SDF text of the model.
    """
    # モデルの原点を壁の表面に置く。u = -y、v = z（wall_geometry と同じ定義）
    visuals = [
        VISUAL.format(cell=t.cell, y=-t.u, z=t.v, size=size, texture=t.texture)
        for t in tiles.itertuples()
    ]
    return (
        '<?xml version="1.0"?>\n<sdf version="1.9">\n  <model name="wall_tiles">\n'
        '    <static>true</static>\n    <link name="link">\n'
        + "\n".join(visuals)
        + "\n    </link>\n  </model>\n</sdf>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the wall tile model for Gazebo.")
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/sdnet_wall.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/SDNET2018"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ros2_ws/src/wall_inspection_gazebo/models/wall_tiles"),
    )
    parser.add_argument("--tile-size", type=float, default=0.2)
    parser.add_argument("--num-crack", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 走査の目標（make_zigzag_goals）と同じ 7 × 5 の格子
    cells = [(u, v) for v in np.arange(0.5, 2.75, 0.5) for u in np.arange(-1.5, 1.75, 0.5)]
    tiles = choose_tiles(pd.read_csv(args.split_csv), cells, args.num_crack, args.seed)
    tiles["texture"] = [f"tile_{c:02d}.jpg" for c in tiles["cell"]]

    texture_dir = args.out / "materials" / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)
    for t in tiles.itertuples():
        shutil.copy(args.image_root / t.source, texture_dir / t.texture)

    tiles.to_csv(args.out / "ground_truth.csv", index=False)
    (args.out / "model.config").write_text(MODEL_CONFIG)
    (args.out / "model.sdf").write_text(make_model_sdf(tiles, args.tile_size))
    print(tiles[["cell", "u", "v", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
