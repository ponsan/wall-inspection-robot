# Robot AI Template

ROS2・Python・機械学習を、研究室で再現可能な形で使うための共通開発テンプレートです。

**v0.1 検証済み:** Windows 11 + WSL2 Ubuntu 24.04 + Docker Desktop + VS Code Dev Containers / Python 3.12 / uv 0.12.9 / ROS2 Jazzy / PyTorch / NVIDIA GPU（任意）/ pytest / Ruff / GitHub Actions

## Quick Start

### 事前準備

- Windows 11
- Windows版 VS Code
- WSL2 + Ubuntu 24.04
- Docker Desktop（WSL Integration有効）
- GitHubアカウントとSSH認証
- NVIDIA GPUは任意

> 詳しいPCセットアップは「Robot AI Template 導入手順書」を参照してください。

### 1. テンプレートから新しいリポジトリを作る

GitHubで `robot-ai-template` を開き、**Use this template → Create a new repository** を選択します。

### 2. WSLへclone

```bash
cd ~/projects
git clone git@github.com:<OWNER>/<NEW-REPOSITORY>.git
cd <NEW-REPOSITORY>
code .
```

### 3. Dev Containerを開く

VS CodeのCommand Paletteから次を実行します。

```text
Dev Containers: Reopen in Container
```

起動時に `scripts/detect_gpu.sh` がGPU有無を判定します。NVIDIA GPUが見つかればGPUモード、なければCPUモードで起動します。

### 4. 動作確認

Dev Container内ターミナルで実行します。

```bash
whoami
which python
uv --version
ros2 --help >/dev/null && echo "ROS2 OK"
pytest
```

GPU搭載PCでは追加で確認します。

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## ディレクトリ

| 場所 | 用途 |
|---|---|
| `src/robot_ai/common/` | 共通Python処理 |
| `src/robot_ai/ml/` | ML・前処理・学習・推論 |
| `src/robot_ai/control/` | 制御ロジック |
| `src/robot_ai/gui/` | Streamlit等のGUI |
| `ros2_ws/src/` | ROS2 package |
| `notebooks/` | 探索・分析・実験 |
| `config/` | 再現に必要な設定 |
| `tests/unit/` | 単体テスト |
| `tests/smoke/` | 最低限の動作確認 |
| `docker/` | Docker環境 |
| `.devcontainer/` | VS Code Dev Container設定 |
| `.github/workflows/` | GitHub Actions |

詳細な設計思想・使い分けは「Robot AI Template 設計・利用ガイド」を参照してください。

## 基本コマンド

```bash
# すべてのテスト
pytest

# 軽量テスト（GitHub Actions相当）
pytest -m "not ml and not gpu and not ros and not slow"

# ML（GPU必須を除く）
pytest -m "ml and not gpu"

# GPUテスト
pytest -m gpu

# コード品質
ruff check src tests
ruff format src tests
pre-commit run --all-files
```

## Python依存の追加

```bash
uv add <package>
uv add --optional ml <package>
uv add --optional gui <package>
uv add --optional data <package>
uv add --optional robot <package>
uv add --group dev <package>
```

`pyproject.toml` と `uv.lock` はセットでGit管理します。

## データの扱い

Dataset、学習済みモデル、checkpoint、rosbag、動画、ログなどの大容量ファイルは原則GitHubへ入れません。外部ストレージ（当面はGoogle Drive等）で管理し、必要なデータだけローカル作業領域へ配置します。

`.env`、秘密鍵、API key等の秘密情報は絶対にcommitしないでください。

## Git / CI

当面の研究・学習段階では軽量運用で構いませんが、次の流れを基本にします。

```text
編集 → pytest / pre-commit → commit → push → GitHub Actions確認
```

GitHubへの `push` は、SSH認証を確認済みのWSL Ubuntu側から実行する運用を推奨します。

## v0.1で未実装のもの

- GitHub上のGPU CI
- ROS2専用CI
- Raspberry Pi / ARM64の本格実機検証
- Serial / CAN / camera / LiDAR等の実機I/O設定
- main branch保護・PR必須運用

必要になった段階で、実プロジェクトで検証してからテンプレートへ汎用化します。
