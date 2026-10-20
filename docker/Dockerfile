FROM ros:jazzy-ros-base-noble

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:/usr/local/bin:${PATH}"

WORKDIR /workspace

# --------------------------------------------------
# 基本開発ツール
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    build-essential \
    ca-certificates \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    sudo \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Gazebo Harmonic + ROS 2 連携、RViz、描画確認ツール
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-gz \
    ros-jazzy-rviz2 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# uv
# バージョンを固定して再現性を確保
# --------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.12.9 \
    /uv /uvx /usr/local/bin/

# --------------------------------------------------
# Python依存関係
# --------------------------------------------------
COPY pyproject.toml uv.lock README.md ./

RUN uv sync \
    --locked \
    --no-install-project \
    --extra ml \
    --extra gui \
    --extra data \
    --extra robot \
    --group dev

COPY src ./src

RUN uv sync \
    --locked \
    --extra ml \
    --extra gui \
    --extra data \
    --extra robot \
    --group dev

# --------------------------------------------------
# 開発ユーザー
# ROS Jazzy/Nobleイメージ既存の ubuntu (UID/GID 1000) を使用
# --------------------------------------------------
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu && \
    chown -R ubuntu:ubuntu /opt/venv

# --------------------------------------------------
# ROS2環境をbash起動時に自動設定
# --------------------------------------------------
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/ubuntu/.bashrc && \
    echo 'if [ -f /workspace/ros2_ws/install/setup.bash ]; then source /workspace/ros2_ws/install/setup.bash; fi' \
    >> /home/ubuntu/.bashrc

USER ubuntu

WORKDIR /workspace

CMD ["bash"]
