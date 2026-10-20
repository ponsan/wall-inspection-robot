#!/usr/bin/env bash

set -e

OUTPUT="compose.override.yaml"

# WSL の中で動いているかを判定する
if grep -qi microsoft /proc/version; then
    IS_WSL=true
else
    IS_WSL=false
fi

# NVIDIA の GPU が使えるかを判定する
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA=true
else
    HAS_NVIDIA=false
fi

# 判定結果を表示する
echo "WSL: $IS_WSL / NVIDIA: $HAS_NVIDIA"

# 判定結果に合わせて、compose の追加設定ファイルを生成する
{
    echo "services:"

    if [ "$HAS_NVIDIA" != true ] && [ "$IS_WSL" != true ]; then
        # GPU も WSL もない場合：空の設定を出す
        echo "  dev: {}"
    else
        echo "  dev:"

        if [ "$HAS_NVIDIA" = true ]; then
            # CUDA（PyTorch の学習）用
            echo "    gpus: all"
        fi

        if [ "$IS_WSL" = true ]; then
            # WSLg の画面表示と、GPU での描画（OpenGL）用
            cat <<'EOF'
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - /mnt/wslg:/mnt/wslg
      - /usr/lib/wsl:/usr/lib/wsl
    devices:
      - /dev/dxg:/dev/dxg
    environment:
      DISPLAY: ":0"
      WAYLAND_DISPLAY: wayland-0
      XDG_RUNTIME_DIR: /mnt/wslg/runtime-dir
      PULSE_SERVER: unix:/mnt/wslg/PulseServer
      LD_LIBRARY_PATH: /usr/lib/wsl/lib
      GALLIUM_DRIVER: d3d12
EOF
        fi
    fi
} > "$OUTPUT"

