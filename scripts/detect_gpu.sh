#!/usr/bin/env bash

set -e

OUTPUT=".devcontainer/compose.hardware.yaml"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected."

    cat > "$OUTPUT" <<'EOF'
services:
  dev:
    gpus: all
EOF

else
    echo "NVIDIA GPU not detected. Using CPU mode."

    cat > "$OUTPUT" <<'EOF'
services:
  dev: {}
EOF
fi