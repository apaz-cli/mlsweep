#!/usr/bin/env bash
# Run the test sweep with each writer backend (-g 3, -j 10).
# TensorBoard is launched in the background before its sweep so results
# stream in live. Wandb runs in online mode and streams automatically.
#
# Usage: bash tests/run_writers.sh [--no-wandb] [--no-tb]

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SWEEP="tests/test_sweep.py"
OUT="outputs/writer_test"
TB_DIR="/tmp/mlsweep_tb_test"
GPUS=3
JOBS=10

NO_WANDB=0
NO_TB=0
for arg in "$@"; do
    case "$arg" in
        --no-wandb) NO_WANDB=1 ;;
        --no-tb)    NO_TB=1 ;;
    esac
done

TB_PID=""
cleanup() {
    if [ -n "$TB_PID" ]; then
        kill "$TB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

header() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════"
}

# Mirrors the IP-resolution logic in mlsweep_viz: open a UDP socket toward
# 8.8.8.8 to discover which local interface the OS would use, then return
# that IP if it is not loopback.
public_ip() {
    python3 - <<'EOF'
import socket
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    if ip != "127.0.0.1":
        print(ip)
except OSError:
    pass
EOF
}

print_url() {
    local port="$1"
    echo "  http://localhost:${port}"
    local ip
    ip=$(public_ip)
    if [ -n "$ip" ]; then
        echo "  http://${ip}:${port}"
    fi
}

# ── 1. Default (jsonl only) ───────────────────────────────────────────────────
header "jsonl (default)"
mlsweep_run "$SWEEP" \
    -g "$GPUS" -j "$JOBS" \
    --output_dir "$OUT/jsonl" \
    --experiment jsonl_test

# ── 2. TensorBoard ────────────────────────────────────────────────────────────
if [ "$NO_TB" -eq 0 ]; then
    header "tensorboard"
    mkdir -p "$TB_DIR"

    tensorboard --logdir "$TB_DIR/tb_test" --port 6006 --bind_all &
    TB_PID=$!
    sleep 1
    print_url 6006
    echo

    mlsweep_run "$SWEEP" \
        -g "$GPUS" -j "$JOBS" \
        --output_dir "$OUT/tb" \
        --experiment tb_test \
        --tensorboard-dir "$TB_DIR"

    kill "$TB_PID" 2>/dev/null || true
    TB_PID=""
fi

# ── 3. Wandb (online) ─────────────────────────────────────────────────────────
if [ "$NO_WANDB" -eq 0 ]; then
    header "wandb  (online — results stream to the W&B UI)"
    mlsweep_run "$SWEEP" \
        -g "$GPUS" -j "$JOBS" \
        --output_dir "$OUT/wandb" \
        --experiment wandb_test \
        --wandb-project mlsweep-test
fi

echo ""
echo "All done."
