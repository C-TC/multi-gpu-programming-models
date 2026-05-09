#!/bin/bash
# Single-node DeepEP V1 vs V2 measurements (8 GPU H100, intranode NVLink only).
#
# Inside the container, on a 1-node 8-GPU allocation:
#   bash repro/deepep/setup.sh   (or `. repro/deepep/setup.sh`)
#   bash repro/deepep/run_1node.sh
#
# Outputs land in repro/deepep/results/.

set -u
DEEP_EP=${DEEP_EP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/DeepEP}
OUT=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/repro/deepep/results
TAG=${TAG:-}    # e.g. TAG=-newcluster-20260504 → log filenames get the suffix
mkdir -p "$OUT"

. "$(dirname "$0")/setup.sh"

cd "$DEEP_EP"

run() {
    local label="$1"; shift
    echo "=== $label ==="
    "$@" > "$OUT/${label}${TAG}.log" 2>&1
    local ec=$?
    echo "  exit=$ec  log: $OUT/${label}${TAG}.log"
    if [[ $ec -eq 0 ]]; then
        grep -E "Best (dispatch|combine)|EP:   0/8|bandwidth:" "$OUT/${label}${TAG}.log" | head -6 | sed 's/^/  | /'
    fi
    echo
}

# === V1 high-throughput (NVSHMEM, intranode NVLink) ===
run v1_intranode \
    timeout 600 python3 tests/legacy/test_intranode.py --num-processes 8

# === V1 low-latency (NVSHMEM IBGDA → falls back to NVLink on single node) ===
run v1_low_latency \
    timeout 600 python3 tests/legacy/test_low_latency.py --num-processes 8 --num-experts 288

# === V2 high-throughput, default config (NCCL Gin, auto SMs) ===
run v2_ep \
    timeout 600 python3 tests/elastic/test_ep.py --num-processes 8 --skip-check --test-first-only

# === V2 high-throughput, matched to V1 (24 SMs, topk=8) for apples-to-apples ===
run v2_ep_sms24_topk8 \
    timeout 600 python3 tests/elastic/test_ep.py --num-processes 8 --num-sms 24 --num-topk 8 \
    --skip-check --test-first-only

# === V2 low-latency-style (small batch, prefer overlap with compute) ===
run v2_ep_lowlat_1node \
    timeout 600 python3 tests/elastic/test_ep.py --num-processes 8 \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288 \
    --num-sms 32 --num-qps 8 --prefer-overlap-with-compute 1 \
    --skip-check --test-first-only

# === V2 small-batch without prefer-overlap (ablation of the flag) ===
run v2_ep_smallbatch_default \
    timeout 600 python3 tests/elastic/test_ep.py --num-processes 8 \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288 \
    --num-sms 32 --num-qps 8 --prefer-overlap-with-compute 0 \
    --skip-check --test-first-only

echo "Done. Logs in $OUT"
