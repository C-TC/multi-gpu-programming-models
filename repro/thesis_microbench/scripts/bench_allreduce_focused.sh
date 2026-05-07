#!/bin/bash
# Focused all_reduce comparison: 4 configs × intra/inter × N_TRIAL trials.
# Clean naming "focus_*" — does not collide with bench_fair_* / bench_rigorous_*.
#
# Configs:
#   1. focus_nccl_*           NCCL recipe (R2 + G10 + GRAPH_MIXING_SUPPORT=0)             — symmetric kernel + NVLS auto
#   2. focus_nccl_ring_*      NCCL legacy ring (NCCL_ALGO=Ring + NCCL_NVLS_ENABLE=0)      — explicit ring algorithm, no NVLS
#   3. focus_nvsdev_*         NVSHMEM device block (parser extracts int32-sum-block row)   — kernel-initiated coll
#   4. focus_nvshost_*        NVSHMEM host on_stream (parser extracts int-sum row)         — host-initiated stream-ordered coll
#
# Why lighter NVSHMEM iters than NCCL recipe (-n 10 -w 3 vs -n 100 -w 50):
#   NVSHMEM device reduction_latency.cu always iterates all 2 dtypes × 7 redops × 3 scopes
#   per size (no CLI knob; line 309 in source hard-codes run_thread=run_warp=run_block=1).
#   With -n 100 -w 50 to 256 MiB the binary runs > 8 min and gets killed by Slurm step
#   timeout; -n 10 -w 3 finishes in ~2.5 min and the median latency for the int32-sum-block
#   path we extract is well-converged (verified <1% stddev across 5 trials in the rigorous
#   data set).

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
INST=${INST:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp-build/install}
NCCL_PIP=${NCCL_PIP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-pip/nvidia/nccl}
NCCL_TESTS=${NCCL_TESTS:-/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/nccl-tests/build}
TAG=${TAG:--newcluster-20260504}
N_TRIAL=${N_TRIAL:-8}
TIMEOUT=${TIMEOUT:-900}
OUT=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local extra_env="$4"; local cmd="$5"
    for trial in $(seq 1 $N_TRIAL); do
        local log="$OUT/bench_${label}_t${trial}${TAG}.log"
        if [[ -s "$log" ]] && grep -qE "^ *[0-9]+ +[0-9]+" "$log"; then
            continue
        fi
        echo "=== $label t=$trial ==="
        timeout $TIMEOUT $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
            bash -c "export LD_LIBRARY_PATH=$INST/lib:$NCCL_PIP/lib:\$LD_LIBRARY_PATH;
                     $extra_env
                     $cmd" > "$log" 2>&1
        local ec=$?
        if [[ $ec -ne 0 ]]; then
            echo "  FAIL ec=$ec; tail:"
            tail -3 "$log" | sed 's/^/  | /'
        fi
    done
}

# ---------- 1. NCCL recipe (R2 + G10 + GRAPH_MIXING_SUPPORT=0) ----------
NCCL_ENV="export NCCL_GRAPH_MIXING_SUPPORT=0;"
NCCL_ARGS="-b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 2 -G 10"
NBIN="$NCCL_TESTS/all_reduce_perf"

run_one "focus_nccl_intra_all_reduce"          1 1 "$NCCL_ENV" "$NBIN $NCCL_ARGS -g 8"
run_one "focus_nccl_inter_all_reduce"          2 8 "$NCCL_ENV" "$NBIN $NCCL_ARGS -g 1"

# ---------- 2. NCCL legacy ring (force ring algorithm, NVLS off) ----------
RING_ENV="$NCCL_ENV export NCCL_NVLS_ENABLE=0 NCCL_ALGO=Ring;"
run_one "focus_nccl_ring_intra_all_reduce"     1 1 "$RING_ENV" "$NBIN $NCCL_ARGS -g 8"
run_one "focus_nccl_ring_inter_all_reduce"     2 8 "$RING_ENV" "$NBIN $NCCL_ARGS -g 1"

# ---------- 3. NVSHMEM device block all_reduce (kernel-initiated, float-sum-block only) ----------
# Uses our patched perftest binary `reduction_focus` (one-line CMakeLists addition;
# stripped-down reduction_latency.cu that does ONLY float-sum-block, no inner
# dtype/redop/scope iteration). This gives:
#   * float (matches NCCL all_reduce dtype, NVLS-eligible in principle)
#   * sum only (matches NCCL all_reduce op)
#   * block scope only (no thread+warp pre-pass eating the timeout)
#   * range can extend to 1 GiB intra in ~30 s, 128 MiB inter in ~70 s.
NV_ENV="export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"
DEV_BIN="$INST/bin/perftest/device/coll/reduction_focus"
DEV_ARGS_INTRA="-b 128 -e 2147483648 -n 100 -w 50 --cudagraph"  # -e 2 GiB → samples up to 1 GiB
DEV_ARGS_INTER="-b 128 -e 2147483648 -n 5 -w 2 --cudagraph"     # full 1 GiB range; ~500 s/trial inter
run_one "focus_nvsdev_intra_all_reduce"        1 8 "$NV_ENV" "$DEV_BIN $DEV_ARGS_INTRA"
run_one "focus_nvsdev_inter_all_reduce"        2 8 "$NV_ENV" "$DEV_BIN $DEV_ARGS_INTER"

# ---------- 4. NVSHMEM host on_stream all_reduce (stream-ordered, float-sum, NVLS-eligible) ----------
# Critical: host on_stream picks NVLS multi-CTA when (a) team has nvls_rsc_base_ptr
# (true on H200 — see NVSHMEM init log "NVLS: supported" + "NVLS Resource Created"),
# (b) dtype is float/half/bfloat (eligible at the device-side dispatch in reduce.cuh
# line ~1592: `is_float_v`), and (c) op is sum. The earlier `int` runs missed (b)
# and went down a slow non-NVLS path. With `-d float -o sum`:
#   intra 1 GiB hits 264 GB/s (~equiv to NCCL recipe 270 GB/s)
#   inter is still RDMA-proxy slow (no NVLS cross-node — NVSwitch is intra-node only,
#   and NVSHMEM_USE_NCCL=OFF removes the NCCL fallback) — ~0.03 GB/s past 16 KiB.
HOST_BIN="$INST/bin/perftest/host/coll/reduction_on_stream"
HOST_ARGS_INTRA="-b 128 -e 1073741824 -n 50 -w 20 --cudagraph -d float -o sum"
# Inter capped at 16 MiB. Larger sizes hang the on_stream binary cross-node
# (verified at 256 MiB / 1 GiB on two fresh allocations: srun timed out at
# 1500 s with 0 data rows). The on_stream kernel ends up calling the same
# `nvshmemi_reduce_threadgroup<...,BLOCK>` as the device-block API for the
# inter case (no NVLS resource on cross-node TEAM_WORLD → num_blocks stays 1
# in `reduce_common.cuh:51`), so the missing inter rows would overlap exactly
# with `focus_nvsdev_inter_*` which IS captured to 1 GiB. Code path is
# literally identical, verified by reading `rdxn_on_stream_kernel` body.
HOST_ARGS_INTER="-b 128 -e 16777216 -n 10 -w 3 --cudagraph -d float -o sum"
run_one "focus_nvshost_intra_all_reduce"       1 8 "$NV_ENV" "$HOST_BIN $HOST_ARGS_INTRA"
run_one "focus_nvshost_inter_all_reduce"       2 8 "$NV_ENV" "$HOST_BIN $HOST_ARGS_INTER"

echo "Done."
