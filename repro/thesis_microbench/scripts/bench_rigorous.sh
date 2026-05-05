#!/bin/bash
# Unified rigorous benchmark: 8 trials × (20 warmup + 50 timed iters) per data point.
#
# Domains (set DOMAINS env to subset):
#   nccl              - 5 NCCL collectives, intra+inter, {default, sym(-R 2), nvls_off}
#   nvshmem_device    - 5 NVSHMEM device collectives, intra+inter, {nvls_on, nvls_off}
#   nvshmem_host      - 5 NVSHMEM host on_stream collectives, intra+inter, {nvls_on, nvls_off}
#   p2p               - 6 NVSHMEM P2P APIs, intra+inter (no NVLS knob)
#
# Output naming: bench_<config>_<scenario>_<op>_t<trial>{TAG}.log
# where config encodes the implementation + flags so a single sweep script
# fully describes the configuration matrix.

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
INST=${INST:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp-build/install}
NCCL_PIP=${NCCL_PIP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-pip/nvidia/nccl}
NCCL_TESTS=${NCCL_TESTS:-/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/nccl-tests/build}
TAG=${TAG:--newcluster-20260504}
N_TRIAL=${N_TRIAL:-8}
DOMAINS=${DOMAINS:-nccl nvshmem_device p2p}
OUT=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"

# Rigorous iter counts.
NCCL_ARGS="-b 4 -e 33554432 -f 2 -n 50 -w 20"
# NVSHMEM perftest: -n iters, -w warmup. Default in args files is 10/5; override.
NVS_COLL_ARGS="-b 4 -e 33554432 -n 50 -w 20"
NVS_P2P_ARGS="-b 4 -e 33554432 -n 50 -w 20 -c 32 -t 256"

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local extra_env="$4"; local cmd="$5"
    for trial in $(seq 1 $N_TRIAL); do
        local log="$OUT/bench_${label}_t${trial}${TAG}.log"
        # Resume: skip already-good trial logs (non-empty + has at least one numeric data row)
        # NCCL format has leading spaces; NVSHMEM perftest format has no leading spaces -- accept both.
        if [[ -s "$log" ]] && grep -qE "^ *[0-9]+ +[0-9]+" "$log"; then
            echo "=== $label t=$trial (skip, already done) ==="
            continue
        fi
        echo "=== $label t=$trial ==="
        timeout 90 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
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

# ---------- NCCL ----------
if [[ " $DOMAINS " =~ " nccl " ]]; then
    NCCL_OPS="all_reduce alltoall broadcast all_gather reduce_scatter"
    for op in $NCCL_OPS; do
        bin="$NCCL_TESTS/${op}_perf"
        [ -x "$bin" ] || { echo "skip $bin"; continue; }
        # default (-R 0)
        run_one "nccl_default_intra_${op}" 1 1 "" "$bin $NCCL_ARGS -g 8 -R 0"
        run_one "nccl_default_inter_${op}" 2 8 "" "$bin $NCCL_ARGS -g 1 -R 0"
        # sym (-R 2 → ncclSymmetricTaskScheduler)
        run_one "nccl_sym_intra_${op}"     1 1 "" "$bin $NCCL_ARGS -g 8 -R 2"
        run_one "nccl_sym_inter_${op}"     2 8 "" "$bin $NCCL_ARGS -g 1 -R 2"
        # NVLS off (-R 0 + NCCL_NVLS_ENABLE=0)
        run_one "nccl_nvlsoff_intra_${op}" 1 1 "export NCCL_NVLS_ENABLE=0;" "$bin $NCCL_ARGS -g 8 -R 0"
        run_one "nccl_nvlsoff_inter_${op}" 2 8 "export NCCL_NVLS_ENABLE=0;" "$bin $NCCL_ARGS -g 1 -R 0"
    done
fi

# ---------- NVSHMEM device collectives ----------
if [[ " $DOMAINS " =~ " nvshmem_device " ]]; then
    NV_ENV="export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"
    DEV=$INST/bin/perftest/device/coll
    for c in alltoall_latency bcast_latency fcollect_latency reduction_latency reducescatter_latency; do
        bin="$DEV/$c"
        [ -x "$bin" ] || continue
        # reduction/reducescatter perftest internally loops over 5 dtype × 5 op × 3 scope per size,
        # so it's ~15× heavier than alltoall/bcast/fcollect. Use lighter iters for those.
        if [[ "$c" == reduction_latency || "$c" == reducescatter_latency ]]; then
            ARGS="-b 4 -e 33554432 -n 5 -w 2"
        else
            ARGS="$NVS_COLL_ARGS"
        fi
        # NVLS on (default)
        run_one "nvsdev_nvlson_intra_${c}"  1 8 "$NV_ENV"                                "$bin $ARGS"
        run_one "nvsdev_nvlson_inter_${c}"  2 8 "$NV_ENV"                                "$bin $ARGS"
        # NVLS off
        run_one "nvsdev_nvlsoff_intra_${c}" 1 8 "$NV_ENV export NVSHMEM_DISABLE_NVLS=1;" "$bin $ARGS"
        run_one "nvsdev_nvlsoff_inter_${c}" 2 8 "$NV_ENV export NVSHMEM_DISABLE_NVLS=1;" "$bin $ARGS"
    done
fi

# ---------- NVSHMEM host on_stream collectives ----------
if [[ " $DOMAINS " =~ " nvshmem_host " ]]; then
    NV_ENV="export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"
    HOST=$INST/bin/perftest/host/coll
    for c in alltoall_on_stream broadcast_on_stream fcollect_on_stream reduction_on_stream reducescatter_on_stream; do
        bin="$HOST/$c"
        [ -x "$bin" ] || continue
        run_one "nvshost_nvlson_intra_${c}"  1 8 "$NV_ENV"                                "$bin $NVS_COLL_ARGS"
        run_one "nvshost_nvlson_inter_${c}"  2 8 "$NV_ENV"                                "$bin $NVS_COLL_ARGS"
        run_one "nvshost_nvlsoff_intra_${c}" 1 8 "$NV_ENV export NVSHMEM_DISABLE_NVLS=1;" "$bin $NVS_COLL_ARGS"
        run_one "nvshost_nvlsoff_inter_${c}" 2 8 "$NV_ENV export NVSHMEM_DISABLE_NVLS=1;" "$bin $NVS_COLL_ARGS"
    done
fi

# ---------- NVSHMEM P2P (intra=2 ranks, inter=1 rank/node) ----------
if [[ " $DOMAINS " =~ " p2p " ]]; then
    NV_ENV="export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"
    P2P=$INST/bin/perftest/device/pt-to-pt
    for api in shmem_g_bw shmem_get_bw shmem_p_bw shmem_put_bw shmem_st_bw shmem_atomic_bw; do
        bin="$P2P/$api"
        [ -x "$bin" ] || continue
        run_one "p2p_intra_${api}" 1 2 "$NV_ENV" "$bin $NVS_P2P_ARGS"
        run_one "p2p_inter_${api}" 2 1 "$NV_ENV" "$bin $NVS_P2P_ARGS"
    done
fi

echo "Done."
