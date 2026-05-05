#!/bin/bash
# Extended 4.2.1: more NCCL collectives + NVSHMEM host-side equivalents.
#
# Adds per-collective NCCL tests (broadcast, all_gather, reduce_scatter) and
# NVSHMEM host-on-stream variants so that the comparison plot shows three
# series side-by-side per collective:
#   * NVSHMEM device (kernel-initiated, 32-bit, block scope)  -- already in coll_*_msgsize logs
#   * NVSHMEM host on_stream (CPU-initiated)                  -- run here
#   * NCCL                                                    -- run here

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
INST=${INST:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp-build/install}
NCCL_PIP=${NCCL_PIP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-pip/nvidia/nccl}
NCCL_TESTS=${NCCL_TESTS:-/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/nccl-tests/build}
TAG=${TAG:--newcluster-20260504}
OUT=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"
NV_ENV="export LD_LIBRARY_PATH=$INST/lib:\$LD_LIBRARY_PATH; export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"
NCCL_ENV="export LD_LIBRARY_PATH=$NCCL_PIP/lib:\$LD_LIBRARY_PATH;"

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local env="$4"; shift 4
    local cmd="$@"
    local log="$OUT/${label}${TAG}.log"
    echo "=== $label  (-N $nnodes --ntasks-per-node=$tpn) ==="
    timeout 120 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
        bash -c "$env $cmd" > "$log" 2>&1
    local ec=$?
    echo "  log: $log  (exit $ec)"
    [[ $ec -ne 0 ]] && tail -8 "$log" | sed 's/^/  | /'
    echo
}

# ---- NCCL: 5 collectives × 2 scenarios ----
NCCL_ARGS="-b 4 -e 33554432 -f 2 -n 20 -w 5"
for op in all_reduce alltoall broadcast all_gather reduce_scatter; do
    bin="$NCCL_TESTS/${op}_perf"
    [ -x "$bin" ] || { echo "SKIP missing $bin"; continue; }
    # intranode 1 process × 8 GPUs (no MPI)
    run_one "nccl_intra_${op}_8g" 1 1 "$NCCL_ENV" "$bin $NCCL_ARGS -g 8"
    # internode 2×8 = 16 ranks (MPI=1 build)
    run_one "nccl_inter_${op}_2x8" 2 8 "$NCCL_ENV" "$bin $NCCL_ARGS -g 1"
done

# ---- NVSHMEM host on_stream collectives (CPU-initiated, 5 collectives × 2 scenarios) ----
HOST_COLL=$INST/bin/perftest/host/coll
for c in alltoall_on_stream broadcast_on_stream fcollect_on_stream reducescatter_on_stream reduction_on_stream; do
    bin="$HOST_COLL/$c"
    [ -x "$bin" ] || { echo "SKIP missing $bin"; continue; }
    run_one "coll_intra_${c}_8r_msgsize"  1 8 "$NV_ENV" "$bin -b 4 -e 33554432"
    run_one "coll_inter_${c}_16r_msgsize" 2 8 "$NV_ENV" "$bin -b 4 -e 33554432"
done

echo "Done. Logs: $OUT"
