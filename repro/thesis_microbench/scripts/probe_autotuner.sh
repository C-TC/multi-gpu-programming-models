#!/bin/bash
# Capture NCCL's per-call autotuner decisions for 4 configurations:
#   1. default                                     (NVLS available, sym off)
#   2. NCCL_NVLS_ENABLE=0                          (no NVLS algo in candidate set)
#   3. NCCL_SYM_NOWIN_ENABLE=1                     (sym kernel scheduler engaged)
#   4. NCCL_SYM_NOWIN_ENABLE=1 + NCCL_NVLS_ENABLE=0 (sym only)
#
# We log just COLL+TUNING subsystems and write to .nccldbg, so the data
# stays on stdout while NCCL's per-call decisions go to stderr.

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
NCCL_PIP=${NCCL_PIP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nccl-pip/nvidia/nccl}
NCCL_TESTS=${NCCL_TESTS:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/nccl-tests/build}
TAG=${TAG:--newcluster-20260504}
OUT=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"

# Use a short range so we get a manageable number of decisions per run.
ARGS="-b 4 -e 33554432 -f 32 -n 2 -w 1"

probe() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local extra_env="$4"; shift 4
    local cmd="$@"
    local dbg="$OUT/probe_${label}${TAG}.nccldbg"
    echo "=== $label  (${extra_env:-default}) ==="
    timeout 60 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
        bash -c "export LD_LIBRARY_PATH=$NCCL_PIP/lib:\$LD_LIBRARY_PATH;
                 export NCCL_DEBUG=INFO
                 export NCCL_DEBUG_SUBSYS=COLL,TUNING
                 $extra_env
                 $cmd" > /dev/null 2> "$dbg"
    echo "  log: $dbg"
    echo "  -- algo/proto picks (Algo ... proto ...): --"
    grep -E "Bytes -> Algo|Bytes -> Kernel|Symmetric\]" "$dbg" | sort -u | head -10 | sed 's/^/  | /'
    echo
}

# all_reduce on intra 8 GPU
probe "intra_allreduce_default"        1 1 ""                                          "$NCCL_TESTS/all_reduce_perf $ARGS -g 8"
probe "intra_allreduce_nvls_off"       1 1 "export NCCL_NVLS_ENABLE=0;"                "$NCCL_TESTS/all_reduce_perf $ARGS -g 8"
probe "intra_allreduce_sym_on"         1 1 "export NCCL_SYM_NOWIN_ENABLE=1;"           "$NCCL_TESTS/all_reduce_perf $ARGS -g 8"
probe "intra_allreduce_sym_only"       1 1 "export NCCL_SYM_NOWIN_ENABLE=1; export NCCL_NVLS_ENABLE=0;"  "$NCCL_TESTS/all_reduce_perf $ARGS -g 8"

# broadcast on intra 8 GPU (NVLS-friendly op)
probe "intra_broadcast_default"        1 1 ""                                          "$NCCL_TESTS/broadcast_perf $ARGS -g 8"
probe "intra_broadcast_nvls_off"       1 1 "export NCCL_NVLS_ENABLE=0;"                "$NCCL_TESTS/broadcast_perf $ARGS -g 8"
probe "intra_broadcast_sym_on"         1 1 "export NCCL_SYM_NOWIN_ENABLE=1;"           "$NCCL_TESTS/broadcast_perf $ARGS -g 8"

# all_reduce on inter 16 GPU (NVLS doesn't apply across nodes)
probe "inter_allreduce_default"        2 8 ""                                          "$NCCL_TESTS/all_reduce_perf $ARGS -g 1"
probe "inter_allreduce_sym_on"         2 8 "export NCCL_SYM_NOWIN_ENABLE=1;"           "$NCCL_TESTS/all_reduce_perf $ARGS -g 1"

echo "Done."
