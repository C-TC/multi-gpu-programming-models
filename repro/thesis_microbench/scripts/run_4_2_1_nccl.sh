#!/bin/bash
# Thesis Chapter 4.2.1 -- NCCL vs NVSHMEM device collectives.
#
# This script captures the NCCL side; the NVSHMEM side comes from the 4.1.2
# coll_*_reduction_*  and coll_*_alltoall_*  logs already produced.
#
# nccl-tests `-g N` means N GPUs per process (intranode only). For internode
# we'd need `mpi=1` in the nccl-tests build + mpirun bootstrap; the simpler
# path that maps to the same hardware is to use the NCCL-tests intranode
# results vs NVSHMEM intranode, plus NCCL-tests MPI run for internode.

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
INST=${INST:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nvshmem-3.3.9-ibp-build/install}
NCCL_PIP=${NCCL_PIP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nccl-pip/nvidia/nccl}
NCCL_TESTS=${NCCL_TESTS:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/nccl-tests/build}
TAG=${TAG:--newcluster-20260504}
OUT=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"
NCCL_ENV="export LD_LIBRARY_PATH=$NCCL_PIP/lib:\$LD_LIBRARY_PATH;"

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; shift 3
    local cmd="$@"
    local log="$OUT/${label}${TAG}.log"
    echo "=== $label  (-N $nnodes --ntasks-per-node=$tpn) ==="
    timeout 120 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
        bash -c "$NCCL_ENV $cmd" > "$log" 2>&1
    local ec=$?
    echo "  log: $log  (exit $ec)"
    [[ $ec -ne 0 ]] && tail -8 "$log" | sed 's/^/  | /'
    echo
}

# ---- Intranode 8 GPU ----
# `-g 8` puts 8 GPUs in 1 process (single rank).
run_one "nccl_intra_allreduce_8g"  1 1 "$NCCL_TESTS/all_reduce_perf -b 4 -e 33554432 -f 2 -g 8 -n 20 -w 5"
run_one "nccl_intra_alltoall_8g"   1 1 "$NCCL_TESTS/alltoall_perf  -b 4 -e 33554432 -f 2 -g 8 -n 20 -w 5"
run_one "nccl_intra_reducescatter_8g" 1 1 "$NCCL_TESTS/reduce_scatter_perf -b 4 -e 33554432 -f 2 -g 8 -n 20 -w 5"

# ---- Internode 16 GPU (8 per node) -- uses NCCL with srun pmi2 + each rank picks its GPU via LOCAL_RANK ----
# nccl-tests built without MPI=1: each process is a single-rank, single-GPU client.
# srun launches 16 ranks; nccl-tests autodiscovers via env (NCCL bootstrap) using NCCL_COMM_ID/MASTER_ADDR.
run_one "nccl_inter_allreduce_2x8" 2 8 "$NCCL_TESTS/all_reduce_perf -b 4 -e 33554432 -f 2 -g 1 -n 20 -w 5"
run_one "nccl_inter_alltoall_2x8"  2 8 "$NCCL_TESTS/alltoall_perf  -b 4 -e 33554432 -f 2 -g 1 -n 20 -w 5"

echo "Done. Logs: $OUT"
