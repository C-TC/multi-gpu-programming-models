#!/bin/bash
# Rigorous comparison: nccl-tests with vs without symmetric registration.
#
# Each config runs N_TRIAL times (8 by default).  Each trial uses 100 warmup
# iters + 200 timed iters per data point (vs the default 5+20).  We then take
# the median and report mean ± stddev across trials.
#
# Configs:
#   default              : -R 0  (no registration)
#   sym (window reg)     : -R 2  (ncclCommWindowRegister with NCCL_WIN_COLL_SYMMETRIC)
#                          → triggers ncclSymmetricTaskScheduler, log shows [Symmetric]
#   nvls_off             : -R 0 + NCCL_NVLS_ENABLE=0
#
# Each call's per-call kernel decision is captured and summarized.

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
NCCL_PIP=${NCCL_PIP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nccl-pip/nvidia/nccl}
NCCL_TESTS=${NCCL_TESTS:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/nccl-tests/build}
TAG=${TAG:--newcluster-20260504}
N_TRIAL=${N_TRIAL:-8}
OUT=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"

# 100 warmup + 200 timed; sweep step ×2; up to 32 MiB.
NCCL_ARGS="-b 4 -e 33554432 -f 2 -n 200 -w 100"

run_n() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local extra_env="$4"; local extra_args="$5"
    local op="$6"
    local bin="$NCCL_TESTS/${op}_perf"
    [ -x "$bin" ] || { echo "skip $bin"; return; }
    for trial in $(seq 1 $N_TRIAL); do
        local log="$OUT/bench_${label}_${op}_t${trial}${TAG}.log"
        echo "=== $label  trial=$trial $op  (-N $nnodes --ntasks-per-node=$tpn) ==="
        timeout 180 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
            bash -c "export LD_LIBRARY_PATH=$NCCL_PIP/lib:\$LD_LIBRARY_PATH;
                     $extra_env
                     $bin $NCCL_ARGS $extra_args" > "$log" 2>&1
        local ec=$?
        echo "  log: $log  (exit $ec)"
        if [[ $ec -ne 0 ]]; then
            tail -3 "$log" | sed 's/^/  | /'
        fi
    done
}

# Capture one debug log per config (separate from trials, with NCCL_DEBUG)
capture_kernel_evidence() {
    local label="$1"; local extra_env="$2"; local extra_args="$3"; local op="$4"
    local dbg="$OUT/bench_${label}_${op}_evidence${TAG}.log"
    echo "=== capturing kernel evidence for $label / $op ==="
    timeout 60 $SRUN_BASE -N 1 --ntasks-per-node=1 \
        bash -c "export LD_LIBRARY_PATH=$NCCL_PIP/lib:\$LD_LIBRARY_PATH;
                 export NCCL_DEBUG=INFO; export NCCL_DEBUG_SUBSYS=COLL,TUNING;
                 $extra_env
                 $NCCL_TESTS/${op}_perf -b 1024 -e 1048576 -f 32 -n 5 -w 2 -g 8 $extra_args" \
        > "$dbg" 2>&1
    echo "  evidence: $dbg"
    echo "  -- kernel/algo picks --"
    grep -E "Bytes -> Algo|Symmetric\]" "$dbg" | sort -u | head -8 | sed 's/^/  | /'
    echo
}

# Capture evidence first (3 configs × 1 op = 3 small runs)
capture_kernel_evidence "default"  ""                                  "-R 0" "all_reduce"
capture_kernel_evidence "sym"      ""                                  "-R 2" "all_reduce"
capture_kernel_evidence "nvls_off" "export NCCL_NVLS_ENABLE=0;"        "-R 0" "all_reduce"

# Now the rigorous timing: 3 configs × 1 op × N_TRIAL trials.
run_n "default"  1 1 ""                              "-R 0 -g 8" "all_reduce"
run_n "sym"      1 1 ""                              "-R 2 -g 8" "all_reduce"
run_n "nvls_off" 1 1 "export NCCL_NVLS_ENABLE=0;"    "-R 0 -g 8" "all_reduce"

echo "Done.  Use the analyze script to compute mean ± stddev."
