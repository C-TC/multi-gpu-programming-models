#!/bin/bash
# Re-run NCCL collectives with NCCL_SYM_NOWIN_ENABLE=0 (default) and =1 to compare
# the "ring/tree" path vs the new symmetric-memory kernels (NCCL 2.30+).
#
# nccl-tests uses cudaMalloc'd buffers, not ncclMemAlloc'd windows; without
# NCCL_SYM_NOWIN_ENABLE=1, NCCL takes the legacy ring/tree path. With it set,
# NCCL converts the task into a symmetric task and dispatches the new sym kernel.
#
# We also capture NCCL_DEBUG=INFO+TUNING to verify the chosen kernel per call.

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
NCCL_PIP=${NCCL_PIP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nccl-pip/nvidia/nccl}
NCCL_TESTS=${NCCL_TESTS:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/nccl-tests/build}
TAG=${TAG:--newcluster-20260504}
OUT=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local sym="$4"; shift 4
    local cmd="$@"
    local log="$OUT/${label}_sym${sym}${TAG}.log"
    local dbg="$OUT/${label}_sym${sym}${TAG}.debug"
    echo "=== $label  sym=$sym  (-N $nnodes --ntasks-per-node=$tpn) ==="
    timeout 120 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
        bash -c "export LD_LIBRARY_PATH=$NCCL_PIP/lib:\$LD_LIBRARY_PATH;
                 export NCCL_SYM_NOWIN_ENABLE=$sym
                 # NCCL_DEBUG removed, was interleaving with perftest stdout
                 # subsys removed
                 $cmd " > "$log" 2>&1
    local ec=$?
    echo "  log: $log  (exit $ec)"
    [[ $ec -ne 0 ]] && tail -8 "$log" | sed 's/^/  | /'
    # Quick scan of the debug log for which kernel actually ran
    if [[ -s "$dbg" ]]; then
        echo "  -- kernel signature seen in debug log: --"
        grep -E "Symmetric\]|Algorithm|Selecting" "$dbg" | head -3 | sed 's/^/  | /'
    fi
    echo
}

NCCL_ARGS="-b 4 -e 33554432 -f 2 -n 20 -w 5"

for sym in 0 1; do
    for op in all_reduce alltoall broadcast all_gather reduce_scatter; do
        bin="$NCCL_TESTS/${op}_perf"
        [ -x "$bin" ] || continue
        run_one "nccl_intra_${op}_8g" 1 1 "$sym" "$bin $NCCL_ARGS -g 8"
        run_one "nccl_inter_${op}_2x8" 2 8 "$sym" "$bin $NCCL_ARGS -g 1"
    done
done

echo "Done. Logs: $OUT"
