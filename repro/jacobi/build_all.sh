#!/bin/bash
# Build the 7 jacobi variants for sm_90.
# Source repro/jacobi/setup_env.sh first.

set -u
REPO=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models
LOG=$REPO/repro/jacobi/results/logs
mkdir -p "$LOG"

VARIANTS=(single_gpu mpi mpi_overlap nccl nccl_graphs nccl_overlap nvshmem)

cd "$REPO"
fail=0
for v in "${VARIANTS[@]}"; do
    echo "=== build $v ==="
    if ( cd "$v" && make clean && make BUILD_SM_ARCH=90 -j ) > "$LOG/build-$v.log" 2>&1; then
        ls -la "$v/jacobi" 2>&1 | tail -1
    else
        echo "  FAILED — see $LOG/build-$v.log (last 8 lines below)"
        tail -8 "$LOG/build-$v.log" | sed 's/^/  | /'
        fail=$((fail+1))
    fi
done
echo
echo "Built $((${#VARIANTS[@]} - fail))/${#VARIANTS[@]} variants. Failures: $fail."
exit $fail
