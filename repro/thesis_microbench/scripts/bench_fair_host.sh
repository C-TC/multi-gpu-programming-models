#!/bin/bash
# Fair recipe NVSHMEM host on_stream collectives — companion to bench_fair.sh
# (which covers NCCL + NVSHMEM device coll). Same methodology:
#
#   * --cudagraph           CUDA graph mode (matches NCCL's -G 10 amortisation)
#   * -w 50 -n 100          warmup + timed iters (matches NCCL recipe)
#   * -b 128 -e 256MiB      same range as NVSHMEM device coll in bench_fair.sh
#   * 8 trials per (op, scenario), fresh srun each
#
# NVSHMEM build was NVSHMEM_USE_NCCL=OFF + NCCL fallback verified absent
# at runtime (nm -D libnvshmem_host.so | grep nccl → 0).

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
INST=${INST:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp-build/install}
TAG=${TAG:--newcluster-20260504}
N_TRIAL=${N_TRIAL:-8}
OUT=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"

NV_ENV="export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"
HOST=$INST/bin/perftest/host/coll
NVS_ARGS="-b 128 -e 268435456 -n 100 -w 50 --cudagraph"

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local cmd="$4"
    for trial in $(seq 1 $N_TRIAL); do
        local log="$OUT/bench_${label}_t${trial}${TAG}.log"
        if [[ -s "$log" ]] && grep -qE "^ *[0-9]+ +[0-9]+" "$log"; then
            continue
        fi
        echo "=== $label t=$trial ==="
        timeout 120 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
            bash -c "export LD_LIBRARY_PATH=$INST/lib:\$LD_LIBRARY_PATH;
                     $NV_ENV
                     $cmd" > "$log" 2>&1
        local ec=$?
        if [[ $ec -ne 0 ]]; then
            echo "  FAIL ec=$ec; tail:"
            tail -3 "$log" | sed 's/^/  | /'
        fi
    done
}

# Map: NCCL op name → NVSHMEM host on_stream binary name
for c_pair in "all_reduce:reduction_on_stream" "alltoall:alltoall_on_stream" "broadcast:broadcast_on_stream"; do
    op="${c_pair%%:*}"; nvs_bin="${c_pair##*:}"
    bin="$HOST/$nvs_bin"
    [ -x "$bin" ] || { echo "skip $bin"; continue; }
    # reduction is heavier internally → use lighter iter to fit
    if [[ "$nvs_bin" == reduction_on_stream || "$nvs_bin" == reducescatter_on_stream ]]; then
        ARGS="-b 128 -e 268435456 -n 20 -w 5 --cudagraph"
    else
        ARGS="$NVS_ARGS"
    fi
    run_one "fair_nvshmem_host_intra_${op}" 1 8 "$bin $ARGS"
    run_one "fair_nvshmem_host_inter_${op}" 2 8 "$bin $ARGS"
done

echo "Done."
