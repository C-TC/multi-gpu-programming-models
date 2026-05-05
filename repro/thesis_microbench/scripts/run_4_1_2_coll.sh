#!/bin/bash
# Thesis Chapter 4.1.2 -- Collective micro-benchmarks (NVSHMEM perftest device).
#
# 5 collectives: alltoall, bcast, fcollect, reduction (allreduce), reducescatter.
# Sweeps: message-size at default 32 CTAs × 256 TPB; multi-rank scaling on 2 nodes.
# Synchronization barrier as a baseline.

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
INST=${INST:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp-build/install}
TAG=${TAG:--newcluster-20260504}
OUT=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

COLL=$INST/bin/perftest/device/coll
SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"
ENV_PREAMBLE="export LD_LIBRARY_PATH=$INST/lib:\$LD_LIBRARY_PATH; export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"

COLLS=(alltoall_latency bcast_latency fcollect_latency reduction_latency reducescatter_latency barrier_latency sync_latency)

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; shift 3
    local cmd="$@"
    local log="$OUT/${label}${TAG}.log"
    echo "=== $label  (-N $nnodes --ntasks-per-node=$tpn) ==="
    timeout 90 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
        bash -c "$ENV_PREAMBLE $cmd" > "$log" 2>&1
    local ec=$?
    echo "  log: $log  (exit $ec)"
    if [[ $ec -ne 0 ]]; then
        tail -8 "$log" | sed 's/^/  | /'
    fi
    echo
}

# ---- MESSAGE-SIZE SWEEP ----
# intranode: 1 node, 8 ranks (the standard NVL collective scenario)
# internode: 2 nodes, 8 ranks per node (16 total)
for c in "${COLLS[@]}"; do
    bin="$COLL/$c"
    [ -x "$bin" ] || { echo "SKIP missing $bin"; continue; }
    if [[ "$c" == barrier_latency || "$c" == sync_latency ]]; then
        # barrier/sync don't take a size arg
        run_one "coll_intra_${c}_8r"  1 8 "$bin"
        run_one "coll_inter_${c}_16r" 2 8 "$bin"
    else
        run_one "coll_intra_${c}_8r_msgsize"  1 8 "$bin -b 4 -e 33554432"
        run_one "coll_inter_${c}_16r_msgsize" 2 8 "$bin -b 4 -e 33554432"
    fi
done

# ---- RANK SCALING (intranode 2/4/8 ranks; internode 2x2, 2x4, 2x8 = 4/8/16 ranks) ----
for c in alltoall_latency fcollect_latency reduction_latency; do
    bin="$COLL/$c"
    [ -x "$bin" ] || continue
    for r in 2 4 8; do
        run_one "coll_intra_${c}_${r}r" 1 "$r" "$bin -b 65536 -e 65536"
    done
    for r in 2 4 8; do
        n=$((r * 2))
        run_one "coll_inter_${c}_${n}r_2x${r}" 2 "$r" "$bin -b 65536 -e 65536"
    done
done

echo "Done. Logs: $OUT"
