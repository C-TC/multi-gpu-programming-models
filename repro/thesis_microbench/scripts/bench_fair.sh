#!/bin/bash
# Fair NCCL vs NVSHMEM comparison following the recommended NCCL recipe:
#
#   NCCL_GRAPH_MIXING_SUPPORT=0 -b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 2 -G 10
#
# What this changes vs bench_rigorous.sh:
#   * -G 10  CUDA graphs replay the collective 10× per timed iter, amortising
#            launch overhead (this is the real "modern" measurement).
#   * -R 2   nccl-tests calls ncclMemAlloc + ncclCommWindowRegister(SYMMETRIC)
#            so the symmetric kernel scheduler engages.
#   * -c 0   skip per-call validation (faster, doesn't change perf numbers).
#   * -w 50 -n 100   tighter convergence than the earlier -w 20 -n 50.
#   * -b 128 -e 1G   skip the < 128 B regime (where graph launch dominates anyway)
#                    and extend up to 1 GB to see large-message frontiers.
#   * NCCL_GRAPH_MIXING_SUPPORT=0  removes the per-launch sync NCCL adds for
#                                  graph/non-graph mixing safety; saves ~1 µs/launch.
#
# NVSHMEM equivalent (same idea, matched to the tools we have):
#   * --cudagraph     NVSHMEM perftest supports CUDA graph mode (sets use_graph=1).
#   * -w 50 -n 100    same iter counts.
#   * -b 128 -e ?     match NCCL range as far as the perftest allows.
# NVSHMEM device kernels DO use symmetric memory by design (it's the symmetric
# heap), so there's no "-R 2 vs -R 0" knob — the kernel-initiated path is
# always "symmetric". The fair comparison is:
#   NCCL with -R 2 + -G 10  vs  NVSHMEM device with --cudagraph

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
INST=${INST:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nvshmem-3.3.9-ibp-build/install}
NCCL_PIP=${NCCL_PIP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nccl-pip/nvidia/nccl}
NCCL_TESTS=${NCCL_TESTS:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/nccl-tests/build}
TAG=${TAG:--newcluster-20260504}
N_TRIAL=${N_TRIAL:-8}
OUT=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"

# THE RECIPE
NCCL_ARGS="-b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 2 -G 10"
NCCL_ENV="export NCCL_GRAPH_MIXING_SUPPORT=0;"

# NVSHMEM equivalent: 1 GB on collectives is too big for our 8 GB symmetric heap →
# stop at 256 MiB. --cudagraph for graph mode. -w/-n match NCCL.
NVS_COLL_ARGS="-b 128 -e 268435456 -n 100 -w 50 --cudagraph"

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local extra_env="$4"; local cmd="$5"
    for trial in $(seq 1 $N_TRIAL); do
        local log="$OUT/bench_${label}_t${trial}${TAG}.log"
        if [[ -s "$log" ]] && grep -qE "^ *[0-9]+ +[0-9]+" "$log"; then
            continue
        fi
        echo "=== $label t=$trial ==="
        timeout 120 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
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

# ---------- NCCL with the full recipe ----------
# Headline 3 ops (most-used in modern training).
NCCL_OPS="all_reduce alltoall broadcast"
for op in $NCCL_OPS; do
    bin="$NCCL_TESTS/${op}_perf"
    [ -x "$bin" ] || { echo "skip $bin"; continue; }
    # default (-R 2 + -G 10 + NCCL_GRAPH_MIXING_SUPPORT=0)
    run_one "fair_nccl_intra_${op}" 1 1 "$NCCL_ENV" "$bin $NCCL_ARGS -g 8"
    run_one "fair_nccl_inter_${op}" 2 8 "$NCCL_ENV" "$bin $NCCL_ARGS -g 1"
    # NVLS off (same NCCL recipe minus NVLS) so we can isolate that knob too
    run_one "fair_nccl_nvlsoff_intra_${op}" 1 1 "$NCCL_ENV export NCCL_NVLS_ENABLE=0;" "$bin $NCCL_ARGS -g 8"
    run_one "fair_nccl_nvlsoff_inter_${op}" 2 8 "$NCCL_ENV export NCCL_NVLS_ENABLE=0;" "$bin $NCCL_ARGS -g 1"
done

# ---------- NVSHMEM device coll with --cudagraph ----------
NV_ENV="export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"
DEV=$INST/bin/perftest/device/coll
for c_pair in "all_reduce:reduction_latency" "alltoall:alltoall_latency" "broadcast:bcast_latency"; do
    op="${c_pair%%:*}"; nvs_bin="${c_pair##*:}"
    bin="$DEV/$nvs_bin"
    [ -x "$bin" ] || continue
    # reduction is heavier internally → use lighter iter to fit in 120s timeout
    if [[ "$nvs_bin" == reduction_latency ]]; then
        ARGS="-b 128 -e 268435456 -n 20 -w 5 --cudagraph"
    else
        ARGS="$NVS_COLL_ARGS"
    fi
    run_one "fair_nvshmem_intra_${op}" 1 8 "$NV_ENV" "$bin $ARGS"
    run_one "fair_nvshmem_inter_${op}" 2 8 "$NV_ENV" "$bin $ARGS"
done

echo "Done."
