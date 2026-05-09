#!/bin/bash
# Thesis Chapter 4.1.1 -- P2P micro-benchmarks (NVSHMEM perftest device).
#
# 6 APIs: g, get, p, put, st, atomic_inc
# 3 sweep types per API: message-size, TPB, CTA
# 2 scenarios: intranode (1 node, 2 ranks) and internode (2 nodes, 1 rank each).
#
# Run from the LOGIN NODE with JOBID=<salloc job id> + CNAME=thmicro_$JOBID set.
# The named container must already be primed on both nodes (-N 2 hostname call).
#
# Outputs one log per (api, sweep_type, scenario) in $OUT.

set -u
JOBID=${JOBID:?must set JOBID to your salloc job id}
CNAME=${CNAME:-thmicro_$JOBID}
INST=${INST:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nvshmem-3.3.9-ibp-build/install}
TAG=${TAG:--newcluster-20260504}
OUT=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

P2P=$INST/bin/perftest/device/pt-to-pt
SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"
ENV_PREAMBLE="export LD_LIBRARY_PATH=$INST/lib:\$LD_LIBRARY_PATH; export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"

# 6 P2P APIs (the thesis ones; perftest binary names).
APIS=(shmem_g_bw shmem_get_bw shmem_p_bw shmem_put_bw shmem_st_bw shmem_atomic_bw)
# Pinned mid-size for TPB/CTA sweeps: 1 MiB (clearly bandwidth-bound for all APIs that have sizes; atomics ignore size).
PIN_SIZE=1048576

# Each scenario calls (-N nnodes -np_per_node tasks_per_node).
run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; shift 3
    local cmd="$@"
    local log="$OUT/${label}${TAG}.log"
    echo "=== $label  (-N $nnodes --ntasks-per-node=$tpn)  $cmd ==="
    timeout 90 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
        bash -c "$ENV_PREAMBLE $cmd" > "$log" 2>&1
    local ec=$?
    echo "  log: $log  (exit $ec)"
    if [[ $ec -ne 0 ]]; then
        tail -8 "$log" | sed 's/^/  | /'
    fi
    echo
}

# ---- MESSAGE-SIZE SWEEP (default 32 CTAs × 256 TPB) ----
for api in "${APIS[@]}"; do
    bin="$P2P/$api"
    [ -x "$bin" ] || { echo "SKIP missing $bin"; continue; }
    # intranode: 1 node, 2 ranks
    run_one "p2p_intra_${api}_msgsize" 1 2 "$bin -b 4 -e 33554432 -c 32 -t 256"
    # internode: 2 nodes, 1 rank each
    run_one "p2p_inter_${api}_msgsize" 2 1 "$bin -b 4 -e 33554432 -c 32 -t 256"
done

# ---- TPB SWEEP (pin size = 1 MiB, 32 CTAs) ----
for api in "${APIS[@]}"; do
    bin="$P2P/$api"
    [ -x "$bin" ] || continue
    for tpb in 32 64 128 256 512 1024; do
        run_one "p2p_intra_${api}_tpb${tpb}" 1 2 "$bin -b $PIN_SIZE -e $PIN_SIZE -c 32 -t $tpb"
        run_one "p2p_inter_${api}_tpb${tpb}" 2 1 "$bin -b $PIN_SIZE -e $PIN_SIZE -c 32 -t $tpb"
    done
done

# ---- CTA SWEEP (pin size = 1 MiB, 256 TPB) ----
for api in "${APIS[@]}"; do
    bin="$P2P/$api"
    [ -x "$bin" ] || continue
    for cta in 1 2 4 8 16 32; do
        run_one "p2p_intra_${api}_cta${cta}" 1 2 "$bin -b $PIN_SIZE -e $PIN_SIZE -c $cta -t 256"
        run_one "p2p_inter_${api}_cta${cta}" 2 1 "$bin -b $PIN_SIZE -e $PIN_SIZE -c $cta -t 256"
    done
done

# ---- PING-PONG LATENCY (low-water-mark for round-trip) ----
PP_BINS=(shmem_p_ping_pong_latency shmem_put_ping_pong_latency shmem_signal_ping_pong_latency shmem_atomic_ping_pong_latency)
for pp in "${PP_BINS[@]}"; do
    bin="$P2P/$pp"
    [ -x "$bin" ] || continue
    run_one "p2p_intra_${pp}" 1 2 "$bin -b 4 -e 1048576 -n 500 -w 50"
    run_one "p2p_inter_${pp}" 2 1 "$bin -b 4 -e 1048576 -n 500 -w 50"
done

echo "Done. Logs: $OUT"
