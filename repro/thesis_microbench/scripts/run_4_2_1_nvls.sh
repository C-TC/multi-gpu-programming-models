#!/bin/bash
# Re-run NCCL collectives with NCCL_NVLS_ENABLE=0 (NVLink-SHARP off) and =1 (default).
# Quantifies the contribution of NVLS multicast on intra-node H200 NVL.
# Inter-node NVLS only applies if NVLS-on-Ethernet/IB is configured (it isn't here),
# so the inter run is mostly a sanity check that disabling NVLS doesn't change much.
#
# This is orthogonal to the NCCL_SYM_NOWIN_ENABLE sweep -- symmetric kernels and
# NVLS multicast can both be in play simultaneously by default.

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
NCCL_PIP=${NCCL_PIP:-/mnt/vast/home/tiancheng.chen/workspace/nccl-pip/nvidia/nccl}
NCCL_TESTS=${NCCL_TESTS:-/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/nccl-tests/build}
TAG=${TAG:--newcluster-20260504}
OUT=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local nvls="$4"; shift 4
    local cmd="$@"
    local log="$OUT/${label}_nvls${nvls}${TAG}.log"
    local dbg="$OUT/${label}_nvls${nvls}${TAG}.debug"
    echo "=== $label  nvls=$nvls  (-N $nnodes --ntasks-per-node=$tpn) ==="
    timeout 120 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
        bash -c "export LD_LIBRARY_PATH=$NCCL_PIP/lib:\$LD_LIBRARY_PATH;
                 export NCCL_NVLS_ENABLE=$nvls
                 # debug removed
                 # subsys removed
                 $cmd " > "$log" 2> "$dbg"
    local ec=$?
    echo "  log: $log  (exit $ec)"
    [[ $ec -ne 0 ]] && tail -8 "$log" | sed 's/^/  | /'
    if [[ -s "$dbg" ]]; then
        echo "  -- evidence: --"
        grep -iE "NVLS multicast support|NVLS tuning|Algorithm.*selected|^.*NVLS" "$dbg" | head -2 | sed 's/^/  | /'
    fi
    echo
}

NCCL_ARGS="-b 4 -e 33554432 -f 2 -n 20 -w 5"

# We only run the high-impact 3 collectives × intra+inter to keep it short.
for nvls in 0 1; do
    for op in all_reduce alltoall broadcast; do
        bin="$NCCL_TESTS/${op}_perf"
        [ -x "$bin" ] || continue
        run_one "nccl_intra_${op}_8g" 1 1 "$nvls" "$bin $NCCL_ARGS -g 8"
        run_one "nccl_inter_${op}_2x8" 2 8 "$nvls" "$bin $NCCL_ARGS -g 1"
    done
done

echo "Done. Logs: $OUT"
