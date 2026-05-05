#!/bin/bash
# Re-run the same NVSHMEM device collectives that 4.1.2 ran, but with
# NVSHMEM_DISABLE_NVLS=1 (NVLink-SHARP off in NVSHMEM).  This isolates how much
# the NVSHMEM device-side tree/ring pays without NVLS multicast.
#
# Default in our build: NVLS support compiled in, runtime auto-uses it when
# the platform reports NVLS multicast (we confirmed it does on H200).

set -u
JOBID=${JOBID:?must set JOBID}
CNAME=${CNAME:-thmicro_$JOBID}
INST=${INST:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp-build/install}
TAG=${TAG:--newcluster-20260504}
OUT=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/thesis_microbench/results
mkdir -p "$OUT"

SRUN_BASE="srun --jobid=$JOBID --mpi=pmi2 --overlap --container-name=$CNAME --container-mounts=/mnt/vast:/mnt/vast,/etc/slurm:/etc/slurm"

run_one() {
    local label="$1"; local nnodes="$2"; local tpn="$3"; local nvls_off="$4"; shift 4
    local cmd="$@"
    local log="$OUT/${label}_nvls${nvls_off}${TAG}.log"
    echo "=== $label  NVSHMEM_DISABLE_NVLS=$nvls_off  (-N $nnodes --ntasks-per-node=$tpn) ==="
    timeout 90 $SRUN_BASE -N "$nnodes" --ntasks-per-node="$tpn" \
        bash -c "export LD_LIBRARY_PATH=$INST/lib:\$LD_LIBRARY_PATH;
                 export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;
                 export NVSHMEM_DISABLE_NVLS=$nvls_off;
                 $cmd" > "$log" 2>&1
    local ec=$?
    echo "  log: $log  (exit $ec)"
    [[ $ec -ne 0 ]] && tail -6 "$log" | sed 's/^/  | /'
    echo
}

# Map nvls_off→label: 0 = NVLS allowed (default), 1 = NVLS disabled
COLL=$INST/bin/perftest/device/coll
for nvls_off in 0 1; do
    for c in alltoall_latency bcast_latency fcollect_latency reduction_latency reducescatter_latency; do
        bin="$COLL/$c"
        [ -x "$bin" ] || continue
        # 'nvls0' = default (NVLS on); 'nvls1' = NVSHMEM_DISABLE_NVLS=1 (NVLS off)
        run_one "coll_intra_${c}_8r_msgsize"  1 8 "$nvls_off" "$bin -b 4 -e 33554432"
        run_one "coll_inter_${c}_16r_msgsize" 2 8 "$nvls_off" "$bin -b 4 -e 33554432"
    done
done

echo "Done."
