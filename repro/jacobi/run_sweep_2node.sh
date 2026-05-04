#!/bin/bash
# 2-node Jacobi sweep — one srun per config (PMI2 has to be re-bootstrapped per
# MPI app, so we can't loop multiple jacobi runs inside a single srun step).
#
# srun container init runs ~45 s on this cluster regardless of payload, so each
# row in the CSV costs ~50 s. Total ≈ 50 min for 51 rows. Trim REPS or NXS if
# you need it shorter.
#
# Run from the LOGIN NODE with JOBID=<salloc id> set.
# NPN=4 (default) → 8 ranks total; NPN=8 → 16 ranks total.

set -u
JOBID=${JOBID:?must set JOBID to your salloc job id}
NPN=${NPN:-4}
CONTAINER=${CONTAINER:-/mnt/vast/containers/gpu_882f6e72.sqsh}
REPO=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models
NODES=$(squeue -j $JOBID -h -o "%N" | head -1)
HEAD=$(scontrol show hostnames "$NODES" | head -1)
echo "Job $JOBID nodes: $NODES; head: $HEAD; ntasks-per-node=$NPN"

OUT=$REPO/repro/jacobi/results
LOGD=$OUT/logs
mkdir -p "$OUT" "$LOGD"
TS="$(date +%Y%m%d-%H%M%S)"
CSV="$OUT/results-2node-${TS}.csv"
LOG="$LOGD/run-2node-${TS}.log"
echo "sweep,variant,nx,ny,niter,num_gpus,run_id,runtime_s,runtime_serial_s,extra" > "$CSV"
echo "Writing $CSV  (log $LOG)"

SRUN_ARGS="--jobid=$JOBID --mpi=pmi2 --overlap \
    --container-image=$CONTAINER \
    --container-workdir=$REPO \
    --container-mounts=/mnt/vast:/mnt/vast \
    --container-remap-root --container-env=HOME"

run_one() {
    local sweep=$1 variant=$2 nx=$3 ny=$4 niter=$5 npn=$6 rep=$7 extra=$8
    local extra_argv=""
    [[ "$extra" == *"block"* ]]   && extra_argv+=" -use_block_comm"
    [[ "$extra" == *"nbsync"* ]]  && extra_argv+=" -neighborhood_sync"
    local ng=$((2 * npn))
    local bin
    case "$variant" in
        nccl)        bin="./nccl/jacobi" ;;
        nccl_graphs) bin="./nccl_graphs/jacobi" ;;
        nvshmem|nvshmem_block|nvshmem_block_nbsync)
                     bin="./nvshmem/jacobi" ;;
        *) echo "unknown variant $variant"; return 1 ;;
    esac
    local label="${sweep}-${variant}-nx${nx}-ng${ng}-r${rep}"
    echo "  $label  extra='$extra_argv'" | tee -a "$LOG"

    out=$(srun $SRUN_ARGS -N 2 --ntasks-per-node=$npn \
            bash -c ". repro/jacobi/setup_env.sh > /dev/null 2>&1; $bin -csv -nx $nx -ny $ny -niter $niter $extra_argv" 2>>"$LOG")
    echo "    raw: $out" >> "$LOG"

    local line=$(echo "$out" | grep -E '^[a-z_-]+,' | tail -1)
    if [[ -z "$line" ]]; then
        echo "    NO CSV LINE; skipping row" | tee -a "$LOG"
        return
    fi
    local nf=$(echo "$line" | awk -F, '{print NF}')
    local runtime runtime_serial
    if [[ "$nf" == "6" ]]; then
        runtime=$(echo "$line" | awk -F, '{print $6}' | tr -d ' ')
        runtime_serial="$runtime"
    else
        runtime=$(echo "$line" | awk -F, '{print $8}' | tr -d ' ')
        runtime_serial=$(echo "$line" | awk -F, '{print $9}' | tr -d ' ')
    fi
    echo "$sweep,$variant,$nx,$ny,$niter,$ng,$rep,$runtime,$runtime_serial,$extra" >> "$CSV"
}

NITER=1000; REPS=${REPS:-3}

# Sweep A2 — scale-up at 16384^2.
for v in nccl nccl_graphs nvshmem nvshmem_block nvshmem_block_nbsync; do
    extra=""
    [[ "$v" == "nvshmem_block" ]]        && extra="block"
    [[ "$v" == "nvshmem_block_nbsync" ]] && extra="block,nbsync"
    for r in $(seq 1 $REPS); do
        run_one A2 "$v" 16384 16384 $NITER $NPN $r "$extra"
    done
done

# Sweep B2 — nx axis at full 2-node.
for nx in 2048 4096 8192 16384; do
    for v in nccl nccl_graphs nvshmem_block_nbsync; do
        extra=""
        [[ "$v" == "nvshmem_block_nbsync" ]] && extra="block,nbsync"
        for r in $(seq 1 $REPS); do
            run_one B2 "$v" $nx 16384 $NITER $NPN $r "$extra"
        done
    done
done

echo "Done. CSV: $CSV"
