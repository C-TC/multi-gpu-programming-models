#!/bin/bash
# 2-node Jacobi sweep, optimized for the H200 cluster's per-srun overhead.
#
# Trick: `srun --container-name=<name>` lets pyxis reuse the same container
# instance across srun calls. The first srun loads the container (~32 s),
# every subsequent srun attaches to the named container (~11 s instead of
# 50+ s). For 51 runs that's ~10 min instead of ~45 min.
#
# Architecture is still one srun per (variant, nx, rep) — jacobi binaries
# do MPI_Init/Finalize per process, and srun's PMI2 server only handles
# one MPI session per step. So we can't fold all configs into a single srun
# bash loop. But the per-srun cost is now low enough that this isn't
# painful.
#
# Run from the LOGIN NODE with JOBID=<salloc id> set.
# NPN=4 (default) → 8 ranks total; NPN=8 → 16 ranks total.

set -u
JOBID=${JOBID:?must set JOBID to your salloc job id}
NPN=${NPN:-4}
REPS=${REPS:-3}
CONTAINER=${CONTAINER:-/mnt/vast/containers/gpu_882f6e72.sqsh}
CONTAINER_NAME=${CONTAINER_NAME:-jacobi_sweep_$JOBID}
REPO=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models
NODES=$(squeue -j $JOBID -h -o "%N" | head -1)
HEAD=$(scontrol show hostnames "$NODES" | head -1)
echo "Job $JOBID nodes: $NODES; head: $HEAD; ntasks-per-node=$NPN; container_name=$CONTAINER_NAME"

OUT=$REPO/repro/jacobi/results
LOGD=$OUT/logs
mkdir -p "$OUT" "$LOGD"
TS="$(date +%Y%m%d-%H%M%S)"
CSV="$OUT/results-2node-${TS}.csv"
LOG="$LOGD/run-2node-${TS}.log"
echo "sweep,variant,nx,ny,niter,num_gpus,run_id,runtime_s,runtime_serial_s,extra" > "$CSV"
echo "Writing $CSV  (log $LOG)"

# Step 1: prime the named container on both nodes (~30 s, one-time).
echo "Priming container '$CONTAINER_NAME'..."
srun --jobid=$JOBID --overlap \
     --container-image=$CONTAINER \
     --container-name=$CONTAINER_NAME \
     --container-mounts=/mnt/vast:/mnt/vast \
     --container-remap-root --container-env=HOME \
     -N 2 --ntasks-per-node=1 \
     true 2>>"$LOG"

# Step 2: per-config srun, reusing the named container (~11 s/call).
SRUN_ARGS="--jobid=$JOBID --mpi=pmi2 --overlap \
    --container-name=$CONTAINER_NAME \
    --container-mounts=/mnt/vast:/mnt/vast"

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
            bash -c ". $REPO/repro/jacobi/setup_env.sh > /dev/null 2>&1; cd $REPO; $bin -csv -nx $nx -ny $ny -niter $niter $extra_argv" 2>>"$LOG")
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

NITER=1000

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
