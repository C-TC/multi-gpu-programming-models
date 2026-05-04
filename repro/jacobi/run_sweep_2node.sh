#!/bin/bash
# 2-node Jacobi sweep — ONE srun, all configs in a single shell loop, mpirun
# inside for the inner parallelism. Avoids the per-srun container init cost
# (~45 s) by amortizing it across the whole sweep.
#
# Architecture:
#   * outer srun -N 2 --ntasks-per-node=1 launches one bash on each node.
#     On node 0 (rank 0) the bash runs the controller loop; on node 1 the
#     bash blocks waiting on a sentinel file so it stays alive (and the
#     container stays loaded) for the duration of the sweep.
#   * Controller loop iterates over (variant, nx, ny, niter, rep) configs
#     and shells out to `mpirun -np N -hostfile H ...` for each. mpirun's
#     own bootstrap (orte over ssh — pyxis container exposes ssh between
#     nodes via --container-mounts of /run/munge etc.) handles the per-call
#     parallelism without going back through srun.
#   * Tear-down: controller writes the sentinel file when all configs done.
#
# If ssh between nodes inside the container is unavailable, mpirun won't
# work and this script falls back to invoking srun --jobid for each config
# (slow path, set FALLBACK=1 to force).

set -u
JOBID=${JOBID:?must set JOBID to your salloc job id}
NPN=${NPN:-4}
REPS=${REPS:-3}
CONTAINER=${CONTAINER:-/mnt/vast/containers/gpu_882f6e72.sqsh}
FALLBACK=${FALLBACK:-0}
REPO=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models
NODES=$(squeue -j $JOBID -h -o "%N" | head -1)
HEAD=$(scontrol show hostnames "$NODES" | head -1)
ALL_HOSTS=$(scontrol show hostnames "$NODES" | paste -sd,)
echo "Job $JOBID nodes: $NODES; head: $HEAD; ntasks-per-node=$NPN"

OUT=$REPO/repro/jacobi/results
LOGD=$OUT/logs
mkdir -p "$OUT" "$LOGD"
TS="$(date +%Y%m%d-%H%M%S)"
CSV="$OUT/results-2node-${TS}.csv"
LOG="$LOGD/run-2node-${TS}.log"
echo "sweep,variant,nx,ny,niter,num_gpus,run_id,runtime_s,runtime_serial_s,extra" > "$CSV"
echo "Writing $CSV  (log $LOG)"

NG=$((2 * NPN))

# Build the configs file (mounted on /mnt/vast so the container sees it).
CONFIGS=$OUT/sweep2node-configs-${TS}.txt
{
    for v in nccl nccl_graphs nvshmem nvshmem_block nvshmem_block_nbsync; do
        bin=./nccl/jacobi
        extra=""
        case "$v" in
            nccl_graphs) bin=./nccl_graphs/jacobi ;;
            nvshmem|nvshmem_block|nvshmem_block_nbsync) bin=./nvshmem/jacobi ;;
        esac
        [[ "$v" == nvshmem_block || "$v" == nvshmem_block_nbsync ]] && extra="-use_block_comm"
        [[ "$v" == nvshmem_block_nbsync ]] && extra="$extra -neighborhood_sync"
        for r in $(seq 1 $REPS); do
            echo "A2 $v $bin 16384 16384 1000 $r $extra"
        done
    done
    for nx in 2048 4096 8192 16384; do
        for v in nccl nccl_graphs nvshmem_block_nbsync; do
            bin=./nccl/jacobi
            extra=""
            case "$v" in
                nccl_graphs)         bin=./nccl_graphs/jacobi ;;
                nvshmem_block_nbsync) bin=./nvshmem/jacobi; extra="-use_block_comm -neighborhood_sync" ;;
            esac
            for r in $(seq 1 $REPS); do
                echo "B2 $v $bin $nx 16384 1000 $r $extra"
            done
        done
    done
} > "$CONFIGS"
echo "Configs: $(wc -l < $CONFIGS) runs."

if [[ "$FALLBACK" == "1" ]]; then
    # Slow path: one srun per config (~50 s/call).
    echo "FALLBACK=1: per-config srun"
    SRUN_ARGS="--jobid=$JOBID --mpi=pmi2 --overlap --container-image=$CONTAINER --container-workdir=$REPO --container-mounts=/mnt/vast:/mnt/vast --container-remap-root --container-env=HOME"
    while read sweep variant bin nx ny niter rep extra; do
        out=$(srun $SRUN_ARGS -N 2 --ntasks-per-node=$NPN \
                bash -c ". repro/jacobi/setup_env.sh > /dev/null 2>&1; $bin -csv -nx $nx -ny $ny -niter $niter $extra" 2>>"$LOG")
        line=$(echo "$out" | grep -E '^[a-z_-]+,' | tail -1)
        nf=$(echo "$line" | awk -F, '{print NF}')
        if [[ "$nf" == "6" ]]; then
            rt=$(echo "$line" | awk -F, '{print $6}' | tr -d ' '); rtser=$rt
        else
            rt=$(echo "$line" | awk -F, '{print $8}' | tr -d ' ')
            rtser=$(echo "$line" | awk -F, '{print $9}' | tr -d ' ')
        fi
        echo "$sweep,$variant,$nx,$ny,$niter,$NG,$rep,$rt,$rtser,$extra" >> "$CSV"
        echo "  $sweep-$variant-nx$nx-r$rep  rt=$rt"
    done < "$CONFIGS"
    rm -f "$CONFIGS"
    exit 0
fi

# Fast path: ONE srun -N 2 --ntasks-per-node=1 spawns a controller bash on
# each node. The whole sweep runs inside via mpirun-with-srun (using
# `srun --overlap` from inside but without re-loading the container — the
# step is shared with the parent).
RUNNER=$OUT/sweep2node-runner-${TS}.sh
cat > "$RUNNER" <<INNER_EOF
#!/bin/bash
# Per-rank inner script. Rank 0 is the controller, others idle until done.
set -u
. $REPO/repro/jacobi/setup_env.sh > /dev/null 2>&1
SENTINEL=$OUT/sweep2node-done-${TS}
if [[ "\${SLURM_PROCID:-0}" != "0" ]]; then
    # idle rank: just wait for sentinel (keeps container alive)
    until [[ -f "\$SENTINEL" ]]; do sleep 5; done
    exit 0
fi

# Controller (rank 0). Spawn each test with srun --overlap inside the parent step.
# srun without --container-image inside the parent srun's container REUSES
# the container (much faster than re-loading per call).
SRUN_INNER="srun --jobid=$JOBID --mpi=pmi2 --overlap -N 2 --ntasks-per-node=$NPN"
while read sweep variant bin nx ny niter rep extra; do
    out=\$(\$SRUN_INNER bash -c "cd $REPO; \$bin -csv -nx \$nx -ny \$ny -niter \$niter \$extra" 2>>"$LOG")
    line=\$(echo "\$out" | grep -E '^[a-z_-]+,' | tail -1)
    nf=\$(echo "\$line" | awk -F, '{print NF}')
    if [[ "\$nf" == "6" ]]; then
        rt=\$(echo "\$line" | awk -F, '{print \$6}' | tr -d ' '); rtser=\$rt
    else
        rt=\$(echo "\$line" | awk -F, '{print \$8}' | tr -d ' ')
        rtser=\$(echo "\$line" | awk -F, '{print \$9}' | tr -d ' ')
    fi
    echo "\$sweep,\$variant,\$nx,\$ny,\$niter,$NG,\$rep,\$rt,\$rtser,\$extra" >> "$CSV"
    echo "  \$sweep-\$variant-nx\$nx-r\$rep  rt=\$rt"
done < "$CONFIGS"
touch "\$SENTINEL"
INNER_EOF
chmod +x "$RUNNER"

srun --jobid=$JOBID --mpi=none --overlap \
     --container-image=$CONTAINER \
     --container-workdir=$REPO \
     --container-mounts=/mnt/vast:/mnt/vast \
     --container-remap-root --container-env=HOME \
     -N 2 --ntasks-per-node=1 \
     "$RUNNER" 2>>"$LOG"

rm -f "$RUNNER" "$CONFIGS" "$OUT/sweep2node-done-${TS}"
echo "Done. CSV: $CSV"
