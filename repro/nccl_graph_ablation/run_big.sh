#!/bin/bash
# Ablate NCCL_GRAPH_MIXING_SUPPORT for nccl_graphs at the standard point.
# Compares: default (1, enabled) vs disabled (0).
# Same shape as Sweep B: nx=ny=16384, niter=1000, 1/2/4/8 GPUs, 3 reps.
#
# Run from inside the container at the repo root.

set -u
REPO=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models
cd "$REPO"
. repro/jacobi/setup_env.sh

OUT_DIR="$REPO/repro/nccl_graph_ablation/results"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
RESULTS_CSV="$OUT_DIR/results-graph-ablation-${TS}.csv"
LOG="$OUT_DIR/run-graph-ablation-${TS}.log"

echo "sweep,variant,nx,ny,niter,num_gpus,run_id,runtime_s,runtime_serial_s,nccl_graph_mixing" > "$RESULTS_CSV"
echo "Writing $RESULTS_CSV"

NX=16384; NY=16384; NITER=1000; REPS=3

run_one() {
    local mix="$1" ng="$2" rep="$3"
    DEVS=$(seq -s, 0 $((ng-1)))
    local out
    out=$(NCCL_GRAPH_MIXING_SUPPORT=$mix CUDA_VISIBLE_DEVICES=$DEVS \
          mpirun $MPIRUN_BASE_ARGS -np $ng \
              -x CUDA_VISIBLE_DEVICES -x NCCL_GRAPH_MIXING_SUPPORT \
              ./nccl_graphs/jacobi -csv -niter $NITER -nx $NX -ny $NY \
          2>>"$LOG")
    local line
    line=$(printf '%s\n' "$out" | grep -E '^nccl' | tail -1)
    if [[ -z "$line" ]]; then
        echo "[run] empty for mix=$mix ng=$ng rep=$rep" | tee -a "$LOG"
        return
    fi
    local rt rt_ser
    rt=$(echo "$line" | awk -F',' '{gsub(/ /,"",$8); print $8}')
    rt_ser=$(echo "$line" | awk -F',' '{gsub(/ /,"",$9); print $9}')
    echo "Cgraph,nccl_graphs,$NX,$NY,$NITER,$ng,$rep,$rt,$rt_ser,$mix" >> "$RESULTS_CSV"
    printf "  mix=%d ng=%d rep=%d  rt=%ss\n" "$mix" "$ng" "$rep" "$rt"
}

for ng in 1 2 4 8; do
    echo "--- ng=$ng ---"
    for r in $(seq 1 $REPS); do
        run_one 1 $ng $r
        run_one 0 $ng $r
    done
done

echo "Done. $RESULTS_CSV"
