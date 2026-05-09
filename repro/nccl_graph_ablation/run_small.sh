#!/bin/bash
# Latency-bound version: small problem so per-iter cost is dominated by
# CUDA-graph launch overhead rather than compute. This is where
# NCCL_GRAPH_MIXING_SUPPORT=0 should pay off if anywhere.
#
# nx=128 keeps the jacobi kernel essentially trivial (~µs/iter); 5000 iter
# multiplies any per-launch saving by 5000.

set -u
REPO=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models
cd "$REPO"
. repro/jacobi/setup_env.sh

OUT_DIR="$REPO/repro/nccl_graph_ablation/results"
TS="$(date +%Y%m%d-%H%M%S)"
RESULTS_CSV="$OUT_DIR/results-graph-ablation-small-${TS}.csv"
LOG="$OUT_DIR/run-graph-ablation-small-${TS}.log"

echo "sweep,variant,nx,ny,niter,num_gpus,run_id,runtime_s,runtime_serial_s,nccl_graph_mixing" > "$RESULTS_CSV"
echo "Writing $RESULTS_CSV"

NY=16384; NITER=5000; REPS=5

run_one() {
    local mix="$1" ng="$2" nx="$3" rep="$4"
    DEVS=$(seq -s, 0 $((ng-1)))
    local out
    out=$(NCCL_GRAPH_MIXING_SUPPORT=$mix CUDA_VISIBLE_DEVICES=$DEVS \
          mpirun $MPIRUN_BASE_ARGS -np $ng \
              -x CUDA_VISIBLE_DEVICES -x NCCL_GRAPH_MIXING_SUPPORT \
              ./nccl_graphs/jacobi -csv -niter $NITER -nx $nx -ny $NY \
          2>>"$LOG")
    local line
    line=$(printf '%s\n' "$out" | grep -E '^nccl' | tail -1)
    [[ -z "$line" ]] && return
    local rt rt_ser
    rt=$(echo "$line" | awk -F',' '{gsub(/ /,"",$8); print $8}')
    rt_ser=$(echo "$line" | awk -F',' '{gsub(/ /,"",$9); print $9}')
    echo "Csmall,nccl_graphs,$nx,$NY,$NITER,$ng,$rep,$rt,$rt_ser,$mix" >> "$RESULTS_CSV"
    printf "  mix=%d ng=%d nx=%-4d rep=%d  rt=%ss\n" "$mix" "$ng" "$nx" "$rep" "$rt"
}

for nx in 128 512; do
    for ng in 4 8; do
        echo "--- nx=$nx ng=$ng ---"
        for r in $(seq 1 $REPS); do
            run_one 1 $ng $nx $r
            run_one 0 $ng $nx $r
        done
    done
done
echo "Done. $RESULTS_CSV"
