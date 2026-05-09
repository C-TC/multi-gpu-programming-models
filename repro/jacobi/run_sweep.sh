#!/bin/bash
# 1-node Jacobi sweep on a single 8-GPU H100 node.
#
# Sweep A: scale-up — 16384^2, niter=1000, GPUs in {1,2,4,8}, all variants, 3 reps each.
# Sweep B: nx-axis  — GPUs=8, niter=1000, nx in {2048,4096,8192,16384}, ny=16384,
#                     subset of variants {nccl,nccl_graphs,nvshmem,nvshmem-block-nbsync},
#                     3 reps each.
#
# CSV columns: sweep,variant,nx,ny,niter,num_gpus,run_id,runtime_s,runtime_serial_s,extra
# Run from inside the container after sourcing repro/jacobi/setup_env.sh.

set -u
REPO=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models
cd "$REPO"
. repro/jacobi/setup_env.sh

OUT=$REPO/repro/jacobi/results
LOGD=$OUT/logs
mkdir -p "$OUT" "$LOGD"
TS="$(date +%Y%m%d-%H%M%S)"
CSV="$OUT/results-1node-${TS}.csv"
LOG="$LOGD/run-1node-${TS}.log"

echo "sweep,variant,nx,ny,niter,num_gpus,run_id,runtime_s,runtime_serial_s,extra" > "$CSV"
echo "Writing $CSV  (log $LOG)"

# Run a single configuration; appends to CSV. Parses the variant's csv-mode output.
# Output line shape (from the binaries):  <variant>, <runtime_serial>, <runtime_parallel>, ...
run_one() {
    local sweep=$1 variant=$2 nx=$3 ny=$4 niter=$5 ng=$6 rep=$7 extra=$8
    local extra_argv=""
    [[ "$extra" == *"block"* ]]   && extra_argv+=" -use_block_comm"
    [[ "$extra" == *"nbsync"* ]]  && extra_argv+=" -neighborhood_sync"

    local DEVS=$(seq -s, 0 $((ng-1)))
    local bin extra_env
    case "$variant" in
        single_gpu)   bin="./single_gpu/jacobi"; ng=1; DEVS=0 ;;
        mpi)          bin="./mpi/jacobi" ;;
        mpi_overlap)  bin="./mpi_overlap/jacobi" ;;
        nccl)         bin="./nccl/jacobi" ;;
        nccl_graphs)  bin="./nccl_graphs/jacobi" ;;
        nccl_overlap) bin="./nccl_overlap/jacobi" ;;
        nvshmem|nvshmem_block|nvshmem_block_nbsync)
                      bin="./nvshmem/jacobi" ;;
        *) echo "unknown variant $variant"; return 1 ;;
    esac

    local label="${sweep}-${variant}-nx${nx}-ng${ng}-r${rep}"
    echo "  $label  extra='$extra_argv'" | tee -a "$LOG"

    local out
    if [[ "$variant" == nvshmem* ]]; then
        out=$(CUDA_VISIBLE_DEVICES=$DEVS \
              mpirun $MPIRUN_BASE_ARGS -np $ng \
                -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x NVSHMEM_SYMMETRIC_SIZE \
                $bin -csv -nx $nx -ny $ny -niter $niter $extra_argv 2>>"$LOG")
    else
        out=$(CUDA_VISIBLE_DEVICES=$DEVS \
              mpirun $MPIRUN_BASE_ARGS -np $ng \
                -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH \
                $bin -csv -nx $nx -ny $ny -niter $niter $extra_argv 2>>"$LOG")
    fi
    echo "    raw: $out" >> "$LOG"

    # Pull the last comma-separated line. Binary variant id may contain '-' (e.g. "nvshmem-use_block_comm").
    local line=$(echo "$out" | grep -E '^[a-z_-]+,' | tail -1)
    if [[ -z "$line" ]]; then
        echo "    NO CSV LINE; skipping row" | tee -a "$LOG"
        return
    fi
    # Field count distinguishes the two csv formats:
    #   single_gpu: variant, nx, ny, niter, nccheck, runtime              (NF=6)
    #   multi-GPU : variant, nx, ny, niter, nccheck, size, 1, runtime, runtime_serial (NF=9)
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

# Sweep A — scale-up at 16384^2 / niter=1000
NX=16384; NY=16384; NITER=1000; REPS=3
for ng in 1 2 4 8; do
    for v in single_gpu mpi mpi_overlap nccl nccl_graphs nccl_overlap nvshmem; do
        # single_gpu only at 1 GPU; nvshmem default is no extra flags.
        if [[ "$v" == single_gpu && "$ng" -ne 1 ]]; then continue; fi
        if [[ "$v" != single_gpu && "$ng" -eq 1 ]]; then continue; fi
        for r in $(seq 1 $REPS); do
            run_one A "$v" $NX $NY $NITER $ng $r ""
        done
    done
    # nvshmem variants with -use_block_comm and -nbsync
    if [[ "$ng" -ne 1 ]]; then
        for r in $(seq 1 $REPS); do
            run_one A nvshmem_block        $NX $NY $NITER $ng $r "block"
            run_one A nvshmem_block_nbsync $NX $NY $NITER $ng $r "block,nbsync"
        done
    fi
done

# Sweep B — nx axis at 8 GPU
NG=8
for NX in 2048 4096 8192 16384; do
    for v in nccl nccl_graphs nvshmem; do
        for r in $(seq 1 $REPS); do
            run_one B "$v" $NX $NY $NITER $NG $r ""
        done
    done
    for r in $(seq 1 $REPS); do
        run_one B nvshmem_block_nbsync $NX $NY $NITER $NG $r "block,nbsync"
    done
done

echo "Done. CSV: $CSV"
