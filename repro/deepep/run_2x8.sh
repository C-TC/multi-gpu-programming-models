#!/bin/bash
# Run V1 internode HT, V2 ep, V1 LL multi-node, V2 LL multi-node
# on a 2-node x 8-GPU allocation.

set -u
JOBID=${JOBID:?must set JOBID to your salloc job id}
DEEP_EP=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/DeepEP
NVSHMEM_HOME=${NVSHMEM_HOME:-/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nvshmem-pip/nvidia/nvshmem}
NCCL_PIP_LIB=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/nccl-pip/nvidia/nccl/lib
CONTAINER=${CONTAINER:-/mnt/vast/containers/gpu_882f6e72.sqsh}
CONTAINER_NAME=${CONTAINER_NAME:-deepep_2x8_$JOBID}
TAG=${TAG:-}

# Prime the named container once (~30 s); subsequent srun calls reuse it (~11 s vs ~50 s).
echo "Priming container '$CONTAINER_NAME'..."
srun --jobid=$JOBID --overlap \
     --container-image=$CONTAINER --container-name=$CONTAINER_NAME \
     --container-workdir=$DEEP_EP --container-mounts=/mnt/vast:/mnt/vast \
     --container-remap-root --container-env=HOME \
     -N 2 --ntasks-per-node=1 true

SRUN_CONTAINER_ARGS="--container-name=$CONTAINER_NAME --container-mounts=/mnt/vast:/mnt/vast"
OUT=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/repro/deepep/results
NODES=$(squeue -j $JOBID -h -o "%N" | head -1)
HEAD=$(scontrol show hostnames "$NODES" | head -1)
echo "Job $JOBID nodes: $NODES; head: $HEAD; tag=$TAG"

# Per-srun setup that all ranks need.
SETUP="
pip install --quiet --no-deps 'nvidia-nccl-cu13>=2.30.4' 2>&1 | tail -1 >/dev/null
pip uninstall -y deep_ep > /dev/null 2>&1
[ -L $DEEP_EP/deep_ep/_C.cpython-312-x86_64-linux-gnu.so ] || \\
    ln -sf $DEEP_EP/build/lib.linux-x86_64-cpython-312/deep_ep/_C.cpython-312-x86_64-linux-gnu.so \\
           $DEEP_EP/deep_ep/_C.cpython-312-x86_64-linux-gnu.so
"

ENV="$SETUP
export PYTHONPATH=$DEEP_EP
export NVSHMEM_HOME=$NVSHMEM_HOME
export LD_LIBRARY_PATH=$NCCL_PIP_LIB:$NVSHMEM_HOME/lib:\$LD_LIBRARY_PATH
export NVSHMEM_SYMMETRIC_SIZE=8G
export MASTER_ADDR=$HEAD
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=\$SLURM_NODEID"

# Optional IBGDA tuning env (used only by V1 LL test runs)
IBGDA_ENV='
export NVSHMEM_HCA_LIST="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8"
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_DISABLE_NVLS=1
'

run_test() {
    local label="$1"
    local extra_env="$2"
    local cmd="$3"
    echo "=== $label ==="
    srun --jobid=$JOBID --mpi=pmi2 --overlap $SRUN_CONTAINER_ARGS -N 2 --ntasks-per-node=1 \
        bash -c "$ENV
$extra_env
cd $DEEP_EP && $cmd" \
        > "$OUT/${label}${TAG}.log" 2>&1
    local ec=$?
    echo "  log: $OUT/${label}${TAG}.log  (exit $ec)"
    if [[ $ec -ne 0 ]]; then
        echo "  -- last 8 lines --"
        tail -8 "$OUT/${label}${TAG}.log" | sed 's/^/  | /'
    else
        echo "  -- summary lines --"
        grep -E '\* EP:   0/|@ EP:   0/|Best dispatch|Best combine|bandwidth:' "$OUT/${label}${TAG}.log" | head -8 | sed 's/^/  | /'
    fi
    echo
}

# 1. V1 internode HT (now works because we have 8 GPU/node!)
run_test v1_internode_2x8 "" \
    "python3 tests/legacy/test_internode.py --num-processes 8 --num-tokens 4096 --hidden 7168 --num-experts 64"

# 2. V2 ep multi-node 2x8 = 16 ranks, matching V1 setup
run_test v2_ep_2x8_topk8_e64 "" \
    "python3 tests/elastic/test_ep.py --num-processes 8 --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 64 --num-sms 24 --num-qps 16 --skip-check --test-first-only"

# 3. V2 ep multi-node 2x8 with default config (topk=6, e=256)
run_test v2_ep_2x8_topk6_e256 "" \
    "python3 tests/elastic/test_ep.py --num-processes 8 --num-tokens 4096 --hidden 7168 --num-topk 6 --num-experts 256 --num-sms 32 --num-qps 16 --skip-check --test-first-only"

# 4. V1 LL multi-node, with explicit IBGDA env tuning
run_test v1_low_latency_2x8 "$IBGDA_ENV" \
    "python3 tests/legacy/test_low_latency.py --num-processes 8 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288"

# 5. V2 LL multi-node
run_test v2_ep_lowlat_2x8 "" \
    "python3 tests/elastic/test_ep.py --num-processes 8 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288 --num-sms 32 --num-qps 16 --prefer-overlap-with-compute 1 --skip-check --test-first-only"

echo "Done."
