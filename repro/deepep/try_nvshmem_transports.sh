#!/bin/bash
# Try a few NVSHMEM transport / HCA configurations to get V1 LL multi-node working.

set -u
JOBID=${JOBID:?must set JOBID}
DEEP_EP=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/DeepEP
OUT=/mnt/vast/home/tiancheng.chen/workspace/nccl-nvshmem-repro/multi-gpu-programming-models/repro/deepep/results
NODES=$(squeue -j $JOBID -h -o "%N" | head -1)
HEAD=$(scontrol show hostnames "$NODES" | head -1)

SETUP="
pip install --quiet --no-deps 'nvidia-nccl-cu13>=2.30.4' 2>&1 | tail -1 >/dev/null
pip uninstall -y deep_ep > /dev/null 2>&1
[ -L $DEEP_EP/deep_ep/_C.cpython-312-x86_64-linux-gnu.so ] || \\
    ln -sf $DEEP_EP/build/lib.linux-x86_64-cpython-312/deep_ep/_C.cpython-312-x86_64-linux-gnu.so \\
           $DEEP_EP/deep_ep/_C.cpython-312-x86_64-linux-gnu.so
"

BASE_ENV="$SETUP
export PYTHONPATH=$DEEP_EP
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/lib/x86_64-linux-gnu/nvshmem/13:/opt/nvshmem/lib:\$LD_LIBRARY_PATH
export NVSHMEM_SYMMETRIC_SIZE=8G
export MASTER_ADDR=$HEAD
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=\$SLURM_NODEID"

CMD="python3 tests/legacy/test_low_latency.py --num-processes 8 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288"

try() {
    local label="$1"
    local nv_env="$2"
    echo "=== $label ==="
    srun --jobid=$JOBID --mpi=pmi2 -N 2 --ntasks-per-node=1 \
        bash -c "$BASE_ENV
$nv_env
cd $DEEP_EP && $CMD" \
        > "$OUT/v1ll_2x8_${label}.log" 2>&1
    local ec=$?
    echo "  exit=$ec  log: $OUT/v1ll_2x8_${label}.log"
    if [[ $ec -eq 0 ]]; then
        echo "  SUCCESS, summary:"
        grep -E "bandwidth|avg_t" "$OUT/v1ll_2x8_${label}.log" | head -3 | sed 's/^/  | /'
    else
        echo "  failure tail:"
        grep -E "init failed|not accessible|building transport|^Allocating|exit code" "$OUT/v1ll_2x8_${label}.log" | head -3 | sed 's/^/  | /'
    fi
}

# Force IBRC (no IBGDA), let NVSHMEM auto-pick HCAs
try ibrc_only "
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_DISABLE_NVLS=1
"

# Force IBGDA with explicit HCA list (8 mlx5 GPU NICs, skipping mlx5_5 which had different sriov state)
try ibgda_hca '
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_6:1,mlx5_7:1,mlx5_8:1
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_DISABLE_NVLS=1
'

# Explicit DEVX path
try ibgda_devx '
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_IBGDA_NUM_RC_PER_PE=2
export NVSHMEM_IBGDA_NUM_DCI=2
export NVSHMEM_DISABLE_NVLS=1
'

# Last resort: libfabric
try libfabric '
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=verbs
export NVSHMEM_DISABLE_NVLS=1
'

echo "Done."
