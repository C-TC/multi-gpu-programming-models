#!/bin/bash
# DeepEP V1 (legacy NVSHMEM) tests using the **working** version combination
# we discovered after looking at mistral's `runtime/vllm-internal/tools/ep_kernels/`:
#
#   * DeepEP commit `73b6ea4` (pre-V2 layout: csrc/kernels/, deep_ep/buffer.py,
#     tests/test_*.py — no `legacy/` subdir). The V2 release `b306af0` regressed
#     the V1 LL kernel (cudaErrorIllegalAddress) and the V1 internode wrapper
#     (TypeError); 73b6ea4 has neither regression.
#   * NVSHMEM 3.4.5 (cu13 wheel, install with `pip install nvidia-nvshmem-cu13==3.4.5`).
#     The reason this version specifically: NVSHMEM 3.4.5 has v1 device_state
#     struct layout (matching DeepEP's bundled ibgda_device.cuh), while
#     NVSHMEM 3.5+ switched to v2.
#   * `NVSHMEM_HCA_PREFIX=` (empty string) — bypasses the IBGDA enumeration
#     filter that defaults to `mlx5*`. This cluster's IB devices are named
#     `ibp0..ibp7` via ibverbs even though /sys/class/infiniband shows mlx5_*;
#     setting the prefix to empty allows any device.
#   * NVSHMEM_IB_ENABLE_IBGDA=1, NVSHMEM_DISABLE_NVLS=1 (standard IBGDA env).
#
# Run from the LOGIN NODE with JOBID=<salloc id> set; this script handles
# the pyxis container priming and the full run.

set -u
JOBID=${JOBID:?must set JOBID to your salloc job id}
CONTAINER=${CONTAINER:-/mnt/vast/containers/gpu_882f6e72.sqsh}
CONTAINER_NAME=${CONTAINER_NAME:-deepep_v1_$JOBID}
DEEP_EP=${DEEP_EP:-/mnt/vast/home/tiancheng.chen/workspace/DeepEP-pre-v2}
NVSHMEM_HOME=${NVSHMEM_HOME:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-pip-3.4.5/nvidia/nvshmem}
TAG=${TAG:-}
OUT=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/deepep/results
NODES=$(squeue -j $JOBID -h -o "%N" | head -1)
HEAD=$(scontrol show hostnames "$NODES" | head -1)
mkdir -p "$OUT"
echo "Job $JOBID nodes: $NODES; head: $HEAD; container_name=$CONTAINER_NAME; tag=$TAG"
echo "DeepEP: $DEEP_EP (pre-V2 commit 73b6ea4)"
echo "NVSHMEM: $NVSHMEM_HOME (3.4.5)"

# Step 1: prime the named container on both nodes (~30 s once).
echo "Priming container '$CONTAINER_NAME'..."
srun --jobid=$JOBID --overlap \
     --container-image=$CONTAINER --container-name=$CONTAINER_NAME \
     --container-workdir=$DEEP_EP \
     --container-mounts=/mnt/vast:/mnt/vast --container-remap-root --container-env=HOME \
     -N 2 --ntasks-per-node=1 true

# Step 2: ensure DeepEP-pre-v2 is built against NVSHMEM 3.4.5.
SO=$DEEP_EP/build/lib.linux-x86_64-cpython-312/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so
if [[ ! -f "$SO" ]]; then
    echo "Building DeepEP-pre-v2 against NVSHMEM 3.4.5..."
    srun --jobid=$JOBID --overlap \
         --container-name=$CONTAINER_NAME \
         --container-mounts=/mnt/vast:/mnt/vast \
         -N 1 --ntasks-per-node=1 \
         bash -c "cd $DEEP_EP && rm -rf build && CPATH=/usr/local/cuda/include/cccl NVSHMEM_DIR=$NVSHMEM_HOME TORCH_CUDA_ARCH_LIST='9.0' python3 setup.py build > /tmp/deepep_v1_build.log 2>&1; ls $SO"
fi
ln -sf "$SO" "$DEEP_EP/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so" 2>/dev/null

# Step 3: shared env that all V1 tests need.
ENV_VARS="
pip uninstall -y deep_ep > /dev/null 2>&1
export PYTHONPATH=$DEEP_EP
export NVSHMEM_HOME=$NVSHMEM_HOME
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:\$LD_LIBRARY_PATH
export NVSHMEM_SYMMETRIC_SIZE=8G
export MASTER_ADDR=$HEAD
export WORLD_SIZE=2
export RANK=\$SLURM_NODEID
# THE working IBGDA env on this cluster:
export NVSHMEM_HCA_PREFIX=
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_DISABLE_NVLS=1
"

run_one() {
    local label=$1; local nnodes=$2; local cmd=$3; local mport=${4:-29560}
    echo "=== $label (nnodes=$nnodes, port=$mport) ==="
    srun --jobid=$JOBID --mpi=pmi2 --overlap \
         --container-name=$CONTAINER_NAME \
         --container-mounts=/mnt/vast:/mnt/vast \
         -N $nnodes --ntasks-per-node=1 \
         bash -c "$ENV_VARS
export MASTER_PORT=$mport
cd $DEEP_EP
$cmd" \
         > "$OUT/${label}${TAG}.log" 2>&1
    local ec=$?
    echo "  log: $OUT/${label}${TAG}.log  (exit $ec)"
    if [[ $ec -eq 0 ]]; then
        # Pull a few summary lines depending on test type.
        grep -E "Best (dispatch|combine)|Dispatch \+ combine bandwidth|Dispatch bandwidth.*Combine bandwidth" \
            "$OUT/${label}${TAG}.log" | head -3 | sed 's/^/  | /'
    else
        echo "  -- last 6 lines --"
        tail -6 "$OUT/${label}${TAG}.log" | sed 's/^/  | /'
    fi
    echo
}

# V1 intranode (1 node, 8 GPU, NVLink only — should match V2-release V1 result)
run_one v1_intranode_mistral_recipe 1 \
    "python3 tests/test_intranode.py --num-processes 8" 29560

# V1 internode HT (2 nodes × 8 GPU = 16 ranks, NVSHMEM IBGDA cross-node)
run_one v1_internode_2x8_mistral_recipe 2 \
    "python3 tests/test_internode.py --num-processes 8 --num-tokens 4096 --hidden 7168 --num-experts 64" 29561

# V1 low-latency (2 nodes × 8 GPU = 16 ranks, NVSHMEM IBGDA cross-node)
run_one v1_low_latency_2x8_mistral_recipe 2 \
    "python3 tests/test_low_latency.py --num-processes 8 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288" 29562

# V1 low-latency 2×4 (8 ranks total — half the GPUs)
run_one v1_low_latency_2x4_mistral_recipe 2 \
    "python3 tests/test_low_latency.py --num-processes 4 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 64" 29563

echo "Done."
