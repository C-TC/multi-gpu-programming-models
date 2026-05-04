#!/bin/bash
# Run V2 test_ep across 2 nodes via srun + torchrun-style env.
#
# V1 test_internode is INCOMPATIBLE with our 2x4 allocation: it asserts
# num_local_ranks == 8 (LEGACY_NUM_MAX_NVL_PEERS), so it requires 2x8=16 GPUs.
# V1 low_latency does not have this restriction and is run.

set -u
JOBID=${JOBID:?must set JOBID to your salloc job id}
DEEP_EP=/mnt/vast/home/tiancheng.chen/workspace/DeepEP
NVSHMEM_HOME=${NVSHMEM_HOME:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-pip/nvidia/nvshmem}
NCCL_PIP_LIB=/mnt/vast/home/tiancheng.chen/workspace/nccl-pip/nvidia/nccl/lib
CONTAINER=${CONTAINER:-/mnt/vast/containers/gpu_882f6e72.sqsh}
CONTAINER_NAME=${CONTAINER_NAME:-deepep_2x4_$JOBID}
TAG=${TAG:-}

echo "Priming container '$CONTAINER_NAME'..."
srun --jobid=$JOBID --overlap \
     --container-image=$CONTAINER --container-name=$CONTAINER_NAME \
     --container-workdir=$DEEP_EP --container-mounts=/mnt/vast:/mnt/vast \
     --container-remap-root --container-env=HOME \
     -N 2 --ntasks-per-node=1 true

SRUN_CONTAINER_ARGS="--container-name=$CONTAINER_NAME --container-mounts=/mnt/vast:/mnt/vast"
OUT=/mnt/vast/home/tiancheng.chen/workspace/multi-gpu-programming-models/repro/deepep/results
NODES=$(squeue -j $JOBID -h -o "%N" | head -1)
HEAD=$(scontrol show hostnames "$NODES" | head -1)
echo "Job $JOBID nodes: $NODES; head: $HEAD; tag=$TAG"

# Per-srun setup: every container is fresh, so re-install the right NCCL +
# uninstall the stale dist-packages deep_ep + symlink our locally-built _C.so.
SETUP="
pip install --quiet --no-deps 'nvidia-nccl-cu13>=2.30.4' 2>&1 | tail -1
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

run_test() {
    local label="$1"
    local cmd="$2"
    echo "=== $label ==="
    srun --jobid=$JOBID --mpi=pmi2 --overlap $SRUN_CONTAINER_ARGS -N 2 --ntasks-per-node=1 \
        bash -c "$ENV
cd $DEEP_EP && $cmd" \
        > "$OUT/${label}${TAG}.log" 2>&1
    echo "log: $OUT/${label}${TAG}.log  (exit $?)"
    tail -8 "$OUT/${label}${TAG}.log"
    echo
}

# V2 elastic ep, 2x4=8 ranks, explicit num-sms to bypass auto-detect bug
run_test v2_ep_2x4_topk6_sms24 \
    "python3 tests/elastic/test_ep.py --num-processes 4 --num-tokens 4096 --hidden 7168 --num-topk 6 --num-experts 256 --num-sms 24 --num-qps 8 --skip-check --test-first-only"

run_test v2_ep_2x4_topk8_sms24_e64 \
    "python3 tests/elastic/test_ep.py --num-processes 4 --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 64 --num-sms 24 --num-qps 8 --skip-check --test-first-only"

# V1 low-latency on 2 nodes (uses RDMA between nodes, NVLink within)
run_test v1_low_latency_2x4 \
    "python3 tests/legacy/test_low_latency.py --num-processes 4 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 64"

# V2 low-latency on 2 nodes 2x4 (was missing on prior cluster, added now)
run_test v2_ep_lowlat_2x4 \
    "python3 tests/elastic/test_ep.py --num-processes 4 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288 --num-sms 32 --num-qps 16 --prefer-overlap-with-compute 1 --skip-check --test-first-only"

echo "Done."
