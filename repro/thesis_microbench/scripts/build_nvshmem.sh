#!/bin/bash
# Build NVSHMEM 3.3.9-ibp from internal fork (~/workspace/nvshmem-3.3.9-ibp,
# branch 3.3.9-ibp) with perftest binaries enabled.
#
# Run inside the H200 container (any one node — does not need 2 nodes for build).
# Output: $WORKSPACE/install/{lib,bin/perftest/...}

set -eu
SRC=${SRC:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp}
WORKSPACE=${WORKSPACE:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-3.3.9-ibp-build}

mkdir -p "$WORKSPACE"

export CUDA_HOME=/usr/local/cuda
# OPAL_PREFIX=/opt/hpcx/ompi is set in this container's env, but FindMPI
# produces malformed -I paths from it. Drop it and use PMI2 bootstrap (no MPI).
unset OPAL_PREFIX

# Features: enable transports we need; skip MPI bootstrap (pmi2 bootstrap + srun --mpi=pmi2 is enough).
export NVSHMEM_IBGDA_SUPPORT=1   # cross-node IBGDA (kernel-side RDMA)
export NVSHMEM_IBRC_SUPPORT=1    # cross-node IBRC (host proxy) -- alternative for comparison
export NVSHMEM_USE_GDRCOPY=0     # gdrcopy headers absent in this container
export NVSHMEM_MPI_SUPPORT=0     # avoid the broken hpcx FindMPI; pmi2 bootstrap is enough
export NVSHMEM_BUILD_TESTS=1     # built-in tests + perftest in-tree
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0

echo "=== Step 1: build NVSHMEM 3.3.9-ibp ==="
cmake -G Ninja -S "$SRC" -B "$WORKSPACE/build" \
    -DCMAKE_INSTALL_PREFIX="$WORKSPACE/install" \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DNVSHMEM_PREFIX="$WORKSPACE/install"
cmake --build "$WORKSPACE/build" --target install

# Step 2 (standalone perftest build) is unnecessary -- Step 1 with NVSHMEM_BUILD_TESTS=1
# already builds the device-side perftest binaries in-tree and installs them under
# $install/bin/perftest/{device,host}/{pt-to-pt,coll}/. The standalone perftest
# project also fails on this container (CMake imports a malformed cccl include path
# from the installed nvshmem_device target), so we skip it.

echo "=== Done ==="
echo "Install prefix: $WORKSPACE/install"
ls "$WORKSPACE/install/bin/perftest/device/pt-to-pt/" 2>&1 | head -20
ls "$WORKSPACE/install/bin/perftest/device/coll/" 2>&1 | head -20
