# Source this from inside the container at the multi-gpu repo root.
# Adjust the four *_HOME paths if the container layout differs.

# Container paths (verified on h200 cluster via repro/jacobi/recon.sh).
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export NCCL_HOME=${NCCL_HOME:-/usr}                       # NCCL header /usr/include/nccl.h, lib via ldconfig
export MPI_HOME=${MPI_HOME:-/usr/local/mpi}               # Open MPI 4.1.9a1 here (NOT /usr)
# NVSHMEM 3.6.5 from pip wheel — /opt/nvshmem on this container has a broken
# header symlink (points at /usr/include/nvshmem_13/ which doesn't exist), so
# we install nvidia-nvshmem-cu13 to a workspace prefix instead.
export NVSHMEM_HOME=${NVSHMEM_HOME:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-pip/nvidia/nvshmem}

# Build flags consumed by the variant Makefiles.
export GENCODE_SM90=1                                     # H100/H200 share sm_90
export USE_NVTX=1

# All transport + bootstrap plugins ship inside the pip wheel's lib/. CUDA 13's
# libcudart.so.13 lives in $CUDA_HOME/lib64; not always already on the path
# inside non-interactive bash sessions launched by srun, so add it explicitly.
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PATH=$CUDA_HOME/bin:$MPI_HOME/bin:${PATH:-}

# Default mpirun args for benchmark runs (-np supplied per call).
export MPIRUN_BASE_ARGS="--allow-run-as-root --oversubscribe -bind-to none"

# NVSHMEM symmetric heap and standard env knobs.
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-4G}
export NVSHMEM_BOOTSTRAP=MPI

if [[ "${VERBOSE:-0}" == "1" ]]; then
    echo "[setup_env] CUDA_HOME=$CUDA_HOME"
    echo "[setup_env] NCCL_HOME=$NCCL_HOME"
    echo "[setup_env] MPI_HOME=$MPI_HOME"
    echo "[setup_env] NVSHMEM_HOME=$NVSHMEM_HOME"
    which mpirun
    which nvcc
fi
