#!/bin/bash
# Toolchain inventory for the current container. Run inside the GPU container.

echo "=== HOSTNAME ==="
hostname
echo
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -10
echo
echo "=== CUDA / nvcc ==="
nvcc --version 2>&1 | tail -3
ls /usr/local/cuda* -d 2>&1
echo
echo "=== MPI ==="
which mpirun mpicxx 2>&1
mpirun --version 2>&1 | head -3
echo
echo "=== NCCL ==="
ldconfig -p 2>&1 | grep libnccl | head -5
find /usr -name "nccl.h" 2>/dev/null | head -3
strings $(ldconfig -p | grep libnccl.so | head -1 | awk '{print $NF}') 2>/dev/null | grep -E "^NCCL [0-9]" | head -3
echo
echo "=== NVSHMEM ==="
ls /opt/nvshmem 2>&1 | head -10
ls /usr/lib/x86_64-linux-gnu/nvshmem 2>&1 | head -10
/opt/nvshmem/bin/nvshmem-info -a 2>&1 | head -25
echo
echo "=== InfiniBand ==="
ibv_devices 2>&1 | head -15
ls /dev/infiniband 2>&1
ls -la /dev/gdrdrv 2>&1
echo
echo "=== Kernel modules ==="
lsmod 2>/dev/null | grep -E '^mlx5|^nvidia_peermem|^gdrdrv' | head
echo
echo "=== Python / Torch / NCCL pip ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version())" 2>&1
pip show nvidia-nccl-cu13 2>/dev/null | head -3
echo
echo "=== Capabilities ==="
capsh --print 2>&1 | head -5
