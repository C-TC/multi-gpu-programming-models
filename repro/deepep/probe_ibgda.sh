#!/bin/bash
# Probe IBGDA prerequisites on a single node.
echo "=== HOSTNAME ==="
hostname
echo "=== ibstat HCAs ==="
ibstat 2>&1 | head -40
echo "=== ibv_devices ==="
ibv_devices 2>&1 | head
echo "=== mlx5 driver loaded ==="
lsmod 2>/dev/null | grep -E '^mlx5|^nvidia_peermem|^gdrdrv|^ib_uverbs|^ib_core' | head
echo "=== /dev/infiniband ==="
ls /dev/infiniband 2>&1
echo "=== /dev/gdrdrv ==="
ls -la /dev/gdrdrv 2>&1
echo "=== nvidia-peermem (newer way) ==="
modinfo nvidia_peermem 2>&1 | head -5
echo "=== devx access ==="
ls /sys/class/infiniband 2>&1
for dev in $(ls /sys/class/infiniband 2>/dev/null); do
    cap=$(cat /sys/class/infiniband/$dev/device/sriov_drivers_autoprobe 2>/dev/null || echo missing)
    echo "  $dev sriov_drivers_autoprobe=$cap"
done
echo "=== nvshmem-info ==="
/opt/nvshmem/bin/nvshmem-info -a 2>&1 | head -40
echo "=== Try IBGDA env to enable ==="
echo "Check if NVSHMEM_DISABLE_NVLS / NVSHMEM_IBGDA_NIC_HANDLER vars are needed"
