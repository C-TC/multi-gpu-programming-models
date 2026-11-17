#!/bin/bash
# Build NCCL 2.32.6-1, NVSHMEM 3.8.0-0 (+ reduction_focus perftest) and nccl-tests 2.21.1 for B200 (sm_100).
# Runs INSIDE the container on ONE node; everything is built on node-local disk ($LOCAL).
# The finished install tree is tarred to $ARTIFACTS (NFS) so other nodes/jobs can unpack it locally.
set -euo pipefail
source /home/siyshen/workspace/b200-microbench/repro/scripts/env.sh
NPROC=${NPROC:-96}
SRC=$B200MB/src
W=$LOCAL/work
mkdir -p "$W" "$INST" "$ARTIFACTS"
export CUDA_HOME=/usr/local/cuda
GENCODE="-gencode=arch=compute_100,code=sm_100"

echo "=== toolchain ==="; nvcc --version | tail -2; cmake --version | head -1; gcc --version | head -1
ls /opt/hpcx 2>/dev/null | head; which mpicc || true

echo "=== copy sources to local disk ==="
rm -rf "$W/src"; mkdir -p "$W/src"
cp -r "$SRC/nccl-2.32.6" "$SRC/nvshmem-3.8.0" "$SRC/nccl-tests-2.21.1" "$W/src/"

echo "=== NCCL 2.32.6-1 ==="
cd "$W/src/nccl-2.32.6"
make -j"$NPROC" src.build BUILDDIR="$INST/nccl" NVCC_GENCODE="$GENCODE" > "$W/nccl_build.log" 2>&1 || { tail -50 "$W/nccl_build.log"; exit 1; }
ls "$INST/nccl/lib"

echo "=== NVSHMEM 3.8.0-0 (+ reduction_focus) ==="
NS="$W/src/nvshmem-3.8.0"
cp "$REPRO/nvshmem_patches/reduction_focus.cu" "$NS/perftest/device/coll/"
grep -q 'reduction_focus.cu' "$NS/perftest/device/coll/CMakeLists.txt" || \
  sed -i 's/^nvshmem_add_perftest(reduction_latency.cu)/nvshmem_add_perftest(reduction_latency.cu)\nnvshmem_add_perftest(reduction_focus.cu)/' "$NS/perftest/device/coll/CMakeLists.txt"
grep -n 'reduction' "$NS/perftest/device/coll/CMakeLists.txt"
unset OPAL_PREFIX || true
GDR=OFF; [ -e /usr/include/gdrapi.h ] && GDR=ON
cmake -G Ninja -S "$NS" -B "$W/nvshmem-build" \
  -DCMAKE_INSTALL_PREFIX="$INST/nvshmem" -DNVSHMEM_PREFIX="$INST/nvshmem" \
  -DCMAKE_CUDA_ARCHITECTURES=100 \
  -DNVSHMEM_IBGDA_SUPPORT=ON -DNVSHMEM_IBRC_SUPPORT=ON -DNVSHMEM_USE_NCCL=OFF \
  -DNVSHMEM_MPI_SUPPORT=OFF -DNVSHMEM_PMIX_SUPPORT=OFF -DNVSHMEM_SHMEM_SUPPORT=OFF \
  -DNVSHMEM_USE_GDRCOPY=$GDR -DNVSHMEM_BUILD_TESTS=ON -DNVSHMEM_BUILD_EXAMPLES=OFF \
  -DNVSHMEM_BUILD_PYTHON_LIB=OFF -DNVSHMEM_BUILD_HYDRA_LAUNCHER=OFF -DNVSHMEM_BUILD_TXZ_PACKAGE=OFF \
  -DNVSHMEM_BUILD_BITCODE_LIBRARY=OFF > "$W/nvshmem_cmake.log" 2>&1 || { tail -60 "$W/nvshmem_cmake.log"; exit 1; }
grep -E 'NVSHMEM_(IBGDA|IBRC|USE_NCCL|MPI|USE_GDRCOPY|BUILD_TESTS)_?[A-Z]*:' "$W/nvshmem_cmake.log" || true
cmake --build "$W/nvshmem-build" --target install -j "$NPROC" > "$W/nvshmem_build.log" 2>&1 || { grep -n -m5 -i 'error' "$W/nvshmem_build.log"; tail -40 "$W/nvshmem_build.log"; exit 1; }
ls "$INST/nvshmem/bin/perftest/device/coll" "$INST/nvshmem/bin/perftest/device/pt-to-pt" | head -60
echo "NCCL symbols in libnvshmem_host (expect 0): $(nm -D "$INST/nvshmem/lib/libnvshmem_host.so" | grep -ci nccl || true)"

echo "=== nccl-tests 2.21.1 (MPI=1) ==="
MPI_HOME=${MPI_HOME:-$( [ -d /usr/local/mpi ] && echo /usr/local/mpi || echo /opt/hpcx/ompi )}
cd "$W/src/nccl-tests-2.21.1"
make -j"$NPROC" MPI=1 MPI_HOME="$MPI_HOME" NCCL_HOME="$INST/nccl" CUDA_HOME="$CUDA_HOME" \
     NVCC_GENCODE="$GENCODE" BUILDDIR="$INST/nccl-tests" > "$W/nccltests_build.log" 2>&1 || { tail -40 "$W/nccltests_build.log"; exit 1; }
ls "$INST/nccl-tests"
cp "$W"/*.log "$INST/" 2>/dev/null || true

echo "=== package install tree -> $ARTIFACTS ==="
tar -C "$LOCAL" -czf "$ARTIFACTS/b200mb-install.tgz.tmp" install && mv "$ARTIFACTS/b200mb-install.tgz.tmp" "$ARTIFACTS/b200mb-install.tgz"
ls -la "$ARTIFACTS"
echo BUILD_OK
