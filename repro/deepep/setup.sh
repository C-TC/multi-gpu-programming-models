#!/bin/bash
# DeepEP container prep. Run inside the container before any test, on every node.
# Idempotent — symlink + pip-uninstall + pip-install are all no-ops if already done.
#
# What this fixes for the current cluster's container (h200 / gpu_882f6e72.sqsh):
#   1. Container's PyTorch links NCCL 2.28.8, but DeepEP V2 wants ≥2.30.4
#      → pip-install nvidia-nccl-cu13>=2.30.4 to a separate location, prepend to LD_LIBRARY_PATH
#   2. Container ships an outdated `deep_ep 1.2.1` flat-layout package in dist-packages
#      → uninstall it so our locally-built deep_ep takes precedence
#   3. /opt/nvshmem on this container has a broken header symlink → install
#      nvidia-nvshmem-cu13 to a workspace prefix and use that.
#   4. Symlink the locally-built _C.so into deep_ep/ so dev import works
#
# After sourcing this, $PYTHONPATH points at the local DeepEP checkout; you can run
# `python3 tests/...` or `python3 tests/elastic/test_ep.py ...` directly.

DEEP_EP=${DEEP_EP:-/mnt/vast/home/tiancheng.chen/workspace/DeepEP}
NVSHMEM_HOME=${NVSHMEM_HOME:-/mnt/vast/home/tiancheng.chen/workspace/nvshmem-pip/nvidia/nvshmem}

# 1. NCCL ABI ≥ 2.30.4 (V2 requirement)
pip install --quiet --no-deps 'nvidia-nccl-cu13>=2.30.4' 2>&1 | tail -1 >/dev/null

# 1b. NVSHMEM 3.x with usable headers — container's /opt/nvshmem ships a broken
#     symlink for nvshmem.h. Install our own copy if missing.
if [[ ! -f $NVSHMEM_HOME/include/nvshmem.h ]]; then
    pip install --quiet --no-deps --target=$(dirname $(dirname $NVSHMEM_HOME)) nvidia-nvshmem-cu13 >/dev/null 2>&1
    ln -sf libnvshmem_host.so.3 $NVSHMEM_HOME/lib/libnvshmem_host.so 2>/dev/null
fi

# 2. Remove stale dist-packages deep_ep
pip uninstall -y deep_ep > /dev/null 2>&1

# 3. Symlink locally-built _C.so into the package dir for dev mode
PY_TAG=cpython-312-x86_64-linux-gnu
SO=_C.${PY_TAG}.so
if [[ ! -L $DEEP_EP/deep_ep/$SO ]]; then
    ln -sf $DEEP_EP/build/lib.linux-x86_64-cpython-312/deep_ep/$SO \
           $DEEP_EP/deep_ep/$SO
fi

# 4. PYTHONPATH + LD_LIBRARY_PATH for the right NCCL + NVSHMEM bootstrap
export PYTHONPATH=$DEEP_EP
export NVSHMEM_HOME
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-8G}

if [[ "${VERBOSE:-0}" == "1" ]]; then
    echo "[deepep/setup] PYTHONPATH=$PYTHONPATH"
    echo "[deepep/setup] LD_LIBRARY_PATH starts with $(echo $LD_LIBRARY_PATH | cut -d: -f1)"
    python3 -c "import deep_ep; print('[deepep/setup] deep_ep', deep_ep.__version__, 'Buffer:', hasattr(deep_ep, 'Buffer'), 'ElasticBuffer:', hasattr(deep_ep, 'ElasticBuffer'))"
fi
