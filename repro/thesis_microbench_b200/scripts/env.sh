# Shared settings for the B200 rerun of the paper's micro-benchmarks (sourced by all scripts).
# Storage policy (MARS cluster rules): nothing is executed from Lustre. Sources/scripts/results live on
# NFS home; the container image and all build/install trees are copied to node-local disk at job start.
B200MB=/home/siyshen/workspace/b200-microbench
REPRO=$B200MB/repro
RESULTS=$REPRO/results
ARTIFACTS=$B200MB/artifacts                 # install tarball (NFS, data only)
SQSH_SRC=/lustre/fsw/coreai_libraries_nccl/siyshen/workspace/containers/pytorch_26.06-py3.sqsh
LOCAL=${LOCAL:-/raid/scratch/siyshen/b200mb}  # node-local disk (same path on every node)
SQSH=$LOCAL/pytorch_26.06-py3.sqsh
INST=$LOCAL/install
NCCL_HOME_L=$INST/nccl
NVSHMEM_HOME_L=$INST/nvshmem
NCCL_TESTS_BIN=$INST/nccl-tests
MOUNTS="$LOCAL:$LOCAL,$B200MB:$B200MB"
