#!/bin/bash
# Prepare every node of a (new) hold job: local image + install + functional tests + ar_check + named container.
#   bash prep_hold.sh <JOBID>
set -euo pipefail
source /home/siyshen/workspace/b200-microbench/repro/scripts/env.sh
J=$1
NN=$(squeue -h -j "$J" -o %D)
srun --jobid="$J" --overlap -N "$NN" --ntasks-per-node=1 bash -c "
  bash $REPRO/scripts/stage_node.sh >/dev/null
  if [ ! -x $INST/nvshmem/tests/test_install/host/coll/reduce ]; then mkdir -p $INST/nvshmem/tests && tar -C $INST/nvshmem/tests -xzf $ARTIFACTS/nvshmem-tests.tgz; fi
  mkdir -p $LOCAL/validate && cp $ARTIFACTS/ar_check $LOCAL/validate/ar_check && chmod +x $LOCAL/validate/ar_check
  echo \$(hostname -s) staged"
srun --jobid="$J" --overlap -N "$NN" --ntasks-per-node=1 --export=NONE --container-image="$SQSH" --container-name=b200mb \
  --container-mounts="$MOUNTS" --no-container-remap-root bash -c 'echo $(hostname -s) container ok'
