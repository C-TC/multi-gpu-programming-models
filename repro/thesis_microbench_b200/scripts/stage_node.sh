#!/bin/bash
# Per-node staging (run OUTSIDE the container, 1 task per node):
#   copy the container image from Lustre to node-local disk (read once, sequentially),
#   and unpack the install tarball (if built) onto node-local disk.
set -euo pipefail
source /home/siyshen/workspace/b200-microbench/repro/scripts/env.sh
mkdir -p "$LOCAL"
if [ ! -s "$SQSH" ] || [ "$(stat -c %s "$SQSH")" != "$(stat -c %s "$SQSH_SRC")" ]; then
  cp "$SQSH_SRC" "$SQSH.tmp" && mv "$SQSH.tmp" "$SQSH"
fi
if [ -s "$ARTIFACTS/b200mb-install.tgz" ] && [ "${SKIP_INSTALL:-0}" != 1 ]; then
  if [ ! -f "$INST/.stamp" ] || [ "$ARTIFACTS/b200mb-install.tgz" -nt "$INST/.stamp" ]; then
    rm -rf "$INST"; tar -C "$LOCAL" -xzf "$ARTIFACTS/b200mb-install.tgz"; touch "$INST/.stamp"
  fi
fi
echo "$(hostname): $(ls -la $SQSH | awk '{print $5}') bytes sqsh; install: $(ls $INST 2>/dev/null | tr '\n' ' ')"
df -h "$LOCAL" | tail -1
