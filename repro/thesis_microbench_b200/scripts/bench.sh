#!/bin/bash
# B200 rerun driver (run from the LOGIN node; attaches steps to the persistent hold job).
#   JOBID=<hold job> GROUPS_="p2p_paper ar_paper ..." N_TRIAL=8 bash bench.sh
# Mirrors multi-gpu-programming-models/repro/thesis_microbench/scripts/{bench_rigorous,bench_allreduce_focused}.sh
# (the scripts that produced the paper's Fig. "nvshmem-p2p-bandwidth" and "nvshmem-allreduce-bandwidth"),
# with the same CLI args, trial counts and env, plus clearly-labelled tuned NVSHMEM variants.
set -u
source /home/siyshen/workspace/b200-microbench/repro/scripts/env.sh
JOBID=${JOBID:?set JOBID}
CNAME=${CNAME:-b200mb}
N_TRIAL=${N_TRIAL:-8}
GROUPS_=${GROUPS_:-"p2p_paper ar_paper"}
RAW=$RESULTS/raw
mkdir -p "$RAW"
NODES=($(scontrol show hostnames "$(squeue -h -j "$JOBID" -o %N)"))
NODEA=${NODEA:-${NODES[0]}}

LDP="$NVSHMEM_HOME_L/lib:$NCCL_HOME_L/lib"
# run_one <label> <nnodes> <tasks/node> <mpi> <timeout_s> <env-exports> <cmd>
run_one() {
  local label="$1" nn="$2" tpn="$3" mpi="$4" to="$5" envs="$6" cmd="$7"
  # ONLY=intra|inter lets an intra-only 1-node hold and the 2-node hold work through the same groups in parallel
  [ "${ONLY:-}" = intra ] && [ "$nn" = 2 ] && return 0
  [ "${ONLY:-}" = inter ] && [ "$nn" = 1 ] && return 0
  local nodelist=$NODEA; [ "$nn" = 2 ] && nodelist="${NODES[0]},${NODES[1]}"
  for trial in $(seq 1 "$N_TRIAL"); do
    local log="$RAW/${label}_t${trial}.log"
    if [ -s "$log" ] && grep -q '^#BENCH_EXIT 0' "$log"; then continue; fi
    echo "[$(date +%T)] $label t$trial"
    {
      echo "#BENCH label=$label trial=$trial nodes=$nodelist tpn=$tpn mpi=$mpi date=$(date -Is)"
      echo "#BENCH env: $envs"
      echo "#BENCH cmd: $cmd"
      timeout "$to" srun --jobid="$JOBID" --overlap -N "$nn" -w "$nodelist" --ntasks-per-node="$tpn" \
        --mpi="$mpi" --export=NONE --container-name="$CNAME" --container-mounts="$MOUNTS" --no-container-remap-root \
        bash -c "export LD_LIBRARY_PATH=$LDP:\$LD_LIBRARY_PATH NCCL_NET_PLUGIN=none; $envs $cmd" 2>&1
      echo "#BENCH_EXIT $?"
    } > "$log"
    grep -q '^#BENCH_EXIT 0' "$log" || { echo "  FAIL: $(tail -3 "$log" | tr '\n' ' ')"; }
  done
}

NVS_P2P=$NVSHMEM_HOME_L/bin/perftest/device/pt-to-pt
NVS_DEV=$NVSHMEM_HOME_L/bin/perftest/device/coll
NVS_HOST=$NVSHMEM_HOME_L/bin/perftest/host/coll
NV_ENV="export NVSHMEM_BOOTSTRAP=PMI NVSHMEM_BOOTSTRAP_PMI=PMI2;"
# --use_smem 0: NVSHMEM 3.8 perftests register smem for TMA by default (TMA_POLICY=DISABLE makes it a no-op,
# but we turn it off to match the 3.3.9 kernels byte-for-byte in behaviour).
P2P_ARGS="-b 4 -e 33554432 -n 50 -w 20 -c 32 -t 256 --use_smem 0"            # paper (bench_rigorous.sh)
P2P_TUNED_ARGS="-b 4 -e 33554432 -n 20 -w 5 -c 64 -t 1024 --use_smem 0"      # paper inter "ibgdatuned"
# Paper's IBGDA env (HCA_PREFIX= was an ibp* naming workaround on CoreWeave; harmless here, kept for parity).
IBGDA_ENV="$NV_ENV export NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_DISABLE_NVLS=1;"
# 3.8 changed the default RC QP map from 'cta' (3.3.9) to 'none'; pin 'cta' to reproduce the paper setup.
IBGDA_TUNED_ENV="$IBGDA_ENV export NVSHMEM_IBGDA_NUM_RC_PER_PE=64 NVSHMEM_IBGDA_RC_MAP_BY=cta;"

for G in $GROUPS_; do
case $G in
# ------------------------------------------------------------------ P2P (paper Fig. one-sided RMA)
p2p_paper)
  for api in put get p g; do
    b=$NVS_P2P/shmem_${api}_bw
    run_one "p2p_paper_intra_${api}"        1 2 pmi2 300 "$NV_ENV"          "$b $P2P_ARGS"
    run_one "p2p_paper_inter_ibgdatuned_${api}" 2 1 pmi2 600 "$IBGDA_TUNED_ENV" "$b $P2P_TUNED_ARGS"
  done ;;
p2p_extra)   # same as paper's other inter configs (IBGDA untuned c32xt256, and default IBRC proxy)
  for api in put get p g; do
    b=$NVS_P2P/shmem_${api}_bw
    run_one "p2p_paper_inter_ibgda_${api}"  2 1 pmi2 600 "$IBGDA_ENV" "$b $P2P_ARGS"
    run_one "p2p_paper_inter_ibrc_${api}"   2 1 pmi2 900 "$NV_ENV"    "$b $P2P_ARGS"
  done ;;
# ------------------------------------------------------------------ AllReduce (paper Fig. allreduce)
ar_paper)
  NCCL_ENV="export NCCL_GRAPH_MIXING_SUPPORT=0;"
  NARGS="-b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 2 -G 10"
  RING_ENV="$NCCL_ENV export NCCL_NVLS_ENABLE=0 NCCL_ALGO=Ring;"
  NB=$NCCL_TESTS_BIN/all_reduce_perf
  run_one ar_paper_intra_nccl_nvls   1 1 pmix 900 "$NCCL_ENV" "$NB $NARGS -g 8"
  run_one ar_paper_inter_nccl_nvls   2 8 pmix 900 "$NCCL_ENV" "$NB $NARGS -g 1"
  run_one ar_paper_intra_nccl_ring   1 1 pmix 900 "$RING_ENV" "$NB $NARGS -g 8"
  run_one ar_paper_inter_nccl_ring   2 8 pmix 900 "$RING_ENV" "$NB $NARGS -g 1"
  run_one ar_paper_intra_nvsdev      1 8 pmi2 900  "$NV_ENV" "$NVS_DEV/reduction_focus -b 128 -e 2147483648 -n 100 -w 50 --cudagraph"
  run_one ar_paper_intra_nvshost     1 8 pmi2 900  "$NV_ENV" "$NVS_HOST/reduction_on_stream -b 128 -e 1073741824 -n 50 -w 20 --cudagraph -d float -o sum"
  ;;
ar_paper_inter_nvs)   # paper inter NVSHMEM configs (default transport = IBRC proxy); very slow -> own group
  run_one ar_paper_inter_nvsdev      2 8 pmi2 ${AR_INTER_TO:-1500} "$NV_ENV" "$NVS_DEV/reduction_focus -b 128 -e ${AR_INTER_E:-2147483648} -n 5 -w 2 --cudagraph"
  run_one ar_paper_inter_nvshost     2 8 pmi2 ${AR_INTER_TO:-1500} "$NV_ENV" "$NVS_HOST/reduction_on_stream -b 128 -e ${AR_INTER_E:-2147483648} -n 5 -w 2 -d float -o sum"
  ;;
*) source $REPRO/scripts/groups_tune.sh "$G" ;;
esac
done
echo "[$(date +%T)] done: $GROUPS_"
