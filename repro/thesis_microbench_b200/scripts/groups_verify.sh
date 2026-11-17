# Extra groups for bench.sh (sourced with the group name as $1): verification + tuning exploration.
# Uses run_one/env from bench.sh. All logs land in $RAW with distinct prefixes (vfy_*, tune_*).
G1="$1"
NB=$NCCL_TESTS_BIN/all_reduce_perf
NCCL_ENV="export NCCL_GRAPH_MIXING_SUPPORT=0;"
RING_ENV="$NCCL_ENV export NCCL_NVLS_ENABLE=0 NCCL_ALGO=Ring;"
DBG_NCCL="export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,NET;"
NVSHMEM_TESTS_DEV_REDUCE=$NVSHMEM_HOME_L/tests/test_install/device/coll/reduce_block_sum
NVSHMEM_TESTS_HOST_REDUCE=$NVSHMEM_HOME_L/tests/test_install/host/coll/reduce
case $G1 in
vfy_versions)   # which libraries are really loaded
  N_TRIAL=1 run_one vfy_ldd 1 1 none 120 "" "ldd $NB | grep -E 'nccl|mpi'; ldd $NVS_P2P/shmem_put_bw | grep -E 'nvshmem|nccl'; ls $NVSHMEM_HOME_L/lib; nvidia-smi -q | grep -E 'Driver Version|CUDA Version'; nvcc --version | tail -1"
  ;;
vfy_transport)  # NVSHMEM remote transport selected by default vs with IBGDA env (inter, 1 PE/node)
  N_TRIAL=1 run_one vfy_transport_default 2 1 pmi2 300 "$NV_ENV export NVSHMEM_DEBUG=INFO;"          "$NVS_P2P/shmem_put_bw -b 1048576 -e 1048576 -n 5 -w 2 --use_smem 0"
  N_TRIAL=1 run_one vfy_transport_ibgda   2 1 pmi2 300 "$IBGDA_TUNED_ENV export NVSHMEM_DEBUG=INFO;" "$NVS_P2P/shmem_put_bw -b 1048576 -e 1048576 -n 5 -w 2 --use_smem 0"
  N_TRIAL=1 run_one vfy_transport_ar16_default 2 8 pmi2 600 "$NV_ENV export NVSHMEM_DEBUG=INFO;" "$NVS_HOST/reduction_on_stream -b 1024 -e 1024 -n 5 -w 2 -d float -o sum"
  ;;
vfy_zerobyte)   # put/get_bw with nCTAs: sizes < 8B*nCTAs move nothing -> flat per-iteration time
  for api in put get; do
    N_TRIAL=3 run_one vfy_zerobyte_inter_c64_${api} 2 1 pmi2 300 "$IBGDA_TUNED_ENV" "$NVS_P2P/shmem_${api}_bw -b 4 -e 8192 -n 20 -w 5 -c 64 -t 1024 --use_smem 0"
    N_TRIAL=3 run_one vfy_zerobyte_inter_c32_${api} 2 1 pmi2 300 "$IBGDA_ENV"       "$NVS_P2P/shmem_${api}_bw -b 4 -e 8192 -n 50 -w 20 -c 32 -t 256 --use_smem 0"
    N_TRIAL=3 run_one vfy_zerobyte_inter_c1_${api}  2 1 pmi2 300 "$IBGDA_TUNED_ENV" "$NVS_P2P/shmem_${api}_bw -b 4 -e 8192 -n 20 -w 5 -c 1 -t 1024 --use_smem 0"
  done ;;
vfy_nccl_algo)  # which algorithm / symmetric kernel NCCL really runs for each paper config
  for sc in intra inter; do
    if [ $sc = intra ]; then nn=1; tpn=1; g="-g 8"; else nn=2; tpn=8; g="-g 1"; fi
    N_TRIAL=1 run_one vfy_nccl_algo_${sc}_nvls $nn $tpn pmix 600 "$NCCL_ENV $DBG_NCCL" "$NB -b 128 -e 1G -f 8 -w 1 -n 1 -c 0 -R 2 -G 10 $g"
    N_TRIAL=1 run_one vfy_nccl_algo_${sc}_ring $nn $tpn pmix 600 "$RING_ENV $DBG_NCCL" "$NB -b 128 -e 1G -f 8 -w 1 -n 1 -c 0 -R 2 -G 10 $g"
  done ;;
vfy_correct)    # data correctness: nccl-tests -c 1 (both configs), NVSHMEM functional reduce tests
  for sc in intra inter; do
    if [ $sc = intra ]; then nn=1; tpn=1; g="-g 8"; else nn=2; tpn=8; g="-g 1"; fi
    N_TRIAL=1 run_one vfy_correct_${sc}_nccl_nvls $nn $tpn pmix 900 "$NCCL_ENV" "$NB -b 128 -e 1G -f 4 -w 5 -n 5 -c 1 -R 2 -G 10 $g"
    N_TRIAL=1 run_one vfy_correct_${sc}_nccl_ring $nn $tpn pmix 900 "$RING_ENV" "$NB -b 128 -e 1G -f 4 -w 5 -n 5 -c 1 -R 2 -G 10 $g"
    if [ $sc = intra ]; then nn=1; tpn=8; else nn=2; tpn=8; fi
    for t in $NVSHMEM_TESTS_DEV_REDUCE $NVSHMEM_TESTS_HOST_REDUCE; do
      N_TRIAL=1 run_one vfy_correct_${sc}_nvs_$(basename $t) $nn $tpn pmi2 900 "$NV_ENV ${VFY_NVS_ENV:-}" "$t"
    done
  done ;;
*) echo "unknown group $G1" ;;
esac
# --- smoke tests (appended)
case $G1 in
smoke_intra)
  N_TRIAL=1 run_one smoke_p2p_put 1 2 pmi2 120 "$NV_ENV" "$NVS_P2P/shmem_put_bw -b 256 -e 33554432 -f 16 -n 20 -w 5 -c 32 -t 256 --use_smem 0"
  N_TRIAL=1 run_one smoke_nccl 1 1 pmix 300 "$NCCL_ENV export NCCL_DEBUG=VERSION;" "$NB -b 128 -e 1G -f 64 -w 5 -n 10 -c 1 -R 2 -G 10 -g 8"
  N_TRIAL=1 run_one smoke_nvsdev 1 8 pmi2 300 "$NV_ENV" "$NVS_DEV/reduction_focus -b 128 -e 4194304 -n 20 -w 5 --cudagraph"
  N_TRIAL=1 run_one smoke_nvshost 1 8 pmi2 300 "$NV_ENV" "$NVS_HOST/reduction_on_stream -b 128 -e 4194304 -n 20 -w 5 --cudagraph -d float -o sum" ;;
esac
# --- targeted validation + investigations (appended)
ARC=$LOCAL/validate/ar_check
case $G1 in
vfy_archeck)   # float-sum allreduce correctness on the plotted NVSHMEM paths (on_stream + device block)
  N_TRIAL=1 run_one vfy_archeck_intra 1 8 pmi2 900 "$NV_ENV" "$ARC 536870912 536870912"
  N_TRIAL=1 run_one vfy_archeck_inter_ibrc 2 8 pmi2 1500 "$NV_ENV" "$ARC 536870912 536870912"
  N_TRIAL=1 run_one vfy_archeck_inter_ibgda 2 8 pmi2 1500 "$NV_ENV export NVSHMEM_IB_ENABLE_IBGDA=1;" "$ARC 536870912 536870912" ;;
vfy_correct_teams)   # rerun NVSHMEM functional reduce tests with enough team slots (intra NVLS dup teams)
  for t in $NVSHMEM_TESTS_DEV_REDUCE $NVSHMEM_TESTS_HOST_REDUCE; do
    N_TRIAL=1 run_one vfy_correct_intra_nvs_$(basename $t)_maxteams 1 8 pmi2 900 "$NV_ENV export NVSHMEM_MAX_TEAMS=1024;" "$t"
  done ;;
inv_ring_intra)   # why is intra NCCL Ring small-message latency ~22us? vary registration / graphs / plugin
  A="-b 128 -e 1048576 -f 8 -w 50 -n 100 -c 0"
  N_TRIAL=1 run_one inv_ring_intra_R2_G10 1 1 pmix 300 "$RING_ENV" "$NB $A -R 2 -G 10 -g 8"
  N_TRIAL=1 run_one inv_ring_intra_R0_G10 1 1 pmix 300 "$RING_ENV" "$NB $A -R 0 -G 10 -g 8"
  N_TRIAL=1 run_one inv_ring_intra_R2_G0  1 1 pmix 300 "$RING_ENV" "$NB $A -R 2 -G 0 -g 8"
  N_TRIAL=1 run_one inv_ring_intra_R0_G0  1 1 pmix 300 "$RING_ENV" "$NB $A -R 0 -G 0 -g 8"
  N_TRIAL=1 run_one inv_ring_intra_R1_G10 1 1 pmix 300 "$RING_ENV" "$NB $A -R 1 -G 10 -g 8"
  N_TRIAL=1 run_one inv_ring_intra_R2_G10_mp 1 8 pmix 300 "$RING_ENV" "$NB $A -R 2 -G 10 -g 1"
  N_TRIAL=1 run_one inv_ring_intra_R2_G10_dbg 1 1 pmix 300 "$RING_ENV export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,REG,P2P,GRAPH,ENV;" "$NB -b 1024 -e 1024 -w 5 -n 5 -c 0 -R 2 -G 10 -g 8" ;;
esac
case $G1 in
inv_ring_nccl230)   # same Ring config, but run against the container's NCCL 2.30.5 (paper used 2.30.4)
  L230="export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH;"
  N_TRIAL=1 run_one inv_ring230_intra_dbg 1 1 pmix 300 "$L230 $RING_ENV export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING;" "$NB -b 128 -e 1G -f 8 -w 5 -n 5 -c 0 -R 2 -G 10 -g 8"
  N_TRIAL=1 run_one inv_ring230_intra_perf 1 1 pmix 300 "$L230 $RING_ENV" "$NB -b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 2 -G 10 -g 8"
  N_TRIAL=1 run_one inv_ring230_intra_R0_perf 1 1 pmix 300 "$L230 $RING_ENV" "$NB -b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 0 -G 10 -g 8" ;;
esac
case $G1 in
tune_ar_intra_host)   # on-stream NVLS multi-CTA count (default 16 above threshold) and threads/CTA
  for c in 16 32 64 128; do
    N_TRIAL=1 run_one tune_ar_intra_nvshost_ctas$c 1 8 pmi2 600 "$NV_ENV export NVSHMEM_MAX_CTAS=$c NVSHMEM_MAX_TEAMS=1024;" "$NVS_HOST/reduction_on_stream -b 128 -e 1073741824 -f 4 -n 50 -w 20 --cudagraph -d float -o sum"
  done
  N_TRIAL=1 run_one tune_ar_intra_nvshost_ctas32_nt1024 1 8 pmi2 600 "$NV_ENV export NVSHMEM_MAX_CTAS=32 NVSHMEM_MAX_TEAMS=1024 NVSHMEM_REDUCE_NTHREADS=1024;" "$NVS_HOST/reduction_on_stream -b 128 -e 1073741824 -f 4 -n 50 -w 20 --cudagraph -d float -o sum" ;;
esac
case $G1 in
tuned_intra)   # final tuned intra configs, paper iteration counts, 8 trials
  TA="-b 4 -e 33554432 -n 50 -w 20"
  run_one p2p_tuned_intra_put 1 2 pmi2 300 "$NV_ENV" "$NVS_P2P/shmem_put_bw $TA -c 64 -t 1024 --use_smem 0"
  run_one p2p_tuned_intra_get 1 2 pmi2 300 "$NV_ENV" "$NVS_P2P/shmem_get_bw $TA -c 64 -t 1024 --use_smem 0"
  # (TMA variant -c 32 -t 256 --use_smem 1 + NVSHMEM_TMA_POLICY=ENABLE kept as p2p_tunedtma_intra_get: 579 GB/s peak but ~8us small-msg latency, bimodal noise)
  run_one p2p_tuned_intra_p   1 2 pmi2 300 "$NV_ENV" "$NVS_P2P/shmem_p_bw $TA -c 296 -t 1024 --use_smem 0"
  run_one p2p_tuned_intra_g   1 2 pmi2 300 "$NV_ENV" "$NVS_P2P/shmem_g_bw $TA -c 296 -t 1024 --use_smem 0"
  run_one ar_tuned_intra_nvshost 1 8 pmi2 900 "$NV_ENV export NVSHMEM_MAX_CTAS=64 NVSHMEM_MAX_TEAMS=1024;" "$NVS_HOST/reduction_on_stream -b 128 -e 1073741824 -n 50 -w 20 --cudagraph -d float -o sum" ;;
esac
case $G1 in
vfy_archeck_tuned_intra)
  N_TRIAL=1 run_one vfy_archeck_intra_maxctas64 1 8 pmi2 900 "$NV_ENV export NVSHMEM_MAX_CTAS=64 NVSHMEM_MAX_TEAMS=1024;" "$ARC 536870912 536870912" ;;
vfy_archeck_tuned_inter)
  N_TRIAL=1 run_one vfy_archeck_inter_tuned 2 8 pmi2 1500 "$NV_ENV ${TUNED_INTER_AR_ENV:-}" "$ARC 536870912 536870912" ;;
esac
case $G1 in
ar_paper_inter_nvs_v2)   # same as bench.sh ar_paper_inter_nvs; on-stream max size capped at the plotted 2^29 (paper: 2 GiB, unplotted)
  run_one ar_paper_inter_nvsdev  2 8 pmi2 1500 "$NV_ENV" "$NVS_DEV/reduction_focus -b 128 -e 2147483648 -n 5 -w 2 --cudagraph"
  run_one ar_paper_inter_nvshost 2 8 pmi2 1500 "$NV_ENV" "$NVS_HOST/reduction_on_stream -b 128 -e 536870912 -n 5 -w 2 -d float -o sum" ;;
esac
case $G1 in
inv_ring_nccl230_inter)   # does NCCL 2.30.5 also replace the inter-node "Ring" config by symmetric kernels?
  L230="export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH;"
  N_TRIAL=1 run_one inv_ring230_inter_dbg 2 8 pmix 600 "$L230 $RING_ENV export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING;" "$NB -b 128 -e 1G -f 8 -w 5 -n 5 -c 0 -R 2 -G 10 -g 1"
  N_TRIAL=1 run_one inv_nvls230_inter_dbg 2 8 pmix 600 "$L230 $NCCL_ENV export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING;" "$NB -b 128 -e 1G -f 8 -w 5 -n 5 -c 0 -R 2 -G 10 -g 1" ;;
tuned_intra_get)   # re-run of the final tuned intra get (non-TMA) on whichever node pair is available
  run_one p2p_tuned_intra_get 1 2 pmi2 300 "$NV_ENV" "$NVS_P2P/shmem_get_bw -b 4 -e 33554432 -n 50 -w 20 -c 64 -t 1024 --use_smem 0" ;;
esac
case $G1 in
tune_ar_inter2)
  for cfg in "ibrc 268435456" "ibrc 1073741824" "ibgda 1073741824" "ibgdarc16 268435456"; do set -- $cfg
    tenv="$NV_ENV export NVSHMEM_REDUCE_SCRATCH_SIZE=$2;"
    case $1 in ibgda) tenv="$tenv export NVSHMEM_IB_ENABLE_IBGDA=1;";; ibgdarc16) tenv="$tenv export NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NUM_RC_PER_PE=16;";; esac
    N_TRIAL=1 run_one tune_ar_inter_nvsdev_$1_s$2_t256 2 8 pmi2 900 "$tenv" "$NVS_DEV/reduction_focus -b 128 -e 134217728 -n 5 -w 2"
    N_TRIAL=1 run_one tune_ar_inter_nvshost_$1_s$2     2 8 pmi2 900 "$tenv" "$NVS_HOST/reduction_on_stream -b 128 -e 67108864 -f 8 -n 5 -w 2 -d float -o sum"
  done ;;
esac
case $G1 in
tuned_inter)   # final tuned inter configs, 8 trials (args otherwise identical to the paper's inter runs)
  TUNED_P2P_ENV="$IBGDA_ENV export NVSHMEM_IBGDA_NUM_RC_PER_PE=64 NVSHMEM_IBGDA_RC_MAP_BY=warp;"
  for api in put get p g; do
    run_one p2p_tuned_inter_${api} 2 1 pmi2 600 "$TUNED_P2P_ENV" "$NVS_P2P/shmem_${api}_bw -b 4 -e 33554432 -n 20 -w 5 -c 64 -t 1024 --use_smem 0"
  done
  TUNED_AR_ENV="$NV_ENV export NVSHMEM_REDUCE_SCRATCH_SIZE=268435456;"
  N_TRIAL=1 run_one vfy_archeck_inter_tuned 2 8 pmi2 1500 "$TUNED_AR_ENV" "$ARC 536870912 536870912"
  run_one ar_tuned_inter_nvsdev  2 8 pmi2 1500 "$TUNED_AR_ENV" "$NVS_DEV/reduction_focus -b 128 -e 2147483648 -n 5 -w 2 --cudagraph"
  run_one ar_tuned_inter_nvshost 2 8 pmi2 1500 "$TUNED_AR_ENV" "$NVS_HOST/reduction_on_stream -b 128 -e 536870912 -n 5 -w 2 -d float -o sum" ;;
esac
case $G1 in
rearch)   # NCCL 2.32.6 with the refactored enqueue pipeline (NCCL_ENQUEUE_REARCH_ENABLE=1; default only on NCCL dev branch)
  RA="export NCCL_ENQUEUE_REARCH_ENABLE=1;"
  NARGS="-b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 2 -G 10"
  for sc in intra inter; do
    if [ $sc = intra ]; then nn=1; tpn=1; g="-g 8"; else nn=2; tpn=8; g="-g 1"; fi
    N_TRIAL=1 run_one vfy_rearch_algo_${sc}_nvls $nn $tpn pmix 600 "$RA $NCCL_ENV $DBG_NCCL export NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING;" "$NB -b 128 -e 1G -f 8 -w 1 -n 1 -c 0 -R 2 -G 10 $g"
    N_TRIAL=1 run_one vfy_rearch_algo_${sc}_ring $nn $tpn pmix 600 "$RA $RING_ENV $DBG_NCCL export NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING;" "$NB -b 128 -e 1G -f 8 -w 1 -n 1 -c 0 -R 2 -G 10 $g"
    N_TRIAL=1 run_one vfy_rearch_correct_${sc}_nvls $nn $tpn pmix 900 "$RA $NCCL_ENV" "$NB -b 128 -e 1G -f 4 -w 5 -n 5 -c 1 -R 2 -G 10 $g"
    N_TRIAL=1 run_one vfy_rearch_correct_${sc}_ring $nn $tpn pmix 900 "$RA $RING_ENV" "$NB -b 128 -e 1G -f 4 -w 5 -n 5 -c 1 -R 2 -G 10 $g"
    run_one ar_rearch_${sc}_nccl_nvls $nn $tpn pmix 900 "$RA $NCCL_ENV" "$NB $NARGS $g"
    run_one ar_rearch_${sc}_nccl_ring $nn $tpn pmix 900 "$RA $RING_ENV" "$NB $NARGS $g"
  done ;;
esac
case $G1 in
ab_samepair)   # same-node-pair A/B: legacy enqueue path, to compare with ar_rearch_* measured on this pair
  NARGS="-b 128 -e 1G -f 2 -w 50 -n 100 -c 0 -R 2 -G 10"
  N_TRIAL=3 run_one ab_legacy_inter_nccl_nvls 2 8 pmix 900 "$NCCL_ENV" "$NB $NARGS -g 1"
  N_TRIAL=3 run_one ab_legacy_inter_nccl_ring 2 8 pmix 900 "$RING_ENV" "$NB $NARGS -g 1"
  N_TRIAL=3 run_one ab_legacy_intra_nccl_nvls 1 1 pmix 900 "$NCCL_ENV" "$NB $NARGS -g 8"
  N_TRIAL=3 run_one ab_rearch_intra_nccl_nvls 1 1 pmix 900 "export NCCL_ENQUEUE_REARCH_ENABLE=1; $NCCL_ENV" "$NB $NARGS -g 8" ;;
esac
case $G1 in
latency)   # true one-sided latency (NOT ping-pong RTT): NVSHMEM shmem_{put,get,p,g}_latency, shipped .args settings, 8 trials
  # put/get: 1 CTA issues {put,get}_nbi + nvshmem_quiet per iteration; reported for Thread/Warp/Block scope (Block = bulk op as in *_bw)
  # p: 1 CTA x 512 threads, scalar p per element + quiet per iteration; g: 1 CTA x 512 threads, blocking scalar g per element
  for api in put get p g; do
    case $api in put|get) LA="-b 4 -e 65536 -n 200 -w 20 -t 1024 --repetitions 3 --use_smem 0";;
                 p)       LA="-b 4 -e 65536 -n 50 -w 5 -t 512 --repetitions 3";;
                 g)       LA="-b 4 -e 65536 -n 200 -w 20 -t 512 --repetitions 3";; esac
    run_one lat_intra_${api}       1 2 pmi2 300 "$NV_ENV"          "$NVS_P2P/shmem_${api}_latency $LA"
    run_one lat_inter_ibgda_${api} 2 1 pmi2 600 "$IBGDA_TUNED_ENV" "$NVS_P2P/shmem_${api}_latency $LA"
    run_one lat_inter_ibrc_${api}  2 1 pmi2 900 "$NV_ENV"          "$NVS_P2P/shmem_${api}_latency $LA"
  done ;;
esac
case $G1 in
latency2)
  ROT=$LOCAL/validate
  LPG="-b 4 -e 65536 -n 200 -w 20 -t 1024 --repetitions 3 --use_smem 0"; LG="-b 4 -e 65536 -n 200 -w 20 -t 512 --repetitions 3"
  LP="-b 4 -e 65536 -n 50 -w 5 -t 512 --repetitions 3"
  # (1) intra g/get without the same-address cache artifact
  run_one lat_intra_g_rot   1 2 pmi2 300 "$NV_ENV" "$ROT/shmem_g_latency_rot $LG"
  run_one lat_intra_get_rot 1 2 pmi2 300 "$NV_ENV" "$ROT/shmem_get_latency_rot $LPG"
  # (2) inter IBGDA with default QP count (latency config) — the 64-RC/PE env is a bandwidth config whose quiet drains 64 QPs
  for api in put get p g; do
    case $api in put|get) LA="$LPG";; p) LA="$LP";; g) LA="$LG";; esac
    run_one lat_inter_ibgdadef_${api} 2 1 pmi2 600 "$IBGDA_ENV" "$NVS_P2P/shmem_${api}_latency $LA"
  done
  run_one lat_inter_ibgdadef_g_rot   2 1 pmi2 600 "$IBGDA_ENV" "$ROT/shmem_g_latency_rot $LG"
  run_one lat_inter_ibgdadef_get_rot 2 1 pmi2 600 "$IBGDA_ENV" "$ROT/shmem_get_latency_rot $LPG"
  # (3) probe: put/get latency vs RC QPs per PE
  for rc in 1 2 8 64; do for api in put get; do
    N_TRIAL=1 run_one tune_lat_inter_${api}_rc$rc 2 1 pmi2 600 "$IBGDA_ENV export NVSHMEM_IBGDA_NUM_RC_PER_PE=$rc;" "$NVS_P2P/shmem_${api}_latency -b 8 -e 65536 -f 64 -n 200 -w 20 -t 1024 --repetitions 3 --use_smem 0"
  done; done ;;
esac
