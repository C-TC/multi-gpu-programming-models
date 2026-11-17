# Tuning exploration groups (sourced by bench.sh via EXTRA_GROUPS_FILE). Single-trial sweeps; prefixes tune_*.
G1="$1"
case $G1 in
tune_p2p_intra)   # CTA/thread sweep + TMA for bulk ops, 1 MiB..32 MiB
  for api in put get p g; do
    for ct in "32 256" "64 1024" "148 1024" "296 1024" "148 512"; do set -- $ct
      N_TRIAL=1 run_one tune_p2p_intra_${api}_c$1t$2 1 2 pmi2 300 "$NV_ENV" "$NVS_P2P/shmem_${api}_bw -b 1048576 -e 33554432 -n 20 -w 5 -c $1 -t $2 --use_smem 0"
    done
  done
  for api in put get; do for ct in "32 256" "148 1024"; do set -- $ct
    N_TRIAL=1 run_one tune_p2p_intra_${api}_c$1t$2_tma 1 2 pmi2 300 "$NV_ENV export NVSHMEM_TMA_POLICY=ENABLE;" "$NVS_P2P/shmem_${api}_bw -b 1048576 -e 33554432 -n 20 -w 5 -c $1 -t $2 --use_smem 1"
  done; done ;;
tune_p2p_inter)   # IBGDA QP count / mapping / concurrency, 64 KiB..32 MiB
  for api in p g put get; do
    for cfg in "64 cta 64 1024" "128 cta 128 1024" "64 warp 64 1024" "128 none 128 1024" "256 cta 256 1024"; do set -- $cfg
      N_TRIAL=1 run_one tune_p2p_inter_${api}_rc$1$2_c$3t$4 2 1 pmi2 400 \
        "$IBGDA_ENV export NVSHMEM_IBGDA_NUM_RC_PER_PE=$1 NVSHMEM_IBGDA_RC_MAP_BY=$2;" \
        "$NVS_P2P/shmem_${api}_bw -b 65536 -e 33554432 -f 8 -n 20 -w 5 -c $3 -t $4 --use_smem 0"
    done
  done ;;
tune_ar_intra)    # device-block threads per block; on-stream as-is
  for t in 256 512 1024; do
    N_TRIAL=1 run_one tune_ar_intra_nvsdev_t$t 1 8 pmi2 600 "$NV_ENV" "$NVS_DEV/reduction_focus -b 128 -e 2147483648 -n 20 -w 5 -t $t"
  done ;;
tune_ar_inter)    # transport + scratch size + threads; sizes 128 B .. 64 MiB (x8 steps)
  for cfg in "ibrc 524288 256" "ibgda 524288 256" "ibgda 16777216 256" "ibgda 268435456 256" "ibgda 268435456 1024" "ibrc 268435456 1024"; do set -- $cfg
    tenv="$NV_ENV export NVSHMEM_REDUCE_SCRATCH_SIZE=$2;"
    [ $1 = ibgda ] && tenv="$tenv export NVSHMEM_IB_ENABLE_IBGDA=1;"
    N_TRIAL=1 run_one tune_ar_inter_nvsdev_$1_s$2_t$3  2 8 pmi2 900 "$tenv" "$NVS_DEV/reduction_focus -b 128 -e 134217728 -n 5 -w 2 -t $3"
    [ $3 = 256 ] && N_TRIAL=1 run_one tune_ar_inter_nvshost_$1_s$2 2 8 pmi2 900 "$tenv" "$NVS_HOST/reduction_on_stream -b 128 -e 67108864 -f 8 -n 5 -w 2 -d float -o sum"
  done ;;
*) source $REPRO/scripts/groups_verify.sh "$G1" ;;
esac
