| Quantity (paper text) | H200 (paper) | B200 (this rerun, paper config) | Note |
|---|---|---|---|
| intra put peak (GB/s) | 313 | 672.4 |  |
| intra get peak (GB/s) | 141 | 72.7 |  |
| intra p peak (GB/s) | 172 | 180.6 |  |
| intra g peak (GB/s) | < 9 | 18.1 |  |
| inter put peak (GB/s), IBGDA tuned | 48.0 | 48.87 |  |
| inter get peak (GB/s), IBGDA tuned | 48.2 | 49.12 |  |
| inter p peak (GB/s), IBGDA tuned | 15.6 | 15.69 |  |
| inter g peak (GB/s), IBGDA tuned | 1.28 | 5.55 |  |
| intra latency, bulk put/get, ≤64 KiB (µs) | 1.8–2.5 | 2.3–4.0 |  |
| intra latency, scalar p/g, ≤64 KiB (µs) | 1.3–2.2 | 1.7–5.3 | H200 text range omits g's rise at ≥16 KiB (8.8 µs @64 KiB in the H200 data) |
| inter put/get latency at smallest valid size (512 B) (µs) | 9.4–9.5 @256 B* | 21.8 / 18.5 | *H200 text value came from the untuned c32 run; the tuned c64 run plotted moves 0 B at 256 B |
| inter put/get latency at 64 KiB (µs) | ~9.7* | 21.7 / 18.4 | *untuned-run value; plotted tuned run: 15.0 / 14.6 |
| inter p / g latency at 256 B (µs) | 7.5 / 25.3 | 12.0 / 19.9 | H200 data actually gives 10.8 / 25.3 |
| MEASURED intra [figure] put latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 2.57 / 2.49 / 2.57 / 3.86 | lat_intra_put (Block scope) |
| MEASURED intra [figure] get latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 4.20 / 4.17 / 4.29 / 9.95 | lat_intra_get_rot (Block scope) |
| MEASURED intra [figure] p latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 2.22 / 2.26 / 2.34 / 8.31 | lat_intra_p |
| MEASURED intra [figure] g latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 1.90 / 1.97 / 3.89 / 58.60 | lat_intra_g_rot |
| MEASURED intra, upstream perftest (g/get same-address, cached) get latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 4.10 / 3.99 / 4.07 / 9.73 | lat_intra_get (Block scope) |
| MEASURED intra, upstream perftest (g/get same-address, cached) g latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 0.22 / 0.18 / 0.28 / 2.75 | lat_intra_g |
| MEASURED inter IBGDA default QPs [figure] put latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 20.35 / 20.42 / 20.61 / 22.05 | lat_inter_ibgdadef_put (Block scope) |
| MEASURED inter IBGDA default QPs [figure] get latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 17.88 / 17.91 / 18.13 / 19.73 | lat_inter_ibgdadef_get_rot (Block scope) |
| MEASURED inter IBGDA default QPs [figure] p latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 14.78 / 15.87 / 29.13 / 205.05 | lat_inter_ibgdadef_p |
| MEASURED inter IBGDA default QPs [figure] g latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 14.27 / 15.02 / 39.78 / 586.39 | lat_inter_ibgdadef_g_rot |
| MEASURED inter IBGDA 64 RC/PE (paper bw env) put latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 76.71 / 76.78 / 76.97 / 78.40 | lat_inter_ibgda_put (Block scope) |
| MEASURED inter IBGDA 64 RC/PE (paper bw env) get latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 73.45 / 73.45 / 73.45 / 73.45 | lat_inter_ibgda_get (Block scope) |
| MEASURED inter IBGDA 64 RC/PE (paper bw env) p latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 68.90 / 70.57 / 82.91 / 258.27 | lat_inter_ibgda_p |
| MEASURED inter IBGDA 64 RC/PE (paper bw env) g latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 14.24 / 15.21 / 44.49 / 655.66 | lat_inter_ibgda_g |
| MEASURED inter IBRC (CPU proxy) put latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 13.91 / 13.91 / 13.99 / 15.91 | lat_inter_ibrc_put (Block scope) |
| MEASURED inter IBRC (CPU proxy) get latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 9.90 / 9.86 / 9.85 / 11.51 | lat_inter_ibrc_get (Block scope) |
| MEASURED inter IBRC (CPU proxy) p latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 8.30 / 21.59 / 270.40 / 4249.95 | lat_inter_ibrc_p |
| MEASURED inter IBRC (CPU proxy) g latency @ 8 B / 256 B / 4 KiB / 64 KiB (µs), n=8 | (not measured) | 10.07 / 11.04 / 53.38 / 884.46 | lat_inter_ibrc_g |
| AllReduce intra peak algbw, NVSHMEM on-stream (GB/s) | 264* | 303.5 | *H200 text used the unplotted 1 GiB point; plotted peaks were 262.5 / 274.8 |
| AllReduce intra peak algbw, NCCL NVLS (GB/s) | 276* | 479.2 | *H200 text used the unplotted 1 GiB point; plotted peaks were 262.5 / 274.8 |
| AllReduce intra peak algbw, NVSHMEM device block (GB/s) | 30 | 15.4 |  |
| AllReduce intra peak algbw, NCCL ring (GB/s) | (below on-stream) | 398.7 |  |
| AllReduce intra latency ≤64 KiB, NVSHMEM device (µs) | 3.8–7.1 | 5.9–11.5 |  |
| AllReduce intra latency ≤64 KiB, NCCL ring (µs) | 4.7–8.9** | 21.7–24.4 | **H200 'ring' was actually NCCL symmetric kernels (NCCL 2.30 ignores NCCL_ALGO=Ring with -R 2) |
| AllReduce intra latency ≤64 KiB, NCCL NVLS (µs) | 5.6–5.9 | 4.1–24.4 |  |
| AllReduce inter peak algbw, NVSHMEM on-stream (GB/s) | < 0.20 | 0.14 |  |
| AllReduce inter peak algbw, NVSHMEM device (GB/s) | < 0.20 | 0.13 |  |
| AllReduce inter peak algbw, NCCL ring (GB/s) | 180 | 195.71 |  |
| AllReduce inter peak algbw, NCCL NVLS-Tree (GB/s) | 252* | 366.05 | *H200 text used the unplotted 1 GiB point; plotted peak 246.4 |
| AllReduce inter latency at 64 KiB, nvshost (µs) | ms (NVSHMEM) / tens of µs (NCCL) | 2265.6 |  |
| AllReduce inter latency at 64 KiB, nvsdev (µs) | ms (NVSHMEM) / tens of µs (NCCL) | 2399.1 |  |
| AllReduce inter latency at 64 KiB, nccl_ring (µs) | ms (NVSHMEM) / tens of µs (NCCL) | 61.9 |  |
| AllReduce inter latency at 64 KiB, nccl_nvls (µs) | ms (NVSHMEM) / tens of µs (NCCL) | 36.8 |  |
