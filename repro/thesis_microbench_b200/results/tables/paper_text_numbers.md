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
