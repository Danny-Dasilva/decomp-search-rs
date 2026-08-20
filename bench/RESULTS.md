# Benchmark: Python (LanceDB) vs Rust — end-to-end CLI, ms (median / p90)

iters=8, warm page cache, full process cost per call

| command | python | rust | speedup |
|---|---|---|---|
| find lbHeap_80015900 (hashed) | 853.4 / 884.0 | 9.6 / 10.9 | 89x |
| find lbHeap_80015900 (local) | 862.7 / 891.8 | 9.5 / 10.5 | 91x |
| findw lbHeap_80015900 (local) | 3838.5 / 3881.6 | 39.7 / 42.4 | 97x |
| findw mpRightWallGetTop (local) | 2414.1 / 2474.4 | 30.4 / 33.5 | 80x |
| findw OSInit (local) | 12164.2 / 12494.8 | 86.2 / 95.4 | 141x |
| donors lbHeap_80015900 (find local + findw hashed) | 4505.8 / 4637.9 | 46.8 / 49.2 | 96x |

## Internal (warm process — server-mode floor)

| metric | value |
|---|---|
| find p50 / p90 / p99 (44.6k fns) | 1.47 / 2.02 / 2.58 ms |
| findw p50 / p90 / max (209k windows) | 4.9 / 45.7 / 48.1 ms |
| ingest-dtk full project (16.3k fns + 56k windows, hashed) | 0.56 s |
| ingest-dtk re-run, no changes (all vectors reused) | 0.71 s |
| sweep (solvability, whole 16k-fn project) | 0.20 s |

Baseline (Python/LanceDB, same index, warm cache): find 867 ms end-to-end
(769 ms = imports), findw 3.6–12.2 s; production decomp-agent logs: findw median
7.5 s, p99 61.6 s across 18,913 calls.

Hardware: Threadripper 7970X 32C/64T, AVX-512, 183 GiB RAM. Rank-parity
vs Python verified on 160 sampled queries (ties permute); hashed embedding
bit-exact (self-sim 1.0); ingest token docs byte-identical on
15,978/15,979 common functions (1 diff = repo drift since index snapshot);
eval recall 7/8, identical to Python (miss = documented findw-only case).
