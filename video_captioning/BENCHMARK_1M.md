# Benchmark: 1M-caption dense video captioning — Ray Core vs Ray Data

Runs executed 2026-07-14 on Anyscale staging (CoreWeave US-EAST-14A).
This document collects every performance metric from the two full-scale runs;
the narrative comparison lives in the [README](./README.md#what-a-256-gpu-run-shows).

## Task

Caption the full [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
corpus densely with [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct):
each clip is split into 12-second windows (`CAPTION_WINDOW_SEC=12`) and every
window is captioned once from 6 uniformly-sampled keyframes.

| Workload parameter | Value |
|---|---|
| Source clips | 43,7xx (whole corpus, no `NUM_VIDEOS` limit) |
| Total video | ~3,443 video-hours (~12.4M video-seconds) |
| Caption windows produced | 1,055,410 (Ray Core) / 1,055,221 (Ray Data) |
| Windows per clip (mean) | ~24 |
| Keyframes per window | 6, resized to 448×448, JPEG q85 |
| Model | Qwen3-VL-8B-Instruct, 1 replica/GPU, no TP |
| Weights loading | Run:ai streamer directly from S3 mirror |
| Sampling | temperature 0.2, max 256 output tokens |
| Dispatch batch size | 32 rows (`VLM_BATCH_SIZE`) |
| `max_model_len` | 8192 |

## Hardware (identical for both pipelines)

| | |
|---|---|
| GPU workers | 32 nodes × 8× RTX PRO 6000 (Blackwell, 96 GB) = 256 GPUs, 120 CPU / 960 Gi each |
| CPU-only workers | up to 8 nodes × 120 CPU / 480 Gi |
| Head node | 32 CPU / 128 Gi, control-plane only |
| Cluster total | ~4,800 CPUs, 256 GPUs |

Both drivers call `wait_for_gpus(256)` before starting their clock, so the
full cluster exists at t0 in both runs and hardware is identical.

## Methodology

- **Timed region is end-to-end**: the clock starts before any caption engine
  exists (Ray Core: before actor creation and weight loading; Ray Data: before
  the terminal `write_parquet` that triggers the stream, whose actor pool
  boots inside the execution) and stops when the last caption is written.
- **GPUs held**: sampled every 5 s on the driver as
  `cluster_resources().GPU − available_resources().GPU`; `gpu_seconds_held`
  is the step integral over the timed region.
- **GPU/CPU utilization**: NVML / psutil, sampled every 5 s by one actor
  pinned to each worker node; reported as mean and p95 over all samples.
- Both runs succeeded on their first attempt (no job retries).

## Headline comparison

The "tuned" column is the same Ray Data pipeline after the three efficiency
changes described in [Efficiency tuning](#efficiency-tuning-from-43-to-19-minutes).

| Metric | Ray Core (pre-provisioned) | Ray Data (first run) | **Ray Data (tuned)** | Best ratio |
|---|---|---|---|---|
| Captions written | 1,055,410 | 1,055,221 | 1,055,409 | — |
| End-to-end wall time | 1,735.6 s (28.9 min) | 2,601.0 s (43.4 min) | **1,124.6 s (18.7 min)** | **Data 1.54× faster** |
| Captions / sec | 608.1 | 405.7 | **938.5** | Data 1.54× |
| GPUs held, mean (peak) | 255.3 (256) | 89.8 (256) | 105.1 (256) | Data holds 2.4× fewer |
| GPU-seconds held | 443,034 (123.1 GPU-h) | 233,587 (64.9 GPU-h) | **118,481 (32.9 GPU-h)** | **Data 3.74× cheaper** |
| **Captions / GPU-hour** | **8,576** | 16,263 | **32,068** | **Data 3.74×** |
| Captions / held-GPU-second | 2.38 | 4.52 | **8.91** | Data 3.74× |
| GPU util, mean / p95 | 23.8% / 100% | 13.2% / 87% | **30.3% / 99.4%** | — |
| CPU util, mean / p95 | 4.8% / 7.3% | 6.4% / 21.1% | **16.7% / 94.5%** | — |

![Caption progress over time](./figures/progress_1m.png)

![GPUs held over time](./figures/gpus_held_1m.png)

![GPU-hours and captions per GPU-hour](./figures/efficiency_1m.png)

## Time to completion

Total wall time hides the shape of the runs, so here is time to each
completion fraction (parsed from the driver logs):

| Fraction complete | Ray Core | Ray Data (first run) | **Ray Data (tuned)** |
|---|---|---|---|
| 90% | 26.7 min | 15.2 min | **14.4 min** |
| 95% | 27.9 min | 15.6 min | **14.6 min** |
| 99% | 28.8 min | 22.8 min | **14.7 min** |
| 100% (GPU stage) | 28.9 min | 43.4 min | **14.9 min** |
| 100% (all written) | 28.9 min | 43.4 min | 18.7 min |

Ray Data reaches 95% of the corpus **1.9× sooner** than Ray Core — its decode
spreads across all ~4,800 cluster CPUs instead of the driver-funneled ~190
tasks, and the engine pool bursts to 256 right as decoded windows flood in.
The first run's long total wall was entirely the drain: the last ~1% of
captions took ~21 minutes on a scaled-in pool waiting for a replacement
engine to boot. The tuned run's `(8, n)` pool floor eliminates it — 99% to
100% of GPU work now takes 12 seconds, and the remaining ~3.8 min to "all
written" is the CPU-side detokenize/write drain after every GPU has already
been released.

## Efficiency tuning: from 43 to 19 minutes

Three measurement-driven changes to the Ray Data workload (none to the
work itself):

1. **Engine pool floor — `concurrency=(8, 256)`** (was `(1, 256)`). The
   autoscaled pool used to scale in during the drain, and straggler blocks
   then waited minutes for a replacement engine boot. A floor of 8 live
   engines removes that wait for a few GPU-minutes of extra hold.
2. **Deeper per-engine queues — `max_concurrent_batches=8`** (default 4).
   Engines were running ~75 requests at 17–19% KV-cache usage; doubling the
   in-flight batches raises continuous-batching occupancy with no preemption
   risk at 8k context. Per-held-GPU throughput went from 4.5 to 8.9
   captions/GPU-second.
3. **Truthful CPU reservations — `num_cpus=0.25` on the pinned shard reads.**
   The Ray autoscaler provisions against *reserved* CPUs of pending tasks,
   not utilization. IO-bound read tasks that reserved a full core each
   inflated the opening burst's demand ~4×, summoning CPU nodes that arrived
   after the ~2-minute burst ended and then idled for the whole run (they
   could not scale back down because Ray Data SPREAD-schedules blocks, and a
   node holding live object-store copies is never idle-reclaimable). The
   fractional reservation keeps `max_nodes` as a genuine elastic ceiling
   while the autoscaler stops at the nodes the work actually needs —
   validated: with 0.25-CPU reservations the pool scaled to 4 nodes, exactly
   the demand arithmetic (1,357 read tasks × 0.25 CPU ≈ 339 reserved CPUs ≈
   3–4 nodes), instead of slamming into the 8-node ceiling.

Net effect: wall 2,601 s → **1,124.6 s**, GPU-seconds held 233,587 →
**118,481**, captions/GPU-hour 16,263 → **32,068**, mean GPU util 13.2% →
**30.3%**, mean CPU util 6.4% → **16.7%** — and the tuned pipeline now beats
the hand-rolled Ray Core baseline on wall clock (1.5×) and GPU-hours (3.7×)
simultaneously.

### Production mode: nodes follow the pool (PREPROVISION_GPUS=0)

The benchmark numbers above pre-provision all 32 GPU nodes at t0 for
hardware parity, so fleet-mean utilization is capped by GPUs the pool never
holds. `PREPROVISION_GPUS=0` drops that gate — the engine pool's pending
actors drive the node autoscaler directly — and `DATASET_REPEATS=3` streams
the corpus three times to emulate the continuous-workload regime
(job `prodjob_aa4exhmvx8ynsjqpfm2ns1b29f`, 2026-07-15):

| Metric | Benchmark mode (tuned, 1×) | Production mode (3×) |
|---|---|---|
| Captions | 1,055,409 | 3,166,045 |
| Captions/s | 938.5 | **1,236.9** |
| Fleet-mean GPU util (p95) | 30.3% (99.4) | **48.8% (99.5)** |
| Utilization of *held* GPUs | ~74% | **~87%** |
| Captions per GPU-hour held | 32,068 | **38,563** |
| GPU-s provisioned per 1k captions | 273 | **167 (−39%)** |
| GPUs provisioned / held, mean | 256 / 105 | 207 / 115 |

The remaining provisioned-vs-held gap (1.8×) is ramp overhang (nodes arrive
~2 min before their engines finish booting), 8-GPU node granularity, and
idle-node termination lag — all of which amortize with run length. The cost
of production mode is ~7–10 min of extra ramp (node provisioning moves
inside the pipeline), which is why it suits recurring/long workloads and
benchmark mode suits one-shot sprints.

A second-order effect worth understanding: the *decode stage itself* got
3× faster (whole corpus in 5.4 min vs ~18) without any decode change,
because Ray Data's backpressure paces producers to consumers — once the
engine pool ramped faster and consumed deeper, decode was released to run
wide open: **3,816 windows/s sustained, ~22,900 frames/s encoded, ~46,000×
real-time video ingest on 2,917 concurrent cores at peak**. The 16.7% mean
CPU figure is a time-dilution artifact of a phased pipeline on GPU-heavy
nodes: ~70% of the cluster's CPUs are busy during the 5-minute decode burst,
after which 3,840 of the 4,080 cores — welded to the GPU nodes — have only
tokenize/detokenize/write left to do. On a continuous (petabyte-scale)
stream, decode never drains and the fleet rides at decode-phase utilization
throughout.

## CAIOS (object storage) traffic

Everything durable in these runs moves through CoreWeave AI Object Storage
via the in-cluster [LOTA](https://docs.coreweave.com/products/storage/object-storage/improving-performance/about-lota)
endpoint (`http://cwlota.com`): model weights in, mp4 shards in, captions and
reports out. Two LOTA properties matter for reading these numbers: it
requires virtual-hosted addressing (hence
`RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=1` in the job YAMLs), and it
accelerates GET requests through a distributed LRU cache spread across the
cluster's nodes (1 TiB per node by default) — 256 engines fetching the *same*
~17 GiB of weight objects is that cache's best case, which is consistent with
per-engine load times staying flat under 256-way concurrency.

![CAIOS weight streaming dot plot](./figures/caios_weight_streaming.png)

- **Model weight streaming (read)** — every caption engine streams ~17 GiB of
  Qwen3-VL-8B safetensors from CAIOS via vLLM's Run:ai loader: ≈4.3 TiB read
  per run across 256 engines plus the driver. Sampled per-engine load times
  (vLLM `Model loading took …` lines survive log dedup for a subset of
  engines): Ray Data median ~9 s (8.3–17.2 s, n=23), Ray Core median ~12 s
  (10.6–35.6 s, n=6) — roughly **0.5–2.0 GiB/s per engine while up to 256
  peers load concurrently**, with no visible collapse under the Ray Core
  run's simultaneous 256-engine burst.
- **Dataset reads** — 1,357 parquet shards holding 43,751 mp4 clips
  (~3,443 video-hours, **630.7 GiB of mp4 payload** per Ray Data's ReadFiles
  operator metrics) are read once per run.
- **Caption writes** — 1,055,221 rows / 283.1 MiB of output parquet
  (Ray Data sink; the Ray Core run writes ~528 parts of ~2,000 rows each),
  plus one `report.json` per run.
- The optional `<model>/hash` object is absent from the mirror, so every
  engine logs a benign `NoSuchKey` before falling back to the default
  snapshot hash — see the README if you want to silence it.

### How CAIOS removes the storage bottleneck (vs vanilla S3)

Two access patterns in this benchmark are exactly the ones that hurt on a
plain regional object store, and both were non-events on CAIOS:

**1. Model loading — a 256-way read storm of the same hot object.**
Every engine streams the same **~17 GiB** of Qwen3-VL safetensors, so booting
the fleet demands **≈4.3 TB of reads of one model per run**. Measured: each
engine loaded in a **median of 9–12 s ≈ 1.9 GiB/s per engine — with up to 255
peers streaming the same bytes at the same time** and no per-engine
degradation (at full overlap that is an instantaneous aggregate demand on the
order of **0.5 TB/s**). The sampled completions cluster inside a ~2-minute
boot window, so the fleet-wide fill sustained **at least ~37 GiB/s**. This is
LOTA's best case by design: after the first pulls, the distributed LRU cache
(1 TiB per node) serves the hot model from inside the cluster instead of
hammering the backend.

**2. Dataset reads — a wide, one-shot corpus scan.**
1,357 shard reads (43,751 clips, **630.7 GiB of mp4**) completed in the first
**140 s** of the run across ~960 single-CPU readers — **~4.5 GiB/s aggregate,
with zero tuning**: no prefix sharding, no retry/backoff engineering, no
`503 SlowDown` handling, one endpoint URL and one addressing env var.

| | CAIOS + LOTA | Vanilla AWS S3 |
|---|---|---|
| Hot-object fan-out (256 × same 17 GiB) | Absorbed by a node-local distributed cache built into the endpoint | Every read returns to the regional service; hot prefixes throttle (`503 SlowDown`, ~5,500 GET/s per prefix guidance) |
| Per-client throughput | **~1.9 GiB/s per engine measured** under full-fleet concurrency | ~85–90 MB/s per connection guidance — clients stack dozens of parallel ranged GETs to compete |
| Tuning needed for this workload | Virtual-hosted addressing flag | Prefix design, request-rate backoff, CRT/multipart tuning; hot-weight patterns are typically *staged out* of S3 first (local NVMe, FSx for Lustre, CloudFront, or the premium Express One Zone class) |
| Cache | Included, transparent, read-after-write consistent | A second system you provision, pay for, and keep coherent |

The punchline for this pipeline: with storage serving 630 GiB in two minutes
and 4.3 TB of weight reads in seconds per engine, **the bottleneck moved to
where it belongs — video decode CPU — instead of the storage tier.** The
section below quantifies that.

### CAIOS raw limits and published specs (reference)

Official limits and guidance, for interpreting the measurements above
([quotas](https://docs.coreweave.com/products/storage/object-storage/manage-quotas),
[best practices](https://docs.coreweave.com/products/storage/object-storage/improving-performance/best-practices),
[warp benchmarking](https://docs.coreweave.com/products/storage/object-storage/improving-performance/benchmarking-warp),
[benchmark blog](https://www.coreweave.com/blog/benchmark-results-coreweave-ai-object-storage-delivers-2-gb-s-per-gpu-throughput-across-any-number-of-gpus)):

| Spec | Value |
|---|---|
| Published read throughput | up to 2 GB/s per GPU at launch; current marketing up to 7 GB/s per GPU (cache-served) |
| CoreWeave's own 20-node benchmark | uncached ~24 GiB/s aggregate (1.2 GiB/s/node); cache-hot 368 GiB/s (18.4 GiB/s/node, 2.3 GiB/s/GPU) |
| Warp sample (docs, 1 node, GET, 300 conc., 15 MiB objects) | 4.2 GiB/s avg; latency p50 38 ms / p99 125 ms |
| Hard rate limit | none published — throttling is per-partition `503 Slow Down`, handled by client back-off |
| Recommended concurrency | ~300 concurrent ops per node to start, tune toward 500+ while throughput climbs |
| Read sizing | ≥15 MB per request recommended; <1 MB degrades; objects <4 MB bypass the LOTA cache |
| Writes | not LOTA-accelerated (proxied); multipart with ≥50 MB parts spreads load across nodes — a single `PutObject` lands the whole object on one node |
| Multipart limits | parts 5 MiB–5 GiB, ≤10,000 parts, max object ~48.8 TiB |
| Default storage quota | **20 TiB STANDARD per AZ per account** (quota raises via support) |
| LOTA cache | 1 TiB per node, LRU, GET-only, read-after-write consistent |

Context for our numbers, per node (the unit CoreWeave publishes):

- **Weight streaming reproduces the published envelope.** Eight engines per
  node at ~1.9 GiB/s each is **~15.2 GiB/s per GPU node** — ~83% of the
  published cache-hot 18.4 GiB/s/node, with each engine at the 2 GB/s-per-GPU
  launch spec, on default Run:ai streamer settings and zero tuning. (The
  current 7 GB/s-per-GPU marketing tier assumes prestaged cache and tuned
  streaming concurrency.)
- **The corpus scan is client-bound, not storage-bound**: ~0.56 GiB/s per CPU
  node vs 1.2 GiB/s/node published *uncached*, because each reader is one
  sequential pyarrow stream whose task time includes parquet decode — far
  from warp's tuned 300-way per-node concurrency. The same nodes sustained
  15 GiB/s on the weight path, and the scan finished in the first 140 s of a
  43-minute job, so pushing read parallelism would not move the benchmark.
- Cache state is not fully controlled: earlier same-day runs touched the same
  objects on the same underlying nodes, so reads were somewhere between cold
  and hot. Cold/hot separation needs a dedicated warp run.

### First-party measured ceilings (US-EAST-14A, this fleet)

[`storage_bench.py`](./storage_bench.py) / [`job_storage_bench.yaml`](./job_storage_bench.yaml)
sweep concurrent 15 MiB ranged GETs and 50 MiB PUTs from one bench actor per
node (4 GPU + 4 CPU nodes, 8 processes each so the Python GIL never caps the
measurement), against both endpoints. The measured optima:

| Direction | Optimal config | Per node | Aggregate, 8 nodes |
|---|---|---|---|
| **Read, LOTA** (`cwlota.com`) | 64–200 concurrency/node | **25–30 GiB/s per GPU node** | **135 GiB/s** |
| Read, gateway (`cwobject.com`) | ~500 concurrency/node | 11–12.6 GiB/s | 95 GiB/s |
| **Write, PUT 50 MiB** | 100–150 concurrency/node | ~4–6.5 GiB/s | **40–41 GiB/s** |

Findings from the sweep:

- **The LOTA read ceiling here beats CoreWeave's published cache-hot figure**:
  25–30 GiB/s per GPU node vs their 18.4 GiB/s/node benchmark, at p50 latency
  of ~21 ms (vs ~90–600 ms via the gateway). Extrapolated to the 32-node
  caption fleet, that is on the order of **0.8+ TiB/s** of aggregate read
  capacity — ~200× what the captioning pipeline actually drew.
- Concurrency has a cliff, not a plateau, past the optimum: LOTA throughput
  halves at 500/node, and at 800/node the node's agent stopped responding
  entirely (which ended one benchmark attempt). Treat ~300/node (CoreWeave's
  own guidance) as the safe operating point, 500 as the edge.
- Writes saturate at ~40 GiB/s aggregate from concurrency ~100/node; more
  concurrency only adds latency (p50 1.1 s → 6.6 s from 100 → 500). The
  latency-throughput knee is at ~100/node.
- CPU-only nodes show an asymmetry: ~6.5 GiB/s via LOTA but 12.6 GiB/s via
  the gateway — on this fleet the fat LOTA cache path belongs to GPU nodes.
- The `WRITE_KEYFRAMES=1` job knob (persist each window's JPEG frames next to
  its caption) turns the captioning run's output from 283 MiB into a ~185 GiB
  frames+captions dataset — a realistic write-heavy variant that still sits
  comfortably under the measured 40 GiB/s write ceiling.

Full sweep data (per-node values are averages within each pool; passes marked
† included one degraded GPU node that dragged its pool average down):

**Reads — LOTA (`cwlota.com`), 15 MiB ranged GETs**

| Concurrency/node | GPU node GiB/s | CPU node GiB/s | Aggregate GiB/s | p50 |
|---|---|---|---|---|
| 32 | 16.8 † (healthy: 21–23) | 4.9 | 86.8 | 17 ms |
| 64 | 21.8 † / **27.3** (healthy: 25–30) | 5.3–6.4 | 108.7 / **134.9** | 21 ms |
| 100 | **25.1** | 7.2 | 129.2 | 22 ms |
| 200 | 21.9 † (healthy: 25–30) | 5.0 | 107.3 | 29 ms |
| 300 | 24.9 | 6.0 | 123.6 | 36 ms |
| 500 | 12.3 | 6.8 | 76.4 | 237 ms |
| 800 | — node agent became unresponsive; run aborted | | | |

**Reads — gateway (`cwobject.com`)**

| Concurrency/node | GPU node GiB/s | CPU node GiB/s | Aggregate GiB/s | p50 |
|---|---|---|---|---|
| 64 | 8.9 | 9.2 | 72.1 | 90 ms |
| 100 | 4.9 | 5.0 | 39.5 | 187 ms |
| 300 | 7.9 | 8.5 | 65.3 | 410 ms |
| 500 | **11.2** | **12.6** | **95.2** | 600 ms |
| 800 | 10.0 | 11.8 | 87.0 | 1,043 ms |

**Writes — PUT, 50 MiB objects, via `cwlota.com`**

| Concurrency/node | Aggregate GiB/s | p50 per object |
|---|---|---|
| 25 | 20.6 | 0.41 s |
| 50 | 33.8 | 0.58 s |
| 100 | **40.0** | 1.14 s |
| 150 | **41.2** | 1.78 s |
| 300 | 37.0 | 3.58 s |
| 500 | 27.8 | 6.55 s |

### CPU-only fleet ceilings (augmentation fleet, 2026-07-15)

[`job_storage_bench_cpu.yaml`](./job_storage_bench_cpu.yaml) reran the sweep
on a GPU-free fleet (8 × 120-CPU nodes) — the shape the
[1000× augmentation job](./README.md#scaling-the-corpus-1000-ffmpeg-augmentation--caios-write-benchmark)
runs on, since no stage of that workload touches a GPU
(job `prodjob_bengys2v2hquz9djg5in9m87kk`):

| Direction | Optimal config | Per node | Aggregate, 8 nodes |
|---|---|---|---|
| **Write, PUT 50 MiB** | 100 concurrency/node | **5.16 GiB/s** | **41.3 GiB/s** |
| Read, gateway (`cwobject.com`) | 500 concurrency/node | **10.6 GiB/s** | 84.9 GiB/s |
| Read, LOTA (`cwlota.com`) | 64–300 concurrency/node | ~8.5–9.0 GiB/s | 60–72 GiB/s |

- **The write ceiling is not GPU-node-gated**: 41.3 GiB/s aggregate from CPU
  nodes alone reproduces the mixed-fleet 40–41 GiB/s, with the same knee at
  ~100/node (150/node drops to 3.96 GiB/s/node while p50 doubles to 1.9 s).
- **The fat LOTA path is**: hot-subset passes measured the same ~8.9 GiB/s as
  the whole-corpus passes (no cache lift), vs 25–30 GiB/s/node on GPU nodes.
  On CPU nodes LOTA's win is latency — p50 35 ms vs 103–655 ms via the
  gateway — so cold, wide scans belong on the gateway (10.6 GiB/s/node at
  500/node) and low-latency hot reads on LOTA.
- LOTA passes logged a low error rate (~0.1% of requests, absorbed by client
  retries); the gateway swept clean at every concurrency.

### Read vs decode — measured, not asserted

Ray Data's per-operator metrics (from `ray-data-dataset_10_0.log` of the 1M
run) separate the storage read from the CPU decode:

| Operator | Tasks | Output | Worker-seconds | Share of CPU work |
|---|---|---|---|---|
| `ReadFiles` (S3 → mp4 blocks) | 1,357 | 43,751 rows, 630.7 GiB | 22,891 (6.4 CPU-h) | 8.5% |
| `FlatMap(decode_row)` (mp4 → keyframes) | 5,015 | 1,055,221 rows, 183.5 GiB | 246,227 (68.4 CPU-h) | 91.5% |

The entire 630.7 GiB dataset was read in the first ~140 s of the 43-minute
run (ListFiles done 15:17:52, ReadFiles operator complete 15:20:12) — an
aggregate draw of **~4.5 GiB/s from CAIOS** across ~1,000 concurrent
single-CPU readers, ~17 s of pure worker time per ~476 MiB shard. Decoding
those bytes then took **10.8× the worker-seconds of reading them** (decode
task p50 73 s, p99 205 s, max 979 s — those slowest decode tasks are the
straggler blocks behind the drain tail). So: storage was never the
bottleneck at this scale — weight loads complete in seconds per engine, the
corpus reads in ~2 minutes, and the pipeline is CPU-bound on video decoding.

The same arithmetic explains the Ray Core run's GPU starvation: read+decode
are fused there and throttled to ~190 concurrent tasks by the head-node row
buffer, and 269k worker-seconds ÷ 190 tasks ≈ 24 minutes of decode-limited
runtime — almost exactly its observed steady-state duration.

## Per-pipeline detail

### Ray Core — hand-rolled tasks + actors (`caption_ray_core.py`)

- Job: `prodjob_ahv3si4f9k7u89l7stm3wevz2r`
  ([console](https://console.anyscale-staging.com/jobs/prodjob_ahv3si4f9k7u89l7stm3wevz2r))
- Output: `…/outputs/ray_core/20260714-144309/`
- Pre-provisions 256 engines and blocks on weight load (~3 min of the wall,
  256 GPUs held throughout) before the first caption.
- **Decode-bound at this scale**: every decoded row funnels through the
  driver's queue, and the head-node buffer cap that prevents OOM at 1M rows
  (`MAX_BUFFERED_ROWS=150000`) limits decode to ~190 concurrent shard tasks.
  Mid-run steady state showed `decoded_q≈24` with only ~30 of 512 allowed
  caption batches in flight — the pinned fleet idled at 23.8% mean GPU
  utilization waiting for frames.
- Sustained ~470–610 captions/s.

```json
{"pipeline": "ray_core", "num_gpus": 256, "num_captions": 1055410,
 "wall_seconds": 1735.6, "captions_per_sec": 608.098,
 "captions_per_gpu_per_sec": 2.3754, "video_hours_per_wall_hour": 7142.34,
 "gpu_seconds_held": 443033.6, "captions_per_gpu_hour": 8576.0,
 "utilization": {"gpu_util": {"mean": 23.8, "p95": 100.0, "n": 11072},
                 "cpu_util": {"mean": 4.8, "p95": 7.3, "n": 11418},
                 "gpus_held": {"mean": 255.3, "peak": 256.0, "gpu_seconds": 443033.6}}}
```

### Ray Data — streaming pipeline + autoscaled engine pool (`caption_ray_data.py`)

- Job: `prodjob_lhqcatakzmt2ekghuxp7am1qjb`
  ([console](https://console.anyscale-staging.com/jobs/prodjob_lhqcatakzmt2ekghuxp7am1qjb))
- Output: `…/outputs/ray_data/20260714-151624/`
- Engine pool `concurrency=(1, 256)`: ramped from 1 engine to **254–256
  active** once the window queue deepened, sustaining ~900 captions/s at full
  fleet mid-run; scaled back in during the drain (hence mean 89.8 GPUs held
  vs peak 256).
- Decode (`flat_map`) is unpinned and spread across all ~4,800 cluster CPUs
  with framework backpressure — no driver funnel, no decode starvation.
- Wall time carries the ramp plus a drain tail: straggler blocks late in the
  stream can wait on a ~2–3 min replacement-engine boot after the pool has
  scaled in. Raising the pool floor (`concurrency=(m, 256)`) trades GPU-hours
  back for tail latency.

```json
{"pipeline": "ray_data", "num_gpus": 256, "num_captions": 1055221,
 "wall_seconds": 2601.0, "captions_per_sec": 405.696,
 "captions_per_gpu_per_sec": 1.5848, "video_hours_per_wall_hour": 4765.05,
 "gpu_seconds_held": 233586.9, "captions_per_gpu_hour": 16262.9,
 "utilization": {"gpu_util": {"mean": 13.2, "p95": 87.0, "n": 16608},
                 "cpu_util": {"mean": 6.4, "p95": 21.1, "n": 17128},
                 "gpus_held": {"mean": 89.8, "peak": 256.0, "gpu_seconds": 233586.9}}}
```

Tuned run (`prodjob_6ak48lybysi4cff3wz33pdzz3w`,
[console](https://console.anyscale-staging.com/jobs/prodjob_6ak48lybysi4cff3wz33pdzz3w),
output `…/outputs/ray_data/20260715-132556/`):

```json
{"pipeline": "ray_data", "num_gpus": 256, "num_captions": 1055409,
 "wall_seconds": 1124.6, "captions_per_sec": 938.485,
 "captions_per_gpu_per_sec": 3.666, "video_hours_per_wall_hour": 11022.85,
 "gpu_seconds_held": 118480.6, "captions_per_gpu_hour": 32068.3,
 "utilization": {"gpu_util": {"mean": 30.3, "p95": 99.4, "n": 7168},
                 "cpu_util": {"mean": 16.7, "p95": 94.5, "n": 7392},
                 "gpus_held": {"mean": 105.1, "peak": 256.0, "gpu_seconds": 118480.6}}}
```

Note on `captions_per_gpu_per_sec` and `gpu_util` means: both are normalized
over the full 256-GPU fleet and the full wall clock, so they penalize the
autoscaled run for GPUs it deliberately never held. `captions_per_gpu_hour`
(normalized by GPU-seconds actually held) is the cost-comparable number.

## Context: the same corpus at one caption per clip (43.7k captions)

The shallow-queue regime, where autoscaling's advantage is largest — the
fleet is only warranted for ~2 minutes:

| Metric | Ray Core | Ray Data |
|---|---|---|
| End-to-end wall time | 308 s | 726–1,041 s across runs |
| GPUs held, mean (peak) | 252 (256) | 16–22 (82) |
| GPU-seconds held | 77,599 | ~16,200 |
| Captions / GPU-hour | 2,028 | ~9,700 (**4.8×**) |

## Reproduce

```bash
cd video_captioning
anyscale job submit -f job_ray_core.yaml   # pre-provisioned baseline
anyscale job submit -f job_ray_data.yaml   # autoscaled Ray Data pipeline
```

`report.json` with all metrics above is written next to each run's captions.
`CAPTION_WINDOW_SEC` (default 12) sets caption density; `EXPECTED_GPUS`
(default 256) sets the fleet / pool ceiling.
