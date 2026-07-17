"""CAIOS read-throughput ceiling benchmark, run from inside the cluster.

Follows CoreWeave's published warp methodology — many concurrent ~15 MiB
ranged GETs per node — but as a read-only Ray job against the already-mirrored
FineVideo shards, so nothing is wiped or written. The sweep measures:

  - endpoint: https://cwobject.com (gateway, no cache) vs http://cwlota.com
    (LOTA, node-local distributed cache)
  - concurrency per node: 100 / 300 / 500 (CoreWeave's tuning guidance)
  - cache state: a repeated pass over a small hot subset isolates cache-hot
    LOTA throughput from the cold/backend numbers

One bench actor is pinned per node (GPU pool and CPU pool separately); each
actor fans work across processes to keep the Python GIL out of the
measurement. Results print as STORAGE_BENCH JSON lines.
"""

import json
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import ray

from bench_workers import PASS_SECONDS, _proc_worker, _put_worker
from utils import dataset_source, require_s3_uri

CONCURRENCIES = [int(c) for c in os.environ.get("BENCH_READ_CONC", "100,300,500").split(",")]
WRITE_CONCURRENCIES = [int(c) for c in os.environ.get("BENCH_WRITE_CONC", "100,300,500").split(",")]
HOT_PASSES = os.environ.get("BENCH_HOT", "1") == "1"
PROCS_PER_NODE = int(os.environ.get("BENCH_PROCS_PER_NODE", "8"))
HOT_SUBSET_SHARDS = 64  # ~30 GiB — fits every node's LOTA cache slice
BENCH_READ = os.environ.get("BENCH_READ", "1") == "1"
BENCH_WRITE = os.environ.get("BENCH_WRITE", "0") == "1"

ENDPOINTS = [
    ("cwobject", "https://cwobject.com"),  # gateway: backend-direct, TLS
    ("cwlota", "http://cwlota.com"),  # LOTA: cache-accelerated, in-cluster
]


@ray.remote
class NodeBench:
    """One per node; spreads GETs across processes so the GIL never caps it."""

    def __init__(self, tag: str):
        self.tag = tag
        self.ip = ray.util.get_node_ip_address()

    def ping(self) -> str:
        return self.ip

    def run(self, endpoint: str, bucket: str, keys: List[Tuple[str, int]],
            concurrency: int, duration_s: int, mode: str = "get") -> Dict:
        deadline = time.time() + duration_s
        per_proc = max(1, concurrency // PROCS_PER_NODE)
        worker = _put_worker if mode == "put" else _proc_worker
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=PROCS_PER_NODE) as pool:
            parts = list(pool.map(
                worker,
                [(endpoint, bucket, keys, per_proc, deadline)] * PROCS_PER_NODE,
            ))
        wall = time.time() - t0
        total_bytes = sum(p["bytes"] for p in parts)
        lat = sorted(x for p in parts for x in p["lat"])
        pct = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))] * 1000, 1) if lat else None
        return {
            "node": self.ip,
            "tag": self.tag,
            "gib_per_s": round(total_bytes / wall / 2**30, 2),
            "reqs_per_s": round(sum(p["reqs"] for p in parts) / wall, 1),
            "errors": sum(p["errs"] for p in parts),
            "p50_ms": pct(0.50),
            "p90_ms": pct(0.90),
            "p99_ms": pct(0.99),
        }


def list_shards(bucket: str, prefix: str) -> List[Tuple[str, int]]:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3", endpoint_url="http://cwlota.com",
        region_name=os.environ.get("AWS_REGION", "US-EAST-14A"),
        config=Config(s3={"addressing_style": "virtual"}),
    )
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [(o["Key"], o["Size"]) for o in page.get("Contents", [])
                 if o["Key"].endswith(".parquet")]
    return keys


def main():
    ray.init(ignore_reinit_error=True)
    uri = require_s3_uri(dataset_source(), "dataset")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    keys = list_shards(bucket, prefix)
    total_gib = sum(s for _, s in keys) / 2**30
    print(f"dataset: {len(keys)} shards, {total_gib:.1f} GiB in s3://{bucket}/{prefix}")

    # One actor per node, per pool. num_cpus=100 admits exactly one per
    # 120-CPU node, which is what makes the per-node numbers meaningful.
    actors = []
    for tag, selector, count in (
        ("gpu-node", {"ray.io/accelerator-type": "RTX-PRO-6000"}, int(os.environ.get("BENCH_GPU_NODES", "4"))),
        ("cpu-node", {"cpu_only": "true"}, int(os.environ.get("BENCH_CPU_NODES", "4"))),
    ):
        actors += [NodeBench.options(num_cpus=100, label_selector=selector).remote(tag)
                   for _ in range(count)]
    ips = ray.get([a.ping.remote() for a in actors])
    print(f"bench actors on nodes: {ips}")

    hot_subset = keys[:HOT_SUBSET_SHARDS]

    def one_pass(name: str, endpoint: str, key_set, concurrency: int, mode: str = "get"):
        per_node = ray.get([
            a.run.remote(endpoint, bucket, key_set, concurrency, PASS_SECONDS, mode)
            for a in actors
        ])
        agg = round(sum(r["gib_per_s"] for r in per_node), 1)
        by_tag: Dict[str, List[float]] = {}
        for r in per_node:
            by_tag.setdefault(r["tag"], []).append(r["gib_per_s"])
        line = {
            "config": name,
            "concurrency_per_node": concurrency,
            "aggregate_gib_per_s": agg,
            "per_node_gib_per_s": {t: round(sum(v) / len(v), 2) for t, v in by_tag.items()},
            "errors": sum(r["errors"] for r in per_node),
            "p50_ms": per_node[0]["p50_ms"],
            "p99_ms": max(r["p99_ms"] or 0 for r in per_node),
            "nodes": per_node,
        }
        print("STORAGE_BENCH " + json.dumps(line), flush=True)

    if BENCH_READ:
        # Cold/backend vs LOTA, swept across concurrency, whole 630 GiB keyspace.
        for ep_name, ep_url in ENDPOINTS:
            for conc in CONCURRENCIES:
                one_pass(f"{ep_name}", ep_url, keys, conc)

        if HOT_PASSES:
            # Cache-hot LOTA: warm a small subset, then measure the repeat pass.
            one_pass("cwlota-warmup", "http://cwlota.com", hot_subset, 300)
            one_pass("cwlota-hot", "http://cwlota.com", hot_subset, 300)
            one_pass("cwlota-hot-500", "http://cwlota.com", hot_subset, 500)

    if BENCH_WRITE:
        # PUT ceiling: 50 MiB objects to a scratch prefix, then delete them.
        # Writes are not LOTA-cached (proxied through), so cwlota is the same
        # path the pipelines use for their own output.
        scratch = prefix.rsplit("/datasets/", 1)[0] + "/bench_scratch"
        for conc in WRITE_CONCURRENCIES:
            one_pass(f"put-c{conc}", "http://cwlota.com", scratch, conc, mode="put")

        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3", endpoint_url="http://cwlota.com",
            region_name=os.environ.get("AWS_REGION", "US-EAST-14A"),
            config=Config(s3={"addressing_style": "virtual"}),
        )
        deleted = 0
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=scratch
        ):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            for i in range(0, len(objs), 1000):
                client.delete_objects(Bucket=bucket, Delete={"Objects": objs[i:i + 1000]})
                deleted += len(objs[i:i + 1000])
        print(f"cleaned up {deleted} scratch objects under {scratch}", flush=True)


if __name__ == "__main__":
    main()
