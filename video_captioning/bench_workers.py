"""Process-pool workers for storage_bench.py.

Lives in its own importable module (not __main__) so ProcessPoolExecutor
children can resolve the functions by reference after fork.
"""

import os
import random
import threading
import time
from typing import Dict

RANGE_BYTES = 15 * 1024 * 1024  # CoreWeave guidance: >=15 MiB per request
PUT_OBJECT_BYTES = 50 * 1024 * 1024  # CoreWeave guidance: >=50 MB objects/parts
PASS_SECONDS = int(os.environ.get("BENCH_PASS_SECONDS", "45"))

def _proc_worker(args) -> Dict:
    """One process: `threads` looping ranged GETs until the deadline."""
    endpoint, bucket, keys, threads, deadline = args
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_REGION", "US-EAST-14A"),
        config=Config(
            s3={"addressing_style": "virtual"},  # path-style errors on CAIOS
            max_pool_connections=threads + 16,
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )
    lock = threading.Lock()
    stats = {"bytes": 0, "reqs": 0, "errs": 0, "lat": []}

    def loop():
        rnd = random.Random(threading.get_ident() ^ os.getpid())
        while time.time() < deadline:
            key, size = rnd.choice(keys)
            start = 0 if size <= RANGE_BYTES else rnd.randrange(0, size - RANGE_BYTES)
            t0 = time.time()
            try:
                r = client.get_object(
                    Bucket=bucket, Key=key,
                    Range=f"bytes={start}-{start + RANGE_BYTES - 1}",
                )
                n = len(r["Body"].read())
            except Exception:
                with lock:
                    stats["errs"] += 1
                continue
            dt = time.time() - t0
            with lock:
                stats["bytes"] += n
                stats["reqs"] += 1
                if len(stats["lat"]) < 20000:
                    stats["lat"].append(dt)

    ts = [threading.Thread(target=loop, daemon=True) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(PASS_SECONDS + 120)
    return stats


def _put_worker(args) -> Dict:
    """One process: `threads` looping 50 MiB PutObject calls until deadline.

    One shared read-only payload per process — generating fresh random bytes
    per request would benchmark os.urandom, not the object store.
    """
    endpoint, bucket, prefix, threads, deadline = args
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_REGION", "US-EAST-14A"),
        config=Config(
            s3={"addressing_style": "virtual"},
            max_pool_connections=threads + 16,
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )
    payload = os.urandom(PUT_OBJECT_BYTES)
    lock = threading.Lock()
    stats = {"bytes": 0, "reqs": 0, "errs": 0, "lat": []}

    def loop():
        i = 0
        base = f"{prefix}/{os.getpid()}-{threading.get_ident()}"
        while time.time() < deadline:
            t0 = time.time()
            try:
                client.put_object(Bucket=bucket, Key=f"{base}-{i}", Body=payload)
            except Exception:
                with lock:
                    stats["errs"] += 1
                continue
            finally:
                i += 1
            dt = time.time() - t0
            with lock:
                stats["bytes"] += PUT_OBJECT_BYTES
                stats["reqs"] += 1
                if len(stats["lat"]) < 20000:
                    stats["lat"].append(dt)

    ts = [threading.Thread(target=loop, daemon=True) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(PASS_SECONDS + 300)
    return stats
