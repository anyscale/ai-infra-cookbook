"""Mirror FineVideo and Qwen3-VL into CoreWeave AI Object Storage.

The job resolves each Hugging Face repository to an immutable commit, then
streams every file directly from the Hub to S3-compatible object storage. It
never stages a repository on shared cluster storage. Existing destination
objects with the expected size are skipped, so interrupted runs are resumable.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import ray
import requests
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.hf_api import RepoFile
from pyarrow.fs import FileType
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils import ai_storage_uri, filesystem_and_path


MODEL_ID = os.environ.get("HF_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
DATASET_ID = os.environ.get("HF_DATASET_ID", "HuggingFaceFV/finevideo")
MODEL_REVISION = os.environ.get("HF_MODEL_REVISION", "main")
DATASET_REVISION = os.environ.get("HF_DATASET_REVISION", "main")
COPY_CHUNK_BYTES = int(os.environ.get("COPY_CHUNK_MB", "16")) * 1024 * 1024
MIRROR_CONCURRENCY = int(os.environ.get("MIRROR_CONCURRENCY", "16"))
FORCE_COPY = os.environ.get("FORCE_COPY", "0") == "1"


def _http_session() -> requests.Session:
    """Build a session that honors Retry-After and backs off on Hub/CDN limits."""
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


@ray.remote(num_cpus=1, max_retries=3, retry_exceptions=True)
def copy_file(
    repo_id: str,
    repo_type: str,
    revision: str,
    relative_path: str,
    expected_size: int,
    expected_sha256: str,
    destination_root: str,
) -> Tuple[str, str, int]:
    """Stream one immutable Hub file to object storage with one resolver GET."""
    destination_fs, destination_prefix = filesystem_and_path(destination_root)
    destination_path = f"{destination_prefix}/{relative_path}"

    if not FORCE_COPY:
        info = destination_fs.get_file_info(destination_path)
        if info.type == FileType.File and info.size == expected_size:
            return "skipped", relative_path, expected_size

    source_url = hf_hub_url(
        repo_id,
        relative_path,
        repo_type=repo_type,
        revision=revision,
    )
    copied = 0
    digest = hashlib.sha256() if expected_sha256 else None
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        "Accept-Encoding": "identity",
    }
    with _http_session().get(
        source_url,
        headers=headers,
        stream=True,
        timeout=(30, 900),
    ) as source:
        source.raise_for_status()
        with destination_fs.open_output_stream(destination_path) as destination:
            for chunk in source.iter_content(chunk_size=COPY_CHUNK_BYTES):
                if not chunk:
                    continue
                destination.write(chunk)
                copied += len(chunk)
                if digest:
                    digest.update(chunk)

    if copied != expected_size:
        raise IOError(
            f"Size mismatch for {relative_path}: copied {copied}, "
            f"expected {expected_size}"
        )
    if digest and digest.hexdigest() != expected_sha256:
        destination_fs.delete_file(destination_path)
        raise IOError(
            f"SHA-256 mismatch for {relative_path}: copied {digest.hexdigest()}, "
            f"expected {expected_sha256}"
        )
    return "copied", relative_path, copied


def _repo_files(
    api: HfApi, repo_id: str, repo_type: str, revision: str
) -> Tuple[str, List[RepoFile]]:
    info = api.repo_info(repo_id, repo_type=repo_type, revision=revision)
    resolved_revision = info.sha
    files = [
        entry
        for entry in api.list_repo_tree(
            repo_id,
            repo_type=repo_type,
            revision=resolved_revision,
            recursive=True,
        )
        if isinstance(entry, RepoFile)
    ]
    return resolved_revision, files


def _manifest_file(entry: RepoFile) -> Dict[str, Any]:
    lfs = getattr(entry, "lfs", None)
    return {
        "path": entry.path,
        "size": entry.size,
        "blob_id": getattr(entry, "blob_id", None),
        "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
    }


def _write_json(uri: str, value: Dict[str, Any]) -> None:
    fs, path = filesystem_and_path(uri)
    body = json.dumps(value, indent=2, sort_keys=True).encode()
    with fs.open_output_stream(path) as stream:
        stream.write(body)


def _drain_copies(
    requests: Iterable[Tuple[str, str, str, str, int, str, str]]
) -> Dict[str, int]:
    pending = iter(requests)
    inflight: Dict[Any, str] = {}
    stats = {"copied_files": 0, "copied_bytes": 0, "skipped_files": 0}

    def submit_one() -> bool:
        try:
            request = next(pending)
        except StopIteration:
            return False
        ref = copy_file.options(
            label_selector={"cpu_only": "true"}
        ).remote(*request)
        inflight[ref] = request[3]
        return True

    for _ in range(MIRROR_CONCURRENCY):
        if not submit_one():
            break

    while inflight:
        ready, _ = ray.wait(list(inflight), num_returns=1)
        ref = ready[0]
        inflight.pop(ref)
        status, path, size = ray.get(ref)
        stats[f"{status}_files"] += 1
        if status == "copied":
            stats["copied_bytes"] += size
        print(f"{status:7s} {size:>12,d}  {path}", flush=True)
        submit_one()

    return stats


def mirror_repo(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    requested_revision: str,
    destination_root: str,
) -> None:
    revision, files = _repo_files(
        api, repo_id, repo_type, requested_revision
    )
    total_bytes = sum(entry.size for entry in files)
    print(
        f"Mirroring {repo_type} {repo_id}@{revision}: {len(files)} files, "
        f"{total_bytes:,} bytes -> {destination_root}",
        flush=True,
    )

    requests = (
        (
            repo_id,
            repo_type,
            revision,
            entry.path,
            entry.size,
            getattr(getattr(entry, "lfs", None), "sha256", None) or "",
            destination_root,
        )
        for entry in files
    )
    stats = _drain_copies(requests)
    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "source": {
            "provider": "huggingface",
            "repo_id": repo_id,
            "repo_type": repo_type,
            "requested_revision": requested_revision,
            "resolved_revision": revision,
        },
        "destination": destination_root,
        "completed_at": completed_at,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "transfer": stats,
        "files": [_manifest_file(entry) for entry in files],
    }
    _write_json(f"{destination_root}/_mirror_manifest.json", manifest)
    _write_json(
        f"{destination_root}/_SUCCESS",
        {
            "completed_at": completed_at,
            "resolved_revision": revision,
            "file_count": len(files),
            "total_bytes": total_bytes,
        },
    )
    print(f"Mirror complete: {destination_root}", flush=True)


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required because FineVideo is a gated dataset."
        )
    if MIRROR_CONCURRENCY < 1:
        raise ValueError("MIRROR_CONCURRENCY must be at least 1")

    model_destination = ai_storage_uri("models/Qwen3-VL-8B-Instruct")
    dataset_destination = ai_storage_uri("datasets/finevideo")
    print(f"AI storage root: {ai_storage_uri('')}", flush=True)

    ray.init(ignore_reinit_error=True)
    api = HfApi(token=token)
    mirror_repo(
        api,
        MODEL_ID,
        "model",
        MODEL_REVISION,
        model_destination,
    )
    mirror_repo(
        api,
        DATASET_ID,
        "dataset",
        DATASET_REVISION,
        dataset_destination,
    )


if __name__ == "__main__":
    main()
