"""End-to-end check of the Modal-compatible API with the unmodified Modal SDK.

Connection settings come from the environment (MODAL_SERVER_URL,
MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, MODAL_OVERRIDE_HEADERS); see README.md.
"""

import time

import modal
from modal.exception import SandboxFilesystemNotFoundError

IMAGE = "python:3.12-slim"


def step(label: str, start: float) -> float:
    now = time.monotonic()
    print(f"ok  {label} ({now - start:.2f}s)", flush=True)
    return now


def main() -> None:
    t = time.monotonic()
    app = modal.App.lookup("ray-sandbox-smoke", create_if_missing=True)
    sb = modal.Sandbox.create(
        app=app,
        image=modal.Image.from_registry(IMAGE),
        cpu=1,
        memory=1024,
        timeout=600,
        workdir="/tmp",
        secrets=[modal.Secret.from_dict({"SMOKE_GREETING": "hi"})],
    )
    t = step(f"create {sb.object_id}", t)

    p = sb.exec("python", "-c", "import sys; print('hello'); sys.exit(3)")
    assert p.wait() == 3
    assert p.stdout.read() == "hello\n"
    t = step("exec: exit code and stdout", t)

    p = sb.exec("sh", "-c", 'echo "$SMOKE_GREETING $(pwd)"')
    assert p.wait() == 0 and p.stdout.read() == "hi /tmp\n"
    t = step("exec: secret env and workdir", t)

    p = sb.exec("sh", "-c", "echo oops >&2")
    assert p.wait() == 0 and p.stderr.read() == "oops\n"
    t = step("exec: stderr", t)

    p = sb.exec(
        "sh", "-c", "for i in 1 2 3; do echo line $i; sleep 0.2; done", bufsize=1
    )
    assert list(p.stdout) == ["line 1\n", "line 2\n", "line 3\n"]
    assert p.wait() == 0
    t = step("exec: stream stdout lines", t)

    payload = bytes(range(256)) * 4096  # 1 MiB, every byte value
    sb.filesystem.write_bytes(payload, "/tmp/data.bin")
    assert sb.filesystem.read_bytes("/tmp/data.bin") == payload
    try:
        sb.filesystem.read_bytes("/tmp/missing.txt")
        raise AssertionError("reading a missing file succeeded")
    except SandboxFilesystemNotFoundError:
        pass
    t = step("filesystem: 1 MiB write/read, missing file", t)

    p = sb.exec(
        "python",
        "-c",
        "import urllib.request; "
        "print(urllib.request.urlopen('https://pypi.org/simple/', timeout=20).status)",
    )
    assert p.wait() == 0 and p.stdout.read().strip() == "200", p.stderr.read()
    t = step("network: HTTPS egress from the sandbox", t)

    same = modal.Sandbox.from_id(sb.object_id)
    assert same.exec("true").wait() == 0
    t = step("reattach by id", t)

    sb.terminate()
    assert sb.poll() is not None
    step("terminate", t)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
