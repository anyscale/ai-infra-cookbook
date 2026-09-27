# Modal-compatible sandboxes on Ray

Run unmodified [Modal](https://github.com/modal-labs/modal-client) sandbox code
(`Sandbox.create`, `exec`, `filesystem`) on your own Ray cluster. Each sandbox
is a [gVisor](https://github.com/google/gvisor) container managed by
[Ray Sandbox](https://docs.ray.io/en/master/ray-core/sandboxes.html), and an
Anyscale service serves Ray's Modal-compatible gRPC API.

## The stack

| Layer | Library | Role in this example |
|---|---|---|
| Isolation | [gVisor](https://github.com/google/gvisor) | Runs each sandbox under a user-space kernel |
| Sandboxes | [Ray Sandbox](https://github.com/ray-project/ray/tree/master/python/ray/experimental/sandbox) | One Ray actor per sandbox, plus the Modal-compatible gRPC facade |
| Orchestration | [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) | Serves the facade over gRPC |
| Client | [Modal SDK](https://github.com/modal-labs/modal-client) | Your existing sandbox code, unchanged |
| Platform | [Anyscale](https://www.anyscale.com) | Image build, compute provisioning, TLS and token auth for the service |

## Install the Anyscale CLI

```bash
pip install -U anyscale
anyscale login
```

## Deploy the service

Clone the example from GitHub.

```bash
git clone https://github.com/anyscale/examples.git
cd examples/ray_sandbox_modal_api
```

Deploy the service. It builds the image from the Dockerfile first and needs an
Anyscale cloud on Kubernetes (set `cloud:` in `service.yaml` if it isn't your
default).

```bash
anyscale service deploy -f service.yaml
```

## Connect the Modal SDK

Start the relay on the machine that runs your Modal code, then point the SDK at
it.

```bash
pip install "modal==1.5.5"
STATUS=$(anyscale service status -n ray-sandbox-modal --json)
export SANDBOX_SERVICE_TOKEN=$(jq -r .query_auth_token <<<"$STATUS")
python modal_relay.py --url "$(jq -r .query_url <<<"$STATUS")" &

export MODAL_SERVER_URL=http://127.0.0.1:50051
export MODAL_TOKEN_ID=ak-unused MODAL_TOKEN_SECRET=as-unused  # required by the SDK, not checked
python smoke_test.py
```

The last line of the smoke test is `SMOKE TEST PASSED`. From then on, Modal code
runs unchanged:

```python
import modal

app = modal.App.lookup("my-app", create_if_missing=True)
sb = modal.Sandbox.create(
    app=app, image=modal.Image.from_registry("python:3.12-slim"), cpu=1, memory=1024, timeout=600
)
p = sb.exec("python", "-c", "print('hello')")
print(p.wait(), p.stdout.read())  # 0 hello
sb.terminate()
```

Tools built on the Modal SDK work the same way. For example, Harbor runs
Terminal-Bench with `harbor run -p <tasks> -a oracle -e modal` (install
`harbor[modal]`).

## Understanding the example

- [Dockerfile](https://github.com/anyscale/examples/blob/main/ray_sandbox_modal_api/Dockerfile)
  adds what Ray Sandbox needs on each node (gVisor's `runsc`, `slirp4netns`,
  `erofs-utils`) to Ray's nightly image, following the Ray docs. It also
  writes `modal_grpc.py`, the facade's gRPC routes, which Serve's proxies
  import when they start. While the nightly image predates the facade's
  merge into Ray master
  ([ray-project/ray#65839](https://github.com/ray-project/ray/pull/65839)),
  it also overlays master's sandbox package.
- [modal_facade.py](https://github.com/anyscale/examples/blob/main/ray_sandbox_modal_api/modal_facade.py)
  is the Serve app. It runs Ray's facade (`ray.experimental.sandbox.http.grpc_facade`)
  as one replica on the head node. Each sandbox is a `SandboxHost` actor on a
  worker node.
- [service.yaml](https://github.com/anyscale/examples/blob/main/ray_sandbox_modal_api/service.yaml)
  gives the worker pods what gVisor needs and keeps the service's token auth
  on, because the facade has no authentication of its own.
- [modal_relay.py](https://github.com/anyscale/examples/blob/main/ray_sandbox_modal_api/modal_relay.py)
  adapts the SDK to three limits of the Anyscale service edge:

  | Edge limit | What the relay does |
  |---|---|
  | One gRPC service name per service, but the SDK calls two | Forwards every call under the one name and gives the SDK its own address for the second service |
  | A request body over 1 MiB gets HTTP 413 | Splits file uploads into 768 KiB streams |
  | A stream idle for 60 s gets HTTP 504 | Reopens a read that's waiting on a quiet command |

What works: `Sandbox.create` from a registry image (with `cpu`, `memory`,
`timeout`, `workdir`, `secrets`, `block_network`, `name`, and `tags`), `exec`,
`filesystem` read, write, and list, `from_id`, `poll`, `wait`, and `terminate`.

What doesn't: image builds beyond a single `FROM`, writing to an exec's stdin,
network allowlists, and Modal Functions, Volumes, or tunnels.

Sandboxes get open internet access by default, and that includes the cluster's
internal endpoints. Use `block_network=True` for untrusted code.

## Shutdown

```bash
anyscale service terminate -n ray-sandbox-modal
```

## Position in the stack

**Stage:** Post-train

- **Related:** [skyrl](../skyrl/) — RL post-training, where agent rollouts
  need sandboxed code execution

Part of the [Open-Source Frontier Infra Stack](../README.md) — explore the
map in the [interactive explorer](../README.md#interactive-explorer).
