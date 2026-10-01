# syntax=docker/dockerfile:1.7
# Ray nightly plus what Ray Sandbox needs on every node (gVisor's runsc,
# slirp4netns, erofs-utils; recipe from
# https://docs.ray.io/en/master/ray-core/sandboxes.html), grpclib for the
# Modal-compatible gRPC facade, and the facade's gRPC routes.
ARG GVISOR_VERSION=20260921.0
ARG SLIRP4NETNS_VERSION=1.3.5
ARG EROFS_UTILS_VERSION=1.9.4
# The facade's merge into Ray master (ray-project/ray#65839).
ARG RAY_FACADE_COMMIT=ce7f78f2c2b6c3dfc425ac759155cfeaa157ff51

FROM anyscale/ray:nightly-py312 AS build
ARG GVISOR_VERSION
ARG SLIRP4NETNS_VERSION
ARG EROFS_UTILS_VERSION
ARG RAY_FACADE_COMMIT

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends autoconf automake bzip2 \
        ca-certificates curl gcc git libtool liblz4-dev make pkg-config uuid-dev

# gVisor, with the gvisor-bin/ helpers that must stay next to runsc
RUN mkdir -p /out/bin \
    && curl -fsSL "https://storage.googleapis.com/gvisor/releases/release/${GVISOR_VERSION}/$(uname -m)/gvisor.tar.bz2" \
        | tar -xj -C /out/bin

# slirp4netns, for network="public"
RUN curl -fsSL -o /out/bin/slirp4netns \
        "https://github.com/rootless-containers/slirp4netns/releases/download/v${SLIRP4NETNS_VERSION}/slirp4netns-$(uname -m)" \
    && chmod a+rx /out/bin/slirp4netns

# erofs-utils
RUN curl -fsSL "https://github.com/erofs/erofs-utils/archive/refs/tags/v${EROFS_UTILS_VERSION}.tar.gz" \
        | tar -xz -C /tmp \
    && cd "/tmp/erofs-utils-${EROFS_UTILS_VERSION}" \
    && ./autogen.sh \
    && ./configure --disable-fuse --prefix=/out \
    && make -j"$(nproc)" \
    && make install

# Temporary: a nightly built before the facade's merge lacks it, so take
# master's ray.experimental.sandbox package from that commit. Once the nightly
# ships the facade, this step leaves /out/overlay empty.
RUN mkdir -p /out/overlay \
    && if ! python -c "import ray.experimental.sandbox.http.grpc_facade" 2>/dev/null; then \
        git init -q /tmp/ray && cd /tmp/ray \
        && git remote add origin https://github.com/ray-project/ray.git \
        && git sparse-checkout init --cone \
        && git sparse-checkout set python/ray/experimental/sandbox \
        && git fetch -q --depth 1 --filter=blob:none origin "${RAY_FACADE_COMMIT}" \
        && git checkout -q FETCH_HEAD \
        && mkdir -p /out/overlay/pkg \
        && cp -r python/ray/experimental/sandbox /out/overlay/pkg/sandbox; \
    fi

FROM anyscale/ray:nightly-py312
COPY --from=build /out/bin/ /usr/local/bin/
COPY --from=build /out/overlay/ /opt/ray-sandbox-overlay/
RUN pip install --no-cache-dir grpclib \
    && sudo mkdir -p /opt/ray-sandbox-overlay/pylib /opt/ray-sandbox-modal \
    && sudo chown -R ray /opt/ray-sandbox-overlay /opt/ray-sandbox-modal

# If the build stage made an overlay, load this import hook in every Python
# process (from a .pth file) so ray.experimental.sandbox resolves to it.
RUN cat > /opt/ray-sandbox-overlay/pylib/ray_sandbox_overlay_graft.py <<'EOF'
"""Route ``ray.experimental.sandbox`` to the overlay copy under /opt.

Anyscale re-syncs its managed ``ray`` package into the image's Python env at
cluster start, so files placed inside ``site-packages/ray`` do not survive.
This finder is installed from a ``.pth`` file and wins over the shipped copy
in every Python process (drivers, workers, actors).
"""

import importlib.abc
import importlib.util
import os
import sys

_ROOT = "/opt/ray-sandbox-overlay/pkg/sandbox"
_PREFIX = "ray.experimental.sandbox"


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _PREFIX and not fullname.startswith(_PREFIX + "."):
            return None
        rel = fullname[len(_PREFIX):].lstrip(".").replace(".", os.sep)
        base = os.path.join(_ROOT, rel) if rel else _ROOT
        if os.path.isdir(base):
            init = os.path.join(base, "__init__.py")
            if not os.path.exists(init):
                return None
            return importlib.util.spec_from_file_location(
                fullname, init, submodule_search_locations=[base]
            )
        module = base + ".py"
        if os.path.exists(module):
            return importlib.util.spec_from_file_location(fullname, module)
        return None


if not any(isinstance(f, _OverlayFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _OverlayFinder())
EOF
RUN if [ -d /opt/ray-sandbox-overlay/pkg/sandbox ]; then \
        SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
        && printf '/opt/ray-sandbox-overlay/pylib\nimport ray_sandbox_overlay_graft\n' > "$SITE/zz_ray_sandbox_overlay.pth"; \
    fi

# The facade's gRPC routes. Serve's gRPC proxies import them when they start,
# so they live in the image; the Serve app itself is modal_facade.py in the
# working directory.
RUN cat > /opt/ray-sandbox-modal/modal_grpc.py <<'EOF'
"""gRPC method tables that route Modal SDK calls to the Ray Sandbox facade.

Serve's gRPC proxy takes grpcio ``add_*Servicer_to_server`` functions but only
reads their method tables; each call then goes to the ingress deployment's
method named after the RPC (see modal_facade.py). The tables come from the
facade's vendored protos, under the service names the Modal SDK calls.

An Anyscale service routes gRPC for one service name, so the command router's
methods are also served under ``modal.client.ModalClient``; modal_relay.py
sends router calls there. The two services share no method names.
"""

import grpc
from google.protobuf import message_factory

from ray.experimental.sandbox.http._proto import (
    sandbox_control_pb2,
    sandbox_exec_pb2,
)

_CONTROL = sandbox_control_pb2.DESCRIPTOR.services_by_name["ModalClient"]
_ROUTER = sandbox_exec_pb2.DESCRIPTOR.services_by_name["TaskCommandRouter"]
# The vendored protos carry this package prefix; it never reaches the wire.
_PACKAGE_PREFIX = "ray_sandbox_facade."
_HANDLER_KINDS = {
    (False, False): grpc.unary_unary_rpc_method_handler,
    (False, True): grpc.unary_stream_rpc_method_handler,
    (True, False): grpc.stream_unary_rpc_method_handler,
    (True, True): grpc.stream_stream_rpc_method_handler,
}


def _method_table(servicer, methods):
    return {
        m.name: _HANDLER_KINDS[(m.client_streaming, m.server_streaming)](
            getattr(servicer, m.name),
            request_deserializer=message_factory.GetMessageClass(m.input_type).FromString,
            response_serializer=message_factory.GetMessageClass(
                m.output_type
            ).SerializeToString,
        )
        for m in methods
    }


def add_to_server(servicer, server):
    """Register both services, plus the router's methods under the control plane."""
    for service, methods in (
        (_CONTROL, [*_CONTROL.methods, *_ROUTER.methods]),
        (_ROUTER, list(_ROUTER.methods)),
    ):
        name = service.full_name[len(_PACKAGE_PREFIX) :]
        server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler(name, _method_table(servicer, methods)),)
        )
EOF
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
    && echo /opt/ray-sandbox-modal > "$SITE/ray_sandbox_modal.pth" \
    && runsc --version | head -1 && slirp4netns --version | head -1 && mkfs.erofs --version \
    && python -c "import ray, ray.experimental.sandbox.http.grpc_facade as f, modal_grpc; print('ray', ray.__version__, ray.__commit__[:10], '| facade', f.__file__)"
