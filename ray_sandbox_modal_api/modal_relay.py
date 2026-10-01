"""Localhost relay between an unmodified Modal SDK and the ray-sandbox-modal service.

The Modal SDK talks gRPC to two services: the control plane
(``modal.client.ModalClient``, at MODAL_SERVER_URL) and the command router
(``modal.task_command_router.TaskCommandRouter``, at a URL the control plane
hands out). An Anyscale service routes gRPC for a single service name, so the
service also serves the router's methods under the control-plane name. This
relay listens on localhost and forwards every call to the service over TLS
with the service's bearer token. It moves router calls under the
control-plane name and answers ``TaskGetCommandRouterAccess`` with its own
address, so the SDK sends router calls back through it.

It also hides two limits of the service's edge from the SDK:

- the edge rejects a request body over 1 MiB (HTTP 413), and a client stream
  is one body, so stdin uploads (file writes) go up in segments that the
  facade resumes by offset;
- the edge cuts a stream that has been idle for 60 s, so a stdout/stderr read
  waiting on a quiet command is re-opened at its offset while the SDK's
  stream stays open.

    export SANDBOX_SERVICE_TOKEN=<the service's query_auth_token>
    python modal_relay.py --url <the service's query_url> [--port 50051]
    export MODAL_SERVER_URL=http://127.0.0.1:50051
    export MODAL_TOKEN_ID=ak-unused MODAL_TOKEN_SECRET=as-unused

Needs only the Modal SDK, which brings grpclib and the proto descriptors.
"""

import argparse
import asyncio
import logging
import os
import ssl
import time
import urllib.parse

from grpclib import GRPCError
from grpclib.client import Channel
from grpclib.const import Cardinality, Handler, Status
from grpclib.encoding.base import CodecBase
from grpclib.exceptions import StreamTerminatedError
from grpclib.server import Server
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

logger = logging.getLogger("modal_relay")

_CONTROL = "modal.client.ModalClient"
# Set by the relay itself or specific to the local hop.
_DROPPED_METADATA = {"authorization", "user-agent"}
# The edge answers HTTP 413 once one request body (a whole client stream)
# passes 1 MiB; three of the SDK's 256 KiB chunks per stream stay under it.
_STDIN_SEGMENT_BYTES = 768 * 1024
# A read that fails after this long without data was cut for idling.
_IDLE_CUT_SECONDS = 30.0
_EDGE_ERRORS = (StreamTerminatedError, ConnectionError, OSError, asyncio.TimeoutError)


class _RawCodec(CodecBase):
    """Passes serialized messages through untouched."""

    __content_subtype__ = "proto"

    def encode(self, message, message_type):
        return message

    def decode(self, data, message_type):
        return data


def _logged(name, handler):
    async def handle(stream) -> None:
        try:
            await handler(stream)
        except Exception as exc:
            logger.debug("%s failed: %r", name, exc)
            raise

    return handle


class Relay:
    def __init__(self, url: str, token: str, own_url: str) -> None:
        o = urllib.parse.urlparse(url)
        tls = None
        if o.scheme == "https":
            tls = ssl.create_default_context()
            # Without ALPN h2 the service's ingress falls back to HTTP/1.1.
            tls.set_alpn_protocols(["h2"])
        port = o.port or (443 if tls else 80)
        self._channel = Channel(o.hostname, port, ssl=tls, codec=_RawCodec())
        self._auth = [("authorization", f"Bearer {token}")] if token else []
        self._router_access = api_pb2.TaskGetCommandRouterAccessResponse(
            url=own_url, jwt="modal-relay"
        ).SerializeToString()

    def __mapping__(self):
        special = {
            "TaskGetCommandRouterAccess": self._router_access_handler,
            "TaskExecStdinWriteStream": self._stdin_stream_handler,
            "TaskExecStdioRead": self._stdio_read_handler,
        }
        mapping = {}
        for pb2, name in ((api_pb2, "ModalClient"), (sr_pb2, "TaskCommandRouter")):
            service = pb2.DESCRIPTOR.services_by_name[name]
            for method in service.methods:
                cardinality = Cardinality(
                    (method.client_streaming, method.server_streaming)
                )
                path = f"/{_CONTROL}/{method.name}"
                handler = special.get(method.name, self._forward_handler)(
                    path, cardinality
                )
                mapping[f"/{service.full_name}/{method.name}"] = Handler(
                    _logged(method.name, handler), cardinality, None, None
                )
        return mapping

    def _upstream(self, stream, path: str, cardinality: Cardinality):
        metadata = [
            (key, value)
            for key, value in stream.metadata.items()
            if key not in _DROPPED_METADATA
        ] + self._auth
        return self._channel.request(
            path, cardinality, None, None, deadline=stream.deadline, metadata=metadata
        )

    async def _unary_reply(self, up) -> bytes:
        reply = await up.recv_message()
        # Raises the service's error status, which the relay passes on in
        # place of the reply.
        await up.recv_trailing_metadata()
        return reply

    def _router_access_handler(self, path, cardinality):
        async def handle(stream) -> None:
            await stream.recv_message()
            await stream.send_message(self._router_access)

        return handle

    def _forward_handler(self, path, cardinality):
        async def handle(stream) -> None:
            try:
                async with self._upstream(stream, path, cardinality) as up:
                    if cardinality.client_streaming:
                        while (message := await stream.recv_message()) is not None:
                            await up.send_message(message)
                        await up.end()
                    else:
                        await up.send_message(await stream.recv_message(), end=True)
                    if cardinality.server_streaming:
                        async for message in up:
                            await stream.send_message(message)
                        await up.recv_trailing_metadata()
                    else:
                        await stream.send_message(await self._unary_reply(up))
            except _EDGE_ERRORS as exc:
                # The SDK retries UNAVAILABLE.
                raise GRPCError(Status.UNAVAILABLE, f"relay upstream: {exc!r}")

        return handle

    def _stdin_stream_handler(self, path, cardinality):
        async def handle(stream) -> None:
            first = sr_pb2.TaskExecStdinWriteStreamRequest.FromString(
                await stream.recv_message()
            )
            start = first.start
            offset = start.offset
            segment, size, reply = [], 0, None

            async def flush(final: bool) -> None:
                nonlocal offset, segment, size, reply
                head = sr_pb2.TaskExecStdinWriteStreamRequest(
                    start=sr_pb2.TaskExecStdinWriteStreamStart(
                        task_id=start.task_id, exec_id=start.exec_id, offset=offset
                    )
                )
                try:
                    async with self._upstream(stream, path, cardinality) as up:
                        await up.send_message(head.SerializeToString())
                        for message in segment:
                            await up.send_message(message)
                        if final:
                            await up.send_message(
                                sr_pb2.TaskExecStdinWriteStreamRequest(
                                    end=sr_pb2.TaskExecStdinWriteStreamEnd()
                                ).SerializeToString()
                            )
                        await up.end()
                        reply = await self._unary_reply(up)
                except _EDGE_ERRORS as exc:
                    raise GRPCError(Status.UNAVAILABLE, f"relay upstream: {exc!r}")
                offset += size
                segment, size = [], 0

            while (raw := await stream.recv_message()) is not None:
                request = sr_pb2.TaskExecStdinWriteStreamRequest.FromString(raw)
                if request.WhichOneof("payload") == "end":
                    await flush(final=True)
                    break
                segment.append(raw)
                size += len(request.data)
                if size >= _STDIN_SEGMENT_BYTES:
                    await flush(final=False)
            else:
                await flush(final=False)
            await stream.send_message(reply)

        return handle

    def _stdio_read_handler(self, path, cardinality):
        async def handle(stream) -> None:
            request = sr_pb2.TaskExecStdioReadRequest.FromString(
                await stream.recv_message()
            )
            while True:
                last_data = time.monotonic()
                try:
                    async with self._upstream(stream, path, cardinality) as up:
                        await up.send_message(request.SerializeToString(), end=True)
                        async for raw in up:
                            await stream.send_message(raw)
                            request.offset += len(
                                sr_pb2.TaskExecStdioReadResponse.FromString(raw).data
                            )
                            last_data = time.monotonic()
                        await up.recv_trailing_metadata()
                    return
                except (GRPCError, *_EDGE_ERRORS) as exc:
                    idle = time.monotonic() - last_data
                    expired = (
                        stream.deadline is not None
                        and stream.deadline.time_remaining() <= 0
                    )
                    service_error = isinstance(exc, GRPCError) and exc.status not in (
                        Status.CANCELLED,
                        Status.UNAVAILABLE,
                        Status.UNKNOWN,
                        Status.INTERNAL,
                    )
                    if service_error or expired or idle < _IDLE_CUT_SECONDS:
                        if isinstance(exc, GRPCError):
                            raise
                        raise GRPCError(Status.UNAVAILABLE, f"relay upstream: {exc!r}")
                    logger.debug("Re-opening stdio read at %d after %r", request.offset, exc)

        return handle


async def serve(url: str, token: str, host: str, port: int) -> None:
    relay = Relay(url, token, own_url=f"http://{host}:{port}")
    server = Server([relay], codec=_RawCodec())
    await server.start(host, port)
    logger.info("Relaying %s:%d to %s", host, port, url)
    await server.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True, help="the service's query_url")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    token = os.environ.get("SANDBOX_SERVICE_TOKEN", "")
    asyncio.run(serve(args.url, token, args.host, args.port))


if __name__ == "__main__":
    main()
