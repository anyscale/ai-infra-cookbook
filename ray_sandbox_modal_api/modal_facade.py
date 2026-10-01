"""Ray Serve app that serves Ray Sandbox's Modal-compatible gRPC facade.

Serve's gRPC proxy calls the ingress deployment's method named after each RPC;
the routes come from ``modal_grpc.add_to_server``, which the image provides.
Each method here runs the facade's own grpclib handler on a minimal stream
object and turns the handler's ``GRPCError`` into the call's status. The
facade keeps its exec table in memory, so the deployment runs one replica.
"""

import asyncio
import os
from typing import Any, AsyncIterator

import grpc
from google.protobuf import message_factory
from grpclib import GRPCError

from ray import serve
from ray.experimental.sandbox.http._proto import (
    sandbox_control_pb2,
    sandbox_exec_pb2,
)
from ray.experimental.sandbox.http.grpc_facade import build_servicers

_CONTROL = sandbox_control_pb2.DESCRIPTOR.services_by_name["ModalClient"]
_ROUTER = sandbox_exec_pb2.DESCRIPTOR.services_by_name["TaskCommandRouter"]
_STATUS_CODES = {code.value[0]: code for code in grpc.StatusCode}
_END = object()


class _Stream:
    """The two calls of a grpclib server stream that the facade's handlers make."""

    def __init__(self, requests: AsyncIterator[Any]) -> None:
        self._requests = requests
        self.responses: asyncio.Queue = asyncio.Queue()

    async def recv_message(self) -> Any:
        return await anext(self._requests, None)

    async def send_message(self, message: Any) -> None:
        self.responses.put_nowait(message)


async def _single(request: Any) -> AsyncIterator[Any]:
    yield request


def _set_status(grpc_context: Any, exc: GRPCError) -> None:
    grpc_context.set_code(_STATUS_CODES[exc.status.value])
    grpc_context.set_details(exc.message or exc.status.name)


def _unary_method(method: Any):
    response_cls = message_factory.GetMessageClass(method.output_type)

    async def call(self, request, grpc_context):
        stream = _Stream(request if method.client_streaming else _single(request))
        try:
            await self._handlers[method.name](stream)
        except GRPCError as exc:
            # Serve sends the status set on the context with the reply.
            _set_status(grpc_context, exc)
            return response_cls()
        return response_cls() if stream.responses.empty() else stream.responses.get_nowait()

    return call


def _streaming_method(method: Any):
    async def call(self, request, grpc_context):
        stream = _Stream(request if method.client_streaming else _single(request))

        async def run() -> None:
            try:
                await self._handlers[method.name](stream)
            finally:
                stream.responses.put_nowait(_END)

        task = asyncio.ensure_future(run())
        try:
            while (message := await stream.responses.get()) is not _END:
                yield message
            await task
        except GRPCError as exc:
            # A generator's status only travels with an exception.
            _set_status(grpc_context, exc)
            raise
        finally:
            task.cancel()

    return call


class ModalFacade:
    def __init__(self) -> None:
        # Command-router URL for clients that reach Serve directly; clients of
        # the Anyscale service go through modal_relay.py, which answers
        # TaskGetCommandRouterAccess itself.
        control, router = build_servicers(
            advertise_url=os.environ.get("SANDBOX_ROUTER_URL", "")
        )
        self._handlers = {
            **{m.name: getattr(control, m.name) for m in _CONTROL.methods},
            **{m.name: getattr(router, m.name) for m in _ROUTER.methods},
        }


for _method in [*_CONTROL.methods, *_ROUTER.methods]:
    _make = _streaming_method if _method.server_streaming else _unary_method
    setattr(ModalFacade, _method.name, _make(_method))


app = serve.deployment(
    ModalFacade,
    num_replicas=1,
    # Long polls (SandboxWait, TaskExecWait, stdio reads) hold a slot each.
    max_ongoing_requests=10_000,
    ray_actor_options={"num_cpus": 0, "resources": {"sandbox_facade": 1}},
).bind()
