"""HTTP client for UI Bridge API."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, NamedTuple
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_PORT = 9876
DEFAULT_TIMEOUT = 30.0
ELEMENT_DISCOVERY_TIMEOUT = 60.0
# Seconds an execute-with-diff HTTP call is given beyond its recording window.
EXECUTE_WITH_DIFF_TIMEOUT_SLACK = 10.0


# The default runner host. The runner binds the IPv4 loopback only, so this is
# the one address that reaches it from the same OS. ``localhost`` is not used
# because it resolves to ``::1`` first on Windows and pays a doomed IPv6
# connect. The first /etc/resolv.conf nameserver (the old default) is a DNS
# resolver on native Linux and macOS, and under NAT-mode WSL2 it is typically
# WSL's DNS-tunnelling stub or the Windows host's vEthernet IP, neither of which
# the runner binds; see the README for reaching a Windows runner from WSL.
LOOPBACK_HOST = "127.0.0.1"


def _port_from_environment() -> int:
    """Read QONTINUI_RUNNER_PORT, else the default; an unset or blank value is the default.

    A value that is not a TCP port raises a ``ValueError`` naming the variable,
    rather than ``int()``'s bare "invalid literal" or a URL that cannot connect.
    Only ASCII digits are accepted: ``int()`` alone would also take ``+9876``,
    ``9_876`` and non-ASCII digits.
    """
    raw = os.environ.get("QONTINUI_RUNNER_PORT", "").strip()
    if not raw:
        return DEFAULT_RUNNER_PORT
    if not (raw.isascii() and raw.isdigit() and 1 <= int(raw) <= 65535):
        raise ValueError(
            f"QONTINUI_RUNNER_PORT={raw!r} is not a TCP port (1-65535); "
            f"unset it to use the default {DEFAULT_RUNNER_PORT}"
        )
    return int(raw)


# Observation ``UnknownCode`` values (wire contract, snake_case) that this client
# assigns itself when the failure happened before any body could carry a code.
CODE_APP_UNREACHABLE = "app_unreachable"
CODE_PRODUCER_FAILED = "producer_failed"

# The runner's capture route and the code of its over-the-inline-cap 413
# (``vision_capture_access.rs`` ``INLINE_TOO_LARGE``; ``hint`` carries ``url``).
VISION_CAPTURE_ROUTE = "/ui-bridge/vision/capture"
CODE_INLINE_TOO_LARGE = "INLINE_TOO_LARGE"
# Minted by the screenshot tools (not the runner) when a capture of a
# WebSocket-transport app failed on a runner build with no WebSocket frame
# source (plan 2026-11-21-vision-capture-cannot-frame-a-websocket-transport-app).
CODE_FRAME_SOURCE_MISSING = "FRAME_SOURCE_MISSING"
# The runner's cache-read route serves raw bytes, not a JSON envelope. A path
# is followed only when it is exactly what the runner's handler accepts
# (`vision_cache_get_handler`: 64 hex chars), so a hint cannot steer the read.
VISION_CACHE_PREFIX = "/ui-bridge/vision/cache/"
_VISION_CACHE_PATH = re.compile(r"/ui-bridge/vision/cache/[0-9a-fA-F]{64}")
# The most a cached capture read may be. The runner caps INLINE replies at
# 5 MiB, and each INLINE_TOO_LARGE answer reports the capture's encoded size
# in ``hint.bytes``; the author of this bound measured none. A raw 4K RGBA
# buffer is ~33 MB before PNG compression, so the bound is a ceiling, not a
# typical size. Beyond it the read is refused (by ``hint.bytes`` before any
# request, else by content-length or the streamed count), never buffered.
MAX_CACHED_CAPTURE_BYTES = 64 * 1024 * 1024
# The most of a non-200 cache-read error body that is read to report it.
MAX_CACHE_ERROR_BODY_BYTES = 64 * 1024

# The closed Observation ``UnknownCode`` set, in the canonical wire-contract
# order: the twin of the ui-bridge SDK's
# ``packages/ui-bridge/src/observation/observation.ts`` ``UNKNOWN_CODES``, and
# of qontinui-schemas ``rust-vision-core/src/observation.rs``
# ``UnknownCode::ALL`` (16 members in both; the last seven landed in schemas
# with qontinui-schemas#264). A code outside it still renders as
# UNKNOWN with the code it carried (``timeline.render_unknown``), flagged as
# one this server does not know, never as an error.
UNKNOWN_CODES: tuple[str, ...] = (
    "app_unreachable",
    "capability_unavailable_in_build",
    "producer_failed",
    "producer_not_run",
    "input_missing",
    "below_confidence_floor",
    "model_reply_unparseable",
    "needs_multi_frame_input",
    "stale_input",
    "frontier_open",
    "ledger_schema_absent",
    "ledger_write_failing",
    "no_observations",
    "to_node_unobserved",
    "performance_observer_unsupported",
    "no_event_before_timeout",
)


class AssertState(NamedTuple):
    """One element state ``sdk_ai_assert`` can assert (see ``AI_ASSERT_STATES``)."""

    claim: str
    """How a result states the assertion after the target, e.g. ``is visible``."""
    sentence: str
    """The natural-language form of the assertion; ``{target}`` is substituted."""


# The states sdk_ai_assert offers: every ui-bridge ``AssertionType`` that needs no
# expected value (``ai/types.ts``). A request carries two forms of one assertion.
#
# - ``target`` + ``type``: the SDK's parser passes these through verbatim when
#   both are present (``ai/nl-assertion-parser.ts`` parseNLAssertion, and the
#   server's own guard in ``server/handlers.ts``), so a connected SDK app checks
#   exactly the element and state that were asked.
# - ``assertion``: the only field the runner's own UI reads, which is what
#   answers when no SDK app is connected (runner ``useAISearchEvents.ts``
#   ``ai_assert``). Each sentence below was run through that parser on
#   2026-10-13 and parses back to its own state for a plain target. "{T} does
#   not exist" is not used: it parses with the whole sentence as the target,
#   which no element matches, so notExists would pass vacuously. The parser
#   still lowercases the target, drops words such as "a"/"the"/"page", and reads
#   a target containing "no"/"not" or a state word as part of the assertion, so
#   on the runner's own UI such a target can be checked as a different state.
AI_ASSERT_STATES: dict[str, AssertState] = {
    "visible": AssertState("is visible", "{target} is visible"),
    "hidden": AssertState("is hidden", "{target} is hidden"),
    "enabled": AssertState("is enabled", "{target} is enabled"),
    "disabled": AssertState("is disabled", "{target} is disabled"),
    "focused": AssertState("is focused", "{target} is focused"),
    "checked": AssertState("is checked", "{target} is checked"),
    "unchecked": AssertState("is unchecked", "{target} is unchecked"),
    "exists": AssertState("exists", "{target} exists"),
    "notExists": AssertState("does not exist", "there is no {target} on the page"),
}


def _network_wait_body(
    url_pattern: str | None, method: str | None, timeout: int
) -> dict[str, Any]:
    """The body of a wait-for-network-request call, in every reader's field names.

    The SDK server reads ``urlPattern`` (``network/types.ts``
    WaitForRequestOptions); the browser relay reads ``url`` (``react/
    commandHandlers.ts`` waitForNetworkRequest). Both match it as a substring,
    so the pattern goes under both names: the snake_case ``url_pattern`` this
    client used to send is read by neither, so every wait matched any request.
    ``timeout`` is always sent, because the two readers' defaults differ (30 s
    and 10 s) and the tool documents 30 s. The SDK server also refuses an empty
    body (its route is ``bodyRequired``).
    """
    body: dict[str, Any] = {"timeout": timeout}
    if url_pattern:
        body["urlPattern"] = url_pattern
        body["url"] = url_pattern
    if method:
        body["method"] = method
    return body


def _wait_for_change_body(
    predicate: dict[str, Any], options: dict[str, Any] | None
) -> tuple[dict[str, Any], float]:
    """The body of a wait-for-change call, and an HTTP timeout that outlasts it.

    A wait longer than the default HTTP timeout would otherwise be cut off by
    this client and reported as a transport failure while the app was still
    waiting.
    """
    body: dict[str, Any] = {"predicate": predicate}
    if options:
        body["options"] = options
    wait_ms = (options or {}).get("timeout")
    http_timeout = DEFAULT_TIMEOUT
    if isinstance(wait_ms, int | float) and not isinstance(wait_ms, bool):
        http_timeout = max(DEFAULT_TIMEOUT, wait_ms / 1000 + 10.0)
    return body, http_timeout


def _execute_with_diff_timeout(request: dict[str, Any]) -> float:
    """An HTTP timeout that outlasts an execute-with-diff's own recording.

    ``settleTimeout`` is spent twice: the browser relay settles the DOM BEFORE
    the action (``react/commandHandlers.ts`` relayExecuteWithDiff), and
    executeWithDiff settles again after it. With ``expectStates`` the SDK then
    records until every expected state has entered or ``withinMs`` has
    elapsed, even past settle. A window longer than the default HTTP timeout
    would otherwise be cut off here and reported as a transport failure while
    the app was still recording.
    """

    def _ms(key: str) -> float:
        value = request.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        return 0.0

    wait_ms = 2 * _ms("settleTimeout") + _ms("withinMs")
    return max(DEFAULT_TIMEOUT, wait_ms / 1000 + EXECUTE_WITH_DIFF_TIMEOUT_SLACK)


@dataclass
class UIBridgeResponse:
    """Response from the UI Bridge API.

    ``code`` is the machine-readable cause of a failure (or of an ``unknown``
    Observation carried in a successful body): the body's own diagnostic code
    when the producer sent one (``UB-NET-ERROR``, ``RUNNER_REQUIRED``, an
    Observation ``unknown.code`` …), else a code this client assigned from the
    transport failure (``app_unreachable`` / ``producer_failed``). ``status`` is
    the HTTP status, ``None`` when no HTTP response was received.
    """

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    code: str | None = None
    status: int | None = None
    # A failure body's own ``hint`` (runner ``ApiResponse.hint``), unrendered,
    # so a caller reads a structured field rather than parsing ``error``. Set
    # on every failure that carried a body (non-2xx, and a 2xx whose
    # ``success`` is false); ``None`` when the body had none.
    hint: Any = None
    # Set by ``vision_capture`` on a failure that came AFTER the runner took
    # the frame: an INLINE_TOO_LARGE answer (the frame was taken and cached),
    # whatever its cache follow-up then did, or a 200 whose frame could not
    # be read. Such a failure is not a frame-source failure.
    capture_taken: bool = False

    def describe_error(self) -> str:
        """Render the failure for a tool result: typed code and status first.

        The MCP consumer must be able to branch on the code without parsing the
        English, so the code is always emitted as a ``[code=…]`` prefix when set.
        """
        tags: list[str] = []
        if self.code is not None:
            tags.append(f"code={self.code}")
        if self.status is not None:
            tags.append(f"status={self.status}")
        message = self.error if self.error is not None else "no error message"
        if not tags:
            return message
        return f"[{' '.join(tags)}] {message}"


@dataclass(frozen=True)
class Unknown:
    """A read whose answer is UNKNOWN: it failed, or was not interpretable.

    Distinct from ``None``, which a reader returns for a field or row that was
    read and is absent. ``reason`` names what failed.
    """

    reason: str


@dataclass(frozen=True)
class RunnerFrameSources:
    """The runner's ``/health`` frame-source advertisement, as read.

    ``sources`` is ``None`` when ``visionFrameSources`` is absent (a runner
    build that predates it); ``build_id`` is ``None`` when ``buildId`` is.
    """

    sources: list[str] | None
    build_id: str | None


def _str_field(obj: Any, key: str) -> str | None:
    """Return ``obj[key]`` when ``obj`` is a dict and the value is a non-empty str."""
    if isinstance(obj, dict):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _envelope_unknown(obj: Any) -> dict[str, Any] | None:
    """The ``unknown`` object of an Observation envelope in ``unknown`` state."""
    if isinstance(obj, dict) and obj.get("status") == "unknown":
        unknown = obj.get("unknown")
        if isinstance(unknown, dict):
            return unknown
    return None


def _envelope_unknown_code(obj: Any) -> str | None:
    """Return ``unknown.code`` when ``obj`` is an Observation envelope in ``unknown``."""
    return _str_field(_envelope_unknown(obj), "code")


def extract_code(body: Any) -> str | None:
    """Extract the producer's own machine-readable code from a response body.

    Handles every shape a UI Bridge producer answers with:

    - SDK ``APIResponse`` — top-level ``code``;
    - legacy / recovery bodies — top-level ``error_code``;
    - SDK action results (``NLActionResponse``) — top-level ``errorCode``;
    - runner ``ApiResponse`` — ``error_detail.code``, then ``refusal.code``;
    - a bare Observation envelope — ``unknown.code`` when ``status == "unknown"``;
    - an ``ApiResponse`` wrapping an envelope — ``data.unknown.code``.

    Explicit error codes win over an envelope's code. Returns ``None`` when the
    body carries no code at all.
    """
    if not isinstance(body, dict):
        return None
    return (
        _str_field(body, "code")
        or _str_field(body, "error_code")
        or _str_field(body, "errorCode")
        or _str_field(body.get("error_detail"), "code")
        or _str_field(body.get("refusal"), "code")
        or _envelope_unknown_code(body)
        or _envelope_unknown_code(body.get("data"))
    )


RAW_BODY_LIMIT = 500
FIELD_LIMIT = 300


def _compact(value: Any, limit: int = FIELD_LIMIT) -> str:
    """One-line rendering of a JSON value, truncated to ``limit`` characters."""
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)
    )
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def render_next_action(value: Any) -> str | None:
    """One-line rendering of a recovery value, in either runner wire shape.

    Runners are deployed independently of this client, so one MCP session can
    talk to builds on either side of the change that replaced the runner's
    ``RecoveryHint`` with the shared refusal vocabulary's ``NextAction``
    (qontinui-runner#3131). Both shapes are read:

    - ``NextAction`` — ``{"kind": "retry_later", "target"?: str,
      "retry_after_s"?: int}``, rendered ``retry_later target=<t>
      retry_after_s=<n>`` with the absent parts left out;
    - the older ``RecoveryHint`` serde spelling — a bare string
      (``"RESNAPSHOT"``) or a one-key object (``{"RETRY_AFTER_MS": 1000}``) —
      rendered verbatim, unmapped, so the reader sees what the runner said.

    Returns ``None`` for an absent value. Any other shape is compacted as JSON
    rather than dropped.
    """
    if value is None:
        return None
    if isinstance(value, dict) and isinstance(value.get("kind"), str):
        parts = [value["kind"]]
        if value.get("target") is not None:
            parts.append(f"target={_compact(value['target'])}")
        if value.get("retry_after_s") is not None:
            parts.append(f"retry_after_s={_compact(value['retry_after_s'])}")
        return _compact(" ".join(parts))
    if isinstance(value, dict) and len(value) == 1:
        ((name, arg),) = value.items()
        return _compact(f"{name} {_compact(arg)}")
    return _compact(value)


def render_refusal(refusal: Any) -> list[str]:
    """The lines a runner's typed ``refusal`` contributes to a failure render.

    ``refusal`` is the qontinui-schemas ``Refusal``: ``{code, discriminator?,
    next_action, glossary_terms, detail?, observed_at, source}``. Its sentence
    is the body's own ``error`` (rendered first by the caller), so this adds
    the machine half: the code and its discriminator, what to do next, the raw
    detail, and the glossary terms. A body without one (every runner predating
    qontinui-runner#3131) contributes nothing.
    """
    if not isinstance(refusal, dict):
        return []
    head = _str_field(refusal, "code") or "(no code)"
    discriminator = _str_field(refusal, "discriminator")
    if discriminator is not None:
        head = f"{head} ({discriminator})"
    lines = [f"refusal: {_compact(head)}"]
    next_action = render_next_action(refusal.get("next_action"))
    if next_action is not None:
        lines.append(f"next: {next_action}")
    if refusal.get("detail") is not None:
        lines.append(f"refusal detail: {_compact(refusal['detail'])}")
    terms = refusal.get("glossary_terms")
    if isinstance(terms, list) and terms:
        lines.append(f"glossary: {_compact(', '.join(map(str, terms)))}")
    return lines


def render_body_error(body: Any, raw_text: str) -> str:
    """Render a failure body for a human AND keep every recovery field.

    The body's ``error`` comes first, then one compact line per recovery field
    the producer sent: the runner's typed ``refusal`` (``render_refusal``),
    runner ``hint`` / SDK ``suggestions`` (no longer sent by a runner carrying
    qontinui-runner#3131, so absent is normal) / ``error_detail.recovery`` in
    either wire shape (``render_next_action``) / ``error_detail.context``, and
    an Observation envelope's ``unknown.detail``. A body that is not a JSON object falls back
    to the raw response text, truncated.
    """
    if not isinstance(body, dict):
        raw = " ".join(raw_text.split())
        if not raw:
            return "(empty response body)"
        if len(raw) > RAW_BODY_LIMIT:
            raw = raw[:RAW_BODY_LIMIT] + f"\u2026 ({len(raw_text)} chars)"
        return raw

    detail = body.get("error_detail")
    detail = detail if isinstance(detail, dict) else {}
    unknown = _envelope_unknown(body) or _envelope_unknown(body.get("data")) or {}

    head = _str_field(body, "error") or _str_field(detail, "message")
    unknown_detail = unknown.get("detail")
    if head is None and unknown_detail is not None:
        # The envelope's detail IS the message; do not repeat it below.
        head = _compact(unknown_detail)
        unknown_detail = None
    lines = [head if head is not None else "(no error message in body)"]

    lines.extend(render_refusal(body.get("refusal")))
    if body.get("hint") is not None:
        lines.append(f"hint: {_compact(body['hint'])}")
    suggestions = body.get("suggestions")
    if isinstance(suggestions, list) and suggestions:
        lines.append(f"suggestions: {_compact('; '.join(map(str, suggestions)))}")
    recovery = render_next_action(detail.get("recovery"))
    if recovery is not None:
        lines.append(f"recovery: {recovery}")
    if detail.get("context") is not None:
        lines.append(f"context: {_compact(detail['context'])}")
    if unknown_detail is not None:
        lines.append(f"detail: {_compact(unknown_detail)}")
    elif _str_field(body, "detail") is not None:
        # A plain ``{error, detail}`` body (the runner's 400 refusals, FastAPI).
        lines.append(f"detail: {_compact(body['detail'])}")
    return "\n".join(lines)


def _inline_too_large_url(response: UIBridgeResponse) -> str | None:
    """The ``hint.url`` of an ``INLINE_TOO_LARGE`` answer, when it names one.

    Only a path the runner's cache route itself accepts —
    ``/ui-bridge/vision/cache/<64 hex>``, nothing before or after it — is
    returned; anything else (a ``../`` or ``?`` suffix, another route) is not.
    """
    hint = response.hint if isinstance(response.hint, dict) else {}
    url = hint.get("url")
    if isinstance(url, str) and _VISION_CACHE_PATH.fullmatch(url):
        return url
    return None


def sniff_image_format(data: bytes) -> str | None:
    """``png`` / ``jpeg`` / ``webp`` from the magic bytes, else ``None``."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _inline_capture(response: UIBridgeResponse) -> UIBridgeResponse:
    """A successful ``vision/capture`` answer narrowed to its inline image.

    An answer without the inline bytes or the pixel size is a failure, not an
    empty image: the caller asked for ``inline: true``.
    """
    data = response.data if isinstance(response.data, dict) else {}
    b64 = data.get("base64")
    width = data.get("width")
    height = data.get("height")
    missing = [
        name
        for name, ok in (
            ("base64", isinstance(b64, str) and bool(b64)),
            ("width", isinstance(width, int) and not isinstance(width, bool)),
            ("height", isinstance(height, int) and not isinstance(height, bool)),
        )
        if not ok
    ]
    if missing:
        shape = (
            "a multi-output {captures} answer"
            if "captures" in data
            else f"keys {sorted(data.keys())}"
        )
        return UIBridgeResponse(
            success=False,
            error=(
                "vision/capture answered without "
                f"{', '.join(missing)} (got {shape}); the image is UNKNOWN"
            ),
            code=CODE_PRODUCER_FAILED,
            status=response.status,
        )
    sha = data.get("sha256")
    fmt = data.get("format")
    return UIBridgeResponse(
        success=True,
        data={
            "base64": b64,
            "width": width,
            "height": height,
            "sha256": sha if isinstance(sha, str) else None,
            "format": fmt if isinstance(fmt, str) else None,
        },
        status=response.status,
    )


class UIBridgeClient:
    """HTTP client for the UI Bridge API.

    This client provides access to both:
    - Control mode: Runner's own Tauri webview UI (/ui-bridge/control/*)
    - SDK mode: External SDK-integrated apps via runner proxy (/ui-bridge/sdk/*)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """Initialize the client.

        An explicit argument wins, then the environment variable, then the default.

        Args:
            host: Runner host. Defaults to QONTINUI_RUNNER_HOST, else 127.0.0.1.
            port: Runner port. Defaults to QONTINUI_RUNNER_PORT, else 9876. Only
                the environment value is validated; an explicit port is used as given.
        """
        self.host = (
            host or os.environ.get("QONTINUI_RUNNER_HOST", "").strip() or LOOPBACK_HOST
        )
        self.port = port if port is not None else _port_from_environment()
        self.base_url = f"http://{self.host}:{self.port}"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        params: dict[str, str] | None = None,
        bare_list_key: str | None = None,
        *,
        unwrapped: bool = False,
    ) -> UIBridgeResponse:
        """Make an HTTP request to the UI Bridge API.

        Args:
            method: HTTP method (GET or POST).
            endpoint: API endpoint path.
            json_data: Optional JSON body for POST requests.
            timeout: Request timeout in seconds.
            params: Optional query parameters for GET requests.
            bare_list_key: For a route the runner serves UNWRAPPED (no
                ``success`` envelope): a 200 body that has no ``success`` key
                but carries a list under this key is the route's answer, and
                is returned whole as ``data``. Every other route keeps the
                rule that an absent ``success`` is a failure.
            unwrapped: The route answers its body directly, with no
                ``{success, data}`` wrapper (the journey routes answer an
                Observation envelope or a health object). A 2xx JSON object is
                then returned whole as ``data``; it is the caller's to read.
        """
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = await client.get(url, params=params, timeout=timeout)
            elif method == "POST":
                response = await client.post(url, json=json_data, timeout=timeout)
            elif method == "DELETE":
                response = await client.delete(url, timeout=timeout)
            else:
                return UIBridgeResponse(
                    success=False,
                    error=f"Unsupported method: {method} (client did not send a request)",
                    code=CODE_PRODUCER_FAILED,
                )

            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return UIBridgeResponse(
                    success=False,
                    error=(
                        f"Unexpected response body from {url}: expected a JSON "
                        f"object, got {type(data).__name__}"
                    ),
                    code=CODE_PRODUCER_FAILED,
                    status=response.status_code,
                )
            if unwrapped or (
                bare_list_key is not None
                and "success" not in data
                and isinstance(data.get(bare_list_key), list)
            ):
                return UIBridgeResponse(
                    success=True, data=data, status=response.status_code
                )
            # Absent `success` is a failure, never a success.
            success = data.get("success", False) is True
            code = extract_code(data)
            inner = data.get("data")
            if success and isinstance(inner, dict) and inner.get("success") is False:
                # A failure inside a success envelope: the request worked, the
                # operation did not. The SDK in-process handlers return
                # success(result) whatever the result says (aiExecute's
                # NLActionResponse, fillForm's FillResult), and runner routes
                # wrap their own UI's {success: false} in {success: true}.
                # ui-bridge's relay already treats an inner `success: false`
                # as the failure (relay-handlers.ts relayCommand); so does this.
                return UIBridgeResponse(
                    success=False,
                    data=inner,
                    error=render_body_error(inner, response.text),
                    code=extract_code(inner) or code or CODE_PRODUCER_FAILED,
                    status=response.status_code,
                    hint=inner.get("hint") or data.get("hint"),
                )
            if success:
                return UIBridgeResponse(
                    success=True,
                    data=data.get("data"),
                    error=data.get("error"),
                    code=code,
                    status=response.status_code,
                )
            return UIBridgeResponse(
                success=False,
                data=data.get("data"),
                error=render_body_error(data, response.text),
                code=code or CODE_PRODUCER_FAILED,
                status=response.status_code,
                hint=data.get("hint"),
            )
        except httpx.ConnectTimeout:
            # The connection itself never completed: the app was not reached.
            return UIBridgeResponse(
                success=False,
                error=f"Cannot connect to runner at {url}: connect timeout after {timeout}s",
                code=CODE_APP_UNREACHABLE,
            )
        except httpx.ConnectError as e:
            return UIBridgeResponse(
                success=False,
                error=f"Cannot connect to runner at {url}. Is qontinui-runner running? Error: {e}",
                code=CODE_APP_UNREACHABLE,
            )
        except httpx.HTTPStatusError as e:
            return self._http_status_failure(e.response)
        except httpx.TimeoutException:
            return UIBridgeResponse(
                success=False,
                error=f"Request timed out after {timeout}s (timeout)",
                code=CODE_PRODUCER_FAILED,
            )
        except Exception as e:
            return UIBridgeResponse(
                success=False,
                error=f"{type(e).__name__}: {e}",
                code=CODE_PRODUCER_FAILED,
            )

    @staticmethod
    def _http_status_failure(response: httpx.Response) -> UIBridgeResponse:
        """Build the failure for a non-2xx answer, keeping the body's own code."""
        status = response.status_code
        body: Any = None
        try:
            body = response.json()
        except ValueError:
            body = None
        code = extract_code(body) or CODE_PRODUCER_FAILED
        error = f"API error {status}: {render_body_error(body, response.text)}"
        hint = body.get("hint") if isinstance(body, dict) else None
        return UIBridgeResponse(
            success=False, error=error, code=code, status=status, hint=hint
        )

    # -------------------------------------------------------------------------
    # Health & Status
    # -------------------------------------------------------------------------

    async def health(self) -> UIBridgeResponse:
        """Check runner health."""
        return await self._request("GET", "/health")

    async def vision_frame_sources(
        self, timeout: float = DEFAULT_TIMEOUT
    ) -> RunnerFrameSources | Unknown:
        """The runner's ``/health`` ``visionFrameSources``, and its ``buildId``.

        Read from ``data`` first, then the top-level mirror the runner serves
        beside it (as it does for ``buildId``). ``sources`` is ``None`` when the
        field is absent: a runner build that predates the frame-source
        advertisement. A read that failed, or a field that is present but not
        a list of strings, is :class:`Unknown`, never ``None``.
        """
        resp = await self._request("GET", "/health", timeout=timeout, unwrapped=True)
        if not resp.success or not isinstance(resp.data, dict):
            return Unknown(f"GET /health failed: {resp.describe_error()}")
        body = resp.data
        inner = body.get("data")
        layers = [inner, body] if isinstance(inner, dict) else [body]
        sources: list[str] | None = None
        for layer in layers:
            if "visionFrameSources" not in layer:
                continue
            value = layer["visionFrameSources"]
            if not isinstance(value, list) or not all(
                isinstance(v, str) for v in value
            ):
                return Unknown(
                    "GET /health visionFrameSources is not a list of strings "
                    f"(got {_compact(value)})"
                )
            sources = value
            break
        build_id = next(
            (
                layer["buildId"]
                for layer in layers
                if isinstance(layer.get("buildId"), str) and layer["buildId"]
            ),
            None,
        )
        return RunnerFrameSources(sources=sources, build_id=build_id)

    async def registered_app(
        self, target: str, timeout: float = DEFAULT_TIMEOUT
    ) -> dict[str, Any] | None | Unknown:
        """``target``'s row in ``GET /ui-bridge/apps/registered``.

        ``None`` means the registry was read and holds no live row whose
        ``appId`` is ``target`` (a paired device or an adb serial, say). A read
        that failed, or an answer that is not a list, is :class:`Unknown`.
        """
        route = "/ui-bridge/apps/registered"
        resp = await self._request("GET", route, timeout=timeout, unwrapped=True)
        if not resp.success or not isinstance(resp.data, dict):
            return Unknown(f"GET {route} failed: {resp.describe_error()}")
        body = resp.data
        rows = body.get("data")
        if body.get("success") is not True or not isinstance(rows, list):
            return Unknown(
                f"GET {route} answered no app list "
                f"(success={_compact(body.get('success'))}, "
                f"data={_compact(rows)})"
            )
        for row in rows:
            if isinstance(row, dict) and row.get("appId") == target:
                return row
        return None

    # -------------------------------------------------------------------------
    # Journey queries (runner ``journey/routes.rs``)
    # -------------------------------------------------------------------------

    async def journey(
        self, app_id: str, route: str, params: dict[str, str] | None = None
    ) -> UIBridgeResponse:
        """``GET /apps/{app_id}/journey/{route}``; ``data`` is the body as served.

        Every route but ``health`` answers an Observation envelope; ``health``
        answers ``JourneyHealthResponse``. The runner refuses an unknown or
        unused query parameter with a 400, so only parameters the caller set
        are sent.
        """
        return await self._request(
            "GET",
            f"/apps/{quote(app_id, safe='')}/journey/{route}",
            params=params or None,
            unwrapped=True,
        )

    # -------------------------------------------------------------------------
    # Control Mode - Runner's Own UI (/ui-bridge/control/*)
    # -------------------------------------------------------------------------

    async def control_snapshot(self) -> UIBridgeResponse:
        """Get a full UI snapshot of the runner's webview.

        Returns all registered elements, components, and workflows
        with their current state (visibility, position, text content).
        """
        return await self._request("GET", "/ui-bridge/control/snapshot")

    async def control_clipboard_read(self) -> UIBridgeResponse:
        """Read the current system clipboard content."""
        return await self._request("GET", "/ui-bridge/control/clipboard")

    async def control_clipboard_write(
        self, text: str, html: str | None = None
    ) -> UIBridgeResponse:
        """Write text to the system clipboard."""
        data: dict[str, Any] = {"text": text}
        if html is not None:
            data["html"] = html
        return await self._request("POST", "/ui-bridge/control/clipboard", data)

    async def control_forms(self) -> UIBridgeResponse:
        """Get form state data from the runner's webview.

        Returns all detected forms with field values, validation errors,
        dirty state, required fields, and constraint information.
        """
        return await self._request("GET", "/ui-bridge/control/forms")

    async def control_fill_form(
        self,
        fields: dict[str, Any],
        trigger_validation: bool = True,
        clear_first: bool = True,
    ) -> UIBridgeResponse:
        """Fill multiple form fields atomically in the runner's webview.

        Args:
            fields: Map of element ID to value (string, boolean, or string[]).
            trigger_validation: Whether to trigger validation after filling.
            clear_first: Whether to clear existing values first.
        """
        return await self._request(
            "POST",
            "/ui-bridge/control/fill",
            {
                "fields": fields,
                "triggerValidation": trigger_validation,
                "clearFirst": clear_first,
            },
        )

    async def control_form_snapshot(self) -> UIBridgeResponse:
        """Capture a snapshot of all form state in the runner's webview.

        Returns a FormSnapshot with all forms and their field states.
        Use before and after an action to track form changes via control_form_diff.
        """
        return await self._request("POST", "/ui-bridge/control/forms/snapshot")

    async def control_form_diff(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> UIBridgeResponse:
        """Compare two form snapshots from the runner's webview.

        Args:
            before: The before snapshot from control_form_snapshot.
            after: The after snapshot from control_form_snapshot.
        """
        return await self._request(
            "POST",
            "/ui-bridge/control/forms/diff",
            {"before": before, "after": after},
        )

    async def control_discover(
        self, interactive_only: bool = False
    ) -> UIBridgeResponse:
        """Trigger element discovery in the runner's webview.

        Args:
            interactive_only: If True, only return interactive elements.
        """
        return await self._request(
            "POST",
            "/ui-bridge/control/discover",
            {"interactive_only": interactive_only},
        )

    async def discover_media(self) -> UIBridgeResponse:
        """Discover every element, hidden and non-interactive ones included, with
        the media rows stated, as the runner's own sweeps ask."""
        return await self._request(
            "POST",
            "/ui-bridge/control/discover",
            {"interactive_only": False, "includeHidden": True, "includeMedia": True},
        )

    async def vision_assert(
        self, snapshot: dict[str, Any], assertions: list[dict[str, Any]]
    ) -> UIBridgeResponse:
        """Evaluate vision-core assertions over a caller-supplied ElementSnapshot."""
        return await self._request(
            "POST",
            "/ui-bridge/vision/assert",
            {"snapshot": snapshot, "assertions": assertions},
        )

    async def vision_capture(
        self,
        target: str | None = None,
        region: dict[str, int] | None = None,
        element: str | None = None,
        *,
        contract: str | None = None,
        follow_oversize: bool = False,
    ) -> UIBridgeResponse:
        """Capture a frame through ``POST /ui-bridge/vision/capture``, inline.

        The runner's one capture route (its ``CaptureRequest``,
        ``src-tauri/src/mcp/ui_bridge/vision_routes.rs``). The SDK screenshot
        and control annotated-screenshot routes this replaced were deleted
        from the runner on 2026-05-14 (runner commit 940fd0f40).

        Args:
            target: Frame source: a registered device / app id or an adb serial.
                ``None`` captures the runner's own window.
            region: Pixel-space crop ``{x, y, w, h}`` applied after capture.
            element: Element id; the runner crops to its rect.
            contract: Output contract: ``"claude"`` (the runner default, lossy
                and size-bounded), ``"webp"``, or ``"png_strict"`` (lossless,
                not resized, so it can pass the inline cap).
            follow_oversize: On ``INLINE_TOO_LARGE``, read the capture the
                runner already took and cached from the answer's ``hint.url``
                (``GET /ui-bridge/vision/cache/<sha>``, raw bytes) instead of
                failing. Without it the failure is returned, naming the url.

        Every capture is taken fresh (``force``): the runner's cache is keyed
        on its UI-mutation counter, which a change it did not drive never bumps.

        On success ``data`` is ``{base64, width, height, sha256, format}``.
        ``sha256`` is the runner's hash of the cache KEY, not of the image
        bytes. A capture over the runner's inline cap answers 413
        ``INLINE_TOO_LARGE``; that failure is returned with the ``url`` the
        bytes are served from, and is not retried.
        """
        body: dict[str, Any] = {"inline": True, "force": True}
        if target is not None:
            body["target"] = target
        if region is not None:
            body["region"] = region
        if element is not None:
            body["element"] = element
        if contract is not None:
            body["contract"] = contract
        response = await self._request("POST", VISION_CAPTURE_ROUTE, body)
        if not response.success and response.code != CODE_INLINE_TOO_LARGE:
            return response
        if response.success:
            result = _inline_capture(response)
        else:
            result = await self._oversize_capture(response, follow_oversize)
        if not result.success:
            result.capture_taken = True
        return result

    async def _oversize_capture(
        self, response: UIBridgeResponse, follow_oversize: bool
    ) -> UIBridgeResponse:
        """The answer to a capture the runner took but would not return inline.

        ``response`` is the runner's 413 ``INLINE_TOO_LARGE``. With
        ``follow_oversize`` the cached capture is read from its ``hint.url``;
        otherwise the failure is returned naming that url.
        """
        url = _inline_too_large_url(response)
        hint = response.hint if isinstance(response.hint, dict) else {}
        size = hint.get("bytes")
        if follow_oversize and url is not None:
            if (
                isinstance(size, int)
                and not isinstance(size, bool)
                and size > MAX_CACHED_CAPTURE_BYTES
            ):
                return UIBridgeResponse(
                    success=False,
                    error=(
                        f"capture is {size} bytes (the runner's hint.bytes), "
                        "over this client's MAX_CACHED_CAPTURE_BYTES="
                        f"{MAX_CACHED_CAPTURE_BYTES}; it was not read from "
                        f"GET {self.base_url}{url}"
                    ),
                    code=CODE_INLINE_TOO_LARGE,
                    status=response.status,
                    hint=response.hint,
                )
            return await self._read_cached_capture(url)
        where = (
            f"Fetch it with GET {self.base_url}{url}."
            if url is not None
            else (
                "The answer carried no followable cache url in its hint "
                "(only /ui-bridge/vision/cache/<64 hex> is followed)."
            )
        )
        response.error = (
            "capture is too large to return inline; it was taken and "
            f"cached, and is not retried here. {where} "
            f"Runner said: {response.error}"
        )
        return response

    async def _read_cached_capture(self, url: str) -> UIBridgeResponse:
        """A cached capture read from ``GET <url>``, as an inline frame.

        The runner's cache route answers the raw image bytes with their
        content type (``vision_cache_get_handler``). The pixel size is read
        from the image itself; the format from its magic bytes.
        """
        from PIL import Image

        full = f"{self.base_url}{url}"
        client = await self._get_client()
        too_big = (
            f"capture over the inline cap; {full} serves more than "
            f"MAX_CACHED_CAPTURE_BYTES={MAX_CACHED_CAPTURE_BYTES} bytes, so it "
            "was not read"
        )
        try:
            async with client.stream("GET", full, timeout=DEFAULT_TIMEOUT) as response:
                if response.status_code != 200:
                    head = b""
                    cut = False
                    async for chunk in response.aiter_bytes():
                        head += chunk
                        if len(head) > MAX_CACHE_ERROR_BODY_BYTES:
                            head = head[:MAX_CACHE_ERROR_BODY_BYTES]
                            cut = True
                            break
                    text = head.decode("utf-8", "replace")
                    cut_note = (
                        f" (truncated at {MAX_CACHE_ERROR_BODY_BYTES} bytes)"
                        if cut
                        else ""
                    )
                    try:
                        error_body: Any = json.loads(text)
                    except ValueError:
                        error_body = None
                    return UIBridgeResponse(
                        success=False,
                        error=(
                            f"capture over the inline cap; reading it from {full} "
                            f"failed: API error {response.status_code}: "
                            f"{render_body_error(error_body, text)}{cut_note}"
                        ),
                        code=extract_code(error_body) or CODE_PRODUCER_FAILED,
                        status=response.status_code,
                        hint=error_body.get("hint")
                        if isinstance(error_body, dict)
                        else None,
                    )
                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit():
                    if int(declared) > MAX_CACHED_CAPTURE_BYTES:
                        return UIBridgeResponse(
                            success=False,
                            error=f"{too_big} (declared content-length {declared})",
                            code=CODE_PRODUCER_FAILED,
                            status=response.status_code,
                        )
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_CACHED_CAPTURE_BYTES:
                        return UIBridgeResponse(
                            success=False,
                            error=f"{too_big} (counted while streaming)",
                            code=CODE_PRODUCER_FAILED,
                            status=response.status_code,
                        )
                    chunks.append(chunk)
        except httpx.HTTPError as e:
            return UIBridgeResponse(
                success=False,
                error=f"capture over the inline cap; reading it from {full} failed: {e}",
                code=CODE_PRODUCER_FAILED,
            )
        data = b"".join(chunks)
        fmt = sniff_image_format(data)

        def decode_size() -> tuple[int, int]:
            with Image.open(io.BytesIO(data)) as img:
                # Decode fully: the header alone gives a size for truncated bytes.
                img.load()
                size: tuple[int, int] = img.size
                return size

        try:
            # A full decode of a large capture is CPU-bound; keep it off the loop.
            width, height = await asyncio.to_thread(decode_size)
        except Exception as e:  # any decode failure leaves the frame UNKNOWN
            return UIBridgeResponse(
                success=False,
                error=(
                    f"capture read from {full} is not a decodable image "
                    f"({type(e).__name__}: {e}); the frame is UNKNOWN"
                ),
                code=CODE_PRODUCER_FAILED,
                status=response.status_code,
            )
        return UIBridgeResponse(
            success=True,
            data={
                "base64": base64.b64encode(data).decode(),
                "width": width,
                "height": height,
                "sha256": url[len(VISION_CACHE_PREFIX) :],
                "format": fmt,
                "fetched_from": url,
            },
            status=response.status_code,
        )

    async def control_list_elements(self) -> UIBridgeResponse:
        """List all registered UI elements in the runner's webview."""
        return await self._request("GET", "/ui-bridge/control/elements")

    async def control_get_element(self, element_id: str) -> UIBridgeResponse:
        """Get details for a specific element.

        Args:
            element_id: The element's registered ID.

        Returns:
            Element details including bounds, state, actions, etc.
        """
        return await self._request("GET", f"/ui-bridge/control/element/{element_id}")

    async def control_click(self, element_id: str) -> UIBridgeResponse:
        """Click an element in the runner's webview.

        Args:
            element_id: The element's registered ID.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "click"},
        )

    async def control_type(self, element_id: str, text: str) -> UIBridgeResponse:
        """Type text into an element in the runner's webview.

        Args:
            element_id: The element's registered ID.
            text: Text to type.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "type", "params": {"text": text}},
        )

    async def control_focus(self, element_id: str) -> UIBridgeResponse:
        """Focus an element in the runner's webview.

        Args:
            element_id: The element's registered ID.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "focus"},
        )

    async def control_hover(self, element_id: str) -> UIBridgeResponse:
        """Hover over an element in the runner's webview.

        Args:
            element_id: The element's registered ID.
        """
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            {"action": "hover"},
        )

    async def control_action(
        self,
        element_id: str,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> UIBridgeResponse:
        """Execute any action on an element in the runner's webview.

        Generic method for all UI Bridge actions.

        Args:
            element_id: The element's registered ID.
            action: Action name (click, type, focus, blur, hover, etc.).
            params: Optional action parameters.
        """
        body: dict[str, Any] = {"action": action}
        if params:
            body["params"] = params
        return await self._request(
            "POST",
            f"/ui-bridge/control/element/{element_id}/action",
            body,
        )

    # -------------------------------------------------------------------------
    # Control Mode - Undo/Redo Awareness
    # -------------------------------------------------------------------------

    async def control_undo_state(self) -> UIBridgeResponse:
        """Get undo/redo availability and state from the runner's webview.

        Returns whether undo/redo is available, what it would reverse,
        stack depth, and detection sources.
        """
        return await self._request("GET", "/ui-bridge/control/undo-state")

    async def control_undo(self) -> UIBridgeResponse:
        """Execute undo in the runner's webview.

        Uses the developer-declared handler if available, otherwise
        dispatches Ctrl+Z (Cmd+Z on Mac) keyboard event.
        """
        return await self._request("POST", "/ui-bridge/control/undo")

    async def control_redo(self) -> UIBridgeResponse:
        """Execute redo in the runner's webview.

        Uses the developer-declared handler if available, otherwise
        dispatches Ctrl+Shift+Z (Cmd+Shift+Z on Mac) keyboard event.
        """
        return await self._request("POST", "/ui-bridge/control/redo")

    # -------------------------------------------------------------------------
    # SDK Mode - External SDK-Integrated Apps (/ui-bridge/sdk/*)
    # -------------------------------------------------------------------------

    async def sdk_connect(self, url: str) -> UIBridgeResponse:
        """Connect to an SDK-integrated app.

        Args:
            url: The app URL (e.g., 'http://localhost:3001').
        """
        return await self._request("POST", "/ui-bridge/sdk/connect", {"url": url})

    async def sdk_disconnect(self) -> UIBridgeResponse:
        """Disconnect from the SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/disconnect")

    async def sdk_status(self) -> UIBridgeResponse:
        """Check SDK app connection status."""
        return await self._request("GET", "/ui-bridge/sdk/status")

    async def sdk_elements(
        self,
        content_only: bool = False,
        content_types: list[str] | None = None,
    ) -> UIBridgeResponse:
        """List all registered UI elements in the connected SDK app.

        Args:
            content_only: If True, filter to only content (non-interactive) elements.
            content_types: Filter to elements matching these content types
                (e.g., ['heading', 'paragraph', 'badge']).

        The runner (qontinui-runner ``handle_elements``) reads these query
        parameters and filters before answering; the tool filters again on
        what comes back.
        """
        params: dict[str, str] | None = None
        if content_only or content_types:
            params = {}
            if content_only:
                params["contentOnly"] = "true"
            if content_types:
                params["contentTypes"] = ",".join(content_types)
        return await self._request("GET", "/ui-bridge/sdk/elements", params=params)

    async def sdk_element(self, element_id: str) -> UIBridgeResponse:
        """Get details for a specific element by its registered ID.

        Args:
            element_id: The element's registered ID.
        """
        return await self._request("GET", f"/ui-bridge/sdk/element/{element_id}")

    async def sdk_element_action(
        self, element_id: str, action: str, params: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Execute an action on an element.

        Args:
            element_id: The element's registered ID.
            action: Action to perform (click, type, focus, hover).
            params: Optional params (e.g., {"text": "hello"} for type).
        """
        body: dict[str, Any] = {"action": action}
        if params:
            body["params"] = params
        return await self._request(
            "POST", f"/ui-bridge/sdk/element/{element_id}/action", body
        )

    async def sdk_snapshot(
        self,
        include_content: bool = True,
    ) -> UIBridgeResponse:
        """Get a complete UI snapshot with all elements and their state.

        Args:
            include_content: Include content (non-interactive) elements in the snapshot.
                Defaults to True. Set to False to only get interactive elements.

        Note: The includeContent query parameter requires SDK handler support.
        The Rust relay currently does not forward query params for GET requests,
        so this parameter will only take effect once the relay is updated or
        the snapshot endpoint is changed to accept POST body params.
        """
        params: dict[str, str] | None = None
        if not include_content:
            params = {"includeContent": "false"}
        return await self._request("GET", "/ui-bridge/sdk/snapshot", params=params)

    async def sdk_clipboard_read(self) -> UIBridgeResponse:
        """Read the current system clipboard content via SDK endpoint."""
        return await self._request("GET", "/ui-bridge/sdk/clipboard")

    async def sdk_clipboard_write(
        self, text: str, html: str | None = None
    ) -> UIBridgeResponse:
        """Write text to the system clipboard via SDK endpoint."""
        data: dict[str, Any] = {"text": text}
        if html is not None:
            data["html"] = html
        return await self._request("POST", "/ui-bridge/sdk/clipboard", data)

    async def sdk_undo_state(self) -> UIBridgeResponse:
        """Get undo/redo availability and state from the connected SDK app."""
        return await self._request("GET", "/ui-bridge/sdk/undo-state")

    async def sdk_undo(self) -> UIBridgeResponse:
        """Execute undo in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/undo")

    async def sdk_redo(self) -> UIBridgeResponse:
        """Execute redo in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/redo")

    async def sdk_forms(self) -> UIBridgeResponse:
        """Get form state data from the connected SDK app.

        Returns all detected forms with field values, validation errors,
        dirty state, required fields, and constraint information.
        """
        return await self._request("GET", "/ui-bridge/sdk/forms")

    async def sdk_fill_form(
        self,
        fields: dict[str, Any],
        trigger_validation: bool = True,
        clear_first: bool = True,
    ) -> UIBridgeResponse:
        """Fill multiple form fields atomically in the connected SDK app.

        Args:
            fields: Map of element ID to value (string, boolean, or string[]).
            trigger_validation: Whether to trigger validation after filling.
            clear_first: Whether to clear existing values first.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/fill",
            {
                "fields": fields,
                "triggerValidation": trigger_validation,
                "clearFirst": clear_first,
            },
        )

    async def sdk_form_snapshot(self) -> UIBridgeResponse:
        """Capture a snapshot of all form state in the connected SDK app.

        Returns a FormSnapshot with all forms and their field states.
        Use before and after an action to track form changes via sdk_form_diff.
        """
        return await self._request("POST", "/ui-bridge/sdk/forms/snapshot")

    async def sdk_form_diff(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> UIBridgeResponse:
        """Compare two form snapshots from the connected SDK app.

        Args:
            before: The before snapshot from sdk_form_snapshot.
            after: The after snapshot from sdk_form_snapshot.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/forms/diff",
            {"before": before, "after": after},
        )

    async def sdk_discover(
        self,
        interactive_only: bool = False,
        include_content: bool = True,
        content_roles: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Trigger element discovery in the SDK app.

        Args:
            interactive_only: If True, only return interactive elements.
            include_content: Include content (non-interactive) elements in discovery.
                Defaults to True. Ignored if interactive_only is True.
            content_roles: Filter content elements to these roles
                (e.g., ['heading', 'body-text', 'metric']).
                Only applies when content elements are included.
        """
        body: dict[str, Any] = {"interactive_only": interactive_only}
        if not include_content:
            body["includeContent"] = False
        if content_roles:
            body["contentRoles"] = content_roles
        return await self._request(
            "POST",
            "/ui-bridge/sdk/discover",
            body,
        )

    async def sdk_ai_search(
        self,
        text: str,
        content_role: str | None = None,
        content_types: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Search for elements by natural language description.

        Args:
            text: Natural language description of the element.
            content_role: Filter results to elements with this content role
                (e.g., 'heading', 'body-text', 'metric', 'badge').
            content_types: Filter results to elements with these content types
                (e.g., ['heading', 'paragraph', 'metric-value']).
        """
        body: dict[str, Any] = {"text": text}
        if content_role:
            body["contentRole"] = content_role
        if content_types:
            body["contentTypes"] = content_types
        # The runner unwraps this route: both its SDK path and its IPC fallback
        # (qontinui-runner sdk_client.rs handle_ai_search) answer the bare
        # search response, {results: [...], ...}, with no success envelope.
        return await self._request(
            "POST", "/ui-bridge/sdk/ai/search", body, bare_list_key="results"
        )

    async def sdk_ai_find(
        self,
        query: str,
        context: str | None = None,
        confidence_threshold: float | None = None,
    ) -> UIBridgeResponse:
        """Find an element by natural language description with spatial/relational context.

        Supports queries like "close button near Terminal 1 tab" or
        "email input in the login form". More capable than sdk_ai_search
        because it handles spatial references, container scoping, ordinals,
        and state filters.

        Args:
            query: Natural language element description.
            context: Optional context hint (e.g., 'in the dialog').
            confidence_threshold: Minimum confidence threshold (0-1).
        """
        body: dict[str, Any] = {"query": query}
        if context:
            body["context"] = {"sectionHint": context}
        if confidence_threshold is not None:
            body["confidenceThreshold"] = confidence_threshold
        return await self._request("POST", "/ui-bridge/sdk/ai/find", body)

    async def sdk_ai_execute(self, instruction: str) -> UIBridgeResponse:
        """Execute an action by natural language instruction.

        Args:
            instruction: Natural language instruction (e.g., 'click the Submit button').
        """
        return await self._request(
            "POST", "/ui-bridge/sdk/ai/execute", {"instruction": instruction}
        )

    async def sdk_ai_assert(self, text: str, state: str = "exists") -> UIBridgeResponse:
        """Assert that the element ``text`` names is in ``state``.

        The body carries the assertion both structured and as a sentence; see
        ``AI_ASSERT_STATES`` for why, and for which reader takes which form.

        Args:
            text: Element ID or description to find.
            state: A key of ``AI_ASSERT_STATES``.

        Raises:
            ValueError: ``state`` is not one this client can express. Nothing is
                sent, so it is never checked as some other state.
        """
        assert_state = AI_ASSERT_STATES.get(state)
        if assert_state is None:
            raise ValueError(
                f"cannot assert state {state!r}; use one of: "
                f"{', '.join(AI_ASSERT_STATES)}"
            )
        body: dict[str, Any] = {
            "target": text,
            "type": state,
            "assertion": assert_state.sentence.format(target=text),
        }
        return await self._request("POST", "/ui-bridge/sdk/ai/assert", body)

    async def sdk_ai_summary(self) -> UIBridgeResponse:
        """Get an AI-friendly summary of the current page."""
        return await self._request("GET", "/ui-bridge/sdk/ai/summary")

    async def sdk_components(self) -> UIBridgeResponse:
        """List all registered components in the connected SDK app."""
        return await self._request("GET", "/ui-bridge/sdk/components")

    # -------------------------------------------------------------------------
    # SDK Mode - Cross-App Analysis (/ui-bridge/sdk/ai/analyze/*)
    # -------------------------------------------------------------------------

    async def sdk_ai_analyze_data(self) -> UIBridgeResponse:
        """Extract labeled data values from the connected SDK app's page."""
        return await self._request("GET", "/ui-bridge/sdk/ai/analyze/data")

    async def sdk_ai_analyze_regions(self) -> UIBridgeResponse:
        """Segment the connected SDK app's page into semantic regions."""
        return await self._request("GET", "/ui-bridge/sdk/ai/analyze/regions")

    async def sdk_ai_analyze_structured_data(self) -> UIBridgeResponse:
        """Extract tables and lists from the connected SDK app's page."""
        return await self._request("GET", "/ui-bridge/sdk/ai/analyze/structured-data")

    async def sdk_ai_cross_app_compare(
        self,
        source_snapshot: dict[str, Any],
        target_snapshot: dict[str, Any],
    ) -> UIBridgeResponse:
        """Compare two semantic snapshots from different apps.

        Args:
            source_snapshot: Semantic snapshot from the source app.
            target_snapshot: Semantic snapshot from the target app.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/ai/analyze/cross-app-compare",
            {
                "sourceSnapshot": source_snapshot,
                "targetSnapshot": target_snapshot,
            },
        )

    async def sdk_ai_snapshot(self) -> UIBridgeResponse:
        """Get a semantic snapshot of the connected SDK app."""
        return await self._request("GET", "/ui-bridge/sdk/ai/snapshot")

    # -------------------------------------------------------------------------
    # SDK Mode - Page Navigation (/ui-bridge/sdk/page/*)
    # -------------------------------------------------------------------------

    async def sdk_page_refresh(self) -> UIBridgeResponse:
        """Refresh the current page in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/page/refresh")

    async def sdk_page_navigate(self, url: str) -> UIBridgeResponse:
        """Navigate the connected SDK app to a URL.

        Args:
            url: The URL to navigate to.
        """
        return await self._request("POST", "/ui-bridge/sdk/page/navigate", {"url": url})

    async def sdk_page_go_back(self) -> UIBridgeResponse:
        """Go back in browser history in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/page/back")

    async def sdk_page_go_forward(self) -> UIBridgeResponse:
        """Go forward in browser history in the connected SDK app."""
        return await self._request("POST", "/ui-bridge/sdk/page/forward")

    # -------------------------------------------------------------------------
    # Control Mode - Idle Detection (/ui-bridge/control/idle-*)
    # -------------------------------------------------------------------------

    async def control_idle_status(self, signal: str | None = None) -> UIBridgeResponse:
        """Get current idle status.

        Args:
            signal: Optional specific signal to query (e.g., 'network', 'dom',
                'loading-indicators'). If None, returns composite status with
                all signals.
        """
        if signal:
            return await self._request(
                "GET", f"/ui-bridge/control/idle-status/{signal}"
            )
        return await self._request("GET", "/ui-bridge/control/idle-status")

    async def control_wait_for_idle(
        self,
        timeout: int = 30000,
        min_stable_ms: int = 500,
        exclude: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Block until the app is idle (all signals stable).

        Args:
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long signals must remain idle before considered
                stable. Defaults to 500.
            exclude: Signal names to ignore (e.g., ['network']).
        """
        body: dict[str, Any] = {
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        if exclude:
            body["exclude"] = exclude
        # HTTP timeout must exceed the wait timeout
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST", "/ui-bridge/control/wait-for-idle", body, timeout=http_timeout
        )

    async def control_wait_for_signal(
        self,
        signal: str,
        timeout: int = 30000,
        min_stable_ms: int = 500,
    ) -> UIBridgeResponse:
        """Block until a specific signal is idle.

        Args:
            signal: Signal name ('network', 'dom', or 'loading-indicators').
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long the signal must remain idle. Defaults to 500.
        """
        body: dict[str, Any] = {
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            f"/ui-bridge/control/wait-for-idle/{signal}",
            body,
            timeout=http_timeout,
        )

    async def control_wait_for_targets(
        self,
        targets: list[str | dict[str, str]],
        timeout: int = 30000,
        min_stable_ms: int = 500,
    ) -> UIBridgeResponse:
        """Wait for specific targets to become idle.

        Args:
            targets: Array of signal names (strings) or indicator objects
                (e.g., {"indicator": ".loading-spinner"}).
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long targets must remain idle. Defaults to 500.
        """
        body: dict[str, Any] = {
            "targets": targets,
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            "/ui-bridge/control/wait-for-targets",
            body,
            timeout=http_timeout,
        )

    # -------------------------------------------------------------------------
    # SDK Mode - Idle Detection (/ui-bridge/sdk/idle-*)
    # -------------------------------------------------------------------------

    async def sdk_idle_status(self, signal: str | None = None) -> UIBridgeResponse:
        """Get current idle status from connected SDK app.

        Args:
            signal: Optional specific signal to query (e.g., 'network', 'dom',
                'loading-indicators'). If None, returns composite status with
                all signals.
        """
        if signal:
            return await self._request("GET", f"/ui-bridge/sdk/idle-status/{signal}")
        return await self._request("GET", "/ui-bridge/sdk/idle-status")

    async def sdk_wait_for_idle(
        self,
        timeout: int = 30000,
        min_stable_ms: int = 500,
        exclude: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Block until the SDK app is idle (all signals stable).

        Args:
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long signals must remain idle before considered
                stable. Defaults to 500.
            exclude: Signal names to ignore (e.g., ['network']).
        """
        body: dict[str, Any] = {
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        if exclude:
            body["exclude"] = exclude
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST", "/ui-bridge/sdk/wait-for-idle", body, timeout=http_timeout
        )

    async def sdk_diagnose_stuck(
        self,
        observation_window_ms: int = 3000,
        dom_mutation_threshold: int = 3,
    ) -> UIBridgeResponse:
        """Diagnose whether the connected SDK app is stuck on a loading screen.

        Args:
            observation_window_ms: How long to observe in ms. Defaults to 3000.
            dom_mutation_threshold: Fewer DOM mutations than this = not changing.
                Defaults to 3.
        """
        body: dict[str, Any] = {
            "observationWindowMs": observation_window_ms,
            "domMutationThreshold": dom_mutation_threshold,
        }
        http_timeout = (observation_window_ms / 1000) + 15.0
        return await self._request(
            "POST",
            "/ui-bridge/sdk/diagnose-stuck",
            body,
            timeout=http_timeout,
        )

    async def sdk_wait_for_signal(
        self,
        signal: str,
        timeout: int = 30000,
        min_stable_ms: int = 500,
    ) -> UIBridgeResponse:
        """Block until a specific signal is idle in the SDK app.

        Args:
            signal: Signal name ('network', 'dom', or 'loading-indicators').
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long the signal must remain idle. Defaults to 500.
        """
        body: dict[str, Any] = {
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            f"/ui-bridge/sdk/wait-for-idle/{signal}",
            body,
            timeout=http_timeout,
        )

    async def sdk_wait_for_targets(
        self,
        targets: list[str | dict[str, str]],
        timeout: int = 30000,
        min_stable_ms: int = 500,
    ) -> UIBridgeResponse:
        """Wait for specific targets to become idle in the SDK app.

        Args:
            targets: Array of signal names (strings) or indicator objects
                (e.g., {"indicator": ".loading-spinner"}).
            timeout: Max time to wait in ms. Defaults to 30000.
            min_stable_ms: How long targets must remain idle. Defaults to 500.
        """
        body: dict[str, Any] = {
            "targets": targets,
            "timeout": timeout,
            "minStableMs": min_stable_ms,
        }
        http_timeout = (timeout / 1000) + 10.0
        return await self._request(
            "POST",
            "/ui-bridge/sdk/wait-for-targets",
            body,
            timeout=http_timeout,
        )

    # -------------------------------------------------------------------------
    # Control Mode - Network Request Monitoring (/ui-bridge/control/network-*)
    # -------------------------------------------------------------------------

    async def control_network_requests(
        self,
        status: str | None = None,
        method: str | None = None,
        url_pattern: str | None = None,
        failures_only: bool = False,
        limit: int = 50,
    ) -> UIBridgeResponse:
        """List recent network requests from the runner's webview.

        Args:
            status: Filter by status: in-flight, completed, failed, cancelled.
            method: Filter by HTTP method (GET, POST, etc.).
            url_pattern: Filter by URL substring match.
            failures_only: Only show failed requests (4xx/5xx/network errors).
            limit: Max number of results. Defaults to 50.
        """
        params: dict[str, str] = {}
        if status:
            params["status"] = status
        if method:
            params["method"] = method
        if url_pattern:
            params["url_pattern"] = url_pattern
        if failures_only:
            params["failures_only"] = "true"
        if limit != 50:
            params["limit"] = str(limit)
        return await self._request(
            "GET",
            "/ui-bridge/control/network-requests",
            params=params or None,
        )

    async def control_network_requests_in_flight(self) -> UIBridgeResponse:
        """Show currently in-flight network requests from the runner's webview."""
        return await self._request(
            "GET", "/ui-bridge/control/network-requests/in-flight"
        )

    async def control_wait_for_network_request(
        self,
        url_pattern: str | None = None,
        method: str | None = None,
        timeout: int = 30000,
    ) -> UIBridgeResponse:
        """Wait for a network request matching the given criteria to complete.

        Args:
            url_pattern: URL substring to match.
            method: HTTP method to match.
            timeout: Timeout in milliseconds. Defaults to 30000.
        """
        return await self._request(
            "POST",
            "/ui-bridge/control/network-requests/wait",
            _network_wait_body(url_pattern, method, timeout),
            timeout=(timeout / 1000) + 10.0,
        )

    # -------------------------------------------------------------------------
    # SDK Mode - Network Request Monitoring (/ui-bridge/sdk/network-*)
    # -------------------------------------------------------------------------

    async def sdk_network_requests(
        self,
        status: str | None = None,
        method: str | None = None,
        url_pattern: str | None = None,
        failures_only: bool = False,
        limit: int = 50,
    ) -> UIBridgeResponse:
        """List recent network requests from the connected SDK app.

        Args:
            status: Filter by status: in-flight, completed, failed, cancelled.
            method: Filter by HTTP method (GET, POST, etc.).
            url_pattern: Filter by URL substring match.
            failures_only: Only show failed requests (4xx/5xx/network errors).
            limit: Max number of results. Defaults to 50.
        """
        params: dict[str, str] = {}
        if status:
            params["status"] = status
        if method:
            params["method"] = method
        if url_pattern:
            params["url_pattern"] = url_pattern
        if failures_only:
            params["failures_only"] = "true"
        if limit != 50:
            params["limit"] = str(limit)
        return await self._request(
            "GET",
            "/ui-bridge/sdk/network-requests",
            params=params or None,
        )

    async def sdk_network_requests_in_flight(self) -> UIBridgeResponse:
        """Show currently in-flight network requests from the connected SDK app."""
        return await self._request("GET", "/ui-bridge/sdk/network-requests/in-flight")

    async def sdk_wait_for_network_request(
        self,
        url_pattern: str | None = None,
        method: str | None = None,
        timeout: int = 30000,
    ) -> UIBridgeResponse:
        """Wait for a network request matching the given criteria to complete.

        Args:
            url_pattern: URL substring to match.
            method: HTTP method to match.
            timeout: Timeout in milliseconds. Defaults to 30000.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/network-requests/wait",
            _network_wait_body(url_pattern, method, timeout),
            timeout=(timeout / 1000) + 10.0,
        )

    # -------------------------------------------------------------------------
    # Control Mode - Design Review (/ui-bridge/control/design/*)
    # -------------------------------------------------------------------------

    async def control_design_snapshot(
        self,
        element_ids: list[str] | None = None,
        include_pseudo_elements: bool = False,
    ) -> UIBridgeResponse:
        """Get design data for all or filtered elements (control/runner path).

        Args:
            element_ids: Optional list of element IDs to include.
            include_pseudo_elements: Whether to include ::before/::after styles.
        """
        body: dict[str, Any] = {}
        if element_ids:
            body["elementIds"] = element_ids
        if include_pseudo_elements:
            body["includePseudoElements"] = True
        return await self._request(
            "POST", "/ui-bridge/control/design/snapshot", body or None
        )

    # -------------------------------------------------------------------------
    # SDK Mode - Design Review (/ui-bridge/sdk/design/*)
    # -------------------------------------------------------------------------

    async def sdk_design_element_styles(self, element_id: str) -> UIBridgeResponse:
        """Get extended computed styles for a specific element.

        Args:
            element_id: The element ID to inspect.
        """
        return await self._request(
            "GET", f"/ui-bridge/sdk/design/element/{element_id}/styles"
        )

    async def sdk_design_state_styles(
        self,
        element_id: str,
        states: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Get styles across interaction states (hover, focus, active, disabled).

        Args:
            element_id: The element ID to inspect.
            states: Interaction states to capture. Defaults to all.
        """
        body: dict[str, Any] = {}
        if states:
            body["states"] = states
        return await self._request(
            "POST",
            f"/ui-bridge/sdk/design/element/{element_id}/state-styles",
            body or None,
        )

    async def sdk_design_snapshot(
        self,
        element_ids: list[str] | None = None,
        include_pseudo_elements: bool = False,
    ) -> UIBridgeResponse:
        """Get design data for all or filtered elements.

        Args:
            element_ids: Optional list of element IDs to include.
            include_pseudo_elements: Whether to include ::before/::after styles.
        """
        body: dict[str, Any] = {}
        if element_ids:
            body["elementIds"] = element_ids
        if include_pseudo_elements:
            body["includePseudoElements"] = True
        return await self._request(
            "POST", "/ui-bridge/sdk/design/snapshot", body or None
        )

    async def sdk_design_responsive(
        self,
        viewports: dict[str, int] | None = None,
        element_ids: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Capture design snapshots at multiple viewport widths.

        Args:
            viewports: Map of label to width in px. Defaults to standard breakpoints.
            element_ids: Optional list of element IDs to include.
        """
        body: dict[str, Any] = {}
        if viewports:
            body["viewports"] = viewports
        if element_ids:
            body["elementIds"] = element_ids
        return await self._request("POST", "/ui-bridge/sdk/design/responsive", body)

    async def sdk_design_audit(
        self,
        guide: dict[str, Any] | None = None,
        element_ids: list[str] | None = None,
        limit: int | None = None,
        max_results: int | None = None,
        include_passed: bool | None = None,
        root_font_size_px: float | None = None,
    ) -> UIBridgeResponse:
        """Run a design audit: against a style guide, else the SDK heuristics.

        Each argument maps onto the SDK's ``DesignAuditRequest``
        (``specs/design-request.ts``); one left ``None`` is not sent, so the
        SDK applies its own default, and neither is an empty ``element_ids``
        (the SDK then audits every element, as the sibling design routes do).
        The SDK validates the values.

        Args:
            guide: Inline style guide config. Uses loaded guide if not provided.
                An empty guide is still sent, so the SDK judges it rather than
                silently falling back to the loaded guide or the heuristics.
            element_ids: Optional list of element IDs to audit.
            limit: Most elements to audit (``limit``).
            max_results: Most findings to list (``maxResults``).
            include_passed: List passing rule results too (``includePassed``,
                style-guide mode).
            root_font_size_px: Root font size ``rem`` resolves against
                (``rootFontSizePx``, style-guide mode).
        """
        body: dict[str, Any] = {}
        if guide is not None:
            body["guide"] = guide
        if element_ids:
            body["elementIds"] = element_ids
        if limit is not None:
            body["limit"] = limit
        if max_results is not None:
            body["maxResults"] = max_results
        if include_passed is not None:
            body["includePassed"] = include_passed
        if root_font_size_px is not None:
            body["rootFontSizePx"] = root_font_size_px
        return await self._request("POST", "/ui-bridge/sdk/design/audit", body or None)

    async def sdk_design_load_guide(self, guide: dict[str, Any]) -> UIBridgeResponse:
        """Load a style guide for subsequent audits.

        Args:
            guide: The style guide configuration (StyleGuideConfig).
        """
        return await self._request(
            "POST", "/ui-bridge/sdk/design/style-guide/load", {"guide": guide}
        )

    async def sdk_design_get_guide(self) -> UIBridgeResponse:
        """Get the currently loaded style guide."""
        return await self._request("GET", "/ui-bridge/sdk/design/style-guide")

    async def sdk_design_clear_guide(self) -> UIBridgeResponse:
        """Clear the currently loaded style guide."""
        return await self._request("DELETE", "/ui-bridge/sdk/design/style-guide")

    # -------------------------------------------------------------------------
    # SDK Mode - Quality Evaluation
    # -------------------------------------------------------------------------

    async def sdk_design_evaluate(
        self,
        context: str | None = None,
        custom_context: dict[str, Any] | None = None,
        element_ids: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Run holistic UI quality evaluation.

        Args:
            context: Built-in context name (general, minimal, data-dense, mobile, accessibility).
            custom_context: Custom context object with metric weights/thresholds.
            element_ids: Optional list of element IDs to evaluate.
        """
        body: dict[str, Any] = {}
        if context:
            body["context"] = context
        if custom_context:
            body["customContext"] = custom_context
        if element_ids:
            body["elementIds"] = element_ids
        return await self._request(
            "POST", "/ui-bridge/sdk/design/evaluate", body or None
        )

    async def sdk_design_evaluate_contexts(self) -> UIBridgeResponse:
        """Get available quality evaluation contexts."""
        return await self._request("GET", "/ui-bridge/sdk/design/evaluate/contexts")

    async def sdk_design_save_baseline(
        self,
        label: str | None = None,
        element_ids: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Save current element state as a baseline for diff comparison.

        Args:
            label: Optional label for the baseline.
            element_ids: Optional list of element IDs to include.
        """
        body: dict[str, Any] = {}
        if label:
            body["label"] = label
        if element_ids:
            body["elementIds"] = element_ids
        return await self._request(
            "POST", "/ui-bridge/sdk/design/evaluate/baseline", body or None
        )

    async def sdk_design_diff_baseline(
        self,
        element_ids: list[str] | None = None,
    ) -> UIBridgeResponse:
        """Diff current elements against saved baseline.

        Args:
            element_ids: Optional list of element IDs to diff.
        """
        body: dict[str, Any] = {}
        if element_ids:
            body["elementIds"] = element_ids
        return await self._request(
            "POST", "/ui-bridge/sdk/design/evaluate/diff", body or None
        )

    # -------------------------------------------------------------------------
    # Control Mode - Change Tracking (/ui-bridge/control/ai/*)
    # -------------------------------------------------------------------------

    async def control_save_bookmark(self, name: str) -> UIBridgeResponse:
        """Save a snapshot bookmark for later diffing (control mode).

        Args:
            name: Unique bookmark name.
        """
        return await self._request(
            "POST", "/ui-bridge/control/ai/bookmarks", {"name": name}
        )

    async def control_list_bookmarks(self) -> UIBridgeResponse:
        """List all saved snapshot bookmarks (control mode)."""
        return await self._request("GET", "/ui-bridge/control/ai/bookmarks")

    async def control_get_bookmark(self, name: str) -> UIBridgeResponse:
        """Get a specific bookmark by name (control mode).

        Args:
            name: Bookmark name.
        """
        return await self._request("GET", f"/ui-bridge/control/ai/bookmark/{name}")

    async def control_delete_bookmark(self, name: str) -> UIBridgeResponse:
        """Delete a saved bookmark (control mode).

        Args:
            name: Bookmark name to delete.
        """
        return await self._request("DELETE", f"/ui-bridge/control/ai/bookmark/{name}")

    async def control_diff_from_bookmark(self, name: str) -> UIBridgeResponse:
        """Diff current UI state against a saved bookmark (control mode).

        Args:
            name: Bookmark name to diff against.
        """
        return await self._request("GET", f"/ui-bridge/control/ai/bookmark/{name}/diff")

    async def control_execute_with_diff(
        self, request: dict[str, Any]
    ) -> UIBridgeResponse:
        """Execute an element action and capture what changed (control mode).

        Args:
            request: Request body with elementAction, settleTimeout, etc.
        """
        return await self._request(
            "POST",
            "/ui-bridge/control/ai/execute-with-diff",
            request,
            timeout=_execute_with_diff_timeout(request),
        )

    async def control_wait_for_change(
        self, predicate: dict[str, Any], options: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Wait for a change matching a predicate (control mode).

        Args:
            predicate: A ui-bridge ``ChangePredicate`` (``ai/types.ts``), e.g.
                ``{"anySignificantChange": True}``.
            options: ``WaitForChangeOptions``: ``timeout``, ``interval`` (ms).
        """
        body, http_timeout = _wait_for_change_body(predicate, options)
        return await self._request(
            "POST",
            "/ui-bridge/control/ai/wait-for-change",
            body,
            timeout=http_timeout,
        )

    async def control_categorize_last_diff(self) -> UIBridgeResponse:
        """Categorize the last computed diff (control mode)."""
        return await self._request("GET", "/ui-bridge/control/ai/categorize-last-diff")

    async def control_scoped_diff(
        self,
        scope: str,
        from_bookmark: str | None = None,
    ) -> UIBridgeResponse:
        """Get a scoped diff (control mode).

        Args:
            scope: Scope for the diff.
            from_bookmark: Optional bookmark name to diff against.
        """
        body: dict[str, Any] = {"scope": scope}
        if from_bookmark:
            body["fromBookmark"] = from_bookmark
        return await self._request("POST", "/ui-bridge/control/ai/scoped-diff", body)

    async def control_summarize_diff(self, body: dict[str, Any]) -> UIBridgeResponse:
        """Get a budget-aware summary of UI changes (control mode).

        Args:
            body: Request body with budget, fromBookmark, includeCategory.
        """
        return await self._request("POST", "/ui-bridge/control/ai/summarize-diff", body)

    async def control_structured_changes(
        self, body: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Analyze table and list changes (control mode).

        Args:
            body: Optional request body with fromBookmark.
        """
        return await self._request(
            "POST", "/ui-bridge/control/ai/structured-changes", body
        )

    async def control_enable_change_buffer(self) -> UIBridgeResponse:
        """Enable change buffering (control mode)."""
        return await self._request("POST", "/ui-bridge/control/ai/change-buffer/enable")

    async def control_disable_change_buffer(self) -> UIBridgeResponse:
        """Disable change buffering (control mode)."""
        return await self._request(
            "POST", "/ui-bridge/control/ai/change-buffer/disable"
        )

    async def control_drain_change_buffer(self) -> UIBridgeResponse:
        """Drain the change buffer (control mode)."""
        return await self._request("POST", "/ui-bridge/control/ai/change-buffer/drain")

    async def control_change_buffer_size(self) -> UIBridgeResponse:
        """Get change buffer size and enabled status (control mode)."""
        return await self._request("GET", "/ui-bridge/control/ai/change-buffer/size")

    # -------------------------------------------------------------------------
    # SDK Mode - Change Tracking (/ui-bridge/sdk/ai/*)
    # -------------------------------------------------------------------------

    async def sdk_save_bookmark(self, name: str) -> UIBridgeResponse:
        """Save a snapshot bookmark for later diffing (SDK mode).

        Args:
            name: Unique bookmark name.
        """
        return await self._request(
            "POST", "/ui-bridge/sdk/ai/bookmarks", {"name": name}
        )

    async def sdk_list_bookmarks(self) -> UIBridgeResponse:
        """List all saved snapshot bookmarks (SDK mode)."""
        return await self._request("GET", "/ui-bridge/sdk/ai/bookmarks")

    async def sdk_get_bookmark(self, name: str) -> UIBridgeResponse:
        """Get a specific bookmark by name (SDK mode).

        Args:
            name: Bookmark name.
        """
        return await self._request("GET", f"/ui-bridge/sdk/ai/bookmark/{name}")

    async def sdk_delete_bookmark(self, name: str) -> UIBridgeResponse:
        """Delete a saved bookmark (SDK mode).

        Args:
            name: Bookmark name to delete.
        """
        return await self._request("DELETE", f"/ui-bridge/sdk/ai/bookmark/{name}")

    async def sdk_diff_from_bookmark(self, name: str) -> UIBridgeResponse:
        """Diff current UI state against a saved bookmark (SDK mode).

        Args:
            name: Bookmark name to diff against.
        """
        return await self._request("GET", f"/ui-bridge/sdk/ai/bookmark/{name}/diff")

    async def sdk_execute_with_diff(self, request: dict[str, Any]) -> UIBridgeResponse:
        """Execute an element action and capture what changed (SDK mode).

        Args:
            request: Request body with elementAction, settleTimeout, etc.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/ai/execute-with-diff",
            request,
            timeout=_execute_with_diff_timeout(request),
        )

    async def sdk_wait_for_change(
        self, predicate: dict[str, Any], options: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Wait for a change matching a predicate (SDK mode).

        Args:
            predicate: A ui-bridge ``ChangePredicate`` (``ai/types.ts``), e.g.
                ``{"anySignificantChange": True}``.
            options: ``WaitForChangeOptions``: ``timeout``, ``interval`` (ms).
        """
        body, http_timeout = _wait_for_change_body(predicate, options)
        return await self._request(
            "POST", "/ui-bridge/sdk/ai/wait-for-change", body, timeout=http_timeout
        )

    async def sdk_categorize_last_diff(self) -> UIBridgeResponse:
        """Categorize the last computed diff (SDK mode)."""
        return await self._request("GET", "/ui-bridge/sdk/ai/categorize-last-diff")

    async def sdk_scoped_diff(
        self,
        scope: str,
        from_bookmark: str | None = None,
    ) -> UIBridgeResponse:
        """Get a scoped diff (SDK mode).

        Args:
            scope: Scope for the diff.
            from_bookmark: Optional bookmark name to diff against.
        """
        body: dict[str, Any] = {"scope": scope}
        if from_bookmark:
            body["fromBookmark"] = from_bookmark
        return await self._request("POST", "/ui-bridge/sdk/ai/scoped-diff", body)

    async def sdk_summarize_diff(self, body: dict[str, Any]) -> UIBridgeResponse:
        """Get a budget-aware summary of UI changes (SDK mode).

        Args:
            body: Request body with budget, fromBookmark, includeCategory.
        """
        return await self._request("POST", "/ui-bridge/sdk/ai/summarize-diff", body)

    async def sdk_structured_changes(
        self, body: dict[str, Any] | None = None
    ) -> UIBridgeResponse:
        """Analyze table and list changes (SDK mode).

        Args:
            body: Optional request body with fromBookmark.
        """
        return await self._request("POST", "/ui-bridge/sdk/ai/structured-changes", body)

    async def sdk_enable_change_buffer(self) -> UIBridgeResponse:
        """Enable change buffering (SDK mode)."""
        return await self._request("POST", "/ui-bridge/sdk/ai/change-buffer/enable")

    async def sdk_disable_change_buffer(self) -> UIBridgeResponse:
        """Disable change buffering (SDK mode)."""
        return await self._request("POST", "/ui-bridge/sdk/ai/change-buffer/disable")

    async def sdk_drain_change_buffer(self) -> UIBridgeResponse:
        """Drain the change buffer (SDK mode)."""
        return await self._request("POST", "/ui-bridge/sdk/ai/change-buffer/drain")

    async def sdk_change_buffer_size(self) -> UIBridgeResponse:
        """Get change buffer size and enabled status (SDK mode)."""
        return await self._request("GET", "/ui-bridge/sdk/ai/change-buffer/size")

    async def control_query_selector(
        self,
        selector: str,
        action: str | None = None,
        index: int | None = None,
    ) -> UIBridgeResponse:
        """Query DOM elements by CSS selector.

        Args:
            selector: CSS selector to query.
            action: Optional action to perform on matched element(s).
            index: Index of matched element to target (0-based).
        """
        body: dict[str, Any] = {"selector": selector}
        if action is not None:
            # The runner flattens this body into the event its own UI handles,
            # which reads the action as `params.action` (runner
            # usePageEvents.ts, case "query_selector"). A top-level `action`
            # was never read, so no click ever happened.
            body["params"] = {"action": action}
        if index is not None:
            body["index"] = index
        return await self._request(
            "POST", "/ui-bridge/sdk/control/query-selector", body
        )

    async def control_page_evaluate(self, expression: str) -> UIBridgeResponse:
        """Evaluate a JavaScript expression in the webview.

        The expression is validated by a regex on the frontend side.
        Only safe property access expressions are allowed.

        Args:
            expression: JavaScript expression to evaluate.
        """
        return await self._request(
            "POST",
            "/ui-bridge/sdk/control/page-evaluate",
            {"expression": expression},
        )

    async def _get(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> UIBridgeResponse:
        """Make a GET request (convenience wrapper)."""
        return await self._request("GET", endpoint, params=params, timeout=timeout)
