"""Project a UI Bridge ``discover`` payload into the snapshot the image assertions read.

Plan 2026-10-13-image-generation-and-visual-judgment, Phase 8. The L0 image
assertions read only ``id``, ``bbox``, ``visibility_reason`` and ``media``, so
that is all :func:`project_audit_snapshot` emits.

Everything from ``MEDIA_TYPE_KINDS`` to the end of :func:`project_media` is copied
from ``project_media`` in qontinui-claude-config
``scripts/uibridge-to-elementsnapshot.py`` (itself a mirror of ``projectMedia`` in
qontinui-web's ``normalize.ts``). A change to one belongs in the others.
``tests/test_audit_images.py`` carries that script's fixtures as of this copy. The
drift guard lives on the other side: qontinui-claude-config's CI checks this repo out
and runs ``scripts/test-media-projection-parity.py``, which feeds both ``project_media``
functions the Phase 7 suites' inputs, the shared SDK capture (enveloped and legacy) and
thousands of mutated rows, and fails on any difference over those inputs. A branch no
input reaches is not covered, so a new branch here wants a case there.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

# SDK `mediaType` -> `MediaKind`, for every type the tag does not refine.
MEDIA_TYPE_KINDS = {
    "picture": "picture",
    "svg": "svg",
    "video": "video",
    "canvas": "canvas",
    "background-image": "css_background",
}
# Tags whose kind is unambiguous even when the facts are unreadable.
TAG_KINDS = {
    "img": "img",
    "picture": "picture",
    "svg": "svg",
    "video": "video",
    "canvas": "canvas",
}
OBJECT_FITS = {"fill", "contain", "cover", "none", "scale-down"}
LOADINGS = {"lazy", "eager"}
DECODINGS = {"sync", "async", "auto"}
FETCH_PRIORITIES = {"high", "low", "auto"}
LOAD_STATES = {"pending", "loaded", "error", "lazy"}
# Image-optimizer endpoints transcode on the fly; their URL says nothing about
# the served format (legacy SDKs inferred one from it anyway).
TRANSCODING_PATH = re.compile(r"/_(?:next|vercel)/image/?$")


#: A UTF-16 surrogate code point. `json.loads` pairs valid ones into a single
#: character, so any left in a decoded string is LONE (a JSON `"\ud800"`).
_SURROGATE = re.compile("[\ud800-\udfff]")

#: HTML's "ASCII whitespace": tab, LF, FF, CR, space. Deliberately not
#: `str.strip()`, which also strips U+00A0 and every other Unicode space.
ASCII_WHITESPACE = "\t\n\f\r "


def src_digest(src: str) -> str:
    """`ElementMedia.src_digest`: lowercase hex SHA-256 of the URL's UTF-8
    bytes, QUERY KEPT — every `next/image` source is `/_next/image?url=…`, so
    stripping the query would make every optimized image one image.

    A lone surrogate becomes U+FFFD first, exactly as the JavaScript producers'
    encoders (Node `Buffer`, `TextEncoder`) do — otherwise the digest disagrees
    with web and native, and `str.encode` raises outright. The shared test
    vectors in `test-uibridge-to-elementsnapshot.py` pin both cases against
    vision-core's own `src_digest_of`."""
    return hashlib.sha256(_SURROGATE.sub("\ufffd", src).encode("utf-8")).hexdigest()


def media_alt(alt_text: Any, decorative: bool) -> dict[str, Any]:
    """`ElementMedia.alt` from the SDK's two halves. The text is stripped of
    ASCII whitespace (HTML's definition), and an alt that is PRESENT and empty
    after stripping is decorative — `alt="  "` reads as `alt=""` does."""
    if isinstance(alt_text, str):
        text = alt_text.strip(ASCII_WHITESPACE)
        if not text:
            return {"decorative": True}
        return {"decorative": decorative, "text": text}
    return {"decorative": decorative}


def _finite(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _size(w: Any, h: Any, *, positive: bool) -> dict[str, float] | None:
    fw, fh = _finite(w), _finite(h)
    if fw is None or fh is None:
        return None
    if (fw <= 0 or fh <= 0) if positive else (fw < 0 or fh < 0):
        return None
    return {"w": fw, "h": fh}


def _keyword(v: Any, allowed: set[str]) -> str | None:
    """A closed-vocabulary value, or None. Outside the vocabulary is UNKNOWN:
    `ElementMedia`'s enums would reject the whole snapshot on it."""
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    return s if s in allowed else None


def _nonempty(v: Any) -> str | None:
    return v if isinstance(v, str) and v.strip() else None


def _byte_count(v: Any) -> int | None:
    """A non-negative integral byte count (`u64` on the Rust side, which
    rejects `1234.0`), or None."""
    f = _finite(v)
    return int(f) if f is not None and f >= 0 and f.is_integer() else None


def _record(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def unwrap_media(raw: Any) -> tuple[dict[str, Any] | None, set[str], bool]:
    """`(facts, unmeasured, legacy)` from a row's `mediaMetadata`.

    `facts` is None when nothing can be read: an `unknown` / `absent` envelope,
    a measured one with no value, or a value that is not a record."""
    if not isinstance(raw, dict):
        return None, set(), False
    if "status" in raw:
        if raw.get("status") != "measured" or not isinstance(raw.get("value"), dict):
            return None, set(), False
        cov = _record(_record(raw.get("provenance")).get("coverage"))
        listed = cov.get("unmeasured")
        if not isinstance(listed, list):
            listed = []
        unmeasured = {
            u["dimension"]
            for u in listed
            if isinstance(u, dict) and isinstance(u.get("dimension"), str)
        }
        return raw["value"], unmeasured, False
    if "mediaType" in raw:
        return raw, set(), True
    return None, set(), False


def media_kind(
    media_type: Any, tag: str | None, facts: dict[str, Any] | None
) -> str | None:
    """`MediaKind` from the SDK's `mediaType`, refined by the tag. The SDK
    spells both an <img> and any `role="img"` element `image`; the tag tells
    them apart. With no tag the src does (a `role="img"` element never carries
    one). With no readable facts the tag alone decides, where it can."""
    if media_type == "image":
        if tag is not None:
            return "img" if tag == "img" else "role_img"
        return (
            "img" if facts is not None and _nonempty(facts.get("src")) else "role_img"
        )
    if isinstance(media_type, str) and media_type in MEDIA_TYPE_KINDS:
        return MEDIA_TYPE_KINDS[media_type]
    return TAG_KINDS.get(tag) if tag is not None else None


def cls_contribution(sources: Any) -> float | None:
    """Summed score of the layout shifts that moved the element, excluding
    shifts within 500 ms of input (CLS excludes those). `[]` is measured: 0.0.
    A malformed entry makes the sum UNKNOWN rather than an undercount."""
    if not isinstance(sources, list):
        return None
    total = 0.0
    for entry in sources:
        if not isinstance(entry, dict):
            return None
        value = _finite(entry.get("value"))
        if value is None or not isinstance(entry.get("hadRecentInput"), bool):
            return None
        if not entry["hadRecentInput"]:
            total += value
    return total


def project_media(el: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    """The `ElementMedia` record for a media row, or None for any other row
    (and for a media row whose kind cannot be named)."""
    raw = el.get("mediaMetadata")
    if not isinstance(raw, dict):
        raw = state.get("mediaMetadata")
    is_media_row = el.get("category") == "media" or isinstance(raw, dict)
    if not is_media_row:
        return None
    raw_tag = el.get("tagName")
    tag = raw_tag.strip().lower() if isinstance(raw_tag, str) else None
    facts, unmeasured, legacy = unwrap_media(raw)
    kind = media_kind(facts.get("mediaType") if facts else None, tag, facts)
    if kind is None:
        return None
    media: dict[str, Any] = {"kind": kind}

    if isinstance(state.get("inViewport"), bool):
        media["in_viewport"] = state["inViewport"]
    if facts is None:
        return media

    def fact(name: str) -> Any:
        # An unmeasured dimension is UNKNOWN even if a value rode along with it.
        return None if name in unmeasured else facts.get(name)

    redacted = _record(state.get("redaction")).get("content") is True

    # `natural_px` is the image RESOURCE's own pixel size. For a srcset
    # candidate the browser's `naturalWidth` is DENSITY-CORRECTED (a 1280w
    # source shown at DPR 2 reads ~640; a `2x` candidate reads half its size),
    # and a legacy SDK reads it raw without saying which candidate it matched —
    # so a legacy row with a `srcset` or `<picture>` sources carries no natural
    # size worth trusting, and it is omitted. An enveloped row's producer
    # de-corrects it by the matched candidate, or lists `naturalWidth` /
    # `naturalHeight` in `coverage.unmeasured` when no candidate resolves; the
    # value passes as given (never re-corrected here) and an unmeasured one is
    # omitted. `effective_resolution` is computed FROM that size, so it is
    # projected only beside a projected `natural_px` (below).
    natural = _size(fact("naturalWidth"), fact("naturalHeight"), positive=True)
    sources = fact("sources")
    density_corrected = legacy and (
        _nonempty(fact("srcset")) is not None
        or (isinstance(sources, list) and len(sources) > 0)
    )
    if natural and not density_corrected:
        media["natural_px"] = natural
    rendered = _size(fact("renderedWidth"), fact("renderedHeight"), positive=False)
    if rendered:
        media["rendered_px"] = rendered

    src = _nonempty(fact("src"))
    if src is not None and not redacted:
        media["src_digest"] = src_digest(src)

    if legacy:
        # Only the facts that mean what `ElementMedia` means; see the header.
        if kind in ("img", "picture") and isinstance(fact("lazyLoading"), bool):
            media["loading"] = "lazy" if fact("lazyLoading") else "eager"
        if (
            kind != "css_background"
        ):  # a legacy background is always "loaded", never measured
            load_state = _keyword(fact("loadingState"), LOAD_STATES)
            if load_state:
                media["load_state"] = load_state
        fmt = _nonempty(fact("format"))
        transcoded = (
            src is not None and TRANSCODING_PATH.search(urlsplit(src).path) is not None
        )
        if fmt and not transcoded:
            media["format"] = fmt
        # Legacy `isDecorative` is `img.alt === ''`, which reads a MISSING alt
        # as decorative. Only an image whose alt text is present can be
        # trusted; the other kinds measure it from attributes that mean it.
        alt_text = fact("altText")
        if not redacted and isinstance(fact("isDecorative"), bool):
            if kind not in ("img", "picture") or isinstance(alt_text, str):
                media["alt"] = media_alt(alt_text, fact("isDecorative"))
        return media

    effective = _finite(fact("effectiveResolution"))
    if effective is not None and effective > 0 and "natural_px" in media:
        media["effective_resolution"] = effective
    for name, field, allowed in (
        ("objectFit", "object_fit", OBJECT_FITS),
        ("loading", "loading", LOADINGS),
        ("decoding", "decoding", DECODINGS),
        ("fetchPriority", "fetch_priority", FETCH_PRIORITIES),
        ("loadingState", "load_state", LOAD_STATES),
    ):
        value = _keyword(fact(name), allowed)
        if value:
            media[field] = value
    position = _nonempty(fact("objectPosition"))
    if position:
        media["object_position"] = position
    if isinstance(fact("dimsReserved"), bool):
        media["dims_reserved"] = fact("dimsReserved")
    fmt = _nonempty(fact("format"))
    if fmt:
        media["format"] = fmt
    weight = _byte_count(fact("encodedBodySize"))
    if weight is not None:
        media["transfer_bytes"] = weight
    if (
        not redacted
        and "altText" not in unmeasured
        and "isDecorative" not in unmeasured
        and isinstance(facts.get("isDecorative"), bool)
    ):
        # The SDK reads both halves together; an absent altText beside them is
        # its MEASURED "this element has no text alternative".
        media["alt"] = media_alt(facts.get("altText"), facts["isDecorative"])
    if isinstance(fact("isLcpCandidate"), bool):
        media["is_lcp_candidate"] = fact("isLcpCandidate")
    cls = cls_contribution(fact("clsSources"))
    if cls is not None:
        media["cls_contribution"] = cls
    slot = _nonempty(fact("slotId"))
    if slot is not None and not redacted:
        media["slot_id"] = slot
    if "placeholder" not in unmeasured:
        # The SDK sets it only when the marker is there: absent-and-unlisted is
        # its measured "no placeholder marker". A PRESENT non-boolean is no
        # reading of the marker at all, so it is omitted, never coerced.
        if "placeholder" not in facts:
            media["placeholder"] = False
        elif isinstance(facts["placeholder"], bool):
            media["placeholder"] = facts["placeholder"]
    return media


def _project_bbox(state: dict[str, Any]) -> dict[str, int] | None:
    """``{x,y,width,height}`` floats as ``{x,y,w,h}`` ints: the origin stays signed,
    the extent is clamped at 0, and a partial, non-finite or out-of-range rect is
    dropped, since one the runner cannot parse would fail the whole snapshot."""
    rect = state.get("rect")
    if not isinstance(rect, dict):
        return None
    vals = [_finite(rect.get(key)) for key in ("x", "y", "width", "height")]
    x, y, w, h = vals
    if x is None or y is None or w is None or h is None:
        return None
    bbox = {"x": round(x), "y": round(y), "w": max(0, round(w)), "h": max(0, round(h))}
    in_range = (
        abs(bbox["x"]) < 2**31
        and abs(bbox["y"]) < 2**31
        and bbox["w"] < 2**32
        and bbox["h"] < 2**32
    )
    return bbox if in_range else None


def project_audit_row(el: Any) -> dict[str, Any] | None:
    """The snapshot element for one ``discover`` row, or None for a row that is not
    a media element. A media row whose facts could not be read stays, with no
    ``media``: naming it answers unknown, where dropping it would read as no image."""
    if not isinstance(el, dict) or not isinstance(el.get("id"), str):
        return None
    state = _record(el.get("state"))
    media = project_media(el, state)
    reports_media = (
        el.get("category") == "media"
        or isinstance(el.get("mediaMetadata"), dict)
        or isinstance(state.get("mediaMetadata"), dict)
    )
    if media is None and not reports_media:
        return None
    out: dict[str, Any] = {"id": el["id"]}
    if media is not None:
        out["media"] = media
    bbox = _project_bbox(state)
    if bbox is not None:
        out["bbox"] = bbox
    reason = state.get("visibilityReason")
    if isinstance(reason, str) and reason:
        out["visibility_reason"] = reason
    return out


def project_audit_snapshot(discover: Any) -> dict[str, Any]:
    """The ``ElementSnapshot`` for a ``discover`` reply: its media rows only."""
    rows = discover.get("elements") if isinstance(discover, dict) else None
    projected = (project_audit_row(el) for el in rows) if isinstance(rows, list) else ()
    return {"elements": [row for row in projected if row is not None]}


def unread_image_ids(discover: Any) -> list[str]:
    """The media rows of a ``discover`` reply with no readable facts. A tagged one
    still projects a kind, so a page-wide check would count it and hold."""
    rows = discover.get("elements") if isinstance(discover, dict) else None
    out: list[str] = []
    for el in rows if isinstance(rows, list) else ():
        row = project_audit_row(el)
        if row is None:
            continue
        raw = el.get("mediaMetadata")
        if not isinstance(raw, dict):
            raw = _record(el.get("state")).get("mediaMetadata")
        if "media" not in row or unwrap_media(raw)[0] is None:
            out.append(row["id"])
    return out


#: The L0 image assertions, each with the caller threshold it needs, if any.
L0_IMAGE_ASSERTIONS: tuple[tuple[str, str | None], ...] = (
    ("image_loaded", None),
    ("image_resolution_sufficient", "min_effective"),
    ("image_not_distorted", "max_aspect_delta"),
    ("image_dims_reserved", None),
    ("lcp_image_prioritized", None),
    ("image_has_text_alternative", None),
    ("image_not_oversized", "max_effective"),
    ("image_weight_within", "max_bytes"),
    ("offscreen_image_lazy", None),
    ("no_duplicate_images", None),
)


def image_assertions(
    ids: list[str],
    thresholds: dict[str, Any],
    *,
    per_image: bool,
    unread: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """The L0 set: one assertion per image and check, or one page-wide per check.

    With no images it stays page-wide, so the runner answers each check's unknown
    rather than an empty list reading as nothing wrong. An omitted threshold is
    left out, and the runner answers unknown for it.

    A page-wide check can hold over an image whose facts could not be read.
    ``unread`` names those images beside it, and each answers unknown."""
    targets: list[str | None] = list(ids) if per_image and ids else [None, *unread]
    out = []
    for kind, threshold in L0_IMAGE_ASSERTIONS:
        fields: dict[str, Any] = {"type": kind}
        if threshold is not None and thresholds.get(threshold) is not None:
            fields[threshold] = thresholds[threshold]
        for target in targets:
            out.append(fields if target is None else {**fields, "element": target})
    return out
