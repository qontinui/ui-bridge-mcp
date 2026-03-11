"""Screenshot annotation, diffing, and visual analysis utilities.

Provides enhanced screenshot annotation with multiple modes, state-colored
element overlays, smart cropping, format conversion, visual diffing, and
element-level screenshots.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ANNOTATION_MODES = (
    "interactive",
    "all",
    "validation",
    "modal",
    "state",
    "relationships",
    "accessibility",
    "boxmodel",
)
CROP_MODES = ("full", "viewport", "modal")
IMAGE_FORMATS = ("png", "jpeg", "webp")

# WCAG minimum touch target size (px)
_WCAG_MIN_TARGET = 44

# Box model overlay colors (CSS DevTools style)
_COLOR_MARGIN = (255, 179, 102, 80)  # orange
_COLOR_BORDER = (255, 255, 102, 100)  # yellow
_COLOR_PADDING = (144, 238, 144, 80)  # green
_COLOR_CONTENT_BOX = (173, 216, 230, 60)  # blue

# Relationship line colors by type
_REL_COLORS: dict[str, str] = {
    "controls": "#2196F3",
    "labels": "#4CAF50",
    "describes": "#8BC34A",
    "validates": "#F44336",
    "toggles": "#9C27B0",
    "activates": "#FF9800",
    "submits": "#00BCD4",
    "owns": "#607D8B",
    "navigatesTo": "#3F51B5",
    "filters": "#CDDC39",
    "populates": "#795548",
    "dependsOn": "#E91E63",
}
_REL_DEFAULT_COLOR = "#9E9E9E"

# Element type strings considered interactive
INTERACTIVE_TYPES = frozenset(
    {
        "button",
        "input",
        "select",
        "textarea",
        "checkbox",
        "radio",
        "link",
        "a",
        "switch",
        "slider",
        "toggle",
        "tab",
        "menuitem",
        "option",
        "combobox",
        "searchbox",
        "spinbutton",
        "scrollbar",
        "listbox",
    }
)


@dataclass
class AnnotationOptions:
    """Options for screenshot annotation."""

    mode: str = "interactive"
    highlight_elements: list[str] | None = None
    crop: str = "full"
    scale: float = 1.0
    format: str = "png"
    quality: int = 85


@dataclass
class DiffResult:
    """Result of a visual diff between two screenshots."""

    diff_image_b64: str
    change_percentage: float
    changed_regions: list[dict[str, int]]
    passed: bool
    threshold_used: float


# ---------------------------------------------------------------------------
# State → color mapping
# ---------------------------------------------------------------------------

# Each style: (outline_color, fill_color, text_color, line_width)
_STYLE_DEFAULT = ("#F44336", "#F44336", "white", 2)  # red
_STYLE_DISABLED = ("#9E9E9E", "#9E9E9E", "white", 1)  # gray
_STYLE_FOCUSED = ("#2196F3", "#2196F3", "white", 3)  # blue
_STYLE_ERROR = ("#D32F2F", "#D32F2F", "white", 3)  # dark red
_STYLE_WARNING = ("#FF9800", "#FF9800", "white", 2)  # orange
_STYLE_VALID = ("#4CAF50", "#4CAF50", "white", 2)  # green
_STYLE_CONTENT = ("#78909C", "#78909C", "white", 1)  # blue-gray
_STYLE_OFFSCREEN = ("#FFD600", "#FFD600", "black", 1)  # yellow


def _element_style(el: dict[str, Any], mode: str) -> tuple[str, str, str, int]:
    """Determine annotation style for an element based on mode and state."""
    state = el.get("state", {})
    category = el.get("category", "interactive")

    if mode == "state":
        # Color by state priority: error > focused > disabled > valid > default
        vs = state.get("validationState", {})
        if vs and vs.get("valid") is False:
            return _STYLE_ERROR
        if state.get("focused"):
            return _STYLE_FOCUSED
        if not state.get("enabled", True):
            return _STYLE_DISABLED
        if vs and vs.get("valid") is True:
            return _STYLE_VALID
        if category == "content":
            return _STYLE_CONTENT
        if state.get("inViewport") is False:
            return _STYLE_OFFSCREEN
        return _STYLE_DEFAULT

    if mode == "validation":
        vs = state.get("validationState", {})
        if vs and vs.get("valid") is False:
            return _STYLE_ERROR
        if vs and vs.get("valid") is True:
            return _STYLE_VALID
        # Neutral/unknown validation
        return _STYLE_WARNING if state.get("required") else _STYLE_DEFAULT

    # Default styling for other modes
    if not state.get("enabled", True):
        return _STYLE_DISABLED
    if state.get("focused"):
        return _STYLE_FOCUSED
    if category == "content":
        return _STYLE_CONTENT
    return _STYLE_DEFAULT


def _should_annotate(
    el: dict[str, Any],
    mode: str,
    highlight_ids: set[str] | None,
    modal_element_ids: set[str] | None,
) -> bool:
    """Decide whether to annotate a given element."""
    state = el.get("state", {})
    rect = state.get("rect")
    if not rect:
        return False
    if not state.get("visible", True):
        return False

    elem_id = el.get("id", "")

    # If explicit highlight list, only those
    if highlight_ids is not None:
        return elem_id in highlight_ids

    category = el.get("category", "interactive")
    el_type = el.get("type", "").lower()

    if mode == "interactive":
        return category == "interactive" or el_type in INTERACTIVE_TYPES

    if mode == "all":
        return True

    if mode == "validation":
        vs = state.get("validationState")
        return vs is not None or state.get("required", False)

    if mode == "modal":
        # Only annotate elements inside modal, or the modal itself
        if modal_element_ids:
            return elem_id in modal_element_ids
        # Fallback: annotate all interactive
        return category == "interactive" or el_type in INTERACTIVE_TYPES

    if mode == "state":
        return True

    if mode == "relationships":
        # Annotate elements that participate in relationships
        return True

    if mode == "accessibility":
        return True

    if mode == "boxmodel":
        return True

    return True


# ---------------------------------------------------------------------------
# Core annotation
# ---------------------------------------------------------------------------


def _load_font(size: int = 12) -> Any:
    """Load a font, trying system fonts then falling back to default."""
    from PIL import ImageFont

    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def annotate_screenshot(
    screenshot_b64: str,
    elements: list[dict[str, Any]],
    width: int,
    height: int,
    rm: Any,
    *,
    options: AnnotationOptions | None = None,
    snapshot: dict[str, Any] | None = None,
    design_data: list[dict[str, Any]] | None = None,
) -> str:
    """Annotate a screenshot with element overlays.

    Enhanced version with annotation modes, state colors, cropping, and
    format conversion.

    Args:
        screenshot_b64: Base64 encoded screenshot PNG.
        elements: Element list from snapshot.
        width: CSS pixel width of the viewport.
        height: CSS pixel height of the viewport.
        rm: RefManager instance for assigning compact refs.
        options: Annotation options (mode, crop, scale, format, etc.).
        snapshot: Full snapshot dict (needed for modal-aware mode and viewport crop).
        design_data: Element design data from sdk_design_snapshot (for boxmodel mode).

    Returns:
        Base64 encoded image in the requested format.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("Pillow not installed. Returning unannotated screenshot.")
        return screenshot_b64

    if options is None:
        options = AnnotationOptions()

    img: Any = Image.open(io.BytesIO(base64.b64decode(screenshot_b64)))

    # For modes that draw semi-transparent overlays, convert to RGBA
    needs_alpha = options.mode in ("boxmodel", "accessibility", "relationships")
    if needs_alpha and img.mode != "RGBA":
        img = img.convert("RGBA")

    draw = ImageDraw.Draw(img)

    # DPI scaling
    scale_x = img.width / width if width else 1.0
    scale_y = img.height / height if height else 1.0

    font = _load_font(12)

    # Build highlight set if specified
    highlight_ids: set[str] | None = None
    if options.highlight_elements:
        highlight_ids = set(options.highlight_elements)

    # Build modal element set for modal-aware mode
    modal_element_ids: set[str] | None = None
    modal_rect: dict[str, float] | None = None
    if options.mode == "modal" and snapshot:
        modal_element_ids, modal_rect = _extract_modal_context(snapshot, elements)

    # Modal dimming overlay
    if options.mode == "modal" and modal_rect:
        _apply_modal_dimming(img, modal_rect, scale_x, scale_y)
        # Recreate draw after image modification
        draw = ImageDraw.Draw(img)

    # Box model overlay (draws before element rectangles for layering)
    if options.mode == "boxmodel":
        draw_box_model_overlay(draw, elements, design_data, scale_x, scale_y, font)

    # Accessibility overlay
    if options.mode == "accessibility":
        draw_accessibility_overlay(draw, elements, scale_x, scale_y, font)

    # Annotate elements (skip for boxmodel — box model draws its own overlays)
    if options.mode != "boxmodel":
        for el in elements:
            if not _should_annotate(el, options.mode, highlight_ids, modal_element_ids):
                continue

            state = el.get("state", {})
            rect = state.get("rect", {})
            elem_id = el.get("id", "?")
            ref = rm.assign(elem_id)

            x = rect.get("x", 0) * scale_x
            y = rect.get("y", 0) * scale_y
            w = rect.get("width", 0) * scale_x
            h = rect.get("height", 0) * scale_y

            outline_color, fill_color, text_color, line_width = _element_style(
                el, options.mode
            )

            # Draw rectangle
            draw.rectangle(
                (x, y, x + w, y + h), outline=outline_color, width=line_width
            )

            # Draw ref label
            text_bbox = draw.textbbox((0, 0), ref, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            label_y = max(y - th - 4, 0)
            draw.rectangle((x, label_y, x + tw + 4, label_y + th + 2), fill=fill_color)
            draw.text((x + 2, label_y), ref, fill=text_color, font=font)

    # Relationship connector lines (drawn on top of elements)
    if options.mode == "relationships" and snapshot:
        _draw_relationship_lines(draw, elements, snapshot, scale_x, scale_y, font)

    # Viewport / scroll indicators (drawn on any mode if snapshot has viewport)
    if snapshot and snapshot.get("viewport"):
        _draw_viewport_indicators(draw, img, snapshot, elements, scale_x, scale_y, font)

    # Crop
    img = _apply_crop(img, options.crop, snapshot, scale_x, scale_y, modal_rect)

    # Scale
    if options.scale != 1.0:
        new_w = max(1, int(img.width * options.scale))
        new_h = max(1, int(img.height * options.scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Encode
    return _encode_image(img, options.format, options.quality)


# ---------------------------------------------------------------------------
# Modal-aware helpers
# ---------------------------------------------------------------------------


def _extract_modal_context(
    snapshot: dict[str, Any],
    elements: list[dict[str, Any]],
) -> tuple[set[str] | None, dict[str, float] | None]:
    """Extract modal element IDs and bounding rect from snapshot."""
    modal_stack = snapshot.get("modalStack", {})
    active_modals = modal_stack.get("activeModals", [])
    if not active_modals:
        return None, None

    # Use the topmost modal
    top_modal = active_modals[-1]
    modal_id = top_modal.get("id", "")

    # Find elements that belong to the modal
    # Heuristic: elements whose ID starts with/contains the modal ID,
    # or elements within the modal's bounding rect
    modal_rect: dict[str, float] | None = None

    # Try to find the modal element itself for its rect
    for el in elements:
        if el.get("id") == modal_id:
            r = el.get("state", {}).get("rect")
            if r:
                modal_rect = r
            break

    if not modal_rect:
        # No rect found for modal, fall back to None
        return None, None

    # Collect elements inside the modal's bounding rect
    modal_ids: set[str] = set()
    mx = modal_rect.get("x", 0)
    my = modal_rect.get("y", 0)
    mw = modal_rect.get("width", 0)
    mh = modal_rect.get("height", 0)

    for el in elements:
        r = el.get("state", {}).get("rect")
        if not r:
            continue
        ex = r.get("x", 0)
        ey = r.get("y", 0)
        # Element center is inside modal bounds
        ecx = ex + r.get("width", 0) / 2
        ecy = ey + r.get("height", 0) / 2
        if mx <= ecx <= mx + mw and my <= ecy <= my + mh:
            modal_ids.add(el.get("id", ""))

    return modal_ids, modal_rect


def _apply_modal_dimming(
    img: Any,
    modal_rect: dict[str, float],
    scale_x: float,
    scale_y: float,
) -> None:
    """Apply a dark semi-transparent overlay outside the modal area."""
    from PIL import Image, ImageDraw

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Dim entire image
    draw.rectangle((0, 0, img.width, img.height), fill=(0, 0, 0, 120))

    # Cut out the modal region (make it transparent)
    mx = int(modal_rect.get("x", 0) * scale_x)
    my = int(modal_rect.get("y", 0) * scale_y)
    mw = int(modal_rect.get("width", 0) * scale_x)
    mh = int(modal_rect.get("height", 0) * scale_y)
    draw.rectangle((mx, my, mx + mw, my + mh), fill=(0, 0, 0, 0))

    # Composite
    if img.mode != "RGBA":
        img_rgba = img.convert("RGBA")
        composited = Image.alpha_composite(img_rgba, overlay)
        # Copy pixels back to original image
        img.paste(composited.convert(img.mode))
    else:
        composited = Image.alpha_composite(img, overlay)
        img.paste(composited)


# ---------------------------------------------------------------------------
# Relationship lines (A3)
# ---------------------------------------------------------------------------


def _draw_relationship_lines(
    draw: Any,
    elements: list[dict[str, Any]],
    snapshot: dict[str, Any],
    scale_x: float,
    scale_y: float,
    font: Any,
) -> None:
    """Draw connector lines between related elements."""
    rels_ctx = snapshot.get("relationships", {})
    relationships = rels_ctx.get("relationships", [])
    if not relationships:
        return

    # Build element center map
    centers: dict[str, tuple[float, float]] = {}
    for el in elements:
        eid = el.get("id", "")
        rect = el.get("state", {}).get("rect")
        if rect:
            cx = (rect.get("x", 0) + rect.get("width", 0) / 2) * scale_x
            cy = (rect.get("y", 0) + rect.get("height", 0) / 2) * scale_y
            centers[eid] = (cx, cy)

    for rel in relationships:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rel_type = rel.get("type", "")
        if src not in centers or tgt not in centers:
            continue

        sx, sy = centers[src]
        tx, ty = centers[tgt]
        color = _REL_COLORS.get(rel_type, _REL_DEFAULT_COLOR)

        # Draw line
        draw.line([(sx, sy), (tx, ty)], fill=color, width=2)

        # Draw arrowhead at target
        _draw_arrowhead(draw, sx, sy, tx, ty, color)

        # Draw relationship type label at midpoint
        mx, my = (sx + tx) / 2, (sy + ty) / 2
        label = rel_type
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle(
            (mx - 2, my - th - 2, mx + tw + 4, my + 2),
            fill=color,
        )
        draw.text((mx, my - th - 1), label, fill="white", font=font)


def _draw_arrowhead(
    draw: Any,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    color: str,
) -> None:
    """Draw a small arrowhead at (tx, ty) pointing from (sx, sy)."""
    import math

    dx = tx - sx
    dy = ty - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    # Normalize
    ux, uy = dx / length, dy / length
    # Arrow size
    size = 8
    # Perpendicular
    px, py = -uy, ux
    # Arrow points
    p1 = (tx - ux * size + px * size / 2, ty - uy * size + py * size / 2)
    p2 = (tx - ux * size - px * size / 2, ty - uy * size - py * size / 2)
    draw.polygon([(tx, ty), p1, p2], fill=color)


# ---------------------------------------------------------------------------
# Viewport / scroll indicators (A4)
# ---------------------------------------------------------------------------


def _draw_viewport_indicators(
    draw: Any,
    img: Any,
    snapshot: dict[str, Any],
    elements: list[dict[str, Any]],
    scale_x: float,
    scale_y: float,
    font: Any,
) -> None:
    """Draw scroll indicators and off-screen element count."""

    viewport = snapshot.get("viewport", {})
    if not viewport:
        return

    vp_w = viewport.get("viewportWidth", 0)
    vp_h = viewport.get("viewportHeight", 0)
    doc_w = viewport.get("documentWidth", vp_w)
    doc_h = viewport.get("documentHeight", vp_h)
    can_down = viewport.get("canScrollDown", False)
    can_right = viewport.get("canScrollRight", False)
    scroll_x = viewport.get("scrollX", 0)
    scroll_y = viewport.get("scrollY", 0)

    iw, ih = img.size

    # Scroll arrows
    arrow_size = 20
    if can_down:
        # Down arrow at bottom center
        cx = iw // 2
        cy = ih - arrow_size - 4
        draw.polygon(
            [(cx - arrow_size, cy), (cx + arrow_size, cy), (cx, cy + arrow_size)],
            fill=(255, 255, 255, 200),
            outline=(100, 100, 100),
        )
    if can_right:
        # Right arrow at right center
        cx = iw - arrow_size - 4
        cy = ih // 2
        draw.polygon(
            [(cx, cy - arrow_size), (cx, cy + arrow_size), (cx + arrow_size, cy)],
            fill=(255, 255, 255, 200),
            outline=(100, 100, 100),
        )
    if scroll_y > 0:
        # Up arrow at top center
        cx = iw // 2
        cy = arrow_size + 4
        draw.polygon(
            [(cx - arrow_size, cy), (cx + arrow_size, cy), (cx, cy - arrow_size)],
            fill=(255, 255, 255, 200),
            outline=(100, 100, 100),
        )
    if scroll_x > 0:
        # Left arrow at left center
        cx = arrow_size + 4
        cy = ih // 2
        draw.polygon(
            [(cx, cy - arrow_size), (cx, cy + arrow_size), (cx - arrow_size, cy)],
            fill=(255, 255, 255, 200),
            outline=(100, 100, 100),
        )

    # Count off-screen elements
    offscreen_count = sum(
        1
        for el in elements
        if el.get("state", {}).get("inViewport") is False
        and el.get("state", {}).get("visible", True)
    )
    if offscreen_count > 0:
        label = f"{offscreen_count} off-screen"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        bx = iw - tw - 12
        by = 4
        draw.rectangle(
            (bx, by, bx + tw + 8, by + th + 6),
            fill=(0, 0, 0, 180),
        )
        draw.text((bx + 4, by + 3), label, fill=(255, 200, 0), font=font)

    # Minimap in bottom-right corner
    if doc_w > vp_w or doc_h > vp_h:
        _draw_minimap(draw, iw, ih, vp_w, vp_h, doc_w, doc_h, scroll_x, scroll_y)


def _draw_minimap(
    draw: Any,
    img_w: int,
    img_h: int,
    vp_w: float,
    vp_h: float,
    doc_w: float,
    doc_h: float,
    scroll_x: float,
    scroll_y: float,
) -> None:
    """Draw a page minimap in the bottom-right corner."""
    max_minimap = 80
    margin = 8

    # Scale factor for minimap
    ratio = min(max_minimap / doc_w, max_minimap / doc_h)
    mm_w = max(20, int(doc_w * ratio))
    mm_h = max(20, int(doc_h * ratio))

    # Position in bottom-right
    mm_x = img_w - mm_w - margin
    mm_y = img_h - mm_h - margin

    # Background
    draw.rectangle(
        (mm_x - 1, mm_y - 1, mm_x + mm_w + 1, mm_y + mm_h + 1),
        fill=(30, 30, 30, 200),
        outline=(100, 100, 100),
    )

    # Viewport rectangle
    vp_rx = int(scroll_x * ratio) + mm_x
    vp_ry = int(scroll_y * ratio) + mm_y
    vp_rw = max(2, int(vp_w * ratio))
    vp_rh = max(2, int(vp_h * ratio))
    draw.rectangle(
        (vp_rx, vp_ry, vp_rx + vp_rw, vp_ry + vp_rh),
        outline=(0, 150, 255),
        width=1,
    )


# ---------------------------------------------------------------------------
# Box model overlay (C1)
# ---------------------------------------------------------------------------


def draw_box_model_overlay(
    draw: Any,
    elements: list[dict[str, Any]],
    design_data: list[dict[str, Any]] | None,
    scale_x: float,
    scale_y: float,
    font: Any,
) -> None:
    """Draw DevTools-style box model overlays on elements.

    Uses design_data (from sdk_design_snapshot) for margin/padding/border info.
    Falls back to element rects if design_data unavailable.
    """
    if not design_data:
        return

    # Map design data by element ID
    design_map: dict[str, dict[str, Any]] = {}
    for entry in design_data:
        eid = entry.get("elementId", "")
        if eid:
            design_map[eid] = entry

    for el in elements:
        eid = el.get("id", "")
        state = el.get("state", {})
        rect = state.get("rect")
        if not rect or not state.get("visible", True):
            continue

        dd = design_map.get(eid)
        if not dd:
            continue

        styles = dd.get("styles", {})
        x = rect.get("x", 0) * scale_x
        y = rect.get("y", 0) * scale_y
        w = rect.get("width", 0) * scale_x
        h = rect.get("height", 0) * scale_y

        # Parse box model values
        mt = _parse_px(styles.get("marginTop", "0"))
        mr = _parse_px(styles.get("marginRight", "0"))
        mb = _parse_px(styles.get("marginBottom", "0"))
        ml = _parse_px(styles.get("marginLeft", "0"))

        pt = _parse_px(styles.get("paddingTop", "0"))
        pr = _parse_px(styles.get("paddingRight", "0"))
        pb = _parse_px(styles.get("paddingBottom", "0"))
        pl = _parse_px(styles.get("paddingLeft", "0"))

        bt = _parse_border_width(styles.get("border", ""))

        # Scale to physical pixels
        mt, mr, mb, ml = mt * scale_y, mr * scale_x, mb * scale_y, ml * scale_x
        pt, pr, pb, pl = pt * scale_y, pr * scale_x, pb * scale_y, pl * scale_x
        bt_s = bt * min(scale_x, scale_y)

        # Margin box (outermost)
        draw.rectangle(
            (x - ml, y - mt, x + w + mr, y + h + mb),
            fill=_COLOR_MARGIN,
        )
        # Border box
        draw.rectangle(
            (x, y, x + w, y + h),
            fill=_COLOR_BORDER,
        )
        # Padding box
        draw.rectangle(
            (x + bt_s, y + bt_s, x + w - bt_s, y + h - bt_s),
            fill=_COLOR_PADDING,
        )
        # Content box (innermost)
        draw.rectangle(
            (x + bt_s + pl, y + bt_s + pt, x + w - bt_s - pr, y + h - bt_s - pb),
            fill=_COLOR_CONTENT_BOX,
        )

        # Font size label
        font_size = styles.get("fontSize", "")
        if font_size:
            label = font_size
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.rectangle(
                (x + w + 2, y, x + w + tw + 6, y + th + 4),
                fill=(0, 0, 0, 180),
            )
            draw.text((x + w + 4, y + 2), label, fill=(173, 216, 230), font=font)


def _parse_px(value: str) -> float:
    """Parse a CSS pixel value like '16px' to float."""
    if not value:
        return 0.0
    v = value.replace("px", "").strip()
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _parse_border_width(border: str) -> float:
    """Parse border shorthand to get width in px."""
    if not border or border == "none":
        return 0.0
    # e.g. "1px solid rgb(0,0,0)"
    parts = border.split()
    for part in parts:
        if part.endswith("px"):
            try:
                return float(part.replace("px", ""))
            except ValueError:
                pass
    return 0.0


# ---------------------------------------------------------------------------
# Accessibility overlay (C2)
# ---------------------------------------------------------------------------


def draw_accessibility_overlay(
    draw: Any,
    elements: list[dict[str, Any]],
    scale_x: float,
    scale_y: float,
    font: Any,
) -> list[dict[str, Any]]:
    """Draw accessibility audit indicators on elements.

    Checks:
    - Touch target size (< 44x44px)
    - Missing labels (interactive elements without label)
    - Focus order numbers (for focusable elements)
    - Validation state indicators

    Returns list of issues found.
    """
    issues: list[dict[str, Any]] = []
    focus_order = 0

    for el in elements:
        state = el.get("state", {})
        rect = state.get("rect")
        if not rect or not state.get("visible", True):
            continue

        eid = el.get("id", "")
        el_type = el.get("type", "").lower()
        category = el.get("category", "interactive")
        label = el.get("label", "")

        x = rect.get("x", 0) * scale_x
        y = rect.get("y", 0) * scale_y
        w = rect.get("width", 0) * scale_x
        h = rect.get("height", 0) * scale_y

        css_w = rect.get("width", 0)
        css_h = rect.get("height", 0)

        is_interactive = category == "interactive" or el_type in INTERACTIVE_TYPES

        if not is_interactive:
            continue

        # Focus order number
        if state.get("enabled", True):
            focus_order += 1
            fo_text = str(focus_order)
            bbox = draw.textbbox((0, 0), fo_text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            # Circle with number in top-left corner
            cx_pos = x - 2
            cy_pos = y - 2
            r = max(tw, th) // 2 + 4
            draw.ellipse(
                (cx_pos - r, cy_pos - r, cx_pos + r, cy_pos + r),
                fill=(33, 150, 243, 200),
                outline="white",
            )
            draw.text(
                (cx_pos - tw // 2, cy_pos - th // 2),
                fo_text,
                fill="white",
                font=font,
            )

        # Touch target too small
        if css_w < _WCAG_MIN_TARGET or css_h < _WCAG_MIN_TARGET:
            draw.rectangle(
                (x, y, x + w, y + h),
                outline="#FF5722",
                width=3,
            )
            size_label = f"{int(css_w)}x{int(css_h)}"
            bbox = draw.textbbox((0, 0), size_label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.rectangle(
                (x + w + 2, y + h - th - 4, x + w + tw + 6, y + h),
                fill=(255, 87, 34, 220),
            )
            draw.text(
                (x + w + 4, y + h - th - 2),
                size_label,
                fill="white",
                font=font,
            )
            issues.append(
                {
                    "element": eid,
                    "issue": "small-target",
                    "detail": f"{int(css_w)}x{int(css_h)}px (min {_WCAG_MIN_TARGET}x{_WCAG_MIN_TARGET}px)",
                }
            )

        # Missing label
        if not label or label.strip() == "":
            draw.rectangle(
                (x, y, x + w, y + h),
                outline="#E91E63",
                width=2,
            )
            no_label_text = "NO LABEL"
            bbox = draw.textbbox((0, 0), no_label_text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            ly = max(y - th - 4, 0)
            draw.rectangle(
                (x, ly, x + tw + 4, ly + th + 2),
                fill=(233, 30, 99, 220),
            )
            draw.text((x + 2, ly), no_label_text, fill="white", font=font)
            issues.append(
                {
                    "element": eid,
                    "issue": "missing-label",
                    "detail": f"Interactive {el_type} has no accessible label",
                }
            )

    return issues


# ---------------------------------------------------------------------------
# Structured visual description (D2)
# ---------------------------------------------------------------------------


def generate_visual_description(
    elements: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> str:
    """Generate a structured text description of the visual layout.

    Analyzes element positions to detect layout regions, count element types,
    and provide a verbal summary AI can use alongside the screenshot.
    """
    viewport = snapshot.get("viewport", {})
    vp_w = viewport.get("viewportWidth", 1920)
    vp_h = viewport.get("viewportHeight", 1080)
    page = snapshot.get("page", {})

    visible_elements = [
        el
        for el in elements
        if el.get("state", {}).get("visible", True) and el.get("state", {}).get("rect")
    ]

    lines: list[str] = []

    # Page info
    title = page.get("title", "")
    pathname = page.get("pathname", "")
    if title or pathname:
        lines.append(f"Page: {title or pathname}")

    lines.append(f"Viewport: {vp_w}x{vp_h}px")
    lines.append(f"Elements: {len(visible_elements)} visible")

    # Detect layout regions
    regions = _detect_layout_regions(visible_elements, vp_w, vp_h)
    if regions:
        lines.append("")
        lines.append("Layout:")
        for region_name, count in regions.items():
            lines.append(f"  {region_name}: {count} elements")

    # Element type breakdown
    type_counts: dict[str, int] = {}
    for el in visible_elements:
        et = el.get("type", "unknown")
        type_counts[et] = type_counts.get(et, 0) + 1

    if type_counts:
        lines.append("")
        lines.append("Element types:")
        for et, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {et}: {count}")

    # Interactive vs content
    interactive = sum(
        1 for el in visible_elements if el.get("category") == "interactive"
    )
    content = sum(1 for el in visible_elements if el.get("category") == "content")
    lines.append(f"\nInteractive: {interactive}, Content: {content}")

    # Modal status
    modal_stack = snapshot.get("modalStack", {})
    active_modals = modal_stack.get("activeModals", [])
    if active_modals:
        top = active_modals[-1]
        lines.append(f"Modal open: {top.get('title', top.get('id', 'untitled'))}")

    # Toast status
    toasts = snapshot.get("toasts", {})
    active_toasts = toasts.get("activeToasts", [])
    if active_toasts:
        lines.append(f"Active toasts: {len(active_toasts)}")
        for t in active_toasts[:3]:
            severity = t.get("severity", "info")
            text = t.get("text", "")[:60]
            lines.append(f"  [{severity}] {text}")

    # Error status
    errors = snapshot.get("errorSummary", {})
    health = errors.get("health", "healthy")
    if health != "healthy":
        lines.append(f"Health: {health}")
        ec = errors.get("errorCount", 0)
        wc = errors.get("warningCount", 0)
        if ec:
            lines.append(f"  Errors: {ec}")
        if wc:
            lines.append(f"  Warnings: {wc}")

    # Scroll status
    can_down = viewport.get("canScrollDown", False)
    scroll_y = viewport.get("scrollY", 0)
    if can_down or scroll_y > 0:
        doc_h = viewport.get("documentHeight", vp_h)
        pct = int(scroll_y / max(doc_h - vp_h, 1) * 100) if doc_h > vp_h else 0
        lines.append(f"Scroll: {pct}% down ({int(scroll_y)}px)")
        if can_down:
            lines.append("  More content below")

    return "\n".join(lines)


def _detect_layout_regions(
    elements: list[dict[str, Any]],
    vp_w: float,
    vp_h: float,
) -> dict[str, int]:
    """Detect common layout regions from element positions."""
    regions: dict[str, int] = {}

    # Thresholds as fractions of viewport
    header_max_y = vp_h * 0.12
    footer_min_y = vp_h * 0.85
    sidebar_max_x = vp_w * 0.25
    sidebar_min_x = vp_w * 0.75

    for el in elements:
        rect = el.get("state", {}).get("rect", {})
        x = rect.get("x", 0)
        y = rect.get("y", 0)
        w = rect.get("width", 0)

        if y < header_max_y:
            regions["header"] = regions.get("header", 0) + 1
        elif y > footer_min_y:
            regions["footer"] = regions.get("footer", 0) + 1
        elif x + w < sidebar_max_x:
            regions["left-sidebar"] = regions.get("left-sidebar", 0) + 1
        elif x > sidebar_min_x:
            regions["right-sidebar"] = regions.get("right-sidebar", 0) + 1
        else:
            regions["main-content"] = regions.get("main-content", 0) + 1

    return regions


# ---------------------------------------------------------------------------
# Delta encoding / incremental screenshots (D3)
# ---------------------------------------------------------------------------


class DeltaEncoder:
    """Tile-based incremental screenshot encoder.

    Divides screenshots into grid tiles, hashes each tile, and only sends
    tiles that changed since the last capture. Dramatically reduces bandwidth
    for AI agents that take many sequential screenshots.
    """

    def __init__(self, tile_size: int = 64) -> None:
        self.tile_size = tile_size
        self._last_hashes: dict[tuple[int, int], str] = {}
        self._last_full_b64: str | None = None

    def encode_delta(
        self,
        screenshot_b64: str,
    ) -> dict[str, Any]:
        """Compare screenshot against previous and return delta.

        Returns:
            Dict with:
                - full_image: base64 of full image (always provided for first capture)
                - changed_tiles: list of {x, y, w, h, data_b64} for changed tiles
                - total_tiles: total grid tiles
                - changed_count: number of changed tiles
                - change_ratio: fraction of tiles changed
                - is_first: True if this is the first capture (no baseline)
        """
        import hashlib

        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(screenshot_b64)))
        iw, ih = img.size
        ts = self.tile_size

        cols = (iw + ts - 1) // ts
        rows = (ih + ts - 1) // ts
        total_tiles = cols * rows

        new_hashes: dict[tuple[int, int], str] = {}
        changed_tiles: list[dict[str, Any]] = []
        is_first = not self._last_hashes

        for row in range(rows):
            for col in range(cols):
                x0 = col * ts
                y0 = row * ts
                x1 = min(x0 + ts, iw)
                y1 = min(y0 + ts, ih)

                tile = img.crop((x0, y0, x1, y1))
                tile_bytes = tile.tobytes()
                tile_hash = hashlib.md5(tile_bytes).hexdigest()
                new_hashes[(col, row)] = tile_hash

                if tile_hash != self._last_hashes.get((col, row)):
                    buf = io.BytesIO()
                    tile.save(buf, format="PNG")
                    changed_tiles.append(
                        {
                            "x": x0,
                            "y": y0,
                            "w": x1 - x0,
                            "h": y1 - y0,
                            "data_b64": base64.b64encode(buf.getvalue()).decode(),
                        }
                    )

        self._last_hashes = new_hashes
        self._last_full_b64 = screenshot_b64

        return {
            "full_image": screenshot_b64 if is_first else None,
            "changed_tiles": changed_tiles,
            "total_tiles": total_tiles,
            "changed_count": len(changed_tiles),
            "change_ratio": round(len(changed_tiles) / total_tiles, 3)
            if total_tiles
            else 0,
            "is_first": is_first,
            "image_width": iw,
            "image_height": ih,
            "tile_size": ts,
        }

    def reset(self) -> None:
        """Reset state, forcing next capture to be a full image."""
        self._last_hashes.clear()
        self._last_full_b64 = None

    def get_last_full(self) -> str | None:
        """Get last full screenshot base64."""
        return self._last_full_b64


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def _apply_crop(
    img: Any,
    crop_mode: str,
    snapshot: dict[str, Any] | None,
    scale_x: float,
    scale_y: float,
    modal_rect: dict[str, float] | None = None,
) -> Any:
    """Crop the image based on the requested crop mode."""
    if crop_mode == "full":
        return img

    if crop_mode == "viewport" and snapshot:
        viewport = snapshot.get("viewport", {})
        if viewport:
            vw = int(viewport.get("viewportWidth", 0) * scale_x)
            vh = int(viewport.get("viewportHeight", 0) * scale_y)
            sx = int(viewport.get("scrollX", 0) * scale_x)
            sy = int(viewport.get("scrollY", 0) * scale_y)
            if vw > 0 and vh > 0:
                # Clamp to image bounds
                right = min(sx + vw, img.width)
                bottom = min(sy + vh, img.height)
                return img.crop((sx, sy, right, bottom))

    if crop_mode == "modal" and modal_rect:
        padding = 20  # pixels of padding around modal
        mx = int(modal_rect.get("x", 0) * scale_x) - padding
        my = int(modal_rect.get("y", 0) * scale_y) - padding
        mw = int(modal_rect.get("width", 0) * scale_x)
        mh = int(modal_rect.get("height", 0) * scale_y)
        # Clamp
        left = max(0, mx)
        top = max(0, my)
        right = min(mx + mw + 2 * padding, img.width)
        bottom = min(my + mh + 2 * padding, img.height)
        if right > left and bottom > top:
            return img.crop((left, top, right, bottom))

    return img


def crop_to_element(
    screenshot_b64: str,
    element: dict[str, Any],
    width: int,
    height: int,
    padding: int = 16,
    scale: float = 1.0,
    fmt: str = "png",
    quality: int = 85,
) -> str | None:
    """Crop a screenshot to a specific element's bounding rect.

    Args:
        screenshot_b64: Base64 encoded screenshot.
        element: Element dict with state.rect.
        width: CSS viewport width.
        height: CSS viewport height.
        padding: Pixels of padding around the element.
        scale: Output scale factor.
        fmt: Output format.
        quality: JPEG/WebP quality.

    Returns:
        Base64 encoded cropped image, or None if element has no rect.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed.")
        return None

    state = element.get("state", {})
    rect = state.get("rect")
    if not rect:
        return None

    img = Image.open(io.BytesIO(base64.b64decode(screenshot_b64)))
    scale_x = img.width / width if width else 1.0
    scale_y = img.height / height if height else 1.0

    x = int(rect.get("x", 0) * scale_x) - padding
    y = int(rect.get("y", 0) * scale_y) - padding
    w = int(rect.get("width", 0) * scale_x)
    h = int(rect.get("height", 0) * scale_y)

    left = max(0, x)
    top = max(0, y)
    right = min(x + w + 2 * padding, img.width)
    bottom = min(y + h + 2 * padding, img.height)

    if right <= left or bottom <= top:
        return None

    cropped = img.crop((left, top, right, bottom))

    if scale != 1.0:
        new_w = max(1, int(cropped.width * scale))
        new_h = max(1, int(cropped.height * scale))
        cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return _encode_image(cropped, fmt, quality)


# ---------------------------------------------------------------------------
# Visual Diffing
# ---------------------------------------------------------------------------


class BaselineStore:
    """In-memory store for screenshot baselines and before/after pairs."""

    def __init__(self) -> None:
        self._baselines: dict[str, bytes] = {}
        self._before: bytes | None = None

    def save_baseline(self, key: str, png_bytes: bytes) -> None:
        """Save a baseline screenshot for a route/page key."""
        self._baselines[key] = png_bytes

    def get_baseline(self, key: str) -> bytes | None:
        """Get baseline for a key, or None."""
        return self._baselines.get(key)

    def list_baselines(self) -> list[str]:
        """List all baseline keys."""
        return list(self._baselines.keys())

    def delete_baseline(self, key: str) -> bool:
        """Delete a baseline. Returns True if it existed."""
        return self._baselines.pop(key, None) is not None

    def save_before(self, png_bytes: bytes) -> None:
        """Save a 'before' screenshot for before/after comparison."""
        self._before = png_bytes

    def get_before(self) -> bytes | None:
        """Get the 'before' screenshot."""
        return self._before

    def clear_before(self) -> None:
        """Clear the before screenshot."""
        self._before = None


def diff_screenshots(
    img1_b64: str,
    img2_b64: str,
    threshold: float = 0.01,
    pixel_tolerance: int = 30,
) -> DiffResult:
    """Compare two screenshots and produce a visual diff.

    Args:
        img1_b64: Baseline screenshot (base64).
        img2_b64: Current screenshot (base64).
        threshold: Maximum allowed change percentage (0.0-1.0) for pass.
        pixel_tolerance: Per-channel pixel difference threshold (0-255).

    Returns:
        DiffResult with diff image, change percentage, and pass/fail.
    """
    from PIL import Image, ImageChops, ImageDraw

    img1 = Image.open(io.BytesIO(base64.b64decode(img1_b64))).convert("RGB")
    img2 = Image.open(io.BytesIO(base64.b64decode(img2_b64))).convert("RGB")

    # Resize to match if needed
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

    # Pixel-level difference
    diff = ImageChops.difference(img1, img2)

    # Create binary mask of changed pixels (above tolerance)
    total_pixels = img1.width * img1.height
    changed_pixels = 0
    diff_data = diff.load()
    mask = Image.new("L", img1.size, 0)
    mask_data = mask.load()

    for y_px in range(img1.height):
        for x_px in range(img1.width):
            r, g, b = diff_data[x_px, y_px]  # type: ignore[index,misc]
            if max(r, g, b) > pixel_tolerance:
                changed_pixels += 1
                mask_data[x_px, y_px] = 255  # type: ignore[index]

    change_pct = changed_pixels / total_pixels if total_pixels > 0 else 0.0

    # Find changed regions (connected component bounding boxes via grid scan)
    changed_regions = _find_changed_regions(mask, grid_size=32)

    # Create diff visualization: original with red overlay on changed areas
    result_img = img2.copy().convert("RGBA")
    overlay = Image.new("RGBA", result_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Red semi-transparent overlay on changed pixels
    for region in changed_regions:
        rx, ry, rw, rh = region["x"], region["y"], region["width"], region["height"]
        overlay_draw.rectangle(
            (rx, ry, rx + rw, ry + rh),
            fill=(255, 0, 0, 80),
            outline=(255, 0, 0, 200),
            width=2,
        )

    result_img = Image.alpha_composite(result_img, overlay)

    diff_b64 = _encode_image(result_img, "png", 85)

    return DiffResult(
        diff_image_b64=diff_b64,
        change_percentage=round(change_pct * 100, 2),
        changed_regions=changed_regions,
        passed=change_pct <= threshold,
        threshold_used=threshold,
    )


def _find_changed_regions(mask: Any, grid_size: int = 32) -> list[dict[str, int]]:
    """Find bounding boxes of changed regions using a grid scan."""
    mask_data = mask.load()
    w, h = mask.size
    regions: list[dict[str, int]] = []

    # Scan grid cells and merge adjacent changed cells
    cells_x = (w + grid_size - 1) // grid_size
    cells_y = (h + grid_size - 1) // grid_size
    changed_cells: set[tuple[int, int]] = set()

    for cy in range(cells_y):
        for cx in range(cells_x):
            # Check if any pixel in this cell is changed
            x0 = cx * grid_size
            y0 = cy * grid_size
            x1 = min(x0 + grid_size, w)
            y1 = min(y0 + grid_size, h)
            has_change = False
            for py in range(y0, y1):
                for px in range(x0, x1):
                    if mask_data[px, py] > 0:
                        has_change = True
                        break
                if has_change:
                    break
            if has_change:
                changed_cells.add((cx, cy))

    # Merge adjacent cells into rectangular regions (simple greedy)
    visited: set[tuple[int, int]] = set()
    for cell in sorted(changed_cells):
        if cell in visited:
            continue
        # BFS to find connected component
        component: set[tuple[int, int]] = set()
        queue = [cell]
        while queue:
            c = queue.pop()
            if c in visited or c not in changed_cells:
                continue
            visited.add(c)
            component.add(c)
            cx, cy = c
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nc = (cx + dx, cy + dy)
                if nc in changed_cells and nc not in visited:
                    queue.append(nc)

        if component:
            min_cx = min(c[0] for c in component)
            min_cy = min(c[1] for c in component)
            max_cx = max(c[0] for c in component)
            max_cy = max(c[1] for c in component)
            regions.append(
                {
                    "x": min_cx * grid_size,
                    "y": min_cy * grid_size,
                    "width": (max_cx - min_cx + 1) * grid_size,
                    "height": (max_cy - min_cy + 1) * grid_size,
                }
            )

    return regions


def create_before_after(
    before_b64: str,
    after_b64: str,
    layout: str = "side-by-side",
) -> str:
    """Create a before/after comparison image.

    Args:
        before_b64: Before screenshot (base64).
        after_b64: After screenshot (base64).
        layout: "side-by-side" or "vertical".

    Returns:
        Base64 encoded comparison PNG.
    """
    from PIL import Image, ImageDraw

    img_before = Image.open(io.BytesIO(base64.b64decode(before_b64))).convert("RGB")
    img_after = Image.open(io.BytesIO(base64.b64decode(after_b64))).convert("RGB")

    # Normalize sizes
    if img_before.size != img_after.size:
        img_after = img_after.resize(img_before.size, Image.Resampling.LANCZOS)

    font = _load_font(16)
    label_height = 28
    divider = 4

    if layout == "vertical":
        total_w = img_before.width
        total_h = img_before.height * 2 + label_height * 2 + divider
        result = Image.new("RGB", (total_w, total_h), (40, 40, 40))
        draw = ImageDraw.Draw(result)

        # Before label
        draw.rectangle((0, 0, total_w, label_height), fill=(33, 33, 33))
        draw.text((8, 4), "BEFORE", fill="white", font=font)
        result.paste(img_before, (0, label_height))

        # Divider
        y_div = label_height + img_before.height
        draw.rectangle((0, y_div, total_w, y_div + divider), fill=(80, 80, 80))

        # After label
        y_after_label = y_div + divider
        draw.rectangle(
            (0, y_after_label, total_w, y_after_label + label_height),
            fill=(33, 33, 33),
        )
        draw.text((8, y_after_label + 4), "AFTER", fill="white", font=font)
        result.paste(img_after, (0, y_after_label + label_height))
    else:
        # Side by side
        total_w = img_before.width * 2 + divider
        total_h = img_before.height + label_height
        result = Image.new("RGB", (total_w, total_h), (40, 40, 40))
        draw = ImageDraw.Draw(result)

        # Before label
        draw.rectangle((0, 0, img_before.width, label_height), fill=(33, 33, 33))
        draw.text((8, 4), "BEFORE", fill="white", font=font)
        result.paste(img_before, (0, label_height))

        # Divider
        draw.rectangle(
            (img_before.width, 0, img_before.width + divider, total_h),
            fill=(80, 80, 80),
        )

        # After label
        x_after = img_before.width + divider
        draw.rectangle((x_after, 0, total_w, label_height), fill=(33, 33, 33))
        draw.text((x_after + 8, 4), "AFTER", fill="white", font=font)
        result.paste(img_after, (x_after, label_height))

    return _encode_image(result, "png", 85)


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------


def _encode_image(img: Any, fmt: str, quality: int) -> str:
    """Encode a PIL Image to base64 in the specified format."""
    buf = io.BytesIO()
    if fmt == "jpeg":
        # JPEG doesn't support alpha
        if img.mode in ("RGBA", "LA", "PA"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality)
    elif fmt == "webp":
        img.save(buf, format="WEBP", quality=quality)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mime_type_for_format(fmt: str) -> str:
    """Get MIME type for an image format string."""
    return {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(fmt, "image/png")
