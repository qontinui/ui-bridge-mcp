"""Tests for screenshot annotation, diffing, and visual analysis utilities."""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from ui_bridge_mcp.screenshot import (
    AnnotationOptions,
    BaselineStore,
    DeltaEncoder,
    _detect_layout_regions,
    _element_style,
    _extract_modal_context,
    _parse_border_width,
    _parse_px,
    _should_annotate,
    annotate_screenshot,
    create_before_after,
    crop_to_element,
    draw_accessibility_overlay,
    generate_visual_description,
    diff_screenshots,
    mime_type_for_format,
)


# =============================================================================
# Helpers
# =============================================================================


class FakeRefManager:
    """Minimal RefManager for testing."""

    def __init__(self) -> None:
        self._counter = 0
        self._id_to_ref: dict[str, str] = {}

    def assign(self, element_id: str) -> str:
        if element_id in self._id_to_ref:
            return self._id_to_ref[element_id]
        self._counter += 1
        ref = f"@e{self._counter}"
        self._id_to_ref[element_id] = ref
        return ref


def _make_screenshot(width: int = 200, height: int = 150, color: str = "white") -> str:
    """Create a test screenshot as base64 PNG."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_element(
    id: str,
    type: str = "button",
    category: str = "interactive",
    x: float = 10,
    y: float = 10,
    w: float = 80,
    h: float = 30,
    visible: bool = True,
    enabled: bool = True,
    focused: bool = False,
    validation_valid: bool | None = None,
    required: bool = False,
    in_viewport: bool = True,
) -> dict[str, Any]:
    """Build a mock element dict."""
    state: dict[str, Any] = {
        "visible": visible,
        "enabled": enabled,
        "focused": focused,
        "rect": {"x": x, "y": y, "width": w, "height": h},
        "inViewport": in_viewport,
    }
    if validation_valid is not None:
        state["validationState"] = {"valid": validation_valid}
    if required:
        state["required"] = True
    return {
        "id": id,
        "type": type,
        "category": category,
        "state": state,
    }


def _decode_b64_image(b64: str) -> Image.Image:
    """Decode a base64 image string to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# =============================================================================
# annotate_screenshot tests
# =============================================================================


class TestAnnotateScreenshot:
    def test_default_mode_annotates_interactive_only(self) -> None:
        screenshot = _make_screenshot()
        elements = [
            _make_element("btn-1", type="button", category="interactive"),
            _make_element("text-1", type="p", category="content", x=10, y=60),
        ]
        rm = FakeRefManager()
        result = annotate_screenshot(screenshot, elements, 200, 150, rm)
        assert result != screenshot  # modified
        img = _decode_b64_image(result)
        assert img.size == (200, 150)
        # Only btn-1 should have been assigned a ref
        assert "btn-1" in rm._id_to_ref
        assert "text-1" not in rm._id_to_ref

    def test_all_mode_annotates_everything(self) -> None:
        screenshot = _make_screenshot()
        elements = [
            _make_element("btn-1", type="button", category="interactive"),
            _make_element("text-1", type="p", category="content", x=10, y=60),
        ]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="all")
        annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        assert "btn-1" in rm._id_to_ref
        assert "text-1" in rm._id_to_ref

    def test_validation_mode_only_annotates_validated(self) -> None:
        screenshot = _make_screenshot()
        elements = [
            _make_element("input-email", type="input", validation_valid=False),
            _make_element(
                "input-name", type="input", validation_valid=True, x=10, y=60
            ),
            _make_element("btn-submit", type="button", x=10, y=100),
        ]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="validation")
        annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        assert "input-email" in rm._id_to_ref
        assert "input-name" in rm._id_to_ref
        # btn-submit has no validationState, not required → excluded
        assert "btn-submit" not in rm._id_to_ref

    def test_validation_mode_includes_required(self) -> None:
        screenshot = _make_screenshot()
        elements = [
            _make_element("input-required", type="input", required=True),
        ]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="validation")
        annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        assert "input-required" in rm._id_to_ref

    def test_state_mode_annotates_all(self) -> None:
        screenshot = _make_screenshot()
        elements = [
            _make_element("btn-1", type="button", focused=True),
            _make_element("text-1", type="p", category="content", x=10, y=60),
        ]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="state")
        annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        assert "btn-1" in rm._id_to_ref
        assert "text-1" in rm._id_to_ref

    def test_highlight_elements_overrides_mode(self) -> None:
        screenshot = _make_screenshot()
        elements = [
            _make_element("btn-1", type="button"),
            _make_element("btn-2", type="button", x=10, y=60),
            _make_element("btn-3", type="button", x=10, y=100),
        ]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="all", highlight_elements=["btn-2"])
        annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        # Only btn-2 highlighted
        assert "btn-1" not in rm._id_to_ref
        assert "btn-2" in rm._id_to_ref
        assert "btn-3" not in rm._id_to_ref

    def test_invisible_elements_skipped(self) -> None:
        screenshot = _make_screenshot()
        elements = [
            _make_element("btn-1", visible=False),
        ]
        rm = FakeRefManager()
        annotate_screenshot(screenshot, elements, 200, 150, rm)
        assert "btn-1" not in rm._id_to_ref

    def test_no_rect_skipped(self) -> None:
        screenshot = _make_screenshot()
        el = _make_element("btn-1")
        del el["state"]["rect"]
        rm = FakeRefManager()
        annotate_screenshot(screenshot, [el], 200, 150, rm)
        assert "btn-1" not in rm._id_to_ref

    def test_scale_changes_dimensions(self) -> None:
        screenshot = _make_screenshot(200, 150)
        elements = [_make_element("btn-1")]
        rm = FakeRefManager()
        opts = AnnotationOptions(scale=0.5)
        result = annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        img = _decode_b64_image(result)
        assert img.size == (100, 75)

    def test_jpeg_format(self) -> None:
        screenshot = _make_screenshot()
        elements = [_make_element("btn-1")]
        rm = FakeRefManager()
        opts = AnnotationOptions(format="jpeg")
        result = annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        img = _decode_b64_image(result)
        assert img.mode == "RGB"

    def test_webp_format(self) -> None:
        screenshot = _make_screenshot()
        elements = [_make_element("btn-1")]
        rm = FakeRefManager()
        opts = AnnotationOptions(format="webp")
        result = annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        # Should decode successfully
        img = _decode_b64_image(result)
        assert img.width > 0

    def test_modal_mode_without_snapshot_falls_back(self) -> None:
        """Modal mode without snapshot data should still annotate interactives."""
        screenshot = _make_screenshot()
        elements = [_make_element("btn-1")]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="modal")
        # No snapshot → falls back to interactive
        annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        assert "btn-1" in rm._id_to_ref

    def test_viewport_crop(self) -> None:
        screenshot = _make_screenshot(400, 300)
        elements = [_make_element("btn-1")]
        rm = FakeRefManager()
        opts = AnnotationOptions(crop="viewport")
        snapshot = {
            "viewport": {
                "viewportWidth": 200,
                "viewportHeight": 150,
                "scrollX": 0,
                "scrollY": 0,
            }
        }
        result = annotate_screenshot(
            screenshot, elements, 400, 300, rm, options=opts, snapshot=snapshot
        )
        img = _decode_b64_image(result)
        assert img.size == (200, 150)


# =============================================================================
# crop_to_element tests
# =============================================================================


class TestCropToElement:
    def test_basic_crop(self) -> None:
        screenshot = _make_screenshot(200, 150)
        el = _make_element("btn-1", x=50, y=40, w=60, h=30)
        result = crop_to_element(screenshot, el, 200, 150, padding=10)
        assert result is not None
        img = _decode_b64_image(result)
        assert img.width == 80  # 60 + 2*10
        assert img.height == 50  # 30 + 2*10

    def test_no_rect_returns_none(self) -> None:
        screenshot = _make_screenshot()
        el = _make_element("btn-1")
        del el["state"]["rect"]
        result = crop_to_element(screenshot, el, 200, 150)
        assert result is None

    def test_scale_factor(self) -> None:
        screenshot = _make_screenshot(200, 150)
        el = _make_element("btn-1", x=50, y=40, w=60, h=30)
        result = crop_to_element(screenshot, el, 200, 150, padding=0, scale=2.0)
        assert result is not None
        img = _decode_b64_image(result)
        assert img.width == 120  # 60 * 2
        assert img.height == 60  # 30 * 2

    def test_clamps_to_image_bounds(self) -> None:
        screenshot = _make_screenshot(100, 80)
        el = _make_element("btn-1", x=80, y=60, w=40, h=30)
        result = crop_to_element(screenshot, el, 100, 80, padding=10)
        assert result is not None
        img = _decode_b64_image(result)
        # Right edge clamped to 100, bottom edge clamped to 80
        assert img.width <= 100
        assert img.height <= 80


# =============================================================================
# BaselineStore tests
# =============================================================================


class TestBaselineStore:
    def test_save_and_get_baseline(self) -> None:
        store = BaselineStore()
        store.save_baseline("dashboard", b"fake-png-data")
        assert store.get_baseline("dashboard") == b"fake-png-data"

    def test_missing_baseline_returns_none(self) -> None:
        store = BaselineStore()
        assert store.get_baseline("nonexistent") is None

    def test_list_baselines(self) -> None:
        store = BaselineStore()
        store.save_baseline("page-a", b"a")
        store.save_baseline("page-b", b"b")
        keys = store.list_baselines()
        assert sorted(keys) == ["page-a", "page-b"]

    def test_delete_baseline(self) -> None:
        store = BaselineStore()
        store.save_baseline("page-a", b"a")
        assert store.delete_baseline("page-a") is True
        assert store.get_baseline("page-a") is None
        assert store.delete_baseline("page-a") is False

    def test_before_after_lifecycle(self) -> None:
        store = BaselineStore()
        assert store.get_before() is None
        store.save_before(b"before-data")
        assert store.get_before() == b"before-data"
        store.clear_before()
        assert store.get_before() is None


# =============================================================================
# diff_screenshots tests
# =============================================================================


class TestDiffScreenshots:
    def test_identical_images_no_changes(self) -> None:
        img = _make_screenshot(100, 80, "white")
        result = diff_screenshots(img, img)
        assert result.change_percentage == 0.0
        assert result.passed is True
        assert len(result.changed_regions) == 0

    def test_different_images_detects_change(self) -> None:
        img1 = _make_screenshot(100, 80, "white")
        img2 = _make_screenshot(100, 80, "red")
        result = diff_screenshots(img1, img2)
        assert result.change_percentage > 0
        assert result.passed is False  # default threshold 0.01 = 1%
        assert len(result.changed_regions) > 0

    def test_threshold_controls_pass_fail(self) -> None:
        img1 = _make_screenshot(100, 80, "white")
        img2 = _make_screenshot(100, 80, "red")
        # Very permissive threshold
        result = diff_screenshots(img1, img2, threshold=1.0)
        assert result.passed is True
        assert result.threshold_used == 1.0

    def test_diff_result_has_image(self) -> None:
        img1 = _make_screenshot(100, 80, "white")
        img2 = _make_screenshot(100, 80, "blue")
        result = diff_screenshots(img1, img2)
        # Should produce a valid image
        diff_img = _decode_b64_image(result.diff_image_b64)
        assert diff_img.width == 100
        assert diff_img.height == 80

    def test_different_sizes_resized(self) -> None:
        img1 = _make_screenshot(100, 80, "white")
        img2 = _make_screenshot(200, 160, "white")
        # Should not crash — img2 gets resized to match img1
        result = diff_screenshots(img1, img2)
        assert result.change_percentage < 5  # minor interpolation artifacts

    def test_partial_change(self) -> None:
        """Only part of the image changes."""
        img = Image.new("RGB", (100, 100), "white")
        buf1 = io.BytesIO()
        img.save(buf1, format="PNG")
        b64_1 = base64.b64encode(buf1.getvalue()).decode()

        # Draw a red square in one corner
        img2 = Image.new("RGB", (100, 100), "white")
        for x in range(50):
            for y in range(50):
                img2.putpixel((x, y), (255, 0, 0))
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG")
        b64_2 = base64.b64encode(buf2.getvalue()).decode()

        result = diff_screenshots(b64_1, b64_2)
        # ~25% of pixels changed (50x50 out of 100x100)
        assert 20 < result.change_percentage < 30
        assert len(result.changed_regions) >= 1


# =============================================================================
# create_before_after tests
# =============================================================================


class TestCreateBeforeAfter:
    def test_side_by_side(self) -> None:
        before = _make_screenshot(100, 80, "white")
        after = _make_screenshot(100, 80, "blue")
        result = create_before_after(before, after, layout="side-by-side")
        img = _decode_b64_image(result)
        # Width should be approximately 2x + divider, height should be image + label
        assert img.width > 200
        assert img.height > 80

    def test_vertical_layout(self) -> None:
        before = _make_screenshot(100, 80, "white")
        after = _make_screenshot(100, 80, "blue")
        result = create_before_after(before, after, layout="vertical")
        img = _decode_b64_image(result)
        # Width should be ~100, height should be ~2x + labels + divider
        assert img.width == 100
        assert img.height > 160

    def test_different_sizes_normalized(self) -> None:
        before = _make_screenshot(100, 80, "white")
        after = _make_screenshot(200, 160, "blue")
        result = create_before_after(before, after)
        # Should not crash
        img = _decode_b64_image(result)
        assert img.width > 0


# =============================================================================
# mime_type_for_format tests
# =============================================================================


class TestMimeType:
    def test_png(self) -> None:
        assert mime_type_for_format("png") == "image/png"

    def test_jpeg(self) -> None:
        assert mime_type_for_format("jpeg") == "image/jpeg"

    def test_webp(self) -> None:
        assert mime_type_for_format("webp") == "image/webp"

    def test_unknown_defaults_to_png(self) -> None:
        assert mime_type_for_format("bmp") == "image/png"


# =============================================================================
# _element_style tests — verify state → color mappings
# =============================================================================


class TestElementStyle:
    def test_default_interactive(self) -> None:
        el = _make_element("btn", type="button")
        outline, fill, text, width = _element_style(el, "interactive")
        assert outline == "#F44336"  # red
        assert width == 2

    def test_disabled_state(self) -> None:
        el = _make_element("btn", enabled=False)
        outline, _, _, width = _element_style(el, "interactive")
        assert outline == "#9E9E9E"  # gray
        assert width == 1

    def test_focused_state(self) -> None:
        el = _make_element("btn", focused=True)
        outline, _, _, width = _element_style(el, "interactive")
        assert outline == "#2196F3"  # blue
        assert width == 3

    def test_content_category(self) -> None:
        el = _make_element("txt", category="content")
        outline, _, _, width = _element_style(el, "interactive")
        assert outline == "#78909C"  # blue-gray
        assert width == 1

    def test_state_mode_error_highest_priority(self) -> None:
        """Error takes priority over focused in state mode."""
        el = _make_element("inp", focused=True, validation_valid=False)
        outline, _, _, width = _element_style(el, "state")
        assert outline == "#D32F2F"  # dark red (error)
        assert width == 3

    def test_state_mode_focused_over_disabled(self) -> None:
        """Focused > disabled in state mode (but disabled+focused is unusual)."""
        el = _make_element("inp", focused=True, enabled=True)
        outline, _, _, _ = _element_style(el, "state")
        assert outline == "#2196F3"

    def test_state_mode_valid(self) -> None:
        el = _make_element("inp", validation_valid=True)
        outline, _, _, _ = _element_style(el, "state")
        assert outline == "#4CAF50"  # green

    def test_state_mode_offscreen(self) -> None:
        el = _make_element("btn", in_viewport=False)
        outline, _, _, _ = _element_style(el, "state")
        assert outline == "#FFD600"  # yellow

    def test_validation_mode_error(self) -> None:
        el = _make_element("inp", validation_valid=False)
        outline, _, _, _ = _element_style(el, "validation")
        assert outline == "#D32F2F"

    def test_validation_mode_valid(self) -> None:
        el = _make_element("inp", validation_valid=True)
        outline, _, _, _ = _element_style(el, "validation")
        assert outline == "#4CAF50"

    def test_validation_mode_required_no_state(self) -> None:
        el = _make_element("inp", required=True)
        outline, _, _, _ = _element_style(el, "validation")
        assert outline == "#FF9800"  # orange warning


# =============================================================================
# _should_annotate tests — all modes
# =============================================================================


class TestShouldAnnotate:
    def test_interactive_includes_button(self) -> None:
        el = _make_element("btn", type="button", category="interactive")
        assert _should_annotate(el, "interactive", None, None) is True

    def test_interactive_excludes_content(self) -> None:
        el = _make_element("txt", type="p", category="content")
        assert _should_annotate(el, "interactive", None, None) is False

    def test_interactive_includes_link_type(self) -> None:
        el = _make_element("lnk", type="link", category="content")
        assert _should_annotate(el, "interactive", None, None) is True

    def test_all_includes_everything(self) -> None:
        el = _make_element("txt", type="p", category="content")
        assert _should_annotate(el, "all", None, None) is True

    def test_validation_includes_validated(self) -> None:
        el = _make_element("inp", validation_valid=False)
        assert _should_annotate(el, "validation", None, None) is True

    def test_validation_includes_required(self) -> None:
        el = _make_element("inp", required=True)
        assert _should_annotate(el, "validation", None, None) is True

    def test_validation_excludes_no_state(self) -> None:
        el = _make_element("btn", type="button")
        assert _should_annotate(el, "validation", None, None) is False

    def test_state_includes_everything(self) -> None:
        el = _make_element("txt", type="p", category="content")
        assert _should_annotate(el, "state", None, None) is True

    def test_modal_with_ids_includes_modal_element(self) -> None:
        el = _make_element("modal-btn")
        assert _should_annotate(el, "modal", None, {"modal-btn"}) is True

    def test_modal_with_ids_excludes_outside(self) -> None:
        el = _make_element("page-btn")
        assert _should_annotate(el, "modal", None, {"modal-btn"}) is False

    def test_modal_without_ids_falls_back_to_interactive(self) -> None:
        el = _make_element("btn", type="button", category="interactive")
        assert _should_annotate(el, "modal", None, None) is True

    def test_highlight_overrides_mode(self) -> None:
        el = _make_element("btn-a")
        assert _should_annotate(el, "all", {"btn-a"}, None) is True
        assert _should_annotate(el, "all", {"btn-b"}, None) is False

    def test_invisible_always_excluded(self) -> None:
        el = _make_element("btn", visible=False)
        assert _should_annotate(el, "all", None, None) is False

    def test_no_rect_excluded(self) -> None:
        el = _make_element("btn")
        del el["state"]["rect"]
        assert _should_annotate(el, "all", None, None) is False


# =============================================================================
# _extract_modal_context tests
# =============================================================================


class TestExtractModalContext:
    def test_no_modal_stack(self) -> None:
        snapshot: dict[str, Any] = {}
        ids, rect = _extract_modal_context(snapshot, [])
        assert ids is None
        assert rect is None

    def test_empty_active_modals(self) -> None:
        snapshot: dict[str, Any] = {"modalStack": {"activeModals": []}}
        ids, rect = _extract_modal_context(snapshot, [])
        assert ids is None
        assert rect is None

    def test_modal_element_not_in_elements(self) -> None:
        snapshot = {"modalStack": {"activeModals": [{"id": "dlg-1"}]}}
        elements = [_make_element("btn-1")]
        ids, rect = _extract_modal_context(snapshot, elements)
        # Modal element has no rect → returns (None, None)
        assert ids is None
        assert rect is None

    def test_modal_with_rect_collects_elements(self) -> None:
        snapshot = {"modalStack": {"activeModals": [{"id": "dlg-1"}]}}
        elements = [
            _make_element("dlg-1", x=100, y=100, w=200, h=200),
            _make_element("btn-inside", x=150, y=150, w=50, h=30),
            _make_element("btn-outside", x=10, y=10, w=50, h=30),
        ]
        ids, rect = _extract_modal_context(snapshot, elements)
        assert rect is not None
        assert rect["x"] == 100
        assert ids is not None
        assert "dlg-1" in ids
        assert "btn-inside" in ids
        assert "btn-outside" not in ids

    def test_uses_topmost_modal(self) -> None:
        """Multiple modals — should use the last (topmost) one."""
        snapshot = {
            "modalStack": {
                "activeModals": [
                    {"id": "dlg-bottom"},
                    {"id": "dlg-top"},
                ]
            }
        }
        elements = [
            _make_element("dlg-top", x=200, y=200, w=100, h=100),
            _make_element("dlg-bottom", x=0, y=0, w=400, h=400),
        ]
        ids, rect = _extract_modal_context(snapshot, elements)
        assert rect is not None
        assert rect["x"] == 200  # topmost modal's position


# =============================================================================
# Modal integration with annotate_screenshot
# =============================================================================


class TestModalAnnotation:
    def test_modal_mode_with_snapshot_annotates_modal_elements(self) -> None:
        screenshot = _make_screenshot(400, 300)
        snapshot = {"modalStack": {"activeModals": [{"id": "dlg-1"}]}}
        elements = [
            _make_element("dlg-1", x=100, y=50, w=200, h=200),
            _make_element("modal-btn", x=150, y=120, w=80, h=30),
            _make_element("bg-btn", x=10, y=10, w=50, h=30),
        ]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="modal")
        annotate_screenshot(
            screenshot, elements, 400, 300, rm, options=opts, snapshot=snapshot
        )
        # Elements inside modal should be annotated
        assert "dlg-1" in rm._id_to_ref
        assert "modal-btn" in rm._id_to_ref
        # Element outside modal should NOT be annotated
        assert "bg-btn" not in rm._id_to_ref

    def test_modal_crop_crops_to_modal(self) -> None:
        screenshot = _make_screenshot(400, 300)
        snapshot = {"modalStack": {"activeModals": [{"id": "dlg-1"}]}}
        elements = [
            _make_element("dlg-1", x=100, y=50, w=200, h=150),
        ]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="modal", crop="modal")
        result = annotate_screenshot(
            screenshot, elements, 400, 300, rm, options=opts, snapshot=snapshot
        )
        img = _decode_b64_image(result)
        # Cropped to modal area + padding (20px each side)
        assert img.width < 400
        assert img.height < 300
        # Should be approximately 200 + 40 = 240 wide
        assert 230 <= img.width <= 250


# =============================================================================
# Empty / edge case tests
# =============================================================================


class TestEdgeCases:
    def test_empty_elements_list(self) -> None:
        screenshot = _make_screenshot()
        rm = FakeRefManager()
        result = annotate_screenshot(screenshot, [], 200, 150, rm)
        # Should return valid image unchanged (except format re-encoding)
        img = _decode_b64_image(result)
        assert img.size == (200, 150)

    def test_zero_width_height(self) -> None:
        screenshot = _make_screenshot()
        rm = FakeRefManager()
        # Should not crash — scale defaults to 1.0
        result = annotate_screenshot(screenshot, [], 0, 0, rm)
        img = _decode_b64_image(result)
        assert img.size == (200, 150)

    def test_element_with_zero_dimensions(self) -> None:
        screenshot = _make_screenshot()
        el = _make_element("btn", w=0, h=0)
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="all")
        # Should not crash
        annotate_screenshot(screenshot, [el], 200, 150, rm, options=opts)

    def test_viewport_crop_without_viewport_data(self) -> None:
        screenshot = _make_screenshot(200, 150)
        rm = FakeRefManager()
        opts = AnnotationOptions(crop="viewport")
        # No viewport in snapshot → returns full image
        result = annotate_screenshot(
            screenshot, [], 200, 150, rm, options=opts, snapshot={}
        )
        img = _decode_b64_image(result)
        assert img.size == (200, 150)

    def test_viewport_crop_with_scroll(self) -> None:
        screenshot = _make_screenshot(400, 300)
        rm = FakeRefManager()
        opts = AnnotationOptions(crop="viewport")
        snapshot = {
            "viewport": {
                "viewportWidth": 200,
                "viewportHeight": 150,
                "scrollX": 50,
                "scrollY": 30,
            }
        }
        result = annotate_screenshot(
            screenshot, [], 400, 300, rm, options=opts, snapshot=snapshot
        )
        img = _decode_b64_image(result)
        assert img.size == (200, 150)


# =============================================================================
# Relationship lines (A3) tests
# =============================================================================


class TestRelationshipLines:
    def test_relationships_mode_draws_lines(self) -> None:
        screenshot = _make_screenshot(400, 300)
        elements = [
            _make_element("btn-toggle", x=50, y=50, w=80, h=30),
            _make_element("panel-1", x=200, y=50, w=150, h=100, category="content"),
        ]
        snapshot = {
            "relationships": {
                "relationships": [
                    {
                        "source": "btn-toggle",
                        "target": "panel-1",
                        "type": "controls",
                        "origin": "aria",
                    }
                ],
                "count": 1,
            }
        }
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="relationships")
        result = annotate_screenshot(
            screenshot, elements, 400, 300, rm, options=opts, snapshot=snapshot
        )
        # Should produce a valid image (with lines drawn)
        img = _decode_b64_image(result)
        assert img.size == (400, 300)
        # Both elements should be annotated
        assert "btn-toggle" in rm._id_to_ref
        assert "panel-1" in rm._id_to_ref

    def test_relationships_mode_no_relationships(self) -> None:
        screenshot = _make_screenshot()
        elements = [_make_element("btn-1")]
        snapshot = {"relationships": {"relationships": [], "count": 0}}
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="relationships")
        result = annotate_screenshot(
            screenshot, elements, 200, 150, rm, options=opts, snapshot=snapshot
        )
        img = _decode_b64_image(result)
        assert img.width > 0

    def test_relationships_missing_element_skipped(self) -> None:
        """Relationship with non-existent element should be silently skipped."""
        screenshot = _make_screenshot()
        elements = [_make_element("btn-1")]
        snapshot = {
            "relationships": {
                "relationships": [
                    {
                        "source": "btn-1",
                        "target": "nonexistent",
                        "type": "controls",
                    }
                ],
                "count": 1,
            }
        }
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="relationships")
        # Should not crash
        annotate_screenshot(
            screenshot, elements, 200, 150, rm, options=opts, snapshot=snapshot
        )


# =============================================================================
# Viewport indicators (A4) tests
# =============================================================================


class TestViewportIndicators:
    def test_indicators_drawn_with_viewport_data(self) -> None:
        screenshot = _make_screenshot(400, 300)
        elements = [
            _make_element("btn-1", in_viewport=False),
            _make_element("btn-2", in_viewport=True, x=50, y=50),
        ]
        snapshot = {
            "viewport": {
                "viewportWidth": 400,
                "viewportHeight": 300,
                "scrollX": 0,
                "scrollY": 100,
                "documentWidth": 400,
                "documentHeight": 800,
                "canScrollDown": True,
                "canScrollRight": False,
            }
        }
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="interactive")
        result = annotate_screenshot(
            screenshot, elements, 400, 300, rm, options=opts, snapshot=snapshot
        )
        img = _decode_b64_image(result)
        assert img.size == (400, 300)

    def test_no_viewport_no_crash(self) -> None:
        screenshot = _make_screenshot()
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="interactive")
        result = annotate_screenshot(
            screenshot, [], 200, 150, rm, options=opts, snapshot={}
        )
        img = _decode_b64_image(result)
        assert img.width > 0


# =============================================================================
# Box model overlay (C1) tests
# =============================================================================


class TestBoxModelOverlay:
    def test_boxmodel_mode_with_design_data(self) -> None:
        screenshot = _make_screenshot(400, 300)
        elements = [
            _make_element("btn-1", x=50, y=50, w=120, h=40),
        ]
        design_data = [
            {
                "elementId": "btn-1",
                "styles": {
                    "marginTop": "8px",
                    "marginRight": "0px",
                    "marginBottom": "8px",
                    "marginLeft": "0px",
                    "paddingTop": "12px",
                    "paddingRight": "16px",
                    "paddingBottom": "12px",
                    "paddingLeft": "16px",
                    "border": "1px solid rgb(0,0,0)",
                    "fontSize": "14px",
                },
            }
        ]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="boxmodel")
        result = annotate_screenshot(
            screenshot,
            elements,
            400,
            300,
            rm,
            options=opts,
            design_data=design_data,
        )
        img = _decode_b64_image(result)
        assert img.size == (400, 300)

    def test_boxmodel_without_design_data(self) -> None:
        screenshot = _make_screenshot()
        elements = [_make_element("btn-1")]
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="boxmodel")
        # No design data — should not crash
        result = annotate_screenshot(screenshot, elements, 200, 150, rm, options=opts)
        img = _decode_b64_image(result)
        assert img.width > 0

    def test_parse_px(self) -> None:
        assert _parse_px("16px") == 16.0
        assert _parse_px("0px") == 0.0
        assert _parse_px("") == 0.0
        assert _parse_px("auto") == 0.0
        assert _parse_px("1.5px") == 1.5

    def test_parse_border_width(self) -> None:
        assert _parse_border_width("1px solid rgb(0,0,0)") == 1.0
        assert _parse_border_width("2px dashed red") == 2.0
        assert _parse_border_width("none") == 0.0
        assert _parse_border_width("") == 0.0


# =============================================================================
# Accessibility overlay (C2) tests
# =============================================================================


class TestAccessibilityOverlay:
    def test_detects_small_target(self) -> None:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (200, 150), "white")
        draw = ImageDraw.Draw(img)
        from ui_bridge_mcp.screenshot import _load_font

        font = _load_font(12)

        elements = [
            _make_element("small-btn", type="button", x=10, y=10, w=30, h=20),
        ]
        issues = draw_accessibility_overlay(draw, elements, 1.0, 1.0, font)
        assert len(issues) >= 1
        assert any(i["issue"] == "small-target" for i in issues)

    def test_detects_missing_label(self) -> None:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (200, 150), "white")
        draw = ImageDraw.Draw(img)
        from ui_bridge_mcp.screenshot import _load_font

        font = _load_font(12)

        el = _make_element("btn-nolabel", type="button", x=10, y=10, w=80, h=40)
        # Element has no label field by default from _make_element
        issues = draw_accessibility_overlay(draw, [el], 1.0, 1.0, font)
        assert any(i["issue"] == "missing-label" for i in issues)

    def test_no_issues_for_labeled_large_element(self) -> None:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (200, 150), "white")
        draw = ImageDraw.Draw(img)
        from ui_bridge_mcp.screenshot import _load_font

        font = _load_font(12)

        el = _make_element("btn-ok", type="button", x=10, y=10, w=80, h=50)
        el["label"] = "Submit"
        issues = draw_accessibility_overlay(draw, [el], 1.0, 1.0, font)
        small = [i for i in issues if i["issue"] == "small-target"]
        no_label = [i for i in issues if i["issue"] == "missing-label"]
        assert len(small) == 0
        assert len(no_label) == 0

    def test_accessibility_mode_produces_valid_image(self) -> None:
        screenshot = _make_screenshot(400, 300)
        elements = [
            _make_element("btn-1", type="button", x=10, y=10, w=30, h=20),
            _make_element("btn-2", type="button", x=100, y=50, w=80, h=50),
        ]
        elements[1]["label"] = "OK"
        rm = FakeRefManager()
        opts = AnnotationOptions(mode="accessibility")
        result = annotate_screenshot(screenshot, elements, 400, 300, rm, options=opts)
        img = _decode_b64_image(result)
        assert img.size == (400, 300)


# =============================================================================
# Visual description (D2) tests
# =============================================================================


class TestVisualDescription:
    def test_basic_description(self) -> None:
        elements = [
            _make_element("btn-1", type="button", x=100, y=20, w=80, h=30),
            _make_element("input-1", type="input", x=100, y=200, w=200, h=30),
            _make_element(
                "text-1", type="p", category="content", x=100, y=400, w=200, h=20
            ),
        ]
        snapshot = {
            "viewport": {
                "viewportWidth": 1920,
                "viewportHeight": 1080,
                "scrollX": 0,
                "scrollY": 0,
                "documentWidth": 1920,
                "documentHeight": 2000,
                "canScrollDown": True,
                "canScrollRight": False,
            },
            "page": {"title": "Dashboard", "pathname": "/dashboard"},
        }
        desc = generate_visual_description(elements, snapshot)
        assert "Dashboard" in desc
        assert "1920x1080" in desc
        assert "3 visible" in desc
        assert "Interactive: 2" in desc
        assert "Content: 1" in desc

    def test_description_with_modal(self) -> None:
        elements: list[dict[str, Any]] = []
        snapshot = {
            "viewport": {"viewportWidth": 1920, "viewportHeight": 1080},
            "modalStack": {"activeModals": [{"id": "dlg-1", "title": "Confirm"}]},
        }
        desc = generate_visual_description(elements, snapshot)
        assert "Modal open" in desc

    def test_description_with_errors(self) -> None:
        elements: list[dict[str, Any]] = []
        snapshot = {
            "viewport": {"viewportWidth": 1920, "viewportHeight": 1080},
            "errorSummary": {
                "health": "degraded",
                "errorCount": 3,
                "warningCount": 1,
            },
        }
        desc = generate_visual_description(elements, snapshot)
        assert "degraded" in desc
        assert "Errors: 3" in desc

    def test_layout_region_detection(self) -> None:
        elements = [
            _make_element("nav", x=100, y=20, w=200, h=40),  # header
            _make_element("sidebar", x=10, y=200, w=100, h=400),  # left sidebar
            _make_element("main", x=400, y=200, w=800, h=400),  # main
            _make_element("footer", x=400, y=950, w=800, h=50),  # footer
        ]
        regions = _detect_layout_regions(elements, 1920.0, 1080.0)
        assert "header" in regions
        assert "left-sidebar" in regions
        assert "main-content" in regions
        assert "footer" in regions


# =============================================================================
# Delta encoder (D3) tests
# =============================================================================


class TestDeltaEncoder:
    def test_first_capture_returns_full(self) -> None:
        enc = DeltaEncoder(tile_size=64)
        img = _make_screenshot(128, 128, "white")
        result = enc.encode_delta(img)
        assert result["is_first"] is True
        assert result["full_image"] is not None
        assert result["total_tiles"] == 4  # (128/64)^2

    def test_identical_second_capture_no_changes(self) -> None:
        enc = DeltaEncoder(tile_size=64)
        img = _make_screenshot(128, 128, "white")
        enc.encode_delta(img)
        result = enc.encode_delta(img)
        assert result["is_first"] is False
        assert result["full_image"] is None
        assert result["changed_count"] == 0
        assert result["change_ratio"] == 0.0

    def test_different_image_detects_changes(self) -> None:
        enc = DeltaEncoder(tile_size=64)
        img1 = _make_screenshot(128, 128, "white")
        img2 = _make_screenshot(128, 128, "red")
        enc.encode_delta(img1)
        result = enc.encode_delta(img2)
        assert result["changed_count"] == 4  # all tiles changed
        assert result["change_ratio"] == 1.0

    def test_partial_change(self) -> None:
        enc = DeltaEncoder(tile_size=64)
        # First capture: all white
        img1 = _make_screenshot(128, 128, "white")
        enc.encode_delta(img1)

        # Second capture: top-left quadrant red
        img2_pil = Image.new("RGB", (128, 128), "white")
        for x in range(64):
            for y in range(64):
                img2_pil.putpixel((x, y), (255, 0, 0))
        buf = io.BytesIO()
        img2_pil.save(buf, format="PNG")
        img2_b64 = base64.b64encode(buf.getvalue()).decode()

        result = enc.encode_delta(img2_b64)
        assert result["changed_count"] == 1  # only 1 tile changed
        assert len(result["changed_tiles"]) == 1
        assert result["changed_tiles"][0]["x"] == 0
        assert result["changed_tiles"][0]["y"] == 0

    def test_reset_forces_full(self) -> None:
        enc = DeltaEncoder(tile_size=64)
        img = _make_screenshot(128, 128, "white")
        enc.encode_delta(img)
        enc.reset()
        result = enc.encode_delta(img)
        assert result["is_first"] is True

    def test_tile_data_is_valid_image(self) -> None:
        enc = DeltaEncoder(tile_size=64)
        img1 = _make_screenshot(128, 128, "white")
        img2 = _make_screenshot(128, 128, "blue")
        enc.encode_delta(img1)
        result = enc.encode_delta(img2)
        # Each changed tile should be a valid PNG
        for tile in result["changed_tiles"]:
            tile_img = _decode_b64_image(tile["data_b64"])
            assert tile_img.width == tile["w"]
            assert tile_img.height == tile["h"]
