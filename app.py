from __future__ import annotations

import base64
import hashlib
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from image_pipeline import (
    apply_erase_mask,
    apply_padding_and_background,
    build_square_sprite_sheet,
    crop_to_alpha_bbox,
    extract_connected_components,
    image_to_png_bytes,
    load_image,
    remove_background_and_crop,
    remove_background_and_extract_sprites,
    resampling_filter,
    rotate_image,
    resize_to_target,
)


OUTPUT_DIR = Path(__file__).parent / "outputs"
ERASER_COMPONENT_DIR = Path(__file__).parent / "components" / "eraser_canvas"
CLIPBOARD_IMAGE_COMPONENT_DIR = Path(__file__).parent / "components" / "clipboard_image"
MANUAL_CROP_COMPONENT_DIR = Path(__file__).parent / "components" / "manual_crop_editor"
CLIPBOARD_IMAGE_STATE_KEY = "cut_fit_clipboard_images"
CLIPBOARD_SEEN_STATE_KEY = "cut_fit_clipboard_seen_ids"
MANUAL_TRANSFORM_CLIPBOARD_IMAGE_STATE_KEY = "manual_transform_clipboard_images"
MANUAL_TRANSFORM_CLIPBOARD_SEEN_STATE_KEY = "manual_transform_clipboard_seen_ids"
SPRITE_SHEET_BUILDER_CLIPBOARD_IMAGE_STATE_KEY = (
    "sprite_sheet_builder_clipboard_images"
)
SPRITE_SHEET_BUILDER_CLIPBOARD_SEEN_STATE_KEY = (
    "sprite_sheet_builder_clipboard_seen_ids"
)
MODEL_NONE = "none"
MODEL_OPTIONS = {
    "u2net": "u2net - 기본",
    "isnet-general-use": "isnet-general-use - 일반 이미지 보존 우선",
    "isnet-anime": "isnet-anime - 일러스트/애니풍",
    MODEL_NONE: "모델 선택하지 않음 - 원본 기준",
}
RESIZE_MODE_OPTIONS = {
    "contain_center": "비율 유지 중앙",
    "contain_top": "비율 유지 상단",
    "contain_bottom": "비율 유지 하단",
    "contain_left": "비율 유지 좌측",
    "contain_right": "비율 유지 우측",
    "stretch": "늘려서 채우기",
}
MAX_OUTPUT_SIZE = 8192
PREVIEW_MAX_WIDTH = 420
PREVIEW_FRAME_HEIGHT = 180
PREVIEW_RENDER_SCALE = 2
CONTROL_WIDTH_XS = 64
CONTROL_WIDTH_SM = 150
CONTROL_WIDTH_MD = 260
CONTROL_WIDTH_LG = 360
CONTROL_WIDTH_XL = 560
ERASER_MAX_DIMENSION = 1024
MIN_ROTATION_DEGREES = -180
MAX_ROTATION_DEGREES = 180
SPRITE_SHEET_RESAMPLE_OPTIONS = {
    "nearest": "Nearest - 픽셀 유지",
    "lanczos": "Lanczos - 부드럽게",
}
TILESET_GUIDE_LAYOUT = (
    (
        ("TL", "top_left", "위쪽 왼쪽 모서리"),
        ("T", "top", "위쪽 가운데"),
        ("TR", "top_right", "위쪽 오른쪽 모서리"),
    ),
    (
        ("L", "left_wall", "왼쪽 벽"),
        ("C", "center", "가운데 내부"),
        ("R", "right_wall", "오른쪽 벽"),
    ),
    (
        ("BL", "bottom_left", "아래쪽 왼쪽 모서리"),
        ("B", "bottom", "아래쪽 가운데"),
        ("BR", "bottom_right", "아래쪽 오른쪽 모서리"),
    ),
)

eraser_canvas = components.declare_component(
    "eraser_canvas",
    path=str(ERASER_COMPONENT_DIR),
)
clipboard_image = components.declare_component(
    "clipboard_image",
    path=str(CLIPBOARD_IMAGE_COMPONENT_DIR),
)
manual_crop_editor = components.declare_component(
    "manual_crop_editor",
    path=str(MANUAL_CROP_COMPONENT_DIR),
)


def safe_stem(original_name: str) -> str:
    stem = Path(original_name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "image"


def safe_output_name(original_name: str, width: int, height: int) -> str:
    return f"{safe_stem(original_name)}_output_{width}x{height}.png"


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"저장 가능한 파일명을 만들지 못했습니다: {path.name}")


def shrink_for_preview(image: Image.Image) -> Image.Image:
    preview = image.copy().convert("RGBA")
    preview.thumbnail(
        (
            PREVIEW_MAX_WIDTH * PREVIEW_RENDER_SCALE,
            PREVIEW_FRAME_HEIGHT * PREVIEW_RENDER_SCALE,
        ),
        Image.Resampling.LANCZOS,
    )
    return preview


def shrink_for_eraser(image: Image.Image) -> Image.Image:
    editor_image = image.copy().convert("RGBA")
    editor_image.thumbnail(
        (ERASER_MAX_DIMENSION, ERASER_MAX_DIMENSION),
        Image.Resampling.LANCZOS,
    )
    return editor_image


def checkerboard_preview(image: Image.Image, square_size: int = 16) -> bytes:
    image = shrink_for_preview(image)
    width, height = image.size
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))

    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                color = (220, 220, 220, 255)
            else:
                color = (248, 248, 248, 255)
            background.paste(
                color,
                (x, y, min(x + square_size, width), min(y + square_size, height)),
            )

    background.alpha_composite(image)
    buffer = BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def png_data_uri(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def png_bytes_from_data_url(data_url: str) -> bytes:
    header, separator, payload = data_url.partition(",")
    if not separator or "base64" not in header or not header.startswith("data:image/png"):
        raise ValueError("지원하지 않는 마스크 데이터 형식입니다.")
    return base64.b64decode(payload)


def image_bytes_from_data_url(data_url: str) -> tuple[str, bytes]:
    header, separator, payload = data_url.partition(",")
    header = header.lower()
    if not separator or "base64" not in header or not header.startswith("data:image/"):
        raise ValueError("지원하지 않는 클립보드 이미지 데이터 형식입니다.")

    mime_type = header.removeprefix("data:").split(";", 1)[0]
    return mime_type, base64.b64decode(payload, validate=True)


def clipboard_image_name(original_name: str, index: int) -> str:
    stem = safe_stem(original_name or f"clipboard_{index:02d}")
    return f"{stem}.png"


def format_file_size(byte_count: int) -> str:
    if byte_count < 1024:
        return f"{byte_count} B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f} KB"
    return f"{byte_count / (1024 * 1024):.1f} MB"


def ensure_clipboard_state(image_state_key: str, seen_state_key: str) -> None:
    if image_state_key not in st.session_state:
        st.session_state[image_state_key] = []
    if seen_state_key not in st.session_state:
        st.session_state[seen_state_key] = []


def absorb_clipboard_images(
    component_value: Any,
    *,
    image_state_key: str,
    seen_state_key: str,
) -> int:
    ensure_clipboard_state(image_state_key, seen_state_key)
    if not isinstance(component_value, dict):
        return 0

    batch_id = str(component_value.get("id") or "")
    seen_ids = st.session_state[seen_state_key]
    if batch_id and batch_id in seen_ids:
        return 0

    image_payloads = component_value.get("images")
    if not isinstance(image_payloads, list):
        image_payloads = [component_value]

    pasted_images = st.session_state[image_state_key]
    existing_digests = {image["digest"] for image in pasted_images}
    added_count = 0
    errors: list[str] = []

    for image_payload in image_payloads:
        if not isinstance(image_payload, dict):
            continue

        try:
            _, raw_bytes = image_bytes_from_data_url(
                str(image_payload.get("data_url") or "")
            )
            image = load_image(raw_bytes)
            image_bytes = image_to_png_bytes(image)
        except Exception as exc:
            errors.append(str(exc))
            continue

        digest = hashlib.sha1(image_bytes).hexdigest()
        if digest in existing_digests:
            continue

        index = len(pasted_images) + 1
        original_name = str(image_payload.get("name") or "")
        pasted_images.append(
            {
                "name": clipboard_image_name(original_name, index),
                "bytes": image_bytes,
                "digest": digest,
                "source": "clipboard",
            }
        )
        existing_digests.add(digest)
        added_count += 1

    if batch_id:
        seen_ids.append(batch_id)

    if added_count:
        st.toast(f"클립보드 이미지 {added_count}개를 추가했습니다.")
    for error in errors:
        st.warning(f"클립보드 이미지를 읽지 못했습니다: {error}")

    return added_count


def render_fixed_preview(image_bytes: bytes, *, alt: str) -> None:
    st.image(image_bytes)


def render_clipboard_upload(
    *,
    component_key: str,
    image_state_key: str = CLIPBOARD_IMAGE_STATE_KEY,
    seen_state_key: str = CLIPBOARD_SEEN_STATE_KEY,
) -> list[dict[str, Any]]:
    ensure_clipboard_state(image_state_key, seen_state_key)
    component_value = clipboard_image(
        image_count=len(st.session_state[image_state_key]),
        default=None,
        key=component_key,
    )
    added_count = absorb_clipboard_images(
        component_value,
        image_state_key=image_state_key,
        seen_state_key=seen_state_key,
    )
    if added_count:
        st.rerun()

    return list(st.session_state[image_state_key])


def clear_clipboard_images(image_state_key: str) -> None:
    st.session_state[image_state_key] = []


def remove_clipboard_image(image_state_key: str, digest: str) -> None:
    st.session_state[image_state_key] = [
        image
        for image in st.session_state.get(image_state_key, [])
        if image.get("digest") != digest
    ]


def render_clipboard_image_manager(
    pasted_images: list[dict[str, Any]],
    *,
    clear_key: str,
    image_state_key: str = CLIPBOARD_IMAGE_STATE_KEY,
) -> None:
    if not pasted_images:
        return

    st.caption(f"클립보드 이미지 {len(pasted_images)}개")
    for index, image in enumerate(pasted_images):
        digest = str(image["digest"])
        name = str(image["name"])
        thumbnail = process_original_preview(image["bytes"])
        size_label = format_file_size(len(image["bytes"]))
        width, height = thumbnail["size"]
        row_cols = st.columns([0.08, 0.72, 0.2], vertical_alignment="center")
        with row_cols[0]:
            st.image(thumbnail["preview_bytes"], width=34)
        with row_cols[1]:
            st.write(name)
            st.caption(f"{width}x{height}px · {size_label}")
        with row_cols[2]:
            st.button(
                "제거",
                key=f"{clear_key}_remove_{digest}_{index}",
                on_click=remove_clipboard_image,
                args=(image_state_key, digest),
                width=CONTROL_WIDTH_SM,
            )

    if len(pasted_images) > 1:
        st.button(
            "클립보드 이미지 모두 제거",
            key=clear_key,
            on_click=clear_clipboard_images,
            args=(image_state_key,),
            width=CONTROL_WIDTH_MD,
        )


def tileset_guide_size(tile_size: int, gap: int, margin: int) -> tuple[int, int]:
    columns = max(len(row) for row in TILESET_GUIDE_LAYOUT)
    rows = len(TILESET_GUIDE_LAYOUT)
    return (
        margin * 2 + columns * tile_size + (columns - 1) * gap,
        margin * 2 + rows * tile_size + (rows - 1) * gap,
    )


def tileset_guide_specs(
    tile_size: int,
    gap: int,
    margin: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row_index, row in enumerate(TILESET_GUIDE_LAYOUT):
        for column_index, (code, file_stem, description) in enumerate(row):
            x0 = margin + column_index * (tile_size + gap)
            y0 = margin + row_index * (tile_size + gap)
            specs.append(
                {
                    "code": code,
                    "file_stem": file_stem,
                    "description": description,
                    "row": row_index + 1,
                    "column": column_index + 1,
                    "box": (x0, y0, x0 + tile_size, y0 + tile_size),
                }
            )
    return specs


def tileset_guide_table(
    tile_size: int,
    gap: int,
    margin: int,
) -> list[dict[str, Any]]:
    rows = []
    for spec in tileset_guide_specs(tile_size, gap, margin):
        x0, y0, x1, y1 = spec["box"]
        rows.append(
            {
                "코드": spec["code"],
                "파일": f"{spec['file_stem']}.png",
                "설명": spec["description"],
                "행": spec["row"],
                "열": spec["column"],
                "박스": f"({x0}, {y0})-({x1}, {y1})",
            }
        )
    return rows


@st.cache_data(show_spinner=False)
def build_tileset_guide_background(
    tile_size: int,
    gap: int,
    margin: int,
    line_width: int,
):
    width, height = tileset_guide_size(tile_size, gap, margin)
    if width > MAX_OUTPUT_SIZE or height > MAX_OUTPUT_SIZE:
        raise ValueError(
            f"가이드 크기가 최대 {MAX_OUTPUT_SIZE}px를 넘습니다: {width}x{height}px"
        )

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    columns = max(len(row) for row in TILESET_GUIDE_LAYOUT)
    rows = len(TILESET_GUIDE_LAYOUT)
    line_color = (255, 75, 75, 230)

    for column in range(columns + 1):
        x = min(margin + column * (tile_size + gap), width - 1)
        draw.line((x, 0, x, height - 1), fill=line_color, width=line_width)

    for row in range(rows + 1):
        y = min(margin + row * (tile_size + gap), height - 1)
        draw.line((0, y, width - 1, y), fill=line_color, width=line_width)

    return {
        "png_bytes": image_to_png_bytes(image),
        "size": image.size,
        "tiles": tileset_guide_table(tile_size, gap, margin),
    }


@st.cache_data(show_spinner=False)
def slice_tileset_guide_image(
    image_bytes: bytes,
    tile_size: int,
    gap: int,
    margin: int,
    file_prefix: str,
    skip_transparent_tiles: bool,
):
    source = load_image(image_bytes)
    expected_width, expected_height = tileset_guide_size(tile_size, gap, margin)
    if source.width < expected_width or source.height < expected_height:
        raise ValueError(
            "입력 이미지가 현재 가이드 설정보다 작습니다: "
            f"입력={source.width}x{source.height}px, "
            f"필요={expected_width}x{expected_height}px"
        )

    prefix = safe_stem(file_prefix)
    tiles = []
    for spec in tileset_guide_specs(tile_size, gap, margin):
        tile = source.crop(spec["box"])
        if skip_transparent_tiles and tile.getchannel("A").getbbox() is None:
            continue

        file_name = f"{prefix}_{spec['file_stem']}.png"
        png_bytes = image_to_png_bytes(tile)
        tiles.append(
            {
                "code": spec["code"],
                "file": file_name,
                "description": spec["description"],
                "size": f"{tile.width}x{tile.height}",
                "png_bytes": png_bytes,
                "preview_bytes": checkerboard_preview(tile),
            }
        )

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for tile in tiles:
            archive.writestr(tile["file"], tile["png_bytes"])

    return {
        "tiles": tiles,
        "zip_bytes": zip_buffer.getvalue(),
        "source_size": source.size,
        "expected_size": (expected_width, expected_height),
    }

@st.cache_resource(show_spinner=False)
def get_rembg_session(model_name: str):
    from rembg import new_session

    return new_session(model_name)


@st.cache_data(show_spinner=False)
def process_original_preview(image_bytes: bytes):
    source = load_image(image_bytes)
    return {
        "preview_bytes": checkerboard_preview(source),
        "size": source.size,
    }


@st.cache_data(show_spinner=False)
def process_crop(
    image_bytes: bytes,
    alpha_threshold: int,
    model_name: str,
    preserve_interior: bool,
    post_process_mask: bool,
):
    if model_name == MODEL_NONE:
        return process_alpha_crop(image_bytes, alpha_threshold)

    fallback_reason = ""
    try:
        result = remove_background_and_crop(
            image_bytes,
            alpha_threshold=alpha_threshold,
            preserve_interior=preserve_interior,
            post_process_mask=post_process_mask,
            session=get_rembg_session(model_name),
        )
    except ValueError as exc:
        source = load_image(image_bytes)
        try:
            cropped, bbox = crop_to_alpha_bbox(source, alpha_threshold=alpha_threshold)
        except ValueError:
            cropped = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            bbox = (0, 0, 1, 1)

        fallback_reason = f"모델 crop 실패, 원본 기준으로 편집합니다: {exc}"
        return {
            "png_bytes": image_to_png_bytes(cropped),
            "preview_bytes": checkerboard_preview(cropped),
            "bbox": bbox,
            "source_size": source.size,
            "cropped_size": cropped.size,
            "removed_size": source.size,
            "fallback_reason": fallback_reason,
        }

    return {
        "png_bytes": result.png_bytes,
        "preview_bytes": checkerboard_preview(result.cropped),
        "bbox": result.bbox,
        "source_size": result.source_size,
        "cropped_size": result.cropped.size,
        "removed_size": result.removed_background.size,
        "fallback_reason": fallback_reason,
    }


@st.cache_data(show_spinner=False)
def process_alpha_crop(image_bytes: bytes, alpha_threshold: int):
    source = load_image(image_bytes)
    try:
        cropped, bbox = crop_to_alpha_bbox(source, alpha_threshold=alpha_threshold)
        fallback_reason = ""
    except ValueError as exc:
        cropped = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        bbox = (0, 0, 1, 1)
        fallback_reason = f"원본 crop 실패, 투명 이미지로 편집합니다: {exc}"

    return {
        "png_bytes": image_to_png_bytes(cropped),
        "preview_bytes": checkerboard_preview(cropped),
        "bbox": bbox,
        "source_size": source.size,
        "cropped_size": cropped.size,
        "removed_size": source.size,
        "fallback_reason": fallback_reason,
    }


@st.cache_data(show_spinner=False)
def process_manual_erase(
    cropped_png_bytes: bytes,
    erase_mask_data_url: str,
    alpha_threshold: int,
):
    cropped = Image.open(BytesIO(cropped_png_bytes)).convert("RGBA")
    bbox = (0, 0, cropped.width, cropped.height)
    if erase_mask_data_url:
        mask = Image.open(BytesIO(png_bytes_from_data_url(erase_mask_data_url)))
        cropped = apply_erase_mask(cropped, mask)
        try:
            cropped, bbox = crop_to_alpha_bbox(
                cropped,
                alpha_threshold=alpha_threshold,
            )
        except ValueError:
            cropped = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            bbox = (0, 0, 1, 1)

    return {
        "png_bytes": image_to_png_bytes(cropped),
        "preview_bytes": checkerboard_preview(cropped),
        "size": cropped.size,
        "bbox": bbox,
    }


@st.cache_data(show_spinner=False)
def process_eraser_editor_image(cropped_png_bytes: bytes):
    cropped = Image.open(BytesIO(cropped_png_bytes)).convert("RGBA")
    editor_image = shrink_for_eraser(cropped)
    return {
        "png_bytes": image_to_png_bytes(editor_image),
        "size": editor_image.size,
    }


@st.cache_data(show_spinner=False)
def process_rotation(
    cropped_png_bytes: bytes,
    rotation_degrees: int,
    alpha_threshold: int,
):
    cropped = Image.open(BytesIO(cropped_png_bytes)).convert("RGBA")
    rotated = rotate_image(
        cropped,
        degrees=rotation_degrees,
        alpha_threshold=alpha_threshold,
    )
    return {
        "png_bytes": image_to_png_bytes(rotated),
        "preview_bytes": checkerboard_preview(rotated),
        "size": rotated.size,
    }


def normalize_manual_crop_box(
    crop_box: Any,
    source_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    default_box = (0, 0, source_width, source_height)

    if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4:
        return default_box

    try:
        x0, y0, x1, y1 = [int(round(float(value))) for value in crop_box]
    except (TypeError, ValueError):
        return default_box

    x0 = max(0, min(x0, source_width - 1))
    y0 = max(0, min(y0, source_height - 1))
    x1 = max(x0 + 1, min(x1, source_width))
    y1 = max(y0 + 1, min(y1, source_height))
    return x0, y0, x1, y1


@st.cache_data(show_spinner=False)
def process_manual_transform_source(image_bytes: bytes):
    source = load_image(image_bytes)
    return {
        "png_bytes": image_to_png_bytes(source),
        "preview_bytes": checkerboard_preview(source),
        "size": source.size,
    }


@st.cache_data(show_spinner=False)
def process_manual_transform(
    image_bytes: bytes,
    paint_data_url: str,
    image_erase_mask_data_url: str,
    crop_box: tuple[int, int, int, int],
    rotation_degrees: int,
    output_width: int,
    output_height: int,
    resize_mode: str,
):
    source = load_image(image_bytes)
    if image_erase_mask_data_url:
        erase_mask = Image.open(
            BytesIO(png_bytes_from_data_url(image_erase_mask_data_url))
        )
        source = apply_erase_mask(source, erase_mask)

    if paint_data_url:
        paint_layer = Image.open(BytesIO(png_bytes_from_data_url(paint_data_url)))
        paint_layer = paint_layer.convert("RGBA")
        if paint_layer.size != source.size:
            paint_layer = paint_layer.resize(source.size, Image.Resampling.LANCZOS)
        source.alpha_composite(paint_layer)

    normalized_crop_box = normalize_manual_crop_box(crop_box, source.size)
    cropped = source.crop(normalized_crop_box)

    degrees = float(rotation_degrees)
    if abs(degrees % 360) < 1e-9:
        rotated = cropped
    else:
        rotated = cropped.rotate(
            -degrees,
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )

    target_width = int(output_width)
    target_height = int(output_height)
    if target_width < 1 or target_height < 1:
        raise ValueError("출력 가로/세로는 1px 이상이어야 합니다.")
    if target_width > MAX_OUTPUT_SIZE or target_height > MAX_OUTPUT_SIZE:
        raise ValueError(
            f"출력 크기가 최대 {MAX_OUTPUT_SIZE}px를 넘습니다: "
            f"{target_width}x{target_height}px"
        )

    output = resize_to_target(
        rotated,
        width=target_width,
        height=target_height,
        mode=normalize_resize_mode(resize_mode),
    )

    return {
        "png_bytes": image_to_png_bytes(output),
        "preview_bytes": checkerboard_preview(output),
        "source_size": source.size,
        "crop_box": normalized_crop_box,
        "cropped_size": cropped.size,
        "rotated_size": rotated.size,
        "resize_mode": normalize_resize_mode(resize_mode),
        "size": output.size,
    }


@st.cache_data(show_spinner=False)
def process_sprite_sheet(
    image_bytes: bytes,
    alpha_threshold: int,
    model_name: str,
    preserve_interior: bool,
    post_process_mask: bool,
    min_area: int,
):
    if model_name == MODEL_NONE:
        components = extract_connected_components(
            load_image(image_bytes),
            alpha_threshold=alpha_threshold,
            min_area=min_area,
        )
    else:
        components = remove_background_and_extract_sprites(
            image_bytes,
            alpha_threshold=alpha_threshold,
            min_area=min_area,
            preserve_interior=preserve_interior,
            post_process_mask=post_process_mask,
            session=get_rembg_session(model_name),
        )
    return [
        {
            "png_bytes": component.png_bytes,
            "bbox": component.bbox,
            "area": component.area,
            "size": component.image.size,
        }
        for component in components
    ]


@st.cache_data(show_spinner=False)
def build_combined_sprite_sheet(
    image_payloads: tuple[tuple[str, bytes], ...],
    scale_factor: float,
    gap: int,
    resampling: str,
):
    images: list[Image.Image] = []
    for file_name, image_bytes in image_payloads:
        image = load_image(image_bytes)
        images.append(image)

    result = build_square_sprite_sheet(
        images,
        scale=float(scale_factor),
        gap=int(gap),
        resampling=resampling,
        max_dimension=MAX_OUTPUT_SIZE,
    )
    placements = []
    for (file_name, _), placement in zip(
        image_payloads,
        result.placements,
        strict=True,
    ):
        paste_x0, paste_y0, paste_x1, paste_y1 = placement.paste_box
        cell_x0, cell_y0, cell_x1, cell_y1 = placement.cell_box
        placements.append(
            {
                "index": placement.index + 1,
                "file": file_name,
                "source": f"{placement.source_size[0]}x{placement.source_size[1]}",
                "cell": f"({cell_x0}, {cell_y0})-({cell_x1}, {cell_y1})",
                "paste": f"({paste_x0}, {paste_y0})-({paste_x1}, {paste_y1})",
            }
        )

    return {
        "png_bytes": result.png_bytes,
        "preview_bytes": checkerboard_preview(result.image),
        "unscaled_size": result.unscaled_size,
        "scaled_size": result.scaled_size,
        "cell_size": result.cell_size,
        "columns": result.columns,
        "rows": result.rows,
        "gap": result.gap,
        "scale": result.scale,
        "placements": placements,
    }


@st.cache_data(show_spinner=False)
def image_size_for_payload(image_bytes: bytes) -> tuple[int, int]:
    return load_image(image_bytes).size


@st.cache_data(show_spinner=False)
def recover_sprite_sheet(
    image_bytes: bytes,
    source_name: str,
    scale_factor: float,
    alpha_threshold: int,
    min_area: int,
    resampling: str,
):
    source = load_image(image_bytes)
    scaled_size = (
        max(1, round(source.width * scale_factor)),
        max(1, round(source.height * scale_factor)),
    )
    if scaled_size[0] > MAX_OUTPUT_SIZE or scaled_size[1] > MAX_OUTPUT_SIZE:
        raise ValueError(
            f"스케일 적용 후 최대 크기 {MAX_OUTPUT_SIZE}px를 넘습니다: "
            f"{scaled_size[0]}x{scaled_size[1]}px"
        )

    scaled = source
    if scaled.size != scaled_size:
        scaled = source.resize(scaled_size, resampling_filter(resampling))

    components = extract_connected_components(
        scaled,
        alpha_threshold=alpha_threshold,
        min_area=min_area,
    )
    prefix = safe_stem(source_name)
    sprites = []
    for index, component in enumerate(components, start=1):
        file_name = f"{prefix}_sprite_{index:02d}.png"
        png_bytes = component.png_bytes
        sprites.append(
            {
                "index": index,
                "file": file_name,
                "bbox": component.bbox,
                "area": component.area,
                "size": component.image.size,
                "png_bytes": png_bytes,
                "preview_bytes": checkerboard_preview(component.image),
            }
        )

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for sprite in sprites:
            archive.writestr(sprite["file"], sprite["png_bytes"])

    return {
        "scaled_png_bytes": image_to_png_bytes(scaled),
        "scaled_preview_bytes": checkerboard_preview(scaled),
        "source_size": source.size,
        "scaled_size": scaled.size,
        "sprites": sprites,
        "zip_bytes": zip_buffer.getvalue(),
    }


@st.cache_data(show_spinner=False)
def build_output_image(
    cropped_png_bytes: bytes,
    width: int,
    height: int,
    resize_mode: str,
    padding_top: int,
    padding_right: int,
    padding_bottom: int,
    padding_left: int,
    transparent_background: bool,
    background_color: str,
):
    cropped = Image.open(BytesIO(cropped_png_bytes)).convert("RGBA")
    resized = resize_to_target(
        cropped,
        width=width,
        height=height,
        mode=resize_mode,
    )
    output = apply_padding_and_background(
        resized,
        padding_top=padding_top,
        padding_right=padding_right,
        padding_bottom=padding_bottom,
        padding_left=padding_left,
        transparent_background=transparent_background,
        background_color=background_color,
    )
    return {
        "png_bytes": image_to_png_bytes(output),
        "preview_bytes": checkerboard_preview(output),
        "resized_size": resized.size,
        "size": output.size,
    }


def image_state_key(image_id: str, name: str) -> str:
    return f"image_{image_id}_{name}"


def manual_image_state_key(image_id: str, name: str) -> str:
    return f"manual_image_{image_id}_{name}"


def normalize_resize_mode(value: Any) -> str:
    if value in RESIZE_MODE_OPTIONS:
        return str(value)
    return "contain_center"


def initialize_image_state(image_id: str) -> None:
    legacy_padding = int(st.session_state.get(image_state_key(image_id, "padding"), 0))
    defaults: dict[str, Any] = {
        "model_name": "u2net",
        "alpha_threshold": 16,
        "preserve_interior": True,
        "post_process_mask": True,
        "aspect_locked": True,
        "output_width": 0,
        "output_height": 0,
        "erase_enabled": False,
        "erase_mask_data_url": "",
        "erase_brush_size": 24,
        "erase_revision": 0,
        "rotation_degrees": 0,
        "resize_mode": "contain_center",
        "padding": 0,
        "padding_top": legacy_padding,
        "padding_right": legacy_padding,
        "padding_bottom": legacy_padding,
        "padding_left": legacy_padding,
        "transparent_background": True,
        "background_color": "#000000",
    }

    for name, value in defaults.items():
        key = image_state_key(image_id, name)
        if key not in st.session_state:
            st.session_state[key] = value


def prune_image_state(active_image_ids: set[str]) -> None:
    active_prefixes = {f"image_{image_id}_" for image_id in active_image_ids}
    for key in list(st.session_state.keys()):
        if key.startswith("image_") and not any(
            key.startswith(prefix) for prefix in active_prefixes
        ):
            del st.session_state[key]


def initialize_manual_image_state(
    image_id: str,
    source_size: tuple[int, int],
) -> None:
    defaults: dict[str, Any] = {
        "crop_box": [0, 0, source_size[0], source_size[1]],
        "crop_revision": 0,
        "paint_data_url": "",
        "paint_revision": 0,
        "image_erase_mask_data_url": "",
        "image_erase_revision": 0,
        "paint_tool": "crop",
        "paint_color": "#ff4b4b",
        "paint_brush_size": 24,
        "editor_zoom": 100,
        "rotation_degrees": 0,
        "output_width": source_size[0],
        "output_height": source_size[1],
        "resize_mode": "contain_center",
    }

    for name, value in defaults.items():
        key = manual_image_state_key(image_id, name)
        if key not in st.session_state:
            st.session_state[key] = value

    crop_key = manual_image_state_key(image_id, "crop_box")
    st.session_state[crop_key] = list(
        normalize_manual_crop_box(st.session_state[crop_key], source_size)
    )
    resize_key = manual_image_state_key(image_id, "resize_mode")
    st.session_state[resize_key] = normalize_resize_mode(st.session_state[resize_key])


def prune_manual_image_state(active_image_ids: set[str]) -> None:
    active_prefixes = {f"manual_image_{image_id}_" for image_id in active_image_ids}
    for key in list(st.session_state.keys()):
        if key.startswith("manual_image_") and not any(
            key.startswith(prefix) for prefix in active_prefixes
        ):
            del st.session_state[key]


def manual_image_settings(
    image_id: str,
    source_size: tuple[int, int],
) -> dict[str, Any]:
    crop_key = manual_image_state_key(image_id, "crop_box")
    paint_tool_key = manual_image_state_key(image_id, "paint_tool")
    paint_brush_key = manual_image_state_key(image_id, "paint_brush_size")
    editor_zoom_key = manual_image_state_key(image_id, "editor_zoom")
    width_key = manual_image_state_key(image_id, "output_width")
    height_key = manual_image_state_key(image_id, "output_height")
    resize_key = manual_image_state_key(image_id, "resize_mode")

    crop_box = normalize_manual_crop_box(st.session_state[crop_key], source_size)
    st.session_state[crop_key] = list(crop_box)

    paint_tool = str(st.session_state[paint_tool_key])
    if paint_tool not in {"crop", "brush", "paint_eraser", "image_eraser"}:
        paint_tool = "crop"

    try:
        paint_brush_size = int(st.session_state[paint_brush_key])
    except (TypeError, ValueError):
        paint_brush_size = 24
    try:
        editor_zoom = int(st.session_state[editor_zoom_key])
    except (TypeError, ValueError):
        editor_zoom = 100

    try:
        output_width = int(st.session_state[width_key])
    except (TypeError, ValueError):
        output_width = source_size[0]
    try:
        output_height = int(st.session_state[height_key])
    except (TypeError, ValueError):
        output_height = source_size[1]

    return {
        "crop_box": crop_box,
        "paint_data_url": st.session_state[
            manual_image_state_key(image_id, "paint_data_url")
        ],
        "image_erase_mask_data_url": st.session_state[
            manual_image_state_key(image_id, "image_erase_mask_data_url")
        ],
        "paint_tool": paint_tool,
        "paint_color": st.session_state[
            manual_image_state_key(image_id, "paint_color")
        ],
        "paint_brush_size": max(1, min(256, paint_brush_size)),
        "editor_zoom": max(50, min(800, editor_zoom)),
        "rotation_degrees": int(
            st.session_state[manual_image_state_key(image_id, "rotation_degrees")]
        ),
        "output_width": max(1, min(MAX_OUTPUT_SIZE, output_width)),
        "output_height": max(1, min(MAX_OUTPUT_SIZE, output_height)),
        "resize_mode": normalize_resize_mode(st.session_state[resize_key]),
    }


def image_settings(image_id: str) -> dict[str, Any]:
    return {
        "model_name": st.session_state[image_state_key(image_id, "model_name")],
        "alpha_threshold": int(
            st.session_state[image_state_key(image_id, "alpha_threshold")]
        ),
        "preserve_interior": bool(
            st.session_state[image_state_key(image_id, "preserve_interior")]
        ),
        "post_process_mask": bool(
            st.session_state[image_state_key(image_id, "post_process_mask")]
        ),
        "aspect_locked": bool(
            st.session_state[image_state_key(image_id, "aspect_locked")]
        ),
        "output_width": int(st.session_state[image_state_key(image_id, "output_width")]),
        "output_height": int(
            st.session_state[image_state_key(image_id, "output_height")]
        ),
        "erase_enabled": bool(
            st.session_state[image_state_key(image_id, "erase_enabled")]
        ),
        "erase_mask_data_url": st.session_state[
            image_state_key(image_id, "erase_mask_data_url")
        ],
        "erase_brush_size": int(
            st.session_state[image_state_key(image_id, "erase_brush_size")]
        ),
        "erase_revision": int(
            st.session_state[image_state_key(image_id, "erase_revision")]
        ),
        "rotation_degrees": int(
            st.session_state[image_state_key(image_id, "rotation_degrees")]
        ),
        "resize_mode": normalize_resize_mode(
            st.session_state[image_state_key(image_id, "resize_mode")]
        ),
        "padding": int(st.session_state[image_state_key(image_id, "padding")]),
        "padding_top": int(
            st.session_state[image_state_key(image_id, "padding_top")]
        ),
        "padding_right": int(
            st.session_state[image_state_key(image_id, "padding_right")]
        ),
        "padding_bottom": int(
            st.session_state[image_state_key(image_id, "padding_bottom")]
        ),
        "padding_left": int(
            st.session_state[image_state_key(image_id, "padding_left")]
        ),
        "transparent_background": bool(
            st.session_state[image_state_key(image_id, "transparent_background")]
        ),
        "background_color": st.session_state[
            image_state_key(image_id, "background_color")
        ],
    }


def image_label(item: dict[str, Any]) -> str:
    return f"{item['name']} · {item['digest'][:8]}"


def toggle_aspect_lock(image_id: str) -> None:
    key = image_state_key(image_id, "aspect_locked")
    st.session_state[key] = not st.session_state.get(key, True)


def ensure_output_size(image_id: str, cropped_size: tuple[int, int]) -> None:
    width_key = image_state_key(image_id, "output_width")
    height_key = image_state_key(image_id, "output_height")

    if int(st.session_state.get(width_key, 0)) <= 0:
        st.session_state[width_key] = cropped_size[0]
    if int(st.session_state.get(height_key, 0)) <= 0:
        st.session_state[height_key] = cropped_size[1]


def target_size_for_settings(
    settings: dict[str, Any],
    cropped_size: tuple[int, int],
) -> tuple[int, int]:
    cropped_width, cropped_height = cropped_size
    width = settings["output_width"] if settings["output_width"] > 0 else cropped_width
    height = settings["output_height"] if settings["output_height"] > 0 else cropped_height

    if settings["aspect_locked"]:
        aspect_ratio = cropped_width / cropped_height
        height = max(1, min(MAX_OUTPUT_SIZE, round(width / aspect_ratio)))

    return width, height


def current_target_size(
    image_id: str,
    cropped_size: tuple[int, int],
) -> tuple[int, int, str]:
    settings = image_settings(image_id)
    target_width, target_height = target_size_for_settings(settings, cropped_size)
    return target_width, target_height, settings["resize_mode"]


def reset_erase_mask(image_id: str) -> None:
    mask_key = image_state_key(image_id, "erase_mask_data_url")
    revision_key = image_state_key(image_id, "erase_revision")
    st.session_state[mask_key] = ""
    st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1


def reset_manual_crop(image_id: str, source_size: tuple[int, int]) -> None:
    crop_key = manual_image_state_key(image_id, "crop_box")
    revision_key = manual_image_state_key(image_id, "crop_revision")
    st.session_state[crop_key] = [0, 0, source_size[0], source_size[1]]
    st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1


def reset_manual_paint(image_id: str) -> None:
    paint_key = manual_image_state_key(image_id, "paint_data_url")
    revision_key = manual_image_state_key(image_id, "paint_revision")
    st.session_state[paint_key] = ""
    st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1


def reset_manual_image_erase(image_id: str) -> None:
    mask_key = manual_image_state_key(image_id, "image_erase_mask_data_url")
    revision_key = manual_image_state_key(image_id, "image_erase_revision")
    st.session_state[mask_key] = ""
    st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1


def reset_manual_transform(image_id: str, source_size: tuple[int, int]) -> None:
    reset_manual_crop(image_id, source_size)
    reset_manual_paint(image_id)
    reset_manual_image_erase(image_id)
    st.session_state[manual_image_state_key(image_id, "rotation_degrees")] = 0
    st.session_state[manual_image_state_key(image_id, "editor_zoom")] = 100
    st.session_state[manual_image_state_key(image_id, "output_width")] = source_size[0]
    st.session_state[manual_image_state_key(image_id, "output_height")] = source_size[1]
    st.session_state[manual_image_state_key(image_id, "resize_mode")] = "contain_center"


def render_erase_editor(image_id: str, cropped_png_bytes: bytes) -> None:
    enabled_key = image_state_key(image_id, "erase_enabled")
    mask_key = image_state_key(image_id, "erase_mask_data_url")
    brush_key = image_state_key(image_id, "erase_brush_size")
    revision_key = image_state_key(image_id, "erase_revision")

    control_cols = st.columns([1, 1, 2], vertical_alignment="bottom")
    with control_cols[0]:
        st.checkbox(
            "수동 지우기",
            key=enabled_key,
            help="켜면 아래 캔버스에 그린 영역이 투명하게 지워집니다.",
        )
    with control_cols[1]:
        st.slider(
            "브러시",
            min_value=1,
            max_value=160,
            step=1,
            key=brush_key,
            disabled=not bool(st.session_state[enabled_key]),
            width=CONTROL_WIDTH_MD,
        )
    with control_cols[2]:
        st.button(
            "마스크 초기화",
            key=image_state_key(image_id, "erase_reset"),
            on_click=reset_erase_mask,
            args=(image_id,),
            disabled=not bool(st.session_state[mask_key]),
            width=CONTROL_WIDTH_MD,
        )

    if not bool(st.session_state[enabled_key]):
        return

    mask_data_url = st.session_state[mask_key]
    editor_image = process_eraser_editor_image(cropped_png_bytes)
    component_value = eraser_canvas(
        image_data_url=png_data_uri(editor_image["png_bytes"]),
        mask_data_url=mask_data_url,
        brush_size=int(st.session_state[brush_key]),
        key=f"{image_state_key(image_id, 'erase_canvas')}_{st.session_state[revision_key]}",
        default=mask_data_url,
    )
    if isinstance(component_value, str) and component_value != mask_data_url:
        st.session_state[mask_key] = component_value
        st.rerun()


def render_compact_controls(
    image_id: str,
    cropped_size: tuple[int, int],
    *,
    show_background_controls: bool,
) -> None:
    cropped_width, cropped_height = cropped_size
    aspect_ratio = cropped_width / cropped_height
    width_key = image_state_key(image_id, "output_width")
    height_key = image_state_key(image_id, "output_height")
    lock_key = image_state_key(image_id, "aspect_locked")
    rotation_key = image_state_key(image_id, "rotation_degrees")
    resize_key = image_state_key(image_id, "resize_mode")
    padding_top_key = image_state_key(image_id, "padding_top")
    padding_right_key = image_state_key(image_id, "padding_right")
    padding_bottom_key = image_state_key(image_id, "padding_bottom")
    padding_left_key = image_state_key(image_id, "padding_left")
    transparent_key = image_state_key(image_id, "transparent_background")
    background_key = image_state_key(image_id, "background_color")

    st.caption("옵션")
    if show_background_controls:
        model_key = image_state_key(image_id, "model_name")
        background_cols = st.columns([1.5, 1.3, 0.9, 0.9], vertical_alignment="bottom")
        with background_cols[0]:
            st.selectbox(
                "모델",
                options=list(MODEL_OPTIONS.keys()),
                format_func=lambda value: MODEL_OPTIONS[value],
                key=model_key,
                help="모델 선택하지 않음은 배경 제거를 건너뛰고 원본 alpha 기준으로 편집합니다.",
                width=CONTROL_WIDTH_LG,
            )
        model_disabled = st.session_state[model_key] == MODEL_NONE
        with background_cols[1]:
            st.slider(
                "Alpha",
                min_value=0,
                max_value=254,
                key=image_state_key(image_id, "alpha_threshold"),
                help="값이 높을수록 거의 투명한 배경 잔여 픽셀을 무시하고 더 강하게 crop합니다.",
                width=CONTROL_WIDTH_LG,
            )
        with background_cols[2]:
            st.checkbox(
                "내부 복원",
                key=image_state_key(image_id, "preserve_interior"),
                help="외곽선 안쪽에서 투명해진 영역을 원본 색상으로 다시 채웁니다.",
                disabled=model_disabled,
            )
        with background_cols[3]:
            st.checkbox(
                "마스크",
                key=image_state_key(image_id, "post_process_mask"),
                help="경계를 매끄럽게 할 수 있지만 작은 내부 디테일은 더 사라질 수 있습니다.",
                disabled=model_disabled,
            )
    else:
        st.slider(
            "Alpha",
            min_value=0,
            max_value=254,
            key=image_state_key(image_id, "alpha_threshold"),
            help="값이 높을수록 거의 투명한 배경 잔여 픽셀을 무시하고 더 강하게 crop합니다.",
            width=CONTROL_WIDTH_LG,
        )

    locked = bool(st.session_state[lock_key])
    size_cols = st.columns([1, 0.22, 1, 1.7], vertical_alignment="bottom")
    with size_cols[0]:
        target_width = int(
            st.number_input(
                "가로",
                min_value=1,
                max_value=MAX_OUTPUT_SIZE,
                step=1,
                key=width_key,
                width=CONTROL_WIDTH_SM,
            )
        )

    with size_cols[1]:
        st.button(
            " ",
            key=image_state_key(image_id, "aspect_lock_button"),
            icon=":material/lock:" if locked else ":material/lock_open:",
            help="가로/세로 비율 잠금",
            on_click=toggle_aspect_lock,
            args=(image_id,),
            width=CONTROL_WIDTH_XS,
        )

    with size_cols[2]:
        if locked:
            target_height = max(
                1,
                min(MAX_OUTPUT_SIZE, round(target_width / aspect_ratio)),
            )
            st.session_state[height_key] = target_height
            st.number_input(
                "세로",
                min_value=1,
                max_value=MAX_OUTPUT_SIZE,
                step=1,
                key=height_key,
                disabled=True,
                width=CONTROL_WIDTH_SM,
            )
        else:
            st.number_input(
                "세로",
                min_value=1,
                max_value=MAX_OUTPUT_SIZE,
                step=1,
                key=height_key,
                width=CONTROL_WIDTH_SM,
            )

    with size_cols[3]:
        st.session_state[resize_key] = normalize_resize_mode(
            st.session_state.get(resize_key)
        )
        st.segmented_control(
            "리사이즈",
            options=list(RESIZE_MODE_OPTIONS.keys()),
            format_func=lambda value: RESIZE_MODE_OPTIONS[value],
            key=resize_key,
            help="비율 유지 옵션은 투명 캔버스 안에 맞춘 뒤 중앙/상단/하단/좌측/우측으로 배치합니다. 늘려서 채우기는 비율을 무시합니다.",
            width="content",
        )

    background_option_cols = st.columns([1.4, 1, 1], vertical_alignment="bottom")
    with background_option_cols[0]:
        st.slider(
            "각도",
            min_value=MIN_ROTATION_DEGREES,
            max_value=MAX_ROTATION_DEGREES,
            step=1,
            key=rotation_key,
            help="양수는 시계 방향, 음수는 반시계 방향으로 회전합니다.",
            width=CONTROL_WIDTH_LG,
        )
    with background_option_cols[1]:
        st.checkbox(
            "투명 배경",
            key=transparent_key,
            help="끄면 지정한 배경색으로 투명 영역과 padding을 채웁니다.",
        )
    with background_option_cols[2]:
        st.color_picker(
            "배경색",
            key=background_key,
            disabled=bool(st.session_state[transparent_key]),
        )

    padding_cols = st.columns(4, vertical_alignment="bottom")
    padding_fields = [
        ("Padding 상", padding_top_key),
        ("Padding 우", padding_right_key),
        ("Padding 하", padding_bottom_key),
        ("Padding 좌", padding_left_key),
    ]
    for column, (label, key) in zip(padding_cols, padding_fields, strict=True):
        with column:
            st.number_input(
                label,
                min_value=0,
                max_value=MAX_OUTPUT_SIZE,
                step=1,
                key=key,
                help="현재 Output 이미지 바깥쪽에 추가할 여백(px)입니다.",
                width=CONTROL_WIDTH_SM,
            )


def process_item_output(
    item: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], int, int]:
    settings = image_settings(item["id"])
    if item.get("kind") == "sprite":
        crop = process_alpha_crop(
            item["bytes"],
            settings["alpha_threshold"],
        )
    else:
        crop = process_crop(
            item["bytes"],
            settings["alpha_threshold"],
            settings["model_name"],
            settings["preserve_interior"],
            settings["post_process_mask"],
        )
    erased = process_manual_erase(
        crop["png_bytes"],
        settings["erase_mask_data_url"] if settings["erase_enabled"] else "",
        settings["alpha_threshold"],
    )
    rotated = process_rotation(
        erased["png_bytes"],
        settings["rotation_degrees"],
        settings["alpha_threshold"],
    )
    ensure_output_size(item["id"], rotated["size"])
    target_width, target_height, resize_mode = current_target_size(
        item["id"],
        rotated["size"],
    )
    output = build_output_image(
        rotated["png_bytes"],
        target_width,
        target_height,
        resize_mode,
        settings["padding_top"],
        settings["padding_right"],
        settings["padding_bottom"],
        settings["padding_left"],
        settings["transparent_background"],
        settings["background_color"],
    )
    return crop, erased, rotated, output, target_width, target_height


def save_output(item: dict[str, Any], png_bytes: bytes, width: int, height: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = safe_output_name(item["name"], width, height)
    output_path = unique_output_path(OUTPUT_DIR / output_name)
    output_path.write_bytes(png_bytes)
    return output_path


def render_image_card(item: dict[str, Any]) -> None:
    with st.container(border=True):
        st.write(f"**{item['name']}**")
        if item.get("kind") == "sprite":
            st.caption(
                f"{item['parent_name']} · sprite #{item['sprite_index']} · "
                f"bbox={item['sheet_bbox']} · area={item['component_area']}"
            )
        else:
            st.caption(f"id={item['digest'][:8]}")

        try:
            with st.spinner(f"{item['name']} 처리 중..."):
                crop, erased, rotated, output, target_width, target_height = (
                    process_item_output(item)
                )
        except Exception as exc:
            ensure_output_size(item["id"], (1, 1))
            preview_col, controls_col = st.columns([1, 2], vertical_alignment="top")
            with preview_col:
                st.caption("Original")
                original_preview = process_original_preview(item["bytes"])
                render_fixed_preview(
                    original_preview["preview_bytes"],
                    alt=f"{item['name']} original",
                )
            with controls_col:
                render_compact_controls(
                    item["id"],
                    (1, 1),
                    show_background_controls=item.get("kind") != "sprite",
                )
                st.error(str(exc))
            return

        settings = image_settings(item["id"])
        original_preview = process_original_preview(item["bytes"])
        preview_cols = st.columns(4, vertical_alignment="top")
        with preview_cols[0]:
            st.caption("Original")
            render_fixed_preview(
                original_preview["preview_bytes"],
                alt=f"{item['name']} original",
            )
        with preview_cols[1]:
            st.caption("Cropped")
            render_fixed_preview(
                erased["preview_bytes"],
                alt=f"{item['name']} cropped",
            )
        with preview_cols[2]:
            st.caption("Rotated")
            render_fixed_preview(
                rotated["preview_bytes"],
                alt=f"{item['name']} rotated",
            )
        with preview_cols[3]:
            st.caption("Output")
            render_fixed_preview(
                output["preview_bytes"],
                alt=f"{item['name']} output",
            )

        x0, y0, x1, y1 = crop["bbox"]
        cropped_width, cropped_height = crop["cropped_size"]
        source_width, source_height = crop["source_size"]
        trimmed_width = source_width - cropped_width
        trimmed_height = source_height - cropped_height
        st.caption(
            f"source={source_width}x{source_height}px · "
            f"bbox=({x0}, {y0}, {x1}, {y1}) · "
            f"cropped={cropped_width}x{cropped_height}px · "
            f"recropped={erased['size'][0]}x{erased['size'][1]}px · "
            f"rotated={rotated['size'][0]}x{rotated['size'][1]}px · "
            f"resized={target_width}x{target_height}px · "
            f"final={output['size'][0]}x{output['size'][1]}px · "
            f"angle={settings['rotation_degrees']}deg · "
            f"erase={'on' if settings['erase_enabled'] and settings['erase_mask_data_url'] else 'off'} · "
            f"padding=top {settings['padding_top']}px, "
            f"right {settings['padding_right']}px, "
            f"bottom {settings['padding_bottom']}px, "
            f"left {settings['padding_left']}px · "
            f"background={'transparent' if settings['transparent_background'] else settings['background_color']} · "
            f"mode={RESIZE_MODE_OPTIONS[settings['resize_mode']]} · "
            f"removed={trimmed_width}px width, {trimmed_height}px height"
        )
        if crop.get("fallback_reason"):
            st.warning(crop["fallback_reason"])

        render_erase_editor(item["id"], crop["png_bytes"])

        render_compact_controls(
            item["id"],
            rotated["size"],
            show_background_controls=item.get("kind") != "sprite",
        )

        action_cols = st.columns([1, 1, 2], vertical_alignment="bottom")
        final_width, final_height = output["size"]
        output_name = safe_output_name(item["name"], final_width, final_height)
        with action_cols[0]:
            st.download_button(
                "PNG 다운로드",
                data=output["png_bytes"],
                file_name=output_name,
                mime="image/png",
                key=image_state_key(item["id"], "download"),
                width=CONTROL_WIDTH_MD,
            )
        with action_cols[1]:
            if st.button(
                "저장",
                key=image_state_key(item["id"], "save"),
                width=CONTROL_WIDTH_MD,
            ):
                output_path = save_output(
                    item,
                    output["png_bytes"],
                    final_width,
                    final_height,
                )
                st.success(f"저장됨: {output_path}")


def render_cut_fit_tab() -> None:
    with st.container(border=True):
        upload_cols = st.columns([1.25, 1], vertical_alignment="top")
        with upload_cols[0]:
            uploaded_files = st.file_uploader(
                "이미지 업로드",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="cut_fit_uploads",
                width=CONTROL_WIDTH_XL,
            )
        with upload_cols[1]:
            st.caption("클립보드")
            pasted_images = render_clipboard_upload(
                component_key="cut_fit_clipboard_image",
            )

        render_clipboard_image_manager(
            pasted_images,
            clear_key="cut_fit_clear_clipboard_images",
        )

    sprite_sheet_mode = st.toggle(
        "스프라이트 시트 분리",
        value=False,
        help="업로드 이미지를 배경 제거하거나 원본 alpha 기준으로 연결되지 않은 영역을 각각의 스프라이트로 분리합니다.",
    )

    sheet_settings: dict[str, Any] = {}
    if sprite_sheet_mode:
        with st.container(border=True):
            sheet_cols = st.columns(
                [1.5, 1.2, 0.9, 0.9, 1],
                vertical_alignment="bottom",
            )
            with sheet_cols[0]:
                sheet_settings["model_name"] = st.selectbox(
                    "시트 모델",
                    options=list(MODEL_OPTIONS.keys()),
                    format_func=lambda value: MODEL_OPTIONS[value],
                    key="sheet_model_name",
                    help="모델 선택하지 않음은 배경 제거 없이 원본 alpha 연결 영역을 분리합니다.",
                    width=CONTROL_WIDTH_LG,
                )
                sheet_model_disabled = sheet_settings["model_name"] == MODEL_NONE
            with sheet_cols[1]:
                sheet_settings["alpha_threshold"] = st.slider(
                    "시트 Alpha",
                    min_value=0,
                    max_value=254,
                    value=16,
                    key="sheet_alpha_threshold",
                    width=CONTROL_WIDTH_LG,
                )
            with sheet_cols[2]:
                sheet_settings["preserve_interior"] = st.checkbox(
                    "내부 복원",
                    value=True,
                    key="sheet_preserve_interior",
                    disabled=sheet_model_disabled,
                )
            with sheet_cols[3]:
                sheet_settings["post_process_mask"] = st.checkbox(
                    "마스크",
                    value=True,
                    key="sheet_post_process_mask",
                    disabled=sheet_model_disabled,
                )
            with sheet_cols[4]:
                sheet_settings["min_area"] = int(
                    st.number_input(
                        "최소 영역",
                        min_value=1,
                        max_value=1_000_000,
                        value=64,
                        step=1,
                        key="sheet_min_area",
                        width=CONTROL_WIDTH_SM,
                    )
                )

    image_sources = [
        {
            "name": uploaded_file.name,
            "bytes": uploaded_file.getvalue(),
            "source": "upload",
        }
        for uploaded_file in uploaded_files or []
    ]
    image_sources.extend(pasted_images)

    if not image_sources:
        st.info("PNG, JPG, JPEG, WebP 이미지를 업로드하거나 클립보드에서 붙여넣으세요.")
        return

    uploaded_items: list[dict[str, Any]] = []
    for index, image_source in enumerate(image_sources):
        image_bytes = image_source["bytes"]
        image_name = image_source["name"]
        digest = hashlib.sha1(image_bytes).hexdigest()
        if sprite_sheet_mode:
            try:
                with st.spinner(f"{image_name} 스프라이트 분리 중..."):
                    sprites = process_sprite_sheet(
                        image_bytes,
                        sheet_settings["alpha_threshold"],
                        sheet_settings["model_name"],
                        sheet_settings["preserve_interior"],
                        sheet_settings["post_process_mask"],
                        sheet_settings["min_area"],
                    )
            except Exception as exc:
                st.error(f"{image_name}: {exc}")
                continue

            if not sprites:
                st.warning(f"{image_name}: 분리된 스프라이트가 없습니다.")
                continue

            for sprite_index, sprite in enumerate(sprites, start=1):
                sprite_bytes = sprite["png_bytes"]
                sprite_digest = hashlib.sha1(sprite_bytes).hexdigest()
                image_id = f"{digest[:12]}_{index}_sprite_{sprite_index}_{sprite_digest[:8]}"
                item = {
                    "id": image_id,
                    "digest": sprite_digest,
                    "name": f"{safe_stem(image_name)}_sprite_{sprite_index:02d}.png",
                    "bytes": sprite_bytes,
                    "kind": "sprite",
                    "parent_name": image_name,
                    "sprite_index": sprite_index,
                    "sheet_bbox": sprite["bbox"],
                    "component_area": sprite["area"],
                }
                uploaded_items.append(item)
                initialize_image_state(image_id)
        else:
            image_id = f"{digest[:16]}_{index}"
            item = {
                "id": image_id,
                "digest": digest,
                "name": image_name,
                "bytes": image_bytes,
                "kind": "image",
            }
            uploaded_items.append(item)
            initialize_image_state(image_id)

    if not uploaded_items:
        return

    active_ids = {item["id"] for item in uploaded_items}
    prune_image_state(active_ids)

    st.caption(f"{len(uploaded_items)}개 이미지 업로드됨")

    for uploaded_item in uploaded_items:
        render_image_card(uploaded_item)

    st.divider()
    if st.button("모든 이미지 저장", type="primary", width=CONTROL_WIDTH_MD):
        saved_paths: list[Path] = []
        errors: list[str] = []
        progress = st.progress(0, text="일괄 저장 준비 중...")

        for index, item in enumerate(uploaded_items, start=1):
            progress.progress(
                (index - 1) / len(uploaded_items),
                text=f"{item['name']} 처리 중...",
            )
            try:
                _, _, _, item_output, _, _ = process_item_output(item)
                final_width, final_height = item_output["size"]
                saved_paths.append(
                    save_output(
                        item,
                        item_output["png_bytes"],
                        final_width,
                        final_height,
                    )
                )
            except Exception as exc:
                errors.append(f"{item['name']}: {exc}")

        progress.progress(1.0, text="일괄 저장 완료")

        if saved_paths:
            st.success(f"{len(saved_paths)}개 이미지를 {OUTPUT_DIR}에 저장했습니다.")
            with st.expander("저장된 파일"):
                for path in saved_paths:
                    st.write(str(path))

        if errors:
            st.error("일부 이미지를 저장하지 못했습니다.")
            for error in errors:
                st.write(error)


def render_manual_transform_card(item: dict[str, Any]) -> None:
    source = process_manual_transform_source(item["bytes"])
    source_size = source["size"]
    initialize_manual_image_state(item["id"], source_size)

    with st.container(border=True):
        st.write(f"**{item['name']}**")
        st.caption(f"id={item['digest'][:8]}")

        rotation_key = manual_image_state_key(item["id"], "rotation_degrees")
        paint_tool_key = manual_image_state_key(item["id"], "paint_tool")
        paint_color_key = manual_image_state_key(item["id"], "paint_color")
        paint_brush_key = manual_image_state_key(item["id"], "paint_brush_size")
        paint_data_key = manual_image_state_key(item["id"], "paint_data_url")
        image_erase_mask_key = manual_image_state_key(
            item["id"],
            "image_erase_mask_data_url",
        )
        editor_zoom_key = manual_image_state_key(item["id"], "editor_zoom")
        width_key = manual_image_state_key(item["id"], "output_width")
        height_key = manual_image_state_key(item["id"], "output_height")
        resize_key = manual_image_state_key(item["id"], "resize_mode")

        control_cols = st.columns(
            [1.2, 0.75, 0.75, 1.8, 0.75],
            vertical_alignment="bottom",
        )
        with control_cols[0]:
            st.slider(
                "각도",
                min_value=MIN_ROTATION_DEGREES,
                max_value=MAX_ROTATION_DEGREES,
                step=1,
                key=rotation_key,
                help="양수는 시계 방향, 음수는 반시계 방향으로 회전합니다.",
                width=CONTROL_WIDTH_LG,
            )
        with control_cols[1]:
            st.number_input(
                "가로",
                min_value=1,
                max_value=MAX_OUTPUT_SIZE,
                step=1,
                key=width_key,
                help="Crop과 회전 후 결과를 배치할 최종 가로 크기입니다.",
                width=CONTROL_WIDTH_SM,
            )
        with control_cols[2]:
            st.number_input(
                "세로",
                min_value=1,
                max_value=MAX_OUTPUT_SIZE,
                step=1,
                key=height_key,
                help="Crop과 회전 후 결과를 배치할 최종 세로 크기입니다.",
                width=CONTROL_WIDTH_SM,
            )
        with control_cols[3]:
            st.session_state[resize_key] = normalize_resize_mode(
                st.session_state.get(resize_key)
            )
            st.segmented_control(
                "리사이즈",
                options=list(RESIZE_MODE_OPTIONS.keys()),
                format_func=lambda value: RESIZE_MODE_OPTIONS[value],
                key=resize_key,
                help="비율 유지 옵션은 지정 크기 안에 맞춘 뒤 위치를 정합니다. 늘려서 채우기는 비율을 무시합니다.",
                width="stretch",
            )
        with control_cols[4]:
            st.button(
                "전체 초기화",
                key=manual_image_state_key(item["id"], "transform_reset"),
                on_click=reset_manual_transform,
                args=(item["id"], source_size),
                width="stretch",
            )

        paint_cols = st.columns(
            [1.8, 0.75, 1, 1, 0.75, 0.75, 0.75],
            vertical_alignment="bottom",
        )
        with paint_cols[0]:
            st.segmented_control(
                "도구",
                options=["crop", "brush", "paint_eraser", "image_eraser"],
                format_func=lambda value: {
                    "crop": "Crop",
                    "brush": "브러시",
                    "paint_eraser": "그림 지우개",
                    "image_eraser": "이미지 지우개",
                }[value],
                key=paint_tool_key,
                width="stretch",
            )
        with paint_cols[1]:
            st.color_picker(
                "색상",
                key=paint_color_key,
                disabled=st.session_state[paint_tool_key] != "brush",
            )
        with paint_cols[2]:
            st.slider(
                "브러시/지우개 크기",
                min_value=1,
                max_value=256,
                step=1,
                key=paint_brush_key,
                disabled=st.session_state[paint_tool_key] == "crop",
                width=CONTROL_WIDTH_LG,
            )
        with paint_cols[3]:
            st.slider(
                "확대",
                min_value=50,
                max_value=800,
                step=25,
                format="%d%%",
                key=editor_zoom_key,
                width=CONTROL_WIDTH_LG,
            )
        with paint_cols[4]:
            st.button(
                "Crop 초기화",
                key=manual_image_state_key(item["id"], "crop_reset"),
                on_click=reset_manual_crop,
                args=(item["id"], source_size),
                width="stretch",
            )
        with paint_cols[5]:
            st.button(
                "그림 지우기",
                key=manual_image_state_key(item["id"], "paint_reset"),
                on_click=reset_manual_paint,
                args=(item["id"],),
                disabled=not bool(st.session_state[paint_data_key]),
                width="stretch",
            )
        with paint_cols[6]:
            st.button(
                "이미지 복원",
                key=manual_image_state_key(item["id"], "image_erase_reset"),
                on_click=reset_manual_image_erase,
                args=(item["id"],),
                disabled=not bool(st.session_state[image_erase_mask_key]),
                width="stretch",
            )

        settings = manual_image_settings(item["id"], source_size)
        editor_col, preview_col = st.columns([1.35, 1], vertical_alignment="top")

        with editor_col:
            crop_key = manual_image_state_key(item["id"], "crop_box")
            revision_key = manual_image_state_key(item["id"], "crop_revision")
            paint_revision_key = manual_image_state_key(item["id"], "paint_revision")
            image_erase_revision_key = manual_image_state_key(
                item["id"],
                "image_erase_revision",
            )
            component_value = manual_crop_editor(
                image_data_url=png_data_uri(source["png_bytes"]),
                crop_box=list(settings["crop_box"]),
                paint_data_url=settings["paint_data_url"],
                image_erase_mask_data_url=settings["image_erase_mask_data_url"],
                tool=settings["paint_tool"],
                brush_color=settings["paint_color"],
                brush_size=settings["paint_brush_size"],
                zoom_factor=settings["editor_zoom"] / 100,
                key=(
                    f"{manual_image_state_key(item['id'], 'crop_canvas')}_"
                    f"{st.session_state[revision_key]}_"
                    f"{st.session_state[paint_revision_key]}_"
                    f"{st.session_state[image_erase_revision_key]}"
                ),
                default={
                    "crop_box": list(settings["crop_box"]),
                    "paint_data_url": settings["paint_data_url"],
                    "image_erase_mask_data_url": settings[
                        "image_erase_mask_data_url"
                    ],
                },
            )
            next_crop_box: tuple[int, int, int, int] | None = None
            next_paint_data_url: str | None = None
            next_image_erase_mask_data_url: str | None = None
            if isinstance(component_value, dict):
                next_crop_box = normalize_manual_crop_box(
                    component_value.get("crop_box"),
                    source_size,
                )
                next_paint_data_url = str(component_value.get("paint_data_url") or "")
                next_image_erase_mask_data_url = str(
                    component_value.get("image_erase_mask_data_url") or ""
                )
            elif isinstance(component_value, (list, tuple)) and len(component_value) == 4:
                next_crop_box = normalize_manual_crop_box(component_value, source_size)

            should_rerun = False
            if next_crop_box is not None and next_crop_box != settings["crop_box"]:
                st.session_state[crop_key] = list(next_crop_box)
                should_rerun = True
            if (
                next_paint_data_url is not None
                and next_paint_data_url != settings["paint_data_url"]
            ):
                st.session_state[paint_data_key] = next_paint_data_url
                should_rerun = True
            if (
                next_image_erase_mask_data_url is not None
                and next_image_erase_mask_data_url
                != settings["image_erase_mask_data_url"]
            ):
                st.session_state[image_erase_mask_key] = next_image_erase_mask_data_url
                should_rerun = True
            if should_rerun:
                st.rerun()

        settings = manual_image_settings(item["id"], source_size)
        try:
            output = process_manual_transform(
                item["bytes"],
                settings["paint_data_url"],
                settings["image_erase_mask_data_url"],
                settings["crop_box"],
                settings["rotation_degrees"],
                settings["output_width"],
                settings["output_height"],
                settings["resize_mode"],
            )
        except Exception as exc:
            with preview_col:
                st.error(str(exc))
            return

        with preview_col:
            st.caption("Output")
            render_fixed_preview(output["preview_bytes"], alt=f"{item['name']} output")

            source_width, source_height = output["source_size"]
            x0, y0, x1, y1 = output["crop_box"]
            crop_width, crop_height = output["cropped_size"]
            rotated_width, rotated_height = output["rotated_size"]
            final_width, final_height = output["size"]
            st.caption(
                f"source={source_width}x{source_height}px · "
                f"crop=({x0}, {y0}, {x1}, {y1}) · "
                f"cropped={crop_width}x{crop_height}px · "
                f"rotated={rotated_width}x{rotated_height}px · "
                f"target={settings['output_width']}x{settings['output_height']}px · "
                f"mode={RESIZE_MODE_OPTIONS[settings['resize_mode']]} · "
                f"paint={'on' if settings['paint_data_url'] else 'off'} · "
                f"image erase={'on' if settings['image_erase_mask_data_url'] else 'off'} · "
                f"zoom={settings['editor_zoom']}% · "
                f"final={final_width}x{final_height}px"
            )

            output_name = safe_output_name(item["name"], final_width, final_height)
            action_cols = st.columns([1, 1], vertical_alignment="bottom")
            with action_cols[0]:
                st.download_button(
                    "PNG 다운로드",
                    data=output["png_bytes"],
                    file_name=output_name,
                    mime="image/png",
                    key=manual_image_state_key(item["id"], "download"),
                    width="stretch",
                )
            with action_cols[1]:
                if st.button(
                    "저장",
                    key=manual_image_state_key(item["id"], "save"),
                    width="stretch",
                ):
                    output_path = save_output(
                        item,
                        output["png_bytes"],
                        final_width,
                        final_height,
                    )
                    st.success(f"저장됨: {output_path}")


def render_manual_transform_tab() -> None:
    with st.container(border=True):
        upload_cols = st.columns([1.25, 1], vertical_alignment="top")
        with upload_cols[0]:
            uploaded_files = st.file_uploader(
                "이미지 업로드",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="manual_transform_uploads",
                width=CONTROL_WIDTH_XL,
            )
        with upload_cols[1]:
            st.caption("클립보드")
            pasted_images = render_clipboard_upload(
                component_key="manual_transform_clipboard_image",
                image_state_key=MANUAL_TRANSFORM_CLIPBOARD_IMAGE_STATE_KEY,
                seen_state_key=MANUAL_TRANSFORM_CLIPBOARD_SEEN_STATE_KEY,
            )

        render_clipboard_image_manager(
            pasted_images,
            clear_key="manual_transform_clear_clipboard_images",
            image_state_key=MANUAL_TRANSFORM_CLIPBOARD_IMAGE_STATE_KEY,
        )

    image_sources = [
        {
            "name": uploaded_file.name,
            "bytes": uploaded_file.getvalue(),
            "source": "upload",
        }
        for uploaded_file in uploaded_files or []
    ]
    image_sources.extend(pasted_images)

    if not image_sources:
        st.info("PNG, JPG, JPEG, WebP 이미지를 업로드하거나 클립보드에서 붙여넣으세요.")
        return

    uploaded_items: list[dict[str, Any]] = []
    for index, image_source in enumerate(image_sources):
        image_bytes = image_source["bytes"]
        digest = hashlib.sha1(image_bytes).hexdigest()
        image_id = f"{digest[:16]}_{index}"
        uploaded_items.append(
            {
                "id": image_id,
                "digest": digest,
                "name": image_source["name"],
                "bytes": image_bytes,
                "kind": "manual",
            }
        )

    active_ids = {item["id"] for item in uploaded_items}
    prune_manual_image_state(active_ids)

    st.caption(f"{len(uploaded_items)}개 이미지 업로드됨")

    for uploaded_item in uploaded_items:
        render_manual_transform_card(uploaded_item)


def render_sprite_sheet_make_mode() -> None:
    with st.container(border=True):
        upload_cols = st.columns([1.25, 1], vertical_alignment="top")
        with upload_cols[0]:
            sheet_files = st.file_uploader(
                "스프라이트 시트용 이미지 업로드",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="sprite_sheet_builder_uploads",
                help="기존 이미지 편집 탭과 별개의 업로드 목록입니다.",
                width=CONTROL_WIDTH_XL,
            )
        with upload_cols[1]:
            st.caption("클립보드")
            pasted_images = render_clipboard_upload(
                component_key="sprite_sheet_builder_clipboard_image",
                image_state_key=SPRITE_SHEET_BUILDER_CLIPBOARD_IMAGE_STATE_KEY,
                seen_state_key=SPRITE_SHEET_BUILDER_CLIPBOARD_SEEN_STATE_KEY,
            )

        render_clipboard_image_manager(
            pasted_images,
            clear_key="sprite_sheet_builder_clear_clipboard_images",
            image_state_key=SPRITE_SHEET_BUILDER_CLIPBOARD_IMAGE_STATE_KEY,
        )

    uploaded_payloads = [
        (sheet_file.name, sheet_file.getvalue()) for sheet_file in sheet_files or []
    ]
    pasted_payloads = [(image["name"], image["bytes"]) for image in pasted_images]
    image_payloads = tuple(uploaded_payloads + pasted_payloads)

    if not image_payloads:
        st.info("스프라이트 시트로 합칠 이미지를 업로드하거나 클립보드에서 붙여넣으세요.")
        return

    first_image_name, first_image_bytes = image_payloads[0]
    first_image_width, first_image_height = image_size_for_payload(first_image_bytes)
    target_width_key = "sprite_sheet_builder_first_image_target_width"
    target_source_key = "sprite_sheet_builder_first_image_target_source"
    first_image_source = hashlib.sha1(first_image_bytes).hexdigest()
    if st.session_state.get(target_source_key) != first_image_source:
        st.session_state[target_source_key] = first_image_source
        st.session_state[target_width_key] = first_image_width

    control_cols = st.columns([1.2, 1, 1, 1.6], vertical_alignment="bottom")
    with control_cols[0]:
        scale_mode = st.segmented_control(
            "스케일 기준",
            options=["scale", "first_width"],
            format_func=lambda value: {
                "scale": "직접 Scale",
                "first_width": "첫 이미지 가로",
            }[value],
            default="scale",
            key="sprite_sheet_builder_scale_mode",
            width="stretch",
        )
    with control_cols[1]:
        if scale_mode == "first_width":
            target_first_width = int(
                st.number_input(
                    "첫 이미지 목표 가로(px)",
                    min_value=1,
                    max_value=MAX_OUTPUT_SIZE,
                    step=1,
                    key=target_width_key,
                    help="첫 번째 이미지의 원본 가로폭을 이 값에 맞추도록 전체 스케일을 계산합니다.",
                )
            )
            scale_factor = target_first_width / first_image_width
        else:
            scale_factor = st.number_input(
                "Scale",
                min_value=0.05,
                max_value=16.0,
                value=1.0,
                step=0.05,
                format="%.2f",
                key="sprite_sheet_builder_scale",
                help="완성된 스프라이트 시트 전체를 몇 배로 조정할지 정합니다.",
            )
    with control_cols[2]:
        gap = int(
            st.number_input(
                "간격(px)",
                min_value=0,
                max_value=512,
                value=0,
                step=1,
                key="sprite_sheet_builder_gap",
                help="각 이미지 셀 사이에 넣을 투명 간격입니다.",
            )
        )
    with control_cols[3]:
        resampling = st.segmented_control(
            "스케일 방식",
            options=list(SPRITE_SHEET_RESAMPLE_OPTIONS.keys()),
            format_func=lambda value: SPRITE_SHEET_RESAMPLE_OPTIONS[value],
            default="nearest",
            key="sprite_sheet_builder_resampling",
            width="stretch",
        )
    if resampling is None:
        resampling = "nearest"

    st.caption(
        f"첫 이미지={first_image_name} · "
        f"원본={first_image_width}x{first_image_height}px · "
        f"적용 scale={float(scale_factor):.4f}x"
    )

    try:
        with st.spinner("스프라이트 시트 생성 중..."):
            sheet = build_combined_sprite_sheet(
                image_payloads,
                float(scale_factor),
                gap,
                str(resampling),
            )
    except Exception as exc:
        st.error(str(exc))
        return

    scaled_width, scaled_height = sheet["scaled_size"]
    unscaled_width, unscaled_height = sheet["unscaled_size"]
    cell_width, cell_height = sheet["cell_size"]

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("이미지", f"{len(image_payloads)}개")
    with metric_cols[1]:
        st.metric("격자", f"{sheet['columns']} x {sheet['rows']}")
    with metric_cols[2]:
        st.metric("셀", f"{cell_width} x {cell_height}px")
    with metric_cols[3]:
        st.metric("최종", f"{scaled_width} x {scaled_height}px")

    st.caption(
        f"원본 시트={unscaled_width}x{unscaled_height}px · "
        f"scale={sheet['scale']:.2f}x · "
        f"간격={sheet['gap']}px · "
        f"방식={SPRITE_SHEET_RESAMPLE_OPTIONS[str(resampling)]}"
    )
    render_fixed_preview(sheet["preview_bytes"], alt="sprite sheet output")

    output_name = safe_output_name("sprite_sheet.png", scaled_width, scaled_height)
    action_cols = st.columns([1, 1, 2], vertical_alignment="bottom")
    with action_cols[0]:
        st.download_button(
            "PNG 다운로드",
            data=sheet["png_bytes"],
            file_name=output_name,
            mime="image/png",
            key="sprite_sheet_builder_download",
            width="stretch",
        )
    with action_cols[1]:
        if st.button(
            "저장",
            key="sprite_sheet_builder_save",
            width="stretch",
        ):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = unique_output_path(OUTPUT_DIR / output_name)
            output_path.write_bytes(sheet["png_bytes"])
            st.success(f"저장됨: {output_path}")

    with st.expander("배치 정보"):
        st.dataframe(sheet["placements"], width="stretch", hide_index=True)


def render_sprite_sheet_recover_mode() -> None:
    sheet_file = st.file_uploader(
        "복구할 스프라이트 시트 업로드",
        type=["png", "jpg", "jpeg", "webp"],
        key="sprite_sheet_recover_upload",
    )
    if sheet_file is None:
        st.info("투명 배경 스프라이트 시트를 업로드하세요.")
        return

    control_cols = st.columns([1, 1, 1, 1.3], vertical_alignment="bottom")
    with control_cols[0]:
        scale_factor = st.number_input(
            "축소 Scale",
            min_value=0.05,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            key="sprite_sheet_recover_scale",
            help="전체 시트를 먼저 축소한 뒤 스프라이트를 분리합니다.",
        )
    with control_cols[1]:
        alpha_threshold = int(
            st.slider(
                "Alpha",
                min_value=0,
                max_value=254,
                value=16,
                key="sprite_sheet_recover_alpha_threshold",
            )
        )
    with control_cols[2]:
        min_area = int(
            st.number_input(
                "최소 영역",
                min_value=1,
                max_value=1_000_000,
                value=16,
                step=1,
                key="sprite_sheet_recover_min_area",
            )
        )
    with control_cols[3]:
        resampling = st.segmented_control(
            "스케일 방식",
            options=list(SPRITE_SHEET_RESAMPLE_OPTIONS.keys()),
            format_func=lambda value: SPRITE_SHEET_RESAMPLE_OPTIONS[value],
            default="nearest",
            key="sprite_sheet_recover_resampling",
            width="stretch",
        )
    if resampling is None:
        resampling = "nearest"

    try:
        with st.spinner("스프라이트 시트 복구 중..."):
            recovered = recover_sprite_sheet(
                sheet_file.getvalue(),
                sheet_file.name,
                float(scale_factor),
                alpha_threshold,
                min_area,
                str(resampling),
            )
    except Exception as exc:
        st.error(str(exc))
        return

    source_width, source_height = recovered["source_size"]
    scaled_width, scaled_height = recovered["scaled_size"]
    sprites = recovered["sprites"]

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("원본", f"{source_width} x {source_height}px")
    with metric_cols[1]:
        st.metric("축소", f"{scaled_width} x {scaled_height}px")
    with metric_cols[2]:
        st.metric("Scale", f"{float(scale_factor):.2f}x")
    with metric_cols[3]:
        st.metric("분리", f"{len(sprites)}개")

    render_fixed_preview(recovered["scaled_preview_bytes"], alt="recovered sprite sheet")

    safe_name = safe_stem(sheet_file.name)
    action_cols = st.columns([1, 1, 2], vertical_alignment="bottom")
    with action_cols[0]:
        st.download_button(
            "축소 시트 PNG",
            data=recovered["scaled_png_bytes"],
            file_name=f"{safe_name}_scaled_{scaled_width}x{scaled_height}.png",
            mime="image/png",
            key="sprite_sheet_recover_scaled_download",
            width="stretch",
        )
    with action_cols[1]:
        st.download_button(
            "분리 PNG ZIP",
            data=recovered["zip_bytes"],
            file_name=f"{safe_name}_sprites.zip",
            mime="application/zip",
            key="sprite_sheet_recover_zip_download",
            width="stretch",
            disabled=not sprites,
        )

    if not sprites:
        st.warning("분리된 스프라이트가 없습니다.")
        return

    with st.expander("분리 정보"):
        st.dataframe(
            [
                {
                    "index": sprite["index"],
                    "file": sprite["file"],
                    "size": f"{sprite['size'][0]}x{sprite['size'][1]}",
                    "bbox": sprite["bbox"],
                    "area": sprite["area"],
                }
                for sprite in sprites
            ],
            width="stretch",
            hide_index=True,
        )

    with st.expander("개별 스프라이트"):
        preview_cols = st.columns(4)
        for index, sprite in enumerate(sprites):
            with preview_cols[index % 4]:
                st.caption(sprite["file"])
                st.image(sprite["preview_bytes"], width=96)
                st.download_button(
                    "PNG",
                    data=sprite["png_bytes"],
                    file_name=sprite["file"],
                    mime="image/png",
                    key=f"sprite_sheet_recover_download_{sprite['file']}",
                    width="stretch",
                )


def render_sprite_sheet_builder_tab() -> None:
    mode = st.segmented_control(
        "모드",
        options=["make", "recover"],
        format_func=lambda value: {
            "make": "스프라이트 시트 만들기",
            "recover": "스프라이트 시트 복구하기",
        }[value],
        default="make",
        key="sprite_sheet_builder_mode",
        width="stretch",
    )
    if mode == "recover":
        render_sprite_sheet_recover_mode()
    else:
        render_sprite_sheet_make_mode()


def render_tileset_guide_tab() -> None:
    gap = 0
    margin = 0
    control_cols = st.columns([1, 1.2, 1, 1.8], vertical_alignment="bottom")
    with control_cols[0]:
        tile_size = int(
            st.number_input(
                "타일 크기(px)",
                min_value=16,
                max_value=256,
                value=64,
                step=8,
                key="tileset_guide_tile_size",
            )
        )
    with control_cols[1]:
        file_prefix = st.text_input(
            "파일 접두어",
            value="ground",
            key="tileset_guide_file_prefix",
        )
    with control_cols[2]:
        line_width = int(
            st.number_input(
                "가이드선(px)",
                min_value=1,
                max_value=6,
                value=1,
                step=1,
                key="tileset_guide_line_width",
            )
        )

    try:
        guide = build_tileset_guide_background(tile_size, gap, margin, line_width)
    except Exception as exc:
        st.error(str(exc))
        return

    guide_width, guide_height = guide["size"]
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("타일", f"{len(guide['tiles'])}개")
    with metric_cols[1]:
        st.metric("크기", f"{tile_size} x {tile_size}px")
    with metric_cols[2]:
        st.metric("격자", "3 x 3")
    with metric_cols[3]:
        st.metric("가이드", f"{guide_width} x {guide_height}px")

    render_fixed_preview(guide["png_bytes"], alt="tileset guide")

    guide_file_name = f"{safe_stem(file_prefix)}_tileset_guide_{tile_size}px.png"
    download_cols = st.columns([1, 1, 2], vertical_alignment="bottom")
    with download_cols[0]:
        st.download_button(
            "가이드 PNG 다운로드",
            data=guide["png_bytes"],
            file_name=guide_file_name,
            mime="image/png",
            key="tileset_guide_download",
            width="stretch",
        )
    with download_cols[1]:
        with st.expander("타일 박스"):
            st.dataframe(guide["tiles"], width="stretch", hide_index=True)

    st.divider()

    slicer_cols = st.columns([1.4, 1], vertical_alignment="bottom")
    with slicer_cols[0]:
        guide_input_file = st.file_uploader(
            "완성된 가이드 이미지 업로드",
            type=["png", "jpg", "jpeg", "webp"],
            key="tileset_guide_input",
        )
    with slicer_cols[1]:
        skip_transparent_tiles = st.checkbox(
            "완전 투명 타일 제외",
            value=False,
            key="tileset_guide_skip_transparent_tiles",
        )

    if guide_input_file is None:
        st.info("완성된 가이드 이미지를 업로드하면 각 타일 박스만 잘라 PNG로 내보냅니다.")
        return

    try:
        sliced = slice_tileset_guide_image(
            guide_input_file.getvalue(),
            tile_size,
            gap,
            margin,
            file_prefix,
            skip_transparent_tiles,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    source_width, source_height = sliced["source_size"]
    expected_width, expected_height = sliced["expected_size"]
    st.caption(
        f"입력={source_width}x{source_height}px · "
        f"가이드 기준={expected_width}x{expected_height}px · "
        f"추출={len(sliced['tiles'])}개"
    )

    zip_name = f"{safe_stem(file_prefix)}_tiles.zip"
    st.download_button(
        "타일 PNG ZIP 다운로드",
        data=sliced["zip_bytes"],
        file_name=zip_name,
        mime="application/zip",
        key="tileset_guide_zip_download",
        width="stretch",
    )

    with st.expander("개별 PNG"):
        preview_cols = st.columns(4)
        for index, tile in enumerate(sliced["tiles"]):
            with preview_cols[index % 4]:
                st.caption(f"{tile['code']} · {tile['file']}")
                st.image(tile["preview_bytes"], width=96)
                st.download_button(
                    "PNG",
                    data=tile["png_bytes"],
                    file_name=tile["file"],
                    mime="image/png",
                    key=f"tileset_guide_download_{tile['file']}",
                    width="stretch",
                )


st.set_page_config(page_title="Image Cut Fit", layout="wide")

cut_fit_tab, manual_transform_tab, sprite_sheet_builder_tab, tileset_guide_tab = st.tabs(
    [
        "자동 배경 제거/리사이즈",
        "수동 자르기/그리기",
        "스프라이트 시트 생성/복구",
        "타일셋 가이드/타일 추출",
    ]
)
with cut_fit_tab:
    render_cut_fit_tab()
with manual_transform_tab:
    render_manual_transform_tab()
with sprite_sheet_builder_tab:
    render_sprite_sheet_builder_tab()
with tileset_guide_tab:
    render_tileset_guide_tab()
