from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from image_pipeline import (
    apply_padding_and_background,
    crop_to_alpha_bbox,
    image_to_png_bytes,
    load_image,
    remove_background_and_crop,
    remove_background_and_extract_sprites,
    resize_to_target,
)


OUTPUT_DIR = Path(__file__).parent / "outputs"
MODEL_OPTIONS = {
    "u2net": "u2net - 기본",
    "isnet-general-use": "isnet-general-use - 일반 이미지 보존 우선",
    "isnet-anime": "isnet-anime - 일러스트/애니풍",
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
PREVIEW_MAX_WIDTH = 520


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


def checkerboard_preview(image: Image.Image, square_size: int = 16) -> bytes:
    image = image.convert("RGBA")
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


@st.cache_resource(show_spinner=False)
def get_rembg_session(model_name: str):
    from rembg import new_session

    return new_session(model_name)


@st.cache_data(show_spinner=False)
def process_crop(
    image_bytes: bytes,
    alpha_threshold: int,
    model_name: str,
    preserve_interior: bool,
    post_process_mask: bool,
):
    result = remove_background_and_crop(
        image_bytes,
        alpha_threshold=alpha_threshold,
        preserve_interior=preserve_interior,
        post_process_mask=post_process_mask,
        session=get_rembg_session(model_name),
    )
    return {
        "png_bytes": result.png_bytes,
        "preview_bytes": checkerboard_preview(result.cropped),
        "bbox": result.bbox,
        "source_size": result.source_size,
        "cropped_size": result.cropped.size,
        "removed_size": result.removed_background.size,
    }


@st.cache_data(show_spinner=False)
def process_alpha_crop(image_bytes: bytes, alpha_threshold: int):
    source = load_image(image_bytes)
    cropped, bbox = crop_to_alpha_bbox(source, alpha_threshold=alpha_threshold)
    return {
        "png_bytes": image_to_png_bytes(cropped),
        "preview_bytes": checkerboard_preview(cropped),
        "bbox": bbox,
        "source_size": source.size,
        "cropped_size": cropped.size,
        "removed_size": source.size,
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
def build_output_image(
    cropped_png_bytes: bytes,
    width: int,
    height: int,
    resize_mode: str,
    padding: int,
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
        padding=padding,
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


def initialize_image_state(image_id: str) -> None:
    defaults: dict[str, Any] = {
        "model_name": "u2net",
        "alpha_threshold": 16,
        "preserve_interior": True,
        "post_process_mask": True,
        "aspect_locked": True,
        "output_width": 0,
        "output_height": 0,
        "resize_mode": "contain_center",
        "padding": 0,
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
        "resize_mode": st.session_state[image_state_key(image_id, "resize_mode")]
        or "contain_center",
        "padding": int(st.session_state[image_state_key(image_id, "padding")]),
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
    resize_key = image_state_key(image_id, "resize_mode")
    padding_key = image_state_key(image_id, "padding")
    transparent_key = image_state_key(image_id, "transparent_background")
    background_key = image_state_key(image_id, "background_color")

    st.caption("옵션")
    if show_background_controls:
        background_cols = st.columns([1.5, 1.3, 0.9, 0.9], vertical_alignment="bottom")
        with background_cols[0]:
            st.selectbox(
                "모델",
                options=list(MODEL_OPTIONS.keys()),
                format_func=lambda value: MODEL_OPTIONS[value],
                key=image_state_key(image_id, "model_name"),
                help="객체 내부가 많이 지워지면 isnet-general-use를, 일러스트/애니풍이면 isnet-anime을 시도하세요.",
            )
        with background_cols[1]:
            st.slider(
                "Alpha",
                min_value=0,
                max_value=254,
                key=image_state_key(image_id, "alpha_threshold"),
                help="값이 높을수록 거의 투명한 배경 잔여 픽셀을 무시하고 더 강하게 crop합니다.",
            )
        with background_cols[2]:
            st.checkbox(
                "내부 복원",
                key=image_state_key(image_id, "preserve_interior"),
                help="외곽선 안쪽에서 투명해진 영역을 원본 색상으로 다시 채웁니다.",
            )
        with background_cols[3]:
            st.checkbox(
                "마스크",
                key=image_state_key(image_id, "post_process_mask"),
                help="경계를 매끄럽게 할 수 있지만 작은 내부 디테일은 더 사라질 수 있습니다.",
            )
    else:
        st.slider(
            "Alpha",
            min_value=0,
            max_value=254,
            key=image_state_key(image_id, "alpha_threshold"),
            help="값이 높을수록 거의 투명한 배경 잔여 픽셀을 무시하고 더 강하게 crop합니다.",
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
            use_container_width=True,
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
            )
        else:
            st.number_input(
                "세로",
                min_value=1,
                max_value=MAX_OUTPUT_SIZE,
                step=1,
                key=height_key,
            )

    with size_cols[3]:
        resize_mode = st.segmented_control(
            "리사이즈",
            options=list(RESIZE_MODE_OPTIONS.keys()),
            format_func=lambda value: RESIZE_MODE_OPTIONS[value],
            key=resize_key,
            help="비율 유지 옵션은 투명 캔버스 안에 맞춘 뒤 중앙/상단/하단/좌측/우측으로 배치합니다. 늘려서 채우기는 비율을 무시합니다.",
            width="stretch",
        )
        if resize_mode is None:
            st.session_state[resize_key] = "contain_center"

    background_option_cols = st.columns([1, 1, 1], vertical_alignment="bottom")
    with background_option_cols[0]:
        st.number_input(
            "Padding",
            min_value=0,
            max_value=MAX_OUTPUT_SIZE,
            step=1,
            key=padding_key,
            help="현재 Output 이미지 바깥쪽에 추가할 여백(px)입니다.",
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


def process_item_output(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int, int]:
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
    ensure_output_size(item["id"], crop["cropped_size"])
    target_width, target_height, resize_mode = current_target_size(
        item["id"],
        crop["cropped_size"],
    )
    output = build_output_image(
        crop["png_bytes"],
        target_width,
        target_height,
        resize_mode,
        settings["padding"],
        settings["transparent_background"],
        settings["background_color"],
    )
    return crop, output, target_width, target_height


def save_output(item: dict[str, Any], png_bytes: bytes, width: int, height: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = safe_output_name(item["name"], width, height)
    output_path = unique_output_path(OUTPUT_DIR / output_name)
    output_path.write_bytes(png_bytes)
    return output_path


def render_image_card(item: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(f"**{item['name']}**")
        if item.get("kind") == "sprite":
            st.caption(
                f"{item['parent_name']} · sprite #{item['sprite_index']} · "
                f"bbox={item['sheet_bbox']} · area={item['component_area']}"
            )
        else:
            st.caption(f"id={item['digest'][:8]}")

        try:
            with st.spinner(f"{item['name']} 처리 중..."):
                crop, output, target_width, target_height = process_item_output(item)
        except Exception as exc:
            ensure_output_size(item["id"], (1, 1))
            preview_col, controls_col = st.columns([1, 2], vertical_alignment="top")
            with preview_col:
                st.image(item["bytes"], use_container_width=True)
            with controls_col:
                render_compact_controls(
                    item["id"],
                    (1, 1),
                    show_background_controls=item.get("kind") != "sprite",
                )
                st.error(str(exc))
            return

        settings = image_settings(item["id"])
        preview_cols = st.columns(3, vertical_alignment="top")
        with preview_cols[0]:
            st.caption("Original")
            st.image(item["bytes"], use_container_width=True)
        with preview_cols[1]:
            st.caption("Cropped")
            preview_width = min(crop["cropped_size"][0], PREVIEW_MAX_WIDTH)
            st.image(crop["preview_bytes"], width=preview_width)
        with preview_cols[2]:
            st.caption("Output")
            output_preview_width = min(output["size"][0], PREVIEW_MAX_WIDTH)
            st.image(output["preview_bytes"], width=output_preview_width)

        x0, y0, x1, y1 = crop["bbox"]
        cropped_width, cropped_height = crop["cropped_size"]
        source_width, source_height = crop["source_size"]
        trimmed_width = source_width - cropped_width
        trimmed_height = source_height - cropped_height
        st.caption(
            f"source={source_width}x{source_height}px · "
            f"bbox=({x0}, {y0}, {x1}, {y1}) · "
            f"cropped={cropped_width}x{cropped_height}px · "
            f"resized={target_width}x{target_height}px · "
            f"final={output['size'][0]}x{output['size'][1]}px · "
            f"padding={settings['padding']}px · "
            f"background={'transparent' if settings['transparent_background'] else settings['background_color']} · "
            f"mode={RESIZE_MODE_OPTIONS[settings['resize_mode']]} · "
            f"removed={trimmed_width}px width, {trimmed_height}px height"
        )

        render_compact_controls(
            item["id"],
            crop["cropped_size"],
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
                use_container_width=True,
            )
        with action_cols[1]:
            if st.button(
                "저장",
                key=image_state_key(item["id"], "save"),
                use_container_width=True,
            ):
                output_path = save_output(
                    item,
                    output["png_bytes"],
                    final_width,
                    final_height,
                )
                st.success(f"저장됨: {output_path}")


st.set_page_config(page_title="Image Cut Fit", layout="wide")

st.title("Image Cut Fit")

uploaded_files = st.file_uploader(
    "이미지 업로드",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

sprite_sheet_mode = st.toggle(
    "스프라이트 시트 분리",
    value=False,
    help="업로드 이미지를 먼저 배경 제거한 뒤 연결되지 않은 알파 영역을 각각의 스프라이트로 분리합니다.",
)

sheet_settings: dict[str, Any] = {}
if sprite_sheet_mode:
    with st.container(border=True):
        sheet_cols = st.columns([1.5, 1.2, 0.9, 0.9, 1], vertical_alignment="bottom")
        with sheet_cols[0]:
            sheet_settings["model_name"] = st.selectbox(
                "시트 모델",
                options=list(MODEL_OPTIONS.keys()),
                format_func=lambda value: MODEL_OPTIONS[value],
                key="sheet_model_name",
            )
        with sheet_cols[1]:
            sheet_settings["alpha_threshold"] = st.slider(
                "시트 Alpha",
                min_value=0,
                max_value=254,
                value=16,
                key="sheet_alpha_threshold",
            )
        with sheet_cols[2]:
            sheet_settings["preserve_interior"] = st.checkbox(
                "내부 복원",
                value=True,
                key="sheet_preserve_interior",
            )
        with sheet_cols[3]:
            sheet_settings["post_process_mask"] = st.checkbox(
                "마스크",
                value=True,
                key="sheet_post_process_mask",
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
                )
            )

if not uploaded_files:
    st.info("PNG, JPG, JPEG, WebP 이미지를 하나 이상 업로드하세요.")
    st.stop()

uploaded_items: list[dict[str, Any]] = []
for index, uploaded_file in enumerate(uploaded_files):
    image_bytes = uploaded_file.getvalue()
    digest = hashlib.sha1(image_bytes).hexdigest()
    if sprite_sheet_mode:
        try:
            with st.spinner(f"{uploaded_file.name} 스프라이트 분리 중..."):
                sprites = process_sprite_sheet(
                    image_bytes,
                    sheet_settings["alpha_threshold"],
                    sheet_settings["model_name"],
                    sheet_settings["preserve_interior"],
                    sheet_settings["post_process_mask"],
                    sheet_settings["min_area"],
                )
        except Exception as exc:
            st.error(f"{uploaded_file.name}: {exc}")
            continue

        if not sprites:
            st.warning(f"{uploaded_file.name}: 분리된 스프라이트가 없습니다.")
            continue

        for sprite_index, sprite in enumerate(sprites, start=1):
            sprite_bytes = sprite["png_bytes"]
            sprite_digest = hashlib.sha1(sprite_bytes).hexdigest()
            image_id = f"{digest[:12]}_{index}_sprite_{sprite_index}_{sprite_digest[:8]}"
            item = {
                "id": image_id,
                "digest": sprite_digest,
                "name": f"{safe_stem(uploaded_file.name)}_sprite_{sprite_index:02d}.png",
                "bytes": sprite_bytes,
                "kind": "sprite",
                "parent_name": uploaded_file.name,
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
            "name": uploaded_file.name,
            "bytes": image_bytes,
            "kind": "image",
        }
        uploaded_items.append(item)
        initialize_image_state(image_id)

if not uploaded_items:
    st.stop()

active_ids = {item["id"] for item in uploaded_items}
prune_image_state(active_ids)

st.caption(f"{len(uploaded_items)}개 이미지 업로드됨")

for uploaded_item in uploaded_items:
    render_image_card(uploaded_item)

st.divider()
if st.button("모든 이미지 저장", type="primary", use_container_width=True):
    saved_paths: list[Path] = []
    errors: list[str] = []
    progress = st.progress(0, text="일괄 저장 준비 중...")

    for index, item in enumerate(uploaded_items, start=1):
        progress.progress(
            (index - 1) / len(uploaded_items),
            text=f"{item['name']} 처리 중...",
        )
        try:
            _, item_output, item_width, item_height = process_item_output(item)
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
