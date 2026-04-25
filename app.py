from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from image_pipeline import image_to_png_bytes, remove_background_and_crop, resize_to_target


OUTPUT_DIR = Path(__file__).parent / "outputs"
MODEL_OPTIONS = {
    "u2net": "u2net - 기본",
    "isnet-general-use": "isnet-general-use - 일반 이미지 보존 우선",
    "isnet-anime": "isnet-anime - 일러스트/애니풍",
}
RESIZE_MODE_OPTIONS = {
    "contain_center": "비율 유지 중앙",
    "stretch": "늘려서 채우기",
}
MAX_OUTPUT_SIZE = 8192


def safe_output_name(original_name: str, width: int, height: int) -> str:
    stem = Path(original_name).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return f"{stem or 'image'}_transparent_{width}x{height}.png"


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
            background.paste(color, (x, y, min(x + square_size, width), min(y + square_size, height)))

    background.alpha_composite(image)
    from io import BytesIO

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
def build_output_image(
    cropped_png_bytes: bytes,
    width: int,
    height: int,
    resize_mode: str,
):
    cropped = Image.open(BytesIO(cropped_png_bytes)).convert("RGBA")
    output = resize_to_target(
        cropped,
        width=width,
        height=height,
        mode=resize_mode,
    )
    return {
        "png_bytes": image_to_png_bytes(output),
        "preview_bytes": checkerboard_preview(output),
        "size": output.size,
    }


def initialize_output_size(image_digest: str, cropped_size: tuple[int, int]) -> None:
    if st.session_state.get("active_image_digest") == image_digest:
        return

    st.session_state["active_image_digest"] = image_digest
    st.session_state["aspect_locked"] = True
    st.session_state["output_width"] = cropped_size[0]
    st.session_state["output_height"] = cropped_size[1]


def toggle_aspect_lock() -> None:
    st.session_state["aspect_locked"] = not st.session_state.get("aspect_locked", True)


def render_output_size_controls(cropped_size: tuple[int, int]) -> tuple[int, int, str]:
    cropped_width, cropped_height = cropped_size
    aspect_ratio = cropped_width / cropped_height

    st.divider()
    st.subheader("출력 크기")

    size_cols = st.columns([1, 0.24, 1])
    with size_cols[0]:
        target_width = int(
            st.number_input(
                "가로",
                min_value=1,
                max_value=MAX_OUTPUT_SIZE,
                value=int(st.session_state["output_width"]),
                step=1,
            )
        )

    locked = st.session_state.get("aspect_locked", True)
    with size_cols[1]:
        st.write("")
        st.button(
            " ",
            key="aspect_lock_button",
            icon=":material/lock:" if locked else ":material/lock_open:",
            help="가로/세로 비율 잠금",
            on_click=toggle_aspect_lock,
            use_container_width=True,
        )

    if locked:
        target_height = max(1, min(MAX_OUTPUT_SIZE, round(target_width / aspect_ratio)))
        st.session_state["output_height"] = target_height
        with size_cols[2]:
            st.number_input(
                "세로",
                min_value=1,
                max_value=MAX_OUTPUT_SIZE,
                value=target_height,
                step=1,
                disabled=True,
            )
    else:
        with size_cols[2]:
            target_height = int(
                st.number_input(
                    "세로",
                    min_value=1,
                    max_value=MAX_OUTPUT_SIZE,
                    value=int(st.session_state["output_height"]),
                    step=1,
                )
            )

    resize_mode = st.segmented_control(
        "리사이즈 방식",
        options=list(RESIZE_MODE_OPTIONS.keys()),
        default="contain_center",
        format_func=lambda value: RESIZE_MODE_OPTIONS[value],
        help="늘려서 채우기는 비율을 무시하고, 비율 유지 중앙은 투명 캔버스 중앙에 배치합니다.",
        width="stretch",
    )
    if resize_mode is None:
        resize_mode = "contain_center"

    return target_width, target_height, resize_mode


st.set_page_config(page_title="Background Cropper", layout="wide")

st.title("Background Cropper")

with st.sidebar:
    model_name = st.selectbox(
        "배경 제거 모델",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda value: MODEL_OPTIONS[value],
        index=0,
        help="객체 내부가 많이 지워지면 isnet-general-use를, 일러스트/애니풍이면 isnet-anime을 시도하세요.",
    )
    alpha_threshold = st.slider(
        "Alpha threshold",
        min_value=0,
        max_value=254,
        value=16,
        help="값이 높을수록 거의 투명한 배경 잔여 픽셀을 무시하고 더 강하게 crop합니다.",
    )
    preserve_interior = st.checkbox(
        "객체 내부 색상 복원",
        value=True,
        help="외곽선 안쪽에서 투명해진 영역을 원본 색상으로 다시 채웁니다.",
    )
    post_process_mask = st.checkbox(
        "마스크 후처리",
        value=True,
        help="경계를 매끄럽게 할 수 있지만 작은 내부 디테일은 더 사라질 수 있습니다.",
    )

uploaded_file = st.file_uploader(
    "이미지 업로드",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("PNG, JPG, JPEG, WebP 이미지를 업로드하세요.")
    st.stop()

image_bytes = uploaded_file.getvalue()
image_digest = hashlib.sha1(image_bytes).hexdigest()

try:
    with st.spinner("배경 제거 및 crop 처리 중..."):
        processed = process_crop(
            image_bytes,
            alpha_threshold,
            model_name,
            preserve_interior,
            post_process_mask,
        )
except Exception as exc:
    st.error(str(exc))
    st.stop()

initialize_output_size(image_digest, processed["cropped_size"])
with st.sidebar:
    target_width, target_height, resize_mode = render_output_size_controls(
        processed["cropped_size"]
    )

try:
    output = build_output_image(
        processed["png_bytes"],
        target_width,
        target_height,
        resize_mode,
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

output_name = safe_output_name(uploaded_file.name, target_width, target_height)

left, middle, right = st.columns(3)

with left:
    st.subheader("Original")
    st.image(image_bytes, use_container_width=True)

with middle:
    st.subheader("Cropped")
    preview_width = min(processed["cropped_size"][0], 520)
    st.image(processed["preview_bytes"], width=preview_width)

with right:
    st.subheader("Output")
    output_preview_width = min(output["size"][0], 520)
    st.image(output["preview_bytes"], width=output_preview_width)

x0, y0, x1, y1 = processed["bbox"]
width, height = processed["cropped_size"]
source_width, source_height = processed["source_size"]
trimmed_width = source_width - width
trimmed_height = source_height - height
st.caption(
    f"source={source_width}x{source_height}px · "
    f"bbox=({x0}, {y0}, {x1}, {y1}) · "
    f"cropped={width}x{height}px · "
    f"output={target_width}x{target_height}px · "
    f"mode={RESIZE_MODE_OPTIONS[resize_mode]} · "
    f"removed={trimmed_width}px width, {trimmed_height}px height"
)

download_col, save_col = st.columns([1, 1])

with download_col:
    st.download_button(
        "PNG 다운로드",
        data=output["png_bytes"],
        file_name=output_name,
        mime="image/png",
        use_container_width=True,
    )

with save_col:
    if st.button("outputs 폴더에 저장", use_container_width=True):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / output_name
        output_path.write_bytes(output["png_bytes"])
        st.success(f"저장됨: {output_path}")
