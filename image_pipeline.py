from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageOps


@dataclass(frozen=True)
class CroppedImageResult:
    cropped: Image.Image
    removed_background: Image.Image
    source_size: tuple[int, int]
    bbox: tuple[int, int, int, int]

    @property
    def png_bytes(self) -> bytes:
        return image_to_png_bytes(self.cropped)


@dataclass(frozen=True)
class SpriteComponent:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    area: int

    @property
    def png_bytes(self) -> bytes:
        return image_to_png_bytes(self.image)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGBA")


def remove_background(
    image: Image.Image,
    session: Any | None = None,
    *,
    post_process_mask: bool = False,
) -> Image.Image:
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError(
            "rembg가 설치되어 있지 않습니다. `pip install -r requirements.txt`를 실행하세요."
        ) from exc

    kwargs = {
        "post_process_mask": post_process_mask,
    }
    if session is not None:
        kwargs["session"] = session
    result = remove(image, **kwargs)

    if isinstance(result, Image.Image):
        return result.convert("RGBA")

    if isinstance(result, bytes):
        return Image.open(BytesIO(result)).convert("RGBA")

    raise TypeError(f"지원하지 않는 rembg 결과 타입입니다: {type(result)!r}")


def alpha_bbox(image: Image.Image, alpha_threshold: int = 0) -> tuple[int, int, int, int]:
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = image.getchannel("A")
    if alpha_threshold > 0:
        alpha = alpha.point(lambda value: 255 if value > alpha_threshold else 0)

    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("투명하지 않은 픽셀을 찾지 못했습니다.")
    return bbox


def crop_to_alpha_bbox(
    image: Image.Image,
    alpha_threshold: int = 0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    bbox = alpha_bbox(image, alpha_threshold=alpha_threshold)
    return image.crop(bbox), bbox


def rotate_image(
    image: Image.Image,
    *,
    degrees: float = 0,
    alpha_threshold: int = 0,
) -> Image.Image:
    image = image.convert("RGBA")
    degrees = float(degrees)
    if math.isclose(degrees % 360, 0, abs_tol=1e-9):
        return image

    rotated = image.rotate(
        -degrees,
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )

    try:
        cropped, _ = crop_to_alpha_bbox(rotated, alpha_threshold=alpha_threshold)
        return cropped
    except ValueError:
        return rotated


def apply_erase_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    mask = mask.convert("RGBA")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)

    image_pixels = np.array(image)
    mask_alpha = np.array(mask.getchannel("A"))
    image_pixels[mask_alpha > 0, 3] = 0
    return Image.fromarray(image_pixels, mode="RGBA")


def extract_connected_components(
    image: Image.Image,
    *,
    alpha_threshold: int = 16,
    min_area: int = 64,
) -> list[SpriteComponent]:
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "스프라이트 분리에는 scipy가 필요합니다. `pip install -r requirements.txt`를 실행하세요."
        ) from exc

    alpha = np.array(image.getchannel("A"))
    foreground = alpha > alpha_threshold
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, component_count = ndimage.label(foreground, structure=structure)
    slices = ndimage.find_objects(labels)

    components: list[SpriteComponent] = []
    for label_index, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue

        y_slice, x_slice = component_slice
        component_mask = labels[component_slice] == label_index
        area = int(component_mask.sum())
        if area < min_area:
            continue

        bbox = (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)
        components.append(
            SpriteComponent(
                image=image.crop(bbox),
                bbox=bbox,
                area=area,
            )
        )

    return sorted(components, key=lambda component: (component.bbox[1], component.bbox[0]))


def resize_to_target(
    image: Image.Image,
    *,
    width: int,
    height: int,
    mode: str,
) -> Image.Image:
    if width < 1 or height < 1:
        raise ValueError("출력 가로/세로는 1px 이상이어야 합니다.")

    image = image.convert("RGBA")
    target_size = (int(width), int(height))

    if mode == "stretch":
        return image.resize(target_size, Image.Resampling.LANCZOS)

    contain_offsets = {
        "contain_center": ("center", "center"),
        "contain_top": ("center", "start"),
        "contain_bottom": ("center", "end"),
        "contain_left": ("start", "center"),
        "contain_right": ("end", "center"),
    }
    if mode in contain_offsets:
        fitted = ImageOps.contain(image, target_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
        horizontal, vertical = contain_offsets[mode]
        extra_width = target_size[0] - fitted.width
        extra_height = target_size[1] - fitted.height
        x_offset = {
            "start": 0,
            "center": extra_width // 2,
            "end": extra_width,
        }[horizontal]
        y_offset = {
            "start": 0,
            "center": extra_height // 2,
            "end": extra_height,
        }[vertical]
        offset = (x_offset, y_offset)
        canvas.alpha_composite(fitted, dest=offset)
        return canvas

    raise ValueError(f"지원하지 않는 리사이즈 모드입니다: {mode}")


def apply_padding_and_background(
    image: Image.Image,
    *,
    padding: int = 0,
    transparent_background: bool = True,
    background_color: str = "#000000",
) -> Image.Image:
    if padding < 0:
        raise ValueError("padding은 0px 이상이어야 합니다.")

    image = image.convert("RGBA")
    width, height = image.size
    canvas_size = (width + padding * 2, height + padding * 2)

    if transparent_background:
        fill = (0, 0, 0, 0)
    else:
        red, green, blue = ImageColor.getrgb(background_color)
        fill = (red, green, blue, 255)

    canvas = Image.new("RGBA", canvas_size, fill)
    canvas.alpha_composite(image, dest=(padding, padding))
    return canvas


def restore_enclosed_transparency(
    source: Image.Image,
    removed_background: Image.Image,
    *,
    alpha_threshold: int = 16,
) -> Image.Image:
    if source.size != removed_background.size:
        raise ValueError("원본 이미지와 배경 제거 이미지의 크기가 다릅니다.")

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "객체 내부 복원에는 scipy가 필요합니다. `pip install -r requirements.txt`를 실행하세요."
        ) from exc

    source_rgba = source.convert("RGBA")
    removed_rgba = removed_background.convert("RGBA")

    alpha = np.array(removed_rgba.getchannel("A"))
    foreground = alpha > alpha_threshold
    filled_foreground = ndimage.binary_fill_holes(foreground)
    enclosed_holes = filled_foreground & ~foreground

    if not enclosed_holes.any():
        return removed_rgba

    restored = np.array(removed_rgba)
    source_pixels = np.array(source_rgba)
    restored[enclosed_holes, :3] = source_pixels[enclosed_holes, :3]
    restored[enclosed_holes, 3] = 255
    return Image.fromarray(restored, mode="RGBA")


def remove_background_and_crop(
    image_bytes: bytes,
    *,
    alpha_threshold: int = 0,
    preserve_interior: bool = True,
    post_process_mask: bool = False,
    session: Any | None = None,
) -> CroppedImageResult:
    source = load_image(image_bytes)
    removed = remove_background(
        source,
        session=session,
        post_process_mask=post_process_mask,
    )
    if preserve_interior:
        removed = restore_enclosed_transparency(
            source,
            removed,
            alpha_threshold=alpha_threshold,
        )
    cropped, bbox = crop_to_alpha_bbox(removed, alpha_threshold=alpha_threshold)
    return CroppedImageResult(
        cropped=cropped,
        removed_background=removed,
        source_size=source.size,
        bbox=bbox,
    )


def remove_background_and_extract_sprites(
    image_bytes: bytes,
    *,
    alpha_threshold: int = 16,
    min_area: int = 64,
    preserve_interior: bool = True,
    post_process_mask: bool = False,
    session: Any | None = None,
) -> list[SpriteComponent]:
    source = load_image(image_bytes)
    removed = remove_background(
        source,
        session=session,
        post_process_mask=post_process_mask,
    )
    if preserve_interior:
        removed = restore_enclosed_transparency(
            source,
            removed,
            alpha_threshold=alpha_threshold,
        )
    return extract_connected_components(
        removed,
        alpha_threshold=alpha_threshold,
        min_area=min_area,
    )
