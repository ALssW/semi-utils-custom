import os.path
import sys
from abc import ABC
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import ImageFont, Image, ImageDraw

from core.configs import fonts_dir
from processor.core import PipelineContext, ImageProcessor, Direction, _parse_color

BASE_FONT_SIZE = 512


def _font_size_for_target(target_height: int) -> int:
    """按目标像素高度选择绘制字号，控制缩小倍数，避免小图文字被从 512px 一次压糊。"""
    target_height = max(1, int(target_height))
    return int(max(64, min(BASE_FONT_SIZE, target_height * 4)))


def load_font(font_path: str, size: int = BASE_FONT_SIZE):
    size = max(1, int(size))
    try:
        if font_path:
            font_file = Path(font_path)

            # 如果是相对路径，转换为基于执行文件所在目录的绝对路径
            if not font_file.is_absolute():
                font_file = fonts_dir / font_path

            return ImageFont.truetype(str(font_file), size)
        else:
            # 尝试常见系统字体
            for fallback in [fonts_dir / "AlibabaPuHuiTi-2-45-Light.otf", "arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
                try:
                    return ImageFont.truetype(fallback, size)
                except OSError:
                    continue
            else:
                return ImageFont.load_default()
    except OSError:
        return ImageFont.load_default()


@dataclass
class TextSegment:
    text: str
    font_path: Optional[str] = None
    height: int = 100
    is_bold: bool = False
    color: str = "black"
    trim: bool = False

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    @staticmethod
    def from_dict(data: dict):
        return TextSegment(
            text=data.get("text"),
            font_path=data.get("font_path", None),
            height=int(data.get("height", 100)),
            color=data.get("color", "black"),
            is_bold=data.get("is_bold", False),
            trim=data.get("trim", False),
        )

    @staticmethod
    def from_dicts(data: List[dict]):
        return [TextSegment.from_dict(datum) for datum in data]


class Generator(ImageProcessor, ABC):

    def category(self) -> str:
        return "generator"


class SolidColorGenerator(Generator):

    def process(self, ctx: PipelineContext):
        width, height = ctx.getint("width"), ctx.getint("height")
        color = ctx.get("color")
        image = Image.new("RGBA", (width, height), color)
        ctx.update_buffer([image]).success()

    def name(self) -> str:
        return "solid_color"


class InterpolateMethod(Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"


# ============ 缓动函数（处理 t 值）============
def _easing_linear(t: np.ndarray) -> np.ndarray:
    """线性"""
    return t


def _easing_ease_in(t: np.ndarray) -> np.ndarray:
    """缓入"""
    return t * t


def _easing_ease_out(t: np.ndarray) -> np.ndarray:
    """缓出"""
    return 1 - (1 - t) ** 2


def _easing_ease_in_out(t: np.ndarray) -> np.ndarray:
    """缓入缓出"""
    return np.where(t < 0.5, 2 * t * t, 1 - (-2 * t + 2) ** 2 / 2)


EASING_FUNCTIONS = {
    InterpolateMethod.LINEAR: _easing_linear,
    InterpolateMethod.EASE_IN: _easing_ease_in,
    InterpolateMethod.EASE_OUT: _easing_ease_out,
    InterpolateMethod.EASE_IN_OUT: _easing_ease_in_out,
}


# ============ NumPy 加速绘制 ============
def _draw_gradient_numpy(
        width: int,
        height: int,
        start_rgba: tuple,
        end_rgba: tuple,
        direction: Direction,
        method: InterpolateMethod = InterpolateMethod.LINEAR
) -> Image.Image:
    """NumPy 加速的渐变绘制"""

    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    # 计算进度 t
    if direction == Direction.HORIZONTAL:
        t = xx / (width - 1) if width > 1 else np.zeros_like(xx, dtype=float)
    elif direction == Direction.VERTICAL:
        t = yy / (height - 1) if height > 1 else np.zeros_like(yy, dtype=float)
    elif direction == Direction.DIAGONAL:
        t = (xx + yy) / (width + height - 2) if (width + height) > 2 else np.zeros_like(xx, dtype=float)
    elif direction == Direction.RADIAL:
        cx, cy = width / 2, height / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        t = np.clip(dist / max_dist, 0, 1)
    else:
        t = np.zeros((height, width), dtype=float)

    # 应用缓动函数
    easing_func = EASING_FUNCTIONS.get(method, _easing_linear)
    t = easing_func(t.astype(float))

    # 向量化颜色插值
    start = np.array(start_rgba, dtype=float)
    end = np.array(end_rgba, dtype=float)

    # 扩展维度 (height, width) -> (height, width, 1)
    t = t[:, :, np.newaxis]

    # 插值计算
    pixels = start + (end - start) * t
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    return Image.fromarray(pixels, mode='RGBA')


class GradientColorGenerator(Generator):
    def process(self, ctx: PipelineContext):
        width, height = ctx.get("width"), ctx.get("height")
        start_color = ctx.get("start_color")
        end_color = ctx.get("end_color")
        direction = ctx.getenum("direction", Direction.HORIZONTAL, Direction)  # horizontal, vertical, diagonal
        method = ctx.getenum("interpolate_method", InterpolateMethod.LINEAR, InterpolateMethod)

        start_rgba = _parse_color(start_color)
        end_rgba = _parse_color(end_color)

        image = _draw_gradient_numpy(
            width, height,
            start_rgba, end_rgba,
            direction, method
        )

        ctx.update_buffer([image]).save_buffer(self.name()).success()

    def name(self) -> str:
        return "gradient_color"


class RichTextGenerator(Generator):
    @staticmethod
    def generate(segment: TextSegment) -> Image.Image:
        """
        按目标高度生成单段文字图。
        :param segment: 文本片段配置
        :return: RGBA 文字图像
        """
        font = load_font(segment.font_path, _font_size_for_target(segment.height))

        # 获取文本尺寸
        metrics = font.getmetrics()
        text = ' ' if not segment.text or segment.text == '' else segment.text
        bbox = font.getbbox(text)
        # 创建透明画布
        image = Image.new('RGBA', (int(bbox[2] - bbox[0]), metrics[0] + abs(metrics[1])), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        # 直接绘制文本，fill 支持 RGBA（第四通道为不透明度）
        draw.text((0, 0), text, font=font, fill=_parse_color(segment.color))

        # 使用 start_process 处理图片，解耦对 Filter 的直接依赖
        pipeline = [
            {
                "processor_name": "trim",
                "trim_top": segment.trim,
                "trim_bottom": segment.trim,
                "save_buffer": False,
            },
            {
                "processor_name": "resize",
                "height": segment.height * 1.13 if segment.is_bold else segment.height,
                "save_buffer": False,
            }
        ]
        from processor.core import start_process
        return start_process(pipeline, input_path=None, output_path=None, initial_buffer=[image])

    @staticmethod
    def render_on_baseline(segment: TextSegment, font_size: int = BASE_FONT_SIZE) -> tuple:
        """
        在 BASE_FONT_SIZE 下按基线绘制文本。
        使用 anchor=ls，保证不同字体共享同一基线坐标系。
        :param segment: 文本片段配置
        :return: (图像, 基线距顶部的距离)
        """
        font = load_font(segment.font_path, font_size)
        ascent, descent = font.getmetrics()
        text = segment.text
        if not text:
            return Image.new('RGBA', (0, 0), (0, 0, 0, 0)), ascent

        bbox = font.getbbox(text)
        width = max(1, int(bbox[2] - bbox[0]))
        height = max(1, ascent + abs(descent))
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        color = _parse_color(segment.color)
        # ls = left + baseline，基线位于 ascent 处
        try:
            draw.text((-bbox[0], ascent), text, font=font, fill=color, anchor='ls')
        except TypeError:
            draw.text((-bbox[0], 0), text, font=font, fill=color)
        return image, ascent

    def process(self, ctx: PipelineContext):
        """
        生成单段富文本图层。
        :param ctx: 管道上下文，字段与 TextSegment 一致
        """
        img = RichTextGenerator.generate(TextSegment.from_dict(ctx))
        ctx.update_buffer([img]).save_buffer(self.name()).success()

    def name(self) -> str:
        return "rich_text"


class MultiRichTextGenerator(Generator):
    @staticmethod
    def _ink_bbox(img: Image.Image):
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img.getchannel('A').getbbox()

    def process(self, ctx: PipelineContext):
        """
        将多段富文本按同一基线拼接后缩放到目标高度。
        :param ctx: 管道上下文，需包含 text_segments、height
        """
        text_segments: List[TextSegment] = TextSegment.from_dicts(ctx.get("text_segments"))
        text_spacing = ctx.getint("text_spacing")
        height = int(float(ctx.get("height", 100)))
        any_bold = any(seg.is_bold for seg in text_segments)

        # 1) 各段按基线绘制
        font_px = _font_size_for_target(height)
        raw_parts: List[tuple] = []  # (image, font_path)
        for segment in text_segments:
            if not segment.text:
                continue
            img, _ = RichTextGenerator.render_on_baseline(segment, font_px)
            if img.width == 0 or img.height == 0:
                continue
            raw_parts.append((img, segment.font_path or ''))

        if not raw_parts:
            ctx.update_buffer([Image.new('RGBA', (0, 0), (0, 0, 0, 0))]).success()
            return

        ref_font = raw_parts[0][1]

        # 2) 以首段墨迹高度为字帽高参考，统一各段视觉高度（解决 ℤ 等符号字体偏大）
        ref_box = self._ink_bbox(raw_parts[0][0])
        ref_ink_h = (ref_box[3] - ref_box[1]) if ref_box else raw_parts[0][0].height

        unified: List[tuple] = []  # (image, font_path)
        for img, font_path in raw_parts:
            box = self._ink_bbox(img)
            if not box or ref_ink_h <= 0:
                unified.append((img, font_path))
                continue
            ink_h = box[3] - box[1]
            if ink_h <= 0:
                unified.append((img, font_path))
                continue
            if abs(ink_h - ref_ink_h) > max(2, ref_ink_h * 0.03):
                scale = ref_ink_h / ink_h
                new_w = max(1, int(round(img.width * scale)))
                new_h = max(1, int(round(img.height * scale)))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            unified.append((img, font_path))

        # 3) 按墨迹底边对齐（大写字母视觉基线）
        ink_boxes = []
        for img, _ in unified:
            box = self._ink_bbox(img)
            ink_boxes.append(box if box else (0, 0, img.width, img.height))

        max_below = max(img.height - box[3] for (img, _), box in zip(unified, ink_boxes))
        max_above = max(box[3] for box in ink_boxes)
        optical_pad = max(2, ref_ink_h // 16)
        canvas_h = max_above + max_below + optical_pad

        spacing_base = 0
        if text_spacing and height > 0:
            spacing_base = max(0, int(round(text_spacing * canvas_h / height)))

        total_w = sum(img.width for img, _ in unified) + spacing_base * (len(unified) - 1)
        canvas = Image.new('RGBA', (max(1, total_w), max(1, canvas_h)), (0, 0, 0, 0))

        ink_bottom_y = max_above
        ref_ink_h_final = ink_boxes[0][3] - ink_boxes[0][1]
        x = 0
        for (img, font_path), box in zip(unified, ink_boxes):
            y = ink_bottom_y - box[3]
            # 符号字体相对正文字体做轻微上移，抵消双线字母的视觉下沉
            if font_path != ref_font:
                y -= max(1, int(round(ref_ink_h_final * 0.08)))
            canvas.paste(img, (x, max(0, y)), img)
            x += img.width + spacing_base

        # 4) 整体缩放
        target_h = int(height * 1.13) if any_bold else height
        from processor.core import start_process
        result = start_process(
            [{"processor_name": "resize", "height": target_h, "save_buffer": False}],
            initial_buffer=[canvas],
        )
        ctx.update_buffer([result]).save_buffer(self.name()).success()

    def name(self) -> str:
        return "multi_rich_text"


class ImageLoader(Generator):
    def process(self, ctx: PipelineContext):
        if isinstance(ctx.get('path'), str):
            ctx.update_buffer([Image.open(ctx.get('path'))]).success()
        elif isinstance(ctx.get('path'), list):
            ctx.update_buffer([Image.open(path) for path in ctx.get('path')])

    def name(self) -> str:
        return "image"
