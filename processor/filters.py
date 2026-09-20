import json
import re
from abc import ABC
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core.util import get_exif
from processor.core import ImageProcessor, PipelineContext, start_process, get_processor
from processor.types import Alignment


class FilterProcessor(ImageProcessor, ABC):
    def category(self) -> str:
        return "filter"


class BlurFilter(FilterProcessor):
    def process(self, ctx: PipelineContext):
        radius = ctx.getint("blur_radius", 5)

        buffer = []
        for img in ctx.get_buffer():
            if img.mode != "RGB":
                img = img.convert("RGB")
            ret_img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            buffer.append(ret_img)
        ctx.update_buffer(buffer).save_buffer(self.name()).success()

    def name(self) -> str:
        return "blur"


class ResizeFilter(FilterProcessor):
    def process(self, ctx: PipelineContext):
        width, height = ctx.get("width"), ctx.get("height")
        scale = ctx.get("scale")

        buffer = []
        for img in ctx.get_buffer():
            if width and height:
                target_size = (int(width), int(height))
            else:
                if width:
                    scale_f = float(width) / img.width
                elif height:
                    scale_f = float(height) / img.height
                elif scale:
                    scale_f = float(scale)
                else:
                    ctx.set("success", False)
                    return
                target_size = (int(img.width * scale_f), int(img.height * scale_f))

            ret_img = img.resize(target_size, resample=Image.Resampling.LANCZOS)
            buffer.append(ret_img)
        ctx.update_buffer(buffer).save_buffer(self.name()).success()

    def name(self) -> str:
        return "resize"


class TrimFilter(FilterProcessor):
    threshold: float = 10.0,
    padding: int = 0

    def process(self, ctx: PipelineContext):
        buffer = []
        for image in ctx.get_buffer():
            if image.height * image.width == 0:
                continue
            bbox = self.get_foreground_bbox(image, trim_left=ctx.get("trim_left", True),
                                            trim_right=ctx.get("trim_right", True),
                                            trim_top=ctx.get("trim_top", True),
                                            trim_bottom=ctx.get("trim_bottom", True))
            buffer.append(image.crop(bbox))
        ctx.update_buffer(buffer).save_buffer(self.name()).success()

    def name(self) -> str:
        return "trim"

    def _get_background_color(self, img_array: np.ndarray) -> np.ndarray:
        """取四角像素均值作为背景色"""
        corners = np.array([
            img_array[0, 0],  # 左上角
            img_array[0, -1],  # 右上角
            img_array[-1, 0],  # 左下角
            img_array[-1, -1]  # 右下角
        ])
        return np.mean(corners, axis=0)

    def _shrink_bbox(
            self,
            diff: np.ndarray,
            threshold: float,
            width: int,
            height: int
    ) -> Tuple[int, int, int, int]:
        """
        从四个方向向内收缩边界框

        Args:
            diff: 每个像素与背景的差异矩阵 shape: (height, width)
            threshold: 差异阈值
            width: 图像宽度
            height: 图像高度

        Returns:
            (left, right, top, bottom) 收缩后的边界
        """
        # 判断每个像素是否超过阈值（与背景有明显差异）
        exceeds = diff > threshold

        # 统计每列是否存在超过阈值的像素
        col_exceeds = np.any(exceeds, axis=0)  # shape: (width,)
        # 统计每行是否存在超过阈值的像素
        row_exceeds = np.any(exceeds, axis=1)  # shape: (height,)

        # 如果整张图都是背景（没有前景），返回原始边界
        if not np.any(col_exceeds):
            return 0, width, 0, height

        # 从左→右扫描：找到第一个超过阈值的列（argmax 返回第一个 True 的索引）
        # 从右→左扫描：反转后找第一个 True，再换算回原索引
        left = int(np.argmax(col_exceeds))
        right = int(width - np.argmax(col_exceeds[::-1]))
        top = int(np.argmax(row_exceeds))
        bottom = int(height - np.argmax(row_exceeds[::-1]))

        return left, right, top, bottom

    def get_foreground_bbox(
            self,
            image: Image.Image,
            threshold: float = 10.0,
            padding: int = 0,
            trim_left: bool = True,
            trim_right: bool = True,
            trim_top: bool = True,
            trim_bottom: bool = True,
    ) -> Tuple[int, int, int, int]:
        img_array = np.array(image, dtype=np.float32)

        # 处理灰度图（2D → 3D）
        if img_array.ndim == 2:
            img_array = img_array[:, :, np.newaxis]

        height, width, channels = img_array.shape

        # ===== 第一步：取四角像素均值作为背景色 =====
        background_color = self._get_background_color(img_array)

        # ===== 第二步：计算每个像素与背景的差异 =====
        diff = np.sqrt(np.sum((img_array - background_color) ** 2, axis=-1))

        # ===== 第三步：从四个方向向内扫描，收缩边界框 =====
        left, right, top, bottom = self._shrink_bbox(diff, threshold, width, height)

        if not trim_left:
            left = 0
        if not trim_right:
            right = width
        if not trim_top:
            top = 0
        if not trim_bottom:
            bottom = height
        # ===== 第四步：应用 padding 并确保边界合法 =====
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)

        return left, top, right, bottom


class MarginFilter(FilterProcessor):

    def process(self, ctx: PipelineContext):
        left_margin = ctx.getint("left_margin", 0)
        right_margin = ctx.getint("right_margin", 0)
        top_margin = ctx.getint("top_margin", 0)
        bottom_margin = ctx.getint("bottom_margin", 0)
        color = ctx.get("margin_color", "white")

        buffer = []
        for img in ctx.get_buffer():
            # 获取原图尺寸
            original_width, original_height = img.size

            # 计算新画布尺寸
            new_width = original_width + left_margin + right_margin
            new_height = original_height + top_margin + bottom_margin

            # 创建新画布，填充指定颜色
            new_img = Image.new(img.mode, (new_width, new_height), color)

            # 计算偏移量（原图粘贴位置）
            offset_x = left_margin
            offset_y = top_margin

            # 将原图粘贴到新画布上
            new_img.paste(img, (offset_x, offset_y))
            buffer.append(new_img)

        ctx.update_buffer(buffer).save_buffer(self.name()).success()

    def name(self) -> str:
        return "margin"


class MarginWithRatioFilter(FilterProcessor):
    ratio_pattern = re.compile('[0-9.]+:[0-9.]+')
    ratio_threshold = 0.01

    def process(self, ctx: PipelineContext):
        buffer = ctx.get_buffer()
        if not buffer:
            return
        real_ratio = 1. * int(ctx.get_exif().get('ImageWidth')) / int(ctx.get_exif().get('ImageHeight'))
        if 'ratio' in ctx and MarginWithRatioFilter.ratio_pattern.match(ctx.get("ratio")):
            ratio_w, ratio_h = ctx.get("ratio").split(':')
            real_ratio = 1. * float(ratio_w) / float(ratio_h)
        img = buffer[0]
        cur_ratio = 1. * img.width / img.height
        if cur_ratio - real_ratio > MarginWithRatioFilter.ratio_threshold:
            # 图片太宽, 增加高度
            new_h = int(img.width / real_ratio)
            pad_vertical = new_h - img.height
            ctx.set('top_margin', pad_vertical / 2)
            ctx.set('bottom_margin', pad_vertical - pad_vertical / 2)
        elif cur_ratio - real_ratio < MarginWithRatioFilter.ratio_threshold:
            # 图片太窄, 增加宽度
            new_w = int(img.height * real_ratio)
            pad_horizontal = new_w - img.width
            ctx.set('left_margin', pad_horizontal / 2)
            ctx.set('right_margin', pad_horizontal - pad_horizontal / 2)
        else:
            return
        MarginFilter().process(ctx)
        ctx.save_buffer(self.name()).success()

    def name(self) -> str:
        return "margin_with_ratio"


class WatermarkFilter(FilterProcessor):
    @staticmethod
    def _paste(canvas: Image.Image, layer: Image.Image, xy: tuple, with_shadow: bool = False):
        """粘贴图层；overlay 模式下加轻微阴影，避免浅色背景上看不清。"""
        if layer is None or layer.width == 0 or layer.height == 0:
            return
        mask = layer if layer.mode == 'RGBA' else None
        if with_shadow and layer.mode == 'RGBA':
            alpha = layer.getchannel('A')
            shadow = Image.new('RGBA', layer.size, (0, 0, 0, 0))
            shadow.putalpha(alpha.point(lambda a: int(a * 0.55)))
            sx, sy = xy[0] + max(1, layer.height // 40), xy[1] + max(1, layer.height // 40)
            canvas.paste(shadow, (sx, sy), shadow)
        canvas.paste(layer, xy, mask=mask)

    @staticmethod
    def _empty_image():
        return Image.new("RGBA", (0, 0), (0, 0, 0, 0))

    def _render_text_slot(self, cfg, default_text_height: int):
        if not cfg:
            return self._empty_image()
        if isinstance(cfg, list):
            for item in cfg:
                if isinstance(item, dict) and item.get("processor_name") in ("rich_text", "multi_rich_text") and "height" not in item:
                    item["height"] = default_text_height
            return start_process(cfg)
        if isinstance(cfg, dict) and "height" not in cfg:
            cfg["height"] = default_text_height
        return start_process([cfg])

    def process(self, ctx: PipelineContext):
        img = ctx.get_buffer()[0]
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        # overlay=true：水印叠在图片内部，不扩展边框
        overlay_raw = ctx.get("overlay", False)
        if isinstance(overlay_raw, str):
            overlay = overlay_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            overlay = bool(overlay_raw)
        color = ctx.get("color", "white")
        delimiter_color = ctx.getcolor("delimiter_color", (0, 0, 0, 255))
        delimiter_width = ctx.getint("delimiter_width", int(img.width * (.002 if overlay else .003)))
        right_alignment = ctx.getenum("right_alignment", Alignment.RIGHT, Alignment)
        # block_align: left|right —— logo+左侧文本块整体靠左或靠右
        block_align_raw = str(ctx.get("block_align", "left")).strip().lower()
        block_align_right = block_align_raw in {"right", "end"}

        if overlay:
            left_margin = ctx.getint("left_margin", int(img.width * .03))
            right_margin = ctx.getint("right_margin", int(img.width * .03))
            top_margin = 0
            bottom_margin = 0
            padding = ctx.getint("padding", int(img.height * .03))
            default_text_height = ctx.getint("text_height", max(int(img.height * .015), 12))
            middle_spacing = ctx.getint("middle_spacing", max(int(img.height * .004), 2))
            common_spacing = ctx.getint("common_spacing", max(int(img.width * .01), 4))
        else:
            left_margin = ctx.getint("left_margin", 0)
            right_margin = ctx.getint("right_margin", 0)
            top_margin = ctx.getint("top_margin", 0)
            bottom_margin = ctx.getint("bottom_margin", int(img.height * .12))
            padding = 0
            default_text_height = int(bottom_margin * .3)
            middle_spacing = ctx.getint("middle_spacing", int(bottom_margin * .05))
            common_spacing = ctx.getint("common_spacing", int(.02 * (img.width + left_margin + right_margin)))

        left_top = self._render_text_slot(ctx.get("left_top"), default_text_height)
        left_middle = self._render_text_slot(ctx.get("left_middle"), default_text_height)
        left_bottom = self._render_text_slot(ctx.get("left_bottom"), default_text_height)
        right_top = self._render_text_slot(ctx.get("right_top"), default_text_height)
        right_bottom = self._render_text_slot(ctx.get("right_bottom"), default_text_height)

        left_logo = Image.open(ctx.get("left_logo")).convert('RGBA') if ctx.get("left_logo") else None
        right_logo = Image.open(ctx.get("right_logo")).convert('RGBA') if ctx.get("right_logo") else None
        center_logo = Image.open(ctx.get("center_logo")).convert('RGBA') if ctx.get("center_logo") else None
        center_logo_height = ctx.getint("center_logo_height")

        if overlay:
            canvas = img.copy()
            canvas_width, canvas_height = canvas.size
            footer_start_y = 0
        else:
            canvas_width = img.width + left_margin + right_margin
            canvas_height = img.height + top_margin + bottom_margin
            canvas = Image.new("RGBA", (canvas_width, canvas_height), color)
            canvas.paste(img, (left_margin, top_margin), mask=img)
            footer_start_y = top_margin + img.height

        left_lines = [im for im in (left_top, left_middle, left_bottom) if im.height > 0]
        if left_lines:
            left_stack_height = sum(im.height for im in left_lines) + middle_spacing * (len(left_lines) - 1)
        else:
            left_stack_height = 0

        right_lines = [im for im in (right_top, right_bottom) if im.height > 0]
        if right_lines:
            right_stack_height = sum(im.height for im in right_lines) + middle_spacing * (len(right_lines) - 1)
        else:
            right_stack_height = 0

        elem_height = max(left_stack_height, right_stack_height, 1)
        if overlay:
            elem_margin = padding
            content_top_y = canvas_height - padding - elem_height
        else:
            elem_margin = int((bottom_margin - elem_height) / 2)
            content_top_y = footer_start_y + elem_margin

        if center_logo:
            logo_height = center_logo_height if center_logo_height else (elem_height if overlay else canvas_height - footer_start_y)
            resize_ctx = PipelineContext({
                'buffer': [center_logo],
                'height': logo_height
            })
            ResizeFilter().process(resize_ctx)
            center_logo = resize_ctx.get_buffer()[0]
            center_x = (canvas.width - center_logo.width) // 2
            if overlay:
                center_y = content_top_y + (elem_height - center_logo.height) // 2
            else:
                center_y = footer_start_y + ((canvas.height - footer_start_y) - center_logo.height) // 2
            self._paste(canvas, center_logo, (center_x, center_y), with_shadow=overlay)

        # 左侧文本块宽度（左对齐）
        text_block_width = max(left_top.width, left_middle.width, left_bottom.width, 0)

        # 预缩放 logo，计算整块宽度
        logo_block_width = 0
        delimiter = None
        if left_logo:
            left_logo = self._resize_keep_ratio(left_logo, elem_height)
            delimiter = Image.new("RGBA", (delimiter_width, int(elem_height * 1.1)), delimiter_color)
            logo_block_width = left_logo.width + common_spacing + delimiter.width + common_spacing

        total_left_block_width = logo_block_width + text_block_width

        if block_align_right:
            block_right = canvas_width - right_margin - common_spacing
            block_left = max(left_margin + common_spacing, block_right - total_left_block_width)
        else:
            block_left = left_margin + common_spacing

        l_x = block_left
        if left_logo:
            left_logo_x = block_left
            left_logo_y = content_top_y + (elem_height - left_logo.height) // 2
            self._paste(canvas, left_logo, (left_logo_x, left_logo_y), with_shadow=overlay)

            delimiter_x = left_logo_x + left_logo.width + common_spacing
            delimiter_y = int(content_top_y - elem_height * .05)
            self._paste(canvas, delimiter, (delimiter_x, delimiter_y), with_shadow=overlay)

            l_x = delimiter_x + delimiter.width + common_spacing

        # 左侧三行：自上而下 left_top / left_middle / left_bottom，整体贴底
        y = canvas_height - elem_margin
        lb_y = y - left_bottom.height
        y = lb_y
        if left_middle.height > 0:
            y -= middle_spacing + left_middle.height
            lm_y = y
        else:
            lm_y = lb_y
        if left_top.height > 0:
            gap = middle_spacing if (left_middle.height > 0 or left_bottom.height > 0) else 0
            lt_y = y - gap - left_top.height
        else:
            lt_y = lm_y

        right_content_end_x = canvas_width - right_margin
        if right_bottom.height > 0 or right_top.height > 0:
            rb_y = canvas_height - elem_margin - right_bottom.height
            rt_y = rb_y - (middle_spacing + right_top.height if right_top.height > 0 else 0)
            if left_bottom.height > 0:
                rb_y = (lb_y + left_bottom.height) - right_bottom.height
            if left_top.height > 0:
                rt_y = (lt_y + left_top.height) - right_top.height
            rt_x = right_content_end_x - right_top.width - common_spacing
            rb_x = right_content_end_x - right_bottom.width - common_spacing
            if Alignment.LEFT == right_alignment:
                rt_x = rb_x = min(rt_x, rb_x)
            self._paste(canvas, right_top, (rt_x, rt_y), with_shadow=overlay)
            self._paste(canvas, right_bottom, (rb_x, rb_y), with_shadow=overlay)

        self._paste(canvas, left_top, (l_x, lt_y), with_shadow=overlay)
        if left_middle.height > 0:
            self._paste(canvas, left_middle, (l_x, lm_y), with_shadow=overlay)
        self._paste(canvas, left_bottom, (l_x, lb_y), with_shadow=overlay)

        if right_logo and (right_top.height > 0 or right_bottom.height > 0):
            right_logo = self._resize_keep_ratio(right_logo, elem_height)
            delimiter = Image.new("RGBA", (delimiter_width, int(elem_height * 1.1)), delimiter_color)
            delimiter_x = canvas_width - right_margin - max(right_top.width,
                                                            right_bottom.width) - 2 * common_spacing - delimiter.width
            delimiter_y = int(content_top_y - elem_height * .05)
            self._paste(canvas, delimiter, (delimiter_x, delimiter_y), with_shadow=overlay)

            right_logo_x = delimiter_x - common_spacing - right_logo.width
            right_logo_y = content_top_y + (elem_height - right_logo.height) // 2
            self._paste(canvas, right_logo, (right_logo_x, right_logo_y), with_shadow=overlay)

        ctx.update_buffer([canvas]).save_buffer(self.name()).success()

    @staticmethod
    def _resize_keep_ratio(logo: Image.Image, target_height: int) -> Image.Image:
        """按目标高度缩放，保持原始宽高比。"""
        if target_height <= 0 or logo.height <= 0:
            return logo
        new_h = target_height
        new_w = max(1, int(round(logo.width * (new_h / logo.height))))
        return logo.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def name(self) -> str:
        return "watermark"


class WatermarkWithTimestampFilter(FilterProcessor):
    def process(self, ctx: PipelineContext):
        img = ctx.get_buffer()[0]

        if "height" not in ctx:
            ctx.set("height", int(img.height * .02))
        # 使用注册表动态获取处理器，避免直接导入
        multi_text_processor = get_processor("multi_rich_text")
        if multi_text_processor:
            multi_text_processor().process(ctx)
        else:
            raise RuntimeError("multi_rich_text processor not found")
        text = ctx.get_buffer()[0]

        text_x = int(img.width * .93) - text.width
        text_y = int(img.height * .95)

        img.paste(text, (text_x, text_y), mask=text)
        ctx.update_buffer([img]).save_buffer(self.name()).success()

    def name(self) -> str:
        return "watermark_with_timestamp"


class RoundedCornerFilter(FilterProcessor):
    def process(self, ctx: PipelineContext):
        # CSS风格: border-radius, 单位px
        radius = ctx.getint("border_radius", 10)

        buffer = []
        for img in ctx.get_buffer():
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            width, height = img.size

            # 创建圆角蒙版
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)

            # 绘制圆角矩形
            draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)

            # 应用蒙版
            output = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            output.paste(img, (0, 0))
            output.putalpha(mask)

            buffer.append(output)
        ctx.update_buffer(buffer).save_buffer(self.name()).success()

    def name(self) -> str:
        return "rounded_corner"


class ShadowFilter(FilterProcessor):

    def process(self, ctx: PipelineContext):
        shadow_color = ctx.getcolor("shadow_color", (0, 0, 0, 180))
        shadow_radius = ctx.getint("shadow_radius", 30)
        # 新参数：衰减强度，值越大边缘越干净（推荐 1.5 ~ 3.0）
        falloff = 1.5
        buffer = []
        for img in ctx.get_buffer():
            if img.mode != 'RGBA':
                original_img = img.convert('RGBA')
            else:
                original_img = img
            w, h = original_img.size
            if shadow_radius <= 0:
                buffer.append(img)
                continue
            padding = int(shadow_radius * 2)
            full_width = w + padding * 2
            full_height = h + padding * 2
            # 1. 生成剪影阴影
            background = Image.new('RGBA', (full_width, full_height), (0, 0, 0, 0))
            shadow_layer = Image.new('RGBA', (w, h), shadow_color)
            shadow_layer.putalpha(original_img.getchannel('A'))
            background.paste(shadow_layer, (padding, padding))
            # 2. 高斯模糊
            shadow_blurred = background.filter(ImageFilter.GaussianBlur(shadow_radius))
            # 3. 关键：应用透明度衰减曲线，消除边缘残留
            shadow_blurred = self._apply_alpha_falloff(shadow_blurred, falloff)
            # 4. 合成原图
            shadow_blurred.paste(original_img, (padding, padding), mask=original_img)
            buffer.append(shadow_blurred)
        ctx.update_buffer(buffer).save_buffer(self.name()).success()

    def process2(self, ctx: PipelineContext):
        shadow_color = ctx.getcolor("shadow_color", (0, 0, 0, 255))
        # 即使radius设为30，为了视觉效果彻底消失，建议不要设太小
        shadow_radius = ctx.getint("shadow_radius", 30)
        buffer = []
        for img in ctx.get_buffer():
            if img.mode != 'RGBA':
                original_img = img.convert('RGBA')
            else:
                original_img = img

            w, h = original_img.size
            if shadow_radius <= 0:
                buffer.append(img)
                continue
            # --- 优化点 1: 扩大画布范围 (3-Sigma 原则) ---
            # 高斯模糊的尾部很长，只有预留 3倍半径 的空间，边缘像素才能自然衰减到 0 (完全透明)
            # 如果只留 1倍，最外圈一定会被切断，留下一圈灰色的“硬边”
            padding = shadow_radius * 3

            full_width = w + padding * 2
            full_height = h + padding * 2

            # 创建底图
            background = Image.new('RGBA', (full_width, full_height), (0, 0, 0, 0))
            # --- 优化点 2: 制作纯色剪影 ---
            shadow_layer = Image.new('RGBA', (w, h), shadow_color)
            shadow_layer.putalpha(original_img.getchannel('A'))

            # 将剪影贴入底图中心
            background.paste(shadow_layer, (padding, padding))
            # --- 优化点 3: 高斯模糊 ---
            shadow_in_process = background.filter(ImageFilter.GaussianBlur(shadow_radius))
            # --- 优化点 4: Alpha 通道非线性衰减 (缓动关键) ---
            # 这一步是为了解决“灰色残留”并让阴影更有层次感。
            # 我们提取阴影的 Alpha 通道，对其进行 指数运算。
            # 作用：让原本很淡的边缘（如 alpha=10）迅速变成 0，而原本浓的地方保持保留。
            # 这是清理 "脏边缘" 最有效的手段。
            r, g, b, a = shadow_in_process.split()

            # lambda x: int(x * ((x / 255.0) ** 0.5)) -> 这种会让阴影更丰满
            # lambda x: int(x * ((x / 255.0) ** 2))   -> 这种会让边缘收得更快(Fade Out)，彻底消除灰边

            # 这里使用平方级衰减 (Quad Ease In)，强力清洗边缘
            a = a.point(lambda p: int(p * (p / 255.0) * 1.2))

            shadow_in_process.putalpha(a)
            # 组合原图
            shadow_in_process.paste(original_img, (padding, padding), mask=original_img)

            # (可选) 如果你不希望图片尺寸暴增，可以在这里 crop 回去，
            # 但既然要阴影，通常就需要保留扩大的尺寸。
            buffer.append(shadow_in_process)
        ctx.update_buffer(buffer).save_buffer(self.name()).success()

    def _apply_alpha_falloff(self, img: Image.Image, gamma: float) -> Image.Image:
        """
        对 Alpha 通道应用幂函数衰减
        公式: new_alpha = (alpha / 255) ^ gamma * 255
        gamma > 1 时，低透明度像素会被压制得更低，边缘更干净
        """
        r, g, b, a = img.split()

        # 转为 numpy 处理
        alpha_array = np.array(a, dtype=np.float32) / 255.0

        # 应用幂函数（缓动曲线）
        alpha_array = np.power(alpha_array, gamma)

        # 可选：设置硬截断阈值，彻底消除极低透明度
        alpha_array[alpha_array < 0.01] = 0

        # 转回 PIL
        new_alpha = Image.fromarray((alpha_array * 255).astype(np.uint8), mode='L')
        img.putalpha(new_alpha)
        return img

    def name(self) -> str:
        return "shadow"

class CropFilter(FilterProcessor):

    def process(self, ctx: PipelineContext):
        width = ctx.getint("width", 0)
        height = ctx.getint("height", 0)
        offset = json.loads(ctx.get("offset", "[]"))

        buffer = []
        for img in ctx.get_buffer():
            img_width, img_height = img.size

            # 默认原图像尺寸
            if width <= 0:
                width = img_width
            if height <= 0:
                height = img_height

            # 默认居中
            left = (img_width - width) // 2
            top = (img_height - height) // 2

            # 处理偏移量
            offset_x = offset[0] if len(offset) > 0 else 0
            offset_y = offset[1] if len(offset) > 1 else 0
            left += offset_x
            top += offset_y

            # 计算边界
            left = max(0, min(left, img_width - width))
            top = max(0, min(top, img_height - height))
            right = left + width
            bottom = top + height

            # 执行裁剪
            cropped_img = img.crop((left, top, right, bottom))
            buffer.append(cropped_img)

        ctx.update_buffer(buffer).save_buffer(self.name()).success()

    def name(self) -> str:
        return "crop"


if __name__ == '__main__':
    buffer_path = '/Users/leslie/Workspace/3_PyProjs/semi-photo-utils/input/元旦/20250406-DSC_5779.jpg'
    ctx = PipelineContext({
        'buffer_path': [buffer_path],
        'exif': get_exif(buffer_path)
    })
    test_filter = MarginWithRatioFilter()
    # test_filter.process(ctx)
    # ctx.save_buffer(test_filter.name(), True).success()

    ctx.set('ratio', '2.25:1')
    # test_filter.process(ctx)
    # ctx.save_buffer(test_filter.name(), True).success()

    ctx.set('margin_color', 'blue')
    test_filter.process(ctx)
    ctx.save_buffer(test_filter.name(), True).success()
