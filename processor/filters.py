import copy
import json
import re
from abc import ABC
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core.logger import logger
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
    # 版权标识烘焙进 logo 时的基准分辨率下限（px），与图片尺寸无关
    LOGO_BAKE_MIN_HEIGHT = 1024

    @staticmethod
    def _as_bool(value, default: bool = False) -> bool:
        """
        将配置值解析为布尔。
        :param value: 原始配置
        :param default: 缺省值
        :return: 布尔结果
        """
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_opacity(value, default: int = 255) -> int:
        """
        解析不透明度。支持 0-255、0-1 小数、百分比字符串。
        :param value: 原始配置
        :param default: 缺省不透明度
        :return: 0-255 的 alpha
        """
        if value is None or value == "":
            return default
        if isinstance(value, str):
            raw = value.strip()
            if raw.endswith("%"):
                return int(np.clip(float(raw[:-1]) / 100.0 * 255.0, 0, 255))
            value = float(raw)
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            return int(np.clip(round(value * 255.0), 0, 255))
        return int(np.clip(int(value), 0, 255))

    @staticmethod
    def _paste(canvas: Image.Image, layer: Image.Image, xy: tuple, with_shadow: bool = False):
        """粘贴图层；overlay 模式下可加轻微阴影。RGBA 用 alpha_composite，避免 mask paste 衰减透明度。"""
        if layer is None or layer.width == 0 or layer.height == 0:
            return
        if canvas.mode != "RGBA":
            # 非 RGBA 画布回退旧逻辑
            mask = layer if layer.mode == "RGBA" else None
            if with_shadow and layer.mode == "RGBA":
                alpha = layer.getchannel("A")
                shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
                shadow.putalpha(alpha.point(lambda a: int(a * 0.55)))
                sx, sy = xy[0] + max(1, layer.height // 40), xy[1] + max(1, layer.height // 40)
                canvas.paste(shadow, (sx, sy), shadow)
            canvas.paste(layer, xy, mask=mask)
            return

        x, y = int(xy[0]), int(xy[1])
        if with_shadow and layer.mode == "RGBA":
            alpha = layer.getchannel("A")
            shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
            shadow.putalpha(alpha.point(lambda a: int(a * 0.55)))
            sx = x + max(1, layer.height // 40)
            sy = y + max(1, layer.height // 40)
            canvas.alpha_composite(shadow, dest=(sx, sy))
        canvas.alpha_composite(layer.convert("RGBA"), dest=(x, y))

    @staticmethod
    def _empty_image():
        return Image.new("RGBA", (0, 0), (0, 0, 0, 0))

    @staticmethod
    def _layer_ink_alpha(img: Image.Image) -> Image.Image:
        """
        从 logo 提取墨迹 alpha：浅色 logo 用亮度，深色 logo 取反，再乘原 alpha。
        :param img: 带透明通道的图层
        :return: L 模式遮罩
        """
        arr = np.array(img.convert("RGBA"))
        rgb = arr[:, :, :3].astype(np.float32)
        a = arr[:, :, 3].astype(np.float32) / 255.0
        lum = rgb.mean(axis=2) / 255.0
        visible = a > 0.08
        if np.any(visible):
            mean_lum = float(lum[visible].mean())
            ink = lum if mean_lum >= 0.5 else (1.0 - lum)
        else:
            ink = np.ones_like(lum)
        new_a = np.clip(a * ink * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(new_a, mode="L")

    @staticmethod
    def _colorize_layer(img: Image.Image, rgba: tuple, is_logo: bool = False) -> Image.Image:
        """
        保留字形/logo 形状，填充指定 RGBA 颜色。
        :param img: 原图层
        :param rgba: 目标 (r,g,b,a)
        :param is_logo: 是否按 logo 墨迹提取遮罩
        :return: 着色后的图层
        """
        if img is None or img.width == 0 or img.height == 0:
            return img
        img = img.convert("RGBA")
        tr, tg, tb, ta = rgba
        alpha = WatermarkFilter._layer_ink_alpha(img) if is_logo else img.getchannel("A")
        if ta < 255:
            alpha = alpha.point(lambda p, t=ta: int(p * t / 255))
        out = Image.new("RGBA", img.size, (tr, tg, tb, 0))
        out.putalpha(alpha)
        return out

    @staticmethod
    def _apply_opacity(img: Image.Image, opacity: int) -> Image.Image:
        """
        按全局不透明度缩放图层 alpha。
        :param img: 原图层
        :param opacity: 0-255
        :return: 调整后的图层
        """
        if img is None or img.width == 0 or img.height == 0 or opacity >= 255:
            return img
        img = img.convert("RGBA")
        alpha = img.getchannel("A").point(lambda p, o=opacity: int(p * o / 255))
        img.putalpha(alpha)
        return img

    @staticmethod
    def _region_luminance(canvas: Image.Image, box: tuple) -> float:
        """
        计算画布指定区域的平均亮度（0-255，Rec.709）。
        :param canvas: 背景图
        :param box: (x0, y0, x1, y1)
        :return: 平均亮度
        """
        stats = WatermarkFilter._region_luminance_stats(canvas, box)
        return stats[0]

    @staticmethod
    def _region_luminance_stats(canvas: Image.Image, box: tuple) -> tuple[float, float, float, float]:
        """
        区域亮度统计：mean, std, p30, p70（0-255）。
        复杂背景用分位差判断，避免只靠均值选错黑白。
        """
        x0, y0, x1, y1 = box
        x0 = int(max(0, min(canvas.width, x0)))
        x1 = int(max(0, min(canvas.width, x1)))
        y0 = int(max(0, min(canvas.height, y0)))
        y1 = int(max(0, min(canvas.height, y1)))
        if x1 <= x0 or y1 <= y0:
            return 0.0, 0.0, 0.0, 0.0
        crop = np.array(canvas.crop((x0, y0, x1, y1)).convert("RGB"), dtype=np.float32)
        y = 0.2126 * crop[:, :, 0] + 0.7152 * crop[:, :, 1] + 0.0722 * crop[:, :, 2]
        return (
            float(y.mean()),
            float(y.std()),
            float(np.percentile(y, 30)),
            float(np.percentile(y, 70)),
        )

    @staticmethod
    def _apply_stroke(layer: Image.Image, stroke_width: int, stroke_rgba: tuple) -> Image.Image:
        """
        给图层加描边：先铺描边色轮廓，再叠原图层。
        返回尺寸会四周各扩展 stroke_width 像素。
        """
        if layer is None or layer.width == 0 or layer.height == 0 or stroke_width <= 0:
            return layer
        layer = layer.convert("RGBA")
        pad = int(stroke_width)
        sr, sg, sb, sa = [int(np.clip(v, 0, 255)) for v in stroke_rgba]
        alpha = layer.getchannel("A")
        # 实心墨迹做描边核，抗锯齿边缘也扩出去
        core = alpha.point(lambda p: 255 if p > 12 else 0)
        sil = Image.new("RGBA", layer.size, (sr, sg, sb, 0))
        sil.putalpha(core.point(lambda p, t=sa: t if p else 0))

        out = Image.new("RGBA", (layer.width + 2 * pad, layer.height + 2 * pad), (0, 0, 0, 0))
        r2 = pad * pad
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                if dx * dx + dy * dy > r2:
                    continue
                out.alpha_composite(sil, dest=(pad + dx, pad + dy))
        out.alpha_composite(layer, dest=(pad, pad))
        return out

    @staticmethod
    def _parse_scale(value) -> float:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            return 1.0
        return scale if scale > 0 else 1.0

    @staticmethod
    def _scaled_px(value, scale: float, minimum: int = 0) -> int:
        px = int(round(float(value) * scale))
        if minimum and float(value) > 0:
            return max(minimum, px)
        return max(0, px)

    def _collect_text_heights(self, cfg, found: list):
        if isinstance(cfg, dict):
            if cfg.get("height") not in (None, ""):
                try:
                    found.append(float(cfg["height"]))
                except (TypeError, ValueError):
                    pass
            for seg in cfg.get("text_segments") or []:
                self._collect_text_heights(seg, found)
        elif isinstance(cfg, list):
            for item in cfg:
                self._collect_text_heights(item, found)

    def _boost_scale_for_min_text(self, ctx: PipelineContext, scale: float, min_text_px: int) -> float:
        """小图上把整体缩放抬到至少 min_text_px，各行相对比例不变。"""
        if min_text_px <= 0:
            return scale
        heights = []
        for key in ("left_top", "left_middle", "left_bottom", "logo_bottom", "right_top", "right_bottom"):
            self._collect_text_heights(ctx.get(key), heights)
        if not heights:
            return scale
        largest = max(heights) * scale
        if largest <= 0 or largest >= min_text_px:
            return scale
        return scale * (min_text_px / largest)

    def _scale_text_cfg(self, cfg, scale: float):
        if not cfg or abs(scale - 1.0) < 1e-6:
            return cfg
        cfg = copy.deepcopy(cfg)

        def apply(node):
            if isinstance(node, dict):
                for key in ("height", "text_spacing"):
                    if node.get(key) not in (None, ""):
                        try:
                            node[key] = max(1, int(round(float(node[key]) * scale)))
                        except (TypeError, ValueError):
                            pass
                for seg in node.get("text_segments") or []:
                    apply(seg)
            elif isinstance(node, list):
                for item in node:
                    apply(item)

        apply(cfg)
        return cfg

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

    def _render_text_slot_at_height(self, cfg, height: int):
        """
        按指定像素高度渲染文字槽，用于在烘焙分辨率下重绘版权标识。
        低分辨率下渲染再放大会让字形与宽度失真，故单独重绘一次。
        :param cfg: 文字槽配置（dict 或 list）
        :param height: 目标高度（px）
        :return: 渲染后的图层
        """
        if not cfg:
            return self._empty_image()
        nodes = copy.deepcopy(cfg) if isinstance(cfg, list) else [copy.deepcopy(cfg)]
        for node in nodes:
            if isinstance(node, dict) and node.get("processor_name") in ("rich_text", "multi_rich_text"):
                node["height"] = int(height)
        return start_process(nodes)

    @staticmethod
    def _logo_bottom_ratio(ctx: PipelineContext, logo_bottom: Image.Image, logo_target_h: int) -> float:
        """
        版权标识相对 logo 的高度比例。
        优先取模板的 logo_bottom_ratio（固定值，与图片尺寸无关）；
        未配置时按已渲染高度推算（旧行为，小图取整会漂移）。
        """
        raw = ctx.get("logo_bottom_ratio")
        if raw not in (None, ""):
            try:
                ratio = float(raw)
            except (TypeError, ValueError):
                ratio = 0.0
            if ratio > 0:
                return max(0.02, min(0.6, ratio))
        return float(logo_bottom.height) / max(float(logo_target_h), 1.0)

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
        auto_color = self._as_bool(ctx.get("auto_color"), overlay)
        opacity = self._parse_opacity(ctx.get("opacity"), delimiter_color[3] if auto_color else 255)
        auto_color_threshold = ctx.getint("auto_color_threshold", 128)
        # 花斑/高反差背景是否自动提高不透明度（默认关，严格使用 opacity）
        busy_opacity_boost = self._as_bool(ctx.get("busy_opacity_boost"), False)
        # 描边：stroke_width>0 时为水印加对比色轮廓；stroke_color=auto 则与填充色相反
        stroke_width_raw = ctx.get("stroke_width", 0)
        try:
            stroke_width_val = int(float(stroke_width_raw)) if stroke_width_raw not in (None, "") else 0
        except (TypeError, ValueError):
            stroke_width_val = 0
        stroke_opacity = self._parse_opacity(ctx.get("stroke_opacity"), min(255, max(opacity, 200)))
        stroke_color_raw = ctx.get("stroke_color", "auto")
        # delimiter_width：vh 过小会变成 0（矮图/横图常见），配置了就至少 1px
        _dw_raw = ctx.get("delimiter_width")
        delimiter_width = ctx.getint("delimiter_width", int(img.width * (.002 if overlay else .003)))
        if _dw_raw not in (None, "") and delimiter_width < 1:
            try:
                if float(_dw_raw) > 0:
                    delimiter_width = 1
            except (TypeError, ValueError):
                pass
        right_alignment = ctx.getenum("right_alignment", Alignment.RIGHT, Alignment)
        # block_align: left|right —— logo+左侧文本块整体靠左或靠右
        block_align_raw = str(ctx.get("block_align", "left")).strip().lower()
        block_align_right = block_align_raw in {"right", "end"}
        # scale：整体缩放；各文字槽的 height 仍是相对细调
        scale = self._parse_scale(ctx.get("scale", 1))
        min_text_px = ctx.getint("min_text_px", 16)
        scale = self._boost_scale_for_min_text(ctx, scale, min_text_px)

        if overlay:
            left_margin = self._scaled_px(ctx.getint("left_margin", int(img.width * .03)), scale)
            right_margin = self._scaled_px(ctx.getint("right_margin", int(img.width * .03)), scale)
            top_margin = 0
            bottom_margin = 0
            padding = self._scaled_px(ctx.getint("padding", int(img.height * .03)), scale)
            default_text_height = self._scaled_px(ctx.getint("text_height", max(int(img.height * .015), 12)), scale, minimum=1)
            middle_spacing = self._scaled_px(ctx.getint("middle_spacing", max(int(img.height * .004), 2)), scale)
            common_spacing = self._scaled_px(ctx.getint("common_spacing", max(int(img.width * .01), 4)), scale)
        else:
            left_margin = self._scaled_px(ctx.getint("left_margin", 0), scale)
            right_margin = self._scaled_px(ctx.getint("right_margin", 0), scale)
            top_margin = self._scaled_px(ctx.getint("top_margin", 0), scale)
            bottom_margin = self._scaled_px(ctx.getint("bottom_margin", int(img.height * .12)), scale, minimum=1)
            padding = 0
            default_text_height = self._scaled_px(int(bottom_margin * .3 / max(scale, 0.01)), scale, minimum=1)
            middle_spacing = self._scaled_px(ctx.getint("middle_spacing", int(bottom_margin * .05 / max(scale, 0.01))), scale)
            common_spacing = self._scaled_px(
                ctx.getint("common_spacing", int(.02 * (img.width + left_margin + right_margin))), scale)

        delimiter_width = self._scaled_px(delimiter_width, scale, minimum=1 if delimiter_width > 0 else 0)
        stroke_width = self._scaled_px(stroke_width_val, scale, minimum=2 if stroke_width_val > 0 else 0)

        left_top = self._render_text_slot(self._scale_text_cfg(ctx.get("left_top"), scale), default_text_height)
        left_middle = self._render_text_slot(self._scale_text_cfg(ctx.get("left_middle"), scale), default_text_height)
        left_bottom = self._render_text_slot(self._scale_text_cfg(ctx.get("left_bottom"), scale), default_text_height)
        logo_bottom = self._render_text_slot(self._scale_text_cfg(ctx.get("logo_bottom"), scale), default_text_height)
        right_top = self._render_text_slot(self._scale_text_cfg(ctx.get("right_top"), scale), default_text_height)
        right_bottom = self._render_text_slot(self._scale_text_cfg(ctx.get("right_bottom"), scale), default_text_height)

        left_logo = Image.open(ctx.get("left_logo")).convert('RGBA') if ctx.get("left_logo") else None
        right_logo = Image.open(ctx.get("right_logo")).convert('RGBA') if ctx.get("right_logo") else None
        center_logo = Image.open(ctx.get("center_logo")).convert('RGBA') if ctx.get("center_logo") else None
        center_logo_height = ctx.getint("center_logo_height")
        if center_logo_height:
            center_logo_height = self._scaled_px(center_logo_height, scale, minimum=1)

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

        # 先按右侧文本高度估算 logo 目标高度；有 logo_bottom 时为其预留空间
        # 若配置了 left_logo_height，则单独指定 logo 高度（仍受 scale 影响）
        # logo_bottom_position: below=logo 下方；br=叠在 logo 右下角
        logo_bottom_pos = str(ctx.get("logo_bottom_position", "below")).strip().lower()
        overlay_logo_bottom = logo_bottom_pos in {
            "br", "bottom_right", "right_bottom", "overlay_br", "overlay",
        }
        text_block_width = max(left_top.width, left_middle.width, left_bottom.width, 0)
        has_side_text = text_block_width > 0
        explicit_logo_h = ctx.get("left_logo_height")
        if explicit_logo_h not in (None, ""):
            logo_target_h = self._scaled_px(explicit_logo_h, scale, minimum=1)
        else:
            logo_target_h = left_stack_height if left_stack_height > 0 else max(default_text_height, 1)
            if left_logo and logo_bottom.height > 0 and not overlay_logo_bottom:
                logo_target_h = max(default_text_height, logo_target_h - logo_bottom.height - middle_spacing)

        logo_col_width = 0
        logo_col_height = 0
        if left_logo:
            # 先裁透明边再按目标高度缩放，避免空白计入高度
            left_logo = self._trim_alpha(left_logo)
            if overlay_logo_bottom and logo_bottom.height > 0:
                # 版权按固定比例在烘焙分辨率下重绘后烙到 logo 右下角，之后只整体缩放，
                # 使两者的相对大小与间距不随图片尺寸变化
                inset_frac = self._parse_logo_bottom_inset(ctx.get("logo_bottom_inset"))
                copy_ratio = self._logo_bottom_ratio(ctx, logo_bottom, logo_target_h)
                bake_h = max(int(logo_target_h) * 8, self.LOGO_BAKE_MIN_HEIGHT)
                logo_bottom_hi = self._trim_alpha_horizontal(self._render_text_slot_at_height(
                    self._scale_text_cfg(ctx.get("logo_bottom"), scale),
                    max(1, int(round(bake_h * copy_ratio))),
                ))
                left_logo = self._bake_logo_bottom_br(
                    left_logo, logo_bottom_hi, max(1, logo_target_h), copy_ratio, inset_frac, bake_h
                )
                logo_bottom = Image.new("RGBA", (0, 0), (0, 0, 0, 0))
            else:
                left_logo = self._resize_keep_ratio(left_logo, max(1, logo_target_h))
            logo_col_width = left_logo.width
            logo_col_height = left_logo.height
        if logo_bottom.height > 0:
            # 版权文字只裁左右，保留 height 参数决定的字号高度
            logo_bottom = self._trim_alpha_horizontal(logo_bottom)
            logo_col_width = max(logo_col_width, logo_bottom.width)
            if logo_col_height > 0:
                logo_col_height += middle_spacing + logo_bottom.height
            else:
                logo_col_height = logo_bottom.height

        elem_height = max(left_stack_height, right_stack_height, logo_col_height, 1)
        if overlay:
            elem_margin = padding
            content_top_y = canvas_height - padding - elem_height
        else:
            elem_margin = int((bottom_margin - elem_height) / 2)
            content_top_y = footer_start_y + elem_margin

        paste_items = []  # (layer, x, y, is_logo)

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
            paste_items.append((center_logo, center_x, center_y, True))

        # logo 列 + 可选分隔符 + 右侧文本
        show_left_delimiter = bool(left_logo) and has_side_text and delimiter_width > 0
        left_delimiter = None
        logo_block_width = logo_col_width
        if left_logo or logo_bottom.height > 0:
            if show_left_delimiter:
                left_delimiter = Image.new("RGBA", (delimiter_width, int(elem_height * 1.1)), delimiter_color)
                logo_block_width = logo_col_width + common_spacing + left_delimiter.width + common_spacing
            elif has_side_text:
                logo_block_width = logo_col_width + common_spacing
            else:
                logo_block_width = logo_col_width

        total_left_block_width = logo_block_width + text_block_width

        if block_align_right:
            block_right = canvas_width - right_margin - common_spacing
            block_left = max(left_margin + common_spacing, block_right - total_left_block_width)
        else:
            block_left = left_margin + common_spacing

        l_x = block_left
        # logo 列：默认 logo 上 / 版权下；logo_bottom_position=br 时版权已烙进 logo
        if left_logo or logo_bottom.height > 0:
            col_top = content_top_y + max(0, (elem_height - logo_col_height) // 2)
            logo_center_x = block_left + logo_col_width // 2
            cursor_y = col_top
            if left_logo:
                left_logo_x = logo_center_x - left_logo.width // 2
                paste_items.append((left_logo, left_logo_x, cursor_y, True))
                cursor_y += left_logo.height
            if logo_bottom.height > 0:
                if left_logo:
                    cursor_y += middle_spacing
                lb_logo_x = logo_center_x - logo_bottom.width // 2
                paste_items.append((logo_bottom, lb_logo_x, cursor_y, False))

            if show_left_delimiter and left_delimiter is not None:
                delimiter_x = block_left + logo_col_width + common_spacing
                delimiter_y = int(content_top_y - elem_height * .05)
                paste_items.append((left_delimiter, delimiter_x, delimiter_y, False))
                l_x = delimiter_x + left_delimiter.width + common_spacing
            elif has_side_text:
                l_x = block_left + logo_col_width + common_spacing

        # 左侧文本列：自上而下 left_top / left_middle / left_bottom
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
            paste_items.append((right_top, rt_x, rt_y, False))
            paste_items.append((right_bottom, rb_x, rb_y, False))

        if left_top.height > 0:
            paste_items.append((left_top, l_x, lt_y, False))
        if left_middle.height > 0:
            paste_items.append((left_middle, l_x, lm_y, False))
        if left_bottom.height > 0:
            paste_items.append((left_bottom, l_x, lb_y, False))

        # 右侧 logo：仅在有右侧参数时绘制分隔符
        if right_logo and (right_top.height > 0 or right_bottom.height > 0):
            right_logo = self._resize_keep_ratio(right_logo, elem_height)
            right_delimiter = Image.new("RGBA", (delimiter_width, int(elem_height * 1.1)), delimiter_color)
            delimiter_x = canvas_width - right_margin - max(right_top.width,
                                                            right_bottom.width) - 2 * common_spacing - right_delimiter.width
            delimiter_y = int(content_top_y - elem_height * .05)
            paste_items.append((right_delimiter, delimiter_x, delimiter_y, False))

            right_logo_x = delimiter_x - common_spacing - right_logo.width
            right_logo_y = content_top_y + (elem_height - right_logo.height) // 2
            paste_items.append((right_logo, right_logo_x, right_logo_y, True))

        fill_rgb = None
        if auto_color and overlay:
            boxes = []
            for layer, px, py, _ in paste_items:
                if layer is None or layer.width == 0 or layer.height == 0:
                    continue
                boxes.append((px, py, px + layer.width, py + layer.height))
            if boxes:
                x0 = min(b[0] for b in boxes)
                y0 = min(b[1] for b in boxes)
                x1 = max(b[2] for b in boxes)
                y1 = max(b[3] for b in boxes)
                pad_box = max(4, int(elem_height * 0.15))
                mean_l, std_l, p30, p70 = self._region_luminance_stats(
                    canvas, (x0 - pad_box, y0 - pad_box, x1 + pad_box, y1 + pad_box)
                )
                # 花斑背景：白字 + 黑描边（黑边在亮纹/碎光上比白边更稳）
                spread = p70 - p30
                busy = std_l >= 40 or spread >= 55
                if busy:
                    fill_rgb = (255, 255, 255)
                else:
                    fill_rgb = (0, 0, 0) if mean_l >= auto_color_threshold else (255, 255, 255)
                use_opacity = opacity
                if busy_opacity_boost and (busy or 70 <= mean_l <= 180):
                    use_opacity = min(255, max(opacity, 230))
                # 花斑时可选增强描边不透明（随 busy_opacity_boost）
                if busy_opacity_boost and busy and stroke_width > 0:
                    stroke_opacity = max(stroke_opacity, 255)
                logger.debug(
                    f"watermark auto_color mean={mean_l:.1f} std={std_l:.1f} "
                    f"p30={p30:.1f} p70={p70:.1f} -> {fill_rgb} opacity={use_opacity} "
                    f"busy_boost={busy_opacity_boost}"
                )
                tint = (*fill_rgb, use_opacity)
                colored = []
                for layer, px, py, is_logo in paste_items:
                    colored.append((self._colorize_layer(layer, tint, is_logo=is_logo), px, py, is_logo))
                paste_items = colored
        elif opacity < 255:
            paste_items = [
                (self._apply_opacity(layer, opacity), px, py, is_logo)
                for layer, px, py, is_logo in paste_items
            ]

        # 描边：默认与填充色相反；厚度不超过最矮图层的约 1/5，避免吞掉字心
        if stroke_width > 0 and paste_items:
            layer_heights = [
                layer.height for layer, _, _, _ in paste_items
                if layer is not None and layer.height > 0
            ]
            if layer_heights:
                stroke_width = min(stroke_width, max(1, min(layer_heights) // 5))
            sc_raw = str(stroke_color_raw).strip().lower() if stroke_color_raw is not None else "auto"
            if sc_raw in ("", "auto"):
                if fill_rgb is not None:
                    stroke_rgb = (0, 0, 0) if fill_rgb[0] > 127 else (255, 255, 255)
                else:
                    # 非 auto_color：按首个非空图层平均亮度粗判
                    stroke_rgb = (0, 0, 0)
                    for layer, _, _, _ in paste_items:
                        if layer and layer.width and layer.mode == "RGBA":
                            arr = np.array(layer)
                            m = arr[:, :, 3] > 12
                            if m.any() and float(arr[:, :, :3][m].mean()) > 127:
                                stroke_rgb = (0, 0, 0)
                            else:
                                stroke_rgb = (255, 255, 255)
                            break
            elif sc_raw in ("black", "dark"):
                stroke_rgb = (0, 0, 0)
            elif sc_raw in ("white", "light"):
                stroke_rgb = (255, 255, 255)
            else:
                try:
                    stroke_rgb = ctx.getcolor("stroke_color", (0, 0, 0, 255))[:3]
                except Exception:
                    stroke_rgb = (0, 0, 0)
            stroke_rgba = (*stroke_rgb, stroke_opacity)
            stroked = []
            for layer, px, py, is_logo in paste_items:
                if layer is None or layer.width == 0 or layer.height == 0:
                    stroked.append((layer, px, py, is_logo))
                    continue
                outlined = self._apply_stroke(layer, stroke_width, stroke_rgba)
                stroked.append((outlined, px - stroke_width, py - stroke_width, is_logo))
            paste_items = stroked

        use_shadow = overlay and not auto_color and stroke_width <= 0
        for layer, px, py, _ in paste_items:
            self._paste(canvas, layer, (px, py), with_shadow=use_shadow)

        # overlay 模式：额外保存透明水印层，供 Motion Photo 视频轨烧录
        # 注意：RGBA paste(..., mask=) 会错误衰减 alpha，须用 alpha_composite
        if overlay and paste_items:
            wm_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            for layer, px, py, _ in paste_items:
                if layer is None or layer.width == 0 or layer.height == 0:
                    continue
                if use_shadow and layer.mode == "RGBA":
                    alpha = layer.getchannel("A")
                    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
                    shadow.putalpha(alpha.point(lambda a: int(a * 0.55)))
                    sx = px + max(1, layer.height // 40)
                    sy = py + max(1, layer.height // 40)
                    wm_layer.alpha_composite(shadow, dest=(sx, sy))
                wm_layer.alpha_composite(layer.convert("RGBA"), dest=(px, py))
            ctx.set("watermark_overlay", wm_layer)

        ctx.update_buffer([canvas]).save_buffer(self.name()).success()

    @staticmethod
    def _trim_alpha(img: Image.Image) -> Image.Image:
        """裁掉四周透明边。"""
        if img is None or img.width == 0 or img.height == 0:
            return img
        img = img.convert("RGBA")
        bbox = img.getchannel("A").getbbox()
        if not bbox:
            return img
        return img.crop(bbox)

    @staticmethod
    def _trim_alpha_horizontal(img: Image.Image) -> Image.Image:
        """只裁左右透明边，保留原高度（字号由 height 参数控制）。"""
        if img is None or img.width == 0 or img.height == 0:
            return img
        img = img.convert("RGBA")
        bbox = img.getchannel("A").getbbox()
        if not bbox:
            return img
        left, _, right, _ = bbox
        return img.crop((left, 0, right, img.height))

    @staticmethod
    def _parse_logo_bottom_inset(value) -> float:
        """logo 右下角内边距占 logo 短边的比例，默认约 2%（贴近大图参考效果）。"""
        try:
            if value in (None, ""):
                return 0.02
            frac = float(value)
        except (TypeError, ValueError):
            return 0.02
        if frac < 0:
            return 0.0
        if frac > 1:
            # 兼容误写成百分数，如 2 → 0.02
            frac = frac / 100.0
        return min(0.2, frac)

    @classmethod
    def _bake_logo_bottom_br(
        cls,
        logo: Image.Image,
        copyright_img: Image.Image,
        target_logo_h: int,
        copy_ratio: float,
        inset_frac: float,
        work_h: int = None,
    ) -> Image.Image:
        """
        在固定高分辨率下把版权烙到 logo 右下角，再整体缩到目标高度。
        copy_ratio 固定了版权相对 logo 的高度比例，work_h 固定了烘焙基准分辨率，
        因此不同尺寸的图片得到的大小比例与间距完全一致。
        """
        if logo is None or logo.width == 0 or logo.height == 0:
            return logo
        if copyright_img is None or copyright_img.width == 0 or copyright_img.height == 0:
            return cls._resize_keep_ratio(logo, target_logo_h)

        work_h = int(work_h) if work_h else max(int(target_logo_h) * 8, cls.LOGO_BAKE_MIN_HEIGHT)
        work_logo = cls._resize_keep_ratio(logo, work_h)
        work_copy_h = max(1, int(round(work_h * float(copy_ratio))))
        work_copy = cls._resize_keep_ratio(copyright_img.convert("RGBA"), work_copy_h)

        pad = max(1, int(round(min(work_logo.width, work_logo.height) * inset_frac)))
        max_w = max(1, work_logo.width - 2 * pad)
        if work_copy.width > max_w:
            new_w = max_w
            new_h = max(1, int(round(work_copy.height * (new_w / work_copy.width))))
            work_copy = work_copy.resize((new_w, new_h), Image.Resampling.LANCZOS)

        x = max(pad, work_logo.width - work_copy.width - pad)
        y = max(pad, work_logo.height - work_copy.height - pad)
        baked = Image.new("RGBA", work_logo.size, (0, 0, 0, 0))
        baked.paste(work_logo, (0, 0), work_logo)
        baked.alpha_composite(work_copy, dest=(x, y))
        return cls._resize_keep_ratio(baked, target_logo_h)

    @staticmethod
    def _resize_keep_ratio(logo: Image.Image, target_height: int) -> Image.Image:
        """按目标高度缩放，保持原始宽高比。大图分步缩小，减轻一次压到很小造成的发糊。"""
        if target_height <= 0 or logo.height <= 0:
            return logo
        new_h = int(target_height)
        new_w = max(1, int(round(logo.width * (new_h / logo.height))))
        img = logo
        while img.height > new_h * 2 and img.width > new_w * 2 and img.height > 2 and img.width > 2:
            img = img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.Resampling.BOX)
        if img.size != (new_w, new_h):
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img

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
