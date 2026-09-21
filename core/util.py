import io
import json
import os
import platform
import re
import shutil
import subprocess
import time
from functools import wraps
from pathlib import Path

from PIL import Image
from jinja2 import Template

from core.configs import templates_dir
from core.jinja2renders import vh, vw, auto_logo
from core.logger import logger

if platform.system() == 'Windows':
    EXIFTOOL_PATH = Path('./exiftool/exiftool.exe')
    ENCODING = 'gbk'
elif shutil.which('exiftool') is not None:
    EXIFTOOL_PATH = shutil.which('exiftool')
    ENCODING = 'utf-8'
else:
    EXIFTOOL_PATH = Path('./exiftool/exiftool')
    ENCODING = 'utf-8'


_DATE_KEYS = (
    'DateTimeOriginal',
    'CreateDate',
    'DateTimeCreated',
    'DigitalCreationDateTime',
    'DateCreated',
    'DigitalCreationDate',
    'ModifyDate',
    'SubSecDateTimeOriginal',
    'SubSecCreateDate',
    'SubSecModifyDate',
    'GPSDateTime',
)


def _parse_epoch_number(raw: str) -> float | None:
    """纯数字时间戳：10~12 位视为秒，13+ 位视为毫秒。"""
    s = str(raw).strip()
    if not re.fullmatch(r'\d{10,16}', s):
        return None
    try:
        n = int(s)
    except ValueError:
        return None
    # 毫秒时间戳通常 13 位（到 2286 年前）；秒级 10 位
    if n >= 10**12:
        return n / 1000.0
    if n >= 10**9:
        return float(n)
    return None


def _format_local_datetime(ts: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def _format_exif_datetime(ts: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime('%Y:%m:%d %H:%M:%S')


def _parse_datetime_string(raw: str) -> float | None:
    """解析常见日期字符串 / 毫秒时间戳，返回 Unix timestamp。"""
    raw = str(raw).strip()
    if not raw:
        return None

    epoch = _parse_epoch_number(raw)
    if epoch is not None:
        return epoch

    cleaned = raw.replace('/', '-')
    # 去掉尾部时区与毫秒，先取日期时间主体
    m = re.match(
        r'^(\d{4})[-:](\d{2})[-:](\d{2})[ T](\d{2}):(\d{2}):(\d{2})',
        cleaned,
    )
    from datetime import datetime
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
            ).timestamp()
        except ValueError:
            return None

    m = re.match(r'^(\d{4})[-:](\d{2})[-:](\d{2})$', cleaned)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).timestamp()
        except ValueError:
            return None
    return None


def normalize_camera_model(model: str) -> str:
    """
    规范化水印用的机身名：
    - Xiaomi 15/24129PN74C → XIAOMI 15
    - NIKON Z50_2 → NIKON Z 50_2
    - NIKON Z 7_2 / NIKON Z 8 保持不变
    """
    if not model:
        return model
    m = str(model).strip()
    if '/' in m:
        m = m.split('/', 1)[0].strip()

    if re.match(r'(?i)^xiaomi\b', m):
        rest = re.sub(r'(?i)^xiaomi\s*', '', m).strip()
        return f'XIAOMI {rest}'.strip() if rest else 'XIAOMI'

    if re.match(r'(?i)^nikon\b', m):
        # NIKON Z50_2 / NIKON Z 50_2 / NIKONZ8 → NIKON Z ...
        m = re.sub(r'(?i)^nikon\s*z\s*', 'NIKON Z ', m)
        m = re.sub(r'\s+', ' ', m).strip()
        return m

    return m


def _normalize_exif_values(exif: dict) -> dict:
    """规范化日期字段与机身名，供模板与时间解析使用。"""
    for key in _DATE_KEYS:
        if key not in exif:
            continue
        raw = exif[key]
        ts = _parse_epoch_number(str(raw))
        if ts is not None:
            exif[key] = _format_local_datetime(ts)

    for key in ('CameraModelName', 'Model'):
        if key in exif and exif[key]:
            exif[key] = normalize_camera_model(exif[key])

    # 保证模板常用的 CameraModelName 有值
    if not exif.get('CameraModelName') and exif.get('Model'):
        exif['CameraModelName'] = exif['Model']
    elif exif.get('CameraModelName') and not exif.get('Model'):
        exif['Model'] = exif['CameraModelName']

    return exif


def get_exif(path) -> dict:
    """
    获取exif信息
    :param path: 照片路径
    :return: exif信息
    """
    exif_dict = {}
    try:
        output_bytes = subprocess.check_output([EXIFTOOL_PATH, '-d', '%Y-%m-%d %H:%M:%S%3f%z', path])
        output = output_bytes.decode('utf-8', errors='ignore')

        lines = output.splitlines()
        utf8_lines = [line for line in lines]

        for line in utf8_lines:
            # 将每一行按冒号分隔成键值对
            kv_pair = line.split(':')
            if len(kv_pair) < 2:
                continue
            key = kv_pair[0].strip()
            value = ':'.join(kv_pair[1:]).strip()
            # 将键中的空格移除
            key = re.sub(r'\s+', '', key)
            key = re.sub(r'/', '', key)
            # 将键值对添加到字典中
            exif_dict[key] = value
        for key, value in exif_dict.items():
            # 过滤非 ASCII 字符
            value_clean = ''.join(c for c in value if ord(c) < 128)
            # 将处理后的值更新到 exif_dict 中
            exif_dict[key] = value_clean

        # 部分软件把拍摄时间写成毫秒时间戳；机身名带内部型号码
        _normalize_exif_values(exif_dict)

        # 无日期字段时，尝试从文件名提取（IMG_YYYYMMDD_HHMMSS / 纯数字时间戳）
        if not any(exif_dict.get(k) for k in ('DateTimeOriginal', 'CreateDate', 'DateTimeCreated')):
            stem = Path(path).stem
            ts = None
            m = re.search(r'(?:^|_)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})(?:_|$)', stem)
            if m:
                from datetime import datetime
                try:
                    ts = datetime(
                        int(m.group(1)), int(m.group(2)), int(m.group(3)),
                        int(m.group(4)), int(m.group(5)), int(m.group(6)),
                    ).timestamp()
                except ValueError:
                    ts = None
            if ts is None:
                digit = re.search(r'(\d{13})', stem) or re.search(r'(\d{10})', stem)
                if digit:
                    ts = _parse_epoch_number(digit.group(1))
            if ts is not None:
                exif_dict['DateTimeOriginal'] = _format_local_datetime(ts)
    except Exception as e:
        logger.error(f'get_exif error: {path} : {e}')

    return exif_dict


def parse_shoot_timestamp(exif: dict) -> float | None:
    """从 EXIF 解析拍摄时间，返回 Unix timestamp；失败返回 None。"""
    if not exif:
        return None

    # 优先真实拍摄时间；ModifyDate 常为导出时间，放最后
    for key in (
        'DateTimeOriginal',
        'SubSecDateTimeOriginal',
        'CreateDate',
        'SubSecCreateDate',
        'DateTimeCreated',
        'DigitalCreationDateTime',
        'DateCreated',
        'DigitalCreationDate',
        'GPSDateTime',
        'ModifyDate',
        'SubSecModifyDate',
    ):
        raw = exif.get(key)
        if not raw:
            continue
        ts = _parse_datetime_string(str(raw))
        if ts is not None:
            return ts
    return None


def apply_shoot_time_mtime(path: str | Path, exif: dict) -> bool:
    """用拍摄时间覆盖文件修改时间（及访问时间）。成功返回 True。"""
    ts = parse_shoot_timestamp(exif)
    if ts is None:
        return False
    try:
        os.utime(path, (ts, ts))
        return True
    except OSError as e:
        logger.warning(f'apply_shoot_time_mtime failed: {path} : {e}')
        return False


def preserve_exif(src_path: str | Path, dst_path: str | Path, exif: dict | None = None) -> bool:
    """
    将输入图 EXIF 尽量复制到输出图，并把修改时间同步为拍摄时间。
    Pillow 重编码会丢掉元数据，因此用 exiftool -TagsFromFile 回写。
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists() or not dst_path.exists():
        return False

    if exif is None:
        exif = get_exif(src_path)

    try:
        # 非法 DateTimeOriginal（毫秒串）无法拷贝，后面按解析结果补写
        # -xmp：按块复制 XMP（保留 GCamera MotionPhoto 等未在 tag 库中定义的字段）
        # -Orientation#=1：输出像素已由 ImageOps.exif_transpose 转正，
        # 若继续沿用源文件的 Orientation（如竖屏照片的 8），观看端会二次旋转
        subprocess.run(
            [
                str(EXIFTOOL_PATH),
                '-TagsFromFile', str(src_path),
                '-all:all',
                '-xmp',
                '-Orientation#=1',
                '-overwrite_original',
                '-q',
                '-m',
                str(dst_path),
            ],
            check=False,
            capture_output=True,
        )
    except Exception as e:
        logger.warning(f'preserve_exif TagsFromFile failed: {dst_path} : {e}')

    ts = parse_shoot_timestamp(exif)
    if ts is None:
        return apply_shoot_time_mtime(dst_path, exif)

    dt_exif = _format_exif_datetime(ts)
    try:
        # 输出侧是否已有可写的 DateTimeOriginal（源为毫秒戳时通常拷不过来）
        dto = subprocess.check_output(
            [str(EXIFTOOL_PATH), '-s3', '-DateTimeOriginal', str(dst_path)],
            stderr=subprocess.DEVNULL,
        ).decode('utf-8', errors='ignore').strip()

        cmd = [
            str(EXIFTOOL_PATH),
            f'-ModifyDate={dt_exif}',
            f'-FileModifyDate={dt_exif}',
            '-overwrite_original',
            '-q',
            '-m',
            str(dst_path),
        ]
        # 仅在缺失或仍是非法数字戳时补写拍摄时间
        if (not dto) or _parse_epoch_number(dto) is not None:
            cmd[1:1] = [
                f'-DateTimeOriginal={dt_exif}',
                f'-CreateDate={dt_exif}',
            ]

        subprocess.run(cmd, check=False, capture_output=True)
    except Exception as e:
        logger.warning(f'preserve_exif write dates failed: {dst_path} : {e}')

    return apply_shoot_time_mtime(dst_path, exif)


def get_motion_photo_info(path: str | Path) -> dict | None:
    """
    检测 Android/Google Motion Photo（动态照片）。
    返回 {'video_length': int, 'presentation_ts_us': int}，非动态照片返回 None。
    """
    path = Path(path)
    try:
        raw = subprocess.check_output(
            [
                str(EXIFTOOL_PATH),
                '-json',
                '-MotionPhoto',
                '-MotionPhotoVersion',
                '-MotionPhotoPresentationTimestampUs',
                '-DirectoryItemLength',
                '-DirectoryItemSemantic',
                '-MicroVideo',
                '-MicroVideoOffset',
                str(path),
            ],
            stderr=subprocess.DEVNULL,
        ).decode('utf-8', errors='ignore')
        import json as _json
        info = (_json.loads(raw) or [{}])[0]
    except Exception as e:
        logger.debug(f'get_motion_photo_info failed: {path} : {e}')
        return None

    video_length = None
    # Container: DirectoryItemLength 可能是单个 int 或列表（仅 MotionPhoto 项有 Length）
    lengths = info.get('DirectoryItemLength')
    semantics = info.get('DirectoryItemSemantic')
    if isinstance(lengths, list) and isinstance(semantics, list):
        for sem, length in zip(semantics, lengths):
            if sem == 'MotionPhoto' and length:
                video_length = int(length)
                break
        if video_length is None:
            # 常见结构：Primary 无 Length，最后一项是视频 Length
            for length in reversed(lengths):
                if length:
                    video_length = int(length)
                    break
    elif lengths:
        video_length = int(lengths if not isinstance(lengths, list) else lengths[-1])

    # Samsung MicroVideo：偏移量是「距文件末尾」的字节数
    if not video_length and info.get('MicroVideoOffset'):
        video_length = int(info['MicroVideoOffset'])

    is_motion = str(info.get('MotionPhoto', '')).strip() in ('1', 'true', 'True')
    is_micro = str(info.get('MicroVideo', '')).strip() in ('1', 'true', 'True')
    if not video_length or not (is_motion or is_micro):
        # 兜底：JPEG EOI 后直接跟 mp4 ftyp
        try:
            data = path.read_bytes()
            eoi = data.rfind(b'\xff\xd9')
            if eoi >= 0 and eoi + 2 < len(data) and data[eoi + 6:eoi + 10] == b'ftyp':
                video_length = len(data) - eoi - 2
                is_motion = True
        except OSError:
            return None

    if not video_length or not (is_motion or is_micro or video_length > 0):
        return None

    file_size = path.stat().st_size
    if video_length <= 0 or video_length >= file_size:
        return None

    # 校验尾部是否像 mp4
    try:
        with path.open('rb') as f:
            f.seek(file_size - video_length)
            head = f.read(12)
        if b'ftyp' not in head:
            return None
    except OSError:
        return None

    ts = info.get('MotionPhotoPresentationTimestampUs')
    try:
        ts = int(ts) if ts is not None else 0
    except (TypeError, ValueError):
        ts = 0

    return {
        'video_length': int(video_length),
        'presentation_ts_us': ts,
    }


def extract_motion_photo_trailer(path: str | Path) -> bytes | None:
    """从 Motion Photo JPEG 末尾提取嵌入的 MP4 视频轨。"""
    info = get_motion_photo_info(path)
    if not info:
        return None
    path = Path(path)
    try:
        with path.open('rb') as f:
            f.seek(path.stat().st_size - info['video_length'])
            trailer = f.read(info['video_length'])
        if b'ftyp' not in trailer[:32]:
            return None
        return trailer
    except OSError as e:
        logger.warning(f'extract_motion_photo_trailer failed: {path} : {e}')
        return None


def append_motion_photo_trailer(dst_path: str | Path, trailer: bytes) -> bool:
    """
    将 Motion Photo 视频轨追加到输出 JPEG 末尾。
    必须在 exiftool 写回元数据之后调用；不要再对文件跑 exiftool，否则会丢掉 trailer。
    """
    dst_path = Path(dst_path)
    if not trailer or not dst_path.exists():
        return False
    try:
        data = dst_path.read_bytes()
        # 已带有相同 trailer 则跳过（避免重复追加）
        if data.endswith(trailer):
            return True
        # Pillow/exiftool 输出应为干净 JPEG；只在「尚无视频轨」时追加
        # 勿对整文件 rfind(FFD9)：MP4 内可能出现相同字节
        if len(data) >= 2 and data[-2:] != b'\xff\xd9':
            # 可能已有其它 trailer：截到第一个完整 JPEG EOI（从文件头解析）
            eoi = _find_jpeg_eoi(data)
            if eoi is None:
                logger.warning(f'append_motion_photo_trailer: no JPEG EOI in {dst_path}')
                return False
            data = data[: eoi + 2]
            dst_path.write_bytes(data)

        with dst_path.open('ab') as f:
            f.write(trailer)
        return True
    except Exception as e:
        logger.warning(f'append_motion_photo_trailer failed: {dst_path} : {e}')
        return False


def _find_jpeg_eoi(data: bytes) -> int | None:
    """从 JPEG 结构扫描找到主图 EOI 位置（返回 FFD9 中 FF 的下标）。"""
    if len(data) < 4 or data[:2] != b'\xff\xd8':
        return None
    i = 2
    n = len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            # 非 marker，可能已进入错误区域
            i += 1
            continue
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        if marker == 0xD9:  # EOI
            return i - 2
        if marker == 0xDA:  # SOS：其后直到 EOI 为熵编码，FF 00 转义
            while i < n - 1:
                if data[i] == 0xFF and data[i + 1] != 0x00:
                    if data[i + 1] == 0xD9:
                        return i
                    # 其它 marker（少见），交给外层
                    break
                i += 1
            continue
        if marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0x01):
            continue  # 无长度 RST/TEM
        if i + 1 >= n:
            break
        seglen = (data[i] << 8) | data[i + 1]
        if seglen < 2:
            break
        i += seglen
    # 兜底：仅在前 95% 里找最后一个 FFD9，降低命中 MP4 的概率
    limit = max(0, int(len(data) * 0.95))
    return data.rfind(b'\xff\xd9', 0, limit) if limit else None


def preserve_motion_photo(
    src_path: str | Path,
    dst_path: str | Path,
    watermark_overlay: "Image.Image | None" = None,
) -> bool:
    """
    若源图为 Motion Photo / 动态照片，把嵌入视频轨恢复到输出图。
    若提供 watermark_overlay（与静态图同尺寸的 RGBA 水印层），则先烧录到视频再追加。
    """
    info = get_motion_photo_info(src_path)
    if not info:
        return False
    trailer = extract_motion_photo_trailer(src_path)
    if not trailer:
        return False

    if watermark_overlay is not None:
        try:
            watermarked = _watermark_motion_photo_video(trailer, watermark_overlay)
            if watermarked:
                trailer = watermarked
                _update_motion_photo_xmp_length(dst_path, len(trailer))
        except Exception as e:
            logger.warning(f'watermark motion photo video failed, fallback raw trailer: {e}')

    ok = append_motion_photo_trailer(dst_path, trailer)
    if ok:
        logger.info(
            f'preserved Motion Photo trailer ({len(trailer)} bytes) -> {dst_path}'
        )
    return ok


def _get_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return exe
    except Exception:
        pass
    # 兜底：系统 PATH 中已安装的 ffmpeg
    return shutil.which("ffmpeg")


def _probe_video_size(ffmpeg: str, video_path: Path) -> tuple[int, int] | None:
    r = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
        errors="replace",
    )
    m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", r.stderr or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _watermark_motion_photo_video(trailer: bytes, overlay: "Image.Image") -> bytes | None:
    """用 ffmpeg 把 RGBA 水印层叠加到 Motion Photo 的 MP4 上，返回新 trailer 字节。"""
    ffmpeg = _get_ffmpeg_exe()
    if not ffmpeg:
        logger.warning("imageio-ffmpeg / ffmpeg 不可用，跳过视频水印")
        return None

    import tempfile

    overlay = overlay.convert("RGBA")
    with tempfile.TemporaryDirectory(prefix="motion_wm_") as td:
        td_path = Path(td)
        src_mp4 = td_path / "src.mp4"
        wm_png = td_path / "wm.png"
        out_mp4 = td_path / "out.mp4"
        src_mp4.write_bytes(trailer)

        size = _probe_video_size(ffmpeg, src_mp4)
        if not size:
            logger.warning("无法解析 Motion Photo 视频尺寸，跳过视频水印")
            return None
        vw, vh = size
        if overlay.size != (vw, vh):
            overlay = overlay.resize((vw, vh), Image.Resampling.LANCZOS)
        overlay.save(wm_png, format="PNG")

        # 高质量重编码，尽量接近原片；音频直接拷贝
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src_mp4),
            "-loop", "1",
            "-i", str(wm_png),
            "-filter_complex",
            f"[1:v]scale={vw}:{vh}:flags=lanczos,format=rgba[wm];"
            f"[0:v][wm]overlay=0:0:format=auto",
            "-c:v", "libx264",
            "-crf", "17",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode != 0 or not out_mp4.exists() or out_mp4.stat().st_size < 1024:
            logger.warning(f"ffmpeg watermark failed: {r.stderr[-500:] if r.stderr else r.returncode}")
            return None
        return out_mp4.read_bytes()


def _update_motion_photo_xmp_length(jpeg_path: str | Path, video_length: int) -> bool:
    """更新 JPEG 内 Motion Photo XMP 的 Item:Length，必须在追加 trailer 之前调用。"""
    jpeg_path = Path(jpeg_path)
    try:
        xmp = subprocess.check_output(
            [str(EXIFTOOL_PATH), "-b", "-XMP", str(jpeg_path)],
            stderr=subprocess.DEVNULL,
        )
        if not xmp:
            return False
        new_xmp, n = re.subn(
            rb'Item:Length="\d+"',
            f'Item:Length="{int(video_length)}"'.encode("ascii"),
            xmp,
            count=1,
        )
        if n == 0:
            return False
        xmp_file = jpeg_path.with_suffix(".motion.xmp")
        try:
            xmp_file.write_bytes(new_xmp)
            subprocess.run(
                [
                    str(EXIFTOOL_PATH),
                    "-overwrite_original",
                    "-q",
                    "-m",
                    f"-xmp<={xmp_file}",
                    str(jpeg_path),
                ],
                check=False,
                capture_output=True,
            )
        finally:
            if xmp_file.exists():
                xmp_file.unlink(missing_ok=True)
        return True
    except Exception as e:
        logger.warning(f'update motion photo XMP length failed: {jpeg_path} : {e}')
        return False


def list_files(path: str, suffixes: set[str], depth: int = 0, max_depth: int = 20):
    """
    使用 pathlib 实现的版本

    Args:
        path: 要扫描的路径
        suffixes: 支持的文件后缀
        depth: 当前递归深度（内部使用）
        max_depth: 最大递归深度，防止无限递归
    """
    result = []
    root = Path(path).resolve()

    if not root.exists():
        return result

    # 防止递归过深
    if depth > max_depth:
        logger.warning(f"list_files: 达到最大递归深度 {max_depth}，跳过 {path}")
        return result

    try:
        # 分离文件夹和文件，分别排序
        items = list(root.iterdir())
        dirs = sorted([i for i in items if i.is_dir()], key=lambda x: x.name.lower(), reverse=True)
        files = sorted([i for i in items if i.is_file()], key=lambda x: (x.stat().st_mtime, x.name.lower()),
                       reverse=True)

        # 先处理文件夹
        for item in dirs:
            if item.name.startswith('.'):
                continue
            # 跳过符号链接，避免无限递归
            if item.is_symlink():
                continue
            children = list_files(str(item), suffixes, depth + 1, max_depth)
            if children:
                result.append({
                    'label': item.name,
                    'value': str(item),
                    'children': children,
                })

        # 再处理文件
        for item in files:
            if item.name.startswith('.'):
                continue
            if item.suffix.lower() in suffixes:
                result.append({
                    'label': item.name,
                    'value': str(item),
                    'is_file': True
                })

    except PermissionError:
        logger.debug(f"list_files: 权限不足，跳过 {path}")
    except Exception as e:
        logger.error(f"list_files: 扫描失败 {path}: {e}")

    return result


def log_rt(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用被装饰的函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = (end_time - start_time) * 1000  # 计算运行时间

        logger.debug(f"[monitor]api#{func.__name__} cost {elapsed_time:.2f}ms")
        return result

    return wrapper


def convert_heic_to_jpeg(path: str, quality: int = 90) -> io.BytesIO:
    """转换 HEIC 为 JPEG 字节流"""
    with Image.open(path) as img:
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return buffer


# ==================== 模板管理相关方法 ====================

def get_template_path(template_name: str) -> Path:
    """
    获取模板文件的完整路径

    Args:
        template_name: 模板名称（不含扩展名），如 "standard1"

    Returns:
        模板文件的完整 Path 对象
    """
    return templates_dir / f"{template_name}.json"


def get_template(template_name: str) -> Template:
    """
    读取并解析模板文件为 Jinja2 Template 对象

    Args:
        template_name: 模板名称（不含扩展名），如 "standard1"

    Returns:
        Jinja2 Template 对象，已注册 vh, vw, auto_logo 全局函数
    """
    template_path = get_template_path(template_name)
    with open(template_path, encoding='utf-8') as f:
        template_str = f.read()
    template = Template(template_str)
    template.globals['vh'] = vh
    template.globals['vw'] = vw
    template.globals['auto_logo'] = auto_logo
    return template


def get_template_content(template_name: str) -> str:
    """
    获取模板文件的内容（原始字符串）

    Args:
        template_name: 模板名称（不含扩展名），如 "standard1"

    Returns:
        模板文件的原始内容字符串
    """
    template_path = get_template_path(template_name)
    with open(template_path, encoding='utf-8') as f:
        return f.read()


def save_template(template_name: str, content: str) -> None:
    """
    保存模板文件

    Args:
        template_name: 模板名称（不含扩展名），如 "standard1"
        content: 模板内容（JSON 字符串）
    """
    template_path = get_template_path(template_name)
    # 确保目录存在
    template_path.parent.mkdir(parents=True, exist_ok=True)
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(content)


def create_template(template_name: str, content: str = '[]') -> None:
    """
    创建新的模板文件

    Args:
        template_name: 模板名称（不含扩展名），如 "my_template"
        content: 模板内容（JSON 字符串），默认为空数组 '[]'

    Raises:
        FileExistsError: 如果模板文件已存在
    """
    template_path = get_template_path(template_name)
    if template_path.exists():
        raise FileExistsError(f"模板 '{template_name}' 已存在")
    save_template(template_name, content)


def list_templates() -> list[str]:
    """
    列出所有可用的模板名称

    Returns:
        模板名称列表（不含扩展名）
    """
    if not templates_dir.exists():
        return []
    return [f.stem for f in templates_dir.glob('*.json')]
