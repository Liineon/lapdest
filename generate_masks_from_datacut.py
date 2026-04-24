"""
从 data_cut 生成与项目一致的二值掩码（PNG）。

规则：像素在「全黑」时为 0，否则为 1。为与 train/utils 中 (m > 128) 一致，
磁盘上保存为 uint8 的 0 与 255（语义仍为 0/1）。

目录结构：与 data_cut 相同，输出到 masks，例如
  data_cut/0分/xxx.tif  ->  masks/0分/xxx_mask.png

用法（在 doing 目录下）:
  python generate_masks_from_datacut.py
  python generate_masks_from_datacut.py --input_dir data/data_cut --output_dir data/masks

依赖：numpy、pillow；若 data_cut 中含 LZW 等压缩的 .tif，需安装 tifffile 与 imagecodecs：
  pip install tifffile imagecodecs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    import tifffile
except ImportError:
    tifffile = None

IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def _is_black_pixels(arr: np.ndarray) -> np.ndarray:
    """与输入同高宽的 bool：True 表示该像素视为全黑（掩码 0）。"""
    if arr.ndim == 2:
        if np.issubdtype(arr.dtype, np.floating):
            return np.isclose(arr, 0.0)
        return arr == 0
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return _is_black_pixels(arr[:, :, 0])
        rgb = arr[:, :, :3]
        if np.issubdtype(rgb.dtype, np.floating):
            return np.all(np.isclose(rgb, 0.0), axis=2)
        return np.all(rgb == 0, axis=2)
    raise ValueError(f"不支持的数组形状: {arr.shape}")


def image_to_mask_u8(arr: np.ndarray) -> np.ndarray:
    """全黑 -> 0，否则 -> 255，uint8 单通道。"""
    black = _is_black_pixels(arr)
    out = np.where(black, np.uint8(0), np.uint8(255))
    return out


def _normalize_filter_size(size: int) -> int:
    """PIL 形态学滤波核大小需为 >=3 的奇数。"""
    if size < 3:
        return 0
    return size if size % 2 == 1 else size + 1


def postprocess_mask_u8(
    mask: np.ndarray,
    open_size: int,
    close_size: int,
    blur_radius: float,
    threshold: int,
) -> np.ndarray:
    """
    对二值掩码做轻量平滑与去噪：
    1) 开运算（先腐蚀后膨胀）去小噪点
    2) 闭运算（先膨胀后腐蚀）补小孔洞并平滑边缘
    3) 高斯模糊后再阈值，弱化锯齿
    """
    img = Image.fromarray(mask, mode="L")

    if open_size >= 3:
        img = img.filter(ImageFilter.MinFilter(open_size)).filter(
            ImageFilter.MaxFilter(open_size)
        )

    if close_size >= 3:
        img = img.filter(ImageFilter.MaxFilter(close_size)).filter(
            ImageFilter.MinFilter(close_size)
        )

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = img.point(lambda p: 255 if p >= threshold else 0, mode="L")

    return np.array(img, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 data_cut 按全黑/非黑生成 masks（0/255 PNG）")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/data_cut",
        help="data_cut 根目录（内含 0分、1分 等子文件夹）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/masks",
        help="掩码输出根目录（与训练 config data.mask_dir 一致）",
    )
    parser.add_argument(
        "--open_size",
        type=int,
        default=3,
        help="开运算核大小（>=3，建议 3 或 5；<3 表示关闭）",
    )
    parser.add_argument(
        "--close_size",
        type=int,
        default=5,
        help="闭运算核大小（>=3，建议 3 或 5；<3 表示关闭）",
    )
    parser.add_argument(
        "--blur_radius",
        type=float,
        default=0.8,
        help="高斯平滑半径（0 表示关闭）",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="平滑后重新二值化阈值（0-255）",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    open_size = _normalize_filter_size(args.open_size)
    close_size = _normalize_filter_size(args.close_size)
    threshold = max(0, min(255, args.threshold))

    if not input_root.is_dir():
        raise SystemExit(f"输入目录不存在: {input_root.resolve()}")

    image_files = sorted(
        p
        for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_files:
        raise SystemExit(f"未找到图片: {input_root.resolve()}")

    n_ok = 0
    for src in image_files:
        rel_parent = src.parent.relative_to(input_root)
        stem = src.stem
        out_path = output_root / rel_parent / f"{stem}_mask.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            suf = src.suffix.lower()
            if suf in (".tif", ".tiff") and tifffile is not None:
                arr = tifffile.imread(str(src))
                if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
                    arr = np.moveaxis(arr, 0, -1)
            else:
                im = Image.open(src)
                arr = np.array(im)
        except OSError as e:
            print(f"[跳过] 无法读取 {src}: {e}")
            continue
        except ValueError as e:
            err = str(e)
            if "imagecodecs" in err.lower():
                raise SystemExit(
                    "读取 TIFF 失败（常见为 LZW 压缩）：请执行 pip install imagecodecs\n" + err
                ) from e
            print(f"[跳过] 读取异常 {src}: {e}")
            continue
        except Exception as e:
            print(f"[跳过] 读取异常 {src}: {e}")
            continue

        mask = image_to_mask_u8(arr)
        mask = postprocess_mask_u8(
            mask=mask,
            open_size=open_size,
            close_size=close_size,
            blur_radius=max(0.0, args.blur_radius),
            threshold=threshold,
        )
        Image.fromarray(mask, mode="L").save(out_path, optimize=True)
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  已处理 {n_ok} ...")

    print(
        "后处理参数: "
        f"open_size={open_size or 'off'}, "
        f"close_size={close_size or 'off'}, "
        f"blur_radius={max(0.0, args.blur_radius):.2f}, "
        f"threshold={threshold}"
    )
    print(f"完成: 共写入 {n_ok} 个掩码 -> {output_root.resolve()}")


if __name__ == "__main__":
    main()
