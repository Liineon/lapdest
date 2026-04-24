"""
将 data（底层）、masks（中间层）、masks_BG（顶层）按对应文件名合并：
- data: 底层原始图
- masks: 中间层，0 和 1 都对 data 有影响：0=隐藏（置 0），1=显示 data
- masks_BG: 顶层，0 和 1 都有影响：0=隐藏（置 0），1=保留下方内容
- 合并顺序：data → 覆盖 masks 得中间图 → 覆盖 masks_BG 得最终图

用法:
  python merge_masks_to_finaldata.py
  python merge_masks_to_finaldata.py --data_dir data --masks_dir masks --masks_bg_dir masks_BG --output_dir finaldata
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def read_data_image(path: str) -> np.ndarray:
    """读取 data 图片（支持 JPG、8/16-bit TIF、多通道），返回与 data 同尺寸的 BGR 或灰度数组。"""
    img = cv2.imread(path)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def binarize_mask(m: np.ndarray) -> np.ndarray:
    """掩码二值化：>0 视为 1（显示），0 为 0（不显示）。"""
    return (m > 0).astype(np.uint8)


def main(
    data_dir: str = "data",
    masks_dir: str = "masks",
    masks_bg_dir: str = "masks_BG",
    output_dir: str = "finaldata",
):
    data_path = Path(data_dir)
    masks_path = Path(masks_dir)
    masks_bg_path = Path(masks_bg_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 处理 data 里的 jpg 或 tif 图片
    exts = ["*.tif", "*.TIF", "*.tiff", "*.TIFF", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    data_files = []
    for ext in exts:
        data_files.extend(data_path.glob(f"**/{ext}"))
    data_files = sorted(data_files)

    done = 0
    skip_no_mask = 0
    skip_read_err = 0

    for f in data_files:
        rel = f.relative_to(data_path)
        subdir = rel.parent
        stem = f.stem
        # 对应 masks（中间层）与 masks_BG（顶层）的文件名：<stem>_mask.png
        mask_file = masks_path / subdir / f"{stem}_mask.png"
        mask_bg_file = masks_bg_path / subdir / f"{stem}_mask.png"

        if not mask_file.exists() or not mask_bg_file.exists():
            skip_no_mask += 1
            continue

        img = read_data_image(str(f))
        if img is None:
            skip_read_err += 1
            continue

        m_sb = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        m_bg = cv2.imread(str(mask_bg_file), cv2.IMREAD_GRAYSCALE)
        if m_sb is None or m_bg is None:
            skip_read_err += 1
            continue

        h, w = img.shape[:2]
        if m_sb.shape[0] != h or m_sb.shape[1] != w:
            m_sb = cv2.resize(m_sb, (w, h), interpolation=cv2.INTER_NEAREST)
        if m_bg.shape[0] != h or m_bg.shape[1] != w:
            m_bg = cv2.resize(m_bg, (w, h), interpolation=cv2.INTER_NEAREST)

        # masks（中间层）: 1=显示 data，0=隐藏（置 0）
        # masks_BG（顶层）: 1=保留下方内容，0=隐藏（置 0）
        bin_masks = binarize_mask(m_sb)
        bin_bg = binarize_mask(m_bg)

        # 分步覆盖：data → masks → masks_BG
        # step1: data 上覆盖 masks → layer1 = data * masks
        # step2: layer1 上覆盖 masks_BG → out = layer1 * masks_BG
        # 等价于 combined = masks=1 且 masks_BG=1 处保留
        combined = bin_masks & bin_bg

        # 底层 data，仅在 combined 为 1 处保留，其余为 0
        if img.ndim == 3:
            combined_3 = combined[:, :, np.newaxis]
            out_img = img * combined_3
        else:
            out_img = img * combined

        out_sub = out_path / subdir
        out_sub.mkdir(parents=True, exist_ok=True)
        # 输出格式与输入一致：jpg/jpeg → jpg，tif/tiff → tif
        out_ext = ".jpg" if f.suffix.lower() in (".jpg", ".jpeg") else ".tif"
        out_file = out_sub / f"{stem}{out_ext}"
        cv2.imwrite(str(out_file), out_img)
        done += 1

    print(f"合并完成: 成功 {done} 张")
    if skip_no_mask:
        print(f"  跳过（缺 masks 或 masks_BG）: {skip_no_mask} 张")
    if skip_read_err:
        print(f"  跳过（读取失败）: {skip_read_err} 张")
    print(f"输出目录: {out_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 data（底层）+ masks（中间层）+ masks_BG（顶层）到 finaldata")
    parser.add_argument("--data_dir", type=str, default="data", help="原始图片目录")
    parser.add_argument("--masks_dir", type=str, default="masks", help="中间层掩码目录")
    parser.add_argument("--masks_bg_dir", type=str, default="masks_BG", help="顶层掩码目录（遮盖用）")
    parser.add_argument("--output_dir", type=str, default="finaldata", help="输出目录")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        masks_dir=args.masks_dir,
        masks_bg_dir=args.masks_bg_dir,
        output_dir=args.output_dir,
    )
