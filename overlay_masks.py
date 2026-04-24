"""
将 masks（鱼鳔掩码）和 masks_BG（背景/鱼体掩码）叠加

- masks: 0=鱼鳔区域，255=其他（>0 视为 1）
- masks_BG: 0=背景，255=鱼体（>0 视为 1）

叠加规则：当且仅当 masks 和 masks_BG 都为 1 时输出 1（255），其他为 0。

用法:
  python overlay_masks.py
  python overlay_masks.py --masks_dir masks --masks_bg_dir masks_BG --output_dir masks_overlay
  python overlay_masks.py --test_image masks/0分/BMU522_mask.png
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def combine_masks(mask_sb: np.ndarray, mask_bg: np.ndarray) -> np.ndarray:
    """
    合并 masks 和 masks_BG：当且仅当两者都为 1 时为 1，其他为 0。
    - masks: >0 视为 1
    - masks_BG: >0 视为 1

    Returns:
        二值掩码 (0 或 255)
    """
    bin_sb = (mask_sb > 0).astype(np.uint8) * 255
    bin_bg = (mask_bg > 0).astype(np.uint8) * 255
    combined = cv2.bitwise_and(bin_sb, bin_bg)
    return combined


def main(
    masks_dir: str = "masks",
    masks_bg_dir: str = "masks_BG",
    output_dir: str = "masks_overlay",
    test_image: str = None,
):
    masks_path = Path(masks_dir)
    masks_bg_path = Path(masks_bg_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 收集所有 mask 文件（以 masks 为准）
    mask_files = sorted(
        p for p in masks_path.rglob("*")
        if p.is_file() and "_mask" in p.name and p.suffix.lower() == ".png"
    )

    if test_image:
        test_path = Path(test_image)
        if test_path.exists():
            try:
                rel = test_path.relative_to(masks_path)
                mask_files = [test_path]
            except ValueError:
                rel = test_path.name
                mask_files = [masks_path / rel] if (masks_path / rel).exists() else []
        else:
            # 按文件名匹配
            stem = Path(test_image).stem.replace("_mask", "")
            matched = [mf for mf in mask_files if stem in mf.stem or mf.stem.replace("_mask", "") == stem]
            mask_files = matched[:1] if matched else []

    if not mask_files:
        print(f"未在 {masks_dir} 中找到掩码文件")
        return

    print(f"找到 {len(mask_files)} 个掩码文件")
    print(f"输出目录: {out_path.absolute()}")
    print()

    done = 0
    for i, mask_file in enumerate(mask_files, 1):
        rel = mask_file.relative_to(masks_path)
        mask_bg_file = masks_bg_path / rel

        if not mask_bg_file.exists():
            print(f"[{i}/{len(mask_files)}] 跳过（无对应 masks_BG）: {rel}")
            continue

        m_sb = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        m_bg = cv2.imread(str(mask_bg_file), cv2.IMREAD_GRAYSCALE)
        if m_sb is None or m_bg is None:
            print(f"[{i}/{len(mask_files)}] 读取失败: {rel}")
            continue

        # 尺寸不一致时调整
        if m_sb.shape != m_bg.shape:
            m_bg = cv2.resize(m_bg, (m_sb.shape[1], m_sb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 叠加：当且仅当 masks 和 masks_BG 都为 1 时为 1，其他为 0
        combined = combine_masks(m_sb, m_bg)
        out_sub = out_path / rel.parent
        out_sub.mkdir(parents=True, exist_ok=True)
        out_file = out_sub / rel.name
        cv2.imwrite(str(out_file), combined)
        done += 1
        print(f"[{i}/{len(mask_files)}] {rel} -> {out_file.name}")

    print(f"\n完成！共处理 {done} 张")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 masks 和 masks_BG 叠加（当且仅当两者都为 1 时输出 1）")
    parser.add_argument("--masks_dir", type=str, default="masks", help="鱼鳔掩码目录")
    parser.add_argument("--masks_bg_dir", type=str, default="masks_BG", help="背景/鱼体掩码目录")
    parser.add_argument("--output_dir", type=str, default="masks_overlay", help="输出目录")
    parser.add_argument("--test_image", type=str, default=None, help="仅处理指定掩码文件（调试用）")
    args = parser.parse_args()

    main(
        masks_dir=args.masks_dir,
        masks_bg_dir=args.masks_bg_dir,
        output_dir=args.output_dir,
        test_image=args.test_image,
    )
