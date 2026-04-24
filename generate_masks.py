"""
鱼鳔掩码生成工具

专门用于生成鱼鳔区域的掩码，鱼鳔通常位于图像中间，呈椭圆形暗色区域。
输出二值掩码：0=鱼鳔区域，255=其他区域。

使用方法:
  # 基本使用（使用测试好的参数：椭圆比例0.6，暗色阈值60）
  python generate_masks.py --input_dir data --output_dir masks

  # 测试单张图像并可视化
  python generate_masks.py --input_dir data --output_dir masks_test --test_image data/0分/BUF279.jpg --visualize

  # 调整参数
  python generate_masks.py --input_dir data --output_dir masks --ellipse_ratio 0.5 --dark_threshold 50 --visualize
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple


def detect_swim_bladder_region(img: np.ndarray, 
                               ellipse_ratio: float = 0.60,
                               center_x_offset: float = 0.0,
                               center_y_offset: float = 0.0,
                               dark_threshold: int = 65,
                               adaptive: bool = True,
                               expand_pixels: int = 500,
                               expand_into_dark: bool = True) -> np.ndarray:
    """
    检测鱼鳔区域并生成掩码（hybrid方法）
    
    结合椭圆形状和暗色检测：在图像中心创建椭圆作为候选区域，然后检测暗色区域，
    取两者的交集作为鱼鳔区域。
    
    Args:
        img: 输入BGR图像
        ellipse_ratio: 椭圆相对图像大小的比例 (0-1, 默认: 0.65)
        center_x_offset: 椭圆中心X方向偏移（比例 -0.3到0.3）
        center_y_offset: 椭圆中心Y方向偏移（比例 -0.3到0.3）
        dark_threshold: 暗色阈值 (0-255, 默认: 65，越小越严格)
        adaptive: 是否使用自适应阈值
        expand_pixels: 鱼鳔区域向外膨胀的像素数，用于覆盖周围黑色轮廓 (默认: 5)
        expand_into_dark: 若为 True，识别区域会向相邻黑色扩展，遇到非黑色（灰度≥dark_threshold+15）则停止扩散 (默认: True)
    
    Returns:
        二值掩码（0=鱼鳔区域，255=其他）
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 步骤1: 创建中心椭圆作为候选区域
    ellipse_mask = create_center_ellipse(gray.shape, ellipse_ratio, center_x_offset, center_y_offset)
    
    # 步骤2: 检测暗色区域
    dark_mask = detect_dark_regions(gray, dark_threshold, adaptive)
    
    # 步骤3: 取交集 - 只有同时在椭圆内且是暗色的区域才是鱼鳔
    mask = cv2.bitwise_and(ellipse_mask, dark_mask)
    
    # 步骤4: 形态学处理，填充空洞，平滑边界
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)  # 多轮闭运算填小空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # 去除噪声
    
    # 步骤5: 找到最大的连通区域（鱼鳔应该是最大的暗色椭圆区域）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # 找到最大的区域（排除背景）
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    
    # 步骤6: 填充内部空洞（中间留白），使区域完整
    inv = cv2.bitwise_not(mask)  # 255=背景与空洞, 0=物体
    inv_padded = np.pad(inv, ((1, 1), (1, 1)), constant_values=255)  # 外扩一圈，保证 (0,0) 在背景上
    ff_h, ff_w = inv_padded.shape[0] + 2, inv_padded.shape[1] + 2
    cv2.floodFill(inv_padded, np.zeros((ff_h, ff_w), dtype=np.uint8), (0, 0), 0)  # 从外缘把背景连通区填成 0
    inv = inv_padded[1:-1, 1:-1]  # 去掉 padding
    # 此时 inv 中 255 仅剩内部空洞，与原始 mask 取并集即得到填满的掩码
    mask = cv2.bitwise_or(mask, inv)
    
    # 鱼鳔=0，其他=255（便于后续“遮住鱼鳔、保留其他”等用途）
    mask = cv2.bitwise_not(mask)
    
    # 步骤7: 向外膨胀鱼鳔区域，覆盖周围的黑色轮廓
    if expand_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bladder_region = (mask == 0).astype(np.uint8) * 255  # 鱼鳔=255 便于膨胀
        bladder_region = cv2.dilate(bladder_region, kernel, iterations=expand_pixels)
        mask = np.where(bladder_region > 0, 0, 255).astype(np.uint8)  # 膨胀后仍为鱼鳔=0
    
    # 步骤8: 若识别区域周围有黑色/暗色部分则继续覆盖；遇到非黑色则停止扩散
    if expand_into_dark:
        # 仅当相邻像素为黑色/深色时才并入；灰度高于此阈值视为非黑色，停止扩散
        expand_dark_threshold = dark_threshold + 22  # 扩散时再放宽，覆盖边缘暗区
        dark_around = (gray < expand_dark_threshold).astype(np.uint8) * 255
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bladder_region = (mask == 0).astype(np.uint8) * 255
        prev_sum = -1
        max_iters = 200
        for _ in range(max_iters):
            # 向外扩 1 像素，只保留扩到暗区的部分
            dilated = cv2.dilate(bladder_region, kernel_small)
            dilated = cv2.bitwise_and(dilated, dark_around)
            bladder_region = cv2.bitwise_or(bladder_region, dilated)
            cur_sum = int(np.sum(bladder_region > 0))
            if cur_sum == prev_sum:
                break
            prev_sum = cur_sum
        mask = np.where(bladder_region > 0, 0, 255).astype(np.uint8)
    
    # 步骤9: 仅当区域为闭合图形时，向中心填充内部空洞；非闭合则不填充
    # 用 flood-fill 从边缘标记“外部”，剩余未被连通的 255 即为真正的内部空洞（闭合才有）
    bladder_region = (mask == 0).astype(np.uint8) * 255
    if np.sum(bladder_region > 0) > 0:
        inv_bladder = cv2.bitwise_not(bladder_region)  # 255=背景+空洞，0=鱼鳔
        inv_padded = np.pad(inv_bladder, ((1, 1), (1, 1)), constant_values=255)
        ff_mask = np.zeros((inv_padded.shape[0] + 2, inv_padded.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(inv_padded, ff_mask, (0, 0), 0)  # 从边缘填满所有与外界连通的 255
        inv_bladder = inv_padded[1:-1, 1:-1]
        internal_holes = (inv_bladder == 255).astype(np.uint8) * 255  # 仅闭合区域会有内部空洞
        if np.sum(internal_holes > 0) > 0:
            # 闭合区域：存在内部空洞，向中心填充完整
            bladder_region = cv2.bitwise_or(bladder_region, internal_holes)
            mask = np.where(bladder_region > 0, 0, 255).astype(np.uint8)
        # 非闭合图形：internal_holes 为空，不填充，保持 mask 不变
    
    # 步骤10: 若有向内凹的形状，只填充贴近原边界的凹陷（更贴合原形）
    bladder_region = (mask == 0).astype(np.uint8) * 255
    if np.sum(bladder_region > 0) > 0:
        contours, _ = cv2.findContours(bladder_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            area_orig = cv2.contourArea(main_contour)
            hull = cv2.convexHull(main_contour)
            area_hull = cv2.contourArea(hull)
            if area_hull > area_orig:
                hull_mask = np.zeros_like(bladder_region)
                cv2.fillConvexPoly(hull_mask, hull, 255)
                # 只保留凸包中“贴近原边界”的凹陷：距原区域不超过 max_fill_dist 像素才填充
                max_fill_dist = 25
                dist_to_bladder = cv2.distanceTransform(
                    (255 - bladder_region).astype(np.uint8), cv2.DIST_L2, 5
                )
                fill_candidates = (hull_mask > 0) & (bladder_region == 0)
                fill_keep = fill_candidates & (dist_to_bladder <= max_fill_dist)
                bladder_region = np.where(
                    (bladder_region > 0) | fill_keep, 255, 0
                ).astype(np.uint8)
                mask = np.where(bladder_region > 0, 0, 255).astype(np.uint8)
    
    # 步骤11: 识别区域内部填充，保证为实心（无内部空洞）
    bladder_region = (mask == 0).astype(np.uint8) * 255
    if np.sum(bladder_region > 0) > 0:
        inv_bladder = cv2.bitwise_not(bladder_region)  # 255=背景+空洞，0=鱼鳔
        inv_padded = np.pad(inv_bladder, ((1, 1), (1, 1)), constant_values=255)
        ff_mask = np.zeros((inv_padded.shape[0] + 2, inv_padded.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(inv_padded, ff_mask, (0, 0), 0)  # 从边缘填满与外界连通的部分
        inv_bladder = inv_padded[1:-1, 1:-1]
        internal_holes = (inv_bladder == 255).astype(np.uint8) * 255
        if np.sum(internal_holes > 0) > 0:
            bladder_region = cv2.bitwise_or(bladder_region, internal_holes)
        mask = np.where(bladder_region > 0, 0, 255).astype(np.uint8)
    
    return mask


def create_center_ellipse(shape: Tuple[int, int], 
                          ratio: float, 
                          offset_x: float = 0.0,
                          offset_y: float = 0.0) -> np.ndarray:
    """在图像中心创建椭圆形掩码"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 计算中心位置
    center_x = int(w / 2 + offset_x * w * 0.3)
    center_y = int(h / 2 + offset_y * h * 0.3)
    
    # 计算椭圆尺寸
    ellipse_w = int(w * ratio)
    ellipse_h = int(h * ratio)
    
    # 绘制椭圆
    cv2.ellipse(mask, (center_x, center_y), (ellipse_w//2, ellipse_h//2), 0, 0, 360, 255, -1)
    
    return mask


def detect_dark_regions(gray: np.ndarray, 
                        threshold: int = 65,
                        adaptive: bool = True) -> np.ndarray:
    """检测暗色区域（放宽：保留灰度 < threshold+22 的像素）"""
    # 预处理：降噪
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 放宽上限：允许略高于 threshold 的暗区参与
    dark_cap = min(255, threshold + 22)
    
    if adaptive:
        # 方法1: 简单阈值（主判据）
        _, thresh1 = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 方法2: Otsu，与放宽阈值相交
        _, thresh2 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh2 = cv2.bitwise_and(thresh2, (filtered < dark_cap).astype(np.uint8) * 255)
        
        # 方法3: 自适应阈值，保留放宽暗色
        thresh3 = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        thresh3 = cv2.bitwise_and(thresh3, (filtered < dark_cap).astype(np.uint8) * 255)
        
        mask = cv2.bitwise_or(thresh1, thresh2)
        mask = cv2.bitwise_or(mask, thresh3)
        # 最终截断：保留灰度 < threshold+22 的像素（再放宽）
        mask = cv2.bitwise_and(mask, (filtered < dark_cap).astype(np.uint8) * 255)
    else:
        _, mask = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return mask


def main(input_dir="data", output_dir="masks", visualize=False,
         dark_threshold=65, ellipse_ratio=0.65, center_x_offset=0.0, center_y_offset=0.0,
         expand_pixels=5, expand_into_dark=True, test_image=None):
    """
    主函数 - 处理所有图像，生成掩码（使用hybrid方法）
    
    Args:
        test_image: 如果指定，只处理这一张图像（用于测试）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 如果指定了test_image，只处理这一张
    if test_image:
        img_paths = [str(Path(test_image))]
        print(f"测试模式: 只处理单张图像")
    else:
        # 查找图像（支持 jpg 与 tif 格式）
        img_paths = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.tif', '*.TIF', '*.tiff', '*.TIFF', '*.png', '*.PNG']:
            img_paths.extend(input_path.glob(f"**/{ext}"))
        
        img_paths = sorted([str(p) for p in img_paths])
    
    print(f"找到 {len(img_paths)} 张图像")
    
    if len(img_paths) == 0:
        print(f"警告: 在 {input_dir} 中未找到图像")
        print(f"查找格式: jpg, jpeg, tif, tiff, png")
        return
    
    print(f"检测方法: hybrid")
    print(f"椭圆比例: {ellipse_ratio}")
    print(f"暗色阈值: {dark_threshold}")
    print(f"膨胀像素: {expand_pixels}")
    print(f"向周围暗区继续覆盖: {'是' if expand_into_dark else '否'}")
    print()
    
    # 处理每张图像
    for i, path in enumerate(img_paths, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[{i}/{len(img_paths)}] 无法读取: {path}")
            continue
        
        try:
            # 使用hybrid检测方法
            mask = detect_swim_bladder_region(
                img,
                ellipse_ratio=ellipse_ratio,
                center_x_offset=center_x_offset,
                center_y_offset=center_y_offset,
                dark_threshold=dark_threshold,
                adaptive=True,
                expand_pixels=expand_pixels,
                expand_into_dark=expand_into_dark
            )
            
            # 确保掩码尺寸与原图尺寸一致
            img_h, img_w = img.shape[:2]
            mask_h, mask_w = mask.shape[:2]
            
            if (mask_h != img_h) or (mask_w != img_w):
                # 如果尺寸不一致，调整掩码尺寸以匹配原图
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                print(f"  警告: 掩码尺寸已调整 {mask_h}x{mask_w} -> {img_h}x{img_w}")
            
            # 计算统计信息（鱼鳔区域 = mask==0）
            mask_pixels = np.sum(mask == 0)
            mask_ratio = mask_pixels / (img_h * img_w) * 100
            
            # 保存掩码（确保尺寸与原图一致）
            if test_image:
                # 测试模式下，直接使用文件名
                rel_path = Path(path).name
                out_file = output_path / (Path(path).stem + "_mask.png")
            else:
                rel_path = Path(path).relative_to(input_path)
                out_file = output_path / rel_path.parent / (rel_path.stem + "_mask.png")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 验证掩码尺寸
            assert mask.shape[0] == img_h and mask.shape[1] == img_w, \
                f"掩码尺寸 ({mask.shape[0]}x{mask.shape[1]}) 与原图尺寸 ({img_h}x{img_w}) 不匹配"
            
            cv2.imwrite(str(out_file), mask)
            print(f"[{i}/{len(img_paths)}] {rel_path}")
            print(f"  掩码像素: {mask_pixels} ({mask_ratio:.2f}%)")
            print(f"  已保存: {out_file}")
            
            # 可视化
            if visualize:
                if test_image:
                    vis_file = output_path / (Path(path).stem + "_vis.jpg")
                else:
                    vis_file = output_path / rel_path.parent / (rel_path.stem + "_vis.jpg")
                vis_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 创建可视化
                overlay = img.copy()
                overlay[mask == 0] = [0, 255, 255]  # 青色叠加在鱼鳔区域（mask==0）
                vis_img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                
                # 掩码热力图
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                
                # 组合显示：原图 | 叠加图 | 掩码图
                vis_combined = np.hstack([img, vis_img, mask_colored])
                
                # 添加信息标签
                info_lines = [
                    f"Method: hybrid",
                    f"Ellipse ratio: {ellipse_ratio}",
                    f"Dark threshold: {dark_threshold}",
                    f"Expand: {expand_pixels} px",
                    f"Mask: {mask_pixels} px ({mask_ratio:.2f}%)"
                ]
                
                y_offset = 30
                for line in info_lines:
                    cv2.putText(vis_combined, line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset += 30
                
                cv2.imwrite(str(vis_file), vis_combined)
                print(f"  可视化: {vis_file}")
            
        except Exception as e:
            print(f"[{i}/{len(img_paths)}] 处理失败 {path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n完成！共处理 {len(img_paths)} 张图像")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="鱼鳔掩码生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（使用测试好的参数：椭圆比例0.6，暗色阈值60）
  python generate_masks.py --input_dir data --output_dir masks

  # 测试单张图像并可视化
  python generate_masks.py --input_dir data --output_dir masks_test --test_image data/0分/BUF279.jpg --visualize

  # 调整参数
  python generate_masks.py --input_dir data --output_dir masks --ellipse_ratio 0.6 --dark_threshold 60 --visualize
        """
    )
    
    parser.add_argument("--input_dir", type=str, default="data",
                       help="输入图像目录 (default: data)")
    parser.add_argument("--output_dir", type=str, default="masks",
                       help="输出掩码目录 (default: masks)")
    parser.add_argument("--ellipse_ratio", type=float, default=0.65,
                       help="椭圆相对图像大小的比例 (0-1, default: 0.65)")
    parser.add_argument("--center_x_offset", type=float, default=0.0,
                       help="椭圆中心X方向偏移，比例值 -0.3到0.3 (default: 0.0)")
    parser.add_argument("--center_y_offset", type=float, default=0.0,
                       help="椭圆中心Y方向偏移，比例值 -0.3到0.3 (default: 0.0)")
    parser.add_argument("--dark_threshold", type=int, default=65,
                       help="暗色阈值 0-255，越小越严格 (default: 65)")
    parser.add_argument("--expand_pixels", type=int, default=5,
                       help="鱼鳔区域向外膨胀像素数，覆盖周围黑色轮廓 (default: 3)")
    parser.add_argument("--no_expand_into_dark", action="store_true",
                       help="禁用「向周围黑色继续覆盖」；默认会向相邻暗区扩展直到无暗色可并")
    parser.add_argument("--visualize", action="store_true",
                       help="生成可视化图像")
    parser.add_argument("--test_image", type=str, default=None,
                       help="测试单张图像路径（用于调试）")
    
    args = parser.parse_args()
    
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ellipse_ratio=args.ellipse_ratio,
        center_x_offset=args.center_x_offset,
        center_y_offset=args.center_y_offset,
        dark_threshold=args.dark_threshold,
        expand_pixels=args.expand_pixels,
        expand_into_dark=not args.no_expand_into_dark,
        visualize=args.visualize,
        test_image=args.test_image
    )