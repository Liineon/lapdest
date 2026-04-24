"""
背景掩码生成工具

专门用于生成鱼和背景的分割掩码，将鱼体区域标记为白色，背景区域标记为黑色。
使用与generate_masks_SB.py相同的方法结构。

使用方法:
  # 基本使用（默认启用自适应中心，根据鱼鳔位置自动调整椭圆中心）
  python generate_masks_BG.py --input_dir data --output_dir masks_BG

  # 禁用自适应中心（如果需要）
  python generate_masks_BG.py --input_dir data --output_dir masks_BG --no_adaptive_center


  # 测试单张图像并可视化
  python generate_masks_BG.py --input_dir data --output_dir masks_BG_test --test_image data/0分/BUF279.jpg --visualize

  # 调整参数
  python generate_masks_BG.py --input_dir data --output_dir masks_BG --ellipse_ratio 0.8 --bright_threshold 150 --visualize
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Optional


def detect_fish_region(img: np.ndarray, 
                       method: str = "hybrid",
                       ellipse_ratio: float = 0.5,
                       center_x_offset: float = 0.0,
                       center_y_offset: float = 0.0,
                       bright_threshold: int = 140,
                       adaptive: bool = True,
                       adaptive_center: bool = True,
                       dark_threshold: int = 60) -> np.ndarray:
    """
    检测鱼体区域并生成掩码
    
    方法: "hybrid" - 结合椭圆形状、亮色检测和暗侧检测（完善版，推荐，默认）
    - 使用椭圆形状约束
    - 检测亮色区域（鱼体通常较亮）
    - 如果检测到鱼鳔且两侧亮度差异明显，在较暗的一侧应用暗色检测
    - 合并多种检测结果，提高准确性
    
    Args:
        img: 输入BGR图像
        method: 检测方法 ("hybrid")
        ellipse_ratio: 椭圆相对图像大小的比例 (0-1, 默认: 0.7，鱼体比鱼鳔大)
        center_x_offset: 椭圆中心X方向偏移（比例 -0.3到0.3）
        center_y_offset: 椭圆中心Y方向偏移（比例 -0.3到0.3）
        bright_threshold: 亮色阈值 (0-255, 默认: 140，鱼体通常较亮)
        adaptive: 是否使用自适应阈值
        adaptive_center: 是否根据鱼鳔位置自适应调整椭圆中心（默认: True）
        dark_threshold: 检测鱼鳔时的暗色阈值 (0-255, 默认: 60)
    
    Returns:
        二值掩码（白色=鱼体区域）
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 如果启用自适应中心，先检测鱼鳔位置
    actual_center_x_offset = center_x_offset
    actual_center_y_offset = center_y_offset
    swim_bladder_center = None
    if adaptive_center:
        swim_bladder_center = detect_swim_bladder_center(gray, dark_threshold)
        if swim_bladder_center is not None:
            # 计算相对于图像中心的偏移（归一化到 -0.3 到 0.3 的范围）
            sb_x, sb_y = swim_bladder_center
            actual_center_x_offset = (sb_x - w / 2) / (w * 0.3)
            actual_center_y_offset = (sb_y - h / 2) / (h * 0.3)
            # 限制偏移范围
            actual_center_x_offset = np.clip(actual_center_x_offset, -0.3, 0.3)
            actual_center_y_offset = np.clip(actual_center_y_offset, -0.3, 0.3)
    
    if method == "hybrid":
        # 方法：结合椭圆形状、亮色检测和暗侧检测（完善版）
        # 步骤1: 创建中心椭圆作为候选区域（鱼体比鱼鳔大，所以用更大的椭圆）
        ellipse_mask = create_center_ellipse(gray.shape, ellipse_ratio, actual_center_x_offset, actual_center_y_offset)
        
        # 步骤2: 检测亮色区域（鱼体通常比背景亮）
        bright_mask = detect_bright_regions(gray, bright_threshold, adaptive)
        
        # 步骤3: 取交集 - 只有同时在椭圆内且是亮色的区域才是鱼体
        mask = cv2.bitwise_and(ellipse_mask, bright_mask)
        
        # 步骤4: 整合dark_side逻辑 - 如果检测到鱼鳔，考虑暗侧检测
        if adaptive_center and swim_bladder_center is not None:
                sb_x, sb_y = swim_bladder_center
                h, w = gray.shape
                
                # 以鱼鳔为中心，将图像分为左右两侧
                left_mask = np.zeros((h, w), dtype=np.uint8)
                right_mask = np.zeros((h, w), dtype=np.uint8)
                left_mask[:, :sb_x] = 255
                right_mask[:, sb_x:] = 255
                
                # 应用椭圆约束
                left_mask = cv2.bitwise_and(left_mask, ellipse_mask)
                right_mask = cv2.bitwise_and(right_mask, ellipse_mask)
                
                # 计算两侧的平均亮度
                left_pixels = gray[left_mask > 0]
                right_pixels = gray[right_mask > 0]
                
                if len(left_pixels) > 0 and len(right_pixels) > 0:
                    left_mean_brightness = np.mean(left_pixels)
                    right_mean_brightness = np.mean(right_pixels)
                    
                    # 如果一侧明显较暗（差异超过10%），在该侧应用暗色检测
                    brightness_diff = abs(left_mean_brightness - right_mean_brightness) / max(left_mean_brightness, right_mean_brightness)
                    if brightness_diff > 0.1:  # 亮度差异超过10%
                        # 选择较暗的一侧
                        if left_mean_brightness < right_mean_brightness:
                            dark_side_mask = left_mask.copy()
                        else:
                            dark_side_mask = right_mask.copy()
                        
                        # 在暗侧检测暗色区域
                        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
                        selected_pixels = gray[dark_side_mask > 0]
                        if len(selected_pixels) > 0:
                            mean_brightness = np.mean(selected_pixels)
                            threshold = int(mean_brightness * 0.8)
                            threshold = max(30, min(120, threshold))
                        else:
                            threshold = dark_threshold
                        
                        _, dark_mask = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY_INV)
                        dark_mask = cv2.bitwise_and(dark_mask, dark_side_mask)
                        
                        # 形态学处理暗色掩码
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                        
                        # 将暗色检测结果与亮色检测结果合并（取并集）
                        mask = cv2.bitwise_or(mask, dark_mask)
        
        # 步骤5: 形态学处理，填充空洞，平滑边界
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 填充空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # 去除噪声
        
        # 步骤6: 找到最大的连通区域（鱼体应该是最大的区域）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # 找到最大的区域（排除背景）
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255
        
        # 步骤7: 填充内部空洞，确保鱼体区域内部全部被填充（保留原有区域）
        mask = fill_internal_holes(mask)
        
        return mask
    
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_swim_bladder_center(gray: np.ndarray, dark_threshold: int = 60) -> Optional[Tuple[int, int]]:
    """
    检测鱼鳔的质心位置（改进版：更鲁棒，尝试多个阈值）
    
    Args:
        gray: 灰度图像
        dark_threshold: 暗色阈值（基准值，会尝试多个阈值）
    
    Returns:
        鱼鳔质心坐标 (x, y)，如果未检测到则返回 None
    """
    h, w = gray.shape
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 尝试多个阈值（从低到高），提高检测成功率
    thresholds = [
        dark_threshold - 20,  # 更宽松的阈值
        dark_threshold - 10,
        dark_threshold,       # 原始阈值
        dark_threshold + 10,
        dark_threshold + 20,  # 更严格的阈值
    ]
    
    # 确保阈值在有效范围内
    thresholds = [max(30, min(120, t)) for t in thresholds]
    thresholds = sorted(set(thresholds))  # 去重并排序
    
    # 尝试不同的中心区域约束宽度
    weight_widths = [0.8, 0.7, 0.6]  # 从宽松到严格
    
    best_center = None
    best_area = 0
    
    for threshold in thresholds:
        for weight_width in weight_widths:
            # 检测暗色区域（鱼鳔通常是暗色的）
            _, dark_mask = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # 应用中心区域约束（鱼鳔通常在中心）
            center_mask = create_center_weight_mask(gray.shape, weight_width=weight_width)
            dark_mask = cv2.bitwise_and(dark_mask, center_mask)
            
            # 形态学处理
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 找到最大的连通区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
            if num_labels > 1:
                # 找到最大的区域（排除背景）
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_idx = np.argmax(areas)
                largest_area = areas[largest_idx]
                
                # 过滤：确保区域足够大（至少是图像面积的0.5%），但不要太大（不超过10%）
                min_area = h * w * 0.005
                max_area = h * w * 0.10
                
                if min_area <= largest_area <= max_area:
                    centroid = centroids[largest_idx + 1]
                    center_x, center_y = int(centroid[0]), int(centroid[1])
                    
                    # 检查质心是否在合理范围内（图像中心附近）
                    center_dist = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
                    max_dist = min(w, h) * 0.3  # 距离中心不超过30%的图像尺寸
                    
                    if center_dist <= max_dist:
                        # 选择面积最接近期望值的（鱼鳔通常占图像面积的1-5%）
                        expected_area = h * w * 0.02  # 期望2%的面积
                        area_score = 1.0 / (1.0 + abs(largest_area - expected_area) / expected_area)
                        
                        # 如果这个结果更好，更新最佳结果
                        if largest_area > best_area * 0.5:  # 至少要有最佳结果的一半面积
                            if best_center is None or abs(largest_area - expected_area) < abs(best_area - expected_area):
                                best_center = (center_x, center_y)
                                best_area = largest_area
    
    # 如果找到了合适的中心，返回它
    if best_center is not None:
        return best_center
    
    # 如果所有方法都失败，尝试使用Otsu自适应阈值
    try:
        _, otsu_mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        center_mask = create_center_weight_mask(gray.shape, weight_width=0.7)
        otsu_mask = cv2.bitwise_and(otsu_mask, center_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu_mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas)
            largest_area = areas[largest_idx]
            
            min_area = h * w * 0.005
            max_area = h * w * 0.10
            
            if min_area <= largest_area <= max_area:
                centroid = centroids[largest_idx + 1]
                center_x, center_y = int(centroid[0]), int(centroid[1])
                center_dist = np.sqrt((center_x - w/2)**2 + (center_y - h/2)**2)
                max_dist = min(w, h) * 0.3
                
                if center_dist <= max_dist:
                    return (center_x, center_y)
    except:
        pass
    
    # 如果所有方法都失败，返回 None（会使用默认的中心位置）
    return None


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


def create_center_weight_mask(shape: Tuple[int, int], 
                              weight_width: float = 0.8) -> np.ndarray:
    """创建中心加权掩码（中间区域权重高，边缘权重低）"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 中心区域矩形（鱼体比鱼鳔大，所以用更大的区域）
    x1 = int(w * (0.5 - weight_width/2))
    y1 = int(h * (0.5 - weight_width/2))
    x2 = int(w * (0.5 + weight_width/2))
    y2 = int(h * (0.5 + weight_width/2))
    
    mask[y1:y2, x1:x2] = 255
    
    return mask


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """
    填充掩码内部的空洞，确保鱼体区域内部全部被填充
    只填充被鱼体区域完全包围的空洞，保留所有原有的鱼体区域
    
    Args:
        mask: 二值掩码（白色=鱼体区域，黑色=背景）
    
    Returns:
        填充后的掩码（保留原有区域，填充内部空洞）
    """
    # 创建填充用的图像（添加边界，确保边界是背景）
    h, w = mask.shape
    # 创建一个更大的图像，边界填充为0（背景）
    filled = np.zeros((h + 2, w + 2), dtype=np.uint8)
    filled[1:h+1, 1:w+1] = mask
    
    # 从边界开始floodFill，标记所有与边界相连的0区域（这些是背景，不是空洞）
    # 使用一个特殊值128来标记背景区域
    mask_flood = filled.copy()
    cv2.floodFill(mask_flood, None, (0, 0), 128)  # 左上角
    cv2.floodFill(mask_flood, None, (w + 1, 0), 128)  # 右上角
    cv2.floodFill(mask_flood, None, (0, h + 1), 128)  # 左下角
    cv2.floodFill(mask_flood, None, (w + 1, h + 1), 128)  # 右下角
    
    # 提取原始区域
    result = mask_flood[1:h+1, 1:w+1].copy()
    
    # 现在result中：
    # - 255: 原有的鱼体区域（保持不变）
    # - 128: 与边界相连的背景区域（需要恢复为0）
    # - 0: 内部空洞（需要填充为255）
    
    # 填充内部空洞（0区域）
    result[result == 0] = 255
    
    # 将背景区域（128）恢复为0
    result[result == 128] = 0
    
    return result


def detect_bright_regions(gray: np.ndarray, 
                         threshold: int = 140,
                         adaptive: bool = True) -> np.ndarray:
    """检测亮色区域（鱼体通常比背景亮）"""
    # 预处理：降噪
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    if adaptive:
        # 自适应阈值（更灵活）
        # 使用多个阈值方法组合
        # 方法1: 简单阈值（检测亮色）
        _, thresh1 = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
        
        # 方法2: Otsu自动阈值
        _, thresh2 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 方法3: 自适应阈值
        thresh3 = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        # 只保留亮色像素
        thresh3 = cv2.bitwise_and(thresh3, (filtered > threshold * 0.7).astype(np.uint8) * 255)
        
        # 合并方法：使用并集，捕获更多亮色区域
        mask = cv2.bitwise_or(thresh1, thresh2)
        mask = cv2.bitwise_or(mask, thresh3)
    else:
        # 简单阈值（检测亮色）
        _, mask = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
    
    return mask


def main(input_dir="data", output_dir="masks_BG", method="opencv",
         min_area=150, max_area=4000, min_circularity=0.3, max_circularity=0.9,
         min_solidity=0.6, use_clahe=True, morph_kernel_size=3, visualize=False,
         bright_threshold=140, ellipse_ratio=0.7, center_x_offset=0.0, center_y_offset=0.0,
         adaptive_center=True, dark_threshold=60):
    """
    主函数 - 处理所有图像，生成掩码
    
    为了保持向后兼容性，保留了旧的参数，但现在默认使用新的hybrid方法
    """
    # 如果method是"opencv"，自动使用hybrid方法
    if method == "opencv":
        method = "hybrid"
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找图像
    img_paths = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        img_paths.extend(input_path.glob(f"**/{ext}"))
    
    img_paths = sorted([str(p) for p in img_paths])
    print(f"找到 {len(img_paths)} 张图像")
    
    if len(img_paths) == 0:
        print(f"警告: 在 {input_dir} 中未找到图像")
        print(f"查找模式: {input_dir}/**/*.jpg")
        return
    
    print(f"检测方法: {method}")
    if method == "hybrid":
        print(f"椭圆比例: {ellipse_ratio}")
        print(f"亮色阈值: {bright_threshold}")
    print(f"自适应中心: {'启用' if adaptive_center else '禁用'}（根据鱼鳔位置调整椭圆中心）")
    if adaptive_center:
        print(f"鱼鳔检测阈值: {dark_threshold}")
    print()
    
    # 处理每张图像
    for i, path in enumerate(img_paths, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[{i}/{len(img_paths)}] 无法读取: {path}")
            continue
        
        try:
            # 使用检测方法
            # 注意：detect_fish_region 内部会处理鱼鳔检测，这里我们需要单独检测以便输出日志
            swim_bladder_info = ""
            if adaptive_center:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                swim_bladder_center = detect_swim_bladder_center(gray, dark_threshold)
                if swim_bladder_center is not None:
                    sb_x, sb_y = swim_bladder_center
                    img_h, img_w = img.shape[:2]
                    swim_bladder_info = f"  鱼鳔位置: ({sb_x}, {sb_y})"
                else:
                    swim_bladder_info = "  鱼鳔位置: 未检测到（使用默认中心）"
            
            mask = detect_fish_region(
                img,
                method=method,
                ellipse_ratio=ellipse_ratio,
                center_x_offset=center_x_offset,
                center_y_offset=center_y_offset,
                bright_threshold=bright_threshold,
                adaptive=True,
                adaptive_center=adaptive_center,
                dark_threshold=dark_threshold
            )
            
            # 确保掩码尺寸与原图尺寸一致
            img_h, img_w = img.shape[:2]
            mask_h, mask_w = mask.shape[:2]
            
            if (mask_h != img_h) or (mask_w != img_w):
                # 如果尺寸不一致，调整掩码尺寸以匹配原图
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                print(f"  警告: 掩码尺寸已调整 {mask_h}x{mask_w} -> {img_h}x{img_w}")
            
            # 计算统计信息
            mask_pixels = np.sum(mask > 0)
            mask_ratio = mask_pixels / (img_h * img_w) * 100
            
            # 保存掩码（确保尺寸与原图一致）
            rel_path = Path(path).relative_to(input_path)
            out_file = output_path / rel_path.parent / (rel_path.stem + "_mask.png")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 验证掩码尺寸
            assert mask.shape[0] == img_h and mask.shape[1] == img_w, \
                f"掩码尺寸 ({mask.shape[0]}x{mask.shape[1]}) 与原图尺寸 ({img_h}x{img_w}) 不匹配"
            
            cv2.imwrite(str(out_file), mask)
            print(f"[{i}/{len(img_paths)}] {rel_path}")
            if swim_bladder_info:
                print(swim_bladder_info)
            print(f"  掩码像素: {mask_pixels} ({mask_ratio:.2f}%)")
            print(f"  已保存: {out_file}")
            
            # 可视化
            if visualize:
                vis_file = output_path / rel_path.parent / (rel_path.stem + "_vis.jpg")
                vis_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 创建可视化
                overlay = img.copy()
                overlay[mask > 0] = [0, 255, 255]  # 青色叠加
                vis_img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                
                # 掩码热力图
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                
                # 组合显示：原图 | 叠加图 | 掩码图
                vis_combined = np.hstack([img, vis_img, mask_colored])
                
                # 添加信息标签
                info_lines = [
                    f"Method: {method}",
                    f"Ellipse ratio: {ellipse_ratio}",
                    f"Bright threshold: {bright_threshold}",
                    f"Adaptive center: {adaptive_center}",
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
        description="鱼和背景分割掩码生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（使用测试好的参数：hybrid方法，椭圆比例0.7，亮色阈值140）
  python generate_masks_BG.py --input_dir data --output_dir masks_BG

  # 测试单张图像并可视化
  python generate_masks_BG.py --input_dir data --output_dir masks_BG_test --test_image data/0分/BUF279.jpg --visualize

  # 调整hybrid方法参数
  python generate_masks_BG.py --input_dir data --output_dir masks_BG --method hybrid --ellipse_ratio 0.7 --bright_threshold 140 --visualize
        """
    )
    
    parser.add_argument("--input_dir", type=str, default="data",
                       help="输入图像目录 (default: data)")
    parser.add_argument("--output_dir", type=str, default="masks_BG",
                       help="输出掩码目录 (default: masks_BG)")
    parser.add_argument("--method", type=str, default="hybrid",
                       choices=["opencv", "hybrid"],
                       help="检测方法: opencv(兼容,自动转为hybrid), hybrid(结合椭圆、亮色和暗侧检测,推荐,默认)")
    parser.add_argument("--ellipse_ratio", type=float, default=0.7,
                       help="椭圆相对图像大小的比例 (0-1, default: 0.7，鱼体比鱼鳔大)")
    parser.add_argument("--center_x_offset", type=float, default=0.0,
                       help="椭圆中心X方向偏移，比例值 -0.3到0.3 (default: 0.0)")
    parser.add_argument("--center_y_offset", type=float, default=0.0,
                       help="椭圆中心Y方向偏移，比例值 -0.3到0.3 (default: 0.0)")
    parser.add_argument("--bright_threshold", type=int, default=140,
                       help="亮色阈值 0-255 (default: 140，鱼体通常较亮)")
    parser.add_argument("--no_adaptive_center", action="store_true",
                       help="禁用自适应中心（默认启用，根据鱼鳔位置调整椭圆中心）")
    parser.add_argument("--dark_threshold", type=int, default=60,
                       help="检测鱼鳔时的暗色阈值 0-255 (default: 60，仅在启用adaptive_center时使用)")
    parser.add_argument("--visualize", action="store_true",
                       help="生成可视化图像")
    parser.add_argument("--test_image", type=str, default=None,
                       help="测试单张图像路径（用于调试）")
    
    # 为了向后兼容，保留旧参数（但不使用）
    parser.add_argument("--min_area", type=int, default=150, help="(已弃用，保留用于兼容)")
    parser.add_argument("--max_area", type=int, default=4000, help="(已弃用，保留用于兼容)")
    parser.add_argument("--min_circularity", type=float, default=0.3, help="(已弃用，保留用于兼容)")
    parser.add_argument("--max_circularity", type=float, default=0.9, help="(已弃用，保留用于兼容)")
    parser.add_argument("--min_solidity", type=float, default=0.6, help="(已弃用，保留用于兼容)")
    parser.add_argument("--no_clahe", action="store_true", help="(已弃用，保留用于兼容)")
    parser.add_argument("--morph_kernel_size", type=int, default=3, help="(已弃用，保留用于兼容)")
    
    args = parser.parse_args()
    
    # 如果指定了test_image，只处理这一张
    test_image = args.test_image
    
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        ellipse_ratio=args.ellipse_ratio,
        center_x_offset=args.center_x_offset,
        center_y_offset=args.center_y_offset,
        bright_threshold=args.bright_threshold,
        adaptive_center=not args.no_adaptive_center,
        dark_threshold=args.dark_threshold,
        visualize=args.visualize,
        # 其他参数保留但不使用
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        max_circularity=args.max_circularity,
        min_solidity=args.min_solidity,
        use_clahe=not args.no_clahe,
        morph_kernel_size=args.morph_kernel_size
    )
