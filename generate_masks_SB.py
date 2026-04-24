"""
鱼鳔掩码生成工具（SAM）

使用 Segment Anything Model (SAM) 生成鱼鳔区域掩码，以图像中心为 prompt 引导分割。

使用方法:
  # 基本使用（需先下载 SAM 模型权重）
  python generate_masks_SB.py --input_dir data --output_dir masks --sam_checkpoint sam_vit_b.pth --sam_model_type vit_b

  # 测试单张图像并可视化
  python generate_masks_SB.py --input_dir data --output_dir masks_test --sam_checkpoint sam_vit_b.pth --test_image data/0分/BUF279.jpg --visualize

  # 调整中心与椭圆约束
  python generate_masks_SB.py --input_dir data --output_dir masks --sam_checkpoint sam_vit_b.pth --ellipse_ratio 0.5 --center_x_offset 0.1
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, TYPE_CHECKING

def read_image_as_bgr(path: str) -> Optional[np.ndarray]:
    """
    读取图片为 BGR 8-bit，兼容 jpg/png/tif/tiff，包括 16-bit TIF。
    若读取失败或格式不支持则返回 None。
    """
    img = cv2.imread(path)
    if img is None:
        # OpenCV 对部分 TIF 可能失败，尝试用 IMREAD_ANYDEPTH 再转 8-bit
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img is None:
            return None
        # 多通道（如 RGB）取前 3 通道
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 16-bit 转为 8-bit：线性缩放到 0-255
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# SAM模型相关导入（可选）
if TYPE_CHECKING:
    from segment_anything import SamPredictor

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    SamPredictor = None  # 类型: ignore
    print("警告: segment_anything 未安装，SAM方法将不可用。请运行: pip install git+https://github.com/facebookresearch/segment-anything.git")


def initialize_sam_predictor(checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda"):
    """
    初始化SAM模型预测器
    
    Args:
        checkpoint_path: SAM模型权重文件路径（如 sam_vit_b.pth）
        model_type: 模型类型 ("vit_h", "vit_l", "vit_b")
        device: 设备（CUDA，如 "cuda"）
    
    Returns:
        SamPredictor对象，如果失败则返回None
    """
    if not SAM_AVAILABLE:
        raise ImportError("segment_anything 未安装，无法使用SAM模型")
    
    try:
        # 检查文件是否存在
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM模型权重文件不存在: {checkpoint_path}")
        
        # 加载模型
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        predictor = SamPredictor(sam)
        print(f"SAM模型已加载: {model_type}, 权重: {checkpoint_path}, 设备: {device}")
        return predictor
    except Exception as e:
        print(f"初始化SAM模型失败: {e}")
        return None


def detect_swim_bladder_sam(img: np.ndarray, 
                            predictor,  # type: ignore  # SamPredictor
                            center_x_offset: float = 0.0,
                            center_y_offset: float = 0.0,
                            ellipse_ratio: float = 0.6) -> np.ndarray:
    """
    使用SAM模型检测鱼鳔区域并生成掩码
    
    策略：使用图像中心点作为prompt，引导SAM分割鱼鳔区域
    
    Args:
        img: 输入BGR图像
        predictor: SAM预测器对象
        center_x_offset: 中心点X方向偏移（比例 -0.3到0.3）
        center_y_offset: 中心点Y方向偏移（比例 -0.3到0.3）
        ellipse_ratio: 用于确定搜索范围的椭圆比例
    
    Returns:
        二值掩码（白色=鱼鳔区域）
    """
    if predictor is None:
        raise ValueError("SAM predictor未初始化")
    
    h, w = img.shape[:2]
    
    # 计算中心点坐标（作为prompt）
    center_x = int(w / 2 + center_x_offset * w * 0.3)
    center_y = int(h / 2 + center_y_offset * h * 0.3)
    
    # 设置图像
    # SAM需要RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    
    # 使用中心点作为prompt（前景点）
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([1])  # 1表示前景点
    
    # 可选：使用椭圆边界框作为辅助约束
    ellipse_w = int(w * ellipse_ratio)
    ellipse_h = int(h * ellipse_ratio)
    box_x1 = max(0, center_x - ellipse_w // 2)
    box_y1 = max(0, center_y - ellipse_h // 2)
    box_x2 = min(w, center_x + ellipse_w // 2)
    box_y2 = min(h, center_y + ellipse_h // 2)
    input_box = np.array([box_x1, box_y1, box_x2, box_y2])
    
    # 使用点+框进行预测
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box[None, :],
        multimask_output=True,
    )
    
    # 选择得分最高的掩码
    best_idx = np.argmax(scores)
    mask = masks[best_idx].astype(np.uint8) * 255
    
    # 可选：形态学后处理，填充空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 找到最大的连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255
    
    # 填充内部空洞
    mask = fill_internal_holes(mask)

    # 边缘平滑，减少锯齿
    mask = smooth_mask_edges(mask, blur_ksize=5, sigma=1.5)

    return mask


def smooth_mask_edges(mask: np.ndarray, blur_ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    """
    对二值掩码边缘做平滑，使轮廓更圆滑、减少锯齿。
    使用高斯模糊后重新二值化，在保持区域大致不变的前提下柔化边界。

    Args:
        mask: 二值掩码（0/255）
        blur_ksize: 高斯核大小（奇数），越大边缘越平滑
        sigma: 高斯标准差

    Returns:
        平滑后的二值掩码
    """
    if blur_ksize <= 1:
        return mask
    # 确保为奇数
    k = (blur_ksize | 1)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigma)
    # 阈值 127 保持与原区域大致一致
    return (blurred >= 127).astype(np.uint8) * 255


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """
    填充掩码内部的空洞，确保鱼鳔区域内部全部被填充
    只填充被鱼鳔区域完全包围的空洞，保留所有原有的鱼鳔区域
    
    Args:
        mask: 二值掩码（白色=鱼鳔区域，黑色=背景）
    
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
    # - 255: 原有的鱼鳔区域（保持不变）
    # - 128: 与边界相连的背景区域（需要恢复为0）
    # - 0: 内部空洞（需要填充为255）
    
    # 填充内部空洞（0区域）
    result[result == 0] = 255
    
    # 将背景区域（128）恢复为0
    result[result == 128] = 0
    
    return result


def main(input_dir="data", output_dir="masks", sam_predictor=None,
         visualize=False, ellipse_ratio=0.6, center_x_offset=0.0, center_y_offset=0.0,
         sam_checkpoint=None, sam_model_type="vit_b", sam_device="cuda", test_image=None):
    """
    主函数 - 使用 SAM 处理所有图像，生成鱼鳔掩码
    """
    if sam_predictor is None:
        if sam_checkpoint is None:
            raise ValueError("必须提供 --sam_checkpoint 参数")
        sam_predictor = initialize_sam_predictor(sam_checkpoint, sam_model_type, sam_device)
        if sam_predictor is None:
            raise RuntimeError("SAM模型初始化失败")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找图像（支持 jpg/jpeg/png/tif/tiff）
    img_paths = []
    exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG',
            '*.tif', '*.TIF', '*.tiff', '*.TIFF']
    for ext in exts:
        img_paths.extend(input_path.glob(f"**/{ext}"))
    
    img_paths = sorted([str(p) for p in img_paths])
    
    # 如果指定了test_image，只处理这一张
    if test_image:
        test_path = Path(test_image)
        if test_path.exists():
            img_paths = [str(test_path)]
            print(f"测试模式: 只处理单张图像 {test_path}")
        else:
            print(f"警告: 测试图像不存在: {test_path}")
            return
    
    print(f"找到 {len(img_paths)} 张图像")
    
    if len(img_paths) == 0:
        print(f"警告: 在 {input_dir} 中未找到图像")
        print(f"支持的格式: jpg, jpeg, png, tif, tiff")
        return
    
    print(f"SAM模型类型: {sam_model_type}")
    print(f"SAM权重: {sam_checkpoint}")
    print()
    
    # 处理每张图像
    for i, path in enumerate(img_paths, 1):
        img = read_image_as_bgr(path)
        if img is None:
            print(f"[{i}/{len(img_paths)}] 无法读取: {path}")
            continue
        
        try:
            if sam_predictor is None:
                print(f"[{i}/{len(img_paths)}] 错误: SAM predictor未初始化")
                continue
            mask = detect_swim_bladder_sam(
                img,
                predictor=sam_predictor,
                center_x_offset=center_x_offset,
                center_y_offset=center_y_offset,
                ellipse_ratio=ellipse_ratio
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
                    f"SAM model: {sam_model_type}",
                    f"Ellipse ratio: {ellipse_ratio}",
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
        description="鱼鳔掩码生成工具（SAM）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（需先下载 SAM 模型权重）
  python generate_masks_SB.py --input_dir data --output_dir masks --sam_checkpoint sam_vit_b.pth --sam_model_type vit_b

  # 测试单张图像并可视化
  python generate_masks_SB.py --input_dir data --output_dir masks_test --sam_checkpoint sam_vit_b.pth --test_image data/0分/BUF279.jpg --visualize

  # 调整中心与椭圆约束
  python generate_masks_SB.py --input_dir data --output_dir masks --sam_checkpoint sam_vit_b.pth --ellipse_ratio 0.5 --center_x_offset 0.1

注意: 需要先安装 segment_anything 并下载权重：
  pip install git+https://github.com/facebookresearch/segment-anything.git
  权重文件: sam_vit_b.pth / sam_vit_l.pth / sam_vit_h.pth
        """
    )
    parser.add_argument("--input_dir", type=str, default="data",
                        help="输入图像目录 (default: data)")
    parser.add_argument("--output_dir", type=str, default="masks",
                        help="输出掩码目录 (default: masks)")
    parser.add_argument("--sam_checkpoint", type=str, default=None, required=True,
                        help="SAM模型权重文件路径 (如 sam_vit_b.pth)")
    parser.add_argument("--sam_model_type", type=str, default="vit_b",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM模型类型 (default: vit_b)")
    parser.add_argument("--sam_device", type=str, default="cuda",
                        choices=["cuda"],
                        help="运行设备：CUDA (default: cuda)")
    parser.add_argument("--ellipse_ratio", type=float, default=0.6,
                        help="椭圆约束相对图像比例 0-1 (default: 0.6)")
    parser.add_argument("--center_x_offset", type=float, default=0.0,
                        help="中心点X偏移比例 -0.3~0.3 (default: 0.0)")
    parser.add_argument("--center_y_offset", type=float, default=0.0,
                        help="中心点Y偏移比例 -0.3~0.3 (default: 0.0)")
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化图像")
    parser.add_argument("--test_image", type=str, default=None,
                        help="仅处理指定单张图像（调试用）")

    args = parser.parse_args()
    if not SAM_AVAILABLE:
        parser.error("segment_anything 未安装。请运行: pip install git+https://github.com/facebookresearch/segment-anything.git")

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ellipse_ratio=args.ellipse_ratio,
        center_x_offset=args.center_x_offset,
        center_y_offset=args.center_y_offset,
        visualize=args.visualize,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        sam_device=args.sam_device,
        test_image=args.test_image,
    )