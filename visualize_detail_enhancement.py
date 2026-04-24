"""
可视化 DetailEnhancementModule 的真实增强效果（基于模型前向中的实际特征）。

示例:
python visualize_detail_enhancement.py --image data/images/2/sample.jpg
python visualize_detail_enhancement.py --image_dir data/images/2 --max_images 20
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

from model import get_model


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        out[nk] = v
    return out


def _load_model_from_cfg(cfg: Dict, cfg_path: Path, device: torch.device) -> torch.nn.Module:
    model_name = str(cfg["model"]["name"])
    num_classes = int(cfg["model"]["num_classes"])
    modules = dict(cfg.get("modules") or {})

    model = get_model(num_classes=num_classes, model_name=model_name, modules=modules)
    ckpt_path = cfg["paths"].get("best_model", "model.pth")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = (cfg_path.parent / ckpt_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到模型权重: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    else:
        state_dict = state
    model.load_state_dict(_strip_module_prefix(state_dict), strict=False)
    model.to(device).eval()
    return model


def _preprocess_image(img_path: str, input_size: int, mean, std) -> Tuple[np.ndarray, torch.Tensor]:
    img = Image.open(img_path).convert("RGB").resize((input_size, input_size))
    rgb = np.array(img).astype(np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    x = (rgb - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
    return rgb, x


def _resolve_mask_path(img_path: str, dataset_root: str, mask_dir: str) -> Optional[str]:
    sidecar = os.path.splitext(img_path)[0] + "_mask.png"
    if not (dataset_root and mask_dir):
        return sidecar if os.path.isfile(sidecar) else None

    img_abs = os.path.abspath(img_path)
    root_abs = os.path.abspath(dataset_root)
    try:
        rel = os.path.relpath(img_abs, root_abs)
    except (ValueError, OSError):
        rel = ""
    if not rel or rel.startswith(".."):
        return sidecar if os.path.isfile(sidecar) else None

    parent = os.path.dirname(rel)
    stem = os.path.splitext(os.path.basename(rel))[0]
    candidate = (
        os.path.join(mask_dir, parent, stem + "_mask.png")
        if parent
        else os.path.join(mask_dir, stem + "_mask.png")
    )
    candidate = os.path.abspath(candidate)
    if os.path.isfile(candidate):
        return candidate
    return sidecar if os.path.isfile(sidecar) else None


def _load_roi_prior(mask_path: Optional[str], input_size: int, device: torch.device) -> Optional[torch.Tensor]:
    if not mask_path or (not os.path.isfile(mask_path)):
        return None
    m = np.array(Image.open(mask_path).convert("L"))
    m = (m > 128).astype(np.float32)
    m = Image.fromarray((m * 255).astype(np.uint8)).resize((input_size, input_size), Image.NEAREST)
    m = (np.array(m).astype(np.float32) / 255.0).clip(0.0, 1.0)
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device).float()


def _robust_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)
    y = (x - lo) / max(hi - lo, eps)
    return np.clip(y, 0.0, 1.0)


def _to_map(feat: torch.Tensor, out_hw: Tuple[int, int]) -> np.ndarray:
    # feat: (1, C, H, W)
    m = feat.detach().abs().mean(dim=1, keepdim=True)  # (1,1,H,W)
    m = F.interpolate(m, size=out_hw, mode="bilinear", align_corners=False)
    m = m[0, 0].cpu().numpy().astype(np.float32)
    return _robust_norm(m)


def _collect_images(args) -> list:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if args.image:
        return [args.image]
    if not args.image_dir:
        return []
    p = Path(args.image_dir)
    if not p.is_dir():
        return []
    imgs = []
    for ext in exts:
        imgs.extend([str(x) for x in p.glob(f"*{ext}")])
    return sorted(imgs)


def _save_panel(
    rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    low_map: np.ndarray,
    detail_map: np.ndarray,
    gate_map: np.ndarray,
    out_map: np.ndarray,
    delta_map: np.ndarray,
    overlay_map: np.ndarray,
    save_path: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(title, fontsize=12)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(enhanced_rgb)
    axes[0, 1].set_title("Enhanced Image (Display Proxy)")
    axes[0, 1].axis("off")

    im1 = axes[0, 2].imshow(low_map, cmap="viridis")
    axes[0, 2].set_title("Low-pass Feature Energy")
    axes[0, 2].axis("off")
    fig.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im2 = axes[0, 3].imshow(detail_map, cmap="magma")
    axes[0, 3].set_title("Detail Component |x-low|")
    axes[0, 3].axis("off")
    fig.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im3 = axes[1, 0].imshow(gate_map, cmap="plasma")
    axes[1, 0].set_title("Detail Gate Weight")
    axes[1, 0].axis("off")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(out_map, cmap="viridis")
    axes[1, 1].set_title("Enhanced Feature Energy")
    axes[1, 1].axis("off")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(delta_map, cmap="inferno")
    axes[1, 2].set_title("Enhancement Delta |out-in|")
    axes[1, 2].axis("off")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    axes[1, 3].imshow(overlay_map)
    axes[1, 3].set_title("Final Overlay on Input")
    axes[1, 3].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _visualize_one(
    model: torch.nn.Module,
    detail_module: torch.nn.Module,
    img_path: str,
    cfg: Dict,
    device: torch.device,
    out_dir: str,
) -> None:
    input_size = int((cfg.get("data") or {}).get("input_size", 384))
    mean = list((cfg.get("image") or {}).get("norm_mean", [0.485, 0.456, 0.406]))
    std = list((cfg.get("image") or {}).get("norm_std", [0.229, 0.224, 0.225]))
    dataset_root = str((cfg.get("paths") or {}).get("dataset_root", "")).strip()
    mask_dir = str((cfg.get("data") or {}).get("mask_dir", "")).strip()

    if dataset_root and (not os.path.isabs(dataset_root)):
        dataset_root = str((Path(__file__).resolve().parent / dataset_root).resolve())
    if mask_dir and (not os.path.isabs(mask_dir)):
        mask_dir = str((Path(__file__).resolve().parent / mask_dir).resolve())

    rgb, x = _preprocess_image(img_path, input_size, mean, std)
    x = x.to(device)
    mask_path = _resolve_mask_path(img_path, dataset_root, mask_dir)
    roi_prior = _load_roi_prior(mask_path, input_size, device)

    cache: Dict[str, torch.Tensor] = {}

    def _hook(_module, inps, out):
        cache["in"] = inps[0].detach()
        cache["out"] = out.detach()

    hook = detail_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(x, roi_prior=roi_prior)
    finally:
        hook.remove()

    if "in" not in cache or "out" not in cache:
        raise RuntimeError("未捕获到细节增强模块输入输出，请检查模型 forward 是否经过该模块。")

    x_in = cache["in"]
    y_out = cache["out"]

    with torch.no_grad():
        low = detail_module.low_pass(x_in)
        detail = x_in - low
        if roi_prior is not None:
            roi_resized = F.interpolate(
                roi_prior, size=detail.shape[2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            roi_dilated = roi_resized
            roi_dilate_iter = int(getattr(detail_module, "roi_dilate_iter", 0))
            roi_dilate_kernel = int(getattr(detail_module, "roi_dilate_kernel", 1))
            if roi_dilate_iter > 0 and roi_dilate_kernel > 1:
                pad = roi_dilate_kernel // 2
                for _ in range(roi_dilate_iter):
                    roi_dilated = F.max_pool2d(
                        roi_dilated,
                        kernel_size=roi_dilate_kernel,
                        stride=1,
                        padding=pad,
                    )
                roi_dilated = roi_dilated.clamp(0.0, 1.0)
            roi_blend_alpha = float(getattr(detail_module, "roi_blend_alpha", 1.0))
            roi_focus_threshold = float(getattr(detail_module, "roi_focus_threshold", 0.0))
            roi_focus_threshold = 0.0 if roi_focus_threshold < 0.0 else (0.95 if roi_focus_threshold > 0.95 else roi_focus_threshold)
            denom = max(1e-6, 1.0 - roi_focus_threshold)
            roi_focused = ((roi_dilated - roi_focus_threshold) / denom).clamp(0.0, 1.0)
            detail_used = detail * (1.0 + roi_blend_alpha * roi_focused)
        else:
            detail_used = detail
        gate = detail_module.detail_gate(detail_used)
        gate_hard_gamma = max(1.0, float(getattr(detail_module, "gate_hard_gamma", 1.0)))
        if gate_hard_gamma > 1.0:
            gate = gate.pow(gate_hard_gamma)
        y_rebuild = x_in + gate * detail_used

    # 精确性校验：重建结果应与模块真实输出近似一致
    max_abs_err = torch.max(torch.abs(y_rebuild - y_out)).item()

    out_hw = (rgb.shape[0], rgb.shape[1])
    low_map = _to_map(low, out_hw)
    detail_map = _to_map(detail_used, out_hw)
    gate_map = _to_map(gate, out_hw)
    out_map = _to_map(y_out, out_hw)
    delta_map = _to_map(y_out - x_in, out_hw)
    delta_u8 = (delta_map * 255.0).astype(np.uint8)
    delta_color_bgr = cv2.applyColorMap(delta_u8, cv2.COLORMAP_INFERNO)
    delta_color_rgb = cv2.cvtColor(delta_color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # 将增强增量热力图叠加在原图上，显示最终增强落点
    overlay_map = np.clip(0.55 * rgb + 0.45 * delta_color_rgb, 0.0, 1.0)
    # 仅用于可视化展示：用增强增量图对原图局部做对比度提升，模拟“增强后外观”。
    gain = 1.0 + 0.35 * delta_map[..., None]
    enhanced_rgb = np.clip(rgb * gain, 0.0, 1.0)

    os.makedirs(out_dir, exist_ok=True)
    stem = Path(img_path).stem
    panel_path = os.path.join(out_dir, f"{stem}_detail_enhance_panel.png")
    title = f"{Path(img_path).name} | rebuild_max_abs_err={max_abs_err:.6e}"
    _save_panel(
        rgb,
        enhanced_rgb,
        low_map,
        detail_map,
        gate_map,
        out_map,
        delta_map,
        overlay_map,
        panel_path,
        title,
    )

    stat_path = os.path.join(out_dir, f"{stem}_detail_enhance_stats.txt")
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write(f"image={img_path}\n")
        f.write(f"mask={mask_path or 'None'}\n")
        f.write(f"rebuild_max_abs_err={max_abs_err:.8e}\n")
        f.write(f"in_mean_abs={x_in.abs().mean().item():.8f}\n")
        f.write(f"out_mean_abs={y_out.abs().mean().item():.8f}\n")
        f.write(f"delta_mean_abs={(y_out - x_in).abs().mean().item():.8f}\n")
        f.write(f"gate_mean={gate.mean().item():.8f}\n")
        f.write(f"detail_mean_abs={detail_used.abs().mean().item():.8f}\n")

    print(f"[OK] {img_path}")
    print(f"  panel: {panel_path}")
    print(f"  stats: {stat_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="可视化 DetailEnhancementModule 的真实增强效果（输入/细节/门控/输出/增量）"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--image", type=str, default=None, help="单张图像路径")
    parser.add_argument("--image_dir", type=str, default=None, help="图像目录（批量）")
    parser.add_argument("--max_images", type=int, default=0, help="批量时最多处理张数，0 表示全部")
    parser.add_argument("--output_dir", type=str, default="outputs/detail_vis", help="输出目录")
    parser.add_argument("--device", type=str, default=None, help="例如 cuda:0；为空自动选择")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image and not args.image_dir:
        raise SystemExit("请至少提供 --image 或 --image_dir 之一。")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path(__file__).resolve().parent / cfg_path).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"找不到配置文件: {cfg_path}")

    cfg = _load_yaml(cfg_path)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _load_model_from_cfg(cfg, cfg_path, device)
    detail_module = getattr(model, "detail_enhance", None)
    if detail_module is None:
        raise SystemExit("当前模型没有 detail_enhance 模块，无法可视化。")
    if not hasattr(detail_module, "low_pass") or not hasattr(detail_module, "detail_gate"):
        raise SystemExit("当前 detail_enhance 不是有效细节增强模块，可能在配置中被关闭。")

    images = _collect_images(args)
    if len(images) == 0:
        raise SystemExit("未找到可处理图像，请检查 --image 或 --image_dir。")

    limit = int(args.max_images) if args.max_images else 0
    if limit > 0:
        images = images[:limit]

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = str((Path(__file__).resolve().parent / out_dir).resolve())

    for img_path in images:
        try:
            _visualize_one(model, detail_module, img_path, cfg, device, out_dir)
        except Exception as e:
            print(f"[FAILED] {img_path} -> {e}")

    print(f"\n完成，共处理 {len(images)} 张。输出目录: {out_dir}")


if __name__ == "__main__":
    main()

