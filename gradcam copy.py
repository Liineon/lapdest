"""
gradcam.py

Usage examples:
  # 只处理单个子文件夹（原有行为）
python gradcam.py --input_dir data/0分 --model_path model.pth

 # 处理 data 下所有子文件夹（0分、1分 等），输出保持相同分类结构
python gradcam.py --input_dir data --all_folders --model_path model.pth

Options:
  --all_folders      When used with --input_dir, process all subdirectories; output mirrors input structure (e.g. outputs/gradcam/0分/, outputs/gradcam/1分/)
  --mask_overlay     If set, and if xxx_mask.png exists, draws mask contour on the saved figure (unchanged). If xxx_mask.png exists, it is always used in the model forward for Grad-CAM / probs (aligned with masked pooling at train/eval).
  --target_class     Optionally force target class index (int). If omitted, the script uses predicted class.
  --device           例如 cuda:0（默认使用 CUDA）
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import yaml

# try to import pytorch-grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except Exception as e:
    raise ImportError("Please install grad-cam (pip install grad-cam). "
                      "Original error: " + str(e))

# Import your model builder: assumes model.py defines get_model(num_classes)
from model import get_model


class ModelWrapperForGradCAM(torch.nn.Module):
    """将 (coral_logits, binary_logits) 转为 3 类概率，供 GradCAM 使用；mask 与训练/评估一致。"""

    def __init__(self, model, mask_overlay: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model
        self.mask_overlay = mask_overlay

    def forward(self, x):
        coral_logits, _ = self.model(x, mask_overlay=self.mask_overlay)
        # CORAL -> 3 类概率: P0=1-s0, P1=s0-s1, P2=s1
        s = torch.sigmoid(coral_logits)  # (B, 2)
        s0 = s[:, 0:1]
        s1 = s[:, 1:2]
        p0 = 1 - s0
        p1 = s0 - s1
        p2 = s1
        probs_3 = torch.cat([p0, p1, p2], dim=1)  # (B, 3)
        return probs_3


def load_model(model_path: str, device: torch.device, config_path: Optional[Path] = None):
    cfg_path = config_path or (Path(__file__).resolve().parent / "config.yaml")
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)
        nc = int(_cfg["model"]["num_classes"])
        mname = str(_cfg["model"]["name"])
        mods = dict(_cfg.get("modules") or {})
    else:
        nc, mname, mods = 3, "convnext_gi", {}
    model = get_model(num_classes=nc, model_name=mname, modules=mods)
    map_location = device
    state = torch.load(model_path, map_location=map_location, weights_only=True)
    # if saved state dict has 'model' or similar wrapper, try common keys
    if isinstance(state, dict) and ('state_dict' in state and isinstance(state['state_dict'], dict)):
        state_dict = state['state_dict']
    else:
        state_dict = state
    # remove possible 'module.' prefixes from DataParallel
    new_state = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_k] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


def find_target_layer(model):
    """
    Attempt to find a reasonable convolutional layer to use as target for GradCAM.
    For torchvision ConvNeXt tiny, model.features is a Sequential of stages; pick the last feature block.
    We'll try a few candidates in order.
    """
    candidates = []
    # common fields
    if hasattr(model, 'features'):
        candidates.append(model.features)
    # also try deeper blocks if exist
    try:
        # for convnext_tiny: model.features[-1] is a sequential block - use its last element
        last = model.features[-1]
        candidates.append(last)
        # if last itself is a sequential with blocks, try its last block
        if hasattr(last, "__len__") and len(last) > 0:
            candidates.append(last[-1])
    except Exception:
        pass

    # fallback: whole model
    candidates.append(model)

    # pick the first candidate that has parameters (heuristic)
    for c in candidates:
        params = list(c.parameters())
        if len(params) > 0:
            return c
    # last resort
    return model


def _load_mask_overlay_tensor(mask_path: str, img_size: int, device: torch.device) -> Optional[torch.Tensor]:
    """与数据集一致：L 图二值化后 NEAREST resize 到 img_size，形状 (1,1,H,W) float。"""
    if not os.path.isfile(mask_path):
        return None
    m = np.array(Image.open(mask_path).convert("L"))
    m_bin = (m > 128).astype(np.float32)
    pil_m = Image.fromarray((m_bin * 255).astype(np.uint8)).resize(
        (img_size, img_size), Image.NEAREST
    )
    m_rs = (np.array(pil_m).astype(np.float32) / 255.0).clip(0.0, 1.0)
    t = torch.from_numpy(m_rs).unsqueeze(0).unsqueeze(0).float().to(device)
    return t


def preprocess_image(img_pil: Image.Image, img_size=384):
    """Return numpy rgb [0,1] and tensor batch for model"""
    img = img_pil.convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0  # RGB 0-1
    # normalization: use ImageNet mean/std (ConvNeXt pretrain)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    inp = (arr - mean) / std
    # convert to tensor CHW
    tensor = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0).float()
    return arr, tensor


def make_gradcam_for_image(model, device, img_path: str, model_path: str,
                           output_dir: str, mask_overlay=False, target_class=None, output_subdir=None):
    # load image
    img_pil = Image.open(img_path).convert("RGB")
    rgb, tensor = preprocess_image(img_pil, img_size=384)
    tensor = tensor.to(device)
    img_size = int(tensor.shape[-1])

    mask_path = img_path.rsplit(".", 1)[0] + "_mask.png"
    mask_t = _load_mask_overlay_tensor(mask_path, img_size, device)

    # 包装模型：CORAL -> 3 类概率；存在 xxx_mask.png 时与训练/评估同样走 mask 池化
    wrapper = ModelWrapperForGradCAM(model, mask_overlay=mask_t).to(device)
    wrapper.eval()

    # get prediction (CORAL -> 3 类概率)
    with torch.no_grad():
        probs_tensor = wrapper(tensor)
        probs = probs_tensor.cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    if target_class is None:
        cam_target = pred_class
    else:
        cam_target = int(target_class)

    # pick target layer (用原始 model 找 features)
    target_layer = find_target_layer(model)

    # create cam (用 wrapper 保证输出为单张量)
    cam = GradCAM(model=wrapper, target_layers=[target_layer])

    tensor_cam = tensor.detach().clone().requires_grad_(True)
    grayscale_cam = cam(
        input_tensor=tensor_cam,
        targets=[ClassifierOutputTarget(cam_target)],
    )[0]
    if grayscale_cam.shape[:2] != rgb.shape[:2]:
        gc = cv2.resize(
            (grayscale_cam * 255).astype(np.uint8),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32) / 255.0
        grayscale_cam = gc

    # overlay cam on image
    visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    # optionally overlay mask boundary (fish bladder mask)
    if mask_overlay and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0)
        if mask is not None:
            # resize mask to visualization size
            h, w = visualization.shape[:2]
            mask_resized = cv2.resize(mask, (w, h))
            # draw contour in cyan
            contours, _ = cv2.findContours((mask_resized>128).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_bgr, contours, -1, (255,255,0), 2)

    # prepare output path (preserve subfolder structure when output_subdir is given)
    out_dir = os.path.join(output_dir, output_subdir) if output_subdir else output_dir
    os.makedirs(out_dir, exist_ok=True)
    fname = Path(img_path).stem + "_gradcam.png"
    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, vis_bgr)
    print("Saved gradcam:", out_path)
    # also save probabilities
    probs_path = os.path.join(out_dir, Path(img_path).stem + "_probs.txt")
    with open(probs_path, "w") as f:
        f.write("pred_class: {}\n".format(pred_class))
        f.write("gradcam_target_class: {}\n".format(cam_target))
        for i,p in enumerate(probs):
            f.write(f"class_{i}: {p:.6f}\n")
    return out_path


def _collect_images(p: Path, all_folders: bool) -> list:
    """
    Collect image paths from a directory.
    If all_folders=True, returns list of (img_path, subfolder_name) to preserve output structure.
    Otherwise returns list of (img_path, None).
    """
    if not p.is_dir():
        return []
    if all_folders:
        result = []
        for subdir in sorted(p.iterdir()):
            if subdir.is_dir():
                subfolder_name = subdir.name
                imgs = [str(x) for x in subdir.glob("*.jpg")] + [str(x) for x in subdir.glob("*.tif")]
                for img_path in sorted(imgs):
                    result.append((img_path, subfolder_name))
        return result
    imgs = sorted([str(x) for x in p.glob("*.jpg")] + [str(x) for x in p.glob("*.tif")])
    return [(path, None) for path in imgs]


def process_batch(input_path, model_path, output_dir, mask_overlay=False, target_class=None, device=None, all_folders=False):
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Grad-CAM 需要 CUDA，请检查驱动与 PyTorch 安装。")
        device = torch.device("cuda:0")
    # load model once
    model = load_model(model_path, device)
    # find list of images
    p = Path(input_path)
    if p.is_dir():
        img_list = _collect_images(p, all_folders)
        if all_folders and img_list:
            print(f"Processing {len(img_list)} images from all subfolders of {input_path}")
    else:
        img_list = [(str(p), None)]

    for img_path, subfolder in img_list:
        try:
            make_gradcam_for_image(model, device, img_path, model_path, output_dir, mask_overlay, target_class,
                                  output_subdir=subfolder)
        except Exception as e:
            print("Failed for", img_path, e)


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single jpg image")
    group.add_argument("--input_dir", type=str, help="Path to a directory of jpg images")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Trained model .pth")
    parser.add_argument("--output_dir", type=str, default="outputs/gradcam", help="Output directory")
    parser.add_argument("--all_folders", action="store_true", help="Process all subdirectories of --input_dir (e.g. data/0分, data/1分, ...)")
    parser.add_argument("--mask_overlay", action="store_true", help="在图上叠加掩码轮廓（需 xxx_mask.png）；前向掩码池化在掩码文件存在时自动启用，不依赖本开关")
    parser.add_argument("--target_class", type=int, default=None, help="Force target class index (0..4)")
    parser.add_argument("--device", type=str, default=None, help="CUDA 设备，如 cuda:0（省略则 cuda:0）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
        if device.type != "cuda":
            raise SystemExit("仅支持 CUDA 设备（例如 cuda:0）。")
    else:
        if not torch.cuda.is_available():
            raise SystemExit("未检测到 CUDA，无法运行 Grad-CAM。")
        device = torch.device("cuda:0")
    inp = args.image if args.image else args.input_dir
    process_batch(inp, args.model_path, args.output_dir, mask_overlay=args.mask_overlay,
                  target_class=args.target_class, device=device, all_folders=args.all_folders)
