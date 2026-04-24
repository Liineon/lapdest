import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from utils import (
    ZebrafishDataset,
    allocate_yyyymmddseq_run_dir,
    format_coral_grade_label,
    save_validation_predictions_excel,
    try_load_val_indices_from_manifest,
)
from model import get_model
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    GradCAM = None
    show_cam_on_image = None
    ClassifierOutputTarget = None

try:
    import cv2
except ImportError:
    cv2 = None


def _find_gradcam_target_layer(model):
    """与 gradcam.find_target_layer 一致：选取含参数的深层特征模块。"""
    if hasattr(model, "cam_target_layer"):
        return model.cam_target_layer
    candidates = []
    if hasattr(model, "features"):
        candidates.append(model.features)
    try:
        last = model.features[-1]
        candidates.append(last)
        if hasattr(last, "__len__") and len(last) > 0:
            candidates.append(last[-1])
    except Exception:
        pass
    if hasattr(model, "backbone"):
        bb = model.backbone
        if isinstance(bb, nn.Sequential) and len(bb) > 0:
            candidates.append(bb[-1])
        else:
            candidates.append(bb)
    candidates.append(model)
    for c in candidates:
        params = list(c.parameters())
        if len(params) > 0:
            return c
    return model

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

NUM_CLASSES = int(cfg["model"]["num_classes"])
RANDOM_SEED = int(cfg["random_seed"])
torch.manual_seed(RANDOM_SEED)

# ===========================
# 设备：仅 CUDA
# ===========================
_cuda_idx = int(cfg["training"].get("cuda_device", 0))
if not torch.cuda.is_available():
    raise RuntimeError(
        "未检测到可用 CUDA。请升级驱动或安装匹配的 PyTorch：https://pytorch.org/"
    )
if _cuda_idx < 0 or _cuda_idx >= torch.cuda.device_count():
    raise RuntimeError(
        f"无效的 cuda_device={_cuda_idx}，当前可见 GPU 数量为 {torch.cuda.device_count()}"
    )
device = torch.device(f"cuda:{_cuda_idx}")
print(f"评估设备：{device} | {_CONFIG_PATH}")

# 评估结果归档：outputs/evaluate/YYYYMMDD###（与 train 的序号规则一致）
_eval_root = cfg["paths"].get("eval_outputs_root")
if _eval_root is None:
    _eval_root = "outputs/evaluate"
EVAL_OUTPUT_DIR = None
if _eval_root not in (False, "") and not (
    isinstance(_eval_root, str) and not str(_eval_root).strip()
):
    EVAL_OUTPUT_DIR = allocate_yyyymmddseq_run_dir(str(_eval_root).strip())
    print(f"📁 本次评估结果归档目录: {os.path.abspath(EVAL_OUTPUT_DIR)}")


# ===========================
# Transform（与训练一致）
# ===========================
MASK_DIR = cfg["data"]["mask_dir"] or ""
NORM_MEAN = list(cfg["image"]["norm_mean"])
NORM_STD = list(cfg["image"]["norm_std"])
INPUT_SIZE = int(cfg["data"]["input_size"])
if MASK_DIR:
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
else:
    transform = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# ===========================
# 加载数据集（只评估验证集；优先读取 val.txt 清单，否则与 train.py 相同比例随机划分）
# ===========================
DATASET_ROOT = cfg["paths"]["dataset_root"]
_smd_ev = cfg["paths"].get("split_manifest_dir")
if _smd_ev in (None, False) or (isinstance(_smd_ev, str) and not str(_smd_ev).strip()):
    SPLIT_MANIFEST_DIR = "data"
else:
    SPLIT_MANIFEST_DIR = str(_smd_ev).strip()
full_dataset = ZebrafishDataset(
    root_dir=DATASET_ROOT,
    transform=transform,
    mask_dir=MASK_DIR,
    mask_augment=False,
)

_eval_cfg = cfg.get("evaluate") or {}
_val_override = _eval_cfg.get("val_manifest")
if _val_override:
    val_manifest = (
        _val_override
        if os.path.isabs(_val_override)
        else os.path.abspath(_val_override)
    )
else:
    val_manifest = os.path.join(SPLIT_MANIFEST_DIR, "val.txt")

val_indices = try_load_val_indices_from_manifest(
    DATASET_ROOT, val_manifest, full_dataset, verbose=True
)
if val_indices is not None:
    dataset = Subset(full_dataset, val_indices)
else:
    print(f"ℹ️ 未使用清单或清单无效：{val_manifest} → 将随机划分验证集")

if val_indices is None:
    _n = len(full_dataset)
    _tr = int(float(cfg["data"]["split_train"]) * _n)
    _va = int(float(cfg["data"]["split_val"]) * _n)
    _te = _n - _tr - _va
    _, val_dataset, _ = random_split(
        full_dataset,
        [_tr, _va, _te],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    dataset = val_dataset
    print(f"✅ 随机划分（与 train.py 一致 70%/15%/15%），取验证子集")

_eval_bs = int(cfg["evaluate"]["batch_size"])
_eval_nw = int(cfg["data"]["num_workers"])
loader = DataLoader(dataset, batch_size=_eval_bs, shuffle=False, num_workers=_eval_nw)
print(f"✅ 只评估验证集（不包含训练数据）")
print(f"验证集样本数: {len(dataset)} / 总样本数: {len(full_dataset)}")


# ===========================
# 加载模型
# ===========================
_model_name = str(cfg["model"]["name"])
_module_cfg = dict(cfg.get("modules") or {})
model = get_model(
    num_classes=NUM_CLASSES, model_name=_model_name, modules=_module_cfg
).to(device)
_model_ckpt = cfg["paths"]["best_model"]
model.load_state_dict(torch.load(_model_ckpt, map_location=device))
print(f"已加载权重: {_model_ckpt}（{_model_name}）")
model.eval()


# ===========================
# 推理（同时收集逐样本预测，供 Excel 导出）
# ===========================
preds, gts = [], []
_pred_rows = []
_pos = 0
_subset_eval = dataset
_full_eval = full_dataset
with torch.no_grad():
    for batch in loader:
        if len(batch) == 3:
            x, y, mask = batch
            x, mask = x.to(device), mask.to(device)
            coral_logits, _ = model(x, mask_overlay=mask)
        else:
            x, y = batch[0], batch[1]
            x = x.to(device)
            coral_logits, _ = model(x)
        probs = torch.sigmoid(coral_logits)
        reg_score = probs.sum(dim=1)
        disc = (probs > 0.5).sum(dim=1)
        p = disc.cpu().numpy()
        preds.extend(p)
        gts.extend(y.numpy())
        bs = int(y.shape[0])
        for j in range(bs):
            gidx = _subset_eval.indices[_pos + j]
            sp = _full_eval.samples[gidx]
            fname = os.path.basename(sp["img_path"])
            gt = int(y[j].item())
            pr = int(disc[j].item())
            rs = float(reg_score[j].item())
            _pred_rows.append(
                {
                    "数据文件名称": fname,
                    "原等级": format_coral_grade_label(gt),
                    "预测等级": format_coral_grade_label(pr),
                    "预测回归值等级": round(rs, 6),
                }
            )
        _pos += bs


# ===========================
# 输出分类报告
# ===========================
# 类别名称：三分类（0和1合并，2保持，3和4合并）
target_names = ["0-1", "2", "3-4"]
_report_str = classification_report(
    gts, preds, labels=[0, 1, 2], target_names=target_names, digits=4
)
print("\n==============================")
print(" Classification Report")
print("==============================")
print(_report_str)


# ===========================
# 混淆矩阵与 Specificity
# ===========================
# 混淆矩阵：行=真实标签，列=预测标签；cm[i,j]=真实i预测为j的样本数
cm = confusion_matrix(gts, preds, labels=[0, 1, 2])
print("Confusion Matrix (行=真实, 列=预测):")
print(cm)


class _CoralProbWrapperForCam(nn.Module):
    """CORAL 输出转为 3 类概率，供 Grad-CAM；可选 mask_overlay 与主模型一致。"""

    def __init__(self, model, mask_overlay=None):
        super().__init__()
        self.model = model
        self.mask_overlay = mask_overlay

    def forward(self, x):
        coral_logits, _ = self.model(x, mask_overlay=self.mask_overlay)
        s = torch.sigmoid(coral_logits)
        s0 = s[:, 0:1]
        s1 = s[:, 1:2]
        p0 = 1 - s0
        p1 = s0 - s1
        p2 = s1
        return torch.cat([p0, p1, p2], dim=1)


def _denorm_tensor_to_rgb01(x_bchw, mean, std):
    """(1,3,H,W) ImageNet 归一化张量 -> (H,W,3) float32 [0,1]。"""
    t = x_bchw[0].detach().cpu().float().numpy()
    m = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    t = (t * s + m).transpose(1, 2, 0)
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def _safe_filename_stem(name: str) -> str:
    stem = Path(name).stem
    for c in '<>:"/\\|?*':
        stem = stem.replace(c, "_")
    return stem[:120] if stem else "sample"


def export_eval_gradcam_heatmaps(
    model,
    dataset,
    device,
    output_dir,
    mean,
    std,
    max_images,
):
    """
    对验证集逐张生成 Grad-CAM 热力图，写入 output_dir/heatmaps/。
    返回成功保存的张数。
    """
    if GradCAM is None or show_cam_on_image is None or ClassifierOutputTarget is None:
        print("⚠️ 未安装 grad-cam，跳过热力图。请执行: pip install grad-cam")
        return 0
    if cv2 is None:
        print("⚠️ 未安装 opencv-python，跳过热力图保存。")
        return 0

    heat_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heat_dir, exist_ok=True)

    target_layer = _find_gradcam_target_layer(model)
    hm_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    n_saved = 0
    limit = int(max_images) if max_images not in (None, 0) else 0

    for i, batch in enumerate(hm_loader):
        if limit > 0 and i >= limit:
            break
        if len(batch) == 3:
            x, y, mask = batch
            x = x.to(device)
            mask = mask.to(device)
            mask_for_wrap = mask
        else:
            x, y = batch[0], batch[1]
            x = x.to(device)
            mask_for_wrap = None

        with torch.no_grad():
            wrap_pred = _CoralProbWrapperForCam(model, mask_for_wrap)
            pr = int(torch.argmax(wrap_pred(x), dim=1).item())

        wrapper = _CoralProbWrapperForCam(model, mask_for_wrap).to(device)
        wrapper.eval()
        # grad-cam 新版不再支持 use_cuda，模型已在 device 上即可
        cam = GradCAM(model=wrapper, target_layers=[target_layer])
        x_cam = x.detach().clone().requires_grad_(True)
        grayscale_cam = cam(
            input_tensor=x_cam,
            targets=[ClassifierOutputTarget(pr)],
        )[0]

        rgb = _denorm_tensor_to_rgb01(x.detach(), mean, std)
        if grayscale_cam.shape[:2] != rgb.shape[:2]:
            gc = cv2.resize(
                (grayscale_cam * 255).astype(np.uint8),
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32) / 255.0
        else:
            gc = grayscale_cam

        vis = show_cam_on_image(rgb, gc, use_rgb=True)
        gidx = dataset.indices[i]
        raw_name = os.path.basename(dataset.dataset.samples[gidx]["img_path"])
        stem = _safe_filename_stem(raw_name)
        gt = int(y.item())
        out_name = f"{i:04d}_{stem}_gt{gt}_pred{pr}_gradcam.png"
        out_path = os.path.join(heat_dir, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        n_saved += 1

    print(f"✅ Grad-CAM 热力图已保存: {heat_dir}（共 {n_saved} 张）")
    return n_saved


def save_confusion_matrix_image(cm, target_names, output_path="confusion_matrix.png"):
    """将混淆矩阵保存为图片。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    # 在格子中显示数值
    thresh = cm.max() / 2.0
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im, ax=ax, label="Number of Samples")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 混淆矩阵已保存至: {output_path}")


_cm_basename = os.path.basename(cfg["evaluate"]["confusion_matrix_png"])
_cm_out = (
    os.path.join(EVAL_OUTPUT_DIR, _cm_basename)
    if EVAL_OUTPUT_DIR
    else cfg["evaluate"]["confusion_matrix_png"]
)
save_confusion_matrix_image(cm, target_names, output_path=_cm_out)
if EVAL_OUTPUT_DIR and os.path.abspath(_cm_out) != os.path.abspath(
    cfg["evaluate"]["confusion_matrix_png"]
):
    try:
        shutil.copy2(_cm_out, cfg["evaluate"]["confusion_matrix_png"])
    except OSError as e:
        print(f"⚠️ 未能同步混淆矩阵到 {cfg['evaluate']['confusion_matrix_png']}: {e}")

# Specificity（macro）
specificity_list = []
for i in range(cm.shape[0]):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity = tn / (tn + fp + 1e-6)
    specificity_list.append(specificity)

macro_specificity = np.mean(specificity_list)
print(f"\nMacro Specificity: {macro_specificity:.4f}")
print("==============================\n")

_n_gradcam = 0
if EVAL_OUTPUT_DIR and bool(_eval_cfg.get("save_gradcam_heatmaps", True)):
    try:
        _gm_max = int(_eval_cfg.get("gradcam_max_images") or 0)
        _n_gradcam = export_eval_gradcam_heatmaps(
            model,
            dataset,
            device,
            EVAL_OUTPUT_DIR,
            NORM_MEAN,
            NORM_STD,
            _gm_max,
        )
    except Exception as e:
        print(f"⚠️ Grad-CAM 热力图生成失败: {e}")

_xlsx_basename = os.path.basename(
    (cfg.get("paths") or {}).get("validation_predictions_xlsx")
    or "validation_sample_predictions.xlsx"
)
_xlsx_out = (
    os.path.join(EVAL_OUTPUT_DIR, _xlsx_basename)
    if EVAL_OUTPUT_DIR
    else (
        (cfg.get("paths") or {}).get("validation_predictions_xlsx")
        or "validation_sample_predictions.xlsx"
    )
)
try:
    save_validation_predictions_excel(_pred_rows, _xlsx_out)
    print(f"✅ 逐样本预测 Excel: {os.path.abspath(_xlsx_out)}")
    _lp_x = (cfg.get("paths") or {}).get("validation_predictions_xlsx")
    if EVAL_OUTPUT_DIR and _lp_x and os.path.abspath(_xlsx_out) != os.path.abspath(_lp_x):
        shutil.copy2(_xlsx_out, _lp_x)
except Exception as e:
    print(f"⚠️ 导出预测 Excel 失败: {e}")

if EVAL_OUTPUT_DIR:
    _rep_path = os.path.join(EVAL_OUTPUT_DIR, "classification_report.txt")
    with open(_rep_path, "w", encoding="utf-8") as f:
        f.write(_report_str)
        f.write(f"\n\nConfusion Matrix (行=真实, 列=预测):\n{cm}\n")
        f.write(f"\nMacro Specificity: {macro_specificity:.6f}\n")
    print(f"✅ 分类报告与矩阵文本已写入: {_rep_path}")

    _meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_output_dir_abs": os.path.abspath(EVAL_OUTPUT_DIR),
        "model_checkpoint": cfg["paths"]["best_model"],
        "val_manifest": val_manifest,
        "val_split_from_manifest": val_indices is not None,
        "n_val_samples": len(dataset),
        "n_full_dataset": len(full_dataset),
        "macro_specificity": float(macro_specificity),
        "confusion_matrix_png": _cm_basename,
        "predictions_xlsx": _xlsx_basename,
        "gradcam_heatmaps_subdir": "heatmaps/" if _n_gradcam else None,
        "gradcam_images_saved": int(_n_gradcam),
        "config_path": str(_CONFIG_PATH.resolve()),
    }
    with open(os.path.join(EVAL_OUTPUT_DIR, "eval_meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(_meta, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    try:
        shutil.copy2(_CONFIG_PATH, os.path.join(EVAL_OUTPUT_DIR, "config_snapshot.yaml"))
    except OSError as e:
        print(f"⚠️ 未能复制配置快照: {e}")
    print(f"✅ 本次评估结果已归档: {os.path.abspath(EVAL_OUTPUT_DIR)}")
