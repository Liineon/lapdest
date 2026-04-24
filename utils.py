import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import os
import glob
import numpy as np
import albumentations as A


def get_train_image_augment(aug_cfg=None):
    """
    训练阶段图像增强 pipeline：增强深色条纹对比度、颜色鲁棒性与噪声鲁棒性。
    - CLAHE：Albumentations 对彩色图自动仅作用于 L 通道
    - RandomGamma：gamma<1 提亮暗部
    - Sharpen：轻微锐化边缘
    - ColorJitter：亮度/对比度/饱和度/色调扰动
    - GaussianNoise：模拟采集噪声
    mask 会随 Compose 同步传递（像素级增强不改变 mask 空间结构）。
    """
    aug_cfg = dict(aug_cfg or {})
    color_jitter_p = float(aug_cfg.get("color_jitter_p", 0.5))
    brightness = float(aug_cfg.get("brightness", 0.2))
    contrast = float(aug_cfg.get("contrast", 0.2))
    saturation = float(aug_cfg.get("saturation", 0.2))
    hue = float(aug_cfg.get("hue", 0.05))
    gauss_noise_p = float(aug_cfg.get("gauss_noise_p", 0.4))
    gauss_noise_var_min = float(aug_cfg.get("gauss_noise_var_min", 10.0))
    gauss_noise_var_max = float(aug_cfg.get("gauss_noise_var_max", 50.0))

    return A.Compose([
        A.CLAHE(clip_limit=(1.5, 2.5), tile_grid_size=(8, 8), p=0.6),
        A.RandomGamma(gamma_limit=(70, 95), p=0.5),  # gamma<1 提亮暗部
        A.Sharpen(alpha=(0.15, 0.35), lightness=(0.5, 0.8), p=0.5),
        # 颜色抖动仅改变像素分布，不改变掩码几何关系
        A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=color_jitter_p,
        ),
        # 高斯噪声仅施加在图像上，mask 保持不变
        A.GaussNoise(var_limit=(gauss_noise_var_min, gauss_noise_var_max), p=gauss_noise_p),
    ], additional_targets={"mask": "mask"})


def _apply_joint_spatial_transform(
    img,
    mask,
    size=(384, 384),
    is_training=False,
    horizontal_flip_p=0.5,
    vertical_flip_p=0.5,
    rotation_degrees=15.0,
):
    """
    对 img 和 mask 施加相同的空间变换（Resize、随机水平/垂直翻转、随机旋转），
    保证掩码与图像对齐。
    """
    img = TF.resize(img, size)
    if mask is not None:
        mask = TF.resize(
            Image.fromarray(mask.astype(np.uint8)),
            size,
            interpolation=Image.NEAREST,
        )
        mask = np.array(mask)

    if is_training:
        if random.random() < float(horizontal_flip_p):
            img = TF.hflip(img)
            if mask is not None:
                mask = np.fliplr(mask).copy()
        if random.random() < float(vertical_flip_p):
            img = TF.vflip(img)
            if mask is not None:
                mask = np.flipud(mask).copy()
        rot = float(rotation_degrees)
        angle = random.uniform(-rot, rot)
        img = TF.rotate(img, angle)
        if mask is not None:
            mask = np.array(TF.rotate(Image.fromarray(mask.astype(np.uint8)), angle, fill=0))

    return img, mask


# 支持的图片扩展名（自动加载多种格式）
IMAGE_EXTENSIONS = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp')


def manifest_file_is_nonempty(path):
    """清单文件存在且至少包含一行非空白内容。"""
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True
    return False


def allocate_yyyymmddseq_run_dir(base_root: str) -> str:
    """
    在 base_root 下分配本次运行目录：YYYYMMDD + 三位序号，如 20260412001。
    当日首次为 001，同日已有子目录则顺延序号。
    """
    os.makedirs(base_root, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    max_seq = 0
    for entry in os.listdir(base_root):
        p = os.path.join(base_root, entry)
        if not os.path.isdir(p) or len(entry) != 11 or not entry.isdigit():
            continue
        if entry[:8] != today:
            continue
        max_seq = max(max_seq, int(entry[8:11]))
    run_name = f"{today}{max_seq + 1:03d}"
    out = os.path.join(base_root, run_name)
    os.makedirs(out, exist_ok=True)
    return out


def _norm_abs_path(p):
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


def indices_from_split_manifest(dataset_root, manifest_path, full_dataset):
    """
    解析 train.txt / val.txt / test.txt，返回在 full_dataset 中的样本下标（去重保序）。
    每行格式：相对 dataset_root 的路径（正斜杠亦可）\\t 标签；标签以磁盘数据集为准。
    无法匹配的行列入 skipped 计数。
    """
    abs_root = _norm_abs_path(dataset_root)
    path_to_idx = {_norm_abs_path(s["img_path"]): i for i, s in enumerate(full_dataset.samples)}

    indices = []
    skipped = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rel = line.split("\t", 1)[0].strip().replace("/", os.sep)
            img_abs = _norm_abs_path(os.path.join(abs_root, rel))
            i = path_to_idx.get(img_abs)
            if i is not None:
                indices.append(i)
            else:
                skipped += 1
                print(f"⚠️ 清单 {manifest_path} 第 {line_no} 行未匹配数据集: {rel}")

    # 去重保序
    indices = list(dict.fromkeys(indices))
    return indices, skipped


def try_load_tri_split_from_manifests(manifest_dir, dataset_root, full_dataset, verbose=True):
    """
    若 manifest_dir 下 train.txt、val.txt、test.txt 均存在且含有效行，且三者解析出的
    样本下标两两不相交，则返回 (train_indices, val_indices, test_indices)；
    清单内相对路径相对 dataset_root（图像根目录）解析。
    否则返回 None，由调用方改用 random_split。
    """
    names = ("train.txt", "val.txt", "test.txt")
    paths = [os.path.join(manifest_dir, n) for n in names]
    if not all(manifest_file_is_nonempty(p) for p in paths):
        return None

    triple = []
    total_skipped = 0
    for p in paths:
        idx, sk = indices_from_split_manifest(dataset_root, p, full_dataset)
        if len(idx) == 0:
            if verbose:
                print(f"⚠️ 清单 {p} 无有效匹配条目，将改用随机划分")
            return None
        triple.append(idx)
        total_skipped += sk

    train_indices, val_indices, test_indices = triple
    s_tr, s_va, s_te = set(train_indices), set(val_indices), set(test_indices)
    if s_tr & s_va or s_tr & s_te or s_va & s_te:
        if verbose:
            print("⚠️ train/val/test 清单存在重叠样本，将改用随机划分")
        return None

    n_all = len(full_dataset)
    union = s_tr | s_va | s_te
    if any(i < 0 or i >= n_all for i in union):
        if verbose:
            print("⚠️ 清单含有非法下标，将改用随机划分")
        return None

    if verbose:
        if len(union) < n_all:
            print(
                f"ℹ️ 清单仅覆盖 {len(union)}/{n_all} 个样本；未出现在三份清单中的样本不参与本次训练划分"
            )
        if total_skipped:
            print(f"ℹ️ 解析清单时有 {total_skipped} 行未能匹配当前数据集扫描结果")

    return train_indices, val_indices, test_indices


def try_load_val_indices_from_manifest(dataset_root, manifest_path, full_dataset, verbose=True):
    """
    若 manifest_path 存在且含有效行，且能解析出至少一个样本下标，则返回下标列表；否则返回 None。
    用于评估脚本优先按 val.txt（或自定义路径）划分验证集。
    """
    if not manifest_file_is_nonempty(manifest_path):
        return None
    val_indices, n_skip = indices_from_split_manifest(
        dataset_root, manifest_path, full_dataset
    )
    if len(val_indices) == 0:
        if verbose:
            print(f"⚠️ {manifest_path} 有内容但无有效路径，将改用随机划分验证集")
        return None
    if verbose:
        print(f"✅ 使用清单划分验证集：{manifest_path}（{len(val_indices)} 条，未匹配行 {n_skip}）")
    return val_indices


_CORAL_GRADE_SUFFIX = {0: "0-1", 1: "2", 2: "3-4"}


def format_coral_grade_label(cls_id: int) -> str:
    """三分类序号 + 档名，如 0（0-1）。"""
    c = int(cls_id)
    name = _CORAL_GRADE_SUFFIX.get(c, "?")
    return f"{c}（{name}）"


def _resolve_subset_from_val_accessor(val_accessor):
    """val_accessor 为 Subset，或 ValDatasetWrapper（内含 .subset）。"""
    if isinstance(val_accessor, Subset):
        return val_accessor
    sub = getattr(val_accessor, "subset", None)
    if isinstance(sub, Subset):
        return sub
    raise TypeError(f"无法从 {type(val_accessor)} 解析 Subset")


def collect_coral_validation_rows(model, val_loader, val_accessor, device, num_classes: int):
    """
    对 val_loader（须 shuffle=False）逐批推理，返回 list[dict]，键为 Excel 列名。
    val_accessor: Subset 或 train 中的 ValDatasetWrapper。
    """
    subset = _resolve_subset_from_val_accessor(val_accessor)
    full_ds = subset.dataset
    model.eval()
    rows = []
    pos = 0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                x, y, mask = batch
                x = x.to(device)
                mask = mask.to(device)
                y_cpu = y
                coral_logits, _ = model(x, mask_overlay=mask)
            else:
                x, y = batch[0], batch[1]
                x = x.to(device)
                y_cpu = y
                coral_logits, _ = model(x)
            assert coral_logits.shape[1] == int(num_classes) - 1
            probs = torch.sigmoid(coral_logits)
            reg_score = probs.sum(dim=1)
            disc = (probs > 0.5).sum(dim=1)
            bs = int(y_cpu.shape[0])
            for j in range(bs):
                gidx = subset.indices[pos + j]
                sp = full_ds.samples[gidx]
                fname = os.path.basename(sp["img_path"])
                gt = int(y_cpu[j].item())
                pr = int(disc[j].item())
                rs = float(reg_score[j].item())
                rows.append(
                    {
                        "数据文件名称": fname,
                        "原等级": format_coral_grade_label(gt),
                        "预测等级": format_coral_grade_label(pr),
                        "预测回归值等级": round(rs, 6),
                    }
                )
            pos += bs
    return rows


def save_validation_predictions_excel(rows, output_path):
    """将逐样本预测写入 xlsx（使用 openpyxl，不依赖 pandas）。"""
    try:
        from openpyxl import Workbook
    except ImportError as e:
        raise ImportError("请安装 openpyxl：pip install openpyxl") from e

    out_abs = os.path.abspath(output_path)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "predictions"
    if not rows:
        wb.save(output_path)
        return
    headers = list(rows[0].keys())
    ws.append(headers)
    for r in rows:
        ws.append([r[h] for h in headers])
    wb.save(output_path)


class ZebrafishDataset(Dataset):
    """
    斑马鱼分类数据集，从 root_dir 下按标签子目录加载多种格式图片（.tif, .jpg 等）。
    所有图像自动转换为 RGB，输出为归一化后的 float tensor。
    mask_dir: 若指定（如 "masks_overlay"），则加载对应掩码，返回 (img, label, mask)。
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        mask_dir=None,
        mask_augment=False,
        augment=None,
        spatial_aug_cfg=None,
    ):
        """
        mask_dir: 掩码目录（如 "masks_overlay"），与图像同结构，文件名为 {stem}_mask.png
        mask_augment: 为 True 时对 mask 施加与图像相同的随机翻转/旋转（训练用）
        augment: Albumentations Compose，训练时对图像做 CLAHE/Gamma/锐化等增强，mask 同步传递
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_dir = mask_dir
        self.mask_augment = mask_augment
        self.augment = augment
        self.spatial_aug_cfg = dict(spatial_aug_cfg or {})

        # ========== 加载带标签的样本路径 ==========
        self.samples = []

        # 原始5分类标签 -> 3分类映射：0和1合并，2保持，3和4合并
        label_map = {
            "0": 0, "0分": 0,
            "1": 1, "1分": 1,
            "2": 2, "2分": 2,
            "3": 3, "3分": 3,
            "4": 4, "4分": 4
        }
        # 三分类映射：0,1->0, 2->1, 3,4->2
        LABEL_5_TO_3 = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
        NUM_CLASSES = 3

        # 支持多种图片扩展名（.tif, .tiff, .jpg, .jpeg, .png, .bmp）
        for label_dir, label_num in label_map.items():
            label_path = os.path.join(root_dir, label_dir)
            if not os.path.isdir(label_path):
                continue
            for ext in IMAGE_EXTENSIONS:
                for img_path in glob.glob(os.path.join(label_path, f"*{ext}")):
                    self.samples.append({
                        "img_path": img_path,
                        "label": LABEL_5_TO_3[label_num]
                    })

        if len(self.samples) == 0:
            raise ValueError(f"❌ 数据集为空，请检查目录: {root_dir}")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        label = sample["label"]

        img = Image.open(img_path).convert("RGB")

        mask = None
        if self.mask_dir:
            rel = os.path.relpath(img_path, self.root_dir)
            parent = os.path.dirname(rel)
            stem = os.path.splitext(os.path.basename(rel))[0]
            mask_path = os.path.join(self.mask_dir, parent, stem + "_mask.png")
            if os.path.exists(mask_path):
                m = np.array(Image.open(mask_path).convert("L"))
                mask = (m > 128).astype(np.float32)  # 255->1, 0->0

        if mask is not None or self.mask_dir:
            img, mask = _apply_joint_spatial_transform(
                img,
                mask,
                size=(384, 384),
                is_training=self.mask_augment,
                horizontal_flip_p=float(self.spatial_aug_cfg.get("horizontal_flip_p", 0.5)),
                vertical_flip_p=float(self.spatial_aug_cfg.get("vertical_flip_p", 0.5)),
                rotation_degrees=float(self.spatial_aug_cfg.get("rotation_degrees", 15.0)),
            )

        # 训练时应用 Albumentations 增强（CLAHE/Gamma/锐化），mask 同步传递
        if self.augment and self.mask_augment:
            img_np = np.array(img)
            mask_for_aug = mask if mask is not None else np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.float32)
            aug_out = self.augment(image=img_np, mask=mask_for_aug)
            img = Image.fromarray(aug_out["image"])
            if mask is not None:
                mask = aug_out["mask"]

        if self.transform:
            img = self.transform(img)

        if mask is not None:
            mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)
            return img, label, mask
        if self.mask_dir:
            return img, label, torch.zeros(1, 384, 384)
        return img, label

    # ------------------------------------------
    def __len__(self):
        return len(self.samples)


# ------------------------------------------
# Weighted Focal Loss（保留你原来的功能）
# ------------------------------------------
def weighted_focal_loss(pred, target, alpha, gamma=2.0, reduction='mean'):
    """
    Weighted Focal Loss for class imbalance handling.
    """
    log_probs = F.log_softmax(pred, dim=1)
    probs = F.softmax(pred, dim=1)

    num_classes = pred.size(1)
    target_one_hot = F.one_hot(target, num_classes).float()

    pt = (probs * target_one_hot).sum(dim=1)
    focal_weight = (1 - pt) ** gamma

    alpha_t = (alpha * target_one_hot).sum(dim=1)
    log_pt = (log_probs * target_one_hot).sum(dim=1)

    loss = -alpha_t * focal_weight * log_pt

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
