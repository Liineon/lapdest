import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, Subset, random_split, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import apply_freeze_backbone, get_model
from utils import (
    ZebrafishDataset,
    allocate_yyyymmddseq_run_dir,
    get_train_image_augment,
    manifest_file_is_nonempty,
    try_load_tri_split_from_manifests,
)
import torchvision.transforms as T
import numpy as np # 新增
import matplotlib.pyplot as plt # 新增
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 新增指标库
from tqdm import tqdm # 导入 tqdm

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 设置随机种子确保可重复性
RANDOM_SEED = int(cfg["random_seed"])
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"✅ 随机种子已设置: {RANDOM_SEED} | 配置文件: {_CONFIG_PATH}")

# CORAL 有序回归：标签为 0-1, 2, 3-4（三分类）
NUM_CLASSES = int(cfg["model"]["num_classes"])


def labels_to_coral(y: torch.Tensor, num_classes: int, device) -> torch.Tensor:
    """将整数标签转为 cumulative ordinal 编码：target[i,j] = 1 if y[i] > j else 0"""
    k = num_classes - 1
    y_expanded = y.unsqueeze(1).to(device)  # (B, 1)
    indices = torch.arange(k, device=device).unsqueeze(0)  # (1, k)
    return (y_expanded > indices).float()  # (B, k)


def _to_builtin(obj):
    """递归转换 numpy 标量/数组为 Python 原生类型，便于 YAML 序列化。"""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_builtin(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# 性能指标评估函数
def evaluate(model, val_loader, device, criterion, num_classes):
    """
    在验证集上评估模型性能，并返回验证损失与常用分类指标。
    """
    model.eval() # 切换到评估模式
    all_preds = []
    all_targets = []
    running_val_loss = 0.0
    
    with torch.no_grad(): # 评估时不计算梯度
        for batch in val_loader:
            if len(batch) == 3:
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                coral_logits, _ = model(x, mask_overlay=mask)
            else:
                x, y = batch[0], batch[1]
                x, y = x.to(device), y.to(device)
                coral_logits, _ = model(x)
            target_coral = labels_to_coral(y, num_classes, device)
            loss = criterion(coral_logits, target_coral)
            running_val_loss += loss.item() * x.size(0)
            # CORAL 预测：pred = sum(sigmoid(logits) > 0.5)
            preds = (torch.sigmoid(coral_logits) > 0.5).sum(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # 转化为 NumPy 数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 计算基本指标
    acc = accuracy_score(all_targets, all_preds)
    
    # 计算 Precision, Recall, F1-Score
    # average='macro' 对每个类别的指标取平均，适用于类别不平衡的情况
    # labels=np.unique(all_targets) 确保只计算实际存在的类别
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro',
            labels=np.unique(all_targets), zero_division=0
        )
    except TypeError:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro',
            labels=np.unique(all_targets)
        )

    # 计算 Specificity (特异性)
    # 多分类 one-vs-rest: Specificity_i = TN_i / (TN_i + FP_i)，宏平均
    try:
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
        # TN_i = 总样本 - 真实为i - 预测为i + TP_i
        tn = np.sum(cm) - np.sum(cm, axis=1) - np.sum(cm, axis=0) + np.diag(cm)
        # TN_i + FP_i = 所有非类别i的样本数 = 总样本 - 真实为i的样本数
        tn_plus_fp = np.sum(cm) - np.sum(cm, axis=1)
        specificity = np.mean(tn / (tn_plus_fp + 1e-8))
    except Exception:
        specificity = 0.0 # 异常处理

    val_loss = running_val_loss / max(1, len(val_loader.dataset))
    return val_loss, acc, precision, recall, f1, specificity


def save_split_manifests(
    dataset_root,
    manifest_dir,
    full_dataset,
    train_indices,
    val_indices,
    test_indices,
    verbose=True,
):
    """
    将划分结果写入 manifest_dir/train.txt、manifest_dir/val.txt、manifest_dir/test.txt。
    每行：相对 dataset_root 的图像路径（POSIX 斜杠）\\t 三分类标签。
    verbose=False 时仅静默写盘并打印一行汇总（用于训练结束再次刷新）。
    """
    abs_root = os.path.abspath(dataset_root)

    def lines_for(indices):
        rows = []
        for i in sorted(indices):
            s = full_dataset.samples[i]
            rel = os.path.relpath(s["img_path"], abs_root)
            rel = rel.replace(os.sep, "/")
            rows.append(f"{rel}\t{s['label']}\n")
        return rows

    os.makedirs(manifest_dir, exist_ok=True)
    paths = {k: os.path.join(manifest_dir, f"{k}.txt") for k in ("train", "val", "test")}
    content = {
        "train": lines_for(train_indices),
        "val": lines_for(val_indices),
        "test": lines_for(test_indices),
    }
    for key, p in paths.items():
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(content[key])
        if verbose:
            print(f"✅ 已写入 {p}（{len(content[key])} 条）")
    if not verbose:
        n_tr, n_va, n_te = len(content["train"]), len(content["val"]), len(content["test"])
        print(f"✅ 已刷新划分清单（训练结束）：train={n_tr} val={n_va} test={n_te}")


# 设备：仅使用 CUDA（需驱动与当前 PyTorch 的 CUDA 构建版本匹配）
_cuda_idx = int(cfg["training"].get("cuda_device", 0))
if not torch.cuda.is_available():
    raise RuntimeError(
        "未检测到可用 CUDA。请升级 NVIDIA 驱动，或安装与驱动匹配的 PyTorch 版本："
        "https://pytorch.org/"
    )
if _cuda_idx < 0 or _cuda_idx >= torch.cuda.device_count():
    raise RuntimeError(
        f"无效的 cuda_device={_cuda_idx}，当前可见 GPU 数量为 {torch.cuda.device_count()}"
    )
device = torch.device(f"cuda:{_cuda_idx}")
try:
    torch.zeros(1, device=device)
except RuntimeError as e:
    raise RuntimeError(f"无法在 {device} 上分配张量，请检查显存占用与驱动：{e}") from e
print(f"训练设备: {device} | {torch.cuda.get_device_name(_cuda_idx)}")
torch.cuda.empty_cache()


# 数据增强和数据集加载
# 使用 masks_overlay 掩码屏蔽黑色背景
MASK_DIR = cfg["data"]["mask_dir"] or ""
NORM_MEAN = list(cfg["image"]["norm_mean"])
NORM_STD = list(cfg["image"]["norm_std"])
INPUT_SIZE = int(cfg["data"]["input_size"])
DATASET_ROOT = cfg["paths"]["dataset_root"]
_split_manifest_dir_cfg = (cfg.get("paths") or {}).get("split_manifest_dir")
SPLIT_MANIFEST_DIR = (
    "data"
    if _split_manifest_dir_cfg in (None, False) or (isinstance(_split_manifest_dir_cfg, str) and not str(_split_manifest_dir_cfg).strip())
    else str(_split_manifest_dir_cfg).strip()
)

# 训练结果归档：outputs/train/YYYYMMDD###（与 evaluate.py 一致）
_train_root = (cfg.get("paths") or {}).get("train_outputs_root")
TRAIN_OUTPUT_DIR = None
if _train_root is None:
    _train_root = "outputs/train"
if _train_root not in (False, "") and not (
    isinstance(_train_root, str) and not str(_train_root).strip()
):
    TRAIN_OUTPUT_DIR = allocate_yyyymmddseq_run_dir(str(_train_root).strip())
    print(f"📁 本次训练结果归档目录: {os.path.abspath(TRAIN_OUTPUT_DIR)}")
    try:
        shutil.copy2(_CONFIG_PATH, os.path.join(TRAIN_OUTPUT_DIR, "config_snapshot.yaml"))
    except OSError as e:
        print(f"⚠️ 未能复制配置快照到训练归档目录: {e}")

if MASK_DIR:
    # 掩码模式下，空间变换在 dataset 内与 mask 同步完成
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
else:
    train_transform = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    val_transform = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

# 加载数据集并分割
# augment: CLAHE(L通道)+Gamma+锐化，增强深色条纹对比度与边缘，mask 同步
full_dataset = ZebrafishDataset(
    root_dir=DATASET_ROOT,
    transform=train_transform,
    mask_dir=MASK_DIR,
    mask_augment=bool(cfg["data"]["mask_augment_train"]),
    augment=get_train_image_augment(),
)

# 划分优先级：
# 1) 若 split_manifest_dir/train.txt、val.txt、test.txt 三者都非空且有效，则按清单划分
# 2) 否则回退到随机划分
_split_tr = float(cfg["data"]["split_train"])
_split_va = float(cfg["data"]["split_val"])
_n = len(full_dataset)
_train_n = int(_split_tr * _n)
_val_n = int(_split_va * _n)
_test_n = _n - _train_n - _val_n

_manifest_paths = {
    name: os.path.join(SPLIT_MANIFEST_DIR, f"{name}.txt")
    for name in ("train", "val", "test")
}
_manifest_status = []
_empty_or_missing = []
for _name, _path in _manifest_paths.items():
    if not os.path.isfile(_path):
        _manifest_status.append(f"{_name}.txt=缺失")
        _empty_or_missing.append(f"{_name}.txt 缺失")
    elif not manifest_file_is_nonempty(_path):
        _manifest_status.append(f"{_name}.txt=为空")
        _empty_or_missing.append(f"{_name}.txt 为空")
    else:
        _manifest_status.append(f"{_name}.txt=非空")
print(f"📄 划分清单检查：{SPLIT_MANIFEST_DIR} -> {', '.join(_manifest_status)}")

_tri_split = try_load_tri_split_from_manifests(
    SPLIT_MANIFEST_DIR, DATASET_ROOT, full_dataset, verbose=True
)
if _tri_split is not None:
    train_indices, val_indices, test_indices = _tri_split
    train_dataset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)
    print(
        f"✅ 本次按清单划分：{SPLIT_MANIFEST_DIR}"
        f"（train={len(train_dataset)} val={len(val_subset)} test={len(test_subset)}）"
    )
else:
    train_dataset, val_subset, test_subset = random_split(
        full_dataset,
        [_train_n, _val_n, _test_n],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    if _empty_or_missing:
        _fallback_reason = "，".join(_empty_or_missing)
        print(f"⚠️ 因 {_fallback_reason}，已回退到随机划分")
    else:
        print("⚠️ 清单虽非空，但存在无效路径、重叠样本或其他解析问题，已回退到随机划分")
    print(
        f"ℹ️ 本次按随机划分数据集："
        f"train={len(train_dataset)} val={len(val_subset)} test={len(test_subset)}"
    )

# 为验证集创建包装类（支持 mask_overlay）
class ValDatasetWrapper(torch.utils.data.Dataset):
    """验证集包装类"""
    def __init__(self, subset, transform, mask_dir=None, input_size=384):
        self.subset = subset
        self.transform = transform
        self.original_dataset = subset.dataset
        self.mask_dir = mask_dir
        self.input_size = int(input_size)
        self.root_dir = getattr(subset.dataset, "root_dir", "data")
        
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        import os
        from utils import _apply_joint_spatial_transform
        original_idx = self.subset.indices[idx]
        sample = self.original_dataset.samples[original_idx]
        img_path = sample["img_path"]
        label = sample["label"]
        
        from PIL import Image
        import numpy as np
        img = Image.open(img_path).convert("RGB")
        
        mask = None
        if self.mask_dir:
            rel = os.path.relpath(img_path, self.root_dir)
            parent = os.path.dirname(rel)
            stem = os.path.splitext(os.path.basename(rel))[0]
            mask_path = os.path.join(self.mask_dir, parent, stem + "_mask.png")
            if os.path.exists(mask_path):
                m = np.array(Image.open(mask_path).convert("L"))
                mask = (m > 128).astype(np.float32)
        
        if mask is not None or self.mask_dir:
            s = self.input_size
            img, mask = _apply_joint_spatial_transform(
                img, mask, size=(s, s), is_training=False
            )
        
        if self.transform:
            img = self.transform(img)
        
        if mask is not None:
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            return img, label, mask
        if self.mask_dir:
            s = self.input_size
            return img, label, torch.zeros(1, s, s)
        return img, label

# 创建验证集
val_dataset = ValDatasetWrapper(
    val_subset, val_transform, mask_dir=MASK_DIR, input_size=INPUT_SIZE
)

# 训练开始前即写入清单，长时间训练中途异常时仍可复现当前子集
save_split_manifests(
    DATASET_ROOT,
    SPLIT_MANIFEST_DIR,
    full_dataset,
    list(train_dataset.indices),
    list(val_subset.indices),
    list(test_subset.indices),
    verbose=True,
)
if TRAIN_OUTPUT_DIR:
    save_split_manifests(
        DATASET_ROOT,
        TRAIN_OUTPUT_DIR,
        full_dataset,
        list(train_dataset.indices),
        list(val_subset.indices),
        list(test_subset.indices),
        verbose=False,
    )

print(
    f"数据集大小: 总计 {len(full_dataset)} | 训练集 {len(train_dataset)} | "
    f"验证集 {len(val_dataset)} | 测试集 {len(test_subset)}"
)

# WeightedRandomSampler：按类别均衡采样，使各类别在训练中出现的频率相近
train_labels = [full_dataset.samples[i]["label"] for i in train_dataset.indices]
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-6)  # 避免除零
sample_weights = np.array([class_weights[l] for l in train_labels], dtype=np.float64)
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)
print(f"类别分布: {dict(zip(range(NUM_CLASSES), class_counts))} | 已启用 WeightedRandomSampler")

# 根据设备与配置决定 batch_size
_bs = cfg["training"]["batch_size"]
if _bs.get("fixed") is not None:
    batch_size = int(_bs["fixed"])
elif "cuda:1" in str(device):
    batch_size = int(_bs["auto_on_cuda_alt"])
else:
    batch_size = int(_bs["auto_on_cuda_default"])

_num_workers = int(cfg["data"]["num_workers"])
# 数据加载器（使用 sampler 时不可再用 shuffle=True）
loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=_num_workers,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=_num_workers
)
print(f"Batch size: {batch_size}")



FREEZE_BACKBONE = bool(cfg["model"]["freeze_backbone"])
_model_name = str(cfg["model"]["name"])
_module_cfg = dict(cfg.get("modules") or {})

model = get_model(
    num_classes=NUM_CLASSES, model_name=_model_name, modules=_module_cfg
).to(device)
if _model_name.lower().replace("-", "_") in (
    "convnext_gi",
    "default",
    "convnext_gi_attention",
):
    print(f"✅ 子模块开关（消融）: {_module_cfg}")
else:
    print("ℹ️ 当前为 baseline 模型，config.modules 仅对 convnext_gi 生效，已忽略")

if FREEZE_BACKBONE:
    apply_freeze_backbone(model)
    print("✅ 已冻结 backbone，仅训练分类相关头部")
else:
    print("✅ 全模型微调")
print(f"模型已加载到 {device}（架构: {_model_name}）")
_lr = float(cfg["training"]["optimizer"]["lr"])
_wd = float(cfg["training"]["optimizer"]["weight_decay"])
opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=_lr, weight_decay=_wd)

# 学习率调度器：当验证 F1 长时间无提升时自动降低学习率
_sch = cfg["scheduler"]
LR_PATIENCE = int(_sch["patience"])
LR_FACTOR = float(_sch["factor"])
MIN_LR = float(_sch["min_lr"])
scheduler = ReduceLROnPlateau(
    opt,
    mode=str(_sch["mode"]),
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=MIN_LR,
    verbose=True,
)
BINARY_LOSS_WEIGHT = float(cfg["training"]["binary_loss_weight"])
USE_BINARY_AUX_LOSS = bool(_module_cfg.get("binary_auxiliary_head", True)) and (
    BINARY_LOSS_WEIGHT > 0
)

# CORAL 损失函数
criterion = torch.nn.BCEWithLogitsLoss()
criterion_binary = torch.nn.BCEWithLogitsLoss()  # 二分类辅助：是否重度炎症

# 训练模型
BEST_MODEL_PATH = cfg["paths"]["best_model"]
BEST_MODEL_PATH_IN_RUN = (
    os.path.join(TRAIN_OUTPUT_DIR, os.path.basename(str(BEST_MODEL_PATH)))
    if TRAIN_OUTPUT_DIR
    else None
)
NUM_EPOCHS = int(cfg["training"]["epochs"])
PATIENCE = int(cfg["early_stopping"]["patience"])

best_f1 = -1.0          # 最佳F1分数
best_epoch = 0
trigger_times = 0     # 早停计数器

# 历史记录字典
history = {
    'train_loss': [],
    'val_loss': [],
    'val_acc': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': [],
    'val_spec': []
}

print(f"--- 开始训练 (共 {NUM_EPOCHS} 个 epoch) ---")

_t0 = datetime.now(timezone.utc)


# 使用 tqdm 包装外部 Epoch 循环
for epoch in tqdm(range(NUM_EPOCHS), desc="Total Training Progress"):
    # 训练阶段
    model.train()
    running_loss = 0.0
    
    # 使用 tqdm 包装内部 DataLoader 循环
    batch_iterator = tqdm(loader, desc=f"Epoch {epoch+1} Batch Progress", leave=False)
    
    for batch in batch_iterator:
        if len(batch) == 3:
            x, y, mask = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            coral_logits, binary_logits = model(x, mask_overlay=mask)
        else:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            coral_logits, binary_logits = model(x)
        target_coral = labels_to_coral(y, NUM_CLASSES, device)
        loss_coral = criterion(coral_logits, target_coral)
        if USE_BINARY_AUX_LOSS:
            binary_target = (y == 2).float().unsqueeze(1)  # 是否重度炎症
            loss_binary = criterion_binary(binary_logits, binary_target)
            loss = loss_coral + BINARY_LOSS_WEIGHT * loss_binary
        else:
            loss = loss_coral
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # 记录训练损失
        current_loss = loss.item()
        running_loss += current_loss * x.size(0)
        
        # 更新内部进度条的实时损失显示
        batch_iterator.set_postfix({"Loss": f"{current_loss:.4f}"})
    
    # 计算平均训练损失
    epoch_loss = running_loss / len(train_dataset)
    history['train_loss'].append(epoch_loss)
    '''
    for epoch in range(NUM_EPOCHS):
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = weighted_focal_loss(pred, y, alpha)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # 记录训练损失
        running_loss += loss.item() * x.size(0)
    
    # 计算平均训练损失
    epoch_loss = running_loss / len(train_dataset)
    history['train_loss'].append(epoch_loss)
    '''
    # 评估阶段
    val_loss, acc, precision, recall, f1, spec = evaluate(
        model, val_loader, device, criterion, NUM_CLASSES
    )

    # 记录性能指标
    history['val_loss'].append(val_loss)
    history['val_acc'].append(acc)
    history['val_precision'].append(precision)
    history['val_recall'].append(recall)
    history['val_f1'].append(f1)
    history['val_spec'].append(spec)

    # 学习率调度：根据验证 F1 决定是否降低学习率
    scheduler.step(f1)

    # 打印和模型保存/早停逻辑
    current_lr = opt.param_groups[0]['lr']
    print(f"\n★ Epoch {epoch+1}/{NUM_EPOCHS}:")
    print(f"  Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
    print(f"  Validation Metrics → ACC={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Spec={spec:.4f}")

    # 自动找到最佳 epoch
    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch + 1
        trigger_times = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        if BEST_MODEL_PATH_IN_RUN:
            try:
                torch.save(model.state_dict(), BEST_MODEL_PATH_IN_RUN)
            except Exception as e:
                print(f"⚠️ 未能将最佳权重写入归档目录: {e}")
        print(f"  *** F1 提升至 {best_f1:.4f}。模型已保存。***")
    else:
        trigger_times += 1
        print(f"  F1 未提升。早停计数: {trigger_times}/{PATIENCE}")

    # 检查早停条件
    if trigger_times >= PATIENCE:
        print(f"\n--- 早停触发！最佳 F1: {best_f1:.4f} 位于 Epoch {best_epoch} ---")
        break

# 训练结束，加载最佳模型
model.load_state_dict(torch.load(BEST_MODEL_PATH))
print(f"\n已加载最佳模型：{BEST_MODEL_PATH}")

# 训练结束后再次刷新同一套清单（内容一致，verbose=False 减少重复日志）
save_split_manifests(
    DATASET_ROOT,
    SPLIT_MANIFEST_DIR,
    full_dataset,
    list(train_dataset.indices),
    list(val_subset.indices),
    list(test_subset.indices),
    verbose=False,
)
if TRAIN_OUTPUT_DIR:
    save_split_manifests(
        DATASET_ROOT,
        TRAIN_OUTPUT_DIR,
        full_dataset,
        list(train_dataset.indices),
        list(val_subset.indices),
        list(test_subset.indices),
        verbose=False,
    )

_t1 = datetime.now(timezone.utc)
if TRAIN_OUTPUT_DIR:
    try:
        _split_test_ratio = float(
            max(
                0.0,
                1.0 - float(cfg["data"]["split_train"]) - float(cfg["data"]["split_val"]),
            )
        )
        _meta = {
            "created_at_utc": _t0.isoformat(),
            "finished_at_utc": _t1.isoformat(),
            "random_seed": int(RANDOM_SEED),
            "split_train_ratio": float(cfg["data"]["split_train"]),
            "split_val_ratio": float(cfg["data"]["split_val"]),
            "split_test_ratio": _split_test_ratio,
            "counts": {
                "total": int(len(full_dataset)),
                "train": int(len(train_dataset)),
                "val": int(len(val_dataset)),
                "test": int(len(test_subset)),
            },
            "dataset_root_abs": os.path.abspath(DATASET_ROOT),
            "dataset_root_config": str(DATASET_ROOT),
            "config_path": str(_CONFIG_PATH.resolve()),
            "run_output_dir_abs": os.path.abspath(TRAIN_OUTPUT_DIR),
            "model_checkpoint_in_run": os.path.basename(str(BEST_MODEL_PATH_IN_RUN)) if BEST_MODEL_PATH_IN_RUN else None,
            "best_epoch": int(best_epoch),
            "best_f1": float(best_f1),
            "epochs_ran": int(len(history["train_loss"])),
            "class_counts_train": {int(i): int(v) for i, v in enumerate(class_counts.tolist())},
            "model_name": str(_model_name),
            "modules": dict(_module_cfg),
            "freeze_backbone": bool(FREEZE_BACKBONE),
            "reproduce_hint_zh": (
                "清单路径相对于 dataset_root；评估可将 evaluate.val_manifest 设为本目录下的 val.txt，"
                "paths.best_model 设为同目录 model.pth。"
            ),
        }
        with open(os.path.join(TRAIN_OUTPUT_DIR, "split_meta.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(
                _to_builtin(_meta),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        with open(os.path.join(TRAIN_OUTPUT_DIR, "history.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(
                _to_builtin(history),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
    except Exception as e:
        print(f"⚠️ 写入训练元信息失败: {e}")

# 自动可视化函数
def plot_metrics(history, best_epoch, save_path="training_performance_curves.png"):
    """绘制训练损失和验证性能曲线"""
    
    num_epochs_ran = len(history['train_loss'])
    epochs = range(1, num_epochs_ran + 1)
    
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：损失曲线
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='darkblue', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation Loss', color='teal', linewidth=2)
    axes[0].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[0].set_title('Train/Validation Loss vs. Epoch', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle='--')

    # 右图：性能指标曲线
    axes[1].plot(epochs, history['val_acc'], label='Validation Accuracy', marker='s', color='green')
    axes[1].plot(epochs, history['val_f1'], label='Validation F1 Score (Best Criterion)', marker='D', color='red', linewidth=2)
    axes[1].plot(epochs, history['val_precision'], label='Validation Precision', marker='^', color='orange')
    axes[1].plot(epochs, history['val_recall'], label='Validation Recall', marker='v', color='purple')
    axes[1].plot(epochs, history['val_spec'], label='Validation Specificity', marker='*', color='brown')
    axes[1].axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[1].set_title('Validation Metrics vs. Epoch', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle='--')
    
    plt.suptitle("Model Performance Over Training Epochs", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

# 训练结束后调用可视化函数
plot_metrics(history, best_epoch, save_path=cfg["paths"]["training_curves"])
if TRAIN_OUTPUT_DIR:
    try:
        _curves_name = os.path.basename(str(cfg["paths"]["training_curves"]) or "training_performance_curves.png")
        _curves_out = os.path.join(TRAIN_OUTPUT_DIR, _curves_name)
        plot_metrics(history, best_epoch, save_path=_curves_out)
    except Exception as e:
        print(f"⚠️ 未能将训练曲线保存到归档目录: {e}")