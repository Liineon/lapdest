from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights


class GastroIntestinalAttention(nn.Module):
    """
    Hybrid channel-spatial attention for ROI-focused representation learning.
    Gaussian anatomical prior has been removed.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ):
        super().__init__()
        reduced = max(1, channels // reduction)

        # Channel attention (SE style) to re-weight feature responses globally.
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial attention guided by local context.
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            # 使用非对称卷积以适配 192x640 的矩形输入特征图。
            nn.Conv2d(reduced, 1, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, roi_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Channel refinement
        channel_map = self.channel_attn(x)
        x = x * channel_map

        # Spatial attention with optional ROI guidance.
        spatial_map = self.spatial_attn(x)
        _, _, h, w = spatial_map.shape
        if roi_prior is not None:
            roi_map = F.interpolate(
                roi_prior, size=(h, w), mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            spatial_map = spatial_map * roi_map

        return x * spatial_map


class _AttentionPassthrough(nn.Module):
    """关闭 GI 注意力时占位，保持 forward(x, roi_prior) 接口。"""

    def forward(self, x: torch.Tensor, roi_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        del roi_prior
        return x


class GLCMFusionHead(nn.Module):
    def __init__(self, deep_dim: int = 768, glcm_dim: int = 4, num_classes: int = 3):
        super().__init__()
        self.glcm_proj = nn.Sequential(
            nn.Linear(glcm_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )
        self.fusion = nn.Sequential(
            nn.Linear(deep_dim + 32, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.coral_out = nn.Linear(256, num_classes - 1)
        self.binary_out = nn.Linear(256, 1)

    def forward(self, feat_vec: torch.Tensor, glcm_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        glcm_feat = self.glcm_proj(glcm_vec)
        fused = self.fusion(torch.cat([feat_vec, glcm_feat], dim=1))
        return self.coral_out(fused), self.binary_out(fused)


class ConvNeXtWithGIAttention(nn.Module):
    """
    Wrap ConvNeXt-Tiny so we can inject ROI-aware attention before pooling.
    """

    def __init__(
        self,
        num_classes: int = 3,
        use_attention: bool = True,
        use_masked_pool: bool = True,
        use_mask_feature_gate: bool = True,
        mask_gate_beta: float = 0.0,
        use_glcm_fusion: bool = True,
        glcm_dim: int = 4,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_masked_pool = use_masked_pool
        self.use_mask_feature_gate = use_mask_feature_gate
        self.mask_gate_beta = float(mask_gate_beta)
        self.use_glcm_fusion = bool(use_glcm_fusion)
        self.glcm_dim = int(glcm_dim)

        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # CORAL ordinal regression: 输出 num_classes-1 维（3类→2维）
        backbone.classifier = nn.Sequential(
            backbone.classifier[0],  # Flatten
            backbone.classifier[1],  # LayerNorm
            nn.Dropout(0.3),
            nn.Linear(768, num_classes - 1),
        )

        self.backbone = backbone
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

        self.attention = (
            GastroIntestinalAttention(channels=768) if use_attention else _AttentionPassthrough()
        )
        # Grad-CAM 目标层：放在掩码门控之后，便于解释最终用于分类的特征。
        self.cam_target_layer = nn.Identity()
        self.glcm_fusion_head = GLCMFusionHead(
            deep_dim=768, glcm_dim=self.glcm_dim, num_classes=self.num_classes
        )

    def _masked_pool(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """对特征做掩码加权平均池化，仅对前景区域求平均。掩码为空时回退到全局平均池化。"""
        sum_mask = mask.sum(dim=(2, 3))  # (B, 1)
        feats_masked = feats * mask
        sum_feats = feats_masked.sum(dim=(2, 3))
        sum_mask_safe = sum_mask.clamp(min=1e-6)
        pooled = (sum_feats / sum_mask_safe).unsqueeze(-1).unsqueeze(-1)
        fallback = (sum_mask < 1.0).squeeze(1)
        if fallback.any():
            pooled[fallback] = self.avgpool(feats[fallback])
        return pooled

    def _apply_mask_feature_gate(self, feats: torch.Tensor, mask_feat: torch.Tensor) -> torch.Tensor:
        """
        特征门控：feats *= beta + (1-beta)*mask。
        - beta=0 -> 硬门控（掩码外完全抑制）
        - 0<beta<1 -> 软门控
        - beta=1 -> 不使用掩码门控
        """
        beta = float(self.mask_gate_beta)
        beta = 0.0 if beta < 0.0 else (1.0 if beta > 1.0 else beta)
        if beta <= 1e-8:
            gate = (mask_feat > 0.5).float()
        else:
            gate = beta + (1.0 - beta) * mask_feat.clamp(0.0, 1.0)
        return feats * gate

    def forward(
        self,
        x: torch.Tensor,
        roi_prior: Optional[torch.Tensor] = None,
        mask_overlay: Optional[torch.Tensor] = None,
        glcm_vec: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        mask_overlay: (B, 1, H, W) 来自 masks_overlay，1=前景 0=背景。None 时使用全局平均池化。
        glcm_vec: (B, glcm_dim) 离线 GLCM 纹理向量，可为 None（自动补零）。
        返回: (coral_logits, aux_logits)
        - coral_logits: (B, num_classes-1) 有序回归
        - aux_logits: (B, 1) 预留辅助输出
        """
        feats = self.features(x)
        feats = self.attention(feats, roi_prior)

        mask_feat = None
        if mask_overlay is not None:
            mask_feat = F.interpolate(
                mask_overlay, size=feats.shape[2:], mode="bilinear", align_corners=False
            )
            if self.use_mask_feature_gate:
                feats = self._apply_mask_feature_gate(feats, mask_feat)

        if self.use_masked_pool and mask_feat is not None:
            feats = self.cam_target_layer(feats)
            pooled = self._masked_pool(feats, mask_feat)
        else:
            feats = self.cam_target_layer(feats)
            pooled = self.avgpool(feats)

        feat_vec = pooled.flatten(1)
        if self.use_glcm_fusion:
            if glcm_vec is None:
                glcm_vec = feat_vec.new_zeros((feat_vec.shape[0], self.glcm_dim))
            else:
                glcm_vec = glcm_vec.to(device=feat_vec.device, dtype=feat_vec.dtype)
            coral_logits, aux_logits = self.glcm_fusion_head(feat_vec, glcm_vec)
            return coral_logits, aux_logits

        coral_logits = self.classifier(feat_vec)
        aux_logits = coral_logits.new_zeros((coral_logits.shape[0], 1))
        return coral_logits, aux_logits


def _module_flags(modules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    m = modules or {}
    return {
        "use_attention": bool(m.get("gi_attention", True)),
        "use_masked_pool": bool(m.get("mask_overlay_pooling", True)),
        "use_mask_feature_gate": bool(m.get("mask_feature_gating", True)),
        "mask_gate_beta": float(m.get("mask_gate_beta", 0.0)),
        "use_glcm_fusion": bool(m.get("glcm_fusion", True)),
        "glcm_dim": int(m.get("glcm_dim", 4)),
    }


def get_model(
    num_classes: int = 3,
    model_name: str = "convnext_gi",
    modules: Optional[Dict[str, Any]] = None,
):
    """
    model_name（对比实验一键切换）:
      - convnext_gi: ConvNeXt-Tiny + 可配置子模块（见 modules）
      - convnext_plain: ConvNeXt-Tiny + CORAL，无注意力/细节/掩码池化
      - resnet18_coral: ResNet-18 + CORAL

    modules 仅对 convnext_gi 生效，键：
    - gi_attention, mask_overlay_pooling, glcm_fusion（bool）
    - glcm_dim（int，离线 GLCM 特征维度）
    - mask_feature_gating（bool，是否启用特征门控）
    - mask_gate_beta（float，0=硬门控，(0,1)=软门控，1=关闭门控）
    """
    name = (model_name or "convnext_gi").lower().replace("-", "_")
    flags = _module_flags(modules)

    if name in ("convnext_gi", "default", "convnext_gi_attention"):
        return ConvNeXtWithGIAttention(num_classes=num_classes, **flags)
    if name in ("convnext_plain", "baseline_convnext", "convnext_baseline"):
        from baselines.convnext_plain import ConvNeXtPlainCoral

        return ConvNeXtPlainCoral(num_classes=num_classes)
    if name in ("resnet18", "resnet18_coral", "baseline_resnet18"):
        from baselines.resnet_coral import ResNet18Coral

        return ResNet18Coral(num_classes=num_classes)

    raise ValueError(
        f"未知 model_name: {model_name!r}。可选: convnext_gi, convnext_plain, resnet18_coral"
    )


def apply_freeze_backbone(model: nn.Module) -> None:
    """
    冻结 backbone 特征，解冻分类头与 GLCM 融合头；若有 GI 注意力模块则一并冻结。
    适用于 convnext_gi / convnext_plain / resnet18_coral。
    """
    if not hasattr(model, "backbone"):
        return
    for param in model.backbone.parameters():
        param.requires_grad = False
    if hasattr(model.backbone, "classifier"):
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True
    # ResNet baseline：分类头在 backbone 外，与 backbone.classifier 非同一引用
    if hasattr(model, "classifier"):
        bb_clf = getattr(model.backbone, "classifier", None)
        if bb_clf is None or id(model.classifier) != id(bb_clf):
            for param in model.classifier.parameters():
                param.requires_grad = True
    if hasattr(model, "glcm_fusion_head"):
        for param in model.glcm_fusion_head.parameters():
            param.requires_grad = True
    if hasattr(model, "attention"):
        for param in model.attention.parameters():
            param.requires_grad = False
