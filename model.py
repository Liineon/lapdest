from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights


class GastroIntestinalAttention(nn.Module):
    """
    Hybrid channel-spatial attention with a soft anatomical prior that nudges
    the receptive field toward the mid-lower abdominal region of the fish.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        prior_center=(0.55, 0.5),
        prior_sigma=(0.12, 0.25),
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
            nn.Conv2d(reduced, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        self.center_y, self.center_x = prior_center
        self.sigma_y, self.sigma_x = prior_sigma

        # Learnable mixing factor between learned spatial map and anatomical prior.
        self.prior_logit = nn.Parameter(torch.tensor(0.0))
        self._cached_prior = None
        self._cached_size = None

    def _build_prior(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        if self._cached_prior is not None and self._cached_size == (height, width):
            return self._cached_prior.to(device)

        yy = torch.linspace(0.0, 1.0, height, device=device).unsqueeze(1)
        xx = torch.linspace(0.0, 1.0, width, device=device).unsqueeze(0)
        gauss = torch.exp(
            -(
                ((yy - self.center_y) ** 2) / (2 * (self.sigma_y ** 2))
                + ((xx - self.center_x) ** 2) / (2 * (self.sigma_x ** 2))
            )
        )
        gauss = gauss / gauss.max().clamp_min(1e-6)
        gauss = gauss.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        self._cached_prior = gauss.detach()
        self._cached_size = (height, width)
        return gauss

    def forward(self, x: torch.Tensor, roi_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Channel refinement
        channel_map = self.channel_attn(x)
        x = x * channel_map

        # Spatial attention + anatomical prior blending
        spatial_map = self.spatial_attn(x)
        b, _, h, w = spatial_map.shape

        if roi_prior is not None:
            prior = F.interpolate(
                roi_prior, size=(h, w), mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
        else:
            prior = self._build_prior(h, w, x.device)

        prior = prior.expand(b, -1, -1, -1)
        prior_weight = torch.sigmoid(self.prior_logit)
        fused_spatial = spatial_map * (1 - prior_weight) + prior * prior_weight

        return x * fused_spatial


class DetailEnhancementModule(nn.Module):
    """
    显式分离并增强高频细节（纹理/边缘），使模型更注重局部纹理。
    通过 原图 - 低通 得到细节分量，再用可学习门控加权增强。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        roi_dilate_kernel: int = 7,
        roi_dilate_iter: int = 1,
        roi_blend_alpha: float = 1.0,
        gate_hard_gamma: float = 1.8,
        roi_focus_threshold: float = 0.4,
    ):
        super().__init__()
        # [MOD] 使用 depthwise Conv 替代 AvgPool，模拟可学习高斯低通滤波。
        self.low_pass = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        # [MOD] 以近似高斯核初始化低通卷积，后续可继续学习微调。
        self._init_low_pass_as_gaussian(channels, kernel_size)
        # [MOD] 增强 detail_gate 表达能力：Conv-BN-ReLU-Conv-Sigmoid。
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.Sigmoid(),
        )
        # ROI 扩张超参：先扩张再参与细节增强，扩大关注范围。
        roi_dilate_kernel = int(roi_dilate_kernel)
        if roi_dilate_kernel < 1:
            roi_dilate_kernel = 1
        if roi_dilate_kernel % 2 == 0:
            roi_dilate_kernel += 1
        self.roi_dilate_kernel = roi_dilate_kernel
        self.roi_dilate_iter = max(0, int(roi_dilate_iter))
        self.roi_blend_alpha = float(roi_blend_alpha)
        self.gate_hard_gamma = max(1.0, float(gate_hard_gamma))
        t = float(roi_focus_threshold)
        self.roi_focus_threshold = 0.0 if t < 0.0 else (0.95 if t > 0.95 else t)

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: Optional[float] = None) -> torch.Tensor:
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum().clamp_min(1e-6)
        kernel2d = torch.outer(g, g)
        return kernel2d / kernel2d.sum().clamp_min(1e-6)

    def _init_low_pass_as_gaussian(self, channels: int, kernel_size: int) -> None:
        kernel2d = self._gaussian_kernel(kernel_size).view(1, 1, kernel_size, kernel_size)
        weight = kernel2d.repeat(channels, 1, 1, 1)
        with torch.no_grad():
            self.low_pass.weight.copy_(weight)

    def forward(
        self,
        x: torch.Tensor,
        roi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        low = self.low_pass(x)
        detail = x - low
        # [MOD] ROI 仅引导 detail 分支，不直接作用于原始特征 x。
        if roi is not None:
            roi_resized = F.interpolate(
                roi, size=detail.shape[2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            roi_dilated = roi_resized
            if self.roi_dilate_iter > 0 and self.roi_dilate_kernel > 1:
                pad = self.roi_dilate_kernel // 2
                for _ in range(self.roi_dilate_iter):
                    roi_dilated = F.max_pool2d(
                        roi_dilated,
                        kernel_size=self.roi_dilate_kernel,
                        stride=1,
                        padding=pad,
                    )
                roi_dilated = roi_dilated.clamp(0.0, 1.0)
            tau = self.roi_focus_threshold
            denom = max(1e-6, 1.0 - tau)
            roi_focused = ((roi_dilated - tau) / denom).clamp(0.0, 1.0)
            detail = detail * (1.0 + self.roi_blend_alpha * roi_focused)
        weight = self.detail_gate(detail)
        if self.gate_hard_gamma > 1.0:
            weight = weight.pow(self.gate_hard_gamma)
        return x + weight * detail


class _AttentionPassthrough(nn.Module):
    """关闭 GI 注意力时占位，保持 forward(x, roi_prior) 接口。"""

    def forward(self, x: torch.Tensor, roi_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        del roi_prior
        return x


class ConvNeXtWithGIAttention(nn.Module):
    """
    Wrap ConvNeXt-Tiny so we can inject ROI-aware attention before pooling.
    """

    def __init__(
        self,
        num_classes: int = 3,
        use_attention: bool = True,
        use_detail_enhance: bool = True,
        use_masked_pool: bool = True,
        use_mask_feature_gate: bool = True,
        mask_gate_beta: float = 0.0,
        use_binary_head: bool = True,
    ):
        super().__init__()
        self.use_masked_pool = use_masked_pool
        self.use_mask_feature_gate = use_mask_feature_gate
        self.mask_gate_beta = float(mask_gate_beta)
        self.use_binary_head = use_binary_head

        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # CORAL ordinal regression: 输出 num_classes-1 维（3类→2维）
        backbone.classifier = nn.Sequential(
            backbone.classifier[0],  # Flatten
            backbone.classifier[1],  # LayerNorm
            nn.Dropout(0.5),
            nn.Linear(768, num_classes - 1),
        )

        self.backbone = backbone
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

        self.attention = (
            GastroIntestinalAttention(channels=768) if use_attention else _AttentionPassthrough()
        )
        self.detail_enhance = (
            DetailEnhancementModule(channels=768) if use_detail_enhance else _AttentionPassthrough()
        )
        # Grad-CAM 目标层：放在掩码门控之后，便于解释最终用于分类的特征。
        self.cam_target_layer = nn.Identity()
        # 辅助二分类头：是否重度炎症（label==2），用于学习"宽间距"
        self.binary_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(768, 1),
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
    ) -> tuple:
        """
        mask_overlay: (B, 1, H, W) 来自 masks_overlay，1=前景 0=背景。None 时使用全局平均池化。
        返回: (coral_logits, binary_logits)
        - coral_logits: (B, num_classes-1) 有序回归
        - binary_logits: (B, 1) 是否重度炎症
        """
        feats = self.features(x)
        feats = self.attention(feats, roi_prior)
        # [MOD] 结构顺序：Stage4 -> GI Attention -> DetailEnhancement -> 分类头。
        # [MOD] roi_prior 传入细节增强模块，用于肠道区域高频细节引导。
        feats = self.detail_enhance(feats, roi_prior)

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
        coral_logits = self.classifier(pooled)
        if self.use_binary_head:
            binary_logits = self.binary_head(pooled)
        else:
            binary_logits = torch.zeros(
                x.size(0), 1, device=x.device, dtype=coral_logits.dtype
            )
        return coral_logits, binary_logits


def _module_flags(modules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    m = modules or {}
    return {
        "use_attention": bool(m.get("gi_attention", True)),
        "use_detail_enhance": bool(m.get("detail_enhancement", True)),
        "use_masked_pool": bool(m.get("mask_overlay_pooling", True)),
        "use_mask_feature_gate": bool(m.get("mask_feature_gating", True)),
        "mask_gate_beta": float(m.get("mask_gate_beta", 0.0)),
        "use_binary_head": bool(m.get("binary_auxiliary_head", True)),
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
    - gi_attention, detail_enhancement, mask_overlay_pooling, binary_auxiliary_head（bool）
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
    冻结 backbone 特征，解冻分类头与二分类头；若有 GI 注意力/细节模块则一并冻结。
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
    if hasattr(model, "binary_head"):
        for param in model.binary_head.parameters():
            param.requires_grad = True
    if hasattr(model, "attention"):
        for param in model.attention.parameters():
            param.requires_grad = False
    if hasattr(model, "detail_enhance"):
        for param in model.detail_enhance.parameters():
            param.requires_grad = False
