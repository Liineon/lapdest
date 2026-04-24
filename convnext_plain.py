import torch.nn as nn
from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtPlainCoral(nn.Module):
    """
    ConvNeXt-Tiny + CORAL + 二分类辅助头，无 GI 注意力、无细节增强、不使用掩码池化。
    forward 签名与主模型一致，mask_overlay / roi_prior 均忽略。
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        backbone.classifier = nn.Sequential(
            backbone.classifier[0],
            backbone.classifier[1],
            nn.Dropout(0.5),
            nn.Linear(768, num_classes - 1),
        )
        self.backbone = backbone
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        self.binary_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(768, 1),
        )

    def forward(self, x, roi_prior=None, mask_overlay=None):
        del roi_prior, mask_overlay
        feats = self.features(x)
        pooled = self.avgpool(feats)
        coral_logits = self.classifier(pooled)
        binary_logits = self.binary_head(pooled)
        return coral_logits, binary_logits
