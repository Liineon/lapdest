import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Coral(nn.Module):
    """
    ResNet-18 ImageNet 预训练骨干 + CORAL + 二分类头；忽略 mask / roi_prior。
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        nf = 512
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(nf, num_classes - 1),
        )
        self.binary_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(nf, 1),
        )

    def forward(self, x, roi_prior=None, mask_overlay=None):
        del roi_prior, mask_overlay
        feats = self.backbone(x)
        coral_logits = self.classifier(feats)
        binary_logits = self.binary_head(feats)
        return coral_logits, binary_logits
