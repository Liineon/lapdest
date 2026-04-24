"""对比实验用 baseline 模型（与主模型相同 CORAL + 二分类头接口）。"""

from .convnext_plain import ConvNeXtPlainCoral
from .resnet_coral import ResNet18Coral

__all__ = ["ConvNeXtPlainCoral", "ResNet18Coral"]
