"""Pretrained backbones and their layer groups for iterative unfreezing."""

from __future__ import annotations

from dataclasses import dataclass

import timm
from timm.models.convnext import ConvNeXt
from timm.models.vision_transformer import VisionTransformer
from torch import nn


@dataclass(frozen=True)
class ArchitectureSpec:
    """A timm model and a short name used in tables and plots."""

    name: str
    timm_name: str


CONVNEXT_TINY = ArchitectureSpec("ConvNeXt-T", "convnext_tiny.fb_in22k_ft_in1k")
DEIT_SMALL = ArchitectureSpec("DeiT-S", "deit_small_patch16_224.fb_in1k")


def create_model(
    spec: ArchitectureSpec,
    n_classes: int,
    image_size: int,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    """timm model with a new classification head for ``n_classes``."""
    kwargs = {}
    if spec.timm_name.startswith(("vit", "deit")):
        # Transformers need the input size to interpolate their position embeddings.
        kwargs["img_size"] = image_size
    return timm.create_model(
        spec.timm_name, pretrained=pretrained, num_classes=n_classes, drop_path_rate=drop_path_rate, **kwargs
    )


def layer_group_prefixes(model: nn.Module) -> list[tuple[str, ...]]:
    """Parameter name prefixes of layer groups ordered from the head (index 0) to the input.

    Unfreezing ``k`` groups means training the ``k`` groups closest to the head.
    """
    if isinstance(model, ConvNeXt):
        return [
            ("head",),
            ("stages.3",),
            ("stages.2",),
            ("stages.1",),
            ("stages.0", "stem", "norm_pre"),
        ]
    if isinstance(model, VisionTransformer):
        n_blocks = len(model.blocks)
        blocks_per_group = 3
        groups: list[tuple[str, ...]] = [("head", "norm", "fc_norm")]
        for end in range(n_blocks, 0, -blocks_per_group):
            groups.append(tuple(f"blocks.{i}" for i in range(max(0, end - blocks_per_group), end)))
        groups[-1] += ("patch_embed", "cls_token", "reg_token", "pos_embed")
        return groups
    raise TypeError(f"Layer groups are not defined for {type(model).__name__}.")


def layer_groups(model: nn.Module) -> list[list[tuple[str, nn.Parameter]]]:
    """Named parameters of every layer group, ordered from the head to the input.

    Raises if some parameter does not belong to exactly one group, so that no
    layer silently stays frozen forever.
    """
    prefixes = layer_group_prefixes(model)
    groups: list[list[tuple[str, nn.Parameter]]] = [[] for _ in prefixes]
    for name, parameter in model.named_parameters():
        matches = [
            index
            for index, group_prefixes in enumerate(prefixes)
            if any(name == prefix or name.startswith(prefix + ".") for prefix in group_prefixes)
        ]
        if len(matches) != 1:
            raise ValueError(f"Parameter {name} matches {len(matches)} layer groups instead of one.")
        groups[matches[0]].append((name, parameter))
    return groups
