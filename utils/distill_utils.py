from pathlib import Path

import torch
import torch.nn.functional as F


def resolve_feature_cache_dir(scene_source_path, configured_dir):
    feature_dir = Path(configured_dir)
    if not feature_dir.is_absolute():
        feature_dir = Path(scene_source_path) / feature_dir
    return feature_dir


class FeatureCacheLoader:
    def __init__(self, feature_dir):
        self.feature_dir = Path(feature_dir)
        if not self.feature_dir.exists():
            raise FileNotFoundError(
                f"DINO feature cache directory not found: {self.feature_dir}"
            )
        self._cache = {}

    def _cache_path(self, image_name):
        return self.feature_dir / f"{image_name}.pt"

    def _load_entry(self, image_name):
        cache_path = self._cache_path(image_name)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing DINO feature cache for image '{image_name}': {cache_path}"
            )

        entry = torch.load(cache_path, map_location="cpu")
        if not isinstance(entry, dict):
            raise ValueError(f"Feature cache must be a dict: {cache_path}")
        if "feature_map" not in entry:
            raise ValueError(f"Feature cache missing 'feature_map': {cache_path}")

        feature_map = entry["feature_map"]
        if not torch.is_tensor(feature_map):
            raise ValueError(f"'feature_map' must be a tensor: {cache_path}")
        if feature_map.dim() == 4:
            if feature_map.size(0) != 1:
                raise ValueError(
                    f"4D feature_map must have batch size 1, got {feature_map.shape}: {cache_path}"
                )
            feature_map = feature_map.squeeze(0)
        if feature_map.dim() != 3:
            raise ValueError(
                f"feature_map must have shape [C,H,W] or [1,C,H,W], got {feature_map.shape}: {cache_path}"
            )

        image_size = entry.get("image_size")
        if image_size is None:
            image_size = tuple(int(v) for v in feature_map.shape[-2:])
        if len(image_size) != 2:
            raise ValueError(f"image_size must be a 2-tuple, got {image_size}: {cache_path}")

        validated = {
            "feature_map": feature_map.float().contiguous(),
            "image_size": (int(image_size[0]), int(image_size[1])),
            "model_name": str(entry.get("model_name", "unknown")),
        }
        self._cache[image_name] = validated
        return validated

    def get_entry(self, image_name):
        return self._cache.get(image_name) or self._load_entry(image_name)

    def get_teacher_dim(self, image_name):
        return int(self.get_entry(image_name)["feature_map"].shape[0])

    def get_feature_map(self, image_name, target_hw=None, device=None, dtype=None):
        entry = self.get_entry(image_name)
        feature_map = entry["feature_map"]

        if target_hw is not None and tuple(feature_map.shape[-2:]) != tuple(target_hw):
            feature_map = F.interpolate(
                feature_map.unsqueeze(0),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if dtype is not None:
            feature_map = feature_map.to(dtype=dtype)
        if device is not None:
            feature_map = feature_map.to(device=device)

        return feature_map, entry


def render_feature_foreground_mask(render_object, eps=1e-6):
    if render_object.dim() == 3:
        activation = torch.norm(render_object, dim=0)
    elif render_object.dim() == 4:
        activation = torch.norm(render_object, dim=1)
    else:
        raise ValueError(
            f"render_object must have shape [C,H,W] or [B,C,H,W], got {render_object.shape}"
        )
    return activation > eps


def cosine_distill_loss(student_map, teacher_map, mask=None, eps=1e-8):
    if student_map.dim() == 3:
        student_map = student_map.unsqueeze(0)
    if teacher_map.dim() == 3:
        teacher_map = teacher_map.unsqueeze(0)
    if student_map.shape != teacher_map.shape:
        raise ValueError(
            f"Student and teacher maps must have the same shape, got {student_map.shape} vs {teacher_map.shape}"
        )

    student_map = F.normalize(student_map, dim=1, eps=eps)
    teacher_map = F.normalize(teacher_map, dim=1, eps=eps)
    loss_map = 1.0 - (student_map * teacher_map).sum(dim=1)

    if mask is None:
        return loss_map.mean()

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    elif mask.dim() != 3:
        raise ValueError(f"Mask must have shape [H,W], [B,H,W], or [B,1,H,W], got {mask.shape}")

    weight = mask.to(loss_map.dtype)
    return (loss_map * weight).sum() / (weight.sum() + eps)
