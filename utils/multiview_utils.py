import torch

from utils.graphics_utils import geom_transform_points


def _prepare_object_mask(camera):
    if getattr(camera, "objects", None) is None:
        return None

    mask = camera.objects
    if mask.dim() == 3:
        mask = mask.squeeze()
    return mask


def project_points_to_camera(points, camera):
    """
    Project world-space points into a camera and return pixel coordinates plus a
    conservative validity mask based on front-facing and image bounds checks.
    """
    cam_xyz = geom_transform_points(points, camera.world_view_transform)
    clip_xyz = geom_transform_points(points, camera.full_proj_transform)

    x_ndc = clip_xyz[:, 0]
    y_ndc = clip_xyz[:, 1]

    width = float(camera.image_width)
    height = float(camera.image_height)

    u = ((x_ndc + 1.0) * 0.5) * max(width - 1.0, 1.0)
    v = ((1.0 - y_ndc) * 0.5) * max(height - 1.0, 1.0)

    valid = (
        (cam_xyz[:, 2] > 0.0)
        & (x_ndc >= -1.0)
        & (x_ndc <= 1.0)
        & (y_ndc >= -1.0)
        & (y_ndc <= 1.0)
    )

    return u, v, valid


def sample_camera_labels(camera, points, ignore_label=-1):
    """
    Sample per-point pseudo labels from a camera's 2D object mask using nearest
    pixel lookup after projection.
    """
    device = points.device
    labels = torch.full((points.shape[0],), -1, device=device, dtype=torch.long)
    valid = torch.zeros((points.shape[0],), device=device, dtype=torch.bool)

    mask = _prepare_object_mask(camera)
    if mask is None or points.numel() == 0:
        return labels, valid

    u, v, proj_valid = project_points_to_camera(points, camera)
    if not proj_valid.any():
        return labels, valid

    mask = mask.to(device)
    pixel_x = torch.round(u[proj_valid]).long().clamp_(0, camera.image_width - 1)
    pixel_y = torch.round(v[proj_valid]).long().clamp_(0, camera.image_height - 1)

    sampled_labels = mask[pixel_y, pixel_x].long()
    sampled_valid = torch.ones_like(sampled_labels, dtype=torch.bool)
    if ignore_label >= 0:
        sampled_valid = sampled_valid & (sampled_labels != ignore_label)

    valid_indices = proj_valid.nonzero(as_tuple=False).squeeze(-1)
    labels[valid_indices] = sampled_labels
    valid[valid_indices] = sampled_valid

    return labels, valid


def collect_multiview_labels(cameras, points, ignore_label=-1):
    """
    Gather per-camera labels and visibility flags for a shared set of 3D points.
    """
    if cameras is None or len(cameras) == 0:
        device = points.device
        empty_labels = torch.empty((0, points.shape[0]), device=device, dtype=torch.long)
        empty_valid = torch.empty((0, points.shape[0]), device=device, dtype=torch.bool)
        return empty_labels, empty_valid

    labels = []
    valid_masks = []
    for camera in cameras:
        cam_labels, cam_valid = sample_camera_labels(camera, points, ignore_label=ignore_label)
        labels.append(cam_labels)
        valid_masks.append(cam_valid)

    return torch.stack(labels, dim=0), torch.stack(valid_masks, dim=0)
