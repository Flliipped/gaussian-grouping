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


def sample_camera_labels(camera, points, point_ids=None, visibility_mask=None, ignore_label=-1):
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
    if visibility_mask is not None and point_ids is not None:
        proj_valid = proj_valid & visibility_mask[point_ids]
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


def collect_multiview_labels(cameras, points, point_ids=None, visibility_masks=None, ignore_label=-1):
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
    for view_idx, camera in enumerate(cameras):
        visibility_mask = None
        if visibility_masks is not None and view_idx < len(visibility_masks):
            visibility_mask = visibility_masks[view_idx]
        cam_labels, cam_valid = sample_camera_labels(
            camera,
            points,
            point_ids=point_ids,
            visibility_mask=visibility_mask,
            ignore_label=ignore_label,
        )
        labels.append(cam_labels)
        valid_masks.append(cam_valid)

    return torch.stack(labels, dim=0), torch.stack(valid_masks, dim=0)


def compute_point_label_consensus(view_labels, view_valid, num_classes=None, min_views=2, conf_tau=0.7):
    """
    Build a Gaussian-level pseudo label from multi-view votes.

    Returns:
        consensus_label: [P] long
        consensus_valid: [P] bool
        consensus_confidence: [P] float in [0, 1], softened instead of hard-thresholded
        valid_view_count: [P] float
    """
    device = view_labels.device
    num_points = view_labels.shape[1]

    consensus_label = torch.full((num_points,), -1, device=device, dtype=torch.long)
    consensus_valid = torch.zeros((num_points,), device=device, dtype=torch.bool)
    consensus_confidence = torch.zeros((num_points,), device=device, dtype=torch.float32)
    valid_view_count = view_valid.to(torch.float32).sum(dim=0)

    if view_labels.numel() == 0 or num_points == 0:
        return consensus_label, consensus_valid, consensus_confidence, valid_view_count

    valid_mask = view_valid.bool()
    if not valid_mask.any():
        return consensus_label, consensus_valid, consensus_confidence, valid_view_count

    valid_labels = view_labels[valid_mask]
    if num_classes is None:
        num_classes = int(valid_labels.max().item()) + 1 if valid_labels.numel() > 0 else 1
    num_classes = max(int(num_classes), 1)

    point_ids = torch.arange(num_points, device=device).unsqueeze(0).expand_as(view_labels)
    flat_point_ids = point_ids[valid_mask]
    flat_labels = valid_labels.long().clamp(min=0, max=num_classes - 1)

    vote_bins = flat_point_ids * num_classes + flat_labels
    vote_counts = torch.bincount(vote_bins, minlength=num_points * num_classes)
    vote_counts = vote_counts.view(num_points, num_classes).to(torch.float32)

    max_votes, best_labels = vote_counts.max(dim=1)
    confidence = max_votes / valid_view_count.clamp_min(1.0)
    is_supported = valid_view_count >= max(int(min_views), 1)

    # Keep a soft confidence instead of hard-rejecting low-confidence votes.
    # This matches the training strategy of downweighting noisy pseudo labels.
    confidence_soft = confidence * torch.sigmoid((confidence - conf_tau) / 0.1)

    consensus_label[is_supported] = best_labels[is_supported]
    consensus_valid = is_supported
    consensus_confidence = torch.where(
        is_supported,
        confidence_soft,
        torch.zeros_like(confidence_soft),
    )

    return consensus_label, consensus_valid, consensus_confidence, valid_view_count


def compute_pair_label_persistence(
    view_labels,
    view_valid,
    anchor_indices,
    neighbor_indices,
    min_views=2,
    conf_tau=0.7,
):
    """
    Estimate whether a pair of Gaussians stays on the same instance boundary
    relation across views.

    Returns:
        same_confidence: [S, K] float
        diff_confidence: [S, K] float
        stability: [S, K] float
        valid_view_count: [S, K] float
    """
    device = view_labels.device
    sample_count = anchor_indices.shape[0]
    effective_k = neighbor_indices.shape[1] if neighbor_indices.dim() == 2 else 0

    zeros = torch.zeros((sample_count, effective_k), device=device, dtype=torch.float32)
    if view_labels.numel() == 0 or sample_count == 0 or effective_k == 0:
        return zeros, zeros, zeros, zeros

    anchor_indices = anchor_indices.long()
    neighbor_indices = neighbor_indices.long()
    num_views = view_labels.shape[0]

    anchor_labels = view_labels[:, anchor_indices]
    anchor_valid = view_valid[:, anchor_indices]

    neighbor_flat = neighbor_indices.reshape(-1)
    gather_index = neighbor_flat.unsqueeze(0).expand(num_views, -1)
    neighbor_labels = torch.gather(view_labels, 1, gather_index).view(num_views, sample_count, effective_k)
    neighbor_valid = torch.gather(view_valid, 1, gather_index).view(num_views, sample_count, effective_k)

    pair_valid = anchor_valid.unsqueeze(-1) & neighbor_valid
    same_votes = pair_valid & (anchor_labels.unsqueeze(-1) == neighbor_labels)
    diff_votes = pair_valid & (anchor_labels.unsqueeze(-1) != neighbor_labels)

    valid_view_count = pair_valid.to(torch.float32).sum(dim=0)
    min_views = max(int(min_views), 1)
    supported = valid_view_count >= min_views

    same_ratio = same_votes.to(torch.float32).sum(dim=0) / valid_view_count.clamp_min(1.0)
    diff_ratio = diff_votes.to(torch.float32).sum(dim=0) / valid_view_count.clamp_min(1.0)

    same_confidence = torch.where(
        supported,
        same_ratio * torch.sigmoid((same_ratio - conf_tau) / 0.1),
        zeros,
    )
    diff_confidence = torch.where(
        supported,
        diff_ratio * torch.sigmoid((diff_ratio - conf_tau) / 0.1),
        zeros,
    )
    stability_margin = torch.abs(same_ratio - diff_ratio)
    stability = torch.where(
        supported,
        torch.maximum(same_confidence, diff_confidence) * stability_margin,
        zeros,
    )

    return same_confidence, diff_confidence, stability, valid_view_count
