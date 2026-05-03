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


def sample_camera_boundary_flags(
    camera,
    points,
    point_ids=None,
    visibility_mask=None,
    ignore_label=-1,
    boundary_radius=1,
):
    """
    Mark projected points that fall on a 2D mask boundary band.

    A point is considered boundary-like if any sampled neighbor pixel within
    the given radius has a different valid label from the center pixel.
    """
    device = points.device
    boundary = torch.zeros((points.shape[0],), device=device, dtype=torch.bool)
    valid = torch.zeros((points.shape[0],), device=device, dtype=torch.bool)

    mask = _prepare_object_mask(camera)
    if mask is None or points.numel() == 0:
        return boundary, valid

    u, v, proj_valid = project_points_to_camera(points, camera)
    if visibility_mask is not None and point_ids is not None:
        proj_valid = proj_valid & visibility_mask[point_ids]
    if not proj_valid.any():
        return boundary, valid

    mask = mask.to(device).long()
    pixel_x = torch.round(u[proj_valid]).long().clamp_(0, camera.image_width - 1)
    pixel_y = torch.round(v[proj_valid]).long().clamp_(0, camera.image_height - 1)
    center_labels = mask[pixel_y, pixel_x]
    center_valid = torch.ones_like(center_labels, dtype=torch.bool)
    if ignore_label >= 0:
        center_valid = center_valid & (center_labels != ignore_label)

    radius = max(int(boundary_radius), 0)
    local_boundary = torch.zeros_like(center_valid)
    if radius > 0 and center_valid.any():
        offsets = [
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dy != 0 or dx != 0
        ]
        for dy, dx in offsets:
            nx = (pixel_x + dx).clamp_(0, camera.image_width - 1)
            ny = (pixel_y + dy).clamp_(0, camera.image_height - 1)
            neighbor_labels = mask[ny, nx]
            neighbor_valid = torch.ones_like(neighbor_labels, dtype=torch.bool)
            if ignore_label >= 0:
                neighbor_valid = neighbor_valid & (neighbor_labels != ignore_label)
            local_boundary = local_boundary | (
                center_valid & neighbor_valid & (neighbor_labels != center_labels)
            )

    valid_indices = proj_valid.nonzero(as_tuple=False).squeeze(-1)
    boundary[valid_indices] = local_boundary
    valid[valid_indices] = center_valid
    return boundary, valid


def collect_multiview_boundary_flags(
    cameras,
    points,
    point_ids=None,
    visibility_masks=None,
    ignore_label=-1,
    boundary_radius=1,
):
    """
    Gather per-camera 2D boundary-band flags for a shared set of 3D points.
    """
    if cameras is None or len(cameras) == 0:
        device = points.device
        empty_boundary = torch.empty((0, points.shape[0]), device=device, dtype=torch.bool)
        empty_valid = torch.empty((0, points.shape[0]), device=device, dtype=torch.bool)
        return empty_boundary, empty_valid

    boundaries = []
    valid_masks = []
    for view_idx, camera in enumerate(cameras):
        visibility_mask = None
        if visibility_masks is not None and view_idx < len(visibility_masks):
            visibility_mask = visibility_masks[view_idx]
        cam_boundary, cam_valid = sample_camera_boundary_flags(
            camera,
            points,
            point_ids=point_ids,
            visibility_mask=visibility_mask,
            ignore_label=ignore_label,
            boundary_radius=boundary_radius,
        )
        boundaries.append(cam_boundary)
        valid_masks.append(cam_valid)

    return torch.stack(boundaries, dim=0), torch.stack(valid_masks, dim=0)


def compute_point_label_distribution(view_labels, view_valid, num_classes=None, eps=1e-8):
    """
    Convert multi-view label votes into a per-point label distribution.

    Returns:
        label_distribution: [P, C]
        mv_confidence: [P], max vote ratio
        mv_entropy: [P], normalized entropy
        valid_view_count: [P]
    """
    device = view_labels.device
    num_points = view_labels.shape[1]

    if view_labels.numel() == 0 or num_points == 0:
        num_classes = max(int(num_classes or 1), 1)
        empty_dist = torch.zeros((num_points, num_classes), device=device, dtype=torch.float32)
        empty = torch.zeros((num_points,), device=device, dtype=torch.float32)
        return empty_dist, empty, empty, empty

    valid_mask = view_valid.bool()
    valid_view_count = valid_mask.to(torch.float32).sum(dim=0)
    if not valid_mask.any():
        num_classes = max(int(num_classes or 1), 1)
        empty_dist = torch.zeros((num_points, num_classes), device=device, dtype=torch.float32)
        empty = torch.zeros((num_points,), device=device, dtype=torch.float32)
        return empty_dist, empty, empty, valid_view_count

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
    label_distribution = vote_counts / valid_view_count.unsqueeze(-1).clamp_min(1.0)

    mv_confidence = label_distribution.max(dim=1).values
    entropy = -(label_distribution * torch.log(label_distribution.clamp_min(eps))).sum(dim=1)
    entropy = entropy / torch.log(label_distribution.new_tensor(float(max(num_classes, 2))))
    entropy = torch.where(valid_view_count > 0, entropy, torch.zeros_like(entropy))

    return label_distribution, mv_confidence, entropy, valid_view_count


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
