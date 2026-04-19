# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scipy.spatial import cKDTree
from utils.general_utils import build_rotation
from utils.multiview_utils import collect_multiview_labels, compute_point_label_consensus

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    """
    Compute the neighborhood consistency loss for a 3D point cloud using Top-k neighbors
    and the KL divergence.
    
    :param features: Tensor of shape (N, D), where N is the number of points and D is the dimensionality of the feature.
    :param predictions: Tensor of shape (N, C), where C is the number of classes.
    :param k: Number of neighbors to consider.
    :param lambda_val: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]


    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = kl.sum(dim=-1).mean()

    # Normalize loss into [0, 1]
    num_classes = predictions.size(1)
    normalized_loss = loss / num_classes

    return lambda_val * normalized_loss

def loss_geo_contrastive(xyz, features, k=5, lambda_val=2.0, lambda_pos=1.0, lambda_neg=3000, max_points=200000, plane_tau = 0.01, normal_tau=0.9, sample_size=800, margin=1.0, hard_neg_k=2 ,eps=1e-8):
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        xyz = xyz[indices]
        features = features[indices]

    indices = torch.randperm(features.size(0))[:sample_size]
    sample_xyz = xyz[indices]
    sample_features = features[indices]

    dists = torch.cdist(sample_xyz, xyz)
    _, neighbor_indices_tensor = dists.topk(k + 1, largest=False)
    neighbor_indices_tensor = neighbor_indices_tensor[:, 1:]

    neighbor_xyz = xyz[neighbor_indices_tensor]
    neighbor_features = features[neighbor_indices_tensor]

    local_center = neighbor_xyz.mean(dim=1, keepdim=True)
    local_xyz = neighbor_xyz - local_center

    cov = torch.matmul(local_xyz.transpose(1, 2), local_xyz) / (k + eps)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]

    rel_xyz = neighbor_xyz - sample_xyz.unsqueeze(1)
    plane_residual = torch.abs((rel_xyz * normals.unsqueeze(1)).sum(dim=-1))

    gate = (plane_residual < plane_tau).float()

    feat_dist = torch.norm(neighbor_features - sample_features.unsqueeze(1), dim=-1)

    pos_loss = (gate * (feat_dist ** 2)).sum() / (gate.sum() + eps)

    # -------------------------
    # Hard negative mining
    # -------------------------
    neg_mask = (1.0 - gate)                                    # [M, k]

    # For hard negatives:
    # among negative neighbors, smaller feat_dist => harder
    masked_neg_dist = feat_dist.clone()

    # set positive positions to large value so they won't be selected
    masked_neg_dist[neg_mask == 0] = 1e6

    # select top hard_neg_k smallest negative distances per anchor
    hard_neg_vals, hard_neg_idx = masked_neg_dist.topk(hard_neg_k, largest=False)

    # build hard negative mask
    hard_neg_mask = torch.zeros_like(neg_mask)                 # [M, k]
    hard_neg_mask.scatter_(1, hard_neg_idx, 1.0)

    # but only keep truly negative positions
    hard_neg_mask = hard_neg_mask * neg_mask

    # negative margin loss
    neg_term = torch.clamp(margin - feat_dist, min=0.0)
    neg_loss = (hard_neg_mask * (neg_term ** 2)).sum() / (hard_neg_mask.sum() + eps)

    loss = lambda_pos * pos_loss + lambda_neg * neg_loss
    
    gate_ratio = gate.mean()
    avg_plane_residual = plane_residual.mean()

    return lambda_val * loss, gate_ratio, avg_plane_residual, pos_loss, neg_loss


def loss_geo_contrastive_boundary(
    xyz,
    features,
    k=8,
    lambda_val=1.0,
    max_points=200000,
    sample_size=800,
    plane_tau=0.01,
    margin=1.0,
    alpha=2.0,
    beta=1.0,
    gamma=1.0,
    weight_power=2.0,
    eps=1e-8,
):
    zero = features.new_tensor(0.0)

    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0), device=features.device)[:max_points]
        xyz = xyz[indices]
        features = features[indices]

    num_points = features.size(0)
    if num_points < 2:
        return zero, zero, zero, zero, zero, zero, zero, zero

    sample_count = min(sample_size, num_points)
    effective_k = min(k, num_points - 1)
    if sample_count <= 0 or effective_k <= 0:
        return zero, zero, zero, zero, zero, zero, zero, zero

    indices = torch.randperm(num_points, device=features.device)[:sample_count]
    sample_xyz = xyz[indices]
    sample_features = features[indices]

    dists = torch.cdist(sample_xyz, xyz)
    _, neighbor_indices_tensor = dists.topk(effective_k + 1, largest=False)
    neighbor_indices_tensor = neighbor_indices_tensor[:, 1:]

    neighbor_xyz = xyz[neighbor_indices_tensor]
    neighbor_features = features[neighbor_indices_tensor]

    local_center = neighbor_xyz.mean(dim=1, keepdim=True)
    local_xyz = neighbor_xyz - local_center
    cov = torch.matmul(local_xyz.transpose(1, 2), local_xyz) / (effective_k + eps)

    _, eigvecs = torch.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]

    rel_xyz = neighbor_xyz - sample_xyz.unsqueeze(1)
    plane_residual = torch.abs((rel_xyz * normals.unsqueeze(1)).sum(dim=-1))
    xyz_dist = torch.norm(rel_xyz, dim=-1)
    feat_dist = torch.norm(neighbor_features - sample_features.unsqueeze(1), dim=-1)

    pos_mask = (plane_residual < plane_tau).to(feat_dist.dtype)
    pos_loss = (pos_mask * (feat_dist ** 2)).sum() / (pos_mask.sum() + eps)

    log_boundary_weight = alpha * plane_residual - beta * feat_dist - gamma * xyz_dist
    log_boundary_weight = log_boundary_weight - log_boundary_weight.max().detach()
    boundary_weight = torch.exp(log_boundary_weight)
    boundary_weight = boundary_weight / (boundary_weight.max() + eps)
    boundary_weight = boundary_weight ** weight_power

    neg_mask = 1.0 - pos_mask
    neg_weight = boundary_weight * neg_mask

    neg_term = torch.clamp(margin - feat_dist, min=0.0)
    neg_loss = (neg_weight * (neg_term ** 2)).sum() / (neg_weight.sum() + eps)

    loss = pos_loss + neg_loss
    gate_ratio = pos_mask.mean()
    avg_plane_residual = plane_residual.mean()
    avg_boundary_weight = boundary_weight.mean()
    max_boundary_weight = boundary_weight.max()
    std_boundary_weight = boundary_weight.std(unbiased=False)

    return (
        lambda_val * loss,
        gate_ratio,
        avg_plane_residual,
        pos_loss,
        neg_loss,
        avg_boundary_weight,
        max_boundary_weight,
        std_boundary_weight,
    )


def loss_sugar_surface_alignment(
    xyz,
    scaling,
    rotation,
    k=8,
    lambda_val=1.0,
    lambda_axis=1.0,
    lambda_plane=0.5,
    lambda_flat=0.1,
    max_points=200000,
    sample_size=800,
    eps=1e-8,
):
    zero = xyz.new_tensor(0.0)

    if xyz.size(0) > max_points:
        selected_indices = torch.randperm(xyz.size(0), device=xyz.device)[:max_points]
        xyz = xyz[selected_indices]
        scaling = scaling[selected_indices]
        rotation = rotation[selected_indices]

    num_points = xyz.size(0)
    if num_points < 2:
        return zero, zero, zero, zero, zero

    sample_count = min(sample_size, num_points)
    effective_k = min(k, num_points - 1)
    if sample_count <= 0 or effective_k <= 0:
        return zero, zero, zero, zero, zero

    sample_indices = torch.randperm(num_points, device=xyz.device)[:sample_count]
    detached_xyz = xyz.detach()
    sample_xyz_detached = detached_xyz[sample_indices]
    dists = torch.cdist(sample_xyz_detached, detached_xyz)
    _, neighbor_indices = dists.topk(effective_k + 1, largest=False)
    neighbor_indices = neighbor_indices[:, 1:]
    neighbor_dists = torch.take_along_dim(dists, neighbor_indices, dim=1)

    sample_xyz = xyz[sample_indices]
    neighbor_xyz = xyz[neighbor_indices]
    sample_scaling = scaling[sample_indices]
    sample_rotation = rotation[sample_indices]

    neighbor_xyz_detached = detached_xyz[neighbor_indices]
    local_center = neighbor_xyz_detached.mean(dim=1, keepdim=True)
    local_xyz = neighbor_xyz_detached - local_center
    cov = torch.matmul(local_xyz.transpose(1, 2), local_xyz) / (effective_k + eps)
    _, eigvecs = torch.linalg.eigh(cov)
    pca_normals = eigvecs[:, :, 0].detach()

    rotation_matrix = build_rotation(sample_rotation)
    min_axis_idx = torch.argmin(sample_scaling, dim=-1)
    surface_axis = torch.gather(
        rotation_matrix,
        2,
        min_axis_idx.view(-1, 1, 1).expand(-1, 3, 1),
    ).squeeze(-1)
    surface_axis = F.normalize(surface_axis, dim=-1, eps=eps)

    axis_align_cosine = torch.abs((surface_axis * pca_normals).sum(dim=-1)).clamp(0.0, 1.0)
    axis_loss = (1.0 - axis_align_cosine).mean()

    rel_xyz = neighbor_xyz - sample_xyz.unsqueeze(1)
    plane_residual = torch.abs((rel_xyz * surface_axis.unsqueeze(1)).sum(dim=-1))
    local_radius = neighbor_dists[:, -1:].clamp_min(eps)
    normalized_plane_residual = plane_residual / local_radius
    plane_loss = normalized_plane_residual.mean()

    sorted_scaling, _ = torch.sort(sample_scaling, dim=-1)
    flat_ratio = sorted_scaling[:, 0] / (0.5 * (sorted_scaling[:, 1] + sorted_scaling[:, 2]) + eps)
    flat_loss = flat_ratio.mean()

    loss = lambda_axis * axis_loss + lambda_plane * plane_loss + lambda_flat * flat_loss

    return (
        lambda_val * loss,
        axis_align_cosine.mean(),
        plane_residual.mean(),
        flat_ratio.mean(),
        loss,
    )


def _graph_zero_outputs(zero):
    return (
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    )


def compute_graph_reliability(
    xyz,
    point_ids=None,
    k=8,
    max_points=200000,
    sample_size=800,
    plane_tau=0.01,
    neg_plane_tau=None,
    spatial_pos_scale=0.75,
    normal_pos_tau=0.75,
    normal_neg_tau=0.4,
    support_cameras=None,
    support_visibility=None,
    sem_min_views=2,
    sem_conf_tau=0.7,
    sem_num_classes=None,
    sem_ignore_label=-1,
    sem_same_boost=1.0,
    sem_neg_boost=1.0,
    sem_conflict_penalty=0.75,
    alpha_dist=2.0,
    alpha_normal=2.0,
    alpha_residual=2.0,
    alpha_mv=1.0,
    pos_reliability_thresh=0.60,
    neg_reliability_thresh=0.30,
    eps=1e-8,
):
    zero = xyz.new_tensor(0.0)
    pair_dtype = xyz.dtype
    feature_indices = torch.arange(xyz.size(0), device=xyz.device)

    if xyz.size(0) > max_points:
        selected_indices = torch.randperm(xyz.size(0), device=xyz.device)[:max_points]
        xyz = xyz[selected_indices]
        feature_indices = feature_indices[selected_indices]
        if point_ids is not None:
            point_ids = point_ids[selected_indices]

    num_points = xyz.size(0)
    if num_points < 2:
        return {"valid": False, "zero": zero}

    sample_count = min(sample_size, num_points)
    effective_k = min(k, num_points - 1)
    if sample_count <= 0 or effective_k <= 0:
        return {"valid": False, "zero": zero}

    sample_indices = torch.randperm(num_points, device=xyz.device)[:sample_count]
    sample_xyz = xyz[sample_indices]

    dists = torch.cdist(sample_xyz, xyz)
    _, neighbor_indices_tensor = dists.topk(effective_k + 1, largest=False)
    neighbor_indices_tensor = neighbor_indices_tensor[:, 1:]
    neighbor_dists = torch.take_along_dim(dists, neighbor_indices_tensor, dim=1)
    neighbor_xyz = xyz[neighbor_indices_tensor]

    flat_point_ids = torch.cat([sample_indices, neighbor_indices_tensor.reshape(-1)], dim=0)
    unique_point_ids, inverse_ids = torch.unique(flat_point_ids, sorted=False, return_inverse=True)
    unique_xyz = xyz[unique_point_ids]

    unique_dists = torch.cdist(unique_xyz, xyz)
    _, unique_neighbor_indices = unique_dists.topk(effective_k + 1, largest=False)
    unique_neighbor_indices = unique_neighbor_indices[:, 1:]
    unique_neighbor_xyz = xyz[unique_neighbor_indices]
    unique_local_center = unique_neighbor_xyz.mean(dim=1, keepdim=True)
    unique_local_xyz = unique_neighbor_xyz - unique_local_center
    unique_cov = torch.matmul(unique_local_xyz.transpose(1, 2), unique_local_xyz) / (effective_k + eps)

    _, unique_eigvecs = torch.linalg.eigh(unique_cov)
    unique_normals = unique_eigvecs[:, :, 0]
    sample_local = inverse_ids[:sample_count]
    neighbor_local = inverse_ids[sample_count:].view(sample_count, effective_k)
    sample_normals = unique_normals[sample_local]
    neighbor_normals = unique_normals[neighbor_local.reshape(-1)].view(sample_count, effective_k, 3)

    rel_xyz = neighbor_xyz - sample_xyz.unsqueeze(1)
    xyz_dist = neighbor_dists
    local_radius = xyz_dist[:, -1:].clamp_min(eps)
    spatial_ratio = xyz_dist / local_radius
    plane_residual = torch.abs((rel_xyz * sample_normals.unsqueeze(1)).sum(dim=-1))
    plane_residual_ratio = plane_residual / local_radius
    normal_cosine = torch.abs((sample_normals.unsqueeze(1) * neighbor_normals).sum(dim=-1)).clamp(0.0, 1.0)

    spatial_pos_mask = (spatial_ratio <= spatial_pos_scale).to(pair_dtype)
    # Use normalized residuals for graph partitioning so the same threshold
    # behaves more consistently across scenes with different local scales.
    depth_pos_mask = (plane_residual_ratio < plane_tau).to(pair_dtype)
    normal_pos_mask = (normal_cosine >= normal_pos_tau).to(pair_dtype)
    geometry_positive_support = spatial_pos_mask * depth_pos_mask * normal_pos_mask

    if neg_plane_tau is None or neg_plane_tau <= plane_tau:
        depth_neg_mask = 1.0 - depth_pos_mask
    else:
        depth_neg_mask = (plane_residual_ratio > neg_plane_tau).to(pair_dtype)
    normal_neg_mask = (normal_cosine <= normal_neg_tau).to(pair_dtype)
    geometry_break_mask = ((depth_neg_mask + normal_neg_mask) > 0).to(pair_dtype)

    avg_sem_valid_views = zero
    avg_sem_confidence = zero
    semantic_pos_keep_ratio = zero
    semantic_neg_keep_ratio = zero
    sem_same_mask = torch.zeros_like(spatial_ratio)
    sem_diff_mask = torch.zeros_like(spatial_ratio)
    sem_pair_conf = torch.zeros_like(spatial_ratio)
    mv_consistency = torch.zeros_like(spatial_ratio)
    semantic_pos_factor = torch.ones_like(spatial_ratio)
    semantic_neg_factor = torch.ones_like(spatial_ratio)
    sem_pair_valid = torch.zeros_like(spatial_ratio, dtype=torch.bool)

    if support_cameras is not None and len(support_cameras) > 0:
        unique_global_point_ids = unique_point_ids if point_ids is None else point_ids[unique_point_ids]
        view_labels, view_valid = collect_multiview_labels(
            support_cameras,
            unique_xyz,
            point_ids=unique_global_point_ids,
            visibility_masks=support_visibility,
            ignore_label=sem_ignore_label,
        )

        if view_labels.shape[0] > 0:
            point_sem_label, point_sem_valid, point_sem_confidence, point_valid_view_count = compute_point_label_consensus(
                view_labels,
                view_valid,
                num_classes=sem_num_classes,
                min_views=sem_min_views,
                conf_tau=sem_conf_tau,
            )

            neighbor_local_flat = neighbor_local.reshape(-1)
            sample_labels = point_sem_label[sample_local]
            sample_valid = point_sem_valid[sample_local]
            sample_confidence = point_sem_confidence[sample_local]
            neighbor_labels = point_sem_label[neighbor_local_flat].view(sample_count, effective_k)
            neighbor_valid = point_sem_valid[neighbor_local_flat].view(sample_count, effective_k)
            neighbor_confidence = point_sem_confidence[neighbor_local_flat].view(sample_count, effective_k)

            sem_pair_valid = sample_valid.unsqueeze(-1) & neighbor_valid
            sem_same_mask = (sem_pair_valid & (sample_labels.unsqueeze(-1) == neighbor_labels)).to(pair_dtype)
            sem_diff_mask = (sem_pair_valid & (sample_labels.unsqueeze(-1) != neighbor_labels)).to(pair_dtype)
            sem_pair_conf = sample_confidence.unsqueeze(-1) * neighbor_confidence
            mv_consistency = sem_same_mask * sem_pair_conf - sem_diff_mask * sem_pair_conf

            sem_valid_point_count = point_sem_valid.to(pair_dtype).sum()
            avg_sem_valid_views = (point_sem_valid.to(point_valid_view_count.dtype) * point_valid_view_count).sum() / (sem_valid_point_count + eps)
            avg_sem_confidence = (point_sem_valid.to(point_sem_confidence.dtype) * point_sem_confidence).sum() / (sem_valid_point_count + eps)

            semantic_pos_factor = 1.0 + sem_same_boost * sem_same_mask * sem_pair_conf
            semantic_pos_factor = semantic_pos_factor * torch.clamp(
                1.0 - sem_conflict_penalty * sem_diff_mask * sem_pair_conf,
                min=0.25,
                max=2.0,
            )
            semantic_neg_factor = 1.0 + sem_neg_boost * sem_diff_mask * sem_pair_conf

    # Geometry does not assign semantics directly here; it only modulates
    # whether semantic propagation between two Gaussians is trustworthy.
    reliability_logit = (
        -alpha_dist * spatial_ratio
        + alpha_normal * normal_cosine
        - alpha_residual * plane_residual_ratio
        + alpha_mv * mv_consistency
    )
    reliability = torch.sigmoid(reliability_logit)

    sem_known_mask = sem_pair_valid.to(pair_dtype)
    semantic_agree_or_unknown = ((sem_same_mask > 0) | (~sem_pair_valid)).to(pair_dtype)
    reliability_high_mask = (reliability >= pos_reliability_thresh).to(pair_dtype)
    reliability_low_mask = (reliability <= neg_reliability_thresh).to(pair_dtype)

    positive_mask = reliability_high_mask * geometry_positive_support * semantic_agree_or_unknown

    spatial_close_mask = (spatial_ratio <= 1.0).to(pair_dtype)
    semantic_negative_promote_mask = spatial_close_mask * sem_diff_mask * (sem_pair_conf >= sem_conf_tau).to(pair_dtype)
    negative_mask = spatial_close_mask * (
        reliability_low_mask * geometry_break_mask + semantic_negative_promote_mask
    )
    negative_mask = torch.clamp(negative_mask * (1.0 - positive_mask), min=0.0, max=1.0)
    ignore_mask = torch.clamp(1.0 - torch.clamp(positive_mask + negative_mask, max=1.0), min=0.0, max=1.0)

    pos_reference = geometry_positive_support.sum()
    neg_reference = (spatial_close_mask * geometry_break_mask + semantic_negative_promote_mask).sum()
    semantic_pos_keep_ratio = (geometry_positive_support * semantic_pos_factor).sum() / (pos_reference + eps)
    semantic_neg_keep_ratio = (negative_mask * semantic_neg_factor).sum() / (neg_reference + eps)

    return {
        "valid": True,
        "zero": zero,
        "xyz": xyz,
        "feature_indices": feature_indices,
        "sample_indices": sample_indices,
        "neighbor_indices": neighbor_indices_tensor,
        "reliability": reliability,
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "ignore_mask": ignore_mask,
        "spatial_ratio": spatial_ratio,
        "plane_residual": plane_residual,
        "plane_residual_ratio": plane_residual_ratio,
        "normal_cosine": normal_cosine,
        "mv_consistency": mv_consistency,
        "sem_known_mask": sem_known_mask,
        "sem_same_mask": sem_same_mask,
        "sem_diff_mask": sem_diff_mask,
        "sem_pair_conf": sem_pair_conf,
        "semantic_pos_factor": semantic_pos_factor,
        "semantic_neg_factor": semantic_neg_factor,
        "avg_sem_valid_views": avg_sem_valid_views,
        "avg_sem_confidence": avg_sem_confidence,
        "semantic_pos_keep_ratio": semantic_pos_keep_ratio,
        "semantic_neg_keep_ratio": semantic_neg_keep_ratio,
    }


def loss_graph_contrastive(
    features,
    graph_data,
    lambda_val=1.0,
    lambda_pos=1.0,
    lambda_neg=1.0,
    neg_margin=0.2,
    hard_neg_k=2,
    normal_weight_lambda=5.0,
    eps=1e-8,
):
    zero = features.new_tensor(0.0)
    if graph_data is None or not graph_data.get("valid", False):
        return _graph_zero_outputs(zero)

    feature_indices = graph_data["feature_indices"]
    selected_features = features[feature_indices]
    raw_feature_norm = torch.norm(selected_features, dim=-1)
    normalized_features = F.normalize(selected_features, dim=-1, eps=eps)

    sample_features = normalized_features[graph_data["sample_indices"]]
    neighbor_features = normalized_features[graph_data["neighbor_indices"]]
    cosine_sim = (neighbor_features * sample_features.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)

    reliability = graph_data["reliability"]
    positive_mask = graph_data["positive_mask"]
    negative_mask = graph_data["negative_mask"]
    ignore_mask = graph_data["ignore_mask"]
    normal_cosine = graph_data["normal_cosine"]
    spatial_ratio = graph_data["spatial_ratio"]
    mv_consistency = graph_data["mv_consistency"]
    semantic_pos_factor = graph_data["semantic_pos_factor"]
    semantic_neg_factor = graph_data["semantic_neg_factor"]

    if normal_weight_lambda > 0:
        normal_weight = torch.exp(-normal_weight_lambda * (1.0 - normal_cosine))
    else:
        normal_weight = torch.ones_like(normal_cosine)
    spatial_weight = torch.exp(-spatial_ratio)

    pos_pair_weight = positive_mask * reliability * spatial_weight * normal_weight * semantic_pos_factor
    pos_loss = (pos_pair_weight * (1.0 - cosine_sim)).sum() / (pos_pair_weight.sum() + eps)

    effective_k = negative_mask.shape[1]
    effective_hard_neg_k = min(max(int(hard_neg_k), 0), effective_k)
    if effective_hard_neg_k > 0:
        hard_neg_scores = (
            cosine_sim
            + 0.5 * (1.0 - normal_cosine)
            + graph_data["sem_diff_mask"] * graph_data["sem_pair_conf"]
        ).masked_fill(negative_mask == 0, -1e6)
        _, hard_neg_idx = hard_neg_scores.topk(effective_hard_neg_k, dim=1, largest=True)
        hard_neg_mask = torch.zeros_like(negative_mask)
        hard_neg_mask.scatter_(1, hard_neg_idx, 1.0)
        hard_neg_mask = hard_neg_mask * negative_mask
    else:
        hard_neg_mask = negative_mask

    neg_term = torch.clamp(cosine_sim - neg_margin, min=0.0)
    active_hard_neg_mask = hard_neg_mask * (cosine_sim > neg_margin).to(cosine_sim.dtype)
    neg_pair_weight = active_hard_neg_mask * (1.0 - reliability) * semantic_neg_factor
    neg_loss = (neg_pair_weight * (neg_term ** 2)).sum() / (neg_pair_weight.sum() + eps)

    loss = lambda_pos * pos_loss + lambda_neg * neg_loss

    pos_count = positive_mask.sum()
    neg_count = negative_mask.sum()
    ignore_count = ignore_mask.sum()
    hard_neg_count = hard_neg_mask.sum()
    active_hard_neg_count = active_hard_neg_mask.sum()
    pair_count = positive_mask.numel()

    avg_reliability = reliability.mean()
    avg_pos_reliability = (positive_mask * reliability).sum() / (pos_count + eps)
    avg_neg_reliability = (negative_mask * reliability).sum() / (neg_count + eps)
    avg_mv_consistency = mv_consistency.mean()
    avg_plane_residual = graph_data["plane_residual"].mean()
    avg_feature_norm = raw_feature_norm.mean()
    avg_normal_cosine = (positive_mask * normal_cosine).sum() / (pos_count + eps)
    avg_pos_cosine = (positive_mask * cosine_sim).sum() / (pos_count + eps)
    avg_neg_cosine = (negative_mask * cosine_sim).sum() / (neg_count + eps)
    avg_hard_neg_cosine = (hard_neg_mask * cosine_sim).sum() / (hard_neg_count + eps)
    active_neg_ratio = active_hard_neg_count / (hard_neg_count + eps)

    return (
        lambda_val * loss,
        pos_count / (pair_count + eps),
        neg_count / (pair_count + eps),
        ignore_count / (pair_count + eps),
        avg_reliability,
        avg_pos_reliability,
        avg_neg_reliability,
        avg_mv_consistency,
        avg_plane_residual,
        pos_loss,
        neg_loss,
        avg_feature_norm,
        avg_normal_cosine,
        avg_pos_cosine,
        avg_neg_cosine,
        avg_hard_neg_cosine,
        active_neg_ratio,
        graph_data["avg_sem_valid_views"],
        graph_data["avg_sem_confidence"],
        graph_data["semantic_pos_keep_ratio"],
        graph_data["semantic_neg_keep_ratio"],
    )


def loss_geo_contrastive_cosine(
    xyz,
    features,
    point_ids=None,
    k=8,
    lambda_val=1.0,
    lambda_pos=1.0,
    lambda_neg=1.0,
    max_points=200000,
    sample_size=800,
    plane_tau=0.01,
    neg_plane_tau=None,
    spatial_pos_scale=0.75,
    normal_pos_tau=0.75,
    normal_neg_tau=0.4,
    neg_margin=0.2,
    hard_neg_k=2,
    support_cameras=None,
    support_visibility=None,
    sem_min_views=2,
    sem_conf_tau=0.7,
    sem_num_classes=None,
    sem_ignore_label=-1,
    normal_weight_lambda=5.0,
    sem_same_boost=1.0,
    sem_neg_boost=1.0,
    sem_conflict_penalty=0.75,
    alpha_dist=2.0,
    alpha_normal=2.0,
    alpha_residual=2.0,
    alpha_mv=1.0,
    pos_reliability_thresh=0.65,
    neg_reliability_thresh=0.35,
    eps=1e-8,
):
    graph_data = compute_graph_reliability(
        xyz=xyz,
        point_ids=point_ids,
        k=k,
        max_points=max_points,
        sample_size=sample_size,
        plane_tau=plane_tau,
        neg_plane_tau=neg_plane_tau,
        spatial_pos_scale=spatial_pos_scale,
        normal_pos_tau=normal_pos_tau,
        normal_neg_tau=normal_neg_tau,
        support_cameras=support_cameras,
        support_visibility=support_visibility,
        sem_min_views=sem_min_views,
        sem_conf_tau=sem_conf_tau,
        sem_num_classes=sem_num_classes,
        sem_ignore_label=sem_ignore_label,
        sem_same_boost=sem_same_boost,
        sem_neg_boost=sem_neg_boost,
        sem_conflict_penalty=sem_conflict_penalty,
        alpha_dist=alpha_dist,
        alpha_normal=alpha_normal,
        alpha_residual=alpha_residual,
        alpha_mv=alpha_mv,
        pos_reliability_thresh=pos_reliability_thresh,
        neg_reliability_thresh=neg_reliability_thresh,
        eps=eps,
    )
    return loss_graph_contrastive(
        features=features,
        graph_data=graph_data,
        lambda_val=lambda_val,
        lambda_pos=lambda_pos,
        lambda_neg=lambda_neg,
        neg_margin=neg_margin,
        hard_neg_k=hard_neg_k,
        normal_weight_lambda=normal_weight_lambda,
        eps=eps,
    )
    
