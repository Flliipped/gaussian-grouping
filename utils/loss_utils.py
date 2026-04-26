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
from math import exp, log
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
    sem_pos_ratio=0.7,
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
    sem_valid_point_ratio = zero
    sem_pair_valid_ratio = zero
    sem_high_conf_same_ratio = zero
    semantic_pos_keep_ratio = zero
    semantic_neg_keep_ratio = zero
    sem_same_mask = torch.zeros_like(spatial_ratio)
    sem_diff_mask = torch.zeros_like(spatial_ratio)
    sem_pair_conf = torch.zeros_like(spatial_ratio)
    mv_consistency = torch.zeros_like(spatial_ratio)
    semantic_pos_factor = torch.ones_like(spatial_ratio)
    semantic_neg_factor = torch.ones_like(spatial_ratio)
    sem_pair_valid = torch.zeros_like(spatial_ratio, dtype=torch.bool)
    point_sem_confidence = torch.zeros((unique_point_ids.shape[0],), device=xyz.device, dtype=pair_dtype)
    point_sem_valid = torch.zeros((unique_point_ids.shape[0],), device=xyz.device, dtype=torch.bool)

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
            sem_valid_point_ratio = sem_valid_point_count / (point_sem_valid.numel() + eps)
            sem_pair_valid_ratio = sem_pair_valid.to(pair_dtype).mean()

            sem_high_conf_same_mask = sem_same_mask * (sem_pair_conf >= sem_conf_tau).to(pair_dtype)
            sem_high_conf_same_ratio = sem_high_conf_same_mask.sum() / (sem_high_conf_same_mask.numel() + eps)
            semantic_pos_factor = 1.0 + sem_pos_ratio * sem_same_boost * sem_high_conf_same_mask * sem_pair_conf

    # Geometry determines whether propagation is trustworthy. Multi-view
    # semantics are kept as a conservative reweighting signal instead of
    # rewriting the graph topology or mining negatives.
    reliability_logit = (
        -alpha_dist * spatial_ratio
        + alpha_normal * normal_cosine
        - alpha_residual * plane_residual_ratio
    )
    reliability = torch.sigmoid(reliability_logit)

    point_reliability_sum = torch.zeros((unique_point_ids.shape[0],), device=xyz.device, dtype=pair_dtype)
    point_reliability_count = torch.zeros((unique_point_ids.shape[0],), device=xyz.device, dtype=pair_dtype)
    sample_reliability = reliability.mean(dim=1)
    point_reliability_sum.scatter_add_(0, sample_local, sample_reliability)
    point_reliability_count.scatter_add_(0, sample_local, torch.ones_like(sample_reliability))
    point_reliability_sum.scatter_add_(0, neighbor_local.reshape(-1), reliability.reshape(-1))
    point_reliability_count.scatter_add_(0, neighbor_local.reshape(-1), torch.ones_like(reliability.reshape(-1)))
    point_reliability = point_reliability_sum / point_reliability_count.clamp_min(1.0)

    sem_known_mask = sem_pair_valid.to(pair_dtype)
    reliability_high_mask = (reliability >= pos_reliability_thresh).to(pair_dtype)
    reliability_low_mask = (reliability <= neg_reliability_thresh).to(pair_dtype)

    positive_mask = reliability_high_mask * geometry_positive_support

    spatial_close_mask = (spatial_ratio <= 1.0).to(pair_dtype)
    negative_mask = spatial_close_mask * (reliability_low_mask * geometry_break_mask)
    negative_mask = torch.clamp(negative_mask * (1.0 - positive_mask), min=0.0, max=1.0)
    ignore_mask = torch.clamp(1.0 - torch.clamp(positive_mask + negative_mask, max=1.0), min=0.0, max=1.0)

    pos_reference = geometry_positive_support.sum()
    neg_reference = (spatial_close_mask * geometry_break_mask).sum()
    semantic_pos_keep_ratio = (geometry_positive_support * semantic_pos_factor).sum() / (pos_reference + eps)
    semantic_neg_keep_ratio = zero

    return {
        "valid": True,
        "zero": zero,
        "xyz": xyz,
        "feature_indices": feature_indices,
        "proto_unique_indices": unique_point_ids,
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
        "point_reliability": point_reliability,
        "point_sem_confidence": point_sem_confidence,
        "point_sem_valid": point_sem_valid,
        "sem_valid_point_ratio": sem_valid_point_ratio,
        "sem_pair_valid_ratio": sem_pair_valid_ratio,
        "sem_high_conf_same_ratio": sem_high_conf_same_ratio,
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
        ).masked_fill(negative_mask == 0, -1e6)
        _, hard_neg_idx = hard_neg_scores.topk(effective_hard_neg_k, dim=1, largest=True)
        hard_neg_mask = torch.zeros_like(negative_mask)
        hard_neg_mask.scatter_(1, hard_neg_idx, 1.0)
        hard_neg_mask = hard_neg_mask * negative_mask
    else:
        hard_neg_mask = negative_mask

    neg_term = torch.clamp(cosine_sim - neg_margin, min=0.0)
    active_hard_neg_mask = hard_neg_mask * (cosine_sim > neg_margin).to(cosine_sim.dtype)
    neg_pair_weight = active_hard_neg_mask * (1.0 - reliability)
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


def _prototype_zero_diagnostics(zero, num_prototypes=0):
    with torch.no_grad():
        zero = zero.detach()
        histogram = zero.new_zeros((max(int(num_prototypes), 0),))
        return {
            "proto_usage_histogram": histogram,
            "proto_update_usage_histogram": histogram.clone(),
            "proto_dead_count": zero,
            "proto_active_update_count": zero,
            "proto_usage_entropy": zero,
            "proto_update_usage_entropy": zero,
            "proto_usage_min": zero,
            "proto_usage_max": zero,
            "proto_usage_std": zero,
            "proto_update_usage_min": zero,
            "proto_update_usage_max": zero,
            "proto_update_usage_std": zero,
            "proto_pair_cosine_mean": zero,
            "proto_pair_cosine_max": zero,
            "proto_pair_cosine_p90": zero,
            "proto_entropy_p10": zero,
            "proto_entropy_p50": zero,
            "proto_entropy_p90": zero,
            "proto_assign_conf_p10": zero,
            "proto_assign_conf_p50": zero,
            "proto_assign_conf_p90": zero,
            "proto_margin_p10": zero,
            "proto_margin_p50": zero,
            "proto_margin_p90": zero,
            "proto_update_selected_count": zero,
            "proto_update_selected_ratio": zero,
            "proto_update_confidence_p50": zero,
            "proto_update_confidence_p90": zero,
            "proto_neg_boundary_entropy": zero,
            "proto_neg_boundary_assign_conf": zero,
            "proto_neg_boundary_selected_ratio": zero,
            "proto_uncertain_boundary_entropy": zero,
            "proto_uncertain_boundary_assign_conf": zero,
            "proto_uncertain_boundary_selected_ratio": zero,
        }


def _safe_quantiles(values, zero):
    values = values.detach().reshape(-1)
    if values.numel() == 0:
        return zero, zero, zero

    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return zero, zero, zero

    sorted_values, _ = torch.sort(finite_values)
    count = sorted_values.numel()
    quantile_positions = sorted_values.new_tensor([0.1, 0.5, 0.9]) * (count - 1)
    lower_indices = quantile_positions.floor().long().clamp(min=0, max=count - 1)
    upper_indices = quantile_positions.ceil().long().clamp(min=0, max=count - 1)
    weights = quantile_positions - lower_indices.to(dtype=quantile_positions.dtype)
    quantiles = (
        sorted_values[lower_indices] * (1.0 - weights)
        + sorted_values[upper_indices] * weights
    ).to(dtype=zero.dtype)
    return quantiles[0], quantiles[1], quantiles[2]


def _normalized_hist_entropy(histogram, zero, eps=1e-8):
    histogram = histogram.detach().reshape(-1)
    if histogram.numel() == 0:
        return zero

    positive_hist = histogram[histogram > 0]
    if positive_hist.numel() == 0:
        return zero

    entropy = -(positive_hist * torch.log(positive_hist.clamp_min(eps))).sum()
    entropy = entropy / max(log(max(histogram.numel(), 2)), eps)
    return entropy.to(dtype=zero.dtype).detach()


def _usage_stats(histogram, zero):
    histogram = histogram.detach().reshape(-1)
    if histogram.numel() == 0:
        return zero, zero, zero

    return (
        histogram.min().to(dtype=zero.dtype).detach(),
        histogram.max().to(dtype=zero.dtype).detach(),
        histogram.std(unbiased=False).to(dtype=zero.dtype).detach(),
    )


def _boundary_unique_mask(graph_data, unique_indices, num_selected_points, edge_mask):
    # Geometry-boundary proxy only: this is not a GT object boundary.
    edge_mask = edge_mask.detach() > 0
    if edge_mask.numel() == 0 or not edge_mask.any():
        return torch.zeros(
            (unique_indices.shape[0],),
            device=unique_indices.device,
            dtype=torch.bool,
        )

    sample_indices = graph_data["sample_indices"].detach()
    neighbor_indices = graph_data["neighbor_indices"].detach()
    active_sample_indices = sample_indices.unsqueeze(-1).expand_as(neighbor_indices)[edge_mask]
    active_neighbor_indices = neighbor_indices[edge_mask]
    active_point_indices = torch.cat([active_sample_indices, active_neighbor_indices], dim=0)

    point_mask = torch.zeros(
        (num_selected_points,),
        device=unique_indices.device,
        dtype=torch.bool,
    )
    point_mask[active_point_indices] = True
    return point_mask[unique_indices.detach()]


def _boundary_proto_stats(boundary_mask, entropy, assign_conf, selected_mask, zero, eps=1e-8):
    boundary_mask = boundary_mask.detach().bool()
    if boundary_mask.numel() == 0 or not boundary_mask.any():
        return zero, zero, zero

    boundary_count = boundary_mask.to(dtype=zero.dtype).sum().clamp_min(eps)
    boundary_entropy = entropy.detach()[boundary_mask].mean().to(dtype=zero.dtype).detach()
    boundary_assign_conf = assign_conf.detach()[boundary_mask].mean().to(dtype=zero.dtype).detach()
    boundary_selected_ratio = (
        (selected_mask.detach().bool() & boundary_mask).to(dtype=zero.dtype).sum()
        / boundary_count
    ).detach()
    return boundary_entropy, boundary_assign_conf, boundary_selected_ratio


def _prototype_diagnostics(
    prototype_bank,
    graph_data,
    unique_indices,
    num_selected_points,
    unique_proto_idx,
    unique_entropy,
    unique_assign_conf,
    unique_assign_margin,
    update_confidence,
    update_confident_mask,
    zero,
    eps=1e-8,
):
    with torch.no_grad():
        num_prototypes = prototype_bank.num_prototypes
        diagnostics = _prototype_zero_diagnostics(zero, num_prototypes)

        proto_idx = unique_proto_idx.detach().long().reshape(-1)
        update_selected = update_confident_mask.detach().bool().reshape(-1)
        if proto_idx.numel() == 0:
            return diagnostics

        usage_counts = torch.bincount(proto_idx, minlength=num_prototypes).to(dtype=zero.dtype)
        usage_histogram = usage_counts / usage_counts.sum().clamp_min(1.0)
        update_proto_idx = proto_idx[update_selected]
        update_usage_counts = torch.bincount(
            update_proto_idx,
            minlength=num_prototypes,
        ).to(dtype=zero.dtype)
        update_usage_histogram = update_usage_counts / update_usage_counts.sum().clamp_min(1.0)

        usage_min, usage_max, usage_std = _usage_stats(usage_histogram, zero)
        update_usage_min, update_usage_max, update_usage_std = _usage_stats(update_usage_histogram, zero)

        entropy_p10, entropy_p50, entropy_p90 = _safe_quantiles(unique_entropy, zero)
        assign_conf_p10, assign_conf_p50, assign_conf_p90 = _safe_quantiles(unique_assign_conf, zero)
        margin_p10, margin_p50, margin_p90 = _safe_quantiles(unique_assign_margin, zero)

        selected_update_confidence = update_confidence.detach().reshape(-1)[update_selected]
        _, update_confidence_p50, update_confidence_p90 = _safe_quantiles(selected_update_confidence, zero)

        pair_cosine_mean = zero
        pair_cosine_max = zero
        pair_cosine_p90 = zero
        if num_prototypes >= 2:
            prototypes = F.normalize(prototype_bank.prototypes.detach(), dim=-1, eps=eps)
            prototype_cosine = torch.matmul(prototypes, prototypes.t())
            off_diag_mask = ~torch.eye(
                num_prototypes,
                device=prototype_cosine.device,
                dtype=torch.bool,
            )
            pair_cosine_values = prototype_cosine[off_diag_mask]
            finite_pair_cosine = pair_cosine_values[torch.isfinite(pair_cosine_values)]
            if finite_pair_cosine.numel() > 0:
                pair_cosine_mean = finite_pair_cosine.mean().to(dtype=zero.dtype).detach()
                pair_cosine_max = finite_pair_cosine.max().to(dtype=zero.dtype).detach()
                _, _, pair_cosine_p90 = _safe_quantiles(finite_pair_cosine, zero)

        neg_edge_mask = graph_data["negative_mask"].detach() > 0
        uncertain_edge_mask = neg_edge_mask | (graph_data["ignore_mask"].detach() > 0)
        neg_boundary_mask = _boundary_unique_mask(
            graph_data,
            unique_indices,
            num_selected_points,
            neg_edge_mask,
        )
        uncertain_boundary_mask = _boundary_unique_mask(
            graph_data,
            unique_indices,
            num_selected_points,
            uncertain_edge_mask,
        )
        (
            neg_boundary_entropy,
            neg_boundary_assign_conf,
            neg_boundary_selected_ratio,
        ) = _boundary_proto_stats(
            neg_boundary_mask,
            unique_entropy,
            unique_assign_conf,
            update_selected,
            zero,
            eps=eps,
        )
        (
            uncertain_boundary_entropy,
            uncertain_boundary_assign_conf,
            uncertain_boundary_selected_ratio,
        ) = _boundary_proto_stats(
            uncertain_boundary_mask,
            unique_entropy,
            unique_assign_conf,
            update_selected,
            zero,
            eps=eps,
        )

        diagnostics.update({
            "proto_usage_histogram": usage_histogram.detach(),
            "proto_update_usage_histogram": update_usage_histogram.detach(),
            "proto_dead_count": (usage_counts == 0).to(dtype=zero.dtype).sum().detach(),
            "proto_active_update_count": (update_usage_counts > 0).to(dtype=zero.dtype).sum().detach(),
            "proto_usage_entropy": _normalized_hist_entropy(usage_histogram, zero, eps=eps),
            "proto_update_usage_entropy": _normalized_hist_entropy(update_usage_histogram, zero, eps=eps),
            "proto_usage_min": usage_min,
            "proto_usage_max": usage_max,
            "proto_usage_std": usage_std,
            "proto_update_usage_min": update_usage_min,
            "proto_update_usage_max": update_usage_max,
            "proto_update_usage_std": update_usage_std,
            "proto_pair_cosine_mean": pair_cosine_mean,
            "proto_pair_cosine_max": pair_cosine_max,
            "proto_pair_cosine_p90": pair_cosine_p90,
            "proto_entropy_p10": entropy_p10,
            "proto_entropy_p50": entropy_p50,
            "proto_entropy_p90": entropy_p90,
            "proto_assign_conf_p10": assign_conf_p10,
            "proto_assign_conf_p50": assign_conf_p50,
            "proto_assign_conf_p90": assign_conf_p90,
            "proto_margin_p10": margin_p10,
            "proto_margin_p50": margin_p50,
            "proto_margin_p90": margin_p90,
            "proto_update_selected_count": update_selected.to(dtype=zero.dtype).sum().detach(),
            "proto_update_selected_ratio": update_selected.to(dtype=zero.dtype).mean().detach(),
            "proto_update_confidence_p50": update_confidence_p50,
            "proto_update_confidence_p90": update_confidence_p90,
            "proto_neg_boundary_entropy": neg_boundary_entropy,
            "proto_neg_boundary_assign_conf": neg_boundary_assign_conf,
            "proto_neg_boundary_selected_ratio": neg_boundary_selected_ratio,
            "proto_uncertain_boundary_entropy": uncertain_boundary_entropy,
            "proto_uncertain_boundary_assign_conf": uncertain_boundary_assign_conf,
            "proto_uncertain_boundary_selected_ratio": uncertain_boundary_selected_ratio,
        })
        return diagnostics


def loss_prototype_learning(
    features,
    prototype_bank,
    graph_data,
    lambda_val=1.0,
    lambda_pull=1.0,
    lambda_sep=0.1,
    lambda_cons=0.1,
    cons_conf_weight=0.0,
    cons_conf_floor=0.25,
    cons_conf_power=1.0,
    cons_conf_normalize=True,
    cons_conf_norm_max=2.0,
    cons_scene_weight=0.0,
    cons_scene_floor=0.5,
    cons_scene_conf_min=0.05,
    cons_scene_conf_target=0.15,
    cons_agree_weight=0.0,
    cons_agree_floor=0.4,
    cons_agree_conf_thresh=0.0,
    conf_thresh=0.2,
    sep_margin=0.2,
    reliability_thresh=0.0,
    entropy_thresh=1.0,
    assign_conf_thresh=0.0,
    sem_invalid_weight=1.0,
    update_conf_thresh=None,
    update_reliability_thresh=None,
    update_entropy_thresh=None,
    update_assign_conf_thresh=None,
    update_sem_invalid_weight=None,
    eps=1e-8,
):
    zero = features.new_tensor(0.0)
    if (
        prototype_bank is None
        or graph_data is None
        or not graph_data.get("valid", False)
        or features.numel() == 0
    ):
        num_prototypes = prototype_bank.num_prototypes if prototype_bank is not None else 0
        return {
            "loss": zero,
            "pull_loss": zero,
            "sep_loss": zero,
            "cons_loss": zero,
            "cons_conf_mean": zero,
            "cons_adaptive_factor_mean": zero,
            "cons_scene_scale": zero,
            "cons_agree_ratio": zero,
            "cons_agree_factor_mean": zero,
            "avg_entropy": zero,
            "avg_assign_conf": zero,
            "avg_proto_confidence": zero,
            "confident_ratio": zero,
            "update_avg_confidence": zero,
            "update_confident_ratio": zero,
            "avg_proto_margin": zero,
            "active_proto_ratio": zero,
            "usage_max": zero,
            **_prototype_zero_diagnostics(zero, num_prototypes),
            "update_features": None,
            "update_probs": None,
            "update_confidence": None,
        }

    if update_conf_thresh is None:
        update_conf_thresh = conf_thresh
    if update_reliability_thresh is None:
        update_reliability_thresh = reliability_thresh
    if update_entropy_thresh is None:
        update_entropy_thresh = entropy_thresh
    if update_assign_conf_thresh is None:
        update_assign_conf_thresh = assign_conf_thresh
    if update_sem_invalid_weight is None:
        update_sem_invalid_weight = sem_invalid_weight

    feature_indices = graph_data["feature_indices"]
    selected_features = F.normalize(features[feature_indices], dim=-1, eps=eps)
    _, assignment_probs = prototype_bank.assign(selected_features)

    entropy = -(assignment_probs * torch.log(assignment_probs.clamp_min(eps))).sum(dim=-1)
    entropy = entropy / max(log(max(assignment_probs.shape[-1], 2)), eps)
    topk = assignment_probs.topk(k=min(2, assignment_probs.shape[-1]), dim=-1)
    assign_conf = topk.values[:, 0]
    if topk.values.shape[-1] > 1:
        assign_margin = topk.values[:, 0] - topk.values[:, 1]
    else:
        assign_margin = topk.values[:, 0]
    assigned_proto_idx = topk.indices[:, 0]

    unique_indices = graph_data["proto_unique_indices"]
    unique_features = selected_features[unique_indices]
    unique_probs = assignment_probs[unique_indices]
    unique_entropy = entropy[unique_indices]
    unique_assign_conf = assign_conf[unique_indices]
    unique_assign_margin = assign_margin[unique_indices]
    point_reliability = graph_data["point_reliability"].to(unique_features.dtype)
    point_sem_confidence = graph_data["point_sem_confidence"].to(unique_features.dtype)
    point_sem_valid = graph_data["point_sem_valid"]
    pull_sem_gate = torch.where(
        point_sem_valid,
        point_sem_confidence,
        torch.full_like(point_sem_confidence, sem_invalid_weight),
    )
    pull_mask = (
        (point_reliability >= reliability_thresh)
        & (unique_entropy <= entropy_thresh)
        & (unique_assign_conf >= assign_conf_thresh)
    ).to(unique_features.dtype)
    update_sem_gate = torch.where(
        point_sem_valid,
        point_sem_confidence,
        torch.full_like(point_sem_confidence, update_sem_invalid_weight),
    )
    update_mask = (
        (point_reliability >= update_reliability_thresh)
        & (unique_entropy <= update_entropy_thresh)
        & (unique_assign_conf >= update_assign_conf_thresh)
    ).to(unique_features.dtype)

    pull_confidence = (1.0 - unique_entropy) * point_reliability * pull_sem_gate * pull_mask
    confident_mask = (pull_confidence >= conf_thresh).to(unique_features.dtype)
    update_confidence = (1.0 - unique_entropy) * point_reliability * update_sem_gate * update_mask
    update_confident_mask = (update_confidence >= update_conf_thresh).to(unique_features.dtype)

    unique_proto_idx = assigned_proto_idx[unique_indices]
    unique_assigned_proto = prototype_bank.prototypes[unique_proto_idx]
    pull_weight = pull_confidence * unique_assign_conf * confident_mask
    pull_distance = 1.0 - (unique_features * unique_assigned_proto).sum(dim=-1).clamp(-1.0, 1.0)
    pull_loss = (pull_weight * pull_distance).sum() / (pull_weight.sum() + eps)

    prototype_cosine = torch.matmul(prototype_bank.prototypes, prototype_bank.prototypes.t())
    off_diag_mask = 1.0 - torch.eye(
        prototype_bank.num_prototypes,
        device=prototype_cosine.device,
        dtype=prototype_cosine.dtype,
    )
    sep_penalty = torch.clamp(prototype_cosine - sep_margin, min=0.0) ** 2
    sep_loss = (sep_penalty * off_diag_mask).sum() / (off_diag_mask.sum() + eps)

    sample_probs = assignment_probs[graph_data["sample_indices"]]
    neighbor_probs = assignment_probs[graph_data["neighbor_indices"]]
    base_cons_weight = (
        graph_data["positive_mask"]
        * graph_data["reliability"]
        * graph_data["semantic_pos_factor"]
    )
    cons_weight = base_cons_weight
    cons_conf_mean = zero
    cons_adaptive_factor_mean = selected_features.new_tensor(1.0)
    cons_scene_scale = selected_features.new_tensor(1.0)
    cons_agree_ratio = zero
    cons_agree_factor_mean = selected_features.new_tensor(1.0)
    positive_mask = graph_data["positive_mask"]
    positive_count = positive_mask.sum()
    pair_conf = None

    if cons_conf_weight > 0.0 or cons_scene_weight > 0.0:
        selected_proto_conf = selected_features.new_zeros(selected_features.shape[0])
        unique_cons_conf = (pull_confidence * unique_assign_conf * confident_mask).detach()
        selected_proto_conf[unique_indices] = unique_cons_conf.clamp_min(0.0)

        sample_conf = selected_proto_conf[graph_data["sample_indices"]]
        neighbor_conf = selected_proto_conf[graph_data["neighbor_indices"]]
        pair_conf = torch.sqrt(sample_conf.unsqueeze(-1) * neighbor_conf).clamp_min(0.0)
        cons_conf_mean = (positive_mask * pair_conf).sum() / (positive_count + eps)

    if cons_conf_weight > 0.0 and pair_conf is not None:
        edge_pair_conf = pair_conf
        if cons_conf_normalize:
            pair_conf_mean = cons_conf_mean.clamp_min(eps)
            edge_pair_conf = edge_pair_conf / pair_conf_mean
            if cons_conf_norm_max is not None and cons_conf_norm_max > 0:
                edge_pair_conf = edge_pair_conf.clamp(max=float(cons_conf_norm_max))

        blend = min(max(float(cons_conf_weight), 0.0), 1.0)
        floor = min(max(float(cons_conf_floor), 0.0), 1.0)
        power = max(float(cons_conf_power), 0.0)
        adaptive_factor = floor + (1.0 - floor) * edge_pair_conf.pow(power)
        adaptive_factor = (1.0 - blend) + blend * adaptive_factor
        cons_weight = base_cons_weight * adaptive_factor
        cons_adaptive_factor_mean = (
            positive_mask * adaptive_factor
        ).sum() / (positive_count + eps)

    if cons_agree_weight > 0.0:
        sample_proto_idx = assigned_proto_idx[graph_data["sample_indices"]]
        neighbor_proto_idx = assigned_proto_idx[graph_data["neighbor_indices"]]
        sample_assign_conf = assign_conf[graph_data["sample_indices"]]
        neighbor_assign_conf = assign_conf[graph_data["neighbor_indices"]]

        same_proto = sample_proto_idx.unsqueeze(-1) == neighbor_proto_idx
        conf_thresh_value = float(cons_agree_conf_thresh)
        if conf_thresh_value > 0.0:
            confident_pair = (
                (sample_assign_conf.unsqueeze(-1) >= conf_thresh_value)
                & (neighbor_assign_conf >= conf_thresh_value)
            )
            agree_mask = same_proto & confident_pair
        else:
            agree_mask = same_proto

        agree_mask = agree_mask.to(base_cons_weight.dtype)
        agree_ratio = (positive_mask * agree_mask).sum() / (positive_count + eps)
        agree_blend = min(max(float(cons_agree_weight), 0.0), 1.0)
        agree_floor = min(max(float(cons_agree_floor), 0.0), 1.0)
        agree_factor = agree_floor + (1.0 - agree_floor) * agree_mask
        agree_factor = (1.0 - agree_blend) + agree_blend * agree_factor
        cons_weight = cons_weight * agree_factor
        cons_agree_ratio = agree_ratio
        cons_agree_factor_mean = (
            positive_mask * agree_factor
        ).sum() / (positive_count + eps)

    prob_diff = ((sample_probs.unsqueeze(1) - neighbor_probs) ** 2).sum(dim=-1)
    cons_loss = (cons_weight * prob_diff).sum() / (cons_weight.sum() + eps)

    if cons_scene_weight > 0.0:
        scene_blend = min(max(float(cons_scene_weight), 0.0), 1.0)
        scene_floor = min(max(float(cons_scene_floor), 0.0), 1.0)
        scene_conf_min = max(float(cons_scene_conf_min), 0.0)
        scene_conf_target = max(float(cons_scene_conf_target), scene_conf_min + eps)
        scene_conf_progress = (
            (cons_conf_mean.detach() - scene_conf_min)
            / (scene_conf_target - scene_conf_min)
        ).clamp(0.0, 1.0)
        scene_scale = scene_floor + (1.0 - scene_floor) * scene_conf_progress
        cons_scene_scale = (1.0 - scene_blend) + scene_blend * scene_scale

    total_loss = (
        lambda_pull * pull_loss
        + lambda_sep * sep_loss
        + lambda_cons * cons_scene_scale * cons_loss
    )
    usage = torch.bincount(
        unique_proto_idx,
        minlength=prototype_bank.num_prototypes,
    ).to(unique_features.dtype)
    usage = usage / usage.sum().clamp_min(1.0)
    active_proto_ratio = (usage > (0.5 / max(prototype_bank.num_prototypes, 1))).to(usage.dtype).mean()
    proto_diagnostics = _prototype_diagnostics(
        prototype_bank=prototype_bank,
        graph_data=graph_data,
        unique_indices=unique_indices,
        num_selected_points=selected_features.shape[0],
        unique_proto_idx=unique_proto_idx,
        unique_entropy=unique_entropy,
        unique_assign_conf=unique_assign_conf,
        unique_assign_margin=unique_assign_margin,
        update_confidence=update_confidence,
        update_confident_mask=update_confident_mask,
        zero=zero,
        eps=eps,
    )

    return {
        "loss": lambda_val * total_loss,
        "pull_loss": pull_loss,
        "sep_loss": sep_loss,
        "cons_loss": cons_loss,
        "cons_conf_mean": cons_conf_mean,
        "cons_adaptive_factor_mean": cons_adaptive_factor_mean,
        "cons_scene_scale": cons_scene_scale,
        "cons_agree_ratio": cons_agree_ratio,
        "cons_agree_factor_mean": cons_agree_factor_mean,
        "avg_entropy": unique_entropy.mean(),
        "avg_assign_conf": unique_assign_conf.mean(),
        "avg_proto_confidence": pull_confidence.mean(),
        "confident_ratio": confident_mask.mean(),
        "update_avg_confidence": update_confidence.mean(),
        "update_confident_ratio": update_confident_mask.mean(),
        "avg_proto_margin": unique_assign_margin.mean(),
        "active_proto_ratio": active_proto_ratio,
        "usage_max": usage.max(),
        **proto_diagnostics,
        "update_features": unique_features.detach(),
        "update_probs": unique_probs.detach(),
        "update_confidence": update_confidence.detach(),
    }


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
    
