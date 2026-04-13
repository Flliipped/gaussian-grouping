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


def loss_sugar_opacity_entropy(opacity, eps=1e-6):
    """SuGaR-style optional opacity entropy regularizer.

    Minimizing the binary entropy gently encourages opacities to move away from
    uncertain mid-values. It is kept separate from the surface alignment loss so
    the semantic grouping pipeline can enable it only in controlled ablations.
    """
    opacity = opacity.clamp(eps, 1.0 - eps)
    entropy = -(opacity * torch.log(opacity) + (1.0 - opacity) * torch.log(1.0 - opacity))
    return entropy.mean()


def loss_geo_gated_contrastive(
    xyz,
    features,
    normals,
    point_ids=None,
    k=8,
    lambda_val=1.0,
    lambda_pos=1.0,
    lambda_neg=1.0,
    smooth_lambda=0.1,
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
    sem_neg_boost=1.0,
    eps=1e-8,
):
    zero = features.new_tensor(0.0)

    if features.size(0) > max_points:
        selected_indices = torch.randperm(features.size(0), device=features.device)[:max_points]
        xyz = xyz[selected_indices]
        features = features[selected_indices]
        normals = normals[selected_indices]
        if point_ids is not None:
            point_ids = point_ids[selected_indices]

    num_points = features.size(0)
    if num_points < 2:
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
            zero,
        )

    sample_count = min(sample_size, num_points)
    effective_k = min(k, num_points - 1)
    if sample_count <= 0 or effective_k <= 0:
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
            zero,
        )

    raw_feature_norm = torch.norm(features, dim=-1)
    sample_indices = torch.randperm(num_points, device=features.device)[:sample_count]
    sample_xyz = xyz[sample_indices]
    sample_features = features[sample_indices]
    sample_normals = F.normalize(normals[sample_indices], dim=-1, eps=eps)

    dists = torch.cdist(sample_xyz, xyz)
    _, neighbor_indices_tensor = dists.topk(effective_k + 1, largest=False)
    neighbor_indices_tensor = neighbor_indices_tensor[:, 1:]
    neighbor_dists = torch.take_along_dim(dists, neighbor_indices_tensor, dim=1)

    neighbor_xyz = xyz[neighbor_indices_tensor]
    neighbor_features = features[neighbor_indices_tensor]
    neighbor_normals = F.normalize(normals[neighbor_indices_tensor], dim=-1, eps=eps)

    rel_xyz = neighbor_xyz - sample_xyz.unsqueeze(1)
    feat_dist = torch.norm(neighbor_features - sample_features.unsqueeze(1), dim=-1)
    aligned_normal_cos = torch.abs(
        (sample_normals.unsqueeze(1) * neighbor_normals).sum(dim=-1)
    ).clamp(0.0, 1.0)
    sample_depth_gap = torch.abs((rel_xyz * sample_normals.unsqueeze(1)).sum(dim=-1))
    neighbor_depth_gap = torch.abs((rel_xyz * neighbor_normals).sum(dim=-1))
    depth_gap = 0.5 * (sample_depth_gap + neighbor_depth_gap)

    if normal_weight_lambda > 0:
        normal_weight = torch.exp(normal_weight_lambda * (aligned_normal_cos - 1.0))
    else:
        normal_weight = torch.ones_like(aligned_normal_cos)

    local_radius = neighbor_dists[:, -1:].clamp_min(eps)
    spatial_ratio = neighbor_dists / local_radius
    spatial_close = (spatial_ratio <= spatial_pos_scale).to(feat_dist.dtype)
    normal_close = (aligned_normal_cos >= normal_pos_tau).to(feat_dist.dtype)
    depth_close = (depth_gap < plane_tau).to(feat_dist.dtype)
    geom_gate = spatial_close * normal_close * depth_close

    if neg_plane_tau is None or neg_plane_tau <= plane_tau:
        depth_far = 1.0 - depth_close
    else:
        depth_far = (depth_gap > neg_plane_tau).to(feat_dist.dtype)
    normal_far = (aligned_normal_cos <= normal_neg_tau).to(feat_dist.dtype)
    boundary_neg_mask = spatial_close * ((depth_far + normal_far) > 0).to(feat_dist.dtype) * (1.0 - geom_gate)

    flat_point_ids = torch.cat([sample_indices, neighbor_indices_tensor.reshape(-1)], dim=0)
    unique_point_ids, inverse_ids = torch.unique(flat_point_ids, sorted=False, return_inverse=True)
    unique_xyz = xyz[unique_point_ids]
    sample_local = inverse_ids[:sample_count]
    neighbor_local = inverse_ids[sample_count:].view(sample_count, effective_k)

    avg_sem_valid_views = zero
    avg_sem_confidence = zero
    semantic_pos_keep_ratio = zero
    semantic_neg_keep_ratio = zero
    sem_same_mask = torch.zeros_like(geom_gate)
    sem_diff_mask = torch.zeros_like(geom_gate)
    semantic_neg_weight = torch.ones_like(geom_gate)
    positive_sem_mask = None
    semantic_cross_neg_mask = torch.zeros_like(geom_gate)

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
            (
                point_sem_label,
                point_sem_valid,
                point_sem_confidence,
                point_valid_view_count,
            ) = compute_point_label_consensus(
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
            sem_pair_min_conf = torch.minimum(sample_confidence.unsqueeze(-1), neighbor_confidence)
            sem_pair_strong = sem_pair_valid & (sem_pair_min_conf >= sem_conf_tau)

            sem_same_mask = (sem_pair_valid & (sample_labels.unsqueeze(-1) == neighbor_labels)).to(feat_dist.dtype)
            sem_diff_mask = (sem_pair_valid & (sample_labels.unsqueeze(-1) != neighbor_labels)).to(feat_dist.dtype)
            positive_sem_mask = (sem_pair_strong & (sample_labels.unsqueeze(-1) == neighbor_labels)).to(feat_dist.dtype)
            semantic_cross_neg_mask = (
                spatial_close
                * (sem_pair_strong & (sample_labels.unsqueeze(-1) != neighbor_labels)).to(feat_dist.dtype)
            )
            semantic_neg_weight = 1.0 + sem_neg_boost * semantic_cross_neg_mask

            sem_valid_point_count = point_sem_valid.to(feat_dist.dtype).sum()
            avg_sem_valid_views = (
                point_sem_valid.to(point_valid_view_count.dtype) * point_valid_view_count
            ).sum() / (sem_valid_point_count + eps)
            avg_sem_confidence = (
                point_sem_valid.to(point_sem_confidence.dtype) * point_sem_confidence
            ).sum() / (sem_valid_point_count + eps)

    if positive_sem_mask is None:
        pos_mask = geom_gate
        semantic_pos_keep_ratio = torch.where(
            geom_gate.sum() > 0,
            torch.ones_like(zero),
            zero,
        )
    else:
        pos_mask = geom_gate * positive_sem_mask
        semantic_pos_keep_ratio = pos_mask.sum() / (geom_gate.sum() + eps)

    neg_candidate_mask = torch.clamp(boundary_neg_mask + semantic_cross_neg_mask, min=0.0, max=1.0)
    semantic_neg_keep_ratio = semantic_cross_neg_mask.sum() / (neg_candidate_mask.sum() + eps)
    ignore_mask = torch.clamp(1.0 - torch.clamp(pos_mask + neg_candidate_mask, max=1.0), min=0.0, max=1.0)

    pos_pair_weight = pos_mask * normal_weight
    pos_loss = (pos_pair_weight * (feat_dist ** 2)).sum() / (pos_pair_weight.sum() + eps)

    smooth_pair_weight = geom_gate * normal_weight
    smooth_loss = (smooth_pair_weight * (feat_dist ** 2)).sum() / (smooth_pair_weight.sum() + eps)

    effective_hard_neg_k = min(max(int(hard_neg_k), 0), effective_k)
    if effective_hard_neg_k > 0:
        masked_neg_dist = feat_dist.masked_fill(neg_candidate_mask == 0, 1e6)
        _, hard_neg_idx = masked_neg_dist.topk(effective_hard_neg_k, dim=1, largest=False)
        hard_neg_mask = torch.zeros_like(neg_candidate_mask)
        hard_neg_mask.scatter_(1, hard_neg_idx, 1.0)
        hard_neg_mask = hard_neg_mask * neg_candidate_mask
    else:
        hard_neg_mask = neg_candidate_mask

    neg_term = torch.clamp(neg_margin - feat_dist, min=0.0)
    active_hard_neg_mask = hard_neg_mask * (feat_dist < neg_margin).to(feat_dist.dtype)
    neg_pair_weight = active_hard_neg_mask * semantic_neg_weight
    neg_loss = (neg_pair_weight * (neg_term ** 2)).sum() / (neg_pair_weight.sum() + eps)

    contrastive_loss = lambda_pos * pos_loss + lambda_neg * neg_loss
    total_loss = contrastive_loss + smooth_lambda * smooth_loss

    pos_count = pos_mask.sum()
    neg_candidate_count = neg_candidate_mask.sum()
    ignore_count = ignore_mask.sum()
    hard_neg_count = hard_neg_mask.sum()
    active_hard_neg_count = active_hard_neg_mask.sum()
    total_pair_count = feat_dist.numel()

    gate_ratio = geom_gate.mean()
    avg_depth_gap = depth_gap.mean()
    avg_feature_norm = raw_feature_norm.mean()
    avg_normal_cosine = (geom_gate * aligned_normal_cos).sum() / (geom_gate.sum() + eps)
    avg_pos_dist = (pos_mask * feat_dist).sum() / (pos_count + eps)
    avg_neg_dist = (neg_candidate_mask * feat_dist).sum() / (neg_candidate_count + eps)
    avg_hard_neg_dist = (hard_neg_mask * feat_dist).sum() / (hard_neg_count + eps)
    active_neg_ratio = active_hard_neg_count / (hard_neg_count + eps)
    neg_candidate_ratio = neg_candidate_count / (total_pair_count + eps)
    ignore_ratio = ignore_count / (total_pair_count + eps)
    boundary_neg_ratio = boundary_neg_mask.sum() / (neg_candidate_count + eps)

    return (
        lambda_val * total_loss,
        gate_ratio,
        avg_depth_gap,
        contrastive_loss,
        smooth_loss,
        pos_loss,
        neg_loss,
        avg_feature_norm,
        avg_normal_cosine,
        avg_pos_dist,
        avg_neg_dist,
        avg_hard_neg_dist,
        active_neg_ratio,
        neg_candidate_ratio,
        ignore_ratio,
        avg_sem_valid_views,
        avg_sem_confidence,
        semantic_pos_keep_ratio,
        semantic_neg_keep_ratio,
        pos_count,
        neg_candidate_count,
        boundary_neg_ratio,
    )
    
