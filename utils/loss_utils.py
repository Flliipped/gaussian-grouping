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


def extract_surface_axis_and_flatness(scaling, rotation, eps=1e-8):
    rotation_matrix = build_rotation(rotation)
    min_axis_idx = torch.argmin(scaling, dim=-1)
    surface_axis = torch.gather(
        rotation_matrix,
        2,
        min_axis_idx.view(-1, 1, 1).expand(-1, 3, 1),
    ).squeeze(-1)
    surface_axis = F.normalize(surface_axis, dim=-1, eps=eps)

    sorted_scaling, _ = torch.sort(scaling, dim=-1)
    flat_ratio = sorted_scaling[:, 0] / (0.5 * (sorted_scaling[:, 1] + sorted_scaling[:, 2]) + eps)
    return surface_axis, flat_ratio


def compute_surface_geometry_confidence(
    surface_axis,
    flat_ratio,
    pca_normals,
    align_tau=0.6,
    flat_tau=0.2,
    temperature=10.0,
):
    surface_dot = (surface_axis * pca_normals).sum(dim=-1)
    aligned_surface_axis = torch.where(surface_dot.unsqueeze(-1) < 0, -surface_axis, surface_axis)
    axis_align_cosine = torch.abs(surface_dot).clamp(0.0, 1.0)

    align_conf = torch.sigmoid(temperature * (axis_align_cosine - align_tau))
    flat_conf = torch.sigmoid(temperature * (flat_tau - flat_ratio))
    surface_confidence = align_conf * flat_conf
    return aligned_surface_axis, axis_align_cosine, surface_confidence


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

    surface_axis, flat_ratio = extract_surface_axis_and_flatness(sample_scaling, sample_rotation, eps=eps)

    axis_align_cosine = torch.abs((surface_axis * pca_normals).sum(dim=-1)).clamp(0.0, 1.0)
    axis_loss = (1.0 - axis_align_cosine).mean()

    rel_xyz = neighbor_xyz - sample_xyz.unsqueeze(1)
    plane_residual = torch.abs((rel_xyz * surface_axis.unsqueeze(1)).sum(dim=-1))
    local_radius = neighbor_dists[:, -1:].clamp_min(eps)
    normalized_plane_residual = plane_residual / local_radius
    plane_loss = normalized_plane_residual.mean()
    flat_loss = flat_ratio.mean()

    loss = lambda_axis * axis_loss + lambda_plane * plane_loss + lambda_flat * flat_loss

    return (
        lambda_val * loss,
        axis_align_cosine.mean(),
        plane_residual.mean(),
        flat_ratio.mean(),
        loss,
    )


def loss_geo_contrastive_cosine(
    xyz,
    features,
    scaling=None,
    rotation=None,
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
    surface_align_tau=0.6,
    surface_flat_tau=0.2,
    surface_fusion_temperature=10.0,
    surface_conf_tau=0.35,
    surface_boundary_tau=1.0,
    surface_axis_tau=0.85,
    surface_depth_weight=1.0,
    surface_axis_weight=0.0,
    surface_pos_damp=0.0,
    surface_neg_boost=0.25,
    eps=1e-8,
):
    zero = features.new_tensor(0.0)

    if features.size(0) > max_points:
        selected_indices = torch.randperm(features.size(0), device=features.device)[:max_points]
        xyz = xyz[selected_indices]
        features = features[selected_indices]
        if scaling is not None:
            scaling = scaling[selected_indices]
        if rotation is not None:
            rotation = rotation[selected_indices]
        if point_ids is not None:
            point_ids = point_ids[selected_indices]

    num_points = features.size(0)
    if num_points < 2:
        return zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero

    sample_count = min(sample_size, num_points)
    effective_k = min(k, num_points - 1)
    if sample_count <= 0 or effective_k <= 0:
        return zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero

    raw_feature_norm = torch.norm(features, dim=-1)
    normalized_features = F.normalize(features, dim=-1, eps=eps)

    sample_indices = torch.randperm(num_points, device=features.device)[:sample_count]
    sample_xyz = xyz[sample_indices]
    sample_features = normalized_features[sample_indices]

    dists = torch.cdist(sample_xyz, xyz)
    _, neighbor_indices_tensor = dists.topk(effective_k + 1, largest=False)
    neighbor_indices_tensor = neighbor_indices_tensor[:, 1:]
    neighbor_dists = torch.take_along_dim(dists, neighbor_indices_tensor, dim=1)

    neighbor_xyz = xyz[neighbor_indices_tensor]
    neighbor_features = normalized_features[neighbor_indices_tensor]

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
    unique_pca_normals = unique_eigvecs[:, :, 0]

    avg_surface_confidence = zero
    avg_surface_axis_align = zero
    avg_surface_axis_cosine = zero
    avg_surface_boundary_score = zero
    unique_normals = unique_pca_normals
    if scaling is not None and rotation is not None:
        unique_scaling = scaling[unique_point_ids]
        unique_rotation = rotation[unique_point_ids]
        unique_surface_axis, unique_flat_ratio = extract_surface_axis_and_flatness(unique_scaling, unique_rotation, eps=eps)
        unique_sorted_scaling, _ = torch.sort(unique_scaling, dim=-1)
        unique_surface_thickness = unique_sorted_scaling[:, 0].clamp_min(eps)
        aligned_surface_axis, unique_axis_align_cosine, unique_surface_confidence = compute_surface_geometry_confidence(
            unique_surface_axis,
            unique_flat_ratio,
            unique_pca_normals,
            align_tau=surface_align_tau,
            flat_tau=surface_flat_tau,
            temperature=surface_fusion_temperature,
        )
        avg_surface_confidence = unique_surface_confidence.mean()
        avg_surface_axis_align = unique_axis_align_cosine.mean()
    else:
        aligned_surface_axis = unique_pca_normals
        unique_surface_confidence = torch.zeros(unique_point_ids.shape[0], device=xyz.device, dtype=xyz.dtype)
        unique_surface_thickness = torch.ones(unique_point_ids.shape[0], device=xyz.device, dtype=xyz.dtype)

    sample_local = inverse_ids[:sample_count]
    neighbor_local = inverse_ids[sample_count:].view(sample_count, effective_k)
    sample_normals = unique_normals[sample_local]
    neighbor_normals = unique_normals[neighbor_local.reshape(-1)].view(sample_count, effective_k, 3)
    sample_surface_axis = aligned_surface_axis[sample_local]
    neighbor_surface_axis = aligned_surface_axis[neighbor_local.reshape(-1)].view(sample_count, effective_k, 3)
    sample_surface_confidence = unique_surface_confidence[sample_local]
    neighbor_surface_confidence = unique_surface_confidence[neighbor_local.reshape(-1)].view(sample_count, effective_k)
    sample_surface_thickness = unique_surface_thickness[sample_local].unsqueeze(-1)

    rel_xyz = neighbor_xyz - sample_xyz.unsqueeze(1)
    xyz_dist = neighbor_dists
    plane_residual = torch.abs((rel_xyz * sample_normals.unsqueeze(1)).sum(dim=-1))
    cosine_sim = (neighbor_features * sample_features.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)
    normal_cosine = torch.abs((sample_normals.unsqueeze(1) * neighbor_normals).sum(dim=-1)).clamp(0.0, 1.0)
    if normal_weight_lambda > 0:
        normal_weight = torch.exp(-normal_weight_lambda * (1.0 - normal_cosine))
    else:
        normal_weight = torch.ones_like(normal_cosine)
    local_radius = xyz_dist[:, -1:].clamp_min(eps)
    spatial_ratio = xyz_dist / local_radius
    spatial_weight = torch.exp(-spatial_ratio)

    surface_axis_cosine = torch.abs(
        (sample_surface_axis.unsqueeze(1) * neighbor_surface_axis).sum(dim=-1)
    ).clamp(0.0, 1.0)
    pair_surface_confidence = torch.sqrt(
        torch.clamp(sample_surface_confidence.unsqueeze(-1) * neighbor_surface_confidence, min=0.0)
    )
    surface_conf_gate = torch.sigmoid(
        surface_fusion_temperature * (pair_surface_confidence - surface_conf_tau)
    )
    thickness_normalized_residual = plane_residual / sample_surface_thickness
    surface_depth_score = torch.sigmoid(
        surface_fusion_temperature * (thickness_normalized_residual - surface_boundary_tau)
    )
    surface_axis_score = torch.sigmoid(
        surface_fusion_temperature * (surface_axis_tau - surface_axis_cosine)
    )
    surface_score_norm = max(surface_depth_weight + surface_axis_weight, eps)
    surface_boundary_raw = (
        surface_depth_weight * surface_depth_score
        + surface_axis_weight * surface_axis_score
    ) / surface_score_norm
    surface_boundary_score = pair_surface_confidence * surface_conf_gate * surface_boundary_raw
    avg_surface_axis_cosine = surface_axis_cosine.mean()
    avg_surface_boundary_score = surface_boundary_score.mean()

    # Geometry gate decides where semantic propagation is allowed.
    spatial_pos_mask = (spatial_ratio <= spatial_pos_scale).to(cosine_sim.dtype)
    normal_pos_mask = (normal_cosine >= normal_pos_tau).to(cosine_sim.dtype)
    depth_pos_mask = (plane_residual < plane_tau).to(cosine_sim.dtype)
    geom_pos_mask = spatial_pos_mask * normal_pos_mask * depth_pos_mask

    if neg_plane_tau is None or neg_plane_tau <= plane_tau:
        depth_neg_mask = 1.0 - depth_pos_mask
    else:
        depth_neg_mask = (plane_residual > neg_plane_tau).to(cosine_sim.dtype)
    normal_neg_mask = (normal_cosine <= normal_neg_tau).to(cosine_sim.dtype)
    geom_neg_mask = ((depth_neg_mask + normal_neg_mask) > 0).to(cosine_sim.dtype) * (1.0 - geom_pos_mask)
    ignore_mask = torch.clamp(1.0 - geom_pos_mask - geom_neg_mask, min=0.0, max=1.0)

    avg_sem_valid_views = zero
    avg_sem_confidence = zero
    semantic_pos_keep_ratio = zero
    semantic_neg_keep_ratio = zero
    sem_same_mask = torch.zeros_like(geom_pos_mask)
    sem_diff_mask = torch.zeros_like(geom_pos_mask)
    sem_pair_conf = torch.zeros_like(geom_pos_mask)
    semantic_pos_factor = torch.ones_like(geom_pos_mask)
    semantic_neg_factor = torch.ones_like(geom_pos_mask)
    pos_mask = geom_pos_mask
    neg_candidate_mask = geom_neg_mask

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

            sample_pair_valid = sample_valid.unsqueeze(-1)
            sem_pair_valid = sample_pair_valid & neighbor_valid
            sem_same_mask = (sem_pair_valid & (sample_labels.unsqueeze(-1) == neighbor_labels)).to(cosine_sim.dtype)
            sem_diff_mask = (sem_pair_valid & (sample_labels.unsqueeze(-1) != neighbor_labels)).to(cosine_sim.dtype)
            sem_pair_conf = sample_confidence.unsqueeze(-1) * neighbor_confidence

            sem_valid_point_count = point_sem_valid.to(cosine_sim.dtype).sum()
            avg_sem_valid_views = (point_sem_valid.to(point_valid_view_count.dtype) * point_valid_view_count).sum() / (sem_valid_point_count + eps)
            avg_sem_confidence = (point_sem_valid.to(point_sem_confidence.dtype) * point_sem_confidence).sum() / (sem_valid_point_count + eps)
            geom_pos_count = geom_pos_mask.sum()
            geom_neg_count = geom_neg_mask.sum()
            # Semantics should enhance or weaken supervision, not delete most pairs.
            semantic_pos_factor = 1.0 + sem_same_boost * sem_same_mask * sem_pair_conf
            semantic_pos_factor = semantic_pos_factor * torch.clamp(
                1.0 - sem_conflict_penalty * sem_diff_mask * sem_pair_conf,
                min=0.25,
                max=2.0,
            )
            semantic_neg_factor = 1.0 + sem_neg_boost * sem_diff_mask * sem_pair_conf

            # High-confidence semantic disagreements can be promoted to local negatives,
            # which helps separate co-planar but semantically different instances.
            semantic_neg_promote_mask = (
                spatial_pos_mask
                * sem_diff_mask
                * (sem_pair_conf >= sem_conf_tau).to(cosine_sim.dtype)
            )
            neg_candidate_mask = torch.clamp(geom_neg_mask + semantic_neg_promote_mask, min=0.0, max=1.0)

            semantic_pos_keep_ratio = (geom_pos_mask * semantic_pos_factor).sum() / (geom_pos_count + eps)
            neg_reference_count = geom_neg_count + semantic_neg_promote_mask.sum()
            semantic_neg_keep_ratio = (neg_candidate_mask * semantic_neg_factor).sum() / (neg_reference_count + eps)
            ignore_mask = torch.clamp(1.0 - torch.clamp(geom_pos_mask + neg_candidate_mask, max=1.0), min=0.0, max=1.0)

    pos_count = pos_mask.sum()
    neg_candidate_count = neg_candidate_mask.sum()
    ignore_count = ignore_mask.sum()

    pos_geometry_factor = torch.clamp(
        1.0 - surface_pos_damp * surface_boundary_score,
        min=0.25,
        max=1.0,
    )
    pos_pair_weight = pos_mask * spatial_weight * normal_weight * semantic_pos_factor * pos_geometry_factor
    pos_loss = (pos_pair_weight * (1.0 - cosine_sim)).sum() / (pos_pair_weight.sum() + eps)

    effective_hard_neg_k = min(max(int(hard_neg_k), 0), effective_k)
    if effective_hard_neg_k > 0:
        hard_neg_scores = (
            cosine_sim
            + 0.5 * (1.0 - normal_cosine)
            + sem_neg_boost * sem_diff_mask * sem_pair_conf
            + surface_neg_boost * surface_boundary_score
        ).masked_fill(neg_candidate_mask == 0, -1e6)
        _, hard_neg_idx = hard_neg_scores.topk(effective_hard_neg_k, dim=1, largest=True)
        hard_neg_mask = torch.zeros_like(neg_candidate_mask)
        hard_neg_mask.scatter_(1, hard_neg_idx, 1.0)
        hard_neg_mask = hard_neg_mask * neg_candidate_mask
    else:
        hard_neg_mask = neg_candidate_mask

    neg_term = torch.clamp(cosine_sim - neg_margin, min=0.0)
    active_hard_neg_mask = hard_neg_mask * (cosine_sim > neg_margin).to(cosine_sim.dtype)
    hard_neg_count = hard_neg_mask.sum()
    active_hard_neg_count = active_hard_neg_mask.sum()

    neg_geometry_factor = 1.0 + surface_neg_boost * surface_boundary_score
    neg_pair_weight = active_hard_neg_mask * semantic_neg_factor * neg_geometry_factor
    neg_loss = (neg_pair_weight * (neg_term ** 2)).sum() / (neg_pair_weight.sum() + eps)

    loss = lambda_pos * pos_loss + lambda_neg * neg_loss
    gate_ratio = pos_mask.mean()
    avg_plane_residual = plane_residual.mean()
    avg_feature_norm = raw_feature_norm.mean()
    avg_normal_cosine = (pos_mask * normal_cosine).sum() / (pos_count + eps)
    avg_pos_cosine = (pos_mask * cosine_sim).sum() / (pos_count + eps)
    avg_neg_cosine = (neg_candidate_mask * cosine_sim).sum() / (neg_candidate_count + eps)
    avg_hard_neg_cosine = (hard_neg_mask * cosine_sim).sum() / (hard_neg_count + eps)
    active_neg_ratio = active_hard_neg_count / (hard_neg_count + eps)
    neg_candidate_ratio = neg_candidate_count / (pos_mask.numel() + eps)
    ignore_ratio = ignore_count / (pos_mask.numel() + eps)

    return (
        lambda_val * loss,
        gate_ratio,
        avg_plane_residual,
        pos_loss,
        neg_loss,
        avg_feature_norm,
        avg_normal_cosine,
        avg_pos_cosine,
        avg_neg_cosine,
        avg_hard_neg_cosine,
        avg_surface_confidence,
        avg_surface_axis_align,
        avg_surface_axis_cosine,
        avg_surface_boundary_score,
        active_neg_ratio,
        neg_candidate_ratio,
        ignore_ratio,
        avg_sem_valid_views,
        avg_sem_confidence,
        semantic_pos_keep_ratio,
        semantic_neg_keep_ratio,
    )
    
