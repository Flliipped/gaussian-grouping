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
from utils.general_utils import build_rotation
from utils.multiview_utils import collect_multiview_labels, compute_point_label_consensus

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

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
    eps=1e-8,
):
    zero = features.new_tensor(0.0)

    if features.size(0) > max_points:
        selected_indices = torch.randperm(features.size(0), device=features.device)[:max_points]
        xyz = xyz[selected_indices]
        features = features[selected_indices]
        if point_ids is not None:
            point_ids = point_ids[selected_indices]

    num_points = features.size(0)
    if num_points < 2:
        return zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero

    sample_count = min(sample_size, num_points)
    effective_k = min(k, num_points - 1)
    if sample_count <= 0 or effective_k <= 0:
        return zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero

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
    unique_normals = unique_eigvecs[:, :, 0]
    sample_local = inverse_ids[:sample_count]
    neighbor_local = inverse_ids[sample_count:].view(sample_count, effective_k)
    sample_normals = unique_normals[sample_local]
    neighbor_normals = unique_normals[neighbor_local.reshape(-1)].view(sample_count, effective_k, 3)

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

    pos_pair_weight = pos_mask * spatial_weight * normal_weight * semantic_pos_factor
    pos_loss = (pos_pair_weight * (1.0 - cosine_sim)).sum() / (pos_pair_weight.sum() + eps)

    effective_hard_neg_k = min(max(int(hard_neg_k), 0), effective_k)
    if effective_hard_neg_k > 0:
        hard_neg_scores = (
            cosine_sim
            + 0.5 * (1.0 - normal_cosine)
            + sem_neg_boost * sem_diff_mask * sem_pair_conf
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

    neg_pair_weight = active_hard_neg_mask * semantic_neg_factor
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
        active_neg_ratio,
        neg_candidate_ratio,
        ignore_ratio,
        avg_sem_valid_views,
        avg_sem_confidence,
        semantic_pos_keep_ratio,
        semantic_neg_keep_ratio,
    )


def loss_object_prototype(
    xyz,
    features,
    prototype_state,
    point_ids=None,
    support_cameras=None,
    support_visibility=None,
    sem_num_classes=None,
    sem_ignore_label=-1,
    sem_min_views=2,
    sem_conf_tau=0.7,
    bank_conf_tau=0.8,
    bank_momentum=0.9,
    temperature=0.2,
    lambda_assign=1.0,
    lambda_pull=1.0,
    lambda_push=0.5,
    lambda_soft=0.25,
    push_margin=0.1,
    max_points=200000,
    sample_size=1200,
    min_proto_points=4,
    active_count_tau=32,
    max_active_prototypes=16,
    soft_conf_floor=0.3,
    eps=1e-8,
):
    zero = features.new_tensor(0.0)

    if prototype_state is None or support_cameras is None or len(support_cameras) == 0:
        return zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero

    if features.size(0) > max_points:
        selected_indices = torch.randperm(features.size(0), device=features.device)[:max_points]
        xyz = xyz[selected_indices]
        features = features[selected_indices]
        if point_ids is not None:
            point_ids = point_ids[selected_indices]

    num_points = features.size(0)
    if num_points == 0:
        return zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero

    sample_count = min(sample_size, num_points)
    sample_indices = torch.randperm(num_points, device=features.device)[:sample_count]
    sample_xyz = xyz[sample_indices]
    sample_features = F.normalize(features[sample_indices], dim=-1, eps=eps)
    sample_point_ids = sample_indices if point_ids is None else point_ids[sample_indices]

    view_labels, view_valid = collect_multiview_labels(
        support_cameras,
        sample_xyz,
        point_ids=sample_point_ids,
        visibility_masks=support_visibility,
        ignore_label=sem_ignore_label,
    )
    point_sem_label, point_sem_valid, point_sem_confidence, point_valid_view_count = compute_point_label_consensus(
        view_labels,
        view_valid,
        num_classes=sem_num_classes,
        min_views=sem_min_views,
        conf_tau=sem_conf_tau,
    )

    prototype_bank = prototype_state["bank"]
    prototype_valid = prototype_state["valid"]
    prototype_counts = prototype_state["counts"]
    num_prototypes = prototype_bank.shape[0]

    update_mask = point_sem_valid & (point_sem_confidence >= bank_conf_tau)
    updated_proto_count = zero
    with torch.no_grad():
        if update_mask.any():
            update_labels = torch.unique(point_sem_label[update_mask])
            updated = 0
            for label in update_labels.tolist():
                label_mask = update_mask & (point_sem_label == label)
                if int(label_mask.sum().item()) < max(int(min_proto_points), 1):
                    continue
                proto_feat = sample_features[label_mask].mean(dim=0)
                proto_feat = F.normalize(proto_feat.unsqueeze(0), dim=-1, eps=eps).squeeze(0)
                if prototype_valid[label]:
                    proto_feat = bank_momentum * prototype_bank[label] + (1.0 - bank_momentum) * proto_feat
                    proto_feat = F.normalize(proto_feat.unsqueeze(0), dim=-1, eps=eps).squeeze(0)
                prototype_bank[label].copy_(proto_feat)
                prototype_valid[label] = True
                prototype_counts[label] += point_sem_confidence[label_mask].sum().to(prototype_counts.dtype)
                updated += 1
            updated_proto_count = zero.new_tensor(float(updated))

    active_proto_ids = (prototype_valid & (prototype_counts >= float(active_count_tau))).nonzero(as_tuple=False).squeeze(-1)
    if max_active_prototypes > 0 and active_proto_ids.numel() > max_active_prototypes:
        active_proto_counts = prototype_counts[active_proto_ids]
        _, topk_idx = active_proto_counts.topk(max_active_prototypes, largest=True)
        active_proto_ids = active_proto_ids[topk_idx]
    active_proto_count = zero.new_tensor(float(active_proto_ids.numel()))
    if active_proto_ids.numel() == 0:
        return zero, zero, zero, zero, zero, active_proto_count, updated_proto_count, zero, zero, zero, zero, zero, zero

    active_prototypes = F.normalize(prototype_bank[active_proto_ids], dim=-1, eps=eps)
    similarity = torch.matmul(sample_features, active_prototypes.transpose(0, 1)).clamp(-1.0, 1.0)
    logits = similarity / max(float(temperature), eps)

    label_to_local = torch.full((num_prototypes,), -1, device=features.device, dtype=torch.long)
    label_to_local[active_proto_ids] = torch.arange(active_proto_ids.numel(), device=features.device)
    safe_labels = point_sem_label.clamp(min=0, max=num_prototypes - 1)
    local_targets = label_to_local[safe_labels]
    supervised_mask = point_sem_valid & (point_sem_confidence >= sem_conf_tau) & (local_targets >= 0)

    assign_loss = zero
    pull_loss = zero
    push_loss = zero
    avg_pos_similarity = zero
    avg_neg_similarity = zero

    if supervised_mask.any():
        supervised_weights = point_sem_confidence[supervised_mask]
        supervised_logits = logits[supervised_mask]
        supervised_targets = local_targets[supervised_mask]

        ce_loss = F.cross_entropy(supervised_logits, supervised_targets, reduction="none")
        assign_loss = (ce_loss * supervised_weights).sum() / (supervised_weights.sum() + eps)
        if active_proto_ids.numel() > 1:
            assign_loss = assign_loss / torch.log(zero.new_tensor(float(active_proto_ids.numel())))

        pos_similarity = similarity[supervised_mask, supervised_targets]
        pull_loss = ((1.0 - pos_similarity) * supervised_weights).sum() / (supervised_weights.sum() + eps)
        avg_pos_similarity = pos_similarity.mean()

        if active_proto_ids.numel() > 1:
            neg_similarity = similarity[supervised_mask].clone()
            neg_similarity.scatter_(1, supervised_targets.unsqueeze(1), -1.0)
            hardest_negative_similarity = neg_similarity.max(dim=1).values
            push_term = torch.clamp(hardest_negative_similarity - pos_similarity + push_margin, min=0.0)
            push_loss = ((push_term ** 2) * supervised_weights).sum() / (supervised_weights.sum() + eps)
            avg_neg_similarity = hardest_negative_similarity.mean()

    soft_loss = zero
    soft_mask = point_sem_valid & (~supervised_mask) & (point_sem_confidence >= soft_conf_floor)
    if soft_mask.any():
        soft_probs = torch.softmax(logits[soft_mask], dim=-1)
        proto_mix = torch.matmul(soft_probs, active_prototypes)
        proto_mix = F.normalize(proto_mix, dim=-1, eps=eps)
        soft_weights = point_sem_confidence[soft_mask]
        soft_similarity = (sample_features[soft_mask] * proto_mix).sum(dim=-1).clamp(-1.0, 1.0)
        soft_loss = ((1.0 - soft_similarity) * soft_weights).sum() / (soft_weights.sum() + eps)

    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
    if active_proto_ids.numel() > 1:
        entropy = entropy / torch.log(zero.new_tensor(float(active_proto_ids.numel())))
    else:
        entropy = torch.zeros_like(entropy)

    prototype_loss = (
        lambda_assign * assign_loss
        + lambda_pull * pull_loss
        + lambda_push * push_loss
        + lambda_soft * soft_loss
    )

    supervised_ratio = supervised_mask.to(zero.dtype).mean()
    soft_ratio = soft_mask.to(zero.dtype).mean()
    avg_confidence = point_sem_confidence[point_valid_view_count > 0].mean() if (point_valid_view_count > 0).any() else zero
    avg_entropy = entropy.mean()

    return (
        prototype_loss,
        assign_loss,
        pull_loss,
        push_loss,
        soft_loss,
        active_proto_count,
        updated_proto_count,
        supervised_ratio,
        soft_ratio,
        avg_confidence,
        avg_entropy,
        avg_pos_similarity,
        avg_neg_similarity,
    )
