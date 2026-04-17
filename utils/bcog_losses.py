import torch
import torch.nn.functional as F

from utils.multiview_utils import collect_multiview_labels, compute_point_label_consensus
from utils.prototype_bank import (
    bootstrap_scene_prototypes,
    compute_slot_assignments,
    get_active_prototype_ids,
    get_valid_prototype_ids,
    update_scene_prototypes,
)


def compute_graph_reliability(
    xyz,
    features,
    point_ids=None,
    scaling=None,
    k=8,
    max_points=200000,
    sample_size=800,
    plane_tau=0.01,
    neg_plane_tau=None,
    spatial_pos_scale=0.75,
    normal_neg_tau=0.4,
    reliability_pos_tau=0.65,
    reliability_neg_tau=0.35,
    reliability_alpha_dist=1.25,
    reliability_alpha_normal=2.0,
    reliability_alpha_plane=1.5,
    reliability_alpha_mv=1.0,
    support_cameras=None,
    support_visibility=None,
    sem_min_views=2,
    sem_conf_tau=0.7,
    sem_num_classes=None,
    sem_ignore_label=-1,
    sem_conflict_penalty=0.75,
    boundary_tau=0.45,
    eps=1e-8,
):
    zero = features.new_tensor(0.0)
    if point_ids is None:
        point_ids = torch.arange(features.shape[0], device=features.device)
    selected_indices = None
    if features.size(0) > max_points:
        selected_indices = torch.randperm(features.size(0), device=features.device)[:max_points]
        xyz = xyz[selected_indices]
        features = features[selected_indices]
        if point_ids is not None:
            point_ids = point_ids[selected_indices]
        if scaling is not None:
            scaling = scaling[selected_indices]

    num_points = features.size(0)
    if num_points < 2:
        return None

    sample_count = min(sample_size, num_points)
    effective_k = min(k, num_points - 1)
    if sample_count <= 0 or effective_k <= 0:
        return None

    raw_feature_norm = torch.norm(features, dim=-1)
    normalized_features = F.normalize(features, dim=-1, eps=eps)

    sample_indices = torch.randperm(num_points, device=features.device)[:sample_count]
    sample_point_ids = point_ids[sample_indices]
    sample_xyz = xyz[sample_indices]
    sample_features = normalized_features[sample_indices]

    dists = torch.cdist(sample_xyz, xyz)
    _, neighbor_indices = dists.topk(effective_k + 1, largest=False)
    neighbor_indices = neighbor_indices[:, 1:]
    neighbor_dists = torch.take_along_dim(dists, neighbor_indices, dim=1)
    neighbor_xyz = xyz[neighbor_indices]
    neighbor_features = normalized_features[neighbor_indices]

    flat_point_ids = torch.cat([sample_indices, neighbor_indices.reshape(-1)], dim=0)
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
    local_radius = neighbor_dists[:, -1:].clamp_min(eps)
    plane_residual = torch.abs((rel_xyz * sample_normals.unsqueeze(1)).sum(dim=-1))
    plane_ratio = plane_residual / local_radius
    cosine_sim = (neighbor_features * sample_features.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)
    normal_cosine = torch.abs((sample_normals.unsqueeze(1) * neighbor_normals).sum(dim=-1)).clamp(0.0, 1.0)
    spatial_ratio = neighbor_dists / local_radius
    spatial_weight = torch.exp(-spatial_ratio)
    spatial_near_mask = (spatial_ratio <= spatial_pos_scale).to(cosine_sim.dtype)

    if neg_plane_tau is None or neg_plane_tau <= plane_tau:
        neg_plane_ratio_tau = plane_tau / local_radius.mean().clamp_min(eps)
    else:
        neg_plane_ratio_tau = neg_plane_tau / local_radius.mean().clamp_min(eps)

    avg_sem_valid_views = zero
    avg_sem_confidence = zero
    sem_same_mask = torch.zeros_like(cosine_sim)
    sem_diff_mask = torch.zeros_like(cosine_sim)
    sem_pair_conf = torch.zeros_like(cosine_sim)
    mv_pair_term = torch.zeros_like(cosine_sim)
    sample_sem_label = torch.full((sample_count,), -1, device=features.device, dtype=torch.long)
    sample_sem_valid = torch.zeros((sample_count,), device=features.device, dtype=torch.bool)
    sample_mv_confidence = torch.zeros((sample_count,), device=features.device)
    sample_valid_view_count = torch.zeros((sample_count,), device=features.device)

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
            sample_sem_label = point_sem_label[sample_local]
            sample_sem_valid = point_sem_valid[sample_local]
            sample_mv_confidence = point_sem_confidence[sample_local]
            sample_valid_view_count = point_valid_view_count[sample_local]

            neighbor_local_flat = neighbor_local.reshape(-1)
            neighbor_labels = point_sem_label[neighbor_local_flat].view(sample_count, effective_k)
            neighbor_valid = point_sem_valid[neighbor_local_flat].view(sample_count, effective_k)
            neighbor_confidence = point_sem_confidence[neighbor_local_flat].view(sample_count, effective_k)

            sem_pair_valid = sample_sem_valid.unsqueeze(-1) & neighbor_valid
            sem_same_mask = (sem_pair_valid & (sample_sem_label.unsqueeze(-1) == neighbor_labels)).to(cosine_sim.dtype)
            sem_diff_mask = (sem_pair_valid & (sample_sem_label.unsqueeze(-1) != neighbor_labels)).to(cosine_sim.dtype)
            sem_pair_conf = sample_mv_confidence.unsqueeze(-1) * neighbor_confidence
            mv_pair_term = sem_same_mask * sem_pair_conf - sem_diff_mask * sem_pair_conf * sem_conflict_penalty

            sem_valid_point_count = point_sem_valid.to(cosine_sim.dtype).sum()
            avg_sem_valid_views = (point_sem_valid.to(point_valid_view_count.dtype) * point_valid_view_count).sum() / (sem_valid_point_count + eps)
            avg_sem_confidence = (point_sem_valid.to(point_sem_confidence.dtype) * point_sem_confidence).sum() / (sem_valid_point_count + eps)

    reliability_logit = (
        -float(reliability_alpha_dist) * spatial_ratio
        + float(reliability_alpha_normal) * normal_cosine
        - float(reliability_alpha_plane) * plane_ratio
        + float(reliability_alpha_mv) * mv_pair_term
    )
    reliability = torch.sigmoid(reliability_logit)

    semantic_pos_gate = torch.where(
        (sem_same_mask + sem_diff_mask) > 0,
        sem_same_mask,
        torch.ones_like(sem_same_mask),
    )
    pos_mask = spatial_near_mask * (reliability >= reliability_pos_tau).to(cosine_sim.dtype) * semantic_pos_gate
    close_neg_mask = spatial_near_mask * (reliability <= reliability_neg_tau).to(cosine_sim.dtype)
    geom_break_mask = (
        (plane_ratio >= float(neg_plane_ratio_tau)).to(cosine_sim.dtype)
        + (normal_cosine <= normal_neg_tau).to(cosine_sim.dtype)
    )
    semantic_neg_mask = spatial_near_mask * sem_diff_mask * (sem_pair_conf >= sem_conf_tau).to(cosine_sim.dtype)
    neg_candidate_mask = torch.clamp(close_neg_mask + semantic_neg_mask + geom_break_mask * (1.0 - pos_mask), min=0.0, max=1.0)
    ignore_mask = torch.clamp(1.0 - torch.clamp(pos_mask + neg_candidate_mask, max=1.0), min=0.0, max=1.0)

    near_weight = spatial_weight * torch.clamp(spatial_near_mask + 0.25 * (1.0 - spatial_near_mask), max=1.0)
    point_reliability = (reliability * near_weight).sum(dim=1) / (near_weight.sum(dim=1) + eps)
    point_plane_ratio = (plane_ratio * near_weight).sum(dim=1) / (near_weight.sum(dim=1) + eps)
    point_boundary_score = 1.0 - point_reliability
    point_boundary_mask = point_reliability <= boundary_tau

    if scaling is not None:
        sample_scaling = scaling[sample_indices]
        scale_score = sample_scaling.max(dim=-1).values / (sample_scaling.mean(dim=-1) + eps)
        scale_score = scale_score / (scale_score.mean().clamp_min(eps))
    else:
        sample_scaling = torch.zeros((sample_count, 3), device=features.device)
        scale_score = torch.zeros((sample_count,), device=features.device)

    return {
        "sample_point_ids": sample_point_ids,
        "sample_xyz": sample_xyz,
        "sample_normals": sample_normals,
        "sample_scaling": sample_scaling,
        "sample_features": sample_features,
        "neighbor_features": neighbor_features,
        "cosine_sim": cosine_sim,
        "normal_cosine": normal_cosine,
        "plane_residual": plane_residual,
        "spatial_weight": spatial_weight,
        "reliability": reliability,
        "pos_mask": pos_mask,
        "neg_candidate_mask": neg_candidate_mask,
        "ignore_mask": ignore_mask,
        "sem_same_mask": sem_same_mask,
        "sem_diff_mask": sem_diff_mask,
        "sem_pair_conf": sem_pair_conf,
        "semantic_pos_factor": 1.0 + sem_same_mask * sem_pair_conf,
        "semantic_neg_factor": 1.0 + sem_diff_mask * sem_pair_conf,
        "raw_feature_norm": raw_feature_norm,
        "point_reliability": point_reliability,
        "point_plane_ratio": point_plane_ratio,
        "point_boundary_score": point_boundary_score,
        "point_boundary_mask": point_boundary_mask,
        "sample_sem_label": sample_sem_label,
        "sample_sem_valid": sample_sem_valid,
        "sample_mv_confidence": sample_mv_confidence,
        "sample_valid_view_count": sample_valid_view_count,
        "sample_scale_score": scale_score,
        "avg_sem_valid_views": avg_sem_valid_views,
        "avg_sem_confidence": avg_sem_confidence,
        "boundary_tau": boundary_tau,
        "zero": zero,
    }


def loss_graph_contrastive(
    graph_state,
    lambda_val=1.0,
    lambda_pos=1.0,
    lambda_neg=1.0,
    neg_margin=0.2,
    hard_neg_k=2,
    eps=1e-8,
):
    if graph_state is None:
        return None, {}

    zero = graph_state["zero"]
    cosine_sim = graph_state["cosine_sim"]
    normal_cosine = graph_state["normal_cosine"]
    pos_mask = graph_state["pos_mask"]
    neg_candidate_mask = graph_state["neg_candidate_mask"]
    ignore_mask = graph_state["ignore_mask"]
    reliability = graph_state["reliability"]
    spatial_weight = graph_state["spatial_weight"]
    semantic_pos_factor = graph_state["semantic_pos_factor"]
    semantic_neg_factor = graph_state["semantic_neg_factor"]
    sem_diff_mask = graph_state["sem_diff_mask"]
    sem_pair_conf = graph_state["sem_pair_conf"]

    pos_count = pos_mask.sum()
    neg_candidate_count = neg_candidate_mask.sum()
    ignore_count = ignore_mask.sum()

    pos_weight = pos_mask * spatial_weight * reliability.clamp_min(0.05) * semantic_pos_factor
    pos_loss = (pos_weight * (1.0 - cosine_sim)).sum() / (pos_weight.sum() + eps)

    effective_hard_neg_k = min(max(int(hard_neg_k), 0), neg_candidate_mask.shape[1])
    if effective_hard_neg_k > 0:
        hard_neg_scores = (
            cosine_sim
            + 0.5 * (1.0 - normal_cosine)
            + sem_diff_mask * sem_pair_conf
            + (1.0 - reliability)
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
    neg_weight = active_hard_neg_mask * (1.0 - reliability).clamp_min(0.05) * semantic_neg_factor
    neg_loss = (neg_weight * (neg_term ** 2)).sum() / (neg_weight.sum() + eps)

    total_loss = lambda_val * (lambda_pos * pos_loss + lambda_neg * neg_loss)
    metrics = {
        "loss": total_loss,
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "gate_ratio": pos_mask.mean(),
        "avg_plane_residual": graph_state["plane_residual"].mean(),
        "avg_feature_norm": graph_state["raw_feature_norm"].mean(),
        "avg_normal_cosine": (pos_mask * normal_cosine).sum() / (pos_count + eps),
        "avg_pos_cosine": (pos_mask * cosine_sim).sum() / (pos_count + eps),
        "avg_neg_cosine": (neg_candidate_mask * cosine_sim).sum() / (neg_candidate_count + eps),
        "avg_hard_neg_cosine": (hard_neg_mask * cosine_sim).sum() / (hard_neg_count + eps),
        "active_neg_ratio": active_hard_neg_count / (hard_neg_count + eps),
        "neg_candidate_ratio": neg_candidate_count / (pos_mask.numel() + eps),
        "ignore_ratio": ignore_count / (pos_mask.numel() + eps),
        "avg_sem_valid_views": graph_state["avg_sem_valid_views"],
        "avg_sem_confidence": graph_state["avg_sem_confidence"],
        "semantic_pos_keep_ratio": (pos_mask * semantic_pos_factor).sum() / (pos_count + eps),
        "semantic_neg_keep_ratio": (neg_candidate_mask * semantic_neg_factor).sum() / (neg_candidate_count + eps),
        "avg_reliability": reliability.mean(),
        "avg_point_reliability": graph_state["point_reliability"].mean(),
        "boundary_ratio": graph_state["point_boundary_mask"].to(cosine_sim.dtype).mean(),
    }
    return total_loss, metrics


def loss_object_prototype(
    graph_state,
    prototype_state,
    temperature=0.2,
    bank_conf_tau=0.65,
    bank_momentum=0.9,
    lambda_pull=1.0,
    lambda_sep=0.25,
    lambda_cons=0.5,
    lambda_soft=0.1,
    sep_margin=0.1,
    min_proto_points=4,
    active_count_tau=16,
    max_active_prototypes=16,
    update_reliability_tau=0.55,
    update_entropy_tau=0.45,
    bootstrap_slots=4,
    bootstrap_novelty_tau=0.9,
    ambiguity_beta_entropy=1.0,
    ambiguity_beta_mv=1.0,
    ambiguity_beta_rel=1.0,
    ambiguity_beta_plane=0.5,
    ambiguity_beta_scale=0.25,
    eps=1e-8,
):
    if graph_state is None or prototype_state is None:
        return None, {}

    zero = graph_state["zero"]
    sample_features = graph_state["sample_features"]
    sample_point_ids = graph_state["sample_point_ids"]
    sample_normals = graph_state["sample_normals"]
    if sample_features.numel() == 0:
        return zero, {}

    point_reliability = graph_state["point_reliability"]
    sample_mv_confidence = graph_state["sample_mv_confidence"]
    sample_sem_label = graph_state["sample_sem_label"]
    sample_sem_valid = graph_state["sample_sem_valid"]
    point_boundary_mask = graph_state["point_boundary_mask"]

    seed_confidence = sample_mv_confidence * point_reliability
    seed_mask = sample_sem_valid & (~point_boundary_mask) & (point_reliability >= update_reliability_tau)
    bootstrap_count = bootstrap_scene_prototypes(
        prototype_state=prototype_state,
        features=sample_features,
        seed_confidence=seed_confidence,
        seed_mask=seed_mask,
        seed_labels=sample_sem_label,
        min_points=min_proto_points,
        max_new_slots=bootstrap_slots,
        novelty_tau=bootstrap_novelty_tau,
        eps=eps,
    )

    update_ids = get_valid_prototype_ids(prototype_state, max_active=max_active_prototypes)
    split_top2_ids = torch.full((sample_features.shape[0], 2), -1, device=sample_features.device, dtype=torch.long)
    split_top2_probs = sample_features.new_zeros((sample_features.shape[0], 2))
    if update_ids.numel() == 0:
        metrics = {
            "loss": zero,
            "pull_loss": zero,
            "sep_loss": zero,
            "cons_loss": zero,
            "soft_loss": zero,
            "active_proto_count": zero,
            "updated_proto_count": zero,
            "update_ratio": zero,
            "soft_ratio": zero,
            "avg_update_confidence": zero,
            "avg_entropy": zero,
            "avg_pos_similarity": zero,
            "avg_neg_similarity": zero,
            "avg_margin": zero,
            "avg_ambiguity": zero,
            "bootstrap_count": zero.new_tensor(float(bootstrap_count)),
            "boundary_ratio": point_boundary_mask.to(zero.dtype).mean(),
            "avg_mv_confidence": sample_mv_confidence.mean(),
            "avg_reliability": point_reliability.mean(),
            "split_point_ids": sample_point_ids.detach(),
            "split_normals": sample_normals.detach(),
            "split_top2_ids": split_top2_ids,
            "split_top2_probs": split_top2_probs,
            "split_entropy": torch.zeros_like(point_reliability),
            "split_margin": torch.zeros_like(point_reliability),
            "split_ambiguity": torch.zeros_like(point_reliability),
            "split_boundary_score": graph_state["point_boundary_score"].detach(),
            "split_boundary_mask": point_boundary_mask.detach(),
            "split_reliability": point_reliability.detach(),
            "split_mv_confidence": sample_mv_confidence.detach(),
            "split_plane_ratio": graph_state["point_plane_ratio"].detach(),
            "split_scale_score": graph_state["sample_scale_score"].detach(),
        }
        return zero, metrics

    _, probs, entropy, margin = compute_slot_assignments(
        sample_features,
        prototype_state=prototype_state,
        active_ids=update_ids,
        temperature=temperature,
        eps=eps,
    )
    if probs.shape[1] > 0:
        topk = torch.topk(probs, k=min(2, probs.shape[1]), dim=-1)
        split_top2_probs[:, :topk.values.shape[1]] = topk.values.detach()
        split_top2_ids[:, :topk.indices.shape[1]] = update_ids[topk.indices].detach()
    update_confidence = (1.0 - entropy.detach()) * point_reliability.detach() * sample_mv_confidence.detach()
    update_mask = (
        sample_sem_valid
        & (~point_boundary_mask)
        & (point_reliability >= update_reliability_tau)
        & (entropy.detach() <= update_entropy_tau)
        & (update_confidence >= bank_conf_tau)
    )
    updated_proto_count = update_scene_prototypes(
        prototype_state=prototype_state,
        features=sample_features,
        probs=probs,
        update_confidence=update_confidence,
        update_mask=update_mask,
        active_ids=update_ids,
        momentum=bank_momentum,
        min_weight=max(float(min_proto_points), 1.0),
        eps=eps,
    )

    active_ids = get_active_prototype_ids(prototype_state, min_mass=active_count_tau, max_active=max_active_prototypes)
    if active_ids.numel() == 0:
        metrics = {
            "loss": zero,
            "pull_loss": zero,
            "sep_loss": zero,
            "cons_loss": zero,
            "soft_loss": zero,
            "active_proto_count": zero,
            "updated_proto_count": zero.new_tensor(float(updated_proto_count)),
            "update_ratio": update_mask.to(zero.dtype).mean(),
            "soft_ratio": zero,
            "avg_update_confidence": update_confidence[update_mask].mean() if update_mask.any() else zero,
            "avg_entropy": entropy.mean(),
            "avg_pos_similarity": zero,
            "avg_neg_similarity": zero,
            "avg_margin": margin.mean(),
            "avg_ambiguity": zero,
            "bootstrap_count": zero.new_tensor(float(bootstrap_count)),
            "boundary_ratio": point_boundary_mask.to(zero.dtype).mean(),
            "avg_mv_confidence": sample_mv_confidence.mean(),
            "avg_reliability": point_reliability.mean(),
            "split_point_ids": sample_point_ids.detach(),
            "split_normals": sample_normals.detach(),
            "split_top2_ids": split_top2_ids,
            "split_top2_probs": split_top2_probs,
            "split_entropy": entropy.detach(),
            "split_margin": margin.detach(),
            "split_ambiguity": torch.zeros_like(entropy),
            "split_boundary_score": graph_state["point_boundary_score"].detach(),
            "split_boundary_mask": point_boundary_mask.detach(),
            "split_reliability": point_reliability.detach(),
            "split_mv_confidence": sample_mv_confidence.detach(),
            "split_plane_ratio": graph_state["point_plane_ratio"].detach(),
            "split_scale_score": graph_state["sample_scale_score"].detach(),
        }
        return zero, metrics
    _, probs, entropy, margin = compute_slot_assignments(
        sample_features,
        prototype_state=prototype_state,
        active_ids=active_ids,
        temperature=temperature,
        eps=eps,
    )
    if active_ids.numel() == 0 or probs.numel() == 0:
        return zero, {}
    topk = torch.topk(probs, k=min(2, probs.shape[1]), dim=-1)
    split_top2_probs = split_top2_probs.zero_()
    split_top2_ids = split_top2_ids.fill_(-1)
    split_top2_probs[:, :topk.values.shape[1]] = topk.values.detach()
    split_top2_ids[:, :topk.indices.shape[1]] = active_ids[topk.indices].detach()

    active_prototypes = F.normalize(prototype_state["bank"][active_ids], dim=-1, eps=eps)
    similarity = torch.matmul(sample_features, active_prototypes.transpose(0, 1)).clamp(-1.0, 1.0)
    top1_similarity, _ = similarity.max(dim=-1)

    pull_weight = point_reliability * torch.clamp(sample_mv_confidence, min=0.1)
    pull_loss = ((1.0 - top1_similarity) * pull_weight).sum() / (pull_weight.sum() + eps)

    soft_mask = sample_sem_valid & (~update_mask)
    soft_weight = sample_mv_confidence[soft_mask] * graph_state["point_boundary_score"][soft_mask]
    soft_loss = zero
    if soft_mask.any():
        proto_mix = torch.matmul(probs[soft_mask], active_prototypes)
        proto_mix = F.normalize(proto_mix, dim=-1, eps=eps)
        soft_similarity = (sample_features[soft_mask] * proto_mix).sum(dim=-1).clamp(-1.0, 1.0)
        soft_loss = ((1.0 - soft_similarity) * soft_weight).sum() / (soft_weight.sum() + eps)

    sep_loss = zero
    avg_neg_similarity = zero
    if active_prototypes.shape[0] > 1:
        proto_cos = torch.matmul(active_prototypes, active_prototypes.transpose(0, 1))
        off_diag_mask = ~torch.eye(active_prototypes.shape[0], device=active_prototypes.device, dtype=torch.bool)
        off_diag = proto_cos[off_diag_mask]
        avg_neg_similarity = off_diag.mean()
        sep_loss = torch.clamp(off_diag - sep_margin, min=0.0).pow(2).mean()

    flat_neighbor_features = graph_state["neighbor_features"].reshape(-1, graph_state["neighbor_features"].shape[-1])
    _, neighbor_probs, _, _ = compute_slot_assignments(
        flat_neighbor_features,
        prototype_state=prototype_state,
        active_ids=active_ids,
        temperature=temperature,
        eps=eps,
    )
    neighbor_probs = neighbor_probs.view(graph_state["neighbor_features"].shape[0], graph_state["neighbor_features"].shape[1], -1)
    cons_weight = graph_state["pos_mask"] * graph_state["reliability"]
    cons_term = (probs.unsqueeze(1) - neighbor_probs).pow(2).sum(dim=-1)
    cons_loss = (cons_weight * cons_term).sum() / (cons_weight.sum() + eps)

    ambiguity = (
        ambiguity_beta_entropy * entropy
        + ambiguity_beta_mv * (1.0 - sample_mv_confidence)
        + ambiguity_beta_rel * (1.0 - point_reliability)
        + ambiguity_beta_plane * graph_state["point_plane_ratio"]
        + ambiguity_beta_scale * graph_state["sample_scale_score"]
    )

    total_loss = (
        lambda_pull * pull_loss
        + lambda_sep * sep_loss
        + lambda_cons * cons_loss
        + lambda_soft * soft_loss
    )
    metrics = {
        "loss": total_loss,
        "pull_loss": pull_loss,
        "sep_loss": sep_loss,
        "cons_loss": cons_loss,
        "soft_loss": soft_loss,
        "active_proto_count": zero.new_tensor(float(active_ids.numel())),
        "updated_proto_count": zero.new_tensor(float(updated_proto_count)),
        "update_ratio": update_mask.to(zero.dtype).mean(),
        "soft_ratio": soft_mask.to(zero.dtype).mean(),
        "avg_update_confidence": update_confidence[update_mask].mean() if update_mask.any() else zero,
        "avg_entropy": entropy.mean(),
        "avg_pos_similarity": top1_similarity.mean(),
        "avg_neg_similarity": avg_neg_similarity,
        "avg_margin": margin.mean(),
        "avg_ambiguity": ambiguity.mean(),
        "bootstrap_count": zero.new_tensor(float(bootstrap_count)),
        "boundary_ratio": point_boundary_mask.to(zero.dtype).mean(),
        "avg_mv_confidence": sample_mv_confidence.mean(),
        "avg_reliability": point_reliability.mean(),
        "split_point_ids": sample_point_ids.detach(),
        "split_normals": sample_normals.detach(),
        "split_top2_ids": split_top2_ids,
        "split_top2_probs": split_top2_probs,
        "split_entropy": entropy.detach(),
        "split_margin": margin.detach(),
        "split_ambiguity": ambiguity.detach(),
        "split_boundary_score": graph_state["point_boundary_score"].detach(),
        "split_boundary_mask": point_boundary_mask.detach(),
        "split_reliability": point_reliability.detach(),
        "split_mv_confidence": sample_mv_confidence.detach(),
        "split_plane_ratio": graph_state["point_plane_ratio"].detach(),
        "split_scale_score": graph_state["sample_scale_score"].detach(),
    }
    return total_loss, metrics
