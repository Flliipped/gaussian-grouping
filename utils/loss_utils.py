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

import torch
import torch.nn.functional as F


def estimate_normals_for_query_points(
    xyz_all,
    query_idx,
    k=8,
    ref_size=4096,
    eps=1e-8,
):
    """
    xyz_all: [N, 3]
    query_idx: [Q]
    return: normals [Q, 3]
    """
    device = xyz_all.device
    N = xyz_all.size(0)
    Q = query_idx.numel()

    # 1) 构造 reference subset
    if N <= ref_size:
        ref_idx = torch.arange(N, device=device)
    else:
        rand_idx = torch.randperm(N, device=device)[:ref_size]
        ref_idx = torch.unique(torch.cat([rand_idx, query_idx], dim=0), sorted=False)

        # 如果拼起来还是太大，就再裁一下，但尽量保留 query
        if ref_idx.numel() > ref_size + Q:
            extra = ref_idx[Q:] if ref_idx.numel() > Q else ref_idx[:0]
            keep_extra = extra[:max(0, ref_size)]
            ref_idx = torch.unique(torch.cat([query_idx, keep_extra], dim=0), sorted=False)

    xyz_query = xyz_all[query_idx]   # [Q, 3]
    xyz_ref = xyz_all[ref_idx]       # [R, 3]

    # 2) query -> ref 距离
    dist = torch.cdist(xyz_query, xyz_ref)   # [Q, R]

    # 3) 如果 query 点自己也在 ref 里，需要排除自身
    ref_map = -torch.ones(N, device=device, dtype=torch.long)
    ref_map[ref_idx] = torch.arange(ref_idx.numel(), device=device)

    self_pos = ref_map[query_idx]   # [Q]
    valid_self = self_pos >= 0
    if valid_self.any():
        dist[torch.arange(Q, device=device)[valid_self], self_pos[valid_self]] = 1e10

    # 4) KNN on ref
    k_eff = min(k, max(1, ref_idx.numel() - 1))
    knn_ref_pos = dist.topk(k_eff, largest=False).indices   # [Q, k_eff]
    knn_idx = ref_idx[knn_ref_pos]                          # [Q, k_eff]

    knn_xyz = xyz_all[knn_idx]                              # [Q, k_eff, 3]
    local = knn_xyz - xyz_query.unsqueeze(1)               # [Q, k_eff, 3]

    cov = torch.matmul(local.transpose(1, 2), local) / (k_eff + eps)   # [Q, 3, 3]
    _, eigvecs = torch.linalg.eigh(cov)
    normals = F.normalize(eigvecs[:, :, 0], dim=-1)        # [Q, 3]

    return normals

def loss_geo_smooth(
    xyz,
    obj_feat,
    k=8,
    plane_tau1=0.35,
    plane_tau2=0.75,
    normal_tau1=0.90,
    normal_tau2=0.75,
    weak_weight=0.3,
    weight_lambda=5.0,
    sample_size=800,
    eps=1e-8,
): 
    """
    xyz: [N, 3]
    obj_feat: [N, D]
    """
    N = xyz.size(0)
    k = min(k, max(1, N - 1))

    # 随机采样中心点
    if N > sample_size:
        indices = torch.randperm(N, device=xyz.device)[:sample_size]
        xyz_sub = xyz[indices]         # [M, 3]
        feat_sub = obj_feat[indices]   # [M, D]
    else:
        indices = None
        xyz_sub = xyz
        feat_sub = obj_feat

    M = xyz_sub.size(0)

    # [M, N]
    dist = torch.cdist(xyz_sub, xyz)

    # 排除自己
    if indices is not None:
        dist[torch.arange(M, device=xyz.device), indices] = 1e10
    else:
        eye_mask = torch.eye(M, device=xyz.device, dtype=torch.bool)
        dist[eye_mask] = 1e10

    # [M, k]
    neighbor_indices_tensor = dist.topk(k, largest=False).indices

    if indices is not None:
        touched_idx = torch.cat([indices, neighbor_indices_tensor.reshape(-1)], dim=0)
    else:
        center_idx = torch.arange(M, device=xyz.device)
        touched_idx = torch.cat([center_idx, neighbor_indices_tensor.reshape(-1)], dim=0)

    touched_unique = torch.unique(touched_idx, sorted=False)
    normals_touched = estimate_normals_for_query_points(xyz, touched_unique, k=k, ref_size=4096, eps=eps)

    index_map = -torch.ones(N, device=xyz.device, dtype=torch.long)
    index_map[touched_unique] = torch.arange(touched_unique.numel(), device=xyz.device)

    if indices is not None:
        normals_sub = normals_touched[index_map[indices]]          # [M, 3]
    else:
        center_idx = torch.arange(M, device=xyz.device)
        normals_sub = normals_touched[index_map[center_idx]]       # [M, 3]
    normals_knn = normals_touched[index_map[neighbor_indices_tensor]]  # [M*k, 3] -> [M, k, 3]

    normal_sim = torch.abs(torch.sum(normals_sub.unsqueeze(1) * normals_knn, dim=-1))  # [M, k]
    # 邻域坐标和特征
    knn_xyz = xyz[neighbor_indices_tensor]          # [M, k, 3]
    feat_knn = obj_feat[neighbor_indices_tensor]    # [M, k, D]

    # 局部坐标
    local = knn_xyz - xyz_sub.unsqueeze(1)          # [M, k, 3]

    # 点到中心点切平面的距离
    plane_residual = torch.abs(
        torch.sum(local * normals_sub.unsqueeze(1), dim=-1)
    )   # [M, k]

    local_scale = local.norm(dim=-1).mean(dim=-1, keepdim=True)  # [M, 1] 
    plane_residual_norm = plane_residual / (local_scale + eps)

    strong_mask = (plane_residual_norm < plane_tau1) & (normal_sim > normal_tau1)
    weak_mask = (plane_residual_norm < plane_tau2) & (normal_sim > normal_tau2) & (~strong_mask)

    weights = strong_mask.float() + weak_mask.float() * weak_weight
    weights = weights * torch.exp(-weight_lambda * plane_residual_norm)

    feat_diff = feat_sub.unsqueeze(1) - feat_knn
    feat_dist2 = torch.sum(feat_diff ** 2, dim=-1)

    loss = (weights * feat_dist2).sum() / (weights.sum() + eps)

    strong_ratio = strong_mask.float().mean().item()
    weak_ratio = weak_mask.float().mean().item()
    avg_plane_residual = plane_residual.mean().item()
    avg_plane_residual_norm = plane_residual_norm.mean().item()
    avg_normal_sim = normal_sim.mean().item()

    return loss, strong_ratio, weak_ratio, avg_plane_residual, avg_plane_residual_norm, avg_normal_sim

def loss_geo_contrastive_v2_1_posonly(xyz, obj_feat, obj_label, k=8, plane_tau=0.01, smooth_weight_lambda=5.0, margin=1.0, sample_size=800, con_weight=0.1, neg_cap=2,eps=1e-8):
    N = xyz.size(0)
    k = min(k, max(1, N-1))

    # 随机采样中心点
    if N > sample_size:
        indices = torch.randperm(N, device=xyz.device)[:sample_size]
        xyz_sub = xyz[indices]
        feat_sub = obj_feat[indices]
        label_sub = obj_label[indices]
    else:
        indices = None
        xyz_sub = xyz
        feat_sub = obj_feat
        label_sub = obj_label
    
    M = xyz_sub.size(0)

    # 2.knn
    dist =  torch.cdist(xyz_sub, xyz) # [M, N]

    if indices is not None:
        dist[torch.arange(M, device=xyz.device), indices] = 1e10
    else:
        eye_mask = torch.eye(M, device=xyz.device, dtype=torch.bool)
        dist[eye_mask] = 1e10

    neighbor_idx = dist.topk(k, largest=False).indices # [M, k]

    knn_xyz = xyz[neighbor_idx]          # [M, k, 3]
    feat_knn = obj_feat[neighbor_idx]    # [M, k, D]
    label_knn = obj_label[neighbor_idx]  # [M, k]

    # 3. 用同一批邻域估中心点法线
    local = knn_xyz - xyz_sub.unsqueeze(1)          # [M, k, 3]
    cov = torch.matmul(local.transpose(1, 2), local) / (k + eps)   # [M, 3, 3]
    _, eigvecs = torch.linalg.eigh(cov)
    normals_sub = F.normalize(eigvecs[:, :, 0], dim=-1)           # [M, 3]

    # 4.几何门控：邻居点到中心点切平面的残差
    plane_residual = torch.abs(
        torch.sum(local * normals_sub.unsqueeze(1), dim=-1)
    ) # [M, k]

    gate_mask = plane_residual < plane_tau

    # 5.当前类别一致性
    same_label = label_knn == label_sub.unsqueeze(1) # [M, k]
    
    # -------------------
    # A.已验证有效的smooth主项
    # -------------------
    smooth_weights = torch.exp(-smooth_weight_lambda * plane_residual) * gate_mask.float()

    feat_diff_raw = feat_sub.unsqueeze(1) - feat_knn
    feat_dist2_raw = torch.sum(feat_diff_raw ** 2, dim=-1)  # [M, k]

    loss_smooth = (smooth_weights * feat_dist2_raw).sum() / (smooth_weights.sum() + eps)
    
    # -------------------
    # B.保守的contrastive辅项
    # -------------------

    # 正样本：几何连续 + 类别一致
    pos_mask = gate_mask & same_label
    
    # 负样本：几何不连续 + 类别不一致
    neg_mask = (~gate_mask) & (~same_label)

    # 每个anchor最多保留neg_cap个负样本，防止负样本过强
    if neg_cap is not None and neg_cap >0:
        neg_mask_limited = torch.zeros_like(neg_mask)
        for i in range(M):
            neg_idx_i = torch.nonzero(neg_mask[i], as_tuple=False).squeeze(-1)
            if neg_idx_i.numel()>0:
                if neg_idx_i.numel() > neg_cap:
                    perm = torch.randperm(neg_idx_i.numel(), device=xyz.device)[:neg_cap]
                    neg_idx_i = neg_idx_i[perm]
                neg_mask_limited[i, neg_idx_i] = True
        neg_mask = neg_mask_limited

    # 6.归一化feature，用平方L2做contrastive
    feat_sub_n = F.normalize(feat_sub, dim=-1)
    feat_knn_n = F.normalize(feat_knn, dim=-1)

    feat_diff_n = feat_sub_n.unsqueeze(1) - feat_knn_n
    feat_dist2_n = torch.sum(feat_diff_n ** 2, dim=-1)

    # 7. 正样本：拉近
    pos_weight = torch.exp(-smooth_weight_lambda * plane_residual) * pos_mask.float()
    loss_pos = (pos_weight * feat_dist2_n).sum() / (pos_weight.sum() + eps)

    # 8. 负样本：推远
    # 只对少量保守负样本做推远，距离至少大于margin
    neg_hinge = F.relu(margin - feat_dist2_n)
    loss_neg = (neg_hinge * neg_mask.float()).sum() / (neg_mask.float().sum() + eps)

    loss_con = loss_pos # + loss_neg

    # 总loss:smooth为主，contrastive为辅
    loss_total = loss_smooth + con_weight * loss_con

    pos_ratio = pos_mask.float().mean().item()
    neg_ratio = neg_mask.float().mean().item()
    gate_ratio = gate_mask.float().mean().item()
    avg_plane_residual = plane_residual.mean().item()

    return loss_total, loss_smooth.item(), loss_con.item(), pos_ratio, neg_ratio, gate_ratio, avg_plane_residual


def loss_geo_contrastive_v1(
    xyz,
    obj_feat,
    obj_label,
    k=8,
    plane_tau=0.01,
    pos_weight_lambda=5.0,
    margin=1.0,
    sample_size=800,
    eps=1e-8,
):
    """
    xyz: [N, 3]
    obj_feat: [N, D]
    obj_label: [N]   当前 3D 预测类别（由 classifier 给出）
    """
    N = xyz.size(0)
    k = min(k, max(1, N - 1))

    # 1. 随机采样中心点
    if N > sample_size:
        indices = torch.randperm(N, device=xyz.device)[:sample_size]
        xyz_sub = xyz[indices]            # [M, 3]
        feat_sub = obj_feat[indices]      # [M, D]
        label_sub = obj_label[indices]    # [M]
    else:
        indices = None
        xyz_sub = xyz
        feat_sub = obj_feat
        label_sub = obj_label

    M = xyz_sub.size(0)

    # 2. KNN
    dist = torch.cdist(xyz_sub, xyz)      # [M, N]

    if indices is not None:
        dist[torch.arange(M, device=xyz.device), indices] = 1e10
    else:
        eye_mask = torch.eye(M, device=xyz.device, dtype=torch.bool)
        dist[eye_mask] = 1e10

    neighbor_idx = dist.topk(k, largest=False).indices   # [M, k]

    knn_xyz = xyz[neighbor_idx]                          # [M, k, 3]
    feat_knn = obj_feat[neighbor_idx]                    # [M, k, D]
    label_knn = obj_label[neighbor_idx]                  # [M, k]

    # 3. 用同一批邻域估中心点法线
    local = knn_xyz - xyz_sub.unsqueeze(1)               # [M, k, 3]
    cov = torch.matmul(local.transpose(1, 2), local) / (k + eps)   # [M, 3, 3]
    _, eigvecs = torch.linalg.eigh(cov)
    normals_sub = F.normalize(eigvecs[:, :, 0], dim=-1)             # [M, 3]

    # 4. 几何门控：邻居点到中心点切平面的残差
    plane_residual = torch.abs(
        torch.sum(local * normals_sub.unsqueeze(1), dim=-1)
    )   # [M, k]

    gate_mask = plane_residual < plane_tau               # [M, k]

    # 5. 当前类别一致性
    same_label = label_knn == label_sub.unsqueeze(1)     # [M, k]

    # 正样本：几何连续 + 类别一致
    pos_mask = gate_mask & same_label

    # 负样本：空间近邻里类别不同
    # 第一版先这样最稳，不用把 gate 也揉进负样本
    neg_mask = ~same_label

    # 6. 归一化 feature，用平方 L2 做 contrastive
    feat_sub_n = F.normalize(feat_sub, dim=-1)                   # [M, D]
    feat_knn_n = F.normalize(feat_knn, dim=-1)                   # [M, k, D]

    feat_diff = feat_sub_n.unsqueeze(1) - feat_knn_n             # [M, k, D]
    feat_dist2 = torch.sum(feat_diff ** 2, dim=-1)               # [M, k], 范围大致 [0, 4]

    # 7. 正样本：拉近
    pos_weight = torch.exp(-pos_weight_lambda * plane_residual) * pos_mask.float()
    loss_pos = (pos_weight * feat_dist2).sum() / (pos_weight.sum() + eps)

    # 8. 负样本：推远
    # 希望负样本距离至少大于 margin
    neg_hinge = F.relu(margin - feat_dist2)
    loss_neg = (neg_hinge * neg_mask.float()).sum() / (neg_mask.sum() + eps)

    loss = loss_pos + loss_neg

    pos_ratio = pos_mask.float().mean().item()
    neg_ratio = neg_mask.float().mean().item()
    gate_ratio = gate_mask.float().mean().item()
    avg_plane_residual = plane_residual.mean().item()

    return loss, pos_ratio, neg_ratio, gate_ratio, avg_plane_residual