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



def loss_geo_smooth(xyz, obj_feat, k=5, normal_tau=0.5, weight_lambda=5.0, max_points=200000, sample_size=800):
    """
    :param xyz: Tensor of shape (N, 3) representing the 3D coordinates of the points.
    :param obj_feat: Tensor of shape (N, D) representing the features of the
    :param k: Number of nearest neighbors to consider.
    :param normal_tau: Threshold for normal similarity.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    """
    N = xyz.size(0)

    # Randomly sample points for which we'll compute the loss
    if N > sample_size:
        indices = torch.randperm(N)[:sample_size]
        xyz_sub = xyz[indices]
        feat_sub = obj_feat[indices]
    else:
        indices = None
        xyz_sub = xyz
        feat_sub = obj_feat
    
    M = xyz_sub.size(0) 

    # [M, N]
    dist = torch.cdist(xyz_sub, xyz)
    
    if indices is not None:
        dist[torch.arange(M, device=xyz.device), indices] = 1e10
    else:
        eye_mask = torch.eye(M, device=xyz.device, dtype=torch.bool)
        dist[:, :M][eye_mask] = 1e10

    # [M, k]
    _, neighbor_indices_tensor = dist.topk(k, largest=False)

    normals_all = estimate_normals_knn(xyz, k=k)
    normals_sub = normals_all if indices is not None else normals_all[indices]
    normals_neighbor = normals_all[neighbor_indices_tensor]

    feat_knn = obj_feat[neighbor_indices_tensor]

    # [M, k]
    normal_sim = torch.abs(torch.sum(normals_sub.unsqueeze(1) * normals_neighbor, dim=-1))

    gate_mask = (normal_sim > normal_tau).float()

    weights = torch.exp(-weight_lambda * (1 - normal_sim)) * gate_mask

    feat_diff = feat_sub.unsqueeze(1) - feat_knn

    feat_diff2 = torch.sum(feat_diff ** 2, dim=-1)

    loss = (weights * feat_diff2).sum() / (weights.sum() + 1e-8)

    gate_ratio = gate_mask.mean()
    avg_normal_sim = normal_sim.mean()
    
    print("xyz in geo:", xyz.shape)
    print("obj_feat in geo:", obj_feat.shape)

    return loss, gate_ratio, avg_normal_sim

def estimate_normals_knn(xyz, k=10, eps=1e-8):
    """
    xyz: [N, 3]
    return: 
        normals:[N, 3]
    """
    N = xyz.size(0)
    dist = torch.cdist(xyz, xyz)
    
    eye_mask = torch.eye(N, device=xyz.device, dtype=torch.bool)
    dist[eye_mask] = 1e10
    # [N, k]
    knn_idx = dist.topk(k, largest=False).indices
    # [N, k, 3]
    knn_xyz = xyz[knn_idx]
    # [N, 1, 3]
    center = xyz.unsqueeze(1)
    # [N, k, 3]
    local = knn_xyz - center
    # 协方差[N, 3, 3]
    cov = torch.matmul(local.transpose(1, 2), local) / (k + eps)

    #对每个点做特征分解
    # [N, 3, 3]
    eigvals, eigvecs = torch.linalg.eigh(cov)

    normals = eigvecs[:, :, 0] # [N, 3]

    normals = F.normalize(normals, dim=-1)


    return normals