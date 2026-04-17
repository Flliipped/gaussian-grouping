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
