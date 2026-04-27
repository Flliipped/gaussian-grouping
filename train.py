# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
from random import randint
from utils.loss_utils import (
    compute_graph_reliability,
    l1_loss,
    loss_cls_3d,
    loss_graph_contrastive,
    loss_prototype_learning,
    loss_sugar_surface_alignment,
    ssim,
)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.prototype_bank import ScenePrototypeBank
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json


def _proto_diag_scalar(proto_diag, key, default=0.0):
    if not proto_diag or key not in proto_diag:
        return default

    value = proto_diag[key]
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() == 0:
            return default
        if value.numel() > 1:
            value = value.float().mean()
        return value.item()
    return float(value)


def _proto_diag_tensor(proto_diag, key):
    if not proto_diag or key not in proto_diag:
        return None

    value = proto_diag[key]
    if not torch.is_tensor(value):
        return None
    value = value.detach()
    if value.numel() == 0:
        return None
    return value


def _add_proto_diag_wandb_logs(log_data, proto_diag, iteration):
    if not proto_diag:
        return

    scalar_keys = [
        "proto_dead_count",
        "proto_active_update_count",
        "proto_usage_entropy",
        "proto_update_usage_entropy",
        "proto_usage_min",
        "proto_usage_max",
        "proto_usage_std",
        "proto_update_usage_min",
        "proto_update_usage_max",
        "proto_update_usage_std",
        "proto_pair_cosine_mean",
        "proto_pair_cosine_max",
        "proto_pair_cosine_p90",
        "proto_entropy_p10",
        "proto_entropy_p50",
        "proto_entropy_p90",
        "proto_assign_conf_p10",
        "proto_assign_conf_p50",
        "proto_assign_conf_p90",
        "proto_margin_p10",
        "proto_margin_p50",
        "proto_margin_p90",
        "proto_update_selected_count",
        "proto_update_selected_ratio",
        "proto_update_confidence_p50",
        "proto_update_confidence_p90",
        "proto_push_loss",
        "proto_push_active_ratio",
        "proto_push_weight_mean",
        "proto_push_penalty_mean",
        "proto_update_boundary_weight_mean",
        "proto_update_neg_boundary_weight_mean",
        "proto_update_ignore_boundary_weight_mean",
        "proto_neg_boundary_entropy",
        "proto_neg_boundary_assign_conf",
        "proto_neg_boundary_selected_ratio",
        "proto_uncertain_boundary_entropy",
        "proto_uncertain_boundary_assign_conf",
        "proto_uncertain_boundary_selected_ratio",
    ]
    for key in scalar_keys:
        if key in proto_diag:
            log_data[f"train_loss_patches/{key}"] = _proto_diag_scalar(proto_diag, key)

    if iteration % 500 != 0:
        return

    for key in ["proto_usage_histogram", "proto_update_usage_histogram"]:
        histogram = _proto_diag_tensor(proto_diag, key)
        if histogram is not None:
            log_data[f"train_loss_patches/{key}"] = wandb.Histogram(histogram.float().cpu().numpy())

    usage_histogram = _proto_diag_tensor(proto_diag, "proto_usage_histogram")
    update_usage_histogram = _proto_diag_tensor(proto_diag, "proto_update_usage_histogram")
    if usage_histogram is not None and usage_histogram.numel() <= 32:
        for proto_idx, usage_value in enumerate(usage_histogram.float().cpu().tolist()):
            log_data[f"train_loss_patches/proto_usage_{proto_idx:02d}"] = usage_value
    if update_usage_histogram is not None and update_usage_histogram.numel() <= 32:
        for proto_idx, usage_value in enumerate(update_usage_histogram.float().cpu().tolist()):
            log_data[f"train_loss_patches/proto_update_usage_{proto_idx:02d}"] = usage_value


def _build_graph_support_cache(cameras, num_support):
    if cameras is None or len(cameras) == 0:
        return {}

    effective_support = min(max(int(num_support), 0), max(len(cameras) - 1, 0))
    if effective_support <= 0:
        return {camera.uid: [camera] for camera in cameras}

    camera_centers = torch.stack(
        [camera.camera_center.detach().float() for camera in cameras],
        dim=0,
    )
    pairwise_dist = torch.cdist(camera_centers, camera_centers)
    pairwise_dist.fill_diagonal_(float("inf"))

    support_cache = {}
    for camera_idx, camera in enumerate(cameras):
        _, neighbor_idx = torch.topk(
            pairwise_dist[camera_idx],
            k=effective_support,
            largest=False,
        )
        support_cache[camera.uid] = [camera] + [cameras[idx.item()] for idx in neighbor_idx]

    return support_cache


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    classifier.cuda()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    prototype_bank = None
    if opt.use_proto:
        prototype_bank = ScenePrototypeBank(
            num_prototypes=opt.num_prototypes,
            feature_dim=gaussians.num_objects,
            tau=opt.proto_tau,
            momentum=opt.proto_ema_momentum,
            device=background.device,
        )

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    graph_support_cache = None
    if opt.graph_use_multiview_semantics:
        graph_support_cache = _build_graph_support_cache(
            scene.getTrainCameras(),
            opt.graph_support_views,
        )
    for iteration in range(first_iter, opt.iterations + 1):
        if iteration == 1:
            print("xyz:", gaussians._xyz.shape)
            print("objects_dc:", gaussians._objects_dc.shape)
    #        print("image:", image.shape)
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

        # Object Loss
        gt_obj = viewpoint_cam.objects.cuda().long()
        logits = classifier(objects)
        loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        loss_obj = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0:
            # regularize at certain intervals
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj + loss_obj_3d
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj          

        loss_graph_raw = torch.tensor(0.0, device=image.device)
        loss_graph = torch.tensor(0.0, device=image.device)
        graph_coeff = torch.tensor(0.0, device=image.device)
        loss_proto_raw = torch.tensor(0.0, device=image.device)
        loss_proto = torch.tensor(0.0, device=image.device)
        proto_coeff = torch.tensor(0.0, device=image.device)
        loss_sugar_raw = torch.tensor(0.0, device=image.device)
        loss_sugar = torch.tensor(0.0, device=image.device)
        sugar_coeff = torch.tensor(0.0, device=image.device)
        sugar_axis_align_cosine = torch.tensor(0.0, device=image.device)
        sugar_plane_residual = torch.tensor(0.0, device=image.device)
        sugar_flat_ratio = torch.tensor(0.0, device=image.device)
        graph_pos_ratio = torch.tensor(0.0, device=image.device)
        graph_neg_ratio = torch.tensor(0.0, device=image.device)
        graph_ignore_ratio = torch.tensor(0.0, device=image.device)
        avg_reliability = torch.tensor(0.0, device=image.device)
        avg_pos_reliability = torch.tensor(0.0, device=image.device)
        avg_neg_reliability = torch.tensor(0.0, device=image.device)
        avg_mv_consistency = torch.tensor(0.0, device=image.device)
        avg_plane_residual = torch.tensor(0.0, device=image.device)
        pos_loss = torch.tensor(0.0, device=image.device)
        neg_loss = torch.tensor(0.0, device=image.device)
        avg_feature_norm = torch.tensor(0.0, device=image.device)
        avg_normal_cosine = torch.tensor(0.0, device=image.device)
        avg_pos_cosine = torch.tensor(0.0, device=image.device)
        avg_neg_cosine = torch.tensor(0.0, device=image.device)
        avg_hard_neg_cosine = torch.tensor(0.0, device=image.device)
        active_neg_ratio = torch.tensor(0.0, device=image.device)
        avg_sem_valid_views = torch.tensor(0.0, device=image.device)
        avg_sem_confidence = torch.tensor(0.0, device=image.device)
        sem_valid_point_ratio = torch.tensor(0.0, device=image.device)
        sem_pair_valid_ratio = torch.tensor(0.0, device=image.device)
        sem_high_conf_same_ratio = torch.tensor(0.0, device=image.device)
        semantic_pos_keep_ratio = torch.tensor(0.0, device=image.device)
        semantic_neg_keep_ratio = torch.tensor(0.0, device=image.device)
        proto_pull_loss = torch.tensor(0.0, device=image.device)
        proto_sep_loss = torch.tensor(0.0, device=image.device)
        proto_cons_loss = torch.tensor(0.0, device=image.device)
        proto_cons_conf_mean = torch.tensor(0.0, device=image.device)
        proto_cons_adaptive_factor_mean = torch.tensor(0.0, device=image.device)
        proto_cons_scene_scale = torch.tensor(0.0, device=image.device)
        proto_cons_agree_ratio = torch.tensor(0.0, device=image.device)
        proto_cons_agree_factor_mean = torch.tensor(0.0, device=image.device)
        proto_avg_entropy = torch.tensor(0.0, device=image.device)
        proto_avg_assign_conf = torch.tensor(0.0, device=image.device)
        proto_avg_confidence = torch.tensor(0.0, device=image.device)
        proto_confident_ratio = torch.tensor(0.0, device=image.device)
        proto_update_avg_confidence = torch.tensor(0.0, device=image.device)
        proto_update_confident_ratio = torch.tensor(0.0, device=image.device)
        proto_avg_margin = torch.tensor(0.0, device=image.device)
        proto_active_ratio = torch.tensor(0.0, device=image.device)
        proto_usage_max = torch.tensor(0.0, device=image.device)
        proto_push_loss = torch.tensor(0.0, device=image.device)
        proto_update_features = None
        proto_update_probs = None
        proto_update_confidence = None
        proto_update_sample_weight = None
        proto_diag = {}
        sugar_active = iteration >= opt.sugar_start_iter and iteration % opt.sugar_interval == 0
        graph_active = iteration >= opt.graph_start_iter and iteration % opt.graph_interval == 0
        proto_active = opt.use_proto and iteration >= opt.proto_start_iter and iteration % opt.proto_interval == 0
        graph_context_active = graph_active or proto_active
        graph_data = None

        if sugar_active:
            sugar_warmup_iters = max(1, opt.sugar_warmup_iters)
            sugar_progress = min(max((iteration - opt.sugar_start_iter) / sugar_warmup_iters, 0.0), 1.0)
            sugar_coeff = image.new_tensor(opt.sugar_weight_lambda * sugar_progress)
            loss_sugar_raw, sugar_axis_align_cosine, sugar_plane_residual, sugar_flat_ratio, _ = loss_sugar_surface_alignment(
                xyz=gaussians._xyz.squeeze(),
                scaling=gaussians.get_scaling,
                rotation=gaussians._rotation,
                k=opt.sugar_knn_k,
                lambda_val=1.0,
                lambda_axis=opt.sugar_lambda_axis,
                lambda_plane=opt.sugar_lambda_plane,
                lambda_flat=opt.sugar_lambda_flat,
                max_points=opt.sugar_max_points,
                sample_size=opt.sugar_sample_size,
            )
            loss_sugar = sugar_coeff * loss_sugar_raw
            loss = loss + loss_sugar

        if graph_context_active:
            support_cameras = None
            support_visibility = None
            if opt.graph_use_multiview_semantics:
                support_cameras = graph_support_cache.get(viewpoint_cam.uid, [viewpoint_cam])

                visibility_masks = [visibility_filter.detach()]
                if len(support_cameras) > 1:
                    with torch.no_grad():
                        for support_cam in support_cameras[1:]:
                            support_render_pkg = render(support_cam, gaussians, pipe, background)
                            visibility_masks.append(support_render_pkg["visibility_filter"].detach())
                support_visibility = torch.stack(visibility_masks, dim=0)

            graph_data = compute_graph_reliability(
                xyz=gaussians._xyz.squeeze().detach(),
                point_ids=torch.arange(gaussians._xyz.shape[0], device=gaussians._xyz.device),
                k=opt.graph_knn_k,
                max_points=opt.graph_max_points,
                sample_size=opt.graph_sample_size,
                plane_tau=opt.graph_plane_tau,
                neg_plane_tau=opt.graph_neg_plane_tau,
                spatial_pos_scale=opt.graph_spatial_pos_scale,
                normal_pos_tau=opt.graph_normal_pos_tau,
                normal_neg_tau=opt.graph_normal_neg_tau,
                support_cameras=support_cameras,
                support_visibility=support_visibility,
                sem_min_views=opt.graph_sem_min_views,
                sem_conf_tau=opt.graph_sem_conf_tau,
                sem_pos_ratio=opt.graph_sem_pos_ratio,
                sem_num_classes=num_classes,
                sem_ignore_label=opt.graph_sem_ignore_label,
                sem_same_boost=opt.graph_sem_same_boost,
                sem_neg_boost=opt.graph_sem_neg_boost,
                sem_conflict_penalty=opt.graph_sem_conflict_penalty,
                alpha_dist=opt.graph_alpha_dist,
                alpha_normal=opt.graph_alpha_normal,
                alpha_residual=opt.graph_alpha_residual,
                alpha_mv=opt.graph_alpha_mv,
                pos_reliability_thresh=opt.graph_pos_reliability_thresh,
                neg_reliability_thresh=opt.graph_neg_reliability_thresh,
            )
            sem_valid_point_ratio = graph_data.get("sem_valid_point_ratio", sem_valid_point_ratio)
            sem_pair_valid_ratio = graph_data.get("sem_pair_valid_ratio", sem_pair_valid_ratio)
            sem_high_conf_same_ratio = graph_data.get("sem_high_conf_same_ratio", sem_high_conf_same_ratio)

        if graph_active and graph_data is not None and graph_data.get("valid", False):
            warmup_iters = max(1, opt.graph_warmup_iters)
            graph_progress = min(max((iteration - opt.graph_start_iter) / warmup_iters, 0.0), 1.0)
            graph_coeff = image.new_tensor(opt.graph_weight_lambda * graph_progress)
            loss_graph_raw, graph_pos_ratio, graph_neg_ratio, graph_ignore_ratio, avg_reliability, avg_pos_reliability, avg_neg_reliability, avg_mv_consistency, avg_plane_residual, pos_loss, neg_loss, avg_feature_norm, avg_normal_cosine, avg_pos_cosine, avg_neg_cosine, avg_hard_neg_cosine, active_neg_ratio, avg_sem_valid_views, avg_sem_confidence, semantic_pos_keep_ratio, semantic_neg_keep_ratio = loss_graph_contrastive(
                features=gaussians._objects_dc.squeeze(1),
                graph_data=graph_data,
                lambda_val=1.0,
                lambda_pos=opt.graph_lambda_pos,
                lambda_neg=opt.graph_lambda_neg,
                neg_margin=opt.graph_neg_margin,
                hard_neg_k=opt.graph_hard_neg_k,
                normal_weight_lambda=opt.graph_normal_weight_lambda,
            )
            loss_graph = graph_coeff * loss_graph_raw
            loss = loss + loss_graph

        if proto_active and graph_data is not None and graph_data.get("valid", False):
            proto_warmup_iters = max(1, opt.proto_warmup_iters)
            proto_progress = min(max((iteration - opt.proto_start_iter) / proto_warmup_iters, 0.0), 1.0)
            proto_coeff = image.new_tensor(opt.proto_weight_lambda * proto_progress)
            proto_outputs = loss_prototype_learning(
                features=gaussians._objects_dc.squeeze(1),
                prototype_bank=prototype_bank,
                graph_data=graph_data,
                lambda_val=1.0,
                lambda_pull=opt.proto_lambda_pull,
                lambda_sep=opt.proto_lambda_sep,
                lambda_cons=opt.proto_lambda_cons,
                cons_conf_weight=opt.proto_cons_conf_weight,
                cons_conf_floor=opt.proto_cons_conf_floor,
                cons_conf_power=opt.proto_cons_conf_power,
                cons_conf_normalize=opt.proto_cons_conf_normalize,
                cons_conf_norm_max=opt.proto_cons_conf_norm_max,
                cons_scene_weight=opt.proto_cons_scene_weight,
                cons_scene_floor=opt.proto_cons_scene_floor,
                cons_scene_conf_min=opt.proto_cons_scene_conf_min,
                cons_scene_conf_target=opt.proto_cons_scene_conf_target,
                cons_agree_weight=opt.proto_cons_agree_weight,
                cons_agree_floor=opt.proto_cons_agree_floor,
                cons_agree_conf_thresh=opt.proto_cons_agree_conf_thresh,
                conf_thresh=opt.proto_conf_thresh,
                sep_margin=opt.proto_sep_margin,
                reliability_thresh=opt.proto_reliability_thresh,
                entropy_thresh=opt.proto_entropy_thresh,
                assign_conf_thresh=opt.proto_assign_conf_thresh,
                sem_invalid_weight=opt.proto_sem_invalid_weight,
                update_conf_thresh=opt.proto_update_conf_thresh,
                update_reliability_thresh=opt.proto_update_reliability_thresh,
                update_entropy_thresh=opt.proto_update_entropy_thresh,
                update_assign_conf_thresh=opt.proto_update_assign_conf_thresh,
                update_sem_invalid_weight=opt.proto_update_sem_invalid_weight,
                boundary_safe_update=opt.proto_boundary_safe_update,
                update_neg_boundary_weight=opt.proto_update_neg_boundary_weight,
                update_ignore_boundary_weight=opt.proto_update_ignore_boundary_weight,
                lambda_push=opt.proto_lambda_push,
                push_mode=opt.proto_push_mode,
                push_margin=opt.proto_push_margin,
                push_temperature=opt.proto_push_temperature,
                push_conf_thresh=opt.proto_push_conf_thresh,
                push_use_confident_mask=opt.proto_push_use_confident_mask,
                push_reliability_thresh=opt.proto_push_reliability_thresh,
                push_entropy_thresh=opt.proto_push_entropy_thresh,
                push_assign_conf_thresh=opt.proto_push_assign_conf_thresh,
                push_neg_boundary_weight=opt.proto_push_neg_boundary_weight,
                push_ignore_boundary_weight=opt.proto_push_ignore_boundary_weight,
            )
            loss_proto_raw = proto_outputs["loss"]
            loss_proto = proto_coeff * loss_proto_raw
            proto_pull_loss = proto_outputs["pull_loss"]
            proto_sep_loss = proto_outputs["sep_loss"]
            proto_cons_loss = proto_outputs["cons_loss"]
            proto_push_loss = proto_outputs["push_loss"]
            proto_cons_conf_mean = proto_outputs["cons_conf_mean"]
            proto_cons_adaptive_factor_mean = proto_outputs["cons_adaptive_factor_mean"]
            proto_cons_scene_scale = proto_outputs["cons_scene_scale"]
            proto_cons_agree_ratio = proto_outputs["cons_agree_ratio"]
            proto_cons_agree_factor_mean = proto_outputs["cons_agree_factor_mean"]
            proto_avg_entropy = proto_outputs["avg_entropy"]
            proto_avg_assign_conf = proto_outputs["avg_assign_conf"]
            proto_avg_confidence = proto_outputs["avg_proto_confidence"]
            proto_confident_ratio = proto_outputs["confident_ratio"]
            proto_update_avg_confidence = proto_outputs["update_avg_confidence"]
            proto_update_confident_ratio = proto_outputs["update_confident_ratio"]
            proto_avg_margin = proto_outputs["avg_proto_margin"]
            proto_active_ratio = proto_outputs["active_proto_ratio"]
            proto_usage_max = proto_outputs["usage_max"]
            proto_update_features = proto_outputs["update_features"]
            proto_update_probs = proto_outputs["update_probs"]
            proto_update_confidence = proto_outputs["update_confidence"]
            proto_update_sample_weight = proto_outputs["update_sample_weight"]
            proto_diag = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in proto_outputs.items()
                if key.startswith("proto_")
            }
            loss = loss + loss_proto

        if iteration % 100 == 0 and (sugar_active or graph_active or proto_active):
            loss_obj_3d_value = loss_obj_3d.item() if loss_obj_3d is not None else 0.0
            message = (
                f"[Iter {iteration}] "
                f"loss_obj={loss_obj.item():.6f}, "
                f"loss_obj_3d={loss_obj_3d_value:.6f}, "
            )
            if sugar_active:
                message += (
                    f"sugar_coeff={sugar_coeff.item():.6f}, "
                    f"loss_sugar_raw={loss_sugar_raw.item():.6f}, "
                    f"loss_sugar={loss_sugar.item():.6f}, "
                    f"sugar_axis_align_cosine={sugar_axis_align_cosine.item():.6f}, "
                    f"sugar_plane_residual={sugar_plane_residual.item():.6f}, "
                    f"sugar_flat_ratio={sugar_flat_ratio.item():.6f}, "
                )
            if graph_active:
                print(
                    message
                    + f"graph_coeff={graph_coeff.item():.6f}, "
                    + f"loss_graph_raw={loss_graph_raw.item():.6f}, "
                    + f"loss_graph={loss_graph.item():.6f}, "
                    + f"pos_loss={pos_loss.item():.6f}, "
                    + f"neg_loss={neg_loss.item():.6f}, "
                    + f"graph_pos_ratio={graph_pos_ratio.item():.4f}, "
                    + f"graph_neg_ratio={graph_neg_ratio.item():.4f}, "
                    + f"graph_ignore_ratio={graph_ignore_ratio.item():.4f}, "
                    + f"avg_reliability={avg_reliability.item():.6f}, "
                    + f"avg_pos_reliability={avg_pos_reliability.item():.6f}, "
                    + f"avg_neg_reliability={avg_neg_reliability.item():.6f}, "
                    + f"avg_mv_consistency={avg_mv_consistency.item():.6f}, "
                    + f"avg_plane_residual={avg_plane_residual.item():.6f}, "
                    + f"avg_feature_norm={avg_feature_norm.item():.6f}, "
                    + f"avg_normal_cosine={avg_normal_cosine.item():.6f}, "
                    + f"avg_pos_cosine={avg_pos_cosine.item():.6f}, "
                    + f"avg_neg_cosine={avg_neg_cosine.item():.6f}, "
                    + f"avg_hard_neg_cosine={avg_hard_neg_cosine.item():.6f}, "
                    + f"active_neg_ratio={active_neg_ratio.item():.6f}, "
                    + f"avg_sem_valid_views={avg_sem_valid_views.item():.6f}, "
                    + f"avg_sem_confidence={avg_sem_confidence.item():.6f}, "
                    + f"sem_valid_point_ratio={sem_valid_point_ratio.item():.6f}, "
                    + f"sem_pair_valid_ratio={sem_pair_valid_ratio.item():.6f}, "
                    + f"sem_high_conf_same_ratio={sem_high_conf_same_ratio.item():.6f}, "
                    + f"semantic_pos_keep_ratio={semantic_pos_keep_ratio.item():.6f}, "
                    + f"semantic_neg_keep_ratio={semantic_neg_keep_ratio.item():.6f}"
                )
            else:
                print(message.rstrip(", "))
            if proto_active:
                print(
                    f"[Iter {iteration}] "
                    + f"proto_coeff={proto_coeff.item():.6f}, "
                    + f"loss_proto_raw={loss_proto_raw.item():.6f}, "
                    + f"loss_proto={loss_proto.item():.6f}, "
                    + f"proto_pull_loss={proto_pull_loss.item():.6f}, "
                    + f"proto_sep_loss={proto_sep_loss.item():.6f}, "
                    + f"proto_cons_loss={proto_cons_loss.item():.6f}, "
                    + f"proto_push_loss={proto_push_loss.item():.6f}, "
                    + f"proto_cons_conf_mean={proto_cons_conf_mean.item():.6f}, "
                    + f"proto_cons_adaptive_factor_mean={proto_cons_adaptive_factor_mean.item():.6f}, "
                    + f"proto_cons_scene_scale={proto_cons_scene_scale.item():.6f}, "
                    + f"proto_cons_agree_ratio={proto_cons_agree_ratio.item():.6f}, "
                    + f"proto_cons_agree_factor_mean={proto_cons_agree_factor_mean.item():.6f}, "
                    + f"proto_avg_entropy={proto_avg_entropy.item():.6f}, "
                    + f"proto_avg_assign_conf={proto_avg_assign_conf.item():.6f}, "
                    + f"proto_avg_confidence={proto_avg_confidence.item():.6f}, "
                    + f"proto_confident_ratio={proto_confident_ratio.item():.6f}, "
                    + f"proto_update_avg_confidence={proto_update_avg_confidence.item():.6f}, "
                    + f"proto_update_confident_ratio={proto_update_confident_ratio.item():.6f}, "
                    + f"proto_avg_margin={proto_avg_margin.item():.6f}, "
                    + f"proto_active_ratio={proto_active_ratio.item():.6f}, "
                    + f"proto_usage_max={proto_usage_max.item():.6f}"
                )
                if proto_diag:
                    print(
                        f"[Iter {iteration}] "
                        + "proto_diag: "
                        + f"dead={_proto_diag_scalar(proto_diag, 'proto_dead_count'):.0f}, "
                        + f"active_update={_proto_diag_scalar(proto_diag, 'proto_active_update_count'):.0f}, "
                        + f"usage_entropy={_proto_diag_scalar(proto_diag, 'proto_usage_entropy'):.6f}, "
                        + f"update_usage_entropy={_proto_diag_scalar(proto_diag, 'proto_update_usage_entropy'):.6f}, "
                        + f"usage_min={_proto_diag_scalar(proto_diag, 'proto_usage_min'):.6f}, "
                        + f"usage_max={_proto_diag_scalar(proto_diag, 'proto_usage_max'):.6f}, "
                        + f"usage_std={_proto_diag_scalar(proto_diag, 'proto_usage_std'):.6f}, "
                        + f"update_usage_min={_proto_diag_scalar(proto_diag, 'proto_update_usage_min'):.6f}, "
                        + f"update_usage_max={_proto_diag_scalar(proto_diag, 'proto_update_usage_max'):.6f}, "
                        + f"update_usage_std={_proto_diag_scalar(proto_diag, 'proto_update_usage_std'):.6f}, "
                        + f"pair_cos_max={_proto_diag_scalar(proto_diag, 'proto_pair_cosine_max'):.6f}, "
                        + f"push_active={_proto_diag_scalar(proto_diag, 'proto_push_active_ratio'):.6f}, "
                        + f"push_weight_mean={_proto_diag_scalar(proto_diag, 'proto_push_weight_mean'):.6f}, "
                        + f"update_boundary_weight_mean={_proto_diag_scalar(proto_diag, 'proto_update_boundary_weight_mean'):.6f}, "
                        + f"assign_conf_p50={_proto_diag_scalar(proto_diag, 'proto_assign_conf_p50'):.6f}, "
                        + f"margin_p50={_proto_diag_scalar(proto_diag, 'proto_margin_p50'):.6f}, "
                        + f"update_selected_ratio={_proto_diag_scalar(proto_diag, 'proto_update_selected_ratio'):.6f}, "
                        + f"neg_boundary_selected_ratio={_proto_diag_scalar(proto_diag, 'proto_neg_boundary_selected_ratio'):.6f}, "
                        + f"uncertain_boundary_selected_ratio={_proto_diag_scalar(proto_diag, 'proto_uncertain_boundary_selected_ratio'):.6f}"
                    )


        loss.backward()
        iter_end.record()


        with torch.no_grad():
            if proto_update_features is not None:
                prototype_bank.ema_update(
                    proto_update_features,
                    proto_update_probs,
                    proto_update_confidence,
                    confidence_thresh=opt.proto_update_conf_thresh,
                    sample_weight=proto_update_sample_weight,
                )

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, loss_sugar_raw, loss_sugar, sugar_coeff, sugar_active, sugar_axis_align_cosine, sugar_plane_residual, sugar_flat_ratio, loss_graph_raw, loss_graph, graph_coeff, graph_active, graph_pos_ratio, graph_neg_ratio, graph_ignore_ratio, avg_reliability, avg_pos_reliability, avg_neg_reliability, avg_mv_consistency, avg_plane_residual, pos_loss, neg_loss, avg_feature_norm, avg_normal_cosine, avg_pos_cosine, avg_neg_cosine, avg_hard_neg_cosine, active_neg_ratio, avg_sem_valid_views, avg_sem_confidence, sem_valid_point_ratio, sem_pair_valid_ratio, sem_high_conf_same_ratio, semantic_pos_keep_ratio, semantic_neg_keep_ratio, loss_proto_raw, loss_proto, proto_coeff, proto_active, proto_pull_loss, proto_sep_loss, proto_cons_loss, proto_cons_conf_mean, proto_cons_adaptive_factor_mean, proto_cons_scene_scale, proto_cons_agree_ratio, proto_cons_agree_factor_mean, proto_avg_entropy, proto_avg_assign_conf, proto_avg_confidence, proto_confident_ratio, proto_update_avg_confidence, proto_update_confident_ratio, proto_avg_margin, proto_active_ratio, proto_usage_max, proto_diag, use_wandb)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))
                if prototype_bank is not None:
                    torch.save(
                        prototype_bank.state_dict(),
                        os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration), 'prototype_bank.pth')
                    )

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                cls_optimizer.step()
                cls_optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, loss_sugar_raw, loss_sugar, sugar_coeff, sugar_active, sugar_axis_align_cosine, sugar_plane_residual, sugar_flat_ratio, loss_graph_raw, loss_graph, graph_coeff, graph_active, graph_pos_ratio, graph_neg_ratio, graph_ignore_ratio, avg_reliability, avg_pos_reliability, avg_neg_reliability, avg_mv_consistency, avg_plane_residual, pos_loss, neg_loss, avg_feature_norm, avg_normal_cosine, avg_pos_cosine, avg_neg_cosine, avg_hard_neg_cosine, active_neg_ratio, avg_sem_valid_views, avg_sem_confidence, sem_valid_point_ratio, sem_pair_valid_ratio, sem_high_conf_same_ratio, semantic_pos_keep_ratio, semantic_neg_keep_ratio, loss_proto_raw, loss_proto, proto_coeff, proto_active, proto_pull_loss, proto_sep_loss, proto_cons_loss, proto_cons_conf_mean, proto_cons_adaptive_factor_mean, proto_cons_scene_scale, proto_cons_agree_ratio, proto_cons_agree_factor_mean, proto_avg_entropy, proto_avg_assign_conf, proto_avg_confidence, proto_confident_ratio, proto_update_avg_confidence, proto_update_confident_ratio, proto_avg_margin, proto_active_ratio, proto_usage_max, proto_diag, use_wandb):

    if use_wandb:
        log_data = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "train_loss_patches/sugar_active": float(sugar_active),
            "train_loss_patches/sugar_coeff": sugar_coeff.item(),
            "train_loss_patches/graph_active": float(graph_active),
            "train_loss_patches/graph_coeff": graph_coeff.item(),
            "train_loss_patches/proto_active": float(proto_active),
            "train_loss_patches/proto_coeff": proto_coeff.item(),
            "iter_time": elapsed,
            "iter": iteration
        }

        if loss_obj_3d is not None:
            log_data["train_loss_patches/loss_obj_3d"] = loss_obj_3d.item()

        if sugar_active:
            log_data.update({
                "train_loss_patches/loss_sugar_raw": loss_sugar_raw.item(),
                "train_loss_patches/loss_sugar": loss_sugar.item(),
                "train_loss_patches/sugar_axis_align_cosine": sugar_axis_align_cosine.item(),
                "train_loss_patches/sugar_plane_residual": sugar_plane_residual.item(),
                "train_loss_patches/sugar_flat_ratio": sugar_flat_ratio.item(),
            })

        if graph_active:
            log_data.update({
                "train_loss_patches/loss_graph_raw": loss_graph_raw.item(),
                "train_loss_patches/loss_graph": loss_graph.item(),
                "train_loss_patches/pos_loss": pos_loss.item(),
                "train_loss_patches/neg_loss": neg_loss.item(),
                "train_loss_patches/graph_pos_ratio": graph_pos_ratio.item(),
                "train_loss_patches/graph_neg_ratio": graph_neg_ratio.item(),
                "train_loss_patches/graph_ignore_ratio": graph_ignore_ratio.item(),
                "train_loss_patches/avg_reliability": avg_reliability.item(),
                "train_loss_patches/avg_pos_reliability": avg_pos_reliability.item(),
                "train_loss_patches/avg_neg_reliability": avg_neg_reliability.item(),
                "train_loss_patches/avg_mv_consistency": avg_mv_consistency.item(),
                "train_loss_patches/avg_plane_residual": avg_plane_residual.item(),
                "train_loss_patches/avg_feature_norm": avg_feature_norm.item(),
                "train_loss_patches/avg_normal_cosine": avg_normal_cosine.item(),
                "train_loss_patches/avg_pos_cosine": avg_pos_cosine.item(),
                "train_loss_patches/avg_neg_cosine": avg_neg_cosine.item(),
                "train_loss_patches/avg_hard_neg_cosine": avg_hard_neg_cosine.item(),
                "train_loss_patches/active_neg_ratio": active_neg_ratio.item(),
                "train_loss_patches/avg_sem_valid_views": avg_sem_valid_views.item(),
                "train_loss_patches/avg_sem_confidence": avg_sem_confidence.item(),
                "train_loss_patches/sem_valid_point_ratio": sem_valid_point_ratio.item(),
                "train_loss_patches/sem_pair_valid_ratio": sem_pair_valid_ratio.item(),
                "train_loss_patches/sem_high_conf_same_ratio": sem_high_conf_same_ratio.item(),
                "train_loss_patches/semantic_pos_keep_ratio": semantic_pos_keep_ratio.item(),
                "train_loss_patches/semantic_neg_keep_ratio": semantic_neg_keep_ratio.item(),
            })

        if proto_active:
            log_data.update({
                "train_loss_patches/loss_proto_raw": loss_proto_raw.item(),
                "train_loss_patches/loss_proto": loss_proto.item(),
                "train_loss_patches/proto_pull_loss": proto_pull_loss.item(),
                "train_loss_patches/proto_sep_loss": proto_sep_loss.item(),
                "train_loss_patches/proto_cons_loss": proto_cons_loss.item(),
                "train_loss_patches/proto_cons_conf_mean": proto_cons_conf_mean.item(),
                "train_loss_patches/proto_cons_adaptive_factor_mean": proto_cons_adaptive_factor_mean.item(),
                "train_loss_patches/proto_cons_scene_scale": proto_cons_scene_scale.item(),
                "train_loss_patches/proto_cons_agree_ratio": proto_cons_agree_ratio.item(),
                "train_loss_patches/proto_cons_agree_factor_mean": proto_cons_agree_factor_mean.item(),
                "train_loss_patches/proto_avg_entropy": proto_avg_entropy.item(),
                "train_loss_patches/proto_avg_assign_conf": proto_avg_assign_conf.item(),
                "train_loss_patches/proto_avg_confidence": proto_avg_confidence.item(),
                "train_loss_patches/proto_confident_ratio": proto_confident_ratio.item(),
                "train_loss_patches/proto_update_avg_confidence": proto_update_avg_confidence.item(),
                "train_loss_patches/proto_update_confident_ratio": proto_update_confident_ratio.item(),
                "train_loss_patches/proto_avg_margin": proto_avg_margin.item(),
                "train_loss_patches/proto_active_ratio": proto_active_ratio.item(),
                "train_loss_patches/proto_usage_max": proto_usage_max.item(),
            })
            _add_proto_diag_wandb_logs(log_data, proto_diag, iteration)

        wandb.log(log_data)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if use_wandb:
                        if idx < 5:
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                            if iteration == testing_iterations[0]:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)]})
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 200)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)

    args.graph_start_iter = config.get("graph_start_iter", config.get("geo_start_iter", args.graph_start_iter))
    args.graph_interval = config.get("graph_interval", config.get("geo_interval", args.graph_interval))
    args.graph_knn_k = config.get("graph_knn_k", config.get("geo_knn_k", args.graph_knn_k))
    args.graph_weight_lambda = config.get("graph_weight_lambda", config.get("geo_weight_lambda", args.graph_weight_lambda))
    args.graph_warmup_iters = config.get("graph_warmup_iters", config.get("geo_warmup_iters", args.graph_warmup_iters))
    args.graph_lambda_pos = config.get("graph_lambda_pos", config.get("geo_lambda_pos", args.graph_lambda_pos))
    args.graph_lambda_neg = config.get("graph_lambda_neg", config.get("geo_lambda_neg", args.graph_lambda_neg))
    args.graph_max_points = config.get("graph_max_points", config.get("geo_max_points", args.graph_max_points))
    args.graph_sample_size = config.get("graph_sample_size", config.get("geo_sample_size", args.graph_sample_size))
    args.graph_plane_tau = config.get("graph_plane_tau", config.get("geo_plane_tau", args.graph_plane_tau))
    args.graph_neg_plane_tau = config.get("graph_neg_plane_tau", config.get("geo_neg_plane_tau", args.graph_neg_plane_tau))
    args.graph_spatial_pos_scale = config.get("graph_spatial_pos_scale", config.get("geo_spatial_pos_scale", args.graph_spatial_pos_scale))
    args.graph_normal_pos_tau = config.get("graph_normal_pos_tau", config.get("geo_normal_pos_tau", args.graph_normal_pos_tau))
    args.graph_normal_neg_tau = config.get("graph_normal_neg_tau", config.get("geo_normal_neg_tau", args.graph_normal_neg_tau))
    args.graph_neg_margin = config.get("graph_neg_margin", config.get("geo_neg_margin", args.graph_neg_margin))
    args.graph_hard_neg_k = config.get("graph_hard_neg_k", config.get("geo_hard_neg_k", args.graph_hard_neg_k))
    args.graph_use_multiview_semantics = config.get("graph_use_multiview_semantics", config.get("geo_use_multiview_semantics", args.graph_use_multiview_semantics))
    args.graph_support_views = config.get("graph_support_views", config.get("geo_support_views", args.graph_support_views))
    args.graph_sem_pos_ratio = config.get("graph_sem_pos_ratio", config.get("geo_sem_pos_ratio", args.graph_sem_pos_ratio))
    args.graph_sem_min_views = config.get("graph_sem_min_views", config.get("geo_sem_min_views", args.graph_sem_min_views))
    args.graph_sem_conf_tau = config.get("graph_sem_conf_tau", config.get("geo_sem_conf_tau", args.graph_sem_conf_tau))
    args.graph_sem_ignore_label = config.get("graph_sem_ignore_label", config.get("geo_sem_ignore_label", args.graph_sem_ignore_label))
    args.graph_normal_weight_lambda = config.get("graph_normal_weight_lambda", config.get("geo_normal_weight_lambda", args.graph_normal_weight_lambda))
    args.graph_sem_same_boost = config.get("graph_sem_same_boost", config.get("geo_sem_same_boost", args.graph_sem_same_boost))
    args.graph_sem_neg_boost = config.get("graph_sem_neg_boost", config.get("geo_sem_neg_boost", args.graph_sem_neg_boost))
    args.graph_sem_conflict_penalty = config.get("graph_sem_conflict_penalty", config.get("geo_sem_conflict_penalty", args.graph_sem_conflict_penalty))
    args.graph_alpha_dist = config.get("graph_alpha_dist", args.graph_alpha_dist)
    args.graph_alpha_normal = config.get("graph_alpha_normal", args.graph_alpha_normal)
    args.graph_alpha_residual = config.get("graph_alpha_residual", args.graph_alpha_residual)
    args.graph_alpha_mv = config.get("graph_alpha_mv", args.graph_alpha_mv)
    args.graph_pos_reliability_thresh = config.get("graph_pos_reliability_thresh", args.graph_pos_reliability_thresh)
    args.graph_neg_reliability_thresh = config.get("graph_neg_reliability_thresh", args.graph_neg_reliability_thresh)
    args.use_proto = config.get("use_proto", args.use_proto)
    args.num_prototypes = config.get("num_prototypes", args.num_prototypes)
    args.proto_start_iter = config.get("proto_start_iter", args.proto_start_iter)
    args.proto_interval = config.get("proto_interval", args.proto_interval)
    args.proto_weight_lambda = config.get("proto_weight_lambda", args.proto_weight_lambda)
    args.proto_warmup_iters = config.get("proto_warmup_iters", args.proto_warmup_iters)
    args.proto_tau = config.get("proto_tau", args.proto_tau)
    args.proto_conf_thresh = config.get("proto_conf_thresh", args.proto_conf_thresh)
    args.proto_ema_momentum = config.get("proto_ema_momentum", args.proto_ema_momentum)
    args.proto_lambda_pull = config.get("proto_lambda_pull", args.proto_lambda_pull)
    args.proto_lambda_sep = config.get("proto_lambda_sep", args.proto_lambda_sep)
    args.proto_lambda_cons = config.get("proto_lambda_cons", args.proto_lambda_cons)
    args.proto_cons_conf_weight = config.get("proto_cons_conf_weight", args.proto_cons_conf_weight)
    args.proto_cons_conf_floor = config.get("proto_cons_conf_floor", args.proto_cons_conf_floor)
    args.proto_cons_conf_power = config.get("proto_cons_conf_power", args.proto_cons_conf_power)
    args.proto_cons_conf_normalize = config.get("proto_cons_conf_normalize", args.proto_cons_conf_normalize)
    args.proto_cons_conf_norm_max = config.get("proto_cons_conf_norm_max", args.proto_cons_conf_norm_max)
    args.proto_cons_scene_weight = config.get("proto_cons_scene_weight", args.proto_cons_scene_weight)
    args.proto_cons_scene_floor = config.get("proto_cons_scene_floor", args.proto_cons_scene_floor)
    args.proto_cons_scene_conf_min = config.get("proto_cons_scene_conf_min", args.proto_cons_scene_conf_min)
    args.proto_cons_scene_conf_target = config.get("proto_cons_scene_conf_target", args.proto_cons_scene_conf_target)
    args.proto_cons_agree_weight = config.get("proto_cons_agree_weight", args.proto_cons_agree_weight)
    args.proto_cons_agree_floor = config.get("proto_cons_agree_floor", args.proto_cons_agree_floor)
    args.proto_cons_agree_conf_thresh = config.get("proto_cons_agree_conf_thresh", args.proto_cons_agree_conf_thresh)
    args.proto_sep_margin = config.get("proto_sep_margin", args.proto_sep_margin)
    args.proto_reliability_thresh = config.get("proto_reliability_thresh", args.proto_reliability_thresh)
    args.proto_entropy_thresh = config.get("proto_entropy_thresh", args.proto_entropy_thresh)
    args.proto_assign_conf_thresh = config.get("proto_assign_conf_thresh", args.proto_assign_conf_thresh)
    args.proto_sem_invalid_weight = config.get("proto_sem_invalid_weight", args.proto_sem_invalid_weight)
    args.proto_update_conf_thresh = config.get("proto_update_conf_thresh", config.get("proto_conf_thresh", args.proto_update_conf_thresh))
    args.proto_update_reliability_thresh = config.get("proto_update_reliability_thresh", config.get("proto_reliability_thresh", args.proto_update_reliability_thresh))
    args.proto_update_entropy_thresh = config.get("proto_update_entropy_thresh", config.get("proto_entropy_thresh", args.proto_update_entropy_thresh))
    args.proto_update_assign_conf_thresh = config.get("proto_update_assign_conf_thresh", config.get("proto_assign_conf_thresh", args.proto_update_assign_conf_thresh))
    args.proto_update_sem_invalid_weight = config.get("proto_update_sem_invalid_weight", config.get("proto_sem_invalid_weight", args.proto_update_sem_invalid_weight))
    args.proto_boundary_safe_update = config.get("proto_boundary_safe_update", args.proto_boundary_safe_update)
    args.proto_update_neg_boundary_weight = config.get("proto_update_neg_boundary_weight", args.proto_update_neg_boundary_weight)
    args.proto_update_ignore_boundary_weight = config.get("proto_update_ignore_boundary_weight", args.proto_update_ignore_boundary_weight)
    args.proto_lambda_push = config.get("proto_lambda_push", args.proto_lambda_push)
    args.proto_push_mode = config.get("proto_push_mode", args.proto_push_mode)
    args.proto_push_margin = config.get("proto_push_margin", args.proto_push_margin)
    args.proto_push_temperature = config.get("proto_push_temperature", args.proto_push_temperature)
    args.proto_push_conf_thresh = config.get("proto_push_conf_thresh", args.proto_push_conf_thresh)
    args.proto_push_use_confident_mask = config.get("proto_push_use_confident_mask", args.proto_push_use_confident_mask)
    args.proto_push_reliability_thresh = config.get("proto_push_reliability_thresh", args.proto_push_reliability_thresh)
    args.proto_push_entropy_thresh = config.get("proto_push_entropy_thresh", args.proto_push_entropy_thresh)
    args.proto_push_assign_conf_thresh = config.get("proto_push_assign_conf_thresh", args.proto_push_assign_conf_thresh)
    args.proto_push_neg_boundary_weight = config.get("proto_push_neg_boundary_weight", args.proto_push_neg_boundary_weight)
    args.proto_push_ignore_boundary_weight = config.get("proto_push_ignore_boundary_weight", args.proto_push_ignore_boundary_weight)
    args.sugar_start_iter = config.get("sugar_start_iter", args.densify_until_iter)
    args.sugar_interval = config.get("sugar_interval", 10)
    args.sugar_warmup_iters = config.get("sugar_warmup_iters", 2000)
    args.sugar_weight_lambda = config.get("sugar_weight_lambda", 0.2)
    args.sugar_lambda_axis = config.get("sugar_lambda_axis", 1.0)
    args.sugar_lambda_plane = config.get("sugar_lambda_plane", 0.5)
    args.sugar_lambda_flat = config.get("sugar_lambda_flat", 0.1)
    args.sugar_knn_k = config.get("sugar_knn_k", 8)
    args.sugar_max_points = config.get("sugar_max_points", 200000)
    args.sugar_sample_size = config.get("sugar_sample_size", 800)
    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="gaussian-splatting")
        wandb.config.args = args
        wandb.run.name = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb)

    # All done
    print("\nTraining complete.")
