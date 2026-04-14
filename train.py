# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
from random import randint, sample
from utils.loss_utils import (
    l1_loss,
    ssim,
    loss_cls_3d,
    loss_geo_gated_contrastive,
    loss_sugar_opacity_entropy,
    loss_sugar_surface_alignment,
)
from utils.distill_utils import (
    FeatureCacheLoader,
    cosine_distill_loss,
    render_feature_foreground_mask,
    resolve_feature_cache_dir,
)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json

def save_auxiliary_heads(model_path, iteration, classifier, distill_head=None):
    aux_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    os.makedirs(aux_dir, exist_ok=True)
    torch.save(classifier.state_dict(), os.path.join(aux_dir, "classifier.pth"))
    if distill_head is not None:
        torch.save(distill_head.state_dict(), os.path.join(aux_dir, "distill_head.pth"))


def restore_auxiliary_heads(model_path, iteration, classifier, distill_head=None):
    aux_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    classifier_path = os.path.join(aux_dir, "classifier.pth")
    if os.path.exists(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        print(f"Loaded classifier from {classifier_path}")

    if distill_head is not None:
        distill_head_path = os.path.join(aux_dir, "distill_head.pth")
        if os.path.exists(distill_head_path):
            distill_head.load_state_dict(torch.load(distill_head_path, map_location="cpu"))
            print(f"Loaded distill head from {distill_head_path}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb):
    first_iter = 0
    checkpoint_classifier_state = None
    checkpoint_distill_state = None
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    num_classes = dataset.num_classes
    print("Num classes: ", num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    classifier.cuda()

    if opt.geo_normal_source != "surface_axis":
        raise ValueError(
            f"Unsupported geo_normal_source '{opt.geo_normal_source}'. Only 'surface_axis' is implemented."
        )

    distill_loader = None
    distill_head = None
    distill_optimizer = None
    distill_teacher_dim = 0
    if opt.distill_enable:
        try:
            if opt.distill_loss_type != "cosine":
                raise ValueError(
                    f"Unsupported distill_loss_type '{opt.distill_loss_type}'. Only 'cosine' is implemented."
                )
            feature_dir = resolve_feature_cache_dir(dataset.source_path, opt.distill_feature_dir)
            distill_loader = FeatureCacheLoader(feature_dir)
            train_cameras = scene.getTrainCameras()
            if len(train_cameras) == 0:
                raise ValueError("No training cameras available for DINO feature distillation.")
            distill_teacher_dim = distill_loader.get_teacher_dim(train_cameras[0].image_name)
            distill_head = torch.nn.Conv2d(gaussians.num_objects, distill_teacher_dim, kernel_size=1)
            distill_head.cuda()
            distill_optimizer = torch.optim.Adam(distill_head.parameters(), lr=5e-4)
            print(
                f"DINO distillation enabled. Feature dir: {feature_dir}, teacher_dim: {distill_teacher_dim}"
            )
        except Exception as exc:
            print(f"[Warning] Disable DINO distillation: {exc}")
            opt.distill_enable = False
            distill_loader = None
            distill_head = None
            distill_optimizer = None
            distill_teacher_dim = 0

    if checkpoint:
        checkpoint_data = torch.load(checkpoint)
        if isinstance(checkpoint_data, tuple) and len(checkpoint_data) == 4:
            model_params, checkpoint_classifier_state, checkpoint_distill_state, first_iter = checkpoint_data
        elif isinstance(checkpoint_data, tuple) and len(checkpoint_data) == 2:
            model_params, first_iter = checkpoint_data
        else:
            raise ValueError(
                f"Unsupported checkpoint format in {checkpoint}. Expected 2 or 4 tuple entries."
            )
        gaussians.restore(model_params, opt)
        if checkpoint_classifier_state is not None:
            classifier.load_state_dict(checkpoint_classifier_state)
        else:
            restore_auxiliary_heads(scene.model_path, first_iter, classifier, distill_head)
        if distill_head is not None and checkpoint_distill_state is not None:
            distill_head.load_state_dict(checkpoint_distill_state)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
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
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        objects = render_pkg["render_object"]

        # Object Loss
        gt_obj = viewpoint_cam.objects.cuda().long()
        logits = classifier(objects)
        loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        loss_obj = loss_obj / torch.log(torch.tensor(num_classes, device=image.device))  # normalize to (0,1)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        recon_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0:
            # regularize at certain intervals
            logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
            prob_obj3d = torch.softmax(logits3d, dim=0).squeeze().permute(1, 0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
            loss = recon_loss + loss_obj + loss_obj_3d
        else:
            loss = recon_loss + loss_obj

        loss_geo_raw = torch.tensor(0.0, device=image.device)
        loss_geo = torch.tensor(0.0, device=image.device)
        geo_coeff = torch.tensor(0.0, device=image.device)
        loss_con_raw = torch.tensor(0.0, device=image.device)
        loss_smooth_raw = torch.tensor(0.0, device=image.device)
        loss_sugar_raw = torch.tensor(0.0, device=image.device)
        loss_sugar = torch.tensor(0.0, device=image.device)
        sugar_coeff = torch.tensor(0.0, device=image.device)
        loss_sugar_opacity_entropy_raw = torch.tensor(0.0, device=image.device)
        loss_sugar_opacity_entropy_term = torch.tensor(0.0, device=image.device)
        sugar_opacity_entropy_coeff = torch.tensor(0.0, device=image.device)
        distill_coeff = torch.tensor(0.0, device=image.device)
        loss_distill_raw = torch.tensor(0.0, device=image.device)
        loss_distill = torch.tensor(0.0, device=image.device)
        distill_active = distill_head is not None and distill_loader is not None
        sugar_axis_align_cosine = torch.tensor(0.0, device=image.device)
        sugar_plane_residual = torch.tensor(0.0, device=image.device)
        sugar_flat_ratio = torch.tensor(0.0, device=image.device)
        gate_ratio = torch.tensor(0.0, device=image.device)
        avg_depth_gap = torch.tensor(0.0, device=image.device)
        pos_loss = torch.tensor(0.0, device=image.device)
        neg_loss = torch.tensor(0.0, device=image.device)
        avg_feature_norm = torch.tensor(0.0, device=image.device)
        avg_normal_cosine = torch.tensor(0.0, device=image.device)
        avg_pos_dist = torch.tensor(0.0, device=image.device)
        avg_neg_dist = torch.tensor(0.0, device=image.device)
        avg_hard_neg_dist = torch.tensor(0.0, device=image.device)
        active_neg_ratio = torch.tensor(0.0, device=image.device)
        neg_candidate_ratio = torch.tensor(0.0, device=image.device)
        ignore_ratio = torch.tensor(0.0, device=image.device)
        avg_sem_valid_views = torch.tensor(0.0, device=image.device)
        avg_sem_confidence = torch.tensor(0.0, device=image.device)
        semantic_pos_keep_ratio = torch.tensor(0.0, device=image.device)
        semantic_neg_keep_ratio = torch.tensor(0.0, device=image.device)
        pos_pairs = torch.tensor(0.0, device=image.device)
        neg_pairs = torch.tensor(0.0, device=image.device)
        boundary_neg_ratio = torch.tensor(0.0, device=image.device)
        sugar_active = iteration >= opt.sugar_start_iter and iteration % opt.sugar_interval == 0
        geo_active = iteration >= opt.geo_start_iter and iteration % opt.geo_interval == 0

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

        sugar_opacity_entropy_active = (
            opt.sugar_lambda_opacity_entropy > 0
            and opt.sugar_opacity_entropy_start_iter <= iteration < opt.sugar_opacity_entropy_end_iter
        )
        if sugar_opacity_entropy_active:
            sugar_opacity_entropy_coeff = image.new_tensor(opt.sugar_lambda_opacity_entropy)
            loss_sugar_opacity_entropy_raw = loss_sugar_opacity_entropy(gaussians.get_opacity)
            loss_sugar_opacity_entropy_term = sugar_opacity_entropy_coeff * loss_sugar_opacity_entropy_raw
            loss = loss + loss_sugar_opacity_entropy_term

        if distill_active:
            try:
                teacher_feature_map, _ = distill_loader.get_feature_map(
                    viewpoint_cam.image_name,
                    target_hw=objects.shape[-2:],
                    device=image.device,
                    dtype=objects.dtype,
                )
                student_feature_map = distill_head(objects)
                distill_mask = render_feature_foreground_mask(objects, eps=opt.distill_mask_eps)
                loss_distill_raw = cosine_distill_loss(
                    student_feature_map,
                    teacher_feature_map,
                    mask=distill_mask,
                )
                distill_coeff = image.new_tensor(opt.distill_weight_lambda)
                loss_distill = distill_coeff * loss_distill_raw
                loss = loss + loss_distill
            except Exception as exc:
                print(f"[Warning] Disable DINO distillation during training: {exc}")
                distill_loader = None
                distill_head = None
                distill_optimizer = None
                distill_teacher_dim = 0
                distill_active = False

        if geo_active:
            warmup_iters = max(1, opt.geo_warmup_iters)
            geo_progress = min(max((iteration - opt.geo_start_iter) / warmup_iters, 0.0), 1.0)
            geo_coeff = image.new_tensor(opt.geo_weight_lambda * geo_progress)

            support_cameras = None
            support_visibility = None
            if opt.geo_use_multiview_semantics:
                candidate_cameras = [cam for cam in scene.getTrainCameras() if cam.uid != viewpoint_cam.uid]
                num_support = min(max(int(opt.geo_support_views), 0), len(candidate_cameras))
                support_cameras = [viewpoint_cam]
                if num_support > 0:
                    support_cameras.extend(sample(candidate_cameras, num_support))

                visibility_masks = [visibility_filter.detach()]
                if len(support_cameras) > 1:
                    with torch.no_grad():
                        for support_cam in support_cameras[1:]:
                            support_render_pkg = render(support_cam, gaussians, pipe, background)
                            visibility_masks.append(support_render_pkg["visibility_filter"].detach())
                support_visibility = torch.stack(visibility_masks, dim=0)

            loss_geo_raw, gate_ratio, avg_depth_gap, loss_con_raw, loss_smooth_raw, pos_loss, neg_loss, avg_feature_norm, avg_normal_cosine, avg_pos_dist, avg_neg_dist, avg_hard_neg_dist, active_neg_ratio, neg_candidate_ratio, ignore_ratio, avg_sem_valid_views, avg_sem_confidence, semantic_pos_keep_ratio, semantic_neg_keep_ratio, pos_pairs, neg_pairs, boundary_neg_ratio = loss_geo_gated_contrastive(
                features=gaussians._objects_dc.squeeze(1),
                xyz=gaussians._xyz.squeeze().detach(),
                normals=gaussians.get_surface_axis.detach(),
                point_ids=torch.arange(gaussians._xyz.shape[0], device=gaussians._xyz.device),
                k=opt.geo_knn_k,
                lambda_val=1.0,
                lambda_pos=opt.geo_lambda_pos,
                lambda_neg=opt.geo_lambda_neg,
                smooth_lambda=opt.geo_smooth_lambda,
                max_points=opt.geo_max_points,
                sample_size=opt.geo_sample_size,
                plane_tau=opt.geo_plane_tau,
                neg_plane_tau=opt.geo_neg_plane_tau,
                spatial_pos_scale=opt.geo_spatial_pos_scale,
                normal_pos_tau=opt.geo_normal_pos_tau,
                normal_neg_tau=opt.geo_normal_neg_tau,
                neg_margin=opt.geo_neg_margin,
                hard_neg_k=opt.geo_hard_neg_k,
                support_cameras=support_cameras,
                support_visibility=support_visibility,
                sem_min_views=opt.geo_sem_min_views,
                sem_conf_tau=opt.geo_sem_conf_tau,
                sem_num_classes=num_classes,
                sem_ignore_label=opt.geo_sem_ignore_label,
                normal_weight_lambda=opt.geo_normal_weight_lambda,
                sem_neg_boost=opt.geo_sem_neg_boost,
            )
            loss_geo = geo_coeff * loss_geo_raw
            loss = loss + loss_geo

        if iteration % 100 == 0 and (sugar_active or sugar_opacity_entropy_active or geo_active or distill_active):
            loss_obj_3d_value = loss_obj_3d.item() if loss_obj_3d is not None else 0.0
            message = (
                f"[Iter {iteration}] "
                f"loss_obj={loss_obj.item():.6f}, "
                f"loss_obj_3d={loss_obj_3d_value:.6f}, "
            )
            if distill_active:
                message += (
                    f"distill_coeff={distill_coeff.item():.6f}, "
                    f"loss_distill_raw={loss_distill_raw.item():.6f}, "
                    f"loss_distill={loss_distill.item():.6f}, "
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
            if sugar_opacity_entropy_active:
                message += (
                    f"sugar_opacity_entropy_coeff={sugar_opacity_entropy_coeff.item():.6f}, "
                    f"loss_sugar_opacity_entropy_raw={loss_sugar_opacity_entropy_raw.item():.6f}, "
                    f"loss_sugar_opacity_entropy={loss_sugar_opacity_entropy_term.item():.6f}, "
                )
            if geo_active:
                print(
                    message
                    + f"geo_coeff={geo_coeff.item():.6f}, "
                    + f"loss_geo_raw={loss_geo_raw.item():.6f}, "
                    + f"loss_geo={loss_geo.item():.6f}, "
                    + f"loss_con={loss_con_raw.item():.6f}, "
                    + f"loss_smooth={loss_smooth_raw.item():.6f}, "
                    + f"pos_loss={pos_loss.item():.6f}, "
                    + f"neg_loss={neg_loss.item():.6f}, "
                    + f"gate_ratio={gate_ratio.item():.4f}, "
                    + f"avg_depth_gap={avg_depth_gap.item():.6f}, "
                    + f"avg_feature_norm={avg_feature_norm.item():.6f}, "
                    + f"avg_normal_cosine={avg_normal_cosine.item():.6f}, "
                    + f"avg_pos_dist={avg_pos_dist.item():.6f}, "
                    + f"avg_neg_dist={avg_neg_dist.item():.6f}, "
                    + f"avg_hard_neg_dist={avg_hard_neg_dist.item():.6f}, "
                    + f"active_neg_ratio={active_neg_ratio.item():.6f}, "
                    + f"neg_candidate_ratio={neg_candidate_ratio.item():.6f}, "
                    + f"ignore_ratio={ignore_ratio.item():.6f}, "
                    + f"avg_sem_valid_views={avg_sem_valid_views.item():.6f}, "
                    + f"avg_sem_confidence={avg_sem_confidence.item():.6f}, "
                    + f"semantic_pos_keep_ratio={semantic_pos_keep_ratio.item():.6f}, "
                    + f"semantic_neg_keep_ratio={semantic_neg_keep_ratio.item():.6f}, "
                    + f"pos_pairs={pos_pairs.item():.2f}, "
                    + f"neg_pairs={neg_pairs.item():.2f}, "
                    + f"boundary_neg_ratio={boundary_neg_ratio.item():.6f}"
                )
            else:
                print(message.rstrip(", "))


        loss.backward()
        iter_end.record()

        metric_logs = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "train_loss_patches/loss_recon": recon_loss.item(),
            "train_loss_patches/loss_obj": loss_obj.item(),
            "train_loss_patches/sugar_active": float(sugar_active),
            "train_loss_patches/sugar_coeff": sugar_coeff.item(),
            "train_loss_patches/sugar_opacity_entropy_active": float(sugar_opacity_entropy_active),
            "train_loss_patches/sugar_opacity_entropy_coeff": sugar_opacity_entropy_coeff.item(),
            "train_loss_patches/geo_active": float(geo_active),
            "train_loss_patches/geo_coeff": geo_coeff.item(),
            "train_loss_patches/distill_active": float(distill_active),
            "train_loss_patches/distill_coeff": distill_coeff.item(),
            "iter_time": iter_start.elapsed_time(iter_end),
            "iter": iteration,
        }

        if loss_obj_3d is not None:
            metric_logs["train_loss_patches/loss_obj_3d"] = loss_obj_3d.item()

        if distill_active:
            metric_logs.update({
                "train_loss_patches/loss_distill_raw": loss_distill_raw.item(),
                "train_loss_patches/loss_distill": loss_distill.item(),
                "train_loss_patches/distill_teacher_dim": float(distill_teacher_dim),
            })

        if sugar_active:
            metric_logs.update({
                "train_loss_patches/loss_sugar_raw": loss_sugar_raw.item(),
                "train_loss_patches/loss_sugar": loss_sugar.item(),
                "train_loss_patches/sugar_axis_align_cosine": sugar_axis_align_cosine.item(),
                "train_loss_patches/sugar_plane_residual": sugar_plane_residual.item(),
                "train_loss_patches/sugar_flat_ratio": sugar_flat_ratio.item(),
            })

        if sugar_opacity_entropy_active:
            metric_logs.update({
                "train_loss_patches/loss_sugar_opacity_entropy_raw": loss_sugar_opacity_entropy_raw.item(),
                "train_loss_patches/loss_sugar_opacity_entropy": loss_sugar_opacity_entropy_term.item(),
            })

        if geo_active:
            metric_logs.update({
                "train_loss_patches/loss_geo_raw": loss_geo_raw.item(),
                "train_loss_patches/loss_geo": loss_geo.item(),
                "train_loss_patches/loss_con": loss_con_raw.item(),
                "train_loss_patches/loss_smooth": loss_smooth_raw.item(),
                "train_loss_patches/pos_loss": pos_loss.item(),
                "train_loss_patches/neg_loss": neg_loss.item(),
                "train_loss_patches/gate_ratio": gate_ratio.item(),
                "train_loss_patches/avg_depth_gap": avg_depth_gap.item(),
                "train_loss_patches/avg_feature_norm": avg_feature_norm.item(),
                "train_loss_patches/avg_normal_cosine": avg_normal_cosine.item(),
                "train_loss_patches/avg_pos_dist": avg_pos_dist.item(),
                "train_loss_patches/avg_neg_dist": avg_neg_dist.item(),
                "train_loss_patches/avg_hard_neg_dist": avg_hard_neg_dist.item(),
                "train_loss_patches/active_neg_ratio": active_neg_ratio.item(),
                "train_loss_patches/neg_candidate_ratio": neg_candidate_ratio.item(),
                "train_loss_patches/ignore_ratio": ignore_ratio.item(),
                "train_loss_patches/avg_sem_valid_views": avg_sem_valid_views.item(),
                "train_loss_patches/avg_sem_confidence": avg_sem_confidence.item(),
                "train_loss_patches/semantic_pos_keep_ratio": semantic_pos_keep_ratio.item(),
                "train_loss_patches/semantic_neg_keep_ratio": semantic_neg_keep_ratio.item(),
                "train_loss_patches/pos_pairs": pos_pairs.item(),
                "train_loss_patches/neg_pairs": neg_pairs.item(),
                "train_loss_patches/boundary_neg_ratio": boundary_neg_ratio.item(),
            })


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                iteration,
                Ll1,
                loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                metric_logs,
                use_wandb,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_auxiliary_heads(scene.model_path, iteration, classifier, distill_head)

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
                cls_optimizer.zero_grad(set_to_none=True)
                if distill_optimizer is not None:
                    distill_optimizer.step()
                    distill_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (
                        gaussians.capture(),
                        classifier.state_dict(),
                        distill_head.state_dict() if distill_head is not None else None,
                        iteration,
                    ),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

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


def training_report(iteration, Ll1, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, metric_logs, use_wandb):

    if use_wandb:
        log_data = dict(metric_logs)
        log_data.setdefault("train_loss_patches/l1_loss", Ll1.item())
        log_data.setdefault("train_loss_patches/total_loss", loss.item())
        log_data.setdefault("iter_time", elapsed)
        log_data.setdefault("iter", iteration)
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

    args.geo_start_iter = config.get("geo_start_iter", args.densify_until_iter + 2000)
    args.geo_interval = config.get("geo_interval", 10)
    args.geo_knn_k = config.get("geo_knn_k", 8)
    args.geo_weight_lambda = config.get("geo_weight_lambda", 1.0)
    args.geo_warmup_iters = config.get("geo_warmup_iters", 4000)
    args.geo_lambda_pos = config.get("geo_lambda_pos", 1.0)
    args.geo_lambda_neg = config.get("geo_lambda_neg", 1.0)
    args.geo_smooth_lambda = config.get("geo_smooth_lambda", 0.1)
    args.geo_max_points = config.get("geo_max_points", 200000)
    args.geo_sample_size = config.get("geo_sample_size", 800)
    args.geo_plane_tau = config.get("geo_plane_tau", 0.01)
    args.geo_neg_plane_tau = config.get("geo_neg_plane_tau", 0.02)
    args.geo_spatial_pos_scale = config.get("geo_spatial_pos_scale", 0.75)
    args.geo_normal_pos_tau = config.get("geo_normal_pos_tau", 0.75)
    args.geo_normal_neg_tau = config.get("geo_normal_neg_tau", 0.4)
    args.geo_neg_margin = config.get("geo_neg_margin", 0.8)
    args.geo_hard_neg_k = config.get("geo_hard_neg_k", 2)
    args.geo_normal_source = config.get("geo_normal_source", "surface_axis")
    args.geo_use_multiview_semantics = config.get("geo_use_multiview_semantics", False)
    args.geo_support_views = config.get("geo_support_views", 3)
    args.geo_sem_pos_ratio = config.get("geo_sem_pos_ratio", 0.7)
    args.geo_sem_min_views = config.get("geo_sem_min_views", 2)
    args.geo_sem_conf_tau = config.get("geo_sem_conf_tau", args.geo_sem_pos_ratio)
    args.geo_sem_ignore_label = config.get("geo_sem_ignore_label", -1)
    args.geo_normal_weight_lambda = config.get("geo_normal_weight_lambda", 5.0)
    args.geo_sem_same_boost = config.get("geo_sem_same_boost", 1.0)
    args.geo_sem_neg_boost = config.get("geo_sem_neg_boost", 1.0)
    args.geo_sem_conflict_penalty = config.get("geo_sem_conflict_penalty", 0.75)
    args.geo_alpha = config.get("geo_alpha", 2.0)
    args.geo_beta = config.get("geo_beta", 1.0)
    args.geo_gamma = config.get("geo_gamma", 1.0)
    args.geo_weight_power = config.get("geo_weight_power", 2.0)
    args.distill_enable = config.get("distill_enable", False)
    args.distill_weight_lambda = config.get("distill_weight_lambda", 0.1)
    args.distill_feature_dir = config.get("distill_feature_dir", "dino_feature")
    args.distill_loss_type = config.get("distill_loss_type", "cosine")
    args.distill_mask_eps = config.get("distill_mask_eps", 1e-6)
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
    args.sugar_lambda_opacity_entropy = config.get("sugar_lambda_opacity_entropy", 0.0)
    args.sugar_opacity_entropy_start_iter = config.get("sugar_opacity_entropy_start_iter", args.sugar_start_iter)
    args.sugar_opacity_entropy_end_iter = config.get("sugar_opacity_entropy_end_iter", args.geo_start_iter)
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
