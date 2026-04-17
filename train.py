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
    loss_sugar_surface_alignment,
)
from utils.bcog_losses import compute_graph_reliability, loss_graph_contrastive, loss_object_prototype
from utils.prototype_bank import create_scene_prototype_bank
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import json

try:
    import wandb
except ImportError:
    wandb = None


def build_multiview_semantic_support(scene, gaussians, pipe, background, viewpoint_cam, visibility_filter, num_support_views):
    candidate_cameras = [cam for cam in scene.getTrainCameras() if cam.uid != viewpoint_cam.uid]
    num_support = min(max(int(num_support_views), 0), len(candidate_cameras))
    support_cameras = [viewpoint_cam]
    if num_support > 0:
        support_cameras.extend(sample(candidate_cameras, num_support))

    visibility_masks = [visibility_filter.detach()]
    if len(support_cameras) > 1:
        with torch.no_grad():
            for support_cam in support_cameras[1:]:
                support_render_pkg = render(support_cam, gaussians, pipe, background)
                visibility_masks.append(support_render_pkg["visibility_filter"].detach())

    return support_cameras, torch.stack(visibility_masks, dim=0)


def metric_or_default(metrics, key, reference_tensor):
    if metrics is None:
        return reference_tensor.new_tensor(0.0)
    value = metrics.get(key, None)
    if value is None:
        return reference_tensor.new_tensor(0.0)
    if torch.is_tensor(value):
        return value
    return reference_tensor.new_tensor(float(value))


def load_config_file(config_path, visited=None):
    resolved_path = os.path.abspath(config_path)
    if visited is None:
        visited = set()
    if resolved_path in visited:
        raise ValueError(f"Cyclic config inheritance detected at '{resolved_path}'.")

    with open(resolved_path, "r") as file:
        config = json.load(file)

    parent_ref = config.pop("extends", None)
    if parent_ref is None:
        return config

    parent_path = parent_ref if os.path.isabs(parent_ref) else os.path.join(os.path.dirname(resolved_path), parent_ref)
    parent_config = load_config_file(parent_path, visited | {resolved_path})
    parent_config.update(config)
    return parent_config


def run_prototype_disagreement_split(gaussians, prototype_state, proto_metrics, opt, reference_tensor):
    zero = reference_tensor.new_tensor(0.0)
    split_metrics = {
        "split_count": zero,
        "split_candidate_count": zero,
        "avg_split_ambiguity": zero,
    }
    if proto_metrics is None or prototype_state is None or int(opt.split_max_points) <= 0:
        return split_metrics

    required_keys = [
        "split_point_ids",
        "split_normals",
        "split_top2_ids",
        "split_top2_probs",
        "split_entropy",
        "split_margin",
        "split_ambiguity",
        "split_boundary_score",
        "split_boundary_mask",
    ]
    if any(proto_metrics.get(key, None) is None for key in required_keys):
        return split_metrics

    point_ids = proto_metrics["split_point_ids"].long()
    normal_dirs = proto_metrics["split_normals"]
    top2_ids = proto_metrics["split_top2_ids"].long()
    top2_probs = proto_metrics["split_top2_probs"]
    entropy = proto_metrics["split_entropy"]
    margin = proto_metrics["split_margin"]
    ambiguity = proto_metrics["split_ambiguity"]
    boundary_score = proto_metrics["split_boundary_score"]
    boundary_mask = proto_metrics["split_boundary_mask"].bool()

    if point_ids.numel() == 0 or top2_ids.numel() == 0:
        return split_metrics

    valid_pair_mask = (
        (top2_ids[:, 0] >= 0)
        & (top2_ids[:, 1] >= 0)
        & (top2_ids[:, 0] != top2_ids[:, 1])
    )
    candidate_mask = valid_pair_mask & boundary_mask
    candidate_mask &= boundary_score >= float(opt.split_boundary_tau)
    candidate_mask &= entropy >= float(opt.split_entropy_tau)
    candidate_mask &= margin <= float(opt.split_margin_tau)
    candidate_mask &= ambiguity >= float(opt.split_ambiguity_tau)

    candidate_count = int(candidate_mask.sum().item())
    split_metrics["split_candidate_count"] = zero.new_tensor(float(candidate_count))
    if candidate_count == 0:
        return split_metrics

    candidate_indices = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
    candidate_score = (
        ambiguity[candidate_indices]
        + boundary_score[candidate_indices]
        + (1.0 - margin[candidate_indices])
    )
    topk = min(candidate_count, int(opt.split_max_points))
    topk_order = torch.topk(candidate_score, k=topk, largest=True).indices
    selected_local = candidate_indices[topk_order]

    split_count = gaussians.split_ambiguous_gaussians(
        selected_idx=point_ids[selected_local],
        prototype_pair_ids=top2_ids[selected_local],
        prototype_bank=prototype_state["bank"].detach(),
        pair_probs=top2_probs[selected_local],
        normal_dirs=normal_dirs[selected_local],
        offset_scale=opt.split_offset_scale,
        scale_shrink=opt.split_scale_shrink,
        opacity_ratio=opt.split_opacity_ratio,
        feature_blend=opt.split_feature_blend,
    )
    split_metrics["split_count"] = zero.new_tensor(float(split_count))
    if split_count > 0:
        split_metrics["avg_split_ambiguity"] = ambiguity[selected_local].mean()
    return split_metrics


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
    prototype_state = create_scene_prototype_bank(opt.proto_num_slots, gaussians.num_objects, device="cuda")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

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

        loss_geo_raw = torch.tensor(0.0, device=image.device)
        loss_geo = torch.tensor(0.0, device=image.device)
        geo_coeff = torch.tensor(0.0, device=image.device)
        loss_proto_raw = torch.tensor(0.0, device=image.device)
        loss_proto = torch.tensor(0.0, device=image.device)
        proto_coeff = torch.tensor(0.0, device=image.device)
        loss_sugar_raw = torch.tensor(0.0, device=image.device)
        loss_sugar = torch.tensor(0.0, device=image.device)
        sugar_coeff = torch.tensor(0.0, device=image.device)
        sugar_axis_align_cosine = torch.tensor(0.0, device=image.device)
        sugar_plane_residual = torch.tensor(0.0, device=image.device)
        sugar_flat_ratio = torch.tensor(0.0, device=image.device)
        graph_metrics = {}
        proto_metrics = {}
        sugar_active = iteration >= opt.sugar_start_iter and iteration % opt.sugar_interval == 0
        geo_active = iteration >= opt.geo_start_iter and iteration % opt.geo_interval == 0
        split_due = (
            opt.split_enabled
            and opt.proto_enabled
            and iteration > opt.densify_until_iter
            and iteration >= opt.split_start_iter
            and iteration % opt.split_interval == 0
        )
        proto_active = (
            opt.proto_enabled
            and iteration >= opt.proto_start_iter
            and (iteration % opt.proto_interval == 0 or split_due)
        )
        graph_state = None

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

        support_cameras = None
        support_visibility = None
        needs_semantic_support = (geo_active and opt.geo_use_multiview_semantics) or proto_active
        if needs_semantic_support:
            support_view_budget = 0
            if geo_active and opt.geo_use_multiview_semantics:
                support_view_budget = max(support_view_budget, int(opt.geo_support_views))
            if proto_active:
                support_view_budget = max(support_view_budget, int(opt.proto_support_views))
            support_cameras, support_visibility = build_multiview_semantic_support(
                scene,
                gaussians,
                pipe,
                background,
                viewpoint_cam,
                visibility_filter,
                support_view_budget,
            )

        if geo_active or proto_active:
            graph_sem_min_views = max(
                int(opt.geo_sem_min_views) if geo_active else 0,
                int(opt.proto_min_views) if proto_active else 0,
            )
            graph_sem_conf_tau = max(
                float(opt.geo_sem_conf_tau) if geo_active else 0.0,
                float(opt.proto_conf_tau) if proto_active else 0.0,
            )
            graph_ignore_label = opt.proto_ignore_label if proto_active else opt.geo_sem_ignore_label
            graph_state = compute_graph_reliability(
                features=gaussians._objects_dc.squeeze(1),
                xyz=gaussians._xyz.squeeze().detach(),
                point_ids=torch.arange(gaussians._xyz.shape[0], device=gaussians._xyz.device),
                scaling=gaussians.get_scaling.detach(),
                k=opt.geo_knn_k,
                max_points=max(int(opt.geo_max_points), int(opt.proto_max_points) if proto_active else 0),
                sample_size=max(int(opt.geo_sample_size), int(opt.proto_sample_size) if proto_active else 0),
                plane_tau=opt.geo_plane_tau,
                neg_plane_tau=opt.geo_neg_plane_tau,
                spatial_pos_scale=opt.geo_spatial_pos_scale,
                normal_neg_tau=opt.geo_normal_neg_tau,
                reliability_pos_tau=opt.geo_reliability_pos_tau,
                reliability_neg_tau=opt.geo_reliability_neg_tau,
                reliability_alpha_dist=opt.geo_reliability_alpha_dist,
                reliability_alpha_normal=opt.geo_reliability_alpha_normal,
                reliability_alpha_plane=opt.geo_reliability_alpha_plane,
                reliability_alpha_mv=opt.geo_reliability_alpha_mv,
                support_cameras=support_cameras,
                support_visibility=support_visibility,
                sem_min_views=graph_sem_min_views,
                sem_conf_tau=graph_sem_conf_tau,
                sem_num_classes=num_classes,
                sem_ignore_label=graph_ignore_label,
                sem_conflict_penalty=opt.geo_sem_conflict_penalty,
                boundary_tau=opt.proto_boundary_tau,
            )

        if geo_active and graph_state is not None:
            warmup_iters = max(1, opt.geo_warmup_iters)
            geo_progress = min(max((iteration - opt.geo_start_iter) / warmup_iters, 0.0), 1.0)
            geo_coeff = image.new_tensor(opt.geo_weight_lambda * geo_progress)
            loss_geo_raw, graph_metrics = loss_graph_contrastive(
                graph_state=graph_state,
                lambda_val=1.0,
                lambda_pos=opt.geo_lambda_pos,
                lambda_neg=opt.geo_lambda_neg,
                neg_margin=opt.geo_neg_margin,
                hard_neg_k=opt.geo_hard_neg_k,
            )
            loss_geo = geo_coeff * loss_geo_raw
            loss = loss + loss_geo

        if proto_active and graph_state is not None:
            proto_warmup_iters = max(1, opt.proto_warmup_iters)
            proto_progress = min(max((iteration - opt.proto_start_iter) / proto_warmup_iters, 0.0), 1.0)
            proto_coeff = image.new_tensor(opt.proto_weight_lambda * proto_progress)
            loss_proto_raw, proto_metrics = loss_object_prototype(
                graph_state=graph_state,
                prototype_state=prototype_state,
                temperature=opt.proto_temperature,
                bank_conf_tau=opt.proto_bank_conf_tau,
                bank_momentum=opt.proto_bank_momentum,
                lambda_pull=opt.proto_lambda_pull,
                lambda_sep=opt.proto_lambda_sep,
                lambda_cons=opt.proto_lambda_cons,
                lambda_soft=opt.proto_lambda_soft,
                min_proto_points=opt.proto_min_points,
                active_count_tau=opt.proto_active_count_tau,
                max_active_prototypes=opt.proto_max_active,
                update_reliability_tau=opt.proto_update_reliability_tau,
                update_entropy_tau=opt.proto_update_entropy_tau,
                bootstrap_slots=opt.proto_bootstrap_slots,
                bootstrap_novelty_tau=opt.proto_bootstrap_novelty_tau,
                sep_margin=opt.proto_sep_margin,
                ambiguity_beta_entropy=opt.ambiguity_beta_entropy,
                ambiguity_beta_mv=opt.ambiguity_beta_mv,
                ambiguity_beta_rel=opt.ambiguity_beta_rel,
                ambiguity_beta_plane=opt.ambiguity_beta_plane,
                ambiguity_beta_scale=opt.ambiguity_beta_scale,
            )
            loss_proto = proto_coeff * loss_proto_raw
            loss = loss + loss_proto

        if iteration % 100 == 0 and (sugar_active or geo_active or proto_active):
            loss_obj_3d_value = loss_obj_3d.item() if loss_obj_3d is not None else 0.0
            message = (
                f"[Iter {iteration}] "
                f"loss_obj={loss_obj.item():.6f}, "
                f"loss_obj_3d={loss_obj_3d_value:.6f}, "
            )
            if sugar_active:
                message += (
                    f"loss_sugar={loss_sugar.item():.6f}, "
                    f"sugar_plane_residual={sugar_plane_residual.item():.6f}, "
                )
            if geo_active:
                message += (
                    f"loss_geo={loss_geo.item():.6f}, "
                    f"pos_loss={metric_or_default(graph_metrics, 'pos_loss', image).item():.6f}, "
                    f"neg_loss={metric_or_default(graph_metrics, 'neg_loss', image).item():.6f}, "
                    f"avg_reliability={metric_or_default(graph_metrics, 'avg_reliability', image).item():.6f}, "
                    f"boundary_ratio={metric_or_default(graph_metrics, 'boundary_ratio', image).item():.6f}, "
                )
            if proto_active:
                message += (
                    f"loss_proto={loss_proto.item():.6f}, "
                    f"proto_pull_loss={metric_or_default(proto_metrics, 'pull_loss', image).item():.6f}, "
                    f"proto_cons_loss={metric_or_default(proto_metrics, 'cons_loss', image).item():.6f}, "
                    f"active_proto_count={metric_or_default(proto_metrics, 'active_proto_count', image).item():.2f}, "
                    f"proto_avg_margin={metric_or_default(proto_metrics, 'avg_margin', image).item():.6f}, "
                    f"proto_avg_ambiguity={metric_or_default(proto_metrics, 'avg_ambiguity', image).item():.6f}, "
                )
            print(message.rstrip(", "))


        loss.backward()
        iter_end.record()


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
                loss_obj_3d,
                loss_sugar,
                sugar_active,
                sugar_plane_residual,
                loss_geo,
                geo_active,
                graph_metrics,
                loss_proto,
                proto_active,
                proto_metrics,
                use_wandb,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))
                if opt.proto_enabled:
                    torch.save(
                        {
                            "bank": prototype_state["bank"].detach().cpu(),
                            "valid": prototype_state["valid"].detach().cpu(),
                            "mass": prototype_state["mass"].detach().cpu(),
                            "age": prototype_state["age"].detach().cpu(),
                        },
                        os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration), "prototype_bank.pth"),
                    )

            if split_due:
                split_metrics = run_prototype_disagreement_split(
                    gaussians=gaussians,
                    prototype_state=prototype_state,
                    proto_metrics=proto_metrics,
                    opt=opt,
                    reference_tensor=image,
                )
                split_count = int(metric_or_default(split_metrics, "split_count", image).item())
                if split_count > 0:
                    print(
                        f"[Iter {iteration}] "
                        f"stage_d_split={split_count}, "
                        f"candidate_count={metric_or_default(split_metrics, 'split_candidate_count', image).item():.0f}, "
                        f"avg_split_ambiguity={metric_or_default(split_metrics, 'avg_split_ambiguity', image).item():.6f}"
                    )
                if use_wandb:
                    wandb.log(
                        {
                            "train_loss_patches/split_count": metric_or_default(split_metrics, "split_count", image).item(),
                            "train_loss_patches/split_candidate_count": metric_or_default(split_metrics, "split_candidate_count", image).item(),
                            "train_loss_patches/avg_split_ambiguity": metric_or_default(split_metrics, "avg_split_ambiguity", image).item(),
                            "iter": iteration,
                        }
                    )
            elif (
                use_wandb
                and opt.split_enabled
                and iteration >= opt.split_start_iter
                and iteration % opt.split_interval == 0
            ):
                wandb.log(
                    {
                        "train_loss_patches/split_count": 0.0,
                        "iter": iteration,
                    }
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


def training_report(iteration, Ll1, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, loss_sugar, sugar_active, sugar_plane_residual, loss_geo, geo_active, graph_metrics, loss_proto, proto_active, proto_metrics, use_wandb):

    if use_wandb:
        log_data = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "iter_time": elapsed,
            "iter": iteration
        }

        if loss_obj_3d is not None:
            log_data["train_loss_patches/loss_obj_3d"] = loss_obj_3d.item()

        if sugar_active:
            log_data.update({
                "train_loss_patches/loss_sugar": loss_sugar.item(),
                "train_loss_patches/sugar_plane_residual": sugar_plane_residual.item(),
            })

        if geo_active:
            log_data.update({
                "train_loss_patches/loss_geo": loss_geo.item(),
                "train_loss_patches/pos_loss": metric_or_default(graph_metrics, "pos_loss", Ll1).item(),
                "train_loss_patches/neg_loss": metric_or_default(graph_metrics, "neg_loss", Ll1).item(),
                "train_loss_patches/avg_reliability": metric_or_default(graph_metrics, "avg_reliability", Ll1).item(),
                "train_loss_patches/boundary_ratio": metric_or_default(graph_metrics, "boundary_ratio", Ll1).item(),
            })

        if proto_active:
            log_data.update({
                "train_loss_patches/loss_proto": loss_proto.item(),
                "train_loss_patches/proto_pull_loss": metric_or_default(proto_metrics, "pull_loss", Ll1).item(),
                "train_loss_patches/proto_cons_loss": metric_or_default(proto_metrics, "cons_loss", Ll1).item(),
                "train_loss_patches/active_proto_count": metric_or_default(proto_metrics, "active_proto_count", Ll1).item(),
                "train_loss_patches/proto_avg_margin": metric_or_default(proto_metrics, "avg_margin", Ll1).item(),
                "train_loss_patches/proto_avg_ambiguity": metric_or_default(proto_metrics, "avg_ambiguity", Ll1).item(),
            })

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
        config = load_config_file(args.config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
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
    args.geo_max_points = config.get("geo_max_points", 200000)
    args.geo_sample_size = config.get("geo_sample_size", 800)
    args.geo_plane_tau = config.get("geo_plane_tau", 0.01)
    args.geo_neg_plane_tau = config.get("geo_neg_plane_tau", 0.02)
    args.geo_spatial_pos_scale = config.get("geo_spatial_pos_scale", 0.75)
    args.geo_normal_neg_tau = config.get("geo_normal_neg_tau", 0.4)
    args.geo_neg_margin = config.get("geo_neg_margin", 0.8)
    args.geo_hard_neg_k = config.get("geo_hard_neg_k", 2)
    args.geo_use_multiview_semantics = config.get("geo_use_multiview_semantics", False)
    args.geo_support_views = config.get("geo_support_views", 3)
    args.geo_sem_min_views = config.get("geo_sem_min_views", 2)
    args.geo_sem_conf_tau = config.get("geo_sem_conf_tau", config.get("geo_sem_pos_ratio", 0.7))
    args.geo_sem_ignore_label = config.get("geo_sem_ignore_label", -1)
    args.geo_sem_conflict_penalty = config.get("geo_sem_conflict_penalty", 0.75)
    args.geo_reliability_pos_tau = config.get("geo_reliability_pos_tau", 0.65)
    args.geo_reliability_neg_tau = config.get("geo_reliability_neg_tau", 0.35)
    args.geo_reliability_alpha_dist = config.get("geo_reliability_alpha_dist", 1.25)
    args.geo_reliability_alpha_normal = config.get("geo_reliability_alpha_normal", 2.0)
    args.geo_reliability_alpha_plane = config.get("geo_reliability_alpha_plane", 1.5)
    args.geo_reliability_alpha_mv = config.get("geo_reliability_alpha_mv", 1.0)
    args.proto_enabled = config.get("proto_enabled", False)
    args.proto_num_slots = config.get("proto_num_slots", 16)
    args.proto_start_iter = config.get("proto_start_iter", max(args.geo_start_iter, args.densify_until_iter + 4000))
    args.proto_interval = config.get("proto_interval", 10)
    args.proto_warmup_iters = config.get("proto_warmup_iters", 3000)
    args.proto_weight_lambda = config.get("proto_weight_lambda", 0.0)
    args.proto_support_views = config.get("proto_support_views", args.geo_support_views)
    args.proto_max_points = config.get("proto_max_points", 200000)
    args.proto_sample_size = config.get("proto_sample_size", 1200)
    args.proto_min_views = config.get("proto_min_views", 2)
    args.proto_conf_tau = config.get("proto_conf_tau", 0.7)
    args.proto_bank_conf_tau = config.get("proto_bank_conf_tau", 0.8)
    args.proto_bank_momentum = config.get("proto_bank_momentum", 0.9)
    args.proto_temperature = config.get("proto_temperature", 0.2)
    args.proto_lambda_pull = config.get("proto_lambda_pull", 1.0)
    args.proto_lambda_sep = config.get("proto_lambda_sep", 0.25)
    args.proto_lambda_cons = config.get("proto_lambda_cons", 0.5)
    args.proto_lambda_soft = config.get("proto_lambda_soft", 0.25)
    args.proto_sep_margin = config.get("proto_sep_margin", 0.1)
    args.proto_min_points = config.get("proto_min_points", 4)
    args.proto_active_count_tau = config.get("proto_active_count_tau", 32)
    args.proto_max_active = config.get("proto_max_active", 16)
    args.proto_update_reliability_tau = config.get("proto_update_reliability_tau", 0.55)
    args.proto_update_entropy_tau = config.get("proto_update_entropy_tau", 0.45)
    args.proto_bootstrap_slots = config.get("proto_bootstrap_slots", 4)
    args.proto_bootstrap_novelty_tau = config.get("proto_bootstrap_novelty_tau", 0.9)
    args.proto_boundary_tau = config.get("proto_boundary_tau", 0.45)
    args.proto_ignore_label = config.get("proto_ignore_label", -1)
    args.ambiguity_beta_entropy = config.get("ambiguity_beta_entropy", 1.0)
    args.ambiguity_beta_mv = config.get("ambiguity_beta_mv", 1.0)
    args.ambiguity_beta_rel = config.get("ambiguity_beta_rel", 1.0)
    args.ambiguity_beta_plane = config.get("ambiguity_beta_plane", 0.5)
    args.ambiguity_beta_scale = config.get("ambiguity_beta_scale", 0.25)
    args.split_enabled = config.get("split_enabled", config.get("use_ambiguous_split", False))
    args.split_start_iter = config.get("split_start_iter", max(args.proto_start_iter + args.proto_warmup_iters, args.densify_until_iter + 12_000))
    args.split_interval = config.get("split_interval", 500)
    args.split_max_points = config.get("split_max_points", config.get("split_topk", 64))
    args.split_ambiguity_tau = config.get("split_ambiguity_tau", 1.1)
    args.split_entropy_tau = config.get("split_entropy_tau", 0.55)
    args.split_margin_tau = config.get("split_margin_tau", 0.2)
    args.split_boundary_tau = config.get("split_boundary_tau", 0.5)
    args.split_scale_shrink = config.get("split_scale_shrink", 0.6)
    args.split_offset_scale = config.get("split_offset_scale", 0.5)
    args.split_opacity_ratio = config.get("split_opacity_ratio", 0.5)
    args.split_feature_blend = config.get("split_feature_blend", 0.7)
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
        if wandb is None:
            raise ImportError("`--use_wandb` was set, but `wandb` is not installed.")
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
