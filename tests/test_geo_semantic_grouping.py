import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from utils.distill_utils import FeatureCacheLoader, cosine_distill_loss
from utils.loss_utils import (
    loss_cls_3d,
    loss_geo_gated_contrastive,
    loss_sugar_surface_alignment,
)


class GeoSemanticGroupingTests(unittest.TestCase):
    def test_geom_gate_accepts_same_plane_same_axis(self):
        torch.manual_seed(0)
        xyz = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=torch.float32)
        features = torch.tensor([[0.0] * 16, [0.1] * 16], dtype=torch.float32)
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

        outputs = loss_geo_gated_contrastive(
            xyz=xyz,
            features=features,
            normals=normals,
            k=1,
            sample_size=2,
            spatial_pos_scale=1.1,
            plane_tau=0.05,
            normal_pos_tau=0.9,
            neg_margin=1.0,
            hard_neg_k=1,
        )

        gate_ratio = outputs[1].item()
        pos_pairs = outputs[19].item()
        neg_pairs = outputs[20].item()
        self.assertGreater(gate_ratio, 0.9)
        self.assertGreater(pos_pairs, 0.0)
        self.assertEqual(neg_pairs, 0.0)

    def test_geom_gate_handles_axis_flip(self):
        torch.manual_seed(0)
        xyz = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=torch.float32)
        features = torch.tensor([[0.0] * 16, [0.2] * 16], dtype=torch.float32)
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float32)

        outputs = loss_geo_gated_contrastive(
            xyz=xyz,
            features=features,
            normals=normals,
            k=1,
            sample_size=2,
            spatial_pos_scale=1.1,
            plane_tau=0.05,
            normal_pos_tau=0.9,
            neg_margin=1.0,
            hard_neg_k=1,
        )

        gate_ratio = outputs[1].item()
        pos_pairs = outputs[19].item()
        self.assertGreater(gate_ratio, 0.9)
        self.assertGreater(pos_pairs, 0.0)

    def test_depth_discontinuity_becomes_boundary_negative(self):
        torch.manual_seed(0)
        xyz = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.35]], dtype=torch.float32)
        features = torch.tensor([[0.0] * 16, [0.05] * 16], dtype=torch.float32)
        normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

        outputs = loss_geo_gated_contrastive(
            xyz=xyz,
            features=features,
            normals=normals,
            k=1,
            sample_size=2,
            spatial_pos_scale=1.1,
            plane_tau=0.05,
            neg_plane_tau=0.1,
            normal_pos_tau=0.9,
            normal_neg_tau=0.2,
            neg_margin=1.0,
            hard_neg_k=1,
        )

        pos_pairs = outputs[19].item()
        neg_pairs = outputs[20].item()
        boundary_neg_ratio = outputs[21].item()
        self.assertEqual(pos_pairs, 0.0)
        self.assertGreater(neg_pairs, 0.0)
        self.assertGreater(boundary_neg_ratio, 0.9)

    def test_multiview_semantics_selects_positive_and_negative_pairs(self):
        torch.manual_seed(0)
        xyz = torch.tensor(
            [
                [-0.6, 0.0, 1.0],
                [-0.3, 0.0, 1.0],
                [0.4, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        features = torch.tensor(
            [
                [0.0] * 16,
                [0.1] * 16,
                [0.2] * 16,
            ],
            dtype=torch.float32,
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=torch.float32)

        object_mask = torch.tensor(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=torch.long,
        )
        identity = torch.eye(4, dtype=torch.float32)
        camera = SimpleNamespace(
            objects=object_mask,
            world_view_transform=identity,
            full_proj_transform=identity,
            image_width=4,
            image_height=4,
        )

        outputs = loss_geo_gated_contrastive(
            xyz=xyz,
            features=features,
            normals=normals,
            point_ids=torch.arange(xyz.shape[0]),
            k=2,
            sample_size=3,
            spatial_pos_scale=1.1,
            plane_tau=0.05,
            normal_pos_tau=0.9,
            neg_margin=1.0,
            hard_neg_k=2,
            support_cameras=[camera],
            sem_min_views=1,
            sem_conf_tau=0.5,
            sem_num_classes=3,
            sem_ignore_label=-1,
        )

        semantic_pos_keep_ratio = outputs[17].item()
        pos_pairs = outputs[19].item()
        neg_pairs = outputs[20].item()
        self.assertGreater(pos_pairs, 0.0)
        self.assertGreater(neg_pairs, 0.0)
        self.assertLess(semantic_pos_keep_ratio, 1.0)

    def test_feature_cache_loader_resizes_and_validates(self):
        loader = FeatureCacheLoader(Path("."))
        with mock.patch.object(FeatureCacheLoader, "_cache_path", return_value=Path(__file__)):
            with mock.patch("utils.distill_utils.torch.load", return_value=
                {
                    "feature_map": torch.randn(8, 2, 3),
                    "image_size": (12, 18),
                    "model_name": "dinov2_vitb14",
                }):
                feature_map, entry = loader.get_feature_map(
                    "frame_000",
                    target_hw=(6, 7),
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

        self.assertEqual(feature_map.shape, (8, 6, 7))
        self.assertEqual(entry["image_size"], (12, 18))
        self.assertEqual(entry["model_name"], "dinov2_vitb14")

    def test_feature_cache_loader_raises_on_missing_cache(self):
        loader = FeatureCacheLoader(Path("."))
        with self.assertRaises(FileNotFoundError):
            loader.get_feature_map("definitely_missing_frame")

    def test_combined_losses_backward_smoke(self):
        torch.manual_seed(0)
        num_points = 6
        object_features = torch.randn(num_points, 16, requires_grad=True)
        xyz = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [0.3, 0.0, 1.0],
                [0.4, 0.0, 1.0],
                [0.5, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        normals = torch.tensor([[0.0, 0.0, 1.0]] * num_points, dtype=torch.float32)

        render_object = torch.randn(16, 4, 4, requires_grad=True)
        classifier = torch.nn.Conv2d(16, 3, kernel_size=1)
        distill_head = torch.nn.Conv2d(16, 8, kernel_size=1)
        gt_obj = torch.randint(0, 3, (4, 4), dtype=torch.long)

        loss_obj = torch.nn.functional.cross_entropy(
            classifier(render_object).unsqueeze(0),
            gt_obj.unsqueeze(0),
        )

        logits3d = classifier(object_features.t().unsqueeze(-1))
        prob_obj3d = torch.softmax(logits3d, dim=0).squeeze(-1).permute(1, 0)
        loss_obj_3d = loss_cls_3d(
            xyz.detach(),
            prob_obj3d,
            k=3,
            lambda_val=1.0,
            max_points=100,
            sample_size=num_points,
        )

        scaling = torch.tensor(
            [
                [0.05, 0.2, 0.2],
                [0.05, 0.2, 0.2],
                [0.05, 0.2, 0.2],
                [0.05, 0.2, 0.2],
                [0.05, 0.2, 0.2],
                [0.05, 0.2, 0.2],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        rotation = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * num_points,
            dtype=torch.float32,
            requires_grad=True,
        )
        loss_sugar = loss_sugar_surface_alignment(
            xyz=xyz,
            scaling=scaling,
            rotation=rotation,
            k=3,
            lambda_val=1.0,
            max_points=100,
            sample_size=num_points,
        )[0]

        loss_geo = loss_geo_gated_contrastive(
            xyz=xyz,
            features=object_features,
            normals=normals,
            k=3,
            sample_size=num_points,
            spatial_pos_scale=1.1,
            plane_tau=0.05,
            normal_pos_tau=0.9,
            neg_margin=1.0,
            hard_neg_k=1,
        )[0]

        teacher_map = torch.randn(8, 4, 4)
        loss_distill = cosine_distill_loss(
            distill_head(render_object),
            teacher_map,
            mask=torch.ones(4, 4, dtype=torch.bool),
        )

        total_loss = loss_obj + loss_obj_3d + loss_sugar + loss_geo + loss_distill
        total_loss.backward()

        self.assertIsNotNone(classifier.weight.grad)
        self.assertIsNotNone(distill_head.weight.grad)
        self.assertIsNotNone(object_features.grad)
        self.assertIsNotNone(scaling.grad)
        self.assertIsNotNone(rotation.grad)


if __name__ == "__main__":
    unittest.main()
