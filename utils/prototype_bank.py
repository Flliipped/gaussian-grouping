import torch
import torch.nn as nn
import torch.nn.functional as F


class ScenePrototypeBank(nn.Module):
    def __init__(
        self,
        num_prototypes,
        feature_dim,
        tau=0.2,
        momentum=0.95,
        init_max_samples=4096,
        eps=1e-8,
        device="cuda",
    ):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.feature_dim = int(feature_dim)
        self.tau = float(tau)
        self.momentum = float(momentum)
        self.init_max_samples = int(init_max_samples)
        self.eps = float(eps)

        prototypes = torch.zeros(
            (self.num_prototypes, self.feature_dim),
            dtype=torch.float32,
            device=device,
        )
        usage = torch.zeros((self.num_prototypes,), dtype=torch.float32, device=device)

        self.register_buffer("prototypes", prototypes)
        self.register_buffer("usage_ema", usage)
        self.register_buffer("initialized", torch.tensor(False, device=device))

    def _normalize(self, x):
        return F.normalize(x, dim=-1, eps=self.eps)

    def _subsample(self, features):
        if features.shape[0] <= self.init_max_samples:
            return features
        indices = torch.randperm(features.shape[0], device=features.device)[: self.init_max_samples]
        return features[indices]

    def _farthest_point_init(self, features):
        features = self._normalize(self._subsample(features))
        if features.shape[0] == 0:
            return None

        num_points = features.shape[0]
        num_select = min(self.num_prototypes, num_points)
        selected = torch.empty((num_select,), dtype=torch.long, device=features.device)
        selected[0] = torch.randint(0, num_points, (1,), device=features.device)

        min_dist = 1.0 - torch.matmul(features, features[selected[0]].unsqueeze(-1)).squeeze(-1)
        for idx in range(1, num_select):
            selected[idx] = torch.argmax(min_dist)
            candidate = 1.0 - torch.matmul(features, features[selected[idx]].unsqueeze(-1)).squeeze(-1)
            min_dist = torch.minimum(min_dist, candidate)

        init_prototypes = features[selected]
        if num_select < self.num_prototypes:
            repeat_count = self.num_prototypes - num_select
            extra_indices = selected[torch.arange(repeat_count, device=features.device) % num_select]
            init_prototypes = torch.cat([init_prototypes, features[extra_indices]], dim=0)
        return init_prototypes

    @torch.no_grad()
    def initialize(self, features):
        init_prototypes = self._farthest_point_init(features)
        if init_prototypes is None:
            return False

        self.prototypes.copy_(self._normalize(init_prototypes))
        self.usage_ema.zero_()
        self.initialized.fill_(True)
        return True

    def assign(self, features):
        normalized_features = self._normalize(features)
        if not bool(self.initialized.item()):
            initialized = self.initialize(normalized_features.detach())
            if not initialized:
                empty_logits = normalized_features.new_zeros((normalized_features.shape[0], self.num_prototypes))
                empty_probs = normalized_features.new_full(
                    (normalized_features.shape[0], self.num_prototypes),
                    1.0 / max(self.num_prototypes, 1),
                )
                return empty_logits, empty_probs

        logits = torch.matmul(normalized_features, self.prototypes.t()) / max(self.tau, self.eps)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    @torch.no_grad()
    def ema_update(self, features, probs, confidence, confidence_thresh=0.2, sample_weight=None):
        if features.numel() == 0:
            return

        normalized_features = self._normalize(features.detach())
        if not bool(self.initialized.item()):
            initialized = self.initialize(normalized_features)
            if not initialized:
                return

        confidence = confidence.detach().clamp(min=0.0)
        active_mask = confidence >= float(confidence_thresh)
        if not active_mask.any():
            return

        selected_features = normalized_features[active_mask]
        selected_probs = probs.detach()[active_mask]
        selected_conf = confidence[active_mask]
        use_sample_weight = sample_weight is not None
        if use_sample_weight:
            sample_weight = sample_weight.detach().to(
                device=confidence.device,
                dtype=confidence.dtype,
            ).reshape(-1).clamp(min=0.0)
            if sample_weight.shape[0] != confidence.shape[0]:
                raise ValueError(
                    "sample_weight must have the same length as confidence in ScenePrototypeBank.ema_update"
                )
            selected_conf = selected_conf * sample_weight[active_mask]
            nonzero_weight_mask = selected_conf > 0
            if not nonzero_weight_mask.any():
                return
            selected_features = selected_features[nonzero_weight_mask]
            selected_probs = selected_probs[nonzero_weight_mask]
            selected_conf = selected_conf[nonzero_weight_mask]

        top_proto_idx = torch.argmax(selected_probs, dim=-1)
        hard_probs = F.one_hot(
            top_proto_idx,
            num_classes=self.num_prototypes,
        ).to(selected_probs.dtype)
        weighted_probs = hard_probs * selected_conf.unsqueeze(-1)

        proto_weight = weighted_probs.sum(dim=0)
        active_proto_mask = proto_weight > 0
        if not active_proto_mask.any():
            return

        proto_mean = torch.matmul(weighted_probs.t(), selected_features)
        proto_mean = proto_mean / proto_weight.unsqueeze(-1).clamp_min(self.eps)
        proto_mean = self._normalize(proto_mean)

        updated = self.prototypes.clone()
        updated[active_proto_mask] = self._normalize(
            self.momentum * self.prototypes[active_proto_mask]
            + (1.0 - self.momentum) * proto_mean[active_proto_mask]
        )
        self.prototypes.copy_(updated)

        if use_sample_weight:
            usage_target = proto_weight / selected_conf.sum().clamp_min(self.eps)
        else:
            usage_target = hard_probs.mean(dim=0)
        self.usage_ema.mul_(self.momentum).add_((1.0 - self.momentum) * usage_target)
