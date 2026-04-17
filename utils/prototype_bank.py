import torch
import torch.nn.functional as F


def create_scene_prototype_bank(num_slots, feature_dim, device, eps=1e-8):
    bank = torch.randn((num_slots, feature_dim), device=device)
    bank = F.normalize(bank, dim=-1, eps=eps)
    return {
        "bank": bank,
        "valid": torch.zeros((num_slots,), device=device, dtype=torch.bool),
        "mass": torch.zeros((num_slots,), device=device),
        "age": torch.zeros((num_slots,), device=device),
    }


def get_active_prototype_ids(prototype_state, min_mass=0.0, max_active=0):
    valid = prototype_state["valid"]
    mass = prototype_state["mass"]
    active_ids = (valid & (mass >= float(min_mass))).nonzero(as_tuple=False).squeeze(-1)
    if active_ids.numel() == 0:
        active_ids = valid.nonzero(as_tuple=False).squeeze(-1)
    if max_active > 0 and active_ids.numel() > max_active:
        active_mass = mass[active_ids]
        _, topk_idx = active_mass.topk(max_active, largest=True)
        active_ids = active_ids[topk_idx]
    return active_ids


def compute_slot_assignments(features, prototype_state, active_ids=None, temperature=0.2, eps=1e-8):
    if features.numel() == 0:
        empty = features.new_zeros((0,))
        return empty, empty, empty, empty

    if active_ids is None:
        active_ids = get_active_prototype_ids(prototype_state)
    if active_ids.numel() == 0:
        empty_logits = features.new_zeros((features.shape[0], 0))
        empty_entropy = features.new_zeros((features.shape[0],))
        return empty_logits, empty_logits, empty_entropy, empty_entropy

    normalized_features = F.normalize(features, dim=-1, eps=eps)
    active_bank = F.normalize(prototype_state["bank"][active_ids], dim=-1, eps=eps)
    logits = torch.matmul(normalized_features, active_bank.transpose(0, 1)) / max(float(temperature), eps)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
    if active_ids.numel() > 1:
        entropy = entropy / torch.log(features.new_tensor(float(active_ids.numel())))
        topk = torch.topk(probs, k=min(2, active_ids.numel()), dim=-1)
        if active_ids.numel() > 1:
            margin = topk.values[:, 0] - topk.values[:, 1]
        else:
            margin = topk.values[:, 0]
    else:
        entropy = torch.zeros_like(entropy)
        margin = torch.ones_like(entropy)
    return logits, probs, entropy, margin


def _is_novel_candidate(feature, existing_bank, novelty_tau):
    if existing_bank.numel() == 0:
        return True
    similarity = torch.matmul(existing_bank, feature)
    return bool((similarity.max() <= float(novelty_tau)).item())


def bootstrap_scene_prototypes(
    prototype_state,
    features,
    seed_confidence,
    seed_mask=None,
    seed_labels=None,
    min_points=4,
    max_new_slots=4,
    novelty_tau=0.9,
    eps=1e-8,
):
    if max_new_slots <= 0 or features.numel() == 0:
        return 0

    valid = prototype_state["valid"]
    invalid_ids = (~valid).nonzero(as_tuple=False).squeeze(-1)
    if invalid_ids.numel() == 0:
        return 0

    if seed_mask is None:
        seed_mask = torch.ones((features.shape[0],), device=features.device, dtype=torch.bool)
    candidate_mask = seed_mask & (seed_confidence > 0)
    if not candidate_mask.any():
        return 0

    normalized_features = F.normalize(features, dim=-1, eps=eps)
    existing_bank = prototype_state["bank"][valid]
    candidate_feats = []

    if seed_labels is not None:
        unique_labels = torch.unique(seed_labels[candidate_mask])
        for label in unique_labels.tolist():
            if label < 0:
                continue
            label_mask = candidate_mask & (seed_labels == label)
            if int(label_mask.sum().item()) < max(int(min_points), 1):
                continue
            weights = seed_confidence[label_mask]
            proto_feat = (normalized_features[label_mask] * weights.unsqueeze(-1)).sum(dim=0)
            proto_feat = F.normalize(proto_feat.unsqueeze(0), dim=-1, eps=eps).squeeze(0)
            score = weights.mean()
            candidate_feats.append((float(score.item()), proto_feat))

    if not candidate_feats:
        top_ids = torch.argsort(seed_confidence[candidate_mask], descending=True)
        candidate_indices = candidate_mask.nonzero(as_tuple=False).squeeze(-1)[top_ids]
        for point_idx in candidate_indices.tolist():
            candidate_feats.append((float(seed_confidence[point_idx].item()), normalized_features[point_idx]))
            if len(candidate_feats) >= max_new_slots * 4:
                break

    candidate_feats.sort(key=lambda item: item[0], reverse=True)
    added = 0

    with torch.no_grad():
        for score, feature in candidate_feats:
            if added >= max_new_slots or invalid_ids.numel() == 0:
                break
            combined_bank = existing_bank if existing_bank.numel() > 0 else prototype_state["bank"].new_zeros((0, feature.shape[0]))
            if not _is_novel_candidate(feature, combined_bank, novelty_tau):
                continue
            slot_id = invalid_ids[0]
            prototype_state["bank"][slot_id].copy_(feature)
            prototype_state["valid"][slot_id] = True
            prototype_state["mass"][slot_id] = max(float(score), 1.0)
            prototype_state["age"][slot_id] = 0.0
            existing_bank = prototype_state["bank"][prototype_state["valid"]]
            invalid_ids = (~prototype_state["valid"]).nonzero(as_tuple=False).squeeze(-1)
            added += 1

    return added


def update_scene_prototypes(
    prototype_state,
    features,
    probs,
    update_confidence,
    update_mask,
    active_ids,
    momentum=0.9,
    min_weight=1.0,
    eps=1e-8,
):
    if active_ids.numel() == 0 or features.numel() == 0 or probs.numel() == 0:
        return 0

    normalized_features = F.normalize(features, dim=-1, eps=eps)
    hard_assign = probs.argmax(dim=-1)
    updated = 0

    with torch.no_grad():
        prototype_state["age"][prototype_state["valid"]] += 1
        for local_slot, global_slot in enumerate(active_ids.tolist()):
            slot_mask = update_mask & (hard_assign == local_slot)
            if not slot_mask.any():
                continue
            slot_weight = update_confidence[slot_mask]
            total_weight = slot_weight.sum()
            if float(total_weight.item()) < float(min_weight):
                continue
            proto_feat = (normalized_features[slot_mask] * slot_weight.unsqueeze(-1)).sum(dim=0)
            proto_feat = F.normalize(proto_feat.unsqueeze(0), dim=-1, eps=eps).squeeze(0)
            if prototype_state["valid"][global_slot]:
                proto_feat = momentum * prototype_state["bank"][global_slot] + (1.0 - momentum) * proto_feat
                proto_feat = F.normalize(proto_feat.unsqueeze(0), dim=-1, eps=eps).squeeze(0)
            prototype_state["bank"][global_slot].copy_(proto_feat)
            prototype_state["valid"][global_slot] = True
            prototype_state["mass"][global_slot] += total_weight.to(prototype_state["mass"].dtype)
            prototype_state["age"][global_slot] = 0.0
            updated += 1

    return updated
