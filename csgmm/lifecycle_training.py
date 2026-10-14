import csv,json,math,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .encoder import PatchEncoder
from .core import seed_all
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED=35001
ITERS=200
PER_CONDITION=4
MARGIN=.10
TEMPERATURE=.10

def normalize_get(x, cond, ids, mean, std):
    ids = np.asarray(ids, dtype=np.int64)
    value = np.asarray(x[ids], dtype=np.float32)
    value = (value - mean[cond[ids], None, :]) / std[cond[ids], None, :]
    return torch.from_numpy(value.transpose(0, 2, 1).copy())


def sample_balanced_ids(rng, by_condition):
    chunks = [rng.choice(ids, size=PER_CONDITION, replace=len(ids) < PER_CONDITION)
              for ids in by_condition]
    ids = np.concatenate(chunks)
    rng.shuffle(ids)
    return ids


def adjacent_positive_ids(ids, cond, rng):
    positives = []
    for index in ids:
        candidates = [candidate for candidate in
                      (index - 2, index - 1, index + 1, index + 2)
                      if 0 <= candidate < len(cond) and cond[candidate] == cond[index]]
        positives.append(int(rng.choice(candidates)) if candidates else int(index))
    return np.asarray(positives, dtype=np.int64)


def pretext_ids(ids, cond):
    output = []
    for index in ids:
        candidate = max(0, int(index) - 8)
        output.append(candidate if cond[candidate] == cond[index] else int(index))
    return np.asarray(output, dtype=np.int64)


def off_diagonal_covariance(value):
    value = value - value.mean(dim=0, keepdim=True)
    covariance = value.T @ value / max(len(value) - 1, 1)
    diagonal = torch.diagonal(covariance)
    return (covariance.square().sum() - diagonal.square().sum()) / covariance.shape[0]


def supcon_loss(features, labels, temperature=TEMPERATURE):
    logits = features @ features.T / temperature
    identity = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~identity)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive.sum(dim=1).clamp_min(1)
    return -((log_prob * positive).sum(dim=1) / positive_count).mean()


def mining_terms(za, zp, labels):
    distance = 1.0 - za @ zp.T
    identity = torch.eye(len(za), dtype=torch.bool, device=za.device)
    same = labels[:, None].eq(labels[None, :]) & ~identity
    different = ~labels[:, None].eq(labels[None, :])
    positive = torch.diagonal(distance)
    far_same = torch.where(same, distance, torch.full_like(distance, -float("inf"))).max(dim=1).values
    same_values = torch.where(same, distance, torch.full_like(distance, -float("inf")))
    top_count = max(1, PER_CONDITION - 1)
    top_same = same_values.topk(k=top_count, dim=1).values
    finite = torch.isfinite(top_same)
    soft_same = torch.where(finite, top_same, torch.zeros_like(top_same)).sum(dim=1) / finite.sum(dim=1).clamp_min(1)
    near_different = torch.where(different, distance, torch.full_like(distance, float("inf"))).min(dim=1).values
    far_different = torch.where(different, distance, torch.full_like(distance, -float("inf"))).max(dim=1).values
    return {
        "positive": positive,
        "far_same": far_same,
        "soft_same": soft_same,
        "near_different": near_different,
        "far_different": far_different,
    }


def objective(variant, ha, hp, za, zp, ua, up, labels):
    term = mining_terms(za, zp, labels)
    pd = term["positive"]
    components = {
        "positive_distance": pd.mean(),
        "hard_positive_distance": term["far_same"].mean(),
        "near_different_distance": term["near_different"].mean(),
        "far_different_distance": term["far_different"].mean(),
    }
    if variant == "cond_farneg":
        raw = F.relu(pd - term["far_different"] + MARGIN).mean()
        loss = raw / 10.0
        components["triplet_raw"] = raw
    elif variant == "cond_nearneg":
        raw = F.relu(pd - term["near_different"] + MARGIN).mean()
        loss = raw / 10.0
        components["triplet_raw"] = raw
    elif variant == "hardpos_max":
        raw = F.relu(pd - term["near_different"] + MARGIN).mean()
        hp_loss = term["far_same"].mean()
        loss = raw / 10.0 + 0.10 * hp_loss
        components.update(triplet_raw=raw, hard_positive_loss=hp_loss)
    elif variant == "hardpos_topk":
        raw = F.relu(pd - term["near_different"] + MARGIN).mean()
        hp_loss = term["soft_same"].mean()
        loss = raw / 10.0 + 0.10 * hp_loss
        components.update(triplet_raw=raw, hard_positive_loss=hp_loss)
    elif variant == "hardpos_vicreg":
        invariance = F.mse_loss(ua, up)
        std_a = torch.sqrt(ua.var(dim=0, unbiased=False) + 1e-4)
        std_p = torch.sqrt(up.var(dim=0, unbiased=False) + 1e-4)
        variance = 0.5 * (F.relu(1.0 - std_a).mean() + F.relu(1.0 - std_p).mean())
        covariance = off_diagonal_covariance(ua) + off_diagonal_covariance(up)
        hp_loss = term["far_same"].mean()
        loss = 25.0 * invariance + 25.0 * variance + covariance + 0.10 * hp_loss
        components.update(invariance=invariance, variance=variance,
                          covariance=covariance, hard_positive_loss=hp_loss)
    elif variant == "supcon_condition":
        features = torch.cat((za, zp), dim=0)
        repeated_labels = torch.cat((labels, labels), dim=0)
        supervised_contrast = supcon_loss(features, repeated_labels)
        loss = supervised_contrast
        components["supcon"] = supervised_contrast
    elif variant == "prototype_margin":
        all_z = torch.cat((za, zp), dim=0)
        all_labels = torch.cat((labels, labels), dim=0)
        prototypes = []
        center_losses = []
        for condition in torch.unique(labels):
            local = all_z[all_labels == condition]
            prototype = F.normalize(local.mean(dim=0), dim=0)
            prototypes.append(prototype)
            center_losses.append((1.0 - local @ prototype).mean())
        prototypes = torch.stack(prototypes)
        center = torch.stack(center_losses).mean()
        proto_distance = 1.0 - prototypes @ prototypes.T
        off = ~torch.eye(len(prototypes), dtype=torch.bool, device=prototypes.device)
        separation = F.relu(0.50 - proto_distance[off]).mean()
        local = pd.mean()
        loss = local + center + 0.25 * separation
        components.update(center=center, prototype_separation=separation, local_pair=local)
    else:
        raise ValueError(variant)
    return loss, components


def fit_variant(variant, x, cond, mean, std, output):
    seed_all(SEED)
    rng = np.random.default_rng(SEED)
    model = PatchEncoder(in_channels=x.shape[-1], projection_dim=256, use_revin=False).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    by_condition = [np.flatnonzero(cond == condition) for condition in sorted(np.unique(cond))]
    history = []
    best = float("inf")
    for step in range(1, ITERS + 1):
        ids = sample_balanced_ids(rng, by_condition)
        positive_ids = adjacent_positive_ids(ids, cond, rng)
        a = normalize_get(x, cond, ids, mean, std).to(DEVICE, non_blocking=True)
        p = normalize_get(x, cond, positive_ids, mean, std).to(DEVICE, non_blocking=True)
        labels = torch.from_numpy(cond[ids].astype(np.int64)).to(DEVICE)
        pretext_weight = max(0.0, 1.0 - step / (ITERS / 5.0))
        if pretext_weight > 0:
            q_ids = pretext_ids(ids, cond)
            q = normalize_get(x, cond, q_ids, mean, std).to(DEVICE, non_blocking=True)
            h = model.embedding(torch.cat((a, p, q), dim=0))
            ha, hp, hq = h[:len(a)], h[len(a):2 * len(a)], h[2 * len(a):]
        else:
            h = model.embedding(torch.cat((a, p), dim=0))
            ha, hp = h[:len(a)], h[len(a):]
            hq = None
        ua, up = model.projection(ha), model.projection(hp)
        za, zp = F.normalize(ua, dim=1), F.normalize(up, dim=1)
        representation_loss, components = objective(variant, ha, hp, za, zp, ua, up, labels)
        if pretext_weight > 0:
            positive_logits = model.classification_head(torch.cat((ha, hq), dim=1)).squeeze(1)
            permutation = torch.randperm(len(ha), device=DEVICE)
            negative_logits = model.classification_head(torch.cat((ha, ha[permutation]), dim=1)).squeeze(1)
            pretext = (F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
                       + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits)))
        else:
            pretext = representation_loss.new_zeros(())
        loss = representation_loss + pretext_weight * pretext
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()),
               "representation_loss": float(representation_loss.detach()),
               "pretext": float(pretext.detach()), "pretext_weight": pretext_weight}
        row.update({key: float(value.detach()) for key, value in components.items()})
        history.append(row)
        value = row["loss"]
        if np.isfinite(value) and value < best:
            best = value
            torch.save(model.state_dict(), output / "model.pt")
        if step == 1 or step % 20 == 0:
            print(variant, step, json.dumps(row), flush=True)
    model.load_state_dict(torch.load(output / "model.pt", map_location=DEVICE, weights_only=True))
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted(set().union(*(row.keys() for row in history)))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(history)
    json.dump({"variant": variant, "iterations": ITERS, "seed": SEED,
               "normal_train_only": True, "test_opened_during_training": False,
               "best_training_loss": best}, open(output / "training_audit.json", "w"), indent=2)
    return model, history
