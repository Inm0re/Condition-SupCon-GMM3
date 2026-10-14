from __future__ import annotations
import csv,json,math,random,itertools,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score,average_precision_score,confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score
from .encoder import PatchEncoder
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITERS=200
PER_CONDITION=4
MARGIN=.10
TEMPERATURE=.10
EMBED_CHUNK=512
GMM_COMPONENTS=3
KNN_K=3
Q_THRESHOLD=.95

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def normalization(x, cond, ids):
    ids = np.asarray(ids, dtype=np.int64)
    ncond, channels = int(cond.max()) + 1, x.shape[-1]
    mean = np.zeros((ncond, channels), dtype=np.float64)
    second = np.zeros_like(mean)
    count = np.zeros(ncond, dtype=np.int64)
    for condition in range(ncond):
        local = ids[cond[ids] == condition]
        if not len(local):
            raise RuntimeError(f"no fit rows for condition {condition}")
        for start in range(0, len(local), 256):
            value = np.asarray(x[local[start:start + 256]], dtype=np.float64)
            count[condition] += value.shape[0] * value.shape[1]
            mean[condition] += value.sum((0, 1))
            second[condition] += np.square(value).sum((0, 1))
        mean[condition] /= count[condition]
        variance = second[condition] / count[condition] - np.square(mean[condition])
        second[condition] = np.sqrt(np.maximum(variance, 1e-8))
    return mean.astype(np.float32), second.astype(np.float32)


def get_array(x, cond, ids, mean, std):
    ids = np.asarray(ids, dtype=np.int64)
    value = np.asarray(x[ids], dtype=np.float32)
    value = (value - mean[cond[ids], None, :]) / std[cond[ids], None, :]
    return torch.from_numpy(value.transpose(0, 2, 1).copy())


def sample_balanced(rng, allowed, cond):
    chunks = []
    for condition in sorted(np.unique(cond[allowed])):
        local = allowed[cond[allowed] == condition]
        chunks.append(rng.choice(local, PER_CONDITION, replace=len(local) < PER_CONDITION))
    answer = np.concatenate(chunks)
    rng.shuffle(answer)
    return answer.astype(np.int64)


def neighbor_indices(ids, allowed_set, cond, group, rng, offsets):
    answer = []
    for index in ids:
        candidates = [
            q for q in (int(index) + offset for offset in offsets)
            if q in allowed_set and cond[q] == cond[index] and group[q] == group[index]
        ]
        answer.append(int(rng.choice(candidates)) if candidates else int(index))
    return np.asarray(answer, dtype=np.int64)


def supcon_loss(features, labels, temperature=TEMPERATURE):
    logits = features @ features.T / temperature
    identity = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = logits - logits.max(1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~identity)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))
    return -((log_prob * positive).sum(1) / positive.sum(1).clamp_min(1)).mean()


def representation(kind, za, zp, conditions):
    if kind == "base_paano":
        similarity = za @ zp.T
        positive = 1.0 - torch.diagonal(similarity)
        masked = similarity.clone()
        masked.diagonal().fill_(float("inf"))
        farthest = (1.0 - masked).max(1).values
        raw = F.relu(positive - farthest + MARGIN).mean()
        return raw / 10.0, {"positive_distance": positive.mean(), "farthest_negative_distance": farthest.mean(), "triplet_raw": raw}
    features = torch.cat((za, zp))
    if kind == "supcon_condition":
        labels = torch.cat((conditions, conditions))
    elif kind == "supcon_global_instance":
        instance = torch.arange(len(za), device=za.device)
        labels = torch.cat((instance, instance))
    else:
        raise ValueError(kind)
    loss = supcon_loss(features, labels)
    return loss, {"contrastive_loss": loss}


def fit_encoder(kind, train, mean, std, seed, output):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    x, cond, group, allowed = train["fit_x"], train["fit_cond"], train["fit_group"], train["fit_ids"]
    allowed_set = set(map(int, allowed))
    model = PatchEncoder(in_channels=x.shape[-1], projection_dim=256, use_revin=False).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history, best = [], float("inf")
    for step in range(1, ITERS + 1):
        ids = sample_balanced(rng, allowed, cond)
        pos_ids = neighbor_indices(ids, allowed_set, cond, group, rng, (-2, -1, 1, 2))
        a = get_array(x, cond, ids, mean, std).to(DEVICE)
        p = get_array(x, cond, pos_ids, mean, std).to(DEVICE)
        conditions = torch.from_numpy(cond[ids].astype(np.int64)).to(DEVICE)
        weight = max(0.0, 1.0 - step / (ITERS / 5.0))
        if weight > 0:
            q_ids = neighbor_indices(ids, allowed_set, cond, group, rng, (-8,))
            q = get_array(x, cond, q_ids, mean, std).to(DEVICE)
            h = model.embedding(torch.cat((a, p, q)))
            ha, hp, hq = h[:len(a)], h[len(a):2 * len(a)], h[2 * len(a):]
        else:
            h = model.embedding(torch.cat((a, p)))
            ha, hp, hq = h[:len(a)], h[len(a):], None
        za = F.normalize(model.projection(ha), dim=1)
        zp = F.normalize(model.projection(hp), dim=1)
        rep, components = representation(kind, za, zp, conditions)
        if weight > 0:
            poslog = model.classification_head(torch.cat((ha, hq), 1)).squeeze(1)
            perm = torch.randperm(len(ha), device=DEVICE)
            neglog = model.classification_head(torch.cat((ha, ha[perm]), 1)).squeeze(1)
            pre = (
                F.binary_cross_entropy_with_logits(poslog, torch.ones_like(poslog))
                + F.binary_cross_entropy_with_logits(neglog, torch.zeros_like(neglog))
            )
        else:
            pre = rep.new_zeros(())
        loss = rep + weight * pre
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()), "representation_loss": float(rep.detach()), "pretext_loss": float(pre.detach()), "pretext_weight": weight}
        row.update({key: float(value.detach()) for key, value in components.items()})
        history.append(row)
        if np.isfinite(row["loss"]) and row["loss"] < best:
            best = row["loss"]
            torch.save(model.state_dict(), output / "model.pt")
        if step == 1 or step % 20 == 0:
            print("TRAIN", output.parts[-4], seed, kind, json.dumps(row), flush=True)
    model.load_state_dict(torch.load(output / "model.pt", map_location=DEVICE, weights_only=True))
    fields = sorted(set().union(*(row.keys() for row in history)))
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    dump({"encoder": kind, "seed": seed, "iterations": ITERS, "best_loss": best, "normal_only": True, "group_boundary_safe_pairs": True, "test_arrays_opened_during_training": False, "fit_count": int(len(allowed))}, output / "training_audit.json")
    return model


@torch.inference_mode()
def embed(model, x, cond, ids, mean, std):
    model.eval()
    ids = np.asarray(ids, dtype=np.int64)
    answer = []
    for start in range(0, len(ids), EMBED_CHUNK):
        local = ids[start:start + EMBED_CHUNK]
        value = get_array(x, cond, local, mean, std).to(DEVICE)
        answer.append(model.embedding(value).float().cpu().numpy())
        if start == 0 or start % 20480 == 0:
            print("EMBED", start, "/", len(ids), flush=True)
    return np.concatenate(answer).astype(np.float32)


def unit(value):
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def total_bank_budget(n, ncond):
    return min(n, max(20 * ncond, min(2048, int(round(0.10 * n)))))


def make_coreset(value, seed, budget):
    budget = min(len(value), max(1, int(budget)))
    if budget >= len(value):
        return value
    normalized = unit(value)
    kmeans = MiniBatchKMeans(n_clusters=budget, random_state=seed, batch_size=max(1024, budget), max_iter=50, n_init=1).fit(normalized)
    selected = [int(np.square(normalized - center).sum(1).argmin()) for center in kmeans.cluster_centers_]
    return value[np.unique(selected)]


def knn_scores(value, bank, k=KNN_K):
    value_n, bank_n = unit(value), unit(bank)
    use_k = min(k, len(bank_n))
    answer = []
    for start in range(0, len(value_n), 1024):
        similarity = value_n[start:start + 1024] @ bank_n.T
        nearest = np.partition(similarity, -use_k, axis=1)[:, -use_k:]
        answer.append((1.0 - nearest).mean(1))
    return np.concatenate(answer).astype(np.float32)


def allocate_condition_budgets(cfit, total):
    conditions = sorted(np.unique(cfit))
    counts = {int(c): int(np.sum(cfit == c)) for c in conditions}
    budgets = {c: max(20, int(round(total * counts[c] / len(cfit)))) for c in counts}
    while sum(budgets.values()) > total:
        c = max((q for q in budgets if budgets[q] > 20), key=lambda q: budgets[q], default=None)
        if c is None:
            break
        budgets[c] -= 1
    while sum(budgets.values()) < total:
        c = max(budgets, key=lambda q: counts[q] - budgets[q])
        if budgets[c] >= counts[c]:
            break
        budgets[c] += 1
    return budgets


def knn_pair(zfit, zval, ztest, cfit, cval, ctest, seed, routing):
    total = total_bank_budget(len(zfit), len(np.unique(cfit)))
    if routing == "global":
        bank = make_coreset(zfit, seed, total)
        return knn_scores(zval, bank), knn_scores(ztest, bank), {"routing": routing, "bank_size": int(len(bank)), "target_budget": total}
    budgets = allocate_condition_budgets(cfit, total)
    val_score = np.empty(len(zval), dtype=np.float32)
    test_score = np.empty(len(ztest), dtype=np.float32)
    actual = {}
    for condition in sorted(np.unique(cfit)):
        bank = make_coreset(zfit[cfit == condition], seed + 100 * int(condition), budgets[int(condition)])
        val_score[cval == condition] = knn_scores(zval[cval == condition], bank)
        test_score[ctest == condition] = knn_scores(ztest[ctest == condition], bank)
        actual[str(int(condition))] = int(len(bank))
    return val_score, test_score, {"routing": routing, "bank_sizes": actual, "total_bank_size": int(sum(actual.values())), "target_budget": total}


def quadratic(value, mean, precision):
    delta = value - mean
    return np.einsum("ni,ij,nj->n", delta, precision, delta, optimize=True).astype(np.float32)


def fit_gmm(value, seed, components=GMM_COMPONENTS):
    use = min(components, len(value))
    km = MiniBatchKMeans(n_clusters=use, random_state=seed, batch_size=512, max_iter=100, n_init=5).fit(value)
    models, priors, counts = [], [], []
    for cluster in range(use):
        member = value[km.labels_ == cluster]
        if len(member) < 2:
            member = value
        models.append(LedoitWolf().fit(member))
        priors.append(max(len(member) / len(value), 1e-8))
        counts.append(int(len(member)))
    return models, priors, counts


def gmm_nll(value, models, priors):
    scores = []
    for model, prior in zip(models, priors):
        # Ledoit-Wolf is positive definite in theory, but very small SEU clusters
        # can still become numerically singular in float64 eigensolvers.  Use a
        # symmetric eigendecomposition with a relative floor so every NLL is
        # finite and the same rule is applied to every GMM result.
        covariance = 0.5 * (model.covariance_ + model.covariance_.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = max(float(np.max(eigenvalues)) * 1e-8, 1e-10)
        eigenvalues = np.maximum(eigenvalues, floor)
        precision = (eigenvectors / eigenvalues[None, :]) @ eigenvectors.T
        logdet = float(np.log(eigenvalues).sum())
        component = 0.5 * (quadratic(value, model.location_, precision) + logdet) - math.log(prior)
        scores.append(np.nan_to_num(component, nan=np.finfo(np.float32).max / 16,
                                    posinf=np.finfo(np.float32).max / 16,
                                    neginf=-np.finfo(np.float32).max / 16))
    return np.min(np.stack(scores, 1), 1).astype(np.float32)


def gmm_pair(zfit, zval, ztest, cfit, cval, ctest, seed, routing):
    if routing == "global":
        models, priors, counts = fit_gmm(zfit, seed)
        return gmm_nll(zval, models, priors), gmm_nll(ztest, models, priors), {"routing": routing, "components": len(models), "cluster_counts": counts}
    val_score = np.empty(len(zval), dtype=np.float32)
    test_score = np.empty(len(ztest), dtype=np.float32)
    audit = {}
    for condition in sorted(np.unique(cfit)):
        models, priors, counts = fit_gmm(zfit[cfit == condition], seed + 100 * int(condition))
        val_score[cval == condition] = gmm_nll(zval[cval == condition], models, priors)
        test_score[ctest == condition] = gmm_nll(ztest[ctest == condition], models, priors)
        audit[str(int(condition))] = counts
    return val_score, test_score, {"routing": routing, "components_per_condition": GMM_COMPONENTS, "cluster_counts": audit}


def robust_scale(reference):
    median = float(np.median(reference))
    q25, q75 = np.quantile(reference, [0.25, 0.75])
    return median, max(float(q75 - q25), 1e-8)


def calibrate(val_raw, test_raw, val_cond, test_cond, routing):
    val_score = np.empty(len(val_raw), dtype=np.float32)
    test_score = np.empty(len(test_raw), dtype=np.float32)
    audit = {}
    if routing == "global":
        median, scale = robust_scale(np.asarray(val_raw, dtype=np.float64))
        val_score[:] = (val_raw - median) / scale
        test_score[:] = (test_raw - median) / scale
        audit["global"] = {"count": int(len(val_raw)), "median": median, "iqr": scale, "test_condition_used": False}
        return val_score, test_score, audit
    for condition in sorted(np.unique(val_cond)):
        reference = np.asarray(val_raw[val_cond == condition], dtype=np.float64)
        median, scale = robust_scale(reference)
        val_score[val_cond == condition] = (reference - median) / scale
        rows = test_cond == condition
        test_score[rows] = (np.asarray(test_raw[rows]) - median) / scale
        audit[str(int(condition))] = {"count": int(len(reference)), "median": median, "iqr": scale, "test_condition_used": True}
    return val_score, test_score, audit

