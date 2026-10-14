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
from .core import seed_all,dump,get_array,sample_balanced,representation

def adjacent(ids, allowed_set, cond, rng):
    answer = []
    for index in ids:
        candidates = [q for q in (index-2, index-1, index+1, index+2)
                      if q in allowed_set and cond[q] == cond[index]]
        answer.append(int(rng.choice(candidates)) if candidates else int(index))
    return np.asarray(answer, dtype=np.int64)


def pretext_indices(ids, allowed_set, cond):
    answer = []
    for index in ids:
        candidate = int(index) - 8
        answer.append(candidate if candidate in allowed_set and cond[candidate] == cond[index] else int(index))
    return np.asarray(answer, dtype=np.int64)


def fit_encoder(kind, train, mean, std, seed, output, smoke=False):
    seed_all(seed); rng = np.random.default_rng(seed)
    x, cond, allowed = train["fit_x"], train["fit_cond"], train["fit_ids"]
    allowed_set = set(map(int, allowed))
    model = PatchEncoder(in_channels=x.shape[-1], projection_dim=256, use_revin=False).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history, best, steps = [], float("inf"), 2 if smoke else ITERS
    for step in range(1, steps + 1):
        ids = sample_balanced(rng, allowed, cond)
        pos_ids = adjacent(ids, allowed_set, cond, rng)
        a = get_array(x, cond, ids, mean, std).to(DEVICE)
        p = get_array(x, cond, pos_ids, mean, std).to(DEVICE)
        conditions = torch.from_numpy(cond[ids].astype(np.int64)).to(DEVICE)
        weight = max(0.0, 1.0 - step / (ITERS / 5.0))
        if weight > 0:
            q_ids = pretext_indices(ids, allowed_set, cond)
            q = get_array(x, cond, q_ids, mean, std).to(DEVICE)
            h = model.embedding(torch.cat((a, p, q)))
            ha, hp, hq = h[:len(a)], h[len(a):2*len(a)], h[2*len(a):]
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
            pre = (F.binary_cross_entropy_with_logits(poslog, torch.ones_like(poslog)) +
                   F.binary_cross_entropy_with_logits(neglog, torch.zeros_like(neglog)))
        else: pre = rep.new_zeros(())
        loss = rep + weight * pre
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        row = dict(step=step, loss=float(loss.detach()), representation_loss=float(rep.detach()),
                   pretext_loss=float(pre.detach()), pretext_weight=weight)
        row.update({k: float(v.detach()) for k, v in components.items()}); history.append(row)
        if np.isfinite(row["loss"]) and row["loss"] < best:
            best = row["loss"]
            if not smoke: torch.save(model.state_dict(), output / "model.pt")
        if step == 1 or step % 20 == 0:
            print("TRAIN", output.parent.parent.name, seed, kind, json.dumps(row), flush=True)
    if smoke: return model
    model.load_state_dict(torch.load(output / "model.pt", map_location=DEVICE, weights_only=True))
    fields = sorted(set().union(*(row.keys() for row in history)))
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(history)
    dump(dict(encoder=kind, seed=seed, iterations=ITERS, best_loss=best, normal_only=True,
              test_arrays_opened_during_training=False, fit_count=int(len(allowed))), output / "training_audit.json")
    return model

