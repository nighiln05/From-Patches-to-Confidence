import os
import itertools
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance

from astra_attn_patch_dataset import ASTRA_EvalRGBDataset, ALL_MACHINE_TYPES
from patch_attn_model import PatchAttentionCLModel

# ─── CONFIG ─────────────────────────────────────────────────────────
ROOT_DIR = r"C:\Users\Nighil Natarajan\ckmam_proj\dcase"
CHECKPOINT_DIR = "checkpoints_april10"
EVAL_EPOCH = 144
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
PATCH_SIZE = 32
STRIDE = 16
NUM_WORKERS = 0

import random
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
# ─────────────────────────────────────────────────────────────────────


def infer_attr_dim(sd):
    key = "attn_pool.attr_bias.weight"
    return sd[key].shape[1] if key in sd else 0


def domain_subset(ds, domain, label_val=None):
    idxs = []
    for i, p in enumerate(ds.samples):
        b = os.path.basename(p)
        if domain in b:
            if label_val is None:
                idxs.append(i)
            else:
                is_normal = ("normal" in b)
                if (0 if is_normal else 1) == label_val:
                    idxs.append(i)
    return Subset(ds, idxs)


@torch.no_grad()
def extract_embeddings(loader, model):
    embs, labels = [], []
    model.eval()
    for batch in tqdm(loader, desc="Extract", leave=False):
        patches = batch["patches"].to(DEVICE)
        attrs = batch["attrs"].to(DEVICE)
        labs = batch["label"].cpu().numpy()
        B, N, C, H, W = patches.shape
        z = model(patches, B, N, attrs)
        z = F.normalize(z, dim=1).cpu().numpy()
        embs.append(z)
        labels.extend(labs.tolist())
    return np.vstack(embs), np.array(labels)


# ---------- modeling helpers ----------
def fit_cov(X, cov_type="lw"):
    if cov_type == "lw":
        est = LedoitWolf().fit(X); return est.location_, est.precision_
    if cov_type == "oas":
        est = OAS().fit(X); return est.location_, est.precision_
    if cov_type == "empirical":
        est = EmpiricalCovariance().fit(X); return est.location_, est.precision_
    if cov_type == "diag":
        mu = X.mean(0); var = X.var(0) + 1e-8
        return mu, np.diag(1.0 / var)
    raise ValueError(f"Unknown COV_TYPE={cov_type}")


def maha_sq_to_centers(X, mus, precisions):
    M = X.shape[0]; dmin = np.full(M, np.inf, dtype=np.float64)
    for mu, prec in zip(mus, precisions):
        d = X - mu[None, :]
        mk = np.einsum("bi,ij,bj->b", d, prec, d, optimize=True)
        dmin = np.minimum(dmin, mk)
    return dmin


def cos_dist_to_centers(Z_unit, centers_unit):
    sims = Z_unit @ centers_unit.T
    sims = np.clip(sims, -1, 1)
    return (1.0 - sims).min(axis=1)


def zscore(scores, mean, std):
    return (scores - mean) / (std + 1e-8)


def quantile(x, q):
    try:
        return float(np.percentile(x, q, method="nearest"))
    except TypeError:
        return float(np.percentile(x, q, interpolation="nearest"))


class DomainModel:
    def __init__(self, use_pca=True, pca_variance=0.98, cov_type="lw",
                 use_cosine=True, w_maha=0.7, w_cos=0.3, k=5,
                 thr_mode="fpr", target_fpr=0.05, perc_q=99.0,
                 std_k=2.5):   # ✅ added

        self.use_pca = use_pca
        self.pca_var = pca_variance
        self.cov_type = cov_type
        self.use_cosine = use_cosine
        self.w_maha = w_maha
        self.w_cos = w_cos
        self.k = k

        self.thr_mode = thr_mode
        self.target_fpr = target_fpr
        self.perc_q = perc_q
        self.std_k = std_k  # ✅ added

        self.pca = None
        self.kmeans = None
        self.mus = None
        self.precs = None
        self.cos_centers = None

        self.maha_mean = 0.0
        self.maha_std = 1.0
        self.cos_mean = 0.0
        self.cos_std = 1.0
        self.threshold = None

    def fit(self, Z_train_unit, rng=SEED):
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_var, random_state=rng)
            X = self.pca.fit_transform(Z_train_unit)
        else:
            X = Z_train_unit

        self.kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=rng)
        labels = self.kmeans.fit_predict(X)

        mus, precs = [], []
        for k in range(self.k):
            idx = np.where(labels == k)[0]
            if idx.size < 2:
                idx = np.arange(X.shape[0])
            mu_k, prec_k = fit_cov(X[idx], cov_type=self.cov_type)
            mus.append(mu_k)
            precs.append(prec_k)

        self.mus, self.precs = mus, precs

        if self.use_cosine:
            centers = []
            for k in range(self.k):
                idx = np.where(labels == k)[0]
                if idx.size == 0:
                    idx = np.arange(Z_train_unit.shape[0])
                c = Z_train_unit[idx].mean(0)
                c = c / (np.linalg.norm(c) + 1e-8)
                centers.append(c)
            self.cos_centers = np.stack(centers)

        X_maha = X
        maha_scores = maha_sq_to_centers(X_maha, self.mus, self.precs)
        self.maha_mean, self.maha_std = maha_scores.mean(), maha_scores.std()

        if self.use_cosine:
            cos_scores = cos_dist_to_centers(Z_train_unit, self.cos_centers)
            self.cos_mean, self.cos_std = cos_scores.mean(), cos_scores.std()
            combo = self.w_maha * zscore(maha_scores, self.maha_mean, self.maha_std) + \
                    self.w_cos * zscore(cos_scores, self.cos_mean, self.cos_std)
        else:
            combo = zscore(maha_scores, self.maha_mean, self.maha_std)

        # ✅ UPDATED threshold logic
        if self.thr_mode == "fpr":
            self.threshold = np.quantile(combo, 1 - self.target_fpr)
        elif self.thr_mode == "percentile":
            self.threshold = quantile(combo, self.perc_q)
        elif self.thr_mode == "std":
            self.threshold = combo.mean() + self.std_k * combo.std()

    def score(self, Z_unit):
        X = self.pca.transform(Z_unit) if self.use_pca else Z_unit
        maha = maha_sq_to_centers(X, self.mus, self.precs)
        maha_z = zscore(maha, self.maha_mean, self.maha_std)

        if self.use_cosine:
            cos = cos_dist_to_centers(Z_unit, self.cos_centers)
            cos_z = zscore(cos, self.cos_mean, self.cos_std)
            return self.w_maha * maha_z + self.w_cos * cos_z
        return maha_z

    def predict(self, Z_unit):
        s = self.score(Z_unit)
        return (s >= self.threshold).astype(int), s


# ---------- tuning ----------
def run_config(train_embs, test_embs, test_labels, cfg):
    dm = DomainModel(**cfg)
    dm.fit(train_embs)
    y_pred, scores = dm.predict(test_embs)

    auc = roc_auc_score(test_labels, scores)
    f1 = f1_score(test_labels, y_pred, zero_division=0)

    return {"auc": auc, "f1": f1, "thr": dm.threshold, "cfg": cfg}


def grid_configs():
    Ks = [1, 3, 5]
    covs = ["lw", "diag"]
    pca_opts = [(False, 1.0), (True, 0.95), (True, 0.98)]
    cos_opts = [(False, 1.0, 0.0), (True, 0.7, 0.3)]

    # ✅ includes std
    thr_opts = [
        ("fpr", 0.05, None),
        ("percentile", None, 99.0),
        ("std", None, None)
    ]

    for K, cov, (use_pca, pca_var), (use_cos, w_m, w_c), (tmode, tfpr, pq) in itertools.product(
        Ks, covs, pca_opts, cos_opts, thr_opts
    ):
        yield dict(
            use_pca=use_pca,
            pca_variance=pca_var,
            cov_type=cov,
            use_cosine=use_cos,
            w_maha=w_m,
            w_cos=w_c,
            k=K,
            thr_mode=tmode,
            target_fpr=(tfpr if tfpr is not None else 0.05),
            perc_q=(pq if pq is not None else 99.0),
            std_k=2.5  # ✅ added
        )


# ---------- MAIN ----------
def main():
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"epoch{EVAL_EPOCH}.pth"),
                      map_location=DEVICE)

    attr_dim = infer_attr_dim(ckpt["model_state"])
    model = PatchAttentionCLModel(embed_dim=128, attr_dim=attr_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    print(f"🔁 Loaded epoch{EVAL_EPOCH}")

    for m in ALL_MACHINE_TYPES:
        print(f"\n=== {m} ===")

        ds_train = ASTRA_EvalRGBDataset(ROOT_DIR, m, split="train",
                                        patch_size=PATCH_SIZE, stride=STRIDE,
                                        global_attr_dim=attr_dim)

        src_train = domain_subset(ds_train, "source", 0)
        tgt_train = domain_subset(ds_train, "target", 0)

        if len(src_train) == 0 or len(tgt_train) == 0:
            continue

        def emb(ds):
            return extract_embeddings(DataLoader(ds, batch_size=BATCH_SIZE), model)

        src_tr_embs, _ = emb(src_train)
        tgt_tr_embs, _ = emb(tgt_train)

        ds_test = ASTRA_EvalRGBDataset(ROOT_DIR, m, split="test",
                                       patch_size=PATCH_SIZE, stride=STRIDE,
                                       global_attr_dim=attr_dim)

        src_test = domain_subset(ds_test, "source")
        tgt_test = domain_subset(ds_test, "target")

        src_te_embs, src_labels = emb(src_test)
        tgt_te_embs, tgt_labels = emb(tgt_test)

        for dom, trE, teE, labels in [
            ("source", src_tr_embs, src_te_embs, src_labels),
            ("target", tgt_tr_embs, tgt_te_embs, tgt_labels)
        ]:

            results = [run_config(trE, teE, labels, cfg) for cfg in grid_configs()]
            best = sorted(results, key=lambda x: -x["auc"])[0]

            print(f"  [{dom}] AUC={best['auc']:.4f} F1={best['f1']:.4f}")
            print(f"       thr={best['thr']:.4f} | cfg={best['cfg']}")

    print("\n✅ Done")


if __name__ == "__main__":
    main()