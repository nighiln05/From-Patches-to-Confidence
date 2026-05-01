import os
import itertools
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import roc_auc_score, f1_score
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

SEED = 42
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


# ---------- modeling ----------
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


def maha_sq_to_centers(X, mus, precs):
    dmin = np.full(X.shape[0], np.inf)
    for mu, prec in zip(mus, precs):
        d = X - mu
        mk = np.einsum("bi,ij,bj->b", d, prec, d)
        dmin = np.minimum(dmin, mk)
    return dmin


def cos_dist(Z, centers):
    sims = Z @ centers.T
    sims = np.clip(sims, -1, 1)
    return (1 - sims).min(axis=1)


def zscore(x, m, s):
    return (x - m) / (s + 1e-8)


class DomainModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, Z):
        if self.cfg["use_pca"]:
            self.pca = PCA(n_components=self.cfg["pca_variance"])
            X = self.pca.fit_transform(Z)
        else:
            self.pca = None
            X = Z

        kmeans = KMeans(n_clusters=self.cfg["k"], n_init=10)
        labels = kmeans.fit_predict(X)

        self.mus, self.precs = [], []

        for i in range(self.cfg["k"]):
            idx = np.where(labels == i)[0]
            if len(idx) < 2:
                idx = np.arange(len(X))
            mu, prec = fit_cov(X[idx], self.cfg["cov_type"])
            self.mus.append(mu)
            self.precs.append(prec)

        if self.cfg["use_cosine"]:
            centers = []
            for i in range(self.cfg["k"]):
                idx = np.where(labels == i)[0]
                if len(idx) == 0:
                    idx = np.arange(len(Z))
                c = Z[idx].mean(0)
                c = c / (np.linalg.norm(c) + 1e-8)
                centers.append(c)
            self.centers = np.stack(centers)

        maha = maha_sq_to_centers(X, self.mus, self.precs)
        self.m_m, self.m_s = maha.mean(), maha.std()

        if self.cfg["use_cosine"]:
            cos = cos_dist(Z, self.centers)
            self.c_m, self.c_s = cos.mean(), cos.std()
            combo = self.cfg["w_maha"] * zscore(maha, self.m_m, self.m_s) + \
                    self.cfg["w_cos"] * zscore(cos, self.c_m, self.c_s)
        else:
            combo = zscore(maha, self.m_m, self.m_s)

        if self.cfg["thr_mode"] == "fpr":
            self.thr = np.quantile(combo, 1 - self.cfg["target_fpr"])
        elif self.cfg["thr_mode"] == "percentile":
            self.thr = np.percentile(combo, self.cfg["perc_q"])
        else:
            self.thr = combo.mean() + 2.5 * combo.std()

    def score(self, Z):
        X = self.pca.transform(Z) if self.pca else Z
        maha = maha_sq_to_centers(X, self.mus, self.precs)
        maha_z = zscore(maha, self.m_m, self.m_s)

        if self.cfg["use_cosine"]:
            cos = cos_dist(Z, self.centers)
            cos_z = zscore(cos, self.c_m, self.c_s)
            return self.cfg["w_maha"] * maha_z + self.cfg["w_cos"] * cos_z

        return maha_z


# ---------- grid ----------
def grid_configs():
    Ks = [1, 3, 5, 10]  # ✅ added 10

    covs = ["lw", "diag"]
    pca_opts = [(False, 1.0), (True, 0.95), (True, 0.98)]
    cos_opts = [(False, 1.0, 0.0), (True, 0.7, 0.3)]

    thr_opts = [
        ("fpr", 0.05),
        ("percentile", 99.0),
        ("std", None)
    ]

    for K, cov, (use_pca, pca_var), (use_cos, w_m, w_c), (tmode, val) in itertools.product(
        Ks, covs, pca_opts, cos_opts, thr_opts
    ):
        yield {
            "k": K,
            "cov_type": cov,
            "use_pca": use_pca,
            "pca_variance": pca_var,
            "use_cosine": use_cos,
            "w_maha": w_m,
            "w_cos": w_c,
            "thr_mode": tmode,
            "target_fpr": val if tmode == "fpr" else 0.05,
            "perc_q": val if tmode == "percentile" else 99.0
        }


# ---------- main ----------
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

        src_tr, _ = emb(src_train)
        tgt_tr, _ = emb(tgt_train)

        ds_test = ASTRA_EvalRGBDataset(ROOT_DIR, m, split="test",
                                       patch_size=PATCH_SIZE, stride=STRIDE,
                                       global_attr_dim=attr_dim)

        src_test = domain_subset(ds_test, "source")
        tgt_test = domain_subset(ds_test, "target")

        src_te, src_y = emb(src_test)
        tgt_te, tgt_y = emb(tgt_test)

        for dom, trE, teE, labels in [
            ("source", src_tr, src_te, src_y),
            ("target", tgt_tr, tgt_te, tgt_y)
        ]:

            print(f"\n  [{dom}]")

            for K in [1, 3, 5, 10]:  # ✅ ablation loop
                cfgs = [cfg for cfg in grid_configs() if cfg["k"] == K]

                results = []
                for cfg in cfgs:
                    dm = DomainModel(cfg)
                    dm.fit(trE)
                    scores = dm.score(teE)

                    auc = roc_auc_score(labels, scores)
                    f1 = f1_score(labels, scores >= dm.thr)

                    results.append((auc, f1))

                best = sorted(results, key=lambda x: -x[0])[0]

                print(f"    K={K} → AUC={best[0]:.4f}, F1={best[1]:.4f}")

    print("\n✅ Done")


if __name__ == "__main__":
    main()