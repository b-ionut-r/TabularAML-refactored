"""Public top-kernel PLAsTiCC features (Olivier's / Siddharth's kernels): per-object and per-passband flux statistics,
flux_ratio_sq / flux_by_flux_ratio_sq, detected-mjd span, and photo-z distance terms."""
import sys, os, time, numpy as np, pandas as pd
H = sys.argv[2] if len(sys.argv) > 2 else "hold"; out = sys.argv[1]; os.makedirs(out, exist_ok=True); t = time.time()
lc = pd.read_parquet(f"{H}/lc.parquet")
lc["flux_ratio_sq"] = (lc.flux / lc.flux_err) ** 2; lc["flux_by_flux_ratio_sq"] = lc.flux * lc.flux_ratio_sq
g = lc.groupby("object_id")
A = g.agg(flux_min=("flux", "min"), flux_max=("flux", "max"), flux_mean=("flux", "mean"), flux_median=("flux", "median"),
          flux_std=("flux", "std"), flux_skew=("flux", "skew"), flux_err_mean=("flux_err", "mean"), detected_mean=("detected_bool", "mean"),
          frs_sum=("flux_ratio_sq", "sum"), frs_skew=("flux_ratio_sq", "skew"), fbf_sum=("flux_by_flux_ratio_sq", "sum"), fbf_skew=("flux_by_flux_ratio_sq", "skew"))
A["flux_diff"] = A.flux_max - A.flux_min; A["flux_dif2"] = A.flux_diff / A.flux_mean; A["flux_w_mean"] = A.fbf_sum / A.frs_sum
A["flux_dif3"] = A.flux_diff / A.flux_w_mean
d = lc[lc.detected_bool == 1].groupby("object_id").mjd.agg(["min", "max"]); A["mjd_det_span"] = (d["max"] - d["min"]).reindex(A.index)
pb = lc.groupby(["object_id", "passband"]).flux.agg(["mean", "std", "max", "min", "skew"]).unstack()
pb.columns = [f"pb{b}_{s}" for s, b in pb.columns]
A = A.join(pb)
for b in range(6):
    A[f"pb{b}_norm_max"] = A[f"pb{b}_max"] / A.flux_max.abs().replace(0, np.nan)
for nm in ("train", "test"):
    m = pd.read_parquet(f"{H}/{nm}.parquet")
    m = m.join(A, on="object_id")
    m["abs_mag_proxy"] = -2.5 * np.log10(m.flux_max.clip(lower=1e-3)) - m.distmod
    m.to_parquet(f"{out}/{nm}_features.parquet", index=False)
print("hand features", A.shape[1] + 1, round(time.time() - t), "s")
