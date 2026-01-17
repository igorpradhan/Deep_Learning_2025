import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import modiscolite
import numpy as np
import pickle

INPUT_FILE = "modisco_windows_200bp.h5"   
OUT_PKL = "modisco_results_200bp.pkl"

print("Loading Data...")
with h5py.File(INPUT_FILE, "r") as f:
    contrib_scores = f["contrib_scores"][:].astype(np.float32)  
    one_hot_seqs   = f["one_hot_seqs"][:].astype(np.float32)    

print("Loaded:", contrib_scores.shape, one_hot_seqs.shape)

# modiscolite expects (N, 4, L)
# contrib_scores = np.transpose(contrib_scores, (0, 2, 1)).copy()
# one_hot_seqs   = np.transpose(one_hot_seqs,   (0, 2, 1)).copy()

print("For modiscolite:", contrib_scores.shape, one_hot_seqs.shape)
print("frac<0:", (contrib_scores < 0).mean(), "min/max:", contrib_scores.min(), contrib_scores.max())

# Optional: tiny noise to avoid edge-case thresholding issues
rng = np.random.RandomState(0)
contrib_scores += rng.normal(scale=1e-6, size=contrib_scores.shape).astype(np.float32)

print("🔬 Running TF-MoDISco...")
pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
    hypothetical_contribs=contrib_scores,
    one_hot=one_hot_seqs,
    sliding_window_size=20,          
    target_seqlet_fdr=0.05,          
    max_seqlets_per_metacluster=20000,
    verbose=True
)

print("Saving results...")
with open(OUT_PKL, "wb") as f:
    pickle.dump((pos_patterns, neg_patterns), f)

n_pos = len(pos_patterns) if pos_patterns else 0
n_neg = len(neg_patterns) if neg_patterns else 0
print(f"Done! Found {n_pos} positive motifs and {n_neg} negative motifs.")
