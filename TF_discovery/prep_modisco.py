import os, glob
import numpy as np
import h5py
from pyfaidx import Fasta
from tqdm import tqdm

SCORES_DIR = "saliency_scores_2"
FASTA_PATH = "data/genome_data/hg19.fa"
OUTPUT_FILE = "modisco_windows_200bp.h5"

SEQ_LEN = 65536
WIN = 200
TOPK = 10
MIN_GAP = 200  # enforce non-overlap

BASE_TO_IDX = {'A':0,'C':1,'G':2,'T':3}

def one_hot_encode(seq_str):
    mat = np.zeros((len(seq_str), 4), dtype=np.float32)
    for i, b in enumerate(seq_str):
        j = BASE_TO_IDX.get(b)
        if j is not None:
            mat[i, j] = 1.0
    return mat

def pick_topk_windows(scores, win=200, topk=10, min_gap=200):
    pos = np.maximum(scores, 0.0)
    wsum = np.convolve(pos, np.ones(win, dtype=np.float32), mode="valid")
    order = np.argsort(wsum)[::-1]
    chosen = []
    for s in order:
        if wsum[s] <= 0:
            break
        if all(abs(s - c) >= min_gap for c in chosen):
            chosen.append(int(s))
            if len(chosen) >= topk:
                break
    return chosen

if __name__ == "__main__":
    score_files = sorted(glob.glob(os.path.join(SCORES_DIR, "score_*.npy")))
    print("found", len(score_files), "score files")

    genome = Fasta(FASTA_PATH)

    all_contrib = []
    all_onehot = []

    for fpath in tqdm(score_files):
        fname = os.path.basename(fpath)
        parts = fname.replace(".npy", "").split("_")
        chrom = parts[-2]
        start0 = int(parts[-1])

        scores_1d = np.load(fpath).astype(np.float32)
        if scores_1d.shape[0] != SEQ_LEN:
            continue

        # fetch 64kb sequence once
        try:
            seq = genome[chrom][start0:start0+SEQ_LEN].seq.upper()
        except KeyError:
            continue
        if len(seq) < SEQ_LEN:
            seq += "N"*(SEQ_LEN-len(seq))

        # choose top windows
        win_starts = pick_topk_windows(scores_1d, win=WIN, topk=TOPK, min_gap=MIN_GAP)
        if not win_starts:
            continue

        for ws in win_starts:
            we = ws + WIN
            seq_win = seq[ws:we]
            oh = one_hot_encode(seq_win)                   # (200,4)
            contrib = oh * scores_1d[ws:we, None]          # (200,4)

            # center per-window (gives + and - values)
            contrib = contrib - contrib.mean()

            all_onehot.append(oh)
            all_contrib.append(contrib)

    X = np.stack(all_onehot).astype(np.float32)   # (Nwin,200,4)
    C = np.stack(all_contrib).astype(np.float32)  # (Nwin,200,4)

    print("Final:", X.shape, C.shape)

    with h5py.File(OUTPUT_FILE, "w") as f:
        f.create_dataset("one_hot_seqs", data=X, compression="gzip")
        f.create_dataset("contrib_scores", data=C, compression="gzip")

    print("Wrote", OUTPUT_FILE)
