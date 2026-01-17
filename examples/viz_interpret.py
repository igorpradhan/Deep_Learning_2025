import os
import sys
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer
from pyfaidx import Fasta

# Import your head
try:
    from caduceus_lrp.epi_head import EPIClassificationHead
except ImportError:
    sys.path.append(".")
    from caduceus_lrp.epi_head import EPIClassificationHead

# ================= CONFIGURATION =================
DEVICE = torch.device("cuda:0")

MODEL_NAME = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
CHECKPOINT_PATH = "final_models/best_model.pt"
PARQUET_PATH = "GM12878_RNAPII_64k_strict.parquet" 
FASTA_PATH = "data/genome_data/hg19.fa"

SEQ_LEN = 65536
# =================================================

class SaliencyModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.gradients = None
        

        self.embedding_layer = self._find_embedding_layer()
        print(f"✅ Hook attached to: {self.embedding_layer}")
        self.hook_handle = self.embedding_layer.register_full_backward_hook(self._hook_fn)

    def _find_embedding_layer(self):
        model_to_search = self.backbone
        if hasattr(self.backbone, "backbone"):
            model_to_search = self.backbone.backbone
        
        # Try standard names
        if hasattr(model_to_search, "embeddings"): return model_to_search.embeddings
        if hasattr(model_to_search, "word_embeddings"): return model_to_search.word_embeddings
        if hasattr(model_to_search, "wte"): return model_to_search.wte
            
        # Recursive search
        for name, module in model_to_search.named_modules():
            if isinstance(module, nn.Embedding):
                return module
        raise AttributeError("Could not find an embedding layer!")

    def _hook_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, input_ids, em, pm):
        outputs = self.backbone(input_ids, output_hidden_states=True)
        hidden = outputs.last_hidden_state
        logits = self.head(hidden, em, pm)
        return logits

def get_saliency_hook(model, input_ids, em, pm):
    """
    Computes importance using AMP (Mixed Precision) to save memory.
    """
    model.eval()
    model.zero_grad(set_to_none=True) # Max memory clearing
    
    # 1. Forward Pass with AMP (Saves RAM)
    with torch.cuda.amp.autocast():
        logits = model(input_ids, em, pm)
        prob = torch.sigmoid(logits)
    
    # 2. Backward Pass
    # We must scale gradients if using scaler, but raw backward works for saliency visualization
    logits.backward()
    
    if model.gradients is None:
        raise RuntimeError("Hook failed to capture gradients!")
        
    saliency = model.gradients.sum(dim=-1)
    
    # Return numpy immediately to free graph
    return saliency.detach().cpu().numpy(), prob.item()

def plot_importance(scores, seq_str, enh_range, prom_range, row_idx):
    e_s, e_e = enh_range
    p_s, p_e = prom_range
    buffer = 100 
    
    def get_slice(start, end, max_len):
        s = max(0, start - buffer)
        e = min(max_len, end + buffer)
        return s, e

    e_view_s, e_view_e = get_slice(e_s, e_e, len(scores))
    p_view_s, p_view_e = get_slice(p_s, p_e, len(scores))
    
    enh_scores = scores[e_view_s:e_view_e]
    prom_scores = scores[p_view_s:p_view_e]
    
    # Normalize locally
    enh_scores = (enh_scores - enh_scores.min()) / (enh_scores.max() - enh_scores.min() + 1e-9)
    prom_scores = (prom_scores - prom_scores.min()) / (prom_scores.max() - prom_scores.min() + 1e-9)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    sns.heatmap([enh_scores], cmap="Reds", cbar=False, ax=axes[0], yticklabels=False)
    axes[0].set_title(f"Enhancer Region (Red=Important) | Global Pos: {e_view_s}-{e_view_e}")
    
    sns.heatmap([prom_scores], cmap="Blues", cbar=False, ax=axes[1], yticklabels=False)
    axes[1].set_title(f"Promoter Region (Blue=Important) | Global Pos: {p_view_s}-{p_view_e}")

    plt.suptitle(f"Interpretation for Sample #{row_idx}", fontsize=16)
    plt.tight_layout()
    output_filename = f"importance_example_{row_idx}.png"
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    plt.close(fig) # Close figure to free memory

# ... (Previous code remains the same) ...

if __name__ == "__main__":
    print(f"🚀 Starting Extraction & Storage on {DEVICE}...")
    
    # 1. Setup Output Directory
    OUTPUT_DIR = "saliency_scores"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Scores will be saved to: {OUTPUT_DIR}/")

    # 2. Load Data (Same as before)
    if not os.path.exists(PARQUET_PATH):
        if os.path.exists(f"./work_dir/DL_2025 copy/{PARQUET_PATH}"):
            PARQUET_PATH = f"./work_dir/DL_2025 copy/{PARQUET_PATH}"
    
    df = pd.read_parquet(PARQUET_PATH)
    # We filter for chr1, but you can remove this to do the whole dataset
    df = df[df['chrom'] == 'chr1'].reset_index(drop=True)
    
    genome = Fasta(FASTA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 3. Load Model (Same as before)
    raw_backbone = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    raw_head = EPIClassificationHead(d_model=512, hidden_dim=512).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    raw_backbone.load_state_dict(checkpoint['backbone'])
    raw_head.load_state_dict(checkpoint['head'])
    model = SaliencyModel(raw_backbone, raw_head).to(DEVICE)
    
    # 4. Processing Loop
    print(f"oProcessing {len(df)} samples...")
    
    # Optional: Metadata list to keep track of what we saved
    saved_metadata = []

    for i in range(len(df)):
        torch.cuda.empty_cache()
        gc.collect()
        
        row = df.iloc[i]
        if row['label'] == 0: continue # Skip negatives
        
        chrom, start = row['chrom'], int(row['win_start'])

        # Robust Chromosome Handling
        if chrom not in genome.keys():
             if chrom.startswith("chr") and chrom[3:] in genome.keys(): chrom = chrom[3:]
             elif ("chr" + chrom) in genome.keys(): chrom = "chr" + chrom
             else: continue

        try:
            seq = genome[chrom][start:start+SEQ_LEN].seq.upper()
        except KeyError: continue
        if len(seq) < SEQ_LEN: seq += "N"*(SEQ_LEN-len(seq))
        
        # Prepare Tensors
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        if inputs.shape[1] > SEQ_LEN: inputs = inputs[:, :SEQ_LEN]
        elif inputs.shape[1] < SEQ_LEN: 
             pad = torch.zeros((1, SEQ_LEN - inputs.shape[1]), dtype=torch.long, device=DEVICE)
             inputs = torch.cat([inputs, pad], dim=1)

        em = torch.zeros((1, SEQ_LEN), device=DEVICE)
        pm = torch.zeros((1, SEQ_LEN), device=DEVICE)
        em[0, int(row['enh_rel_start']):int(row['enh_rel_end'])] = 1.0
        pm[0, int(row['prom_rel_start']):int(row['prom_rel_end'])] = 1.0
        
        try:
            # === COMPUTE SCORES ===
            scores, conf = get_saliency_hook(model, inputs, em, pm)
            
            # === SAVE SCORES ===
            # Only save if the model is somewhat confident (optional, keeps data clean)
            if conf > 0.60:
                # 1. Define filename (e.g., score_SampleID_Chr_Start.npy)
                filename = f"score_{i}_{chrom}_{start}.npy"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # 2. Save the numpy array (Very fast & small)
                np.save(filepath, scores[0])
                
                print(f"💾 Saved Sample #{i} (Conf: {conf:.2f}) -> {filename}")
                
                # 3. Add to metadata
                saved_metadata.append({
                    "sample_idx": i,
                    "chrom": chrom,
                    "start": start,
                    "confidence": conf,
                    "file": filename
                })

        except RuntimeError as e:
            if "out of memory" in str(e): torch.cuda.empty_cache()
            continue
        finally:
            del inputs, em, pm, scores
    
    # Save the index (metadata) so you know which file is which
    pd.DataFrame(saved_metadata).to_csv(f"{OUTPUT_DIR}/metadata.csv", index=False)
    print("Done! Saved all scores and metadata.")