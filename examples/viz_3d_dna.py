import os
import sys
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoModel, AutoTokenizer
from pyfaidx import Fasta

try:
    from caduceus_lrp.epi_head import EPIClassificationHead
except ImportError:
    sys.path.append(".")
    from caduceus_lrp.epi_head import EPIClassificationHead

# ================= CONFIGURATION =================
DEVICE = torch.device("cuda:0")
CHECKPOINT_PATH = "final_models/best_model.pt"
PARQUET_PATH = "GM12878_RNAPII_64k_strict.parquet" 
FASTA_PATH = "data/genome_data/hg19.fa"
MODEL_NAME = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
SEQ_LEN = 65536
# =================================================

# --- Model Classes (Standard) ---
class SaliencyModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.gradients = None
        self.embedding_layer = self._find_embedding_layer()
        self.hook_handle = self.embedding_layer.register_full_backward_hook(self._hook_fn)

    def _find_embedding_layer(self):
        model_to_search = self.backbone
        if hasattr(self.backbone, "backbone"): model_to_search = self.backbone.backbone
        if hasattr(model_to_search, "embeddings"): return model_to_search.embeddings
        if hasattr(model_to_search, "word_embeddings"): return model_to_search.word_embeddings
        if hasattr(model_to_search, "wte"): return model_to_search.wte
        for name, module in model_to_search.named_modules():
            if isinstance(module, nn.Embedding): return module
        raise AttributeError("Could not find embedding layer")

    def _hook_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, input_ids, em, pm):
        outputs = self.backbone(input_ids, output_hidden_states=True)
        hidden = outputs.last_hidden_state
        logits = self.head(hidden, em, pm)
        return logits

def get_saliency_hook(model, input_ids, em, pm):
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        logits = model(input_ids, em, pm)
        prob = torch.sigmoid(logits)
    logits.backward()
    if model.gradients is None: raise RuntimeError("Hook failed")
    saliency = model.gradients.abs().sum(dim=-1)
    return saliency.detach().cpu().numpy(), prob.item()

# --- 🧬 VISUALIZATION LOGIC ---

def generate_chunky_helix(visual_points, loop_radius=100, helix_radius=12, num_twists=15):
    """
    Generates coordinates for a 'fat' double helix that is easy to see.
    """
    t = np.linspace(0, np.pi, visual_points) # The horseshoe path
    
    # 1. Main Backbone Path (The Horseshoe)
    bx = loop_radius * np.cos(t)
    by = loop_radius * np.sin(t) * 0.5 # Flatten slightly for perspective
    bz = np.linspace(-40, 40, visual_points) # Vertical drift

    # 2. Tangent/Normal calculations for correct twisting
    # Tangent
    tx, ty, tz = np.gradient(bx), np.gradient(by), np.gradient(bz)
    norms = np.sqrt(tx**2 + ty**2 + tz**2)
    tx, ty, tz = tx/norms, ty/norms, tz/norms
    
    # Normal (Arbitrary Up vector cross Tangent)
    ux, uy, uz = 0, 0, 1
    nx = uy*tz - uz*ty
    ny = uz*tx - ux*tz
    nz = ux*ty - uy*tx
    nnorms = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx/nnorms, ny/nnorms, nz/nnorms
    
    # Binormal (Tangent cross Normal)
    bnx = ty*nz - tz*ny
    bny = tz*nx - tx*nz
    bnz = tx*ny - ty*nx
    
    # 3. Create Two Strands
    # Twist angle
    theta = np.linspace(0, num_twists * 2 * np.pi, visual_points)
    
    def get_strand_coords(phase_offset):
        # Circle in the local Normal/Binormal plane
        local_x = helix_radius * np.cos(theta + phase_offset)
        local_y = helix_radius * np.sin(theta + phase_offset)
        
        # Project onto global coordinates
        sx = bx + local_x * nx + local_y * bnx
        sy = by + local_x * ny + local_y * bny
        sz = bz + local_x * nz + local_y * bnz
        return sx, sy, sz

    strand1 = get_strand_coords(0)
    strand2 = get_strand_coords(np.pi) # Offset by 180 degrees
    
    return strand1, strand2

def plot_molecular_dna(full_scores, row_idx, e_range, p_range):
    print(f"🎨 Generating Molecular-Style DNA Plot for Sample {row_idx}...")
    
    # --- 1. Downsample Data (The "Chunky" Fix) ---
    # We condense 65,536 points into ~800 visual beads so we can see the helix
    VISUAL_POINTS = 800
    chunk_size = len(full_scores) // VISUAL_POINTS
    
    # Average the scores for each chunk
    visual_scores = full_scores[:VISUAL_POINTS*chunk_size].reshape(VISUAL_POINTS, chunk_size).max(axis=1)
    
    # Map ranges to visual indices
    v_e_start = e_range[0] // chunk_size
    v_e_end = e_range[1] // chunk_size
    v_p_start = p_range[0] // chunk_size
    v_p_end = p_range[1] // chunk_size

    # --- 2. Generate Geometry ---
    (x1, y1, z1), (x2, y2, z2) = generate_chunky_helix(VISUAL_POINTS, helix_radius=8, num_twists=12)
    
    # --- 3. Setup Colors ---
    # Normalize scores for glowing effect
    # We use a non-linear scaling (power of 2) to make high scores REALLY pop
    norm_scores = (visual_scores - visual_scores.min()) / (visual_scores.max() - visual_scores.min() + 1e-9)
    
    # Create Colormap (Dark Blue -> Purple -> Red -> Yellow -> White)
    colors = ["#1a1a40", "#4a1a50", "#cc0000", "#ffaa00", "#ffffff"]
    cm = LinearSegmentedColormap.from_list("bioglow", colors, N=256)
    
    point_colors = cm(norm_scores)
    
    # --- 4. Plotting ---
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Aesthetics
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.axis('off')
    ax.view_init(elev=20, azim=-60)
    
    # A) Plot the "Rungs" (Base Pairs)
    # We draw a line between Strand 1 and Strand 2 for every visual point
    # We color the rung based on the importance score
    for i in range(VISUAL_POINTS):
        # Alpha depends on importance (boring parts are transparent ghost lines)
        alpha = 0.1 + 0.9 * norm_scores[i]
        lw = 1 + 4 * norm_scores[i] # Hotter = Thicker rung
        
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], 
                color=point_colors[i], alpha=alpha, linewidth=lw)

    # B) Plot the Backbones (Space Filling "Atoms")
    # Sugar-Phosphate backbone is usually gray/neutral, but we will make it glow slightly
    
    # Size varies by importance (Hot regions explode)
    sizes = 20 + 200 * (norm_scores ** 2)
    
    # Strand 1 Atoms
    ax.scatter(x1, y1, z1, c=point_colors, s=sizes, edgecolors='none', alpha=0.8, depthshade=True)
    # Strand 2 Atoms
    ax.scatter(x2, y2, z2, c=point_colors, s=sizes, edgecolors='none', alpha=0.8, depthshade=True)

    # --- 5. Highlights ---
    def add_label(start, end, text, color):
        mid = (start + end) // 2
        # Offset label slightly up
        ax.text(x1[mid], y1[mid], z1[mid]+15, text, color=color, fontsize=20, fontweight='bold', ha='center')
        
        # Draw a "Highlight Tube" around the backbone for these regions
        ax.plot(x1[start:end], y1[start:end], z1[start:end], color=color, linewidth=8, alpha=0.6)
        ax.plot(x2[start:end], y2[start:end], z2[start:end], color=color, linewidth=8, alpha=0.6)

    add_label(v_e_start, v_e_end, "Enhancer", "#00ffff") # Cyan
    add_label(v_p_start, v_p_end, "Promoter", "#00ff00") # Lime

    # --- 6. Final Polish ---
    plt.title(f"Predicted Chromatin Loop Interaction\n(Sample #{row_idx})", color='white', fontsize=22)
    
    # Tight layout ensures we don't save huge black borders
    plt.tight_layout()
    
    filename = f"dna_molecular_3d_{row_idx}.png"
    plt.savefig(filename, dpi=150, facecolor='black')
    print(f"✅ Saved Molecular Plot to {filename}")
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 Starting Molecular Visualization on {DEVICE}...")
    
    if not os.path.exists(PARQUET_PATH): PARQUET_PATH = f"work_dir/DL_2025 copy/{PARQUET_PATH}"
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df['chrom'] == 'chr1'].reset_index(drop=True)
    
    genome = Fasta(FASTA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load Model
    raw_backbone = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    raw_head = EPIClassificationHead(d_model=512, hidden_dim=512).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    raw_backbone.load_state_dict(checkpoint['backbone'])
    raw_head.load_state_dict(checkpoint['head'])
    model = SaliencyModel(raw_backbone, raw_head).to(DEVICE)
    
    # Search Loop
    found = 0
    start_index = 0
    
    print("🔍 Searching for a high-confidence positive...")
    
    for i in range(start_index, len(df)):
        torch.cuda.empty_cache()
        gc.collect()
        
        row = df.iloc[i]
        if row['label'] == 0: continue
        
        chrom, start = row['chrom'], int(row['win_start'])
        if chrom not in genome.keys():
             if chrom.startswith("chr") and chrom[3:] in genome.keys(): chrom = chrom[3:]
             elif ("chr" + chrom) in genome.keys(): chrom = "chr" + chrom
             else: continue

        try:
            seq = genome[chrom][start:start+SEQ_LEN].seq.upper()
        except KeyError: continue
        if len(seq) < SEQ_LEN: seq += "N"*(SEQ_LEN-len(seq))
        
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
            scores, conf = get_saliency_hook(model, inputs, em, pm)
            
            if conf > 0.85: # High confidence
                e_range = (int(row['enh_rel_start']), int(row['enh_rel_end']))
                p_range = (int(row['prom_rel_start']), int(row['prom_rel_end']))
                
                # 🎨 GENERATE PLOT
                plot_molecular_dna(scores[0], i, e_range, p_range)
                
                found += 1
                if found >= 1: break # Just 1 good image

        except RuntimeError as e:
            if "out of memory" in str(e): torch.cuda.empty_cache()
            continue
        finally:
            del inputs, em, pm