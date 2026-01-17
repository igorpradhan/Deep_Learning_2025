import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pyfaidx import Fasta
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score
import matplotlib.pyplot as plt

try:
    from caduceus_lrp.epi_head import EPIClassificationHead
except ImportError:
    from epi_head import EPIClassificationHead

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
PARQUET_OUTPUT = os.path.join(BASE_DIR, "GM12878_RNAPII_64k_strict.parquet")
FASTA_PATH = os.path.join(DATA_DIR, "genome_data/hg19.fa")

RESULTS_DIR = "results_finetune"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
SEQ_LEN = 65536
BATCH_SIZE = 4  
EPOCHS = 5
LR = 1e-4


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET CLASS
# ==========================================
class BENGIDataset(Dataset):
    def __init__(self, metadata_df, fasta_path, tokenizer):
        self.metadata = metadata_df.reset_index(drop=True)
        self.genome = Fasta(fasta_path)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        chrom = row['chrom']
        start = int(row['win_start'])
        end = start + SEQ_LEN
        
        try:
            seq_str = self.genome[chrom][start:end].seq.upper()
        except KeyError:
            seq_str = "N" * SEQ_LEN
        
        if len(seq_str) < SEQ_LEN:
            seq_str += "N" * (SEQ_LEN - len(seq_str))

        tokens = self.tokenizer(seq_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        if len(tokens) < SEQ_LEN:
            pad = torch.zeros(SEQ_LEN - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, pad])
        else:
            tokens = tokens[:SEQ_LEN]

        enh_mask = torch.zeros(SEQ_LEN, dtype=torch.float32)
        prom_mask = torch.zeros(SEQ_LEN, dtype=torch.float32)
        
        e_s = max(0, int(row['enh_rel_start']))
        e_e = min(SEQ_LEN, int(row['enh_rel_end']))
        p_s = max(0, int(row['prom_rel_start']))
        p_e = min(SEQ_LEN, int(row['prom_rel_end']))
        
        if e_e > e_s: enh_mask[e_s:e_e] = 1.0
        if p_e > p_s: prom_mask[p_s:p_e] = 1.0
            
        label = torch.tensor(row['label'], dtype=torch.float32)
        return tokens, enh_mask, prom_mask, label

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    print(f"🚀 Starting Fine-Tuning on {DEVICE}...")
    
    if not os.path.exists(PARQUET_OUTPUT):
        print(f"Error: {PARQUET_OUTPUT} not found.")
        sys.exit(1)

    print("Loading Dataset...")
    df = pd.read_parquet(PARQUET_OUTPUT)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    full_ds = BENGIDataset(df, FASTA_PATH, tokenizer)
    
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])
    
    # Using 4 workers for data loading
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print("🏗️ Loading Model...")
    backbone = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    epi_head = EPIClassificationHead(d_model=512, hidden_dim=512).to(DEVICE)

    # Partial Fine-Tuning Configuration
    print("❄️ Configuring Layers...")
    for param in backbone.parameters():
        param.requires_grad = False
        
    # Unfreeze Top 2 Layers
    layers = backbone.backbone.layers 
    for layer in layers[-2:]: 
        for param in layer.parameters():
            param.requires_grad = True
    if hasattr(backbone.backbone, 'norm_f'):
        for param in backbone.backbone.norm_f.parameters():
            param.requires_grad = True

    # Collect parameters
    trainable_params = list(epi_head.parameters())
    for layer in layers[-2:]:
        trainable_params += list(layer.parameters())
    if hasattr(backbone.backbone, 'norm_f'):
        trainable_params += list(backbone.backbone.norm_f.parameters())

    optimizer = optim.AdamW(trainable_params, lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    print(f"\nStarting Training for {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        backbone.train()
        epi_head.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for ids, em, pm, lbl in pbar:
            ids, em, pm, lbl = ids.to(DEVICE), em.to(DEVICE), pm.to(DEVICE), lbl.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                hidden = backbone(ids, output_hidden_states=True).last_hidden_state
                logits = epi_head(hidden, em, pm)
                loss = criterion(logits, lbl)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"   Epoch {epoch+1} Complete. Avg Loss: {avg_train_loss:.4f}")
        
        # Save Checkpoint
        torch.save({
            'backbone': backbone.state_dict(),
            'head': epi_head.state_dict()
        }, f"{RESULTS_DIR}/checkpoint_epoch_{epoch+1}.pt")

    # Evaluation
    print(" Evaluating on Test Set...")
    backbone.eval()
    epi_head.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for ids, em, pm, lbl in tqdm(test_loader, desc="Evaluating"):
            ids, em, pm = ids.to(DEVICE), em.to(DEVICE), pm.to(DEVICE)
            
            with torch.cuda.amp.autocast():
                hidden = backbone(ids, output_hidden_states=True).last_hidden_state
                logits = epi_head(hidden, em, pm)
                probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(lbl.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Metrics
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    print(f" FINAL RESULTS:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   AUROC:    {roc_auc:.4f}")
    print(f"   AUPRC:    {pr_auc:.4f}")

    with open(f"{RESULTS_DIR}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nAUROC: {roc_auc:.4f}\nAUPRC: {pr_auc:.4f}\n")

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{RESULTS_DIR}/roc_curve.png")
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {pr_auc:.2f}')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f"{RESULTS_DIR}/pr_curve.png")
    
    torch.save(epi_head.state_dict(), f"{RESULTS_DIR}/final_head.pt")
    
    print(f"Training Complete. Results saved to {RESULTS_DIR}")