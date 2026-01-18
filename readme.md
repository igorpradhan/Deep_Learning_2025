## 1. Overview

This project investigates how **deep learning models can be trained and interpreted to identify regulatory patterns in DNA sequences**, with a focus on transcription factor (TF) binding and epigenomic signals.

We fine-tune the **Caduceus** sequence model on regulatory genomics data (by unfreezing 2 layers on the backbone) and apply multiple interpretability techniques (saliency maps, Layer-wise Relevance Propagation, and TF-MoDISco) to answer the following research question:

What sequence features and motifs does the model rely on for its predictions?

The project executes these ideas through well-motivated experiments, and extensive post-hoc interpretability analyses.

---

## 2. Repository Structure

```text
DL_final_ver/
├── caduceus_lrp/
│   ├── train_caduceus_finetune.py   # Main training script
│   ├── explanation_model.py         # LRP-enabled model wrapper
│   ├── lrp_layers.py                # Custom LRP layers
│   ├── epi_head.py                  # Prediction head
│   ├── test_gpu.py                  # GPU availability test
│   └── PROGRESS.md                  # Development notes
├── final_models/
│   └── best_model.pt                # Best-performing trained model
├── TF_discovery/
│   └── TF_DISCOVERY.ipynb           # Motif discovery and analysis
├── examples/
│   ├── viz_saliency.py              # Saliency map visualization
│   ├── viz_interpret.py             # Attribution visualization
│   └── viz_3d_dna.py                # 3D DNA visualization
├── modisco_files/
│   ├── modisco_windows_200bp.h5     # TF-MoDISco output
│   ├── big200bp.pos.meme
│   ├── big200bp.neg.meme
│   └── tomtom_large_sample.tsv
├── plots/
│   ├── epoch1_roc.png
│   ├── epoch2_roc.png
│   ├── epoch3_roc.png
│   ├── TF_motifs.png
│   ├── TF_motids.png
│   ├── dist_EPI.png
│   ├── dist_attribution.png
│   └── zoomed_in_sample.png
└── readme.md



## 3. Setup and Execution Guide

### 3.1 Environment Setup

Install dependencies:
pip install -r requirements.txt

3. (Optional) Test GPU availability:

python caduceus_lrp/test_gpu.py

### 3.2 External Downloads Required

* **Pretrained Caduceus weights** (download separately and update path in training script)
* **Regulatory genomics dataset** (formatted as expected by the training code)
* **TF-MoDISco + MEME Suite** if not already installed

### 3.3 Training

Run the training script first:
python caduceus_lrp/train_caduceus_finetune.py

The best model will be saved to `final_models/best_model.pt`.

### 3.4 Interpretability and Motif Discovery

* Saliency maps:


python examples/viz_saliency.py


* LRP attributions:
python examples/viz_interpret.py

* TF-MoDISco analysis:
  Run via script or `TF_discovery/TF_DISCOVERY.ipynb`

---

### 3.5 Recommended Execution Order

1. Install dependencies
2. Download external resources
3. Train model
4. Generate attributions
5. Run TF-MoDISco
6. Analyze motifs in notebook
