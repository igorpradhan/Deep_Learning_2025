"""Comprehensive GPU testing script for Caduceus LRP implementation.
"""

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing on device: {device}\n")

# ============================================================================
# Test 1: Individual LRP Layers
# ============================================================================

print("="*80)
print("TEST 1: Individual LRP Layers")
print("="*80)

from lrp_layers import SiLU_LRP, RMSNorm_LRP, Gate_LRP

# SiLU
x = torch.randn(4, 100, 256, device=device, requires_grad=True)
silu = SiLU_LRP().to(device)
y_std = silu(x, lrp_mode=False)
y_lrp = silu(x.clone().detach().requires_grad_(True), lrp_mode=True)
print(f"[SiLU] Forward match: {torch.allclose(y_std, y_lrp, rtol=1e-5)}")

# RMSNorm
norm = RMSNorm_LRP(dim=256).to(device)
y_std = norm(x, lrp_mode=False)
y_lrp = norm(x, lrp_mode=True)
print(f"[RMSNorm] Forward match: {torch.allclose(y_std, y_lrp, rtol=1e-5)}")

# Gate conservation test
x_g = torch.randn(4, 100, 256, device=device, requires_grad=True)
g_g = torch.randn(4, 100, 256, device=device, requires_grad=True)
gate = Gate_LRP().to(device)
y_g = gate(x_g, g_g, lrp_mode=True)
y_g.sum().backward()
R_x = (x_g * x_g.grad).sum().item()
R_g = (g_g * g_g.grad).sum().item()
R_y = y_g.sum().item()
print(f"[Gate] Conservation: R(x)+R(g)={R_x+R_g:.3f}, R(y)={R_y:.3f}, error={abs(R_x+R_g-R_y)/abs(R_y):.2%}")

# ============================================================================
# Test 2: Mamba Block
# ============================================================================

print("\n" + "="*80)
print("TEST 2: Mamba Block with LRP")
print("="*80)

from mamba_ssm.modules.mamba_simple import Mamba
from explanation_model import MambaBlockLRP

mamba = Mamba(d_model=256, d_state=16, d_conv=4, expand=2).to(device)
mamba_lrp = MambaBlockLRP(mamba).to(device)

# Force slow path for original Mamba to ensure exact comparison
# The fast path uses fused kernels which may have slight numerical differences
original_use_fast_path = mamba.use_fast_path
mamba.use_fast_path = False

x = torch.randn(2, 100, 256, device=device)
with torch.no_grad():
    y_std = mamba(x)
    y_lrp = mamba_lrp(x, lrp_mode=True)

# Restore original setting
mamba.use_fast_path = original_use_fast_path

match = torch.allclose(y_std, y_lrp, rtol=1e-4, atol=1e-6)
max_rel_diff = ((y_std - y_lrp).abs() / (y_std.abs() + 1e-8)).max().item()
print(f"[Mamba] Forward match (slow path): {match}, Max relative diff: {max_rel_diff:.6f}")

# Also test with fast path to see the difference
if original_use_fast_path:
    mamba.use_fast_path = True
    with torch.no_grad():
        y_fast = mamba(x)
    fast_vs_lrp_match = torch.allclose(y_fast, y_lrp, rtol=1e-3, atol=1e-5)
    fast_vs_lrp_diff = ((y_fast - y_lrp).abs() / (y_fast.abs() + 1e-8)).max().item()
    print(f"[Mamba] Forward match (fast path vs LRP): {fast_vs_lrp_match}, Max relative diff: {fast_vs_lrp_diff:.6f}")
    if not fast_vs_lrp_match:
        print(f"  Note: Small differences expected between fast path (fused kernels) and LRP (slow path)")

# ============================================================================
# Test 3: Full Caduceus + EPI
# ============================================================================

print("\n" + "="*80)
print("TEST 3: Full Caduceus with EPI Head")
print("="*80)

from transformers import AutoTokenizer, AutoModelForMaskedLM
from epi_head import EPIClassificationHead_LRP
from explanation_model import CaduceusLRPWrapper

# Use smaller model for testing
model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
caduceus = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
print(f"Model loaded successfully")
# For Caduceus-PS, hidden_states have 2*d_model dimension (forward + RC streams concatenated)
# The base d_model is 256, so hidden states are 512
epi_head = EPIClassificationHead_LRP(d_model=512, hidden_dim=512).to(device)
explainer = CaduceusLRPWrapper(caduceus, epi_head, lrp_mode=True).to(device)

# Test input
B, L = 2, 1000
input_ids = torch.randint(0, 4, (B, L), device=device)
enh_mask = torch.zeros(B, L, device=device)
enh_mask[:, 100:200] = 1.0
prom_mask = torch.zeros(B, L, device=device)
prom_mask[:, 700:800] = 1.0

# Forward pass
with torch.no_grad():
    logits = explainer(input_ids, enh_mask, prom_mask)
print(f"[Forward] Logits: {logits.cpu().numpy()}")

# LRP computation
result = explainer.compute_relevance(input_ids, enh_mask, prom_mask)
relevance = result['relevance']
logits_lrp = result['logits']

# Conservation check
for i in range(B):
    rel_sum = relevance[i].sum().item()
    logit = logits_lrp[i].item()
    error = abs(rel_sum - logit) / max(abs(logit), 1e-8)
    print(f"[Sample {i}] Relevance sum: {rel_sum:.4f}, Logit: {logit:.4f}, Error: {error:.2%}")

# Biological sanity
enh_rel = relevance[:, 100:200].abs().mean().item()
prom_rel = relevance[:, 700:800].abs().mean().item()
bg_rel = relevance[:, 300:400].abs().mean().item()
print(f"[Biology] Enhancer: {enh_rel:.4f}, Promoter: {prom_rel:.4f}, Background: {bg_rel:.4f}")


print("\n" + "="*80)
print("ALL TESTS COMPLETED")
print("="*80)
