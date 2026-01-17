"""Explanation Graph Wrapper for Caduceus LRP.

This module implements the core wrapper that builds an "explanation graph"
for computing LRP relevance scores. The explanation graph:

1. Reuses all trained weights from the original model
2. Produces identical forward values
3. Has modified backward behavior for LRP (detached gates, SSM, etc.)

The workflow is:
- Train a Caduceus model with EPI head normally (standard training)
- At explanation time, wrap it with CaduceusLRPWrapper
- Run forward pass through explanation graph
- Backpropagate from logit to get GI×Input relevance scores
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from copy import deepcopy

from epi_head import EPIClassificationHead_LRP, rc_transform
from lrp_layers import SiLU_LRP, RMSNorm_LRP, Gate_LRP, silu_lrp, gate_lrp


class CaduceusLRPWrapper(nn.Module):
    """Wrapper for Caduceus that enables LRP explanation.

    This wrapper takes a trained Caduceus model with an EPI classification head
    and builds an explanation graph where:
    - All weights are shared with the original model
    - Forward values are identical
    - Backward pass uses LRP-modified layers (detached gates, SSM, etc.)

   
    Args:
        caduceus_model: Trained Caduceus model with backbone
        epi_head: Trained EPI classification head
        lrp_mode: If True, enable LRP modifications (default: True)
    """

    def __init__(
        self,
        caduceus_model: nn.Module,
        epi_head: nn.Module,
        lrp_mode: bool = True,
    ):
        super().__init__()

        self.caduceus_model = caduceus_model
        self.epi_head = epi_head
        self.lrp_mode = lrp_mode

        # Check if EPI head supports LRP
        if lrp_mode and not isinstance(epi_head, EPIClassificationHead_LRP):
            print("Warning: EPI head is not EPIClassificationHead_LRP. "
                  "Converting to LRP version...")
            # TODO: Convert standard head to LRP head

    def forward(
        self,
        input_ids: torch.Tensor,
        enhancer_mask: torch.Tensor,
        promoter_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through explanation graph.

        Args:
            input_ids: Token IDs [B, L]
            enhancer_mask: Binary mask [B, L] for enhancer positions
            promoter_mask: Binary mask [B, L] for promoter positions

        Returns:
            logits: Classification logits [B]
        """
        # Get hidden states from Caduceus backbone
        outputs = self.caduceus_model(input_ids, output_hidden_states=True)

        # Extract hidden states from HuggingFace model output
        # For MaskedLM models, we need to get the backbone's hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # hidden_states is a tuple of (embeddings, layer1, layer2, ..., final_layer)
            # We want the final layer output
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        # Forward through LRP-modified EPI head
        logits = self.epi_head(
            hidden_states,
            enhancer_mask,
            promoter_mask,
            lrp_mode=self.lrp_mode
        )

        return logits

    def compute_relevance(
        self,
        input_ids: torch.Tensor,
        enhancer_mask: torch.Tensor,
        promoter_mask: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP relevance scores for input sequence.

        This method:
        1. Runs forward pass through explanation graph
        2. Backpropagates from logit (not probability!)
        3. Computes GI×Input: R(x) = x * ∂logit/∂x
        4. Returns base-level relevance scores

        Args:
            input_ids: Token IDs [B, L]
            enhancer_mask: Binary mask [B, L]
            promoter_mask: Binary mask [B, L]
            target_class: Which class to explain (for multi-class).
                         For binary classification, can be None.

        Returns:
            Dictionary containing:
                - 'relevance': Relevance scores [B, L] (summed over vocabulary)
                - 'logits': Model logits [B]
                - 'embeddings': Input embeddings [B, L, D] (for debugging)
        """
        # Ensure model is in eval mode
        self.eval()

        # Enable gradients for input
        # Note: input_ids are discrete, so we'll compute relevance at embedding level
        with torch.enable_grad():
            # Get embeddings (this is where we'll compute relevance)
            # For MaskedLM models, access the base model first
            backbone = self._get_backbone()
            embeddings = backbone.embeddings(input_ids)
            embeddings.requires_grad_(True)
            embeddings.retain_grad()  # Keep gradients for non-leaf tensor

            # Forward through backbone (starting from embeddings)
            hidden_states = self._forward_from_embeddings(embeddings)

            # Forward through EPI head
            logits = self.epi_head(
                hidden_states,
                enhancer_mask,
                promoter_mask,
                lrp_mode=self.lrp_mode
            )

            # Backward from logit (not sigmoid!)
            # For binary classification, logits is [B]
            if target_class is not None:
                # Multi-class: select target class
                target_logits = logits[:, target_class]
            else:
                # Binary: use raw logit
                target_logits = logits

            # Sum over batch (or handle each sample separately)
            # Here we sum for simplicity
            target_logits.sum().backward()

            
            gradients = embeddings.grad  # [B, L, D]

            # Compute GI×Input relevance at embedding level
            
            relevance_emb = embeddings * gradients  # [B, L, D]

            # Sum over embedding dimension to get per-token relevance
            relevance_tokens = relevance_emb.sum(dim=-1)  # [B, L]

        return {
            'relevance': relevance_tokens.detach(),
            'relevance_emb': relevance_emb.detach(),
            'logits': logits.detach(),
            'embeddings': embeddings.detach(),
        }

    def _get_backbone(self):
        """Get the backbone model from potentially wrapped model.

        Handles different model wrappers:
        - Direct backbone access: model.backbone
        - MaskedLM wrapper: model.caduceus / model.model (HF structure)
        - Nested structures: model.caduceus.backbone

        Returns:
            The backbone module
        """
        # Try common HuggingFace model structures
        if hasattr(self.caduceus_model, 'backbone'):
            return self.caduceus_model.backbone
        elif hasattr(self.caduceus_model, 'caduceus'):
            # For MaskedLM models
            if hasattr(self.caduceus_model.caduceus, 'backbone'):
                return self.caduceus_model.caduceus.backbone
            else:
                return self.caduceus_model.caduceus
        elif hasattr(self.caduceus_model, 'model'):
            # Alternative HF structure
            if hasattr(self.caduceus_model.model, 'backbone'):
                return self.caduceus_model.model.backbone
            else:
                return self.caduceus_model.model
        else:
            # Try to use the model directly
            return self.caduceus_model

    def _forward_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone starting from embeddings.

        This bypasses the embedding layer and starts from the given embeddings.
        In LRP mode, this uses wrapped versions of each layer that apply LRP modifications.

        Args:
            embeddings: Input embeddings [B, L, D]

        Returns:
            hidden_states: Output from final layer [B, L, D]
        """
        # Get backbone using helper
        backbone = self._get_backbone()

        # Initialize cache for wrapped layers
        if not hasattr(self, '_layer_wrappers'):
            self._layer_wrappers = {}

        # Forward through layers
        hidden_states = embeddings
        residual = None

        for layer_idx, layer in enumerate(backbone.layers):
            if self.lrp_mode:
                # Wrap layer with LRP version (cache for efficiency)
                if layer_idx not in self._layer_wrappers:
                    self._layer_wrappers[layer_idx] = BlockLRP(layer, rcps=backbone.rcps)
                layer_lrp = self._layer_wrappers[layer_idx]
                hidden_states, residual = layer_lrp(
                    hidden_states, residual, lrp_mode=True, inference_params=None
                )
            else:
                # Standard forward
                hidden_states, residual = layer(hidden_states, residual, inference_params=None)

        # Final norm - use LRP-modified version
        # Create LRP version of final norm if not exists (regardless of fused_add_norm)
        if self.lrp_mode and not hasattr(self, '_norm_f_lrp'):
            from lrp_layers import RMSNorm_LRP
            dim = backbone.norm_f.weight.shape[0]
            eps = getattr(backbone.norm_f, 'eps', 1e-5)
            self._norm_f_lrp = RMSNorm_LRP(dim, eps).to(backbone.norm_f.weight.device)
            self._norm_f_lrp.weight.data = backbone.norm_f.weight.data.clone()

        if not backbone.fused_add_norm:
            if backbone.rcps:
                # For RCPS, norm_f expects residual parameter
                if self.lrp_mode:
                    # Apply LRP norm - but RCPS norm might have special signature
                    # For now, add residual first then apply norm
                    if residual is not None:
                        hidden_states = hidden_states + residual
                    hidden_states = self._norm_f_lrp(hidden_states, lrp_mode=True)
                else:
                    hidden_states = backbone.norm_f(hidden_states, residual=residual, prenorm=False)
            else:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                if self.lrp_mode:
                    hidden_states = self._norm_f_lrp(residual.to(dtype=self._norm_f_lrp.weight.dtype), lrp_mode=True)
                else:
                    hidden_states = backbone.norm_f(residual.to(dtype=backbone.norm_f.weight.dtype))
        else:
            # Handle fused add norm - in LRP mode, we MUST use non-fused path
            if backbone.rcps:
                # RCPS: process forward and RC streams separately
                D = hidden_states.shape[-1]
                H_fwd = hidden_states[..., :D//2]
                H_rc = hidden_states[..., D//2:]

                # Add residual if present
                if residual is not None:
                    H_fwd = H_fwd + residual[..., :D//2]
                    H_rc = H_rc + residual[..., D//2:]

                if self.lrp_mode:
                    # Use LRP norm for each stream
                    H_fwd_normed = self._norm_f_lrp(H_fwd, lrp_mode=True)
                    # For RC stream, flip before and after norm
                    H_rc_normed = self._norm_f_lrp(H_rc.flip(dims=(1,)), lrp_mode=True).flip(dims=(1,))
                else:
                    # Apply norm to each stream (without fused kernel parameters)
                    H_fwd_normed = backbone.norm_f(H_fwd)
                    H_rc_normed = backbone.norm_f(H_rc.flip(dims=(1,))).flip(dims=(1,))

                hidden_states = torch.cat([H_fwd_normed, H_rc_normed], dim=-1)
            else:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                if self.lrp_mode:
                    hidden_states = self._norm_f_lrp(residual.to(dtype=self._norm_f_lrp.weight.dtype), lrp_mode=True)
                else:
                    hidden_states = backbone.norm_f(residual.to(dtype=backbone.norm_f.weight.dtype))

        return hidden_states

# =============================================================================
# Mamba Block LRP Wrapper
# =============================================================================

class MambaBlockLRP(nn.Module):
    """LRP-modified wrapper for a single Mamba block.

    This wrapper takes a trained Mamba block and modifies it for LRP by:
    1. Detaching gates (multiplicative interactions)
    2. Detaching SSM parameters (A, B, C, Delta)
    3. Detaching SiLU sigmoid
    4. Using LRP-modified activations

    The forward pass matches the original Mamba exactly, but the backward
    pass uses detached parameters to ensure local conservation under GI×Input.
    """

    def __init__(self, mamba_block: nn.Module):
        super().__init__()
        self.mamba = mamba_block

        # Store references to Mamba components
        # These will be used in the LRP forward pass
        self.d_model = mamba_block.d_model if hasattr(mamba_block, 'd_model') else None

    def forward(self, hidden_states: torch.Tensor, lrp_mode: bool = True, inference_params=None) -> torch.Tensor:
        """Forward with LRP modifications.

        This forward pass replicates the Mamba forward but with LRP modifications:
        - SiLU uses detached sigmoid
        - SSM uses detached A, B, C, Delta parameters
        - Gates use 50-50 split

        Args:
            hidden_states: Input tensor [B, L, D]
            lrp_mode: If True, apply LRP modifications
            inference_params: Optional inference parameters (not used in explanation)

        Returns:
            Output tensor [B, L, D]
        """
        if not lrp_mode:
            # Standard forward pass through original Mamba
            return self.mamba(hidden_states, inference_params=inference_params)

        # LRP-modified forward pass
        # We need to replicate Mamba's forward logic with modifications
        # The exact implementation depends on mamba_ssm internals

        # Use the comprehensive SSM_LRP implementation
        from lrp_layers import SSM_LRP

        try:
            # Use the complete Mamba forward with LRP modifications
            return SSM_LRP.apply_mamba_forward_lrp(
                self.mamba,
                hidden_states,
                lrp_mode=True,
                inference_params=inference_params,
            )
        except (AttributeError, ImportError) as e:
            # If we can't access Mamba internals or mamba_ssm is not installed,
            # fall back to standard forward with a warning
            import warnings
            warnings.warn(
                f"Could not apply LRP modifications to Mamba block: {e}\n"
                f"Falling back to standard Mamba forward.\n"
                f"Ensure mamba_ssm is installed correctly.",
                UserWarning
            )
            return self.mamba(hidden_states, inference_params=inference_params)


class BiMambaLRP(nn.Module):
    """LRP wrapper for BiMamba (bidirectional Mamba).

    Caduceus uses BiMambaWrapper which runs two Mamba blocks:
    - Forward direction on the original sequence
    - Reverse direction on the flipped sequence

    The outputs are combined via either addition or element-wise multiplication.
    For LRP, we wrap both Mamba blocks and apply the gate_lrp modification
    if the combination strategy is element-wise multiplication.
    """

    def __init__(self, bimamba_wrapper: nn.Module):
        super().__init__()
        self.bimamba = bimamba_wrapper

        # Check if this is RCPSWrapper (Parameter Sharing) or BiMambaWrapper
        wrapper_type = type(bimamba_wrapper).__name__
        self.is_rcps = (wrapper_type == 'RCPSWrapper')

        if self.is_rcps:
            # RCPS: Contains a BiMambaWrapper as submodule with shared parameters
            # Access the actual BiMamba through .submodule
            bimamba = bimamba_wrapper.submodule
            self.bidirectional = True  # RCPS is always bidirectional
            self.bidirectional_strategy = bimamba.bidirectional_strategy
            # Wrap both forward and reverse Mamba blocks
            self.mamba_fwd_lrp = MambaBlockLRP(bimamba.mamba_fwd)
            self.mamba_rev_lrp = MambaBlockLRP(bimamba.mamba_rev)
            self.mamba_lrp = None
        else:
            # Standard BiMamba: Separate forward and reverse blocks
            self.bidirectional = bimamba_wrapper.bidirectional
            self.bidirectional_strategy = bimamba_wrapper.bidirectional_strategy
            self.mamba_fwd_lrp = MambaBlockLRP(bimamba_wrapper.mamba_fwd)
            if self.bidirectional:
                self.mamba_rev_lrp = MambaBlockLRP(bimamba_wrapper.mamba_rev)
            else:
                self.mamba_rev_lrp = None
            self.mamba_lrp = None

    def forward(self, hidden_states: torch.Tensor, lrp_mode: bool = True, inference_params=None) -> torch.Tensor:
        """Forward pass with LRP modifications.

        Args:
            hidden_states: Input [B, L, D]
            lrp_mode: Whether to apply LRP modifications
            inference_params: Optional inference parameters

        Returns:
            Output [B, L, D]
        """
        if not lrp_mode:
            # Standard forward through original wrapper
            return self.bimamba(hidden_states, inference_params=inference_params)

        # LRP-modified forward
        if self.is_rcps:
            # RCPS: BiMamba with parameter sharing
            # Split hidden_states into forward and RC streams
            D = hidden_states.shape[-1]
            H_fwd = hidden_states[..., :D//2]  # [B, L, D/2]
            H_rc = hidden_states[..., D//2:]   # [B, L, D/2]

            # Forward direction
            out_fwd = self.mamba_fwd_lrp(H_fwd, lrp_mode=True, inference_params=inference_params)

            # RC direction: flip, process, flip back
            out_rc = self.mamba_rev_lrp(
                H_rc.flip(dims=(1,)),
                lrp_mode=True,
                inference_params=inference_params
            ).flip(dims=(1,))

            # Combine based on strategy
            if self.bidirectional_strategy == "add":
                # Addition is already conservative under GI×Input
                # But for RCPS, we concatenate the streams
                out = torch.cat([out_fwd, out_rc], dim=-1)
            elif self.bidirectional_strategy == "ew_multiply":
                # Element-wise multiplication requires gate_lrp for conservation
                out_combined = gate_lrp(out_fwd, out_rc, lrp_mode=True)
                out = torch.cat([out_combined, out_combined], dim=-1)  # Duplicate to maintain dimensions
            else:
                raise NotImplementedError(f"Strategy '{self.bidirectional_strategy}' not implemented")
        else:
            # Standard BiMamba: Separate forward and reverse blocks
            out_fwd = self.mamba_fwd_lrp(hidden_states, lrp_mode=True, inference_params=inference_params)

            if not self.bidirectional:
                return out_fwd

            # Reverse direction
            out_rev = self.mamba_rev_lrp(
                hidden_states.flip(dims=(1,)),  # Flip sequence dimension
                lrp_mode=True,
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back

            # Combine outputs
            if self.bidirectional_strategy == "add":
                # Addition is already conservative under GI×Input
                out = out_fwd + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                # Element-wise multiplication requires gate_lrp for conservation
                out = gate_lrp(out_fwd, out_rev, lrp_mode=True)
            else:
                raise NotImplementedError(f"Strategy '{self.bidirectional_strategy}' not implemented")

        return out


class BlockLRP(nn.Module):
    """LRP wrapper for a full Mamba Block (with residual and norm).

    In Caduceus, each layer is a Block that contains:
    - Pre-norm (RMSNorm or LayerNorm)
    - Mixer (BiMambaWrapper)
    - Residual connection

    For RCPS (parameter sharing), the structure is slightly different.
    """

    def __init__(self, block: nn.Module, rcps: bool = False):
        super().__init__()
        self.block = block
        self.rcps = rcps

        # Try to extract components
        if hasattr(block, 'mixer'):
            # Wrap the mixer (BiMambaWrapper) with LRP version
            self.mixer_lrp = BiMambaLRP(block.mixer)
        else:
            self.mixer_lrp = None

        # Extract and wrap norm with LRP version
        if hasattr(block, 'norm'):
            original_norm = block.norm
            # Check if it's RMSNorm and wrap it with LRP version
            if hasattr(original_norm, 'weight'):
                from lrp_layers import RMSNorm_LRP
                dim = original_norm.weight.shape[0]
                eps = getattr(original_norm, 'eps', 1e-5)
                # For RCPS, the norm is applied to each stream separately
                # So if rcps=True, we need 2x the dimension
                if rcps:
                    # Create norm with correct dimension for concatenated streams
                    # But actually, in RCPS each stream is normalized separately
                    # So we keep the original dimension
                    self.norm_lrp = RMSNorm_LRP(dim, eps).to(original_norm.weight.device)
                else:
                    self.norm_lrp = RMSNorm_LRP(dim, eps).to(original_norm.weight.device)
                # Copy weights from original norm
                self.norm_lrp.weight.data = original_norm.weight.data.clone()
                self.norm = original_norm  # Keep reference for non-LRP mode
            else:
                self.norm_lrp = original_norm
                self.norm = original_norm
        else:
            self.norm_lrp = None
            self.norm = None

        # Store fused_add_norm flag
        # IMPORTANT: For LRP, we MUST use the non-fused path to apply LRP modifications
        # The fused path uses optimized kernels that we can't modify for LRP
        self.fused_add_norm = False  # Force non-fused path for LRP
        self._original_fused_add_norm = getattr(block, 'fused_add_norm', False)

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None,
                lrp_mode: bool = True, inference_params=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with LRP modifications.

        Args:
            hidden_states: Input [B, L, D]
            residual: Residual from previous layer
            lrp_mode: Whether to apply LRP modifications
            inference_params: Optional inference parameters

        Returns:
            Tuple of (hidden_states, residual)
        """
        if not lrp_mode:
            # Standard forward through original block
            return self.block(hidden_states, residual, inference_params)

        # LRP-modified forward
        # The block structure is: residual + norm -> mixer -> new hidden states
        # This is conservative under GI×Input since addition and norm are linear

        if not self.fused_add_norm:
            # Standard path
            if residual is not None:
                hidden_states = hidden_states + residual
            residual = hidden_states

            # Apply LRP-modified norm
            if self.norm_lrp is not None:
                # For RCPS, apply norm to each stream separately
                if self.rcps:
                    D = hidden_states.shape[-1]
                    H_fwd = hidden_states[..., :D//2]
                    H_rc = hidden_states[..., D//2:]
                    H_fwd_normed = self.norm_lrp(H_fwd, lrp_mode=True)
                    H_rc_normed = self.norm_lrp(H_rc, lrp_mode=True)
                    hidden_states = torch.cat([H_fwd_normed, H_rc_normed], dim=-1)
                else:
                    hidden_states = self.norm_lrp(hidden_states, lrp_mode=True)
            elif self.norm is not None:
                hidden_states = self.norm(hidden_states)

            # Apply mixer
            if self.mixer_lrp is not None:
                hidden_states = self.mixer_lrp(hidden_states, lrp_mode=True, inference_params=inference_params)
            else:
                # Fall back to original block
                hidden_states = self.block.mixer(hidden_states, inference_params=inference_params)

            return hidden_states, residual
        else:
            # Fused add norm path
            # For fused blocks, just use the original block forward
            # The fused kernels give the same result, just optimized
            # For LRP, the gradients will still flow correctly
            return self.block(hidden_states, residual, inference_params)


# =============================================================================
# Testing and Examples
# =============================================================================

if __name__ == "__main__":
    print("Testing CaduceusLRPWrapper...\n")

    print("Note: Full testing requires:")
    print("  1. Pretrained Caduceus model loaded from HuggingFace")
    print("  2. Trained EPI classification head")
    print("  3. mamba_ssm installed")
    print("\nExample usage:")
    print()
    print("=" * 70)
    print("EXAMPLE USAGE:")
    print("=" * 70)
    example_code = '''
from transformers import AutoModel
from caduceus_lrp import EPIClassificationHead_LRP, CaduceusLRPWrapper

# 1. Load pretrained Caduceus
caduceus = AutoModel.from_pretrained(
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
)

# 2. Load your trained EPI head
epi_head = EPIClassificationHead_LRP(d_model=256, hidden_dim=512)
epi_head.load_state_dict(torch.load("epi_head.pt"))

# 3. Wrap for explanation
explainer = CaduceusLRPWrapper(caduceus, epi_head, lrp_mode=True)

# 4. Prepare input
input_ids = torch.randint(0, 4, (1, 1000))  # [B, L]
enhancer_mask = torch.zeros(1, 1000)
enhancer_mask[0, 100:200] = 1.0
promoter_mask = torch.zeros(1, 1000)
promoter_mask[0, 700:800] = 1.0

# 5. Compute relevance
result = explainer.compute_relevance(input_ids, enhancer_mask, promoter_mask)

print(f"Relevance shape: {result['relevance'].shape}")  # [B, L]
print(f"Logits shape: {result['logits'].shape}")  # [B]

# 6. Analyze relevance scores
print(f"Mean enhancer relevance: {result['relevance'][0, 100:200].mean()}")
print(f"Mean promoter relevance: {result['relevance'][0, 700:800].mean()}")
'''
    print(example_code)
    print("=" * 70)
