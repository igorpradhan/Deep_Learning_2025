"""EPI Classification Head for Caduceus.

This module implements the RC-invariant classification head for
enhancer-promoter interaction (EPI) prediction.

The head takes the output of the final MambaDNA block and:
1. Combines the two directional streams (H^(1) and H^(2))
2. Creates RC-invariant representation via symmetrization
3. Pools enhancer and promoter segments separately
4. Constructs pair features [v_E, v_P, |v_E - v_P|, v_E ⊙ v_P]
5. Passes through MLP to get classification logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def rc_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply reverse-complement transformation to tensor.

    Reverses sequence dimension (dim=-2) and channel dimension (dim=-1).

    Args:
        x: Tensor of shape [B, L, D]

    Returns:
        RC-transformed tensor of shape [B, L, D]
    """
    return torch.flip(x, dims=[-2, -1])


def segment_mean(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over a segment.

    Args:
        H: Hidden states [B, L, D]
        mask: Binary mask [B, L] indicating segment positions

    Returns:
        Pooled representation [B, D]
    """
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
    weighted = H * mask.unsqueeze(-1)  # [B, L, D]
    return weighted.sum(dim=1) / denom  # [B, D]


class EPIClassificationHead(nn.Module):
    """RC-invariant classification head for enhancer-promoter interaction prediction.

    This head replaces the standard LM head of Caduceus for the EPI task.

    Args:
        d_model: Hidden dimension of Caduceus (total, before splitting)
        hidden_dim: Hidden dimension for the MLP (default: 512)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.stream_dim = d_model // 2  # Each stream has d_model/2 channels
        self.hidden_dim = hidden_dim

        # MLP for classification: 4 * (d_model/2) -> hidden -> 1
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.stream_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        enhancer_mask: torch.Tensor,
        promoter_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through EPI classification head.

        Args:
            hidden_states: Output from final MambaDNA block [B, L, D]
                          where D = d_model (concatenation of two streams)
            enhancer_mask: Binary mask [B, L] indicating enhancer positions
            promoter_mask: Binary mask [B, L] indicating promoter positions

        Returns:
            logits: Classification logits [B] (pre-sigmoid)
        """
        B, L, D = hidden_states.shape
        assert D == self.d_model, f"Expected {self.d_model} channels, got {D}"

        # Step 1: Split into two directional streams
        # H^(1): forward stream, H^(2): reverse stream
        H1 = hidden_states[..., :self.stream_dim]  # [B, L, D/2]
        H2 = hidden_states[..., self.stream_dim:]  # [B, L, D/2]

        
        H_sum = H1 + H2  

        H_rc = rc_transform(H_sum)  
        H_inv = 0.5 * (H_sum + H_rc)  

        # Step 4: Pool enhancer and promoter segments
        v_E = segment_mean(H_inv, enhancer_mask) 
        v_P = segment_mean(H_inv, promoter_mask) 

    
        # [v_E, v_P, |v_E - v_P|, v_E ⊙ v_P]
        diff = torch.abs(v_E - v_P)  # [B, D/2]
        prod = v_E * v_P  # [B, D/2]

        feat = torch.cat([v_E, v_P, diff, prod], dim=-1)  # [B, 4*D/2]

        logit = self.mlp(feat).squeeze(-1)  # [B]

        return logit

    def forward_with_features(
        self,
        hidden_states: torch.Tensor,
        enhancer_mask: torch.Tensor,
        promoter_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass that also returns intermediate features for LRP.

        This is useful for debugging and for implementing LRP.

        Args:
            hidden_states: Output from final MambaDNA block [B, L, D]
            enhancer_mask: Binary mask [B, L]
            promoter_mask: Binary mask [B, L]

        Returns:
            logits: Classification logits [B]
            features: Dictionary containing intermediate representations
        """
        B, L, D = hidden_states.shape


        H1 = hidden_states[..., :self.stream_dim]
        H2 = hidden_states[..., self.stream_dim:]

        H_sum = H1 + H2
        H_rc = rc_transform(H_sum)
        H_inv = 0.5 * (H_sum + H_rc)

        # Pool segments
        v_E = segment_mean(H_inv, enhancer_mask)
        v_P = segment_mean(H_inv, promoter_mask)

     
        diff = torch.abs(v_E - v_P)
        prod = v_E * v_P
        feat = torch.cat([v_E, v_P, diff, prod], dim=-1)

        
        logit = self.mlp(feat).squeeze(-1)

        features = {
            'H1': H1,
            'H2': H2,
            'H_sum': H_sum,
            'H_rc': H_rc,
            'H_inv': H_inv,
            'v_E': v_E,
            'v_P': v_P,
            'diff': diff,
            'prod': prod,
            'feat': feat,
        }

        return logit, features


class EPIClassificationHead_LRP(EPIClassificationHead):
    """LRP-modified version of EPI classification head.

    This version modifies the elementwise product feature v_E ⊙ v_P
    to use the MambaLRP-style 50-50 split for conservation.

    In the explanation graph:
        prod_expl = 0.5 * (v_E * v_P) + 0.5 * (v_E * v_P).detach()

    This ensures that GI×Input splits relevance equally between v_E and v_P.
    """

    def __init__(self, d_model: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__(d_model, hidden_dim, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        enhancer_mask: torch.Tensor,
        promoter_mask: torch.Tensor,
        lrp_mode: bool = False,
    ) -> torch.Tensor:
        """Forward pass with optional LRP modification.

        Args:
            hidden_states: Output from final MambaDNA block [B, L, D]
            enhancer_mask: Binary mask [B, L]
            promoter_mask: Binary mask [B, L]
            lrp_mode: If True, apply LRP modification to product feature

        Returns:
            logits: Classification logits [B]
        """
        B, L, D = hidden_states.shape


        H1 = hidden_states[..., :self.stream_dim]
        H2 = hidden_states[..., self.stream_dim:]

        H_sum = H1 + H2
        H_rc = rc_transform(H_sum)
        H_inv = 0.5 * (H_sum + H_rc)

        v_E = segment_mean(H_inv, enhancer_mask)
        v_P = segment_mean(H_inv, promoter_mask)

        diff = torch.abs(v_E - v_P)
        prod = v_E * v_P

        if lrp_mode:
            prod = 0.5 * prod + 0.5 * prod.detach()

        feat = torch.cat([v_E, v_P, diff, prod], dim=-1)


        logit = self.mlp(feat).squeeze(-1)

        return logit


# Example usage and testing
if __name__ == "__main__":
    # Test the EPI head
    B, L, D = 4, 1000, 256

    # Create dummy inputs
    hidden_states = torch.randn(B, L, D)
    enhancer_mask = torch.zeros(B, L)
    enhancer_mask[:, 100:200] = 1.0  # Enhancer at positions 100-200
    promoter_mask = torch.zeros(B, L)
    promoter_mask[:, 700:800] = 1.0  # Promoter at positions 700-800

    # Test standard head
    head = EPIClassificationHead(d_model=D, hidden_dim=512)
    logits = head(hidden_states, enhancer_mask, promoter_mask)
    print(f"Standard head output shape: {logits.shape}")  # [B]

    # Test with features
    logits, features = head.forward_with_features(hidden_states, enhancer_mask, promoter_mask)
    print(f"v_E shape: {features['v_E'].shape}")  # [B, D/2]
    print(f"v_P shape: {features['v_P'].shape}")  # [B, D/2]
    print(f"feat shape: {features['feat'].shape}")  # [B, 4*D/2]

    # Test LRP head
    head_lrp = EPIClassificationHead_LRP(d_model=D, hidden_dim=512)
    logits_lrp = head_lrp(hidden_states, enhancer_mask, promoter_mask, lrp_mode=True)
    print(f"LRP head output shape: {logits_lrp.shape}")  # [B]

    # Verify forward values are identical (with and without LRP mode)
    logits_normal = head_lrp(hidden_states, enhancer_mask, promoter_mask, lrp_mode=False)
    print(f"Forward values match: {torch.allclose(logits_normal, logits_lrp, rtol=1e-5)}")

    print("\n EPI classification head tests passed!")
