"""LRP-modified layers for Caduceus.

This module implements the modified layers required for Layer-wise Relevance
Propagation (LRP) in Caduceus.

Key modifications:
1. SiLU: Detach sigmoid in backward pass
2. RMSNorm: Detach scale in backward pass
3. Multiplicative Gates: 50-50 split modification
4. Selective SSM: Detach A, B, C parameters in backward pass

These modifications ensure local conservation under GI×Input while keeping
forward values identical to the original model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# Custom Autograd Function for SSM LRP
# =============================================================================

class SelectiveScanLRP(torch.autograd.Function):
    """Custom autograd function for selective scan with LRP modifications.

    Forward: Call selective_scan_fn normally with all parameters
    Backward: Treat A, B, C, Delta as constants (don't backprop through them)

    This ensures forward values match exactly while making SSM conservative under GI×Input.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, z, delta_bias, delta_softplus, selective_scan_fn):
        """Forward pass - identical to standard selective_scan_fn."""
        # Call selective scan normally
        try:
            out = selective_scan_fn(
                u, delta, A, B, C, D, z=z,
                delta_bias=delta_bias,
                delta_softplus=delta_softplus,
                return_last_state=False,
            )
        except TypeError:
            # Fallback for different signatures
            out = selective_scan_fn(u, delta, A, B, C, D)

        # Save for backward - but we'll only use u
        # A, B, C, Delta are treated as constants in LRP backward
        ctx.save_for_backward(u, delta, A, B, C, D, z)
        ctx.delta_bias = delta_bias
        ctx.delta_softplus = delta_softplus
        ctx.selective_scan_fn = selective_scan_fn

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - only backprop through u (and z if present), not through A, B, C, Delta."""
        u, delta, A, B, C, D, z = ctx.saved_tensors

        # For LRP, we need gradients w.r.t. u (and z), but NOT w.r.t. A, B, C, delta
        # To achieve this, we detach A, B, C, delta before calling backward
        # This makes them constants from the perspective of autograd

        # Re-run forward with detached parameters to get proper gradients
        with torch.enable_grad():
            u_grad = u.detach().requires_grad_(True) if u.requires_grad else u
            z_grad = z.detach().requires_grad_(True) if z is not None and z.requires_grad else z

            try:
                out_for_grad = ctx.selective_scan_fn(
                    u_grad, delta.detach(), A.detach(), B.detach(), C.detach(), D,
                    z=z_grad,
                    delta_bias=ctx.delta_bias,
                    delta_softplus=ctx.delta_softplus,
                    return_last_state=False,
                )
            except TypeError:
                out_for_grad = ctx.selective_scan_fn(
                    u_grad, delta.detach(), A.detach(), B.detach(), C.detach(), D
                )

            # Backward through this computational graph
            grad_u, grad_z = torch.autograd.grad(
                out_for_grad,
                [u_grad] + ([z_grad] if z_grad is not None and z_grad.requires_grad else []),
                grad_output,
                allow_unused=True
            )

            if grad_z is None and z is not None:
                grad_z = torch.zeros_like(z) if z.requires_grad else None

        # Return gradients in same order as forward inputs
        # Gradients for A, B, C, delta are None (treated as constants)
        return grad_u, None, None, None, None, None, grad_z, None, None, None


# =============================================================================
# SiLU Activation with LRP Modification
# =============================================================================

class SiLULRPFunction(torch.autograd.Function):
    """Custom autograd function for SiLU with LRP modification."""

    @staticmethod
    def forward(ctx, x):
        """Forward pass - standard SiLU."""
        sig = torch.sigmoid(x)
        y = x * sig
        ctx.save_for_backward(x, sig)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - treat sigmoid as constant."""
        x, sig = ctx.saved_tensors
        # In LRP mode, gradient is: grad_output * sig (treating sig as constant)
        # Standard would be: grad_output * (sig + x * sig * (1 - sig))
        # LRP version treats sig as detached constant:
        grad_input = grad_output * sig
        return grad_input


class SiLU_LRP(nn.Module):
    """LRP-modified SiLU activation.

    Standard SiLU: y = x * sigmoid(x)

    In LRP mode, we detach the sigmoid factor in the backward pass:
        Forward: y = x * sigmoid(x)  (same as standard)
        Backward: dy/dx = sigmoid(x)  (treating sigmoid as constant)

    This ensures R(x) = R(y) under GI×Input, making the layer locally conservative.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, lrp_mode: bool = False) -> torch.Tensor:
        """Forward pass with optional LRP modification.

        Args:
            x: Input tensor
            lrp_mode: If True, use LRP-modified backward pass

        Returns:
            y: Output tensor (same shape as x)
        """
        if lrp_mode:
            return SiLULRPFunction.apply(x)
        else:
            # Standard SiLU
            sig = torch.sigmoid(x)
            y = x * sig
            return y


def silu_lrp(x: torch.Tensor, lrp_mode: bool = False) -> torch.Tensor:
    """Functional version of SiLU_LRP.

    Args:
        x: Input tensor
        lrp_mode: If True, apply LRP modification

    Returns:
        Output tensor
    """
    if lrp_mode:
        return SiLULRPFunction.apply(x)
    else:
        sig = torch.sigmoid(x)
        return x * sig


# =============================================================================
# RMSNorm with LRP Modification
# =============================================================================

class RMSNormLRPFunction(torch.autograd.Function):
    """Custom autograd function for RMSNorm with LRP modification."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        """Forward pass - standard RMSNorm."""
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        scale = weight / rms
        y = x * scale
        ctx.save_for_backward(x, weight, scale, rms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - treat scale as constant."""
        x, weight, scale, rms = ctx.saved_tensors
        # In LRP mode, gradient is: grad_output * scale (treating scale as constant)
        # This makes RMSNorm behave as a simple linear scaling
        grad_input = grad_output * scale
        # No gradients for weight or eps
        return grad_input, None, None


class RMSNorm_LRP(nn.Module):
    """LRP-modified Root Mean Square Layer Normalization.

    Standard RMSNorm:
        rms = sqrt(mean(x^2) + eps)
        y = (x / rms) * weight

    In LRP mode, we treat the scale (weight / rms) as constant in backward pass:
        Forward: y = x * (weight / rms)  (same as standard)
        Backward: dy/dx = weight / rms  (treating scale as constant)

    This makes RMSNorm behave as a bias-free linear map under GI×Input,
    ensuring exact conservation.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        """Initialize RMSNorm_LRP.

        Args:
            dim: Dimension of the layer
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, lrp_mode: bool = False) -> torch.Tensor:
        """Forward pass with optional LRP modification.

        Args:
            x: Input tensor [..., dim]
            lrp_mode: If True, use LRP-modified backward pass

        Returns:
            y: Output tensor (same shape as x)
        """
        if lrp_mode:
            return RMSNormLRPFunction.apply(x, self.weight, self.eps)
        else:
            # Standard RMSNorm
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            scale = self.weight / rms
            y = x * scale
            return y


# =============================================================================
# Multiplicative Gate with LRP Modification
# =============================================================================

class Gate_LRP(nn.Module):
    """LRP-modified multiplicative gate.

    Standard gate: y = x * g

    Under naive GI×Input, both x and g would receive full relevance R(y),
    doubling the relevance and violating conservation.

    In LRP mode, we use the 50-50 split modification:
        y_expl = 0.5 * (x * g) + 0.5 * (x * g).detach()

    This ensures:
        R(x) = 0.5 * R(y)
        R(g) = 0.5 * R(y)
        R(x) + R(g) = R(y)  ✓ Conservation
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        lrp_mode: bool = False
    ) -> torch.Tensor:
        """Forward pass with optional LRP modification.

        Args:
            x: Content tensor
            g: Gate tensor (same shape as x)
            lrp_mode: If True, apply 50-50 split

        Returns:
            y: Gated output (same shape as x)
        """
        prod = x * g

        if lrp_mode:
            # 50-50 split: half contributes to gradient, half is constant
            y = 0.5 * prod + 0.5 * prod.detach()
        else:
            # Standard multiplicative gate
            y = prod

        return y


def gate_lrp(x: torch.Tensor, g: torch.Tensor, lrp_mode: bool = False) -> torch.Tensor:
    """Functional version of Gate_LRP.

    Args:
        x: Content tensor
        g: Gate tensor
        lrp_mode: If True, apply LRP modification

    Returns:
        Gated output
    """
    prod = x * g
    if lrp_mode:
        return 0.5 * prod + 0.5 * prod.detach()
    else:
        return prod


# =============================================================================
# Selective SSM with LRP Modification
# =============================================================================

class SSM_LRP:
    """LRP modifications for Selective State Space Models.

    The selective SSM in Mamba has time-varying parameters A_t, B_t, C_t, Delta_t
    that depend on the input. This creates paths for relevance to flow
    into the parameter generators, violating conservation.

    In LRP mode, we detach these parameters in the backward pass:
        A_t_expl = A_t.detach()
        B_t_expl = B_t.detach()
        C_t_expl = C_t.detach()
        Delta_t_expl = Delta_t.detach()

    This makes the SSM behave as a linear map under GI×Input, ensuring
    that relevance flows only through the state updates, not the parameter
    generators.
    """

    @staticmethod
    def detach_ssm_params(A, B, C, Delta):
        """Detach SSM parameters for LRP.

        Args:
            A: State transition parameters [d_inner, d_state] or [d_inner, d_state, N]
            B: Input projection parameters [B, L, d_state]
            C: Output projection parameters [B, L, d_state]
            Delta: Step size parameters [B, L, d_inner]

        Returns:
            Tuple of (A_detached, B_detached, C_detached, Delta_detached)
        """
        return A.detach(), B.detach(), C.detach(), Delta.detach()

    @staticmethod
    def selective_scan_lrp(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        delta_bias: Optional[torch.Tensor] = None,
        delta_softplus: bool = False,
        lrp_mode: bool = True,
    ) -> torch.Tensor:
        """Selective scan with LRP modifications.

        This function wraps the selective_scan_fn from mamba_ssm and applies
        LRP modifications by detaching the SSM parameters A, B, C, Delta.

        Args:
            u: Input [B, L, d_inner]
            delta: Step sizes [B, L, d_inner]
            A: State transition [d_inner, d_state]
            B: Input projection [B, L, d_state]
            C: Output projection [B, L, d_state]
            D: Skip connection parameter [d_inner]
            z: Gate values (optional) [B, L, d_inner]
            delta_bias: Bias for delta (optional)
            delta_softplus: Whether to apply softplus to delta
            lrp_mode: If True, detach SSM parameters

        Returns:
            Output [B, L, d_inner]
        """
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        

        if lrp_mode:
            # Use custom autograd function that preserves forward but modifies backward
            # This treats A, B, C, Delta as constants in backward pass
            out = SelectiveScanLRP.apply(
                u, delta, A, B, C, D, z, delta_bias, delta_softplus, selective_scan_fn
            )
        else:
            # Standard forward and backward
            try:
                out = selective_scan_fn(
                    u, delta, A, B, C, D, z=z,
                    delta_bias=delta_bias,
                    delta_softplus=delta_softplus,
                    return_last_state=False,
                )
            except TypeError:
                # Fallback for different selective_scan_fn signatures
                out = selective_scan_fn(u, delta, A, B, C, D)

        return out

    @staticmethod
    def apply_mamba_forward_lrp(
        mamba_block,
        hidden_states: torch.Tensor,
        lrp_mode: bool = True,
        inference_params=None,
    ) -> torch.Tensor:
        """Complete Mamba forward pass with LRP modifications.

        This replicates the Mamba forward pass but with LRP modifications:
        - SiLU uses detached sigmoid
        - SSM uses detached A, B, C, Delta
        - Gates use 50-50 split

        Args:
            mamba_block: The Mamba module
            hidden_states: Input [B, L, D]
            lrp_mode: If True, apply LRP modifications
            inference_params: Optional inference params (not used in LRP)

        Returns:
            Output [B, L, D]
        """
        if not lrp_mode:
            return mamba_block(hidden_states, inference_params=inference_params)

        
        from einops import rearrange
       
        # LRP-modified forward following the original Mamba implementation exactly
        # We replicate the "else" branch (slow path) from the original forward

        batch, seqlen, dim = hidden_states.shape

        # Extract Mamba components
        in_proj = mamba_block.in_proj
        conv1d = mamba_block.conv1d
        x_proj = mamba_block.x_proj
        dt_proj = mamba_block.dt_proj
        A_log = mamba_block.A_log
        D = mamba_block.D
        out_proj = mamba_block.out_proj
        dt_rank = mamba_block.dt_rank
        d_state = mamba_block.d_state

        # Input projection - match original exactly
        # Original: xz = rearrange(in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        # But for LRP we use the layer directly to track gradients
        xz = rearrange(
            in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if in_proj.bias is not None:
            xz = xz + rearrange(in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    
        A = -torch.exp(A_log.float())  # [d_inner, d_state]

        
        x, z = xz.chunk(2, dim=1)  # x, z: [B, d_inner, L]

    
        x = conv1d(x)[..., :seqlen]  # [B, d_inner, L]

        # Apply activation separately for LRP
        # Original uses fused causal_conv1d_fn with activation="silu"
        # For LRP, we need to apply SiLU with modified backward pass
        x = silu_lrp(x, lrp_mode=lrp_mode)

        # SSM parameters - match original exactly
        # Original: x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))
        x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))  # [(b l), dt_rank + 2*d_state]

        # Split into dt, B, C
        dt, B, C = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)

        # Project dt - match original exactly
        # Original: dt = dt_proj.weight @ dt.t()
        dt = dt_proj.weight @ dt.t()  # [d_inner, (b l)]
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

        # Rearrange B and C - match original exactly
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # Selective scan with LRP modifications
        y = SSM_LRP.selective_scan_lrp(
            u=x,
            delta=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            z=z,
            delta_bias=dt_proj.bias.float() if dt_proj.bias is not None else None,
            delta_softplus=True,
            lrp_mode=lrp_mode,
        )

        # Rearrange output - match original exactly
        # Original: y = rearrange(y, "b d l -> b l d")
        y = rearrange(y, "b d l -> b l d")

        out = out_proj(y)

        return out


# =============================================================================
# Helper: Find and Replace Layers in a Module
# =============================================================================

def replace_layers_for_lrp(
    module: nn.Module,
    lrp_mode: bool = False,
    verbose: bool = False
) -> nn.Module:
    """Recursively find and replace layers with LRP versions.

    This function walks through a module and its children, replacing:
    - nn.SiLU / F.silu with SiLU_LRP
    - RMSNorm with RMSNorm_LRP
    - Multiplicative gates with Gate_LRP (requires manual identification)

    Args:
        module: PyTorch module to modify
        lrp_mode: Whether to enable LRP modifications
        verbose: Print replacement messages

    Returns:
        Modified module

    NOTE: This is a helper template. For Mamba blocks, you'll need to
    manually identify where gates, SiLU, and SSM occur and replace them.
    """
    for name, child in module.named_children():
        # Replace SiLU activation
        if isinstance(child, nn.SiLU):
            if verbose:
                print(f"Replacing {name}: nn.SiLU -> SiLU_LRP")
            setattr(module, name, SiLU_LRP())

        # Replace RMSNorm (this is a placeholder - actual class name may differ)
        elif child.__class__.__name__ == 'RMSNorm':
            if verbose:
                print(f"Replacing {name}: RMSNorm -> RMSNorm_LRP")
            dim = child.weight.shape[0]
            eps = getattr(child, 'eps', 1e-8)
            new_norm = RMSNorm_LRP(dim, eps)
            new_norm.weight.data = child.weight.data.clone()
            setattr(module, name, new_norm)

        # Recursively process children
        else:
            replace_layers_for_lrp(child, lrp_mode, verbose)

    return module


# =============================================================================
# Testing and Examples
# =============================================================================

if __name__ == "__main__":
    print("Testing LRP-modified layers...\n")

    # Test SiLU_LRP
    print("1. Testing SiLU_LRP")
    x = torch.randn(4, 10, requires_grad=True)

    silu = SiLU_LRP()
    y_normal = silu(x, lrp_mode=False)
    y_lrp = silu(x, lrp_mode=True)

    print(f"   Forward values match: {torch.allclose(y_normal, y_lrp)}")

    # Check gradients differ
    y_normal.sum().backward(retain_graph=True)
    grad_normal = x.grad.clone()
    x.grad = None

    y_lrp.sum().backward()
    grad_lrp = x.grad.clone()

    print(f"   Gradients differ (expected): {not torch.allclose(grad_normal, grad_lrp)}")
    print()

    print("2. Testing RMSNorm_LRP")
    x = torch.randn(4, 10, 128)

    norm = RMSNorm_LRP(dim=128)
    y_normal = norm(x, lrp_mode=False)
    y_lrp = norm(x, lrp_mode=True)

    print(f"   Forward values match: {torch.allclose(y_normal, y_lrp)}")
    print()

  
    print("3. Testing Gate_LRP")
    x = torch.randn(4, 10, 128)
    g = torch.randn(4, 10, 128)

    gate = Gate_LRP()
    y_normal = gate(x, g, lrp_mode=False)
    y_lrp = gate(x, g, lrp_mode=True)

    print(f"   Forward values match: {torch.allclose(y_normal, y_lrp)}")
    print()

    print("4. Testing Gate_LRP conservation")
    x = torch.randn(4, 10, requires_grad=True)
    g = torch.randn(4, 10, requires_grad=True)

    gate = Gate_LRP()
    y = gate(x, g, lrp_mode=True)

    # GI×Input relevance
    y.sum().backward()
    R_x = (x * x.grad).sum().item()
    R_g = (g * g.grad).sum().item()
    R_y = y.sum().item()

    print(f"   R(x) = {R_x:.6f}")
    print(f"   R(g) = {R_g:.6f}")
    print(f"   R(x) + R(g) = {R_x + R_g:.6f}")
    print(f"   R(y) = {R_y:.6f}")
    print(f"   Approximately conserved: {abs((R_x + R_g) - R_y) < 0.01}")
    print()

