"""
Lightweight LoRA (Low-Rank Adaptation) utilities for EEG foundation models.

Injects trainable low-rank matrices into nn.Linear layers while keeping
the original weights frozen. Compatible with any PyTorch model.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a low-rank adapter: output = W_frozen @ x + (B @ A) @ x * scaling.

    The original weight W is frozen. Only A and B are trainable.
    A is initialized with Kaiming uniform, B is initialized to zero → adapter starts as identity.

    Exposes .weight, .bias, .in_features, .out_features as pass-throughs so that
    nn.MultiheadAttention (which accesses out_proj.weight directly) still works.
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # Low-rank adapter matrices
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.scaling = alpha / rank

        # Initialize A with Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        """Pass-through for nn.MultiheadAttention compatibility."""
        return self.original_linear.weight

    @property
    def bias(self):
        """Pass-through for nn.MultiheadAttention compatibility."""
        return self.original_linear.bias

    def forward(self, x):
        # Original frozen forward
        base_out = self.original_linear(x)
        # LoRA adapter: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def inject_lora(model: nn.Module, target_modules: list = None, rank: int = 8, alpha: float = 16.0):
    """
    Replace nn.Linear layers in target_modules with LoRALinear wrappers.

    IMPORTANT: nn.MultiheadAttention's internal projections (in_proj, out_proj)
    are accessed via F.multi_head_attention_forward which reads .weight/.bias
    directly and never calls forward(). LoRA on those layers would be silently
    ignored. We therefore target feedforward layers and standalone Linear projections
    by default — these are always called through normal forward().

    Args:
        model: The model to inject LoRA into.
        target_modules: List of substrings to match against parameter names.
                       If None, defaults to feedforward and projection layers:
                       ["linear1", "linear2", "fc", "mlp", "proj",
                        "q_proj", "v_proj", "k_proj"].
                       Layers inside nn.MultiheadAttention are always skipped.
        rank: Rank of the low-rank adapter matrices.
        alpha: Scaling factor (effective lr multiplier = alpha/rank).

    Returns:
        Number of LoRA parameters injected.
    """
    if target_modules is None:
        target_modules = [
            "linear1", "linear2",                          # transformer FFN layers
            "fc", "mlp",                                   # generic feedforward
            "q_proj", "v_proj", "k_proj", "out_proj",      # standalone attention projections (not inside MHA)
            "qkv",                                         # fused QKV projections
        ]

    lora_param_count = 0
    replaced = []

    # Build set of module ids that are nn.MultiheadAttention instances
    # so we can skip their children (in_proj, out_proj are not called via forward())
    mha_ids = set()
    for _, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            mha_ids.add(id(module))

    # Collect all replacements first, then apply — avoids mutating the module
    # tree while iterating (which causes infinite recursion).
    def _find_targets(model, target_filter):
        """Return list of (parent_module, child_name, child_linear, full_name)."""
        targets = []
        for name, module in model.named_modules():
            # Skip children of nn.MultiheadAttention
            if id(module) in mha_ids:
                continue
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                if isinstance(child, nn.Linear) and target_filter(full_name, child):
                    targets.append((module, child_name, child, full_name))
        return targets

    # Pass 1: match specific layer names
    targets = _find_targets(
        model,
        lambda fname, _: any(t in fname for t in target_modules),
    )

    # Pass 2 fallback: if no matches, inject into all "large" linear layers
    # (still skipping MHA internals)
    if not targets:
        print(f"  [LoRA] No matches for {target_modules}, trying broader injection...")
        targets = _find_targets(
            model,
            lambda _, child: child.in_features >= 128,
        )

    # Apply all replacements
    for parent, child_name, child_linear, full_name in targets:
        lora_layer = LoRALinear(child_linear, rank=rank, alpha=alpha)
        setattr(parent, child_name, lora_layer)
        n_params = child_linear.in_features * rank + child_linear.out_features * rank
        lora_param_count += n_params
        replaced.append(full_name)

    print(f"  [LoRA] Injected rank-{rank} adapters into {len(replaced)} layers ({lora_param_count:,} params)")
    for r in replaced:
        print(f"    → {r}")

    return lora_param_count
