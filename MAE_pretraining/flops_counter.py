"""
FLOPs Counter for EEG Foundation Models
========================================
Counts FLOPs and parameters for all pretraining model variants.
Works with both MAE-style and BERT-style models.

Usage:
    python -m MAE_pretraining.flops_counter

    # Or from code:
    from MAE_pretraining.flops_counter import count_flops, compare_all_models
    flops_info = count_flops(model, num_channels=32, seq_length=500, batch_size=2)
    compare_all_models()
"""

import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from collections import defaultdict
import time


# ─────────────────────────────────────────────────────────────────────
# Hook-based FLOPs counter (no external dependencies needed)
# ─────────────────────────────────────────────────────────────────────

class FLOPsCounter:
    """
    Counts FLOPs by registering forward hooks on nn.Linear, nn.MultiheadAttention,
    and nn.Embedding layers. Also estimates attention FLOPs from scaled_dot_product calls.
    """

    def __init__(self):
        self.flops = 0
        self.hooks = []
        self.layer_flops = defaultdict(int)

    def _linear_hook(self, module, input, output):
        """Linear: 2 * in_features * out_features * batch_elements"""
        x = input[0]
        batch_elements = x.numel() // x.shape[-1]
        flops = 2 * module.in_features * module.out_features * batch_elements
        self.flops += flops
        self.layer_flops[module.__class__.__name__] += flops

    def _layernorm_hook(self, module, input, output):
        """LayerNorm: ~5 * num_elements (mean, var, sub, div, scale+shift)"""
        x = input[0]
        flops = 5 * x.numel()
        self.flops += flops
        self.layer_flops["LayerNorm"] += flops

    def _embedding_hook(self, module, input, output):
        """Embedding lookup is essentially free (index gather), count as 0"""
        pass

    def _softmax_hook(self, module, input, output):
        """Softmax: ~3 * num_elements (exp, sum, div)"""
        x = input[0]
        flops = 3 * x.numel()
        self.flops += flops
        self.layer_flops["Softmax"] += flops

    def _gelu_hook(self, module, input, output):
        """GELU: ~8 ops per element"""
        x = input[0]
        flops = 8 * x.numel()
        self.flops += flops
        self.layer_flops["GELU"] += flops

    def register_hooks(self, model):
        """Register forward hooks on all relevant layers"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(self._linear_hook)
                self.hooks.append(h)
            elif isinstance(module, nn.LayerNorm):
                h = module.register_forward_hook(self._layernorm_hook)
                self.hooks.append(h)
            elif isinstance(module, nn.Embedding):
                h = module.register_forward_hook(self._embedding_hook)
                self.hooks.append(h)
            elif isinstance(module, nn.Softmax):
                h = module.register_forward_hook(self._softmax_hook)
                self.hooks.append(h)
            elif isinstance(module, nn.GELU):
                h = module.register_forward_hook(self._gelu_hook)
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def reset(self):
        self.flops = 0
        self.layer_flops = defaultdict(int)


def _add_attention_flops_hook(counter, model):
    """
    Monkey-patch torch.nn.functional.scaled_dot_product_attention to count
    attention FLOPs (QK^T matmul + softmax + AV matmul).
    """
    import torch.nn.functional as F
    original_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        # QK^T: (B, H, L, d) x (B, H, d, L) -> (B, H, L, L) = 2*B*H*L*L*d
        B_H = query.shape[:-2].numel()  # product of batch dims and heads
        L_q = query.shape[-2]
        L_k = key.shape[-2]
        d = query.shape[-1]
        # QK^T matmul
        flops = 2 * B_H * L_q * L_k * d
        # Softmax over L_k
        flops += 3 * B_H * L_q * L_k
        # Attention @ V: (B, H, L_q, L_k) x (B, H, L_k, d) -> 2*B*H*L_q*d*L_k
        flops += 2 * B_H * L_q * d * L_k
        counter.flops += flops
        counter.layer_flops["Attention(QK+AV)"] += flops
        return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    F.scaled_dot_product_attention = patched_sdpa
    return original_sdpa


def _add_bmm_flops_hook(counter, model):
    """
    Monkey-patch torch.bmm to count matmul FLOPs for manual attention
    (used in RiemannianSpaceAttention which does manual Q@K^T and attn@V).
    """
    original_bmm = torch.bmm

    def patched_bmm(input, mat2):
        # (B, N, M) x (B, M, P) -> 2*B*N*M*P
        B = input.shape[0]
        N = input.shape[1]
        M = input.shape[2]
        P = mat2.shape[2]
        flops = 2 * B * N * M * P
        counter.flops += flops
        counter.layer_flops["BMM(manual_attn)"] += flops
        return original_bmm(input, mat2)

    torch.bmm = patched_bmm
    return original_bmm


def _add_matmul_flops_hook(counter):
    """
    Monkey-patch torch.matmul for general matrix multiplications
    (e.g., covariance computation in Riemannian bias).
    """
    original_matmul = torch.matmul

    def patched_matmul(input, other):
        if input.dim() >= 2 and other.dim() >= 2:
            batch_dims = input.shape[:-2].numel() if input.dim() > 2 else 1
            M = input.shape[-2]
            K = input.shape[-1]
            N = other.shape[-1]
            flops = 2 * batch_dims * M * K * N
            counter.flops += flops
            counter.layer_flops["Matmul(misc)"] += flops
        return original_matmul(input, other)

    torch.matmul = patched_matmul
    return original_matmul


@torch.no_grad()
def count_flops(model, num_channels=32, seq_length=500, batch_size=2,
                patch_size=16, device="cpu", verbose=True):
    """
    Count FLOPs for a single forward pass of any pretraining model.

    Args:
        model: An EncoderDecoder instance (any variant)
        num_channels: Number of EEG channels (default 32, matching your channel_list)
        seq_length: Temporal length of EEG in samples (default 500)
        batch_size: Batch size for estimation (default 2)
        patch_size: Temporal patch size (default 16, reads from model if available)
        device: Device to run on
        verbose: Print detailed breakdown

    Returns:
        dict with keys: total_flops, total_params, trainable_params,
                        encoder_params, decoder_params, layer_breakdown,
                        flops_per_token, tokens_processed
    """
    model = model.to(device)
    model.eval()

    # Read patch_size from model if available
    if hasattr(model, 'patch_size'):
        patch_size = model.patch_size

    # Ensure seq_length is cleanly divisible by patch_size
    if seq_length % patch_size != 0:
        seq_length = (seq_length // patch_size) * patch_size
        if verbose:
            print(f"  [info] Adjusted seq_length to {seq_length} (divisible by patch_size={patch_size})")

    # Create dummy inputs matching your data format
    dummy_eeg = torch.randn(batch_size, num_channels, seq_length, device=device)
    channel_list = torch.arange(num_channels, dtype=torch.long, device=device)

    # Setup counters
    counter = FLOPsCounter()
    counter.register_hooks(model)

    # Patch attention functions to capture FLOPs
    import torch.nn.functional as F
    original_sdpa = _add_attention_flops_hook(counter, model)
    original_bmm = _add_bmm_flops_hook(counter, model)
    original_matmul = _add_matmul_flops_hook(counter)

    # Forward pass
    try:
        _ = model(dummy_eeg, channel_list)
    except Exception as e:
        print(f"Warning: Forward pass error ({e}), FLOPs may be incomplete")
    finally:
        counter.remove_hooks()
        F.scaled_dot_product_attention = original_sdpa
        torch.bmm = original_bmm
        torch.matmul = original_matmul

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count encoder vs decoder params
    encoder_params = 0
    decoder_params = 0
    if hasattr(model, 'encoder'):
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
    if hasattr(model, 'decoder'):
        decoder_params = sum(p.numel() for p in model.decoder.parameters())

    # Compute per-sample FLOPs
    flops_per_sample = counter.flops / batch_size

    # Compute sequence info
    N = seq_length // patch_size  # number of time patches
    L = N * num_channels          # total tokens

    # Tokens actually processed by encoder
    if hasattr(model, 'mask_prob'):
        mask_prob = model.mask_prob
        # MAE-style: encoder sees (1-mask_prob) * L tokens
        # BERT-style: encoder sees all L tokens
        # Detect by checking for decoder
        if hasattr(model, 'decoder') and len(list(model.decoder.parameters())) > 0:
            tokens_encoder = int(L * (1 - mask_prob))
            style = "MAE"
        else:
            tokens_encoder = L
            style = "BERT"
    else:
        tokens_encoder = L
        style = "Unknown"

    result = {
        "style": style,
        "total_flops": counter.flops,
        "flops_per_sample": flops_per_sample,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "total_tokens": L,
        "tokens_encoder": tokens_encoder,
        "num_patches_time": N,
        "num_channels": num_channels,
        "layer_breakdown": dict(counter.layer_flops),
        "batch_size": batch_size,
    }

    if verbose:
        _print_results(result)

    return result


def _format_number(n, unit=""):
    """Format large numbers with K/M/G/T suffixes"""
    if n >= 1e12:
        return f"{n/1e12:.2f}T{unit}"
    elif n >= 1e9:
        return f"{n/1e9:.2f}G{unit}"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M{unit}"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K{unit}"
    return f"{n:.0f}{unit}"


def _print_results(r):
    """Pretty print FLOPs results"""
    print("\n" + "="*65)
    print(f"  FLOPs Report  ({r['style']}-style)")
    print("="*65)
    print(f"  Total params:        {_format_number(r['total_params'])}")
    print(f"  Trainable params:    {_format_number(r['trainable_params'])}")
    print(f"  Encoder params:      {_format_number(r['encoder_params'])}")
    if r['decoder_params'] > 0:
        print(f"  Decoder params:      {_format_number(r['decoder_params'])}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Sequence: {r['num_patches_time']} time patches × {r['num_channels']} channels = {r['total_tokens']} tokens")
    print(f"  Tokens seen by encoder: {r['tokens_encoder']}")
    print(f"  ─────────────────────────────────────────")
    print(f"  FLOPs (1 sample fwd):  {_format_number(r['flops_per_sample'], 'FLOPs')}")
    print(f"  FLOPs (1 sample train): ~{_format_number(r['flops_per_sample'] * 3, 'FLOPs')}  (fwd + bwd ≈ 3× fwd)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Layer breakdown (total batch = {r['batch_size']}):")
    for layer_type, flops in sorted(r['layer_breakdown'].items(), key=lambda x: -x[1]):
        pct = 100 * flops / max(r['total_flops'], 1)
        print(f"    {layer_type:30s} {_format_number(flops, 'FLOPs'):>14s}  ({pct:5.1f}%)")
    print("="*65 + "\n")


def count_training_flops(model, num_channels=32, seq_length=500,
                          batch_size=64, num_batches_per_epoch=100,
                          num_epochs=50, device="cpu"):
    """
    Estimate total training FLOPs for a full training run.

    Args:
        model: EncoderDecoder model
        num_channels, seq_length: Data dimensions
        batch_size: Training batch size
        num_batches_per_epoch: Number of batches per epoch
        num_epochs: Total training epochs
        device: Device

    Returns:
        dict with total training FLOPs and breakdown
    """
    info = count_flops(model, num_channels=num_channels, seq_length=seq_length,
                       batch_size=batch_size, device=device, verbose=False)

    # Training FLOPs = forward FLOPs × 3 (backward ≈ 2× forward) × batches × epochs
    flops_per_batch_train = info['total_flops'] * 3  # fwd + bwd
    flops_per_epoch = flops_per_batch_train * num_batches_per_epoch
    total_training_flops = flops_per_epoch * num_epochs

    result = {
        **info,
        "training_batch_size": batch_size,
        "num_batches_per_epoch": num_batches_per_epoch,
        "num_epochs": num_epochs,
        "flops_per_batch_train": flops_per_batch_train,
        "flops_per_epoch": flops_per_epoch,
        "total_training_flops": total_training_flops,
    }

    print(f"\n{'='*65}")
    print(f"  Training FLOPs Estimate  ({info['style']}-style)")
    print(f"{'='*65}")
    print(f"  FLOPs/batch (train):     {_format_number(flops_per_batch_train, 'FLOPs')}")
    print(f"  FLOPs/epoch:             {_format_number(flops_per_epoch, 'FLOPs')}")
    print(f"  Total ({num_epochs} epochs):      {_format_number(total_training_flops, 'FLOPs')}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Config: bs={batch_size}, batches/epoch={num_batches_per_epoch}, epochs={num_epochs}")
    print(f"{'='*65}\n")

    return result


def benchmark_throughput(model, num_channels=32, seq_length=500,
                          batch_size=2, num_warmup=3, num_runs=10, device="cpu"):
    """
    Measure wall-clock time per forward pass.

    Args:
        model: EncoderDecoder model
        num_channels, seq_length, batch_size: Data dimensions
        num_warmup: Warmup iterations (not timed)
        num_runs: Timed iterations
        device: Device ("cpu" or "cuda")

    Returns:
        dict with timing info
    """
    model = model.to(device)
    model.eval()

    dummy_eeg = torch.randn(batch_size, num_channels, seq_length, device=device)
    channel_list = list(range(num_channels))

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_eeg, channel_list)

    # Timed runs
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_eeg, channel_list)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)

    result = {
        "mean_ms": times.mean() * 1000,
        "std_ms": times.std() * 1000,
        "min_ms": times.min() * 1000,
        "max_ms": times.max() * 1000,
        "samples_per_sec": batch_size / times.mean(),
        "device": device,
        "batch_size": batch_size,
    }

    print(f"\n  Throughput ({device}, bs={batch_size}): "
          f"{result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms/batch, "
          f"{result['samples_per_sec']:.1f} samples/sec")

    return result


def compare_all_models(num_channels=32, seq_length=512, batch_size=2,
                        enc_dim=512, depth_e=8, patch_size=16, device="cpu"):
    """
    Compare FLOPs and parameters across all model variants.
    Prints a summary table ready for your paper.

    Args:
        num_channels: Number of EEG channels
        seq_length: Temporal length in samples
        batch_size: Batch size for FLOPs estimation
        enc_dim: Encoder embedding dimension
        depth_e: Encoder depth
        patch_size: Temporal patch size
        device: Device

    Returns:
        list of dicts, one per model
    """
    results = []

    # ── 1. MAE (encoder-decoder) ──
    try:
        from MAE_pretraining.pretraining import EncoderDecoder as MAE_ED
        model = MAE_ED(
            num_channels=num_channels, enc_dim=enc_dim, dec_dim=384,
            depth_e=depth_e, depth_d=4, mask_prob=0.7, patch_size=patch_size
        )
        info = count_flops(model, num_channels=num_channels, seq_length=seq_length,
                           batch_size=batch_size, device=device, verbose=False)
        info["name"] = "MAE (ViT enc-dec)"
        results.append(info)
        del model
    except Exception as e:
        print(f"  [skip] MAE: {e}")

    # ── 2. BERT vanilla ──
    try:
        from MAE_pretraining.bert_pretraining import EncoderDecoder as BERT_ED
        model = BERT_ED(
            num_channels=num_channels, enc_dim=enc_dim,
            depth_e=depth_e, mask_prob=0.5, patch_size=patch_size
        )
        info = count_flops(model, num_channels=num_channels, seq_length=seq_length,
                           batch_size=batch_size, device=device, verbose=False)
        info["name"] = "BERT (ViT)"
        results.append(info)
        del model
    except Exception as e:
        print(f"  [skip] BERT vanilla: {e}")

    # ── 3. BERT + SPD loss ──
    try:
        from MAE_pretraining.old_idea.bert_riemaniann_loss import EncoderDecoder as BERT_SPD
        model = BERT_SPD(
            num_channels=num_channels, enc_dim=enc_dim,
            depth_e=depth_e, mask_prob=0.5, patch_size=patch_size
        )
        info = count_flops(model, num_channels=num_channels, seq_length=seq_length,
                           batch_size=batch_size, device=device, verbose=False)
        info["name"] = "BERT (ViT + SPD loss)"
        results.append(info)
        del model
    except Exception as e:
        print(f"  [skip] BERT + SPD loss: {e}")

    # ── 4. BERT + Sequential Riemannian CrissCross ──
    try:
        from MAE_pretraining.old_idea.bert_seq_riemaniann_transformer import EncoderDecoder as BERT_SEQ
        model = BERT_SEQ(
            num_channels=num_channels, enc_dim=enc_dim,
            depth_e=depth_e, mask_prob=0.5, patch_size=patch_size
        )
        info = count_flops(model, num_channels=num_channels, seq_length=seq_length,
                           batch_size=batch_size, device=device, verbose=False)
        info["name"] = "BERT (Seq Riem CrissCross)"
        results.append(info)
        del model
    except Exception as e:
        print(f"  [skip] BERT Seq Riemannian: {e}")

    # ── 5. BERT + Parallel Riemannian CrissCross (CBraMod-style) ──
    try:
        from MAE_pretraining.old_idea.bert_parallel_riemaniann_transformer import EncoderDecoder as BERT_PAR
        model = BERT_PAR(
            num_channels=num_channels, enc_dim=enc_dim,
            depth_e=depth_e, mask_prob=0.5, patch_size=patch_size
        )
        info = count_flops(model, num_channels=num_channels, seq_length=seq_length,
                           batch_size=batch_size, device=device, verbose=False)
        info["name"] = "BERT (Parallel Riem CBraMod)"
        results.append(info)
        del model
    except Exception as e:
        print(f"  [skip] BERT Parallel Riemannian: {e}")

    # ── Print comparison table ──
    if results:
        _print_comparison_table(results)

    return results


def _print_comparison_table(results):
    """Print a formatted comparison table"""
    print("\n" + "="*105)
    print("  MODEL COMPARISON TABLE")
    print("="*105)
    header = f"  {'Model':<32s} {'Style':<6s} {'Total Params':>13s} {'Enc Params':>13s} {'Dec Params':>13s} {'FLOPs/sample':>14s}"
    print(header)
    print("  " + "─"*101)

    for r in results:
        name = r.get("name", "Unknown")
        style = r.get("style", "?")
        total_p = _format_number(r["total_params"])
        enc_p = _format_number(r["encoder_params"])
        dec_p = _format_number(r["decoder_params"]) if r["decoder_params"] > 0 else "—"
        flops = _format_number(r["flops_per_sample"], "FLOPs")
        print(f"  {name:<32s} {style:<6s} {total_p:>13s} {enc_p:>13s} {dec_p:>13s} {flops:>14s}")

    print("="*105)

    # Also print training FLOPs estimate
    print(f"\n  Training FLOPs (×3 for fwd+bwd, per sample):")
    print("  " + "─"*60)
    for r in results:
        name = r.get("name", "Unknown")
        train_flops = r["flops_per_sample"] * 3
        print(f"  {name:<32s} {_format_number(train_flops, 'FLOPs'):>14s}")
    print()


# ─────────────────────────────────────────────────────────────────────
# LaTeX table generation for paper
# ─────────────────────────────────────────────────────────────────────

def generate_latex_table(results, caption="Model comparison", label="tab:model_comparison"):
    """
    Generate a LaTeX table from comparison results.
    Copy-paste directly into your NeurIPS paper.
    """
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & Style & Total Params & Encoder Params & FLOPs/sample \\")
    latex.append(r"\midrule")

    for r in results:
        name = r.get("name", "Unknown").replace("_", r"\_")
        style = r.get("style", "?")
        total_p = _format_number(r["total_params"])
        enc_p = _format_number(r["encoder_params"])
        flops = _format_number(r["flops_per_sample"], "FLOPs")
        latex.append(f"{name} & {style} & {total_p} & {enc_p} & {flops} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    table_str = "\n".join(latex)
    print(table_str)
    return table_str


# ─────────────────────────────────────────────────────────────────────
# Quick single-model usage
# ─────────────────────────────────────────────────────────────────────

def quick_count(model_module_path, num_channels=32, seq_length=500, **kwargs):
    """
    Quick FLOPs count from a module path string.

    Example:
        quick_count("MAE_pretraining.bert_pretraining")
        quick_count("MAE_pretraining.pretraining", num_channels=64, seq_length=1000)
    """
    import importlib
    mod = importlib.import_module(model_module_path)
    model = mod.EncoderDecoder(num_channels=num_channels, **kwargs)
    return count_flops(model, num_channels=num_channels, seq_length=seq_length)


if __name__ == "__main__":
    print("Comparing all model variants with default settings...")
    print("(num_channels=32, seq_length=512, enc_dim=512, depth=8, patch_size=16)\n")

    results = compare_all_models(
        num_channels=32,
        seq_length=512,
        batch_size=2,
        enc_dim=512,
        depth_e=8,
        patch_size=16,
        device="cpu"
    )

    if results:
        print("\n── LaTeX table for paper ──")
        generate_latex_table(results)
