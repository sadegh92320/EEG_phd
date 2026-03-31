"""
Quick diagnostic: check if gradients flow through CBraMod during fine-tuning.
Run from project root:
    python debug_cbramod_ft.py
"""
import torch
import torch.nn as nn
from downstream.models.foundation_models.cbramod import CBraModClassifier
from einops.layers.torch import Rearrange

CKPT = "/Users/sadeghemami/checkpoint_benchmark/CBramod_pretrained_weights.pth"

# Build model exactly as build_cbramod does for full mode
model = CBraModClassifier(
    num_class=2, num_channel=19, data_length=1000,
    pretrained_dir=CKPT,
)

# Unfreeze backbone (same as build_cbramod full mode)
for p in model.backbone.parameters():
    p.requires_grad = True

# Set all_patch_reps classifier (same as build_cbramod full mode)
n_patches = 5
num_channels = 19
num_classes = 2
flat_features = num_channels * n_patches * 200  # 19000

model.classifier = nn.Sequential(
    Rearrange('b c s d -> b (c s d)'),
    nn.Linear(flat_features, n_patches * 200),
    nn.ELU(),
    nn.Dropout(0.1),
    nn.Linear(n_patches * 200, 200),
    nn.ELU(),
    nn.Dropout(0.1),
    nn.Linear(200, num_classes),
)

model.train()
model.eval()  # test in eval mode first (no dropout noise)

# Fake batch
x = torch.randn(8, 19, 1000) * 0.05  # scale ~CBraMod preprocessed data
y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

# Forward
out = model(x)
print("=== Forward pass ===")
print(f"Output shape: {out.shape}")
print(f"Output:\n{out}")
print(f"Softmax:\n{torch.softmax(out, dim=1)}")

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = loss_fn(out, y)
print(f"\nLoss: {loss.item():.4f}  (random baseline = {-torch.log(torch.tensor(0.5)).item():.4f})")

# Backward
loss.backward()

# Check gradients
print("\n=== Gradient check ===")

# Backbone gradients
backbone_grads = []
for name, p in model.backbone.named_parameters():
    if p.grad is not None:
        backbone_grads.append(p.grad.abs().mean().item())
    else:
        print(f"  NO GRAD: backbone.{name}")

if backbone_grads:
    print(f"  Backbone: {len(backbone_grads)} param groups with grad")
    print(f"  Backbone grad mean: {sum(backbone_grads)/len(backbone_grads):.2e}")
    print(f"  Backbone grad min:  {min(backbone_grads):.2e}")
    print(f"  Backbone grad max:  {max(backbone_grads):.2e}")

# Classifier gradients
print()
for i, layer in enumerate(model.classifier):
    if hasattr(layer, 'weight'):
        if layer.weight.grad is not None:
            print(f"  classifier[{i}] {layer.__class__.__name__}: "
                  f"weight grad mean={layer.weight.grad.abs().mean():.2e}, "
                  f"weight scale={layer.weight.abs().mean():.2e}")
        else:
            print(f"  classifier[{i}] {layer.__class__.__name__}: NO GRAD")

# Feature scale check
print("\n=== Feature scale check ===")
model.eval()
with torch.no_grad():
    x_windows = x.unfold(dimension=2, size=200, step=200)
    feats = model.backbone(x_windows)
    print(f"Backbone output shape: {feats.shape}")
    print(f"Backbone output mean:  {feats.mean():.4e}")
    print(f"Backbone output std:   {feats.std():.4e}")
    print(f"Backbone output range: [{feats.min():.4e}, {feats.max():.4e}]")

    feats_flat = feats.contiguous().view(8, -1)
    print(f"Flattened shape: {feats_flat.shape}")
    print(f"Flattened std:   {feats_flat.std():.4e}")

# Quick training test: 20 steps, does the loss go down?
print("\n=== Quick training test (20 steps) ===")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
for step in range(20):
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    if step % 5 == 0:
        with torch.no_grad():
            preds = out.argmax(dim=1)
            acc = (preds == y).float().mean()
        print(f"  step {step:2d}: loss={loss.item():.4f}, acc={acc.item():.2f}, "
              f"logits_mean={out.mean().item():.4e}, logits_std={out.std().item():.4e}")

print("\nDone.")
