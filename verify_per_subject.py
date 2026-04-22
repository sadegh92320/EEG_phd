import copy
from downstream.models.deep_learning_model.eeg_net import EEGNet

model = EEGNet(no_spatial_filters=2, no_channels=22, no_temporal_filters=8,
               temporal_length_1=125, temporal_length_2=31, window_length=1000, num_class=4)
fresh = {k: v.clone() for k, v in model.state_dict().items()}

# simulate one subject of training
import torch
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
x = torch.randn(32, 22, 1000); y = torch.randint(0, 4, (32,))
for _ in range(50):
    opt.zero_grad(); loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward(); opt.step()

# now do what Trainer-2's __init__ does
snap = copy.deepcopy(model.state_dict())

# compare to fresh
import torch
diffs = {k: (snap[k] - fresh[k]).abs().max().item() for k in fresh}
print("max abs diff per layer between Trainer-2 snapshot and fresh random init:")
for k, v in diffs.items():
    if v > 1e-8:
        print(f"  {k}: {v:.4e}")