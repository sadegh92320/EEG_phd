import torch
from downstream.models.foundation_models.cbramod import CBraModClassifier

model = CBraModClassifier(
    num_class=2, num_channel=19, data_length=1000,
    pretrained_dir="/Users/sadeghemami/checkpoint_benchmark/CBramod_pretrained_weights.pth"
)
x = torch.randn(4, 19, 1000)
out = model(x)
print("Output shape:", out.shape)
print("Output values:", out)
print("Loss:", torch.nn.CrossEntropyLoss()(out, torch.tensor([0,1,0,1])))