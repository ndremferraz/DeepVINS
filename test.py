import torch
from model.encoders import InertialBlock, ImageBlock
from model.fusion import AttentionBlock, CausalFusionModel

imu = InertialBlock()
x_imu = torch.randn(4, 6, 20)
y_imu = imu(x_imu)
print("imu:", y_imu.shape)   # expect (4, fc2_dims)

img = ImageBlock()
x_img = torch.randn(4, 2, 64, 64)
y_img = img(x_img)
print("img:", y_img.shape)   # expect (4, fc2_dims)

attn = AttentionBlock()
x_attn = torch.randn(4, 3, 1024)
y_attn = attn(x_attn)
print("attn:", y_attn.shape)   # expect (4, seq_len, embed_dim)

fusion = CausalFusionModel()
y_fusion = fusion(x_imu, x_img)
print("fusion:", y_fusion.shape)   # expect (4, 6)

print(torch.isnan(y_imu).any(), torch.isinf(y_imu).any())
print(torch.isnan(y_img).any(), torch.isinf(y_img).any())
print(torch.isnan(y_attn).any(), torch.isinf(y_attn).any())
print(torch.isnan(y_fusion).any(), torch.isinf(y_fusion).any())

loss = y_imu.mean() + y_img.mean() + y_attn.mean() + y_fusion.mean()
loss.backward()

for name, p in list(imu.named_parameters())[:3]:
    print(name, p.grad is not None)

for name, p in list(attn.named_parameters())[:3]:
    print(name, p.grad is not None)

for name, p in list(fusion.named_parameters())[:3]:
    print(name, p.grad is not None)
