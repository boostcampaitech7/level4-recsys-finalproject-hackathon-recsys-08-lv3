import torch

checkpoint_path = 'outputs_stage2/202501290330/checkpoint_5.pth'
checkpoint = torch.load(checkpoint_path)
print(checkpoint.keys())