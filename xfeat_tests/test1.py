import torch

xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)

#Simple inference with batch sz = 1
output = xfeat.detectAndCompute(torch.randn(1,3,480,640), top_k = 4096)[0]
print(output)