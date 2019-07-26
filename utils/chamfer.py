import torch
from torch.utils.cpp_extension import load

cd = load(name="cd", sources=["utils/chamfer.cpp"])
# cd = load(name="cd", sources=["utils/chamfer.cpp", "utils/chamfer.cu"])

class ChamferDistance(torch.nn.Module):
    def forward(ctx, xyz1, xyz2):
        b, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1, xyz2 = xyz1.contiguous(), xyz2.contiguous()
        d1, d2 = torch.zeros(b, n), torch.zeros(b, m)
        i1, i2 = torch.zeros(b, n, dtype=torch.int), torch.zeros(b, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, d1, d2, i1, i2)
        else:
            d1, d2, i1, i2 = d1.cuda(), d2.cuda(), i1.cuda(), i2.cuda()
            cd.forward_cuda(xyz1, xyz2, d1, d2, i1, i2)

        return d1, d2