from __init__ import *

class HTV_svd_closed():
    def __init__(self, device):
        
        self.filters1 = torch.Tensor([[[[1., -2., 1]]]]).to(device).double()
        self.filters2 = torch.Tensor([[[[1], [-2], [1]]]]).to(device).double()
        self.filters3 = torch.Tensor([[[[1., -1.], [-1., 1]]]]).to(device).double()

    def hessian_2d_closed(self, x): 
        hessian1_valid = F.conv2d(x, self.filters1)
        bound1 = x[:, :, :, -2:-1] - x[:, :, :, -1:]
        hessian1 = torch.cat((hessian1_valid, bound1, bound1), dim=3)
        
        hessian2_valid = F.conv2d(x, self.filters2)
        bound2 = x[:, :, -2:-1, :] - x[:, :, -1:, :]
        hessian2 = torch.cat((hessian2_valid, bound2, bound2 ), dim=2)

        hessian3_valid = F.conv2d(x, self.filters3)
        hessian3 = F.pad(hessian3_valid, (0, 1), "constant", 0)
        hessian3 = F.pad(hessian3, (0, 0, 0, 1), "constant", 0)


        hessian = torch.cat((hessian1, hessian2 , hessian3), dim=1)
        return hessian
    
    def hessian_adj_2d_closed(self, y):
        hessian_adj1 = F.conv_transpose2d(y[:, 0:1, :, :-2], self.filters1)
        bound_adj1 = y[:, 0:1, :, -2:-1] + y[:, 0:1, :, -1:]
        hessian_adj1[:, :, :, -2:-1] += bound_adj1
        hessian_adj1[:, :, :, -1:] -= bound_adj1

        hessian_adj2 = F.conv_transpose2d(y[:, 1:2, :-2, :], self.filters2)
        bound_adj2 = y[:, 1:2, -2:-1, :] + y[:, 1:2, -1:, :]
        hessian_adj2[:, :, -2:-1, :] += bound_adj2
        hessian_adj2[:, :, -1:, :] -= bound_adj2

        hessian_adj3 = F.conv_transpose2d(y[:, 2:3, :-1, :-1], self.filters3)
        
        hessian_adj = hessian_adj1 + hessian_adj2 + hessian_adj3 

        return hessian_adj
    