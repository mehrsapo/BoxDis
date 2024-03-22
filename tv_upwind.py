from __init__ import *

class TV_upwind():
    def __init__(self, device):

        self.filters1 = torch.Tensor([[[[1., -1.]]]]).double().to(device)
        self.filters2 = torch.Tensor([[[[1.], [-1.]]]]).double().to(device)

    def L(self, x, h): 
        conv_1 = F.conv2d(F.pad(x, (1, 1, 1, 0)), self.filters1)
        conv_2 = F.conv2d(F.pad(x, (1, 0, 1, 1)), self.filters2)

        L1_1 = F.pad(conv_1, (0, 1, 0, 1))
        L2_1 = F.pad(conv_2, (0, 1, 0, 1))

        L_1 = torch.cat((L1_1, L2_1), dim=1) 

        L1_2 =  torch.roll(-L1_1, 1, 3)
        L2_2 = torch.roll(-L2_1, 1, 2)

        L_2 = torch.cat((L1_2, L2_2), dim=1) 

        Lx = torch.cat((L_1, L_2), dim=1)

        return Lx * h 
    
    def Lt(self, y, h):


        Lt1y_1 = F.conv_transpose2d(y[:, 0:1, :-1, :-1], self.filters1)[:, :, 1:, 1:-1]
        Lt2y_1 = F.conv_transpose2d(y[:, 1:2, :-1, :-1], self.filters2)[:, :, 1:-1, 1:]

        Lt1y_2 = F.conv_transpose2d(torch.roll(-y[:, 2:3, :, :], -1, 3)[:, :, :-1, :-1], self.filters1)[:, :, 1:, 1:-1]
        Lt2y_2 = F.conv_transpose2d(torch.roll(-y[:, 3:4, :, :], -1, 2)[:, :, :-1, :-1], self.filters2)[:, :, 1:-1, 1:]

        Lty = Lt1y_2 + Lt2y_2 + Lt1y_1 + Lt2y_1 
        
        return Lty * h 
    
