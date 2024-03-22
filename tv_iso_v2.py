from __init__ import *

class TV_ISO_v2():
    def __init__(self, device):

        self.filters1 = torch.Tensor([[[[1., -1.]]]]).double().to(device)
        self.filters2 = torch.Tensor([[[[1.], [-1.]]]]).double().to(device)

    def L(self, x, h): 
        conv_1 = F.conv2d(F.pad(x, (1, 1, 1, 0)), self.filters1)
        conv_2 = F.conv2d(F.pad(x, (1, 0, 1, 1)), self.filters2)


        L1_2 = F.pad(conv_1, (0, 1, 0, 1))
        L2_2 = torch.roll(torch.roll(F.pad(conv_2, (0, 1, 0, 1)), 1, 2), -1, 3)

        Lx = torch.cat((L1_2, L2_2), dim=1) 

        return Lx * h 
    
    def Lt(self, y, h):


        Lt1y_2 = F.conv_transpose2d(y[:, 0:1, :-1, :-1], self.filters1)[:, :, 1:, 1:-1]
        Lt2y_2 = F.conv_transpose2d(torch.roll(torch.roll(y[:, 1:2, :, :], 1, 3), -1, 2)[:, :, :-1, :-1], self.filters2)[:, :, 1:-1, 1:]

        Lty = Lt1y_2 + Lt2y_2 
        
        return Lty * h 
    

if __name__ == '__main__': 

    device = 'cpu'
    x = torch.normal(0, 1, (1, 1, 60, 60)).double().to(device)
    y = torch.normal(0, 1, (1, 2, 62, 62)).double().to(device)
    tv = TV_l2(device)
    h = 1
    print(tv.L(x, h).size(), tv.Lt(y, h).size())
    Hxy = torch.tensordot(tv.L(x, h), y, 4)
    xHty = torch.tensordot(tv.Lt(y, h), x, 4)
    print(Hxy, xHty)
