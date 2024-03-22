from __init__ import * 

class SamplerPixel(): 

    def __init__(self, device):
        
        self.device = device
        self.filter = torch.Tensor([[[[1, 1], [1, 1]]]]).to(device).double()

        return
    
    def upsample(self, x): 
        # factor = 2
        return F.conv_transpose2d(x, self.filter, stride=2)

    def downsample(self, x): 
        return F.conv2d(x, self.filter, stride=2)