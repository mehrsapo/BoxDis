from __init__ import * 

class MRIPixel(): 

    def __init__(self, mask, device) -> None:
    
        self.mask = mask
        self.N = self.mask.size(2)
        self.device = device
        self.h = 1
        self.upsample_filter = torch.Tensor([[[[1, 1], [1, 1]]]]).to(device).double()
        return

    def upsample(self, x): 
        return F.conv_transpose2d(x, self.upsample_filter, stride=2)

    def downsample(self, x): 
        return F.conv2d(x, self.upsample_filter, stride=2)

    def set_h(self, h): 
        self.h = h
        return

    def H(self, x): 
    
        h = self.h 
        s = int(np.log2(h))
        if h >= 1: 
            x_up = x
            for _ in range(s): 
                x_up = self.upsample(x_up)
            Hx = torch.fft.fft2(x_up, norm='ortho') 

        return Hx * self.mask


    def Ht(self, y): 

        h = self.h 
        s = int(np.log2(h))

        y = y * self.mask
        
        if h >= 1: 
            y_down = torch.real(torch.fft.ifft2(y, norm='ortho'))
            for _ in range(s): 
                y_down = self.downsample(y_down)

            Hty = y_down
        
        
        return torch.real(Hty)