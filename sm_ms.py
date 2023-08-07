from __init__ import * 

class SM(): 

    def __init__(self, mask, device) -> None:
    
        self.mask = mask
        self.N = self.mask.size(2)
        self.device = device
        self.sam = Sampler(device)
        self.h = 1
        self.off = 0

        return

    def set_h(self, h, off): 
        self.off = off
        self.h = h
        return

    def H(self, x): 
        up_time = int(np.log2(self.h))
        if up_time >= 1:
            if self.off > 0:
                x = x[:, :, self.off:-self.off, self.off:-self.off]

            for _ in range(up_time):
                x = self.sam.upsample(x)
    
            off_up = 2 ** up_time - 1
            return  x[:, :, off_up:-off_up, off_up:-off_up] * self.mask

        else:
            dist = int(1/self.h)
            if self.off == 0:
                Hx = x[:, :, ::dist, ::dist] * self.mask
            else:
                Hx = x[:, :, self.off:-self.off:dist, self.off:-self.off:dist] * self.mask

        return Hx 


    def Ht(self, y): 

        y_masked = y * self.mask

        up_time = int(np.log2(self.h))

        if up_time >= 1:
            off_up = 2 ** up_time - 1
            y_masked = F.pad(y_masked, (off_up, off_up, off_up, off_up))
            for _ in range(up_time):
                y_masked = self.sam.downsample(y_masked)
            y_masked = F.pad(y_masked, (self.off, self.off, self.off, self.off))
        else:
            s = int(1/self.h)
            dim = s * y.size(2) - (s - 1)
            temp = torch.zeros((y.size(0), y.size(1), dim, dim)).to(self.device)
            temp[:, :, ::s, ::s] = y_masked

            y_masked = F.pad(temp, (self.off, self.off, self.off, self.off))

        return y_masked