from __init__ import * 

class MRIPixel(): 

    def __init__(self, mask, device) -> None:
    
        self.mask = mask
        self.N = self.mask.size(2)
        self.device = device
        self.sam = SamplerPixel(device)
        self.h = 1

        return

    def set_h(self, h): 
        self.h = h
        self.cal_box_corr()
        return

    def cal_box_corr(self): 
        N = self.N
        h = self.h 

        if h >= 1: 
            self.w_1 = torch.fft.fftfreq(N).repeat(N, 1).double()
            self.w_2 = torch.fft.fftfreq(N)[:, None].repeat(1, N).double()
            w_sum = self.w_1 + self.w_2
            shift1 = torch.exp(-1j * w_sum * torch.pi) # + 1j *torch.sin(-w_sum * torch.pi)
            self.box_corr = (torch.sinc(self.w_1) * torch.sinc(self.w_2) * shift1)[None, None, :, :].to(self.device)  

        elif h < 1: 
            self.w_22 = torch.fft.fftfreq(int(N/h)).repeat(int(N/h), 1).double()
            self.w_12 = torch.fft.fftfreq(int(N/h))[:, None].repeat(1, int(N/h)).double()
            w_sum2 = self.w_22 + self.w_12
            shift2 = torch.exp(-1j * w_sum2 * torch.pi) # + 1j *torch.sin(-w_sum2 * torch.pi)
            self.box_corr  = (torch.sinc(self.w_12) * torch.sinc(self.w_22) * shift2)[None, None, :, :].to(self.device) 

        return

    def H(self, x): 
        
        h = self.h; self.set_h(h)
        N = self.N
        s = int(np.log2(h))
        if h >= 1: 
            x_up = x
            for _ in range(s): 
                x_up = self.sam.upsample(x_up)
            Hx = torch.fft.fft2(x_up, norm='ortho') * self.box_corr 

        if h < 1: 
            N_t = N*int(1/h/2)
            N_h = int(N//2)
            c_hat = torch.fft.fft2(x, norm='ortho')
            Hx = torch.fft.ifftshift((torch.fft.fftshift(c_hat * self.box_corr))[:, :, N_t-N_h:N_t+N_h, N_t-N_h:N_t+N_h] * h)

        return Hx * self.mask


    def Ht(self, y): 

        h = self.h; self.set_h(h)
        N = self.N 
        s = int(np.log2(h))

        y = y * self.mask
        
        if h >= 1: 
            y_down = torch.real(torch.fft.ifft2(y  * torch.conj(self.box_corr), norm='ortho'))
            for _ in range(s): 
                y_down = self.sam.downsample(y_down)

            Hty = y_down
            

        if h < 1: 
            N_t = N*int(1/h/2)
            N_h = int(N//2)

            Hty = torch.zeros((1, 1, N*int(1/h), N*int(1/h))).to(self.device) + 1j * torch.zeros((1, 1, N*int(1/h), N*int(1/h))).to(self.device)
            Hty[:, :, N_t-N_h:N_t+N_h, N_t-N_h:N_t+N_h] = torch.fft.fftshift(y) * h
            Hty = torch.fft.ifftshift(Hty) * torch.conj(self.box_corr)
            Hty = torch.real(torch.fft.ifft2(Hty, norm='ortho'))
        

        return Hty